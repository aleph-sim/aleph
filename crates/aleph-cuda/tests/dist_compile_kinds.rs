//! P6-05 report §3.4: plan structure of the Naive, Lookahead and compiled
//! plans at n=28 FP64, model only (no timing, the box need not be idle).
//! Per plan: exchange widths, local swaps, `Local` segments, rank-0 launches
//! after specialise + fusion, the calibrated model's all-ranks compute split by
//! kernel kind, and the all-ranks launch count of each kind. Counts are
//! *counted*, not derived from seconds: a model whose constant is 1.0 for one
//! kind and 0 for the rest, with `m_ref = m` so the `2^(m − m_ref)` scale is 1
//! (PhasePoly: `phase_base` = 1, per-term costs 0).
//! Run: cargo test --release -p aleph-cuda --features cuda --test dist_compile_kinds -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use aleph_cuda::{
    CudaSvBackend, DistSvBackend, GenericTimes, GpuCostModel, KindTimes, LocalExchange,
};
use aleph_ir::dist::{compile_detailed, plan, DistLayout, DistPlan, DistStep, Router};
use aleph_ir::Circuit;
use common::dist::{
    all_ranks, brickwall_bench, ccz_ladder, ghz, grover_iters, qaoa_ring_chords, qft,
};

const ZERO: KindTimes = KindTimes {
    m_ref: 0,
    dense1: 0.0,
    dense2: 0.0,
    dense3: 0.0,
    diag1: 0.0,
    diag_k: 0.0,
    cnot: 0.0,
    phase_base: 0.0,
    phase_term: 0.0,
    phase_term_multi: 0.0,
    generic: GenericTimes::NONE,
};

/// `base` with every kind zeroed except what `keep` sets, at slice size `m_ref`.
fn with_kinds(base: &GpuCostModel, m_ref: u32, keep: impl Fn(&mut KindTimes)) -> GpuCostModel {
    let mut z = KindTimes { m_ref, ..ZERO };
    keep(&mut z);
    GpuCostModel {
        kinds: z,
        ..base.clone()
    }
}

/// Copies the calibrated constant(s) of one kind into a zeroed `KindTimes`.
type Pick = fn(&KindTimes, &mut KindTimes);
/// Sets one kind's per-launch cost to 1 (a launch counter).
type Count = fn(&mut KindTimes);

/// (name, pick, count) per kind.
const KINDS: [(&str, Pick, Count); 7] = [
    (
        "dense1",
        |k, z| {
            z.dense1 = k.dense1;
            z.generic.dense1 = k.generic.dense1;
        },
        |z| z.dense1 = 1.0,
    ),
    (
        "dense2",
        |k, z| {
            z.dense2 = k.dense2;
            z.generic.dense2 = k.generic.dense2;
        },
        |z| z.dense2 = 1.0,
    ),
    (
        "dense3",
        |k, z| {
            z.dense3 = k.dense3;
            z.generic.dense3 = k.generic.dense3;
        },
        |z| z.dense3 = 1.0,
    ),
    (
        "diag1",
        |k, z| {
            z.diag1 = k.diag1;
            z.generic.diag1 = k.generic.diag1;
        },
        |z| z.diag1 = 1.0,
    ),
    (
        "diag_k",
        |k, z| {
            z.diag_k = k.diag_k;
            z.generic.diag_k = k.generic.diag_k;
        },
        |z| z.diag_k = 1.0,
    ),
    (
        "cnot",
        |k, z| {
            z.cnot = k.cnot;
            z.generic.cnot = k.generic.cnot;
        },
        |z| z.cnot = 1.0,
    ),
    (
        "phase",
        |k, z| {
            z.phase_base = k.phase_base;
            z.phase_term = k.phase_term;
            z.phase_term_multi = k.phase_term_multi;
            z.generic.phase_base = k.generic.phase_base;
            z.generic.phase_term = k.generic.phase_term;
            z.generic.phase_term_multi = k.generic.phase_term_multi;
        },
        |z| z.phase_base = 1.0,
    ),
];

fn widths(p: &DistPlan) -> String {
    let w: Vec<String> = p
        .steps
        .iter()
        .filter_map(|s| match s {
            DistStep::Exchange { global_bits } => Some(global_bits.len().to_string()),
            DistStep::Local(_) => None,
        })
        .collect();
    w.join(",")
}

#[test]
#[ignore]
fn compile_kinds_n28_fp64() {
    let Ok(be) = CudaSvBackend::with_seed(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let d = DistSvBackend::new(be, LocalExchange::new());
    let model = GpuCostModel {
        fuse: d.fusion(),
        ..GpuCostModel::rtx4000_fp64()
    };
    let n = 28;
    let cases: Vec<(&str, Circuit)> = vec![
        ("QFT", qft(n)),
        ("GHZ", ghz(n)),
        ("random d=10", brickwall_bench(n, 10)),
        ("QAOA p=2", qaoa_ring_chords(n)),
        ("CCZ ladder d=4", ccz_ladder(n, 4)),
        ("Grover K=3", grover_iters(n, 3)),
    ];
    println!("| circuit | D | plan | exchange widths | local swaps | `Local` segs | rank-0 launches | model all-ranks (s) | dense1 | dense2 | dense3 | diag1 | diag_k | cnot | phase |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (name, c) in &cases {
        for g in [1u32, 2] {
            let l = DistLayout::new(n, g).unwrap();
            let pn = plan(c, l, Router::Naive).unwrap();
            let pl = plan(c, l, Router::Lookahead).unwrap();
            let comp = compile_detailed(c, l, &model).unwrap();
            let chosen = format!(
                "compiled ({:?}, placed={})",
                comp.choice.router, comp.choice.placed
            );
            for (tag, p) in [
                ("naive", &pn),
                ("lookahead", &pl),
                (chosen.as_str(), &comp.plan),
            ] {
                let segs = p
                    .steps
                    .iter()
                    .filter(|s| matches!(s, DistStep::Local(_)))
                    .count();
                let parts: Vec<String> = KINDS
                    .iter()
                    .map(|(_, pick, count)| {
                        let secs = with_kinds(&model, model.kinds.m_ref, |z| pick(&model.kinds, z));
                        let cnt = with_kinds(&model, l.m(), count);
                        format!("{:.3} [{:.0}]", all_ranks(&secs, p), all_ranks(&cnt, p))
                    })
                    .collect();
                println!(
                    "| {name} | {} | {tag} | {} | {} | {segs} | {} | {:.3} | {} |",
                    l.ranks(),
                    widths(p),
                    p.stats.local_swaps,
                    d.rank_pass_count(p).unwrap(),
                    all_ranks(&model, p),
                    parts.join(" | ")
                );
            }
        }
    }
}
