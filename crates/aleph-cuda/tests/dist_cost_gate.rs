//! P6-05 §6.3 model-accuracy gate: the calibrated GpuCostModel's compute term
//! vs measured T_onecard on one GPU (all R ranks, LocalExchange), with the
//! on-card exchange copies subtracted via an exchange-only plan.
//! #538 Stage C: the old cells plus four held-out cells (never used to choose
//! a constant or the rule), with a `generic steps` column from the class walk.
//! Run (idle box): cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use aleph_cuda::{
    state_classes, CudaContext, CudaSvBackend, DistSvBackend, GenericTimes, GpuCostModel,
    KindTimes, LocalExchange,
};
use aleph_ir::dist::{plan, CostModel, DistLayout, DistStep, Router};
use aleph_ir::Circuit;
use common::dist::{
    all_ranks, best_of, brickwall_bench, ccz_ladder, clifford_brickwall, comm_only, ghz,
    grover_iters, hea_bench, qaoa_ring_chords, qaoa_ring_skip7, qft,
};

/// `model` with every kind zeroed except the one `keep` leaves set: the
/// per-kind share of the all-ranks compute (for the report's reading).
fn only(model: &GpuCostModel, keep: impl Fn(&KindTimes, &mut KindTimes)) -> GpuCostModel {
    let k = model.kinds;
    let mut z = KindTimes {
        m_ref: k.m_ref,
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
    keep(&k, &mut z);
    GpuCostModel {
        kinds: z,
        ..model.clone()
    }
}

/// Prints one gate table for `cases` and returns its worst |model/measured − 1|.
fn gate_table(
    title: &str,
    cases: &[(&str, Circuit)],
    n: u32,
    sync: &CudaContext,
    d: &mut DistSvBackend<CudaSvBackend, LocalExchange<CudaSvBackend>>,
    model: &GpuCostModel,
) -> f64 {
    let shares: Vec<(&str, GpuCostModel)> = vec![
        (
            "dense1",
            only(model, |k, z| {
                z.dense1 = k.dense1;
                z.generic.dense1 = k.generic.dense1;
            }),
        ),
        (
            "dense2",
            only(model, |k, z| {
                z.dense2 = k.dense2;
                z.generic.dense2 = k.generic.dense2;
            }),
        ),
        (
            "dense3",
            only(model, |k, z| {
                z.dense3 = k.dense3;
                z.generic.dense3 = k.generic.dense3;
            }),
        ),
        (
            "diag1",
            only(model, |k, z| {
                z.diag1 = k.diag1;
                z.generic.diag1 = k.generic.diag1;
            }),
        ),
        (
            "diag_k",
            only(model, |k, z| {
                z.diag_k = k.diag_k;
                z.generic.diag_k = k.generic.diag_k;
            }),
        ),
        (
            "cnot",
            only(model, |k, z| {
                z.cnot = k.cnot;
                z.generic.cnot = k.generic.cnot;
            }),
        ),
        (
            "phase",
            only(model, |k, z| {
                z.phase_base = k.phase_base;
                z.phase_term = k.phase_term;
                z.phase_term_multi = k.phase_term_multi;
                z.generic.phase_base = k.generic.phase_base;
                z.generic.phase_term = k.generic.phase_term;
                z.generic.phase_term_multi = k.generic.phase_term_multi;
            }),
        ),
    ];
    println!("### {title}");
    println!("| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio | generic steps | verdict |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut worst: f64 = 0.0;
    let mut breakdown = Vec::new();
    for (name, c) in cases {
        for g in [1u32, 2] {
            let l = DistLayout::new(n, g).unwrap();
            let p = plan(c, l, Router::Lookahead).unwrap();
            let comm = comm_only(&p);
            let t_full = best_of(sync, 3, || drop(d.run_plan(&p).unwrap()));
            let t_comm = best_of(sync, 3, || drop(d.run_plan(&comm).unwrap()));
            let measured = t_full - t_comm;
            let all = all_ranks(model, &p);
            let costs = model.step_costs(&p).unwrap();
            let rep: f64 = p
                .steps
                .iter()
                .zip(&costs)
                .filter(|(s, _)| matches!(s, DistStep::Local(_)))
                .map(|(_, c)| f64::from(l.ranks()) * c)
                .sum();
            let classes = state_classes(&p).unwrap();
            let (generic, locals) = p
                .steps
                .iter()
                .zip(&classes)
                .filter(|(s, _)| matches!(s, DistStep::Local(_)))
                .fold((0usize, 0usize), |(g, t), (_, &c)| {
                    (g + usize::from(c), t + 1)
                });
            let ratio = all / measured;
            // is_finite first: f64::max swallows NaN (ADR 0006), so a bad ratio is a MISS.
            let pass = ratio.is_finite() && (ratio - 1.0).abs() <= 0.10;
            worst = if ratio.is_finite() {
                worst.max((ratio - 1.0).abs())
            } else {
                f64::INFINITY
            };
            let verdict = if pass { "PASS" } else { "MISS" };
            println!(
                "| {name} | {} | {measured:.3} | {all:.3} | {ratio:.3} | {rep:.3} | {:.3} | {generic}/{locals} | {verdict} |",
                l.ranks(),
                rep / measured
            );
            let parts: Vec<String> = shares
                .iter()
                .map(|(k, m)| format!("{k}={:.3}", all_ranks(m, &p)))
                .collect();
            breakdown.push(format!(
                "kinds: {name} D={} t_full={t_full:.3} t_comm={t_comm:.3} {}",
                l.ranks(),
                parts.join(" ")
            ));
        }
    }
    for b in &breakdown {
        println!("{b}");
    }
    println!(
        "{title}: worst |model/measured − 1| = {:.1} %",
        100.0 * worst
    );
    worst
}

#[test]
#[ignore]
fn model_gate_n28_fp64() {
    let Ok(sync) = CudaContext::new(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let Ok(be) = CudaSvBackend::with_seed(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let mut d = DistSvBackend::new(be, LocalExchange::new());
    let model = GpuCostModel::rtx4000_fp64();
    let n = 28;
    let old: Vec<(&str, Circuit)> = vec![
        ("QFT", qft(n)),
        ("GHZ", ghz(n)),
        ("random d=10", brickwall_bench(n, 10)),
        ("QAOA p=2", qaoa_ring_chords(n)),
        ("CCZ ladder d=4", ccz_ladder(n, 4)),
        ("Grover K=3", grover_iters(n, 3)),
    ];
    let held_out: Vec<(&str, Circuit)> = vec![
        ("HEA d=4", hea_bench(n)),
        ("random d=20", brickwall_bench(n, 20)),
        ("Clifford brickwall d=10", clifford_brickwall(n, 10)),
        ("QAOA p=2 skip-7", qaoa_ring_skip7(n)),
    ];
    let w_old = gate_table(
        "old cells (seen during design)",
        &old,
        n,
        &sync,
        &mut d,
        &model,
    );
    let w_new = gate_table(
        "held-out cells (never used to choose anything)",
        &held_out,
        n,
        &sync,
        &mut d,
        &model,
    );
    println!(
        "worst |model/measured − 1|: old {:.1} %, held-out {:.1} %",
        100.0 * w_old,
        100.0 * w_new
    );
    assert!(
        w_old.max(w_new) <= 0.10,
        "#538 exit 1: every old and held-out cell within ±10 %"
    );
}
