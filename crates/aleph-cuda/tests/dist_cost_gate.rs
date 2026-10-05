//! P6-05 §6.3 model-accuracy gate: the calibrated GpuCostModel's compute term
//! vs measured T_onecard on one GPU (all R ranks, LocalExchange), with the
//! on-card exchange copies subtracted via an exchange-only plan.
//! Run (idle box): cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use aleph_cuda::{
    CudaContext, CudaSvBackend, DistSvBackend, GpuCostModel, KindTimes, LocalExchange,
};
use aleph_ir::dist::{plan, DistLayout, DistStep, Router};
use aleph_ir::Circuit;
use common::dist::{
    all_ranks, best_of, brickwall_bench, ccz_ladder, comm_only, ghz, grover_iters,
    qaoa_ring_chords, qft,
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
    };
    keep(&k, &mut z);
    GpuCostModel {
        kinds: z,
        ..model.clone()
    }
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
    let cases: Vec<(&str, Circuit)> = vec![
        ("QFT", qft(n)),
        ("GHZ", ghz(n)),
        ("random d=10", brickwall_bench(n, 10)),
        ("QAOA p=2", qaoa_ring_chords(n)),
        ("CCZ ladder d=4", ccz_ladder(n, 4)),
        ("Grover K=3", grover_iters(n, 3)),
    ];
    let shares: Vec<(&str, GpuCostModel)> = vec![
        ("dense1", only(&model, |k, z| z.dense1 = k.dense1)),
        ("dense2", only(&model, |k, z| z.dense2 = k.dense2)),
        ("dense3", only(&model, |k, z| z.dense3 = k.dense3)),
        ("diag1", only(&model, |k, z| z.diag1 = k.diag1)),
        ("diag_k", only(&model, |k, z| z.diag_k = k.diag_k)),
        ("cnot", only(&model, |k, z| z.cnot = k.cnot)),
        (
            "phase",
            only(&model, |k, z| {
                z.phase_base = k.phase_base;
                z.phase_term = k.phase_term;
                z.phase_term_multi = k.phase_term_multi;
            }),
        ),
    ];
    println!("| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio |");
    println!("|---|---|---|---|---|---|---|");
    let mut worst: f64 = 0.0;
    let mut breakdown = Vec::new();
    for (name, c) in &cases {
        for g in [1u32, 2] {
            let l = DistLayout::new(n, g).unwrap();
            let p = plan(c, l, Router::Lookahead).unwrap();
            let comm = comm_only(&p);
            let t_full = best_of(&sync, 3, || drop(d.run_plan(&p).unwrap()));
            let t_comm = best_of(&sync, 3, || drop(d.run_plan(&comm).unwrap()));
            let measured = t_full - t_comm;
            let all = all_ranks(&model, &p);
            let mut rep = 0.0;
            for s in &p.steps {
                if let DistStep::Local(instrs) = s {
                    rep += f64::from(l.ranks())
                        * model.rank_segment(instrs, l, l.ranks() - 1).unwrap();
                }
            }
            let ratio = all / measured;
            worst = worst.max((ratio - 1.0).abs());
            println!(
                "| {name} | {} | {measured:.3} | {all:.3} | {ratio:.3} | {rep:.3} | {:.3} |",
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
    println!("worst |model/measured − 1| = {:.1} %", 100.0 * worst);
    assert!(
        worst <= 0.10,
        "spec §6.3 gate: model compute must be within ±10 % on every cell"
    );
}
