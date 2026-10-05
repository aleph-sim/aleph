//! P6-05 §8: predicted multi-GPU time of the compiled plan vs Naive and
//! Lookahead at n=28, with *measured* compute: one card runs all R ranks,
//! exchange copies are subtracted with the exchange-only plan, and the AWS
//! link table prices the exchanges. Prints the table and the exit verdicts
//! (spec §8); it does not assert them. The `exit3b` lines are an extra
//! plan-level check on compiled plans; the spec's exit 3 is `dist_cost_gate`.
//! Run (idle box): cargo test --release -p aleph-cuda --features cuda --test dist_compile_bench -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use std::time::Instant;

use aleph_cuda::{
    CudaContext, CudaSvBackend, CudaSvBackendF32, DeviceSv, DistSvBackend, GpuCostModel,
    LocalExchange,
};
use aleph_ir::dist::{
    compile_detailed, plan, Candidate, CostModel, DistLayout, DistPlan, DistStep, Router,
};
use aleph_ir::Circuit;
use common::dist::{
    all_ranks, best_of, brickwall_bench, ccz_ladder, comm_only, ghz, grover_iters,
    qaoa_ring_chords, qft,
};

/// Σ over exchanges of the model's link time.
fn link_seconds(model: &GpuCostModel, p: &DistPlan) -> f64 {
    let m = p.layout.m();
    p.steps
        .iter()
        .map(|s| match s {
            DistStep::Exchange { global_bits } => model.exchange(global_bits.len() as u32, m),
            DistStep::Local(_) => 0.0,
        })
        .sum()
}

/// One measured plan.
#[derive(Clone, Copy)]
struct Meas {
    /// Measured compute, all ranks (s).
    all: f64,
    /// Predicted per-GPU time: all / D + link (s).
    t_pred: f64,
    /// Model all-ranks compute / measured.
    ratio: f64,
    /// Measured compute is finite and positive (noise can break this).
    valid: bool,
}

fn measure<B: DeviceSv>(
    sync: &CudaContext,
    d: &mut DistSvBackend<B, LocalExchange<B>>,
    model: &GpuCostModel,
    p: &DistPlan,
) -> Meas {
    let comm = comm_only(p);
    let t_full = best_of(sync, 3, || drop(d.run_plan(p).unwrap()));
    let t_comm = best_of(sync, 3, || drop(d.run_plan(&comm).unwrap()));
    let all = t_full - t_comm;
    let ranks = f64::from(p.layout.ranks());
    Meas {
        valid: all.is_finite() && all > 0.0,
        all,
        t_pred: all / ranks + link_seconds(model, p),
        ratio: all_ranks(model, p) / all,
    }
}

fn label(c: Candidate) -> String {
    let r = match c.router {
        Router::Naive => "naive".to_string(),
        Router::Lookahead => "lookahead".to_string(),
        Router::Reorder { max_k } => format!("reorder k={max_k}"),
    };
    if c.placed {
        format!("{r}+place")
    } else {
        r
    }
}

/// Runs every (circuit, D) cell, printing the table and pushing exit-verdict
/// lines into `verdicts`. The table's `compile (ms)` is a single cold run; the
/// compile-time verdict (in `compile_bench_n28`) is best of 5.
fn run_cells<B: DeviceSv>(
    sync: &CudaContext,
    d: &mut DistSvBackend<B, LocalExchange<B>>,
    model: &GpuCostModel,
    tag: &str,
    cases: &[(&str, Circuit)],
    verdicts: &mut Vec<String>,
) {
    let n = 28;
    println!("\n### {tag}\n");
    println!("| circuit | D | chosen | exch N/L/C | T_pred naive (s) | T_pred lookahead (s) | T_pred compiled (s) | compiled / min(N,L) | compiled / L | model/measured (C) | compile (ms) | measured all N/L/C (s) |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut cand_lines: Vec<String> = Vec::new();
    for (name, c) in cases {
        for g in [1u32, 2] {
            let l = DistLayout::new(n, g).unwrap();
            let pn = plan(c, l, Router::Naive).unwrap();
            let pl = plan(c, l, Router::Lookahead).unwrap();
            let t0 = Instant::now();
            let comp = compile_detailed(c, l, model).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let mn = measure(sync, d, model, &pn);
            let ml = measure(sync, d, model, &pl);
            // Reuse a measurement when the compiled plan *is* a baseline.
            let mc = match (comp.choice.router, comp.choice.placed) {
                (Router::Naive, false) => mn,
                (Router::Lookahead, false) => ml,
                _ => measure(sync, d, model, &comp.plan),
            };
            let base = mn.t_pred.min(ml.t_pred);
            println!(
                "| {name} | {} | {} | {}/{}/{} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {ms:.1} | {:.3}/{:.3}/{:.3} |",
                l.ranks(),
                label(comp.choice),
                pn.stats.exchanges,
                pl.stats.exchanges,
                comp.plan.stats.exchanges,
                mn.t_pred,
                ml.t_pred,
                mc.t_pred,
                mc.t_pred / base,
                mc.t_pred / ml.t_pred,
                mc.ratio,
                mn.all,
                ml.all,
                mc.all,
            );
            let valid = mn.valid && ml.valid && mc.valid;
            const INVALID: &str = "INVALID (measured compute ≤ 0 or non-finite)";
            let cands: Vec<String> = comp
                .candidates
                .iter()
                .map(|&(cd, t)| format!("{}={t:.3}", label(cd)))
                .collect();
            cand_lines.push(format!(
                "candidates: {name} D={} model T: {}",
                l.ranks(),
                cands.join(" ")
            ));
            if tag.starts_with("FP64") {
                let ok1 = mc.t_pred <= 1.03 * base;
                verdicts.push(format!(
                    "exit1 {name} D={}: compiled/min = {:.3} → {}",
                    l.ranks(),
                    mc.t_pred / base,
                    if !valid {
                        INVALID
                    } else if ok1 {
                        "PASS"
                    } else {
                        "MISS"
                    }
                ));
                let ok3 = mc.ratio.is_finite() && (mc.ratio - 1.0).abs() <= 0.10;
                verdicts.push(format!(
                    "exit3b (plan-level compiled-plan check; spec exit 3 = dist_cost_gate) {name} D={}: compiled model/measured = {:.3} (measured all-ranks {:.3} s) → {}",
                    l.ranks(),
                    mc.ratio,
                    mc.all,
                    if !valid {
                        INVALID
                    } else if ok3 {
                        "PASS"
                    } else {
                        "MISS"
                    }
                ));
                if *name == "random d=10" && g == 1 {
                    let gain = 1.0 - mc.t_pred / ml.t_pred;
                    verdicts.push(format!(
                        "exit2 random d=10 D=2: compiled {:.3} s vs lookahead {:.3} s = {:.1} % better (need ≥ 15 %) → {}",
                        mc.t_pred,
                        ml.t_pred,
                        100.0 * gain,
                        if !valid {
                            INVALID
                        } else if gain >= 0.15 {
                            "PASS"
                        } else {
                            "MISS"
                        }
                    ));
                }
            }
        }
    }
    println!();
    for l in &cand_lines {
        println!("{l}");
    }
}

#[test]
#[ignore]
fn compile_bench_n28() {
    let Ok(sync) = CudaContext::new(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let n = 28;
    let mut verdicts = Vec::new();

    // FP64: every §8 workload.
    {
        let Ok(be) = CudaSvBackend::with_seed(0) else {
            eprintln!("skipped: no CUDA");
            return;
        };
        let mut d = DistSvBackend::new(be, LocalExchange::new());
        let model = GpuCostModel {
            fuse: d.fusion(),
            ..GpuCostModel::rtx4000_fp64()
        };
        let cases: Vec<(&str, Circuit)> = vec![
            ("QFT", qft(n)),
            ("GHZ", ghz(n)),
            ("random d=10", brickwall_bench(n, 10)),
            ("QAOA p=2", qaoa_ring_chords(n)),
            ("CCZ ladder d=4", ccz_ladder(n, 4)),
            ("Grover K=3", grover_iters(n, 3)),
        ];
        run_cells(&sync, &mut d, &model, "FP64", &cases, &mut verdicts);

        // Compile time, ~1k gates at g=2 (spec §6.4 target < 50 ms).
        let c = brickwall_bench(n, 15);
        let l = DistLayout::new(n, 2).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            drop(compile_detailed(&c, l, &model).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        verdicts.push(format!(
            "compile time: brickwall d=15 ({} gates) g=2: {best:.1} ms (target < 50 ms) → {}",
            c.instructions().len(),
            if best < 50.0 { "PASS" } else { "MISS" }
        ));
    }

    // FP32: QFT and random only (reported, no exit criterion).
    {
        let Ok(be) = CudaSvBackendF32::with_seed(0) else {
            eprintln!("skipped: no CUDA FP32");
            return;
        };
        let mut d = DistSvBackend::new(be, LocalExchange::new());
        let model = GpuCostModel {
            fuse: d.fusion(),
            ..GpuCostModel::rtx4000_fp32()
        };
        let cases: Vec<(&str, Circuit)> =
            vec![("QFT", qft(n)), ("random d=10", brickwall_bench(n, 10))];
        run_cells(&sync, &mut d, &model, "FP32", &cases, &mut verdicts);
    }

    println!();
    for v in &verdicts {
        println!("{v}");
    }
}
