//! P5.9-06: GPU diagonal phase-polynomial kernel vs the unfused cphase ladder.
//!
//! Baseline = unfused QFT/QPE (each controlled-Phase a full `apply_diag` sweep).
//! Diagonal = `run(fuse_for_gpu(c))`, where `FuseDiagonalRuns` collapses the
//! ladder into a few `DiagonalPhase` blocks applied by `apply_phase_poly` in one
//! coalesced sweep each. This is the lever the P5.9-05 verdict named for QFT/QPE.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{fuse_for_gpu, CudaSvBackend};
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        c.h(j).unwrap();
        for (offset, k) in ((j + 1)..n).enumerate() {
            let theta = std::f64::consts::PI / (1u64 << (offset + 1)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                vec![k],
                vec![j],
            ))
            .unwrap();
        }
    }
    c
}

fn qpe(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let eig = n - 1;
    let m = n - 1;
    c.x(eig).unwrap();
    for q in 0..m {
        c.h(q).unwrap();
    }
    let theta = 2.0 * std::f64::consts::PI * 0.123;
    for k in 0..m {
        let ang = theta * (1u64 << k) as f64;
        c.add_gate(GateInstance::controlled(
            Gate::Phase(Param::Concrete(ang)),
            vec![eig],
            vec![k],
        ))
        .unwrap();
    }
    for j in (0..m).rev() {
        for (offset, k) in (0..j).rev().enumerate() {
            let ang = -std::f64::consts::PI / (1u64 << (offset + 1)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(ang)),
                vec![j],
                vec![k],
            ))
            .unwrap();
        }
        c.h(j).unwrap();
    }
    c
}

fn time(gpu: &mut CudaSvBackend, circ: &Circuit, reps: u32) -> f64 {
    let _ = run(gpu, circ).expect("warmup");
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = run(gpu, circ).expect("run");
        let _ = HasAmplitudes::amplitudes(&st);
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn gpu_diag_phase_bench() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut gpu = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping diag-phase bench: {e}");
            return;
        }
    };
    println!("== P5.9-06 GPU diagonal phase kernel, n={n}, best of {reps} ==");
    println!("workload  unfused(s)  diag-fused(s)  speedup");
    for (name, circ) in [("qft", qft(n)), ("qpe", qpe(n))] {
        let fused = fuse_for_gpu(&circ);
        let unf = time(&mut gpu, &circ, reps);
        let dia = time(&mut gpu, &fused, reps);
        println!("{name:<8}  {unf:8.4}    {dia:9.4}     {:.2}×", unf / dia);
    }
}
