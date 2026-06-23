//! P5.9-06: GPU diagonal phase-polynomial kernel, oracle-equal to unfused.
//!
//! `FuseDiagonalRuns` collapses a controlled-phase ladder (QFT/QPE) into one
//! `Instruction::DiagonalPhase`; the GPU `apply_phase_poly` kernel applies it in
//! one coalesced `amps[x] *= exp(i·φ(x))` sweep. This pins the fused GPU run
//! (both per-gate `run` and `run_layered`) against the unfused per-gate CPU
//! `NaiveSvBackend` at 1e-10, and asserts the fused circuits actually contain a
//! `DiagonalPhase` so the new kernel path is exercised.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{fuse_for_gpu, CudaSvBackend};
use aleph_ir::{Circuit, Instruction};
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;

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

fn n_diagonal_phase(c: &Circuit) -> usize {
    c.instructions()
        .iter()
        .filter(|i| matches!(i, Instruction::DiagonalPhase(_)))
        .count()
}

#[test]
fn gpu_diag_phase_matches_cpu_unfused() {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping diag-phase oracle: {e}");
            return;
        }
    };
    let n = 11;
    let mut n_dp_total = 0usize;
    for (name, circ) in [("qft", qft(n)), ("qpe", qpe(n))] {
        let fused = fuse_for_gpu(&circ);
        let n_dp = n_diagonal_phase(&fused);
        n_dp_total += n_dp;

        let mut cpu = NaiveSvBackend::with_seed(0);
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, &circ).expect("cpu"));

        // Both GPU drivers must apply the fused DiagonalPhase correctly.
        let got = HasAmplitudes::amplitudes(&run(&mut gpu, &fused).expect("gpu run"));
        let got_layered =
            HasAmplitudes::amplitudes(&gpu.run_layered(&fused).expect("gpu run_layered"));

        assert_eq!(got.len(), want.len(), "{name}: len");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).norm() <= 1e-10,
                "{name} run i={i}: |Δ|={:.2e}",
                (a - b).norm()
            );
        }
        for (i, (a, b)) in got_layered.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).norm() <= 1e-10,
                "{name} run_layered i={i}: |Δ|={:.2e}",
                (a - b).norm()
            );
        }
    }
    // The fused QFT/QPE must actually emit DiagonalPhase, or the oracle proves
    // nothing about apply_phase_poly.
    assert!(
        n_dp_total > 0,
        "expected FuseDiagonalRuns to emit DiagonalPhase in the GPU pipeline"
    );
}
