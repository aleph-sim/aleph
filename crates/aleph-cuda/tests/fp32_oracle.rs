//! P5.10-03: FP32 GPU state-vector correctness.
//!
//! `CudaSvBackendF32` stores amplitudes as float for ~2× bandwidth and +1 qubit
//! of in-core reach, at FP32 accuracy. This pins its final state against the
//! exact FP64 CPU `NaiveSvBackend` within **1e-5** across Tier-1 (GHZ, QFT,
//! Grover, random) and Tier-2 (VQE, QAOA) workloads — exercising every FP32
//! kernel: `apply_1q_f32`, `apply_cnot_f32`, `apply_kq_f32`, and the diagonal
//! fast paths (`apply_diag_1q_f32` / `apply_diag_f32`).
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackendF32;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

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

/// One Grover iteration over `n` qubits: H-wrap, a phase oracle (multi-controlled
/// Z marking |1…1⟩) and the diffusion operator — a mix of H, multi-controlled
/// diagonal Z, and X.
fn grover(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    // Oracle: multi-controlled Z marking a subspace (the builder caps controls
    // at 8, so use up to the first 8 qubits as controls on the last target).
    let ctrls: Vec<u32> = (0..(n - 1).min(8)).collect();
    c.add_gate(GateInstance::controlled(
        Gate::Z,
        vec![n - 1],
        ctrls.clone(),
    ))
    .unwrap();
    // Diffusion.
    for q in 0..n {
        c.h(q).unwrap();
        c.x(q).unwrap();
    }
    c.add_gate(GateInstance::controlled(Gate::Z, vec![n - 1], ctrls))
        .unwrap();
    for q in 0..n {
        c.x(q).unwrap();
        c.h(q).unwrap();
    }
    c
}

fn random_brickwall(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            match rng.gen_range(0..3) {
                0 => c.rx(theta, q),
                1 => c.ry(theta, q),
                _ => c.rz(theta, q),
            }
            .unwrap();
        }
        for q in (0..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in (1..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
    }
    c
}

fn vqe(n: u32, layers: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let mut t = 0.1_f64;
    for _ in 0..layers {
        for q in 0..n {
            c.ry(t, q).unwrap();
            t += 0.017;
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in 0..n {
            c.rz(t, q).unwrap();
            t += 0.013;
        }
    }
    c
}

fn qaoa(n: u32, p: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for _ in 0..p {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = (i.min(j), i.max(j));
            c.cnot(a, b).unwrap();
            c.rz(1.4, b).unwrap();
            c.cnot(a, b).unwrap();
        }
        for q in 0..n {
            c.rx(0.8, q).unwrap();
        }
    }
    c
}

/// FP32 GPU must match the exact FP64 CPU oracle within 1e-5 on every workload.
#[test]
fn fp32_matches_cpu_within_1e5() {
    let mut gpu = match CudaSvBackendF32::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 oracle: {e}");
            return;
        }
    };
    let n = 10;
    let mut rng = StdRng::seed_from_u64(0x51003);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("grover", grover(n)),
        ("random(d12)", random_brickwall(&mut rng, n, 12)),
        ("vqe(6L)", vqe(n, 6)),
        ("qaoa(p3)", qaoa(n, 3)),
    ];

    let mut worst = 0.0f64;
    for (name, circ) in &workloads {
        let mut cpu = NaiveSvBackend::with_seed(0);
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu fp64"));
        let got = gpu.run(circ).expect("gpu fp32").amplitudes_vec();
        assert_eq!(got.len(), want.len(), "{name}: len");
        let mut max_d = 0.0f64;
        for (a, b) in got.iter().zip(want.iter()) {
            max_d = max_d.max((a - b).norm());
        }
        worst = worst.max(max_d);
        assert!(max_d <= 1e-5, "{name}: max |Δ|={max_d:.2e} exceeds 1e-5");
        println!("{name:<12} max |Δ| = {max_d:.2e}");
    }
    println!("worst across Tier-1+2: {worst:.2e} (tol 1e-5)");
}
