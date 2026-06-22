//! P5-08 GPU benchmark report harness: our hand-written FP64 state-vector
//! backend (`CudaSvBackend`, with the P5-06 diagonal routing) vs NVIDIA
//! cuStateVec (`CuStateVecBackend`) across the Tier-1 and Tier-2 algorithms, on
//! the same circuit. The Phase-5 exit metric is **our GPU within 1.5× of
//! cuStateVec**; this prints the ratio per workload and flags any cell over 1.5×.
//!
//! `#[ignore]` (needs a GPU + minutes). Run:
//! ```bash
//! ALEPH_REPORT_N=28 cargo test -p aleph-cuda --features cuquantum --release \
//!   -- --ignored --nocapture gpu_report
//! ```
//! An equivalent Qiskit-Aer-GPU column is produced by `gpu_report_bench.py`
//! (separate, since Aer is driven from Python); see the perf doc.

#![cfg(all(target_os = "linux", feature = "cuquantum"))]

use std::time::Instant;

use aleph_backend::{run, Backend};
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{CuStateVecBackend, CudaSvBackend};
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use rand::{rngs::StdRng, Rng, SeedableRng};

// --- Tier-1 + Tier-2 circuit builders (representative gate mixes) ---

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 1..n {
        c.cnot(0, q).unwrap();
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

/// One Grover iteration (phase oracle on |1…1⟩ + diffusion), repeated `iters`
/// times — the oracle/diffusion gate mix without the full √N iteration count.
fn grover(n: u32, iters: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let mcz = |c: &mut Circuit| {
        // (n-1)-controlled Z; IR caps controls at 8, so cap the control fan-in.
        let ctrls: Vec<u32> = (0..(n - 1).min(8)).collect();
        c.add_gate(GateInstance::controlled(Gate::Z, vec![n - 1], ctrls))
            .unwrap();
    };
    for _ in 0..iters {
        mcz(&mut c);
        for q in 0..n {
            c.h(q).unwrap();
            c.x(q).unwrap();
        }
        mcz(&mut c);
        for q in 0..n {
            c.x(q).unwrap();
            c.h(q).unwrap();
        }
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

/// Quantum phase estimation: counting qubits `0..n-1` over a 1-qubit eigenstate
/// `n-1`. H on counting, controlled-`Phase(2^k·θ)`, then inverse QFT.
fn qpe(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let eig = n - 1;
    let m = n - 1; // counting qubits 0..m
    c.x(eig).unwrap(); // |1⟩ eigenstate of Phase
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
    // Inverse QFT on the counting register.
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

/// Hardware-efficient VQE ansatz: `layers` of (Ry on all) + CNOT chain + (Rz on all).
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

/// QAOA Max-Cut on a ring: `p` layers of cost (`Rzz` per ring edge, as
/// CNOT·Rz·CNOT) + mixer (`Rx` on all).
fn qaoa(n: u32, p: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let gamma = 0.7_f64;
    let beta = 0.4_f64;
    for _ in 0..p {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = (i.min(j), i.max(j));
            c.cnot(a, b).unwrap();
            c.rz(2.0 * gamma, b).unwrap();
            c.cnot(a, b).unwrap();
        }
        for q in 0..n {
            c.rx(2.0 * beta, q).unwrap();
        }
    }
    c
}

/// Best-of-`reps` full-run wall-clock (incl. the final device→host sync that a
/// single-amplitude read forces).
fn time_backend<B: Backend>(backend: &mut B, circuit: &Circuit, reps: u32) -> f64
where
    B::State: HasAmplitudes,
{
    let _ = run(backend, circuit).expect("warmup");
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = run(backend, circuit).expect("run");
        let _ = HasAmplitudes::amplitudes(&st);
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn gpu_report() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let mut rng = StdRng::seed_from_u64(0x5208);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("grover(4it)", grover(n, 4)),
        ("random(d20)", random_brickwall(&mut rng, n, 20)),
        ("qpe", qpe(n)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
    ];

    let mut ours = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_report: {e}");
            return;
        }
    };
    let mut cuq = CuStateVecBackend::with_seed(0)
        .expect("cuStateVec")
        .with_qubit_cap(n);

    println!("== P5-08 GPU report: aleph SV vs cuStateVec, n={n}, best of {reps} ==");
    println!("workload       aleph(s)   cuStateVec(s)   ratio(aleph/cuQ)  exit≤1.5×");
    let mut worst = 0.0_f64;
    for (name, circ) in &workloads {
        let a = time_backend(&mut ours, circ, reps);
        let q = time_backend(&mut cuq, circ, reps);
        let ratio = a / q;
        worst = worst.max(ratio);
        let flag = if ratio <= 1.5 { "PASS" } else { "OVER" };
        println!("{name:<14} {a:8.4}    {q:10.4}      {ratio:8.2}×         {flag}");
    }
    println!("\nworst ratio: {worst:.2}× (exit gate: ≤1.5× of cuStateVec)");
}
