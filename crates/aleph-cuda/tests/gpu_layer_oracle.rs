//! P5.9-03: disjoint-1q-layer batched dispatch, oracle-equal to per-gate.
//!
//! `CudaSvBackend::run_layered` folds consecutive plain 1q gates on distinct
//! qubits into one `apply_1q_multi` sweep per `MAX_LAYER_BATCH`-wide chunk,
//! instead of one sweep per gate. This pins it against the unbatched CPU
//! `NaiveSvBackend` per-gate `run` at 1e-10, across the dense workloads plus an
//! adversarial circuit that exercises every flush trigger (same-qubit 1q run,
//! 1q/2q interleave, a diagonal in the batch, a barrier).
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

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

/// Hits every `run_layered` control-flow edge: a multi-gate batch wider than
/// MAX_LAYER_BATCH (forces ≥2 chunks), a *second* gate on an already-pending
/// qubit (collision flush, order-sensitive), a diagonal gate inside the batch,
/// a 2q gate (non-batchable flush), and a barrier.
fn adversarial(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    // Wide disjoint 1q layer (> MAX_LAYER_BATCH=5 ⇒ multiple chunks), mixing a
    // diagonal (rz) in with non-diagonal rotations.
    for q in 0..n {
        if q % 4 == 0 {
            c.rz(0.3 + q as f64 * 0.1, q).unwrap();
        } else {
            c.rx(0.2 + q as f64 * 0.05, q).unwrap();
        }
    }
    // Same-qubit follow-up: must flush before applying (non-commuting order).
    c.ry(0.7, 0).unwrap();
    c.add_gate(GateInstance::new(Gate::T, vec![0])).unwrap();
    // 2q gate breaks the run.
    c.cnot(0, 1).unwrap();
    // Another disjoint 1q layer after the 2q.
    for q in 0..n {
        c.h(q).unwrap();
    }
    c.barrier(0..n).unwrap();
    // A controlled 1q (external control) must NOT be batched.
    c.add_gate(GateInstance::controlled(
        Gate::Phase(Param::Concrete(0.9)),
        vec![2],
        vec![3],
    ))
    .unwrap();
    for q in 0..n {
        c.rx(0.11 * (q as f64 + 1.0), q).unwrap();
    }
    c
}

#[test]
fn run_layered_matches_cpu_per_gate() {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping layer oracle: {e}");
            return;
        }
    };
    let n = 11;
    let mut rng = StdRng::seed_from_u64(0x0590_30a3);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("random(d20)", random_brickwall(&mut rng, n, 20)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
        ("adversarial", adversarial(n)),
    ];
    for (name, circ) in &workloads {
        let mut cpu = NaiveSvBackend::with_seed(0);
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu"));
        let got = HasAmplitudes::amplitudes(&gpu.run_layered(circ).expect("gpu layered"));
        assert_eq!(got.len(), want.len(), "{name}: len");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let d = (a - b).norm();
            assert!(d <= 1e-10, "{name} i={i}: |Δ|={d:.2e}");
        }
    }
}

/// Every batch width 1..=5 must give the identical result (the batching is a
/// pure dispatch optimization, not a numeric approximation).
#[test]
fn run_layered_batch_width_invariant() {
    let n = 10;
    let mut rng = StdRng::seed_from_u64(0x000b_47c4);
    let circ = random_brickwall(&mut rng, n, 12);

    let mut cpu = NaiveSvBackend::with_seed(0);
    let want = HasAmplitudes::amplitudes(&run(&mut cpu, &circ).expect("cpu"));

    for batch in 1..=5 {
        let mut gpu = match CudaSvBackend::with_seed(0).map(|b| b.with_layer_batch(batch)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping batch-width invariant: {e}");
                return;
            }
        };
        let got = HasAmplitudes::amplitudes(&gpu.run_layered(&circ).expect("gpu"));
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let d = (a - b).norm();
            assert!(d <= 1e-10, "batch={batch} i={i}: |Δ|={d:.2e}");
        }
    }
}
