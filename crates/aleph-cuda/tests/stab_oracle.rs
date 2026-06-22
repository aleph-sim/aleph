//! Oracle tests for the GPU stabilizer backend (P5-07).
//!
//! Clifford gate application is deterministic, so a GPU tableau that started
//! from `|0…0⟩` and saw the same gate sequence as the CPU `aleph-stab` tableau
//! must be **bit-for-bit identical** in its `2n` generator rows. Both the
//! per-gate kernel (`apply`) and the batched disjoint-layer kernel
//! (`apply_layer`) are pinned against the CPU tableau over a range of `n` that
//! spans several word boundaries.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! when no CUDA device is present.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance};
use aleph_cuda::{CudaStab, StabOp};
use aleph_ir::Circuit;
use aleph_stab::StabilizerBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A random Clifford circuit as disjoint-qubit layers (so `apply_layer`'s
/// disjointness contract holds). Returns the layers plus the equivalent
/// `Circuit` for the CPU oracle — the two describe the identical gate sequence.
fn random_clifford(rng: &mut StdRng, n: u32, depth: u32) -> (Vec<Vec<StabOp>>, Circuit) {
    let mut layers = Vec::with_capacity(depth as usize);
    let mut circuit = Circuit::new(n, 0);
    for _ in 0..depth {
        // Random permutation of the qubits, then greedily pair / single.
        let mut perm: Vec<u32> = (0..n).collect();
        for i in (1..perm.len()).rev() {
            perm.swap(i, rng.gen_range(0..=i));
        }
        let mut layer = Vec::new();
        let mut i = 0usize;
        while i < perm.len() {
            let q = perm[i];
            // Pair into a CNOT ~half the time when a partner remains.
            if i + 1 < perm.len() && rng.gen_bool(0.5) {
                let t = perm[i + 1];
                layer.push(StabOp::cnot(q, t));
                circuit
                    .add_gate(GateInstance::new(Gate::Cnot, vec![q, t]))
                    .unwrap();
                i += 2;
            } else {
                let (sop, g) = match rng.gen_range(0..5) {
                    0 => (StabOp::h(q), Gate::H),
                    1 => (StabOp::s(q), Gate::S),
                    2 => (StabOp::x(q), Gate::X),
                    3 => (StabOp::y(q), Gate::Y),
                    _ => (StabOp::z(q), Gate::Z),
                };
                layer.push(sop);
                circuit.add_gate(GateInstance::new(g, vec![q])).unwrap();
                i += 1;
            }
        }
        layers.push(layer);
    }
    (layers, circuit)
}

fn cpu_truth(circuit: &Circuit) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let mut be = StabilizerBackend::with_seed(0);
    let t = run(&mut be, circuit).expect("cpu stab run");
    t.export_generators()
}

fn assert_eq_tableau(
    name: &str,
    got: &(Vec<bool>, Vec<bool>, Vec<bool>),
    want: &(Vec<bool>, Vec<bool>, Vec<bool>),
) {
    assert_eq!(got.0, want.0, "{name}: x-bits differ");
    assert_eq!(got.1, want.1, "{name}: z-bits differ");
    assert_eq!(got.2, want.2, "{name}: sign-bits differ");
}

#[test]
fn gpu_stab_matches_cpu_tableau() {
    // n chosen to straddle word boundaries: Wr = ceil((2n+1)/64).
    for n in [5u32, 33, 64, 65, 130, 200] {
        let driver = match CudaStab::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping GPU stab oracle: {e}");
                return;
            }
        };
        let mut rng = StdRng::seed_from_u64(0xC1_1F_F0_00 ^ n as u64);
        let (layers, circuit) = random_clifford(&mut rng, n, 40);
        let want = cpu_truth(&circuit);

        // Path A: batched disjoint-layer kernel.
        let mut sa = driver.allocate(n).unwrap();
        for layer in &layers {
            driver.apply_layer(&mut sa, layer).unwrap();
        }
        assert_eq_tableau(
            &format!("apply_layer n={n}"),
            &sa.export_generators().unwrap(),
            &want,
        );

        // Path B: per-gate kernel, same sequence.
        let mut sb = driver.allocate(n).unwrap();
        for layer in &layers {
            for &g in layer {
                driver.apply(&mut sb, g).unwrap();
            }
        }
        assert_eq_tableau(
            &format!("apply n={n}"),
            &sb.export_generators().unwrap(),
            &want,
        );
    }
}

/// Spot-check the initial `|0…0⟩` tableau matches before any gate (catches a
/// bad `stab_init`).
#[test]
fn gpu_stab_initial_state_matches_cpu() {
    let driver = match CudaStab::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping GPU stab init oracle: {e}");
            return;
        }
    };
    for n in [1u32, 7, 64, 100] {
        let s = driver.allocate(n).unwrap();
        let circuit = Circuit::new(n, 0);
        let want = cpu_truth(&circuit);
        assert_eq_tableau(
            &format!("init n={n}"),
            &s.export_generators().unwrap(),
            &want,
        );
    }
}
