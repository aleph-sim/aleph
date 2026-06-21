//! P5.6-06: property + rejection coverage for `MetalMpsBackend`.
//!
//! * Property: a random **nearest-neighbour** circuit's dense statevector must
//!   match the CPU `aleph_mps::MpsBackend` within the f32 oracle tolerance. The
//!   bond cap stays above the (small-n, bounded-depth) entanglement, so neither
//!   MPS truncates and the compare is exact-to-fp32.
//! * Rejection: every unsupported path (external control, fused `UnitaryKq`,
//!   non-NN 2q, 3q, and the stochastic/readout ops) must return
//!   `UnsupportedInstruction`, not silently mis-handle the input.
//!
//! Device-or-skip so headless/Linux CI stays green.
//!
//! Run: `cargo test -p aleph-metal --features metal --test mps_proptest`

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_backend::{run, Backend, BackendError};
use aleph_core::{Complex, Gate, GateInstance, Param, Pauli, PauliString};
use aleph_ir::{Circuit, Instruction};
use aleph_metal::MetalMpsBackend;
use aleph_mps::MpsBackend;
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Bond cap kept above the entanglement of the bounded-depth NN circuits below,
/// so neither MPS truncates (the truncating regime is covered separately by the
/// P5.6-02 guard test).
const MAX_BOND: usize = 64;

/// Build a random nearest-neighbour circuit: 1q gates anywhere, 2q gates only on
/// adjacent pairs `(q, q+1)` (the scaffold has no SWAP router).
fn random_nn_circuit(rng: &mut StdRng, n: u32, gates: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..gates {
        let theta = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        match rng.gen_range(0..6u32) {
            0 => push(&mut c, Gate::H, &[rng.gen_range(0..n)]),
            1 => push(&mut c, Gate::X, &[rng.gen_range(0..n)]),
            2 => push(
                &mut c,
                Gate::Rz(Param::Concrete(theta)),
                &[rng.gen_range(0..n)],
            ),
            3 => push(
                &mut c,
                Gate::Ry(Param::Concrete(theta)),
                &[rng.gen_range(0..n)],
            ),
            4 if n >= 2 => {
                let q = rng.gen_range(0..n - 1);
                push(&mut c, Gate::Cnot, &[q, q + 1]);
            }
            5 if n >= 2 => {
                let q = rng.gen_range(0..n - 1);
                push(&mut c, Gate::Cz, &[q, q + 1]);
            }
            // n == 1 fallback.
            _ => push(&mut c, Gate::H, &[rng.gen_range(0..n)]),
        }
    }
    c
}

fn push(c: &mut Circuit, g: Gate, qubits: &[u32]) {
    c.add_instruction(Instruction::Gate(GateInstance::new(g, qubits.to_vec())))
        .unwrap();
}

/// CPU MPS dense reference.
fn cpu_mps_dense(circuit: &Circuit) -> Vec<Complex<f64>> {
    let mut be = MpsBackend::new().with_max_bond(MAX_BOND);
    run(&mut be, circuit)
        .expect("cpu mps run")
        .dense_statevector()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// AC: random NN circuits match the CPU MPS within ε.
    #[test]
    fn mps_metal_matches_cpu_on_random_nn(seed in any::<u64>(), n in 2u32..=5, gates in 4usize..=24) {
        let mut gpu = match MetalMpsBackend::with_max_bond(MAX_BOND) {
            Ok(b) => b,
            Err(_) => return Ok(()), // headless: skip
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let circuit = random_nn_circuit(&mut rng, n, gates);
        let want = cpu_mps_dense(&circuit);

        let got = gpu.run(&circuit).expect("gpu mps run").dense_statevector();
        prop_assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = ((g.re - w.re).powi(2) + (g.im - w.im).powi(2)).sqrt();
            prop_assert!(d <= 1e-5, "amp {i}: |Δ|={d:.3e} (n={n}, gates={gates}, seed={seed})");
        }

        // P5.7-04: the layer-batched scheduler must agree with the CPU MPS too.
        let got_b = gpu.run_batched(&circuit).expect("gpu mps run_batched").dense_statevector();
        prop_assert_eq!(got_b.len(), want.len());
        for (i, (g, w)) in got_b.iter().zip(want.iter()).enumerate() {
            let d = ((g.re - w.re).powi(2) + (g.im - w.im).powi(2)).sqrt();
            prop_assert!(d <= 1e-5, "batched amp {i}: |Δ|={d:.3e} (n={n}, gates={gates}, seed={seed})");
        }
    }
}

/// Apply `inst` to a fresh `n`-qubit state and return the backend error.
fn reject(n: u32, inst: GateInstance) -> Option<BackendError> {
    let mut b = MetalMpsBackend::with_max_bond(MAX_BOND).ok()?;
    let mut s = b.allocate(n).unwrap();
    Some(b.apply_gate(&mut s, &inst).unwrap_err())
}

fn assert_unsupported(err: Option<BackendError>) {
    let Some(err) = err else {
        return; // headless: skip
    };
    assert!(
        matches!(err, BackendError::UnsupportedInstruction { .. }),
        "expected UnsupportedInstruction, got {err:?}"
    );
}

#[test]
fn rejects_external_control() {
    // CNOT-on-X expressed as an externally-controlled X — the scaffold has no
    // controlled-gate path.
    let inst = GateInstance::controlled(Gate::X, vec![1u32], vec![0u32]);
    assert_unsupported(reject(3, inst));
}

#[test]
fn rejects_non_nearest_neighbour_2q() {
    // CNOT across a gap (q0, q2) — no SWAP router.
    assert_unsupported(reject(3, GateInstance::new(Gate::Cnot, vec![0u32, 2u32])));
}

#[test]
fn rejects_three_qubit_gate() {
    // Toffoli is a dense 3q (M8x8) block the scaffold does not factorize.
    assert_unsupported(reject(
        3,
        GateInstance::new(Gate::Toffoli, vec![0u32, 1u32, 2u32]),
    ));
}

#[test]
fn rejects_fused_unitary_kq() {
    // A fused dense block (run_optimized output) — unsupported; use `run`.
    let data = vec![Complex::<f64>::new(0.0, 0.0); 16];
    let inst = GateInstance::new(
        Gate::UnitaryKq {
            k: 2,
            data: data.into_boxed_slice(),
        },
        vec![0u32, 1u32],
    );
    assert_unsupported(reject(3, inst));
}

#[test]
fn rejects_stochastic_and_readout_ops() {
    let Some(mut b) = MetalMpsBackend::with_max_bond(MAX_BOND).ok() else {
        return; // headless: skip
    };
    let mut s = b.allocate(3).unwrap();
    assert!(matches!(
        b.measure(&mut s, 0).unwrap_err(),
        BackendError::UnsupportedInstruction { .. }
    ));
    assert!(matches!(
        b.sample(&s, 8).unwrap_err(),
        BackendError::UnsupportedInstruction { .. }
    ));
    let pauli = PauliString::new(1.0, vec![(0u32, Pauli::Z)]).unwrap();
    assert!(matches!(
        b.expectation_value(&s, &pauli).unwrap_err(),
        BackendError::UnsupportedInstruction { .. }
    ));
    assert!(matches!(
        b.probabilities(&s, &[0]).unwrap_err(),
        BackendError::UnsupportedInstruction { .. }
    ));
}
