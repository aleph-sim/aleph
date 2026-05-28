//! Property tests for P1-08 specialised Toffoli + CCZ kernels.
//!
//! Verifies, on random state vectors, that the matrix-shape dispatch
//! chain produces results equivalent to applying the gate twice:
//!
//! - CCX ∘ CCX = I  (involutivity)
//! - CCZ ∘ CCZ = I  (involutivity)
//! - CCZ(q0,q1,q2) ≡ CCZ(q2,q0,q1)  (qubit symmetry)
//!
//! States are built from |0…0⟩ by applying `Ry(θ_i)` on each qubit
//! with a proptest-generated angle, producing a fully general product
//! state without needing direct access to `CpuState`'s private fields.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance, Param};
use aleph_sv::NaiveSvBackend;
use proptest::prelude::*;
use smallvec::smallvec;

const TOL: f64 = 1e-10;

/// Build a non-trivial product state on `n` qubits by applying `Ry(angles[i])`
/// on qubit `i` via the public `Backend::apply_gate` API.
fn product_state(
    backend: &mut NaiveSvBackend,
    n: u32,
    angles: &[f64],
) -> <NaiveSvBackend as Backend>::State {
    let mut state = backend.allocate(n).unwrap();
    for (i, &theta) in angles.iter().take(n as usize).enumerate() {
        let gi = GateInstance::new(Gate::Ry(Param::Concrete(theta)), smallvec![i as u32]);
        backend.apply_gate(&mut state, &gi).unwrap();
    }
    state
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// CCX applied twice equals the identity on any state.
    #[test]
    fn ccx_is_involutive(
        n in 3u32..=6,
        angles in proptest::collection::vec(-std::f64::consts::PI..=std::f64::consts::PI, 3..=6),
        c0 in 0u32..6,
        c1 in 0u32..6,
        t  in 0u32..6,
    ) {
        // Require three distinct indices all within [0, n).
        prop_assume!(c0 != c1 && c0 != t && c1 != t);
        prop_assume!(c0 < n && c1 < n && t < n);

        let mut backend = NaiveSvBackend::with_seed(0);
        let state_before = product_state(&mut backend, n, &angles);
        let original: Vec<_> = state_before.amplitudes().to_vec();

        let mut state = state_before;
        let gi = GateInstance::new(Gate::Toffoli, smallvec![c0, c1, t]);
        backend.apply_gate(&mut state, &gi).unwrap();
        backend.apply_gate(&mut state, &gi).unwrap();

        for (a, b) in state.amplitudes().iter().zip(original.iter()) {
            prop_assert!(
                (a.re - b.re).abs() < TOL,
                "re mismatch: got {}, expected {}", a.re, b.re
            );
            prop_assert!(
                (a.im - b.im).abs() < TOL,
                "im mismatch: got {}, expected {}", a.im, b.im
            );
        }
    }

    /// CCZ applied twice equals the identity on any state.
    #[test]
    fn ccz_is_involutive(
        n in 3u32..=6,
        angles in proptest::collection::vec(-std::f64::consts::PI..=std::f64::consts::PI, 3..=6),
        q0 in 0u32..6,
        q1 in 0u32..6,
        q2 in 0u32..6,
    ) {
        prop_assume!(q0 != q1 && q0 != q2 && q1 != q2);
        prop_assume!(q0 < n && q1 < n && q2 < n);

        let mut backend = NaiveSvBackend::with_seed(0);
        let state_before = product_state(&mut backend, n, &angles);
        let original: Vec<_> = state_before.amplitudes().to_vec();

        let mut state = state_before;
        let gi = GateInstance::new(Gate::Ccz, smallvec![q0, q1, q2]);
        backend.apply_gate(&mut state, &gi).unwrap();
        backend.apply_gate(&mut state, &gi).unwrap();

        for (a, b) in state.amplitudes().iter().zip(original.iter()) {
            prop_assert!(
                (a.re - b.re).abs() < TOL,
                "re mismatch: got {}, expected {}", a.re, b.re
            );
            prop_assert!(
                (a.im - b.im).abs() < TOL,
                "im mismatch: got {}, expected {}", a.im, b.im
            );
        }
    }

    /// CCZ is symmetric in its qubit arguments: CCZ(q0,q1,q2) ≡ CCZ(q2,q0,q1).
    ///
    /// The CCZ matrix is symmetric in all three qubit positions; any permutation
    /// of the qubit list must produce the same output state.
    #[test]
    fn ccz_symmetric_in_qubit_order(
        n in 3u32..=6,
        angles in proptest::collection::vec(-std::f64::consts::PI..=std::f64::consts::PI, 3..=6),
        q0 in 0u32..6,
        q1 in 0u32..6,
        q2 in 0u32..6,
    ) {
        prop_assume!(q0 != q1 && q0 != q2 && q1 != q2);
        prop_assume!(q0 < n && q1 < n && q2 < n);

        let mut backend = NaiveSvBackend::with_seed(0);

        // Apply CCZ(q0, q1, q2).
        let mut state_a = product_state(&mut backend, n, &angles);
        let ga = GateInstance::new(Gate::Ccz, smallvec![q0, q1, q2]);
        backend.apply_gate(&mut state_a, &ga).unwrap();

        // Apply CCZ(q2, q0, q1) — a cyclic permutation — to the same initial state.
        let mut state_b = product_state(&mut backend, n, &angles);
        let gb = GateInstance::new(Gate::Ccz, smallvec![q2, q0, q1]);
        backend.apply_gate(&mut state_b, &gb).unwrap();

        for (x, y) in state_a.amplitudes().iter().zip(state_b.amplitudes().iter()) {
            prop_assert!(
                (x.re - y.re).abs() < TOL,
                "CCZ qubit-order symmetry violated: re {} vs {}", x.re, y.re
            );
            prop_assert!(
                (x.im - y.im).abs() < TOL,
                "CCZ qubit-order symmetry violated: im {} vs {}", x.im, y.im
            );
        }
    }
}
