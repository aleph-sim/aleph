//! Integration test for `expectation_pauli_sum`.
//!
//! Lives in `tests/` (not `#[cfg(test)]` inside `lib.rs`) so that
//! `NaiveSvBackend` from `aleph-sv` and `Backend` from `aleph-backend`
//! resolve to the same version — the trait-version mismatch that prevents
//! using `NaiveSvBackend` inside unit tests is avoided here.

use aleph_backend::{expectation_pauli_sum, run};
use aleph_core::{Pauli, PauliString, PauliSum};
use aleph_ir::Circuit;
use aleph_sv::NaiveSvBackend;

#[test]
fn energy_of_zero_state() {
    // |00>: <Z_q> = +1 each; H = 1.0*Z0 + 2.0*Z1 - 0.5*I  => 1 + 2 - 0.5 = 2.5
    let mut backend = NaiveSvBackend::with_seed(0);
    let circuit = Circuit::new(2, 0);
    let state = run(&mut backend, &circuit).unwrap();
    let ham = PauliSum {
        terms: vec![
            PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap(),
            PauliString::new(2.0, vec![(1, Pauli::Z)]).unwrap(),
            PauliString::identity(-0.5),
        ],
    };
    let e = expectation_pauli_sum(&mut backend, &state, &ham).unwrap();
    assert!((e - 2.5).abs() < 1e-12, "got {e}");
}
