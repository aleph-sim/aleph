//! Cross-backend equivalence: the stabilizer tableau and the state
//! vector backend must agree on the action of every Clifford gate.
//!
//! Method: prepare an identical generic Clifford state in both backends,
//! apply the gate under test, then for each tableau stabilizer generator
//! `g` (sign `s`, unsigned Pauli `P`) assert `⟨ψ|P|ψ⟩ = s` to 1e-10 —
//! i.e. `g|ψ⟩ = |ψ⟩`. (P1-13 lesson: prep a generic state, not |0…0⟩.)

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance, PauliString};
use aleph_stab::{apply_gate, Tableau};
use aleph_sv::NaiveSvBackend;

const N: usize = 4;

/// A fixed generic Clifford preparation applied to both backends.
fn prep() -> Vec<GateInstance> {
    vec![
        GateInstance::new(Gate::H, vec![0u32]),
        GateInstance::new(Gate::S, vec![0u32]),
        GateInstance::new(Gate::Cnot, vec![0u32, 1u32]),
        GateInstance::new(Gate::H, vec![2u32]),
        GateInstance::new(Gate::Cnot, vec![2u32, 3u32]),
        GateInstance::new(Gate::Cnot, vec![1u32, 2u32]),
    ]
}

fn assert_stabilized(gates_under_test: &[GateInstance]) {
    // Tableau side
    let mut t = Tableau::new(N);
    for g in prep().iter().chain(gates_under_test) {
        apply_gate(&mut t, g).unwrap();
    }
    // SV side
    let mut be = NaiveSvBackend::default();
    let mut sv = be.allocate(N as u32).unwrap();
    for g in prep().iter().chain(gates_under_test) {
        be.apply_gate(&mut sv, g).unwrap();
    }
    // Every stabilizer generator must fix the SV state: <psi|P|psi> = sign.
    for gen in t.stabilizers() {
        // sign is ±1.0; unsigned Pauli stripped of coefficient for expectation_value
        let sign = gen.coefficient;
        let unsigned =
            PauliString::new(1.0, gen.terms.clone()).unwrap_or_else(|_| PauliString::identity(1.0));
        let ev = be.expectation_value(&sv, &unsigned).unwrap();
        assert!(
            (ev - sign).abs() < 1e-10,
            "generator {gen:?} not stabilized: <P> = {ev}, expected {sign}"
        );
    }
    // Sanity: the prepared+evolved state has exactly N independent
    // stabilizers (no degeneracy bug).
    assert_eq!(t.stabilizers().len(), N);
}

#[test]
fn native_gates_match_sv() {
    for g in [
        GateInstance::new(Gate::H, vec![1u32]),
        GateInstance::new(Gate::S, vec![1u32]),
        GateInstance::new(Gate::X, vec![1u32]),
        GateInstance::new(Gate::Y, vec![1u32]),
        GateInstance::new(Gate::Z, vec![1u32]),
        GateInstance::new(Gate::Cnot, vec![1u32, 3u32]),
    ] {
        assert_stabilized(&[g]);
    }
}

#[test]
fn composed_gates_match_sv() {
    for g in [
        GateInstance::new(Gate::Sdg, vec![1u32]),
        GateInstance::new(Gate::Cz, vec![0u32, 3u32]),
        GateInstance::new(Gate::Swap, vec![0u32, 3u32]),
        GateInstance::new(Gate::Iswap, vec![0u32, 3u32]),
        GateInstance::new(Gate::IswapDg, vec![0u32, 3u32]),
    ] {
        assert_stabilized(&[g]);
    }
}
