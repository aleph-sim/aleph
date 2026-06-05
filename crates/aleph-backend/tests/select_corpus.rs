//! P3-07 acceptance corpus: a representative circuit per category selects the
//! expected backend. Exercises the full `analyze` + decision-rule path on real
//! circuits (the unit tests in `select.rs` drive the rule from synthetic
//! features; this drives it from actual instructions).

use aleph_backend::{select_backend, BackendKind};
use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::Circuit;

/// GHZ on 6 qubits via H + nearest-neighbor CNOTs — all Clifford.
#[test]
fn clifford_ghz_selects_stabilizer() {
    let mut c = Circuit::new(6, 0);
    c.add_gate(GateInstance::new(Gate::H, vec![0u32])).unwrap();
    for q in 0u32..5 {
        c.add_gate(GateInstance::new(Gate::Cnot, vec![q, q + 1]))
            .unwrap();
    }
    assert_eq!(select_backend(&c), BackendKind::Stabilizer);
}

/// Small (n <= 28) non-Clifford circuit — exact state vector.
#[test]
fn small_nonclifford_selects_statevector() {
    let mut c = Circuit::new(10, 0);
    c.add_gate(GateInstance::new(Gate::T, vec![0u32])).unwrap();
    for q in 0u32..9 {
        c.add_gate(GateInstance::new(Gate::Cnot, vec![q, q + 1]))
            .unwrap();
    }
    assert_eq!(select_backend(&c), BackendKind::Statevector);
}

/// 30-qubit nearest-neighbor shallow non-Clifford brickwork — MPS.
#[test]
fn large_nn_shallow_selects_mps() {
    let mut c = Circuit::new(30, 0);
    // A few shallow layers of nearest-neighbor gates plus a non-Clifford
    // rotation to defeat the Clifford rule.
    for _ in 0..4 {
        for q in (0u32..29).step_by(2) {
            c.add_gate(GateInstance::new(Gate::Cnot, vec![q, q + 1]))
                .unwrap();
        }
    }
    c.add_gate(GateInstance::new(
        Gate::Rz(Param::Concrete(0.3)),
        vec![0u32],
    ))
    .unwrap();
    assert_eq!(select_backend(&c), BackendKind::Mps);
}

/// 30-qubit non-Clifford circuit with a long-range gate — state vector
/// (the CLI layer additionally warns it is too large for exact memory).
#[test]
fn large_longrange_selects_statevector() {
    let mut c = Circuit::new(30, 0);
    c.add_gate(GateInstance::new(
        Gate::Rz(Param::Concrete(0.3)),
        vec![0u32],
    ))
    .unwrap();
    c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 29u32]))
        .unwrap();
    assert_eq!(select_backend(&c), BackendKind::Statevector);
}
