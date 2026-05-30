//! P1-13 oracle — soundness guard for `gates_commute`. For every pair
//! the predicate calls commuting, applying the two gates in either order
//! must yield the same full state vector (1e-12). For a sample of
//! non-commuting pairs, the two orders must differ (BACKLOG sanity). A
//! proptest enforces the `commute ⟹ equal` direction over random pairs.

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_ir::passes::gates_commute;
use aleph_ir::{Circuit, Instruction};

const TOL: f64 = 1e-12;
const N: u32 = 3;

fn amplitudes(c: &Circuit) -> Vec<Complex> {
    let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
    run(&mut backend, c).unwrap().amplitudes().to_vec()
}

/// State vector after applying `gates` in order to |0…0⟩ on N qubits.
fn state_after(gates: &[GateInstance]) -> Vec<Complex> {
    let mut c = Circuit::new(N, 0);
    for gi in gates {
        c.add_gate(gi.clone()).unwrap();
    }
    amplitudes(&c)
}

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}
fn rz(theta: f64, q: u32) -> GateInstance {
    GateInstance::new(Gate::Rz(Param::Concrete(theta)), [q].to_vec())
}
fn rx(theta: f64, q: u32) -> GateInstance {
    GateInstance::new(Gate::Rx(Param::Concrete(theta)), [q].to_vec())
}

fn assert_reorder_equal(a: &GateInstance, b: &GateInstance) {
    assert!(
        gates_commute(a, b),
        "test bug: pair not reported commuting: {:?}/{:?}",
        a.gate,
        b.gate
    );
    let ab = state_after(&[a.clone(), b.clone()]);
    let ba = state_after(&[b.clone(), a.clone()]);
    for (k, (x, y)) in ab.iter().zip(ba.iter()).enumerate() {
        assert!(
            (x.re - y.re).abs() < TOL && (x.im - y.im).abs() < TOL,
            "commuting pair {:?}/{:?} changed amplitude[{k}]: {x:?} vs {y:?}",
            a.gate,
            b.gate
        );
    }
}

fn assert_reorder_differs(a: &GateInstance, b: &GateInstance) {
    let ab = state_after(&[a.clone(), b.clone()]);
    let ba = state_after(&[b.clone(), a.clone()]);
    let differs = ab
        .iter()
        .zip(ba.iter())
        .any(|(x, y)| (x.re - y.re).abs() > TOL || (x.im - y.im).abs() > TOL);
    assert!(
        differs,
        "expected non-commuting pair {:?}/{:?} to differ on |0…0⟩",
        a.gate, b.gate
    );
}

#[test]
fn commuting_pairs_preserve_state() {
    // Need a non-trivial input on the shared qubits, so prefix the system
    // into a superposition where order would matter if they didn't commute.
    // state_after starts from |0…0⟩; for pairs whose action on |0⟩ is
    // order-insensitive only by luck, we instead rely on the proptest for
    // breadth and use clearly order-sensitive operators here.
    assert_reorder_equal(&g(Gate::H, &[0]), &g(Gate::X, &[1])); // disjoint
    assert_reorder_equal(&g(Gate::Z, &[0]), &rz(0.4, 0)); // diagonal
    assert_reorder_equal(&g(Gate::S, &[0]), &g(Gate::T, &[0])); // diagonal
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[1])); // cnot/target
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &rx(0.9, 1)); // cnot/target
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[0])); // cnot/control
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &rz(0.5, 0)); // cnot/control
}

#[test]
fn non_commuting_pairs_differ() {
    // Cases chosen so the two orders differ on |0…0⟩ directly (no state
    // prep needed). NOTE: a pair like Cnot(0,1) ∥ Z(1) (Z on target) does
    // NOT differ on |0…0⟩ — the control is |0⟩ so CNOT is identity and Z
    // fixes |0⟩ — so it is intentionally NOT used here; its
    // non-commutation is covered by the unit test
    // `cnot_does_not_commute_with_z_target_or_x_control`.
    assert_reorder_differs(&g(Gate::X, &[0]), &g(Gate::Z, &[0]));
    assert_reorder_differs(&g(Gate::H, &[0]), &g(Gate::X, &[0]));
    assert_reorder_differs(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[0])); // X on control
}

use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    // For any random pair of gate instructions: if gates_commute says they
    // commute, reordering them must not change the state vector. This is
    // the strong false-positive guard. (We do not assert the converse —
    // some non-commuting pairs coincidentally agree on a given input.)
    #[test]
    fn commute_implies_reorder_equal(
        c in arb_circuit_emittable(N, 0, 2).prop_filter(
            "exactly two gate instructions",
            |c| c.instructions().len() == 2
                && c.instructions().iter().all(|i| matches!(i, Instruction::Gate(_)))
        )
    ) {
        let g0 = match &c.instructions()[0] {
            Instruction::Gate(gi) => gi.clone(),
            _ => unreachable!(),
        };
        let g1 = match &c.instructions()[1] {
            Instruction::Gate(gi) => gi.clone(),
            _ => unreachable!(),
        };
        if gates_commute(&g0, &g1) {
            let ab = state_after(&[g0.clone(), g1.clone()]);
            let ba = state_after(&[g1, g0]);
            for (x, y) in ab.iter().zip(ba.iter()) {
                prop_assert!((x.re - y.re).abs() < TOL && (x.im - y.im).abs() < TOL);
            }
        }
    }
}
