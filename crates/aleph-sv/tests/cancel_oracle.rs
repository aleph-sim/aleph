//! P1-12 oracle — a circuit with `CancelInversePairs` applied yields the
//! same full state vector as the original. Cancellation only ever deletes
//! a gate together with its exact inverse (an identity), so the entire
//! amplitude vector must match to 1e-12, not merely the measurement
//! marginal. Hand-built cases plus a proptest over random circuits guard
//! against false-positive removals.

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance};
use aleph_ir::passes::{CancelInversePairs, Pass};
use aleph_ir::{Circuit, Instruction};
use smallvec::smallvec;

const TOL: f64 = 1e-12;

/// Gate-only twin (drop Measure/Reset/Barrier) so `run` accepts it. The
/// cancellation cases here are unitary; this strips any terminal markers.
fn gate_only(c: &Circuit) -> Circuit {
    let mut out = Circuit::new(c.num_qubits(), c.num_clbits());
    for inst in c.instructions() {
        if let Instruction::Gate(g) = inst {
            out.add_gate(g.clone()).unwrap();
        }
    }
    out
}

fn amplitudes(c: &Circuit) -> Vec<Complex> {
    let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
    run(&mut backend, c).unwrap().amplitudes().to_vec()
}

fn assert_state_preserved(c: &Circuit) {
    let mut cancelled = c.clone();
    CancelInversePairs.run(&mut cancelled).unwrap();

    let before = amplitudes(&gate_only(c));
    let after = amplitudes(&gate_only(&cancelled));

    assert_eq!(before.len(), after.len(), "state dimension changed");
    for (k, (a, b)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            (a.re - b.re).abs() < TOL && (a.im - b.im).abs() < TOL,
            "amplitude[{k}] differs: before={a:?} after={b:?}"
        );
    }
}

#[test]
fn nested_cancellation_preserves_state() {
    // X H H X with a surrounding useful gate that survives.
    let mut c = Circuit::new(2, 0);
    c.ry(0.7, 0).unwrap();
    c.x(0).unwrap();
    c.h(0).unwrap();
    c.h(0).unwrap();
    c.x(0).unwrap();
    c.cnot(0, 1).unwrap();
    assert_state_preserved(&c);
}

#[test]
fn parametric_and_adjoint_pairs_preserve_state() {
    let mut c = Circuit::new(2, 0);
    c.h(0).unwrap();
    c.rz(0.41, 0).unwrap();
    c.rz(-0.41, 0).unwrap(); // cancels
    c.s(1).unwrap();
    c.sdg(1).unwrap(); // cancels
    c.cnot(0, 1).unwrap();
    assert_state_preserved(&c);
}

#[test]
fn two_qubit_and_controlled_pairs_preserve_state() {
    let mut c = Circuit::new(3, 0);
    c.h(0).unwrap();
    c.h(1).unwrap();
    c.cz(0, 1).unwrap();
    c.cz(0, 1).unwrap(); // cancels
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![2u32],
        smallvec![0u32, 1u32],
    ))
    .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![2u32],
        smallvec![1u32, 0u32],
    ))
    .unwrap(); // cancels (controls as a set)
    assert_state_preserved(&c);
}

use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    // For ANY random unitary circuit, cancelling inverse pairs must not
    // change the state vector. This is the real false-positive guard:
    // if the pass ever deletes a non-inverse pair, amplitudes diverge.
    //
    // 0 clbits → no Measure. `arb_op_emittable` can still emit Reset,
    // which `gate_only` drops (changing semantics), so filter Reset out
    // to keep every case a pure unitary whose full state is well-defined.
    #[test]
    fn random_circuit_state_preserved(
        c in arb_circuit_emittable(4, 0, 20)
            .prop_filter(
                "no reset",
                |c| c.instructions().iter().all(|i| !matches!(i, Instruction::Reset(_)))
            )
    ) {
        let mut cancelled = c.clone();
        CancelInversePairs.run(&mut cancelled).unwrap();

        let before = amplitudes(&gate_only(&c));
        let after = amplitudes(&gate_only(&cancelled));
        prop_assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            prop_assert!((a.re - b.re).abs() < TOL && (a.im - b.im).abs() < TOL);
        }
    }
}
