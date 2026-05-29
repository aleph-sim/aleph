//! P1-11 oracle — a DCE'd circuit yields the same measurement distribution
//! as the original. We compare the marginal probability distribution over
//! the measured qubits, computed exactly from `NaiveSvBackend` amplitudes.
//!
//! Test circuits are unitary + terminal measurements (no Reset), so the
//! marginal of the final state over the measured qubits IS the measurement
//! distribution. Reset-sever and barrier behaviour are covered by the
//! structural unit tests in `dce.rs`.

use aleph_backend::run;
use aleph_core::Complex;
use aleph_ir::passes::{DeadCodeElim, Pass};
use aleph_ir::{Circuit, Instruction};

const TOL: f64 = 1e-12;

/// Build a gate-only twin (drop Measure/Reset/Barrier) so `run` accepts it.
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

/// Marginal probability over `measured` qubits: index the outcome by the
/// measured-qubit bit values (packed in the given order), summing |amp|².
fn marginal(state: &[Complex], measured: &[u32]) -> Vec<f64> {
    let mut out = vec![0.0f64; 1 << measured.len()];
    for (i, amp) in state.iter().enumerate() {
        let mut key = 0usize;
        for (bit, &q) in measured.iter().enumerate() {
            if (i >> q) & 1 == 1 {
                key |= 1 << bit;
            }
        }
        out[key] += amp.norm_sqr();
    }
    out
}

/// Assert that running `DeadCodeElim` preserves the marginal distribution
/// over `measured` qubits. `build` populates a circuit whose terminal
/// measurements are exactly `measured` (in any order).
fn assert_dce_preserves_marginal(num_qubits: u32, measured: &[u32], build: impl Fn(&mut Circuit)) {
    let mut original = Circuit::new(num_qubits, measured.len() as u32);
    build(&mut original);

    let mut dced = original.clone();
    DeadCodeElim.run(&mut dced).unwrap();

    let m_orig = marginal(&amplitudes(&gate_only(&original)), measured);
    let m_dced = marginal(&amplitudes(&gate_only(&dced)), measured);

    // Both marginals are `1 << measured.len()` long by construction, so a
    // length check would be tautological; the per-outcome comparison below
    // is the real assertion.
    for (k, (a, b)) in m_orig.iter().zip(m_dced.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL,
            "marginal[{k}] differs: orig={a}, dced={b}"
        );
    }
}

#[test]
fn dead_ancilla_branch_preserves_data_marginal() {
    // q0,q1 data (measured); q2 dead ancilla.
    assert_dce_preserves_marginal(3, &[0, 1], |c| {
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.h(2).unwrap();
        c.x(2).unwrap();
        c.measure(0, 0).unwrap();
        c.measure(1, 1).unwrap();
    });
}

#[test]
fn entangled_unmeasured_qubit_kept_preserves_marginal() {
    // q2 unmeasured but entangled with measured q0 → its gates are kept;
    // marginal over {0} must be unchanged either way.
    assert_dce_preserves_marginal(3, &[0], |c| {
        c.h(0).unwrap();
        c.cnot(0, 2).unwrap();
        c.rz(0.7, 2).unwrap();
        c.measure(0, 0).unwrap();
    });
}

#[test]
fn partial_measurement_preserves_marginal() {
    // 4 qubits, measure {1,3}; gates on {0,2} that don't reach them are dead.
    assert_dce_preserves_marginal(4, &[1, 3], |c| {
        c.h(1).unwrap();
        c.h(3).unwrap();
        c.cnot(1, 3).unwrap();
        c.h(0).unwrap(); // dead
        c.x(2).unwrap(); // dead
        c.measure(1, 0).unwrap();
        c.measure(3, 1).unwrap();
    });
}
