//! Core harness entry point: parse a QASM source, run it through a
//! `Backend`, compare the final amplitudes against a `Fixture`.
//!
//! Tolerance is `STATE_TOLERANCE = 1e-10` on the complex magnitude
//! `|a_ours - a_fixture|`. The first amplitude exceeding tolerance
//! panics with a structured message containing the basis-state
//! label and the deviation; later amplitudes for that fixture are
//! not inspected.

use aleph_backend::{run, Backend};
use aleph_core::Complex;

use crate::error::OracleError;
use crate::fixture::Fixture;
use crate::state::HasAmplitudes;

/// Maximum per-amplitude complex deviation (matches BACKLOG.md §P0-10).
pub const STATE_TOLERANCE: f64 = 1e-10;

/// Parse QASM, run it through `backend`, assert the final state
/// matches `fixture.statevector.amplitudes` to within
/// `STATE_TOLERANCE`. Returns `Err(OracleError)` for harness-level
/// failures (parse error, dim mismatch, …). Correctness failures
/// panic so the offending fixture name shows up in cargo's test
/// output.
pub fn run_state_oracle<B>(
    backend: &mut B,
    fixture: &Fixture,
    qasm_source: &str,
) -> Result<(), OracleError>
where
    B: Backend,
    B::State: HasAmplitudes,
{
    let circuit = aleph_parser::parse(qasm_source)?;
    if circuit.num_qubits() != fixture.num_qubits {
        return Err(OracleError::QubitMismatch {
            name: fixture.name.clone(),
            fixture: fixture.num_qubits,
            circuit: circuit.num_qubits(),
        });
    }
    let state = run(backend, &circuit)?;
    let actual = state.amplitudes();
    let expected = &fixture.statevector.amplitudes;
    if actual.len() != expected.len() {
        return Err(OracleError::DimensionMismatch {
            name: fixture.name.clone(),
            fixture: expected.len(),
            state: actual.len(),
        });
    }
    assert_state_close(&fixture.name, fixture.num_qubits, actual, expected);
    Ok(())
}

fn assert_state_close(name: &str, num_qubits: u32, actual: &[Complex], expected: &[(f64, f64)]) {
    let width = num_qubits as usize;
    for (i, (a, &(er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
        // Explicit NaN/infinity guard. Without it, a backend that
        // produces a NaN amplitude (the exact regression class P0-09
        // hit twice) would silently pass: NaN comparisons against any
        // tolerance return false, so `delta > STATE_TOLERANCE` is
        // false and the loop completes. The check must also reject
        // expected NaNs so a corrupt fixture is loud, not silent.
        if !a.re.is_finite() || !a.im.is_finite() || !er.is_finite() || !ei.is_finite() {
            panic!(
                "oracle: {name} non-finite amplitude\n  \
                 index {i}  basis |{i:0width$b}>\n  \
                 ours      ({:.16e}, {:.16e})\n  \
                 qiskit    ({:.16e}, {:.16e})",
                a.re,
                a.im,
                er,
                ei,
                width = width,
            );
        }
        let dre = a.re - er;
        let dim = a.im - ei;
        let delta = (dre * dre + dim * dim).sqrt();
        if delta > STATE_TOLERANCE {
            panic!(
                "oracle: {name} amplitude mismatch\n  \
                 index {i}  basis |{i:0width$b}>\n  \
                 ours      ({:.16e}, {:.16e})\n  \
                 qiskit    ({:.16e}, {:.16e})\n  \
                 |Δ|       {:.3e}   >  tol {:.3e}",
                a.re,
                a.im,
                er,
                ei,
                delta,
                STATE_TOLERANCE,
                width = width,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Fixture, StateVectorFixture};
    use aleph_sv::NaiveSvBackend;
    use std::collections::BTreeMap;

    fn synth(name: &str, n: u32, amps: Vec<(f64, f64)>) -> Fixture {
        Fixture {
            schema_version: 1,
            name: name.into(),
            num_qubits: n,
            qasm_path: format!("circuits/{name}.qasm"),
            qiskit_version: "test".into(),
            aer_version: "test".into(),
            generated_at: "1970-01-01T00:00:00Z".into(),
            shots: 0,
            rng_seed: 0,
            statevector: StateVectorFixture {
                endianness: "little".into(),
                amplitudes: amps,
            },
            counts: BTreeMap::new(),
        }
    }

    const QASM_H: &str = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\nh q[0];\n";
    const QASM_IDENTITY_1Q: &str = "OPENQASM 3.0;\nqubit[1] q;\n";

    #[test]
    fn h_on_zero_matches_synthetic_fixture() {
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        let fx = synth("kernel_h", 1, vec![(inv, 0.0), (inv, 0.0)]);
        let mut b = NaiveSvBackend::with_seed(0);
        run_state_oracle(&mut b, &fx, QASM_H).unwrap();
    }

    #[test]
    #[should_panic(expected = "amplitude mismatch")]
    fn detects_amplitude_disagreement() {
        // Fixture claims |0⟩ but the circuit lands on |+⟩.
        let fx = synth("kernel_h_wrong", 1, vec![(1.0, 0.0), (0.0, 0.0)]);
        let mut b = NaiveSvBackend::with_seed(0);
        let _ = run_state_oracle(&mut b, &fx, QASM_H);
    }

    #[test]
    fn dimension_mismatch_is_harness_error() {
        // Fixture has 4 amps but the circuit is 1 qubit (2 amps).
        // The qubit-count check fires first.
        let fx = synth(
            "wrong_qubits",
            2,
            vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        );
        let mut b = NaiveSvBackend::with_seed(0);
        let err = run_state_oracle(&mut b, &fx, QASM_IDENTITY_1Q).unwrap_err();
        match err {
            OracleError::QubitMismatch {
                fixture: 2,
                circuit: 1,
                ..
            } => {}
            other => panic!("expected QubitMismatch, got {other:?}"),
        }
    }

    #[test]
    fn passes_within_tolerance_at_1e_minus_11() {
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        // Offset both amplitudes by 1e-11 (< STATE_TOLERANCE).
        let fx = synth(
            "kernel_h_within_tol",
            1,
            vec![(inv + 1e-11, 0.0), (inv - 1e-11, 0.0)],
        );
        let mut b = NaiveSvBackend::with_seed(0);
        run_state_oracle(&mut b, &fx, QASM_H).unwrap();
    }

    #[test]
    #[should_panic(expected = "non-finite amplitude")]
    fn nan_actual_amplitude_is_not_silent() {
        // The exact failure mode P0-09 hit: a backend produces NaN.
        // Without the explicit guard, NaN > 1e-10 is false and the
        // oracle reports success.
        let nan = f64::NAN;
        assert_state_close(
            "nan_in_state",
            1,
            &[Complex::new(nan, 0.0), Complex::new(0.0, 0.0)],
            &[(0.0, 0.0), (0.0, 0.0)],
        );
    }

    #[test]
    #[should_panic(expected = "non-finite amplitude")]
    fn nan_expected_amplitude_is_not_silent() {
        // Symmetric guard: a corrupted fixture containing NaN should
        // fail loudly, not silently match anything.
        let nan = f64::NAN;
        assert_state_close(
            "nan_in_fixture",
            1,
            &[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            &[(nan, 0.0), (0.0, 0.0)],
        );
    }

    #[test]
    #[should_panic(expected = "non-finite amplitude")]
    fn infinite_amplitude_is_not_silent() {
        // +Inf amplitudes are equally pathological — neither side of
        // the oracle should ever produce them in a normalized state.
        assert_state_close(
            "inf_in_state",
            1,
            &[Complex::new(f64::INFINITY, 0.0), Complex::new(0.0, 0.0)],
            &[(0.0, 0.0), (0.0, 0.0)],
        );
    }
}
