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

/// Shot budget for the distribution oracle. Sized so 5σ per-outcome
/// flake probability is ≤ 5.7e-7 (spec §6.2).
pub const DISTRIBUTION_SHOTS: u32 = 100_000;

/// Floor added to every per-outcome band so a forbidden outcome
/// (`p_exact = 0`) tolerates the rare integer round-off but rejects
/// a genuine bug producing several stray samples. Spec §6.2.
pub const DISTRIBUTION_FLOOR: f64 = 1e-6;

/// Sample 100 000 shots through `backend`, then assert the empirical
/// distribution matches `fixture.statevector.amplitudes` (per-outcome
/// `5σ + DISTRIBUTION_FLOOR` band, in probability units).
///
/// `B::State = aleph_sv::CpuState` is pinned because today only
/// `NaiveSvBackend` exists; the bound generalises when a second
/// backend lands (spec §6.4).
pub fn run_distribution_oracle<B>(
    backend: &mut B,
    fixture: &Fixture,
    qasm_source: &str,
) -> Result<(), OracleError>
where
    B: Backend<State = aleph_sv::CpuState>,
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
    let shots = backend.sample(&state, DISTRIBUTION_SHOTS)?;
    let dim = 1usize << fixture.num_qubits;
    let mut empirical = vec![0u64; dim];
    for s in &shots {
        empirical[*s as usize] += 1;
    }
    let exact: Vec<f64> = fixture
        .statevector
        .amplitudes
        .iter()
        .map(|&(re, im)| re * re + im * im)
        .collect();
    assert_distribution_close(
        &fixture.name,
        fixture.num_qubits,
        &empirical,
        &exact,
        DISTRIBUTION_SHOTS,
    );
    Ok(())
}

fn assert_distribution_close(
    name: &str,
    num_qubits: u32,
    empirical: &[u64],
    exact: &[f64],
    shots: u32,
) {
    let width = num_qubits as usize;
    let n_f = shots as f64;
    for (i, (&count, &p)) in empirical.iter().zip(exact.iter()).enumerate() {
        // Defensive: a Qiskit fixture should never carry a NaN or a
        // p outside [0, 1] here (re²+im² is non-negative for finite
        // inputs and ≤ 1 for a normalised state), but mirror
        // `assert_state_close`'s explicit guard so a pathological
        // reference is loud, not silent. Without this, the band
        // computation below would silently produce NaN for p < 0
        // (sqrt of a negative product) and `delta > NaN` would be
        // false — the same regression class P0-10 hardened the state
        // path against.
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            panic!(
                "oracle: {name} distribution non-finite or out-of-range reference\n  \
                 index {i}  basis |{i:0width$b}>\n  p_exact {p}",
                width = width,
            );
        }
        let p_emp = count as f64 / n_f;
        // Clamp the Bernoulli variance `p*(1-p)` at the product level,
        // not just `(1-p)`. The earlier shape — `p * (1.0 - p).max(0.0)`
        // — only saved the second factor; for p outside [0, 1] the
        // product itself can go negative, and `sqrt` of a negative is
        // NaN.  The explicit p-range guard above also rejects that
        // case, but defense-in-depth keeps the formula safe under any
        // future relaxation of the guard.
        let variance = (p * (1.0 - p)).max(0.0);
        let band = 5.0 * (variance / n_f).sqrt() + DISTRIBUTION_FLOOR;
        let delta = (p_emp - p).abs();
        if delta > band {
            panic!(
                "oracle: {name} distribution mismatch\n  \
                 basis  |{i:0width$b}>\n  \
                 exact  {p:.6e}\n  \
                 empir  {p_emp:.6e}   ({count} / {shots})\n  \
                 |Δ|    {delta:.3e}   >  band {band:.3e}   (5σ + {DISTRIBUTION_FLOOR:.0e})",
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

    const QASM_BELL: &str =
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n";

    #[test]
    fn distribution_oracle_passes_on_bell() {
        // |Φ+⟩ → P(00) = P(11) = 0.5; P(01) = P(10) = 0.
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        let fx = synth(
            "bell_phi_plus_test",
            2,
            vec![(inv, 0.0), (0.0, 0.0), (0.0, 0.0), (inv, 0.0)],
        );
        let mut b = NaiveSvBackend::with_seed(0);
        run_distribution_oracle(&mut b, &fx, QASM_BELL).unwrap();
    }

    #[test]
    #[should_panic(expected = "distribution mismatch")]
    fn distribution_oracle_detects_wrong_distribution() {
        // Fixture claims uniform-over-4; circuit actually produces |Φ+⟩.
        let q = 0.5_f64;
        let fx = synth(
            "bell_uniform_wrong",
            2,
            vec![
                (q.sqrt(), 0.0),
                (q.sqrt(), 0.0),
                (q.sqrt(), 0.0),
                (q.sqrt(), 0.0),
            ],
        );
        let mut b = NaiveSvBackend::with_seed(0);
        let _ = run_distribution_oracle(&mut b, &fx, QASM_BELL);
    }

    #[test]
    fn distribution_oracle_qubit_mismatch_is_harness_error() {
        // 1-qubit fixture, 2-qubit circuit.
        let fx = synth("wrong_qubits_dist", 1, vec![(1.0, 0.0), (0.0, 0.0)]);
        let mut b = NaiveSvBackend::with_seed(0);
        let err = run_distribution_oracle(&mut b, &fx, QASM_BELL).unwrap_err();
        match err {
            OracleError::QubitMismatch {
                fixture: 1,
                circuit: 2,
                ..
            } => {}
            other => panic!("expected QubitMismatch, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "non-finite or out-of-range reference")]
    fn distribution_oracle_rejects_negative_reference_p() {
        // Direct call into assert_distribution_close with a pathological
        // negative reference probability — would have computed
        // sqrt(p*(1-p)) on a negative product before the fix, producing
        // a NaN band that silently passes `delta > band`.
        assert_distribution_close("neg_p", 1, &[50_000, 50_000], &[-1e-9, 1.0 + 1e-9], 100_000);
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
