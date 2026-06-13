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

/// Per-amplitude tolerance for the single-precision (FP32) state oracle
/// (BACKLOG.md §P2-08). The trusted fixtures are exact (FP64) Qiskit-Aer
/// statevectors; comparing an `Fp32SvBackend` run against that exact
/// reference at 1e-5 bounds the f32 accumulation error against the true
/// answer — strictly stronger than the §P2-08 acceptance criterion
/// "equivalence vs Qiskit Aer single-precision within 1e-5", and needs
/// no new fixtures. The f64 oracle path stays at `STATE_TOLERANCE` (1e-10).
pub const FP32_STATE_TOLERANCE: f64 = 1e-5;

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
    run_state_oracle_with_tol(backend, fixture, qasm_source, STATE_TOLERANCE)
}

/// Tolerance-parameterized variant of [`run_state_oracle`]. Identical
/// behavior, but asserts each amplitude matches the fixture within `tol`
/// instead of the hard-coded `STATE_TOLERANCE`. Used by the FP32 oracle
/// (`tol = FP32_STATE_TOLERANCE = 1e-5`); the f64 path calls in with
/// `STATE_TOLERANCE` (1e-10) and is byte-identical to the original.
pub fn run_state_oracle_with_tol<B>(
    backend: &mut B,
    fixture: &Fixture,
    qasm_source: &str,
    tol: f64,
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
    assert_state_close(&fixture.name, fixture.num_qubits, &actual, expected, tol);
    Ok(())
}

fn assert_state_close(
    name: &str,
    num_qubits: u32,
    actual: &[Complex],
    expected: &[(f64, f64)],
    tol: f64,
) {
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
        if delta > tol {
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
                tol,
                width = width,
            );
        }
    }
}

/// Shot budget for the distribution oracle. Sized so 5σ per-outcome
/// flake probability is ≤ 5.7e-7 (spec §6.2).
pub const DISTRIBUTION_SHOTS: u32 = 100_000;

/// Floor added to every per-outcome band. At `DISTRIBUTION_SHOTS =
/// 100_000` shots, one stray sample on a forbidden outcome
/// (`p_exact = 0`) already exceeds this floor (1/100_000 = 1e-5 >
/// 1e-6). The floor is therefore a tightness guarantee, not a
/// tolerance: any non-zero count on a forbidden outcome fails the
/// oracle. Spec §6.2.
pub const DISTRIBUTION_FLOOR: f64 = 1e-6;

/// Sample 100 000 shots through `backend`, then assert the empirical
/// distribution matches `fixture.statevector.amplitudes` (per-outcome
/// `5σ + DISTRIBUTION_FLOOR` band, in probability units).
///
/// Backend bound is `Backend` alone: the distribution oracle never
/// pulls live amplitudes off `state` (it samples and compares against
/// the trusted Qiskit reference), so non-AoS backends — including
/// future Phase-2 MPS / stabilizer backends whose state cannot
/// cheaply materialise a full amplitude vector — flow through this
/// function natively. The AoS pin from P0-10 spec §6.4 lifted with
/// P1-01 when SoA landed as the second backend.
pub fn run_distribution_oracle<B>(
    backend: &mut B,
    fixture: &Fixture,
    qasm_source: &str,
) -> Result<(), OracleError>
where
    B: Backend,
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
    // Mirror load_fixture's TooManyQubits guard. load_fixture is the
    // only constructor today, but `Fixture`'s fields are `pub`, so a
    // direct struct-literal construction in a future test/helper
    // could ship a `num_qubits >= usize::BITS` value past load — this
    // line would then shift-overflow and either panic in debug or
    // yield dim=0 in release (then panic on the first empirical[idx]
    // write).
    let dim = 1usize
        .checked_shl(fixture.num_qubits)
        .ok_or_else(|| OracleError::TooManyQubits {
            name: fixture.name.clone(),
            num_qubits: fixture.num_qubits,
            limit: usize::BITS,
        })?;
    let mut empirical = vec![0u64; dim];
    for s in &shots {
        let idx = *s as usize;
        // A well-behaved Backend::sample returns indices in [0, dim).
        // A regression (e.g., a future alias-table bug, or a non-Naive
        // backend that mis-computes its sample dimension) would
        // otherwise panic with a raw `index out of bounds` here —
        // surface a structured oracle panic naming the fixture and
        // the offending index instead, mirroring assert_state_close's
        // structured failure messages.
        if idx >= dim {
            panic!(
                "oracle: {name} sample out of range\n  \
                 sample idx {idx}  >=  2^{nq} = {dim}\n  \
                 (basis space has {dim} outcomes; backend returned an out-of-range index)",
                name = fixture.name,
                nq = fixture.num_qubits,
            );
        }
        empirical[idx] += 1;
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

/// Assert an empirical sample distribution matches an exact one within a
/// calibrated 5σ Bernoulli band (+ a small absolute floor for near-zero
/// probabilities). `empirical[i]` is the shot count for basis state `i`,
/// `exact[i]` its target probability; `shots` is the total drawn. Panics with
/// a structured, basis-labeled message on the first cell outside the band.
///
/// Shared so backends compare sampling against any exact distribution (not just
/// a Qiskit fixture) instead of re-rolling an ad-hoc ±ε check (P3-16).
pub fn assert_distribution_close(
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
        // inputs and bounded by the normalised-state sum), but mirror
        // `assert_state_close`'s explicit guard so a pathological
        // reference is loud, not silent. Without the NaN reject the
        // band computation below would silently produce NaN for p < 0
        // (sqrt of a negative product) and `delta > NaN` would be
        // false — the same regression class P0-10 hardened the state
        // path against.
        //
        // The upper bound is `1.0 + STATE_TOLERANCE` rather than `1.0`
        // exactly: `re*re + im*im` can round UP by 1 ulp on a finite
        // input, and a fixture whose dominant amplitude lands at p ≈ 1
        // (a product/near-product state) would otherwise hard-panic
        // here. The downstream `(p * (1.0 - p)).max(0.0)` clamp keeps
        // the variance well-defined for `p` slightly above 1.
        if !p.is_finite() || !(0.0..=1.0 + STATE_TOLERANCE).contains(&p) {
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
            STATE_TOLERANCE,
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
            STATE_TOLERANCE,
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
    fn distribution_oracle_passes_on_bell_with_soa_backend() {
        // Same fixture as `distribution_oracle_passes_on_bell` — drives
        // the relaxed B::State: HasAmplitudes bound through SoaSvBackend.
        use aleph_sv::SoaSvBackend;
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        let fx = synth(
            "bell_phi_plus_test_soa",
            2,
            vec![(inv, 0.0), (0.0, 0.0), (0.0, 0.0), (inv, 0.0)],
        );
        let mut b = SoaSvBackend::with_seed(0);
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
    fn distribution_oracle_rejects_oversized_num_qubits() {
        // Direct-struct Fixture construction with num_qubits >= usize::BITS
        // must surface as TooManyQubits, not as a shift-overflow panic.
        let mut fx = synth("oversized", 1, vec![(1.0, 0.0), (0.0, 0.0)]);
        fx.num_qubits = 128;
        let mut b = NaiveSvBackend::with_seed(0);
        let err = run_distribution_oracle(&mut b, &fx, QASM_BELL).unwrap_err();
        match err {
            OracleError::TooManyQubits {
                num_qubits: 128, ..
            } => {}
            // QubitMismatch is also acceptable — circuit has 2 qubits,
            // fixture says 128, so the earlier check might fire first.
            OracleError::QubitMismatch {
                fixture: 128,
                circuit: 2,
                ..
            } => {}
            other => panic!("expected TooManyQubits or QubitMismatch, got {other:?}"),
        }
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
    fn distribution_oracle_accepts_p_within_one_ulp_above_one() {
        // A degenerate single-outcome distribution where p_exact rounds
        // up by 1 ulp due to fixture FP rounding. The empirical count
        // matches the rounded reference, and the tightness band
        // tolerates the residual. Pre-G1 this would have hard-panicked.
        let p = 1.0_f64 + f64::EPSILON; // ~2.22e-16 above 1
        assert_distribution_close("p_one_plus_ulp", 1, &[100_000, 0], &[p, 0.0], 100_000);
    }

    #[test]
    #[should_panic(expected = "non-finite or out-of-range reference")]
    fn distribution_oracle_rejects_p_well_above_one() {
        // STATE_TOLERANCE is 1e-10; anything materially above 1 is a
        // genuinely malformed reference and must still hard-panic.
        assert_distribution_close(
            "p_well_above_one",
            1,
            &[100_000, 0],
            &[1.0 + 1e-6, 0.0],
            100_000,
        );
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
            STATE_TOLERANCE,
        );
    }
}
