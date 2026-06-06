//! P4-03 acceptance test: QPE recovers a known eigenphase exactly. The unitary
//! is U = P(2*pi*phi) with the all-ones eigenphase phi = (2^m - 1)/2^m
//! (m = n-1 counting qubits), so QPE collapses to the all-ones basis state
//! |1...1> = amplitude index 2^n - 1, regardless of qubit ordering (ADR 0004)
//! or the inverse-QFT swap convention (all-ones is bit-reversal-invariant). We
//! assert that index carries ~all the probability AND is the most-probable
//! outcome. Reads the committed corpus QASM (shared with Aer) via verbatim `run`.
//!
//! n=10/15/20 run in fast CI (<~1s). n=25 (512 MiB, tens of seconds verbatim
//! single-thread) is #[ignore]d for the nightly ignored-tests schedule, per
//! CLAUDE.md's 30s rule.

use aleph_backend::{run, run_optimized};
use aleph_core::Complex;
use aleph_sv::NaiveSvBackend;
use std::path::PathBuf;

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("workspace root")
        .join(format!("scripts/qiskit-baseline/circuits/qpe_n{n}.qasm"))
}

/// Assert the simulated state collapsed to the all-ones basis state |1...1>
/// (index 2^n - 1): it carries ~all the probability AND is the argmax.
fn assert_all_ones(amps: &[Complex], n: u32, path: &str) {
    let marked = (1usize << n) - 1;
    let p_marked = amps[marked].norm_sqr();
    assert!(
        p_marked > 0.999,
        "qpe_n{n} [{path}]: all-ones probability {p_marked:.6} is not > 0.999"
    );

    let (argmax, _) = amps
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm_sqr().total_cmp(&b.1.norm_sqr()))
        .expect("non-empty state");
    assert_eq!(
        argmax, marked,
        "qpe_n{n} [{path}]: most-probable index {argmax}, expected {marked}"
    );
}

fn assert_recovers_phase(n: u32) {
    let path = corpus_path(n);
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse qpe_n{n}: {e:?}"));

    // Verbatim path (the oracle the corpus is built against).
    let mut raw = NaiveSvBackend::with_seed(0);
    let state = run(&mut raw, &circuit).expect("simulate qpe (run)");
    assert_all_ones(state.amplitudes(), n, "run");

    // Optimized path — the one the phase4_qpe bench actually times. The QPE
    // `cp` ladder + inverse QFT is exactly what the fusion/diagonal passes
    // rewrite, so we must confirm they preserve the exact eigenphase recovery;
    // otherwise the bench could be timing a silently-wrong computation.
    let mut opt = NaiveSvBackend::with_seed(0);
    let state = run_optimized(&mut opt, &circuit).expect("simulate qpe (run_optimized)");
    assert_all_ones(state.amplitudes(), n, "run_optimized");
}

#[test]
fn qpe_n10_recovers_phase() {
    assert_recovers_phase(10);
}

#[test]
fn qpe_n15_recovers_phase() {
    assert_recovers_phase(15);
}

#[test]
fn qpe_n20_recovers_phase() {
    assert_recovers_phase(20);
}

/// n=25 allocates a 512 MiB state vector and runs tens of seconds verbatim
/// single-thread — over the CLAUDE.md 30s limit — so it joins the nightly
/// ignored-tests schedule. n=10/15/20 keep recovery covered in fast CI.
#[test]
#[ignore = "n=25: 512 MiB state, tens of s verbatim single-thread; nightly ignored-tests schedule"]
fn qpe_n25_recovers_phase() {
    assert_recovers_phase(25);
}
