//! P4-02 acceptance test: optimal-iteration Grover converges. For each tested n
//! the marked basis state |0...01> (amplitude index 1 in aleph's MSB-qubit
//! ordering, ADR 0004) must reach probability > 0.9 AND be the most-probable
//! outcome (we amplified the *right* state, not merely *some* state). Reads the
//! committed corpus QASM — the single source of truth shared with Aer — and runs
//! the verbatim `run` path on NaiveSvBackend.
//!
//! CI runs n=4 and n=8 (sub-second). n=12 (264k gates, ~28s debug) and n=16
//! (2.26M gates) are `#[ignore]`d and exercised on the nightly ignored-tests
//! schedule — per CLAUDE.md, oracle tests over ~30s belong on nightly, and a
//! debug `cargo test` of n=12 brushes that limit on slower CI hardware.

use aleph_backend::run;
use aleph_sv::NaiveSvBackend;
use std::path::PathBuf;

/// round(pi/4 * sqrt(2^n)); mirrors run.py::grover_optimal_iters so the corpus
/// filename matches. n in {4,8,12,16} -> {3,13,50,201}.
fn optimal_iters(n: u32) -> u32 {
    (std::f64::consts::PI / 4.0 * (2f64.powi(n as i32)).sqrt()).round() as u32
}

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().expect("workspace root").join(format!(
        "scripts/qiskit-baseline/circuits/grover_n{n}_iters{}.qasm",
        optimal_iters(n)
    ))
}

fn assert_converges(n: u32) {
    let path = corpus_path(n);
    // n=16 (34 MB, 2.26M gates) is generated on demand, not committed (see the
    // circuits/.gitignore). If it is absent on this host, skip with a clear hint
    // rather than failing — this only affects the #[ignore]d nightly path.
    if !path.exists() {
        eprintln!(
            "SKIP grover_n{n}: corpus {} not present; generate it with \
             `python scripts/qiskit-baseline/run.py --gen-only`",
            path.display()
        );
        return;
    }
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse grover_n{n}: {e:?}"));
    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, &circuit).expect("simulate grover");
    let amps = state.amplitudes();

    let p_marked = amps[1].norm_sqr();
    assert!(
        p_marked > 0.9,
        "grover_n{n}: marked-state probability {p_marked:.4} is not > 0.9"
    );

    let (argmax, _) = amps
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm_sqr().total_cmp(&b.1.norm_sqr()))
        .expect("non-empty state");
    assert_eq!(
        argmax, 1,
        "grover_n{n}: most-probable index {argmax}, expected 1"
    );
}

#[test]
fn grover_n4_converges() {
    assert_converges(4);
}

#[test]
fn grover_n8_converges() {
    assert_converges(8);
}

/// n=12 (264k gates) runs ~28s in a debug build — brushing the CLAUDE.md 30s
/// limit on slower CI hardware — so it joins n=16 on the nightly ignored-tests
/// schedule. n=4 and n=8 keep convergence covered in fast default CI.
#[test]
#[ignore = "n=12: 264k gates ~28s debug, run on the nightly ignored-tests schedule"]
fn grover_n12_converges() {
    assert_converges(12);
}

/// n=16 is a 2.26M-gate circuit (~seconds single-thread); per CLAUDE.md it is
/// #[ignore]d for normal CI and exercised on the nightly ignored-tests run.
#[test]
#[ignore = "n=16: 2.26M gates, run on the nightly ignored-tests schedule"]
fn grover_n16_converges() {
    assert_converges(16);
}
