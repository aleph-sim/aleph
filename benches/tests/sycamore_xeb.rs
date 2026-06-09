//! P4-06 acceptance test for the Sycamore-style random circuit. No structural
//! oracle exists for a random circuit, so we gate on three properties over the
//! committed corpus (the SAME QASM Aer runs):
//!
//!  1. `run` ≡ `run_optimized` to 1e-12 — the strong internal oracle. The
//!     fusion / diagonal-fusion / FuseKq passes rewrite the √X/√Y/√W runs and
//!     the CZ brick-wall, so this proves they preserve the exact state on the
//!     path the bench actually times (the P4-03 lesson: oracle must cover
//!     `run_optimized`, not just `run`).
//!  2. Normalization Σp = 1 (1e-10) — unitarity sanity.
//!  3. Linear XEB in a sanity band around 1 (noiseless Porter–Thomas), and
//!     well above 0 (the uniform/depolarized value) — the AC's "XEB ≈ 1".
//!
//! n=16 runs in fast CI (well under the CLAUDE.md 30s budget — both `run` and
//! `run_optimized` over a 64 Ki-amplitude state take ~seconds). n=20 (16 MiB,
//! ~35s both-paths) and n=24 (256 MiB) are #[ignore]d to the nightly schedule.
//!
//! `sycamore_n16_d20.qasm` is a frozen test fixture: it is built by the same
//! `run.py build_sycamore` path as the benchmark corpus (n=20/24/28/30) but is
//! intentionally NOT in `FAMILY_SIZES`, so `--gen-only` leaves it untouched —
//! the same convention as the legacy frozen `grover_n{15..25}_iters5` fixtures.
//! The test checks simulator correctness, not corpus-vs-algorithm identity, so
//! it stays valid even if `build_sycamore` later changes.

use aleph_backend::{run, run_optimized};
use aleph_benches::linear_xeb;
use aleph_sv::NaiveSvBackend;
use std::path::PathBuf;

const DEPTH: u32 = 20;

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().expect("workspace root").join(format!(
        "scripts/qiskit-baseline/circuits/sycamore_n{n}_d{DEPTH}.qasm"
    ))
}

fn check(n: u32) {
    let path = corpus_path(n);
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let circuit =
        aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse sycamore_n{n}: {e:?}"));

    let mut raw = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
    let s_raw = run(&mut raw, &circuit).expect("simulate (run)");
    let mut opt = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
    let s_opt = run_optimized(&mut opt, &circuit).expect("simulate (run_optimized)");

    let a = s_raw.amplitudes();
    let b = s_opt.amplitudes();
    assert_eq!(a.len(), b.len(), "sycamore_n{n}: length mismatch");

    // (1) run ≡ run_optimized.
    let mut max_diff = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((*x - *y).norm());
    }
    assert!(
        max_diff < 1e-12,
        "sycamore_n{n}: run vs run_optimized max amplitude diff {max_diff:.3e} exceeds 1e-12"
    );

    // (2) Normalization on both paths.
    let norm_raw: f64 = a.iter().map(|c| c.norm_sqr()).sum();
    let norm_opt: f64 = b.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (norm_raw - 1.0).abs() < 1e-10,
        "sycamore_n{n}: run Σp = {norm_raw}"
    );
    assert!(
        (norm_opt - 1.0).abs() < 1e-10,
        "sycamore_n{n}: run_optimized Σp = {norm_opt}"
    );

    // (3) XEB sanity band: noiseless Porter–Thomas ⇒ ≈ 1, never the uniform 0.
    // The band is deliberately generous (catches gross corruption / failure to
    // entangle); the precise value is what the benchmark report records.
    let xeb = linear_xeb(b);
    assert!(
        (0.5..=1.5).contains(&xeb),
        "sycamore_n{n}: linear XEB {xeb:.4} outside the sane Porter–Thomas band [0.5, 1.5]"
    );
}

/// Fast-CI gate: n=16 (64 Ki amplitudes) runs both paths in a few seconds,
/// well under the CLAUDE.md 30s budget, while still exercising the full
/// √X/√Y/√W-run + CZ-brick-wall structure the fusion passes rewrite.
#[test]
fn sycamore_n16_is_correct_and_porter_thomas() {
    check(16);
}

/// n=20 (16 MiB) runs both paths in ~35s — over the CLAUDE.md 30s fast-CI
/// budget — so it joins the nightly ignored-tests schedule alongside n=24.
#[test]
#[ignore = "n=20: ~35s both-paths; nightly ignored-tests schedule"]
fn sycamore_n20_is_correct_and_porter_thomas() {
    check(20);
}

/// n=24 allocates a 256 MiB state vector; nightly ignored-tests schedule.
#[test]
#[ignore = "n=24: 256 MiB state; nightly ignored-tests schedule"]
fn sycamore_n24_is_correct_and_porter_thomas() {
    check(24);
}
