//! P5.5-05 Part A: Metal FP32 state oracle vs **Qiskit-Aer** fixtures.
//!
//! Runs `MetalSvBackend` through the committed exact-FP64 Aer fixtures
//! (`oracle/fixtures/*.json`) on the Tier-1 families and asserts each amplitude
//! matches within 1e-5 — the same `FP32_STATE_TOLERANCE` path `Fp32SvBackend`
//! uses. Comparing a single-precision GPU run against the *exact* FP64 reference
//! at 1e-5 bounds the f32 accumulation error against the true answer, which is
//! strictly stronger than "vs Aer single-precision". Both the verbatim `run`
//! path and the fused `run_optimized` path (the one the n~28 bench times) are
//! checked, so the timed code path is the proven-correct one.
//!
//! Skips (passes) when no Metal device is available so headless/Linux CI stays
//! green; runs for real on Apple Silicon.
//!
//! Run: `cargo test -p aleph-metal --features metal --test sv_oracle_aer`

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_core::Complex;
use aleph_metal::MetalSvBackend;
use aleph_oracle::{load_fixture, load_qasm, run_state_oracle_with_tol, workspace_path, Fixture};

/// FP32 oracle tolerance (matches `aleph_oracle::FP32_STATE_TOLERANCE`).
const TOL: f64 = 1e-5;

/// Tier-1 fixture stems present in `oracle/fixtures/`. One entry per family:
/// GHZ, QFT, Grover (2q + 3q multi-controlled), random (Clifford + non-Clifford).
const TIER1_FIXTURES: &[&str] = &[
    "ghz_3",
    "ghz_5",
    "ghz_10",
    "qft_3",
    "qft_5",
    "grover_2q_mark11",
    "multi_ctrl_grover_ccz_3q",
    "random_clifford_n4_d20",
    "random_nonclifford_n4_d20",
];

/// Load a fixture stem -> (Fixture, qasm source). Panics with a clear message on
/// any IO/parse error so a missing fixture names itself.
fn load(stem: &str) -> (Fixture, String) {
    let fx_path = workspace_path(&format!("oracle/fixtures/{stem}.json"));
    let fx = load_fixture(&fx_path).unwrap_or_else(|e| panic!("load fixture {stem}: {e}"));
    let qasm_path = workspace_path(&format!("oracle/{}", fx.qasm_path));
    let qasm = load_qasm(&qasm_path).unwrap_or_else(|e| panic!("load qasm {stem}: {e}"));
    (fx, qasm)
}

/// Assert `MetalSvBackend::run_optimized` matches the fixture amplitudes within
/// `TOL`. Mirrors the harness's finite guard (a NaN amplitude must fail loudly,
/// not slip past a `delta > tol` comparison that NaN makes `false`).
fn assert_optimized_close(gpu: &mut MetalSvBackend, fx: &Fixture, qasm: &str) {
    let circuit = aleph_parser::parse(qasm).expect("parse qasm");
    let state = gpu.run_optimized(&circuit).expect("gpu run_optimized");
    let actual: Vec<Complex<f64>> = aleph_oracle::HasAmplitudes::amplitudes(&state);
    let expected = &fx.statevector.amplitudes;
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: dim mismatch (optimized)",
        fx.name
    );
    for (i, (a, &(er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.re.is_finite() && a.im.is_finite() && er.is_finite() && ei.is_finite(),
            "{}: non-finite amplitude at {i} (optimized): gpu ({}, {}) ref ({er}, {ei})",
            fx.name,
            a.re,
            a.im
        );
        let d = ((a.re - er).powi(2) + (a.im - ei).powi(2)).sqrt();
        assert!(
            d <= TOL,
            "{}: amplitude {i} |Δ|={d:.3e} > {TOL:.0e} (optimized)\n  gpu ({}, {})\n  ref ({er}, {ei})",
            fx.name,
            a.re,
            a.im
        );
    }
}

#[test]
fn fp32_tier1_oracle_matches_aer() {
    let mut gpu = match MetalSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping FP32 Aer oracle: no Metal device available");
            return;
        }
    };

    for stem in TIER1_FIXTURES {
        let (fx, qasm) = load(stem);
        // Verbatim `run` path via the shared harness (reuses its finite guard).
        run_state_oracle_with_tol(&mut gpu, &fx, &qasm, TOL)
            .unwrap_or_else(|e| panic!("{stem}: oracle (run) harness error: {e}"));
        // Fused `run_optimized` path — the one the n~28 bench times.
        assert_optimized_close(&mut gpu, &fx, &qasm);
    }
}
