//! Q1-04 regression: the native [`MwpmDecoder`], wired into the Q0 Monte-Carlo harness, exhibits
//! a surface-code threshold in the right place — **hermetically**, with no PyMatching/Stim.
//!
//! The defining property of a below-threshold code is that adding distance *suppresses* logical
//! error, and above threshold it *amplifies* it. So with the native decoder, rate(d=5) is clearly
//! below rate(d=3) at `p = 0.02` (below the ~2.6–2.8% phenomenological threshold) and clearly above
//! it at `p = 0.045` (above threshold). A decoder that reproduced the wrong threshold — or was
//! simply broken — would fail one of these. The exact agreement with PyMatching's *rate* is
//! covered separately by the (PyMatching-gated) oracle test; here we need only the native decoder
//! and the harness.

use aleph_qec::{run_memory_experiment, MwpmDecoder, PhenomenologicalNoise, SurfaceCode};

/// Native-MWPM logical error rate for the `d`-distance, `d`-round memory-Z experiment at `p`.
fn rate(d: usize, p: f64, shots: u64, seed: u64) -> f64 {
    run_memory_experiment(
        &SurfaceCode::new(d),
        &PhenomenologicalNoise::uniform(p),
        d,
        shots,
        seed,
        MwpmDecoder::new,
    )
    .expect("sweep cell")
    .rate
}

#[test]
fn native_mwpm_shows_threshold_between_d3_and_d5() {
    let shots = 60_000;
    let seed = 2024;

    // Below threshold: distance suppresses logical error.
    let lo_d3 = rate(3, 0.020, shots, seed);
    let lo_d5 = rate(5, 0.020, shots, seed);
    assert!(
        lo_d5 < lo_d3 * 0.85,
        "below threshold (p=0.02): expected rate(d5)={lo_d5:.4} clearly below rate(d3)={lo_d3:.4}"
    );

    // Above threshold: distance amplifies logical error.
    let hi_d3 = rate(3, 0.045, shots, seed);
    let hi_d5 = rate(5, 0.045, shots, seed);
    assert!(
        hi_d5 > hi_d3 * 1.05,
        "above threshold (p=0.045): expected rate(d5)={hi_d5:.4} above rate(d3)={hi_d3:.4}"
    );
}
