//! Q2-01 regression: the [`UnionFindDecoder`], wired into the Q0 Monte-Carlo harness, exhibits a
//! surface-code threshold in the right place — **hermetically**, no PyMatching/Stim.
//!
//! As with the MWPM regression ([`mwpm_threshold`]), the defining property is that adding distance
//! *suppresses* logical error below threshold and *amplifies* it above. Union-Find's (unweighted)
//! threshold sits slightly *below* MWPM's (~2.6–2.8%), but `p = 0.02` is still comfortably below it
//! and `p = 0.05` comfortably above, so the same suppression/amplification bracket holds. The exact
//! UF-vs-MWPM gap is characterised in the Q2-03 report; here we only need the decoder + harness.

use aleph_qec::{run_memory_experiment, PhenomenologicalNoise, SurfaceCode, UnionFindDecoder};

/// Union-Find logical error rate for the `d`-distance, `d`-round memory-Z experiment at `p`.
fn rate(d: usize, p: f64, shots: u64, seed: u64) -> f64 {
    run_memory_experiment(
        &SurfaceCode::new(d),
        &PhenomenologicalNoise::uniform(p),
        d,
        shots,
        seed,
        UnionFindDecoder::new,
    )
    .expect("sweep cell")
    .rate
}

#[test]
fn union_find_shows_threshold_between_d3_and_d5() {
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
    let hi_d3 = rate(3, 0.050, shots, seed);
    let hi_d5 = rate(5, 0.050, shots, seed);
    assert!(
        hi_d5 > hi_d3 * 1.05,
        "above threshold (p=0.05): expected rate(d5)={hi_d5:.4} above rate(d3)={hi_d3:.4}"
    );
}
