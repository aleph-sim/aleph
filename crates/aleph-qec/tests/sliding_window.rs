//! Q4-01 acceptance: the sliding-window streaming decoder must reproduce the **full-batch** logical
//! error rate within CI once the window is large enough, on long syndrome streams, and decode an
//! unbounded stream in bounded memory.

use aleph_qec::{
    build_dem, run_dem_experiment, SlidingWindowDecoder, SurfaceCode, UnionFindDecoder,
};

/// Build a long memory-Z stream DEM + detector rounds at distance `d`, `rounds` rounds.
fn stream(d: usize, rounds: usize, p: f64) -> (aleph_qec::DetectorErrorModel, Vec<usize>) {
    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    (dem, exp.detector_rounds())
}

/// For an adequate window (`buffer ≥ d`), the sliding-window logical-error rate matches the full
/// batch UF rate within the combined 95% CI, at several distances, on a long stream.
///
/// Heavy statistical Monte-Carlo (the per-window graph rebuild makes it minutes), so `#[ignore]`d for
/// the nightly schedule per CLAUDE.md; `full_window_is_batch` is the fast exact CI guard.
#[test]
#[ignore = "minutes-long statistical Monte-Carlo; nightly"]
fn sliding_matches_batch_for_adequate_window() {
    let shots = 60_000u64;
    let seed = 7u64;
    for &d in &[3usize, 5] {
        let rounds = 8 * d; // a long stream
        let (dem, det_rounds) = stream(d, rounds, 0.03);

        let batch = UnionFindDecoder::new(&dem).unwrap();
        let b = run_dem_experiment(&dem, shots, &batch, seed).expect("batch");

        // Commit C = d, buffer = d ⇒ W = 2d (a modest d-dependent bound).
        let sw = SlidingWindowDecoder::new(dem.clone(), det_rounds, 2 * d, d);
        let s = run_dem_experiment(&dem, shots, &sw, seed).expect("sliding");

        let delta = (s.rate - b.rate).abs();
        let bound = 2.0 * (s.ci95 + b.ci95); // ~4σ combined, robustly non-flaky
        assert!(
            delta <= bound,
            "d={d} W={} C={d}: sliding rate {:.5} vs batch {:.5} differ by {delta:.5} > {bound:.5}",
            2 * d,
            s.rate,
            b.rate
        );
    }
}

/// The window working set does not grow with stream length — the bounded-memory property.
#[test]
fn bounded_memory_independent_of_stream_length() {
    let d = 5;
    let (dem_a, ra) = stream(d, 20, 0.03);
    let (dem_b, rb) = stream(d, 80, 0.03);
    let a = SlidingWindowDecoder::new(dem_a, ra, 3 * d, d);
    let b = SlidingWindowDecoder::new(dem_b, rb, 3 * d, d);
    assert_eq!(
        a.max_window_detectors(),
        b.max_window_detectors(),
        "per-window working set must be independent of total stream length"
    );
}

/// A single full-stream window is exactly a batch decode (sanity, also covered by a lib unit test).
#[test]
fn full_window_is_batch() {
    let d = 3;
    let rounds = 10;
    let (dem, det_rounds) = stream(d, rounds, 0.05);
    let num_slices = det_rounds.iter().copied().max().unwrap() + 1;
    let sw = SlidingWindowDecoder::new(dem.clone(), det_rounds, num_slices, num_slices);
    let batch = UnionFindDecoder::new(&dem).unwrap();
    let b = run_dem_experiment(&dem, 20_000, &batch, 1).unwrap();
    let s = run_dem_experiment(&dem, 20_000, &sw, 1).unwrap();
    assert_eq!(
        s.logical_errors, b.logical_errors,
        "full window must equal batch exactly"
    );
}
