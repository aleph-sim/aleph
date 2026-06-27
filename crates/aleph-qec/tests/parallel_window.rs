//! Q4-02 parallel-window decoder oracle tests.
//!
//! * Fast (CI): the parallel two-layer decode produces a *valid* committed correction (residual
//!   clears) on a long stream, and a single full window reproduces a batch decode exactly.
//! * Slow (`#[ignore]`, nightly): the logical-error rate is within CI of full-batch decoding, the
//!   Q4-02 correctness acceptance criterion (mirrors the Q4-01 sliding-window oracle).

use aleph_qec::{
    build_dem, run_dem_experiment, ParallelWindowDecoder, SurfaceCode, Syndrome, UnionFindDecoder,
};

fn stream(d: usize, rounds: usize, p: f64) -> (aleph_qec::DetectorErrorModel, Vec<usize>) {
    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    (dem, exp.detector_rounds())
}

/// On a long stream the two-layer parallel decode always reproduces the input syndrome (the
/// committed correction is syndrome-consistent), so seams compose correctly. Cheap; runs in CI.
#[test]
fn parallel_correction_is_valid_on_long_stream() {
    let d = 5;
    let (dem, rounds) = stream(d, 8 * d, 0.04);
    let pw = ParallelWindowDecoder::new(dem.clone(), rounds, d, d);
    assert!(
        pw.num_windows() >= 5,
        "want several windows across two layers"
    );

    let mut z: u64 = 0x51DE_0001 ^ 0xA5A5;
    let mut next = || {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z
    };
    for _ in 0..200 {
        let bits: Vec<bool> = (0..dem.detectors).map(|_| next() % 12 == 0).collect();
        let syn = Syndrome::from_bits(&bits);
        assert_eq!(
            pw.residual_after_decode(&syn),
            0,
            "committed correction must reproduce the syndrome"
        );
    }
}

/// Logical-error rate within CI of full-batch UF, for buffer `B ≳ d`. Slow Monte-Carlo; nightly.
#[test]
#[ignore = "Monte-Carlo oracle (~minutes); runs on the nightly CI schedule"]
fn parallel_rate_within_ci_of_batch() {
    let shots: u64 = std::env::var("Q4_ORACLE_SHOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let seed = 7;
    for &d in &[3usize, 5] {
        let rounds = 6 * d;
        let (dem, det_rounds) = stream(d, rounds, 0.03);
        let batch = UnionFindDecoder::new(&dem).unwrap();
        let b = run_dem_experiment(&dem, shots, &batch, seed).expect("batch");

        // Buffer B = d (window C + 2d) — the adequate-buffer regime from Q4-01.
        let pw = ParallelWindowDecoder::new(dem.clone(), det_rounds, d, d);
        let s = run_dem_experiment(&dem, shots, &pw, seed).expect("parallel");

        let delta = (s.rate - b.rate).abs();
        let cci = b.ci95 + s.ci95;
        assert!(
            delta <= cci,
            "d={d}: parallel rate {:.5} not within CI of batch {:.5} (Δ={delta:.2e} > ci={cci:.2e})",
            s.rate,
            b.rate
        );
    }
}
