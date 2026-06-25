//! Q1-02 oracle: our from-scratch MWPM decoder ([`MwpmDecoder`]) agrees with PyMatching — the
//! reference minimum-weight perfect matching decoder — on shared surface-code memory DEMs.
//!
//! Two tests, each over the *same* DEM and the *same* sampled syndromes as PyMatching:
//!
//! 1. [`mwpm_logical_error_rate_matches_pymatching`] — at a below-threshold rate (`p=0.03`, d∈{3,5})
//!    the **logical error rate** matches PyMatching's within the combined 95% confidence interval.
//!    This is the Q1-02 exit metric, and it holds even though the two decoders sometimes pick
//!    different equal-weight matchings: MWPM ties (equal-weight matchings in different homology
//!    classes) are genuine, and on a tie either choice is equally (in)correct, so they wash out.
//! 2. [`mwpm_corrections_match_pymatching_when_unambiguous`] — in the sparse regime (`p=0.006`),
//!    where the minimum-weight matching is essentially unique, the **corrections** themselves match
//!    on ≥ 99% of non-empty shots. This pins down per-shot correctness where ties don't muddy it.
//!
//! Together: provably-optimal matching (the hermetic brute-force test in `blossom.rs`) + identical
//! logical performance to PyMatching + identical corrections where the answer is unambiguous.
//!
//! Requires a Python with `numpy` + `stim` + `pymatching`; `#[ignore]`d so the default
//! `cargo test` stays hermetic. Run on a PyMatching-equipped box:
//!
//!   PYMATCHING_PYTHON=/path/to/venv/bin/python \
//!     cargo test -p aleph-qec --test mwpm_pymatching_oracle -- --ignored --nocapture

use aleph_qec::{build_dem, Decoder, DetectorErrorModel, MwpmDecoder, PyMatchingOracle, Syndrome};

/// Sample `shots` syndromes (and true observable flips) from `dem`, deterministically per seed —
/// the same Bernoulli-per-mechanism model the Monte-Carlo harness uses.
fn sample(dem: &DetectorErrorModel, shots: usize, seed: u64) -> (Vec<Syndrome>, Vec<bool>) {
    let mut syndromes = Vec::with_capacity(shots);
    let mut truths = Vec::with_capacity(shots);
    for s in 0..shots as u64 {
        // SplitMix64 per shot.
        let mut z = seed.wrapping_add(s.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut next = || {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut det = vec![false; dem.detectors];
        let mut obs = false;
        for e in &dem.errors {
            if next() < e.prob {
                for &d in &e.dets {
                    det[d as usize] ^= true;
                }
                if e.obs.contains(&0) {
                    obs ^= true;
                }
            }
        }
        syndromes.push(Syndrome::from_bits(&det));
        truths.push(obs);
    }
    (syndromes, truths)
}

/// Outcome of decoding the same shots with both decoders.
struct Compare {
    rate_ours: f64,
    rate_theirs: f64,
    ci: f64,
    /// Agreement over shots whose syndrome is non-empty (the only shots a decoder can disagree
    /// on — both predict "no flip" on an empty syndrome).
    nonempty_agreement: f64,
    nonempty: usize,
}

/// Decode `shots` samples of the `d`-distance memory DEM at physical error `p` with both decoders.
fn compare(d: usize, p: f64, shots: usize, seed: u64) -> Compare {
    use aleph_qec::SurfaceCode;
    let exp = SurfaceCode::new(d).memory_z_experiment(d);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();

    let mwpm = MwpmDecoder::new(&dem).unwrap();
    let oracle = PyMatchingOracle::new(&dem);

    let (syndromes, truths) = sample(&dem, shots, seed);
    let ours = mwpm.decode_batch(&syndromes).unwrap();
    let theirs = oracle
        .decode_batch(&syndromes)
        .expect("PyMatching subprocess (set PYMATCHING_PYTHON)");

    let (mut err_ours, mut err_theirs) = (0u64, 0u64);
    let (mut agree, mut nonempty) = (0usize, 0usize);
    for ((s, (o, t)), truth) in syndromes.iter().zip(ours.iter().zip(&theirs)).zip(&truths) {
        let fo = o.observable_flips.first().copied().unwrap_or(false);
        let ft = t.observable_flips.first().copied().unwrap_or(false);
        if fo != *truth {
            err_ours += 1;
        }
        if ft != *truth {
            err_theirs += 1;
        }
        if s.weight() > 0 {
            nonempty += 1;
            if fo == ft {
                agree += 1;
            }
        }
    }
    let n = shots as f64;
    let (ro, rt) = (err_ours as f64 / n, err_theirs as f64 / n);
    Compare {
        rate_ours: ro,
        rate_theirs: rt,
        ci: 1.96 * ((ro * (1.0 - ro) + rt * (1.0 - rt)) / n).sqrt(),
        nonempty_agreement: if nonempty == 0 {
            1.0
        } else {
            agree as f64 / nonempty as f64
        },
        nonempty,
    }
}

#[test]
#[ignore = "requires python3 + numpy + stim + pymatching; set PYMATCHING_PYTHON"]
fn mwpm_logical_error_rate_matches_pymatching() {
    // Exit metric (Q1-02): aleph-MWPM's logical error rate equals PyMatching's within CI on
    // shared DEMs, for d ∈ {3,5}, at a below-threshold rate with plenty of logical errors.
    for d in [3usize, 5] {
        let c = compare(d, 0.03, 30_000, 0xABCD ^ d as u64);
        eprintln!(
            "d={d} p=0.03: aleph rate={:.4} pymatching rate={:.4} |Δ|={:.4} ci95={:.4} \
             nonempty-agreement={:.4} ({} nonempty)",
            c.rate_ours,
            c.rate_theirs,
            (c.rate_ours - c.rate_theirs).abs(),
            c.ci,
            c.nonempty_agreement,
            c.nonempty,
        );
        assert!(
            (c.rate_ours - c.rate_theirs).abs() <= c.ci + 1e-9,
            "d={d}: logical error rates differ beyond CI: aleph {} vs pymatching {} (ci {})",
            c.rate_ours,
            c.rate_theirs,
            c.ci,
        );
    }
}

#[test]
#[ignore = "requires python3 + numpy + stim + pymatching; set PYMATCHING_PYTHON"]
fn mwpm_corrections_match_pymatching_when_unambiguous() {
    // In the sparse (low-p) regime, syndromes are light and the minimum-weight matching is
    // essentially unique, so a correct MWPM decoder must produce the *same correction* as
    // PyMatching on (almost) every non-empty shot. (At higher p the two correctly disagree on
    // equal-weight ties in different homology classes — that degeneracy is exercised, and bounded,
    // by the rate-within-CI test above, not here.)
    for d in [3usize, 5] {
        let c = compare(d, 0.006, 60_000, 0x1234 ^ d as u64);
        eprintln!(
            "d={d} p=0.006: nonempty-agreement={:.5} over {} nonempty shots",
            c.nonempty_agreement, c.nonempty
        );
        assert!(
            c.nonempty > 1_000,
            "d={d}: too few non-empty shots ({}) to be meaningful",
            c.nonempty
        );
        assert!(
            c.nonempty_agreement >= 0.99,
            "d={d}: corrections agree on only {:.5} of non-empty shots (expected ≥ 0.99)",
            c.nonempty_agreement
        );
    }
}
