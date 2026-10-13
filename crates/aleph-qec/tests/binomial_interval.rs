//! Direct-sampling confidence bounds: references, boundary semantics and exact coverage.

use aleph_qec::{BinomialInterval95, LogicalErrorResult};

#[test]
fn boundaries_no_evidence_and_invalid_tallies() {
    let none = BinomialInterval95::from_counts(0, 0).unwrap();
    assert_eq!((none.lower, none.upper), (0.0, 1.0));
    for n in [1, 10, 1000, 1_000_000, u64::MAX] {
        let zero = BinomialInterval95::from_counts(n, 0).unwrap();
        let all = BinomialInterval95::from_counts(n, n).unwrap();
        assert_eq!(zero.lower, 0.0);
        assert!(zero.upper > 0.0 && zero.upper <= 1.0);
        assert_eq!(all.upper, 1.0);
        assert!(all.lower <= 1.0 - zero.upper && all.lower < 1.0);
    }
    assert!(LogicalErrorResult::try_new(0, 1).is_err());
    assert!(LogicalErrorResult::try_new(10, 11).is_err());
    assert!(LogicalErrorResult::new(10, 11).interval95().is_err());
    let invalid_zero = LogicalErrorResult::new(0, 1);
    assert_eq!(invalid_zero.logical_errors, 1);
    assert!(invalid_zero.rate.is_nan() && invalid_zero.ci95.is_nan());
    assert!(invalid_zero.interval95().is_err());
    assert!(BinomialInterval95::from_counts(u64::MAX, 1).is_err());
    // Public fields remain source-compatible; bounds must be derived from tallies.
    let forged = LogicalErrorResult {
        shots: 1000,
        logical_errors: 0,
        rate: 0.5,
        ci95: 0.3,
    };
    assert_eq!(forged.interval95().unwrap().lower, 0.0);
}

proptest::proptest! {
    #[test]
    fn random_direct_counts_produce_ordered_finite_bounds(n in 1_u64..1_000_001, fraction in 0_u64..1_000_001) {
        let k = n * fraction / 1_000_000;
        let b = BinomialInterval95::from_counts(n, k).unwrap();
        let rate = k as f64 / n as f64;
        proptest::prop_assert!(b.lower.is_finite() && b.upper.is_finite());
        proptest::prop_assert!(0.0 <= b.lower && b.lower <= rate && rate <= b.upper && b.upper <= 1.0);
    }
}

#[test]
fn reference_values() {
    // Independent reference: scipy.stats.binomtest(k, n).proportion_ci(method="exact").
    for (n, k, lo, hi) in [
        (1, 0, 0.0, 0.975),
        (10, 1, 0.0025285785444618, 0.4450161170281954),
        (20, 4, 0.0573339970500327, 0.4366140029966683),
        (1000, 0, 0.0, 0.0036820838968657),
        (1000, 50, 0.037335397604661, 0.0653904879154936),
    ] {
        let bounds = BinomialInterval95::from_counts(n, k).unwrap();
        assert!((bounds.lower - lo).abs() < 1e-10, "n={n} k={k}: {bounds:?}");
        assert!((bounds.upper - hi).abs() < 1e-10, "n={n} k={k}: {bounds:?}");
    }
}

#[test]
fn ordered_monotone_symmetric_bounds() {
    for n in [1, 2, 10, 100, 1000] {
        let (mut lower, mut upper) = (0.0, 0.0);
        for k in 0..=n {
            let b = BinomialInterval95::from_counts(n, k).unwrap();
            let reflected = BinomialInterval95::from_counts(n, n - k).unwrap();
            assert!(0.0 <= b.lower && b.lower <= k as f64 / n as f64);
            assert!(k as f64 / n as f64 <= b.upper && b.upper <= 1.0);
            assert!(b.lower + 1e-14 >= lower && b.upper + 1e-14 >= upper);
            assert!((b.lower - (1.0 - reflected.upper)).abs() < 1e-12);
            (lower, upper) = (b.lower, b.upper);
        }
    }
}

#[test]
fn exact_small_sample_coverage() {
    // Enumerate every possible count with independently evaluated binomial probabilities.
    // No repeated-sampling uncertainty: finite sum covers the entire outcome space.
    for n in [1, 2, 5, 10, 20, 100] {
        let intervals: Vec<_> = (0..=n)
            .map(|k| BinomialInterval95::from_counts(n, k).unwrap())
            .collect();
        for p in [
            0.0_f64, 0.0001, 0.001, 0.01, 0.05, 0.2, 0.5, 0.9, 0.99, 0.9999, 1.0,
        ] {
            let mut coverage = 0.0;
            let mut choose = 1.0;
            for (k, b) in intervals.iter().enumerate() {
                let mass = choose * p.powi(k as i32) * (1.0 - p).powi(n as i32 - k as i32);
                if b.lower <= p && p <= b.upper {
                    coverage += mass;
                }
                choose *= (n as usize - k) as f64 / (k + 1) as f64;
            }
            assert!(coverage >= 0.95 - 1e-12, "n={n} p={p} coverage={coverage}");
        }
    }
}

#[test]
fn large_counts_and_overlap_semantics() {
    let b = BinomialInterval95::from_counts(1_000_000, 1000).unwrap();
    assert!(b.lower < 0.001 && b.upper > 0.001);
    let zero = BinomialInterval95::from_counts(1000, 0).unwrap();
    let all = BinomialInterval95::from_counts(1000, 1000).unwrap();
    assert!(!zero.overlaps(all));
    assert!(zero.overlaps(zero));
}

#[test]
fn conditional_paired_comparison_does_not_assume_independent_rates() {
    // Equal split of discordant errors: equality cannot be rejected, not equivalence.
    let equal = BinomialInterval95::from_counts(20, 10).unwrap();
    assert!(equal.lower <= 0.5 && equal.upper >= 0.5);
    // All discordant errors belong to one decoder: a detectable paired rate difference.
    let different = BinomialInterval95::from_counts(20, 20).unwrap();
    assert!(different.lower > 0.5);
    let no_discordance = BinomialInterval95::from_counts(0, 0).unwrap();
    assert_eq!((no_discordance.lower, no_discordance.upper), (0.0, 1.0));
}
