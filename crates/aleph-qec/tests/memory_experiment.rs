//! Q0-04 integration: the memory-experiment harness end-to-end.
//!
//! The hermetic tests (no Python) exercise the harness with the in-process [`NullDecoder`]:
//! determinism, the `p=0 ⇒ rate 0` invariant, and monotone growth of the *raw* logical-flip
//! rate with `p` (the NullDecoder rate is exactly the rate at which noise flips the observable).
//!
//! The threshold test uses the external [`PyMatchingOracle`] and is `#[ignore]`d — it needs a
//! Python with numpy + stim + pymatching. Run it on a suitably-equipped box, e.g.:
//!
//!   PYMATCHING_PYTHON=/root/pmvenv/bin/python \
//!     cargo test -p aleph-qec --test memory_experiment -- --ignored

use aleph_qec::{
    run_memory_experiment, NullDecoder, PhenomenologicalNoise, PyMatchingOracle, SurfaceCode,
};

#[test]
fn p_zero_is_zero_rate_for_every_distance() {
    for d in [3usize, 5] {
        let res = run_memory_experiment(
            &SurfaceCode::new(d),
            &PhenomenologicalNoise::uniform(0.0),
            d, // rounds = d, the usual choice
            2_000,
            1,
            |dem| Ok(NullDecoder::new(dem.observables)),
        )
        .unwrap();
        assert_eq!(res.logical_errors, 0, "d={d}");
        assert_eq!(res.rate, 0.0, "d={d}");
    }
}

#[test]
fn null_decoder_rate_grows_with_noise() {
    // The NullDecoder predicts no correction, so its logical-error rate is the bare
    // observable-flip rate, which must increase with the physical error rate.
    let code = SurfaceCode::new(3);
    let rate = |p: f64| {
        run_memory_experiment(
            &code,
            &PhenomenologicalNoise::uniform(p),
            3,
            50_000,
            9,
            |dem| Ok(NullDecoder::new(dem.observables)),
        )
        .unwrap()
        .rate
    };
    let (lo, hi) = (rate(0.01), rate(0.08));
    assert!(lo < hi, "rate should grow with p: {lo} !< {hi}");
    assert!(lo > 0.0, "some logical flips expected at p=0.01");
}

/// Below threshold, MWPM logical error rate should *fall* as distance grows. This is the
/// qualitative correctness check the acceptance criteria ask for; it needs real MWPM.
#[test]
#[ignore = "requires python3 + numpy + stim + pymatching; set PYMATCHING_PYTHON"]
fn pymatching_suppresses_errors_with_distance_below_threshold() {
    // Well below the phenomenological threshold (~3%): higher distance ⇒ lower logical error.
    let noise = PhenomenologicalNoise::uniform(0.01);
    let shots = 200_000u64;

    let mut rates = Vec::new();
    for d in [3usize, 5, 7] {
        let code = SurfaceCode::new(d);
        let res = run_memory_experiment(&code, &noise, d, shots, 2024, |dem| {
            Ok(PyMatchingOracle::new(dem))
        })
        .unwrap_or_else(|e| panic!("d={d}: {e}"));
        eprintln!("d={d}: rate = {:.3e} ± {:.1e}", res.rate, res.ci95);
        rates.push(res.rate);
    }

    assert!(
        rates[0] > rates[1] && rates[1] > rates[2],
        "below threshold, logical error rate must fall with distance: {rates:?}"
    );
}

/// Above threshold, adding distance should *not* help (rate flat or rising). Pairs with the
/// test above to bracket the threshold, per the acceptance criteria.
#[test]
#[ignore = "requires python3 + numpy + stim + pymatching; set PYMATCHING_PYTHON"]
fn pymatching_does_not_suppress_above_threshold() {
    // Well above the phenomenological threshold: larger d should not reduce the logical rate.
    let noise = PhenomenologicalNoise::uniform(0.10);
    let shots = 100_000u64;

    let rate = |d: usize| {
        let code = SurfaceCode::new(d);
        run_memory_experiment(&code, &noise, d, shots, 7, |dem| {
            Ok(PyMatchingOracle::new(dem))
        })
        .unwrap_or_else(|e| panic!("d={d}: {e}"))
        .rate
    };
    let (r3, r7) = (rate(3), rate(7));
    eprintln!("above threshold: d=3 {r3:.3e}, d=7 {r7:.3e}");
    assert!(
        r7 >= r3 - 0.02,
        "above threshold d=7 ({r7}) should not beat d=3 ({r3})"
    );
}

/// Q0-05 regression: the threshold recorded in `docs/perf/qec-q0-threshold.md` (~2.6–2.8%)
/// brackets between p=2.5% and p=3.0%. We don't recompute the full sweep — we check that the
/// distance ordering of the logical rate *inverts* across that band: at p=2.5% larger d is
/// better (below threshold), at p=3.0% larger d is worse (above threshold). The crossing
/// therefore lies in (2.5%, 3.0%), pinning the recorded value to its tolerance band.
#[test]
#[ignore = "requires python3 + numpy + stim + pymatching; set PYMATCHING_PYTHON"]
fn threshold_brackets_recorded_value() {
    let shots = 100_000u64;
    let rate = |d: usize, p: f64| {
        run_memory_experiment(
            &SurfaceCode::new(d),
            &PhenomenologicalNoise::uniform(p),
            d,
            shots,
            2024,
            |dem| Ok(PyMatchingOracle::new(dem)),
        )
        .unwrap_or_else(|e| panic!("d={d} p={p}: {e}"))
        .rate
    };

    // Below the threshold: d=9 suppresses relative to d=3.
    let (lo3, lo9) = (rate(3, 0.025), rate(9, 0.025));
    // Above the threshold: d=9 is worse than d=3.
    let (hi3, hi9) = (rate(3, 0.030), rate(9, 0.030));
    eprintln!("p=2.5%: d3={lo3:.4e} d9={lo9:.4e} | p=3.0%: d3={hi3:.4e} d9={hi9:.4e}");

    assert!(
        lo9 < lo3,
        "p=2.5% should be below threshold: d9 {lo9} !< d3 {lo3}"
    );
    assert!(
        hi9 > hi3,
        "p=3.0% should be above threshold: d9 {hi9} !> d3 {hi3}"
    );
}
