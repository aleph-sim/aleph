//! Exact small-model truth, paired calibration, stateful laws and artifact compatibility.
use aleph_qec::*;
use std::time::Duration;
fn tiny() -> DetectorErrorModel {
    DetectorErrorModel::parse(
        "error(0.05) D0 L0\nerror(0.3) D0 L1\nerror(0.15) L0\nerror(0.2) D1 L1",
    )
    .unwrap()
}
fn opts() -> RobustnessOptions {
    RobustnessOptions {
        streams: 8,
        steps: 256,
        seed: 546,
        confidence: 0.95,
        witness_limit: 3,
    }
}
fn exact(dem: &DetectorErrorModel, d: &WitnessDecoder) -> f64 {
    (0..1usize << dem.errors.len())
        .map(|mask| {
            let faults: Vec<_> = (0..dem.errors.len())
                .filter(|i| mask >> i & 1 != 0)
                .collect();
            let p = dem
                .errors
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    if mask >> i & 1 != 0 {
                        e.prob
                    } else {
                        1.0 - e.prob
                    }
                })
                .product::<f64>();
            if d.decode_faults(&faults).unwrap().logical_failure {
                p
            } else {
                0.0
            }
        })
        .sum()
}
#[test]
fn matched_static_is_exactly_the_existing_harness() {
    let dem = tiny();
    let d = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let mut o = opts();
    o.streams = 1;
    let r = run_robustness(&dem, &NoiseProcess::Static, &d, &d, &o).unwrap();
    let direct = run_dem_experiment(&dem, o.steps, &d, o.seed).unwrap();
    assert_eq!(r.streams[0].assumed_failures, direct.logical_errors);
    assert_eq!(r.paired_difference, 0.0);
    assert_eq!(r.decoder_calls, 2 * o.steps);
    assert_eq!(r.assumed_interval.lower, direct.interval95().unwrap().lower);
    assert_eq!(r.assumed_interval.upper, direct.interval95().unwrap().upper);
    assert_eq!(r.generating_sha256, r.assumed_sha256);
}
#[test]
fn exact_heterogeneous_truth_and_paired_mismatch() {
    let dem = tiny();
    let mut wrong = dem.clone();
    wrong.errors[0].prob = 0.4;
    wrong.errors[1].prob = 0.01;
    let a = WitnessDecoder::new(&wrong, DecoderSpec::Mwpm).unwrap();
    let b = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let pa = exact(&dem, &a);
    let pb = exact(&dem, &b);
    assert!((pb - 0.1925).abs() < 1e-12);
    assert!(pa > pb);
    let mut o = opts();
    o.streams = 16;
    o.steps = 512;
    let r = run_robustness(&dem, &NoiseProcess::Static, &a, &b, &o).unwrap();
    assert!(r.assumed_interval.lower <= pa && pa <= r.assumed_interval.upper);
    assert!(r.reference_interval.lower <= pb && pb <= r.reference_interval.upper);
    assert!(r.difference_interval.lower <= pa - pb && pa - pb <= r.difference_interval.upper);
    assert_ne!(r.generating_sha256, r.assumed_sha256);
    for row in &r.streams {
        assert_eq!(
            row.assumed_failures as i64 - row.reference_failures as i64,
            row.worse as i64 - row.better as i64
        );
    }
    for w in &r.witnesses {
        assert_eq!(w.version, 3);
        let decoded = FailureWitness::from_json(&w.to_json().unwrap()).unwrap();
        assert_eq!(decoded, *w);
        assert_eq!(a.replay(w).unwrap(), w.outcome);
    }
}
#[test]
fn zero_drift_and_burst_strength_preserve_shots_and_worker_determinism() {
    let dem = tiny();
    let static_shots = generate_noise_stream(&dem, &NoiseProcess::Static, 512, 546).unwrap();
    for process in [
        NoiseProcess::LinearDrift {
            end_probabilities: dem.errors.iter().map(|e| e.prob).collect(),
        },
        NoiseProcess::MarkovBurst {
            onset: 0.2,
            recovery: 0.1,
            affected: vec![0, 1, 2],
            multiplier: 1.0,
        },
    ] {
        let a = generate_noise_stream(&dem, &process, 512, 546).unwrap();
        assert_eq!(
            a.iter().map(|s| &s.faults).collect::<Vec<_>>(),
            static_shots.iter().map(|s| &s.faults).collect::<Vec<_>>()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        assert_eq!(
            a,
            pool.install(|| generate_noise_stream(&dem, &process, 512, 546).unwrap())
        );
    }
    let mut deterministic = dem.clone();
    for e in &mut deterministic.errors {
        e.prob = 0.0;
    }
    let p = NoiseProcess::LinearDrift {
        end_probabilities: vec![1.0; 4],
    };
    let shots = generate_noise_stream(&deterministic, &p, 2, 546).unwrap();
    assert!(shots[0].faults.is_empty());
    assert_eq!(shots[1].faults, vec![0, 1, 2, 3]);
}
#[test]
fn burst_law_has_expected_occupancy_duration_and_correlation() {
    let dem = tiny();
    let onset = 0.08;
    let recovery = 0.2;
    let p = NoiseProcess::MarkovBurst {
        onset,
        recovery,
        affected: vec![0, 1],
        multiplier: 3.0,
    };
    let mut active = 0;
    let mut starts = 0;
    let mut aa = 0;
    let mut recoveries = 0;
    let mut inactive_trials = 0;
    let mut active_trials = 0;
    let streams = 64;
    let steps = 256;
    for i in 0..streams {
        let shots = generate_noise_stream(&dem, &p, steps, robustness_stream_seed(546, i)).unwrap();
        let mut previous = false;
        for s in shots {
            active += u64::from(s.burst_active);
            starts += u64::from(s.burst_active && !previous);
            aa += u64::from(previous && s.burst_active);
            recoveries += u64::from(previous && !s.burst_active);
            inactive_trials += u64::from(!previous);
            active_trials += u64::from(previous);
            previous = s.burst_active;
        }
    }
    let mut expected = 0.0;
    let mut state = 0.0;
    for _ in 0..steps {
        state = (1.0 - state) * onset + state * (1.0 - recovery);
        expected += state;
    }
    assert!((active as f64 / (streams * steps) as f64 - expected / steps as f64).abs() < 0.025);
    assert!((starts as f64 / inactive_trials as f64 - onset).abs() < 0.02);
    assert!((recoveries as f64 / active_trials as f64 - recovery).abs() < 0.025);
    assert!((aa as f64 / active_trials as f64 - (1.0 - recovery)).abs() < 0.025);
    // Active-run total / starts is right-censored by stream ends; check away from tiny streams.
    assert!((active as f64 / starts as f64 - 1.0 / recovery).abs() < 0.7);
}
#[test]
fn correlated_witnesses_replay_state_and_reject_tampering() {
    let dem = tiny();
    let mut wrong = dem.clone();
    wrong.errors[0].prob = 0.4;
    wrong.errors[1].prob = 0.01;
    let a = WitnessDecoder::new(&wrong, DecoderSpec::Mwpm).unwrap();
    let b = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    for process in [
        NoiseProcess::MarkovBurst {
            onset: 0.3,
            recovery: 0.1,
            affected: vec![0, 1, 2],
            multiplier: 3.0,
        },
        NoiseProcess::LinearDrift {
            end_probabilities: vec![0.2; 4],
        },
    ] {
        let r = run_robustness(&dem, &process, &a, &b, &opts()).unwrap();
        assert!(!r.witnesses.is_empty());
        assert!(r.interval_method.contains("independent stream"));
        assert!(r.difference_interval.upper - r.difference_interval.lower > 0.5);
        for w in r.witnesses {
            let imported = FailureWitness::from_json(&w.to_json().unwrap()).unwrap();
            assert_eq!(w, imported);
            let reduced = a
                .minimize(
                    &w,
                    MinimizeBudget {
                        decoder_calls: 64,
                        time: Duration::from_secs(2),
                    },
                )
                .unwrap();
            FailureWitness::from_json(&reduced.witness.to_json().unwrap()).unwrap();
            let mut bad = serde_json::to_value(&w).unwrap();
            bad["sampling"]["stream"] = serde_json::json!(999);
            assert!(FailureWitness::from_json(&bad.to_string()).is_err());
            let mut bad = serde_json::to_value(&w).unwrap();
            bad["version"] = serde_json::json!(2);
            assert!(FailureWitness::from_json(&bad.to_string()).is_err());
        }
    }
}
#[test]
fn schema_and_process_validation_happens_before_sampling() {
    let parts = DetectorErrorModel::parse("error(0.1) D0 L0 ^ D1").unwrap();
    let flat = DetectorErrorModel::parse("error(0.2) D0 D1 L0").unwrap();
    assert!(validate_noise_compatibility(&parts, &flat).is_err());
    let dem = tiny();
    let mut swapped = dem.clone();
    swapped.errors.swap(0, 1);
    assert!(validate_noise_compatibility(&dem, &swapped).is_err());
    let mut changed = dem.clone();
    changed.errors[0].obs = vec![1];
    assert!(validate_noise_compatibility(&dem, &changed).is_err());
    for p in [
        NoiseProcess::LinearDrift {
            end_probabilities: vec![],
        },
        NoiseProcess::MarkovBurst {
            onset: f64::NAN,
            recovery: 0.1,
            affected: vec![],
            multiplier: 2.0,
        },
        NoiseProcess::MarkovBurst {
            onset: 0.1,
            recovery: 0.1,
            affected: vec![0, 0],
            multiplier: 2.0,
        },
    ] {
        assert!(generate_noise_stream(&dem, &p, 16, 0).is_err());
    }
    assert!(
        generate_noise_stream(&dem, &NoiseProcess::Static, MAX_ROBUSTNESS_STEPS + 1, 0).is_err()
    );
    let d = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let mut o = opts();
    o.streams = u64::MAX;
    assert!(run_robustness(&dem, &NoiseProcess::Static, &d, &d, &o).is_err());
}

#[test]
fn burst_prefix_does_not_depend_on_future_stream_length_or_other_streams() {
    let dem = tiny();
    let p = NoiseProcess::MarkovBurst {
        onset: 0.2,
        recovery: 0.1,
        affected: vec![0, 2],
        multiplier: 2.0,
    };
    let prefix = generate_noise_stream(&dem, &p, 16, 546).unwrap();
    let full = generate_noise_stream(&dem, &p, 64, 546).unwrap();
    assert_eq!(prefix, &full[..16]);
    generate_noise_stream(&dem, &p, 32, robustness_stream_seed(546, 17)).unwrap();
    assert_eq!(prefix, generate_noise_stream(&dem, &p, 16, 546).unwrap());
}
