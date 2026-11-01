//! Exact tiny enumeration, independent conditional frequencies, coverage and provenance.
use aleph_qec::*;
use rayon::prelude::*;
fn model(probs: &[f64]) -> DetectorErrorModel {
    DetectorErrorModel {
        detectors: 2,
        observables: 2,
        errors: probs
            .iter()
            .enumerate()
            .map(|(i, &p)| DemError::new(p, vec![(i % 2) as u32], vec![(i % 2) as u32]))
            .collect(),
    }
}
fn options(budgets: Vec<u64>, enumerate_small: bool) -> StratifiedOptions {
    StratifiedOptions {
        budgets,
        max_decoder_calls: 100000,
        seed: 544,
        confidence: 0.95,
        enumerate_small,
        witness_limit: 4,
        target_width: None,
    }
}
fn probability(dem: &DetectorErrorModel, mask: usize) -> f64 {
    dem.errors
        .iter()
        .enumerate()
        .map(|(i, e)| {
            if mask >> i & 1 != 0 {
                e.prob
            } else {
                1.0 - e.prob
            }
        })
        .product()
}
#[test]
fn exact_weights_for_equal_heterogeneous_and_deterministic_models() {
    for probs in [
        vec![],
        vec![0.0, 1.0, 0.25, 0.5, 0.75],
        vec![0.25; 5],
        vec![0.0, 1.0, 1.0],
    ] {
        let dem = model(&probs);
        let plan = FaultCountPlan::new(&dem, probs.len(), 1000).unwrap();
        let mut weights = vec![0.0; probs.len() + 1];
        for mask in 0..1usize << probs.len() {
            weights[mask.count_ones() as usize] += probability(&dem, mask);
        }
        for (k, &weight) in weights.iter().enumerate() {
            let b = plan.weight(k).unwrap();
            assert!(
                b.lower <= weight && weight <= b.upper,
                "k={k} {b:?} exact={weight}"
            );
            if weight > 0.0 {
                assert!((plan.log_weight(k).unwrap().exp() - weight).abs() < 1e-14);
            } else {
                assert!(plan.log_weight(k).is_none());
            }
        }
        assert_eq!(
            plan.tail(),
            ProbabilityBounds {
                lower: 0.0,
                upper: 0.0
            }
        );
    }
}
#[test]
fn conditional_sampling_is_not_uniform_for_unequal_probabilities() {
    let dem = model(&[0.01, 0.2, 0.7]);
    let plan = FaultCountPlan::new(&dem, 2, 100).unwrap();
    for k in [1, 2] {
        let mut counts = [0usize; 8];
        let n = 30000;
        for shot in 0..n {
            let f = plan.sample(k, 42, shot).unwrap();
            assert_eq!(f.len(), k);
            let mask = f.iter().fold(0, |a, &i| a | (1 << i));
            counts[mask] += 1;
        }
        let weight = plan.log_weight(k).unwrap().exp();
        for (mask, &observed) in counts.iter().enumerate() {
            let expected = if mask.count_ones() as usize == k {
                probability(&dem, mask) / weight
            } else {
                0.0
            };
            let rate = observed as f64 / n as f64;
            let tolerance = 6.0 * (expected * (1.0 - expected) / n as f64).sqrt() + 3.0 / n as f64;
            assert!(
                (rate - expected).abs() < tolerance,
                "{k} {mask} {rate} != {expected}"
            );
        }
    }
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (0..100)
                    .into_par_iter()
                    .map(|s| plan.sample(2, 42, s).unwrap())
                    .collect::<Vec<_>>()
            })
    };
    assert_eq!(run(1), run(4));
}
#[test]
fn deterministic_faults_enumeration_and_underflow() {
    let dem = model(&[0.0, 1.0, 1e-250, 2e-250]);
    let plan = FaultCountPlan::new(&dem, 3, 100).unwrap();
    assert!(plan.log_weight(3).unwrap() < -1000.0);
    assert_eq!(plan.log_weight(3).unwrap().exp(), 0.0);
    let w = plan.weight(3).unwrap();
    assert_eq!(w.lower, 0.0);
    assert!(w.upper > 0.0);
    for i in 0..100 {
        assert_eq!(plan.sample(3, 544, i).unwrap(), vec![1, 2, 3]);
        assert_eq!(plan.sample(1, 544, i).unwrap(), vec![1]);
        assert_eq!(plan.sample(2, 544, i).unwrap().len(), 2);
    }
    assert_eq!(plan.enumerated_faults(2, 0).unwrap(), vec![1, 2]);
    assert_eq!(plan.enumerated_faults(2, 1).unwrap(), vec![1, 3]);
    assert!(plan.enumerated_faults(2, 2).is_err());
    assert!(plan.sample(0, 0, 0).is_err());
    assert!(FaultCountPlan::new(&dem, 0, 100).is_err());
    assert!(FaultCountPlan::new(&dem, 3, 1).is_err());
    let plan = FaultCountPlan::new(&model(&[1e-12; 3]), 1, 100).unwrap();
    assert!(plan.tail().upper > 0.0 && plan.tail().upper < 1e-22);
}
#[test]
fn exact_campaign_agrees_with_enumeration_and_legacy_witnesses_survive() {
    let dem = DetectorErrorModel::parse(
        "error(0.25) D0 L0\nerror(0.125) D0 L1\nerror(0.0625) L0\nerror(0.25) D0 L0 ^ D1 L1",
    )
    .unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let plan = FaultCountPlan::new(&dem, 4, 100).unwrap();
    let result = run_stratified(&plan, &decoder, &options(vec![20; 5], true)).unwrap();
    assert_eq!(result.decoder_calls, 16);
    let mut exact = 0.0;
    for mask in 0..16 {
        let faults = (0..4).filter(|i| mask >> i & 1 != 0).collect::<Vec<_>>();
        if decoder.decode_faults(&faults).unwrap().logical_failure {
            exact += probability(&dem, mask);
        }
    }
    assert!(result.failure_probability.lower <= exact && exact <= result.failure_probability.upper);
    assert!(result.failure_probability.upper - result.failure_probability.lower < 1e-12);
    assert!(!result.witnesses.is_empty());
    for w in &result.witnesses {
        assert_eq!(w.version, 2);
        assert_eq!(
            FailureWitness::from_json(&w.to_json().unwrap()).unwrap(),
            *w
        );
        let minimized = decoder
            .minimize(
                w,
                MinimizeBudget {
                    decoder_calls: 100,
                    time: std::time::Duration::from_secs(5),
                },
            )
            .unwrap();
        assert_eq!(minimized.status, MinimizeStatus::OneMinimal);
        FailureWitness::from_json(&minimized.witness.to_json().unwrap()).unwrap();
    }
    let ordinary = capture_shot(&decoder, 544, 0).unwrap();
    assert_eq!(ordinary.version, 1);
    assert!(ordinary.sampling.is_none());
    FailureWitness::from_json(&ordinary.to_json().unwrap()).unwrap();
    let conditional = capture_stratified_shot(&plan, &decoder, 2, 544, 0).unwrap();
    assert_eq!(conditional.original_faults.len(), 2);
    FailureWitness::from_json(&conditional.to_json().unwrap()).unwrap();
    let mut bad = conditional;
    bad.shot += 1;
    let serialized = serde_json::to_string(&bad).unwrap();
    if plan.sample(2, 544, bad.shot).unwrap() != bad.original_faults {
        assert!(FailureWitness::from_json(&serialized).is_err());
    }
}
#[test]
fn zero_failures_omitted_mass_and_budget_status_are_explicit() {
    let dem = DetectorErrorModel::parse("error(0.01) D0 L0\nerror(0.02) D1 L0").unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let plan = FaultCountPlan::new(&dem, 1, 100).unwrap();
    let mut o = options(vec![10, 0], false);
    o.target_width = Some(1e-10);
    let r = run_stratified(&plan, &decoder, &o).unwrap();
    assert_eq!(r.status, StratifiedStatus::BudgetExhausted);
    assert_eq!(r.strata[0].failures, 0);
    assert!(r.strata[0].conditional.upper > 0.0);
    assert!(r.unsampled_mass.lower > 0.029);
    assert!(r.failure_probability.upper >= r.unsampled_mass.lower);
    let mut bad = o.clone();
    bad.max_decoder_calls = 1;
    assert!(run_stratified(&plan, &decoder, &bad).is_err());
    bad = o.clone();
    bad.confidence = f64::NAN;
    assert!(run_stratified(&plan, &decoder, &bad).is_err());
    o.enumerate_small = true;
    o.budgets = vec![1, 2];
    o.target_width = Some(0.001);
    let exact = run_stratified(&plan, &decoder, &o).unwrap();
    assert_eq!(exact.status, StratifiedStatus::TargetMet);
    assert_eq!(exact.decoder_calls, 3);
}
#[test]
fn repeated_fixed_budget_coverage_on_tiny_heterogeneous_dem() {
    let dem = DetectorErrorModel::parse(
        "error(0.05) D0 L0\nerror(0.3) D0 L1\nerror(0.15) L0\nerror(0.2) D1 L1",
    )
    .unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let plan = FaultCountPlan::new(&dem, 4, 100).unwrap();
    let mut truth = 0.0;
    for mask in 0..16 {
        let faults = (0..4).filter(|i| mask >> i & 1 != 0).collect::<Vec<_>>();
        if decoder.decode_faults(&faults).unwrap().logical_failure {
            truth += probability(&dem, mask);
        }
    }
    let mut covered = 0;
    let trials = 120;
    for seed in 0..trials {
        let mut o = options(vec![32; 5], false);
        o.seed = seed;
        o.witness_limit = 0;
        let r = run_stratified(&plan, &decoder, &o).unwrap();
        assert_eq!(r.decoder_calls, 160);
        covered += usize::from(
            r.failure_probability.lower <= truth && truth <= r.failure_probability.upper,
        );
    }
    assert!(covered >= 110, "coverage {covered}/{trials}");
}

#[test]
fn enumeration_unranking_covers_each_supported_set_once() {
    let plan = FaultCountPlan::new(&model(&[0.2; 6]), 6, 200).unwrap();
    for k in 0..=6 {
        let mut sets = std::collections::BTreeSet::new();
        let mut rank = 0;
        while let Ok(faults) = plan.enumerated_faults(k, rank) {
            assert_eq!(faults.len(), k);
            assert!(sets.insert(faults));
            rank += 1;
        }
        assert_eq!(sets.len(), [1, 6, 15, 20, 15, 6, 1][k]);
    }
    // The cap must include weight storage even with no stochastic suffix rows.
    assert!(FaultCountPlan::new(&model(&[0.0; 100]), 100, 20).is_err());
}

#[test]
fn impossible_strata_are_not_silently_sampled_and_invalid_inputs_spend_no_calls() {
    let dem = model(&[0.0, 1.0, 0.1]);
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm);
    assert!(decoder.is_err()); // Matching priors intentionally exclude p=1.
    let decoder = WitnessDecoder::new(
        &dem,
        DecoderSpec::Relay {
            legs: 1,
            iters: 1,
            seed: 0,
            early_exit: false,
        },
    )
    .unwrap();
    let plan = FaultCountPlan::new(&dem, 3, 100).unwrap();
    let good = options(vec![0, 2, 2, 0], true);
    let result = run_stratified(&plan, &decoder, &good).unwrap();
    assert_eq!(result.decoder_calls, 2);
    assert_eq!(result.strata[0].mode, StratumMode::Impossible);
    assert_eq!(result.strata[3].mode, StratumMode::Impossible);
    let mut bad = good.clone();
    bad.budgets[0] = 1;
    assert!(run_stratified(&plan, &decoder, &bad).is_err());
    let mut nan = model(&[0.1]);
    nan.errors[0].prob = f64::NAN;
    assert!(FaultCountPlan::new(&nan, 1, 100).is_err());
    assert!(run_stratified(
        &plan,
        &WitnessDecoder::new(&model(&[0.2]), DecoderSpec::Mwpm).unwrap(),
        &good
    )
    .is_err());
}

#[test]
fn automatic_routing_is_pre_observation_and_preserves_selected_estimator() {
    for (name, dem, spec, k, expected) in [
        (
            "surface-low",
            {
                let e = SurfaceCode::new(3).memory_z_experiment(3);
                build_dem(&e.annotated, &e.phenomenological_mechanisms(0.0001, 0.0001)).unwrap()
            },
            DecoderSpec::Mwpm,
            5,
            RareEventMethod::Stratified,
        ),
        (
            "surface-accessible",
            {
                let e = SurfaceCode::new(3).memory_z_experiment(3);
                build_dem(&e.annotated, &e.phenomenological_mechanisms(0.03, 0.03)).unwrap()
            },
            DecoderSpec::Mwpm,
            5,
            RareEventMethod::Direct,
        ),
        (
            "bb-low",
            BBCode::gross().code_capacity_dem(0.001),
            DecoderSpec::Relay {
                legs: 6,
                iters: 10,
                seed: 0x5E1A_4B9C,
                early_exit: false,
            },
            5,
            RareEventMethod::Stratified,
        ),
        (
            "bb-accessible",
            BBCode::gross().code_capacity_dem(0.03),
            DecoderSpec::Relay {
                legs: 6,
                iters: 10,
                seed: 0x5E1A_4B9C,
                early_exit: false,
            },
            8,
            RareEventMethod::Direct,
        ),
    ] {
        let plan = FaultCountPlan::new(&dem, k, 4_000_000).unwrap();
        let decoder = WitnessDecoder::new(&dem, spec).unwrap();
        let mut opts = options(vec![256; k + 1], true);
        opts.target_width = Some(1e-6);
        let decision = select_rare_event_method(&plan, &opts).unwrap();
        assert_eq!(decision.method, expected, "{name}: {decision:?}");
        opts.seed += 7;
        opts.witness_limit = 0;
        let other = select_rare_event_method(&plan, &opts).unwrap();
        assert_eq!(
            serde_json::to_value(&decision).unwrap(),
            serde_json::to_value(&other).unwrap()
        );
        let auto = run_automatic_rare_event(&plan, &decoder, &opts).unwrap();
        match expected {
            RareEventMethod::Direct => {
                let manual =
                    run_dem_experiment(&dem, decision.declared_decoder_calls, &decoder, opts.seed)
                        .unwrap();
                assert_eq!(auto.direct_failures, Some(manual.logical_errors));
                let ci = manual.interval95().unwrap();
                assert_eq!(
                    auto.failure_probability,
                    ProbabilityBounds {
                        lower: ci.lower,
                        upper: ci.upper
                    }
                );
                assert_eq!(auto.decoder_calls, decision.declared_decoder_calls);
                assert!(auto.stratified.is_none());
            }
            RareEventMethod::Stratified => {
                let manual = run_stratified(&plan, &decoder, &opts).unwrap();
                assert_eq!(auto.failure_probability, manual.failure_probability);
                assert_eq!(auto.decoder_calls, manual.decoder_calls);
                assert_eq!(auto.status, manual.status);
                assert!(auto.direct_failures.is_none());
                assert!(auto.stratified.is_some());
            }
        }
    }
}

#[test]
fn automatic_preflight_and_zero_budget_are_explicit() {
    let dem = model(&[0.1, 0.2]);
    let plan = FaultCountPlan::new(&dem, 2, 100).unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let mut opts = options(vec![0; 3], false);
    let auto = run_automatic_rare_event(&plan, &decoder, &opts).unwrap();
    assert_eq!(auto.decision.method, RareEventMethod::Direct);
    assert_eq!(auto.decoder_calls, 0);
    assert_eq!(
        auto.failure_probability,
        ProbabilityBounds {
            lower: 0.0,
            upper: 1.0
        }
    );
    opts.confidence = 0.9;
    opts.budgets = vec![3; 3];
    let auto = run_automatic_rare_event(&plan, &decoder, &opts).unwrap();
    let ci = BinomialInterval95::from_counts(9, auto.direct_failures.unwrap()).unwrap();
    assert!(auto.failure_probability.upper < ci.upper);
    opts.max_decoder_calls = 8;
    assert!(select_rare_event_method(&plan, &opts).is_err());
    opts.max_decoder_calls = 100;
    opts.target_width = Some(f64::NAN);
    assert!(run_automatic_rare_event(&plan, &decoder, &opts).is_err());
}
