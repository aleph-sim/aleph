//! Independent reconstruction, exhaustive tiny reductions, sampler stream and artifact checks.
use aleph_qec::*;
use rayon::prelude::*;
use std::{collections::BTreeSet, time::Duration};
fn budget() -> MinimizeBudget {
    MinimizeBudget {
        decoder_calls: 10000,
        time: Duration::from_secs(30),
    }
}
fn tiny() -> DetectorErrorModel {
    DetectorErrorModel::parse(
        "error(0.5) D0 L0\nerror(0.5) D0 L1\nerror(0.5) L0\nerror(0.5) D0 L0 ^ D1 L1",
    )
    .unwrap()
}
#[test]
fn exhaustive_tiny_reconstruction_and_one_minimality() {
    let dem = tiny();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let mut seen = BTreeSet::new();
    for shot in 0..2048 {
        let witness = capture_shot(&decoder, 543, shot).unwrap();
        if !seen.insert(witness.faults.clone()) {
            continue;
        }
        let mut expected_d = [false; 2];
        let mut expected_o = [false; 2];
        for &f in &witness.faults {
            for &d in &dem.errors[f].dets {
                expected_d[d as usize] ^= true;
            }
            for &o in &dem.errors[f].obs {
                expected_o[o as usize] ^= true;
            }
        }
        assert_eq!(
            reconstruct_faults(&dem, &witness.faults).unwrap(),
            (Syndrome::from_bits(&expected_d), expected_o.to_vec())
        );
        assert_eq!(
            FailureWitness::from_json(&witness.to_json().unwrap()).unwrap(),
            witness
        );
        if witness.outcome.logical_failure {
            let result = decoder.minimize(&witness, budget()).unwrap();
            assert_eq!(result.status, MinimizeStatus::OneMinimal);
            assert!(result.witness.outcome.logical_failure);
            assert_eq!(result.witness.original_faults, witness.original_faults);
            assert_eq!(result.witness.dem, witness.dem);
            for i in 0..result.witness.faults.len() {
                let mut faults = result.witness.faults.clone();
                faults.remove(i);
                let (s, t) = reconstruct_faults(&dem, &faults).unwrap();
                assert_eq!(
                    MwpmDecoder::new_with_split_components(&dem)
                        .unwrap()
                        .decode(&s)
                        .observable_flips,
                    t
                );
            }
        }
    }
    assert_eq!(seen.len(), 16);
}
#[test]
fn capture_matches_ordinary_sampler_and_worker_counts() {
    let dem = tiny();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let (syndromes, truths) = sample_shots(&dem, 128, 7);
    let run = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| {
                (0..128)
                    .into_par_iter()
                    .map(|i| capture_shot(&decoder, 7, i).unwrap())
                    .collect::<Vec<_>>()
            })
    };
    let single = run(1);
    assert_eq!(single, run(4));
    for (i, w) in single.iter().enumerate() {
        assert_eq!(w.outcome.syndrome, syndromes[i].fired);
        assert_eq!(w.outcome.truth, truths[i]);
    }
}
#[test]
fn budgets_tampering_and_incompatible_replay() {
    let dem = DetectorErrorModel::parse("error(0.1) L0").unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let w = (0..100)
        .map(|i| capture_shot(&decoder, 11, i).unwrap())
        .find(|w| w.outcome.logical_failure)
        .unwrap();
    assert!(w.outcome.syndrome.is_empty());
    for calls in [0, 1] {
        let r = decoder
            .minimize(
                &w,
                MinimizeBudget {
                    decoder_calls: calls,
                    ..budget()
                },
            )
            .unwrap();
        assert_eq!(r.status, MinimizeStatus::Incomplete);
        assert_eq!(r.decoder_calls, calls);
        assert!(r.witness.outcome.logical_failure);
    }
    let r = decoder
        .minimize(
            &w,
            MinimizeBudget {
                time: Duration::ZERO,
                ..budget()
            },
        )
        .unwrap();
    assert_eq!(r.status, MinimizeStatus::Incomplete);
    let passing = (0..100)
        .map(|i| capture_shot(&decoder, 11, i).unwrap())
        .find(|w| !w.outcome.logical_failure)
        .unwrap();
    assert!(decoder.minimize(&passing, budget()).is_err());
    let mut bad = w.clone();
    bad.dem.push(' ');
    assert!(bad.to_json().is_err());
    let mut bad = w.clone();
    bad.faults.push(0);
    assert!(bad.to_json().is_err());
    let mut bad = w.clone();
    bad.outcome.prediction[0] ^= true;
    assert!(FailureWitness::from_json(&serde_json::to_string(&bad).unwrap()).is_err());
    let mut bad = w.clone();
    bad.shot = passing.shot;
    assert!(bad.to_json().is_err());
    let mut bad = w.clone();
    bad.version = 2;
    assert!(bad.to_json().is_err());
    assert!(WitnessDecoder::new(&tiny(), DecoderSpec::Mwpm)
        .unwrap()
        .replay(&w)
        .is_err());
    assert_ne!(w.signature().unwrap(), passing.signature().unwrap());
}
#[test]
fn relay_seed_is_stable_and_nonconvergence_is_not_failure() {
    let dem = BBCode::gross().code_capacity_dem(0.03);
    for spec in [
        DecoderSpec::Relay {
            legs: 2,
            iters: 2,
            seed: 19,
            early_exit: false,
        },
        DecoderSpec::FixedRelay {
            legs: 2,
            iters: 2,
            seed: 19,
            msg_bits: 8,
            frac_bits: 3,
            early_exit: false,
        },
    ] {
        let decoder = WitnessDecoder::new(&dem, spec.clone()).unwrap();
        let other = WitnessDecoder::new(&dem, spec).unwrap();
        for shot in 0..10 {
            let w = capture_shot(&decoder, 43, shot).unwrap();
            assert_eq!(decoder.replay(&w).unwrap(), other.replay(&w).unwrap());
            assert_eq!(
                w.outcome.logical_failure,
                w.outcome.truth != w.outcome.prediction
            );
            assert_eq!(FailureWitness::from_json(&w.to_json().unwrap()).unwrap(), w);
        }
    }
    assert!(WitnessDecoder::new(
        &dem,
        DecoderSpec::FixedRelay {
            legs: 1,
            iters: 1,
            seed: 0,
            msg_bits: 32,
            frac_bits: 3,
            early_exit: false
        }
    )
    .is_err());
}

#[test]
fn invalid_diagnostics_are_separate_and_fixed_codec_is_checked() {
    let mut dem = BBCode::gross().code_capacity_dem(0.03);
    dem.observables = 0;
    for e in &mut dem.errors {
        e.obs.clear();
    }
    let spec = DecoderSpec::FixedRelay {
        legs: 1,
        iters: 1,
        seed: 19,
        msg_bits: 8,
        frac_bits: 3,
        early_exit: false,
    };
    let decoder = WitnessDecoder::new(&dem, spec).unwrap();
    let w = (0..100)
        .map(|i| capture_shot(&decoder, 543, i).unwrap())
        .find(|w| w.outcome.valid == Some(false))
        .unwrap();
    assert!(!w.outcome.logical_failure);
    let (syn, reference) = decoder.fixed_reference(&w).unwrap();
    let rec = read_ref(&mut reference.as_slice()).unwrap();
    assert_eq!(rec[0].true_obs, 0);
    assert_eq!(rec[0].sw_obs, 0);
    assert!(!rec[0].valid);
    let reconstructed: Vec<u32> = syn
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(j, b)| {
            let word = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            (0..32).filter_map(move |i| (word >> i & 1 != 0).then_some((j * 32 + i) as u32))
        })
        .collect();
    assert_eq!(reconstructed, w.outcome.syndrome);
    assert_eq!(rec[0].iters, 1);
    let dem = DetectorErrorModel::parse("error(0.1) L0").unwrap();
    assert!(WitnessDecoder::new(&dem, DecoderSpec::Mwpm)
        .unwrap()
        .fixed_reference(
            &capture_shot(
                &WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap(),
                543,
                0
            )
            .unwrap()
        )
        .is_err());
}

#[test]
fn real_fixtures_replay_certify_and_match_reference_bytes() {
    for json in [
        include_str!("data/witness/surface-d3-r3.json"),
        include_str!("data/witness/bb-gross-capacity.json"),
    ] {
        let w = FailureWitness::from_json(json).unwrap();
        assert!(w.outcome.logical_failure);
        assert_eq!(
            w.minimization.as_ref().unwrap().status,
            MinimizeStatus::OneMinimal
        );
        let dem = DetectorErrorModel::parse(&w.dem).unwrap();
        let decoder = WitnessDecoder::new(&dem, w.decoder.clone()).unwrap();
        if matches!(w.decoder, DecoderSpec::FixedRelay { .. }) {
            let (syn, reference) = decoder.fixed_reference(&w).unwrap();
            assert_eq!(syn, include_bytes!("data/witness/bb-gross-capacity.syn"));
            assert_eq!(
                reference,
                include_bytes!("data/witness/bb-gross-capacity.ref")
            );
            let relay = WitnessDecoder::new(
                &dem,
                DecoderSpec::Relay {
                    legs: 6,
                    iters: 10,
                    seed: 0x5E1A_4B9C,
                    early_exit: false,
                },
            )
            .unwrap();
            assert!(relay.replay(&w).unwrap().logical_failure);
        } else {
            assert!(
                WitnessDecoder::new(&dem, DecoderSpec::UnionFind)
                    .unwrap()
                    .replay(&w)
                    .unwrap()
                    .logical_failure
            );
        }
        // A forged certification for the unreduced original set must be rejected.
        let mut fake = w.clone();
        fake.faults = fake.original_faults.clone();
        fake.outcome = fake.original_outcome.clone();
        assert!(FailureWitness::from_json(&serde_json::to_string(&fake).unwrap()).is_err());
        // Incomplete work remains labelled after round-trip, even when starting from a certified fixture.
        let incomplete = decoder
            .minimize(
                &w,
                MinimizeBudget {
                    decoder_calls: 0,
                    ..budget()
                },
            )
            .unwrap();
        assert_eq!(
            FailureWitness::from_json(&incomplete.witness.to_json().unwrap())
                .unwrap()
                .minimization
                .unwrap()
                .status,
            MinimizeStatus::Incomplete
        );
    }
}

#[test]
fn malformed_inputs_return_errors_and_pure_logical_tiny_models_certify() {
    for text in [
        "error(0.5) L0\nerror(0.5) L1\nerror(0.5) L0",
        "error(0.5) D0\nerror(0.5) D0 L0",
    ] {
        let dem = DetectorErrorModel::parse(text).unwrap();
        let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
        let mut seen = BTreeSet::new();
        for shot in 0..256 {
            let w = capture_shot(&decoder, 543, shot).unwrap();
            if !seen.insert(w.faults.clone()) {
                continue;
            }
            if w.outcome.logical_failure {
                let reduced = decoder.minimize(&w, budget()).unwrap();
                assert_eq!(reduced.status, MinimizeStatus::OneMinimal);
                FailureWitness::from_json(&reduced.witness.to_json().unwrap()).unwrap();
            }
        }
        assert_eq!(seen.len(), 1 << dem.errors.len());
    }
    let mut dem = tiny();
    dem.errors[0].prob = f64::NAN;
    assert!(WitnessDecoder::new(&dem, DecoderSpec::Mwpm).is_err());
    let mut dem = tiny();
    dem.errors[0].dets.push(100);
    assert!(WitnessDecoder::new(&dem, DecoderSpec::Mwpm).is_err());
    let dem = tiny();
    assert!(reconstruct_faults(&dem, &[dem.errors.len()]).is_err());
    assert!(reconstruct_faults(&dem, &[1, 0]).is_err());
    assert!(WitnessDecoder::new(
        &dem,
        DecoderSpec::Relay {
            legs: 0,
            iters: 1,
            seed: 0,
            early_exit: false
        }
    )
    .is_err());
}

#[test]
fn signatures_group_exact_examples_across_shots_but_keep_decoder_choices() {
    let dem = DetectorErrorModel::parse("error(0.1) L0").unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let witnesses: Vec<_> = (0..100)
        .map(|shot| capture_shot(&decoder, 543, shot).unwrap())
        .filter(|w| w.outcome.logical_failure)
        .collect();
    assert!(witnesses.len() > 1);
    assert_ne!(witnesses[0].shot, witnesses[1].shot);
    assert_eq!(
        witnesses[0].signature().unwrap(),
        witnesses[1].signature().unwrap()
    );
    let uf = WitnessDecoder::new(&dem, DecoderSpec::UnionFind).unwrap();
    let other = capture_shot(&uf, 543, witnesses[0].shot).unwrap();
    assert_ne!(
        other.signature().unwrap(),
        witnesses[0].signature().unwrap()
    );
}
