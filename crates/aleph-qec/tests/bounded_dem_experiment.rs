//! Bounded experiments must score the entire deterministic stream or return an error.
use std::cell::{Cell, RefCell};

use aleph_qec::{
    run_dem_experiment, run_dem_experiment_with_options, sample_shots, Correction, Decoder,
    DemError, DemExperimentOptions, DetectorErrorModel, Error, LogicalErrorResult, NullDecoder,
    Result, Syndrome,
};

fn model() -> DetectorErrorModel {
    DetectorErrorModel {
        detectors: 4,
        observables: 3,
        errors: vec![
            DemError::new(0.19, vec![0, 1], vec![0, 2]),
            DemError::new(0.37, vec![1, 3], vec![1]),
            DemError::new(1.0, vec![2], vec![2]),
            DemError::new(0.0, vec![0, 2], vec![0, 1]),
        ],
    }
}

struct DetectorDecoder;
impl Decoder for DetectorDecoder {
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        Correction::new(vec![syndrome.is_fired(0), syndrome.is_fired(3), false])
    }
}

#[test]
fn fixed_seed_matches_full_artifact_path_across_batches_and_workers() {
    let dem = model();
    let shots = 103;
    for seed in [0, 11, u64::MAX] {
        // Independent scoring of the pre-existing full-shot artifact API, with a decoder
        // sensitive to the syndrome and all three observable columns.
        let (syndromes, truths) = sample_shots(&dem, shots, seed);
        let expected = LogicalErrorResult::try_new(
            shots,
            syndromes
                .iter()
                .zip(&truths)
                .filter(|(s, t)| DetectorDecoder.decode(s).observable_flips != **t)
                .count() as u64,
        )
        .unwrap();
        for workers in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                for batch_size in [1, 2, 7, 32, 103, 1000] {
                    let actual = run_dem_experiment_with_options(
                        &dem,
                        shots,
                        &DetectorDecoder,
                        seed,
                        &DemExperimentOptions { batch_size },
                    )
                    .unwrap();
                    assert_eq!(
                        actual, expected,
                        "seed={seed}, workers={workers}, batch={batch_size}"
                    );
                }
            });
        }
        assert_eq!(
            run_dem_experiment(&dem, shots, &DetectorDecoder, seed).unwrap(),
            expected
        );
    }
}

struct RecordingDecoder {
    batches: RefCell<Vec<Vec<Syndrome>>>,
    observables: usize,
}
impl Decoder for RecordingDecoder {
    fn decode(&self, _: &Syndrome) -> Correction {
        Correction::none(self.observables)
    }
    fn decode_batch(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
        self.batches.borrow_mut().push(syndromes.to_vec());
        Ok(syndromes.iter().map(|s| self.decode(s)).collect())
    }
}

#[test]
fn batch_bound_tail_and_absolute_stream_are_preserved() {
    let dem = model();
    for (shots, batch_size, sizes) in [(10, 4, vec![4, 4, 2]), (8, 4, vec![4, 4]), (0, 4, vec![])] {
        let dec = RecordingDecoder {
            batches: RefCell::default(),
            observables: dem.observables,
        };
        let result = run_dem_experiment_with_options(
            &dem,
            shots,
            &dec,
            55,
            &DemExperimentOptions { batch_size },
        )
        .unwrap();
        assert_eq!(result.shots, shots);
        let batches = dec.batches.into_inner();
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), sizes);
        assert_eq!(
            batches.into_iter().flatten().collect::<Vec<_>>(),
            sample_shots(&dem, shots, 55).0
        );
    }
}

struct Malformed {
    calls: Cell<usize>,
    fail_call: usize,
    count_delta: isize,
    width: usize,
}
impl Decoder for Malformed {
    fn decode(&self, _: &Syndrome) -> Correction {
        Correction::none(3)
    }
    fn decode_batch(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call != self.fail_call {
            return Ok(syndromes.iter().map(|s| self.decode(s)).collect());
        }
        let count = syndromes
            .len()
            .checked_add_signed(self.count_delta)
            .unwrap();
        Ok((0..count).map(|_| Correction::none(self.width)).collect())
    }
}

#[test]
fn short_long_empty_or_wrong_width_predictions_are_rejected_in_later_batches() {
    let dem = model();
    for (delta, width) in [(-1, 3), (1, 3), (-4, 3), (0, 0), (0, 2), (0, 4)] {
        let dec = Malformed {
            calls: Cell::new(0),
            fail_call: 1,
            count_delta: delta,
            width,
        };
        let err = run_dem_experiment_with_options(
            &dem,
            10,
            &dec,
            1,
            &DemExperimentOptions { batch_size: 4 },
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::DecoderContract { shot: 4, .. }),
            "{err}"
        );
        assert_eq!(
            dec.calls.get(),
            2,
            "stop immediately, never score the remaining tail"
        );
    }
}

#[test]
fn zero_batch_size_is_rejected_even_for_zero_shots() {
    for shots in [0, 1] {
        assert!(matches!(
            run_dem_experiment_with_options(
                &model(),
                shots,
                &DetectorDecoder,
                1,
                &DemExperimentOptions { batch_size: 0 }
            ),
            Err(Error::InvalidExperimentBatchSize)
        ));
    }
}

#[test]
fn zero_observables_require_exact_empty_corrections() {
    let dem = DetectorErrorModel {
        detectors: 0,
        observables: 0,
        errors: vec![],
    };
    assert_eq!(
        run_dem_experiment(&dem, 7, &NullDecoder::new(0), 1)
            .unwrap()
            .logical_errors,
        0
    );
    assert!(matches!(
        run_dem_experiment(&dem, 7, &NullDecoder::new(1), 1),
        Err(Error::DecoderContract { .. })
    ));
}

#[test]
fn huge_requested_count_does_not_materialize_all_shots() {
    struct Failing;
    impl Decoder for Failing {
        fn decode(&self, _: &Syndrome) -> Correction {
            unreachable!()
        }
        fn decode_batch(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
            assert_eq!(syndromes.len(), 2);
            Err(Error::Oracle("intentional failure".into()))
        }
    }
    let err = run_dem_experiment_with_options(
        &model(),
        u64::MAX,
        &Failing,
        1,
        &DemExperimentOptions { batch_size: 2 },
    )
    .unwrap_err();
    assert!(matches!(err, Error::Oracle(_)));
}

#[test]
fn deterministic_mechanisms_produce_exact_tallies() {
    let dem = DetectorErrorModel {
        detectors: 0,
        observables: 2,
        errors: vec![
            DemError::new(1.0, vec![], vec![1]),
            DemError::new(0.0, vec![], vec![0]),
        ],
    };
    for batch_size in [1, 9, 100] {
        let result = run_dem_experiment_with_options(
            &dem,
            23,
            &NullDecoder::new(2),
            1,
            &DemExperimentOptions { batch_size },
        )
        .unwrap();
        assert_eq!((result.shots, result.logical_errors), (23, 23));
    }
}

#[test]
#[ignore = "requires numpy + stim + pymatching; set PYMATCHING_PYTHON"]
fn pymatching_batch_boundaries_and_zero_observables() {
    use aleph_qec::PyMatchingOracle;
    for observables in [0, 3] {
        let dem = DetectorErrorModel {
            detectors: 2,
            observables,
            errors: vec![DemError::new(
                0.2,
                vec![0, 1],
                if observables == 0 { vec![] } else { vec![0, 2] },
            )],
        };
        let decoder = PyMatchingOracle::new(&dem);
        let (syndromes, truths) = sample_shots(&dem, 35, 565);
        let predictions = decoder.decode_batch(&syndromes).unwrap();
        assert_eq!(predictions.len(), truths.len());
        assert!(predictions
            .iter()
            .all(|p| p.observable_flips.len() == observables));
        let errors = predictions
            .iter()
            .zip(&truths)
            .filter(|(p, t)| p.observable_flips != **t)
            .count() as u64;
        let batched = run_dem_experiment_with_options(
            &dem,
            35,
            &decoder,
            565,
            &DemExperimentOptions { batch_size: 16 },
        )
        .unwrap();
        assert_eq!(batched, LogicalErrorResult::try_new(35, errors).unwrap());
    }
}

#[test]
fn default_limit_splits_campaign_and_reports_late_schema_index() {
    struct Inspect {
        calls: RefCell<Vec<usize>>,
    }
    impl Decoder for Inspect {
        fn decode(&self, _: &Syndrome) -> Correction {
            Correction::none(0)
        }
        fn decode_batch(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
            self.calls.borrow_mut().push(syndromes.len());
            Ok(syndromes.iter().map(|s| self.decode(s)).collect())
        }
    }
    let dem = DetectorErrorModel {
        detectors: 0,
        observables: 0,
        errors: vec![],
    };
    let decoder = Inspect {
        calls: RefCell::default(),
    };
    let size = DemExperimentOptions::default().batch_size;
    assert_eq!(
        run_dem_experiment(&dem, size as u64 + 3, &decoder, 1)
            .unwrap()
            .shots,
        size as u64 + 3
    );
    assert_eq!(*decoder.calls.borrow(), [size, 3]);

    struct LastWrong;
    impl Decoder for LastWrong {
        fn decode(&self, _: &Syndrome) -> Correction {
            Correction::none(3)
        }
        fn decode_batch(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
            let mut predictions: Vec<_> = syndromes.iter().map(|s| self.decode(s)).collect();
            predictions.last_mut().unwrap().observable_flips.pop();
            Ok(predictions)
        }
    }
    let err = run_dem_experiment_with_options(
        &model(),
        10,
        &LastWrong,
        1,
        &DemExperimentOptions { batch_size: 4 },
    )
    .unwrap_err();
    assert!(matches!(err, Error::DecoderContract { shot: 3, .. }));
}
