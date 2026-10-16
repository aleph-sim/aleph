//! Matching priors are rejected before building a decoder, consistently in debug/release.
use aleph_qec::{
    BpDecoder, Decoder, DemError, DetectorErrorModel, Error, MatchingGraph, MwpmDecoder, Syndrome,
    UnionFindDecoder,
};

fn model(prob: f64) -> DetectorErrorModel {
    DetectorErrorModel {
        detectors: 2,
        observables: 1,
        errors: vec![
            DemError::new(0.1, vec![0], vec![]),
            DemError::new(prob, vec![1], vec![0]),
        ],
    }
}

fn assert_prior_error<T>(result: aleph_qec::Result<T>, column: usize, prob: f64) {
    match result {
        Err(Error::InvalidMatchingPrior {
            column: actual,
            prob: rejected,
        }) => {
            assert_eq!(actual, column);
            assert_eq!(
                rejected.to_bits(),
                prob.to_bits(),
                "input must not be clamped"
            );
        }
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("unsupported prior accepted: {prob}"),
    }
}

#[test]
fn all_matching_constructors_reject_unsupported_priors() {
    for p in [
        0.5f64.next_up(),
        0.6,
        1.0,
        1.01,
        -0.1,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let dem = model(p);
        assert_prior_error(MatchingGraph::from_dem(&dem), 1, p);
        assert_prior_error(MwpmDecoder::new(&dem), 1, p);
        assert_prior_error(UnionFindDecoder::new(&dem), 1, p);
        assert_prior_error(UnionFindDecoder::new_weighted(&dem), 1, p);
        assert_prior_error(MwpmDecoder::new_with_split_components(&dem), 1, p);
        assert_prior_error(UnionFindDecoder::new_with_split_components(&dem), 1, p);
    }
}

#[test]
fn source_priors_are_checked_before_dropping_or_merging_edges() {
    // Two inadmissible 0.6 priors would XOR-merge to 0.48. Validate each source
    // column, not just the resulting graph, so invalid inputs cannot disappear.
    let dem = DetectorErrorModel {
        detectors: 1,
        observables: 0,
        errors: vec![
            DemError::new(0.6, vec![0], vec![]),
            DemError::new(0.6, vec![0], vec![]),
        ],
    };
    assert_prior_error(MatchingGraph::from_dem(&dem), 0, 0.6);
    for error in [
        DemError::new(0.6, vec![], vec![]),
        DemError::new(1.0, vec![], vec![0]),
        DemError::new(f64::NAN, vec![], vec![]),
        DemError::with_components(0.6, vec![(vec![0], vec![]), (vec![1], vec![0])]),
    ] {
        let p = error.prob;
        let dem = DetectorErrorModel {
            detectors: 2,
            observables: 1,
            errors: vec![error],
        };
        assert_prior_error(MatchingGraph::from_dem(&dem), 0, p);
        assert_prior_error(MwpmDecoder::new_with_split_components(&dem), 0, p);
    }
}

#[test]
fn split_constructor_errors_use_split_column_indices() {
    let dem = DetectorErrorModel {
        detectors: 2,
        observables: 0,
        errors: vec![
            DemError::with_components(0.1, vec![(vec![0], vec![]), (vec![1], vec![])]),
            DemError::new(0.7, vec![], vec![]),
        ],
    };
    assert_prior_error(MwpmDecoder::new(&dem), 1, 0.7);
    assert_prior_error(MwpmDecoder::new_with_split_components(&dem), 2, 0.7);
}

#[test]
fn supported_boundary_priors_decode_and_project_in_debug_and_release() {
    for p in [
        f64::from_bits(1),
        1e-320,
        f64::MIN_POSITIVE,
        0.1,
        0.5f64.next_down(),
        0.5,
    ] {
        // Distinct boundary edges give unique supports even when their weight is zero.
        let dem = DetectorErrorModel {
            detectors: 2,
            observables: 2,
            errors: vec![
                DemError::new(0.0, vec![0, 1], vec![]),
                DemError::new(p, vec![0], vec![0]),
                DemError::new(p, vec![1], vec![1]),
            ],
        };
        let graph = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(graph.num_columns(), 3);
        assert_eq!(graph.edges().len(), 2);
        assert!(graph
            .edges()
            .iter()
            .all(|e| e.weight.is_finite() && e.weight >= 0.0));
        let mwpm = MwpmDecoder::new(&dem).unwrap();
        let uf = UnionFindDecoder::new(&dem).unwrap();
        let weighted = UnionFindDecoder::new_weighted(&dem).unwrap();
        for mask in 0..4 {
            let bits = vec![mask & 1 != 0, mask & 2 != 0];
            let s = Syndrome::from_bits(&bits);
            for (decoder, estimate) in [
                (&mwpm as &dyn Decoder, mwpm.decode_errors(&s)),
                (&uf as &dyn Decoder, uf.decode_errors(&s)),
                (&weighted as &dyn Decoder, weighted.decode_errors(&s)),
            ] {
                assert_eq!(estimate, vec![0, bits[0] as u8, bits[1] as u8]);
                assert_eq!(decoder.decode(&s).observable_flips, bits);
            }
        }
    }
}

#[test]
fn zero_negative_zero_and_empty_models_are_supported() {
    let empty = DetectorErrorModel::default();
    assert!(MatchingGraph::from_dem(&empty).unwrap().edges().is_empty());
    MwpmDecoder::new(&empty).unwrap();
    UnionFindDecoder::new_weighted(&empty).unwrap();
    for p in [0.0, -0.0] {
        let dem = model(p);
        let graph = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(graph.num_columns(), 2);
        assert_eq!(graph.edges().len(), 1);
        assert_eq!(graph.edges()[0].column, 0);
    }
}

#[test]
fn bp_and_dem_construction_keep_the_full_probability_range() {
    for p in [0.6, 1.0] {
        let dem =
            DetectorErrorModel::from_check_matrices(1, 1, &[vec![0]], &[vec![0]], &[p]).unwrap();
        let _bp = BpDecoder::new(&dem);
        assert_eq!(dem.errors[0].prob.to_bits(), p.to_bits());
    }
    let dem = DetectorErrorModel::parse("error(0.6) D0 L0").unwrap();
    assert_prior_error(MwpmDecoder::new(&dem), 0, 0.6);
}

#[test]
fn ordinary_supported_weights_remain_bit_identical() {
    for p in [1e-300, 0.0001, 0.01, 0.2, 0.49, 0.5f64.next_down(), 0.5] {
        let dem = DetectorErrorModel {
            detectors: 1,
            observables: 0,
            errors: vec![DemError::new(p, vec![0], vec![])],
        };
        let graph = MatchingGraph::from_dem(&dem).unwrap();
        let legacy = ((1.0 - p) / p).ln();
        assert_eq!(graph.edges()[0].weight.to_bits(), legacy.to_bits());
    }
}
