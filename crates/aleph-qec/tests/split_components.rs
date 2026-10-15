//! Per-part error columns must project onto the same detector/observable supports as edges.
use aleph_qec::{
    Decoder, DemError, DetectorErrorModel, Error, MatchingGraph, MwpmDecoder, Syndrome,
    UnionFindDecoder,
};

#[test]
fn split_retains_order_priors_counts_empty_parts_and_independent_parity() {
    let mut dem = DetectorErrorModel {
        detectors: 6,
        observables: 5,
        errors: vec![
            DemError::new(0.0, vec![4, 4], vec![3, 3]),
            DemError::with_components(
                0.13,
                vec![(vec![1, 0, 0], vec![0, 0, 2]), (vec![2, 1], vec![1, 2])],
            ),
            DemError::with_components(1.0, vec![(vec![], vec![0]), (vec![], vec![])]),
            DemError::new(0.2, vec![3], vec![4]),
        ],
    };
    let before = dem.clone();
    let split = dem.split_components();
    assert_eq!(dem, before, "source DEM must remain unchanged");
    assert_eq!((split.detectors, split.observables), (6, 5));
    assert_eq!(
        split.errors,
        vec![
            before.errors[0].clone(),
            DemError::new(0.13, vec![1], vec![2]),
            DemError::new(0.13, vec![1, 2], vec![1, 2]),
            DemError::new(1.0, vec![], vec![0]),
            DemError::new(1.0, vec![], vec![]),
            before.errors[3].clone(),
        ]
    );
    assert!(split.errors.iter().all(|e| e.components.is_empty()));
    assert_eq!(split.split_components(), split, "idempotence");
    dem.errors[1].components[0].0.push(5);
    assert_eq!(
        split.errors[1].dets,
        [1],
        "returned model owns its supports"
    );
}

#[test]
fn undecomposed_and_empty_models_are_unchanged() {
    for dem in [
        DetectorErrorModel::default(),
        DetectorErrorModel {
            detectors: 5,
            observables: 2,
            errors: vec![
                DemError::new(0.2, vec![0, 1], vec![1]),
                DemError::new(0.0, vec![], vec![]),
            ],
        },
    ] {
        assert_eq!(dem.split_components(), dem);
    }
}

fn project(dem: &DetectorErrorModel, estimate: &[u8]) -> (Vec<bool>, Vec<bool>) {
    assert_eq!(estimate.len(), dem.errors.len());
    let (mut detectors, mut observables) =
        (vec![false; dem.detectors], vec![false; dem.observables]);
    for (&bit, error) in estimate.iter().zip(&dem.errors) {
        assert!(bit <= 1);
        if bit != 0 {
            for &d in &error.dets {
                detectors[d as usize] ^= true;
            }
            for &o in &error.obs {
                observables[o as usize] ^= true;
            }
        }
    }
    (detectors, observables)
}

#[test]
fn split_constructors_fix_shared_columns_for_all_syndromes_without_ties() {
    let dem = DetectorErrorModel::parse("error(0.1) D0 L0 ^ D1 L1\nerror(0.17) D2 L0").unwrap();
    let split = dem.split_components();
    let mwpm = MwpmDecoder::new_with_split_components(&dem).unwrap();
    let uf = UnionFindDecoder::new_with_split_components(&dem).unwrap();
    let weighted = UnionFindDecoder::new_with_split_components(&dem)
        .unwrap()
        .weighted(true);
    let reference = MwpmDecoder::new(&split).unwrap();
    for mask in 0..8 {
        let bits: Vec<_> = (0..3).map(|d| mask & (1 << d) != 0).collect();
        let syndrome = Syndrome::from_bits(&bits);
        for (decoder, estimate) in [
            (&mwpm as &dyn Decoder, mwpm.decode_errors(&syndrome)),
            (&uf as &dyn Decoder, uf.decode_errors(&syndrome)),
            (&weighted as &dyn Decoder, weighted.decode_errors(&syndrome)),
        ] {
            let (detectors, observables) = project(&split, &estimate);
            assert_eq!(detectors, bits, "H ehat = syndrome, mask={mask}");
            assert_eq!(
                observables,
                decoder.decode(&syndrome).observable_flips,
                "O ehat = decode, mask={mask}"
            );
        }
        assert_eq!(
            mwpm.decode_errors(&syndrome),
            reference.decode_errors(&syndrome)
        );
    }
    // Pin the original regression: original col 0 flips D0 AND D1, although the
    // matching correction to this syndrome uses only the first component.
    let syndrome = Syndrome::new(3, vec![0]);
    let legacy = MwpmDecoder::new(&dem).unwrap().decode_errors(&syndrome);
    assert_ne!(project(&dem, &legacy).0, vec![true, false, false]);
    assert_eq!(mwpm.decode_errors(&syndrome), [1, 0, 0]);
}

#[test]
fn graph_edges_are_identical_except_for_column_indices() {
    let dem = DetectorErrorModel::parse(
        "error(0.1) D0 L0 ^ D1 L1\nerror(0.2) D0 L0\nerror(0.3) D2 L0 ^ L1 ^",
    )
    .unwrap();
    let split = dem.split_components();
    let original = MatchingGraph::from_dem(&dem).unwrap();
    let graph = MatchingGraph::from_dem(&split).unwrap();
    assert_eq!(graph.num_columns(), split.errors.len());
    assert_eq!(original.edges().len(), graph.edges().len());
    for (a, b) in original.edges().iter().zip(graph.edges()) {
        let mut expected = a.clone();
        expected.column = b.column;
        assert_eq!(
            expected, *b,
            "endpoints, observables, probability and weight must be unchanged"
        );
        let part = &split.errors[b.column as usize];
        let expected_dets: Vec<_> = [b.a, b.b]
            .into_iter()
            .filter(|&d| d != graph.boundary())
            .map(|d| d as u32)
            .collect();
        assert_eq!(part.dets, expected_dets);
        assert_eq!(part.obs, b.observables);
    }
}

#[test]
fn hyperedge_parts_still_fail_explicitly() {
    let dem = DetectorErrorModel::parse("error(0.1) D0 D1 D2 ^ D3").unwrap();
    assert!(matches!(
        MwpmDecoder::new_with_split_components(&dem),
        Err(Error::NonGraphlike { dets: 3 })
    ));
    assert!(matches!(
        UnionFindDecoder::new_with_split_components(&dem),
        Err(Error::NonGraphlike { dets: 3 })
    ));
}

#[test]
fn splitting_is_not_an_equivalent_noise_sampling_model() {
    let dem = DetectorErrorModel::parse("error(0.1) D0 ^ D1").unwrap();
    let split = dem.split_components();
    let possible_original = [project(&dem, &[0]).0, project(&dem, &[1]).0];
    let independent_part = project(&split, &[1, 0]).0;
    assert!(
        !possible_original.contains(&independent_part),
        "original parts always fire together"
    );
}

proptest::proptest! {
    #[test]
    fn component_parity_is_preserved_without_cross_contamination(
        parts in proptest::collection::vec(
            (proptest::collection::vec(0u32..4, 0..8), proptest::collection::vec(0u32..3, 0..8)),
            2..6
        )
    ) {
        let source = DemError::with_components(0.25, parts.clone());
        let dem = DetectorErrorModel { detectors: 4, observables: 3, errors: vec![source] };
        let split = dem.split_components();
        proptest::prop_assert_eq!(split.errors.len(), parts.len());
        proptest::prop_assert_eq!(project(&split, &vec![1; parts.len()]), project(&dem, &[1]));
        for (part, error) in parts.iter().zip(&split.errors) {
            let mut dets = vec![false; 4]; let mut obs = vec![false; 3];
            for &d in &part.0 { dets[d as usize] ^= true; }
            for &o in &part.1 { obs[o as usize] ^= true; }
            let one = DetectorErrorModel { detectors: 4, observables: 3, errors: vec![error.clone()] };
            proptest::prop_assert_eq!(project(&one, &[1]), (dets, obs));
            proptest::prop_assert_eq!(error.prob.to_bits(), 0.25f64.to_bits());
        }
        proptest::prop_assert_eq!(split.split_components(), split);
    }
}
