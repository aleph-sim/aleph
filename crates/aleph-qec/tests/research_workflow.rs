#[path = "../examples/support/research_qec.rs"]
mod integration;
#[test]
fn sampling_estimation_and_saved_witness_replay_share_model_identity() {
    let path = std::env::temp_dir().join(format!(
        "aleph-p7-qec-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let report = integration::study(&path).unwrap();
    assert_eq!(report["surface"]["shots"], 512);
    assert!(report["surface"]["failures"].as_u64().unwrap() > 0);
    assert_eq!(report["surface"]["witness"]["schema_version"], 1);
    assert_eq!(report["tiny_exact"]["decoder_calls"], 8);
    assert_eq!(report["tiny_exact"]["witnesses"][0]["schema_version"], 2);
    assert_eq!(report["tiny_dynamic"]["witnesses"][0]["schema_version"], 3);
    assert!(
        integration::study(&path).is_err(),
        "must reject mixed/stale output directory"
    );
    std::fs::remove_dir_all(path).unwrap();
}
