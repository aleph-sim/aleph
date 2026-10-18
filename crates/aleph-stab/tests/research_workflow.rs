#[path = "../examples/support/research_observables.rs"]
mod integration;
#[test]
fn same_saved_queries_cross_full_cone_sparse_and_analytic_references() {
    let path = std::env::temp_dir().join(format!(
        "aleph-p7-query-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let report = integration::study(&path, true).unwrap();
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let large = cases.iter().find(|c| c["case"] == "ghz128").unwrap();
    assert_eq!(large["compact_sv"]["status"], "resource_limit");
    assert_eq!(large["full_sv"]["status"], "not_attempted");
    assert_eq!(large["sparse"][0]["max_active_qubits"], 128);
    assert_eq!(large["sparse"][1]["max_active_qubits"], 128);
    let stress = cases.iter().find(|c| c["case"] == "product6").unwrap();
    assert_eq!(stress["term_limit_control"]["required"], 16);
    assert!(
        integration::study(&path, true).is_err(),
        "must reject mixed/stale output directory"
    );
    std::fs::remove_dir_all(path).unwrap();
}
