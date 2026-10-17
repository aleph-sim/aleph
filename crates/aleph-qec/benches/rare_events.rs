//! Statistical-efficiency feature costs at a fixed, matched decoder-call budget.
use aleph_qec::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
fn bench(c: &mut Criterion) {
    let exp = SurfaceCode::new(3).memory_z_experiment(3);
    let dem = build_dem(
        &exp.annotated,
        &exp.phenomenological_mechanisms(0.0001, 0.0001),
    )
    .unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let plan = FaultCountPlan::new(&dem, 5, 4_000_000).unwrap();
    let options = StratifiedOptions {
        budgets: vec![256; 6],
        max_decoder_calls: 10000,
        seed: 544,
        confidence: 0.95,
        enumerate_small: true,
        witness_limit: 0,
        target_width: None,
    };
    let calls = run_stratified(&plan, &decoder, &options)
        .unwrap()
        .decoder_calls;
    let direct = MwpmDecoder::new_with_split_components(&dem).unwrap();
    let mut group = c.benchmark_group("rare_events");
    group.sample_size(10);
    group.bench_function("plan_d3_k5", |b| {
        b.iter(|| FaultCountPlan::new(black_box(&dem), 5, 4_000_000).unwrap())
    });
    group.bench_function("conditional_k2", |b| {
        b.iter(|| plan.sample(2, 544, black_box(0)).unwrap())
    });
    group.bench_function("stratified_fixed_budget", |b| {
        b.iter(|| run_stratified(&plan, &decoder, black_box(&options)).unwrap())
    });
    group.bench_function("direct_matched_calls", |b| {
        b.iter(|| {
            let result = run_dem_experiment(black_box(&dem), calls, &direct, 544).unwrap();
            (result, result.interval95().unwrap())
        })
    });
    group.finish();
}
fn config() -> Criterion {
    let mut c = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500));
    if let Some(path) = std::env::var_os("ALEPH_BENCH_OUTPUT") {
        c = c.output_directory(&std::path::PathBuf::from(path));
    }
    c
}
criterion_group! {name=benches;config=config();targets=bench}
criterion_main!(benches);
