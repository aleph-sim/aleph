//! Identical before/after harness: public complete experiment plus isolated stages.
use aleph_qec::{
    build_dem, run_dem_experiment, sample_shots, Decoder, DemError, DetectorErrorModel,
    MwpmDecoder, NullDecoder, SurfaceCode,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench(c: &mut Criterion) {
    let dem = DetectorErrorModel {
        detectors: 128,
        observables: 3,
        errors: (0..128)
            .map(|d| DemError::new(0.02, vec![d], vec![d % 3]))
            .collect(),
    };
    let decoder = NullDecoder::new(dem.observables);
    let mut group = c.benchmark_group("dem_experiment");
    group.sample_size(10);
    for shots in [1024, 131_072] {
        group.bench_with_input(
            BenchmarkId::new("null_total", shots),
            &shots,
            |b, &shots| {
                b.iter(|| run_dem_experiment(black_box(&dem), shots, &decoder, 565).unwrap());
            },
        );
    }
    // Stage microbenchmarks use a fixed 1024-shot batch. Sampling includes RNG creation
    // and allocations; decode includes prediction allocation; score excludes both.
    group.bench_function("sample_1024", |b| {
        b.iter(|| sample_shots(black_box(&dem), 1024, 565))
    });
    let (syndromes, truths) = sample_shots(&dem, 1024, 565);
    group.bench_function("null_decode_1024", |b| {
        b.iter(|| decoder.decode_batch(black_box(&syndromes)).unwrap())
    });
    let predictions = decoder.decode_batch(&syndromes).unwrap();
    group.bench_function("score_1024", |b| {
        b.iter(|| {
            black_box(&predictions)
                .iter()
                .zip(black_box(&truths))
                .filter(|(p, t)| p.observable_flips != **t)
                .count()
        })
    });
    let exp = SurfaceCode::new(3).memory_z_experiment(3);
    let surface = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).unwrap();
    let mwpm = MwpmDecoder::new(&surface).unwrap();
    group.bench_function("mwpm_d3_total_1024", |b| {
        b.iter(|| run_dem_experiment(black_box(&surface), 1024, &mwpm, 565).unwrap())
    });
    group.finish();
}
fn config() -> Criterion {
    let mut c = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(200))
        .measurement_time(std::time::Duration::from_millis(500));
    if let Some(path) = std::env::var_os("ALEPH_BENCH_OUTPUT") {
        c = c.output_directory(&std::path::PathBuf::from(path));
    }
    c
}
criterion_group! {name=benches; config=config(); targets=bench}
criterion_main!(benches);
