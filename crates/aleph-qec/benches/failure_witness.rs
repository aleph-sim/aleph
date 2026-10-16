//! Opt-in capture and reduction cost; ordinary sample timing lives in dem_experiment.
use aleph_qec::{
    build_dem, capture_shot, DecoderSpec, MinimizeBudget, SurfaceCode, WitnessDecoder,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
fn bench(c: &mut Criterion) {
    let exp = SurfaceCode::new(3).memory_z_experiment(3);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    // Fixed predeclared surface example: five starting mechanisms, reduced to two.
    let witness = capture_shot(&decoder, 543, 39).unwrap();
    assert!(witness.outcome.logical_failure);
    let mut group = c.benchmark_group("failure_witness");
    group.sample_size(10);
    group.bench_function("capture_d3", |b| {
        b.iter(|| capture_shot(black_box(&decoder), 543, witness.shot).unwrap())
    });
    group.bench_function("minimize_d3", |b| {
        b.iter(|| {
            decoder
                .minimize(
                    black_box(&witness),
                    MinimizeBudget {
                        decoder_calls: 10000,
                        time: Duration::from_secs(30),
                    },
                )
                .unwrap()
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
criterion_group! {name=benches; config=config(); targets=bench}
criterion_main!(benches);
