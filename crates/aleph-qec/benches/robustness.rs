//! Descriptive paired-harness overhead and process generation costs; no speedup target.
use aleph_qec::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
fn bench(c: &mut Criterion) {
    let e = SurfaceCode::new(3).memory_z_experiment(3);
    let dem = build_dem(&e.annotated, &e.phenomenological_mechanisms(0.03, 0.03)).unwrap();
    let d = WitnessDecoder::new(&dem, DecoderSpec::Mwpm).unwrap();
    let opts = RobustnessOptions {
        streams: 1,
        steps: 256,
        seed: 546,
        confidence: 0.95,
        witness_limit: 0,
    };
    c.bench_function("robustness/direct_two_decoders_512_calls", |b| {
        b.iter(|| {
            black_box(run_dem_experiment(&dem, 256, &d, 546).unwrap());
            black_box(run_dem_experiment(&dem, 256, &d, 546).unwrap());
        })
    });
    c.bench_function("robustness/static_paired_512_calls", |b| {
        b.iter(|| black_box(run_robustness(&dem, &NoiseProcess::Static, &d, &d, &opts).unwrap()))
    });
    for (name, process) in [
        ("static", NoiseProcess::Static),
        (
            "drift",
            NoiseProcess::LinearDrift {
                end_probabilities: vec![0.1; dem.errors.len()],
            },
        ),
        (
            "bursts",
            NoiseProcess::MarkovBurst {
                onset: 0.05,
                recovery: 0.2,
                affected: (0..dem.errors.len()).step_by(2).collect(),
                multiplier: 8.0,
            },
        ),
    ] {
        c.bench_function(&format!("robustness/generate_{name}_256"), |b| {
            b.iter(|| black_box(generate_noise_stream(&dem, &process, 256, 546).unwrap()))
        });
    }
}
fn config() -> Criterion {
    let c = Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(2));
    match std::env::var_os("ALEPH_BENCH_OUTPUT") {
        Some(path) => c.output_directory(&std::path::PathBuf::from(path)),
        None => c,
    }
}
criterion_group! {name=benches;config=config();targets=bench}
criterion_main!(benches);
