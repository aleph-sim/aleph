//! Public fused-gate validation cost; identical harness on baseline/candidate.
use aleph_backend::Backend;
use aleph_core::{Complex, Gate, GateInstance};
use aleph_sv::{Fp32SvBackend, NaiveSvBackend, SoaSvBackend};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::{path::PathBuf, time::Duration};
fn gate(k: u8) -> GateInstance {
    let d = 1usize << k;
    let data = (0..d * d)
        .map(|i| {
            let (r, c) = (i / d, i % d);
            Complex::from_polar(
                1.0 / (d as f64).sqrt(),
                std::f64::consts::TAU * (r * c) as f64 / d as f64 + 0.137 * r as f64,
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    GateInstance::new(
        Gate::UnitaryKq { k, data },
        &(0..k as u32).collect::<Vec<_>>()[..],
    )
}
fn backend<B: Backend>(c: &mut Criterion, name: &str, factory: impl Fn() -> B) {
    let mut backend = factory();
    let mut group = c.benchmark_group(format!("p8_fused/{name}"));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(300));
    for n in [10, 20] {
        for k in [2, 3, 5] {
            let mut state = backend.allocate(n).unwrap();
            let gate = gate(k);
            group.bench_function(format!("n{n}_k{k}"), |b| {
                b.iter(|| {
                    backend
                        .apply_gate(
                            criterion::black_box(&mut state),
                            criterion::black_box(&gate),
                        )
                        .unwrap()
                })
            });
        }
    }
    group.finish();
    let gate = gate(5);
    let mut cold = c.benchmark_group(format!("p8_fused/cold_{name}"));
    cold.sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(300));
    cold.bench_function("n10_k5", |b| {
        b.iter_batched(
            || {
                let mut backend = factory();
                let state = backend.allocate(10).unwrap();
                (backend, state)
            },
            |(mut backend, mut state)| {
                backend.apply_gate(&mut state, &gate).unwrap();
                (backend, state)
            },
            BatchSize::PerIteration,
        )
    });
    cold.finish();
}
fn bench(c: &mut Criterion) {
    backend(c, "aos", || NaiveSvBackend::with_seed(573));
    backend(c, "soa", || SoaSvBackend::with_seed(573));
    backend(c, "fp32", || Fp32SvBackend::with_seed(573));
}
fn config() -> Criterion {
    let mut c = Criterion::default();
    if let Some(path) = std::env::var_os("ALEPH_BENCH_OUTPUT") {
        c = c.output_directory(&PathBuf::from(path));
    }
    c
}
criterion_group! {name=benches;config=config();targets=bench}
criterion_main!(benches);
