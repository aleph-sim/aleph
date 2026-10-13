//! Micro-bench for `NaiveSvBackend::sample`. Quantifies the
//! alias-method (P0-11) vs inverse-CDF (P0-09) speedup. Save a
//! baseline from `main` with `--save-baseline pre-alias` before the
//! P0-11 merge and compare with `--baseline pre-alias` after.
//!
//! States are built via `apply_gate` (the public path); we set the
//! state up once outside the timing loop so the bench measures
//! `sample` in isolation.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_sv::{CpuState, NaiveSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallvec::smallvec;
use std::{path::PathBuf, time::Duration};

fn uniform_plus_n(n: u32) -> CpuState {
    let mut b = NaiveSvBackend::with_seed(0);
    let mut s = b.allocate(n).unwrap();
    for q in 0..n {
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q]))
            .unwrap();
    }
    s
}

fn ghz_n(n: u32) -> CpuState {
    let mut b = NaiveSvBackend::with_seed(0);
    let mut s = b.allocate(n).unwrap();
    b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
        .unwrap();
    for t in 1u32..n {
        b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0u32, t]))
            .unwrap();
    }
    s
}

fn bench_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample");
    for &(n, shots, label) in &[
        (4u32, 1_000u32, "uniform_n4_shots1k"),
        (10, 100_000, "uniform_n10_shots100k"),
        (16, 100_000, "uniform_n16_shots100k"),
    ] {
        let s = uniform_plus_n(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &shots,
            |bencher, &shots| {
                bencher.iter_with_setup(
                    || NaiveSvBackend::with_seed(0),
                    |mut backend| {
                        let v = backend.sample(&s, shots).unwrap();
                        criterion::black_box(v);
                    },
                );
            },
        );
    }
    {
        let s = ghz_n(10);
        let shots = 100_000u32;
        group.bench_with_input(
            BenchmarkId::from_parameter("ghz_n10_shots100k"),
            &shots,
            |bencher, &shots| {
                bencher.iter_with_setup(
                    || NaiveSvBackend::with_seed(0),
                    |mut backend| {
                        let v = backend.sample(&s, shots).unwrap();
                        criterion::black_box(v);
                    },
                );
            },
        );
    }
    group.finish();
}

fn sampler_matrix<B: Backend>(c: &mut Criterion, name: &str, mut make: impl FnMut() -> B) {
    let mut group = c.benchmark_group(format!("p8_single_shot/{name}"));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(700));
    for n in [4, 12, 20] {
        let mut backend = make();
        let mut state = backend.allocate(n).unwrap();
        for q in 0..n {
            backend
                .apply_gate(&mut state, &GateInstance::new(Gate::H, &[q][..]))
                .unwrap();
        }
        for shots in [1, 2, 4, 16, 1024] {
            // Advance the RNG between iterations so a CDF does not benchmark only
            // one fixed early/late exit. State preparation is outside the timed API.
            group.bench_function(format!("uniform_n{n}_shots{shots}"), |b| {
                b.iter(|| criterion::black_box(backend.sample(&state, shots).unwrap()));
            });
        }
    }
    group.finish();
}

fn bench_low_shot(c: &mut Criterion) {
    sampler_matrix(c, "aos", || NaiveSvBackend::with_seed(556));
    sampler_matrix(c, "soa", || aleph_sv::SoaSvBackend::with_seed(556));
    sampler_matrix(c, "f32", || aleph_sv::Fp32SvBackend::with_seed(556));
    let mut group = c.benchmark_group("p8_single_shot/noisy");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(700));
    for (n, shots) in [(4, 1024), (16, 64), (20, 8)] {
        let mut circuit = aleph_ir::Circuit::new(n, 0);
        for q in 0..n {
            circuit
                .add_gate(GateInstance::new(Gate::H, &[q][..]))
                .unwrap();
        }
        let mut model = aleph_sv::noise::NoiseModel::new();
        model.add_all_qubit_quantum_error(aleph_sv::noise::depolarizing_error(0.05, 1), &["H"]);
        model.add_readout_error(
            aleph_sv::noise::ReadoutError::new([[0.97, 0.03], [0.07, 0.93]]),
            0,
        );
        group.bench_function(format!("pauli_readout_n{n}_shots{shots}"), |b| {
            b.iter(|| {
                criterion::black_box(
                    aleph_sv::noise::run_noisy(&circuit, &model, shots, 556).unwrap(),
                )
            });
        });
    }
    group.finish();
}

fn config() -> Criterion {
    let mut c = Criterion::default();
    if let Some(path) = std::env::var_os("ALEPH_BENCH_OUTPUT") {
        c = c.output_directory(&PathBuf::from(path));
    }
    c
}

criterion_group! {name=benches; config=config(); targets=bench_sample, bench_low_shot}
criterion_main!(benches);
