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
        group.bench_with_input(BenchmarkId::from_parameter(label), &shots, |bencher, &shots| {
            bencher.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut backend| {
                    let v = backend.sample(&s, shots).unwrap();
                    criterion::black_box(v);
                },
            );
        });
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

criterion_group!(benches, bench_sample);
criterion_main!(benches);
