//! Baseline benchmark for [`aleph_sv::NaiveSvBackend`]: apply an H to
//! every qubit on `n ∈ {10, 15, 20}`. Establishes the curve P0-11 will
//! measure against.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallvec::smallvec;

fn h_wall(c: &mut Criterion) {
    let mut group = c.benchmark_group("h_wall");
    for &n in &[10u32, 15, 20] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &n| {
            bencher.iter_with_setup(
                || {
                    let mut b = NaiveSvBackend::with_seed(0);
                    let s = b.allocate(n).unwrap();
                    (b, s)
                },
                |(mut b, mut s)| {
                    for q in 0..n {
                        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q]))
                            .unwrap();
                    }
                    criterion::black_box(&s);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, h_wall);
criterion_main!(benches);
