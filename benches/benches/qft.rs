//! QFT end-to-end benchmark for n ∈ {10, 15, 20}.
//!
//! Drives `NaiveSvBackend` through the textbook QFT circuit per
//! Nielsen & Chuang § 5.1: per-qubit Hadamard followed by a
//! descending ladder of controlled-Phase gates.  Real cost is
//! `O(n · 2^n)` so throughput is reported as `n · 2^n` elements,
//! letting bencher.dev plot a stable elements/s metric across the
//! P0/P1 backend evolution.
//!
//! Reference: aleph's own `QFT.md` playbook at the repo root.

use aleph_backend::run;
use aleph_benches::qft_circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const QUBIT_COUNTS: &[u32] = &[10, 15, 20];

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft");
    for &n in QUBIT_COUNTS {
        let circuit = qft_circuit(n);
        group.throughput(Throughput::Elements(u64::from(n) * (1u64 << n)));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut backend| {
                    let state = run(&mut backend, &circuit).unwrap();
                    black_box(state);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qft);
criterion_main!(benches);
