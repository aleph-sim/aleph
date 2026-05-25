//! GHZ-state preparation end-to-end benchmark for n ∈ {10, 15, 20, 25}.
//!
//! Drives `NaiveSvBackend` through the canonical GHZ circuit
//! (`H q[0]; CX q[0],q[1]; CX q[1],q[2]; …`).  Throughput is
//! reported in amplitudes (`2^n`) so bencher.dev plots elements/s,
//! the metric SoA/SIMD work in Phase 1 will move.
//!
//! Memory budget at n=25: state vector is 2^25 × 16 B ≈ 512 MiB.
//! Criterion drops the result between iterations so peak resident
//! memory is one buffer — comfortable on the EPYC runner (123 GiB)
//! and a 16 GiB laptop alike.

use aleph_backend::run;
use aleph_benches::ghz_circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const QUBIT_COUNTS: &[u32] = &[10, 15, 20, 25];

fn bench_ghz(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz");
    for &n in QUBIT_COUNTS {
        let circuit = ghz_circuit(n);
        group.throughput(Throughput::Elements(1u64 << n));
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

criterion_group!(benches, bench_ghz);
criterion_main!(benches);
