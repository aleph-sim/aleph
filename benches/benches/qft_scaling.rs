//! P2-01 parallel-scaling benchmark: QFT at n ∈ {22, 25} through the
//! AoS + AVX-512 `NaiveSvBackend`, driving the rayon-parallel gate
//! kernels (state length 2^n ≫ the `ALEPH_PAR_MIN_AMPS` threshold, so
//! the parallel path is active by default).
//!
//! This bench is gated behind the `scaling-bench` feature so the routine
//! `cargo bench --workspace` (and CI) skip it — n=25 allocates a 512 MiB
//! state vector per iteration and is meant for deliberate, pinned runs on
//! the EPYC bench server. Measure scaling by sweeping `RAYON_NUM_THREADS`:
//!
//!   RAYON_NUM_THREADS=1  cargo bench -p aleph-benches --bench qft_scaling \
//!       --features scaling-bench -- --save-baseline t1
//!   RAYON_NUM_THREADS=8  cargo bench -p aleph-benches --bench qft_scaling \
//!       --features scaling-bench -- --baseline t1
//!
//! The P2-01 acceptance gate: QFT-25 ≥ 6× faster at 8 cores than at 1.

use aleph_backend::run;
use aleph_benches::qft_circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const QUBIT_COUNTS: &[u32] = &[22, 25];

fn bench_qft_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft_scaling");
    // n=25 is a 512 MiB state vector; keep the sample count low so a
    // sweep finishes in minutes, not hours. Criterion treats this as a
    // floor for slow benches.
    group.sample_size(10);
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

criterion_group!(benches, bench_qft_scaling);
criterion_main!(benches);
