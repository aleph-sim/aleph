//! P2-08 FP32-mode benchmark: fused QFT at n ∈ {22, 25} through the f64
//! `NaiveSvBackend` vs the single-precision `Fp32SvBackend`. The state
//! vector is bandwidth-bound at these n (P2-05), so halving bytes/amp
//! (`Complex<f32>` = 8 B vs 16) should yield ~1.5–2× wall-clock. Gated
//! behind `scaling-bench` (n=25 f64 is a 512 MiB state); meant for
//! deliberate EPYC runs:
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches \
//!       --bench qft_precision --features scaling-bench
//!
//! Both backends time the FUSED QFT path: the circuit is pre-optimized
//! once per `n` outside the timed loop (the `run_optimized` default
//! pipeline — cancellation, DCE, 1q/2q/diagonal/k-qubit fusion), which
//! raises arithmetic intensity so the per-amp byte traffic dominates and
//! the f32/f64 ratio reflects the memory-bandwidth halving. Each `n`
//! emits a `f64` and an `f32` entry under the `qft_precision` group so
//! the speedup is a direct read-off.
//!
//! AC (BACKLOG §P2-08): f32 ≥ ~1.5–2× faster than f64 at n ≥ 24, EPYC.

use aleph_backend::run;
use aleph_benches::qft_circuit;
use aleph_sv::{Fp32SvBackend, NaiveSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const QUBIT_COUNTS: &[u32] = &[22, 25];

fn bench_qft_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft_precision");
    // n=25 is a 512 MiB f64 state vector; keep the sample count low so a
    // run finishes in minutes, not hours. Criterion treats this as a
    // floor for slow benches.
    group.sample_size(10);
    for &n in QUBIT_COUNTS {
        // Pre-optimize once, outside the timed loop, so the one-time
        // fusion cost is excluded and only the parallel-kernel sweep is
        // measured. Same circuit drives both backends.
        let mut circuit = qft_circuit(n);
        circuit.optimize().expect("optimize pipeline");
        group.throughput(Throughput::Elements(u64::from(n) * (1u64 << n)));
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut backend| {
                    let state = run(&mut backend, &circuit).unwrap();
                    black_box(state);
                },
            );
        });
        group.bench_with_input(BenchmarkId::new("f32", n), &n, |b, _| {
            b.iter_with_setup(
                || Fp32SvBackend::with_seed(0),
                |mut backend| {
                    let state = run(&mut backend, &circuit).unwrap();
                    black_box(state);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qft_precision);
criterion_main!(benches);
