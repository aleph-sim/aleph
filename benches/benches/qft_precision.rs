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
use aleph_benches::{qft_circuit, random_brickwall_circuit};
use aleph_sv::{Fp32SvBackend, NaiveSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const QUBIT_COUNTS: &[u32] = &[22, 25];

/// Depth for the random brick-wall comparison. After fusion the
/// per-layer `Rz`+`Rx` collapse to dense `Unitary1q` and the CNOT layer
/// to `Unitary2q`/`UnitaryKq`, so this exercises the f32 dense AVX-512
/// kernels (16 f32 lanes/zmm) — unlike the diagonal-heavy fused QFT,
/// which is dominated by `DiagonalPhase` (whose f32 kernel shares the
/// f64 transcendental `sin_cos`, diluting the byte-traffic win).
const RANDOM_DEPTH: usize = 30;

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

/// Fused random brick-wall f32-vs-f64 at n ∈ {22, 25}. This is the
/// dense-kernel counterpart to `bench_qft_precision`: where fused QFT is
/// `DiagonalPhase`-bound, the fused brick-wall is dominated by dense
/// `Unitary1q`/`Unitary2q`, so the f32 AVX-512 kernels get both the
/// byte-traffic halving AND the 16-lane (vs 8) SIMD width. Reported
/// alongside QFT to characterise where single precision actually helps.
fn bench_random_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_precision");
    group.sample_size(10);
    for &n in QUBIT_COUNTS {
        let mut circuit = random_brickwall_circuit(n, RANDOM_DEPTH);
        circuit.optimize().expect("optimize pipeline");
        group.throughput(Throughput::Elements(RANDOM_DEPTH as u64 * (1u64 << n)));
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

criterion_group!(benches, bench_qft_precision, bench_random_precision);
criterion_main!(benches);
