//! P5.5-05 Part B: Tier-1 GPU-vs-CPU statevector exit bench.
//!
//! Three arms, all on the same default-optimized IR pipeline so the comparison
//! is backend-only:
//!   * `gpu`     — `MetalSvBackend` (FP32, fused `apply_kq` path).
//!   * `cpu_f32` — `Fp32SvBackend`  (CPU FP32, apples-to-apples).
//!   * `cpu_f64` — `NaiveSvBackend` (CPU FP64, the headline CPU statevector).
//!
//! Workloads: GHZ, QFT, random brickwall over the n-sweep {24, 26, 28}
//! (n=28 is the headline cell), plus one Grover cell at n=20 (its
//! multi-controlled decomposition makes n=28 Grover intractable on CPU).
//!
//! Correctness-at-scale guard: there is no Aer fixture at n=24..28, so before
//! timing each workload we run it once on the GPU and once on `NaiveSvBackend`
//! and assert a sampled set of amplitudes agree within 1e-5. The kernels are
//! size-agnostic (same dispatch, larger grid), so guarding at GUARD_N on the
//! real Tier-1 circuit — combined with Part A's Aer-fixture oracle — defends the
//! timed n=28 path against a fast-but-wrong kernel.
//!
//! Run on the M4 Mac Mini (a Metal device is required):
//!   cargo bench -p aleph-metal --features metal --bench sv_vs_cpu

use aleph_backend::run_optimized;
use aleph_benches::{ghz_circuit, qft_circuit, random_brickwall_circuit};
use aleph_core::Complex;
use aleph_ir::Circuit;
use aleph_metal::MetalSvBackend;
use aleph_sv::{Fp32SvBackend, NaiveSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Headline n-sweep. n=28 is the exit cell; all three backends cap at 28 qubits.
const NS: &[u32] = &[24, 26, 28];
/// Random brickwall depth — dense enough to be a real GPU workload, shallow
/// enough to keep the n=28 sample tractable.
const RANDOM_DEPTH: usize = 20;
/// Qubit count at which the scale self-consistency guard runs (cheapest in the
/// sweep: 2^24 f64 reference ≈ 268 MB, seconds to compute).
const GUARD_N: u32 = 24;
/// Grover cell size (separate from the sweep; n=28 Grover is CPU-intractable).
const GROVER_N: u32 = 20;
/// Grover baseline circuit (measurement-free), anchored to the crate manifest.
const GROVER_N20: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/qiskit-baseline/circuits/grover_n20_iters5.qasm"
);
/// Tolerance for the scale self-consistency guard (GPU f32 vs CPU f64).
const GUARD_TOL: f64 = 1e-5;

/// Assert a sampled set of amplitudes agree within `GUARD_TOL`. Samples a fixed
/// stride plus the index of the largest-magnitude reference amplitude so a
/// degenerate "all zero but one" mismatch can't hide between stride points.
fn assert_sampled_close(name: &str, gpu: &[Complex<f64>], refr: &[Complex<f64>]) {
    assert_eq!(gpu.len(), refr.len(), "{name}: guard dim mismatch");
    let n = gpu.len();
    let stride = (n / 4096).max(1);
    let mut indices: Vec<usize> = (0..n).step_by(stride).collect();
    // Add the peak-magnitude reference index.
    let peak = refr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            (a.re * a.re + a.im * a.im)
                .partial_cmp(&(b.re * b.re + b.im * b.im))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    indices.push(peak);
    for i in indices {
        let (g, r) = (gpu[i], refr[i]);
        assert!(
            g.re.is_finite() && g.im.is_finite() && r.re.is_finite() && r.im.is_finite(),
            "{name}: non-finite guard amplitude at {i}: gpu {g:?} ref {r:?}"
        );
        let d = ((g.re - r.re).powi(2) + (g.im - r.im).powi(2)).sqrt();
        assert!(
            d <= GUARD_TOL,
            "{name}: guard amplitude {i} |Δ|={d:.3e} > {GUARD_TOL:.0e}\n  gpu {g:?}\n  ref {r:?}"
        );
    }
}

/// Run `circuit` on the GPU and on `NaiveSvBackend`, assert sampled agreement.
/// Panics (fails the bench) on mismatch — a fast-but-wrong kernel cannot pass.
fn guard(name: &str, circuit: &Circuit) {
    let mut gpu = MetalSvBackend::with_seed(0).expect("Metal device required for sv_vs_cpu bench");
    let mut cpu = NaiveSvBackend::with_seed(0);
    let gpu_state = gpu.run_optimized(circuit).expect("gpu guard run");
    let cpu_state = run_optimized(&mut cpu, circuit).expect("cpu guard run");
    let gpu_amps = aleph_oracle::HasAmplitudes::amplitudes(&gpu_state);
    let cpu_amps = aleph_oracle::HasAmplitudes::amplitudes(&cpu_state);
    assert_sampled_close(name, &gpu_amps, &cpu_amps);
}

/// Bench all three arms for one built circuit at size `n`. Backends are built in
/// the untimed setup closure; the zero-state allocate inside `run_optimized` is
/// timed but identical across arms, so ratios are unaffected (P5.5-04 lesson).
fn bench_three_arms(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: u32,
    circuit: &Circuit,
) {
    group.bench_with_input(BenchmarkId::new("gpu", n), &n, |b, _| {
        b.iter_with_setup(
            || MetalSvBackend::with_seed(0).expect("Metal device required"),
            |mut be| {
                black_box(be.run_optimized(black_box(circuit)).unwrap());
            },
        );
    });
    group.bench_with_input(BenchmarkId::new("cpu_f32", n), &n, |b, _| {
        b.iter_with_setup(
            || Fp32SvBackend::with_seed(0),
            |mut be| {
                black_box(run_optimized(&mut be, black_box(circuit)).unwrap());
            },
        );
    });
    group.bench_with_input(BenchmarkId::new("cpu_f64", n), &n, |b, _| {
        b.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut be| {
                black_box(run_optimized(&mut be, black_box(circuit)).unwrap());
            },
        );
    });
}

fn bench_ghz(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_vs_cpu/ghz");
    group.sample_size(10);
    guard("ghz", &ghz_circuit(GUARD_N));
    for &n in NS {
        bench_three_arms(&mut group, n, &ghz_circuit(n));
    }
    group.finish();
}

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_vs_cpu/qft");
    group.sample_size(10);
    guard("qft", &qft_circuit(GUARD_N));
    for &n in NS {
        bench_three_arms(&mut group, n, &qft_circuit(n));
    }
    group.finish();
}

fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_vs_cpu/random");
    group.sample_size(10);
    guard("random", &random_brickwall_circuit(GUARD_N, RANDOM_DEPTH));
    for &n in NS {
        bench_three_arms(&mut group, n, &random_brickwall_circuit(n, RANDOM_DEPTH));
    }
    group.finish();
}

fn bench_grover(c: &mut Criterion) {
    let qasm = std::fs::read_to_string(GROVER_N20).expect("read grover_n20 qasm");
    let circuit = aleph_parser::parse(&qasm).expect("parse grover_n20");
    let mut group = c.benchmark_group("sv_vs_cpu/grover");
    group.sample_size(10);
    guard("grover", &circuit);
    bench_three_arms(&mut group, GROVER_N, &circuit);
    group.finish();
}

criterion_group!(benches, bench_ghz, bench_qft, bench_random, bench_grover);
criterion_main!(benches);
