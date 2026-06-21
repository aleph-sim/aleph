//! P5.7-08: Metal GPU MPS vs CPU MPS exit benchmark.
//!
//! Three arms on the same circuits, so the comparison is backend-only:
//! - `cpu` — `aleph_mps::MpsBackend` (f64, faer SVD, rayon): the Phase-3 CPU MPS.
//! - `gpu` — `MetalMpsBackend::run` (FP32, GPU contract/apply + GPU-resident Jacobi
//!   SVD, canonical truncation), gate-by-gate.
//! - `gpu_batched` — `MetalMpsBackend::run_batched` (P5.7-04): a brickwall layer's
//!   disjoint splits in one batched dispatch.
//!
//! Workloads: the NN random brickwall (`random_brickwall_circuit`, small bond) and
//! the bond-saturating brickwall (`brickwall_ry_cnot_rz`, central bond → 2^(n/2))
//! across an n-sweep. Bond cap is generous so nothing truncates (an exact, apples-
//! to-apples compare; truncation correctness is gated by the oracle, not timed).
//!
//! Correctness guard: before timing each workload, assert the GPU (batched) dense
//! state matches the CPU MPS within the FP32 tolerance — a fast-but-wrong backend
//! cannot post a number. Perf is never a CI gate; run on the M4 Mac Mini:
//!   cargo bench -p aleph-metal --features metal --bench mps_vs_cpu

use aleph_backend::run;
use aleph_benches::{brickwall_ry_cnot_rz, random_brickwall_circuit};
use aleph_core::Complex;
use aleph_ir::Circuit;
use aleph_metal::MetalMpsBackend;
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Bond cap above every workload's natural entanglement (n ≤ 14 ⇒ central bond ≤
/// 2^7 = 128), so neither MPS truncates and the compare is exact-to-fp32.
const MAX_BOND: usize = 256;
/// FP32 dense-state agreement tolerance for the pre-timing guard.
const GUARD_TOL: f64 = 1e-4;

fn cpu_dense(circuit: &Circuit) -> Vec<Complex<f64>> {
    let mut be = MpsBackend::new().with_max_bond(MAX_BOND);
    run(&mut be, circuit)
        .expect("cpu mps run")
        .dense_statevector()
}

/// Assert the GPU (batched) state matches the CPU MPS before timing.
fn guard(name: &str, circuit: &Circuit) {
    let mut gpu = MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required");
    let got = gpu
        .run_batched(circuit)
        .expect("gpu run_batched")
        .dense_statevector();
    let want = cpu_dense(circuit);
    assert_eq!(got.len(), want.len(), "{name}: dim mismatch");
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(
            g.re.is_finite() && g.im.is_finite(),
            "{name}: non-finite GPU amplitude"
        );
        worst = worst.max(((g.re - w.re).powi(2) + (g.im - w.im).powi(2)).sqrt());
    }
    assert!(
        worst <= GUARD_TOL,
        "{name}: max |Δ|={worst:.3e} > {GUARD_TOL:.0e}"
    );
}

fn bench_three_arms(c: &mut Criterion, group_name: &str, n: u32, circuit: &Circuit) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    guard(group_name, circuit);

    group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
        b.iter_with_setup(
            || MpsBackend::new().with_max_bond(MAX_BOND),
            |mut be| {
                black_box(run(&mut be, black_box(circuit)).unwrap());
            },
        );
    });
    group.bench_with_input(BenchmarkId::new("gpu", n), &n, |b, _| {
        b.iter_with_setup(
            || MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required"),
            |mut be| {
                black_box(be.run(black_box(circuit)).unwrap());
            },
        );
    });
    group.bench_with_input(BenchmarkId::new("gpu_batched", n), &n, |b, _| {
        b.iter_with_setup(
            || MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required"),
            |mut be| {
                black_box(be.run_batched(black_box(circuit)).unwrap());
            },
        );
    });
    group.finish();
}

/// NN random brickwall (small bond): the per-gate dispatch regime.
fn bench_nn_brickwall(c: &mut Criterion) {
    for &n in &[8u32, 10, 12, 14] {
        bench_three_arms(
            c,
            "mps_vs_cpu/nn_brickwall",
            n,
            &random_brickwall_circuit(n, 24),
        );
    }
}

/// Bond-saturating brickwall (central bond → 2^(n/2)): the large-bond regime where
/// the GPU SVD has the most work to amortise the dispatch overhead.
fn bench_bond_saturating(c: &mut Criterion) {
    for &n in &[8u32, 10, 12, 14] {
        bench_three_arms(
            c,
            "mps_vs_cpu/bond_saturating",
            n,
            &brickwall_ry_cnot_rz(n, 12),
        );
    }
}

criterion_group!(benches, bench_nn_brickwall, bench_bond_saturating);
criterion_main!(benches);
