//! P5.7-08 / P5.8-01: Metal GPU MPS vs CPU MPS exit benchmark.
//!
//! Two regimes, same circuits per cell so the comparison is backend-only:
//!
//! **Small-n exact** (`mps_vs_cpu/nn_brickwall`, `…/bond_saturating`, n ≤ 14): three
//! arms — `cpu` (`aleph_mps::MpsBackend`, f64 faer SVD, rayon), `gpu`
//! (`MetalMpsBackend::run`, FP32 GPU-resident Jacobi SVD + canonical truncation,
//! gate-by-gate) and `gpu_batched` (`run_batched`, P5.7-04, a brickwall layer's
//! disjoint splits in one dispatch). Bond cap (256) sits above the natural
//! entanglement so nothing truncates — an apples-to-apples exact compare, guarded
//! by a dense `2^n` agreement check before timing.
//!
//! **Large-n χ-sweep** (`mps_vs_cpu/large_n`, n ∈ {16, 20, 24}): the regime the
//! Phase 5.7 audit (`docs/perf/phase5.7-audit.md`) predicts the GPU can finally win
//! — large bond χ, large n. A single bond-*saturating* circuit per n (central bond →
//! 2^(n/2)) is run under a swept bond cap χ ∈ {256, 512, 1024}; cells where χ would
//! not bind (χ > 2^(n/2)) are skipped. Only `cpu` and `gpu` (canonical `run`) arms:
//! `run_batched` is exact-only and *refuses* a real truncation
//! (`MpsTruncationUnsupported`), so it cannot run the truncating sweep. There is no
//! dense `2^n` allocation anywhere on this path. Large-n correctness is asserted
//! `2^n`-free in `tests/mps_large_n.rs` (norm=1, analytic GHZ `Z`-string
//! expectation, `run` vs `run_batched` agreement) — not re-guarded here, where an
//! extra per-cell `gpu.run` would cost minutes at n=20/24.
//!
//! Perf is never a CI gate; run on the M4 Mac Mini:
//!   cargo bench -p aleph-metal --features metal --bench mps_vs_cpu
//! The heaviest large-n cells are long-running by design (they characterise the slow
//! regime); filter to a subset with e.g. `-- large_n/.*n16`.

use aleph_backend::run;
use aleph_benches::{brickwall_ry_cnot_rz, random_brickwall_circuit};
use aleph_core::Complex;
use aleph_ir::Circuit;
use aleph_metal::MetalMpsBackend;
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Bond cap above every small-n workload's natural entanglement (n ≤ 14 ⇒ central
/// bond ≤ 2^7 = 128), so neither MPS truncates and the compare is exact-to-fp32.
const MAX_BOND: usize = 256;
/// FP32 dense-state agreement tolerance for the small-n pre-timing guard.
const GUARD_TOL: f64 = 1e-4;

/// Large-n χ-sweep grid. The circuit per n saturates the central bond to 2^(n/2);
/// the cap χ is swept, so the GPU SVD has progressively more work to amortise the
/// per-dispatch overhead. Cells with χ > 2^(n/2) are skipped (χ would not bind).
const LARGE_N: &[u32] = &[16, 20, 24];
const CHI_SWEEP: &[usize] = &[256, 512, 1024];

fn cpu_dense(circuit: &Circuit) -> Vec<Complex<f64>> {
    let mut be = MpsBackend::new().with_max_bond(MAX_BOND);
    run(&mut be, circuit)
        .expect("cpu mps run")
        .dense_statevector()
}

/// Small-n guard: assert the GPU (batched) state matches the CPU MPS before timing.
fn guard_dense(name: &str, circuit: &Circuit) {
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
    guard_dense(group_name, circuit);

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

/// Large-n χ-sweep: a bond-saturating circuit per n run under a swept bond cap.
/// Two arms (`cpu`, `gpu` canonical `run`); `run_batched` is excluded — it refuses
/// truncation. The χ-sweep is reported with no `2^n` allocation anywhere.
fn bench_large_n_chi_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_vs_cpu/large_n");
    group.sample_size(10);
    for &n in LARGE_N {
        // One deep circuit per n, saturating the central bond to ≈ 2^(n/2); the cap
        // χ (not the circuit) is what varies across the sweep.
        let natural_bond = 1usize << (n / 2);
        let circuit = brickwall_ry_cnot_rz(n, n + 6);
        for &chi in CHI_SWEEP {
            // Skip caps that cannot bind: χ ≥ the natural bond never truncates, so it
            // duplicates the largest binding cell at extra cost.
            if chi > natural_bond {
                continue;
            }
            let label = format!("n{n}_chi{chi}");
            // No correctness guard here: a guard would do a full extra `gpu.run` per
            // cell — multi-minute at n=20/24 — and would fire even for criterion-
            // filtered-out cells. Large-n `2^n`-free correctness is asserted instead
            // in `tests/mps_large_n.rs` (norm=1, GHZ `Z`-string, run vs run_batched).

            group.bench_with_input(
                BenchmarkId::new("cpu", &label),
                &(n, chi),
                |b, &(_, chi)| {
                    b.iter_with_setup(
                        || MpsBackend::new().with_max_bond(chi),
                        |mut be| {
                            black_box(run(&mut be, black_box(&circuit)).unwrap());
                        },
                    );
                },
            );
            group.bench_with_input(
                BenchmarkId::new("gpu", &label),
                &(n, chi),
                |b, &(_, chi)| {
                    b.iter_with_setup(
                        || MetalMpsBackend::with_max_bond(chi).expect("Metal device required"),
                        |mut be| {
                            black_box(be.run(black_box(&circuit)).unwrap());
                        },
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_nn_brickwall,
    bench_bond_saturating,
    bench_large_n_chi_sweep
);
criterion_main!(benches);
