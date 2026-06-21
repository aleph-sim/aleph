//! P5.7-04: layer-batched vs gate-by-gate MPS-on-Metal SVD dispatch.
//!
//! Two arms per workload, both on `MetalMpsBackend`:
//!   * `gate_by_gate` — `run` (P5.7-03): one SVD dispatch + GPU sync per 2q gate.
//!   * `batched`      — `run_batched` (P5.7-04): each brickwall layer's disjoint
//!     two-site splits factored in a *single* batched-Jacobi dispatch, one
//!     `commit`/`wait` per layer.
//!
//! The per-gate SVD dispatch was ~85% of per-gate time after P5.7-03
//! (`docs/perf/phase5.7.md`); collapsing a layer's launches into one removes the
//! per-gate dispatch latency. Two cases: the report's NN brickwall (n=12 d=24,
//! small bond) and a bond-saturating brickwall (`brickwall_ry_cnot_rz`, central
//! bond → 64 at n=12) so the larger-block regime is covered too.
//!
//! Correctness guard: before timing, assert the two paths agree amplitude-for-
//! amplitude (a fast-but-wrong batched path cannot pass). Perf is never a CI gate.
//!
//! Run on the M4 Mac Mini (a Metal device is required):
//!   cargo bench -p aleph-metal --features metal --bench mps_batched

use aleph_benches::{brickwall_ry_cnot_rz, random_brickwall_circuit};
use aleph_ir::Circuit;
use aleph_metal::MetalMpsBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Bond cap above the n=12 workloads' natural entanglement (central bond ≤ 64),
/// so neither path truncates and the compare is exact-to-fp32.
const MAX_BOND: usize = 128;
const GUARD_TOL: f64 = 1e-5;

/// Assert the gate-by-gate and batched dense states agree everywhere. Panics
/// (fails the bench) on mismatch, so a fast-but-wrong batched path cannot pass.
fn guard(name: &str, circuit: &Circuit) {
    let mut gpu = MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required");
    let seq = gpu
        .run(circuit)
        .expect("gate-by-gate run")
        .dense_statevector();
    let bat = gpu
        .run_batched(circuit)
        .expect("batched run")
        .dense_statevector();
    assert_eq!(seq.len(), bat.len(), "{name}: dim mismatch");
    let mut worst = 0.0f64;
    for (s, b) in seq.iter().zip(bat.iter()) {
        assert!(
            s.re.is_finite() && s.im.is_finite() && b.re.is_finite() && b.im.is_finite(),
            "{name}: non-finite amplitude"
        );
        let d: f64 = ((s.re - b.re).powi(2) + (s.im - b.im).powi(2)).sqrt();
        worst = worst.max(d);
    }
    assert!(
        worst <= GUARD_TOL,
        "{name}: max |Δ|={worst:.3e} > {GUARD_TOL:.0e}"
    );
}

/// Bench both arms for one built circuit. The backend is built in the untimed
/// setup closure; the zero-state allocate inside `run`/`run_batched` is timed but
/// identical across arms, so the ratio is unaffected.
fn bench_two_arms(c: &mut Criterion, group_name: &str, label: &str, circuit: &Circuit) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    guard(label, circuit);
    group.bench_with_input(BenchmarkId::new("gate_by_gate", label), &label, |b, _| {
        b.iter_with_setup(
            || MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required"),
            |mut be| {
                black_box(be.run(black_box(circuit)).unwrap());
            },
        );
    });
    group.bench_with_input(BenchmarkId::new("batched", label), &label, |b, _| {
        b.iter_with_setup(
            || MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required"),
            |mut be| {
                black_box(be.run_batched(black_box(circuit)).unwrap());
            },
        );
    });
    group.finish();
}

fn bench_small_bond(c: &mut Criterion) {
    // The report's headline workload: NN brickwall n=12, depth=24 (small bond).
    bench_two_arms(
        c,
        "mps_batched/nn_brickwall",
        "n12_d24",
        &random_brickwall_circuit(12, 24),
    );
}

fn bench_large_bond(c: &mut Criterion) {
    // Bond-saturating brickwall: central bond reaches χ=64 at n=12, so the blocks
    // (rows/cols up to 128) are the larger-bond regime the AC asks for.
    bench_two_arms(
        c,
        "mps_batched/bond_saturating",
        "n12_l16",
        &brickwall_ry_cnot_rz(12, 16),
    );
}

criterion_group!(benches, bench_small_bond, bench_large_bond);
criterion_main!(benches);
