//! P5.5-04 fused-vs-unfused state-vector bench for `MetalSvBackend`.
//!
//! For each Tier-1 workload, two benchmarks are measured:
//!   * `unfused` — `MetalSvBackend::run` (gate-by-gate, one GPU dispatch +
//!     `wait_until_completed` per gate).
//!   * `fused`   — `MetalSvBackend::run_optimized` (default IR pipeline:
//!     FuseKq collapses adjacent gates into dense `UnitaryKq` blocks that ride
//!     the `apply_kq` kernel, cutting the number of GPU round-trips).
//!
//! The AC needs fused to beat unfused on >=1 workload; QFT and random brickwall
//! are the strong cases. Run on the dev Mac (an Apple M3; the AC references an
//! M4 base — see docs/perf/phase5.5.md). A device is required.
//!
//! Run: `cargo bench -p aleph-metal --features metal --bench sv_fused`

use aleph_benches::{ghz_circuit, qft_circuit, random_brickwall_circuit};
use aleph_ir::Circuit;
use aleph_metal::MetalSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Path to the n=15 Grover baseline fixture (measurement-free), anchored to the
/// crate manifest dir so it resolves regardless of the bench's working dir.
const GROVER_N15: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/qiskit-baseline/circuits/grover_n15_iters5.qasm"
);

/// Benchmark `unfused` (run) vs `fused` (run_optimized) for one built circuit.
/// The backend is built inside `iter_with_setup`'s setup closure, so the Metal
/// pipeline compile runs untimed before each timed sample, matching the CPU
/// benches' `with_seed` setup idiom. The zero-state `allocate` happens inside
/// `run`/`run_optimized` and is therefore part of the timed work — but it is
/// identical across both arms, so the fused/unfused ratio is unaffected.
fn bench_pair(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    id: u32,
    circuit: &Circuit,
) {
    group.bench_with_input(BenchmarkId::new("unfused", id), &id, |b, _| {
        b.iter_with_setup(
            || MetalSvBackend::with_seed(0).expect("Metal device required for sv_fused bench"),
            |mut backend| {
                let s = backend.run(black_box(circuit)).unwrap();
                black_box(s);
            },
        );
    });
    group.bench_with_input(BenchmarkId::new("fused", id), &id, |b, _| {
        b.iter_with_setup(
            || MetalSvBackend::with_seed(0).expect("Metal device required for sv_fused bench"),
            |mut backend| {
                let s = backend.run_optimized(black_box(circuit)).unwrap();
                black_box(s);
            },
        );
    });
}

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_fused/qft");
    for &n in &[14u32, 16, 18] {
        let circuit = qft_circuit(n);
        bench_pair(&mut group, n, &circuit);
    }
    group.finish();
}

fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_fused/random");
    for &n in &[14u32, 16, 18] {
        // Fixed depth; deterministic builder.
        let circuit = random_brickwall_circuit(n, 12);
        bench_pair(&mut group, n, &circuit);
    }
    group.finish();
}

fn bench_ghz(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_fused/ghz");
    for &n in &[16u32, 18] {
        let circuit = ghz_circuit(n);
        bench_pair(&mut group, n, &circuit);
    }
    group.finish();
}

fn bench_grover(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_fused/grover");
    let src = std::fs::read_to_string(GROVER_N15).expect("read grover_n15 fixture");
    let circuit = aleph_parser::parse(&src).expect("parse grover_n15");
    bench_pair(&mut group, 15, &circuit);
    group.finish();
}

criterion_group!(benches, bench_qft, bench_random, bench_ghz, bench_grover);
criterion_main!(benches);
