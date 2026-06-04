//! P2-09 cache-blocking benchmark: tiled (`TileBlock` in the pipeline) vs
//! non-tiled (same passes minus `TileBlock`) execution of a low-qubit-heavy
//! circuit — the regime where applying a run of gates per cache tile turns
//! N DRAM passes into 1.  Plus a high-qubit-spanning counter-case (random
//! brick-wall) where tiling does not help.  Gated behind `scaling-bench`;
//! meant for EPYC `perf stat -e cache-misses,LLC-load-misses` runs:
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches \
//!       --bench cache_blocking --features scaling-bench
//!
//! AC (BACKLOG §P2-09): L2/L3 cache-miss reduction on the low-qubit case;
//! speedup in the cache-resident regime; oracle equivalence preserved
//! (tested separately at 1e-12).
//!
//! The two forms differ ONLY in whether `TileBlock` ran — fusion,
//! cancellation, DCE, relabelling, diagonal fusion, 1q/2q/kq fusion are
//! identical — so the wall-clock delta isolates the tile-major executor.

use aleph_backend::run;
use aleph_benches::{low_qubit_heavy_circuit, random_brickwall_circuit};
use aleph_ir::passes::{
    CancelInversePairs, DeadCodeElim, Fuse1qRuns, Fuse2q, FuseDiagonalRuns, FuseKq, PassPipeline,
    RelabelQubits,
};
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

/// Build the non-tiled baseline: the default pipeline minus `TileBlock`.
///
/// This replicates the pass order from [`PassPipeline::default_pipeline`]
/// verbatim, only omitting the final `TileBlock` step.  All fusion,
/// cancellation, DCE, and relabelling passes are present so the only
/// structural difference between `tiled` and `untiled` is the absence of
/// `TiledBlock` instructions and the tile-major execution path.
fn optimize_no_tiling(mut c: aleph_ir::Circuit) -> aleph_ir::Circuit {
    PassPipeline::new(vec![
        Box::new(RelabelQubits::default()),
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        Box::new(FuseDiagonalRuns),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
        Box::new(FuseKq::default()),
        // TileBlock intentionally omitted — this is the baseline.
    ])
    .run(&mut c)
    .expect("non-tiling pipeline");
    c
}

/// Isolation pipeline: fusion WITHOUT `FuseKq`, optionally with `TileBlock`.
///
/// `FuseKq` (default `max_qubits = 4`) merges runs of low-qubit 1q/2q gates
/// into ≥3q `UnitaryKq` blocks, which the tile executor cannot group
/// (`TileBlock::confinable` requires arity ≤ 2). On a low-qubit-heavy
/// circuit that lets `FuseKq` consume most of the run before `TileBlock`
/// sees it, masking the tile-major win. Dropping `FuseKq` keeps the run as
/// tileable 1q/2q gates, isolating what cache-blocking delivers when the two
/// memory-pass optimizations don't compete. (The two are alternative
/// strategies for the same gates — a regime-aware pipeline would pick one;
/// see the P2-09 follow-up note.)
fn optimize_no_kq(mut c: aleph_ir::Circuit, with_tiling: bool) -> aleph_ir::Circuit {
    let mut passes: Vec<Box<dyn aleph_ir::passes::Pass>> = vec![
        Box::new(RelabelQubits::default()),
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        Box::new(FuseDiagonalRuns),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
        // FuseKq intentionally omitted — keep runs tileable.
    ];
    if with_tiling {
        passes.push(Box::new(aleph_ir::passes::TileBlock::default()));
    }
    PassPipeline::new(passes).run(&mut c).expect("no-kq pipeline");
    c
}

fn bench_cache_blocking(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_blocking");
    // n=25 is 512 MiB; keep sample count low so runs finish in minutes.
    group.sample_size(10);

    // ── low-qubit-heavy: the expected win ─────────────────────────────────
    // width=6 < tile_bits=15: all active-window gates are tile-confinable.
    // At n=22/25 the state spans multiple tiles, so the tile-major executor
    // makes multiple single-tile sweeps — the cache win is visible here.
    for &n in &[22u32, 25] {
        let circuit = low_qubit_heavy_circuit(n, 6, 40);

        let mut tiled = circuit.clone();
        tiled.optimize().expect("tiled optimize");

        let untiled = optimize_no_tiling(circuit.clone());

        // Throughput unit: gates × amplitudes (depth × state size).
        group.throughput(Throughput::Elements(40u64 * (1u64 << n)));

        group.bench_with_input(BenchmarkId::new("lowqubit_tiled", n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut bk| black_box(run(&mut bk, &tiled).unwrap()),
            );
        });
        group.bench_with_input(BenchmarkId::new("lowqubit_untiled", n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut bk| black_box(run(&mut bk, &untiled).unwrap()),
            );
        });
    }

    // ── isolation: cache-blocking without FuseKq competition ──────────────
    // FuseKq eats low-qubit runs into ≥3q UnitaryKq blocks the tile executor
    // can't group, masking the win on the default pipeline. Here both forms
    // drop FuseKq, so the runs stay tileable 1q/2q — this isolates the
    // tile-major win when the two optimizations don't compete.
    for &n in &[22u32, 25] {
        let circuit = low_qubit_heavy_circuit(n, 6, 40);
        let tiled = optimize_no_kq(circuit.clone(), true);
        let untiled = optimize_no_kq(circuit.clone(), false);
        group.throughput(Throughput::Elements(40u64 * (1u64 << n)));
        group.bench_with_input(BenchmarkId::new("lowqubit_nokq_tiled", n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut bk| black_box(run(&mut bk, &tiled).unwrap()),
            );
        });
        group.bench_with_input(BenchmarkId::new("lowqubit_nokq_untiled", n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut bk| black_box(run(&mut bk, &untiled).unwrap()),
            );
        });
    }

    // ── counter-case: random brick-wall (no win expected) ─────────────────
    // Gates span all n qubits; TileBlock produces no grouping, so tiled ≈
    // untiled.  Reported to confirm the benchmark is not systematically
    // measuring something other than tiling.
    {
        let n = 25u32;
        let circuit = random_brickwall_circuit(n, 30);

        let mut tiled = circuit.clone();
        tiled.optimize().expect("tiled optimize");

        let untiled = optimize_no_tiling(circuit.clone());

        group.throughput(Throughput::Elements(30u64 * (1u64 << n)));

        group.bench_with_input(BenchmarkId::new("random_tiled", n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut bk| black_box(run(&mut bk, &tiled).unwrap()),
            );
        });
        group.bench_with_input(BenchmarkId::new("random_untiled", n), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut bk| black_box(run(&mut bk, &untiled).unwrap()),
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cache_blocking);
criterion_main!(benches);
