//! P2-05 Tier-1 parallel-scaling benchmark: GHZ / QFT / Grover / random
//! brick-wall at n = 25 through the AoS + AVX-512 `NaiveSvBackend`, driving
//! the rayon-parallel gate kernels. Companion to `qft_scaling.rs` (P2-01),
//! extended to the full Tier-1 set for the Phase-2 scaling report (#31).
//!
//! Gated behind `scaling-bench` so `cargo bench --workspace` / CI skip the
//! 512 MiB n=25 runs. Measure scaling by sweeping `RAYON_NUM_THREADS` across
//! processes, exactly as P2-01:
//!
//!   RAYON_NUM_THREADS=1  cargo bench -p aleph-benches --bench tier1_scaling \
//!       --features scaling-bench -- --save-baseline t1
//!   RAYON_NUM_THREADS=8  cargo bench -p aleph-benches --bench tier1_scaling \
//!       --features scaling-bench -- --baseline t1
//!
//! Circuits are the canonical n=25 fixtures under
//! `scripts/qiskit-baseline/circuits/` — the same circuits the Stage-0 Qiskit
//! Aer baseline used, so scaling lines up with the Aer comparison. GHZ-25 is
//! trivial (25 gates, allocation-bound) — included for spec completeness; its
//! efficiency number is not a meaningful bandwidth signal (see docs/perf/phase2.md).

use aleph_backend::run;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::PathBuf;

/// (group label, fixture stem). n = 25 only — the Phase-2 scaling target size.
const WORKLOADS: &[(&str, &str)] = &[
    ("ghz", "ghz_n25"),
    ("qft", "qft_n25"),
    ("grover", "grover_n25_iters5"),
    ("random", "random_brickwall_n25_d20"),
];

const N: u32 = 25;

fn fixture_path(stem: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("benches crate is one dir deep from repo root")
        .join("scripts/qiskit-baseline/circuits")
        .join(format!("{stem}.qasm"))
}

fn load(stem: &str) -> aleph_ir::Circuit {
    let path = fixture_path(stem);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
    aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse {} failed: {:?}", path.display(), e))
}

/// Raw `run`: parallelism lives in the gate kernels, so this isolates kernel
/// scaling. Headline scaling group.
fn bench_tier1_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_scaling");
    // n=25 is a 512 MiB state vector; keep the sample count low so a sweep
    // finishes in minutes. Criterion treats this as a floor for slow benches.
    group.sample_size(10);
    for &(label, stem) in WORKLOADS {
        let circuit = load(stem);
        group.throughput(Throughput::Elements(u64::from(N) * (1u64 << N)));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &circuit,
            |b, circuit| {
                b.iter_with_setup(
                    || NaiveSvBackend::with_seed(0),
                    |mut backend| {
                        let state = run(&mut backend, circuit).unwrap();
                        black_box(state);
                    },
                );
            },
        );
    }
    group.finish();
}

/// Fused path: optimize once outside the timed loop (the `run_optimized`
/// default pipeline), then time the parallel kernels on the fused circuit.
/// The honest end-to-end shape; QFT is known fused == raw, Grover/random may
/// differ. Compare its T1->Tn curve against the raw `tier1_scaling` group.
fn bench_tier1_scaling_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_scaling_fused");
    group.sample_size(10);
    for &(label, stem) in WORKLOADS {
        let mut circuit = load(stem);
        circuit.optimize().expect("optimize pipeline");
        group.throughput(Throughput::Elements(u64::from(N) * (1u64 << N)));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &circuit,
            |b, circuit| {
                b.iter_with_setup(
                    || NaiveSvBackend::with_seed(0),
                    |mut backend| {
                        let state = run(&mut backend, circuit).unwrap();
                        black_box(state);
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_tier1_scaling, bench_tier1_scaling_fused);
criterion_main!(benches);
