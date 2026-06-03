//! P2-06 diagonal-fusion before/after benchmark.
//!
//! Isolates the impact of [`FuseDiagonalRuns`] on the QFT controlled-phase
//! ladder: the same circuit is optimized two ways — with the pre-P2-06
//! pipeline (`Cancel, DCE, Fuse1q, Fuse2q`) and with the current default
//! pipeline (which inserts `FuseDiagonalRuns` before `Fuse2q`) — then each
//! optimized circuit is executed through the AoS + AVX-512 `NaiveSvBackend`.
//! The wall-clock delta is the memory-pass reduction P2-06 buys.
//!
//! Two QFT encodings are measured (they fuse very differently — see
//! docs/perf/phase2.md / the P2-06 design doc):
//!   * `builder`  — controlled-`Phase` gates (the whole ladder is diagonal).
//!   * `fixture`  — the Aer-comparable `qft_n25.qasm`, decomposed to `p`+`cx`
//!     (diagonal fusion must absorb the interleaved `cx`s to collapse it).
//!
//! Gated behind `scaling-bench` so `cargo bench --workspace` / CI skip the
//! 512 MiB n=25 runs. Run on a verified-idle box (CLAUDE.md):
//!
//!   cargo bench -p aleph-benches --bench diagonal_fusion --features scaling-bench
//!
//! The instruction-count reduction (the "≥5× fewer passes" acceptance
//! criterion) is printed to stderr once at startup.

use aleph_backend::run;
use aleph_ir::passes::{
    CancelInversePairs, DeadCodeElim, Fuse1qRuns, Fuse2q, FuseDiagonalRuns, PassPipeline,
};
use aleph_ir::Circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// Builder QFT (controlled-Phase form).
fn builder_qft(n: u32) -> Circuit {
    aleph_benches::qft_circuit(n)
}

/// Decomposed fixture QFT (`p`+`cx`), parsed from the Aer-comparable file.
fn fixture_qft() -> Circuit {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .parent()
        .expect("benches crate is one dir deep from repo root")
        .join("scripts/qiskit-baseline/circuits/qft_n25.qasm");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
    aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse {} failed: {:?}", path.display(), e))
}

/// The pre-P2-06 pipeline: no diagonal fusion.
fn pipeline_without_diagonal() -> PassPipeline {
    PassPipeline::new(vec![
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
    ])
}

/// The current default pipeline: includes `FuseDiagonalRuns` before `Fuse2q`.
fn pipeline_with_diagonal() -> PassPipeline {
    PassPipeline::new(vec![
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        Box::new(FuseDiagonalRuns),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
    ])
}

fn optimized(base: &Circuit, pipeline: &PassPipeline) -> Circuit {
    let mut c = base.clone();
    pipeline.run(&mut c).expect("pipeline run");
    c
}

/// Print the instruction-count (memory-pass) reduction once, for the PR.
fn report_pass_counts() {
    let without = pipeline_without_diagonal();
    let with = pipeline_with_diagonal();
    eprintln!("--- P2-06 instruction-count (memory-pass) reduction ---");
    let mut cases: Vec<(String, Circuit)> = Vec::new();
    for n in [20u32, 22, 25] {
        cases.push((format!("builder_qft_n{n}"), builder_qft(n)));
    }
    cases.push(("fixture_qft_n25".to_string(), fixture_qft()));
    for (label, base) in &cases {
        let a = optimized(base, &without).len();
        let b = optimized(base, &with).len();
        let ratio = a as f64 / b.max(1) as f64;
        eprintln!("  {label:<18} without={a:>5}  with={b:>5}  reduction={ratio:>5.2}x");
    }
    eprintln!("-------------------------------------------------------");
}

fn bench_diagonal_fusion(c: &mut Criterion) {
    report_pass_counts();

    let without = pipeline_without_diagonal();
    let with = pipeline_with_diagonal();

    let mut group = c.benchmark_group("qft_diagonal_fusion");
    // n=25 is 512 MiB / run — criterion's sample_size is a floor for slow
    // benches, so keep it small (see CLAUDE.md / P2-05 ops notes).
    group.sample_size(10);

    // Builder QFT at a few sizes.
    for n in [22u32, 25] {
        let base = builder_qft(n);
        let opt_without = optimized(&base, &without);
        let opt_with = optimized(&base, &with);

        group.bench_with_input(BenchmarkId::new("builder_without", n), &opt_without, |b, c| {
            b.iter(|| {
                let mut be = NaiveSvBackend::new();
                black_box(run(&mut be, black_box(c)).unwrap());
            })
        });
        group.bench_with_input(BenchmarkId::new("builder_with", n), &opt_with, |b, c| {
            b.iter(|| {
                let mut be = NaiveSvBackend::new();
                black_box(run(&mut be, black_box(c)).unwrap());
            })
        });
    }

    // Decomposed fixture QFT-25.
    let base = fixture_qft();
    let opt_without = optimized(&base, &without);
    let opt_with = optimized(&base, &with);
    group.bench_with_input(BenchmarkId::new("fixture_without", 25), &opt_without, |b, c| {
        b.iter(|| {
            let mut be = NaiveSvBackend::new();
            black_box(run(&mut be, black_box(c)).unwrap());
        })
    });
    group.bench_with_input(BenchmarkId::new("fixture_with", 25), &opt_with, |b, c| {
        b.iter(|| {
            let mut be = NaiveSvBackend::new();
            black_box(run(&mut be, black_box(c)).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench_diagonal_fusion);
criterion_main!(benches);
