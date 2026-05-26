//! Phase 1, Stage 0: time `NaiveSvBackend` on the same QASM circuits Qiskit
//! Aer runs (`scripts/qiskit-baseline/circuits/`).  Report numbers feed
//! `docs/perf/phase1-vs-qiskit.md`.
//!
//! `NaiveSvBackend` is the AoS + AVX-512 path post-P1-03 (see ADR 0008) — the
//! canonical fast x86 backend.  Runs scalar on non-AVX-512 hosts; that's fine
//! locally but EPYC is the authoritative measurement target.
//!
//! `SoaSvBackend` is included for appendix-table triangulation per the spec
//! (`docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md` § 4.2).

use aleph_backend::run;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::PathBuf;

const WORKLOADS: &[&str] = &["qft_n20", "grover_n20_iters5", "random_brickwall_n20_d20"];

fn fixture_path(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("benches crate is one dir deep from repo root")
        .join("scripts/qiskit-baseline/circuits")
        .join(format!("{name}.qasm"))
}

fn bench_qiskit_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("qiskit_baseline");
    // n·2^n keeps bencher.dev's elements/s axis aligned with existing
    // benches/qft.rs.  All three workloads run at n=20 so the throughput
    // setting is per-group rather than per-workload.
    group.throughput(Throughput::Elements(20u64 * (1u64 << 20)));

    for &name in WORKLOADS {
        let path = fixture_path(name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
        let circuit = aleph_parser::parse(&src)
            .unwrap_or_else(|e| panic!("parse {} failed: {:?}", path.display(), e));

        // Headline: NaiveSvBackend (AoS + AVX-512).
        group.bench_with_input(
            BenchmarkId::new("naive_aos_avx512", name),
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

        // Appendix triangulation: SoaSvBackend.
        group.bench_with_input(BenchmarkId::new("soa", name), &circuit, |b, circuit| {
            b.iter_with_setup(
                || SoaSvBackend::with_seed(0),
                |mut backend| {
                    let state = run(&mut backend, circuit).unwrap();
                    black_box(state);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qiskit_baseline);
criterion_main!(benches);
