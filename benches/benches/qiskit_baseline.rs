//! Phase 1 performance matrix: time `NaiveSvBackend` on the same QASM circuits
//! Qiskit Aer runs (`scripts/qiskit-baseline/circuits/`).  Report numbers feed
//! `docs/perf/phase1.md` (P1-14); the Stage-0 snapshot was
//! `docs/perf/phase1-vs-qiskit.md`.
//!
//! Matrix: {ghz, qft, grover, random_brickwall} × n ∈ {15, 20, 22, 25}.  The
//! full matrix runs only under `ALEPH_BENCH_FULL_MATRIX=1` (a manual EPYC run);
//! the default is a cheap CI subset (n ≤ 20, no grover) that stays under the
//! Bench workflow's 30-minute timeout.
//!
//! `NaiveSvBackend` is the AoS + AVX-512 path post-P1-03 (see ADR 0008) — the
//! canonical fast x86 backend.  Runs scalar on non-AVX-512 hosts; that's fine
//! locally but EPYC is the authoritative measurement target.
//!
//! `SoaSvBackend` is included for appendix-table triangulation (n ≤ 20 only).

use aleph_backend::run;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

const N_LIST: &[u32] = &[15, 20, 22, 25];
const FAMILIES: &[&str] = &["ghz", "qft", "grover", "random_brickwall"];

fn workload_name(family: &str, n: u32) -> String {
    match family {
        "grover" => format!("grover_n{n}_iters5"),
        "random_brickwall" => format!("random_brickwall_n{n}_d20"),
        _ => format!("{family}_n{n}"),
    }
}

/// (name, n). Full matrix only when `ALEPH_BENCH_FULL_MATRIX=1`; otherwise a
/// cheap CI subset (n<=20, no grover) that stays well under the Bench
/// workflow's 30-minute timeout. The full matrix is a manual EPYC run.
fn selected_workloads() -> Vec<(String, u32)> {
    let full = std::env::var("ALEPH_BENCH_FULL_MATRIX")
        .map(|v| v != "0")
        .unwrap_or(false);
    let mut out = Vec::new();
    for &family in FAMILIES {
        for &n in N_LIST {
            if !full && (n > 20 || family == "grover") {
                continue; // CI subset: fast cells only
            }
            out.push((workload_name(family, n), n));
        }
    }
    out
}

fn fixture_path(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("benches crate is one dir deep from repo root")
        .join("scripts/qiskit-baseline/circuits")
        .join(format!("{name}.qasm"))
}

/// Per-cell criterion budget. Large n is minutes/iter, so shrink the sample
/// count (disclosed in the report's RSD table). Grover is the most expensive.
fn sample_budget_for(name: &str, n: u32) -> (usize, Duration) {
    if name.starts_with("grover_") && n >= 22 {
        (10, Duration::from_secs(30))
    } else if n >= 22 {
        (10, Duration::from_secs(20))
    } else if name.starts_with("grover_") {
        (10, Duration::from_secs(20))
    } else {
        (50, Duration::from_secs(10))
    }
}

fn bench_qiskit_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("qiskit_baseline");

    for (name, n) in selected_workloads() {
        // n·2^n elements/s axis (matches benches/qft.rs).
        group.throughput(Throughput::Elements(n as u64 * (1u64 << n)));
        let (samples, m_time) = sample_budget_for(&name, n);
        group.sample_size(samples).measurement_time(m_time);

        let path = fixture_path(&name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
        let circuit = aleph_parser::parse(&src)
            .unwrap_or_else(|e| panic!("parse {} failed: {:?}", path.display(), e));

        // Headline: NaiveSvBackend (AoS + AVX-512).
        group.bench_with_input(
            BenchmarkId::new("naive_aos_avx512", &name),
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

        // SoA appendix only at n<=20 (known ~2.3x slower; skip the n>=22 time).
        if n <= 20 {
            group.bench_with_input(BenchmarkId::new("soa", &name), &circuit, |b, circuit| {
                b.iter_with_setup(
                    || SoaSvBackend::with_seed(0),
                    |mut backend| {
                        let state = run(&mut backend, circuit).unwrap();
                        black_box(state);
                    },
                );
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_qiskit_baseline);
criterion_main!(benches);
