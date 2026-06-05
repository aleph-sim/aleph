//! P4-01 QFT benchmark over the committed corpus QASM (the SAME files Aer
//! times), run through the canonical optimized state-vector path. This is the
//! aleph half of the Phase-4 QFT report row.
//!
//! n=10/15/20/25 run anywhere; n=28/30 allocate ≥4/16 GiB and are gated behind
//! the `scaling-bench` feature so default `cargo bench --workspace` / CI skip
//! them. Measure on the EPYC box:
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches \
//!       --bench phase4_qft --features scaling-bench
//!
//! The corpus QASM lives at `scripts/qiskit-baseline/circuits/qft_n{N}.qasm`.

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// QFT sizes always benched (small enough for any host / CI bench job).
const SMALL_N: &[u32] = &[10, 15, 20, 25];
/// Large sizes gated behind `scaling-bench` (≥4 GiB state vector).
#[cfg(feature = "scaling-bench")]
const LARGE_N: &[u32] = &[28, 30];

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("workspace root")
        .join(format!("scripts/qiskit-baseline/circuits/qft_n{n}.qasm"))
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n: u32) {
    let src = std::fs::read_to_string(corpus_path(n))
        .unwrap_or_else(|e| panic!("read qft_n{n}.qasm: {e}"));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse qft_n{n}: {e:?}"));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        b.iter(|| {
            // Raised cap so n=30 is permitted on a host with enough RAM; small
            // n are unaffected (cap only gates allocate()).
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
            let state = run_optimized(&mut backend, black_box(&circuit)).expect("simulate");
            black_box(state.amplitudes().len())
        })
    });
}

fn qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase4_qft");
    // n=25 already allocates 512 MiB; keep sample counts modest.
    group.sample_size(10);
    for &n in SMALL_N {
        bench_one(&mut group, n);
    }
    #[cfg(feature = "scaling-bench")]
    for &n in LARGE_N {
        bench_one(&mut group, n);
    }
    group.finish();
}

criterion_group!(benches, qft);
criterion_main!(benches);
