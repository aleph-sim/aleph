//! P4-03 QPE benchmark over the committed corpus QASM (the SAME files Aer
//! times), run through the canonical optimized state-vector path. This is the
//! aleph half of the Phase-4 QPE report row. Mirrors phase4_qft.rs.
//!
//! n in {10,15,20,25} (n=25 => 512 MiB state). Every size has a committed
//! corpus and is criterion-measurable single-thread, so there is no
//! scaling-bench / oneshot tier (unlike QFT-30 or Grover-16).
//!
//! Corpus: scripts/qiskit-baseline/circuits/qpe_n{N}.qasm.

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// QPE sizes (all committed, all criterion-measurable single-thread).
const SMALL_N: &[u32] = &[10, 15, 20, 25];

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("workspace root")
        .join(format!("scripts/qiskit-baseline/circuits/qpe_n{n}.qasm"))
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n: u32) {
    let src = std::fs::read_to_string(corpus_path(n))
        .unwrap_or_else(|e| panic!("read qpe_n{n}.qasm: {e}"));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse qpe_n{n}: {e:?}"));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        b.iter(|| {
            // Raised cap matches phase4_qft (only gates allocate(); small n
            // unaffected).
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
            let state = run_optimized(&mut backend, black_box(&circuit)).expect("simulate");
            black_box(state.amplitudes().len())
        })
    });
}

fn qpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase4_qpe");
    // n=25 allocates 512 MiB; keep sample counts modest.
    group.sample_size(10);
    for &n in SMALL_N {
        bench_one(&mut group, n);
    }
    group.finish();
}

criterion_group!(benches, qpe);
criterion_main!(benches);
