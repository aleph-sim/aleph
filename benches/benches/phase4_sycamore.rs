//! P4-06 Sycamore-style random-circuit benchmark over the committed corpus
//! QASM (the SAME files Aer times), run through the optimized state-vector
//! path. The aleph half of the Phase-4 Sycamore report row. Mirrors
//! phase4_qpe.rs.
//!
//! Criterion sizes n in {20,24} (n=24 => 256 MiB state). The heavy n=28
//! (4 GiB) and n=30 (16 GiB) are measured single-shot via the `oneshot`
//! bin instead, exactly like QFT-30.
//!
//! Corpus: scripts/qiskit-baseline/circuits/sycamore_n{N}_d20.qasm.

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

const SMALL_N: &[u32] = &[20, 24];
const DEPTH: u32 = 20;

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().expect("workspace root").join(format!(
        "scripts/qiskit-baseline/circuits/sycamore_n{n}_d{DEPTH}.qasm"
    ))
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n: u32) {
    let src = std::fs::read_to_string(corpus_path(n))
        .unwrap_or_else(|e| panic!("read sycamore_n{n}_d{DEPTH}.qasm: {e}"));
    let circuit =
        aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse sycamore_n{n}: {e:?}"));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        b.iter(|| {
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
            let state = run_optimized(&mut backend, black_box(&circuit)).expect("simulate");
            black_box(state.amplitudes().len())
        })
    });
}

fn sycamore(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase4_sycamore");
    // n=24 allocates 256 MiB; keep sample counts modest.
    group.sample_size(10);
    for &n in SMALL_N {
        bench_one(&mut group, n);
    }
    group.finish();
}

criterion_group!(benches, sycamore);
criterion_main!(benches);
