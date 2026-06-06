//! P4-02 Grover benchmark over the committed corpus QASM (the SAME files Aer
//! times), run through the canonical optimized state-vector path. This is the
//! aleph half of the Phase-4 Grover report row. Mirrors phase4_qft.rs.
//!
//! n=4/8/12 run anywhere (state <= 4096 amplitudes). n=16 is 2.26M gates; its
//! corpus is generated on demand (gitignored, ~34 MB) and its report number is
//! taken from the `oneshot` single-shot path in the EPYC run (same split as QFT
//! n=30). It is available here behind `scaling-bench` for spot checks — generate
//! the corpus first with `python scripts/qiskit-baseline/run.py --gen-only`.
//!
//! Corpus: scripts/qiskit-baseline/circuits/grover_n{N}_iters{opt}.qasm.

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// Grover sizes always benched (tiny state, fine for any host / CI bench job).
const SMALL_N: &[u32] = &[4, 8, 12];
/// Large size gated behind `scaling-bench` (2.26M gates, on-demand corpus). The
/// report number comes from `oneshot`; this path stays runnable for spot checks.
#[cfg(feature = "scaling-bench")]
const LARGE_N: &[u32] = &[16];

/// round(pi/4 * sqrt(2^n)); mirrors run.py::grover_optimal_iters so the corpus
/// filename matches.
fn optimal_iters(n: u32) -> u32 {
    (std::f64::consts::PI / 4.0 * (2f64.powi(n as i32)).sqrt()).round() as u32
}

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().expect("workspace root").join(format!(
        "scripts/qiskit-baseline/circuits/grover_n{n}_iters{}.qasm",
        optimal_iters(n)
    ))
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n: u32) {
    let src =
        std::fs::read_to_string(corpus_path(n)).unwrap_or_else(|e| panic!("read grover_n{n}: {e}"));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse grover_n{n}: {e:?}"));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        b.iter(|| {
            // Raised cap matches phase4_qft (only gates allocate(); small n
            // unaffected). All grover n <= 16 are well under the default anyway.
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
            let state = run_optimized(&mut backend, black_box(&circuit)).expect("simulate");
            black_box(state.amplitudes().len())
        })
    });
}

fn grover(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase4_grover");
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

criterion_group!(benches, grover);
criterion_main!(benches);
