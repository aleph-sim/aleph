//! P1-05 micro-bench: anti-diagonal kernels vs generic 2×2.
//!
//! `n = 14` (state = 256 KiB) lands in L2 on EPYC 8124P (1 MiB L2 per
//! core). At this size SIMD per-µop reduction translates to wall-clock,
//! avoiding the bandwidth-bound regime documented in P1-06 lessons
//! (ADR 0008). For larger n=20 numbers, see the workload benches
//! and `docs/perf/phase1-vs-qiskit.md`.

use aleph_core::Complex;
use aleph_sv::kernels;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const N: u32 = 14;
const LEN: usize = 1 << N;

fn random_state() -> Vec<Complex> {
    (0..LEN)
        .map(|k| {
            let r = ((k as u64).wrapping_mul(2_654_435_761) as f64) * 1e-19;
            Complex::new(r.sin(), r.cos())
        })
        .collect()
}

fn pauli_x_matrix() -> [[Complex; 2]; 2] {
    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    [[z, o], [o, z]]
}

fn pauli_y_matrix() -> [[Complex; 2]; 2] {
    let z = Complex::new(0.0, 0.0);
    let pi = Complex::new(0.0, 1.0);
    let ni = Complex::new(0.0, -1.0);
    [[z, ni], [pi, z]]
}

fn generic_antidiag_matrix() -> [[Complex; 2]; 2] {
    let z = Complex::new(0.0, 0.0);
    // a = 0.6 + 0.8i, b = 0.6 − 0.8i: |a| = |b| = 1, unitary anti-diagonal.
    let a = Complex::new(0.6, 0.8);
    let b = Complex::new(0.6, -0.8);
    [[z, a], [b, z]]
}

/// Apply via the dispatch path (engages specialised kernels via
/// classifier + AVX-512 Tier A on EPYC, scalar fallback elsewhere).
fn bench_specialised(c: &mut Criterion, label: &str, m: [[Complex; 2]; 2]) {
    let mut state = random_state();
    c.bench_function(&format!("p1_05_specialised_{label}"), |b| {
        b.iter(|| {
            kernels::aos::apply_1q(black_box(&mut state), 8, black_box(&[]), black_box(&m));
        })
    });
}

/// Baseline: generic 2×2 scalar loop with no dispatch overhead.
/// Inlines the scalar inner body of `apply_1q` directly, giving a
/// fair upper bound for the specialised path to beat.
fn bench_generic_baseline(c: &mut Criterion, label: &str, m: [[Complex; 2]; 2]) {
    let mut state = random_state();
    c.bench_function(&format!("p1_05_generic_baseline_{label}"), |b| {
        b.iter(|| {
            let t_bit = 1usize << 8;
            let mut i = 0usize;
            while i < state.len() {
                if i & t_bit == 0 {
                    let j = i | t_bit;
                    let a = state[i];
                    let bv = state[j];
                    state[i] = m[0][0] * a + m[0][1] * bv;
                    state[j] = m[1][0] * a + m[1][1] * bv;
                }
                i += 1;
            }
            black_box(&state);
        })
    });
}

fn benches(c: &mut Criterion) {
    bench_specialised(c, "x", pauli_x_matrix());
    bench_specialised(c, "y", pauli_y_matrix());
    bench_specialised(c, "antidiag", generic_antidiag_matrix());

    bench_generic_baseline(c, "x", pauli_x_matrix());
    bench_generic_baseline(c, "y", pauli_y_matrix());
    bench_generic_baseline(c, "antidiag", generic_antidiag_matrix());
}

criterion_group!(p1_05, benches);
criterion_main!(p1_05);
