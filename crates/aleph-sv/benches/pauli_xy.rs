//! P1-05 micro-bench: anti-diagonal kernels vs generic 2×2.
//!
//! `n = 14` (state = 256 KiB) lands in L2 on EPYC 8124P (1 MiB L2 per
//! core). At this size SIMD per-µop reduction translates to wall-clock,
//! avoiding the bandwidth-bound regime documented in P1-06 lessons
//! (ADR 0008). For larger n=20 numbers, see the workload benches
//! and `docs/perf/phase1-vs-qiskit.md`.
//!
//! Bench inventory (12 total):
//!   * `p1_05_specialised_{x,y,antidiag}`             — Tier A target=8 via dispatch
//!   * `p1_05_specialised_{x,y,antidiag}_tier_b`      — Tier B target=0 via dispatch
//!   * `p1_05_generic_baseline_{x,y,antidiag}`        — scalar inner-loop (upper bound)
//!   * `p1_05_generic_avx512_baseline_{x,y,antidiag}` — generic packed-complex AVX-512
//!     (honest pre-P1-05 baseline)

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

/// Tier-A path: target=8 → `1 << 8 = 256 ≥ LANES=4`.
/// Engages the specialised AVX-512 kernels via the dispatcher on EPYC;
/// falls through to scalar on non-AVX-512 hosts.
fn bench_specialised(c: &mut Criterion, label: &str, m: [[Complex; 2]; 2]) {
    let mut state = random_state();
    c.bench_function(&format!("p1_05_specialised_{label}"), |b| {
        b.iter(|| {
            kernels::aos::apply_1q(black_box(&mut state), 8, black_box(&[]), black_box(&m));
        })
    });
}

/// Tier-B path: target=0 → `1 << 0 = 1 < LANES=4` (in-register lane permute).
/// On EPYC this exercises the Tier-B lowbit AVX-512 kernels when amps.len()
/// is divisible by LANES=4 (always true here with LEN=16384).
fn bench_specialised_tier_b(c: &mut Criterion, label: &str, m: [[Complex; 2]; 2]) {
    let mut state = random_state();
    c.bench_function(&format!("p1_05_specialised_{label}_tier_b"), |b| {
        b.iter(|| {
            kernels::aos::apply_1q(black_box(&mut state), 0, black_box(&[]), black_box(&m));
        })
    });
}

/// Scalar upper-bound baseline: generic 2×2 scalar loop with no dispatch
/// overhead. Gives a fair upper bound for the specialised path to beat.
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

/// Honest AVX-512 baseline: the generic packed-complex `apply_1q_avx512`
/// kernel at target=8 with no controls — this is what pre-P1-05 dispatch
/// would have called for any 1q gate on EPYC.  Comparing `p1_05_specialised_*`
/// against this (not against the scalar baseline) gives the REAL speedup.
fn bench_generic_avx512_baseline(c: &mut Criterion, label: &str, m: [[Complex; 2]; 2]) {
    #[cfg(target_arch = "x86_64")]
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("skipping generic_avx512_baseline_{label}: avx512f not available");
        return;
    }
    #[allow(unused_mut)] // mut only used inside the x86_64 cfg block below
    let mut state = random_state();
    c.bench_function(&format!("p1_05_generic_avx512_baseline_{label}"), |b| {
        b.iter(|| {
            // SAFETY: feature checked + target_bit=256 ≥ LANES=4 + no controls.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                aleph_sv::kernels::aos::apply_1q_avx512(
                    black_box(&mut state),
                    8,
                    black_box(&[]),
                    black_box(&m),
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                // Non-x86_64: fall through to the scalar inner body as a no-op
                // placeholder (bench won't run on these hosts anyway).
                let _ = &m;
                black_box(&state);
            }
        })
    });
}

fn benches(c: &mut Criterion) {
    // Tier-A specialised kernels (honest comparison: vs generic_avx512_baseline)
    bench_specialised(c, "x", pauli_x_matrix());
    bench_specialised(c, "y", pauli_y_matrix());
    bench_specialised(c, "antidiag", generic_antidiag_matrix());

    // Tier-B specialised kernels (target=0, in-register lane permute)
    bench_specialised_tier_b(c, "x", pauli_x_matrix());
    bench_specialised_tier_b(c, "y", pauli_y_matrix());
    bench_specialised_tier_b(c, "antidiag", generic_antidiag_matrix());

    // Scalar upper-bound baseline (illustrates the gap over a naive loop)
    bench_generic_baseline(c, "x", pauli_x_matrix());
    bench_generic_baseline(c, "y", pauli_y_matrix());
    bench_generic_baseline(c, "antidiag", generic_antidiag_matrix());

    // Honest AVX-512 baseline (pre-P1-05 dispatch; the ADR 0011 reference point)
    bench_generic_avx512_baseline(c, "x", pauli_x_matrix());
    bench_generic_avx512_baseline(c, "y", pauli_y_matrix());
    bench_generic_avx512_baseline(c, "antidiag", generic_antidiag_matrix());
}

criterion_group!(p1_05, benches);
criterion_main!(p1_05);
