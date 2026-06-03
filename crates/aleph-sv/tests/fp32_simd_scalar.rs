//! P2-08 (Phase B): the f32 AVX-512 kernels must agree closely with the
//! f32 scalar reference across the index-coverage space (all target
//! positions, with and without an external control above the target).
//!
//! The dispatcher `apply_1q_f32` routes to the AVX-512 kernel on
//! x86_64 + AVX-512 when the dispatch contract holds (`target_bit ≥
//! LANES_F32`, all controls > target); `apply_1q_dense_scalar_f32` is the
//! forced scalar reference. On non-AVX-512 hosts (e.g. the aarch64 dev
//! box) the dispatcher itself routes to scalar, so these comparisons are
//! trivially equal there — the test still compiles and runs, and becomes
//! a genuine SIMD≡scalar check when run on an AVX-512 (EPYC) box.
//!
//! Run with: `cargo test -p aleph-sv --features internal-bench
//! --test fp32_simd_scalar` (the `internal-bench` feature flips
//! `kernels` to `pub`, mirroring `policy_invariance`).

use aleph_core::Complex;
use aleph_sv::kernels::aos_f32::{
    apply_1q_dense_scalar_f32, apply_1q_diag_scalar_f32, apply_1q_f32,
};

/// Deterministic, non-trivial, well-spread complex pattern.
fn fill(n: u32) -> Vec<Complex<f32>> {
    (0..(1usize << n))
        .map(|i| Complex::new((i as f32 * 0.013).sin(), (i as f32 * 0.027).cos()))
        .collect()
}

/// A generic (non-diagonal, non-anti-diagonal) 2×2 unitary so the
/// dispatcher takes the dense path (and the SIMD arm on AVX-512), not the
/// diagonal fast path. Entries mix real and imaginary parts to exercise
/// the full complex-multiply (`fmaddsub` + `vpermilps` swap).
fn test_m() -> [[Complex<f32>; 2]; 2] {
    let a = std::f32::consts::FRAC_1_SQRT_2;
    [
        [
            Complex::new(a * 0.6, a * 0.3),
            Complex::new(-a * 0.2, a * 0.7),
        ],
        [
            Complex::new(a * 0.4, -a * 0.5),
            Complex::new(a * 0.8, a * 0.1),
        ],
    ]
}

#[test]
fn dense_1q_simd_matches_scalar() {
    let m = test_m();
    // Cover targets across the full 0..n range, including low-bit targets
    // (< log2(LANES_F32) = 3) which route to scalar in BOTH paths — that
    // boundary is still worth exercising on EPYC.
    for n in [4u32, 8, 12] {
        for target in 0..n {
            // Two control configurations: none, and (when room exists) one
            // external control strictly ABOVE the target — the SIMD
            // contract requires every control > target.
            let mut control_cases: Vec<Vec<u32>> = vec![vec![]];
            // Pick the highest qubit above target (and != target) as a
            // control when available.
            if target + 1 < n {
                control_cases.push(vec![n - 1]);
            }

            for ctrls in control_cases {
                // Filter to honour the contract (control strictly above
                // target and in range); empties stay empty.
                let ctrls: Vec<u32> = ctrls
                    .iter()
                    .copied()
                    .filter(|&c| c > target && c < n)
                    .collect();

                let base = fill(n);
                let mut a_simd = base.clone();
                let mut a_scalar = base.clone();

                apply_1q_f32(&mut a_simd, target, &ctrls, &m);
                apply_1q_dense_scalar_f32(&mut a_scalar, target, &ctrls, &m);

                for i in 0..a_simd.len() {
                    assert!(
                        (a_simd[i].re - a_scalar[i].re).abs() < 1e-5
                            && (a_simd[i].im - a_scalar[i].im).abs() < 1e-5,
                        "mismatch n={n} target={target} ctrls={ctrls:?} i={i}: \
                         simd={:?} scalar={:?}",
                        a_simd[i],
                        a_scalar[i]
                    );
                }
            }
        }
    }
}

/// Two non-trivial diagonal 2×2 matrices (off-diagonals exactly zero) so
/// the dispatcher takes the diagonal fast path (and its AVX-512 arm on
/// AVX-512 hardware). `d0` carries a non-unit modulus, `d1` is an S-gate
/// phase (`e^{iπ/2} = i`) — together they exercise the broadcast complex
/// scale (`fmaddsub` + `vpermilps` swap) on both the 0-side and 1-side.
fn diag_m() -> [[Complex<f32>; 2]; 2] {
    let zero = Complex::new(0.0f32, 0.0);
    [
        [Complex::new(0.6, -0.8), zero], // d0: |d0| = 1 but non-trivial re+im
        [zero, Complex::new(0.0, 1.0)],  // d1: S gate (i)
    ]
}

#[test]
fn diag_1q_simd_matches_scalar() {
    let m = diag_m();
    // Off-diagonals must be exactly zero so the dispatcher routes to the
    // diagonal path (not the dense arm).
    debug_assert!(m[0][1].re == 0.0 && m[0][1].im == 0.0);
    debug_assert!(m[1][0].re == 0.0 && m[1][0].im == 0.0);

    for n in [4u32, 8, 12] {
        for target in 0..n {
            // None, and (when room exists) one external control strictly
            // ABOVE the target — the SIMD contract requires control > target.
            let mut control_cases: Vec<Vec<u32>> = vec![vec![]];
            if target + 1 < n {
                control_cases.push(vec![n - 1]);
            }

            for ctrls in control_cases {
                let ctrls: Vec<u32> = ctrls
                    .iter()
                    .copied()
                    .filter(|&c| c > target && c < n)
                    .collect();

                let base = fill(n);
                let mut a_simd = base.clone();
                let mut a_scalar = base.clone();

                // Dispatcher → diagonal SIMD arm on AVX-512, scalar elsewhere.
                apply_1q_f32(&mut a_simd, target, &ctrls, &m);
                // Forced scalar reference.
                apply_1q_diag_scalar_f32(&mut a_scalar, target, &ctrls, m[0][0], m[1][1]);

                for i in 0..a_simd.len() {
                    assert!(
                        (a_simd[i].re - a_scalar[i].re).abs() < 1e-5
                            && (a_simd[i].im - a_scalar[i].im).abs() < 1e-5,
                        "diag mismatch n={n} target={target} ctrls={ctrls:?} i={i}: \
                         simd={:?} scalar={:?}",
                        a_simd[i],
                        a_scalar[i]
                    );
                }
            }
        }
    }
}
