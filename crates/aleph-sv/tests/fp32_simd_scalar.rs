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
    apply_1q_dense_scalar_f32, apply_1q_diag_scalar_f32, apply_1q_f32, apply_2q_dense_scalar_f32,
    apply_2q_f32, apply_kq_f32, apply_kq_scalar_f32,
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

/// A genuinely dense (all entries nonzero) real 4×4 unitary: H⊗H, every
/// entry ±1/2. Exercises the full 4×4 matvec (every `m[r][c] * z_c` term)
/// through the broadcast complex-multiply building block.
fn dense_2q_m() -> [[Complex<f32>; 4]; 4] {
    let h = 0.5f32; // (1/√2)^2
    let s = |x: f32| Complex::new(x, 0.0);
    [
        [s(h), s(h), s(h), s(h)],
        [s(h), s(-h), s(h), s(-h)],
        [s(h), s(h), s(-h), s(-h)],
        [s(h), s(-h), s(-h), s(h)],
    ]
}

/// A dense 4×4 with nonzero imaginary parts in every entry, so the
/// `m_im` broadcast / `vpermilps`-swap half of the complex-multiply is
/// exercised too (H⊗H alone has zero imaginary parts). Not unitary — the
/// SIMD≡scalar comparison does not require unitarity, only that both
/// paths apply the same linear map.
fn dense_2q_m_complex() -> [[Complex<f32>; 4]; 4] {
    let mut m = [[Complex::new(0.0f32, 0.0); 4]; 4];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, e) in row.iter_mut().enumerate() {
            let rr = ((r * 4 + c) as f32 * 0.11 + 0.2).sin() * 0.5;
            let ii = ((r * 4 + c) as f32 * 0.17 + 0.3).cos() * 0.5;
            *e = Complex::new(rr, ii);
        }
    }
    m
}

#[test]
fn dense_2q_simd_matches_scalar() {
    // LANES_F32 = 8 ⇒ log2(LANES_F32) = 3; the SIMD arm requires
    // `1 << t_lo ≥ LANES_F32` (i.e. t_lo ≥ 3) AND every control > t_hi.
    // Lower-target pairs route to scalar in BOTH paths — included
    // deliberately so EPYC exercises the dispatcher boundary (this is
    // where P1-07's low-bit-target SIGSEGV/underflow hid) and confirms no
    // panic.
    for m in [dense_2q_m(), dense_2q_m_complex()] {
        for n in [4u32, 8, 12] {
            // Target pairs spanning positions, INCLUDING at least one pair
            // where a target ∈ {0, 1} (low-bit tier → scalar in both).
            let mut target_cases: Vec<[u32; 2]> = vec![
                [0, 1], // both low-bit (scalar path)
                [0, n - 1],
                [1, 2],
            ];
            // A high-aligned pair (both t_lo ≥ 3) that hits the SIMD arm on
            // AVX-512 when n is large enough.
            if n >= 8 {
                target_cases.push([3, 5]);
                target_cases.push([4, 6]);
                // Reversed orientation (targets[0] > targets[1]) to cover
                // the offset_k1/offset_k2 swap branch.
                target_cases.push([6, 4]);
            }

            for targets in target_cases {
                let t_lo = targets[0].min(targets[1]);
                let t_hi = targets[0].max(targets[1]);
                if t_lo == t_hi || t_hi >= n {
                    continue; // distinct + in range
                }

                // Control configurations: none, and (when room exists) one
                // external control strictly ABOVE t_hi — the SIMD contract.
                let mut control_cases: Vec<Vec<u32>> = vec![vec![]];
                if t_hi + 1 < n {
                    control_cases.push(vec![n - 1]);
                }

                for ctrls in control_cases {
                    let ctrls: Vec<u32> = ctrls
                        .iter()
                        .copied()
                        .filter(|&c| c > t_hi && c < n && c != targets[0] && c != targets[1])
                        .collect();

                    let base = fill(n);
                    let mut a_simd = base.clone();
                    let mut a_scalar = base.clone();

                    // Dispatcher → dense 2q SIMD arm on AVX-512 when the
                    // contract holds, scalar elsewhere.
                    apply_2q_f32(&mut a_simd, targets, &ctrls, &m);
                    // Forced scalar reference.
                    apply_2q_dense_scalar_f32(&mut a_scalar, targets, &ctrls, &m);

                    for i in 0..a_simd.len() {
                        assert!(
                            (a_simd[i].re - a_scalar[i].re).abs() < 1e-5
                                && (a_simd[i].im - a_scalar[i].im).abs() < 1e-5,
                            "2q mismatch n={n} targets={targets:?} ctrls={ctrls:?} i={i}: \
                             simd={:?} scalar={:?}",
                            a_simd[i],
                            a_scalar[i]
                        );
                    }
                }
            }
        }
    }
}

/// Deterministic dense `2^k × 2^k` matrix with every entry nonzero in both
/// real and imaginary parts. Not unitary — the SIMD≡scalar comparison only
/// needs both paths to apply the SAME linear map; varied magnitudes exercise
/// the full `2^k × 2^k` matvec (every broadcast `data[r*dim+c]` term and the
/// `m_im` half of the complex-multiply).
fn dense_kq_m(k: u8) -> Vec<Complex<f32>> {
    let dim = 1usize << k;
    (0..dim * dim)
        .map(|i| {
            let rr = ((i as f32) * 0.07 + 0.3).sin() * 0.4 - 0.2;
            let ii = ((i as f32) * 0.11 + 0.5).cos() * 0.4 - 0.15;
            Complex::new(rr, ii)
        })
        .collect()
}

#[test]
fn kq_simd_matches_scalar() {
    // The f32 kq SIMD arm engages when `outer = (1<<n) >> k >= 16` (= the
    // 16-lane i32 gather group) AND `k <= 4`. At n=12 every k ∈ {2,3,4} has
    // `outer ∈ {2^10, 2^9, 2^8} ≥ 16`, so the dispatcher takes the SIMD path
    // on AVX-512 (and scalar on aarch64, where the comparison is trivially
    // equal but the harness still exercises the index algebra).
    let n = 12u32;
    for k in [2u8, 3, 4] {
        let data = dense_kq_m(k);

        // Qubit subsets covering low and high positions (and non-adjacent
        // spreads). `targets_offsets_fixed` sorts internally, so order in the
        // slice is irrelevant; pick distinct, in-range subsets of size k.
        let subsets: Vec<Vec<u32>> = match k {
            2 => vec![
                vec![0, 1],   // lowest pair
                vec![0, 11],  // low + top
                vec![10, 11], // top pair
                vec![3, 7],   // mid spread
            ],
            3 => vec![
                vec![0, 1, 2],   // lowest triple
                vec![0, 5, 11],  // low + mid + top
                vec![9, 10, 11], // top triple
                vec![1, 4, 8],   // non-adjacent spread
            ],
            4 => vec![
                vec![0, 1, 2, 3],   // lowest quad
                vec![0, 4, 8, 11],  // spread incl top
                vec![8, 9, 10, 11], // top quad
                vec![2, 5, 7, 10],  // non-adjacent spread
            ],
            _ => unreachable!(),
        };

        for qubits in subsets {
            let base = fill(n);
            let mut a_simd = base.clone();
            let mut a_scalar = base.clone();

            // Dispatcher → kq SIMD arm on AVX-512, scalar elsewhere.
            apply_kq_f32(&mut a_simd, &qubits, k, &data);
            // Forced scalar reference.
            apply_kq_scalar_f32(&mut a_scalar, &qubits, k, &data);

            for i in 0..a_simd.len() {
                assert!(
                    (a_simd[i].re - a_scalar[i].re).abs() < 1e-5
                        && (a_simd[i].im - a_scalar[i].im).abs() < 1e-5,
                    "kq mismatch n={n} k={k} qubits={qubits:?} i={i}: \
                     simd={:?} scalar={:?}",
                    a_simd[i],
                    a_scalar[i]
                );
            }
        }
    }
}
