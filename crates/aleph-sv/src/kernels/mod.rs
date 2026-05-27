//! Indexed gate application kernels.
//!
//! Two layouts share the same MSB qubit-ordering convention (ADR 0004
//! / P0-06 spec §6): `qubits[0]` is the MSB of the matrix index. They
//! diverge only in storage:
//!
//! * `aos` — `Vec<Complex<f64>>` (the naive `Vec<num_complex::Complex>`
//!   layout used by `NaiveSvBackend`).
//! * `soa` — paired `Vec<f64>` (real, imaginary) used by `SoaSvBackend`
//!   (P1-01). Same algorithms, layout chosen for SIMD-friendly
//!   sequential reads — explicit vectorisation lands in P1-03 / P1-04.

pub(crate) mod aos;
pub(crate) mod soa;

/// Bitwise-OR of `1 << q` over `controls`. Layout-agnostic — used by
/// both AoS and SoA kernels to compute the control gate-mask.
///
/// Returns `usize` so the result composes directly with index
/// arithmetic in the kernel loops; `q` is bounded by `state.num_qubits`
/// at the apply_gate boundary, which itself is capped at `MAX_*_QUBITS
/// ≤ 28`, so `1 << q` never overflows on any supported platform.
pub(crate) fn control_mask(controls: &[u32]) -> usize {
    let mut mask: usize = 0;
    for &c in controls {
        mask |= 1usize << c;
    }
    mask
}

/// Expand a "free-bit counter" `k` into a full bit index by
/// interleaving `k`'s bits into the **free** positions, with the
/// **fixed** bit positions set to their prescribed value. `fixed`
/// MUST be sorted by ascending position (caller's responsibility —
/// the SIMD kernels hoist this sort once outside their outer loops).
///
/// Used by the controlled AVX-512 kernel (P1-03,
/// `aos::apply_1q_avx512`): the outer loop counts `k` over
/// `2^(n_qubits − target − 1 − controls.len())` free-bit values; for
/// each `k`, `expand_with_fixed(k, &sorted_controls_renormalised)`
/// is the base index of the next outer block where every control is
/// set and the target + below-target bits are clear (the inner SIMD
/// walk fills those).
///
/// Bit positions in `fixed.0` are `u32` to match `Gate` qubit
/// indices; the caller guarantees they are < 64 (in practice < 28
/// since `MAX_*_QUBITS ≤ 28`), so the `1usize << pos` shifts never
/// overflow.
// Allow dead_code: the only caller (avx512 path in aos.rs) is
// `#[cfg(target_arch = "x86_64")]`, so on ARM / WASM / RISC-V the
// helper is unreferenced. Unit tests below run on all targets.
#[allow(dead_code)]
pub(crate) fn expand_with_fixed(k: usize, fixed: &[(u32, bool)]) -> usize {
    let mut result: usize = 0;
    let mut k_bit: u32 = 0;
    let mut fixed_iter = fixed.iter().peekable();
    let mut pos: u32 = 0;
    let k_bits_needed = usize::BITS - k.leading_zeros();
    while k_bit < k_bits_needed || fixed_iter.peek().is_some() {
        match fixed_iter.peek() {
            Some(&&(fpos, fval)) if fpos == pos => {
                if fval {
                    result |= 1usize << pos;
                }
                fixed_iter.next();
            }
            _ => {
                if (k >> k_bit) & 1 == 1 {
                    result |= 1usize << pos;
                }
                k_bit += 1;
            }
        }
        pos += 1;
    }
    result
}

/// Tolerance (squared magnitude) for the diagonal-2x2 detection
/// heuristic.  `EPS_SQ = 1e-30` ⇒ `|m_off| < ~3.16e-16`, just above
/// FP64 machine epsilon (~2.22e-16), so an off-diagonal entry the
/// caller produced as a "true" zero (e.g. `Phase::matrix()` literal
/// `0.0`) detects as diagonal while any caller-supplied off-diagonal
/// of magnitude ≥ machine eps falls through.
// Allow dead_code: callers in `aos.rs` and `soa.rs` land in P1-06
// Tasks 5 + 6.  Until then, the constant is unreferenced.
#[allow(dead_code)]
const DIAGONAL_EPS_SQ: f64 = 1e-30;

/// Returns true iff both off-diagonal entries of a 2×2 matrix have
/// squared magnitude below `DIAGONAL_EPS_SQ`.
///
/// Used as the dispatch heuristic for the 1q diagonal fast path
/// (P1-06).  The cost is 2 complex `norm_sqr` calls + 2 comparisons,
/// roughly 5 ns per call — negligible against any reasonable
/// state-vector kernel.
// Allow dead_code: callers land in P1-06 Tasks 5 + 6.
#[allow(dead_code)]
#[inline]
pub(crate) fn is_diagonal_2x2(m: &[[aleph_core::Complex; 2]; 2]) -> bool {
    m[0][1].norm_sqr() < DIAGONAL_EPS_SQ && m[1][0].norm_sqr() < DIAGONAL_EPS_SQ
}

#[cfg(test)]
mod tests {
    use super::control_mask;

    #[test]
    fn control_mask_empty_is_zero() {
        assert_eq!(control_mask(&[]), 0);
    }

    #[test]
    fn control_mask_combines_bits() {
        // Controls on qubits 0, 2, 5 → bit positions 0, 2, 5 → 0b100101 = 37.
        assert_eq!(control_mask(&[0, 2, 5]), 0b100101);
    }

    #[test]
    fn control_mask_is_order_independent() {
        assert_eq!(control_mask(&[5, 0, 2]), control_mask(&[0, 2, 5]));
    }

    #[test]
    fn expand_with_fixed_target_only_passthroughs_k() {
        // fixed = [(target=2, false)] → bit 2 cleared, other bits from k.
        // Free positions: 0, 1, 3, 4, ...  k = 0b011 → set positions 0 and 1.
        // Expected: 0b0011.
        assert_eq!(super::expand_with_fixed(0b011, &[(2, false)]), 0b0011);
    }

    #[test]
    fn expand_with_fixed_control_set_high() {
        // fixed sorted: (1, false), (3, true). Free positions: 0, 2, 4, ...
        // k = 0b010 → free bit at position 2; plus control bit at position 3.
        // Expected: bit 2 + bit 3 = 0b1100.
        assert_eq!(
            super::expand_with_fixed(0b010, &[(1, false), (3, true)]),
            0b1100
        );
    }

    #[test]
    fn expand_with_fixed_empty_fixed_is_identity() {
        assert_eq!(super::expand_with_fixed(0xDEAD, &[]), 0xDEAD);
    }

    use super::is_diagonal_2x2;
    use aleph_core::Complex;

    fn z(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    #[test]
    fn is_diagonal_2x2_pauli_z() {
        // diag(1, -1) — both off-diagonals exactly zero
        let m = [[z(1.0, 0.0), z(0.0, 0.0)], [z(0.0, 0.0), z(-1.0, 0.0)]];
        assert!(is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rz_random_theta() {
        // diag(e^{-iθ/2}, e^{+iθ/2}) for θ = 1.234
        let theta = 1.234_f64;
        let m = [
            [z((theta / 2.0).cos(), -(theta / 2.0).sin()), z(0.0, 0.0)],
            [z(0.0, 0.0), z((theta / 2.0).cos(), (theta / 2.0).sin())],
        ];
        assert!(is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rejects_hadamard() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let m = [[z(s, 0.0), z(s, 0.0)], [z(s, 0.0), z(-s, 0.0)]];
        assert!(!is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rejects_pauli_x() {
        let m = [[z(0.0, 0.0), z(1.0, 0.0)], [z(1.0, 0.0), z(0.0, 0.0)]];
        assert!(!is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_accepts_subepsilon_off_diagonal() {
        // |m_off| = 1e-17, well below FP64 eps — counts as zero
        let m = [[z(1.0, 0.0), z(1e-17, 0.0)], [z(0.0, 1e-17), z(-1.0, 0.0)]];
        assert!(is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rejects_superepsilon_off_diagonal() {
        // |m_off| = 1e-8, well above FP64 eps — counts as non-zero
        let m = [[z(1.0, 0.0), z(1e-8, 0.0)], [z(0.0, 0.0), z(-1.0, 0.0)]];
        assert!(!is_diagonal_2x2(&m));
    }

    #[test]
    fn expand_with_fixed_two_controls_around_target() {
        // fixed sorted: [(0, true), (2, false), (4, true)].
        // Free positions: 1, 3, 5, 6, 7, ...
        // k = 0b011 → free bits at positions 1 and 3.
        // Plus fixed: bit 0 set, bit 2 clear, bit 4 set.
        // Expected: 1 + 2 + 8 + 16 = 0b11011 = 27.
        assert_eq!(
            super::expand_with_fixed(0b011, &[(0, true), (2, false), (4, true)]),
            0b11011,
        );
    }
}
