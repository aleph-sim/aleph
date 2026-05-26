//! AVX-512F `apply_1q` kernel — 8-lane f64 SIMD.
//!
//! Structurally identical to the AVX2 path; differs only in lane
//! count (`__m512d` vs `__m256d`) and intrinsic prefix. When
//! `target_bit < 8` the AVX-512 path delegates to `avx2::apply_1q`
//! (which itself handles `target_bit < 4` via scalar tail). This
//! cascade keeps the AVX-512 main loop free of masked-store code; the
//! AVX2 implementation is already correct and well-tested.

use aleph_core::Complex;
use core::arch::x86_64::*;

const LANES: usize = 8;

/// # Safety
///
/// Caller MUST ensure the host CPU supports AVX-512F. The
/// `kernels::soa::apply_1q` dispatcher checks this before invoking.
/// Other safety invariants: see `super::avx2::apply_1q` doc-block
/// (`re.len() == im.len()`, power-of-two, `target` and `controls`
/// in qubit range, controls distinct from target).
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    let target_bit = 1usize << target;

    // Sub-512-bit target: hand off to AVX2 (which handles down to
    // target_bit = 1 via its scalar tail). Faster than `__mmask8`
    // masked stores for small targets, and reuses tested code.
    if target_bit < LANES {
        // SAFETY: AVX-512F implies AVX2 + FMA on every shipping CPU
        // (Skylake-X 2017 onwards, every AMD EPYC since Zen 4 2022).
        // The runtime dispatcher confirmed avx512f availability
        // before reaching this kernel.
        super::avx2::apply_1q(re, im, target, controls, m);
        return;
    }

    let len = re.len();

    let m00r = _mm512_set1_pd(m[0][0].re);
    let m00i = _mm512_set1_pd(m[0][0].im);
    let m01r = _mm512_set1_pd(m[0][1].re);
    let m01i = _mm512_set1_pd(m[0][1].im);
    let m10r = _mm512_set1_pd(m[1][0].re);
    let m10i = _mm512_set1_pd(m[1][0].im);
    let m11r = _mm512_set1_pd(m[1][1].re);
    let m11i = _mm512_set1_pd(m[1][1].im);

    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let mut block = 0usize;
        while block < len {
            apply_block_8(
                re, im, block, target_bit, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i,
            );
            block += outer_step;
        }
        return;
    }

    // Same shape as AVX2: fall back to scalar when any control is
    // at-or-below target (the SIMD inner walk only handles
    // `controls > target`; see avx2::apply_1q for the full rationale).
    if controls.iter().any(|&c| c <= target) {
        super::scalar::apply_1q(re, im, target, controls, m);
        return;
    }

    // Renormalise control positions so the outer loop only places
    // bits above `target + 1`, leaving the low bits zero for the
    // contiguous SIMD inner walk. See avx2::apply_1q for the full
    // derivation.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    for k in 0..outer_count {
        let block = crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1);
        apply_block_8(
            re, im, block, target_bit, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i,
        );
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn apply_block_8(
    re: &mut [f64],
    im: &mut [f64],
    block: usize,
    target_bit: usize,
    m00r: __m512d,
    m00i: __m512d,
    m01r: __m512d,
    m01i: __m512d,
    m10r: __m512d,
    m10i: __m512d,
    m11r: __m512d,
    m11i: __m512d,
) {
    // Invariant: caller routes target_bit < LANES to AVX2, so here
    // target_bit ≥ LANES and LANES (= 8) divides target_bit (both
    // powers of two). No tail needed.
    debug_assert!(target_bit >= LANES);
    debug_assert!(target_bit.is_power_of_two());

    let mut j = 0usize;
    while j + LANES <= target_bit {
        let i0 = block | j;
        let i1 = i0 + target_bit;

        // SAFETY: i0 + LANES ≤ block + target_bit ≤ block + 2*target_bit ≤ len
        //         (outer block stride is 2*target_bit, stops at len).
        let re0 = _mm512_loadu_pd(re.as_ptr().add(i0));
        let im0 = _mm512_loadu_pd(im.as_ptr().add(i0));
        let re1 = _mm512_loadu_pd(re.as_ptr().add(i1));
        let im1 = _mm512_loadu_pd(im.as_ptr().add(i1));

        let new_re0 = {
            let t1 = _mm512_fmsub_pd(m00r, re0, _mm512_mul_pd(m00i, im0));
            let t2 = _mm512_fmsub_pd(m01r, re1, _mm512_mul_pd(m01i, im1));
            _mm512_add_pd(t1, t2)
        };
        let new_im0 = {
            let t1 = _mm512_fmadd_pd(m00r, im0, _mm512_mul_pd(m00i, re0));
            let t2 = _mm512_fmadd_pd(m01r, im1, _mm512_mul_pd(m01i, re1));
            _mm512_add_pd(t1, t2)
        };
        let new_re1 = {
            let t1 = _mm512_fmsub_pd(m10r, re0, _mm512_mul_pd(m10i, im0));
            let t2 = _mm512_fmsub_pd(m11r, re1, _mm512_mul_pd(m11i, im1));
            _mm512_add_pd(t1, t2)
        };
        let new_im1 = {
            let t1 = _mm512_fmadd_pd(m10r, im0, _mm512_mul_pd(m10i, re0));
            let t2 = _mm512_fmadd_pd(m11r, im1, _mm512_mul_pd(m11i, re1));
            _mm512_add_pd(t1, t2)
        };

        _mm512_storeu_pd(re.as_mut_ptr().add(i0), new_re0);
        _mm512_storeu_pd(im.as_mut_ptr().add(i0), new_im0);
        _mm512_storeu_pd(re.as_mut_ptr().add(i1), new_re1);
        _mm512_storeu_pd(im.as_mut_ptr().add(i1), new_im1);

        j += LANES;
    }
    debug_assert_eq!(j, target_bit);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use aleph_core::GateMatrix;
    use aleph_test::gate::arb_1q_gate;
    use aleph_test::state::arb_state_vector;
    use proptest::prelude::*;

    fn host_has_avx512f() -> bool {
        std::is_x86_feature_detected!("avx512f")
    }

    fn run_avx512(
        re: &mut [f64],
        im: &mut [f64],
        target: u32,
        controls: &[u32],
        m: &[[Complex; 2]; 2],
    ) -> bool {
        if !host_has_avx512f() {
            return false;
        }
        // SAFETY: feature gate above.
        unsafe { super::apply_1q(re, im, target, controls, m) };
        true
    }

    fn hadamard() -> [[Complex; 2]; 2] {
        let s = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        [[s, s], [s, -s]]
    }

    fn pauli_x() -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[z, o], [o, z]]
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn avx512_matches_scalar(
            gate in arb_1q_gate(),
            // Target up to 6 exercises both sub-LANES delegation
            // (target ∈ {0,1,2}) and the main 8-lane loop (target ≥ 3).
            target in 0u32..7u32,
            ctrl_count in 0u32..=2u32,
            ctrl_seeds in proptest::collection::vec(0u32..7u32, 0..=2),
            amps in arb_state_vector(7),
        ) {
            if !host_has_avx512f() {
                return Ok(());
            }
            let mut controls: Vec<u32> = ctrl_seeds.into_iter().take(ctrl_count as usize).collect();
            controls.retain(|c| *c != target);
            controls.sort_unstable();
            controls.dedup();

            let m = match gate.matrix().unwrap() {
                GateMatrix::M2x2(m) => m,
                _ => unreachable!("arb_1q_gate yields 1q gates"),
            };
            let re_src: Vec<f64> = amps.iter().map(|c| c.re).collect();
            let im_src: Vec<f64> = amps.iter().map(|c| c.im).collect();
            let mut re_ref = re_src.clone();
            let mut im_ref = im_src.clone();
            super::super::scalar::apply_1q(&mut re_ref, &mut im_ref, target, &controls, &m);
            let mut re_simd = re_src.clone();
            let mut im_simd = im_src.clone();
            // SAFETY: feature gate.
            unsafe { super::apply_1q(&mut re_simd, &mut im_simd, target, &controls, &m); }
            for k in 0..re_ref.len() {
                prop_assert!(
                    (re_ref[k] - re_simd[k]).abs() < 1e-12
                        && (im_ref[k] - im_simd[k]).abs() < 1e-12,
                    "k={k} target={target} controls={:?}: scalar=({}, {}) avx512=({}, {})",
                    controls, re_ref[k], im_ref[k], re_simd[k], im_simd[k]
                );
            }
        }
    }

    /// target=3, target_bit=8 == LANES → exact fit on AVX-512, zero tail.
    #[test]
    fn avx512_target_three_exact_fit() {
        let mut re = vec![0.0_f64; 16];
        let mut im = vec![0.0_f64; 16];
        re[0] = 1.0;
        if !run_avx512(&mut re, &mut im, 3, &[], &hadamard()) {
            return;
        }
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((re[0] - s).abs() < 1e-12);
        assert!((re[8] - s).abs() < 1e-12);
    }

    /// target=0, target_bit=1 < LANES → AVX-512 delegates to AVX2,
    /// which delegates to scalar tail. End-to-end must still match.
    #[test]
    fn avx512_target_zero_delegates_to_avx2() {
        let mut re = vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut im = vec![0.0_f64; 8];
        if !run_avx512(&mut re, &mut im, 0, &[], &pauli_x()) {
            return;
        }
        assert_eq!(re, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// Control above target on a target large enough that AVX-512's
    /// main loop engages (target=3, target_bit=8 == LANES).
    #[test]
    fn avx512_control_above_target_simd_path() {
        let mut re = vec![0.0_f64; 32]; // 5 qubits
        let mut im = vec![0.0_f64; 32];
        re[24] = 1.0; // 24 = 0b11000 → q3=1, q4=1
        if !run_avx512(&mut re, &mut im, 3, &[4], &pauli_x()) {
            return;
        }
        // X on q3 with q4-control set: q3 flips 1→0. 24 - 8 (bit 3) = 16.
        assert!(re[24].abs() < 1e-12);
        assert!((re[16] - 1.0).abs() < 1e-12);
    }
}
