//! AVX2 + FMA `apply_1q` kernel — 4-lane f64 SIMD.
//!
//! Task 4 implements the uncontrolled fast path (`controls.is_empty()`);
//! controlled path lands in Task 5 and currently falls through to the
//! scalar fallback.

use aleph_core::Complex;
use core::arch::x86_64::*;

const LANES: usize = 4;

/// AVX2 + FMA `apply_1q` — 4-lane f64 SIMD.
///
/// # Safety
///
/// Caller MUST ensure the host CPU supports AVX2 + FMA. The
/// `kernels::soa::apply_1q` dispatcher checks this before invoking;
/// tests call `is_x86_feature_detected!` inline. Additional
/// invariants enforced by the caller (`SoaSvBackend::apply_gate`):
///
/// * `re.len() == im.len()` and is a power of two.
/// * `target < re.len().trailing_zeros()` (qubit index in range).
/// * `controls` contains no duplicates, none equal to `target`, all < n_qubits.
///
/// Inner-block reads at offset `j` cover `re[block + j .. block + j + LANES]`
/// and `re[block + target_bit + j .. block + target_bit + j + LANES]`; both
/// spans lie within `[block, block + 2 * target_bit)`, which is `≤ re.len()`
/// because the outer block stride is `2 * target_bit` and stops at `len`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    let target_bit = 1usize << target;
    let len = re.len();

    // Broadcast matrix entries — constant across all iterations.
    let m00r = _mm256_set1_pd(m[0][0].re);
    let m00i = _mm256_set1_pd(m[0][0].im);
    let m01r = _mm256_set1_pd(m[0][1].re);
    let m01i = _mm256_set1_pd(m[0][1].im);
    let m10r = _mm256_set1_pd(m[1][0].re);
    let m10i = _mm256_set1_pd(m[1][0].im);
    let m11r = _mm256_set1_pd(m[1][1].re);
    let m11i = _mm256_set1_pd(m[1][1].im);

    if controls.is_empty() {
        // Nested block / pair iteration. Outer steps by 2*target_bit;
        // inner sweeps `j ∈ [0, target_bit)` in LANES-sized chunks.
        let outer_step = target_bit << 1;
        let mut block = 0usize;
        while block < len {
            apply_block_4(
                re, im, block, target_bit, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i, m,
            );
            block += outer_step;
        }
        return;
    }

    // If any control is at-or-below target, the inner SIMD walk would
    // either toggle a control bit (c < target) or collide with target
    // (c == target — also blocked by backend duplicate-qubit check).
    // Fall back to scalar. The QFT controlled-Phase shape always has
    // control > target, so this fall-through doesn't fire on the hot
    // path. See spec §7.
    if controls.iter().any(|&c| c <= target) {
        super::scalar::apply_1q(re, im, target, controls, m);
        return;
    }

    // SIMD controlled path: all controls > target (guarded above).
    //
    // The inner SIMD walk reads / writes `LANES` *contiguous* doubles
    // starting at `block | j` and `block | j + target_bit`. For the
    // loads / stores to address consecutive amplitudes, `block` must
    // have its low `target + 1` bits all zero — otherwise `block | j`
    // for j ∈ [0, LANES) is not equal to `block + j` (the bit-OR
    // collapses to the same index for j-bits that are already set in
    // `block`), and we'd both skip valid amplitudes and run off the
    // end of the buffer.
    //
    // To guarantee that, the outer loop iterates only over bits
    // *above* target. Renormalise control positions by subtracting
    // `target + 1` so they start at 0 in the "above" coordinate
    // system, then `expand_with_fixed` lays out free + control bits
    // densely; left-shift the result by `target + 1` to put them
    // back at their actual qubit positions.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    // Each control is > target and < n_qubits, distinct from target,
    // so `target + 1 + controls.len() ≤ n_qubits` — the subtraction
    // never underflows.
    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    for k in 0..outer_count {
        let block = crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1);
        apply_block_4(
            re, im, block, target_bit, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i, m,
        );
    }
}

#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn apply_block_4(
    re: &mut [f64],
    im: &mut [f64],
    block: usize,
    target_bit: usize,
    m00r: __m256d,
    m00i: __m256d,
    m01r: __m256d,
    m01i: __m256d,
    m10r: __m256d,
    m10i: __m256d,
    m11r: __m256d,
    m11i: __m256d,
    m: &[[Complex; 2]; 2],
) {
    let mut j = 0usize;
    while j + LANES <= target_bit {
        let i0 = block | j;
        let i1 = i0 + target_bit;

        // SAFETY: i0 + LANES ≤ block + target_bit ≤ block + 2*target_bit ≤ len.
        let re0 = _mm256_loadu_pd(re.as_ptr().add(i0));
        let im0 = _mm256_loadu_pd(im.as_ptr().add(i0));
        let re1 = _mm256_loadu_pd(re.as_ptr().add(i1));
        let im1 = _mm256_loadu_pd(im.as_ptr().add(i1));

        // new_re0 = m00r*re0 - m00i*im0 + m01r*re1 - m01i*im1
        let new_re0 = {
            let t1 = _mm256_fmsub_pd(m00r, re0, _mm256_mul_pd(m00i, im0));
            let t2 = _mm256_fmsub_pd(m01r, re1, _mm256_mul_pd(m01i, im1));
            _mm256_add_pd(t1, t2)
        };
        // new_im0 = m00r*im0 + m00i*re0 + m01r*im1 + m01i*re1
        let new_im0 = {
            let t1 = _mm256_fmadd_pd(m00r, im0, _mm256_mul_pd(m00i, re0));
            let t2 = _mm256_fmadd_pd(m01r, im1, _mm256_mul_pd(m01i, re1));
            _mm256_add_pd(t1, t2)
        };
        // new_re1 = m10r*re0 - m10i*im0 + m11r*re1 - m11i*im1
        let new_re1 = {
            let t1 = _mm256_fmsub_pd(m10r, re0, _mm256_mul_pd(m10i, im0));
            let t2 = _mm256_fmsub_pd(m11r, re1, _mm256_mul_pd(m11i, im1));
            _mm256_add_pd(t1, t2)
        };
        // new_im1 = m10r*im0 + m10i*re0 + m11r*im1 + m11i*re1
        let new_im1 = {
            let t1 = _mm256_fmadd_pd(m10r, im0, _mm256_mul_pd(m10i, re0));
            let t2 = _mm256_fmadd_pd(m11r, im1, _mm256_mul_pd(m11i, re1));
            _mm256_add_pd(t1, t2)
        };

        _mm256_storeu_pd(re.as_mut_ptr().add(i0), new_re0);
        _mm256_storeu_pd(im.as_mut_ptr().add(i0), new_im0);
        _mm256_storeu_pd(re.as_mut_ptr().add(i1), new_re1);
        _mm256_storeu_pd(im.as_mut_ptr().add(i1), new_im1);

        j += LANES;
    }
    // Tail: leftover pairs for target_bit ∈ {1, 2, 3}.
    while j < target_bit {
        let i0 = block | j;
        let i1 = i0 + target_bit;
        let a0_re = re[i0];
        let a0_im = im[i0];
        let a1_re = re[i1];
        let a1_im = im[i1];
        re[i0] = m[0][0].re * a0_re - m[0][0].im * a0_im + m[0][1].re * a1_re - m[0][1].im * a1_im;
        im[i0] = m[0][0].re * a0_im + m[0][0].im * a0_re + m[0][1].re * a1_im + m[0][1].im * a1_re;
        re[i1] = m[1][0].re * a0_re - m[1][0].im * a0_im + m[1][1].re * a1_re - m[1][1].im * a1_im;
        im[i1] = m[1][0].re * a0_im + m[1][0].im * a0_re + m[1][1].re * a1_im + m[1][1].im * a1_re;
        j += 1;
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use aleph_core::GateMatrix;
    use aleph_test::gate::arb_1q_gate;
    use aleph_test::state::arb_state_vector;
    use proptest::prelude::*;

    /// Host capability check — proptests below skip silently if AVX2 + FMA
    /// are unavailable. CI matrix on the EPYC runner exercises the path
    /// for real; M-series dev hosts skip (this module is `cfg`-gated off
    /// on ARM, so this check only fires on x86 hosts without AVX2/FMA).
    fn host_has_avx2_fma() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// AVX2 controlled path matches scalar reference. Covers both
        /// orientations: control above target (SIMD path engages) and
        /// control below target (scalar fall-through engages — same
        /// kernel under test, the guard delegates correctly).
        #[test]
        fn avx2_matches_scalar(
            gate in arb_1q_gate(),
            target in 0u32..6u32,
            ctrl_count in 0u32..=2u32,
            // Controls deliberately span both sides of target.
            ctrl_seeds in proptest::collection::vec(0u32..6u32, 0..=2),
            amps in arb_state_vector(6),
        ) {
            if !host_has_avx2_fma() {
                return Ok(());
            }
            // Build a duplicate-free control list distinct from target.
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

            // Scalar reference
            let mut re_ref = re_src.clone();
            let mut im_ref = im_src.clone();
            super::super::scalar::apply_1q(&mut re_ref, &mut im_ref, target, &controls, &m);

            // AVX2 candidate — direct call.
            let mut re_simd = re_src.clone();
            let mut im_simd = im_src.clone();
            // SAFETY: host_has_avx2_fma() guard above.
            unsafe { super::apply_1q(&mut re_simd, &mut im_simd, target, &controls, &m); }

            for k in 0..re_ref.len() {
                prop_assert!(
                    (re_ref[k] - re_simd[k]).abs() < 1e-12
                        && (im_ref[k] - im_simd[k]).abs() < 1e-12,
                    "k={k} target={target} controls={:?}: scalar=({}, {}) avx2=({}, {})",
                    controls, re_ref[k], im_ref[k], re_simd[k], im_simd[k]
                );
            }
        }
    }

    /// Helper: AVX2 direct call wrapped in feature-detection skip.
    /// Returns true if the call actually executed; false if skipped.
    fn run_avx2(
        re: &mut [f64],
        im: &mut [f64],
        target: u32,
        controls: &[u32],
        m: &[[Complex; 2]; 2],
    ) -> bool {
        if !host_has_avx2_fma() {
            return false;
        }
        // SAFETY: host_has_avx2_fma() above.
        unsafe { super::apply_1q(re, im, target, controls, m) };
        true
    }

    fn pauli_x() -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[z, o], [o, z]]
    }

    fn hadamard() -> [[Complex; 2]; 2] {
        let s = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        [[s, s], [s, -s]]
    }

    fn pauli_y() -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let pi = Complex::new(0.0, 1.0);
        let mi = Complex::new(0.0, -1.0);
        [[z, mi], [pi, z]]
    }

    /// target=0, target_bit=1 → all tail. AVX2 inner loop runs zero
    /// iterations; the scalar tail must produce the same result as
    /// straight scalar apply.
    #[test]
    fn avx2_target_zero_all_tail() {
        let mut re = vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut im = vec![0.0_f64; 8];
        if !run_avx2(&mut re, &mut im, 0, &[], &pauli_x()) {
            return;
        }
        assert_eq!(re, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(im.iter().all(|x| x.abs() < 1e-12));
    }

    /// target=1, target_bit=2 < LANES → partial fit; SIMD loop runs zero
    /// inner iters (`j + 4 ≤ 2` is false), all work falls to tail.
    #[test]
    fn avx2_target_one_partial_tail() {
        let mut re = vec![0.0_f64, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut im = vec![0.0_f64; 8];
        if !run_avx2(&mut re, &mut im, 1, &[], &pauli_y()) {
            return;
        }
        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩. State amps[1]=1 means q0=1,q1=0;
        // Y on q1 maps |q1=0⟩ → i|q1=1⟩, so amps[1] → i·amps[3].
        assert!(re[1].abs() < 1e-12);
        assert!((im[3] - 1.0).abs() < 1e-12);
    }

    /// target=2, target_bit=4 == LANES → exact fit, zero tail.
    #[test]
    fn avx2_target_two_exact_fit() {
        let mut re = vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut im = vec![0.0_f64; 8];
        if !run_avx2(&mut re, &mut im, 2, &[], &hadamard()) {
            return;
        }
        let s = std::f64::consts::FRAC_1_SQRT_2;
        // H on q2 mixes amps[0] ↔ amps[4]: both go to s.
        assert!((re[0] - s).abs() < 1e-12);
        assert!((re[4] - s).abs() < 1e-12);
        for k in [1, 2, 3, 5, 6, 7] {
            assert!(re[k].abs() < 1e-12, "k={k} re={}", re[k]);
        }
    }

    /// Control above target — SIMD controlled path engages. n=3,
    /// target=0, control=2 → CX(c=q2, t=q0). amps[5] (q0=1, q1=0, q2=1)
    /// flips q0 → moves mass to amps[4].
    #[test]
    fn avx2_control_above_target_simd_path() {
        let mut re = vec![0.0_f64; 8];
        let mut im = vec![0.0_f64; 8];
        re[5] = 1.0;
        if !run_avx2(&mut re, &mut im, 0, &[2], &pauli_x()) {
            return;
        }
        assert!(re[5].abs() < 1e-12);
        assert!((re[4] - 1.0).abs() < 1e-12);
    }

    /// Control below target — scalar fall-through engages. n=3,
    /// target=2, control=0 → CX(c=q0, t=q2). amps[1] (q0=1) flips q2 →
    /// moves to amps[5].
    #[test]
    fn avx2_control_below_target_scalar_fallthrough() {
        let mut re = vec![0.0_f64; 8];
        let mut im = vec![0.0_f64; 8];
        re[1] = 1.0;
        if !run_avx2(&mut re, &mut im, 2, &[0], &pauli_x()) {
            return;
        }
        assert!(re[1].abs() < 1e-12);
        assert!((re[5] - 1.0).abs() < 1e-12);
    }
}
