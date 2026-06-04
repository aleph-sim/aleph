//! Diagonal-phase kernel: ψ[x] *= exp(i·phase(x)) in one streaming pass.
//!
//! Two implementations of the same map live here:
//!
//! * **Scalar** (`*_scalar_aos` / `*_scalar_soa`) — the reference, runs on
//!   every target, rayon-parallel per amplitude.
//! * **AVX-512** (`*_avx512_aos` / `*_avx512_soa`, x86_64-only) — processes
//!   8 consecutive amplitude indices per step. The phase accumulation is the
//!   hot part: each `PhaseTerm` becomes one `VPOPCNTQ`-based parity test
//!   (`avx512vpopcntdq`) feeding a masked `VADDPD`. `e^{iφ}` is computed by a
//!   scalar-extract `sin_cos` per lane (we are bandwidth-bound; the transcendental
//!   cost hides), then the per-lane (cos, sin) drive a packed complex multiply.
//!
//! The public `apply_diagonal_phase_aos` / `_soa` dispatchers pick the SIMD
//! path at runtime when both `avx512f` and `avx512vpopcntdq` are present, and
//! fall back to the scalar kernel otherwise. The two paths are bit-for-bit
//! equivalent modulo FP rounding (gated by an equivalence test, run for real
//! on an AVX-512 box).

use aleph_core::Complex;
use aleph_ir::DiagonalPhase;

/// Evaluate the real phase for amplitude index `x`. Hot inner helper —
/// kept tiny so it inlines into the per-amplitude loop.
#[inline(always)]
pub(crate) fn phase_at(dp: &DiagonalPhase, x: u64) -> f64 {
    let mut phi = 0.0;
    for t in &dp.terms {
        let mut all = true;
        for &m in &t.conds {
            if (m & x).count_ones() & 1 == 0 {
                all = false;
                break;
            }
        }
        if all {
            phi += t.angle;
        }
    }
    phi
}

/// Scalar, rayon-parallel application over SoA (split re/im) arrays.
pub(crate) fn apply_diagonal_phase_scalar_soa(re: &mut [f64], im: &mut [f64], dp: &DiagonalPhase) {
    use crate::kernels::tuning::DEFAULT_POLICY;
    use crate::kernels::{par_blocks, BlockPtr};
    let len = re.len();
    debug_assert_eq!(len, im.len());
    let rp = BlockPtr(re.as_mut_ptr());
    let ip = BlockPtr(im.as_mut_ptr());
    par_blocks(
        DEFAULT_POLICY,
        len,
        len,
        |k| k,
        move |k| {
            // SAFETY: each k in 0..len is a distinct index; par_blocks calls
            // body on disjoint indices, so writes never alias across rayon
            // tasks. rp/ip point into two separate buffers and never alias
            // each other. Both BlockPtrs are Send+Sync.
            let r = unsafe { &mut *rp.ptr().add(k) };
            let i = unsafe { &mut *ip.ptr().add(k) };
            let phi = phase_at(dp, k as u64);
            let (s, co) = phi.sin_cos();
            let nr = *r * co - *i * s;
            let ni = *r * s + *i * co;
            *r = nr;
            *i = ni;
        },
    );
}

/// Scalar, rayon-parallel application over an AoS amplitude slice.
pub(crate) fn apply_diagonal_phase_scalar_aos(amps: &mut [Complex], dp: &DiagonalPhase) {
    use crate::kernels::tuning::DEFAULT_POLICY;
    use crate::kernels::{par_blocks, ComplexPtr};
    let len = amps.len();
    let p = ComplexPtr(amps.as_mut_ptr());
    par_blocks(
        DEFAULT_POLICY,
        len,
        len,
        |k| k,
        move |k| {
            // SAFETY: each k in 0..len is a distinct index; par_blocks calls
            // body on disjoint indices, so these per-element writes never
            // alias across rayon tasks. `p` (ComplexPtr) is Send+Sync.
            let amp = unsafe { &mut *p.ptr().add(k) };
            let phi = phase_at(dp, k as u64);
            let (s, co) = phi.sin_cos();
            let re = amp.re * co - amp.im * s;
            let im = amp.re * s + amp.im * co;
            amp.re = re;
            amp.im = im;
        },
    );
}

// ---------------------------------------------------------------------------
// Runtime dispatchers — pick the AVX-512 path when the host supports it.
// ---------------------------------------------------------------------------

/// Apply the diagonal phase to an AoS amplitude slice, dispatching to the
/// AVX-512 kernel when the host supports `avx512f` + `avx512vpopcntdq`,
/// otherwise the scalar reference.
pub(crate) fn apply_diagonal_phase_aos(amps: &mut [Complex], dp: &DiagonalPhase) {
    #[cfg(target_arch = "x86_64")]
    {
        if amps.len() >= LANES
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vpopcntdq")
        {
            // SAFETY: both required features are detected immediately above,
            // and `amps.len() >= LANES` guarantees at least one full 8-block;
            // since `len` is a power of two ≥ 8 it is a multiple of LANES, so
            // the SIMD kernel processes the whole buffer with no scalar tail.
            unsafe { apply_diagonal_phase_avx512_aos(amps, dp) };
            return;
        }
    }
    apply_diagonal_phase_scalar_aos(amps, dp);
}

/// Apply the diagonal phase to SoA (split re/im) arrays, dispatching to the
/// AVX-512 kernel when the host supports `avx512f` + `avx512vpopcntdq`,
/// otherwise the scalar reference.
pub(crate) fn apply_diagonal_phase_soa(re: &mut [f64], im: &mut [f64], dp: &DiagonalPhase) {
    #[cfg(target_arch = "x86_64")]
    {
        if re.len() >= LANES
            && re.len() == im.len()
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vpopcntdq")
        {
            // SAFETY: features detected above; `re.len() == im.len()` and is a
            // power of two ≥ 8 (state-vector invariant), so it is a multiple of
            // LANES and the SIMD kernel has no scalar tail.
            unsafe { apply_diagonal_phase_avx512_soa(re, im, dp) };
            return;
        }
    }
    apply_diagonal_phase_scalar_soa(re, im, dp);
}

// ---------------------------------------------------------------------------
// AVX-512 kernels (x86_64 only).
// ---------------------------------------------------------------------------

/// SIMD-lane width in `f64`s: one `__m512d` / `__m512i` holds 8 amplitudes.
#[cfg(target_arch = "x86_64")]
const LANES: usize = 8;

/// Compute the 8 per-lane phases for the 8 amplitude indices starting at
/// `base` (lanes are `base + 0 .. base + 7`), returning them as a `__m512d`.
///
/// Mirrors `phase_at` exactly, but evaluated across 8 lanes at once: for each
/// `PhaseTerm`, a lane "fires" iff every `cond` mask has odd popcount against
/// that lane's index; firing lanes get `term.angle` added.
///
/// # Safety
///
/// Caller MUST ensure the host supports `avx512f` and `avx512vpopcntdq`
/// (`_mm512_popcnt_epi64` is VPOPCNTDQ).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub(crate) unsafe fn phases_at_block(
    dp: &DiagonalPhase,
    base: usize,
) -> std::arch::x86_64::__m512d {
    use std::arch::x86_64::*;

    // Lane k holds index (base + k). `_mm512_set_epi64` takes args
    // high-lane-first, so lane 0 gets the last argument (0).
    let lane_offsets = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    let idx = _mm512_add_epi64(_mm512_set1_epi64(base as i64), lane_offsets);

    let one = _mm512_set1_epi64(1);
    let mut phi = _mm512_setzero_pd();

    for t in &dp.terms {
        // Start all-true: a term with no conds (global phase) fires on every
        // lane, matching the scalar `all = true` initial state.
        let mut fire: __mmask8 = 0xFF;
        for &m in &t.conds {
            let anded = _mm512_and_epi64(idx, _mm512_set1_epi64(m as i64));
            let pc = _mm512_popcnt_epi64(anded);
            // `_mm512_test_epi64_mask(pc, one)` sets mask bit k iff
            // `(pc_k & 1) != 0`, i.e. iff the popcount is odd — exactly the
            // scalar `count_ones() & 1 == 1` parity test.
            let odd = _mm512_test_epi64_mask(pc, one);
            fire &= odd;
        }
        // Masked add: lanes where `fire` is set get `+ angle`; others keep phi.
        phi = _mm512_mask_add_pd(phi, fire, phi, _mm512_set1_pd(t.angle));
    }
    phi
}

/// Compute per-lane `(cos, sin)` of the 8 phases in `phi` via scalar-extract.
///
/// We are bandwidth-bound, so a vector polynomial `sin_cos` buys little; the
/// scalar libm `sin_cos` per lane is simplest and matches the scalar kernel's
/// rounding exactly (same `f64::sin_cos`).
///
/// # Safety
///
/// Caller MUST ensure the host supports `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn sincos_block(
    phi: std::arch::x86_64::__m512d,
) -> (std::arch::x86_64::__m512d, std::arch::x86_64::__m512d) {
    use std::arch::x86_64::*;
    let mut phi_arr = [0.0f64; LANES];
    _mm512_storeu_pd(phi_arr.as_mut_ptr(), phi);
    let mut cos_arr = [0.0f64; LANES];
    let mut sin_arr = [0.0f64; LANES];
    for j in 0..LANES {
        let (s, c) = phi_arr[j].sin_cos();
        sin_arr[j] = s;
        cos_arr[j] = c;
    }
    let cos = _mm512_loadu_pd(cos_arr.as_ptr());
    let sin = _mm512_loadu_pd(sin_arr.as_ptr());
    (cos, sin)
}

/// AVX-512 SoA diagonal-phase kernel. Processes 8 amplitudes per step over
/// the split `re` / `im` streams — the natural SIMD case (no de-interleave).
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host supports `avx512f` and `avx512vpopcntdq`.
/// * `re.len() == im.len()` and is a multiple of `LANES` (= 8). Guaranteed for
///   any state vector of `n ≥ 3` qubits; the dispatcher only calls this when
///   `len ≥ 8`, and `len` is a power of two.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn apply_diagonal_phase_avx512_soa(re: &mut [f64], im: &mut [f64], dp: &DiagonalPhase) {
    use crate::kernels::tuning::DEFAULT_POLICY;
    use crate::kernels::{par_blocks, BlockPtr};
    use std::arch::x86_64::*;

    let len = re.len();
    debug_assert_eq!(len, im.len());
    debug_assert_eq!(len % LANES, 0);

    let rp = BlockPtr(re.as_mut_ptr());
    let ip = BlockPtr(im.as_mut_ptr());
    let count = len / LANES;
    par_blocks(
        DEFAULT_POLICY,
        count,
        len,
        |k| k * LANES,
        move |base| {
            // SAFETY: par_blocks hands each task a distinct `base` that is a
            // multiple of LANES; the 8-lane load/store windows [base, base+8)
            // are therefore pairwise-disjoint and in-bounds (base + 8 ≤ len).
            // rp/ip point into two separate, non-aliasing buffers; both
            // BlockPtrs are Send+Sync. AVX-512F + VPOPCNTDQ are guaranteed by
            // this fn's #[target_feature] contract (dispatcher checked them).
            let r_ptr = rp.ptr().add(base);
            let i_ptr = ip.ptr().add(base);

            let phi = phases_at_block(dp, base);
            let (cos, sin) = sincos_block(phi);

            let re8 = _mm512_loadu_pd(r_ptr);
            let im8 = _mm512_loadu_pd(i_ptr);

            // new_re = re*cos - im*sin ; new_im = re*sin + im*cos
            let new_re = _mm512_fmsub_pd(re8, cos, _mm512_mul_pd(im8, sin));
            let new_im = _mm512_fmadd_pd(re8, sin, _mm512_mul_pd(im8, cos));

            _mm512_storeu_pd(r_ptr, new_re);
            _mm512_storeu_pd(i_ptr, new_im);
        },
    );
}

/// AVX-512 AoS diagonal-phase kernel. Processes 8 consecutive `Complex`
/// (16 interleaved `f64`) per step.
///
/// **Complex-multiply approach: full-SIMD de-interleave.** Two `__m512d`
/// loads cover the 8 `Complex` as `(re0,im0,re1,im1,...,re3,im3)` and
/// `(re4,im4,...,re7,im7)`. `_mm512_permutex2var_pd` with even/odd index
/// vectors gathers the 8 real parts into one `__m512d` and the 8 imaginary
/// parts into another (de-interleave). After the per-lane `cos/sin` multiply
/// we re-interleave with the inverse permute and store. This keeps the whole
/// multiply in registers (no scalar per-lane fallback).
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host supports `avx512f` and `avx512vpopcntdq`.
/// * `amps.len()` is a multiple of `LANES` (= 8) — guaranteed for `n ≥ 3`;
///   the dispatcher only calls this when `len ≥ 8` and `len` is a power of two.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn apply_diagonal_phase_avx512_aos(amps: &mut [Complex], dp: &DiagonalPhase) {
    use crate::kernels::tuning::DEFAULT_POLICY;
    use crate::kernels::{par_blocks, BlockPtr};
    use std::arch::x86_64::*;

    let len = amps.len();
    debug_assert_eq!(len % LANES, 0);

    // View the Complex slice as a flat `*mut f64` (re,im interleaved).
    let bp = BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Gather even f64 positions (the real parts) from a concat of two zmm:
    // permutex2var index lane k selects element `idx[k]` from {a (0..7),
    // b (8..15)}. Reals sit at flat offsets 0,2,4,...,14 → lanes
    // [0,2,4,6,8,10,12,14]; imags at 1,3,...,15.
    let even_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    let odd_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    // Re-interleave: produce (re0,im0,re1,im1,...) from the de-interleaved
    // real-vector `r` (lanes 0..7) and imag-vector `i` (treated as b, lanes
    // 8..15). Low half lanes are re0,im0,re1,im1,re2,im2,re3,im3 →
    // [0,8,1,9,2,10,3,11]; high half re4,im4,...,re7,im7 → [4,12,5,13,6,14,7,15].
    let lo_inter_idx = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
    let hi_inter_idx = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);

    let count = len / LANES;
    par_blocks(
        DEFAULT_POLICY,
        count,
        len,
        |k| k * LANES,
        move |base| {
            // SAFETY: par_blocks hands each task a distinct `base` (multiple of
            // LANES); the two 8-f64 windows [2*base, 2*base+8) and
            // [2*base+8, 2*base+16) cover the 8 Complex at amps[base..base+8],
            // pairwise-disjoint across tasks and in-bounds (base + 8 ≤ len ⇒
            // 2*base + 16 ≤ 2*len). BlockPtr is Send+Sync; AVX-512F + VPOPCNTDQ
            // guaranteed by this fn's #[target_feature] contract.
            let p = bp.ptr().add(base * 2);

            let phi = phases_at_block(dp, base);
            let (cos, sin) = sincos_block(phi);

            let lo = _mm512_loadu_pd(p); // re0,im0,...,re3,im3
            let hi = _mm512_loadu_pd(p.add(LANES)); // re4,im4,...,re7,im7

            // De-interleave into 8 reals and 8 imags.
            let re8 = _mm512_permutex2var_pd(lo, even_idx, hi);
            let im8 = _mm512_permutex2var_pd(lo, odd_idx, hi);

            // new_re = re*cos - im*sin ; new_im = re*sin + im*cos
            let new_re = _mm512_fmsub_pd(re8, cos, _mm512_mul_pd(im8, sin));
            let new_im = _mm512_fmadd_pd(re8, sin, _mm512_mul_pd(im8, cos));

            // Re-interleave: treat new_re as `a` (lanes 0..7) and new_im as `b`
            // (lanes 8..15).
            let out_lo = _mm512_permutex2var_pd(new_re, lo_inter_idx, new_im);
            let out_hi = _mm512_permutex2var_pd(new_re, hi_inter_idx, new_im);

            _mm512_storeu_pd(p, out_lo);
            _mm512_storeu_pd(p.add(LANES), out_hi);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_ir::PhaseTerm;
    use smallvec::smallvec;

    #[test]
    fn applies_controlled_phase_to_amplitudes() {
        // cp(π/2) on (0,1): multiply ψ[11] by i, others unchanged.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![0b01, 0b10],
                angle: std::f64::consts::FRAC_PI_2,
            }],
        };
        let mut amps = vec![Complex::new(1.0, 0.0); 4];
        apply_diagonal_phase_scalar_aos(&mut amps, &dp);
        for (x, amp) in amps.iter().enumerate().take(3) {
            assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12, "x={x}");
        }
        assert!((amps[3] - Complex::new(0.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn global_phase_applies_to_all() {
        // empty-conds term = global phase π → multiply everything by -1.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![],
                angle: std::f64::consts::PI,
            }],
        };
        let mut amps = vec![Complex::new(1.0, 0.0); 4];
        apply_diagonal_phase_scalar_aos(&mut amps, &dp);
        for (x, amp) in amps.iter().enumerate() {
            assert!((amp - Complex::new(-1.0, 0.0)).norm() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn applies_controlled_phase_soa() {
        use super::apply_diagonal_phase_scalar_soa;
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![0b01, 0b10],
                angle: std::f64::consts::FRAC_PI_2,
            }],
        };
        let mut re = vec![1.0f64; 4];
        let mut im = vec![0.0f64; 4];
        apply_diagonal_phase_scalar_soa(&mut re, &mut im, &dp);
        for x in 0..3 {
            assert!((re[x] - 1.0).abs() < 1e-12 && im[x].abs() < 1e-12, "x={x}");
        }
        // ψ[11] = e^{iπ/2} = i
        assert!(re[3].abs() < 1e-12 && (im[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn soa_matches_aos_on_random_terms() {
        // Cross-check the two scalar kernels agree on a multi-term diagonal.
        let dp = DiagonalPhase {
            n_qubits: 3,
            terms: vec![
                PhaseTerm {
                    conds: smallvec![0b001],
                    angle: 0.37,
                },
                PhaseTerm {
                    conds: smallvec![0b110],
                    angle: -1.2,
                },
                PhaseTerm {
                    conds: smallvec![0b010, 0b100],
                    angle: 2.1,
                },
                PhaseTerm {
                    conds: smallvec![],
                    angle: 0.5,
                }, // global
            ],
        };
        // seed an arbitrary non-uniform state
        let aos: Vec<Complex> = (0..8)
            .map(|k| Complex::new(0.1 * k as f64 + 1.0, 0.2 - 0.05 * k as f64))
            .collect();
        let mut aos_out = aos.clone();
        super::apply_diagonal_phase_scalar_aos(&mut aos_out, &dp);
        let mut re: Vec<f64> = aos.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = aos.iter().map(|c| c.im).collect();
        super::apply_diagonal_phase_scalar_soa(&mut re, &mut im, &dp);
        for k in 0..8 {
            assert!((re[k] - aos_out[k].re).abs() < 1e-12, "re k={k}");
            assert!((im[k] - aos_out[k].im).abs() < 1e-12, "im k={k}");
        }
    }

    /// Build a non-trivial DiagonalPhase on n=5 (len 32) exercising a global
    /// term, single-cond terms, and a multi-cond term, with angles chosen so
    /// per-lane phases are distinct and include values > 2π and negative
    /// (so a broken VPOPCNTQ parity, lane-offset order, or sincos shows up).
    fn equiv_fixture_dp() -> DiagonalPhase {
        DiagonalPhase {
            n_qubits: 5,
            terms: vec![
                PhaseTerm {
                    conds: smallvec![],
                    angle: 0.9,
                }, // global
                PhaseTerm {
                    conds: smallvec![0b00001],
                    angle: 7.3, // > 2π
                },
                PhaseTerm {
                    conds: smallvec![0b00010],
                    angle: -2.4, // negative
                },
                PhaseTerm {
                    conds: smallvec![0b10000],
                    angle: 1.05,
                },
                PhaseTerm {
                    conds: smallvec![0b00100, 0b01000],
                    angle: -3.9,
                },
                PhaseTerm {
                    conds: smallvec![0b00001, 0b10000],
                    angle: 5.5,
                },
            ],
        }
    }

    fn equiv_fixture_state() -> Vec<Complex> {
        // Non-uniform; distinct nonzero re/im per index so a swapped/zeroed
        // component is visible.
        (0..32)
            .map(|k| Complex::new(1.0 + 0.13 * k as f64, -0.7 + 0.09 * k as f64))
            .collect()
    }

    /// Scalar kernel vs the runtime dispatcher (SIMD on AVX-512 hosts, scalar
    /// elsewhere) must agree to 1e-13 — AoS. On aarch64 this is scalar-vs-scalar
    /// (smoke); on EPYC it gates the AVX-512 kernel against the reference.
    #[test]
    fn dispatcher_matches_scalar_aos() {
        let dp = equiv_fixture_dp();
        let state = equiv_fixture_state();

        let mut scalar = state.clone();
        apply_diagonal_phase_scalar_aos(&mut scalar, &dp);

        let mut dispatched = state.clone();
        apply_diagonal_phase_aos(&mut dispatched, &dp);

        for k in 0..state.len() {
            assert!(
                (scalar[k].re - dispatched[k].re).abs() < 1e-13,
                "re mismatch at k={k}: {} vs {}",
                scalar[k].re,
                dispatched[k].re
            );
            assert!(
                (scalar[k].im - dispatched[k].im).abs() < 1e-13,
                "im mismatch at k={k}: {} vs {}",
                scalar[k].im,
                dispatched[k].im
            );
        }
    }

    /// As `dispatcher_matches_scalar_aos`, for the SoA path.
    #[test]
    fn dispatcher_matches_scalar_soa() {
        let dp = equiv_fixture_dp();
        let state = equiv_fixture_state();

        let mut re_s: Vec<f64> = state.iter().map(|c| c.re).collect();
        let mut im_s: Vec<f64> = state.iter().map(|c| c.im).collect();
        apply_diagonal_phase_scalar_soa(&mut re_s, &mut im_s, &dp);

        let mut re_d: Vec<f64> = state.iter().map(|c| c.re).collect();
        let mut im_d: Vec<f64> = state.iter().map(|c| c.im).collect();
        apply_diagonal_phase_soa(&mut re_d, &mut im_d, &dp);

        for k in 0..state.len() {
            assert!(
                (re_s[k] - re_d[k]).abs() < 1e-13,
                "re mismatch at k={k}: {} vs {}",
                re_s[k],
                re_d[k]
            );
            assert!(
                (im_s[k] - im_d[k]).abs() < 1e-13,
                "im mismatch at k={k}: {} vs {}",
                im_s[k],
                im_d[k]
            );
        }
    }

    /// Cross-check the dispatcher's AoS and SoA outputs agree with each other
    /// on the same fixture (catches a layout-specific de-interleave bug that a
    /// single-layout scalar comparison would miss on an AVX-512 host).
    #[test]
    fn dispatcher_aos_matches_soa() {
        let dp = equiv_fixture_dp();
        let state = equiv_fixture_state();

        let mut aos = state.clone();
        apply_diagonal_phase_aos(&mut aos, &dp);

        let mut re: Vec<f64> = state.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = state.iter().map(|c| c.im).collect();
        apply_diagonal_phase_soa(&mut re, &mut im, &dp);

        for k in 0..state.len() {
            assert!((aos[k].re - re[k]).abs() < 1e-13, "re k={k}");
            assert!((aos[k].im - im[k]).abs() < 1e-13, "im k={k}");
        }
    }
}
