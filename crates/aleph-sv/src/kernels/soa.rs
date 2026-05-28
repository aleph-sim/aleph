//! SoA gate application kernels — paired `Vec<f64>` storage.
//!
//! Convention identical to the AoS path (ADR 0004 / P0-06 spec §6):
//! `target` / `targets[0]` is the MSB of the matrix-index group;
//! `controls` are external (no row in the matrix). Real-arithmetic
//! expansion of `m * (re + i·im)` runs entirely on f64 pairs — the
//! compiler can vectorise the inner loop, and P1-03 will land an
//! explicit AVX2 specialisation on top.

use aleph_core::Complex;

/// SIMD-lane width (in **doubles**) for the SoA AVX-512 2q kernels.
/// One `_mm512_loadu_pd` reads 8 contiguous `f64`s from either the `re`
/// or `im` stream — i.e. 8 amplitudes' worth of one component.  Tier A
/// of the SoA SIMD kernels therefore requires
/// `1 << min(targets) >= LANES_SOA`.
///
/// Mirrors `aos.rs`'s in-function `LANES = 4` (which counts complex
/// pairs per zmm, since AoS interleaves re/im in a `Vec<Complex>`).  The
/// numeric difference reflects the layout: 8 lanes of one component
/// (SoA) vs 4 lanes of paired components (AoS) — both fill one zmm.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // referenced only by the avx512 paths below
const LANES_SOA: usize = 8;

/// Apply a 3-qubit matrix to `targets = [t0, t1, t2]` (with external
/// `controls`) in place over paired SoA storage. MSB convention:
/// `targets[0]` is bit 2 of the matrix index, `targets[1]` is bit 1,
/// `targets[2]` is bit 0 (matches `aos::apply_3q`).
pub(crate) fn apply_3q(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 3],
    controls: &[u32],
    m: &[[Complex; 8]; 8],
) {
    debug_assert_eq!(re.len(), im.len());
    let t_bits = [
        1usize << targets[0],
        1usize << targets[1],
        1usize << targets[2],
    ];
    let t_mask = t_bits[0] | t_bits[1] | t_bits[2];
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            let mut idx = [0usize; 8];
            for (k, slot) in idx.iter_mut().enumerate() {
                let bit_t0 = if k & 4 != 0 { t_bits[0] } else { 0 };
                let bit_t1 = if k & 2 != 0 { t_bits[1] } else { 0 };
                let bit_t2 = if k & 1 != 0 { t_bits[2] } else { 0 };
                *slot = i | bit_t0 | bit_t1 | bit_t2;
            }
            let v_re = [
                re[idx[0]], re[idx[1]], re[idx[2]], re[idx[3]], re[idx[4]], re[idx[5]], re[idx[6]],
                re[idx[7]],
            ];
            let v_im = [
                im[idx[0]], im[idx[1]], im[idx[2]], im[idx[3]], im[idx[4]], im[idx[5]], im[idx[6]],
                im[idx[7]],
            ];
            for r in 0..8 {
                let mut acc_re = 0.0_f64;
                let mut acc_im = 0.0_f64;
                for c in 0..8 {
                    acc_re += m[r][c].re * v_re[c] - m[r][c].im * v_im[c];
                    acc_im += m[r][c].re * v_im[c] + m[r][c].im * v_re[c];
                }
                re[idx[r]] = acc_re;
                im[idx[r]] = acc_im;
            }
        }
        i += 1;
    }
}

/// Scalar fallback for 2-qubit gate application over paired SoA storage.
///
/// Handles the cases where the AVX-512 SoA path's safety contract is
/// not satisfied: `1 << min(targets) < LANES`, non-AVX-512 host, or
/// external controls below `max(targets)`. Also the only entry-point
/// on non-x86_64 targets.
///
/// **MSB convention (P0-06):** `targets[0]` is the *high* bit of the
/// matrix index `k`, `targets[1]` is the *low* bit (matches
/// `aos::apply_2q_dense_scalar`).
///
/// Targets must be distinct; the caller (`apply_gate`) enforces this.
pub(crate) fn apply_2q_dense_scalar(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex; 4]; 4],
) {
    debug_assert_eq!(re.len(), im.len());
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let t_mask = t0_bit | t1_bit;
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            let idx = [
                i,          // k = 00
                i | t1_bit, // k = 01
                i | t0_bit, // k = 10
                i | t_mask, // k = 11
            ];
            let v_re = [re[idx[0]], re[idx[1]], re[idx[2]], re[idx[3]]];
            let v_im = [im[idx[0]], im[idx[1]], im[idx[2]], im[idx[3]]];
            for r in 0..4 {
                let mut acc_re = 0.0_f64;
                let mut acc_im = 0.0_f64;
                for c in 0..4 {
                    acc_re += m[r][c].re * v_re[c] - m[r][c].im * v_im[c];
                    acc_im += m[r][c].re * v_im[c] + m[r][c].im * v_re[c];
                }
                re[idx[r]] = acc_re;
                im[idx[r]] = acc_im;
            }
        }
        i += 1;
    }
}

/// Scalar CNOT specialisation over paired SoA storage.  For amplitudes
/// where bit `control` = 1 AND every external control bit is set, swap
/// `(re[i], im[i])` with `(re[i | t_bit], im[i | t_bit])`.  Zero
/// multiplies; pure swap-pair traffic on both streams.
///
/// `control` and `target` are passed separately (vs the generic 2q
/// kernel's `targets[2]`) because the dispatch prelude has already
/// disambiguated the orientation via `Perm2qKind`.  External
/// `controls` are appended to the implicit control mask.  Mirror of
/// `aos::apply_2q_cnot_scalar` with the `re` / `im` split.
pub(crate) fn apply_2q_cnot_scalar(
    re: &mut [f64],
    im: &mut [f64],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    debug_assert_eq!(re.len(), im.len());
    let c_bit = 1usize << control;
    let t_bit = 1usize << target;
    let ctrl_mask = c_bit | super::control_mask(external_controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask && (i & t_bit) == 0 {
            re.swap(i, i | t_bit);
            im.swap(i, i | t_bit);
        }
        i += 1;
    }
}

/// Scalar SWAP specialisation over paired SoA storage.  Walks every
/// base index `i` with both target bits zero (and external controls
/// set); for each such `i`, swap `(re[i | a_bit], im[i | a_bit])`
/// (a=0, b=1) with `(re[i | b_bit], im[i | b_bit])` (a=1, b=0).
/// Mirror of `aos::apply_2q_swap_scalar` with the `re` / `im` split.
pub(crate) fn apply_2q_swap_scalar(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    debug_assert_eq!(re.len(), im.len());
    let a_bit = 1usize << targets[0];
    let b_bit = 1usize << targets[1];
    let t_mask = a_bit | b_bit;
    let ctrl_mask = super::control_mask(external_controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            re.swap(i | a_bit, i | b_bit);
            im.swap(i | a_bit, i | b_bit);
        }
        i += 1;
    }
}

/// Scalar CZ specialisation over paired SoA storage.  Negate
/// `(re[i], im[i])` for amplitudes where both target bits are 1 (and
/// external controls satisfied).  Touches 1/4 of the state vector;
/// no multiplies — single sign-flip per stream.  Mirror of
/// `aos::apply_2q_cz_scalar` with the `re` / `im` split.
pub(crate) fn apply_2q_cz_scalar(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    debug_assert_eq!(re.len(), im.len());
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let t_mask = t0_bit | t1_bit;
    let ctrl_mask = super::control_mask(external_controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == t_mask && (i & ctrl_mask) == ctrl_mask {
            re[i] = -re[i];
            im[i] = -im[i];
        }
        i += 1;
    }
}

/// Scalar 2q-diagonal specialisation over paired SoA storage.  For
/// each amplitude `(re[i], im[i])`, multiply by `d[k]` where
/// `k = ((i >> targets[0]) & 1) << 1 | ((i >> targets[1]) & 1)`.
///
/// MSB convention matches `aos::apply_2q_diagonal_scalar`:
/// `targets[0]` is the high bit of `k`, `targets[1]` is the low bit
/// (per ADR 0004 / P0-06 §6).  Each amp is a 2-stream complex
/// multiply by `d[k]`.
pub(crate) fn apply_2q_diagonal_scalar(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
    d: [Complex; 4],
) {
    debug_assert_eq!(re.len(), im.len());
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let ctrl_mask = super::control_mask(external_controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask {
            let k_hi = ((i & t0_bit) != 0) as usize;
            let k_lo = ((i & t1_bit) != 0) as usize;
            let k = (k_hi << 1) | k_lo;
            let d_re = d[k].re;
            let d_im = d[k].im;
            let r = re[i];
            let im_v = im[i];
            re[i] = r * d_re - im_v * d_im;
            im[i] = r * d_im + im_v * d_re;
        }
        i += 1;
    }
}

/// Packed AVX-512 CNOT specialisation over paired SoA storage — Tier A
/// (`1 << target >= LANES_SOA = 8` AND `control > target`).
///
/// Pure swap-pair traffic, mirror of `aos::apply_2q_cnot_avx512` (Task 6)
/// but on paired `f64` streams.  For every outer index with bit
/// `control = 1`, bit `target = 0`, and every external control set,
/// swap the LANES_SOA-wide window starting at `outer` with the matching
/// window at `outer | t_bit`.  Per `LANES_SOA` amps: 4 loads + 4 stores
/// (two zmm loads per stream, two zmm stores per stream) — zero
/// multiplies, bandwidth-bound.
///
/// **Outer-walk (bit-disjointness).** Identical renormalise-then-shift
/// idiom as the AoS Tier A kernel.  The inner SIMD walk owns bits
/// `[0, target)` via `j`, and bit `target` is split by the `t_bit`
/// offset between the two halves of the swap pair.  The outer walk
/// reserves bits `[0, target]` and injects `control` (fixed=true) plus
/// every external control (fixed=true) in the above-target subspace,
/// renormalised by `-(target + 1)`.
///
/// The "loose" form — `expand_with_fixed(k, &[(target, false),
/// (control, true), ...])` — pins target and control to the right
/// values but lets `k`'s free bits fall into positions below target
/// where they collide with `j` in the inner walk.  Same bug class as
/// the AoS Task 5/6 first-fix lessons.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << target >= LANES_SOA` (= 8) — inner SIMD walk has ≥ LANES_SOA
///   contiguous amps per half on each stream.
/// * `control > target` — Tier A's control-above-target invariant
///   (matches AoS Tier A; the reverse orientation is not yet wired on
///   the SoA path).
/// * Every external control's qubit index is strictly greater than
///   `max(control, target)`, so the renormalisation subtraction is safe
///   and the outer-walk's bit-expansion never toggles an external
///   control bit.
/// * Distinct + in-range qubits; `re.len() == im.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cnot_avx512(
    re: &mut [f64],
    im: &mut [f64],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let t_bit = 1usize << target;
    let len = re.len();

    debug_assert!(
        t_bit >= LANES_SOA,
        "t_bit < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        control > target,
        "Tier A requires control > target (Tier B not wired on SoA path)"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > target.max(control)),
        "external control at-or-below max(control, target)"
    );

    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    let inner_walk = |outer: usize| {
        let mut j = 0usize;
        while j + LANES_SOA <= t_bit {
            let i0 = outer | j;
            let i1 = i0 | t_bit;
            // SAFETY: bit-disjointness invariant — `outer`'s bits ≥
            // target+1 (with control + every external control bit set,
            // `t_bit` clear), `j` ⊆ [0, target), `t_bit` is bit `target`.
            // All three pairwise disjoint, so i0 + LANES_SOA ≤ outer +
            // t_bit ≤ len and i1 + LANES_SOA ≤ outer + 2*t_bit ≤ len.
            let ar = _mm512_loadu_pd(re_ptr.add(i0));
            let ai = _mm512_loadu_pd(im_ptr.add(i0));
            let br = _mm512_loadu_pd(re_ptr.add(i1));
            let bi = _mm512_loadu_pd(im_ptr.add(i1));
            _mm512_storeu_pd(re_ptr.add(i0), br);
            _mm512_storeu_pd(im_ptr.add(i0), bi);
            _mm512_storeu_pd(re_ptr.add(i1), ar);
            _mm512_storeu_pd(im_ptr.add(i1), ai);
            j += LANES_SOA;
        }
        debug_assert_eq!(j, t_bit);
    };

    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((control - target - 1, true));
    for &c in external_controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - target - 2 - external_controls.len() as u32);
    for k in 0..outer_count {
        let outer = crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1);
        inner_walk(outer);
    }
}

/// Packed AVX-512 SWAP specialisation over paired SoA storage — Tier A
/// (`1 << min(targets) >= LANES_SOA = 8`).
///
/// Mirror of `aos::apply_2q_swap_avx512` (Task 8) on paired `f64`
/// streams.  Per outer block with bits `[0, lo]` zero, swap the
/// LANES_SOA-wide window at `outer | lo_bit` with the window at
/// `outer | hi_bit` on both streams — same swap-pair traffic as CNOT
/// but with `hi_bit` as the partner offset instead of `t_bit`.
///
/// **Outer-walk.** Same renormalise-then-shift as the AoS analogue.
/// The inner walk owns bits `[0, lo)` via `j`; bit `hi` enters as a
/// `fixed=false` slot above lo (the inner walk visits both `hi=0` and
/// `hi=1` per outer iter); external controls are `fixed=true` above lo.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << min(targets) >= LANES_SOA` (= 8) — inner SIMD walk has ≥
///   LANES_SOA contiguous amps per half on each stream.
/// * Distinct targets, both in qubit range.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the renormalisation subtraction is safe and
///   the outer-walk's bit-expansion never toggles an external control
///   bit.
/// * `re.len() == im.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_swap_avx512(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let lo_bit = 1usize << lo;
    let hi_bit = 1usize << hi;
    let len = re.len();

    debug_assert!(
        lo != hi,
        "SWAP requires distinct targets: dispatch contract violated"
    );
    debug_assert!(
        lo_bit >= LANES_SOA,
        "lo_bit < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    let inner_walk = |outer: usize| {
        let mut j = 0usize;
        while j + LANES_SOA <= lo_bit {
            // SAFETY: bit-disjointness — `outer` ⊆ bits ≥ lo+1 with
            // `hi` clear and every external control set; `j` ⊆ [0, lo);
            // `lo_bit`/`hi_bit` are bits `lo`/`hi`.  All pairwise
            // disjoint for both i_01 and i_10, so each + LANES_SOA ≤
            // len.
            let i_01 = outer | lo_bit | j;
            let i_10 = outer | hi_bit | j;
            let ar = _mm512_loadu_pd(re_ptr.add(i_01));
            let ai = _mm512_loadu_pd(im_ptr.add(i_01));
            let br = _mm512_loadu_pd(re_ptr.add(i_10));
            let bi = _mm512_loadu_pd(im_ptr.add(i_10));
            _mm512_storeu_pd(re_ptr.add(i_01), br);
            _mm512_storeu_pd(im_ptr.add(i_01), bi);
            _mm512_storeu_pd(re_ptr.add(i_10), ar);
            _mm512_storeu_pd(im_ptr.add(i_10), ai);
            j += LANES_SOA;
        }
        debug_assert_eq!(j, lo_bit);
    };

    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((hi - lo - 1, false));
    for &ec in external_controls {
        fixed_above.push((ec - lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - lo - 2 - external_controls.len() as u32);
    for k in 0..outer_count {
        let outer = crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1);
        inner_walk(outer);
    }
}

/// Packed AVX-512 CZ specialisation over paired SoA storage — Tier A
/// (`1 << min(targets) >= LANES_SOA = 8`).
///
/// Mirror of `aos::apply_2q_cz_avx512` (Task 10) on paired `f64`
/// streams.  Sign-flip every amp in the `(1, 1)` sub-block via a single
/// `vxorpd` against the IEEE-754 sign-bit mask, applied independently
/// to the `re` and `im` zmm registers.  Per LANES_SOA amps: 2 loads + 2
/// xors + 2 stores ≈ 6 µops; zero multiplies, bandwidth-bound.
///
/// **Outer-walk.** Same renormalise-then-shift as the AoS analogue.
/// `hi` enters `fixed_above` as `fixed=true` (we target the `(1,1)`
/// sub-block); `lo_bit` is OR'd into `outer` after the shift so both
/// target bits are set; external controls are `fixed=true` above lo.
///
/// # Safety
///
/// Same contract as `apply_2q_swap_avx512` above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cz_avx512(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let lo_bit = 1usize << lo;
    let len = re.len();

    debug_assert!(
        lo != hi,
        "CZ requires distinct targets: dispatch contract violated"
    );
    debug_assert!(
        lo_bit >= LANES_SOA,
        "lo_bit < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();
    // Sign-mask: each double-lane has only its IEEE-754 sign bit set,
    // so `vxorpd(z, sign_mask)` flips the sign of every double in `z` —
    // equivalent to `z = -z` for both real and imaginary parts.
    let sign_mask = _mm512_set1_pd(-0.0_f64);

    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((hi - lo - 1, true));
    for &ec in external_controls {
        fixed_above.push((ec - lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - lo - 2 - external_controls.len() as u32);
    for k in 0..outer_count {
        let base = crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1);
        // base has: bit hi = 1, every external_control bit = 1, bits
        // [0, lo] all zero.  ORing in `lo_bit` sets bit `lo`, so the
        // resulting `outer` lands on the (1, 1) sub-block.
        let outer = base | lo_bit;
        let mut j = 0usize;
        while j + LANES_SOA <= lo_bit {
            // SAFETY: bit-disjointness — `base` ⊆ bits ≥ lo+1 (hi set,
            // every external control set), `lo_bit` is bit `lo`, `j`
            // ⊆ [0, lo).  Pairwise disjoint, so `i = outer | j =
            // base + lo_bit + j` and `i + LANES_SOA ≤ len`.
            let i = outer | j;
            let r = _mm512_loadu_pd(re_ptr.add(i));
            let m = _mm512_loadu_pd(im_ptr.add(i));
            _mm512_storeu_pd(re_ptr.add(i), _mm512_xor_pd(r, sign_mask));
            _mm512_storeu_pd(im_ptr.add(i), _mm512_xor_pd(m, sign_mask));
            j += LANES_SOA;
        }
        debug_assert_eq!(j, lo_bit);
    }
}

/// Packed AVX-512 general-diagonal 2q specialisation over paired SoA
/// storage — Tier A (`1 << min(targets) >= LANES_SOA = 8`).
///
/// Mirror of `aos::apply_2q_diagonal_avx512` (Task 11) on paired `f64`
/// streams.  Per outer block we iterate the four `(q_hi, q_lo) ∈
/// {0,1}²` sub-blocks; each sub-block multiplies LANES_SOA contiguous
/// amps by a single broadcast `d[k]`.  The complex multiply expands to
/// two-stream FMAs:
///
/// ```text
/// new_re = re * d_re - im * d_im   ≈ vfmsub231pd(re, d_re_bc, vmulpd(im, d_im_bc))
/// new_im = re * d_im + im * d_re   ≈ vfmadd231pd(im, d_re_bc, vmulpd(re, d_im_bc))
/// ```
///
/// Per LANES_SOA amps per sub-block: ~6 µops (2 muls + 2 FMAs + 2
/// stores plus 2 loads) ≈ 0.75 µops/amp.
///
/// **Sub-block to d[k] mapping.** Same disambiguation as the AoS
/// analogue: `k` is defined by `targets[0]` (MSB) and `targets[1]`
/// (LSB), but the outer-walk thinks in `(q_hi, q_lo)` coordinates.
/// When `targets[0] < targets[1]` (`targets[0] = lo`), `k = (q_lo << 1)
/// | q_hi`; otherwise the usual `k = (q_hi << 1) | q_lo`.
///
/// # Safety
///
/// Same contract as `apply_2q_swap_avx512` above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_diagonal_avx512(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
    d: [Complex; 4],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let lo_bit = 1usize << lo;
    let hi_bit = 1usize << hi;
    let len = re.len();

    debug_assert!(
        lo != hi,
        "2q-diagonal requires distinct targets: dispatch contract violated"
    );
    debug_assert!(
        lo_bit >= LANES_SOA,
        "lo_bit < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    // Disambiguate which d[k] each (q_hi, q_lo) sub-block hits.  MSB
    // convention: k bit 1 = targets[0], k bit 0 = targets[1].
    let (d_for_hi0_lo0, d_for_hi0_lo1, d_for_hi1_lo0, d_for_hi1_lo1) = if targets[0] < targets[1] {
        // targets[0] = lo, targets[1] = hi → k = (q_lo << 1) | q_hi
        (d[0], d[2], d[1], d[3])
    } else {
        // targets[0] = hi, targets[1] = lo → k = (q_hi << 1) | q_lo
        (d[0], d[1], d[2], d[3])
    };

    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    let multiply_block = |base: usize, d_k: Complex| {
        let d_re_bc = _mm512_set1_pd(d_k.re);
        let d_im_bc = _mm512_set1_pd(d_k.im);
        let mut j = 0usize;
        while j + LANES_SOA <= lo_bit {
            // SAFETY: bit-disjointness (see doc-comment).  `base` ⊆
            // bits ≥ lo+1 OR'd with at most {lo_bit, hi_bit}; `j` ⊆
            // [0, lo).  Pairwise disjoint, so `i = base + j` and
            // `i + LANES_SOA ≤ base + lo_bit ≤ len`.
            let i = base | j;
            let r = _mm512_loadu_pd(re_ptr.add(i));
            let m = _mm512_loadu_pd(im_ptr.add(i));
            // new_re = r * d_re - m * d_im
            let new_r = _mm512_sub_pd(_mm512_mul_pd(r, d_re_bc), _mm512_mul_pd(m, d_im_bc));
            // new_im = r * d_im + m * d_re
            let new_m = _mm512_add_pd(_mm512_mul_pd(r, d_im_bc), _mm512_mul_pd(m, d_re_bc));
            _mm512_storeu_pd(re_ptr.add(i), new_r);
            _mm512_storeu_pd(im_ptr.add(i), new_m);
            j += LANES_SOA;
        }
        debug_assert_eq!(j, lo_bit);
    };

    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((hi - lo - 1, false));
    for &ec in external_controls {
        fixed_above.push((ec - lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(p, _)| p);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - lo - 2 - external_controls.len() as u32);
    for k in 0..outer_count {
        let base = crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1);
        // base has: bit hi = 0, every external_control bit = 1, bits
        // [0, lo] all zero.  Iterate the 4 sub-blocks:
        multiply_block(base, d_for_hi0_lo0); // (q_hi=0, q_lo=0)
        multiply_block(base | lo_bit, d_for_hi0_lo1); // (q_hi=0, q_lo=1)
        multiply_block(base | hi_bit, d_for_hi1_lo0); // (q_hi=1, q_lo=0)
        multiply_block(base | hi_bit | lo_bit, d_for_hi1_lo1); // (q_hi=1, q_lo=1)
    }
}

/// Top-level SoA 2q dispatch.  Mirrors `aos::apply_2q` — see spec § 4.9.
/// Detection order:
/// 1. `classify_2q_permutation` → Identity / CnotHi / CnotLo / Swap fast paths.
/// 2. `is_diagonal_4x4` → CZ (`is_cz_signature` shortcut) / general diagonal fast path.
/// 3. Otherwise: `apply_2q_dense_scalar`.
///
/// All paths are scalar in this task; AVX-512 specialisations land in
/// Tasks 13/14 (mirror of AoS Tasks 5-11).
pub(crate) fn apply_2q(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex; 4]; 4],
) {
    // 1. Permutation detection (Identity / CNOT / SWAP).
    match super::classify_2q_permutation(m) {
        Some(super::Perm2qKind::Identity) => return,
        Some(super::Perm2qKind::CnotHi) => {
            dispatch_cnot_soa(re, im, targets[0], targets[1], controls);
            return;
        }
        Some(super::Perm2qKind::CnotLo) => {
            dispatch_cnot_soa(re, im, targets[1], targets[0], controls);
            return;
        }
        Some(super::Perm2qKind::Swap) => {
            dispatch_swap_soa(re, im, targets, controls);
            return;
        }
        None => {}
    }

    // 2. Diagonal-4x4 (catches Cz, controlled-Phase, Rzz, user diagonals).
    if super::is_diagonal_4x4(m) {
        let d = [m[0][0], m[1][1], m[2][2], m[3][3]];
        let is_cz = super::is_cz_signature(d);
        dispatch_diagonal_or_cz_soa(re, im, targets, controls, d, is_cz);
        return;
    }

    // 3. Generic dense 4×4 — scalar for now; AVX-512 lands in Task 14.
    apply_2q_dense_scalar(re, im, targets, controls, m);
}

/// Dispatch helper for SoA CNOT specialisations.  Routes to the Tier A
/// AVX-512 kernel when the host + qubit orientation satisfies the
/// safety contract (AVX-512F detected, `1 << target >= LANES_SOA`,
/// `control > target`, every external control above `max(control,
/// target)`); otherwise falls through to the scalar specialised
/// kernel.  Mirror of `aos::dispatch_cnot` (Tier A only — SoA Tier B/C
/// are not yet wired).
fn dispatch_cnot_soa(re: &mut [f64], im: &mut [f64], control: u32, target: u32, controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << target) >= LANES_SOA
            && control > target
            && controls.iter().all(|&c| c > target.max(control))
        {
            // SAFETY: AVX-512F detected; `1 << target >= LANES_SOA`;
            // `control > target`; every external control > max(control,
            // target).  Distinct qubits + in-range are enforced by the
            // parent `apply_gate` boundary.
            unsafe {
                apply_2q_cnot_avx512(re, im, control, target, controls);
            }
            return;
        }
    }
    apply_2q_cnot_scalar(re, im, control, target, controls);
}

/// Dispatch helper for SoA SWAP.  Routes to the Tier A AVX-512 kernel
/// when the host + qubit orientation satisfies the safety contract;
/// otherwise falls through to the scalar specialised kernel.  Mirror
/// of `aos::dispatch_swap` (Tier A only).
fn dispatch_swap_soa(re: &mut [f64], im: &mut [f64], targets: [u32; 2], controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let lo = targets[0].min(targets[1]);
        let hi = targets[0].max(targets[1]);
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << lo) >= LANES_SOA
            && controls.iter().all(|&c| c > hi)
        {
            // SAFETY: AVX-512F detected; `1 << lo >= LANES_SOA`; every
            // external control > hi; distinct qubits enforced by parent
            // `apply_gate`.
            unsafe {
                apply_2q_swap_avx512(re, im, targets, controls);
            }
            return;
        }
    }
    apply_2q_swap_scalar(re, im, targets, controls);
}

/// Dispatch helper for the diagonal-4x4 branch (catches CZ,
/// controlled-Phase, Rzz, user diagonals).  Routes to the matching
/// Tier A AVX-512 kernel (CZ via `apply_2q_cz_avx512`, general
/// diagonal via `apply_2q_diagonal_avx512`) when the host + qubit
/// orientation satisfies the safety contract; otherwise falls through
/// to the scalar specialised kernels.  Mirror of
/// `aos::dispatch_diagonal_or_cz` (Tier A only).
fn dispatch_diagonal_or_cz_soa(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    controls: &[u32],
    d: [Complex; 4],
    is_cz: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        let lo = targets[0].min(targets[1]);
        let hi = targets[0].max(targets[1]);
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << lo) >= LANES_SOA
            && controls.iter().all(|&c| c > hi)
        {
            // SAFETY: AVX-512F detected; `1 << lo >= LANES_SOA`; every
            // external control > hi; distinct qubits enforced by
            // parent `apply_gate`.
            if is_cz {
                unsafe {
                    apply_2q_cz_avx512(re, im, targets, controls);
                }
            } else {
                unsafe {
                    apply_2q_diagonal_avx512(re, im, targets, controls, d);
                }
            }
            return;
        }
    }
    if is_cz {
        apply_2q_cz_scalar(re, im, targets, controls);
    } else {
        apply_2q_diagonal_scalar(re, im, targets, controls, d);
    }
}

/// Apply a 1-qubit matrix to `target` (with external `controls`) in
/// place over a paired `(re, im)` SoA state. See the `aos.rs` analogue
/// for the index-pair convention.
pub(crate) fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    debug_assert_eq!(re.len(), im.len());
    // Diagonal fast path (P1-06).  Same heuristic as the AoS path.
    if super::is_diagonal_2x2(m) {
        apply_1q_diagonal_soa(re, im, target, controls, m[0][0], m[1][1]);
        return;
    }
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let a0_re = re[i];
            let a0_im = im[i];
            let a1_re = re[j];
            let a1_im = im[j];
            // row 0
            re[i] =
                m[0][0].re * a0_re - m[0][0].im * a0_im + m[0][1].re * a1_re - m[0][1].im * a1_im;
            im[i] =
                m[0][0].re * a0_im + m[0][0].im * a0_re + m[0][1].re * a1_im + m[0][1].im * a1_re;
            // row 1
            re[j] =
                m[1][0].re * a0_re - m[1][0].im * a0_im + m[1][1].re * a1_re - m[1][1].im * a1_im;
            im[j] =
                m[1][0].re * a0_im + m[1][0].im * a0_re + m[1][1].re * a1_im + m[1][1].im * a1_re;
        }
        i += 1;
    }
}

/// SoA diagonal 1q fast path.  Each amplitude is a complex pair
/// `(re[i], im[i])`; the diagonal multiply by `d = (d_re, d_im)` is
/// `new_re = re*d_re - im*d_im` and `new_im = re*d_im + im*d_re`.
///
/// Only the current amp's two streams mix — no cross-amp coupling.
/// LLVM should auto-vectorise the inner block to 4-lane `vmulpd ymm`
/// or 8-lane `vmulpd zmm` depending on host features and walk
/// granularity.
pub(crate) fn apply_1q_diagonal_soa(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m00: Complex,
    m11: Complex,
) {
    debug_assert_eq!(re.len(), im.len());
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask {
            let (d_re, d_im) = if (i & t_bit) == 0 {
                (m00.re, m00.im)
            } else {
                (m11.re, m11.im)
            };
            let r = re[i];
            let im_v = im[i];
            re[i] = r * d_re - im_v * d_im;
            im[i] = r * d_im + im_v * d_re;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::aos;

    fn pauli_x() -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[z, o], [o, z]]
    }

    fn hadamard() -> [[Complex; 2]; 2] {
        let s = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        [[s, s], [s, -s]]
    }

    #[test]
    fn apply_1q_diagonal_soa_matches_aos_phase() {
        let theta = 1.7_f64;
        let m = [
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [
                Complex::new(0.0, 0.0),
                Complex::new(theta.cos(), theta.sin()),
            ],
        ];
        let aos_state_init: Vec<Complex> = (0..8)
            .map(|k| Complex::new(0.2 * k as f64, -0.05 * k as f64))
            .collect();
        let mut aos_state = aos_state_init.clone();
        let mut soa_re: Vec<f64> = aos_state_init.iter().map(|c| c.re).collect();
        let mut soa_im: Vec<f64> = aos_state_init.iter().map(|c| c.im).collect();
        aos::apply_1q(&mut aos_state, 1, &[], &m);
        apply_1q(&mut soa_re, &mut soa_im, 1, &[], &m); // exercises diagonal route
        for k in 0..aos_state.len() {
            assert!((aos_state[k].re - soa_re[k]).abs() < 1e-14);
            assert!((aos_state[k].im - soa_im[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn apply_1q_diagonal_soa_matches_aos_with_control() {
        // diag(2, -1) on q=0, controlled by q=2.  4 qubits, 16 amps.
        let m00 = Complex::new(2.0, 0.0);
        let m11 = Complex::new(-1.0, 0.0);
        let m = [[m00, Complex::new(0.0, 0.0)], [Complex::new(0.0, 0.0), m11]];
        let aos_state_init: Vec<Complex> = (0..16)
            .map(|k| Complex::new(0.11 * k as f64, 0.05 * k as f64))
            .collect();
        let mut aos_state = aos_state_init.clone();
        let mut soa_re: Vec<f64> = aos_state_init.iter().map(|c| c.re).collect();
        let mut soa_im: Vec<f64> = aos_state_init.iter().map(|c| c.im).collect();
        aos::apply_1q(&mut aos_state, 0, &[2], &m);
        apply_1q(&mut soa_re, &mut soa_im, 0, &[2], &m);
        for k in 0..aos_state.len() {
            assert!((aos_state[k].re - soa_re[k]).abs() < 1e-14);
            assert!((aos_state[k].im - soa_im[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn x_flips_single_qubit_soa() {
        let mut re = vec![1.0, 0.0];
        let mut im = vec![0.0, 0.0];
        apply_1q(&mut re, &mut im, 0, &[], &pauli_x());
        assert_eq!(re, vec![0.0, 1.0]);
        assert_eq!(im, vec![0.0, 0.0]);
    }

    #[test]
    fn h_on_zero_yields_plus_soa() {
        let mut re = vec![1.0, 0.0];
        let mut im = vec![0.0, 0.0];
        apply_1q(&mut re, &mut im, 0, &[], &hadamard());
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((re[0] - s).abs() < 1e-12);
        assert!((re[1] - s).abs() < 1e-12);
        assert!(im[0].abs() < 1e-12);
        assert!(im[1].abs() < 1e-12);
    }

    #[test]
    fn external_control_skips_when_unset_soa() {
        // 2-qubit state amps[0] = 1 (q0 = 0, q1 = 0); external control q0.
        let mut re = vec![1.0, 0.0, 0.0, 0.0];
        let mut im = vec![0.0; 4];
        apply_1q(&mut re, &mut im, 1, &[0], &pauli_x());
        assert_eq!(re, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn external_control_fires_when_set_soa() {
        // 2-qubit state amps[1] = 1 (q0 = 1, q1 = 0); CX(c=q0,t=q1) flips q1.
        let mut re = vec![0.0, 1.0, 0.0, 0.0];
        let mut im = vec![0.0; 4];
        apply_1q(&mut re, &mut im, 1, &[0], &pauli_x());
        assert_eq!(re, vec![0.0, 0.0, 0.0, 1.0]);
    }

    /// Helper: build an AoS state from paired (re, im) slices.
    fn aos_from(re: &[f64], im: &[f64]) -> Vec<Complex> {
        re.iter()
            .zip(im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect()
    }

    fn cnot() -> [[Complex; 4]; 4] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    }

    #[test]
    fn cnot_creates_bell_soa() {
        // Start from |+0⟩ encoded as amps[0]=amps[1]=inv (after H on q0
        // applied to |00⟩). CNOT(c=q0,t=q1) routes the q0=1 mass from
        // amps[1] to amps[3].
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        let mut re = vec![inv, inv, 0.0, 0.0];
        let mut im = vec![0.0; 4];
        apply_2q(&mut re, &mut im, [0, 1], &[], &cnot());
        assert!((re[0] - inv).abs() < 1e-12);
        assert!(re[1].abs() < 1e-12);
        assert!(re[2].abs() < 1e-12);
        assert!((re[3] - inv).abs() < 1e-12);
        assert!(im.iter().all(|x| x.abs() < 1e-12));
    }

    fn toffoli() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for (i, row) in m.iter_mut().enumerate().take(6) {
            row[i] = o;
        }
        m[6][7] = o;
        m[7][6] = o;
        m
    }

    #[test]
    fn toffoli_flips_target_when_both_controls_set_soa() {
        // amps[3] = 1.0 → (q0 = 1, q1 = 1, q2 = 0); Toffoli swaps k=6 ↔ k=7
        // → amps[3] moves to amps[7].
        let mut re = vec![0.0; 8];
        let mut im = vec![0.0; 8];
        re[3] = 1.0;
        apply_3q(&mut re, &mut im, [0, 1, 2], &[], &toffoli());
        assert!((re[7] - 1.0).abs() < 1e-12);
        assert!(re[3].abs() < 1e-12);
    }

    #[test]
    fn toffoli_with_single_control_set_is_identity_soa() {
        let mut re = vec![0.0; 8];
        let mut im = vec![0.0; 8];
        re[1] = 1.0;
        apply_3q(&mut re, &mut im, [0, 1, 2], &[], &toffoli());
        assert!((re[1] - 1.0).abs() < 1e-12);
    }

    use aleph_core::GateMatrix;
    use aleph_test::gate::{arb_1q_gate, arb_2q_gate};
    use aleph_test::state::arb_state_vector;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// AoS / SoA equivalence on `apply_1q`: for any 1q gate and any
        /// normalised state, applying through both kernels yields
        /// matching amplitudes within 1e-12.
        #[test]
        fn apply_1q_soa_matches_aos(
            gate in arb_1q_gate(),
            q in 0u32..5,
            amps in arb_state_vector(5),
        ) {
            let m = match gate.matrix().unwrap() {
                GateMatrix::M2x2(m) => m,
                _ => unreachable!("arb_1q_gate yields 1q gates"),
            };
            let re: Vec<f64> = amps.iter().map(|c| c.re).collect();
            let im: Vec<f64> = amps.iter().map(|c| c.im).collect();
            // AoS reference
            let mut aos_state = amps.clone();
            aos::apply_1q(&mut aos_state, q, &[], &m);
            // SoA candidate
            let mut soa_re = re.clone();
            let mut soa_im = im.clone();
            apply_1q(&mut soa_re, &mut soa_im, q, &[], &m);
            let soa_state = aos_from(&soa_re, &soa_im);
            for (a, b) in aos_state.iter().zip(soa_state.iter()) {
                prop_assert!((a - b).norm() < 1e-12, "aos {a} vs soa {b}");
            }
        }

        /// AoS / SoA equivalence on `apply_2q`. Distinct targets are
        /// enforced by `prop_assume` (the strategy generates qubits
        /// independently; the kernel itself only requires `t0 != t1`
        /// via the parent backend's duplicate-qubit check).
        #[test]
        fn apply_2q_soa_matches_aos(
            gate in arb_2q_gate(),
            t0 in 0u32..5,
            t1 in 0u32..5,
            amps in arb_state_vector(5),
        ) {
            prop_assume!(t0 != t1);
            let m = match gate.matrix().unwrap() {
                GateMatrix::M4x4(m) => m,
                _ => unreachable!("arb_2q_gate yields 2q gates"),
            };
            let re: Vec<f64> = amps.iter().map(|c| c.re).collect();
            let im: Vec<f64> = amps.iter().map(|c| c.im).collect();
            let mut aos_state = amps.clone();
            aos::apply_2q(&mut aos_state, [t0, t1], &[], &m);
            let mut soa_re = re.clone();
            let mut soa_im = im.clone();
            apply_2q(&mut soa_re, &mut soa_im, [t0, t1], &[], &m);
            let soa_state = aos_from(&soa_re, &soa_im);
            for (a, b) in aos_state.iter().zip(soa_state.iter()) {
                prop_assert!((a - b).norm() < 1e-12, "aos {a} vs soa {b}");
            }
        }
    }

    fn random_re_im(n_qubits: u32, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut s = seed.wrapping_add(1);
        let mut lcg = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
        };
        let len = 1usize << n_qubits;
        let mut re = Vec::with_capacity(len);
        let mut im = Vec::with_capacity(len);
        for _ in 0..len {
            re.push(lcg());
            im.push(lcg());
        }
        (re, im)
    }

    fn assert_re_im_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b.iter()) {
            assert!(
                (ai - bi).abs() < tol,
                "diff {} > tol {}",
                (ai - bi).abs(),
                tol
            );
        }
    }

    #[test]
    fn soa_apply_2q_cnot_scalar_matches_dense_scalar() {
        let n = 6;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][3] = Complex::new(1.0, 0.0);
            m[3][2] = Complex::new(1.0, 0.0);
            m
        };
        for (c, t) in [(0u32, 1), (1, 0), (2, 5), (3, 5)] {
            let (r0, i0) = random_re_im(n, 0xabcd);
            let mut ra = r0.clone();
            let mut ia = i0.clone();
            let mut rb = r0;
            let mut ib = i0;
            apply_2q_dense_scalar(&mut ra, &mut ia, [c, t], &[], &m);
            apply_2q_cnot_scalar(&mut rb, &mut ib, c, t, &[]);
            assert_re_im_close(&ra, &rb, 1e-14);
            assert_re_im_close(&ia, &ib, 1e-14);
        }
    }

    #[test]
    fn soa_apply_2q_prelude_dispatches_identity_as_noop() {
        let n = 5;
        let id = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = Complex::new(1.0, 0.0);
            }
            m
        };
        let (r0, i0) = random_re_im(n, 0x4242);
        let mut r = r0.clone();
        let mut imv = i0.clone();
        apply_2q(&mut r, &mut imv, [0, 1], &[], &id);
        assert_re_im_close(&r, &r0, 1e-15);
        assert_re_im_close(&imv, &i0, 1e-15);
    }

    #[test]
    fn soa_apply_2q_cz_scalar_matches_dense_scalar() {
        let n = 6;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][2] = Complex::new(1.0, 0.0);
            m[3][3] = Complex::new(-1.0, 0.0);
            m
        };
        for t in [[0u32, 1], [2, 5]] {
            let (r0, i0) = random_re_im(n, 0xfeed);
            let mut ra = r0.clone();
            let mut ia = i0.clone();
            let mut rb = r0;
            let mut ib = i0;
            apply_2q_dense_scalar(&mut ra, &mut ia, t, &[], &m);
            apply_2q_cz_scalar(&mut rb, &mut ib, t, &[]);
            assert_re_im_close(&ra, &rb, 1e-14);
            assert_re_im_close(&ia, &ib, 1e-14);
        }
    }

    /// Tier A AVX-512 CNOT equivalence vs scalar SoA CNOT.  All
    /// `(control, target)` cases satisfy `control > target` and
    /// `1 << target >= LANES_SOA = 8`, i.e. `target >= 3`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_cnot_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [7u32, 8, 10] {
            for (c, t) in [(4u32, 3), (5, 3), (6, 4), (7, 5)] {
                if n <= c.max(t) {
                    continue;
                }
                if (1usize << t) < LANES_SOA || c <= t {
                    continue;
                }
                let (r0, i0) = random_re_im(n, 0xc01f_5a + n as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_cnot_scalar(&mut ra, &mut ia, c, t, &[]);
                // SAFETY: AVX-512F detected; t_bit = 1<<t ≥ LANES_SOA;
                // c > t (Tier A); no external controls.
                unsafe {
                    super::apply_2q_cnot_avx512(&mut rb, &mut ib, c, t, &[]);
                }
                assert_re_im_close(&ra, &rb, 1e-14);
                assert_re_im_close(&ia, &ib, 1e-14);
            }
        }
    }

    /// Tier A AVX-512 SWAP equivalence vs scalar SoA SWAP.  All
    /// `targets` cases satisfy `1 << min(targets) >= LANES_SOA = 8`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_swap_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [8u32, 10] {
            for t in [[3u32, 4], [3, 5], [4, 6], [5, 7]] {
                let hi = t[0].max(t[1]);
                if n <= hi {
                    continue;
                }
                let lo = t[0].min(t[1]);
                if (1usize << lo) < LANES_SOA {
                    continue;
                }
                let (r0, i0) = random_re_im(n, 0x5a5a_de + n as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_swap_scalar(&mut ra, &mut ia, t, &[]);
                // SAFETY: AVX-512F detected; lo_bit ≥ LANES_SOA;
                // distinct targets; no external controls.
                unsafe {
                    super::apply_2q_swap_avx512(&mut rb, &mut ib, t, &[]);
                }
                assert_re_im_close(&ra, &rb, 1e-14);
                assert_re_im_close(&ia, &ib, 1e-14);
            }
        }
    }

    /// Tier A AVX-512 CZ equivalence vs scalar SoA CZ.  All `targets`
    /// cases satisfy `1 << min(targets) >= LANES_SOA = 8`.  Pure
    /// sign-flip so 1e-15 tolerance is appropriate.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_cz_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [8u32, 10] {
            for t in [[3u32, 4], [3, 5], [4, 6], [5, 7]] {
                let hi = t[0].max(t[1]);
                if n <= hi {
                    continue;
                }
                let lo = t[0].min(t[1]);
                if (1usize << lo) < LANES_SOA {
                    continue;
                }
                let (r0, i0) = random_re_im(n, 0xfeed_cz + n as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_cz_scalar(&mut ra, &mut ia, t, &[]);
                // SAFETY: AVX-512F detected; lo_bit ≥ LANES_SOA;
                // distinct targets; no external controls.
                unsafe {
                    super::apply_2q_cz_avx512(&mut rb, &mut ib, t, &[]);
                }
                assert_re_im_close(&ra, &rb, 1e-15);
                assert_re_im_close(&ia, &ib, 1e-15);
            }
        }
    }

    /// Tier A AVX-512 general-diagonal equivalence vs scalar SoA
    /// diagonal.  Non-CZ d-tuple (the CZ path is exercised by the
    /// dedicated CZ test above).  Exercises both `targets[0] <
    /// targets[1]` and `targets[0] > targets[1]` to cover the
    /// d-tuple disambiguation.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_diagonal_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let d = [
            Complex::new(0.6, 0.8),
            Complex::new(-0.7, 0.71_428_571_428_571_43),
            Complex::new(0.99, -0.141_421_356_237_309_5),
            Complex::new(-0.5, -0.866_025_403_784_438_6),
        ];
        for n in [8u32, 10] {
            for t in [[3u32, 4], [3, 5], [4, 6], [7, 5], [6, 4]] {
                let hi = t[0].max(t[1]);
                if n <= hi {
                    continue;
                }
                let lo = t[0].min(t[1]);
                if (1usize << lo) < LANES_SOA {
                    continue;
                }
                let (r0, i0) = random_re_im(n, 0x1357_d2 + n as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_diagonal_scalar(&mut ra, &mut ia, t, &[], d);
                // SAFETY: AVX-512F detected; lo_bit ≥ LANES_SOA;
                // distinct targets; no external controls.
                unsafe {
                    super::apply_2q_diagonal_avx512(&mut rb, &mut ib, t, &[], d);
                }
                assert_re_im_close(&ra, &rb, 1e-14);
                assert_re_im_close(&ia, &ib, 1e-14);
            }
        }
    }

    /// Portable indexing-coverage check for the SoA Tier A outer-walk
    /// and inner-walk pattern shared by SWAP / CZ / diagonal.  Uses
    /// only integer arithmetic (no AVX-512 intrinsics) so it runs on
    /// aarch64 too; asserts every amp in the swap-pair subspace is
    /// touched exactly once and every fixed-point (lo, hi) = (0, 0)
    /// or (1, 1) amp is touched zero times.  Catches bit-collision
    /// bugs in the renormalise-then-shift outer-walk that would
    /// otherwise surface only on a real AVX-512 host.
    #[test]
    fn soa_apply_2q_tier_a_indexing_covers_state_exactly_once() {
        // SWAP-shape (hi enters as fixed=false; both i_01 and i_10 get
        // visited).  Same renormalisation pattern as the diagonal
        // kernel sans the (1,1) sub-block enumeration — covering it
        // here covers all four Tier A kernels' outer-walks.
        let cases: &[(u32, [u32; 2], &[u32])] = &[(8, [3, 4], &[]), (10, [3, 5], &[7])];
        for &(n_qubits, targets, external_controls) in cases {
            let len = 1usize << n_qubits;
            let lo = targets[0].min(targets[1]);
            let hi = targets[0].max(targets[1]);
            let lo_bit = 1usize << lo;
            let hi_bit = 1usize << hi;
            // LANES_SOA = 8 (one zmm of f64s); hard-coded here so the
            // test stays portable to non-x86_64 (where the const is
            // gated out).
            let lanes = 8usize;

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            fixed_above.push((hi - lo - 1, false));
            for &ec in external_controls {
                fixed_above.push((ec - lo - 1, true));
            }
            fixed_above.sort_unstable_by_key(|&(p, _)| p);

            let outer_count = 1usize << (n_qubits - lo - 2 - external_controls.len() as u32);

            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }

            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let outer = crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1);
                assert_eq!(
                    outer & ec_mask,
                    ec_mask,
                    "outer={outer:#b} missing ec bits {ec_mask:#b}"
                );
                assert_eq!(
                    outer & hi_bit,
                    0,
                    "outer={outer:#b} has hi_bit={hi_bit:#b} set"
                );
                assert_eq!(
                    outer & ((1usize << (lo + 1)) - 1),
                    0,
                    "outer={outer:#b} has bits set in [0, lo]"
                );
                for &off in &[lo_bit, hi_bit] {
                    let mut j = 0usize;
                    while j + lanes <= lo_bit {
                        let i = outer | off | j;
                        for d in 0..lanes {
                            assert!(
                                i + d < len,
                                "n={n_qubits} t={targets:?}: OOB i+d={} len={}",
                                i + d,
                                len
                            );
                            touched[i + d] += 1;
                        }
                        j += lanes;
                    }
                }
            }
            for (idx, &count) in touched.iter().enumerate() {
                let bit_lo = (idx & lo_bit) != 0;
                let bit_hi = (idx & hi_bit) != 0;
                let in_ec_subspace = (idx & ec_mask) == ec_mask;
                let swap_pair_member = bit_lo ^ bit_hi;
                let expected = if in_ec_subspace && swap_pair_member {
                    1u32
                } else {
                    0u32
                };
                assert_eq!(
                    count, expected,
                    "n={n_qubits} t={targets:?}: amp {idx} touched {count} times \
                     (expected {expected}; ec_subspace={in_ec_subspace} \
                     swap_pair_member={swap_pair_member})"
                );
            }
        }
    }

    #[test]
    fn soa_apply_2q_diagonal_scalar_matches_dense_scalar() {
        let n = 6;
        let d = [
            Complex::new(0.6, 0.8),
            Complex::new(-0.7, 0.7142857142857143),
            Complex::new(0.99, -0.1414213562373095),
            Complex::new(-0.5, -0.8660254037844386),
        ];
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        for (k, row) in m.iter_mut().enumerate() {
            row[k] = d[k];
        }
        for t in [[0u32, 1], [2, 5]] {
            let (r0, i0) = random_re_im(n, 0x1357);
            let mut ra = r0.clone();
            let mut ia = i0.clone();
            let mut rb = r0;
            let mut ib = i0;
            apply_2q_dense_scalar(&mut ra, &mut ia, t, &[], &m);
            apply_2q_diagonal_scalar(&mut rb, &mut ib, t, &[], d);
            assert_re_im_close(&ra, &rb, 1e-14);
            assert_re_im_close(&ia, &ib, 1e-14);
        }
    }
}
