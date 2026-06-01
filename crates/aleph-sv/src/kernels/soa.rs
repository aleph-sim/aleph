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

/// Top-level 3q dispatch for SoA. Matrix-detects Toffoli (CCX) and CCZ
/// shapes per spec §3.1 and routes to specialised SoA paths.  Identity
/// short-circuits.  Falls through to the generic 8×8 scalar kernel for
/// arbitrary matrices.  Mirror of `aos::apply_3q`.
pub(crate) fn apply_3q(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 3],
    controls: &[u32],
    m: &[[Complex; 8]; 8],
) {
    if super::is_identity_8x8(m) {
        return;
    }
    if super::is_toffoli(m) {
        dispatch_toffoli_soa(re, im, targets, controls);
        return;
    }
    if super::is_ccz(m) {
        dispatch_ccz_soa(re, im, targets, controls);
        return;
    }
    apply_3q_generic_soa(re, im, targets, controls, m);
}

/// Scalar fallback for arbitrary 8×8 matrices over SoA storage.
/// Renamed from the pre-P1-08 `apply_3q`; the public entry-point is now
/// `apply_3q` (with dispatch).  MSB convention identical to `aos`.
fn apply_3q_generic_soa(
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

/// Scalar Tier-C reference for Toffoli (CCX) on SoA storage.
///
/// Mirror of `aos::apply_toffoli_scalar` operating on separate `re` and
/// `im` streams.  Walks every amplitude index `i`; swaps `re[i] ↔
/// re[i | target_bit]` AND `im[i] ↔ im[i | target_bit]` when every
/// control bit (c0, c1, external) is set in `i` and the target bit is
/// clear.
pub(crate) fn apply_toffoli_scalar_soa(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 3],
    external_controls: &[u32],
) {
    debug_assert_eq!(re.len(), im.len());
    let c0 = targets[0];
    let c1 = targets[1];
    let t = targets[2];
    let target_bit = 1usize << t;
    let mut ctrl_mask = (1usize << c0) | (1usize << c1);
    for &e in external_controls {
        ctrl_mask |= 1usize << e;
    }
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask && (i & target_bit) == 0 {
            re.swap(i, i | target_bit);
            im.swap(i, i | target_bit);
        }
        i += 1;
    }
}

/// Scalar Tier-C reference for CCZ on SoA storage.
///
/// Mirror of `aos::apply_ccz_scalar` operating on separate `re` and `im`
/// streams.  Negates `re[i]` AND `im[i]` when all three qubits AND any
/// external controls are |1⟩, i.e. `(i & mask) == mask`.
pub(crate) fn apply_ccz_scalar_soa(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 3],
    external_controls: &[u32],
) {
    debug_assert_eq!(re.len(), im.len());
    let mut mask = (1usize << targets[0]) | (1usize << targets[1]) | (1usize << targets[2]);
    for &e in external_controls {
        mask |= 1usize << e;
    }
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if (i & mask) == mask {
            re[i] = -re[i];
            im[i] = -im[i];
        }
        i += 1;
    }
}

/// Toffoli Tier-A (clean) for SoA: target bit ≥ LANES_SOA=8, every
/// control strictly above target.  Mirror of `aos::apply_toffoli_avx512_tier_a`
/// but operating on paired `re`/`im` streams.  Per matching LANES_SOA-wide
/// block: swap the lo-half and hi-half windows on BOTH streams.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * `target >= 3` i.e. `1 << target >= LANES_SOA` (= 8).
/// * Every qubit position in `sorted_controls` is strictly greater than
///   `target` — guarantees no control bit aliases into the inner j-sweep
///   range `[0, target)`.
/// * `re.len() == im.len() == 1 << n` for some n ≥ 4 (circuit invariant
///   with target ≥ 3).
/// * All elements of `sorted_controls` are distinct, differ from `target`,
///   and are valid qubit indices (< n).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_a_soa(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;

    debug_assert!(
        target >= 3,
        "SoA Tier-A contract: target must be >= LANES_SOA_BITS (3)"
    );
    debug_assert!(
        sorted_controls.iter().all(|&c| c > target),
        "SoA Tier-A contract: every control bit must be strictly above target"
    );

    let target_bit = 1usize << target;
    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = re.len();
    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    let count = len / LANES_SOA;
    crate::kernels::par_blocks(
        count,
        len,
        |k| k * LANES_SOA,
        |block_base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            if (block_base & target_bit) != 0 {
                return;
            }
            if (block_base & ctrl_mask) != ctrl_mask {
                return;
            }
            // SAFETY: block_base & target_bit == 0, all control bits set.
            // target_bit >= LANES_SOA ensures block_base | target_bit ≥
            // block_base + LANES_SOA; both windows are within [0, len).
            let lo = block_base;
            let hi = block_base | target_bit;
            let ar = _mm512_loadu_pd(re_ptr.add(lo));
            let ai = _mm512_loadu_pd(im_ptr.add(lo));
            let br = _mm512_loadu_pd(re_ptr.add(hi));
            let bi = _mm512_loadu_pd(im_ptr.add(hi));
            _mm512_storeu_pd(re_ptr.add(lo), br);
            _mm512_storeu_pd(im_ptr.add(lo), bi);
            _mm512_storeu_pd(re_ptr.add(hi), ar);
            _mm512_storeu_pd(im_ptr.add(hi), ai);
        },
    );
}

/// Toffoli Tier-A outer-walk for SoA: handles controls at-or-above
/// `LANES_SOA_BITS = 3` but not strictly above target. Controls below
/// LANES_SOA_BITS are NOT supported — `block_base` advances by
/// `LANES_SOA = 8`, so its low 3 bits are always zero, and any
/// sub-LANES control bit makes the mask test always fail, silently
/// dropping the gate. `dispatch_toffoli_soa` enforces this; do not
/// invoke directly without verifying.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * `target >= 3` i.e. `1 << target >= LANES_SOA` (= 8).
/// * Every element of `sorted_controls` is `>= LANES_SOA_BITS = 3`.
///   Sub-LANES controls silently disable the gate.
/// * All elements of `sorted_controls` are distinct, differ from `target`,
///   and are valid qubit indices (< n).
/// * `re.len() == im.len() == 1 << n` for some n ≥ 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_a_outer_walk_soa(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;

    debug_assert!(
        target >= 3,
        "SoA Tier-A outer-walk contract: target must be >= LANES_SOA_BITS (3)"
    );

    let target_bit = 1usize << target;
    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = re.len();
    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    let count = len / LANES_SOA;
    crate::kernels::par_blocks(
        count,
        len,
        |k| k * LANES_SOA,
        |block_base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            if (block_base & target_bit) != 0 {
                return;
            }
            if (block_base & ctrl_mask) != ctrl_mask {
                return;
            }
            // SAFETY: target_bit >= LANES_SOA ensures the hi window is in bounds.
            let lo = block_base;
            let hi = block_base | target_bit;
            let ar = _mm512_loadu_pd(re_ptr.add(lo));
            let ai = _mm512_loadu_pd(im_ptr.add(lo));
            let br = _mm512_loadu_pd(re_ptr.add(hi));
            let bi = _mm512_loadu_pd(im_ptr.add(hi));
            _mm512_storeu_pd(re_ptr.add(lo), br);
            _mm512_storeu_pd(im_ptr.add(lo), bi);
            _mm512_storeu_pd(re_ptr.add(hi), ar);
            _mm512_storeu_pd(im_ptr.add(hi), ai);
        },
    );
}

/// Toffoli Tier-B.0 for SoA: `target=0`, in-register permute swap.
///
/// For `t=0` the target bit is bit 0 of the amplitude index.  Within
/// each 8-double zmm on each stream, consecutive lane pairs `(0↔1)`,
/// `(2↔3)`, `(4↔5)`, `(6↔7)` differ only in bit 0.  One
/// `vpermutexvar_pd` swaps them in place.
///
/// **Permute index (lane-0-first):** `(1,0, 3,2, 5,4, 7,6)`.
/// `_mm512_set_epi64` takes HIGH-to-LOW args (arg 0 → lane 7, arg 7 →
/// lane 0), so the call is `_mm512_set_epi64(6,7, 4,5, 2,3, 0,1)`.
///
/// The SAME permute is applied independently to the `re` and `im` streams.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * Every entry in `sorted_controls` is ≥ `LANES_SOA_BITS = 3`.
/// * `re.len() == im.len() == 1 << n` for some `n ≥ 4`.
/// * All entries distinct, differ from 0, in-range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_b0_soa(
    re: &mut [f64],
    im: &mut [f64],
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;

    debug_assert!(
        sorted_controls.iter().all(|&c| c >= 3),
        "SoA Tier-B.0 contract: every control must be at qubit index >= LANES_SOA_BITS = 3"
    );

    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = re.len();
    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    // Swap lane pairs (0↔1, 2↔3, 4↔5, 6↔7): output = (1,0, 3,2, 5,4, 7,6).
    // _mm512_set_epi64 HIGH-to-LOW: arg0=lane7, arg7=lane0.
    let perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);

    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & ctrl_mask) != ctrl_mask {
            block_base += LANES_SOA;
            continue;
        }
        // SAFETY: block_base + LANES_SOA ≤ len; len is a power of two,
        // loop steps by LANES_SOA, and the last valid block_base is len - LANES_SOA.
        let zr = _mm512_loadu_pd(re_ptr.add(block_base));
        let zi = _mm512_loadu_pd(im_ptr.add(block_base));
        _mm512_storeu_pd(re_ptr.add(block_base), _mm512_permutexvar_pd(perm_idx, zr));
        _mm512_storeu_pd(im_ptr.add(block_base), _mm512_permutexvar_pd(perm_idx, zi));
        block_base += LANES_SOA;
    }
}

/// Toffoli Tier-B.1 for SoA: `target=1`, in-register permute swap.
///
/// For `t=1` the target bit is bit 1 of the amplitude index.  Swap
/// pairs `(0↔2)`, `(1↔3)`, `(4↔6)`, `(5↔7)`.
///
/// **Permute index (lane-0-first):** `(2,3, 0,1, 6,7, 4,5)`.
/// `_mm512_set_epi64` HIGH-to-LOW: `_mm512_set_epi64(5,4, 7,6, 1,0, 3,2)`.
///
/// # Safety  (same as Tier-B.0 but target is 1)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_b1_soa(
    re: &mut [f64],
    im: &mut [f64],
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;

    debug_assert!(
        sorted_controls.iter().all(|&c| c >= 3),
        "SoA Tier-B.1 contract: every control must be at qubit index >= LANES_SOA_BITS = 3"
    );

    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = re.len();
    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    // Swap pairs (0↔2, 1↔3, 4↔6, 5↔7): output = (2,3, 0,1, 6,7, 4,5).
    // _mm512_set_epi64 HIGH-to-LOW: arg0=lane7, arg7=lane0.
    let perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & ctrl_mask) != ctrl_mask {
            block_base += LANES_SOA;
            continue;
        }
        // SAFETY: block_base + LANES_SOA ≤ len (same as Tier-B.0).
        let zr = _mm512_loadu_pd(re_ptr.add(block_base));
        let zi = _mm512_loadu_pd(im_ptr.add(block_base));
        _mm512_storeu_pd(re_ptr.add(block_base), _mm512_permutexvar_pd(perm_idx, zr));
        _mm512_storeu_pd(im_ptr.add(block_base), _mm512_permutexvar_pd(perm_idx, zi));
        block_base += LANES_SOA;
    }
}

/// Toffoli Tier-B.2 for SoA: `target=2`, in-register cross-256 swap.
///
/// For `t=2` the target bit is bit 2 of the amplitude index.  Swap the
/// low 4 lanes with the high 4 lanes: `(0↔4)`, `(1↔5)`, `(2↔6)`, `(3↔7)`.
///
/// **Permute index (lane-0-first):** `(4,5,6,7, 0,1,2,3)`.
/// `_mm512_set_epi64` HIGH-to-LOW: `_mm512_set_epi64(3,2,1,0, 7,6,5,4)`.
///
/// # Safety  (same as Tier-B.0 but target is 2)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_b2_soa(
    re: &mut [f64],
    im: &mut [f64],
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;

    debug_assert!(
        sorted_controls.iter().all(|&c| c >= 3),
        "SoA Tier-B.2 contract: every control must be at qubit index >= LANES_SOA_BITS = 3"
    );

    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = re.len();
    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    // Swap halves (0↔4, 1↔5, 2↔6, 3↔7): output = (4,5,6,7, 0,1,2,3).
    // _mm512_set_epi64 HIGH-to-LOW: arg0=lane7, arg7=lane0.
    let perm_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & ctrl_mask) != ctrl_mask {
            block_base += LANES_SOA;
            continue;
        }
        // SAFETY: block_base + LANES_SOA ≤ len (same as Tier-B.0).
        let zr = _mm512_loadu_pd(re_ptr.add(block_base));
        let zi = _mm512_loadu_pd(im_ptr.add(block_base));
        _mm512_storeu_pd(re_ptr.add(block_base), _mm512_permutexvar_pd(perm_idx, zr));
        _mm512_storeu_pd(im_ptr.add(block_base), _mm512_permutexvar_pd(perm_idx, zi));
        block_base += LANES_SOA;
    }
}

/// CCZ Tier-A for SoA: all mask bits ≥ LANES_SOA_BITS=3.  Sign-flips the
/// amplitude at every index where the full mask is set.  Applied
/// independently to both `re` and `im` streams via `_mm512_xor_pd` with
/// the IEEE-754 sign-bit mask.
///
/// Mirror of `aos::apply_ccz_avx512_tier_a` with LANES_SOA=8.  Because
/// each lane is one f64 (not one Complex), the per-block mask check is
/// simpler: the full `mask` is either entirely set in `block_base` or
/// not — no intra-block ambiguity possible when `mask_lo >= 3`.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * All entries of `mask_bits` are distinct, < n.
/// * `mask_bits[0]` (the minimum) is ≥ LANES_SOA_BITS = 3.
/// * `re.len() == im.len() == 1 << n` for some n ≥ 4 (circuit invariant).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ccz_avx512_tier_a_soa(re: &mut [f64], im: &mut [f64], mask_bits: &[u32]) {
    use std::arch::x86_64::*;

    debug_assert!(
        mask_bits.iter().min().copied().unwrap_or(0) >= 3,
        "SoA CCZ Tier-A contract: every mask bit must be >= LANES_SOA_BITS (3)"
    );

    let mut mask: usize = 0;
    for &b in mask_bits {
        mask |= 1usize << b;
    }

    let len = re.len();
    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    // IEEE-754: XOR with -0.0 sign mask flips the sign bit of every f64 lane.
    let sign_mask = _mm512_set1_pd(-0.0_f64);

    let count = len / LANES_SOA;
    crate::kernels::par_blocks(
        count,
        len,
        |k| k * LANES_SOA,
        |block_base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            if (block_base & mask) != mask {
                return;
            }
            // SAFETY: block_base + LANES_SOA ≤ len because mask_lo ≥ 3 →
            // mask_lo_bit ≥ 8 = LANES_SOA; matching block_base values are
            // spaced ≥ LANES_SOA apart.  State vector is 1<<n ≥ 16 (n≥4).
            let pr = re_ptr.add(block_base);
            let pi = im_ptr.add(block_base);
            let zr = _mm512_loadu_pd(pr);
            let zi = _mm512_loadu_pd(pi);
            _mm512_storeu_pd(pr, _mm512_xor_pd(zr, sign_mask));
            _mm512_storeu_pd(pi, _mm512_xor_pd(zi, sign_mask));
        },
    );
}

/// CCZ Tier-A outer-walk for SoA: handles mask bits below LANES_SOA_BITS=3.
///
/// For each 8-amp block, the "high" mask bits (≥ 3) are checked at
/// `block_base` level (uniform within block).  The "low" mask bits (< 3)
/// drive a per-lane blend inside the block: for amp `k ∈ {0..8}`, flip
/// iff `(k & mask_low) == mask_low`.
///
/// **SoA simplification vs AoS.** In SoA each lane IS one amplitude's
/// `re` (or `im`) component — no AoS "doubling" of lane bits.  The
/// `lane_mask` is therefore exactly the bitfield of matching `k` values
/// in `{0..8}`, with no interleaving.  The same `lane_mask` applies to
/// both the `re` and `im` streams.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * All entries of `mask_bits` are distinct, < n.
/// * `re.len() == im.len() == 1 << n` for some n ≥ 4 (circuit invariant).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ccz_avx512_tier_a_outer_walk_soa(
    re: &mut [f64],
    im: &mut [f64],
    mask_bits: &[u32],
) {
    use std::arch::x86_64::*;

    const LANES_SOA_BITS: u32 = 3; // log2(LANES_SOA)

    // Partition mask bits into low (< 3) and high (≥ 3).
    let mut mask_low: usize = 0;
    let mut mask_high: usize = 0;
    for &b in mask_bits {
        if b < LANES_SOA_BITS {
            mask_low |= 1usize << b;
        } else {
            mask_high |= 1usize << b;
        }
    }

    // Per-block lane mask: for amp k in {0..8}, flip iff (k & mask_low) == mask_low.
    // In SoA each lane position = one amplitude component (no re/im doubling).
    let lane_mask: u8 = {
        let mut m = 0u8;
        for k in 0..8u32 {
            if (k as usize & mask_low) == mask_low {
                m |= 1 << k;
            }
        }
        m
    };

    let len = re.len();
    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    let sign = _mm512_set1_pd(-0.0_f64);

    let count = len / LANES_SOA;
    crate::kernels::par_blocks(
        count,
        len,
        |k| k * LANES_SOA,
        |block_base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            if (block_base & mask_high) != mask_high {
                return;
            }
            // SAFETY: block_base + LANES_SOA ≤ len; state vector is 1<<n with
            // n ≥ 4 and block_base is always LANES_SOA-aligned.
            let pr = re_ptr.add(block_base);
            let pi = im_ptr.add(block_base);
            let zr = _mm512_loadu_pd(pr);
            let zi = _mm512_loadu_pd(pi);
            let neg_r = _mm512_xor_pd(zr, sign);
            let neg_i = _mm512_xor_pd(zi, sign);
            // _mm512_mask_blend_pd(mask, a, b): selects b where bit=1, a where bit=0.
            _mm512_storeu_pd(pr, _mm512_mask_blend_pd(lane_mask, zr, neg_r));
            _mm512_storeu_pd(pi, _mm512_mask_blend_pd(lane_mask, zi, neg_i));
        },
    );
}

/// Routes Toffoli to the best available SoA tier.  Mirror of
/// `aos::dispatch_toffoli` with LANES_SOA=8 → `LANES_SOA_BITS=3`.
///
/// Tier-A (clean): every control strictly above target AND `target >= 3`.
/// Tier-A (outer-walk): `target >= 3` but some control at or below target.
/// Tier-B.0/1/2: `target ∈ {0,1,2}` AND every control ≥ 3.
/// Tier-C (scalar): all other cases.
fn dispatch_toffoli_soa(re: &mut [f64], im: &mut [f64], targets: [u32; 3], controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        const LANES_SOA_BITS: u32 = 3; // log2(LANES_SOA) where LANES_SOA = 8
        let t = targets[2];

        // Merge the CCX's inner control pair with any external controls.
        let mut all_ctrls: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
        all_ctrls.push(targets[0]);
        all_ctrls.push(targets[1]);
        for &c in controls {
            all_ctrls.push(c);
        }
        all_ctrls.sort();
        let c_lo = all_ctrls[0];

        if std::is_x86_feature_detected!("avx512f") {
            if t >= LANES_SOA_BITS {
                if c_lo > t {
                    // Tier-A clean: every control strictly above target.
                    // SAFETY: AVX-512F detected, target ≥ 3 (LANES_SOA_BITS),
                    // every control in all_ctrls is strictly above target.
                    unsafe {
                        apply_toffoli_avx512_tier_a_soa(re, im, t, &all_ctrls);
                    }
                    return;
                }
                if c_lo >= LANES_SOA_BITS {
                    // Tier-A outer-walk: controls at-or-above LANES_SOA_BITS=3
                    // but some lie between LANES_SOA_BITS and target. The flat
                    // block-stride walk has block_base advancing by LANES_SOA=8,
                    // so its low LANES_SOA_BITS bits are always zero; any control
                    // bit below LANES_SOA_BITS would silently disable the gate
                    // (the mask test could never pass). We REQUIRE c_lo >=
                    // LANES_SOA_BITS before invoking the SIMD kernel; sub-LANES
                    // controls fall through to scalar.
                    // SAFETY: AVX-512F detected, target ≥ 3, every control
                    // ≥ LANES_SOA_BITS; qubits distinct + in-range by invariant.
                    unsafe {
                        apply_toffoli_avx512_tier_a_outer_walk_soa(re, im, t, &all_ctrls);
                    }
                    return;
                }
                // c_lo < LANES_SOA_BITS: SIMD contract violated, fall through to scalar.
            }
            if t == 0 && c_lo >= LANES_SOA_BITS {
                // Tier-B.0: target=0, in-zmm permute swap.
                // SAFETY: AVX-512F detected, target=0 (implicit), every
                // control ≥ LANES_SOA_BITS=3 (c_lo >= 3).
                unsafe {
                    apply_toffoli_avx512_tier_b0_soa(re, im, &all_ctrls);
                }
                return;
            }
            if t == 1 && c_lo >= LANES_SOA_BITS {
                // Tier-B.1: target=1, in-zmm permute swap.
                // SAFETY: AVX-512F detected, target=1 (implicit), every
                // control ≥ LANES_SOA_BITS=3 (c_lo >= 3).
                unsafe {
                    apply_toffoli_avx512_tier_b1_soa(re, im, &all_ctrls);
                }
                return;
            }
            if t == 2 && c_lo >= LANES_SOA_BITS {
                // Tier-B.2: target=2, cross-256 in-zmm permute swap.
                // SAFETY: AVX-512F detected, target=2 (implicit), every
                // control ≥ LANES_SOA_BITS=3 (c_lo >= 3).
                unsafe {
                    apply_toffoli_avx512_tier_b2_soa(re, im, &all_ctrls);
                }
                return;
            }
        }
    }
    apply_toffoli_scalar_soa(re, im, targets, controls);
}

/// Routes CCZ to the best available SoA tier.  Mirror of
/// `aos::dispatch_ccz` with LANES_SOA=8 → `LANES_SOA_BITS=3`.
fn dispatch_ccz_soa(re: &mut [f64], im: &mut [f64], targets: [u32; 3], controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        const LANES_SOA_BITS: u32 = 3; // log2(LANES_SOA) where LANES_SOA = 8

        // Build sorted combined mask of all qubit positions.
        let mut all_mask: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
        for &q in &targets {
            all_mask.push(q);
        }
        for &c in controls {
            all_mask.push(c);
        }
        all_mask.sort();
        let mask_lo = all_mask[0];

        if std::is_x86_feature_detected!("avx512f") {
            if mask_lo >= LANES_SOA_BITS {
                // Tier-A clean: every mask bit ≥ LANES_SOA_BITS=3, per-block
                // mask check is uniform — no intra-block ambiguity.
                // SAFETY: AVX-512F detected, mask_lo ≥ 3, all qubit positions
                // distinct + in-range (Circuit invariant), n ≥ 4.
                unsafe {
                    apply_ccz_avx512_tier_a_soa(re, im, &all_mask);
                }
            } else {
                // Tier-A outer-walk: some mask bit < LANES_SOA_BITS=3, so
                // per-lane blend inside each block.
                // SAFETY: AVX-512F detected, all qubit positions distinct +
                // in-range (Circuit invariant), n ≥ 4.
                unsafe {
                    apply_ccz_avx512_tier_a_outer_walk_soa(re, im, &all_mask);
                }
            }
            return;
        }
    }
    apply_ccz_scalar_soa(re, im, targets, controls);
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

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    let inner_walk = |outer: usize| {
        let re_ptr = re_bp.ptr();
        let im_ptr = im_bp.ptr();
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
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        inner_walk,
    );
}

/// Packed AVX-512 CNOT specialisation over paired SoA storage — Tier B
/// (`target ∈ {0, 1, 2}` AND `1 << control >= LANES_SOA = 8`).
///
/// In Tier A the `target` bit sits ≥ LANES_SOA, so a swap-pair touches
/// two disjoint LANES_SOA-wide windows on each stream.  In Tier B the
/// target bit sits *inside* one LANES_SOA-wide window — a swap-pair
/// lives entirely within one zmm register per stream.  A single load +
/// `vpermutexvar_pd` + store on each of `re` and `im` swaps the
/// matching doubles in place.  Per LANES_SOA amps: 2 loads + 2 permutes
/// + 2 stores ≈ 6 µops; zero multiplies, bandwidth-bound.
///
/// **Permute-index tables** (8 doubles per zmm; each amp = 1 double per
/// stream — SoA, NOT AoS).  Position `p` in the zmm corresponds to amp
/// `outer + j + p`, so positions vary by bits `[0, 3)` (the low 3 bits
/// of `j + p`).  Output position → source index in the same zmm:
/// * `target = 0` (bit 0 varies within zmm) — swap pairs (0↔1),
///   (2↔3), (4↔5), (6↔7) → `[1, 0, 3, 2, 5, 4, 7, 6]`
/// * `target = 1` (bit 1 varies) — swap pairs (0↔2), (1↔3), (4↔6),
///   (5↔7) → `[2, 3, 0, 1, 6, 7, 4, 5]`
/// * `target = 2` (bit 2 varies) — swap halves
///   → `[4, 5, 6, 7, 0, 1, 2, 3]`
///
/// The SAME permute index applies independently to the `re` and `im`
/// streams — both have identical bit-layout per amp.
///
/// **Outer-walk.** Same renormalise-then-shift pattern as the AoS Tier
/// B CNOT.  The inner SIMD walk owns bits `[0, control)` via `j` (which
/// steps by LANES_SOA, so `j`'s low 3 bits are zero — the in-register
/// permute handles those swaps).  The outer walk reserves bits
/// `[0, control]` and injects every external control as `fixed=true`
/// (renormalised by `-(control + 1)`) in the above-control subspace,
/// then ORs in `c_bit` to pin the control bit.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `target ∈ {0, 1, 2}` (i.e. `1 << target < LANES_SOA = 8`).
/// * `1 << control >= LANES_SOA` (= 8) — the inner SIMD walk has ≥
///   LANES_SOA contiguous amps per stream per outer block.
/// * Every external control's qubit index is strictly greater than
///   `max(control, target) = control`, so the renormalisation
///   subtraction is safe and external controls land above the
///   reserved span.
/// * Distinct + in-range qubits; `re.len() == im.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cnot_avx512_tier_b(
    re: &mut [f64],
    im: &mut [f64],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let c_bit = 1usize << control;
    let len = re.len();

    debug_assert!(target <= 2, "Tier B requires target ∈ {{0, 1, 2}}");
    debug_assert!(
        c_bit >= LANES_SOA,
        "c_bit < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > control),
        "external control at-or-below control: dispatch contract violated"
    );

    // SAFETY: permute indices derived above.  `_mm512_setr_epi64` is an
    // immediate-builder, no UB possible.
    let permute_idx = match target {
        0 => _mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6),
        1 => _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5),
        2 => _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3),
        _ => unreachable!(),
    };

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    // Outer-walk: reserve bits `[0, control]` for the inner walk + the
    // control bit itself, inject every external control as `fixed=true`
    // (renormalised by `-(control + 1)`) in the above-control subspace,
    // then shift up by `control + 1` and OR in `c_bit` to pin the
    // control bit.  Subtractions are safe by the safety contract.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - control - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - control - 1 - external_controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (control + 1),
        |base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            let outer = base | c_bit;
            // outer has: bit control = 1, every external_control = 1, bits
            // [0, control) all zero.  Bit `target` is in [0, control) and
            // is therefore 0 in outer — the LANES_SOA-wide load picks up
            // amps with mixed target-bit values (j enumerates bits
            // [0, control) and the in-register permute swaps matching pairs).
            let mut j = 0usize;
            while j + LANES_SOA <= c_bit {
                // SAFETY: bit-disjointness — `outer`'s bits ≥ control+1
                // (with control set), `j` ⊆ [0, c_bit).  Disjoint, so
                // `i + LANES_SOA ≤ outer + c_bit ≤ len`.
                let i = outer | j;
                let zr = _mm512_loadu_pd(re_ptr.add(i));
                let zi = _mm512_loadu_pd(im_ptr.add(i));
                _mm512_storeu_pd(re_ptr.add(i), _mm512_permutexvar_pd(permute_idx, zr));
                _mm512_storeu_pd(im_ptr.add(i), _mm512_permutexvar_pd(permute_idx, zi));
                j += LANES_SOA;
            }
            debug_assert_eq!(j, c_bit);
        },
    );
}

/// Packed AVX-512 CNOT specialisation over paired SoA storage — Tier C
/// (both `control` and `target` in `{0, 1, 2}`).
///
/// Both qubits' bits fit inside a single LANES_SOA-wide window (8 amps
/// per stream).  One load + `vpermutexvar_pd` + store per stream
/// effects the CNOT.  No inner walk needed.  There is a third
/// "irrelevant" bit (the one of `{0, 1, 2}` neither control nor target)
/// that varies across positions but is preserved by the permute — it
/// labels which two CNOT events live in the same zmm.
///
/// **Permute-index tables** (8 doubles per zmm; bits b0, b1, b2 of the
/// position label = state-vector bits 0, 1, 2 within the window).
/// Output position → source position:
/// * `(c=0, t=1)` — flip b1 when b0=1: positions (1, 3) ↔ (1+2, 3+2)
///   plus offset 4 → `[0, 3, 2, 1, 4, 7, 6, 5]`
/// * `(c=1, t=0)` — flip b0 when b1=1: positions 2 ↔ 3, 6 ↔ 7
///   → `[0, 1, 3, 2, 4, 5, 7, 6]`
/// * `(c=0, t=2)` — flip b2 when b0=1: 1 ↔ 5, 3 ↔ 7
///   → `[0, 5, 2, 7, 4, 1, 6, 3]`
/// * `(c=2, t=0)` — flip b0 when b2=1: 4 ↔ 5, 6 ↔ 7
///   → `[0, 1, 2, 3, 5, 4, 7, 6]`
/// * `(c=1, t=2)` — flip b2 when b1=1: 2 ↔ 6, 3 ↔ 7
///   → `[0, 1, 6, 7, 4, 5, 2, 3]`
/// * `(c=2, t=1)` — flip b1 when b2=1: 4 ↔ 6, 5 ↔ 7
///   → `[0, 1, 2, 3, 6, 7, 4, 5]`
///
/// SAME permute applied independently to `re` and `im` streams.
///
/// **Outer-walk.** External controls all sit above the 8-amp window
/// (positions ≥ 3) by safety contract.  Each is renormalised by `-3`
/// and injected as `fixed=true`; free positions ≥ 3 are enumerated by
/// `k`.  The resulting `base` — shifted left by 3 — has bits 0, 1, 2
/// zero and every external control bit set.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * Both `control` and `target` ∈ `{0, 1, 2}`, distinct.
/// * Every external control's qubit index is strictly greater than 2
///   (so the renormalisation by `-3` is safe and external controls
///   land above the in-register window).
/// * `re.len() == im.len() >= 8` (at least one 8-amp window per stream).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cnot_avx512_tier_c(
    re: &mut [f64],
    im: &mut [f64],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let len = re.len();

    debug_assert!(
        control <= 2 && target <= 2 && control != target,
        "Tier C requires distinct control,target ∈ {{0, 1, 2}}"
    );
    debug_assert!(
        len >= LANES_SOA,
        "len < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > 2),
        "external control at-or-below 2: dispatch contract violated"
    );

    let permute_idx = match (control, target) {
        (0, 1) => _mm512_setr_epi64(0, 3, 2, 1, 4, 7, 6, 5),
        (1, 0) => _mm512_setr_epi64(0, 1, 3, 2, 4, 5, 7, 6),
        (0, 2) => _mm512_setr_epi64(0, 5, 2, 7, 4, 1, 6, 3),
        (2, 0) => _mm512_setr_epi64(0, 1, 2, 3, 5, 4, 7, 6),
        (1, 2) => _mm512_setr_epi64(0, 1, 6, 7, 4, 5, 2, 3),
        (2, 1) => _mm512_setr_epi64(0, 1, 2, 3, 6, 7, 4, 5),
        _ => unreachable!("Tier C requires distinct control,target ∈ {{0, 1, 2}}"),
    };

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    // Outer-walk: reserve bits [0, 3) for the in-register 8-amp window,
    // inject every external control (renormalised by -3) as `fixed=true`
    // in the above-window subspace, then shift up by 3 to land on an
    // LANES_SOA-aligned boundary.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - 3, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - 3 - external_controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << 3,
        |base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            debug_assert_eq!(base & 7, 0);
            // SAFETY: bit-disjointness — `base` has bits 0,1,2 = 0 and
            // base + LANES_SOA ≤ len (one zmm load per stream).
            let zr = _mm512_loadu_pd(re_ptr.add(base));
            let zi = _mm512_loadu_pd(im_ptr.add(base));
            _mm512_storeu_pd(re_ptr.add(base), _mm512_permutexvar_pd(permute_idx, zr));
            _mm512_storeu_pd(im_ptr.add(base), _mm512_permutexvar_pd(permute_idx, zi));
        },
    );
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

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    let inner_walk = |outer: usize| {
        let re_ptr = re_bp.ptr();
        let im_ptr = im_bp.ptr();
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
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1),
        inner_walk,
    );
}

/// Packed AVX-512 SWAP specialisation over paired SoA storage — Tier B
/// (`min(targets) ∈ {0, 1, 2}` AND `1 << max(targets) >= LANES_SOA = 8`).
///
/// In Tier A the lo-target bit sits ≥ LANES_SOA, so a swap-pair touches
/// two disjoint LANES_SOA-wide windows.  In Tier B the lo bit sits
/// *inside* one LANES_SOA-wide window — and the hi bit selects between
/// two adjacent windows.  A swap-pair therefore spans both windows
/// (one half lives in the `hi = 0` zmm, the other in the `hi = 1`
/// zmm), so a single `vpermutex2var_pd` per output zmm shuffles the
/// matching doubles between the two inputs.  Per LANES_SOA amps:
/// 4 loads + 4 permutes + 4 stores per pair of windows; zero
/// multiplies.  Same shape applied independently to `re` and `im`.
///
/// **Permute-index tables** (8 doubles per zmm; doubles `[0, 7]` =
/// hi=0 input, doubles `[8, 15]` = hi=1 input; output position →
/// source index).  Within a LANES_SOA-wide zmm position `p`
/// corresponds to amp `outer + j + p`, so the within-zmm bits are
/// `[0, 3)`:
/// * `lo = 0` — bit 0 varies; the (lo=1, hi=0) amps at positions
///   1, 3, 5, 7 swap with the (lo=0, hi=1) amps at hi=1 positions
///   0, 2, 4, 6.
///   * `idx_for_hi0 = [0, 8, 2, 10, 4, 12, 6, 14]`
///   * `idx_for_hi1 = [1, 9, 3, 11, 5, 13, 7, 15]`
/// * `lo = 1` — bit 1 varies; (lo=1, hi=0) at 2, 3, 6, 7 swap with
///   (lo=0, hi=1) at hi=1 positions 0, 1, 4, 5.
///   * `idx_for_hi0 = [0, 1, 8, 9, 4, 5, 12, 13]`
///   * `idx_for_hi1 = [2, 3, 10, 11, 6, 7, 14, 15]`
/// * `lo = 2` — bit 2 varies; (lo=1, hi=0) at 4, 5, 6, 7 swap with
///   (lo=0, hi=1) at hi=1 positions 0, 1, 2, 3.
///   * `idx_for_hi0 = [0, 1, 2, 3, 8, 9, 10, 11]`
///   * `idx_for_hi1 = [4, 5, 6, 7, 12, 13, 14, 15]`
///
/// **Outer-walk.** Same renormalise-then-shift idiom as the AoS Tier
/// B SWAP.  The inner walk owns bits `[0, hi)` via `j` (which steps
/// by LANES_SOA) and the choice between the two halves of the swap
/// pair.  The outer walk reserves bits `[0, hi]` and injects every
/// external control as `fixed=true` (renormalised by `-(hi + 1)`)
/// in the above-hi subspace, then shifts up by `hi + 1`.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `min(targets) ∈ {0, 1, 2}` (i.e. `1 << lo < LANES_SOA = 8`).
/// * `1 << max(targets) >= LANES_SOA` (= 8) — the inner SIMD walk
///   has ≥ LANES_SOA contiguous amps per zmm half on each stream.
/// * Distinct targets, both in qubit range.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the renormalisation subtraction is safe.
/// * `re.len() == im.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_swap_avx512_tier_b(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let hi_bit = 1usize << hi;
    let len = re.len();

    debug_assert!(
        lo != hi,
        "SWAP requires distinct targets: dispatch contract violated"
    );
    debug_assert!(lo <= 2, "Tier B requires lo ∈ {{0, 1, 2}}");
    debug_assert!(
        hi_bit >= LANES_SOA,
        "hi_bit < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    // SAFETY: permute indices derived in the doc comment above.
    // `_mm512_setr_epi64` is an immediate-builder, no UB possible.
    let (idx_for_hi0, idx_for_hi1) = match lo {
        0 => (
            _mm512_setr_epi64(0, 8, 2, 10, 4, 12, 6, 14),
            _mm512_setr_epi64(1, 9, 3, 11, 5, 13, 7, 15),
        ),
        1 => (
            _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13),
            _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15),
        ),
        2 => (
            _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11),
            _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15),
        ),
        _ => unreachable!("Tier B requires lo ∈ {{0, 1, 2}}"),
    };

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    // Outer-walk: reserve bits `[0, hi]` for the inner walk + the
    // hi-bit half-selector, inject every external control as
    // `fixed=true` in the above-hi subspace, then shift up by `hi + 1`.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - hi - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - hi - 1 - external_controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (hi + 1),
        |outer| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            // outer has: bit hi = 0, every external_control bit set,
            // bits [0, hi) all zero.  OR in `j` (step over [0, hi)) and
            // either 0 or `hi_bit` to select between the two zmm halves.
            let mut j = 0usize;
            while j + LANES_SOA <= hi_bit {
                // SAFETY: bit-disjointness — `outer`'s bits ≥ hi+1 (with
                // hi cleared and ec bits set), `j` ⊆ [0, hi_bit), `hi_bit`
                // is bit hi.  Pairwise disjoint, so each LANES_SOA-wide
                // load + store stays within `len`.
                let i_0 = outer | j;
                let i_1 = i_0 | hi_bit;
                let zr0 = _mm512_loadu_pd(re_ptr.add(i_0));
                let zr1 = _mm512_loadu_pd(re_ptr.add(i_1));
                let zi0 = _mm512_loadu_pd(im_ptr.add(i_0));
                let zi1 = _mm512_loadu_pd(im_ptr.add(i_1));
                let new_zr0 = _mm512_permutex2var_pd(zr0, idx_for_hi0, zr1);
                let new_zr1 = _mm512_permutex2var_pd(zr0, idx_for_hi1, zr1);
                let new_zi0 = _mm512_permutex2var_pd(zi0, idx_for_hi0, zi1);
                let new_zi1 = _mm512_permutex2var_pd(zi0, idx_for_hi1, zi1);
                _mm512_storeu_pd(re_ptr.add(i_0), new_zr0);
                _mm512_storeu_pd(re_ptr.add(i_1), new_zr1);
                _mm512_storeu_pd(im_ptr.add(i_0), new_zi0);
                _mm512_storeu_pd(im_ptr.add(i_1), new_zi1);
                j += LANES_SOA;
            }
            debug_assert_eq!(j, hi_bit);
        },
    );
}

/// Packed AVX-512 SWAP specialisation over paired SoA storage — Tier C
/// (both targets in `{0, 1, 2}`).
///
/// Both qubits' bits fit inside a single LANES_SOA-wide window (8 amps
/// per stream).  One load + `vpermutexvar_pd` + store per stream
/// effects the SWAP by exchanging the (lo=1, hi=0) amps with the
/// (lo=0, hi=1) amps inside the window.  The third "irrelevant" bit
/// (the one of `{0, 1, 2}` not in `targets`) labels which two SWAP
/// events live in the same zmm.
///
/// **Permute-index tables** (output position → source position).  Same
/// permute applied to both `re` and `im` streams (SoA: identical
/// bit-layout per amp on each stream).
/// * targets `{0, 1}`: swap (b0=1, b1=0) ↔ (b0=0, b1=1) — positions
///   (1 ↔ 2) and (5 ↔ 6) → `[0, 2, 1, 3, 4, 6, 5, 7]`
/// * targets `{0, 2}`: swap (b0=1, b2=0) ↔ (b0=0, b2=1) — positions
///   (1 ↔ 4) and (3 ↔ 6) → `[0, 4, 2, 6, 1, 5, 3, 7]`
/// * targets `{1, 2}`: swap (b1=1, b2=0) ↔ (b1=0, b2=1) — positions
///   (2 ↔ 4) and (3 ↔ 5) → `[0, 1, 4, 5, 2, 3, 6, 7]`
///
/// SWAP is symmetric in its targets, so the same permute serves both
/// orientations of `targets[0]`/`targets[1]`.
///
/// **Outer-walk.** External controls all sit above the 8-amp window
/// (positions ≥ 3) by safety contract.  Each is renormalised by `-3`
/// and injected as `fixed=true`; free positions ≥ 3 are enumerated by
/// `k`.  The resulting `base` — shifted left by 3 — has bits 0, 1, 2
/// zero and every external control bit set.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * Both targets in `{0, 1, 2}`, distinct.
/// * Every external control's qubit index is strictly greater than 2
///   (so the renormalisation by `-3` is safe).
/// * `re.len() == im.len() >= 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_swap_avx512_tier_c(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(re.len(), im.len());
    let len = re.len();
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);

    debug_assert!(
        lo <= 2 && hi <= 2 && lo != hi,
        "Tier C requires distinct targets in {{0, 1, 2}}"
    );
    debug_assert!(
        len >= LANES_SOA,
        "len < LANES_SOA: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > 2),
        "external control at-or-below 2: dispatch contract violated"
    );

    let permute_idx = match (lo, hi) {
        (0, 1) => _mm512_setr_epi64(0, 2, 1, 3, 4, 6, 5, 7),
        (0, 2) => _mm512_setr_epi64(0, 4, 2, 6, 1, 5, 3, 7),
        (1, 2) => _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7),
        _ => unreachable!("Tier C requires distinct targets in {{0, 1, 2}}"),
    };

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    // Outer-walk: reserve bits [0, 3), inject ec (renormalised by -3)
    // as `fixed=true`, shift up by 3 to land on an LANES_SOA-aligned
    // boundary with every external control set.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - 3, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - 3 - external_controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << 3,
        |base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            debug_assert_eq!(base & 7, 0);
            // SAFETY: bit-disjointness — `base` has bits 0,1,2 = 0 and
            // base + LANES_SOA ≤ len (one zmm load per stream).
            let zr = _mm512_loadu_pd(re_ptr.add(base));
            let zi = _mm512_loadu_pd(im_ptr.add(base));
            _mm512_storeu_pd(re_ptr.add(base), _mm512_permutexvar_pd(permute_idx, zr));
            _mm512_storeu_pd(im_ptr.add(base), _mm512_permutexvar_pd(permute_idx, zi));
        },
    );
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

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());
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
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1),
        |base| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
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
        },
    );
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

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());

    let multiply_block = |base: usize, d_k: Complex| {
        let re_ptr = re_bp.ptr();
        let im_ptr = im_bp.ptr();
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
            //        = -(m * d_im) + (r * d_re)   via fnmadd(a,b,c) = c - a*b
            let new_r = _mm512_fnmadd_pd(m, d_im_bc, _mm512_mul_pd(r, d_re_bc));
            // new_im = r * d_im + m * d_re
            //        = (m * d_re) + (r * d_im)    via fmadd(a,b,c) = a*b + c
            let new_m = _mm512_fmadd_pd(m, d_re_bc, _mm512_mul_pd(r, d_im_bc));
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
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1),
        |base| {
            // base has: bit hi = 0, every external_control bit = 1, bits
            // [0, lo] all zero.  Iterate the 4 sub-blocks:
            multiply_block(base, d_for_hi0_lo0); // (q_hi=0, q_lo=0)
            multiply_block(base | lo_bit, d_for_hi0_lo1); // (q_hi=0, q_lo=1)
            multiply_block(base | hi_bit, d_for_hi1_lo0); // (q_hi=1, q_lo=0)
            multiply_block(base | hi_bit | lo_bit, d_for_hi1_lo1); // (q_hi=1, q_lo=1)
        },
    );
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

/// Dispatch helper for SoA CNOT specialisations.  Routes to the
/// matching AVX-512 Tier A / B / C kernel when the host + qubit
/// orientation satisfies the safety contract; otherwise falls through
/// to the scalar specialised kernel.  Mirror of `aos::dispatch_cnot`
/// with the SoA's LANES_SOA = 8 → bigger in-zmm window
/// (`{0, 1, 2}` instead of `{0, 1}` for the inside-zmm slot).
///
/// Tier coverage:
/// * **Tier A** (`1 << target >= LANES_SOA` AND `control > target`):
///   classic LANES_SOA-block swap-pair across two disjoint windows on
///   each stream.
/// * **Tier B** (`target ∈ {0, 1, 2}` AND `1 << control >= LANES_SOA`):
///   in-register `vpermutexvar_pd` swap inside one LANES_SOA-wide
///   window per stream.
/// * **Tier C** (both `control, target ∈ {0, 1, 2}`): one
///   LANES_SOA-aligned 8-amp window per stream, one permute per
///   window.
///
/// Uncovered orientation: `target >= 3` AND `control < target`.
/// Falls through to scalar.
fn dispatch_cnot_soa(re: &mut [f64], im: &mut [f64], control: u32, target: u32, controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && controls.iter().all(|&c| c > target.max(control))
        {
            let t_bit = 1usize << target;
            let c_bit = 1usize << control;
            // Tier A: control > target AND t_bit ≥ LANES_SOA.
            if t_bit >= LANES_SOA && control > target {
                // SAFETY: Tier A contract — AVX-512F detected, t_bit ≥
                // LANES_SOA, control above target, every external
                // control above max(control, target).
                unsafe {
                    apply_2q_cnot_avx512(re, im, control, target, controls);
                }
                return;
            }
            // Tier B: target ∈ {0, 1, 2} AND c_bit ≥ LANES_SOA.
            if target <= 2 && c_bit >= LANES_SOA {
                // SAFETY: Tier B contract — AVX-512F detected, target ∈
                // {0, 1, 2}, c_bit ≥ LANES_SOA, every external control >
                // control = max(c, t).
                unsafe {
                    apply_2q_cnot_avx512_tier_b(re, im, control, target, controls);
                }
                return;
            }
            // Tier C: both control and target ∈ {0, 1, 2}.  Requires
            // len ≥ LANES_SOA (= 8) so a single zmm load covers the
            // 2-qubit subspace and any "extra" low bits.  Two distinct
            // qubits in {0, 1, 2} only guarantee n_qubits ≥ 2; if
            // n_qubits == 2 then len == 4 < LANES_SOA and the SIMD load
            // would OOB.  Fall through to scalar when len is too small.
            //
            // Additionally, the kernel renormalises external controls
            // by `(c - 3, true)`, which is sound only when every
            // external control sits strictly above the in-quartet
            // subspace, i.e. `c > 2`.  The outer-walk guard `c > max(
            // control, target)` admits ec=2 when both quartet qubits
            // are in {0, 1}, so we must add the tighter constraint
            // `c >= 3` here.  Without it, `2u32 - 3` underflows to
            // 0xFFFFFFFF and `expand_with_fixed`'s downstream shift
            // panics in debug / silently corrupts in release.
            if target <= 2
                && control <= 2
                && re.len() >= LANES_SOA
                && controls.iter().all(|&c| c >= 3)
            {
                // SAFETY: Tier C contract — AVX-512F detected, both ∈
                // {0, 1, 2}, every external control ≥ 3, len ≥
                // LANES_SOA.
                unsafe {
                    apply_2q_cnot_avx512_tier_c(re, im, control, target, controls);
                }
                return;
            }
            // Remaining case: target ≥ 3 AND control < target.  Falls
            // through to scalar.
        }
    }
    apply_2q_cnot_scalar(re, im, control, target, controls);
}

/// Dispatch helper for SoA SWAP.  Routes to the matching AVX-512
/// Tier A / B / C kernel when the host + qubit orientation satisfies
/// the safety contract; otherwise falls through to the scalar
/// specialised kernel.  Mirror of `aos::dispatch_swap` with
/// LANES_SOA = 8.
///
/// Tier coverage (SWAP is symmetric, so keyed on `lo = min(targets)`,
/// `hi = max(targets)`):
/// * **Tier A** (`1 << lo >= LANES_SOA`): classic LANES_SOA-block
///   swap-pair across two disjoint windows.
/// * **Tier B** (`lo ∈ {0, 1, 2}` AND `1 << hi >= LANES_SOA`): a
///   swap-pair spans two adjacent LANES_SOA-wide windows; one
///   `vpermutex2var_pd` per output zmm per stream.
/// * **Tier C** (both targets in `{0, 1, 2}`): both targets fit
///   inside one LANES_SOA-aligned 8-amp window per stream; a single
///   in-register permute per stream effects the SWAP.
fn dispatch_swap_soa(re: &mut [f64], im: &mut [f64], targets: [u32; 2], controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let lo = targets[0].min(targets[1]);
        let hi = targets[0].max(targets[1]);
        if std::is_x86_feature_detected!("avx512f") && controls.iter().all(|&c| c > hi) {
            let lo_bit = 1usize << lo;
            let hi_bit = 1usize << hi;
            if lo_bit >= LANES_SOA {
                // SAFETY: Tier A contract — AVX-512F detected, lo_bit ≥
                // LANES_SOA, targets distinct (lo ≠ hi by min/max of
                // distinct inputs), every external control > hi.
                unsafe {
                    apply_2q_swap_avx512(re, im, targets, controls);
                }
                return;
            }
            if hi_bit >= LANES_SOA && lo <= 2 {
                // SAFETY: Tier B contract — AVX-512F detected, lo ∈
                // {0, 1, 2}, hi_bit ≥ LANES_SOA, distinct targets,
                // every external control > hi.
                unsafe {
                    apply_2q_swap_avx512_tier_b(re, im, targets, controls);
                }
                return;
            }
            // Tier C: both targets in {0, 1, 2}.  Requires len ≥
            // LANES_SOA (= 8): n_qubits == 2 with targets {0, 1} gives
            // len == 4 < LANES_SOA and the SIMD load would OOB.  Fall
            // through to scalar when len is too small.
            //
            // Additionally, the kernel renormalises external controls
            // by `(c - 3, true)`; the outer-walk guard `c > hi` admits
            // ec=2 whenever hi ≤ 1, but the kernel's subtraction would
            // then underflow.  Tighten to `c >= 3` so the
            // renormalisation is safe across the full {0,1,2}-quartet
            // surface.
            if lo <= 2 && hi <= 2 && re.len() >= LANES_SOA && controls.iter().all(|&c| c >= 3) {
                // SAFETY: Tier C contract — AVX-512F detected, both
                // targets in {0, 1, 2}, distinct, every external
                // control ≥ 3, len ≥ LANES_SOA.
                unsafe {
                    apply_2q_swap_avx512_tier_c(re, im, targets, controls);
                }
                return;
            }
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

    // 1. Diagonal fast path (P1-06).
    if super::is_diagonal_2x2(m) {
        apply_1q_diagonal_soa(re, im, target, controls, m[0][0], m[1][1]);
        return;
    }

    // 2. Anti-diagonal fast path (P1-05). Per-arm dispatch picks
    // AVX-512 in T9/T10; T8 wires the scalar fallback chain.
    if super::is_antidiagonal_2x2(m) {
        match super::classify_1q_antidiag(m) {
            Some(super::Perm1qKind::X) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && controls.iter().all(|&c| c > target)
                    {
                        if (1usize << target) >= 8 {
                            // SAFETY: feature detected + Tier-A SoA contract.
                            unsafe {
                                apply_1q_x_soa_avx512(re, im, target, controls);
                            }
                            return;
                        } else if target <= 2
                            && re.len().is_multiple_of(8)
                            && controls.iter().all(|&c| c >= 3)
                        {
                            // SAFETY: feature detected + Tier-B SoA contract.
                            unsafe {
                                apply_1q_x_soa_avx512_lowbit(re, im, target, controls);
                            }
                            return;
                        }
                    }
                }
                apply_1q_x_soa_scalar(re, im, target, controls);
                return;
            }
            Some(super::Perm1qKind::YPos) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && (1usize << target) >= 8
                        && controls.iter().all(|&c| c > target)
                    {
                        // SAFETY: feature detected + Tier-A SoA contract.
                        unsafe {
                            apply_1q_y_soa_avx512(re, im, target, controls, 1.0);
                        }
                        return;
                    }
                }
                apply_1q_y_soa_scalar(re, im, target, controls, 1.0);
                return;
            }
            Some(super::Perm1qKind::YNeg) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && (1usize << target) >= 8
                        && controls.iter().all(|&c| c > target)
                    {
                        // SAFETY: feature detected + Tier-A SoA contract.
                        unsafe {
                            apply_1q_y_soa_avx512(re, im, target, controls, -1.0);
                        }
                        return;
                    }
                }
                apply_1q_y_soa_scalar(re, im, target, controls, -1.0);
                return;
            }
            None => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && (1usize << target) >= 8
                        && controls.iter().all(|&c| c > target)
                    {
                        // SAFETY: feature detected + Tier-A SoA contract.
                        unsafe {
                            apply_1q_antidiag_soa_avx512(
                                re, im, target, controls, m[0][1], m[1][0],
                            );
                        }
                        return;
                    }
                }
                apply_1q_antidiag_soa_scalar(re, im, target, controls, m[0][1], m[1][0]);
                return;
            }
        }
    }

    // 3. Generic 2×2 path (scalar with LLVM auto-vec; same as before).
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
            re[i] =
                m[0][0].re * a0_re - m[0][0].im * a0_im + m[0][1].re * a1_re - m[0][1].im * a1_im;
            im[i] =
                m[0][0].re * a0_im + m[0][0].im * a0_re + m[0][1].re * a1_im + m[0][1].im * a1_re;
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

/// Scalar Pauli-X kernel on SoA streams. Swaps both `re` and `im`
/// at index pairs `(i, i | (1 << target))` for every `i` with target
/// bit clear and every control bit set.
pub(crate) fn apply_1q_x_soa_scalar(re: &mut [f64], im: &mut [f64], target: u32, controls: &[u32]) {
    debug_assert_eq!(re.len(), im.len());
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            re.swap(i, j);
            im.swap(i, j);
        }
        i += 1;
    }
}

/// Scalar Pauli-Y kernel on SoA streams. Per pair `(i, j)`:
///   re[i], im[i] = phase_sign *  im[j], -phase_sign * re[j]
///   re[j], im[j] = -phase_sign * im[i_old], phase_sign * re[i_old]
/// `phase_sign = +1.0` is YPos (canonical); `-1.0` is YNeg.
pub(crate) fn apply_1q_y_soa_scalar(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    phase_sign: f64,
) {
    debug_assert_eq!(re.len(), im.len());
    debug_assert!(phase_sign == 1.0 || phase_sign == -1.0);
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let r0 = re[i];
            let i0 = im[i];
            let r1 = re[j];
            let i1 = im[j];
            re[i] = phase_sign * i1;
            im[i] = -phase_sign * r1;
            re[j] = -phase_sign * i0;
            im[j] = phase_sign * r0;
        }
        i += 1;
    }
}

/// Scalar generic anti-diagonal kernel on SoA. `m = [[0, a], [b, 0]]`.
/// `new[i] = a * amps[j]_old`, `new[j] = b * amps[i]_old`.
/// (Corrected form: `a` applies to `j` index, `b` applies to `i` index,
/// matching `m[0][1]` → row 0 col 1 and `m[1][0]` → row 1 col 0.)
pub(crate) fn apply_1q_antidiag_soa_scalar(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    a: Complex,
    b: Complex,
) {
    debug_assert_eq!(re.len(), im.len());
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let r0 = re[i];
            let i0 = im[i];
            let r1 = re[j];
            let i1 = im[j];
            // new[i] = a * (r1, i1)
            re[i] = a.re * r1 - a.im * i1;
            im[i] = a.re * i1 + a.im * r1;
            // new[j] = b * (r0, i0)
            re[j] = b.re * r0 - b.im * i0;
            im[j] = b.re * i0 + b.im * r0;
        }
        i += 1;
    }
}

/// AVX-512 Pauli-X on SoA streams (Tier A).
///
/// # Safety
/// * Host must support AVX-512F.
/// * `1usize << target ≥ LANES_SOA = 8`.
/// * Every control > target.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_x_soa_avx512(re: &mut [f64], im: &mut [f64], target: u32, controls: &[u32]) {
    use core::arch::x86_64::*;

    const LANES: usize = 8;
    let target_bit = 1usize << target;
    debug_assert!(target_bit >= LANES);
    debug_assert!(controls.iter().all(|&c| c > target));

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());
    let len = re.len();

    let outer_iter = |block: usize| {
        let re_ptr = re_bp.ptr();
        let im_ptr = im_bp.ptr();
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            let i1 = block | target_bit | j;
            // SAFETY: in-bounds by outer-walk construction.
            let r0 = _mm512_loadu_pd(re_ptr.add(i0));
            let r1 = _mm512_loadu_pd(re_ptr.add(i1));
            let m0 = _mm512_loadu_pd(im_ptr.add(i0));
            let m1 = _mm512_loadu_pd(im_ptr.add(i1));
            _mm512_storeu_pd(re_ptr.add(i0), r1);
            _mm512_storeu_pd(re_ptr.add(i1), r0);
            _mm512_storeu_pd(im_ptr.add(i0), m1);
            _mm512_storeu_pd(im_ptr.add(i1), m0);
            j += LANES;
        }
        debug_assert_eq!(j, target_bit);
    };

    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let count = len / outer_step;
        crate::kernels::par_blocks(count, len, |k| k * outer_step, outer_iter);
        return;
    }

    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        outer_iter,
    );
}

/// AVX-512 Pauli-Y on SoA streams (Tier A).
///
/// For YPos (phase_sign=+1):
///   re[i0_block] ←  im[i1_block]    im[i0_block] ← -re[i1_block]
///   re[i1_block] ← -im[i0_block]    im[i1_block] ←  re[i0_block]
///
/// Sign flip via `_mm512_xor_pd` with broadcast `-0.0` mask.
///
/// # Safety
/// Same as `apply_1q_x_soa_avx512`. `phase_sign ∈ {+1,-1}`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_y_soa_avx512(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    phase_sign: f64,
) {
    use core::arch::x86_64::*;

    const LANES: usize = 8;
    debug_assert!(phase_sign == 1.0 || phase_sign == -1.0);
    let target_bit = 1usize << target;
    debug_assert!(target_bit >= LANES);
    debug_assert!(controls.iter().all(|&c| c > target));

    let sign_mask = _mm512_set1_pd(-0.0f64); // toggles sign bit on xor
    let zero_mask = _mm512_set1_pd(0.0f64);
    // YPos pattern:
    //   new_re[i0] =  im[i1]   → no sign flip       (xor with zero_mask)
    //   new_im[i0] = -re[i1]   → sign flip          (xor with sign_mask)
    //   new_re[i1] = -im[i0]   → sign flip          (xor with sign_mask)
    //   new_im[i1] =  re[i0]   → no sign flip       (xor with zero_mask)
    // YNeg: every sign flips.
    let (mask_new_re_i0, mask_new_im_i0, mask_new_re_i1, mask_new_im_i1) = if phase_sign == 1.0 {
        (zero_mask, sign_mask, sign_mask, zero_mask)
    } else {
        (sign_mask, zero_mask, zero_mask, sign_mask)
    };

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());
    let len = re.len();

    let outer_iter = |block: usize| {
        let re_ptr = re_bp.ptr();
        let im_ptr = im_bp.ptr();
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            let i1 = block | target_bit | j;
            // SAFETY: in-bounds.
            let r0 = _mm512_loadu_pd(re_ptr.add(i0));
            let r1 = _mm512_loadu_pd(re_ptr.add(i1));
            let m0 = _mm512_loadu_pd(im_ptr.add(i0));
            let m1 = _mm512_loadu_pd(im_ptr.add(i1));
            _mm512_storeu_pd(re_ptr.add(i0), _mm512_xor_pd(m1, mask_new_re_i0));
            _mm512_storeu_pd(im_ptr.add(i0), _mm512_xor_pd(r1, mask_new_im_i0));
            _mm512_storeu_pd(re_ptr.add(i1), _mm512_xor_pd(m0, mask_new_re_i1));
            _mm512_storeu_pd(im_ptr.add(i1), _mm512_xor_pd(r0, mask_new_im_i1));
            j += LANES;
        }
        debug_assert_eq!(j, target_bit);
    };

    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let count = len / outer_step;
        crate::kernels::par_blocks(count, len, |k| k * outer_step, outer_iter);
        return;
    }

    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        outer_iter,
    );
}

/// AVX-512 generic anti-diagonal on SoA streams (Tier A).
///
/// `m = [[0, a], [b, 0]]`. new[i] = a * old[j]; new[j] = b * old[i].
/// (CORRECTED math: m[0][1]=a goes to new_i0, m[1][0]=b goes to new_i1.
///  Plan's original prose had a/b swapped; T2/T6 oracle-verified the
///  correct form on EPYC.)
///
/// Complex multiply on SoA: (new_re, new_im) = (s.re * z.re - s.im * z.im,
///                                              s.re * z.im + s.im * z.re).
///
/// # Safety
/// Same as `apply_1q_x_soa_avx512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_antidiag_soa_avx512(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    a: Complex,
    b: Complex,
) {
    use core::arch::x86_64::*;

    const LANES: usize = 8;
    let target_bit = 1usize << target;
    debug_assert!(target_bit >= LANES);
    debug_assert!(controls.iter().all(|&c| c > target));

    let ar = _mm512_set1_pd(a.re);
    let ai = _mm512_set1_pd(a.im);
    let br = _mm512_set1_pd(b.re);
    let bi = _mm512_set1_pd(b.im);

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());
    let len = re.len();

    let outer_iter = |block: usize| {
        let re_ptr = re_bp.ptr();
        let im_ptr = im_bp.ptr();
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            let i1 = block | target_bit | j;
            // SAFETY: in-bounds.
            let r0 = _mm512_loadu_pd(re_ptr.add(i0));
            let r1 = _mm512_loadu_pd(re_ptr.add(i1));
            let m0 = _mm512_loadu_pd(im_ptr.add(i0));
            let m1 = _mm512_loadu_pd(im_ptr.add(i1));

            // new[i0] = a * (r1, m1)
            // new_re_i0 = a.re * r1 - a.im * m1
            // new_im_i0 = a.re * m1 + a.im * r1
            let new_r_i0 = _mm512_fmsub_pd(ar, r1, _mm512_mul_pd(ai, m1));
            let new_m_i0 = _mm512_fmadd_pd(ar, m1, _mm512_mul_pd(ai, r1));

            // new[i1] = b * (r0, m0)
            let new_r_i1 = _mm512_fmsub_pd(br, r0, _mm512_mul_pd(bi, m0));
            let new_m_i1 = _mm512_fmadd_pd(br, m0, _mm512_mul_pd(bi, r0));

            _mm512_storeu_pd(re_ptr.add(i0), new_r_i0);
            _mm512_storeu_pd(im_ptr.add(i0), new_m_i0);
            _mm512_storeu_pd(re_ptr.add(i1), new_r_i1);
            _mm512_storeu_pd(im_ptr.add(i1), new_m_i1);

            j += LANES;
        }
        debug_assert_eq!(j, target_bit);
    };

    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let count = len / outer_step;
        crate::kernels::par_blocks(count, len, |k| k * outer_step, outer_iter);
        return;
    }

    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        outer_iter,
    );
}

/// AVX-512 Pauli-X on SoA streams (Tier B): `target ∈ {0, 1, 2}`.
///
/// Each `__m512d` of `re[]` (resp. `im[]`) holds 8 amplitudes; since
/// `1 << target < LANES_SOA = 8`, both elements of every swap pair live
/// inside a single zmm register.  A lane permute swaps them in-register:
///
/// * `target = 0` → swap adjacent doubles within each 128-bit lane:
///   `_mm512_permute_pd::<0x55>` (control `01 01 01 01`).
/// * `target = 1` → swap doubles at distance 2 within each 256-bit lane:
///   `_mm512_permutex_pd::<0x4E>` (control `01 00 11 10`).
/// * `target = 2` → swap halves of the zmm:
///   `_mm512_permutexvar_pd` with index `[4,5,6,7,0,1,2,3]`.
///
/// For X there is no sign change — same permute applied to both `re` and `im`.
///
/// Control mask: if `controls` is non-empty, only outer blocks satisfying
/// `(block & ctrl_mask) == ctrl_mask` are processed.  The contract
/// `controls.iter().all(|&c| c > target)` ensures every control bit is
/// above the 3-bit window, so the mask test on `block` is safe (block
/// steps by LANES_SOA = 8, which clears bits 0..=2 — i.e. exactly the
/// target-bit positions).
///
/// # Safety
/// * Host must support AVX-512F.
/// * `target ∈ {0, 1, 2}` (i.e. `1 << target < LANES_SOA = 8`).
/// * Every control > target (i.e. `controls.iter().all(|&c| c > target)`).
/// * `re.len() % LANES_SOA == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_x_soa_avx512_lowbit(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
) {
    use core::arch::x86_64::*;

    const LANES: usize = 8;
    debug_assert!((1usize << target) < LANES);
    // Tier-B SoA contract: the block-level `(block & ctrl_mask) == ctrl_mask`
    // gate only inspects bits ≥ log2(LANES) = 3 (since block addresses are
    // LANES-aligned). Any control at qubit index < 3 would alias to 0 in
    // `block` and the gate would silently no-op for amplitudes that DO have
    // the control bit set within the LANES-block. Dispatch must filter such
    // configurations to the scalar fallback.
    debug_assert!(controls.iter().all(|&c| c >= 3));
    debug_assert_eq!(re.len() % LANES, 0);

    // Index vector for target=2 swap (lanes [0..3] ↔ lanes [4..7] within zmm).
    // permutexvar lane k receives src[idx[k]]; lane-order idx = [4,5,6,7,0,1,2,3].
    // _mm512_set_epi64 args: arg 0 → lane 7, arg 7 → lane 0 (reversed).
    let idx_t2 = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    let ctrl_mask = if controls.is_empty() {
        0
    } else {
        crate::kernels::control_mask(controls)
    };

    let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());
    let len = re.len();

    let count = len / LANES;
    crate::kernels::par_blocks(
        count,
        len,
        |k| k * LANES,
        |block| {
            let re_ptr = re_bp.ptr();
            let im_ptr = im_bp.ptr();
            if (block & ctrl_mask) == ctrl_mask {
                // SAFETY: in-bounds — block + LANES ≤ len by loop invariant.
                let r = _mm512_loadu_pd(re_ptr.add(block));
                let m = _mm512_loadu_pd(im_ptr.add(block));
                let (r_p, m_p) = match target {
                    0 => (_mm512_permute_pd::<0x55>(r), _mm512_permute_pd::<0x55>(m)),
                    1 => (_mm512_permutex_pd::<0x4E>(r), _mm512_permutex_pd::<0x4E>(m)),
                    2 => (
                        _mm512_permutexvar_pd(idx_t2, r),
                        _mm512_permutexvar_pd(idx_t2, m),
                    ),
                    _ => unreachable!("Tier-B SoA: target out of {{0, 1, 2}}"),
                };
                _mm512_storeu_pd(re_ptr.add(block), r_p);
                _mm512_storeu_pd(im_ptr.add(block), m_p);
            }
        },
    );
}

// Tier-B Y and generic anti-diag SoA: NOT a separate SIMD kernel.
// The SoA dispatch routes both directly to the scalar kernels in
// `apply_1q_y_soa_scalar` / `apply_1q_antidiag_soa_scalar` when the
// Tier-A contract fails (target < 3 OR controls below 3). Lane-by-lane
// sign-mask construction on split re/im streams at target ∈ {0,1,2}
// is bug-prone and the workload payoff is minimal. See ADR 0011
// Open Question 1.

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
                let (r0, i0) = random_re_im(n, 0x00c0_1f5a + n as u64);
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
                let (r0, i0) = random_re_im(n, 0x005a_5ade + n as u64);
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
                let (r0, i0) = random_re_im(n, 0x00fe_edc2 + n as u64);
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
            Complex::new(-0.7, 0.7142857142857143),
            Complex::new(0.99, -0.1414213562373095),
            Complex::new(-0.5, -0.8660254037844386),
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
                let (r0, i0) = random_re_im(n, 0x0013_57d2 + n as u64);
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

    /// Tier B AVX-512 CNOT equivalence vs scalar SoA CNOT.  Cases:
    /// `target ∈ {0, 1, 2}` and `control >= 3` (so `c_bit ≥ LANES_SOA`).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_cnot_avx512_tier_b_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [7u32, 9] {
            for (c, t) in [(3u32, 0), (3, 1), (3, 2), (4, 0), (5, 1), (7, 2)] {
                if n <= c.max(t) {
                    continue;
                }
                let (r0, i0) =
                    random_re_im(n, 0x00b0_005a + n as u64 + ((c as u64) << 8) + t as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_cnot_scalar(&mut ra, &mut ia, c, t, &[]);
                // SAFETY: AVX-512F detected; target ∈ {0,1,2};
                // c_bit ≥ LANES_SOA; no external controls.
                unsafe {
                    super::apply_2q_cnot_avx512_tier_b(&mut rb, &mut ib, c, t, &[]);
                }
                assert_re_im_close(&ra, &rb, 1e-14);
                assert_re_im_close(&ia, &ib, 1e-14);
            }
        }
    }

    /// Tier C AVX-512 CNOT equivalence vs scalar SoA CNOT.  All 6 (c, t)
    /// pairs with both in {0, 1, 2}.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_cnot_avx512_tier_c_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [4u32, 6, 8] {
            for (c, t) in [(0u32, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)] {
                let (r0, i0) =
                    random_re_im(n, 0x00c0_005a + n as u64 + ((c as u64) << 8) + t as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_cnot_scalar(&mut ra, &mut ia, c, t, &[]);
                // SAFETY: AVX-512F detected; both control,target ∈
                // {0,1,2}; len = 1<<n ≥ 16 ≥ 8; no external controls.
                unsafe {
                    super::apply_2q_cnot_avx512_tier_c(&mut rb, &mut ib, c, t, &[]);
                }
                assert_re_im_close(&ra, &rb, 1e-14);
                assert_re_im_close(&ia, &ib, 1e-14);
            }
        }
    }

    /// Tier B AVX-512 SWAP equivalence vs scalar SoA SWAP.  Cases:
    /// `min(targets) ∈ {0, 1, 2}` and `max(targets) >= 3`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_swap_avx512_tier_b_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [7u32, 9] {
            for t in [[0u32, 3], [1, 3], [2, 3], [0, 5], [1, 5], [2, 5]] {
                let hi = t[0].max(t[1]);
                if n <= hi {
                    continue;
                }
                let (r0, i0) =
                    random_re_im(n, 0xb50a + n as u64 + ((t[0] as u64) << 8) + t[1] as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_swap_scalar(&mut ra, &mut ia, t, &[]);
                // SAFETY: AVX-512F detected; lo ∈ {0,1,2}; hi_bit ≥
                // LANES_SOA; distinct targets; no external controls.
                unsafe {
                    super::apply_2q_swap_avx512_tier_b(&mut rb, &mut ib, t, &[]);
                }
                assert_re_im_close(&ra, &rb, 1e-14);
                assert_re_im_close(&ia, &ib, 1e-14);
            }
        }
    }

    /// Tier C AVX-512 SWAP equivalence vs scalar SoA SWAP.  All 3
    /// distinct target pairs in `{0, 1, 2}`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_apply_2q_swap_avx512_tier_c_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [4u32, 6, 8] {
            for t in [[0u32, 1], [0, 2], [1, 2]] {
                let (r0, i0) =
                    random_re_im(n, 0xc50a + n as u64 + ((t[0] as u64) << 8) + t[1] as u64);
                let mut ra = r0.clone();
                let mut ia = i0.clone();
                let mut rb = r0;
                let mut ib = i0;
                apply_2q_swap_scalar(&mut ra, &mut ia, t, &[]);
                // SAFETY: AVX-512F detected; both targets in {0,1,2};
                // distinct; len = 1<<n ≥ 16 ≥ 8; no external controls.
                unsafe {
                    super::apply_2q_swap_avx512_tier_c(&mut rb, &mut ib, t, &[]);
                }
                assert_re_im_close(&ra, &rb, 1e-14);
                assert_re_im_close(&ia, &ib, 1e-14);
            }
        }
    }

    /// Portable indexing-coverage check for the Tier B / C outer-walks.
    /// Replays the renormalise-then-shift bit-arithmetic without any
    /// AVX-512 intrinsics — runs on aarch64 too — and asserts every
    /// amp that should be touched is touched exactly once.  Catches
    /// bit-collision bugs that would otherwise surface only on a real
    /// AVX-512 host.
    ///
    /// Two configs:
    /// 1. **Tier B CNOT** with one external control: outer-walk shifts
    ///    by `control + 1` and ORs in `c_bit`.  Every amp in the
    ///    LANES_SOA-aligned subspace with `(control_bit = 1, every ec
    ///    bit = 1, target_bit = 0)` plus the matching `target_bit = 1`
    ///    sibling must be touched exactly once (by the in-register
    ///    permute, which loads LANES_SOA contiguous amps that include
    ///    both target-bit values).
    /// 2. **Tier C SWAP** with one external control: outer-walk shifts
    ///    by 3 and lands on LANES_SOA-aligned bases.  Every amp in the
    ///    8-amp window with every ec bit set must be touched exactly
    ///    once (the in-register permute reads + writes all 8 doubles
    ///    per stream).
    #[test]
    fn soa_apply_2q_tier_bc_indexing_covers_state_exactly_once() {
        // ----- Config 1: Tier B CNOT, control=4, target=0, ec=[6]. -----
        {
            let n_qubits = 7u32;
            let len = 1usize << n_qubits;
            let control = 4u32;
            let c_bit = 1usize << control;
            let lanes = 8usize;
            let external_controls: &[u32] = &[6];

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            for &ec in external_controls {
                fixed_above.push((ec - control - 1, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

            let outer_count = 1usize << (n_qubits - control - 1 - external_controls.len() as u32);
            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }

            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let base = crate::kernels::expand_with_fixed(k, &fixed_above) << (control + 1);
                let outer = base | c_bit;
                assert_eq!(outer & ec_mask, ec_mask, "outer missing ec bits");
                assert_eq!(outer & c_bit, c_bit, "outer missing control bit");
                assert_eq!(outer & ((1usize << (control + 1)) - 1), c_bit);
                let mut j = 0usize;
                while j + lanes <= c_bit {
                    let i = outer | j;
                    for d in 0..lanes {
                        assert!(i + d < len, "OOB i+d={} len={}", i + d, len);
                        touched[i + d] += 1;
                    }
                    j += lanes;
                }
            }
            for (idx, &count) in touched.iter().enumerate() {
                let in_ec_subspace = (idx & ec_mask) == ec_mask;
                let control_set = (idx & c_bit) != 0;
                let expected = if in_ec_subspace && control_set {
                    1u32
                } else {
                    0u32
                };
                assert_eq!(
                    count, expected,
                    "Tier B CNOT: amp {idx} touched {count} times (expected {expected}; \
                     ec_subspace={in_ec_subspace} control_set={control_set})"
                );
            }
        }

        // ----- Config 2: Tier C SWAP, targets=[0,1], ec=[4]. -----
        {
            let n_qubits = 6u32;
            let len = 1usize << n_qubits;
            let external_controls: &[u32] = &[4];

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            for &ec in external_controls {
                fixed_above.push((ec - 3, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

            let outer_count = 1usize << (n_qubits - 3 - external_controls.len() as u32);
            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }
            let lanes = 8usize;

            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let base = crate::kernels::expand_with_fixed(k, &fixed_above) << 3;
                assert_eq!(base & 7, 0, "Tier C base not 8-aligned");
                assert_eq!(base & ec_mask, ec_mask, "Tier C base missing ec bits");
                for d in 0..lanes {
                    let i = base + d;
                    assert!(i < len, "Tier C OOB i={i} len={len}");
                    touched[i] += 1;
                }
            }
            for (idx, &count) in touched.iter().enumerate() {
                let in_ec_subspace = (idx & ec_mask) == ec_mask;
                let expected = if in_ec_subspace { 1u32 } else { 0u32 };
                assert_eq!(
                    count, expected,
                    "Tier C SWAP: amp {idx} touched {count} times (expected {expected}; \
                     ec_subspace={in_ec_subspace})"
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

    /// Regression test for the SoA Tier C dispatch bug found by
    /// high-effort code review on the P1-07 PR.  The bug: `dispatch_
    /// cnot_soa` / `dispatch_swap_soa` Tier C arms admitted external
    /// controls at qubit 2 while the kernels' `(ec - 3, true)`
    /// renormalisation requires `ec >= 3`.  With ec=2, the subtraction
    /// underflowed to 0xFFFFFFFF and crashed `expand_with_fixed`
    /// downstream (debug panic / release UB).
    ///
    /// We verify by routing a CnotHi-shaped matrix through `apply_2q`
    /// with `targets = [0, 1]` and `external_controls = [2]` on n=4
    /// (len = 16, well above LANES_SOA = 8 so the Tier-C `len` guard
    /// would otherwise admit this call).  The dispatch must NOT enter
    /// Tier C — it must fall through to scalar — and the result must
    /// match `apply_2q_dense_scalar`.
    #[test]
    fn soa_dispatch_cnot_tier_c_rejects_external_control_at_qubit_2() {
        let n = 4u32;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][3] = Complex::new(1.0, 0.0);
            m[3][2] = Complex::new(1.0, 0.0);
            m
        };
        let (r0, i0) = random_re_im(n, 0xc7c2);
        let mut ra = r0.clone();
        let mut ia = i0.clone();
        let mut rb = r0;
        let mut ib = i0;
        // Routed through the full apply_2q prelude — pre-fix this
        // would panic in debug or silently corrupt state in release.
        apply_2q(&mut ra, &mut ia, [0, 1], &[2], &m);
        // Reference via dense scalar.
        apply_2q_dense_scalar(&mut rb, &mut ib, [0, 1], &[2], &m);
        assert_re_im_close(&ra, &rb, 1e-14);
        assert_re_im_close(&ia, &ib, 1e-14);
    }

    /// Mirror of the CNOT regression test for SWAP.  Targets {0, 1}
    /// with external control at qubit 2 — pre-fix `dispatch_swap_soa`
    /// admitted ec=2 into the Tier C kernel where the `(ec - 3)`
    /// renormalisation underflowed.
    #[test]
    fn soa_dispatch_swap_tier_c_rejects_external_control_at_qubit_2() {
        let n = 4u32;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][2] = Complex::new(1.0, 0.0);
            m[2][1] = Complex::new(1.0, 0.0);
            m[3][3] = Complex::new(1.0, 0.0);
            m
        };
        let (r0, i0) = random_re_im(n, 0x5402);
        let mut ra = r0.clone();
        let mut ia = i0.clone();
        let mut rb = r0;
        let mut ib = i0;
        apply_2q(&mut ra, &mut ia, [0, 1], &[2], &m);
        apply_2q_dense_scalar(&mut rb, &mut ib, [0, 1], &[2], &m);
        assert_re_im_close(&ra, &rb, 1e-14);
        assert_re_im_close(&ia, &ib, 1e-14);
    }

    // ----- P1-05 T8: SoA anti-diagonal scalar oracles vs AoS scalars -----

    #[test]
    fn apply_1q_x_soa_scalar_matches_aos() {
        let mut re: Vec<f64> = (0..16).map(|k| k as f64).collect();
        let mut im: Vec<f64> = (0..16).map(|k| -(k as f64)).collect();
        let mut amps_aos: Vec<Complex> = (0..16)
            .map(|k| Complex::new(k as f64, -(k as f64)))
            .collect();
        super::apply_1q_x_soa_scalar(&mut re, &mut im, 2, &[]);
        aos::apply_1q_x_scalar(&mut amps_aos, 2, &[]);
        for k in 0..16 {
            assert!((re[k] - amps_aos[k].re).abs() < 1e-12);
            assert!((im[k] - amps_aos[k].im).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_1q_y_soa_scalar_pos_matches_aos() {
        let mut re: Vec<f64> = (0..16).map(|k| k as f64 * 0.13).collect();
        let mut im: Vec<f64> = (0..16).map(|k| k as f64 * 0.27 + 1.0).collect();
        let mut amps_aos: Vec<Complex> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        super::apply_1q_y_soa_scalar(&mut re, &mut im, 1, &[], 1.0);
        aos::apply_1q_y_scalar(&mut amps_aos, 1, &[], 1.0);
        for k in 0..16 {
            assert!((re[k] - amps_aos[k].re).abs() < 1e-12);
            assert!((im[k] - amps_aos[k].im).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_1q_antidiag_soa_scalar_matches_aos() {
        let a = Complex::new(0.6, 0.8);
        let b = Complex::new(0.6, -0.8);
        let mut re: Vec<f64> = (0..16).map(|k| k as f64 * 0.05).collect();
        let mut im: Vec<f64> = (0..16).map(|k| k as f64 * 0.11 - 0.3).collect();
        let mut amps_aos: Vec<Complex> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        super::apply_1q_antidiag_soa_scalar(&mut re, &mut im, 2, &[], a, b);
        aos::apply_1q_antidiag_scalar(&mut amps_aos, 2, &[], a, b);
        for k in 0..16 {
            assert!((re[k] - amps_aos[k].re).abs() < 1e-12);
            assert!((im[k] - amps_aos[k].im).abs() < 1e-12);
        }
    }

    // ----- P1-05 T9: SoA Tier-A AVX-512 anti-diagonal kernels vs scalar -----

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_x_soa_avx512_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=8, target=3 → target_bit=8 = LANES_SOA.
        let mut re_avx: Vec<f64> = (0..256).map(|k| k as f64 * 0.05).collect();
        let mut im_avx: Vec<f64> = (0..256).map(|k| k as f64 * 0.11 - 0.5).collect();
        let mut re_sca = re_avx.clone();
        let mut im_sca = im_avx.clone();
        // SAFETY: avx512 detected + target_bit=8 >= LANES_SOA + no controls.
        unsafe {
            super::apply_1q_x_soa_avx512(&mut re_avx, &mut im_avx, 3, &[]);
        }
        super::apply_1q_x_soa_scalar(&mut re_sca, &mut im_sca, 3, &[]);
        assert_eq!(re_avx, re_sca);
        assert_eq!(im_avx, im_sca);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_y_soa_avx512_pos_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut re_avx: Vec<f64> = (0..256).map(|k| k as f64 * 0.07 - 2.0).collect();
        let mut im_avx: Vec<f64> = (0..256).map(|k| k as f64 * 0.13 + 1.0).collect();
        let mut re_sca = re_avx.clone();
        let mut im_sca = im_avx.clone();
        // SAFETY: avx512 detected + target_bit=8 >= LANES_SOA + no controls.
        unsafe {
            super::apply_1q_y_soa_avx512(&mut re_avx, &mut im_avx, 3, &[], 1.0);
        }
        super::apply_1q_y_soa_scalar(&mut re_sca, &mut im_sca, 3, &[], 1.0);
        for k in 0..256 {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-12);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_antidiag_soa_avx512_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = Complex::new(0.6, 0.8);
        let b = Complex::new(0.6, -0.8);
        let mut re_avx: Vec<f64> = (0..256).map(|k| k as f64 * 0.05).collect();
        let mut im_avx: Vec<f64> = (0..256).map(|k| -(k as f64) * 0.03).collect();
        let mut re_sca = re_avx.clone();
        let mut im_sca = im_avx.clone();
        // SAFETY: avx512 detected + target_bit=8 >= LANES_SOA + no controls.
        unsafe {
            super::apply_1q_antidiag_soa_avx512(&mut re_avx, &mut im_avx, 3, &[], a, b);
        }
        super::apply_1q_antidiag_soa_scalar(&mut re_sca, &mut im_sca, 3, &[], a, b);
        for k in 0..256 {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-12);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_x_soa_avx512_lowbit_all_targets_match_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for &target in &[0u32, 1u32, 2u32] {
            let mut re_avx: Vec<f64> = (0..64).map(|k| k as f64 * 0.05).collect();
            let mut im_avx: Vec<f64> = (0..64).map(|k| -(k as f64) * 0.07).collect();
            let mut re_sca = re_avx.clone();
            let mut im_sca = im_avx.clone();
            // SAFETY: avx512 detected + target ∈ {0,1,2} + no controls +
            // len = 64 is divisible by LANES_SOA = 8.
            unsafe {
                super::apply_1q_x_soa_avx512_lowbit(&mut re_avx, &mut im_avx, target, &[]);
            }
            super::apply_1q_x_soa_scalar(&mut re_sca, &mut im_sca, target, &[]);
            assert_eq!(re_avx, re_sca, "target={}", target);
            assert_eq!(im_avx, im_sca, "target={}", target);
        }
    }

    // ---- P1-05 review B2: SoA Tier-B dispatch boundary oracle ----

    #[test]
    fn apply_1q_y_soa_dispatch_low_target_routes_to_scalar() {
        // n=3 (8 amps), target=0. SoA dispatch: 1<<0=1 < LANES_SOA=8 → no Tier-A.
        // YPos arm has NO Tier-B branch, so falls through to scalar. Verify
        // equivalence to apply_1q_y_soa_scalar directly.
        let z = Complex::new(0.0, 0.0);
        let pi = Complex::new(0.0, 1.0);
        let ni = Complex::new(0.0, -1.0);
        let pauli_y_pos = [[z, ni], [pi, z]];
        let mut re_dispatch: Vec<f64> = (0..8).map(|k| k as f64 * 0.13 - 0.5).collect();
        let mut im_dispatch: Vec<f64> = (0..8).map(|k| k as f64 * 0.27 + 1.0).collect();
        let mut re_direct = re_dispatch.clone();
        let mut im_direct = im_dispatch.clone();
        super::apply_1q(&mut re_dispatch, &mut im_dispatch, 0, &[], &pauli_y_pos);
        super::apply_1q_y_soa_scalar(&mut re_direct, &mut im_direct, 0, &[], 1.0);
        for k in 0..8 {
            assert!(
                (re_dispatch[k] - re_direct[k]).abs() < 1e-12,
                "re[{}] mismatch",
                k
            );
            assert!(
                (im_dispatch[k] - im_direct[k]).abs() < 1e-12,
                "im[{}] mismatch",
                k
            );
        }
    }

    #[test]
    fn apply_1q_antidiag_soa_dispatch_low_target_routes_to_scalar() {
        // Same shape for generic anti-diag.
        // n=3 (8 amps), target=1. SoA dispatch: 1<<1=2 < LANES_SOA=8 → no Tier-A.
        // Generic anti-diag arm has NO Tier-B branch, falls through to scalar.
        let z = Complex::new(0.0, 0.0);
        let a = Complex::new(0.6, 0.8);
        let b = Complex::new(0.6, -0.8);
        let m = [[z, a], [b, z]];
        let mut re_dispatch: Vec<f64> = (0..8).map(|k| k as f64 * 0.05).collect();
        let mut im_dispatch: Vec<f64> = (0..8).map(|k| k as f64 * 0.11 - 0.3).collect();
        let mut re_direct = re_dispatch.clone();
        let mut im_direct = im_dispatch.clone();
        super::apply_1q(&mut re_dispatch, &mut im_dispatch, 1, &[], &m);
        super::apply_1q_antidiag_soa_scalar(&mut re_direct, &mut im_direct, 1, &[], a, b);
        for k in 0..8 {
            assert!(
                (re_dispatch[k] - re_direct[k]).abs() < 1e-12,
                "re[{}] mismatch: dispatch={} direct={}",
                k,
                re_dispatch[k],
                re_direct[k]
            );
            assert!(
                (im_dispatch[k] - im_direct[k]).abs() < 1e-12,
                "im[{}] mismatch: dispatch={} direct={}",
                k,
                im_dispatch[k],
                im_direct[k]
            );
        }
    }

    // ---- P1-05 T12: SoA boundary-n test ----

    #[test]
    fn apply_1q_x_soa_dispatch_boundary_n() {
        // n=2 < LANES_SOA=8 → scalar path; n=3..=5 spans the tier-B target range.
        for n in 2..=5u32 {
            let len = 1usize << n;
            let mut re: Vec<f64> = (0..len).map(|k| k as f64 * 0.13).collect();
            let mut im: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.07).collect();
            let mut re_ref = re.clone();
            let mut im_ref = im.clone();
            super::apply_1q(&mut re, &mut im, 0, &[], &pauli_x());
            super::apply_1q_x_soa_scalar(&mut re_ref, &mut im_ref, 0, &[]);
            assert_eq!(re, re_ref, "n={}", n);
            assert_eq!(im, im_ref, "n={}", n);
        }
    }
}

// ---- P1-08 T15: SoA Toffoli + CCZ kernels ----

#[cfg(test)]
mod soa_multi_controlled_tests {
    use aleph_core::Complex;

    // Helper: build a Toffoli (CCX) 8×8 matrix (rows 6 ↔ 7 swapped).
    fn toffoli_m() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = o;
        }
        // Swap rows 6 and 7 (Toffoli convention: targets=[c0,c1,t]; matrix
        // index 6 = 0b110 → c0=1, c1=1, t=0; 7 = 0b111 → c0=1, c1=1, t=1).
        m[6][6] = z;
        m[6][7] = o;
        m[7][7] = z;
        m[7][6] = o;
        m
    }

    // Helper: build a CCZ 8×8 matrix (diag with -1 at index 7).
    fn ccz_m() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let neg = Complex::new(-1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = o;
        }
        m[7][7] = neg;
        m
    }

    // ---- apply_toffoli_scalar_soa ----

    #[test]
    fn toffoli_scalar_soa_flips_target_when_both_controls_set() {
        // 4 qubits (16 amps), targets=[0,1,2], no external controls.
        // When both c0=0 and c1=1 are set (amp index has bits 0 and 1 set),
        // bit 2 (the target) should be flipped.  Amp 3 (=0b011) ↔ amp 7 (=0b111).
        let n = 4;
        let len = 1usize << n;
        let mut re: Vec<f64> = (0..len).map(|k| k as f64 * 0.1).collect();
        let mut im: Vec<f64> = (0..len).map(|k| k as f64 * 0.05 + 0.01).collect();
        let re_orig = re.clone();
        let im_orig = im.clone();
        super::apply_toffoli_scalar_soa(&mut re, &mut im, [0, 1, 2], &[]);
        // Amps 3 and 7 should be swapped; all others unchanged.
        assert!((re[3] - re_orig[7]).abs() < 1e-15);
        assert!((im[3] - im_orig[7]).abs() < 1e-15);
        assert!((re[7] - re_orig[3]).abs() < 1e-15);
        assert!((im[7] - im_orig[3]).abs() < 1e-15);
        // Spot-check an amp that should be unchanged (amp 5 = 0b0101; c0=1, c1=0 → no fire).
        assert!((re[5] - re_orig[5]).abs() < 1e-15);
        assert!((im[5] - im_orig[5]).abs() < 1e-15);
    }

    #[test]
    fn toffoli_scalar_soa_involutive() {
        let n = 5;
        let len = 1usize << n;
        let re_orig: Vec<f64> = (0..len).map(|k| (k as f64).sin() * 0.3).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| (k as f64).cos() * 0.4).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_toffoli_scalar_soa(&mut re, &mut im, [0, 1, 2], &[]);
        super::apply_toffoli_scalar_soa(&mut re, &mut im, [0, 1, 2], &[]);
        for k in 0..len {
            assert!((re[k] - re_orig[k]).abs() < 1e-14, "re[{}]", k);
            assert!((im[k] - im_orig[k]).abs() < 1e-14, "im[{}]", k);
        }
    }

    #[test]
    fn toffoli_scalar_soa_external_control_gates() {
        // External control qubit 3; gate fires only when bits 0,1,3 all set.
        let n = 4;
        let len = 1usize << n;
        let mut re: Vec<f64> = (0..len).map(|k| k as f64 * 0.1).collect();
        let mut im: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.07).collect();
        let re_orig = re.clone();
        let im_orig = im.clone();
        super::apply_toffoli_scalar_soa(&mut re, &mut im, [0, 1, 2], &[3]);
        // ctrl_mask = 0b1011 = 11; target_bit = 4.
        // Only amp 11 (0b1011) and amp 15 (0b1111) should swap.
        assert!((re[11] - re_orig[15]).abs() < 1e-15);
        assert!((im[11] - im_orig[15]).abs() < 1e-15);
        assert!((re[15] - re_orig[11]).abs() < 1e-15);
        assert!((im[15] - im_orig[11]).abs() < 1e-15);
        // Amp 3 (0b0011): bits 0,1 set but not bit 3 → no swap.
        assert!((re[3] - re_orig[3]).abs() < 1e-15);
        assert!((im[3] - im_orig[3]).abs() < 1e-15);
    }

    #[test]
    fn toffoli_scalar_soa_matches_dispatch() {
        // Verify that apply_3q routes to the scalar SoA Toffoli correctly
        // (small n=3, Tier-C path).
        let len = 8usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.13 - 0.5).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.09 + 0.3).collect();
        let mut re_disp = re_init.clone();
        let mut im_disp = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        super::apply_3q(&mut re_disp, &mut im_disp, [0, 1, 2], &[], &toffoli_m());
        super::apply_toffoli_scalar_soa(&mut re_sca, &mut im_sca, [0, 1, 2], &[]);
        for k in 0..len {
            assert!((re_disp[k] - re_sca[k]).abs() < 1e-14, "re[{}]", k);
            assert!((im_disp[k] - im_sca[k]).abs() < 1e-14, "im[{}]", k);
        }
    }

    // ---- apply_ccz_scalar_soa ----

    #[test]
    fn ccz_scalar_soa_sign_flips_only_111() {
        let n = 4;
        let len = 1usize << n;
        let mut re: Vec<f64> = (0..len).map(|k| k as f64 * 0.1 + 1.0).collect();
        let mut im: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.05).collect();
        let re_orig = re.clone();
        let im_orig = im.clone();
        super::apply_ccz_scalar_soa(&mut re, &mut im, [0, 1, 2], &[]);
        // mask = 0b0111 = 7; only amp 7 (and amp 15 where bits 0,1,2 all set)
        // should be negated.
        for k in 0..len {
            if (k & 7) == 7 {
                assert!(
                    (re[k] + re_orig[k]).abs() < 1e-15,
                    "re[{}] should negate",
                    k
                );
                assert!(
                    (im[k] + im_orig[k]).abs() < 1e-15,
                    "im[{}] should negate",
                    k
                );
            } else {
                assert!((re[k] - re_orig[k]).abs() < 1e-15, "re[{}] unchanged", k);
                assert!((im[k] - im_orig[k]).abs() < 1e-15, "im[{}] unchanged", k);
            }
        }
    }

    #[test]
    fn ccz_scalar_soa_involutive() {
        let n = 5;
        let len = 1usize << n;
        let re_orig: Vec<f64> = (0..len).map(|k| (k as f64).sin()).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| (k as f64).cos()).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_ccz_scalar_soa(&mut re, &mut im, [0, 1, 2], &[]);
        super::apply_ccz_scalar_soa(&mut re, &mut im, [0, 1, 2], &[]);
        for k in 0..len {
            assert!((re[k] - re_orig[k]).abs() < 1e-14, "re[{}]", k);
            assert!((im[k] - im_orig[k]).abs() < 1e-14, "im[{}]", k);
        }
    }

    #[test]
    fn ccz_scalar_soa_matches_dispatch() {
        // Small n=3 → Tier-C scalar path.
        let len = 8usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.17 - 0.6).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.11 + 0.4).collect();
        let mut re_disp = re_init.clone();
        let mut im_disp = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        super::apply_3q(&mut re_disp, &mut im_disp, [0, 1, 2], &[], &ccz_m());
        super::apply_ccz_scalar_soa(&mut re_sca, &mut im_sca, [0, 1, 2], &[]);
        for k in 0..len {
            assert!((re_disp[k] - re_sca[k]).abs() < 1e-14, "re[{}]", k);
            assert!((im_disp[k] - im_sca[k]).abs() < 1e-14, "im[{}]", k);
        }
    }

    // ---- SoA dispatch matches AoS ----

    #[test]
    fn toffoli_soa_dispatch_matches_aos_n5() {
        // n=5 (32 amps). targets=[0,1,2] → Tier-B or Tier-A depending on AVX-512.
        let n = 5;
        let len = 1usize << n;
        let aos_init: Vec<Complex> = (0..len)
            .map(|k| Complex::new(k as f64 * 0.07, -(k as f64) * 0.03))
            .collect();
        let mut aos = aos_init.clone();
        let mut re: Vec<f64> = aos_init.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = aos_init.iter().map(|c| c.im).collect();
        crate::kernels::aos::apply_3q(&mut aos, [0, 1, 2], &[], &toffoli_m());
        super::apply_3q(&mut re, &mut im, [0, 1, 2], &[], &toffoli_m());
        for k in 0..len {
            assert!((re[k] - aos[k].re).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - aos[k].im).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn ccz_soa_dispatch_matches_aos_n5() {
        let n = 5;
        let len = 1usize << n;
        let aos_init: Vec<Complex> = (0..len)
            .map(|k| Complex::new((k as f64).sin() * 0.5, (k as f64).cos() * 0.5))
            .collect();
        let mut aos = aos_init.clone();
        let mut re: Vec<f64> = aos_init.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = aos_init.iter().map(|c| c.im).collect();
        crate::kernels::aos::apply_3q(&mut aos, [0, 1, 2], &[], &ccz_m());
        super::apply_3q(&mut re, &mut im, [0, 1, 2], &[], &ccz_m());
        for k in 0..len {
            assert!((re[k] - aos[k].re).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - aos[k].im).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn toffoli_soa_dispatch_matches_aos_n8_tier_a() {
        // n=8 (256 amps), targets=[2,3,4] → Tier-A (target=4 >= 3=LANES_SOA_BITS).
        let n = 8;
        let len = 1usize << n;
        let aos_init: Vec<Complex> = (0..len)
            .map(|k| Complex::new(k as f64 * 0.01 - 1.28, -(k as f64) * 0.007))
            .collect();
        let mut aos = aos_init.clone();
        let mut re: Vec<f64> = aos_init.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = aos_init.iter().map(|c| c.im).collect();
        crate::kernels::aos::apply_3q(&mut aos, [2, 3, 4], &[], &toffoli_m());
        super::apply_3q(&mut re, &mut im, [2, 3, 4], &[], &toffoli_m());
        for k in 0..len {
            assert!((re[k] - aos[k].re).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - aos[k].im).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn ccz_soa_dispatch_matches_aos_n8_tier_a() {
        let n = 8;
        let len = 1usize << n;
        let aos_init: Vec<Complex> = (0..len)
            .map(|k| Complex::new((k as f64).sin(), (k as f64).cos()))
            .collect();
        let mut aos = aos_init.clone();
        let mut re: Vec<f64> = aos_init.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = aos_init.iter().map(|c| c.im).collect();
        crate::kernels::aos::apply_3q(&mut aos, [3, 4, 5], &[], &ccz_m());
        super::apply_3q(&mut re, &mut im, [3, 4, 5], &[], &ccz_m());
        for k in 0..len {
            assert!((re[k] - aos[k].re).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - aos[k].im).abs() < 1e-13, "im[{}]", k);
        }
    }

    // ---- Tier-A outer-walk (control below target) ----

    #[test]
    fn toffoli_soa_outer_walk_control_below_target_matches_scalar() {
        // targets=[0,3,4]: c0=0 is BELOW t=4. This triggers outer-walk path.
        let n = 6;
        let len = 1usize << n;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.03 - 1.0).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.02 + 0.5).collect();
        let mut re_disp = re_init.clone();
        let mut im_disp = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        super::apply_3q(&mut re_disp, &mut im_disp, [0, 3, 4], &[], &toffoli_m());
        super::apply_toffoli_scalar_soa(&mut re_sca, &mut im_sca, [0, 3, 4], &[]);
        for k in 0..len {
            assert!(
                (re_disp[k] - re_sca[k]).abs() < 1e-13,
                "re[{}] disp={} sca={}",
                k,
                re_disp[k],
                re_sca[k]
            );
            assert!((im_disp[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn ccz_soa_outer_walk_low_bit_matches_scalar() {
        // targets=[0,1,4]: mask_lo=0 < 3 → outer-walk path.
        let n = 6;
        let len = 1usize << n;
        let re_init: Vec<f64> = (0..len).map(|k| (k as f64).sin() * 0.7).collect();
        let im_init: Vec<f64> = (0..len).map(|k| (k as f64).cos() * 0.7).collect();
        let mut re_disp = re_init.clone();
        let mut im_disp = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        super::apply_3q(&mut re_disp, &mut im_disp, [0, 1, 4], &[], &ccz_m());
        super::apply_ccz_scalar_soa(&mut re_sca, &mut im_sca, [0, 1, 4], &[]);
        for k in 0..len {
            assert!(
                (re_disp[k] - re_sca[k]).abs() < 1e-13,
                "re[{}] disp={} sca={}",
                k,
                re_disp[k],
                re_sca[k]
            );
            assert!((im_disp[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    // ---- AVX-512 Tier-A kernels directly ----

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_toffoli_avx512_tier_a_soa_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=6 (64 amps), target=3 >= LANES_SOA_BITS=3, controls=[4, 5] > 3.
        // Clean Tier-A contract: c_lo > target. Scalar takes the same
        // control set via targets=[c0, c1, target].
        let len = 64usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.05 - 0.8).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.04 + 0.6).collect();
        let mut re_avx = re_init.clone();
        let mut im_avx = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        // SAFETY: AVX-512F detected, target=3 >= 3, controls=[4, 5] both > 3.
        unsafe {
            super::apply_toffoli_avx512_tier_a_soa(&mut re_avx, &mut im_avx, 3, &[4, 5]);
        }
        super::apply_toffoli_scalar_soa(&mut re_sca, &mut im_sca, [4, 5, 3], &[]);
        for k in 0..len {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_toffoli_avx512_tier_a_outer_walk_soa_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=7 (128 amps), target=5 >= 3, controls=[3, 6]: c_lo=3 ≤ target
        // (outer-walk path) but both controls ≥ LANES_SOA_BITS=3 (valid).
        let len = 128usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.03).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.02).collect();
        let mut re_avx = re_init.clone();
        let mut im_avx = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        // SAFETY: AVX-512F detected, target=5 ≥ 3, both controls ≥ LANES_SOA_BITS=3.
        unsafe {
            super::apply_toffoli_avx512_tier_a_outer_walk_soa(&mut re_avx, &mut im_avx, 5, &[3, 6]);
        }
        super::apply_toffoli_scalar_soa(&mut re_sca, &mut im_sca, [3, 6, 5], &[]);
        for k in 0..len {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    /// Verifies that `dispatch_toffoli_soa` falls through to the SoA scalar
    /// path when a control sits below LANES_SOA_BITS=3 — without this
    /// fallback, the outer-walk SoA SIMD path would silently no-op.
    #[test]
    fn dispatch_toffoli_soa_falls_through_to_scalar_when_control_below_lanes_bits() {
        // n=6, controls=[0, 5], target=3: c_lo=0 < LANES_SOA_BITS=3.
        let len = 64usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.05 + 0.1).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.03).collect();
        let mut re_d = re_init.clone();
        let mut im_d = im_init.clone();
        let mut re_s = re_init.clone();
        let mut im_s = im_init.clone();
        super::dispatch_toffoli_soa(&mut re_d, &mut im_d, [0, 5, 3], &[]);
        super::apply_toffoli_scalar_soa(&mut re_s, &mut im_s, [0, 5, 3], &[]);
        for k in 0..len {
            assert!((re_d[k] - re_s[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_d[k] - im_s[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    // ---- AVX-512 Tier-B kernels directly ----

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_toffoli_avx512_tier_b0_soa_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=5 (32 amps), target=0, controls=[3,4] >= LANES_SOA_BITS=3.
        let len = 32usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.1 - 1.5).collect();
        let im_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.07).collect();
        let mut re_avx = re_init.clone();
        let mut im_avx = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        // SAFETY: AVX-512F detected, target=0 (implicit), controls=[3,4] >= 3.
        unsafe {
            super::apply_toffoli_avx512_tier_b0_soa(&mut re_avx, &mut im_avx, &[3, 4]);
        }
        super::apply_toffoli_scalar_soa(&mut re_sca, &mut im_sca, [3, 4, 0], &[]);
        for k in 0..len {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_toffoli_avx512_tier_b1_soa_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=5 (32 amps), target=1, controls=[3,4] >= LANES_SOA_BITS=3.
        let len = 32usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.09 - 1.0).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.06).collect();
        let mut re_avx = re_init.clone();
        let mut im_avx = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        // SAFETY: AVX-512F detected, target=1 (implicit), controls=[3,4] >= 3.
        unsafe {
            super::apply_toffoli_avx512_tier_b1_soa(&mut re_avx, &mut im_avx, &[3, 4]);
        }
        super::apply_toffoli_scalar_soa(&mut re_sca, &mut im_sca, [3, 4, 1], &[]);
        for k in 0..len {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_toffoli_avx512_tier_b2_soa_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=5 (32 amps), target=2, controls=[3,4] >= LANES_SOA_BITS=3.
        let len = 32usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.11 - 0.5).collect();
        let im_init: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.08 + 0.3).collect();
        let mut re_avx = re_init.clone();
        let mut im_avx = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        // SAFETY: AVX-512F detected, target=2 (implicit), controls=[3,4] >= 3.
        unsafe {
            super::apply_toffoli_avx512_tier_b2_soa(&mut re_avx, &mut im_avx, &[3, 4]);
        }
        super::apply_toffoli_scalar_soa(&mut re_sca, &mut im_sca, [3, 4, 2], &[]);
        for k in 0..len {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    // ---- AVX-512 CCZ Tier-A ----

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_ccz_avx512_tier_a_soa_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=6 (64 amps), mask_bits=[3,4,5] — all >= LANES_SOA_BITS=3.
        let len = 64usize;
        let re_init: Vec<f64> = (0..len).map(|k| (k as f64).sin() * 0.6).collect();
        let im_init: Vec<f64> = (0..len).map(|k| (k as f64).cos() * 0.6).collect();
        let mut re_avx = re_init.clone();
        let mut im_avx = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        // SAFETY: AVX-512F detected, mask_bits all >= 3.
        unsafe {
            super::apply_ccz_avx512_tier_a_soa(&mut re_avx, &mut im_avx, &[3, 4, 5]);
        }
        super::apply_ccz_scalar_soa(&mut re_sca, &mut im_sca, [3, 4, 5], &[]);
        for k in 0..len {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_ccz_avx512_tier_a_outer_walk_soa_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=6 (64 amps), mask_bits=[0,1,4] — bits 0,1 < LANES_SOA_BITS=3.
        let len = 64usize;
        let re_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.04 - 1.28).collect();
        let im_init: Vec<f64> = (0..len).map(|k| k as f64 * 0.03 - 0.96).collect();
        let mut re_avx = re_init.clone();
        let mut im_avx = im_init.clone();
        let mut re_sca = re_init.clone();
        let mut im_sca = im_init.clone();
        // SAFETY: AVX-512F detected; some mask bits < 3 — outer-walk path.
        unsafe {
            super::apply_ccz_avx512_tier_a_outer_walk_soa(&mut re_avx, &mut im_avx, &[0, 1, 4]);
        }
        super::apply_ccz_scalar_soa(&mut re_sca, &mut im_sca, [0, 1, 4], &[]);
        for k in 0..len {
            assert!((re_avx[k] - re_sca[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im_avx[k] - im_sca[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    // ---- Involutivity under dispatch (Tier-A / Tier-B paths) ----

    #[test]
    fn toffoli_dispatch_soa_involutive_n6_tier_a() {
        // n=6 (64 amps), targets=[3,4,5] → Tier-A on AVX-512, scalar otherwise.
        let n = 6;
        let len = 1usize << n;
        let re_orig: Vec<f64> = (0..len).map(|k| (k as f64).sin()).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| (k as f64).cos()).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_3q(&mut re, &mut im, [3, 4, 5], &[], &toffoli_m());
        super::apply_3q(&mut re, &mut im, [3, 4, 5], &[], &toffoli_m());
        for k in 0..len {
            assert!((re[k] - re_orig[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - im_orig[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn toffoli_dispatch_soa_involutive_n5_tier_b0() {
        // n=5 (32 amps), targets=[3,4,0] → Tier-B.0 (t=0, c_lo=3 >= 3).
        let n = 5;
        let len = 1usize << n;
        let re_orig: Vec<f64> = (0..len).map(|k| k as f64 * 0.11 - 0.5).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.09 + 0.3).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_3q(&mut re, &mut im, [3, 4, 0], &[], &toffoli_m());
        super::apply_3q(&mut re, &mut im, [3, 4, 0], &[], &toffoli_m());
        for k in 0..len {
            assert!((re[k] - re_orig[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - im_orig[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn toffoli_dispatch_soa_involutive_n5_tier_b1() {
        // n=5 (32 amps), targets=[3,4,1] → Tier-B.1 (t=1, c_lo=3 >= 3).
        let n = 5;
        let len = 1usize << n;
        let re_orig: Vec<f64> = (0..len).map(|k| k as f64 * 0.13).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.11).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_3q(&mut re, &mut im, [3, 4, 1], &[], &toffoli_m());
        super::apply_3q(&mut re, &mut im, [3, 4, 1], &[], &toffoli_m());
        for k in 0..len {
            assert!((re[k] - re_orig[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - im_orig[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn toffoli_dispatch_soa_involutive_n5_tier_b2() {
        // n=5 (32 amps), targets=[3,4,2] → Tier-B.2 (t=2, c_lo=3 >= 3).
        let n = 5;
        let len = 1usize << n;
        let re_orig: Vec<f64> = (0..len).map(|k| k as f64 * 0.17 - 1.0).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.14 + 0.7).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_3q(&mut re, &mut im, [3, 4, 2], &[], &toffoli_m());
        super::apply_3q(&mut re, &mut im, [3, 4, 2], &[], &toffoli_m());
        for k in 0..len {
            assert!((re[k] - re_orig[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - im_orig[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    #[test]
    fn ccz_dispatch_soa_involutive_n6_tier_a() {
        let n = 6;
        let len = 1usize << n;
        let re_orig: Vec<f64> = (0..len).map(|k| (k as f64 * 0.07).sin()).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| (k as f64 * 0.07).cos()).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_3q(&mut re, &mut im, [3, 4, 5], &[], &ccz_m());
        super::apply_3q(&mut re, &mut im, [3, 4, 5], &[], &ccz_m());
        for k in 0..len {
            assert!((re[k] - re_orig[k]).abs() < 1e-13, "re[{}]", k);
            assert!((im[k] - im_orig[k]).abs() < 1e-13, "im[{}]", k);
        }
    }

    // ---- apply_3q identity short-circuit ----

    #[test]
    fn apply_3q_soa_identity_is_noop() {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut identity = [[z; 8]; 8];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = o;
        }
        let len = 16usize;
        let re_orig: Vec<f64> = (0..len).map(|k| k as f64 * 0.1).collect();
        let im_orig: Vec<f64> = (0..len).map(|k| -(k as f64) * 0.05).collect();
        let mut re = re_orig.clone();
        let mut im = im_orig.clone();
        super::apply_3q(&mut re, &mut im, [0, 1, 2], &[], &identity);
        assert_eq!(re, re_orig);
        assert_eq!(im, im_orig);
    }
}
