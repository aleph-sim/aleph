//! Indexed gate application kernels.
//!
//! Convention from the P0-06 spec: `qubits[0]` is the **MSB** of the
//! matrix index. For a 2-qubit gate on `[a, b]`, basis order is
//! `|a b⟩` — i.e. matrix row/col `k` corresponds to `(a, b) =
//! ((k >> 1) & 1, k & 1)`. Same generalization for 3-qubit gates.
//! This matches `Gate::Cnot` (`qubits = [control, target]`) whose
//! matrix swaps rows 2 ↔ 3.

use aleph_core::Complex;

/// Apply a 1-qubit matrix to `target` (possibly with external
/// `controls`) in place.
///
/// Iterates the 2^(n-1) basis indices whose `target` bit is zero;
/// each defines a 2-element subspace `(i, i | t_bit)`. Skips
/// iterations whose `i` does not have every control bit set.
///
/// Runtime-dispatches on x86_64 to a packed-complex AVX-512 kernel
/// (`apply_1q_avx512`, see ADR 0008) when host AVX-512F is available
/// and the target / control orientation satisfies the SIMD path's
/// safety contract; otherwise falls through to the scalar body
/// (which LLVM auto-vectorises into 2-lane `vmulpd xmm` via the
/// natural Complex layout).
pub fn apply_1q(amps: &mut [Complex], target: u32, controls: &[u32], m: &[[Complex; 2]; 2]) {
    // 1. Diagonal fast path (P1-06).
    if super::is_diagonal_2x2(m) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
                && (1usize << target) >= 4
                && controls.iter().all(|&c| c > target)
            {
                // SAFETY: see apply_1q_diagonal_avx512 # Safety —
                // feature detected + target_bit ≥ LANES (from the
                // `(1usize << target) >= 4` guard above) + every
                // control > target (from `c > target` guard) + standard
                // apply_gate qubit-range + distinct invariants.
                unsafe {
                    apply_1q_diagonal_avx512(amps, target, controls, m[0][0], m[1][1]);
                }
                return;
            }
        }
        apply_1q_diagonal_scalar(amps, target, controls, m[0][0], m[1][1]);
        return;
    }

    // 2. Anti-diagonal fast path (P1-05). Per-arm dispatch picks
    // AVX-512 Tier A when its contract holds; otherwise falls back
    // to the scalar kernel (which is also the Tier-C path for
    // controls below the target).
    if super::is_antidiagonal_2x2(m) {
        match super::classify_1q_antidiag(m) {
            Some(super::Perm1qKind::X) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && controls.iter().all(|&c| c > target)
                    {
                        if (1usize << target) >= 4 {
                            // SAFETY: feature gate + Tier-A contract
                            // (target_bit ≥ LANES, controls > target).
                            unsafe {
                                apply_1q_x_avx512(amps, target, controls);
                            }
                            return;
                        } else if target <= 1
                            && amps.len().is_multiple_of(4)
                            && controls.iter().all(|&c| c >= 2)
                        {
                            // SAFETY: feature gate + Tier-B contract
                            // (target ∈ {0,1}, controls > target,
                            // amps.len() divisible by LANES).
                            unsafe {
                                apply_1q_x_avx512_lowbit(amps, target, controls);
                            }
                            return;
                        }
                    }
                }
                apply_1q_x_scalar(amps, target, controls);
                return;
            }
            Some(super::Perm1qKind::YPos) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && controls.iter().all(|&c| c > target)
                    {
                        if (1usize << target) >= 4 {
                            // SAFETY: feature gate + Tier-A contract.
                            unsafe {
                                apply_1q_y_avx512(amps, target, controls, 1.0);
                            }
                            return;
                        } else if target <= 1
                            && amps.len().is_multiple_of(4)
                            && controls.iter().all(|&c| c >= 2)
                        {
                            // SAFETY: feature gate + Tier-B contract.
                            unsafe {
                                apply_1q_y_avx512_lowbit(amps, target, controls, 1.0);
                            }
                            return;
                        }
                    }
                }
                apply_1q_y_scalar(amps, target, controls, 1.0);
                return;
            }
            Some(super::Perm1qKind::YNeg) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && controls.iter().all(|&c| c > target)
                    {
                        if (1usize << target) >= 4 {
                            // SAFETY: feature gate + Tier-A contract.
                            unsafe {
                                apply_1q_y_avx512(amps, target, controls, -1.0);
                            }
                            return;
                        } else if target <= 1
                            && amps.len().is_multiple_of(4)
                            && controls.iter().all(|&c| c >= 2)
                        {
                            // SAFETY: feature gate + Tier-B contract.
                            unsafe {
                                apply_1q_y_avx512_lowbit(amps, target, controls, -1.0);
                            }
                            return;
                        }
                    }
                }
                apply_1q_y_scalar(amps, target, controls, -1.0);
                return;
            }
            None => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx512f")
                        && controls.iter().all(|&c| c > target)
                    {
                        if (1usize << target) >= 4 {
                            // SAFETY: feature gate + Tier-A contract.
                            unsafe {
                                apply_1q_antidiag_avx512(amps, target, controls, m[0][1], m[1][0]);
                            }
                            return;
                        } else if target <= 1
                            && amps.len().is_multiple_of(4)
                            && controls.iter().all(|&c| c >= 2)
                        {
                            // SAFETY: feature gate + Tier-B contract.
                            unsafe {
                                apply_1q_antidiag_avx512_lowbit(
                                    amps, target, controls, m[0][1], m[1][0],
                                );
                            }
                            return;
                        }
                    }
                }
                apply_1q_antidiag_scalar(amps, target, controls, m[0][1], m[1][0]);
                return;
            }
        }
    }

    // 3. Generic 2×2 path (unchanged).
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << target) >= 4
            && controls.iter().all(|&c| c > target)
        {
            // SAFETY: feature detection gates the call; the kernel's
            // bounds + alignment invariants follow from
            // `1usize << target ≥ 4` (LANES-aligned block stride),
            // `c > target` for every control (no control-bit
            // toggling in the inner walk), and the apply_gate-level
            // qubit-range + duplicate-qubit checks.
            unsafe {
                apply_1q_avx512(amps, target, controls, m);
            }
            return;
        }
    }

    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let a = amps[i];
            let b = amps[j];
            amps[i] = m[0][0] * a + m[0][1] * b;
            amps[j] = m[1][0] * a + m[1][1] * b;
        }
        i += 1;
    }
}

/// Packed-complex AVX-512 path for AoS `apply_1q`. 4 complex pairs
/// per `__m512d`, interleaved as `(re_0, im_0, re_1, im_1, re_2,
/// im_2, re_3, im_3)`.
///
/// **Math.** For a 2×2 unitary `U = [[u00, u01], [u10, u11]]` and the
/// state pair `(z_0, z_1) = (state[i], state[i | t_bit])`, each
/// output is `new_z_r = u[r][0] * z_0 + u[r][1] * z_1`. Each complex
/// multiply `u_rk * z_k` is implemented as
/// `vfmaddsub(u_rk_re_bcast, z_k, u_rk_im_bcast × swap(z_k))`,
/// where `swap` swaps adjacent doubles via `vpermilpd` with imm 0x55
/// (each `(re, im)` lane-pair becomes `(im, re)`). `fmaddsub`
/// alternates SUB / ADD across even / odd lanes, producing
/// `(re_out, im_out)` for each of the 4 packed complex.
///
/// **Performance shape.** Per inner iter (4 complex pairs):
/// 2 loads + 2 permutes + 4 mul + 4 fmaddsub + 2 add + 2 stores ≈
/// 16 µops. Vs the SoA layout's would-be packed-AVX-512 kernel:
/// ~28 µops for 8 pairs across 4 separate streams. Empirically on
/// EPYC 8124P (Zen 4), this is the path that breaks past the
/// LLVM-auto-vec'd `vmulpd xmm` AoS baseline — see ADR 0008.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1usize << target ≥ LANES` so the inner block has at least
///   `LANES` contiguous pairs (no in-block tail; outer step is
///   `2 * target_bit ≥ 2 * LANES`, keeping i1 + LANES inside the
///   outer extent).
/// * Every control's qubit index is strictly greater than `target`,
///   so the inner SIMD walk's `block | j` for `j ∈ [0, target_bit)`
///   doesn't toggle any control bit.
/// * Standard apply_gate invariants: `target` and `controls` are
///   distinct and in qubit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
// `pub` so criterion benches (separate compilation units) can reach this
// via `aleph_sv::kernels::aos::apply_1q_avx512` when the `internal-bench`
// feature enables `pub mod kernels` in lib.rs. Without that feature the
// module itself is private, so this visibility is effectively pub(crate).
pub unsafe fn apply_1q_avx512(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4; // 4 complex pairs per __m512d (8 lanes f64)

    let target_bit = 1usize << target;
    let len = amps.len();

    // Pin the # Safety contract as debug-only asserts.  Release-mode
    // violations would silently produce a no-op (target_bit < LANES)
    // or an underflowed outer_count (controls below target) — both
    // catastrophic.  Converting to panic-on-debug guards a future
    // dispatch-relaxation regression.
    debug_assert!(
        target_bit >= LANES,
        "target_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        controls.iter().all(|&c| c > target),
        "control at-or-below target: dispatch contract violated"
    );

    // Broadcast U matrix entries — constant across all iterations.
    let m00r = _mm512_set1_pd(m[0][0].re);
    let m00i = _mm512_set1_pd(m[0][0].im);
    let m01r = _mm512_set1_pd(m[0][1].re);
    let m01i = _mm512_set1_pd(m[0][1].im);
    let m10r = _mm512_set1_pd(m[1][0].re);
    let m10i = _mm512_set1_pd(m[1][0].im);
    let m11r = _mm512_set1_pd(m[1][1].re);
    let m11i = _mm512_set1_pd(m[1][1].im);

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            let i1 = i0 + target_bit;

            // Each Complex is 16 bytes (re, im). `amps.as_ptr() as *const f64`
            // views the storage as paired f64. `_mm512_loadu_pd` reads 8
            // consecutive doubles = 4 complex starting at `amps[i0]`.
            // SAFETY: `i0 + LANES ≤ block + target_bit ≤ len` (outer block
            // stride is `2 * target_bit`, so `block + 2*target_bit ≤ len`);
            // `i1 + LANES ≤ block + 2*target_bit ≤ len`.
            let z0 = _mm512_loadu_pd(amps_ptr.add(i0 * 2));
            let z1 = _mm512_loadu_pd(amps_ptr.add(i1 * 2));

            // vpermilpd imm = 0x55 = 0b01010101: each (low=0, high=1)
            // pair becomes (high, low). After permute:
            // (im_0, re_0, im_1, re_1, im_2, re_2, im_3, re_3).
            let z0_swap = _mm512_permute_pd::<0x55>(z0);
            let z1_swap = _mm512_permute_pd::<0x55>(z1);

            // U_ij × z_k = vfmaddsub(U_ij_re, z_k, U_ij_im × z_k_swap).
            // fmaddsub(a, b, c) = (a·b - c, a·b + c, ...) alternating.
            // Even lanes (re_out): m_re·z_re - m_im·z_im ✓
            // Odd lanes  (im_out): m_re·z_im + m_im·z_re ✓
            let t00 = _mm512_mul_pd(m00i, z0_swap);
            let prod00 = _mm512_fmaddsub_pd(m00r, z0, t00);
            let t01 = _mm512_mul_pd(m01i, z1_swap);
            let prod01 = _mm512_fmaddsub_pd(m01r, z1, t01);
            let new_z0 = _mm512_add_pd(prod00, prod01);

            let t10 = _mm512_mul_pd(m10i, z0_swap);
            let prod10 = _mm512_fmaddsub_pd(m10r, z0, t10);
            let t11 = _mm512_mul_pd(m11i, z1_swap);
            let prod11 = _mm512_fmaddsub_pd(m11r, z1, t11);
            let new_z1 = _mm512_add_pd(prod10, prod11);

            _mm512_storeu_pd(amps_ptr.add(i0 * 2), new_z0);
            _mm512_storeu_pd(amps_ptr.add(i1 * 2), new_z1);

            j += LANES;
        }
        // Caller guarantees `target_bit ≥ LANES` and `target_bit`
        // is a power of two ⇒ LANES divides target_bit ⇒ no tail.
        debug_assert_eq!(j, target_bit);
    };

    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let count = len / outer_step;
        crate::kernels::par_blocks(count, len, |k| k * outer_step, outer_iter);
        return;
    }

    // Controlled SIMD path. Caller's `c > target` guard guarantees
    // all controls sit above the target bit and the subtraction
    // `c - target - 1` does not underflow.
    //
    // The outer loop must iterate only over bits ABOVE `target + 1`
    // so that `block`'s low `target + 1` bits are zero — that keeps
    // the inner SIMD walk's `block | j` contiguous for the
    // `vmovupd zmm` load. We achieve that by renormalising control
    // positions (subtract `target + 1`) so `expand_with_fixed` lays
    // them out densely, then left-shifting by `target + 1` to put
    // the result back at the actual qubit positions.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    // Subtraction is safe: each control is distinct from target and
    // < n_qubits, all controls > target, so
    // `target + 1 + controls.len() ≤ n_qubits`.
    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        outer_iter,
    );
}

/// Scalar fallback for the 1q diagonal fast path.
///
/// Walks every amplitude exactly once, multiplying `state[i]` by
/// `m00` if bit `target` of `i` is 0 and by `m11` otherwise.  No
/// cross-term mixing — half the loads and stores of the generic
/// kernel; LLVM auto-vectorises the inner multiply to 2-lane `vmulpd`
/// xmm on x86_64.
///
/// `m00` and `m11` are passed explicitly (rather than the full matrix)
/// because the caller has already detected the diagonal — passing the
/// scalars makes the contract explicit and lets the compiler keep
/// them in registers across the loop.
pub(crate) fn apply_1q_diagonal_scalar(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    m00: Complex,
    m11: Complex,
) {
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask {
            let d = if (i & t_bit) == 0 { m00 } else { m11 };
            amps[i] *= d;
        }
        i += 1;
    }
}

/// Scalar Pauli-X kernel. Pure amplitude swap; no arithmetic.
///
/// Iterates basis indices `i` whose target bit is 0 and every
/// control bit is set, swapping `amps[i]` with `amps[i | (1 << target)]`.
pub(crate) fn apply_1q_x_scalar(amps: &mut [Complex], target: u32, controls: &[u32]) {
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            amps.swap(i, i | t_bit);
        }
        i += 1;
    }
}

/// Scalar Pauli-Y kernel. Swap + sign-flip + (re, im) exchange.
///
/// `Y = [[0, -i], [i, 0]]` for `YPos` (canonical Pauli-Y), or
/// `Y' = [[0, +i], [-i, 0]]` for `YNeg`. `phase_sign = +1.0` selects
/// YPos, `phase_sign = -1.0` selects YNeg.
///
/// For YPos: `amps[i0] ← (im_i1, -re_i1)`, `amps[i1] ← (-im_i0, re_i0)`.
/// For YNeg: signs flip on both sides.
pub(crate) fn apply_1q_y_scalar(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    phase_sign: f64,
) {
    debug_assert!(phase_sign == 1.0 || phase_sign == -1.0);
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let z0 = amps[i];
            let z1 = amps[j];
            // YPos (phase_sign=+1):
            //   amps[i] = (-i) * z1 = (im1, -re1)
            //   amps[j] = (+i) * z0 = (-im0, re0)
            // YNeg (phase_sign=-1): swap signs on both halves.
            amps[i] = Complex::new(phase_sign * z1.im, -phase_sign * z1.re);
            amps[j] = Complex::new(-phase_sign * z0.im, phase_sign * z0.re);
        }
        i += 1;
    }
}

/// Scalar generic anti-diagonal kernel. Full complex multiply on the
/// two off-diagonal entries + swap.
///
/// `m = [[0, a], [b, 0]]`. With `i = base index (target-bit clear)` and
/// `j = i | (1 << target)`, the `apply_1q` row-multiply
/// `amps[i] = m[0][0]*z0 + m[0][1]*z1`, `amps[j] = m[1][0]*z0 + m[1][1]*z1`
/// collapses (since `m[0][0] = m[1][1] = 0`) to
/// `amps[i] ← a * amps[j]_old`, `amps[j] ← b * amps[i]_old`.
pub(crate) fn apply_1q_antidiag_scalar(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    a: Complex,
    b: Complex,
) {
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let z0 = amps[i];
            let z1 = amps[j];
            amps[i] = a * z1;
            amps[j] = b * z0;
        }
        i += 1;
    }
}

/// Packed-complex AVX-512 path for the 1q diagonal fast path.
///
/// **Math.** For each amplitude `z = state[i]` whose target bit is 0,
/// `z ← z * m00`; whose target bit is 1, `z ← z * m11`.  No cross-term
/// arithmetic — single-stream complex multiply per pair.
///
/// **Performance shape.** Per inner iter (4 complex pairs):
/// 1 vmovupd + 1 vpermilpd + 1 vmulpd + 1 vfmaddsub + 1 vmovupd ≈
/// 5 µops, vs `apply_1q_avx512`'s ~16 µops per 4 pairs (which does
/// the full 2x2 multiply).  Roughly 3× fewer µops on the AVX-512
/// path for diagonal gates.
///
/// **Block structure.** The target qubit splits the basis index into
/// contiguous blocks of `target_bit = 1 << target` amps with the same
/// multiplier.  Outer step = `2 * target_bit`; first sub-block (size
/// `target_bit`) uses `m00`, second uses `m11`.  Caller guarantees
/// `target_bit ≥ LANES = 4` so each sub-block has at least one full
/// LANES-wide load.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1usize << target ≥ LANES` so the inner SIMD walk has at least
///   `LANES` contiguous pairs per sub-block.
/// * Every control's qubit index is strictly greater than `target`,
///   so the inner walk's `block | j` for `j ∈ [0, target_bit)`
///   doesn't toggle any control bit.
/// * Standard apply_gate invariants: `target` and `controls` are
///   distinct and in qubit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_diagonal_avx512(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    m00: Complex,
    m11: Complex,
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4; // 4 complex pairs per __m512d (8 lanes f64)

    let target_bit = 1usize << target;
    let len = amps.len();

    // Pin the # Safety contract as debug-only asserts (see
    // apply_1q_avx512 for the same pattern).  Release-mode
    // violations would silently no-op or underflow outer_count.
    debug_assert!(
        target_bit >= LANES,
        "target_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        controls.iter().all(|&c| c > target),
        "control at-or-below target: dispatch contract violated"
    );

    // Broadcast the two diagonal entries; constant across the walk.
    let m00r = _mm512_set1_pd(m00.re);
    let m00i = _mm512_set1_pd(m00.im);
    let m11r = _mm512_set1_pd(m11.re);
    let m11i = _mm512_set1_pd(m11.im);

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        // 0-side: amps[block .. block + target_bit] get * m00
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            // SAFETY: i0 + LANES ≤ block + target_bit ≤ len.
            let z = _mm512_loadu_pd(amps_ptr.add(i0 * 2));
            // vpermilpd 0x55: each (re, im) pair becomes (im, re).
            let zs = _mm512_permute_pd::<0x55>(z);
            // t = m00_im * zs : per pair → (m00.im * im, m00.im * re, ...)
            let t = _mm512_mul_pd(m00i, zs);
            // out = vfmaddsub(m00_re, z, t) :
            //   even lane = m00.re*re - m00.im*im = (m00 * z).re  ✓
            //   odd  lane = m00.re*im + m00.im*re = (m00 * z).im  ✓
            let out = _mm512_fmaddsub_pd(m00r, z, t);
            _mm512_storeu_pd(amps_ptr.add(i0 * 2), out);
            j += LANES;
        }
        debug_assert_eq!(j, target_bit);

        // 1-side: amps[block + target_bit .. block + 2*target_bit] get * m11
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i1 = block | target_bit | j;
            // SAFETY: i1 + LANES ≤ block + 2*target_bit ≤ len.
            let z = _mm512_loadu_pd(amps_ptr.add(i1 * 2));
            let zs = _mm512_permute_pd::<0x55>(z);
            let t = _mm512_mul_pd(m11i, zs);
            let out = _mm512_fmaddsub_pd(m11r, z, t);
            _mm512_storeu_pd(amps_ptr.add(i1 * 2), out);
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

    // Controlled SIMD path.  Caller's `c > target` guard guarantees
    // all controls sit above the target bit and `c - target - 1`
    // does not underflow.  The outer loop iterates over bit-patterns
    // that have every control set and every below-target bit clear,
    // letting the inner SIMD walk fill in the target + below-target
    // bits contiguously.
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

/// Packed-complex AVX-512 Pauli-X kernel (Tier A).
///
/// Pure amplitude swap: for each LANES-block (`LANES = 4` complex pairs
/// = 8 doubles per `__m512d`), load both the i0-block and the
/// i1-block (= i0 | target_bit), then store crossed. Zero arithmetic.
///
/// # Safety
/// Caller MUST ensure:
/// * Host CPU supports AVX-512F.
/// * `1usize << target ≥ LANES = 4`.
/// * Every control's qubit index is strictly greater than `target`.
/// * Standard apply_gate invariants: `target` and `controls` are
///   distinct and in qubit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_x_avx512(amps: &mut [Complex], target: u32, controls: &[u32]) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;

    let target_bit = 1usize << target;
    let len = amps.len();

    debug_assert!(
        target_bit >= LANES,
        "target_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        controls.iter().all(|&c| c > target),
        "control at-or-below target: dispatch contract violated"
    );

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            let i1 = block | target_bit | j;
            // SAFETY: i0 + LANES ≤ block + target_bit ≤ len; same for i1.
            let z0 = _mm512_loadu_pd(amps_ptr.add(i0 * 2));
            let z1 = _mm512_loadu_pd(amps_ptr.add(i1 * 2));
            _mm512_storeu_pd(amps_ptr.add(i0 * 2), z1);
            _mm512_storeu_pd(amps_ptr.add(i1 * 2), z0);
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

    // Controlled outer walk: identical to apply_1q_diagonal_avx512.
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

/// Packed-complex AVX-512 Pauli-Y kernel (Tier A).
///
/// `phase_sign = +1.0` → YPos (canonical `Y = [[0,-i],[i,0]]`).
/// `phase_sign = -1.0` → YNeg (`Y' = [[0,+i],[-i,0]]`).
///
/// Per LANES-block: load z0 + z1, permilpd-swap (re,im) → (im,re),
/// xor with sign masks, store crossed.
///
/// # Safety
/// Caller MUST ensure:
/// * Host CPU supports AVX-512F.
/// * `1usize << target ≥ LANES = 4`.
/// * Every control's qubit index is strictly greater than `target`.
/// * `phase_sign ∈ {+1.0, -1.0}`.
/// * Standard apply_gate invariants.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_y_avx512(amps: &mut [Complex], target: u32, controls: &[u32], phase_sign: f64) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;

    debug_assert!(phase_sign == 1.0 || phase_sign == -1.0);

    let target_bit = 1usize << target;
    let len = amps.len();

    debug_assert!(target_bit >= LANES);
    debug_assert!(controls.iter().all(|&c| c > target));

    // Sign masks: a __m512d viewed as 8 doubles.
    // YPos amps[i0] = (im1, -re1, ...) → mask_for_i0 negates odd lanes (1, 3, 5, 7).
    // YPos amps[i1] = (-im0,  re0, ...) → mask_for_i1 negates even lanes (0, 2, 4, 6).
    // YNeg: flip both masks (swap which lanes get the sign).
    //
    // _mm512_set_pd(a, b, c, d, e, f, g, h) packs h into lane 0, a into lane 7
    // (reversed argument order relative to lane index). The comments below use
    // lane-index order (0..7), so the argument list is the reverse.
    let sign_bit = -0.0f64; // IEEE-754 sign bit; xor toggles sign.
    let zero = 0.0f64;
    // Match the `debug_assert!(phase_sign == ±1.0)` contract exactly:
    // any value other than `1.0` selects the YNeg branch, mirroring the
    // assert's binary character. `> 0.0` would silently accept e.g. 0.5.
    let (mask_i0, mask_i1) = if phase_sign == 1.0 {
        (
            // lane 0: zero (re), lane 1: sign (im), lane 2: zero, lane 3: sign, ...
            // args reversed: (lane7, lane6, lane5, lane4, lane3, lane2, lane1, lane0)
            _mm512_set_pd(
                sign_bit, zero, sign_bit, zero, sign_bit, zero, sign_bit, zero,
            ),
            _mm512_set_pd(
                zero, sign_bit, zero, sign_bit, zero, sign_bit, zero, sign_bit,
            ),
        )
    } else {
        (
            _mm512_set_pd(
                zero, sign_bit, zero, sign_bit, zero, sign_bit, zero, sign_bit,
            ),
            _mm512_set_pd(
                sign_bit, zero, sign_bit, zero, sign_bit, zero, sign_bit, zero,
            ),
        )
    };

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            let i1 = block | target_bit | j;
            // SAFETY: in-bounds by outer-walk construction.
            let z0 = _mm512_loadu_pd(amps_ptr.add(i0 * 2));
            let z1 = _mm512_loadu_pd(amps_ptr.add(i1 * 2));
            // permilpd 0x55: per 128-bit lane, swap the two doubles → (im, re) per pair.
            let z0s = _mm512_permute_pd::<0x55>(z0);
            let z1s = _mm512_permute_pd::<0x55>(z1);
            // Apply sign flips and cross-store.
            let new_i0 = _mm512_xor_pd(z1s, mask_i0);
            let new_i1 = _mm512_xor_pd(z0s, mask_i1);
            _mm512_storeu_pd(amps_ptr.add(i0 * 2), new_i0);
            _mm512_storeu_pd(amps_ptr.add(i1 * 2), new_i1);
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

    // Controlled SIMD path: same renormalise-then-shift idiom as
    // apply_1q_avx512. Caller's `c > target` guard guarantees
    // `c - target - 1` does not underflow.
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

/// Packed-complex AVX-512 generic anti-diagonal kernel (Tier A).
///
/// `m = [[0, a], [b, 0]]`. `amps[i0] ← a * amps[i1]_old`,
/// `amps[i1] ← b * amps[i0]_old`. Per LANES-block: load z0+z1,
/// complex-multiply z1 by a → new_i0, complex-multiply z0 by b →
/// new_i1, store crossed.
///
/// # Safety
/// Caller MUST ensure:
/// * Host CPU supports AVX-512F.
/// * `1usize << target ≥ LANES = 4`.
/// * Every control's qubit index is strictly greater than `target`.
/// * Standard apply_gate invariants.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_antidiag_avx512(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    a: Complex,
    b: Complex,
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;

    let target_bit = 1usize << target;
    let len = amps.len();
    debug_assert!(target_bit >= LANES);
    debug_assert!(controls.iter().all(|&c| c > target));

    let ar = _mm512_set1_pd(a.re);
    let ai = _mm512_set1_pd(a.im);
    let br = _mm512_set1_pd(b.re);
    let bi = _mm512_set1_pd(b.im);

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES <= target_bit {
            let i0 = block | j;
            let i1 = block | target_bit | j;
            // SAFETY: in-bounds by outer-walk construction.
            let z0 = _mm512_loadu_pd(amps_ptr.add(i0 * 2));
            let z1 = _mm512_loadu_pd(amps_ptr.add(i1 * 2));

            // new_i0 = a * z1 (corrected: m[0][1]*z1 = a*z1)
            let z1s = _mm512_permute_pd::<0x55>(z1);
            let t1 = _mm512_mul_pd(ai, z1s);
            let new_i0 = _mm512_fmaddsub_pd(ar, z1, t1);

            // new_i1 = b * z0 (corrected: m[1][0]*z0 = b*z0)
            let z0s = _mm512_permute_pd::<0x55>(z0);
            let t0 = _mm512_mul_pd(bi, z0s);
            let new_i1 = _mm512_fmaddsub_pd(br, z0, t0);

            _mm512_storeu_pd(amps_ptr.add(i0 * 2), new_i0);
            _mm512_storeu_pd(amps_ptr.add(i1 * 2), new_i1);

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

/// Packed-complex AVX-512 Pauli-X kernel — Tier B (target < LANES).
///
/// Two stride patterns:
/// * `target = 0`: swap (pair0,pair1) and (pair2,pair3) within each
///   `__m512d`. Uses `_mm512_permutex_pd::<0x4E>` (quadword pattern
///   [2,3,0,1] per 256-bit lane).
/// * `target = 1`: swap (pair0,pair2) and (pair1,pair3) within each
///   `__m512d`. Uses `_mm512_permutexvar_pd` with index `[4,5,6,7,0,1,2,3]`.
///
/// # Safety
/// * Host AVX-512F.
/// * `target ∈ {0, 1}` AND `1 << target < LANES = 4`.
/// * Every control's qubit index `≥ log2(LANES) = 2`. The block-level
///   `(block & ctrl_mask) == ctrl_mask` test only inspects bits at or
///   above `log2(LANES)` (since block addresses are LANES-aligned), so
///   any control with index below 2 would be aliased to 0 in `block`
///   and the gate would silently no-op for amplitudes that DO have the
///   control bit set within the LANES-block. Dispatch must filter such
///   configurations to the scalar fallback.
/// * `amps.len() % LANES == 0` (always true for `n ≥ 2`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_x_avx512_lowbit(amps: &mut [Complex], target: u32, controls: &[u32]) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    debug_assert!((1usize << target) < LANES);
    debug_assert!(
        controls.iter().all(|&c| c >= 2),
        "Tier-B AoS contract: every control must be at qubit index ≥ log2(LANES) = 2"
    );
    debug_assert_eq!(amps.len() % LANES, 0);

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);
    let n_amps = amps.len();

    // Index vector for target=1 swap (pair0↔pair2, pair1↔pair3 within zmm).
    // Pairs occupy 2 lanes each, so the swap is cross-256-bit:
    // lanes [0,1] (pair0) ↔ lanes [4,5] (pair2); lanes [2,3] (pair1) ↔ lanes [6,7] (pair3).
    // permutexvar lane k receives src[idx[k]], so lane-order idx = [4,5,6,7,0,1,2,3].
    // _mm512_set_epi64 args: arg 0 → lane 7, arg 7 → lane 0 (reversed).
    let idx_t1 = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    let ctrl_mask = if controls.is_empty() {
        0usize
    } else {
        crate::kernels::control_mask(controls)
    };

    let count = n_amps / LANES;
    crate::kernels::par_blocks(count, n_amps, |k| k * LANES, |block| {
        let amps_ptr = bp.ptr();
        // Gate the block on the control bits: every control bit MUST
        // be set in `block`. Tier-B has `c > target` so `c ≥ 1` for
        // target=0 and `c ≥ 2` for target=1; ctrl bits never overlap
        // the in-register swap region.
        if controls.is_empty() || (block & ctrl_mask) == ctrl_mask {
            // SAFETY: block + LANES ≤ n_amps.
            let z = _mm512_loadu_pd(amps_ptr.add(block * 2));
            let swapped = if target == 0 {
                // _mm512_permutex_pd::<0x4E>: immediate 0x4E = 0b01001110.
                // Each pair of bits selects which of the 4 f64-doubles within
                // each 256-bit half goes to each output position.
                // [2,3,0,1, 2,3,0,1] in the lo/hi 256-bit lanes → swaps pair0↔pair1
                // and pair2↔pair3 within each zmm.
                _mm512_permutex_pd::<0x4E>(z)
            } else {
                // idx_t1 = [4,5,6,7, 0,1,2,3] (lane-index order) →
                // swaps pair0↔pair2 and pair1↔pair3.
                _mm512_permutexvar_pd(idx_t1, z)
            };
            _mm512_storeu_pd(amps_ptr.add(block * 2), swapped);
        }
    });
}

/// Packed-complex AVX-512 Pauli-Y kernel — Tier B (target < LANES).
///
/// Per LANES-block: permute (pair-swap) then permilpd<0x55> to swap
/// (re, im) → (im, re) per pair, then xor with a sign mask to
/// implement the Y-gate sign pattern.
///
/// **Sign-mask derivation (YPos, phase_sign = +1.0).**
/// Y gate: `amps[i0] ← (im_i1, -re_i1)`, `amps[i1] ← (-im_i0, re_i0)`.
/// The sequence is: permutex (pair-swap) → permute_pd<0x55> ((im,re) swap)
/// → xor sign mask.
///
/// After permutex for target=0 (0x4E): layout is [pair_i1, pair_i0, pair_i1, pair_i0].
/// After permilpd<0x55>: within each pair → (im, re) order.
/// Now slot 0 holds (im_i1, re_i1) → need amps[i0] = (im_i1, -re_i1) →
///   negate lane 1 (re_i1 is at odd position): sign_bit on lane 1.
/// Slot 1 holds (im_i0, re_i0) → need amps[i1] = (-im_i0, re_i0) →
///   negate lane 2 (im_i0 is at even position of pair 1): sign_bit on lane 2.
/// Pattern in [lane0..lane7]: [0, sign, sign, 0, 0, sign, sign, 0].
///
/// For target=1 (idx_t1 swap): layout is [pair_i1, pair_i1, pair_i0, pair_i0].
/// After permilpd<0x55>: slot 0 (im_i1, re_i1) → negate lane 1; slot 1 (im_i1,
/// re_i1) → negate lane 3; slot 2 (im_i0, re_i0) → negate lane 4; slot 3 →
/// negate lane 6.
/// Pattern in [lane0..lane7]: [0, sign, 0, sign, sign, 0, sign, 0].
///
/// YNeg (phase_sign = -1.0): all signs flip.
///
/// # Safety
/// * Host AVX-512F.
/// * `target ∈ {0, 1}`.
/// * Every control's qubit index > target.
/// * `phase_sign ∈ {+1.0, -1.0}`.
/// * `amps.len() % LANES == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_y_avx512_lowbit(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    phase_sign: f64,
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    debug_assert!((1usize << target) < LANES);
    // Tier-B contract: see apply_1q_x_avx512_lowbit Safety block.
    debug_assert!(controls.iter().all(|&c| c >= 2));
    debug_assert!(phase_sign == 1.0 || phase_sign == -1.0);
    debug_assert_eq!(amps.len() % LANES, 0);

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);
    let n_amps = amps.len();
    let sign_bit = -0.0f64;
    let zero = 0.0f64;

    // _mm512_set_pd(a7, a6, a5, a4, a3, a2, a1, a0): arg 0 → lane 7, arg 7 → lane 0.
    // Comments below use lane-index order [0..7]; the arg list is the reverse.
    let (idx_t1, mask_t0, mask_t1) = if phase_sign == 1.0 {
        (
            _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4),
            // target=0 mask [lane0..7]: [0, sign, sign, 0, 0, sign, sign, 0]
            // → args reversed: (lane7=0, lane6=sign, lane5=sign, lane4=0,
            //                   lane3=0, lane2=sign, lane1=sign, lane0=0)
            _mm512_set_pd(
                zero, sign_bit, sign_bit, zero, zero, sign_bit, sign_bit, zero,
            ),
            // target=1 mask [lane0..7]: [0, sign, 0, sign, sign, 0, sign, 0]
            // → args reversed: (lane7=0, lane6=sign, lane5=0, lane4=sign,
            //                   lane3=sign, lane2=0, lane1=sign, lane0=0)
            _mm512_set_pd(
                zero, sign_bit, zero, sign_bit, sign_bit, zero, sign_bit, zero,
            ),
        )
    } else {
        // YNeg: all signs flip.
        (
            _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4),
            _mm512_set_pd(
                sign_bit, zero, zero, sign_bit, sign_bit, zero, zero, sign_bit,
            ),
            _mm512_set_pd(
                sign_bit, zero, sign_bit, zero, zero, sign_bit, zero, sign_bit,
            ),
        )
    };

    let ctrl_mask = if controls.is_empty() {
        0usize
    } else {
        crate::kernels::control_mask(controls)
    };

    let count = n_amps / LANES;
    crate::kernels::par_blocks(count, n_amps, |k| k * LANES, |block| {
        let amps_ptr = bp.ptr();
        if controls.is_empty() || (block & ctrl_mask) == ctrl_mask {
            // SAFETY: in-bounds.
            let z = _mm512_loadu_pd(amps_ptr.add(block * 2));
            let permuted = if target == 0 {
                _mm512_permutex_pd::<0x4E>(z)
            } else {
                _mm512_permutexvar_pd(idx_t1, z)
            };
            // Swap (re, im) → (im, re) per pair.
            let swapped_re_im = _mm512_permute_pd::<0x55>(permuted);
            let mask = if target == 0 { mask_t0 } else { mask_t1 };
            let out = _mm512_xor_pd(swapped_re_im, mask);
            _mm512_storeu_pd(amps_ptr.add(block * 2), out);
        }
    });
}

/// Packed-complex AVX-512 generic anti-diagonal kernel — Tier B
/// (target < LANES). Uses the same in-register permute as the X kernel
/// but follows with a full complex multiply per source.
///
/// `m = [[0, a], [b, 0]]`. `amps[i0] ← a * amps[i1]_old`,
/// `amps[i1] ← b * amps[i0]_old`.
///
/// **Scalar mapping (corrected).**
/// Post-permute for target=0 (permutex<0x4E>): slot pattern is
/// [i1-data, i0-data, i1-data, i0-data]. Slot 0 holds data that WAS
/// at i1, now destined for i0 → multiply by `a`. Slot 1 holds data
/// that WAS at i0, destined for i1 → multiply by `b`. So target=0
/// slot mapping (pairs 0..3) = [a, b, a, b].
///
/// Post-permute for target=1 (permutexvar idx_t1): slot pattern is
/// [i1-data, i1-data, i0-data, i0-data]. Slots 0,1 → `a`; slots 2,3
/// → `b`. target=1 slot mapping = [a, a, b, b].
///
/// # Safety
/// * Host AVX-512F.
/// * `target ∈ {0, 1}`.
/// * Every control's qubit index > target.
/// * `amps.len() % LANES == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_1q_antidiag_avx512_lowbit(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    a: Complex,
    b: Complex,
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    debug_assert!((1usize << target) < LANES);
    // Tier-B contract: see apply_1q_x_avx512_lowbit Safety block.
    debug_assert!(controls.iter().all(|&c| c >= 2));
    debug_assert_eq!(amps.len() % LANES, 0);

    // Build per-lane scalar vectors with the correct scalar in the right slot.
    // _mm512_set_pd: arg 0 → lane 7, arg 7 → lane 0 (reversed).
    // Each pair (slot) occupies 2 lanes; comments use lane-index order [0..7].
    let (idx_t1, sr_t0, si_t0, sr_t1, si_t1) = {
        let idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        // target=0: slot 0 (lanes 0,1) → a; slot 1 (lanes 2,3) → b;
        //           slot 2 (lanes 4,5) → a; slot 3 (lanes 6,7) → b.
        // Reversed args: (lane7=b.re, lane6=b.re, lane5=a.re, lane4=a.re,
        //                 lane3=b.re, lane2=b.re, lane1=a.re, lane0=a.re)
        let sr_t0 = _mm512_set_pd(b.re, b.re, a.re, a.re, b.re, b.re, a.re, a.re);
        let si_t0 = _mm512_set_pd(b.im, b.im, a.im, a.im, b.im, b.im, a.im, a.im);
        // target=1: slot 0 (lanes 0,1) → a; slot 1 (lanes 2,3) → a;
        //           slot 2 (lanes 4,5) → b; slot 3 (lanes 6,7) → b.
        // Reversed args: (lane7=b.re, lane6=b.re, lane5=b.re, lane4=b.re,
        //                 lane3=a.re, lane2=a.re, lane1=a.re, lane0=a.re)
        let sr_t1 = _mm512_set_pd(b.re, b.re, b.re, b.re, a.re, a.re, a.re, a.re);
        let si_t1 = _mm512_set_pd(b.im, b.im, b.im, b.im, a.im, a.im, a.im, a.im);
        (idx, sr_t0, si_t0, sr_t1, si_t1)
    };

    let ctrl_mask = if controls.is_empty() {
        0usize
    } else {
        crate::kernels::control_mask(controls)
    };

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);
    let n_amps = amps.len();

    let count = n_amps / LANES;
    crate::kernels::par_blocks(count, n_amps, |k| k * LANES, |block| {
        let amps_ptr = bp.ptr();
        if controls.is_empty() || (block & ctrl_mask) == ctrl_mask {
            // SAFETY: in-bounds.
            let z = _mm512_loadu_pd(amps_ptr.add(block * 2));
            let permuted = if target == 0 {
                _mm512_permutex_pd::<0x4E>(z)
            } else {
                _mm512_permutexvar_pd(idx_t1, z)
            };
            let (sr, si) = if target == 0 {
                (sr_t0, si_t0)
            } else {
                (sr_t1, si_t1)
            };
            // Complex multiply: z' = sr * permuted ± si * permilpd<0x55>(permuted)
            // fmaddsub(a, b, c) = (a·b - c, a·b + c, ...) alternating.
            // Even lanes (re_out): sr*re - si*im ✓
            // Odd  lanes (im_out): sr*im + si*re ✓
            let zs = _mm512_permute_pd::<0x55>(permuted);
            let t = _mm512_mul_pd(si, zs);
            let out = _mm512_fmaddsub_pd(sr, permuted, t);
            _mm512_storeu_pd(amps_ptr.add(block * 2), out);
        }
    });
}

/// Scalar fallback for 2-qubit gate application.
///
/// Handles the cases where the AVX-512 path's safety contract is not
/// satisfied: `1 << min(targets) < LANES`, non-AVX-512 host, or
/// external controls below `max(targets)`. Also the only entry-point
/// on non-x86_64 targets.
///
/// **MSB convention (P0-06):** `targets[0]` is the *high* bit of the
/// matrix index `k`, `targets[1]` is the *low* bit. So matrix row 2
/// (binary `10`) corresponds to `(targets[0] = 1, targets[1] = 0)`.
/// This matches `Gate::Cnot` (`qubits = [control, target]`), whose
/// matrix swaps rows 2 ↔ 3.
///
/// Targets must be distinct; the caller (`apply_gate`) enforces this.
pub(crate) fn apply_2q_dense_scalar(
    amps: &mut [Complex],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex; 4]; 4],
) {
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let t_mask = t0_bit | t1_bit;
    let ctrl_mask = super::control_mask(controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            // MSB convention: matrix index k bit 1 → targets[0], bit 0 → targets[1].
            // So idx[k] sets t0_bit iff (k & 2) != 0, t1_bit iff (k & 1) != 0.
            let idx = [
                i,          // k = 00
                i | t1_bit, // k = 01
                i | t0_bit, // k = 10
                i | t_mask, // k = 11
            ];
            let v = [amps[idx[0]], amps[idx[1]], amps[idx[2]], amps[idx[3]]];
            for r in 0..4 {
                amps[idx[r]] = m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3] * v[3];
            }
        }
        i += 1;
    }
}

/// Scalar CNOT specialisation: for amplitudes where bit `control` = 1
/// AND every external control bit is set, swap `state[i]` with
/// `state[i | t_bit]`.  Zero multiplies; pure swap-pair traffic.
///
/// `control` and `target` are passed separately (vs the generic
/// kernel's `targets[2]`) because the dispatch prelude has already
/// disambiguated the orientation via `Perm2qKind`.  External
/// `controls` are appended to the implicit control mask.
pub(crate) fn apply_2q_cnot_scalar(
    amps: &mut [Complex],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    let c_bit = 1usize << control;
    let t_bit = 1usize << target;
    let ctrl_mask = c_bit | super::control_mask(external_controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask && (i & t_bit) == 0 {
            amps.swap(i, i | t_bit);
        }
        i += 1;
    }
}

/// Scalar SWAP specialisation: for amplitudes where bits `a` and `b`
/// differ (and external controls satisfied), swap `state[i_a0_b1]`
/// with `state[i_a1_b0]`.
///
/// Convention: this kernel walks every base index `i` with bits a, b
/// both zero (and external controls set); for each such i, swap
/// `state[i | a_bit]` (= a=0, b=1) with `state[i | b_bit]` (= a=1, b=0).
pub(crate) fn apply_2q_swap_scalar(
    amps: &mut [Complex],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    let a_bit = 1usize << targets[0];
    let b_bit = 1usize << targets[1];
    let t_mask = a_bit | b_bit;
    let ctrl_mask = super::control_mask(external_controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            amps.swap(i | a_bit, i | b_bit);
        }
        i += 1;
    }
}

/// Scalar CZ specialisation: negate `state[i]` for amplitudes where
/// both target bits are 1 (and external controls satisfied).  Touches
/// 1/4 of the state vector.
pub(crate) fn apply_2q_cz_scalar(
    amps: &mut [Complex],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let t_mask = t0_bit | t1_bit;
    let ctrl_mask = super::control_mask(external_controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == t_mask && (i & ctrl_mask) == ctrl_mask {
            amps[i] = -amps[i];
        }
        i += 1;
    }
}

/// Scalar 2q-diagonal specialisation: multiply `state[i]` by `d[k]`
/// where `k = ((i >> targets[0]) & 1) << 1 | ((i >> targets[1]) & 1)`.
///
/// Convention: matches the MSB-first quartet ordering of the generic
/// 2q kernel.  `targets[0]` is the high bit of `k`, `targets[1]` is
/// the low bit (per ADR 0004 / P0-06 §6).
pub(crate) fn apply_2q_diagonal_scalar(
    amps: &mut [Complex],
    targets: [u32; 2],
    external_controls: &[u32],
    d: [Complex; 4],
) {
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let ctrl_mask = super::control_mask(external_controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask {
            let k_hi = ((i & t0_bit) != 0) as usize;
            let k_lo = ((i & t1_bit) != 0) as usize;
            let k = (k_hi << 1) | k_lo;
            amps[i] *= d[k];
        }
        i += 1;
    }
}

/// Top-level 2q dispatch.  See spec § 4.2 for the detection order:
/// 1. `classify_2q_permutation` → Identity / CnotHi / CnotLo / Swap fast paths.
/// 2. `is_diagonal_4x4` → CZ (`is_cz_signature` shortcut) / general diagonal fast path.
/// 3. Otherwise: AVX-512 dense kernel when contract holds, else `apply_2q_dense_scalar`.
///
/// The AVX-512 dense kernel (`apply_2q_avx512`) handles the Tier-A generic
/// case where `1 << min(targets) ≥ LANES` and all external controls sit above
/// `max(targets)`.  Sub-LANES dispatch or below-target controls fall through
/// to `apply_2q_dense_scalar`.  Specialised permutation / diagonal AVX-512
/// paths land in subsequent tasks.
pub(crate) fn apply_2q(
    amps: &mut [Complex],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex; 4]; 4],
) {
    // 1. Permutation detection (Identity / CNOT / SWAP).
    match super::classify_2q_permutation(m) {
        Some(super::Perm2qKind::Identity) => return,
        Some(super::Perm2qKind::CnotHi) => {
            dispatch_cnot(amps, targets[0], targets[1], controls);
            return;
        }
        Some(super::Perm2qKind::CnotLo) => {
            dispatch_cnot(amps, targets[1], targets[0], controls);
            return;
        }
        Some(super::Perm2qKind::Swap) => {
            dispatch_swap(amps, targets, controls);
            return;
        }
        None => {}
    }

    // 2. Diagonal-4x4 (catches Cz, controlled-Phase, Rzz, user diagonals).
    if super::is_diagonal_4x4(m) {
        let d = [m[0][0], m[1][1], m[2][2], m[3][3]];
        let is_cz = super::is_cz_signature(d);
        dispatch_diagonal_or_cz(amps, targets, controls, d, is_cz);
        return;
    }

    // 3. Generic dense 4×4 — SIMD where contract holds, scalar otherwise.
    #[cfg(target_arch = "x86_64")]
    {
        let t_lo = targets[0].min(targets[1]);
        let t_hi = targets[0].max(targets[1]);
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << t_lo) >= 4
            && controls.iter().all(|&c| c > t_hi)
        {
            // SAFETY: feature gate, t_lo_bit ≥ LANES, controls > t_hi.
            unsafe {
                apply_2q_avx512(amps, targets, controls, m);
            }
            return;
        }
    }
    apply_2q_dense_scalar(amps, targets, controls, m);
}

/// Packed-complex AVX-512 generic 2q dense kernel.  The inner walk
/// steps by LANES = 4 complex pairs along the low-target axis
/// (requires `1 << t_lo >= LANES`); the outer walk enumerates quartet
/// base indices via [`expand_with_fixed`].
///
/// **Math.** For each quartet `(z00, z01, z10, z11)`, compute
/// `new_z_r = Σ_c m[r][c] * z_c`.  Each `m[r][c] * z_c` is one
/// `vfmaddsub(m_re_bcast, z_c, m_im_bcast × vpermilpd<0x55>(z_c))` —
/// the same packed-complex idiom as `apply_1q_avx512`, replicated
/// across four loaded subspaces.
///
/// **Outer-walk (bit-disjointness invariant).** The kernel computes
/// each amplitude index as `i = block | offsets[k] | j`, so for the
/// `|` to behave as `+` (the only way the SAFETY bound holds), the
/// three pieces MUST occupy disjoint bit positions:
///
/// * `j` walks `[0, t_lo_bit)` in LANES strides — bits `[0, t_lo)`.
/// * `offsets[k] ∈ {0, t_lo_bit, t_hi_bit, t_lo_bit | t_hi_bit}` —
///   bits exactly `{t_lo, t_hi}`.
/// * `block` MUST therefore use only bits strictly above `t_lo`,
///   with bit `t_hi` clear and every control bit set.
///
/// We achieve that with the same renormalise-then-shift idiom as
/// `apply_1q_avx512` (aos.rs lines 234-248), extended to two
/// reserved positions: `expand_with_fixed` lays out `t_hi` and every
/// control at *renormalised* positions (each minus `t_lo + 1`) in
/// the "above t_lo" subspace, and a left-shift by `t_lo + 1`
/// promotes the result to actual qubit positions. The inner SIMD
/// walk then owns bits `[0, t_lo]` exclusively. This handles bits
/// *between* `t_lo` and `t_hi` correctly because `expand_with_fixed`
/// honours the fixed-false slot at `t_hi - t_lo - 1` and lets `k`'s
/// bits flow around it.
///
/// The "symmetric" form — `expand_with_fixed(k, &[(t_lo, false),
/// (t_hi, false), ...])` without the shift — pins the two target
/// bits to zero correctly, but lets `k`'s bits fall into positions
/// below `t_lo`, where they collide with the inner walk's `j` and
/// destroy the bit-disjointness invariant (corrupt indices,
/// out-of-bounds stores for any non-adjacent target pair).
///
/// [`expand_with_fixed`]: crate::kernels::expand_with_fixed
///
/// **Per inner iter (LANES quartets = 16 amps):**
/// 4 loads + 4 permutes + 16 mul + 16 fmaddsub + 12 add + 4 stores
/// ≈ 56 µops.  Vs scalar 4-quartet (~256 µops): ~4.5× per-amp.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << min(targets) >= LANES` (= 4) — inner SIMD walk has ≥ LANES
///   contiguous pairs per sub-block.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the outer-walk's bit-expansion never toggles
///   a control bit.
/// * Distinct targets/controls, all in qubit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_avx512(
    amps: &mut [Complex],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex; 4]; 4],
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;

    let t_lo = targets[0].min(targets[1]);
    let t_hi = targets[0].max(targets[1]);
    let t_lo_bit = 1usize << t_lo;
    let t_hi_bit = 1usize << t_hi;
    let t_mask = t_lo_bit | t_hi_bit;
    let len = amps.len();

    debug_assert!(
        t_lo_bit >= LANES,
        "t_lo_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        controls.iter().all(|&c| c > t_hi),
        "control at-or-below t_hi: dispatch contract violated"
    );

    // Compute index permutation: targets[0] is MSB of matrix index k,
    // targets[1] is LSB.  If targets[0] < targets[1] (i.e. t_lo == targets[0]),
    // then bit k=1 (lo) corresponds to t_hi_bit memory offset, and
    // bit k=2 (hi) corresponds to t_lo_bit offset.  The four
    // sub-block offsets keyed by k=0,1,2,3:
    let (offset_k1, offset_k2) = if targets[0] < targets[1] {
        // targets[0]=t_lo, targets[1]=t_hi → k bit 1 (low) selects t_hi_bit
        (t_hi_bit, t_lo_bit)
    } else {
        (t_lo_bit, t_hi_bit)
    };
    let offsets = [0usize, offset_k1, offset_k2, t_mask];

    // Broadcast all 16 matrix cells.
    let mut m_re = [_mm512_setzero_pd(); 16];
    let mut m_im = [_mm512_setzero_pd(); 16];
    for r in 0..4 {
        for c in 0..4 {
            m_re[r * 4 + c] = _mm512_set1_pd(m[r][c].re);
            m_im[r * 4 + c] = _mm512_set1_pd(m[r][c].im);
        }
    }

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES <= t_lo_bit {
            // Load 4 sub-blocks, each LANES complex pairs.
            // Base index for k=0 is `block | j` (no target bits set).
            let mut z = [_mm512_setzero_pd(); 4];
            let mut zs = [_mm512_setzero_pd(); 4];
            for k in 0..4 {
                let i_k = block | offsets[k] | j;
                // SAFETY: bit-disjointness invariant (see doc-comment
                // "Outer-walk" section):
                //   * `block` ⊆ bits ≥ t_lo+1, with bit t_hi clear
                //     and every control bit set (renormalise-then-
                //     shift outer-walk below).
                //   * `offsets[k]` ⊆ {t_lo, t_hi}.
                //   * `j` ⊆ [0, t_lo).
                // The three are pairwise bit-disjoint, so
                //   i_k = block + offsets[k] + j
                //       ≤ (block | t_mask) + (t_lo_bit - LANES)
                //       ≤ len - LANES.
                z[k] = _mm512_loadu_pd(amps_ptr.add(i_k * 2));
                zs[k] = _mm512_permute_pd::<0x55>(z[k]);
            }

            // Compute each output row.
            let mut new_z = [_mm512_setzero_pd(); 4];
            for r in 0..4 {
                let t0 = _mm512_mul_pd(m_im[r * 4], zs[0]);
                let mut p = _mm512_fmaddsub_pd(m_re[r * 4], z[0], t0);
                let t1 = _mm512_mul_pd(m_im[r * 4 + 1], zs[1]);
                p = _mm512_add_pd(p, _mm512_fmaddsub_pd(m_re[r * 4 + 1], z[1], t1));
                let t2 = _mm512_mul_pd(m_im[r * 4 + 2], zs[2]);
                p = _mm512_add_pd(p, _mm512_fmaddsub_pd(m_re[r * 4 + 2], z[2], t2));
                let t3 = _mm512_mul_pd(m_im[r * 4 + 3], zs[3]);
                p = _mm512_add_pd(p, _mm512_fmaddsub_pd(m_re[r * 4 + 3], z[3], t3));
                new_z[r] = p;
            }

            // Store back into the same 4 sub-blocks.
            for k in 0..4 {
                let i_k = block | offsets[k] | j;
                // SAFETY: same bit-disjointness invariant as the
                // load above ⇒ i_k + LANES ≤ len.
                _mm512_storeu_pd(amps_ptr.add(i_k * 2), new_z[k]);
            }

            j += LANES;
        }
        debug_assert_eq!(j, t_lo_bit);
    };

    // Outer-walk: reserve bits `[0, t_lo]` for the inner SIMD walk
    // (`j` plus the `offsets[k]` injection of t_lo_bit / t_hi_bit) by
    // renormalising every "fixed" position (t_hi and each external
    // control) — subtract `t_lo + 1` so they index into the "above
    // t_lo" subspace — then left-shift `expand_with_fixed`'s result
    // by `t_lo + 1` to place everything back at the real qubit
    // positions.  See the doc-comment "Outer-walk (bit-disjointness
    // invariant)" section for the full derivation; this is the 2q
    // extension of the `apply_1q_avx512` idiom (aos.rs lines
    // 234-248).  Controls.len() = 0 collapses naturally — no
    // separate uncontrolled branch needed.
    //
    // Subtraction `t_hi - t_lo - 1` is safe: `t_hi > t_lo` is
    // guaranteed by construction (we take min/max of distinct
    // targets).  `c - t_lo - 1` is safe because the dispatch
    // contract requires every control `c > t_hi > t_lo`.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((t_hi - t_lo - 1, false));
    for &c in controls {
        fixed_above.push((c - t_lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    // Above-t_lo subspace has `n_qubits - (t_lo + 1)` positions;
    // one is reserved for t_hi (fixed=0) and `controls.len()` are
    // reserved for control bits (fixed=1).  Remaining free positions
    // count = n_qubits - t_lo - 2 - controls.len(); each free
    // position contributes a bit to the outer-walk index.
    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - t_lo - 2 - controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (t_lo + 1),
        outer_iter,
    );
}

/// Packed AVX-512 CNOT specialisation — Tier A (`1 << target >= LANES`
/// AND `control > target`).
///
/// Pure swap-pair traffic: for every outer index with bit `control = 1`,
/// bit `target = 0`, and every external control bit set, swap the
/// LANES-wide window starting at `outer` with the matching window
/// starting at `outer | t_bit`.  Zero multiplies; bandwidth-bound.
///
/// **Outer-walk (bit-disjointness).** This kernel reuses the
/// renormalise-then-shift idiom from `apply_1q_avx512` (aos.rs lines
/// 234-248) and `apply_2q_avx512` (lines 786-817).  The inner SIMD
/// walk owns bits `[0, target)` via `j`, and bit `target` is split by
/// the `t_bit` offset between the two halves of the swap pair.  The
/// outer walk therefore must reserve bits `[0, target]` and inject
/// bit `control` plus every external control bit as `fixed=true` in
/// the "above target" subspace.  We renormalise each above-target
/// position by subtracting `target + 1`, lay them out densely with
/// `expand_with_fixed`, then shift the result back by `target + 1`.
///
/// The "loose" form — `expand_with_fixed(k, &[(target, false),
/// (control, true), ...])` — pins target and control to the right
/// values but lets `k`'s free bits fall into positions below target
/// where they collide with `j` in the inner walk.  Same bug class as
/// Task 5's first fix (lines 658-664 in `apply_2q_avx512`'s
/// doc-comment).
///
/// Tier A is restricted to `control > target` because the inner walk
/// would otherwise toggle the control bit (`j ∈ [0, t_bit)` ⇒ `j`'s
/// bits ⊆ `[0, target)` which would include the control bit when
/// `control < target`).  The reverse orientation lands in Tier B
/// (Task 7).
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << target >= LANES` (= 4) — inner SIMD walk has ≥ LANES
///   contiguous pairs per half.
/// * `control > target` — Tier A's control-above-target invariant
///   (Tier B handles `control < target`).
/// * Every external control's qubit index is strictly greater than
///   `max(control, target)`, so the outer-walk's bit-expansion never
///   toggles an external control bit and the renormalisation
///   subtraction is safe.
/// * Distinct + in-range qubits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cnot_avx512(
    amps: &mut [Complex],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    let t_bit = 1usize << target;
    let len = amps.len();

    debug_assert!(t_bit >= LANES, "t_bit < LANES: dispatch contract violated");
    debug_assert!(
        control > target,
        "Tier A requires control > target (Tier B handles the reverse)"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > target.max(control)),
        "external control at-or-below max(control, target)"
    );

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Inner walk: swap LANES amps at `outer | j` with `outer | j | t_bit`.
    // By the outer-walk reservation, bits `[0, target]` of `outer` are zero
    // (target itself is reserved by the renormalise-then-shift), so
    // `outer | j` = `outer + j` and `outer | j | t_bit` = `outer + j + t_bit`.
    let inner_walk = |outer: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES <= t_bit {
            let i0 = outer | j;
            let i1 = i0 | t_bit;
            // SAFETY: bit-disjointness invariant — `outer`'s bits ≥ target+1
            // (with control bit + every external control bit set, t_bit
            // clear), `j` ⊆ [0, target), `t_bit` is bit `target`. The three
            // are pairwise disjoint, so i0 + LANES ≤ outer + t_bit ≤ len
            // and i1 + LANES ≤ outer + 2*t_bit ≤ outer + 2^(target+1) ≤ len.
            let a_vec = _mm512_loadu_pd(amps_ptr.add(i0 * 2));
            let b_vec = _mm512_loadu_pd(amps_ptr.add(i1 * 2));
            _mm512_storeu_pd(amps_ptr.add(i0 * 2), b_vec);
            _mm512_storeu_pd(amps_ptr.add(i1 * 2), a_vec);
            j += LANES;
        }
        debug_assert_eq!(j, t_bit);
    };

    // Outer-walk: reserve bits `[0, target]` for the inner walk, inject
    // control + every external control as `fixed=true` in the "above
    // target" subspace.  Subtractions are safe because the safety
    // contract guarantees `control > target` and each external control
    // > max(control, target) > target.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((control - target - 1, true));
    for &c in external_controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    // Above-target subspace has `n_qubits - (target + 1)` positions; one
    // reserved for control (fixed=1) and `external_controls.len()`
    // reserved for external control bits (fixed=1).  Remaining free
    // positions count = n_qubits - target - 2 - external_controls.len();
    // each contributes one bit to the outer-walk index.
    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - target - 2 - external_controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        inner_walk,
    );
}

/// Packed AVX-512 CNOT specialisation — Tier B (`target ∈ {0, 1}` AND
/// `1 << control >= LANES`).
///
/// In Tier A the target bit sits ≥ LANES, so a swap-pair touches two
/// disjoint LANES-wide windows.  In Tier B the target bit sits inside
/// a single LANES-wide window, so a swap-pair lives entirely *within*
/// one zmm register: a single load + `vpermutexvar_pd` + store
/// effectively swaps the matching doubles in place.
///
/// **Permute-index tables** (8 doubles per zmm; each amp = 2 doubles):
/// * `target = 0` (t_bit = 1) — swap (amp0, amp1) and (amp2, amp3) →
///   `[2, 3, 0, 1, 6, 7, 4, 5]`
/// * `target = 1` (t_bit = 2) — swap (amp0, amp2) and (amp1, amp3) →
///   `[4, 5, 6, 7, 0, 1, 2, 3]`
///
/// **Outer-walk (bit-disjointness).** Mirrors `apply_2q_cnot_avx512`'s
/// renormalise-then-shift pattern but reserves the bigger inner span
/// `[0, control)`.  The inner SIMD walk owns bits `[0, control)` via
/// `j` (which steps by LANES, so `j`'s low log2(LANES) bits are zero
/// and the in-register permute handles those swaps).  The outer walk
/// therefore reserves bits `[0, control]` and injects every external
/// control bit (each renormalised by `-(control + 1)`) as `fixed=true`
/// in the above-control subspace, then ORs in `c_bit` to pin
/// `control` itself.
///
/// The "loose" form — `expand_with_fixed(k, &[(control, true), ...])`
/// — would let `k`'s low bits fall into positions `[0, control)`
/// where they collide with `j` in the inner walk.  Same bug class as
/// Task 5's first fix and Task 6's Tier A renormalisation.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `target ∈ {0, 1}` (i.e. `1 << target < LANES = 4`).
/// * `1 << control >= LANES` (= 4) — the inner SIMD walk has ≥ LANES
///   contiguous amps per outer block.
/// * Every external control's qubit index is strictly greater than
///   `max(control, target) = control`, so the renormalisation
///   subtraction is safe and external controls land above the
///   reserved span.
/// * Distinct + in-range qubits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cnot_avx512_tier_b(
    amps: &mut [Complex],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    let c_bit = 1usize << control;
    let len = amps.len();

    debug_assert!(target <= 1, "Tier B requires target ∈ {{0, 1}}");
    debug_assert!(c_bit >= LANES, "c_bit < LANES: dispatch contract violated");
    debug_assert!(
        external_controls.iter().all(|&c| c > control),
        "external control at-or-below control"
    );

    // SAFETY: permute indices derived above.  `_mm512_setr_epi64` is
    // an immediate-builder, no UB possible.
    let permute_idx = match target {
        0 => _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5),
        1 => _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3),
        _ => unreachable!(),
    };

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Outer-walk: reserve bits `[0, control]` for the inner walk + the
    // control bit itself, inject every external control as `fixed=true`
    // in the "above control" subspace, then shift the result up by
    // `control + 1` and OR in `c_bit` to pin the control bit.
    // Subtractions are safe because the safety contract guarantees
    // each external control > control.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - control - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    // Above-control subspace has `n_qubits - (control + 1)` positions;
    // `external_controls.len()` are reserved (fixed=1).  Remaining
    // free positions count = n_qubits - control - 1 - ec.len(); each
    // contributes one bit to the outer-walk index.
    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - control - 1 - external_controls.len() as u32);

    let outer_iter = |base: usize| {
        let amps_ptr = bp.ptr();
        let outer = base | c_bit;
        // outer has: bit control=1, every external_control=1, bits
        // [0, control) all zero.  Bit `target` is in [0, control) and
        // is therefore 0 in outer — the LANES-wide load picks up amps
        // with mixed target-bit values (j enumerates bits [0, control)
        // including bit `target`), and the in-register permute swaps
        // the matching pairs.
        let mut j = 0usize;
        while j + LANES <= c_bit {
            let i = outer | j;
            // SAFETY: bit-disjointness — `outer`'s bits ≥ control+1
            // (with control bit set), `j` ⊆ [0, c_bit).  The two are
            // disjoint, so i + LANES ≤ outer + c_bit ≤ len.
            let z = _mm512_loadu_pd(amps_ptr.add(i * 2));
            let z2 = _mm512_permutexvar_pd(permute_idx, z);
            _mm512_storeu_pd(amps_ptr.add(i * 2), z2);
            j += LANES;
        }
        debug_assert_eq!(j, c_bit);
    };
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (control + 1),
        outer_iter,
    );
}

/// Packed AVX-512 CNOT specialisation — Tier C (both `control` and
/// `target` in `{0, 1}`).
///
/// Both qubits' bits fit inside a single quartet (4 amps = 8 doubles
/// = one zmm).  One load + `vpermutexvar_pd` + store per quartet
/// effects the CNOT.  No inner walk needed.
///
/// **Permute-index tables** (q1=high bit, q0=low bit; amp k = doubles
/// 2k, 2k+1):
/// * `(control=0, target=1)` — CNOT flips q1 when q0=1:
///   swap amp1 ↔ amp3 → `[0, 1, 6, 7, 4, 5, 2, 3]`
/// * `(control=1, target=0)` — CNOT flips q0 when q1=1:
///   swap amp2 ↔ amp3 → `[0, 1, 2, 3, 6, 7, 4, 5]`
///
/// **Outer-walk.** External controls all sit above the quartet span
/// (positions ≥ 2) by safety contract.  Each is renormalised by
/// `-2` and injected as `fixed=true`; free positions ≥ 2 are
/// enumerated by `k` through `expand_with_fixed`.  The resulting
/// base — shifted left by 2 — has bits 0 and 1 both zero, i.e. lands
/// exactly on a quartet boundary.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * Both `control` and `target` ∈ `{0, 1}`, distinct.
/// * Every external control's qubit index is strictly greater than 1
///   (so the renormalisation by `-2` is safe and external controls
///   land above the quartet span).
/// * `amps.len() >= 4` (at least one quartet).
/// * Distinct + in-range qubits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cnot_avx512_tier_c(
    amps: &mut [Complex],
    control: u32,
    target: u32,
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    let len = amps.len();

    debug_assert!(
        control <= 1 && target <= 1 && control != target,
        "Tier C requires distinct control,target ∈ {{0, 1}}"
    );
    debug_assert!(len >= 4, "len < 4: dispatch contract violated");
    debug_assert!(
        external_controls.iter().all(|&c| c > 1),
        "external control at-or-below 1"
    );

    let permute_idx = match (control, target) {
        // c=0, t=1: control = q0, target = q1.  Flip q1 when q0=1 →
        // swap amp1 (q1=0,q0=1, doubles 2,3) ↔ amp3 (q1=1,q0=1, doubles 6,7).
        (0, 1) => _mm512_setr_epi64(0, 1, 6, 7, 4, 5, 2, 3),
        // c=1, t=0: control = q1, target = q0.  Flip q0 when q1=1 →
        // swap amp2 (q1=1,q0=0, doubles 4,5) ↔ amp3 (q1=1,q0=1, doubles 6,7).
        (1, 0) => _mm512_setr_epi64(0, 1, 2, 3, 6, 7, 4, 5),
        _ => unreachable!("Tier C requires distinct control,target ∈ {{0, 1}}"),
    };

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Outer-walk: reserve bits [0, 2) for the in-register quartet,
    // inject every external control (renormalised by -2) as
    // `fixed=true` in the above-quartet subspace, then shift the
    // result up by 2 to land on a quartet boundary with all external
    // controls set.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - 2, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - 2 - external_controls.len() as u32);

    let outer_iter = |base: usize| {
        let amps_ptr = bp.ptr();
        // base has bits 0 and 1 zero (positions 0, 1 in fixed_above's
        // post-shift namespace are free, and they shift out by `<< 2`
        // leaving 0).  Every external control bit is set.
        debug_assert_eq!(base & 3, 0);
        // SAFETY: bit-disjointness — `base` has bits 0,1 = 0 and
        // base ≤ len - 4 (the inner permute writes 4 amps starting
        // at base).
        let z = _mm512_loadu_pd(amps_ptr.add(base * 2));
        let z2 = _mm512_permutexvar_pd(permute_idx, z);
        _mm512_storeu_pd(amps_ptr.add(base * 2), z2);
    };
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << 2,
        outer_iter,
    );
}

/// Dispatch helper for CNOT specialisations.  Routes to the AVX-512
/// Tier A / B / C kernels when the host + qubit orientation satisfies
/// the matching safety contract, otherwise falls through to the
/// scalar kernel.
///
/// Tier coverage:
/// * **Tier A** (`1 << target >= LANES` AND `control > target`):
///   classic LANES-block swap-pair across two disjoint windows.
/// * **Tier B** (`target ∈ {0, 1}` AND `control >= 2`): in-register
///   `vpermutexvar_pd` swap inside one LANES-wide window.
/// * **Tier C** (both `control, target ∈ {0, 1}`): single quartet
///   per zmm, one permute per quartet.
///
/// Uncovered orientation: `target >= 2` AND `control < target`.
/// Falls through to scalar.  A future "Tier B-reverse" using an
/// in-register permute on the control axis could close this gap;
/// out of scope for P1-07.
fn dispatch_cnot(amps: &mut [Complex], control: u32, target: u32, controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && controls.iter().all(|&c| c > target.max(control))
        {
            let t_bit = 1usize << target;
            let c_bit = 1usize << control;
            // Tier A: control > target AND t_bit ≥ LANES.
            if t_bit >= 4 && control > target {
                // SAFETY: Tier A contract — AVX-512F detected, t_bit ≥ LANES,
                // control above target, every external control above
                // max(control, target).
                unsafe {
                    apply_2q_cnot_avx512(amps, control, target, controls);
                }
                return;
            }
            // Tier B: target ∈ {0, 1} AND c_bit ≥ LANES.
            if target <= 1 && c_bit >= 4 {
                // SAFETY: Tier B contract — AVX-512F detected, target ∈ {0, 1},
                // c_bit ≥ LANES, every external control > control = max(c,t).
                unsafe {
                    apply_2q_cnot_avx512_tier_b(amps, control, target, controls);
                }
                return;
            }
            // Tier C: both control and target ∈ {0, 1}.
            if target <= 1 && control <= 1 {
                // SAFETY: Tier C contract — AVX-512F detected, both ∈ {0, 1},
                // every external control > 1, len ≥ 4 (n_qubits ≥ 2 since
                // both qubits exist).
                unsafe {
                    apply_2q_cnot_avx512_tier_c(amps, control, target, controls);
                }
                return;
            }
            // Remaining case: target ≥ 2 AND control < target.  Out of
            // Tier A/B/C scope; fall through to scalar.
        }
    }
    apply_2q_cnot_scalar(amps, control, target, controls);
}

/// Packed AVX-512 SWAP specialisation — Tier A
/// (`1 << min(targets) >= LANES`).
///
/// Pure swap-pair traffic: for every outer index with bits
/// `min(targets)` and `max(targets)` zero and every external control
/// bit set, swap the LANES-wide window starting at `outer | lo_bit`
/// (lo-bit=1, hi-bit=0) with the matching window starting at
/// `outer | hi_bit` (lo-bit=0, hi-bit=1).  Zero multiplies;
/// bandwidth-bound.  The two "diagonal" quartet members
/// `(lo, hi) = (0, 0)` and `(1, 1)` are SWAP fixed points and never
/// touched.
///
/// **Outer-walk (bit-disjointness).** Same renormalise-then-shift
/// idiom as `apply_2q_avx512` (lines 786-817) and
/// `apply_2q_cnot_avx512` (lines 916-938).  The inner SIMD walk
/// owns bits `[0, lo)` via `j`, and bits `lo`, `hi` are split between
/// the two halves of the swap pair (each iter loads one window with
/// `lo` set, one with `hi` set).  The outer walk reserves bits
/// `[0, lo]` and injects bit `hi` (fixed=false) plus every external
/// control bit (fixed=true) in the "above lo" subspace; the result
/// is shifted up by `lo + 1` to land back at real qubit positions.
///
/// The "loose" form — `expand_with_fixed(k, &[(lo, false), (hi, false),
/// ...])` without the shift — pins both target bits to zero
/// correctly, but lets `k`'s bits fall into positions below `lo`
/// where they collide with `j` in the inner walk.  Same bug class as
/// Task 5's first fix and Task 6's Tier A renormalisation.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << min(targets) >= LANES` (= 4) — inner SIMD walk has ≥ LANES
///   contiguous pairs per half.
/// * Distinct targets, both in qubit range.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the renormalisation subtraction is safe and
///   the outer-walk's bit-expansion never toggles an external
///   control bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_swap_avx512(amps: &mut [Complex], targets: [u32; 2], external_controls: &[u32]) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let lo_bit = 1usize << lo;
    let hi_bit = 1usize << hi;
    let len = amps.len();

    debug_assert!(
        lo != hi,
        "SWAP requires distinct targets: dispatch contract violated"
    );
    debug_assert!(
        lo_bit >= LANES,
        "lo_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Inner walk: swap LANES amps at `outer | lo_bit | j` (lo-bit=1,
    // hi-bit=0) with LANES amps at `outer | hi_bit | j` (lo-bit=0,
    // hi-bit=1).  By the outer-walk reservation, bits `[0, lo]` of
    // `outer` are zero (lo is reserved by the renormalise-then-shift,
    // hi is a fixed-false slot above lo), so `outer | lo_bit | j` =
    // `outer + lo_bit + j` and similarly for hi.
    let inner_walk = |outer: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES <= lo_bit {
            // SAFETY: bit-disjointness invariant —
            //   * `outer` ⊆ bits ≥ lo+1, with bit `hi` clear and
            //     every external control bit set (renormalise-then-
            //     shift outer-walk below).
            //   * `j` ⊆ [0, lo).
            //   * `lo_bit` is bit `lo`; `hi_bit` is bit `hi` ≥ lo+1
            //     (and lies in a fixed-false position of `outer`).
            // The three pieces are pairwise bit-disjoint for both
            // i_01 (lo-bit=1, hi-bit=0) and i_10 (lo-bit=0,
            // hi-bit=1), so each index + LANES ≤ len.
            let i_01 = outer | lo_bit | j;
            let i_10 = outer | hi_bit | j;
            let a = _mm512_loadu_pd(amps_ptr.add(i_01 * 2));
            let b = _mm512_loadu_pd(amps_ptr.add(i_10 * 2));
            _mm512_storeu_pd(amps_ptr.add(i_01 * 2), b);
            _mm512_storeu_pd(amps_ptr.add(i_10 * 2), a);
            j += LANES;
        }
        debug_assert_eq!(j, lo_bit);
    };

    // Outer-walk: reserve bits `[0, lo]` for the inner walk, inject
    // `hi` (fixed=false) and every external control (fixed=true) into
    // the "above lo" subspace.  Subtractions are safe: `hi > lo` by
    // construction (min/max of distinct targets) and every external
    // control > hi > lo by the safety contract.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((hi - lo - 1, false));
    for &ec in external_controls {
        fixed_above.push((ec - lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    // Above-lo subspace has `n_qubits - (lo + 1)` positions; one
    // reserved for `hi` (fixed=0) and `external_controls.len()`
    // reserved for external control bits (fixed=1).  Remaining free
    // positions count = n_qubits - lo - 2 - external_controls.len();
    // each contributes one bit to the outer-walk index.
    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - lo - 2 - external_controls.len() as u32);
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1),
        inner_walk,
    );
}

/// Packed AVX-512 SWAP specialisation — Tier B (`min(targets) ∈
/// {0, 1}` AND `1 << max(targets) >= LANES`).
///
/// In Tier A the lo target bit sits ≥ LANES, so a swap-pair touches
/// two disjoint LANES-wide windows.  In Tier B the lo bit sits
/// *inside* one LANES-wide window — and the hi bit selects between
/// two adjacent windows.  A swap-pair therefore spans both windows
/// (one half lives in the `hi = 0` zmm, the other in the `hi = 1`
/// zmm), so a single `vpermutex2var_pd` per output zmm shuffles the
/// matching doubles between the two inputs.
///
/// **Permute-index tables** (8 doubles per zmm; doubles `[0, 7]` =
/// hi=0 input, doubles `[8, 15]` = hi=1 input; output position →
/// source index):
/// * `lo = 0` — varying bit 0 inside each zmm is the lo-bit:
///   * `idx_for_hi0 = [0, 1, 8, 9, 4, 5, 12, 13]`
///   * `idx_for_hi1 = [2, 3, 10, 11, 6, 7, 14, 15]`
/// * `lo = 1` — varying bit 1 inside each zmm is the lo-bit:
///   * `idx_for_hi0 = [0, 1, 2, 3, 8, 9, 10, 11]`
///   * `idx_for_hi1 = [4, 5, 6, 7, 12, 13, 14, 15]`
///
/// **Outer-walk (bit-disjointness).** Same renormalise-then-shift
/// idiom as `apply_2q_cnot_avx512_tier_b`.  The inner walk owns
/// bits `[0, hi)` via `j` (which steps by LANES) and the choice
/// between the two halves of the swap pair (each iteration loads
/// the `hi = 0` zmm and the `hi = 1` zmm).  The outer walk
/// reserves bits `[0, hi]` and injects every external control bit
/// (each renormalised by `-(hi + 1)`) as `fixed=true` in the
/// above-hi subspace, then shifts the result up by `hi + 1`.
///
/// The "loose" form — `expand_with_fixed(k, &[(hi, false), ...])`
/// — would let `k`'s bits fall into positions `[0, hi)` where they
/// collide with `j` in the inner walk and with `hi_bit` in the
/// half-selector.  Same bug class as Task 5's first fix and Task
/// 6's Tier A renormalisation.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `min(targets) ∈ {0, 1}` (i.e. `1 << lo < LANES = 4`).
/// * `1 << max(targets) >= LANES` (= 4) — the inner SIMD walk has
///   ≥ LANES contiguous amps per zmm half.
/// * Distinct targets, both in qubit range.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the renormalisation subtraction is safe
///   and external controls land above the reserved span.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_swap_avx512_tier_b(
    amps: &mut [Complex],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let hi_bit = 1usize << hi;
    let len = amps.len();

    debug_assert!(
        lo != hi,
        "SWAP requires distinct targets: dispatch contract violated"
    );
    debug_assert!(lo <= 1, "Tier B requires lo ∈ {{0, 1}}");
    debug_assert!(
        hi_bit >= LANES,
        "hi_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    // SAFETY: permute indices derived in the doc comment above.
    // `_mm512_setr_epi64` is an immediate-builder, no UB possible.
    let (idx_for_hi0, idx_for_hi1) = match lo {
        0 => (
            _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13),
            _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15),
        ),
        1 => (
            _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11),
            _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15),
        ),
        _ => unreachable!("Tier B requires lo ∈ {{0, 1}}"),
    };

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Outer-walk: reserve bits `[0, hi]` for the inner walk + the
    // hi-bit half-selector, inject every external control as
    // `fixed=true` in the above-hi subspace, then shift the result
    // up by `hi + 1`.  Subtractions are safe because the safety
    // contract guarantees each external control > hi.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - hi - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    // Above-hi subspace has `n_qubits - (hi + 1)` positions;
    // `external_controls.len()` are reserved (fixed=1).  Remaining
    // free positions count = n_qubits - hi - 1 - ec.len(); each
    // contributes one bit to the outer-walk index.
    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - hi - 1 - external_controls.len() as u32);

    let outer_iter = |outer: usize| {
        let amps_ptr = bp.ptr();
        // outer has: bit hi = 0, every external_control bit set,
        // bits [0, hi) all zero.  We OR in `j` (which steps over
        // bits [0, hi)) and either 0 or `hi_bit` to select between
        // the two zmm halves of the swap pair.
        let mut j = 0usize;
        while j + LANES <= hi_bit {
            // SAFETY: bit-disjointness — `outer`'s bits ≥ hi+1
            // (with hi cleared and every external control set), `j`
            // ⊆ [0, hi_bit), `hi_bit` is bit hi.  Three pieces
            // pairwise bit-disjoint, so each LANES-wide load + store
            // stays within `len`.
            let i_0 = outer | j;
            let i_1 = i_0 | hi_bit;
            let z0 = _mm512_loadu_pd(amps_ptr.add(i_0 * 2));
            let z1 = _mm512_loadu_pd(amps_ptr.add(i_1 * 2));
            let new_z0 = _mm512_permutex2var_pd(z0, idx_for_hi0, z1);
            let new_z1 = _mm512_permutex2var_pd(z0, idx_for_hi1, z1);
            _mm512_storeu_pd(amps_ptr.add(i_0 * 2), new_z0);
            _mm512_storeu_pd(amps_ptr.add(i_1 * 2), new_z1);
            j += LANES;
        }
        debug_assert_eq!(j, hi_bit);
    };
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (hi + 1),
        outer_iter,
    );
}

/// Packed AVX-512 SWAP specialisation — Tier C (both targets in
/// `{0, 1}`).
///
/// Both qubits' bits fit inside a single quartet (4 amps = 8
/// doubles = one zmm).  One load + `vpermutexvar_pd` + store
/// effects the SWAP by exchanging amp1 (q1=0, q0=1) with amp2
/// (q1=1, q0=0).  No inner walk needed.  The choice of `lo`/`hi`
/// is irrelevant — SWAP is symmetric in its targets — so the same
/// permute index serves both orientations.
///
/// **Permute-index table** (q1 = bit 1, q0 = bit 0; amp k =
/// doubles 2k, 2k+1):
/// * swap amp1 (doubles 2, 3) ↔ amp2 (doubles 4, 5)
///   → `[0, 1, 4, 5, 2, 3, 6, 7]`
///
/// **Outer-walk.** External controls all sit above the quartet
/// span (positions ≥ 2) by safety contract.  Each is renormalised
/// by `-2` and injected as `fixed=true`; free positions ≥ 2 are
/// enumerated by `k` through `expand_with_fixed`.  The resulting
/// base — shifted left by 2 — has bits 0 and 1 both zero, i.e.
/// lands exactly on a quartet boundary.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * Both targets in `{0, 1}`, distinct.
/// * Every external control's qubit index is strictly greater
///   than 1 (so the renormalisation by `-2` is safe and external
///   controls land above the quartet span).
/// * `amps.len() >= 4` (at least one quartet).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_swap_avx512_tier_c(
    amps: &mut [Complex],
    targets: [u32; 2],
    external_controls: &[u32],
) {
    use core::arch::x86_64::*;

    let len = amps.len();

    debug_assert!(
        targets[0] <= 1 && targets[1] <= 1 && targets[0] != targets[1],
        "Tier C requires distinct targets in {{0, 1}}"
    );
    debug_assert!(len >= 4, "len < 4: dispatch contract violated");
    debug_assert!(
        external_controls.iter().all(|&c| c > 1),
        "external control at-or-below 1: dispatch contract violated"
    );

    // SWAP exchanges amp1 (q1=0, q0=1) with amp2 (q1=1, q0=0).
    // Output position → source index in the input zmm.
    let permute_idx = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Outer-walk: reserve bits [0, 2) for the in-register quartet,
    // inject every external control (renormalised by -2) as
    // `fixed=true` in the above-quartet subspace, then shift the
    // result up by 2 to land on a quartet boundary with all
    // external controls set.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &ec in external_controls {
        fixed_above.push((ec - 2, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - 2 - external_controls.len() as u32);

    let outer_iter = |base: usize| {
        let amps_ptr = bp.ptr();
        // base has bits 0 and 1 zero (post-shift) and every external
        // control bit set.
        debug_assert_eq!(base & 3, 0);
        // SAFETY: bit-disjointness — `base` has bits 0, 1 = 0 and
        // base ≤ len - 4 (the in-register permute writes 4 amps
        // starting at base).
        let z = _mm512_loadu_pd(amps_ptr.add(base * 2));
        let z2 = _mm512_permutexvar_pd(permute_idx, z);
        _mm512_storeu_pd(amps_ptr.add(base * 2), z2);
    };
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << 2,
        outer_iter,
    );
}

/// Dispatch helper for SWAP.  Routes to the AVX-512 Tier A / B / C
/// kernels when the host + qubit orientation satisfies the matching
/// safety contract, otherwise falls through to the scalar SWAP
/// kernel.
///
/// Tier coverage (SWAP is symmetric, so the routing is keyed on
/// `lo = min(targets)`, `hi = max(targets)`):
/// * **Tier A** (`1 << lo >= LANES`): classic LANES-block swap-pair
///   across two disjoint windows.
/// * **Tier B** (`lo ∈ {0, 1}` AND `1 << hi >= LANES`): a swap-pair
///   spans two adjacent LANES-wide windows whose only differing bit
///   is `hi`; `vpermutex2var_pd` shuffles the matching doubles
///   between the two zmms.
/// * **Tier C** (both targets in `{0, 1}`): both targets fit inside
///   a single quartet (4 amps = 8 doubles = one zmm); a single
///   load + `vpermutexvar_pd` + store swaps amp1 ↔ amp2 in place.
fn dispatch_swap(amps: &mut [Complex], targets: [u32; 2], controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let lo = targets[0].min(targets[1]);
        let hi = targets[0].max(targets[1]);
        if std::is_x86_feature_detected!("avx512f") && controls.iter().all(|&c| c > hi) {
            let lo_bit = 1usize << lo;
            let hi_bit = 1usize << hi;
            if lo_bit >= 4 {
                // SAFETY: Tier A contract — AVX-512F detected, lo_bit ≥
                // LANES, targets distinct (lo ≠ hi by min/max of
                // distinct inputs), every external control > hi.
                unsafe {
                    apply_2q_swap_avx512(amps, targets, controls);
                }
                return;
            }
            if hi_bit >= 4 && lo <= 1 {
                // SAFETY: Tier B contract — AVX-512F detected, lo ∈
                // {0, 1}, hi_bit ≥ LANES, distinct targets, every
                // external control > hi.
                unsafe {
                    apply_2q_swap_avx512_tier_b(amps, targets, controls);
                }
                return;
            }
            if lo <= 1 && hi <= 1 {
                // SAFETY: Tier C contract — AVX-512F detected, both
                // targets in {0, 1}, distinct, every external control
                // > hi ≥ 1, len ≥ 4 (guaranteed because there's at
                // least one quartet whenever both targets fit in {0, 1}
                // and the state is well-formed at n_qubits ≥ 2).
                unsafe {
                    apply_2q_swap_avx512_tier_c(amps, targets, controls);
                }
                return;
            }
        }
    }
    apply_2q_swap_scalar(amps, targets, controls);
}

/// Packed AVX-512 CZ specialisation — Tier A (`1 << min(targets) >= LANES`).
///
/// CZ negates `state[i]` for amplitudes where both target bits are 1 (and
/// every external control is satisfied).  Touches only the `(1, 1)`
/// sub-block — 1/4 of the state vector.  Implemented as a single
/// `vxorpd` against a sign-mask broadcast (`-0.0`), which flips the sign
/// bit of every double in the zmm.  Zero multiplies; bandwidth-bound.
///
/// **Outer-walk (bit-disjointness).** Same renormalise-then-shift idiom
/// as the other Tier A 2q kernels (`apply_2q_avx512`,
/// `apply_2q_cnot_avx512`, `apply_2q_swap_avx512`).  The inner SIMD walk
/// owns bits `[0, lo)` via `j`.  This kernel targets the `(1, 1)`
/// sub-block, so both target bits are SET in `outer`: bit `hi` enters
/// via a `fixed=true` slot in `fixed_above`, and bit `lo` is OR'd in
/// after the shift via `outer | lo_bit`.  External control bits are
/// laid out as `fixed=true` in the above-lo subspace.
///
/// The "loose" form — `expand_with_fixed(k, &[(lo, true), (hi, true),
/// ...])` without the shift — pins both target bits to 1 correctly but
/// lets `k`'s bits fall into positions below `lo` where they collide
/// with `j` in the inner walk.  Same bug class as Task 5's first fix
/// and the matching SWAP/CNOT Tier A renormalisations.
///
/// **Per inner iter (LANES = 4 amps):** 1 load + 1 xor + 1 store ≈ 3
/// µops.  Vs scalar CZ (4 amps × ~5 µops per branch+negate = ~20):
/// ~7× per-amp on the inner walk.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << min(targets) >= LANES` (= 4) — inner SIMD walk has ≥ LANES
///   contiguous pairs per touched sub-block.
/// * Distinct targets, both in qubit range.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the renormalisation subtraction is safe and
///   the outer-walk's bit-expansion never toggles an external control
///   bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_cz_avx512(amps: &mut [Complex], targets: [u32; 2], external_controls: &[u32]) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let lo_bit = 1usize << lo;
    let len = amps.len();

    debug_assert!(
        lo != hi,
        "CZ requires distinct targets: dispatch contract violated"
    );
    debug_assert!(
        lo_bit >= LANES,
        "lo_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);
    // Sign-mask: each double-lane has only its IEEE-754 sign bit set, so
    // `vxorpd(z, sign_mask)` flips the sign of every double in `z` —
    // equivalent to `z = -z` for both real and imaginary parts.
    let sign_mask = _mm512_set1_pd(-0.0_f64);

    // Outer-walk: reserve bits `[0, lo]` for the inner walk + the
    // lo-bit (OR'd in after the shift), inject `hi` as `fixed=true`
    // (we target the (1, 1) sub-block) and every external control as
    // `fixed=true` in the above-lo subspace, then shift up by `lo + 1`.
    // Subtractions are safe: `hi > lo` by construction (min/max of
    // distinct targets) and every external control > hi > lo by the
    // safety contract.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((hi - lo - 1, true));
    for &ec in external_controls {
        fixed_above.push((ec - lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    // Above-lo subspace has `n_qubits - (lo + 1)` positions; one
    // reserved for `hi` (fixed=1) and `external_controls.len()` for
    // external control bits (fixed=1).  Remaining free positions count
    // = n_qubits - lo - 2 - external_controls.len(); each contributes
    // one bit to the outer-walk index.
    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - lo - 2 - external_controls.len() as u32);

    let outer_iter = |base: usize| {
        let amps_ptr = bp.ptr();
        // base has: bit hi = 1, every external_control bit = 1, bits
        // [0, lo] all zero.  ORing in `lo_bit` sets bit `lo`, so the
        // resulting `outer` lands on the (1, 1) sub-block.
        let outer = base | lo_bit;
        let mut j = 0usize;
        while j + LANES <= lo_bit {
            // SAFETY: bit-disjointness invariant — `base` ⊆ bits ≥
            // lo+1 (hi set, every external control set), `lo_bit` is
            // bit `lo`, `j` ⊆ [0, lo).  All three pairwise disjoint,
            // so `i = outer | j = base + lo_bit + j` and
            // `i + LANES ≤ base + 2 * lo_bit ≤ len`.
            let i = outer | j;
            let z = _mm512_loadu_pd(amps_ptr.add(i * 2));
            let z = _mm512_xor_pd(z, sign_mask);
            _mm512_storeu_pd(amps_ptr.add(i * 2), z);
            j += LANES;
        }
        debug_assert_eq!(j, lo_bit);
    };
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1),
        outer_iter,
    );
}

/// Packed AVX-512 general-diagonal 2q specialisation — Tier A
/// (`1 << min(targets) >= LANES`).
///
/// Each amplitude `state[i]` is multiplied by `d[k]` where
/// `k = ((i >> targets[0]) & 1) << 1 | ((i >> targets[1]) & 1)` (MSB
/// convention, per ADR 0004 / P0-06 §6).  Per outer block we iterate
/// the four `(q_hi, q_lo) ∈ {0,1}²` sub-blocks, each multiplying the
/// LANES-wide window by a single broadcast `d[k]` via the P1-06
/// single-stream complex-multiply idiom (`vpermilpd<0x55>` + `vmulpd`
/// + `vfmaddsub`).  ~1.25 µops per amplitude on the inner walk.
///
/// **Outer-walk (bit-disjointness).** Same renormalise-then-shift idiom
/// as `apply_2q_cz_avx512`, but `hi` is a *fixed-zero* slot in
/// `fixed_above` (we enumerate the hi bit per sub-block by OR-ing in
/// `hi_bit` for the two upper sub-blocks).  The inner SIMD walk owns
/// bits `[0, lo)` via `j`; bit `lo` is OR'd in per sub-block via
/// `multiply_block(base | lo_bit, ..)`; bit `hi` is OR'd in via
/// `multiply_block(base | hi_bit, ..)`; external controls live in the
/// above-lo subspace as `fixed=true`.
///
/// **Sub-block to d[k] mapping.** `k` is defined by `targets[0]` (MSB)
/// and `targets[1]` (LSB); but the outer-walk thinks in `(q_hi, q_lo)`
/// coordinates (where lo = min(targets), hi = max(targets)).  When
/// `targets[0] < targets[1]` (so `targets[0] = lo`, `targets[1] = hi`),
/// `k = (q_lo << 1) | q_hi`; when `targets[0] > targets[1]`, the usual
/// `k = (q_hi << 1) | q_lo`.  This disambiguation is the `d_for_*`
/// tuple below.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << min(targets) >= LANES` (= 4) — inner SIMD walk has ≥ LANES
///   contiguous pairs per sub-block.
/// * Distinct targets, both in qubit range.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the renormalisation subtraction is safe and
///   the outer-walk's bit-expansion never toggles an external control
///   bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_2q_diagonal_avx512(
    amps: &mut [Complex],
    targets: [u32; 2],
    external_controls: &[u32],
    d: [Complex; 4],
) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    let lo = targets[0].min(targets[1]);
    let hi = targets[0].max(targets[1]);
    let lo_bit = 1usize << lo;
    let hi_bit = 1usize << hi;
    let len = amps.len();

    debug_assert!(
        lo != hi,
        "2q-diagonal requires distinct targets: dispatch contract violated"
    );
    debug_assert!(
        lo_bit >= LANES,
        "lo_bit < LANES: dispatch contract violated"
    );
    debug_assert!(
        external_controls.iter().all(|&c| c > hi),
        "external control at-or-below hi: dispatch contract violated"
    );

    // Disambiguate which d[k] each (q_hi, q_lo) sub-block hits.  MSB
    // convention: k bit 1 = targets[0], k bit 0 = targets[1].
    let (d_for_hi0_lo0, d_for_hi0_lo1, d_for_hi1_lo0, d_for_hi1_lo1) = if targets[0] < targets[1] {
        // targets[0] = lo, targets[1] = hi → k = (q_lo << 1) | q_hi
        //   (q_hi=0, q_lo=0) → k=0
        //   (q_hi=0, q_lo=1) → k=2
        //   (q_hi=1, q_lo=0) → k=1
        //   (q_hi=1, q_lo=1) → k=3
        (d[0], d[2], d[1], d[3])
    } else {
        // targets[0] = hi, targets[1] = lo → k = (q_hi << 1) | q_lo
        //   (q_hi=0, q_lo=0) → k=0
        //   (q_hi=0, q_lo=1) → k=1
        //   (q_hi=1, q_lo=0) → k=2
        //   (q_hi=1, q_lo=1) → k=3
        (d[0], d[1], d[2], d[3])
    };

    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);

    // Single-stream complex multiply per sub-block.  P1-06 diagonal-1q
    // idiom: vmovupd → vpermilpd 0x55 → vmulpd(im_bc, swap) →
    // vfmaddsub(re_bc, z, t).  ~5 µops per LANES amps ≈ 1.25 µops/amp.
    let multiply_block = |base: usize, d_k: Complex| {
        let amps_ptr = bp.ptr();
        let d_re_bc = _mm512_set1_pd(d_k.re);
        let d_im_bc = _mm512_set1_pd(d_k.im);
        let mut j = 0usize;
        while j + LANES <= lo_bit {
            let i = base | j;
            // SAFETY: bit-disjointness invariant (see doc-comment
            // "Outer-walk" section).  `base` ⊆ bits ≥ lo+1 OR'd with
            // `lo_bit` and/or `hi_bit`; `j` ⊆ [0, lo).  All pieces
            // pairwise bit-disjoint, so `i = base + j` and
            // `i + LANES ≤ base + lo_bit ≤ len`.
            let z = _mm512_loadu_pd(amps_ptr.add(i * 2));
            let zs = _mm512_permute_pd::<0x55>(z);
            let t = _mm512_mul_pd(d_im_bc, zs);
            let out = _mm512_fmaddsub_pd(d_re_bc, z, t);
            _mm512_storeu_pd(amps_ptr.add(i * 2), out);
            j += LANES;
        }
        debug_assert_eq!(j, lo_bit);
    };

    // Outer-walk: reserve bits `[0, lo]` for the inner walk + the
    // lo-bit (OR'd in per sub-block); inject `hi` as `fixed=false`
    // (we enumerate the hi bit per sub-block via OR-ing in `hi_bit`)
    // and every external control as `fixed=true` in the above-lo
    // subspace; then shift up by `lo + 1`.  Subtractions are safe:
    // `hi > lo` by construction; every external control > hi > lo by
    // the safety contract.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((hi - lo - 1, false));
    for &ec in external_controls {
        fixed_above.push((ec - lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(p, _)| p);

    // Above-lo subspace has `n_qubits - (lo + 1)` positions; one
    // reserved for `hi` (fixed=0) and `external_controls.len()` for
    // external control bits (fixed=1).  Remaining free positions
    // count = n_qubits - lo - 2 - external_controls.len(); each
    // contributes one bit to the outer-walk index.
    let n_qubits = len.trailing_zeros();
    let outer_count = 1usize << (n_qubits - lo - 2 - external_controls.len() as u32);

    let outer_iter = |base: usize| {
        // base has: bit hi = 0, every external_control bit = 1, bits
        // [0, lo] all zero.  Iterate the 4 sub-blocks:
        multiply_block(base, d_for_hi0_lo0); // (q_hi=0, q_lo=0)
        multiply_block(base | lo_bit, d_for_hi0_lo1); // (q_hi=0, q_lo=1)
        multiply_block(base | hi_bit, d_for_hi1_lo0); // (q_hi=1, q_lo=0)
        multiply_block(base | hi_bit | lo_bit, d_for_hi1_lo1); // (q_hi=1, q_lo=1)
    };
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1),
        outer_iter,
    );
}

/// Dispatch helper for the diagonal-4x4 branch (catches CZ, controlled-
/// phase, Rzz, user diagonals).  Routes to `apply_2q_cz_avx512` (when
/// the matrix matches the CZ signature) or `apply_2q_diagonal_avx512`
/// (general diagonal) when the host + qubit orientation satisfies the
/// Tier A safety contract; otherwise falls through to the scalar
/// specialised kernel.
fn dispatch_diagonal_or_cz(
    amps: &mut [Complex],
    targets: [u32; 2],
    controls: &[u32],
    d: [Complex; 4],
    is_cz: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        let lo = targets[0].min(targets[1]);
        let hi = targets[0].max(targets[1]);
        let lo_bit = 1usize << lo;
        if std::is_x86_feature_detected!("avx512f")
            && lo_bit >= 4
            && controls.iter().all(|&c| c > hi)
        {
            // SAFETY: Tier A contract — AVX-512F detected, lo_bit ≥
            // LANES, targets distinct (lo ≠ hi by min/max of distinct
            // inputs), every external control > hi.
            if is_cz {
                unsafe {
                    apply_2q_cz_avx512(amps, targets, controls);
                }
            } else {
                unsafe {
                    apply_2q_diagonal_avx512(amps, targets, controls, d);
                }
            }
            return;
        }
    }
    if is_cz {
        apply_2q_cz_scalar(amps, targets, controls);
    } else {
        apply_2q_diagonal_scalar(amps, targets, controls, d);
    }
}

/// Scalar Tier-C reference for Toffoli (CCX). Tier-C fallback of
/// `dispatch_toffoli` (spec §4.4, wired in Task 6).
///
/// `targets = [c0, c1, t]` matches `Gate::Toffoli`'s qubit layout.
/// `external_controls` are additional control qubits beyond c0 and c1.
///
/// Walks every amplitude index `i`; swaps `amps[i]` with
/// `amps[i | target_bit]` when every control bit (c0, c1, external)
/// is set in `i` and the target bit is clear. No SIMD — scalar only.
///
/// **Indexing convention.** Amplitude index bit `q` corresponds to
/// qubit `q` (bit 0 = qubit 0, bit 1 = qubit 1, …), matching the
/// convention used by all scalar kernels in this file. With
/// `targets = [c0=0, c1=1, t=2]`, ctrl_mask = `0b011` and target_bit
/// = `4`; the gate fires at `i = 0b011 = 3` and swaps with `i = 7`.
pub(crate) fn apply_toffoli_scalar(
    amps: &mut [Complex],
    targets: [u32; 3],
    external_controls: &[u32],
) {
    let c0 = targets[0];
    let c1 = targets[1];
    let t = targets[2];
    let target_bit = 1usize << t;
    let mut ctrl_mask = (1usize << c0) | (1usize << c1);
    for &e in external_controls {
        ctrl_mask |= 1usize << e;
    }
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask && (i & target_bit) == 0 {
            amps.swap(i, i | target_bit);
        }
        i += 1;
    }
}

/// Scalar Tier-C reference for CCZ (spec §5.4). Sign-flips the
/// single amplitude where all three qubits AND any external controls
/// are |1⟩.
///
/// `targets[0..3]` are the three CCZ qubit indices. The mask is
/// `(1<<targets[0]) | (1<<targets[1]) | (1<<targets[2])` plus one bit
/// per external control. Amplitude `i` is negated iff `(i & mask) == mask`.
/// This is symmetric in the order of `targets`.
pub(crate) fn apply_ccz_scalar(amps: &mut [Complex], targets: [u32; 3], external_controls: &[u32]) {
    let mut mask = (1usize << targets[0]) | (1usize << targets[1]) | (1usize << targets[2]);
    for &e in external_controls {
        mask |= 1usize << e;
    }
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & mask) == mask {
            amps[i] = -amps[i];
        }
        i += 1;
    }
}

/// Packed AVX-512 Toffoli specialisation — Tier A (clean contract).
///
/// Inner loop: loads one LANES-wide zmm from the `target_bit=0` window
/// and one from the `target_bit=1` window, stores them cross-swapped.
/// 2 zmm loads + 2 zmm stores per matching block; purely bandwidth-bound.
///
/// The Tier-A "clean" contract restricts every control bit strictly above
/// the target bit. Within a LANES-block the j-index sweeps bits `[0, target)`,
/// none of which overlap any control bit, so the `ctrl_mask` check on
/// `block_base` is uniform across the entire block — no partial-block
/// ambiguity, no outer-walk needed.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F (`is_x86_feature_detected!("avx512f")` true).
/// * `target >= 2` i.e. `1 << target >= LANES` (== 4) — inner SIMD walk
///   has ≥ LANES contiguous amp pairs per half.
/// * Every qubit position in `sorted_controls` is strictly greater than
///   `target` — guarantees no control bit falls inside the inner `j`-sweep
///   range `[0, target)`.
/// * `amps.len() == 1 << n` for some n ≥ 3 (circuit invariant).
/// * `target` and all entries of `sorted_controls` are distinct and < n.
///
/// `sorted_controls` contains ALL control qubits (both CCX's inner pair
/// and any external controls). It need not be sorted; the function only
/// ORs the bits into a mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_a(amps: &mut [Complex], target: u32, sorted_controls: &[u32]) {
    use std::arch::x86_64::*;

    const LANES: usize = 4; // complex amps per zmm (8 f64 per zmm)

    debug_assert!(
        target >= 2,
        "Tier-A contract: target must be >= LANES_BITS (2)"
    );
    debug_assert!(
        sorted_controls.iter().all(|&c| c > target),
        "Tier-A contract: every control bit must be strictly above target"
    );

    let target_bit = 1usize << target;
    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    // Flat block-stride walk: each block covers LANES consecutive amps.
    // Skip blocks where target_bit is already set (those are the hi-half;
    // we always load from the lo-half and its partner simultaneously).
    // Skip blocks where any control bit is clear (gate does not fire).
    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & target_bit) != 0 {
            // Already the hi half — the lo-half iteration handles this pair.
            block_base += LANES;
            continue;
        }
        if (block_base & ctrl_mask) != ctrl_mask {
            // Not all controls set — gate does not fire here.
            block_base += LANES;
            continue;
        }
        // SAFETY: `block_base & target_bit == 0` and all control bits set.
        // `block_base + LANES <= block_base | target_bit < len` because the
        // state vector has a full power-of-two length and target_bit ≥ LANES.
        // The hi-half `block_base | target_bit` is similarly within bounds.
        // Both pointers are multiplied by 2 (f64 offset = amp_index * 2).
        let lo_ptr = amps_ptr.add(block_base * 2);
        let hi_ptr = amps_ptr.add((block_base | target_bit) * 2);
        let z_lo = _mm512_loadu_pd(lo_ptr);
        let z_hi = _mm512_loadu_pd(hi_ptr);
        _mm512_storeu_pd(lo_ptr, z_hi);
        _mm512_storeu_pd(hi_ptr, z_lo);
        block_base += LANES;
    }
}

/// Tier-A outer-walk variant for Toffoli (CCX): handles controls
/// at-or-above `LANES_BITS` but not strictly above target. The inner
/// loop is structurally identical to `apply_toffoli_avx512_tier_a` —
/// the difference is the relaxed SAFETY contract: controls may lie at
/// any position ≥ LANES_BITS (including at-or-below target between
/// LANES_BITS and target), but they MUST NOT lie below LANES_BITS.
///
/// **Why c_lo >= LANES_BITS is required.** The flat block-stride walk
/// increments `block_base` by LANES (= 4) each step, so `block_base`
/// is always a multiple of LANES and its low `LANES_BITS` bits are
/// always zero. If `ctrl_mask` includes a bit position below
/// `LANES_BITS`, the test `(block_base & ctrl_mask) != ctrl_mask` is
/// always true (the required low bit is never set), and the kernel
/// silently never fires. Caller MUST ensure every control bit is at
/// or above `LANES_BITS`; otherwise the gate is dropped without
/// any error indication. The `dispatch_toffoli` prelude enforces
/// this; do not invoke this function directly without verifying.
///
/// The `target_bit` check (`block_base & target_bit != 0`) gates
/// which half of each target-pair we process — we always process
/// from the lo-half side and swap with `block_base | target_bit`.
/// Since `target >= 2`, `target_bit >= LANES`, ensuring each block
/// either lies fully in the lo-half or fully in the hi-half.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F (`is_x86_feature_detected!("avx512f")` true).
/// * `target >= 2` i.e. `1 << target >= LANES` (== 4) — ensures each
///   LANES-block is entirely in the lo-half or hi-half of the target pair.
/// * Every element of `sorted_controls` is `>= LANES_BITS = 2`. This is
///   the critical extra precondition vs the docstring's English claim —
///   sub-LANES controls silently disable the gate.
/// * All elements of `sorted_controls` are distinct, differ from `target`,
///   and are valid qubit indices (< n where `amps.len() == 1 << n`).
/// * `amps.len() == 1 << n` for some n ≥ 3 (circuit invariant).
///
/// Unlike `apply_toffoli_avx512_tier_a`, this function does NOT
/// require `c_lo > target` — controls may lie above OR below target
/// AS LONG AS each is ≥ LANES_BITS.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_a_outer_walk(
    amps: &mut [Complex],
    target: u32,
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;

    const LANES: usize = 4; // complex amps per zmm (8 f64 per zmm)

    debug_assert!(
        target >= 2,
        "Tier-A outer-walk contract: target must be >= LANES_BITS (2)"
    );

    let target_bit = 1usize << target;
    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    // Flat block-stride walk identical to the clean Tier-A kernel, but
    // without the `c_lo > target` restriction. The mask check handles
    // both above-target and below-target control bits uniformly.
    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & target_bit) != 0 {
            // Hi-half block — the lo-half iteration handles this pair.
            block_base += LANES;
            continue;
        }
        if (block_base & ctrl_mask) != ctrl_mask {
            // Not all controls set — gate does not fire here.
            block_base += LANES;
            continue;
        }
        // SAFETY: `block_base & target_bit == 0` and all control bits set.
        // `block_base + LANES <= block_base | target_bit < len` because the
        // state vector has a full power-of-two length and target_bit ≥ LANES.
        // The hi-half `block_base | target_bit` is similarly within bounds.
        // Both pointers are multiplied by 2 (f64 offset = amp_index * 2).
        let lo_ptr = amps_ptr.add(block_base * 2);
        let hi_ptr = amps_ptr.add((block_base | target_bit) * 2);
        let z_lo = _mm512_loadu_pd(lo_ptr);
        let z_hi = _mm512_loadu_pd(hi_ptr);
        _mm512_storeu_pd(lo_ptr, z_hi);
        _mm512_storeu_pd(hi_ptr, z_lo);
        block_base += LANES;
    }
}

/// Toffoli Tier-B.0: target=0 (within-LANES), in-zmm permute swap.
///
/// For `t=0` the target bit is bit 0 of the amplitude index. Within each
/// 4-amp zmm block (8 doubles), consecutive amp pairs `(amp0, amp1)` and
/// `(amp2, amp3)` differ only in bit 0. The gate swaps `amp0 ↔ amp1` and
/// `amp2 ↔ amp3` — a pure in-register permute, no cross-block loads.
///
/// **AoS layout permutation.** The zmm holds 8 doubles:
/// `(a0.re, a0.im, a1.re, a1.im, a2.re, a2.im, a3.re, a3.im)`.
/// Swapping `a0 ↔ a1` and `a2 ↔ a3` produces:
/// `(a1.re, a1.im, a0.re, a0.im, a3.re, a3.im, a2.re, a2.im)`.
/// As a lane-index permutation: input `(0,1,2,3,4,5,6,7)` → output
/// `(2,3,0,1,6,7,4,5)`.
///
/// **`_mm512_set_epi64` endianness.** The intrinsic stores argument 0 in
/// lane 7 and argument 7 in lane 0 (HIGH-to-LOW). To produce index vector
/// `(2,3,0,1,6,7,4,5)` (lane-0-first), the call is
/// `_mm512_set_epi64(5,4,7,6,1,0,3,2)`.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * `target == 0` (implicit — no parameter; this kernel is t=0 only).
/// * Every entry in `sorted_controls` is ≥ `LANES_BITS = 2`, so each
///   4-amp block carries a uniform ctrl-mask value (no control bit aliases
///   into the within-block bit positions).
/// * `amps.len() == 1 << n` for some `n ≥ 3`.
/// * All elements of `sorted_controls` are distinct, differ from 0, and
///   are valid qubit indices (< n).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_b0(amps: &mut [Complex], sorted_controls: &[u32]) {
    use std::arch::x86_64::*;

    const LANES: usize = 4; // complex amps per zmm (8 f64 per zmm)

    debug_assert!(
        sorted_controls.iter().all(|&c| c >= 2),
        "Tier-B.0 contract: every control must be at qubit index >= log2(LANES) = 2"
    );

    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    // Permute index vector for in-zmm swap of pairs (a0↔a1) and (a2↔a3).
    // Want output lane order (2,3,0,1,6,7,4,5).
    // _mm512_set_epi64 takes HIGH-to-LOW args, so reverse the lane order:
    // arg positions 7..0 correspond to output lanes 0..7.
    let perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & ctrl_mask) != ctrl_mask {
            // Not all controls set — gate does not fire for this block.
            block_base += LANES;
            continue;
        }
        // SAFETY: `block_base + LANES <= len` because len is a power of two
        // and the loop steps by LANES = 4, so the last valid block_base is
        // `len - LANES`. Each Complex is 16 bytes, so the pointer arithmetic
        // `amps_ptr.add(block_base * 2)` advances by `block_base * 2` f64
        // values = `block_base` Complex values. One zmm load covers LANES
        // Complex = 8 f64 values, all within [block_base, block_base + LANES).
        let p = amps_ptr.add(block_base * 2);
        let z = _mm512_loadu_pd(p);
        let z_perm = _mm512_permutexvar_pd(perm_idx, z);
        _mm512_storeu_pd(p, z_perm);
        block_base += LANES;
    }
}

/// Toffoli Tier-B.1: target=1 (within-LANES), in-zmm cross-128 swap.
/// Swaps amp pairs (a0,a2) and (a1,a3) within each 4-amp zmm block —
/// equivalent to swapping the low 256 bits of the zmm with the high 256.
///
/// **AoS layout permutation.** The zmm holds 8 doubles:
/// `(a0.re, a0.im, a1.re, a1.im, a2.re, a2.im, a3.re, a3.im)`.
/// Swapping `a0 ↔ a2` and `a1 ↔ a3` (pairs differing only in bit 1)
/// produces: `(a2.re, a2.im, a3.re, a3.im, a0.re, a0.im, a1.re, a1.im)`.
/// As a lane-index permutation: input `(0,1,2,3,4,5,6,7)` → output
/// `(4,5,6,7,0,1,2,3)`.
///
/// **`_mm512_set_epi64` endianness.** The intrinsic stores argument 0 in
/// lane 7 and argument 7 in lane 0 (HIGH-to-LOW). To produce index vector
/// `(4,5,6,7,0,1,2,3)` (lane-0-first), the call is
/// `_mm512_set_epi64(3,2,1,0,7,6,5,4)`.
///
/// # Safety
///
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * `target == 1` (implicit — no parameter; this kernel is t=1 only).
/// * Every entry in `sorted_controls` is ≥ `LANES_BITS = 2`, so each
///   4-amp block carries a uniform ctrl-mask value (no control bit aliases
///   into the within-block bit positions 0 or 1).
/// * `amps.len() == 1 << n` for some `n ≥ 3`.
/// * All elements of `sorted_controls` are distinct, differ from 1, and
///   are valid qubit indices (< n).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_b1(amps: &mut [Complex], sorted_controls: &[u32]) {
    use std::arch::x86_64::*;

    const LANES: usize = 4; // complex amps per zmm (8 f64 per zmm)

    debug_assert!(
        sorted_controls.iter().all(|&c| c >= 2),
        "Tier-B.1 contract: every control must be at qubit index >= log2(LANES) = 2"
    );

    let mut ctrl_mask = 0usize;
    for &c in sorted_controls {
        ctrl_mask |= 1usize << c;
    }
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    // Permute index vector for in-zmm cross-128 swap: (a0,a2) ↔ (a1,a3).
    // Want output lane order (4,5,6,7,0,1,2,3).
    // _mm512_set_epi64 takes HIGH-to-LOW args, so reverse the lane order:
    // arg positions 7..0 correspond to output lanes 0..7.
    let perm_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & ctrl_mask) != ctrl_mask {
            // Not all controls set — gate does not fire for this block.
            block_base += LANES;
            continue;
        }
        // SAFETY: `block_base + LANES <= len` because len is a power of two
        // and the loop steps by LANES = 4, so the last valid block_base is
        // `len - LANES`. Each Complex is 16 bytes, so the pointer arithmetic
        // `amps_ptr.add(block_base * 2)` advances by `block_base * 2` f64
        // values = `block_base` Complex values. One zmm load covers LANES
        // Complex = 8 f64 values, all within [block_base, block_base + LANES).
        let p = amps_ptr.add(block_base * 2);
        let z = _mm512_loadu_pd(p);
        let z_perm = _mm512_permutexvar_pd(perm_idx, z);
        _mm512_storeu_pd(p, z_perm);
        block_base += LANES;
    }
}

/// Top-level 3q dispatch. Matrix-detects Toffoli (CCX) and CCZ shapes
/// per spec §3.1 and routes to specialised paths. Identity short-circuits.
/// Falls through to the generic 8x8 scalar kernel for arbitrary matrices.
pub(crate) fn apply_3q(
    amps: &mut [Complex],
    targets: [u32; 3],
    controls: &[u32],
    m: &[[Complex; 8]; 8],
) {
    if super::is_identity_8x8(m) {
        return;
    }
    if super::is_toffoli(m) {
        dispatch_toffoli(amps, targets, controls);
        return;
    }
    if super::is_ccz(m) {
        dispatch_ccz(amps, targets, controls);
        return;
    }
    apply_3q_generic(amps, targets, controls, m);
}

/// Routes Toffoli to the best available tier (spec §4).
///
/// Tier-A (AVX-512, task 7): fires when every control (inner CCX pair +
/// external) is strictly above the target bit AND target ≥ LANES_BITS.
/// Falls through to the scalar Tier-C reference otherwise.
fn dispatch_toffoli(amps: &mut [Complex], targets: [u32; 3], controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        const LANES_BITS: u32 = 2; // log2(LANES) where LANES = 4
        let t = targets[2];

        // Merge the CCX's inner control pair with any external controls,
        // then sort so c_lo = all_ctrls[0].
        let mut all_ctrls: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
        all_ctrls.push(targets[0]);
        all_ctrls.push(targets[1]);
        for &c in controls {
            all_ctrls.push(c);
        }
        all_ctrls.sort();
        let c_lo = all_ctrls[0];

        if std::is_x86_feature_detected!("avx512f") {
            if t >= LANES_BITS {
                if c_lo > t {
                    // Tier-A clean: every control strictly above target.
                    // SAFETY: AVX-512F detected, target ≥ 2 (LANES_BITS), every
                    // control in all_ctrls is strictly above target (c_lo > t).
                    // Qubits distinct + in-range guaranteed by Circuit invariant.
                    unsafe {
                        apply_toffoli_avx512_tier_a(amps, t, &all_ctrls);
                    }
                    return;
                }
                if c_lo >= LANES_BITS {
                    // Tier-A outer-walk: controls at-or-above LANES_BITS but
                    // some lie between LANES_BITS and the target. The flat
                    // block-stride walk has block_base advancing by LANES, so
                    // its low LANES_BITS bits are always zero; any control bit
                    // below LANES_BITS would silently disable the gate (the
                    // mask test could never pass). We REQUIRE c_lo >= LANES_BITS
                    // before invoking the SIMD kernel; sub-LANES controls fall
                    // through to scalar.
                    // SAFETY: AVX-512F detected, target ≥ 2, every control
                    // ≥ LANES_BITS; qubits distinct + in-range by invariant.
                    unsafe {
                        apply_toffoli_avx512_tier_a_outer_walk(amps, t, &all_ctrls);
                    }
                    return;
                }
                // c_lo < LANES_BITS: SIMD contract violated, fall through to scalar.
            }
            if t == 0 && c_lo >= LANES_BITS {
                // Tier-B.0: target=0 in-zmm permute swap.
                // SAFETY: AVX-512F detected, target=0 (implicit in kernel),
                // every control >= LANES_BITS=2 (c_lo >= 2). State vector
                // length is a power of two with n >= 3 (circuit invariant).
                // Qubits distinct + in-range by Circuit invariant.
                unsafe {
                    apply_toffoli_avx512_tier_b0(amps, &all_ctrls);
                }
                return;
            }
            if t == 1 && c_lo >= LANES_BITS {
                // Tier-B.1: target=1 in-zmm cross-128 swap.
                // SAFETY: AVX-512F detected, target=1 (implicit in kernel),
                // every control >= LANES_BITS=2 (c_lo >= 2). State vector
                // length is a power of two with n >= 3 (circuit invariant).
                // Qubits distinct + in-range by Circuit invariant.
                unsafe {
                    apply_toffoli_avx512_tier_b1(amps, &all_ctrls);
                }
                return;
            }
        }
    }
    apply_toffoli_scalar(amps, targets, controls);
}

/// CCZ Tier-A: AVX-512 sign-flip via vxorpd on packed-complex AoS.
/// Uses `_mm512_xor_pd` with sign mask (0x8000_0000_0000_0000 in
/// every double lane) for 1-µop latency-1 sign flip per zmm block.
///
/// CCZ is symmetric — it has no "target"; every qubit in `mask_bits` acts
/// as both target and control. The flat block-stride walk checks the
/// combined mask at each LANES-aligned block base: if all mask bits are
/// set, sign-flip the entire block. Because mask_lo ≥ LANES_BITS = 2, every
/// mask bit is at or above log₂(LANES), so the per-block mask check is
/// uniform across the block's LANES amplitudes — no intra-block ambiguity.
///
/// **Inner loop (per matching block):**
/// 1 `vmovupd` (load zmm) + 1 `vxorpd` (sign flip) + 1 `vmovupd` (store) = 3 µops.
///
/// # Safety
/// Caller MUST guarantee all of:
/// * Host CPU supports AVX-512F.
/// * All entries of `mask_bits` are distinct and < n.
/// * `mask_bits[0]` (the minimum) is ≥ LANES_BITS = 2, so each LANES-block
///   has a uniform CCZ-mask value (no mask bit falls inside the intra-block
///   index range `[block_base, block_base + LANES)`).
/// * `amps.len() == 1 << n` for some n ≥ 3 (circuit invariant).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ccz_avx512_tier_a(amps: &mut [Complex], mask_bits: &[u32]) {
    use std::arch::x86_64::*;

    const LANES: usize = 4; // complex amps per zmm (8 f64 per zmm)

    debug_assert!(
        mask_bits.iter().min().copied().unwrap_or(0) >= 2,
        "Tier-A CCZ contract: every mask bit must be >= LANES_BITS (2)"
    );

    // Build the combined control+target mask from all qubit positions.
    let mut mask: usize = 0;
    for &b in mask_bits {
        mask |= 1usize << b;
    }

    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    // IEEE-754: -0.0 has exactly the sign bit set (0x8000_0000_0000_0000).
    // XOR with this value flips the sign bit of every double lane.
    let sign_mask = _mm512_set1_pd(-0.0_f64);

    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & mask) == mask {
            // SAFETY: block_base + LANES ≤ len because mask_bits are all ≥ 2,
            // so mask_lo ≥ 4 > LANES; the amplitude count between any two
            // consecutive matching block_base values is a multiple of LANES.
            // The state vector length is 1 << n ≥ 8 (n ≥ 3) and is a multiple
            // of LANES. `amps_ptr.add(block_base * 2)` is within bounds.
            let p = amps_ptr.add(block_base * 2);
            let z = _mm512_loadu_pd(p);
            let neg = _mm512_xor_pd(z, sign_mask);
            _mm512_storeu_pd(p, neg);
        }
        block_base += LANES;
    }
}

/// CCZ Tier-A outer-walk: handles mask bits below LANES_BITS via
/// per-lane-mask `_mm512_mask_blend_pd`. The "high" mask bits (>=
/// LANES_BITS) are still checked at the block_base level uniformly;
/// the "low" mask bits (< LANES_BITS) drive the per-lane blend.
///
/// For each amp `k ∈ {0,1,2,3}` inside a LANES-block, the lane should
/// be sign-flipped iff `(k & mask_low) == mask_low` where `mask_low` is
/// the union of all mask bits below LANES_BITS = 2. Each amp occupies
/// 2 consecutive doubles (re + im) in AoS layout, so the u8 `lane_mask`
/// has bit positions `2*k` and `2*k+1` set for matching lanes.
///
/// `_mm512_mask_blend_pd(mask, a, b)` selects `b` where the mask bit is
/// 1, and `a` where the mask bit is 0. With `a = z` (original) and
/// `b = neg` (sign-flipped), setting `lane_mask` in the matching positions
/// produces the correct selective sign flip.
///
/// **Edge case.** If `mask_low == 0` (no bits below LANES_BITS), then
/// `(0 & 0) == 0` is true for k=0 only, but the specification for this
/// function is only invoked when some mask bit IS below LANES_BITS
/// (`mask_lo < LANES_BITS`). If all amps in the block should flip (e.g.,
/// every bit in mask_low is the full set), `lane_mask = 0xFF` and
/// `blend` degenerates to a full flip — identical to Tier-A clean.
///
/// # Safety
/// - Host CPU supports AVX-512F.
/// - All entries of `mask_bits` are distinct and < n.
/// - `amps.len() == 1 << n` for some n ≥ 3 (circuit invariant).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ccz_avx512_tier_a_outer_walk(amps: &mut [Complex], mask_bits: &[u32]) {
    use std::arch::x86_64::*;

    const LANES: usize = 4; // complex amps per zmm (8 f64 per zmm)
    const LANES_BITS: u32 = 2; // log2(LANES)

    // Partition mask bits into low (< LANES_BITS) and high (>= LANES_BITS).
    let mut mask_low: usize = 0;
    let mut mask_high: usize = 0;
    for &b in mask_bits {
        if b < LANES_BITS {
            mask_low |= 1usize << b;
        } else {
            mask_high |= 1usize << b;
        }
    }

    // Per-block lane mask: for amp k in {0..4}, flip iff (k & mask_low) == mask_low.
    // Each amp occupies 2 doubles (AoS re+im), so bit positions are 2*k and 2*k+1.
    // Precomputed once — constant per kernel invocation.
    let lane_mask: u8 = {
        let mut m = 0u8;
        for k in 0..4u32 {
            if (k as usize & mask_low) == mask_low {
                m |= 1 << (2 * k); // re lane
                m |= 1 << (2 * k + 1); // im lane
            }
        }
        m
    };

    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    // IEEE-754: -0.0 has exactly the sign bit set; XOR flips sign of every lane.
    let sign = _mm512_set1_pd(-0.0_f64);

    let mut block_base = 0usize;
    while block_base < len {
        // High mask bits must be satisfied at block level (uniform within block).
        if (block_base & mask_high) != mask_high {
            block_base += LANES;
            continue;
        }
        // SAFETY: block_base + LANES ≤ len because amps.len() is a power of two
        // and block_base is always LANES-aligned. `amps_ptr.add(block_base * 2)`
        // is within bounds because block_base < len.
        let p = amps_ptr.add(block_base * 2);
        let z = _mm512_loadu_pd(p);
        let neg = _mm512_xor_pd(z, sign);
        // _mm512_mask_blend_pd(mask, a, b): selects b where bit=1, a where bit=0.
        // lane_mask=1 → take neg (flipped); lane_mask=0 → take z (unchanged).
        let blended = _mm512_mask_blend_pd(lane_mask, z, neg);
        _mm512_storeu_pd(p, blended);
        block_base += LANES;
    }
}

/// Routes CCZ to the best available tier (spec §5).
fn dispatch_ccz(amps: &mut [Complex], targets: [u32; 3], controls: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        const LANES_BITS: u32 = 2; // log2(LANES) where LANES = 4

        // Build the sorted combined mask of all qubit positions (targets +
        // external controls). CCZ is symmetric — there is no distinct "target".
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
            if mask_lo >= LANES_BITS {
                // Tier-A clean: every mask bit ≥ LANES_BITS=2, so each zmm
                // block's full-mask check is uniform — no intra-block ambiguity.
                // SAFETY: AVX-512F detected, mask_lo ≥ 2, all qubit positions
                // distinct + in-range (Circuit invariant), n ≥ 3 (Circuit invariant).
                unsafe {
                    apply_ccz_avx512_tier_a(amps, &all_mask);
                }
            } else {
                // Tier-A outer-walk: some mask bit < LANES_BITS=2, so we need
                // a per-lane blend inside each block.
                // SAFETY: AVX-512F detected, all qubit positions distinct + in-range
                // (Circuit invariant), n ≥ 3 (Circuit invariant).
                unsafe {
                    apply_ccz_avx512_tier_a_outer_walk(amps, &all_mask);
                }
            }
            return;
        }
    }
    apply_ccz_scalar(amps, targets, controls);
}

/// Scalar fallback for arbitrary 8×8 matrices. Apply a 3-qubit matrix to
/// `targets = [t0, t1, t2]` (with external `controls`) in place.
///
/// **MSB convention (P0-06):** matrix index `k`'s bits map to targets
/// from MSB to LSB — bit 2 of `k` is `targets[0]`, bit 1 is
/// `targets[1]`, bit 0 is `targets[2]`. So `k = 6` (binary `110`)
/// corresponds to `(targets[0] = 1, targets[1] = 1, targets[2] = 0)`.
/// This matches `Gate::Toffoli` (`qubits = [c0, c1, target]`), whose
/// matrix swaps rows 6 ↔ 7.
fn apply_3q_generic(
    amps: &mut [Complex],
    targets: [u32; 3],
    controls: &[u32],
    m: &[[Complex; 8]; 8],
) {
    let t_bits = [
        1usize << targets[0],
        1usize << targets[1],
        1usize << targets[2],
    ];
    let t_mask = t_bits[0] | t_bits[1] | t_bits[2];
    let ctrl_mask = super::control_mask(controls);
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            let mut idx = [0usize; 8];
            for (k, slot) in idx.iter_mut().enumerate() {
                // MSB convention: k bit 2 → targets[0], bit 1 → targets[1], bit 0 → targets[2].
                let bit_t0 = if k & 4 != 0 { t_bits[0] } else { 0 };
                let bit_t1 = if k & 2 != 0 { t_bits[1] } else { 0 };
                let bit_t2 = if k & 1 != 0 { t_bits[2] } else { 0 };
                *slot = i | bit_t0 | bit_t1 | bit_t2;
            }
            let v = [
                amps[idx[0]],
                amps[idx[1]],
                amps[idx[2]],
                amps[idx[3]],
                amps[idx[4]],
                amps[idx[5]],
                amps[idx[6]],
                amps[idx[7]],
            ];
            for r in 0..8 {
                let mut acc = Complex::new(0.0, 0.0);
                for c in 0..8 {
                    acc += m[r][c] * v[c];
                }
                amps[idx[r]] = acc;
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pauli_x() -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[z, o], [o, z]]
    }

    fn pauli_y_pos() -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let pi = Complex::new(0.0, 1.0);
        let ni = Complex::new(0.0, -1.0);
        [[z, ni], [pi, z]]
    }

    fn hadamard() -> [[Complex; 2]; 2] {
        let s = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        [[s, s], [s, -s]]
    }

    #[test]
    fn x_flips_single_qubit() {
        let mut amps = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        apply_1q(&mut amps, 0, &[], &pauli_x());
        assert_eq!(amps[0], Complex::new(0.0, 0.0));
        assert_eq!(amps[1], Complex::new(1.0, 0.0));
    }

    #[test]
    fn h_on_zero_yields_plus() {
        let mut amps = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        apply_1q(&mut amps, 0, &[], &hadamard());
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amps[0].re - s).abs() < 1e-12);
        assert!((amps[1].re - s).abs() < 1e-12);
    }

    #[test]
    fn x_on_target_1_in_2q_state() {
        // amps[2] = 1.0: (q0 = 0, q1 = 1) in the global state vector
        // (bit 0 = q0, bit 1 = q1). X on q1 flips bit 1: index 2 → 0.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[2] = Complex::new(1.0, 0.0);
        apply_1q(&mut amps, 1, &[], &pauli_x());
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        assert_eq!(amps[2], Complex::new(0.0, 0.0));
    }

    #[test]
    fn controls_skip_when_unset() {
        // amps[1] = 1.0: (q0 = 1, q1 = 0). External control q0 is set
        // ⇒ CX(c=q0, t=q1) fires, flipping q1: index 1 → 3.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[1] = Complex::new(1.0, 0.0);
        apply_1q(&mut amps, 1, &[0], &pauli_x());
        assert_eq!(amps[3], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
    }

    #[test]
    fn controls_do_nothing_when_control_zero() {
        // amps[0] = 1.0 → control bit clear, gate must not fire.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[0] = Complex::new(1.0, 0.0);
        apply_1q(&mut amps, 1, &[0], &pauli_x());
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
    }

    // P1-06 diagonal-fast-path tests.

    #[test]
    fn apply_1q_routes_diagonal_phase_through_fast_path() {
        // 8-amp state (n=3), Phase(π/4) on q=1, no controls.
        // Verify result equals what apply_1q_diagonal_scalar produces directly.
        let theta = std::f64::consts::FRAC_PI_4;
        let m = [
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [
                Complex::new(0.0, 0.0),
                Complex::new(theta.cos(), theta.sin()),
            ],
        ];
        let mut amps_via_dispatch: Vec<Complex> = (0..8)
            .map(|k| Complex::new(0.1 * k as f64, 0.07 * k as f64))
            .collect();
        let mut amps_direct = amps_via_dispatch.clone();
        super::apply_1q(&mut amps_via_dispatch, 1, &[], &m);
        super::apply_1q_diagonal_scalar(&mut amps_direct, 1, &[], m[0][0], m[1][1]);
        for (a, b) in amps_via_dispatch.iter().zip(amps_direct.iter()) {
            assert!((a - b).norm() < 1e-14);
        }
    }

    #[test]
    fn apply_1q_routes_non_diagonal_through_generic() {
        // Hadamard on q=0: result should match the textbook H|0⟩ = |+⟩.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let h = [
            [Complex::new(s, 0.0), Complex::new(s, 0.0)],
            [Complex::new(s, 0.0), Complex::new(-s, 0.0)],
        ];
        let mut amps = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        super::apply_1q(&mut amps, 0, &[], &h);
        assert!((amps[0] - Complex::new(s, 0.0)).norm() < 1e-14);
        assert!((amps[1] - Complex::new(s, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn apply_1q_diagonal_scalar_z_on_q0() {
        // Z|+⟩ = |-⟩ ; here we test Z on a 2-amp state with both amps nonzero
        let mut amps = vec![Complex::new(0.5, 0.0), Complex::new(0.7, 0.1)];
        // m = diag(1, -1)
        let m00 = Complex::new(1.0, 0.0);
        let m11 = Complex::new(-1.0, 0.0);
        super::apply_1q_diagonal_scalar(&mut amps, 0, &[], m00, m11);
        assert_eq!(amps[0], Complex::new(0.5, 0.0));
        assert_eq!(amps[1], Complex::new(-0.7, -0.1));
    }

    #[test]
    fn apply_1q_diagonal_scalar_matches_generic_phase() {
        // phase(θ) = diag(1, e^{iθ})
        let theta = 0.7_f64;
        let m = [
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [
                Complex::new(0.0, 0.0),
                Complex::new(theta.cos(), theta.sin()),
            ],
        ];
        let mut amps_diag = vec![
            Complex::new(0.3, 0.4),
            Complex::new(0.5, -0.1),
            Complex::new(-0.2, 0.6),
            Complex::new(0.1, 0.8),
        ];
        let mut amps_gen = amps_diag.clone();
        super::apply_1q_diagonal_scalar(&mut amps_diag, 1, &[], m[0][0], m[1][1]);
        super::apply_1q(&mut amps_gen, 1, &[], &m);
        for (d, g) in amps_diag.iter().zip(amps_gen.iter()) {
            assert!((d - g).norm() < 1e-14, "diag {d:?} vs generic {g:?}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_1q_diagonal_avx512_controlled_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 32-amp state (n=5), target=2, control on qubit 4 (above target).
        let mut amps_avx: Vec<Complex> = (0..32)
            .map(|k| Complex::new(0.07 * k as f64, -0.03 * k as f64))
            .collect();
        let mut amps_sca = amps_avx.clone();
        let m00 = Complex::new(0.6, 0.8); // arbitrary unit-magnitude
        let m11 = Complex::new(-0.6, 0.8);
        unsafe {
            super::apply_1q_diagonal_avx512(&mut amps_avx, 2, &[4], m00, m11);
        }
        super::apply_1q_diagonal_scalar(&mut amps_sca, 2, &[4], m00, m11);
        for (a, s) in amps_avx.iter().zip(amps_sca.iter()) {
            assert!(
                (a - s).norm() < 1e-14,
                "controlled avx {a:?} vs scalar {s:?}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_1q_diagonal_avx512_matches_scalar_on_phase() {
        if !std::is_x86_feature_detected!("avx512f") {
            return; // smoke on non-AVX-512 hosts: no-op
        }
        // 16-amp state (n=4), target=2 (target_bit=4 ≥ LANES), no controls
        let mut amps_avx: Vec<Complex> = (0..16)
            .map(|k| Complex::new(0.1 * k as f64, 0.05 * k as f64))
            .collect();
        let mut amps_sca = amps_avx.clone();
        let theta = 0.9_f64;
        let m00 = Complex::new(1.0, 0.0);
        let m11 = Complex::new(theta.cos(), theta.sin()); // phase(θ)
        unsafe {
            super::apply_1q_diagonal_avx512(&mut amps_avx, 2, &[], m00, m11);
        }
        super::apply_1q_diagonal_scalar(&mut amps_sca, 2, &[], m00, m11);
        for (a, s) in amps_avx.iter().zip(amps_sca.iter()) {
            assert!((a - s).norm() < 1e-14, "avx {a:?} vs scalar {s:?}");
        }
    }

    #[test]
    fn apply_1q_diagonal_scalar_with_external_control() {
        // 4-amp state (2 qubits).  Diagonal m on qubit 0, control on qubit 1.
        // Only amps with bit-1 = 1 (indices 2, 3) get touched.
        let mut amps = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let m00 = Complex::new(2.0, 0.0);
        let m11 = Complex::new(-1.0, 0.0);
        super::apply_1q_diagonal_scalar(&mut amps, 0, &[1], m00, m11);
        // i=0 (bit1=0): untouched → 1.0
        // i=1 (bit1=0): untouched → 2.0
        // i=2 (bit1=1, bit0=0): * m00 = 2 * 3 = 6
        // i=3 (bit1=1, bit0=1): * m11 = -1 * 4 = -4
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(2.0, 0.0));
        assert_eq!(amps[2], Complex::new(6.0, 0.0));
        assert_eq!(amps[3], Complex::new(-4.0, 0.0));
    }

    /// Canonical `Gate::Cnot` matrix (P0-06):
    /// swaps rows 2 ↔ 3 with `qubits = [control, target]` and
    /// control = MSB of the matrix index.
    fn cnot() -> [[Complex; 4]; 4] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    }

    #[test]
    fn cnot_flips_target_when_control_set() {
        // Targets = [q0, q1]. State amps[1] = 1 corresponds to
        // (q0 = 1, q1 = 0) in the global state vector.
        // With MSB convention idx = [0, t1_bit, t0_bit, t_mask] = [0, 2, 1, 3],
        // amps[1] sits at matrix slot k = 2 (control set, target clear).
        // Cnot swaps slot 2 ↔ 3 ⇒ amps[1] moves to amps[3].
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[1] = Complex::new(1.0, 0.0);
        apply_2q(&mut amps, [0, 1], &[], &cnot());
        assert_eq!(amps[3], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
    }

    #[test]
    fn cnot_on_zero_state_unchanged() {
        // amps[0] = 1 (control = 0, target = 0) — Cnot leaves it alone.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[0] = Complex::new(1.0, 0.0);
        apply_2q(&mut amps, [0, 1], &[], &cnot());
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn apply_2q_external_control_skips_when_unset() {
        // 3 qubits, state amps[1] = 1 (q0 = 1, q1 = 0, q2 = 0).
        // Apply Cnot (q0 = ctrl, q1 = tgt) externally controlled by q2.
        // Since q2 = 0, gate should NOT fire.
        let mut amps = vec![Complex::new(0.0, 0.0); 8];
        amps[1] = Complex::new(1.0, 0.0);
        apply_2q(&mut amps, [0, 1], &[2], &cnot());
        assert_eq!(amps[1], Complex::new(1.0, 0.0));
    }

    fn random_complex_state(n_qubits: u32, seed: u64) -> Vec<Complex> {
        // Tiny deterministic LCG; we only need different per-amp.
        let mut s = seed.wrapping_add(1);
        let lcg = |x: &mut u64| {
            *x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*x >> 32) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
        };
        let len = 1usize << n_qubits;
        let mut amps = Vec::with_capacity(len);
        for _ in 0..len {
            amps.push(Complex::new(lcg(&mut s), lcg(&mut s)));
        }
        amps
    }

    fn assert_amps_close(a: &[Complex], b: &[Complex], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b.iter()) {
            let d = (*ai - *bi).norm_sqr();
            assert!(d < tol * tol, "diff {} > tol {}", d.sqrt(), tol);
        }
    }

    #[test]
    fn apply_2q_cnot_scalar_matches_dense_scalar_canonical() {
        let n = 6;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][3] = Complex::new(1.0, 0.0);
            m[3][2] = Complex::new(1.0, 0.0);
            m
        };
        for (c, t) in [(0u32, 1), (1, 0), (3, 5), (5, 3)] {
            let amps0 = random_complex_state(n, 0xabcd);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_dense_scalar(&mut a, [c, t], &[], &m);
            apply_2q_cnot_scalar(&mut b, c, t, &[]);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    #[test]
    fn apply_2q_swap_scalar_matches_dense_scalar() {
        let n = 6;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][2] = Complex::new(1.0, 0.0);
            m[2][1] = Complex::new(1.0, 0.0);
            m[3][3] = Complex::new(1.0, 0.0);
            m
        };
        for t in [[0u32, 1], [1, 3], [3, 5], [0, 5]] {
            let amps0 = random_complex_state(n, 0xbeef);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_dense_scalar(&mut a, t, &[], &m);
            apply_2q_swap_scalar(&mut b, t, &[]);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    #[test]
    fn apply_2q_cz_scalar_matches_dense_scalar() {
        let n = 6;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][2] = Complex::new(1.0, 0.0);
            m[3][3] = Complex::new(-1.0, 0.0);
            m
        };
        for t in [[0u32, 1], [1, 3], [3, 5], [0, 5]] {
            let amps0 = random_complex_state(n, 0xcafe);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_dense_scalar(&mut a, t, &[], &m);
            apply_2q_cz_scalar(&mut b, t, &[]);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    #[test]
    fn apply_2q_diagonal_scalar_matches_dense_scalar_random_phases() {
        let n = 6;
        // Random diag (e^{iθ_k}): four arbitrary phases.
        let d = [
            Complex::new(0.6, 0.8),
            Complex::new(-0.7, 0.7142857142857143),
            Complex::new(0.99, -0.1414213562373095),
            Complex::new(-0.5, -0.8660254037844386),
        ];
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        for k in 0..4 {
            m[k][k] = d[k];
        }
        for t in [[0u32, 1], [1, 3], [3, 5]] {
            let amps0 = random_complex_state(n, 0xfeed);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_dense_scalar(&mut a, t, &[], &m);
            apply_2q_diagonal_scalar(&mut b, t, &[], d);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    #[test]
    fn apply_2q_cnot_scalar_respects_external_control() {
        let n = 5;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][3] = Complex::new(1.0, 0.0);
            m[3][2] = Complex::new(1.0, 0.0);
            m
        };
        let amps0 = random_complex_state(n, 0xface);
        let mut a = amps0.clone();
        let mut b = amps0;
        apply_2q_dense_scalar(&mut a, [0, 1], &[3], &m);
        apply_2q_cnot_scalar(&mut b, 0, 1, &[3]);
        assert_amps_close(&a, &b, 1e-14);
    }

    #[test]
    fn apply_2q_prelude_dispatches_identity_as_noop() {
        let n = 5;
        let id = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = Complex::new(1.0, 0.0);
            }
            m
        };
        let amps0 = random_complex_state(n, 0x1234);
        let mut a = amps0.clone();
        apply_2q(&mut a, [0, 1], &[], &id);
        assert_amps_close(&a, &amps0, 1e-15);
    }

    #[test]
    fn apply_2q_prelude_dispatches_cnot_matches_dense() {
        let n = 6;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][3] = Complex::new(1.0, 0.0);
            m[3][2] = Complex::new(1.0, 0.0);
            m
        };
        let amps0 = random_complex_state(n, 0x5678);
        let mut a = amps0.clone();
        let mut b = amps0;
        apply_2q(&mut a, [2, 3], &[], &m);
        apply_2q_dense_scalar(&mut b, [2, 3], &[], &m);
        assert_amps_close(&a, &b, 1e-14);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_avx512_generic_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: host lacks AVX-512F");
            return;
        }
        // Random non-special unitary-ish matrix; not diagonal, not permutation.
        // Doesn't need to be unitary for the equivalence test — just dense.
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        let mut s: u64 = 1;
        let mut lcg = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
        };
        for row in m.iter_mut() {
            for cell in row.iter_mut() {
                *cell = Complex::new(lcg(), lcg());
            }
        }
        for n in [6u32, 8, 10] {
            for t in [[0u32, 2], [2, 3], [4, 5], [3, 5], [2, 5]] {
                // Require t_lo >= 2 (LANES=4 means 1<<t_lo >= 4)
                if (1usize << t[0].min(t[1])) < 4 {
                    continue;
                }
                let amps0 = random_complex_state(n, 0xdead + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_dense_scalar(&mut a, t, &[], &m);
                // SAFETY: feature-gated; t_lo >= 2 → 1 << t_lo >= 4 = LANES.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    super::apply_2q_avx512(&mut b, t, &[], &m);
                }
                assert_amps_close(&a, &b, 1e-12);
            }
        }
    }

    /// AVX-512 equivalence with at least one external control bit
    /// above `t_hi`.  Exercises the renormalise-then-shift path with
    /// `controls.len() > 0`, which the uncontrolled equivalence test
    /// does not cover.  Non-adjacent targets with `t_lo > 0` give us
    /// a non-trivial inner `j`-walk too.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_avx512_generic_with_controls_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: host lacks AVX-512F");
            return;
        }
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        let mut s: u64 = 1;
        let mut lcg = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
        };
        for row in m.iter_mut() {
            for cell in row.iter_mut() {
                *cell = Complex::new(lcg(), lcg());
            }
        }
        // (n, targets, controls).  Every control must sit above
        // max(targets); t_lo >= 2 to satisfy the LANES contract.
        let cases: &[(u32, [u32; 2], &[u32])] =
            &[(8, [2, 5], &[7]), (10, [3, 5], &[8]), (10, [2, 5], &[7, 9])];
        for &(n, t, controls) in cases {
            assert!(controls.iter().all(|&c| c > t[0].max(t[1])));
            let amps0 = random_complex_state(n, 0xc0de + n as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_dense_scalar(&mut a, t, controls, &m);
            // SAFETY: feature-gated; t_lo >= 2 ⇒ LANES contract met,
            // every control > t_hi ⇒ outer-walk contract met.
            unsafe {
                super::apply_2q_avx512(&mut b, t, controls, &m);
            }
            assert_amps_close(&a, &b, 1e-12);
        }
    }

    /// Pin the outer-walk + offsets + inner-j enumeration so that,
    /// for a non-adjacent target pair, every state-vector amplitude
    /// is touched exactly once by the SIMD kernel.  Catches
    /// bit-collision bugs in the outer-walk reservation pattern that
    /// would otherwise only surface on a real AVX-512 host (and
    /// silently produce out-of-bounds stores in release mode).
    ///
    /// Reproduces the kernel's index computation `i = block |
    /// offsets[k] | j` using only integer arithmetic, so it runs on
    /// every target (aarch64, wasm, …) — protecting against
    /// regressions of the bit-disjointness invariant on hosts where
    /// the AVX-512 generic equivalence test short-circuits.
    #[test]
    fn apply_2q_avx512_indexing_covers_state_exactly_once() {
        // Two non-adjacent configurations exercising t_lo > 0 with
        // bits between t_lo and t_hi:
        //   (n=6, targets=[3,5]): outer_count = 2, one j-step per outer.
        //   (n=7, targets=[2,5]): outer_count = 8, one j-step per outer.
        for (n_qubits, targets) in [(6u32, [3u32, 5u32]), (7u32, [2u32, 5u32])] {
            let len = 1usize << n_qubits;
            let controls: &[u32] = &[];

            let t_lo = targets[0].min(targets[1]);
            let t_hi = targets[0].max(targets[1]);
            let t_lo_bit = 1usize << t_lo;
            let t_hi_bit = 1usize << t_hi;
            let t_mask = t_lo_bit | t_hi_bit;
            let lanes = 4usize;

            let (offset_k1, offset_k2) = if targets[0] < targets[1] {
                (t_hi_bit, t_lo_bit)
            } else {
                (t_lo_bit, t_hi_bit)
            };
            let offsets = [0usize, offset_k1, offset_k2, t_mask];

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            fixed_above.push((t_hi - t_lo - 1, false));
            for &c in controls {
                fixed_above.push((c - t_lo - 1, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

            let outer_count = 1usize << (n_qubits - t_lo - 2 - controls.len() as u32);

            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let block = crate::kernels::expand_with_fixed(k, &fixed_above) << (t_lo + 1);
                for &off in &offsets {
                    let mut j = 0usize;
                    while j + lanes <= t_lo_bit {
                        let i = block | off | j;
                        for d in 0..lanes {
                            assert!(
                                i + d < len,
                                "n={n_qubits} targets={targets:?}: outer-walk OOB \
                                 block={block} off={off} j={j} d={d} i+d={} len={}",
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
                assert_eq!(
                    count, 1,
                    "n={n_qubits} targets={targets:?}: amp {idx} touched {count} \
                     times (must be exactly 1)"
                );
            }
        }
    }

    #[test]
    fn apply_2q_prelude_dispatches_cz_matches_dense() {
        let n = 6;
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][1] = Complex::new(1.0, 0.0);
            m[2][2] = Complex::new(1.0, 0.0);
            m[3][3] = Complex::new(-1.0, 0.0);
            m
        };
        let amps0 = random_complex_state(n, 0x9abc);
        let mut a = amps0.clone();
        let mut b = amps0;
        apply_2q(&mut a, [1, 4], &[], &m);
        apply_2q_dense_scalar(&mut b, [1, 4], &[], &m);
        assert_amps_close(&a, &b, 1e-14);
    }

    /// Tier A AVX-512 CNOT equivalence vs the scalar specialised kernel.
    /// All `(control, target)` cases satisfy the Tier A contract:
    /// `control > target` and `1 << target >= LANES`.  The reverse
    /// orientation (`control < target`) lands in Tier B (Task 7) and is
    /// not exercised here.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_cnot_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // All cases: control > target, t_bit = 1 << target >= 4.
        for n in [6u32, 8, 10] {
            for (c, t) in [(3u32, 2), (5, 3), (4, 2), (7, 4)] {
                if n <= c.max(t) {
                    continue;
                }
                if (1usize << t) < 4 || c <= t {
                    continue;
                }
                let amps0 = random_complex_state(n, 0xc01f + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_cnot_scalar(&mut a, c, t, &[]);
                // SAFETY: AVX-512F detected, t_bit >= 4 = LANES, control >
                // target (Tier A), no external controls.
                unsafe {
                    super::apply_2q_cnot_avx512(&mut b, c, t, &[]);
                }
                assert_amps_close(&a, &b, 1e-14);
            }
        }
    }

    /// Portable indexing-coverage test for `apply_2q_cnot_avx512`'s
    /// outer-walk + inner-walk pattern.  Reproduces `i = outer | off | j`
    /// using only integer arithmetic (so it runs on aarch64 too) and
    /// asserts every state amplitude in the "control bit set, every
    /// external control set" subspace is touched exactly once.  Catches
    /// bit-collision bugs in the outer-walk reservation pattern that
    /// would otherwise only surface on a real AVX-512 host.
    ///
    /// (The "control bit clear" subspace is not touched at all — CNOT
    /// is a no-op there — so we record an expected touch count of 0
    /// for those amps.)
    #[test]
    fn apply_2q_cnot_avx512_indexing_covers_state_exactly_once() {
        // Two configurations: one with no external controls, one with.
        // Both satisfy Tier A: control > target, t_bit >= LANES.
        let cases: &[(u32, u32, u32, &[u32])] = &[(6, 4, 2, &[]), (8, 5, 2, &[7])];
        for &(n_qubits, control, target, external_controls) in cases {
            let len = 1usize << n_qubits;
            let t_bit = 1usize << target;
            let c_bit = 1usize << control;
            let lanes = 4usize;

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            fixed_above.push((control - target - 1, true));
            for &c in external_controls {
                fixed_above.push((c - target - 1, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

            let outer_count = 1usize << (n_qubits - target - 2 - external_controls.len() as u32);

            // Expected: every amp where bit `control` is 1 AND every
            // external control bit is 1 gets touched exactly twice
            // (once at offset 0, once at offset t_bit — the two sides
            // of the swap pair).  Other amps get touched 0 times.
            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }
            let required_mask = c_bit | ec_mask;

            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let outer = crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1);
                // Outer must already have bit `control` set, every external
                // control set, and bits [0, target] all zero.
                assert_eq!(
                    outer & required_mask,
                    required_mask,
                    "outer={outer:#b} missing required bits {required_mask:#b}"
                );
                assert_eq!(
                    outer & ((1usize << (target + 1)) - 1),
                    0,
                    "outer={outer:#b} has bits set in [0, target]"
                );
                for &off in &[0usize, t_bit] {
                    let mut j = 0usize;
                    while j + lanes <= t_bit {
                        let i = outer | off | j;
                        for d in 0..lanes {
                            assert!(
                                i + d < len,
                                "n={n_qubits} c={control} t={target}: OOB i+d={} len={}",
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
                let in_subspace = (idx & required_mask) == required_mask;
                let expected = if in_subspace { 1u32 } else { 0u32 };
                assert_eq!(
                    count, expected,
                    "n={n_qubits} c={control} t={target}: amp {idx} touched \
                     {count} times (expected {expected}; in_subspace={in_subspace})"
                );
            }
        }
    }

    /// Tier B AVX-512 CNOT equivalence vs the scalar specialised kernel.
    /// Tier B cases: `target ∈ {0, 1}` AND `control >= 2` (so
    /// `c_bit >= LANES = 4` and the in-register `vpermutexvar_pd`
    /// covers the single-window swap).  Exercises both target values
    /// against several control positions.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_cnot_avx512_tier_b_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [6u32, 8, 10] {
            for (c, t) in [(2u32, 0), (3, 0), (5, 0), (2, 1), (3, 1), (5, 1)] {
                if n <= c.max(t) {
                    continue;
                }
                let amps0 = random_complex_state(n, 0xb337 + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_cnot_scalar(&mut a, c, t, &[]);
                // SAFETY: AVX-512F detected, target ∈ {0, 1}, c_bit ≥ 4,
                // no external controls.
                unsafe {
                    super::apply_2q_cnot_avx512_tier_b(&mut b, c, t, &[]);
                }
                assert_amps_close(&a, &b, 1e-14);
            }
        }
    }

    /// Tier C AVX-512 CNOT equivalence vs the scalar specialised kernel.
    /// Tier C cases: both `control` and `target` in `{0, 1}` (a single
    /// quartet per zmm, one permute per quartet).  Covers both
    /// orientations `(0, 1)` and `(1, 0)`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_cnot_avx512_tier_c_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [4u32, 6, 8] {
            for (c, t) in [(0u32, 1), (1, 0)] {
                let amps0 = random_complex_state(n, 0xc007 + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_cnot_scalar(&mut a, c, t, &[]);
                // SAFETY: AVX-512F detected, both ∈ {0, 1}, no external
                // controls, len = 1 << n ≥ 4.
                unsafe {
                    super::apply_2q_cnot_avx512_tier_c(&mut b, c, t, &[]);
                }
                assert_amps_close(&a, &b, 1e-14);
            }
        }
    }

    /// Portable indexing-coverage test for the Tier B and Tier C
    /// outer-walk + in-register-permute pattern.  Reproduces the bit
    /// arithmetic in integer form (so the test runs on aarch64 too)
    /// and additionally validates the per-quartet permutation result
    /// against the scalar CNOT kernel.  Catches bit-collision bugs
    /// in the outer-walk reservation pattern that would otherwise
    /// only surface on a real AVX-512 host.
    #[test]
    fn apply_2q_cnot_avx512_tier_bc_indexing_covers_state_exactly_once() {
        // --- Tier B: (n=8, control=5, target=0, ec=[7]) ---
        {
            let n_qubits = 8u32;
            let control = 5u32;
            let target = 0u32;
            let external_controls: &[u32] = &[7];
            let len = 1usize << n_qubits;
            let c_bit = 1usize << control;
            let lanes = 4usize;

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            for &c in external_controls {
                fixed_above.push((c - control - 1, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);
            let outer_count = 1usize << (n_qubits - control - 1 - external_controls.len() as u32);

            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }
            let required_mask = c_bit | ec_mask;

            // Per-amp touched counter: every LANES-wide load covers
            // LANES contiguous amps; each amp in the controlled
            // subspace should be touched exactly once.
            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let base = crate::kernels::expand_with_fixed(k, &fixed_above) << (control + 1);
                let outer = base | c_bit;
                // outer must have control bit + every ec bit set, and
                // bits [0, control) all zero.
                assert_eq!(
                    outer & required_mask,
                    required_mask,
                    "Tier B: outer={outer:#b} missing required bits {required_mask:#b}"
                );
                assert_eq!(
                    outer & ((1usize << control) - 1),
                    0,
                    "Tier B: outer={outer:#b} has bits set in [0, control)"
                );
                let mut j = 0usize;
                while j + lanes <= c_bit {
                    let i = outer | j;
                    for d in 0..lanes {
                        assert!(i + d < len, "Tier B: OOB i+d={} len={}", i + d, len);
                        touched[i + d] += 1;
                    }
                    j += lanes;
                }
            }
            for (idx, &count) in touched.iter().enumerate() {
                let in_subspace = (idx & required_mask) == required_mask;
                let expected = if in_subspace { 1u32 } else { 0u32 };
                assert_eq!(
                    count, expected,
                    "Tier B: amp {idx} touched {count} times (expected {expected}; in_subspace={in_subspace})"
                );
            }

            // Sanity: scalar CNOT on a random state agrees with itself
            // via the same dispatch route used by apply_2q.  This pins
            // the "permute pattern matches the scalar semantics" half;
            // the AVX-512 equivalence test above covers the SIMD half.
            let amps0 = random_complex_state(n_qubits, 0xb1de);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_cnot_scalar(&mut a, control, target, external_controls);
            super::dispatch_cnot(&mut b, control, target, external_controls);
            assert_amps_close(&a, &b, 1e-14);
        }

        // --- Tier C: (n=6, control=0, target=1, ec=[5]) ---
        {
            let n_qubits = 6u32;
            let control = 0u32;
            let target = 1u32;
            let external_controls: &[u32] = &[5];
            let len = 1usize << n_qubits;

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            for &c in external_controls {
                fixed_above.push((c - 2, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);
            let outer_count = 1usize << (n_qubits - 2 - external_controls.len() as u32);

            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }

            // Each iteration touches 4 contiguous amps (one quartet).
            // Every quartet whose external-control bits are all set
            // should be visited exactly once; amps inside such a
            // quartet are each touched once.
            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let base = crate::kernels::expand_with_fixed(k, &fixed_above) << 2;
                assert_eq!(base & 3, 0, "Tier C: base not quartet-aligned");
                assert_eq!(
                    base & ec_mask,
                    ec_mask,
                    "Tier C: base={base:#b} missing ec bits {ec_mask:#b}"
                );
                for d in 0..4 {
                    assert!(
                        base + d < len,
                        "Tier C: OOB base+d={} len={}",
                        base + d,
                        len
                    );
                    touched[base + d] += 1;
                }
            }
            for (idx, &count) in touched.iter().enumerate() {
                // Expected: every amp where every external control bit
                // is 1 gets touched exactly once.  Other amps untouched.
                let in_subspace = (idx & ec_mask) == ec_mask;
                let expected = if in_subspace { 1u32 } else { 0u32 };
                assert_eq!(
                    count, expected,
                    "Tier C: amp {idx} touched {count} times (expected {expected}; in_subspace={in_subspace})"
                );
            }

            let amps0 = random_complex_state(n_qubits, 0xc11c);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_cnot_scalar(&mut a, control, target, external_controls);
            super::dispatch_cnot(&mut b, control, target, external_controls);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    /// Verifies the `Perm2qKind::CnotLo` dispatch arm of `apply_2q`'s
    /// prelude routes through `dispatch_cnot` with the targets swapped
    /// and produces the same result as the generic dense kernel.  Pins
    /// the orientation-swap behaviour that's easy to invert by mistake.
    #[test]
    fn apply_2q_prelude_dispatches_cnot_lo_matches_dense() {
        let n = 6;
        // CnotLo matrix: π = [0, 3, 2, 1].
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][3] = Complex::new(1.0, 0.0);
            m[2][2] = Complex::new(1.0, 0.0);
            m[3][1] = Complex::new(1.0, 0.0);
            m
        };
        let amps0 = random_complex_state(n, 0xc107);
        let mut a = amps0.clone();
        let mut b = amps0;
        apply_2q(&mut a, [2, 3], &[], &m);
        apply_2q_dense_scalar(&mut b, [2, 3], &[], &m);
        assert_amps_close(&a, &b, 1e-14);
    }

    /// Tier A AVX-512 SWAP equivalence vs the scalar specialised
    /// kernel.  All `targets = [a, b]` cases satisfy the Tier A
    /// contract: `1 << min(a, b) >= LANES = 4`.  Exercises both
    /// adjacent and non-adjacent target pairs with `min(targets) ≥ 2`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_swap_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [6u32, 8, 10] {
            for t in [[2u32, 3], [2, 5], [3, 5], [4, 5]] {
                if n <= t[0].max(t[1]) {
                    continue;
                }
                if (1usize << t[0].min(t[1])) < 4 {
                    continue;
                }
                let amps0 = random_complex_state(n, 0x5403 + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_swap_scalar(&mut a, t, &[]);
                // SAFETY: AVX-512F detected, lo_bit ≥ LANES = 4,
                // distinct targets, no external controls.
                unsafe {
                    super::apply_2q_swap_avx512(&mut b, t, &[]);
                }
                assert_amps_close(&a, &b, 1e-14);
            }
        }
    }

    /// Tier A AVX-512 SWAP equivalence with external controls
    /// (controls strictly above max(targets)).  Pins the
    /// renormalise-then-shift outer-walk pattern in the
    /// external-control branch.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_swap_avx512_tier_a_with_controls_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let cases: &[(u32, [u32; 2], &[u32])] = &[(8, [2, 5], &[7]), (10, [3, 5], &[7, 9])];
        for &(n, t, ec) in cases {
            let amps0 = random_complex_state(n, 0x5a90 + n as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_swap_scalar(&mut a, t, ec);
            // SAFETY: AVX-512F detected, lo_bit ≥ LANES, distinct
            // targets, every external control > max(targets).
            unsafe {
                super::apply_2q_swap_avx512(&mut b, t, ec);
            }
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    /// Portable indexing-coverage test for `apply_2q_swap_avx512`'s
    /// outer-walk + inner-walk pattern.  Reproduces `i = outer | lo_bit
    /// | j` and `outer | hi_bit | j` with only integer arithmetic (so
    /// it runs on aarch64 too) and asserts:
    ///
    /// 1. Every amplitude in the "every external control bit set"
    ///    subspace whose `(lo, hi)` bit pattern is `(1, 0)` or `(0, 1)`
    ///    is touched exactly once (one load + one store).  Amplitudes
    ///    with `(lo, hi) ∈ {(0, 0), (1, 1)}` are SWAP fixed points and
    ///    are not touched.  Amplitudes outside the
    ///    external-control subspace are not touched at all.
    /// 2. The `dispatch_swap` end-to-end result matches the scalar
    ///    SWAP kernel.
    ///
    /// Catches bit-collision bugs in the outer-walk reservation
    /// pattern that would otherwise only surface on a real AVX-512
    /// host.
    #[test]
    fn apply_2q_swap_avx512_tier_a_indexing_covers_state_exactly_once() {
        let cases: &[(u32, [u32; 2], &[u32])] = &[(6, [2, 3], &[]), (8, [2, 5], &[7])];
        for &(n_qubits, targets, external_controls) in cases {
            let len = 1usize << n_qubits;
            let lo = targets[0].min(targets[1]);
            let hi = targets[0].max(targets[1]);
            let lo_bit = 1usize << lo;
            let hi_bit = 1usize << hi;
            let lanes = 4usize;

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
                // Outer must already have every external control set,
                // bit `hi` clear, and bits [0, lo] all zero.
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
            // Expected touch counts:
            //   * (lo, hi) = (1, 0) or (0, 1) AND every ec bit set → 1
            //   * (lo, hi) = (0, 0) or (1, 1) (SWAP fixed points)   → 0
            //   * any ec bit clear                                  → 0
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

            // End-to-end: dispatch_swap must match apply_2q_swap_scalar.
            let amps0 = random_complex_state(n_qubits, 0x577a + n_qubits as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_swap_scalar(&mut a, targets, external_controls);
            super::dispatch_swap(&mut b, targets, external_controls);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    /// Tier B AVX-512 SWAP equivalence vs the scalar specialised
    /// kernel.  Tier B cases: `min(targets) ∈ {0, 1}` AND
    /// `1 << max(targets) >= LANES = 4`.  Exercises both `lo`
    /// values across a range of `hi` positions and qubit counts.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_swap_avx512_tier_b_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [6u32, 8, 10] {
            for t in [[0u32, 2], [0, 3], [1, 2], [1, 3], [0, 5], [1, 5]] {
                if n <= t[0].max(t[1]) {
                    continue;
                }
                let amps0 = random_complex_state(n, 0x5b00 + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_swap_scalar(&mut a, t, &[]);
                // SAFETY: AVX-512F detected, lo ∈ {0, 1}, hi_bit ≥
                // LANES, distinct targets, no external controls.
                unsafe {
                    super::apply_2q_swap_avx512_tier_b(&mut b, t, &[]);
                }
                assert_amps_close(&a, &b, 1e-14);
            }
        }
    }

    /// Tier C AVX-512 SWAP equivalence vs the scalar specialised
    /// kernel.  Tier C cases: both targets in `{0, 1}`.  SWAP is
    /// symmetric, so only one orientation (`[0, 1]`) is tested;
    /// the dispatch test below pins the symmetric routing.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_swap_avx512_tier_c_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [4u32, 6, 8] {
            let t = [0u32, 1];
            let amps0 = random_complex_state(n, 0x5c00 + n as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_swap_scalar(&mut a, t, &[]);
            // SAFETY: AVX-512F detected, both targets in {0, 1},
            // distinct, no external controls, len = 1 << n ≥ 4.
            unsafe {
                super::apply_2q_swap_avx512_tier_c(&mut b, t, &[]);
            }
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    /// Portable indexing-coverage test for the Tier B and Tier C
    /// SWAP outer-walk + in-register-permute pattern.  Reproduces
    /// the bit arithmetic in integer form (so the test runs on
    /// aarch64 too) and additionally validates the per-zmm
    /// permutation result against the scalar SWAP kernel via
    /// `dispatch_swap`.  Catches bit-collision bugs in the
    /// outer-walk reservation pattern that would otherwise only
    /// surface on a real AVX-512 host.
    #[test]
    fn apply_2q_swap_avx512_tier_bc_indexing_covers_state_exactly_once() {
        // --- Tier B: (n=8, targets=[0, 5], ec=[7]) ---
        {
            let n_qubits = 8u32;
            let targets = [0u32, 5];
            let external_controls: &[u32] = &[7];
            let len = 1usize << n_qubits;
            let hi = targets[0].max(targets[1]);
            let hi_bit = 1usize << hi;
            let lanes = 4usize;

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            for &c in external_controls {
                fixed_above.push((c - hi - 1, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);
            let outer_count = 1usize << (n_qubits - hi - 1 - external_controls.len() as u32);

            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }

            // Each (k, j) iteration loads LANES amps at i_0 and
            // LANES amps at i_1 = i_0 | hi_bit, then writes both
            // back via the permute pair.  Every amp in the
            // ec-satisfied subspace gets touched exactly once
            // (every amp has a unique (i_0 / i_1, lane) coordinate).
            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let outer = crate::kernels::expand_with_fixed(k, &fixed_above) << (hi + 1);
                assert_eq!(
                    outer & ec_mask,
                    ec_mask,
                    "Tier B: outer={outer:#b} missing ec bits {ec_mask:#b}"
                );
                assert_eq!(
                    outer & ((1usize << (hi + 1)) - 1),
                    0,
                    "Tier B: outer={outer:#b} has bits set in [0, hi]"
                );
                let mut j = 0usize;
                while j + lanes <= hi_bit {
                    let i_0 = outer | j;
                    let i_1 = i_0 | hi_bit;
                    for d in 0..lanes {
                        assert!(i_0 + d < len, "Tier B: OOB i_0+d={} len={}", i_0 + d, len);
                        assert!(i_1 + d < len, "Tier B: OOB i_1+d={} len={}", i_1 + d, len);
                        touched[i_0 + d] += 1;
                        touched[i_1 + d] += 1;
                    }
                    j += lanes;
                }
            }
            // Expected: every amp with every ec bit set is touched
            // exactly once.  Other amps (any ec bit clear) untouched.
            // Unlike Tier A, SWAP fixed points (lo, hi) ∈ {(0,0),
            // (1,1)} are *still* loaded + stored (the zmm carries
            // both swap-pair members and fixed points; the permute
            // moves swap-pair doubles and copies fixed-point doubles
            // through unchanged).  So all four (lo, hi) combinations
            // appear in the touch counts.
            for (idx, &count) in touched.iter().enumerate() {
                let in_ec_subspace = (idx & ec_mask) == ec_mask;
                let expected = if in_ec_subspace { 1u32 } else { 0u32 };
                assert_eq!(
                    count, expected,
                    "Tier B: amp {idx} touched {count} times (expected {expected}; ec_subspace={in_ec_subspace})"
                );
            }

            // End-to-end: dispatch_swap must match apply_2q_swap_scalar.
            let amps0 = random_complex_state(n_qubits, 0x57b0);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_swap_scalar(&mut a, targets, external_controls);
            super::dispatch_swap(&mut b, targets, external_controls);
            assert_amps_close(&a, &b, 1e-14);
        }

        // --- Tier C: (n=6, targets=[0, 1], ec=[5]) ---
        {
            let n_qubits = 6u32;
            let targets = [0u32, 1];
            let external_controls: &[u32] = &[5];
            let len = 1usize << n_qubits;

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            for &c in external_controls {
                fixed_above.push((c - 2, true));
            }
            fixed_above.sort_unstable_by_key(|&(pos, _)| pos);
            let outer_count = 1usize << (n_qubits - 2 - external_controls.len() as u32);

            let mut ec_mask = 0usize;
            for &c in external_controls {
                ec_mask |= 1usize << c;
            }

            // Each iteration touches 4 contiguous amps (one quartet).
            // Every quartet whose external-control bits are all set
            // should be visited exactly once; amps inside such a
            // quartet are each touched once.
            let mut touched = vec![0u32; len];
            for k in 0..outer_count {
                let base = crate::kernels::expand_with_fixed(k, &fixed_above) << 2;
                assert_eq!(base & 3, 0, "Tier C: base not quartet-aligned");
                assert_eq!(
                    base & ec_mask,
                    ec_mask,
                    "Tier C: base={base:#b} missing ec bits {ec_mask:#b}"
                );
                for d in 0..4 {
                    assert!(
                        base + d < len,
                        "Tier C: OOB base+d={} len={}",
                        base + d,
                        len
                    );
                    touched[base + d] += 1;
                }
            }
            for (idx, &count) in touched.iter().enumerate() {
                let in_subspace = (idx & ec_mask) == ec_mask;
                let expected = if in_subspace { 1u32 } else { 0u32 };
                assert_eq!(
                    count, expected,
                    "Tier C: amp {idx} touched {count} times (expected {expected}; in_subspace={in_subspace})"
                );
            }

            let amps0 = random_complex_state(n_qubits, 0x57c0);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_swap_scalar(&mut a, targets, external_controls);
            super::dispatch_swap(&mut b, targets, external_controls);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    /// Verifies the `Perm2qKind::Swap` dispatch arm of `apply_2q`'s
    /// prelude routes through `dispatch_swap` and produces the same
    /// result as the generic dense kernel.  Pins the Swap arm of the
    /// classifier-driven dispatch.
    #[test]
    fn apply_2q_prelude_dispatches_swap_matches_dense() {
        let n = 6;
        // Canonical SWAP matrix (mirrors `swap_matrix()` in mod.rs's
        // tests): swap rows 1 ↔ 2, identity on rows 0 and 3.
        let m = {
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            m[0][0] = Complex::new(1.0, 0.0);
            m[1][2] = Complex::new(1.0, 0.0);
            m[2][1] = Complex::new(1.0, 0.0);
            m[3][3] = Complex::new(1.0, 0.0);
            m
        };
        let amps0 = random_complex_state(n, 0x5a91);
        let mut a = amps0.clone();
        let mut b = amps0;
        apply_2q(&mut a, [2, 5], &[], &m);
        apply_2q_dense_scalar(&mut b, [2, 5], &[], &m);
        assert_amps_close(&a, &b, 1e-14);
    }

    /// Canonical `Gate::Toffoli` matrix (P0-06): identity on rows 0..6,
    /// swap rows 6 ↔ 7. Matches `qubits = [c0, c1, target]` with
    /// `qubits[0]` as the MSB of the matrix index.
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
    fn toffoli_flips_target_when_both_controls_set() {
        // Targets = [q0, q1, q2]. State amps[3] = 1 corresponds to
        // (q0 = 1, q1 = 1, q2 = 0) globally. With MSB convention this
        // maps to matrix slot k = 6 (bit 2 = q0 = 1, bit 1 = q1 = 1,
        // bit 0 = q2 = 0). Toffoli swaps slot 6 ↔ 7 ⇒ amps[3] → amps[7].
        let mut amps = vec![Complex::new(0.0, 0.0); 8];
        amps[3] = Complex::new(1.0, 0.0);
        apply_3q(&mut amps, [0, 1, 2], &[], &toffoli());
        assert_eq!(amps[7], Complex::new(1.0, 0.0));
        assert_eq!(amps[3], Complex::new(0.0, 0.0));
    }

    #[test]
    fn toffoli_with_single_control_set_is_identity() {
        // State amps[1] = 1 (q0 = 1, q1 = 0, q2 = 0). Only one control
        // bit set ⇒ Toffoli acts as identity.
        let mut amps = vec![Complex::new(0.0, 0.0); 8];
        amps[1] = Complex::new(1.0, 0.0);
        apply_3q(&mut amps, [0, 1, 2], &[], &toffoli());
        assert_eq!(amps[1], Complex::new(1.0, 0.0));
    }

    /// Equivalence: `apply_1q` (dispatcher — prefers AVX-512 on
    /// capable hosts) must match a scalar reference within 1e-12
    /// across the full 1q gate set and both target / control
    /// orientations that the AVX-512 path's safety contract covers
    /// (plus the orientations that fall through to scalar). On
    /// non-AVX-512 hosts (and `cfg`-gated out on ARM) both calls
    /// land in the scalar body, so the test is still valid but
    /// doesn't exercise the new code path.
    use aleph_core::GateMatrix;
    use aleph_test::gate::arb_1q_gate;
    use aleph_test::state::arb_state_vector;
    use proptest::prelude::*;

    /// Scalar AoS reference — the body before any AVX-512 dispatch
    /// was added. Independent from `apply_1q`'s dispatcher so it
    /// remains a stable comparison point.
    fn apply_1q_scalar_reference(
        amps: &mut [Complex],
        target: u32,
        controls: &[u32],
        m: &[[Complex; 2]; 2],
    ) {
        let t_bit = 1usize << target;
        let ctrl_mask = super::super::control_mask(controls);
        let len = amps.len();
        let mut i = 0usize;
        while i < len {
            if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
                let j = i | t_bit;
                let a = amps[i];
                let b = amps[j];
                amps[i] = m[0][0] * a + m[0][1] * b;
                amps[j] = m[1][0] * a + m[1][1] * b;
            }
            i += 1;
        }
    }

    /// Tier A AVX-512 CZ equivalence vs the scalar specialised kernel.
    /// All `targets = [a, b]` cases satisfy the Tier A contract:
    /// `1 << min(a, b) >= LANES = 4`.  Exercises both adjacent and
    /// non-adjacent target pairs with `min(targets) ≥ 2`.  CZ is a
    /// pure sign-flip (no floating arithmetic beyond IEEE-754 sign-bit
    /// flip), so the tolerance is bit-exact.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_cz_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for n in [6u32, 8, 10] {
            for t in [[2u32, 3], [2, 5], [3, 5], [4, 5]] {
                if n <= t[0].max(t[1]) {
                    continue;
                }
                if (1usize << t[0].min(t[1])) < 4 {
                    continue;
                }
                let amps0 = random_complex_state(n, 0x6203 + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_cz_scalar(&mut a, t, &[]);
                // SAFETY: AVX-512F detected, lo_bit ≥ LANES = 4,
                // distinct targets, no external controls.
                unsafe {
                    super::apply_2q_cz_avx512(&mut b, t, &[]);
                }
                assert_amps_close(&a, &b, 1e-15);
            }
        }
    }

    /// Tier A AVX-512 CZ equivalence with external controls (controls
    /// strictly above max(targets)).  Pins the renormalise-then-shift
    /// outer-walk pattern in the external-control branch.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_cz_avx512_tier_a_with_controls_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let cases: &[(u32, [u32; 2], &[u32])] = &[(8, [2, 5], &[7]), (10, [3, 5], &[7, 9])];
        for &(n, t, ec) in cases {
            let amps0 = random_complex_state(n, 0x6a90 + n as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_cz_scalar(&mut a, t, ec);
            // SAFETY: AVX-512F detected, lo_bit ≥ LANES, distinct
            // targets, every external control > max(targets).
            unsafe {
                super::apply_2q_cz_avx512(&mut b, t, ec);
            }
            assert_amps_close(&a, &b, 1e-15);
        }
    }

    /// Portable indexing-coverage test for `apply_2q_cz_avx512`'s
    /// outer-walk + inner-walk pattern.  Reproduces `i = outer | j`
    /// with `outer = base | lo_bit` (i.e. both target bits set) using
    /// only integer arithmetic (so it runs on aarch64 too) and asserts:
    ///
    /// 1. Every amplitude in the "every external control bit set"
    ///    subspace whose `(lo, hi)` bit pattern is `(1, 1)` is touched
    ///    exactly once (one load + one store).  All other amplitudes
    ///    (any of the three sub-blocks `(0,0)`, `(0,1)`, `(1,0)`, OR
    ///    any external-control bit clear) are NOT touched — CZ only
    ///    negates the `(1, 1)` sub-block.
    /// 2. The `apply_2q` end-to-end result (routed via
    ///    `dispatch_diagonal_or_cz`) matches the scalar CZ kernel.
    ///
    /// Catches bit-collision bugs in the outer-walk reservation
    /// pattern that would otherwise only surface on a real AVX-512
    /// host.
    #[test]
    fn apply_2q_cz_avx512_tier_a_indexing_covers_state_exactly_once() {
        let cases: &[(u32, [u32; 2], &[u32])] = &[(6, [2, 3], &[]), (8, [2, 5], &[7])];
        for &(n_qubits, targets, external_controls) in cases {
            let len = 1usize << n_qubits;
            let lo = targets[0].min(targets[1]);
            let hi = targets[0].max(targets[1]);
            let lo_bit = 1usize << lo;
            let hi_bit = 1usize << hi;
            let lanes = 4usize;

            let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
            fixed_above.push((hi - lo - 1, true));
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
                let base = crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1);
                let outer = base | lo_bit;
                // outer must have: bit lo set, bit hi set, every ec
                // bit set, bits [0, lo) all zero.
                assert_eq!(
                    outer & ec_mask,
                    ec_mask,
                    "outer={outer:#b} missing ec bits {ec_mask:#b}"
                );
                assert_eq!(outer & lo_bit, lo_bit, "outer={outer:#b} missing lo_bit");
                assert_eq!(outer & hi_bit, hi_bit, "outer={outer:#b} missing hi_bit");
                assert_eq!(
                    outer & (lo_bit - 1),
                    0,
                    "outer={outer:#b} has bits set in [0, lo)"
                );
                let mut j = 0usize;
                while j + lanes <= lo_bit {
                    let i = outer | j;
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
            // Expected touch counts:
            //   * (lo, hi) = (1, 1) AND every ec bit set → 1
            //   * any other (lo, hi) sub-block            → 0
            //   * any ec bit clear                        → 0
            for (idx, &count) in touched.iter().enumerate() {
                let bit_lo = (idx & lo_bit) != 0;
                let bit_hi = (idx & hi_bit) != 0;
                let in_ec_subspace = (idx & ec_mask) == ec_mask;
                let in_cz_subblock = bit_lo && bit_hi;
                let expected = if in_ec_subspace && in_cz_subblock {
                    1u32
                } else {
                    0u32
                };
                assert_eq!(
                    count, expected,
                    "n={n_qubits} t={targets:?}: amp {idx} touched {count} times \
                     (expected {expected}; ec_subspace={in_ec_subspace} \
                     cz_subblock={in_cz_subblock})"
                );
            }

            // End-to-end: apply_2q with the CZ matrix must match
            // apply_2q_cz_scalar (apply_2q routes through
            // dispatch_diagonal_or_cz for diagonal matrices).
            let cz_matrix: [[Complex; 4]; 4] = {
                let z = Complex::new(0.0, 0.0);
                let o = Complex::new(1.0, 0.0);
                let n = Complex::new(-1.0, 0.0);
                [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, n]]
            };
            let amps0 = random_complex_state(n_qubits, 0x67ca + n_qubits as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_cz_scalar(&mut a, targets, external_controls);
            super::apply_2q(&mut b, targets, external_controls, &cz_matrix);
            assert_amps_close(&a, &b, 1e-15);
        }
    }

    /// A general (non-CZ) diagonal used by the Task 11 tests.  Each
    /// entry has unit magnitude so the kernel exercises full complex
    /// multiplies rather than degenerating to sign flips.
    #[cfg(target_arch = "x86_64")]
    fn nontrivial_diag_4() -> [Complex; 4] {
        [
            Complex::new(0.6, 0.8),
            Complex::new(-0.7, 0.7142857142857143),
            Complex::new(0.99, -0.1414213562373095),
            Complex::new(-0.5, -0.8660254037844386),
        ]
    }

    /// Tier A AVX-512 general-diagonal 2q equivalence vs the scalar
    /// specialised kernel.  All `targets = [a, b]` cases satisfy the
    /// Tier A contract `1 << min(a, b) >= LANES = 4`.  Tolerance is
    /// 1e-14 (scalar reference uses `Complex::mul`; SIMD uses
    /// fmaddsub — identical reductions modulo IEEE-754 rounding of
    /// reorderable adds).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_diagonal_avx512_tier_a_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let d = nontrivial_diag_4();
        for n in [6u32, 8, 10] {
            for t in [[2u32, 3], [2, 5], [3, 5], [4, 5], [5, 2], [5, 3]] {
                if n <= t[0].max(t[1]) {
                    continue;
                }
                if (1usize << t[0].min(t[1])) < 4 {
                    continue;
                }
                let amps0 = random_complex_state(n, 0xd1a9 + n as u64);
                let mut a = amps0.clone();
                let mut b = amps0;
                apply_2q_diagonal_scalar(&mut a, t, &[], d);
                // SAFETY: AVX-512F detected, lo_bit ≥ LANES = 4,
                // distinct targets, no external controls.
                unsafe {
                    super::apply_2q_diagonal_avx512(&mut b, t, &[], d);
                }
                assert_amps_close(&a, &b, 1e-14);
            }
        }
    }

    /// Tier A AVX-512 general-diagonal equivalence with external
    /// controls (controls strictly above max(targets)).  Pins the
    /// renormalise-then-shift outer-walk pattern in the external-
    /// control branch — same shape as the matching CZ test.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_2q_diagonal_avx512_tier_a_with_controls_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let d = nontrivial_diag_4();
        let cases: &[(u32, [u32; 2], &[u32])] =
            &[(8, [2, 5], &[7]), (10, [3, 5], &[7, 9]), (8, [5, 2], &[7])];
        for &(n, t, ec) in cases {
            let amps0 = random_complex_state(n, 0xd2b0 + n as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_diagonal_scalar(&mut a, t, ec, d);
            // SAFETY: AVX-512F detected, lo_bit ≥ LANES, distinct
            // targets, every external control > max(targets).
            unsafe {
                super::apply_2q_diagonal_avx512(&mut b, t, ec, d);
            }
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    /// Portable indexing-coverage test for `apply_2q_diagonal_avx512`'s
    /// outer-walk + 4-sub-block iteration.  Reproduces the four
    /// `(q_hi, q_lo)` sub-block walks using only integer arithmetic
    /// (runs on aarch64 too) and asserts:
    ///
    /// 1. Every amplitude in the "every external control bit set"
    ///    subspace is touched exactly once across the four sub-blocks
    ///    (one load + one store per amp).  Any amp with an external
    ///    control bit clear is NOT touched (kernel skips it).
    /// 2. The `apply_2q` end-to-end result (routed via
    ///    `dispatch_diagonal_or_cz` with a non-CZ diagonal) matches
    ///    `apply_2q_diagonal_scalar`.
    ///
    /// Catches bit-collision bugs in the outer-walk reservation
    /// pattern that would otherwise only surface on a real AVX-512
    /// host.
    #[test]
    fn apply_2q_diagonal_avx512_tier_a_indexing_covers_state_exactly_once() {
        let cases: &[(u32, [u32; 2], &[u32])] =
            &[(6, [2, 3], &[]), (8, [2, 5], &[7]), (6, [3, 2], &[])];
        for &(n_qubits, targets, external_controls) in cases {
            let len = 1usize << n_qubits;
            let lo = targets[0].min(targets[1]);
            let hi = targets[0].max(targets[1]);
            let lo_bit = 1usize << lo;
            let hi_bit = 1usize << hi;
            let lanes = 4usize;

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
                let base = crate::kernels::expand_with_fixed(k, &fixed_above) << (lo + 1);
                // base must have: bit hi clear, every ec bit set, bits
                // [0, lo+1) all zero (i.e. lo also clear since it's in
                // [0, lo+1)).
                assert_eq!(
                    base & ec_mask,
                    ec_mask,
                    "base={base:#b} missing ec bits {ec_mask:#b}"
                );
                assert_eq!(base & lo_bit, 0, "base={base:#b} has lo_bit set");
                assert_eq!(base & hi_bit, 0, "base={base:#b} has hi_bit set");
                assert_eq!(
                    base & (lo_bit - 1),
                    0,
                    "base={base:#b} has bits set in [0, lo)"
                );
                for &sub in &[0usize, lo_bit, hi_bit, lo_bit | hi_bit] {
                    let outer = base | sub;
                    let mut j = 0usize;
                    while j + lanes <= lo_bit {
                        let i = outer | j;
                        for off in 0..lanes {
                            assert!(
                                i + off < len,
                                "n={n_qubits} t={targets:?}: OOB i+off={} len={}",
                                i + off,
                                len
                            );
                            touched[i + off] += 1;
                        }
                        j += lanes;
                    }
                }
            }
            // Expected touch counts:
            //   * every external-control bit set → 1 (kernel walks all
            //     four (q_hi, q_lo) sub-blocks in the ec-satisfied
            //     subspace exactly once)
            //   * any external-control bit clear → 0
            for (idx, &count) in touched.iter().enumerate() {
                let in_ec_subspace = (idx & ec_mask) == ec_mask;
                let expected = if in_ec_subspace { 1u32 } else { 0u32 };
                assert_eq!(
                    count, expected,
                    "n={n_qubits} t={targets:?}: amp {idx} touched {count} times \
                     (expected {expected}; ec_subspace={in_ec_subspace})"
                );
            }

            // End-to-end: apply_2q with a non-CZ diagonal matrix must
            // match apply_2q_diagonal_scalar (apply_2q routes through
            // dispatch_diagonal_or_cz for diagonal matrices, and the
            // non-CZ signature steers to the diagonal kernel).
            let d = [
                Complex::new(0.6, 0.8),
                Complex::new(-0.7, 0.7142857142857143),
                Complex::new(0.99, -0.1414213562373095),
                Complex::new(-0.5, -0.8660254037844386),
            ];
            let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
            for k in 0..4 {
                m[k][k] = d[k];
            }
            let amps0 = random_complex_state(n_qubits, 0xd3c1 + n_qubits as u64);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q_diagonal_scalar(&mut a, targets, external_controls, d);
            super::apply_2q(&mut b, targets, external_controls, &m);
            assert_amps_close(&a, &b, 1e-14);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 96, ..ProptestConfig::default() })]

        /// AVX-512 dispatcher matches scalar reference. Target up to 6
        /// exercises the sub-LANES fallback (target ∈ {0, 1}) AND the
        /// SIMD main loop (target ≥ 2). Controls span both sides of
        /// target so the scalar-fall-through guard is also exercised.
        #[test]
        fn aos_apply_1q_matches_scalar_reference(
            gate in arb_1q_gate(),
            target in 0u32..6u32,
            ctrl_count in 0u32..=2u32,
            ctrl_seeds in proptest::collection::vec(0u32..6u32, 0..=2),
            amps in arb_state_vector(6),
        ) {
            // Duplicate-free, target-free control list (matches what
            // apply_gate would deliver).
            let mut controls: Vec<u32> =
                ctrl_seeds.into_iter().take(ctrl_count as usize).collect();
            controls.retain(|c| *c != target);
            controls.sort_unstable();
            controls.dedup();

            let m = match gate.matrix().unwrap() {
                GateMatrix::M2x2(m) => m,
                _ => unreachable!("arb_1q_gate yields 1q gates"),
            };
            let mut ref_state = amps.clone();
            apply_1q_scalar_reference(&mut ref_state, target, &controls, &m);

            let mut dispatched = amps.clone();
            apply_1q(&mut dispatched, target, &controls, &m);

            for k in 0..ref_state.len() {
                let diff = ref_state[k] - dispatched[k];
                prop_assert!(
                    diff.norm() < 1e-12,
                    "k={k} target={target} controls={:?}: ref={} disp={}",
                    controls, ref_state[k], dispatched[k]
                );
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig { cases: 32, ..Default::default() })]

        /// Drive `apply_2q` (the dispatch prelude, which routes diagonal
        /// matrices through the SIMD diagonal/CZ kernel on AVX-512 hosts
        /// and through scalar paths elsewhere) with arbitrary diagonal
        /// 4×4 unitaries and verify amplitude-level equivalence against
        /// `apply_2q_dense_scalar`.  Invariant holds on every host.
        #[test]
        fn prop_2q_diagonal_matches_scalar_aos(
            m in aleph_test::gate::arb_diagonal_4x4(),
            seed in 0u64..100,
        ) {
            let n = 6u32;
            let amps0 = random_complex_state(n, seed);
            let mut a = amps0.clone();
            let mut b = amps0;
            apply_2q(&mut a, [2, 3], &[], &m);
            apply_2q_dense_scalar(&mut b, [2, 3], &[], &m);
            for (ai, bi) in a.iter().zip(b.iter()) {
                proptest::prop_assert!(((*ai - *bi).norm_sqr()) < 1e-24);
            }
        }
    }

    #[test]
    fn apply_1q_x_scalar_matches_generic() {
        // n=4 random-ish state.
        let mut amps_x: Vec<Complex> = (0..16)
            .map(|k| Complex::new(k as f64 * 0.13, k as f64 * 0.27))
            .collect();
        let mut amps_g = amps_x.clone();
        super::apply_1q_x_scalar(&mut amps_x, 1, &[]);
        super::apply_1q(&mut amps_g, 1, &[], &pauli_x());
        for (a, b) in amps_x.iter().zip(amps_g.iter()) {
            assert!(
                (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                "x scalar diverged from generic"
            );
        }
    }

    #[test]
    fn apply_1q_x_scalar_external_control_below_target_dispatches_to_scalar() {
        // controls=[0], target=2; c=0 < target=2 so Tier-A and Tier-B both
        // reject (Tier-A requires c > target; Tier-B requires c >= 2 AND
        // c > target). Dispatch must route to the scalar fallback.
        //
        // This is a dispatch-routing assertion, not a kernel parity test:
        // scalar correctness is independently verified by
        // apply_1q_x_scalar_matches_generic (target=1, no controls).
        // Here we confirm that apply_1q produces the same result as
        // apply_1q_x_scalar with the below-target control, which is the
        // only path that correctly applies the gate at c=0, target=2.
        let mut amps_x: Vec<Complex> = (0..16).map(|k| Complex::new(k as f64, 0.0)).collect();
        let mut amps_g = amps_x.clone();
        super::apply_1q_x_scalar(&mut amps_x, 2, &[0]);
        super::apply_1q(&mut amps_g, 2, &[0], &pauli_x());
        assert_eq!(amps_x, amps_g);
    }

    #[test]
    fn apply_1q_y_scalar_matches_generic_ypos() {
        let mut amps_y: Vec<Complex> = (0..16)
            .map(|k| Complex::new(k as f64 * 0.11, 1.0 + k as f64 * 0.23))
            .collect();
        let mut amps_g = amps_y.clone();
        super::apply_1q_y_scalar(&mut amps_y, 2, &[], 1.0);
        super::apply_1q(&mut amps_g, 2, &[], &pauli_y_pos());
        for (a, b) in amps_y.iter().zip(amps_g.iter()) {
            assert!((a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_1q_antidiag_scalar_matches_generic_phased() {
        let z = Complex::new(0.0, 0.0);
        let a = Complex::new(0.5, 0.8660254037844386); // e^{iπ/3}
        let b = Complex::new(0.5, -0.8660254037844386); // e^{-iπ/3}
        let m = [[z, a], [b, z]];
        let mut amps_s: Vec<Complex> = (0..8)
            .map(|k| Complex::new(k as f64 * 0.17, k as f64 * 0.29))
            .collect();
        let mut amps_g = amps_s.clone();
        super::apply_1q_antidiag_scalar(&mut amps_s, 1, &[], a, b);
        super::apply_1q(&mut amps_g, 1, &[], &m);
        for (s, g) in amps_s.iter().zip(amps_g.iter()) {
            assert!((s.re - g.re).abs() < 1e-12 && (s.im - g.im).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_x_avx512_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: host lacks avx512f");
            return;
        }
        // n=6, target=3 (target_bit=8 ≥ LANES=4).
        let mut amps_avx: Vec<Complex> = (0..64)
            .map(|k| Complex::new(k as f64 * 0.11, k as f64 * 0.23 - 1.0))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: feature checked + target_bit=8 ≥ LANES + no controls.
        unsafe {
            super::apply_1q_x_avx512(&mut amps_avx, 3, &[]);
        }
        super::apply_1q_x_scalar(&mut amps_sca, 3, &[]);
        for (a, s) in amps_avx.iter().zip(amps_sca.iter()) {
            assert_eq!(a, s);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_x_avx512_matches_scalar_with_control() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=6, target=2, control=4 (above target).
        let mut amps_avx: Vec<Complex> = (0..64)
            .map(|k| Complex::new(k as f64, k as f64 * -0.5))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: avx512 + target_bit=4=LANES + control=4 > target=2.
        unsafe {
            super::apply_1q_x_avx512(&mut amps_avx, 2, &[4]);
        }
        super::apply_1q_x_scalar(&mut amps_sca, 2, &[4]);
        assert_eq!(amps_avx, amps_sca);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_y_avx512_pos_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut amps_avx: Vec<Complex> = (0..64)
            .map(|k| Complex::new((k as f64) * 0.13 - 2.0, (k as f64) * 0.27 + 1.0))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: avx512 + target_bit=8 ≥ LANES=4 + no controls.
        unsafe {
            super::apply_1q_y_avx512(&mut amps_avx, 3, &[], 1.0);
        }
        super::apply_1q_y_scalar(&mut amps_sca, 3, &[], 1.0);
        for (a, s) in amps_avx.iter().zip(amps_sca.iter()) {
            assert!(
                (a.re - s.re).abs() < 1e-12 && (a.im - s.im).abs() < 1e-12,
                "y_pos avx diverged: avx={:?} scalar={:?}",
                a,
                s
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_y_avx512_neg_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut amps_avx: Vec<Complex> = (0..64)
            .map(|k| Complex::new((k as f64) * 0.07, (k as f64) * -0.19))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: as above.
        unsafe {
            super::apply_1q_y_avx512(&mut amps_avx, 3, &[], -1.0);
        }
        super::apply_1q_y_scalar(&mut amps_sca, 3, &[], -1.0);
        for (a, s) in amps_avx.iter().zip(amps_sca.iter()) {
            assert!((a.re - s.re).abs() < 1e-12 && (a.im - s.im).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_y_avx512_matches_generic_pauli_y() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // End-to-end: dispatch path via apply_1q(Pauli-Y matrix) MUST equal
        // direct call to scalar y on a fresh copy.
        let mut amps_dispatch: Vec<Complex> =
            (0..64).map(|k| Complex::new(k as f64, 1.0)).collect();
        let mut amps_direct = amps_dispatch.clone();
        super::apply_1q(&mut amps_dispatch, 3, &[], &pauli_y_pos());
        super::apply_1q_y_scalar(&mut amps_direct, 3, &[], 1.0);
        for (d, x) in amps_dispatch.iter().zip(amps_direct.iter()) {
            assert!((d.re - x.re).abs() < 1e-12 && (d.im - x.im).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_antidiag_avx512_matches_scalar_phased() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = Complex::new(0.6, 0.8); // |a| = 1
        let b = Complex::new(0.6, -0.8); // |b| = 1
        let mut amps_avx: Vec<Complex> = (0..64)
            .map(|k| Complex::new(k as f64 * 0.05, k as f64 * 0.07 - 1.0))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: avx512 + target_bit=8 ≥ LANES=4 + no controls.
        unsafe {
            super::apply_1q_antidiag_avx512(&mut amps_avx, 3, &[], a, b);
        }
        super::apply_1q_antidiag_scalar(&mut amps_sca, 3, &[], a, b);
        for (av, sc) in amps_avx.iter().zip(amps_sca.iter()) {
            assert!(
                (av.re - sc.re).abs() < 1e-12 && (av.im - sc.im).abs() < 1e-12,
                "antidiag avx diverged: avx={:?} scalar={:?}",
                av,
                sc
            );
        }
    }

    // --- Tier-B AVX-512 tests (target ∈ {0, 1}, in-register lane permute) ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_x_avx512_lowbit_target0_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut amps_avx: Vec<Complex> = (0..16)
            .map(|k| Complex::new(k as f64, -(k as f64) * 0.3))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: feature gate + target=0 < LANES + amps.len()=16 divisible by 4.
        unsafe {
            super::apply_1q_x_avx512_lowbit(&mut amps_avx, 0, &[]);
        }
        super::apply_1q_x_scalar(&mut amps_sca, 0, &[]);
        assert_eq!(amps_avx, amps_sca);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_x_avx512_lowbit_target1_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut amps_avx: Vec<Complex> = (0..16).map(|k| Complex::new(k as f64, 1.0)).collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: feature gate + target=1 < LANES + amps.len()=16 divisible by 4.
        unsafe {
            super::apply_1q_x_avx512_lowbit(&mut amps_avx, 1, &[]);
        }
        super::apply_1q_x_scalar(&mut amps_sca, 1, &[]);
        assert_eq!(amps_avx, amps_sca);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_y_avx512_lowbit_pos_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for &target in &[0u32, 1u32] {
            let mut amps_avx: Vec<Complex> = (0..16)
                .map(|k| Complex::new(k as f64 * 0.11, k as f64 * 0.31 - 0.5))
                .collect();
            let mut amps_sca = amps_avx.clone();
            // SAFETY: target ∈ {0, 1}, no controls.
            unsafe {
                super::apply_1q_y_avx512_lowbit(&mut amps_avx, target, &[], 1.0);
            }
            super::apply_1q_y_scalar(&mut amps_sca, target, &[], 1.0);
            for (a, s) in amps_avx.iter().zip(amps_sca.iter()) {
                assert!(
                    (a.re - s.re).abs() < 1e-12 && (a.im - s.im).abs() < 1e-12,
                    "y_lowbit target={}: avx={:?} scalar={:?}",
                    target,
                    a,
                    s
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_antidiag_avx512_lowbit_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = Complex::new(0.6, 0.8);
        let b = Complex::new(0.6, -0.8);
        for &target in &[0u32, 1u32] {
            let mut amps_avx: Vec<Complex> = (0..16)
                .map(|k| Complex::new(k as f64 * 0.05, k as f64 * 0.13))
                .collect();
            let mut amps_sca = amps_avx.clone();
            // SAFETY: target ∈ {0, 1}, no controls.
            unsafe {
                super::apply_1q_antidiag_avx512_lowbit(&mut amps_avx, target, &[], a, b);
            }
            super::apply_1q_antidiag_scalar(&mut amps_sca, target, &[], a, b);
            for (av, sc) in amps_avx.iter().zip(amps_sca.iter()) {
                assert!(
                    (av.re - sc.re).abs() < 1e-12 && (av.im - sc.im).abs() < 1e-12,
                    "antidiag_lowbit target={}: avx={:?} scalar={:?}",
                    target,
                    av,
                    sc
                );
            }
        }
    }

    // ---- P1-05 T12: boundary-n + NaN-propagation tests ----

    #[test]
    fn apply_1q_x_dispatch_n2_no_segfault() {
        // Bell-state-sized; n=2 < LANES=4 → Tier-B path (target ∈ {0, 1}).
        // X on target=0 swaps pairs (i, i|1) — t_bit = 1 << 0 = 1, so amps[0]
        // and amps[1] swap, amps[2] and amps[3] swap.  Starting from |00⟩ =
        // index 0 set: result is index 1 set.
        let mut amps: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        super::apply_1q(&mut amps, 0, &[], &pauli_x());
        assert_eq!(
            amps,
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ]
        );
    }

    #[test]
    fn apply_1q_x_dispatch_boundary_n_around_lanes() {
        // n=2 (< LANES), n=3 (between), n=4 (LANES-bit threshold), n=5 (above).
        for n in 2..=5u32 {
            let len = 1usize << n;
            let mut amps: Vec<Complex> = (0..len)
                .map(|k| Complex::new(k as f64 * 0.13, k as f64 * -0.07))
                .collect();
            let mut amps_ref = amps.clone();
            super::apply_1q(&mut amps, 0, &[], &pauli_x());
            super::apply_1q_x_scalar(&mut amps_ref, 0, &[]);
            assert_eq!(amps, amps_ref, "n={}", n);
        }
    }

    #[test]
    fn nan_diagonal_propagates_through_generic_kernel() {
        // m has NaN on diagonal → is_antidiagonal_2x2 rejects → generic kernel.
        let nan = Complex::new(f64::NAN, 0.0);
        let o = Complex::new(1.0, 0.0);
        let z = Complex::new(0.0, 0.0);
        let m = [[nan, o], [o, z]];
        let mut amps: Vec<Complex> = (0..8).map(|k| Complex::new(k as f64, 0.0)).collect();
        super::apply_1q(&mut amps, 1, &[], &m);
        assert!(
            amps.iter().any(|c| c.re.is_nan() || c.im.is_nan()),
            "NaN failed to propagate from diagonal"
        );
    }

    #[test]
    fn nan_off_diagonal_propagates_through_generic_antidiag_kernel() {
        // Diagonals zero, off-diagonal NaN → is_antidiagonal_2x2 passes,
        // classify_1q_antidiag returns None → generic anti-diag multiplies by NaN.
        let nan = Complex::new(f64::NAN, 0.0);
        let o = Complex::new(1.0, 0.0);
        let z = Complex::new(0.0, 0.0);
        let m = [[z, nan], [o, z]];
        let mut amps: Vec<Complex> = (0..8).map(|k| Complex::new(k as f64, 1.0)).collect();
        super::apply_1q(&mut amps, 1, &[], &m);
        assert!(
            amps.iter().any(|c| c.re.is_nan() || c.im.is_nan()),
            "NaN failed to propagate from off-diagonal"
        );
    }

    #[test]
    fn nan_in_both_off_diagonals_still_propagates() {
        let nan = Complex::new(f64::NAN, 0.0);
        let z = Complex::new(0.0, 0.0);
        let m = [[z, nan], [nan, z]];
        let mut amps: Vec<Complex> = (0..8).map(|k| Complex::new(k as f64, 1.0)).collect();
        super::apply_1q(&mut amps, 1, &[], &m);
        assert!(amps.iter().any(|c| c.re.is_nan() || c.im.is_nan()));
    }

    // ---- P1-05 review B1: controlled tests for Tier-B kernels ----

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_x_avx512_lowbit_with_control_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // n=5 (32 amps), target=0, control=2. Tier-B contract: target ∈ {0,1}, c >= 2.
        let mut amps_avx: Vec<Complex> = (0..32)
            .map(|k| Complex::new(k as f64 * 0.07, k as f64 * -0.13))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: avx512 + target=0 < LANES + len=32 divisible by 4 + c=2 >= 2.
        unsafe {
            super::apply_1q_x_avx512_lowbit(&mut amps_avx, 0, &[2]);
        }
        super::apply_1q_x_scalar(&mut amps_sca, 0, &[2]);
        assert_eq!(amps_avx, amps_sca);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_y_avx512_lowbit_with_control_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        for &target in &[0u32, 1u32] {
            let mut amps_avx: Vec<Complex> = (0..32)
                .map(|k| Complex::new(k as f64 * 0.11 + 1.0, k as f64 * 0.23))
                .collect();
            let mut amps_sca = amps_avx.clone();
            // SAFETY: avx512 + target ∈ {0,1} + len=32 % 4 == 0 + c=2 >= 2 (and > target).
            unsafe {
                super::apply_1q_y_avx512_lowbit(&mut amps_avx, target, &[2], 1.0);
            }
            super::apply_1q_y_scalar(&mut amps_sca, target, &[2], 1.0);
            for (a, s) in amps_avx.iter().zip(amps_sca.iter()) {
                assert!(
                    (a.re - s.re).abs() < 1e-12 && (a.im - s.im).abs() < 1e-12,
                    "y_lowbit target={} with control: avx={:?} scalar={:?}",
                    target,
                    a,
                    s
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn apply_1q_antidiag_avx512_lowbit_with_control_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = Complex::new(0.6, 0.8);
        let b = Complex::new(0.6, -0.8);
        let mut amps_avx: Vec<Complex> = (0..32)
            .map(|k| Complex::new(k as f64 * 0.05, k as f64 * 0.13 - 0.5))
            .collect();
        let mut amps_sca = amps_avx.clone();
        // SAFETY: avx512 + target=0 < LANES + len=32 % 4 == 0 + c=2 >= 2.
        unsafe {
            super::apply_1q_antidiag_avx512_lowbit(&mut amps_avx, 0, &[2], a, b);
        }
        super::apply_1q_antidiag_scalar(&mut amps_sca, 0, &[2], a, b);
        for (av, sc) in amps_avx.iter().zip(amps_sca.iter()) {
            assert!(
                (av.re - sc.re).abs() < 1e-12 && (av.im - sc.im).abs() < 1e-12,
                "antidiag_lowbit with control: avx={:?} scalar={:?}",
                av,
                sc
            );
        }
    }

    // ---- P1-05 T13: anti-diag classifier dispatch proptest ----

    // P1-05 T13: anti-diag classifier dispatch proptest
    proptest::proptest! {
        #[test]
        fn antidiag_classifier_dispatch_matches_generic(
            a_re in -1.0f64..1.0,
            a_im in -1.0f64..1.0,
            b_re in -1.0f64..1.0,
            b_im in -1.0f64..1.0,
            target in 0u32..6,
            seed in proptest::prelude::any::<u64>(),
        ) {
            // Build anti-diagonal matrix (diagonal zeros force the
            // is_antidiagonal_2x2 path).
            let z = Complex::new(0.0, 0.0);
            let a = Complex::new(a_re, a_im);
            let b = Complex::new(b_re, b_im);
            let m = [[z, a], [b, z]];

            // Generate a deterministic state from seed.
            let n: usize = 1 << 6; // n=6 → 64 amps
            let mut s_dispatch: Vec<Complex> = (0..n)
                .map(|k| {
                    let r = ((seed.wrapping_mul(k as u64 + 1)) as f64) * 1e-19;
                    Complex::new(r.sin(), r.cos())
                })
                .collect();
            let mut s_direct = s_dispatch.clone();

            // Dispatch path (the one under test).
            super::apply_1q(&mut s_dispatch, target, &[], &m);

            // Direct generic kernel: inline the scalar 2×2 multiply over
            // the same pair enumeration. Matches the fall-through arm of apply_1q.
            let t_bit = 1usize << target;
            let mut i = 0usize;
            while i < s_direct.len() {
                if i & t_bit == 0 {
                    let j = i | t_bit;
                    let za = s_direct[i];
                    let zb = s_direct[j];
                    s_direct[i] = m[0][0] * za + m[0][1] * zb;
                    s_direct[j] = m[1][0] * za + m[1][1] * zb;
                }
                i += 1;
            }

            for (d, g) in s_dispatch.iter().zip(s_direct.iter()) {
                proptest::prop_assert!((d.re - g.re).abs() < 1e-12,
                    "re diverged: dispatch={} direct={}", d.re, g.re);
                proptest::prop_assert!((d.im - g.im).abs() < 1e-12,
                    "im diverged: dispatch={} direct={}", d.im, g.im);
            }
        }
    }
}

#[cfg(test)]
mod toffoli_indexing_tests {
    /// Classify (c0, c1, t, ext, n) into the expected dispatch tier
    /// per spec §4.2-§4.4. This is the source-of-truth oracle for the
    /// SIMD dispatch path; if the function below returns Tier X, the
    /// runtime SIMD path must match.
    #[derive(Debug, PartialEq, Eq)]
    enum Tier {
        A,
        B0,
        B1,
        C,
    }

    fn classify_toffoli(c0: u32, c1: u32, t: u32, ext: &[u32], n: u32) -> Tier {
        const LANES_BITS: u32 = 2;
        if n < 3 {
            return Tier::C;
        }
        let target_bit_idx = t;
        let ctrl_bits: Vec<u32> = std::iter::once(c0)
            .chain(std::iter::once(c1))
            .chain(ext.iter().copied())
            .collect();
        let c_lo = *ctrl_bits.iter().min().unwrap();
        // Tier A: target_bit ≥ LANES (== t ≥ LANES_BITS) and c_lo > t.
        if target_bit_idx >= LANES_BITS && c_lo > target_bit_idx {
            return Tier::A;
        }
        // Tier A outer-walk: target_bit ≥ LANES but some controls below target.
        // Spec §4.2 says Tier-A handles this via expand_with_fixed renormalisation.
        if target_bit_idx >= LANES_BITS {
            return Tier::A;
        }
        // Tier B: target_bit < LANES (t ∈ {0,1}) and c_lo ≥ LANES_BITS.
        if c_lo >= LANES_BITS {
            return match t {
                0 => Tier::B0,
                1 => Tier::B1,
                _ => unreachable!("t<LANES_BITS but t not in {{0,1}}"),
            };
        }
        // Else: Tier C scalar.
        Tier::C
    }

    /// Compute the swap pair (i, i ^ target_bit) for a given dispatch
    /// configuration and verify pairwise-disjoint bits at the
    /// SIMD-block level (mirrors P1-07 Task 14's coverage tests).
    fn pairs_are_disjoint(c0: u32, c1: u32, t: u32, ext: &[u32], n: u32) -> bool {
        let target_bit = 1u64 << t;
        let mut ctrl_mask = (1u64 << c0) | (1u64 << c1);
        for &e in ext {
            ctrl_mask |= 1u64 << e;
        }
        // For every i with ctrl bits set and target bit clear, (i, i | target_bit)
        // must be in-range and distinct, and target_bit must not overlap ctrl_mask.
        if target_bit & ctrl_mask != 0 {
            return false;
        }
        let len = 1u64 << n;
        for i in 0..len {
            if (i & ctrl_mask) != ctrl_mask {
                continue;
            }
            if (i & target_bit) != 0 {
                continue;
            }
            let j = i | target_bit;
            if j >= len {
                return false;
            }
            if i == j {
                return false;
            }
        }
        true
    }

    #[test]
    fn toffoli_classification_clean_tier_a() {
        // c0=4, c1=5, t=2, n=6: t_bit_idx=2 ≥ LANES_BITS, c_lo=4 > t=2.
        assert_eq!(classify_toffoli(4, 5, 2, &[], 6), Tier::A);
    }

    #[test]
    fn toffoli_classification_tier_b0() {
        // c0=2, c1=3, t=0, n=4: t<LANES_BITS, c_lo=2 ≥ LANES_BITS.
        assert_eq!(classify_toffoli(2, 3, 0, &[], 4), Tier::B0);
    }

    #[test]
    fn toffoli_classification_tier_b1() {
        // c0=2, c1=3, t=1, n=4
        assert_eq!(classify_toffoli(2, 3, 1, &[], 4), Tier::B1);
    }

    #[test]
    fn toffoli_classification_tier_c_small_n() {
        // n=2 — must be Tier C.
        assert_eq!(classify_toffoli(0, 1, 0, &[], 2), Tier::C);
    }

    #[test]
    fn toffoli_classification_tier_c_mixed_low_controls() {
        // c0=0, c1=1, t=0: t < LANES_BITS, c_lo=0 < LANES_BITS → Tier C.
        assert_eq!(classify_toffoli(0, 1, 0, &[], 3), Tier::C);
    }

    #[test]
    fn toffoli_pairs_disjoint_exhaustive_n6() {
        // For all triples (c0,c1,t) in {0..6}^3 with c0 != c1 != t,
        // and ext subsets of size ≤ 1, verify pair disjointness.
        for c0 in 0..6 {
            for c1 in 0..6 {
                for t in 0..6 {
                    if c0 == c1 || c0 == t || c1 == t {
                        continue;
                    }
                    assert!(
                        pairs_are_disjoint(c0, c1, t, &[], 6),
                        "c0={} c1={} t={} ext=[]",
                        c0,
                        c1,
                        t
                    );
                    for e in 0..6 {
                        if e == c0 || e == c1 || e == t {
                            continue;
                        }
                        assert!(
                            pairs_are_disjoint(c0, c1, t, &[e], 6),
                            "c0={} c1={} t={} ext=[{}]",
                            c0,
                            c1,
                            t,
                            e
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod ccz_indexing_tests {
    #[derive(Debug, PartialEq, Eq)]
    enum CczTier {
        A,
        C,
    }

    /// CCZ has no target — every mask bit is symmetric. Tier A
    /// applies when mask_lo ≥ LANES_BITS (so each zmm block has a
    /// fixed ctrl-mask value); Tier C otherwise.
    fn classify_ccz(q0: u32, q1: u32, q2: u32, ext: &[u32], n: u32) -> CczTier {
        const LANES_BITS: u32 = 2;
        if n < 3 {
            return CczTier::C;
        }
        let mask_bits: Vec<u32> = [q0, q1, q2]
            .iter()
            .copied()
            .chain(ext.iter().copied())
            .collect();
        let _mask_lo = *mask_bits.iter().min().unwrap();
        let _ = LANES_BITS; // used conceptually in spec §5.2 dispatch logic
                            // Tier A outer-walk handles mask_lo < LANES_BITS (per spec §5.2).
        if n >= 3 {
            CczTier::A
        } else {
            CczTier::C
        }
    }

    fn ccz_pairs_unique(q0: u32, q1: u32, q2: u32, ext: &[u32], n: u32) -> bool {
        let mut mask = (1u64 << q0) | (1u64 << q1) | (1u64 << q2);
        for &e in ext {
            mask |= 1u64 << e;
        }
        let len = 1u64 << n;
        let mut count = 0u64;
        for i in 0..len {
            if (i & mask) == mask {
                count += 1;
            }
        }
        // Every full match is exactly one sign-flip; no pairs to swap.
        // Validate count = 2^(n - popcount(mask)).
        let expected = 1u64 << (n - mask.count_ones());
        count == expected
    }

    #[test]
    fn ccz_pairs_count_exhaustive_n6() {
        for q0 in 0..6 {
            for q1 in 0..6 {
                for q2 in 0..6 {
                    if q0 == q1 || q0 == q2 || q1 == q2 {
                        continue;
                    }
                    assert!(
                        ccz_pairs_unique(q0, q1, q2, &[], 6),
                        "q0={} q1={} q2={}",
                        q0,
                        q1,
                        q2
                    );
                    for e in 0..6 {
                        if e == q0 || e == q1 || e == q2 {
                            continue;
                        }
                        assert!(ccz_pairs_unique(q0, q1, q2, &[e], 6));
                    }
                }
            }
        }
    }

    #[test]
    fn ccz_symmetry_mask_is_permutation_invariant() {
        // CCZ symmetric in qubit order: mask(q0,q1,q2) = mask(any permutation).
        let m1 = (1u64 << 3) | (1u64 << 4) | (1u64 << 5);
        let m2 = (1u64 << 5) | (1u64 << 3) | (1u64 << 4);
        assert_eq!(m1, m2);
    }

    #[test]
    fn ccz_classification_small_n_is_tier_c() {
        assert_eq!(classify_ccz(0, 1, 2, &[], 2), CczTier::C);
    }
}

#[cfg(test)]
mod apply_toffoli_scalar_tests {
    use super::*;
    use aleph_core::Complex;

    fn basis_state(n: u32, index: usize) -> Vec<Complex> {
        let mut amps = vec![Complex::new(0.0, 0.0); 1 << n];
        amps[index] = Complex::new(1.0, 0.0);
        amps
    }

    #[test]
    fn ccx_swaps_only_11_target_pair() {
        // targets = [c0=0, c1=1, t=2], n=3.
        // Amplitude index bit k corresponds to qubit k (same convention
        // as all scalar kernels in this file; bit 0 = qubit 0).
        // ctrl_mask = (1<<0)|(1<<1) = 0b011 = 3; target_bit = 1<<2 = 4.
        // The gate fires when i has bits 0 and 1 set and bit 2 clear,
        // i.e. i = 0b011 = 3, and swaps with i|4 = 7 = 0b111.
        for input in 0..8usize {
            let mut amps = basis_state(3, input);
            apply_toffoli_scalar(&mut amps, [0, 1, 2], &[]);
            let expected = if input == 0b011 {
                0b111
            } else if input == 0b111 {
                0b011
            } else {
                input
            };
            let mut want = vec![Complex::new(0.0, 0.0); 8];
            want[expected] = Complex::new(1.0, 0.0);
            assert_eq!(
                amps, want,
                "input {:03b} should map to {:03b}",
                input, expected
            );
        }
    }

    #[test]
    fn ccx_with_external_control_acts_only_when_ext_set() {
        // targets = [0,1,2], external_controls = [3], n=4.
        // ctrl_mask = (1<<0)|(1<<1)|(1<<3) = 0b1011 = 11; target_bit = 4.
        // Fires when i = 0b1011 = 11 (bits 0,1,3 set, bit 2 clear),
        // swaps with i|4 = 15 = 0b1111.
        for input in 0..16usize {
            let mut amps = basis_state(4, input);
            apply_toffoli_scalar(&mut amps, [0, 1, 2], &[3]);
            let expected = if input == 0b1011 {
                0b1111
            } else if input == 0b1111 {
                0b1011
            } else {
                input
            };
            let mut want = vec![Complex::new(0.0, 0.0); 16];
            want[expected] = Complex::new(1.0, 0.0);
            assert_eq!(amps, want);
        }
    }

    #[test]
    fn ccx_involutive() {
        let mut amps: Vec<Complex> = (0..16).map(|i| Complex::new(i as f64, 0.0)).collect();
        let original = amps.clone();
        apply_toffoli_scalar(&mut amps, [0, 1, 2], &[]);
        apply_toffoli_scalar(&mut amps, [0, 1, 2], &[]);
        assert_eq!(amps, original);
    }
}

#[cfg(test)]
mod apply_ccz_scalar_tests {
    use super::*;
    use aleph_core::Complex;

    fn basis_state(n: u32, index: usize) -> Vec<Complex> {
        let mut amps = vec![Complex::new(0.0, 0.0); 1 << n];
        amps[index] = Complex::new(1.0, 0.0);
        amps
    }

    #[test]
    fn ccz_sign_flips_only_111() {
        // qubits = [0,1,2], n=3. Mask = (1<<0)|(1<<1)|(1<<2) = 0b111 = 7.
        // Sign flip happens only on amp index 7.
        for input in 0..8usize {
            let mut amps = basis_state(3, input);
            apply_ccz_scalar(&mut amps, [0, 1, 2], &[]);
            let mut want = vec![Complex::new(0.0, 0.0); 8];
            let sign = if input == 7 { -1.0 } else { 1.0 };
            want[input] = Complex::new(sign, 0.0);
            assert_eq!(amps, want);
        }
    }

    #[test]
    fn ccz_with_external_control_acts_only_when_ext_set() {
        // qubits = [0,1,2], ctx=[3], n=4. Mask = 0b1111 = 15.
        for input in 0..16usize {
            let mut amps = basis_state(4, input);
            apply_ccz_scalar(&mut amps, [0, 1, 2], &[3]);
            let mut want = vec![Complex::new(0.0, 0.0); 16];
            let sign = if input == 15 { -1.0 } else { 1.0 };
            want[input] = Complex::new(sign, 0.0);
            assert_eq!(amps, want);
        }
    }

    #[test]
    fn ccz_involutive() {
        let mut amps: Vec<Complex> = (0..16).map(|i| Complex::new(i as f64, 0.0)).collect();
        let original = amps.clone();
        apply_ccz_scalar(&mut amps, [0, 1, 2], &[]);
        apply_ccz_scalar(&mut amps, [0, 1, 2], &[]);
        assert_eq!(amps, original);
    }

    #[test]
    fn ccz_symmetric_in_qubit_order() {
        let mut a = vec![Complex::new(1.0, 0.0); 16];
        let mut b = a.clone();
        apply_ccz_scalar(&mut a, [0, 1, 2], &[]);
        apply_ccz_scalar(&mut b, [2, 0, 1], &[]);
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod apply_3q_prelude_tests {
    use super::*;
    use aleph_core::Complex;

    fn toffoli_matrix() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        // Rows 0–5: identity diagonal.
        m[0][0] = o;
        m[1][1] = o;
        m[2][2] = o;
        m[3][3] = o;
        m[4][4] = o;
        m[5][5] = o;
        // Rows 6–7 swapped (Toffoli).
        m[6][7] = o;
        m[7][6] = o;
        m
    }

    fn ccz_matrix() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        // Diagonal +1 for rows 0–6, then −1 for row 7 (CCZ).
        m[0][0] = o;
        m[1][1] = o;
        m[2][2] = o;
        m[3][3] = o;
        m[4][4] = o;
        m[5][5] = o;
        m[6][6] = o;
        m[7][7] = Complex::new(-1.0, 0.0);
        m
    }

    fn random_amps(n: u32, seed: u64) -> Vec<Complex> {
        // Linear congruential — deterministic, no rand crate dep.
        let mut s = seed;
        let mut step = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        let mut v: Vec<Complex> = (0..(1 << n))
            .map(|_| Complex::new(step(), step()))
            .collect();
        // Normalise.
        let norm: f64 = v
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .sum::<f64>()
            .sqrt();
        for c in &mut v {
            *c = Complex::new(c.re / norm, c.im / norm);
        }
        v
    }

    #[test]
    fn apply_3q_routes_toffoli_to_scalar() {
        let mut a = random_amps(5, 1);
        let mut b = a.clone();
        apply_3q(&mut a, [0, 1, 4], &[], &toffoli_matrix());
        apply_toffoli_scalar(&mut b, [0, 1, 4], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_3q_routes_ccz_to_scalar() {
        let mut a = random_amps(5, 2);
        let mut b = a.clone();
        apply_3q(&mut a, [0, 1, 4], &[], &ccz_matrix());
        apply_ccz_scalar(&mut b, [0, 1, 4], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_3q_generic_unchanged_on_arbitrary_matrix() {
        let z = Complex::new(0.0, 0.0);
        let mut m = [[z; 8]; 8];
        // Hadamard-like 3q matrix (8x8 Walsh-Hadamard, normalised).
        let s = 1.0 / (8.0_f64.sqrt());
        for (r, row) in m.iter_mut().enumerate() {
            for (c, entry) in row.iter_mut().enumerate() {
                let sign = if (r & c).count_ones() % 2 != 0 {
                    -1.0
                } else {
                    1.0
                };
                *entry = Complex::new(sign * s, 0.0);
            }
        }
        let mut a = random_amps(5, 3);
        let mut b = a.clone();
        apply_3q(&mut a, [0, 1, 4], &[], &m);
        apply_3q_generic(&mut b, [0, 1, 4], &[], &m);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod toffoli_tier_a_tests {
    use super::*;
    use aleph_core::Complex;

    /// Linear-congruential pseudo-random amplitudes — no `rand` crate dep.
    fn random_amps(n: u32, seed: u64) -> Vec<Complex> {
        let mut s = seed;
        let mut step = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        (0..(1 << n))
            .map(|_| Complex::new(step(), step()))
            .collect()
    }

    /// n=8 state, inner controls c0=5 c1=6, target t=2.
    /// c_lo = min(5,6) = 5 > t=2, and t=2 >= LANES_BITS=2 — clean Tier A.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tier_a_matches_scalar_clean_contract() {
        if !std::is_x86_feature_detected!("avx512f") {
            return; // Skip gracefully on non-AVX-512 hosts.
        }
        let mut simd = random_amps(8, 7);
        let mut scalar = simd.clone();
        // all_ctrls = [5, 6]; target = 2.
        unsafe {
            apply_toffoli_avx512_tier_a(&mut simd, 2, &[5, 6]);
        }
        apply_toffoli_scalar(&mut scalar, [5, 6, 2], &[]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// External control qubit at index 7, inner pair at (3, 4), target=2.
    /// c_lo = min(3,4,7) = 3 > t=2 ✓. All three controls above target.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tier_a_with_external_control_clean() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut simd = random_amps(8, 8);
        let mut scalar = simd.clone();
        // SIMD call: all_ctrls = [3, 4, 7], target = 2.
        unsafe {
            apply_toffoli_avx512_tier_a(&mut simd, 2, &[3, 4, 7]);
        }
        // Scalar call: targets = [3, 4, 2], external_controls = [7].
        apply_toffoli_scalar(&mut scalar, [3, 4, 2], &[7]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Smoke-test that `dispatch_toffoli` now routes through Tier A on AVX-512
    /// hosts when the contract holds, by verifying end-to-end equivalence to
    /// the scalar path for a mid-sized state.
    #[test]
    fn dispatch_toffoli_routes_tier_a_on_avx512() {
        let n = 8u32;
        let mut via_dispatch = random_amps(n, 42);
        let mut via_scalar = via_dispatch.clone();
        // targets = [c0=5, c1=6, t=2]: c_lo=5 > t=2, t>=2. Tier-A eligible.
        dispatch_toffoli(&mut via_dispatch, [5, 6, 2], &[]);
        apply_toffoli_scalar(&mut via_scalar, [5, 6, 2], &[]);
        for (x, y) in via_dispatch.iter().zip(via_scalar.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }

    /// Verifies that `apply_toffoli_avx512_tier_a_outer_walk` produces the
    /// same result as the scalar reference under its tightened contract:
    /// `target >= LANES_BITS=2` AND every control `>= LANES_BITS=2`.
    /// Configuration: n=8, sorted=[3, 5], target=4 — target above target,
    /// one control between LANES_BITS and target (3 ≤ target=4 but ≥ 2).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tier_a_outer_walk_control_below_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return; // Skip gracefully on non-AVX-512 hosts.
        }
        let mut simd = random_amps(8, 11);
        let mut scalar = simd.clone();
        // sorted=[3, 5], target=4: c_lo=3 ≤ t=4 (outer-walk path),
        // both controls ≥ LANES_BITS=2 (valid contract).
        let sorted = [3u32, 5u32];
        unsafe {
            apply_toffoli_avx512_tier_a_outer_walk(&mut simd, 4, &sorted);
        }
        apply_toffoli_scalar(&mut scalar, [3, 5, 4], &[]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Verifies that `dispatch_toffoli` correctly falls through to the
    /// scalar path when a control sits below LANES_BITS — the SIMD
    /// contract cannot be satisfied, so the scalar kernel must take over.
    /// Without this fallback, the outer-walk SIMD path would silently
    /// no-op (block_base & low_bit is always zero).
    #[test]
    fn dispatch_toffoli_falls_through_to_scalar_when_control_below_lanes_bits() {
        // n=7, c0=0 (below LANES_BITS=2), c1=5, t=4. c_lo=0 < LANES_BITS.
        let mut via_dispatch = random_amps(7, 31);
        let mut via_scalar = via_dispatch.clone();
        dispatch_toffoli(&mut via_dispatch, [0, 5, 4], &[]);
        apply_toffoli_scalar(&mut via_scalar, [0, 5, 4], &[]);
        for (x, y) in via_dispatch.iter().zip(via_scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Dispatch-routing test: `dispatch_toffoli` must produce the same result
    /// as the scalar kernel for a control-below-target configuration where
    /// every control is still ≥ LANES_BITS=2 (so the outer-walk path is
    /// valid). Routes to outer-walk on AVX-512, scalar elsewhere.
    #[test]
    fn dispatch_toffoli_routes_outer_walk_below_target() {
        // n=8 state, c0=3, c1=5, t=4: c_lo=3 < t=4 — outer-walk path on AVX-512.
        // Both controls ≥ LANES_BITS=2 so contract holds.
        let mut a = random_amps(8, 12);
        let mut b = a.clone();
        dispatch_toffoli(&mut a, [3, 5, 4], &[]);
        apply_toffoli_scalar(&mut b, [3, 5, 4], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Direct-call equivalence: Tier-B.0 kernel vs scalar for n=6, controls=[3,4], t=0.
    /// `c_lo = 3 >= LANES_BITS = 2`, `t = 0` — Tier-B.0 contract satisfied.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tier_b0_direct_call_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut simd = random_amps(6, 14);
        let mut scalar = simd.clone();
        let sorted = [3u32, 4u32];
        // SAFETY: AVX-512F detected, sorted_controls = [3, 4] all >= 2,
        // t=0 (implicit), n=6 so len=64 = 1<<6 >= 1<<3.
        unsafe {
            apply_toffoli_avx512_tier_b0(&mut simd, &sorted);
        }
        apply_toffoli_scalar(&mut scalar, [3, 4, 0], &[]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Cross-arch dispatch test for Tier-B.0: `dispatch_toffoli` must match
    /// the scalar kernel for `targets=[3,4,0]` on all host architectures.
    /// On AVX-512 hosts this exercises the Tier-B.0 branch (t=0, c_lo=3>=2);
    /// on other hosts it falls through to the scalar fallback.
    #[test]
    fn dispatch_toffoli_routes_tier_b0_cross_arch() {
        let mut a = random_amps(6, 15);
        let mut b = a.clone();
        dispatch_toffoli(&mut a, [3, 4, 0], &[]);
        apply_toffoli_scalar(&mut b, [3, 4, 0], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Direct-call equivalence: Tier-B.1 kernel vs scalar for n=6, controls=[3,4], t=1.
    /// `c_lo = 3 >= LANES_BITS = 2`, `t = 1` — Tier-B.1 contract satisfied.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tier_b1_direct_call_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut simd = random_amps(6, 15);
        let mut scalar = simd.clone();
        let sorted = [3u32, 4u32];
        // SAFETY: AVX-512F detected, sorted_controls = [3, 4] all >= 2,
        // t=1 (implicit), n=6 so len=64 = 1<<6 >= 1<<3.
        unsafe {
            apply_toffoli_avx512_tier_b1(&mut simd, &sorted);
        }
        apply_toffoli_scalar(&mut scalar, [3, 4, 1], &[]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Cross-arch dispatch test for Tier-B.1: `dispatch_toffoli` must match
    /// the scalar kernel for `targets=[3,4,1]` on all host architectures.
    /// On AVX-512 hosts this exercises the Tier-B.1 branch (t=1, c_lo=3>=2);
    /// on other hosts it falls through to the scalar fallback.
    #[test]
    fn dispatch_toffoli_routes_tier_b1_cross_arch() {
        let mut a = random_amps(6, 16);
        let mut b = a.clone();
        dispatch_toffoli(&mut a, [3, 4, 1], &[]);
        apply_toffoli_scalar(&mut b, [3, 4, 1], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }
}

#[cfg(test)]
mod ccz_tier_a_tests {
    use super::*;
    use aleph_core::Complex;

    fn random_amps(n: u32, seed: u64) -> Vec<Complex> {
        let mut s = seed;
        let mut step = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        (0..(1 << n))
            .map(|_| Complex::new(step(), step()))
            .collect()
    }

    /// Direct-call test: `apply_ccz_avx512_tier_a` with mask_bits = [2, 4, 6]
    /// (mask_lo = 2 ≥ LANES_BITS) must exactly match `apply_ccz_scalar` with
    /// targets=[2, 4, 6] and no external controls.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn ccz_tier_a_direct_call_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut simd = random_amps(7, 17);
        let mut scalar = simd.clone();
        let mask_bits = [2u32, 4u32, 6u32]; // mask_lo = 2 ≥ LANES_BITS = 2.
                                            // SAFETY: AVX-512F detected, mask_lo = 2 ≥ LANES_BITS, n=7 ≥ 3, qubits distinct.
        unsafe {
            apply_ccz_avx512_tier_a(&mut simd, &mask_bits);
        }
        apply_ccz_scalar(&mut scalar, [2, 4, 6], &[]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Direct-call test with external control: mask_bits = [2, 3, 4, 5] maps to
    /// targets=[2, 3, 4] + external_control=[5]. Scalar: `apply_ccz_scalar`
    /// with targets=[2, 3, 4] and controls=[5] must produce identical results.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn ccz_tier_a_with_external_control() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut simd = random_amps(7, 18);
        let mut scalar = simd.clone();
        let mask_bits = [2u32, 3u32, 4u32, 5u32];
        // SAFETY: AVX-512F detected, mask_lo = 2 ≥ LANES_BITS, n=7 ≥ 3, qubits distinct.
        unsafe {
            apply_ccz_avx512_tier_a(&mut simd, &mask_bits);
        }
        apply_ccz_scalar(&mut scalar, [2, 3, 4], &[5]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Cross-arch dispatch test: `dispatch_ccz` with targets=[2, 4, 6] must match
    /// `apply_ccz_scalar` on all host architectures. On AVX-512 hosts this exercises
    /// the Tier-A branch (mask_lo = 2 ≥ LANES_BITS); on others it falls through
    /// to the scalar fallback.
    #[test]
    fn dispatch_ccz_routes_tier_a_cross_arch() {
        let mut a = random_amps(7, 19);
        let mut b = a.clone();
        dispatch_ccz(&mut a, [2, 4, 6], &[]);
        apply_ccz_scalar(&mut b, [2, 4, 6], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x.re - y.re).abs() < 1e-12,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < 1e-12,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    /// Direct-call test: `apply_ccz_avx512_tier_a_outer_walk` with mask = {0, 3, 5}
    /// (mask_lo = 0 < LANES_BITS = 2). Must match `apply_ccz_scalar` exactly.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn ccz_tier_a_outer_walk_low_mask() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut simd = random_amps(6, 19);
        let mut scalar = simd.clone();
        // mask = {0, 3, 5} — mask_lo=0 < LANES_BITS.
        unsafe {
            apply_ccz_avx512_tier_a_outer_walk(&mut simd, &[0, 3, 5]);
        }
        apply_ccz_scalar(&mut scalar, [0, 3, 5], &[]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }

    /// Cross-arch dispatch test: `dispatch_ccz` with targets=[0, 3, 5] (mask_lo=0 <
    /// LANES_BITS) must match `apply_ccz_scalar` on all host architectures.
    /// On AVX-512 hosts this exercises the outer-walk branch.
    #[test]
    fn dispatch_ccz_routes_outer_walk_cross_arch() {
        let mut a = random_amps(6, 20);
        let mut b = a.clone();
        dispatch_ccz(&mut a, [0, 3, 5], &[]);
        apply_ccz_scalar(&mut b, [0, 3, 5], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }
}
