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
pub(crate) fn apply_1q(amps: &mut [Complex], target: u32, controls: &[u32], m: &[[Complex; 2]; 2]) {
    // Diagonal fast path (P1-06).  Detection cost is ~5 ns per call;
    // negligible vs even the cheapest state-vector kernel.  Catches
    // Z/S/T/Sdg/Tdg/Rz/Phase intrinsic gates AND any user-supplied
    // diagonal GenericUnitary(M2x2).
    if super::is_diagonal_2x2(m) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
                && (1usize << target) >= 4
                && controls.iter().all(|&c| c > target)
            {
                // SAFETY: identical contract to apply_1q_avx512 — feature gate +
                // target_bit ≥ LANES + every control above target.
                unsafe {
                    apply_1q_diagonal_avx512(amps, target, controls, m[0][0], m[1][1]);
                }
                return;
            }
        }
        apply_1q_diagonal_scalar(amps, target, controls, m[0][0], m[1][1]);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 packed-complex kernel: 1 `vmovupd zmm` reads 4
        // complex pairs (1 cache-line per side, 2 streams total),
        // versus the SoA SIMD attempt's 4 separate streams (see
        // ADR 0008). Engages when target_bit ≥ LANES (=4) so the
        // inner loop's contiguous unit-stride load is safe, AND
        // every control sits above target so the inner walk
        // doesn't toggle a control bit.
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
unsafe fn apply_1q_avx512(
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

    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    let outer_iter = |block: usize| {
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
        let mut block = 0usize;
        while block < len {
            outer_iter(block);
            block += outer_step;
        }
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
    for k in 0..outer_count {
        let block = crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1);
        outer_iter(block);
    }
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

    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    let outer_iter = |block: usize| {
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
        let mut block = 0usize;
        while block < len {
            outer_iter(block);
            block += outer_step;
        }
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
    for k in 0..outer_count {
        let block = crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1);
        outer_iter(block);
    }
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
            apply_2q_cnot_scalar(amps, targets[0], targets[1], controls);
            return;
        }
        Some(super::Perm2qKind::CnotLo) => {
            apply_2q_cnot_scalar(amps, targets[1], targets[0], controls);
            return;
        }
        Some(super::Perm2qKind::Swap) => {
            apply_2q_swap_scalar(amps, targets, controls);
            return;
        }
        None => {}
    }

    // 2. Diagonal-4x4 (catches Cz, controlled-Phase, Rzz, user diagonals).
    if super::is_diagonal_4x4(m) {
        let d = [m[0][0], m[1][1], m[2][2], m[3][3]];
        if super::is_cz_signature(d) {
            apply_2q_cz_scalar(amps, targets, controls);
        } else {
            apply_2q_diagonal_scalar(amps, targets, controls, d);
        }
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

    let amps_ptr = amps.as_mut_ptr() as *mut f64;

    let outer_iter = |block: usize| {
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
    for k in 0..outer_count {
        let block = crate::kernels::expand_with_fixed(k, &fixed_above) << (t_lo + 1);
        outer_iter(block);
    }
}

/// Apply a 3-qubit matrix to `targets = [t0, t1, t2]` (with external
/// `controls`) in place.
///
/// **MSB convention (P0-06):** matrix index `k`'s bits map to targets
/// from MSB to LSB — bit 2 of `k` is `targets[0]`, bit 1 is
/// `targets[1]`, bit 0 is `targets[2]`. So `k = 6` (binary `110`)
/// corresponds to `(targets[0] = 1, targets[1] = 1, targets[2] = 0)`.
/// This matches `Gate::Toffoli` (`qubits = [c0, c1, target]`), whose
/// matrix swaps rows 6 ↔ 7.
pub(crate) fn apply_3q(
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
        for r in 0..4 {
            for c in 0..4 {
                m[r][c] = Complex::new(lcg(), lcg());
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
        for r in 0..4 {
            for c in 0..4 {
                m[r][c] = Complex::new(lcg(), lcg());
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
}
