//! Single-precision (`Complex<f32>`) AoS gate kernels (P2-08).
//!
//! Scalar f32 kernels cover every gate type (correctness on any circuit
//! and on non-AVX-512 hosts); f32 AVX-512 kernels accelerate the fused
//! hot-types the optimized pipeline emits. Mirrors `kernels::aos` per the
//! f64→f32 substitution rules in the P2-08 plan; the FP64 path is untouched.

#![allow(dead_code)]

use aleph_core::Complex;

use crate::kernels::tuning::{self, GateClass};
#[cfg(target_arch = "x86_64")]
use crate::kernels::BlockPtrF32;
use crate::kernels::{control_mask, is_diagonal_2x2_f32, par_blocks, ComplexF32Ptr};

/// Generic 2×2 application on an f32 AoS slice. Correct for any 2×2
/// matrix (diagonal, anti-diagonal, dense); the diagonal fast path in
/// `apply_1q_f32` (Task 4) routes diagonal matrices away from here for
/// speed.
///
/// Mirrors `aos::apply_1q`'s scalar generic path (lines 192–223) with
/// `Complex<f32>` substituted for `Complex<f64>` and `ComplexF32Ptr`
/// substituted for `ComplexPtr`. Index algebra and SAFETY reasoning are
/// identical.
// `pub` (like `aos::apply_1q_avx512`) so the `internal-bench`-gated
// integration test `tests/fp32_simd_scalar.rs` can call the forced scalar
// reference directly. Without that feature `kernels` is private, so the
// effective visibility is `pub(crate)`.
pub fn apply_1q_dense_scalar_f32(
    amps: &mut [Complex<f32>],
    target: u32,
    controls: &[u32],
    m: &[[Complex<f32>; 2]; 2],
) {
    let t_bit = 1usize << target;
    let ctrl_mask = control_mask(controls);
    let len = amps.len();
    let cp = ComplexF32Ptr(amps.as_mut_ptr());
    let policy = tuning::resolve_policy(
        GateClass::OneQGeneric,
        tuning::pos_class(target, len.trailing_zeros()),
    );
    // Flat per-amplitude walk: parallel over the full index range.
    // Each base index `i` with bit `target` clear writes its own
    // disjoint pair `{i, i|t_bit}` — no count-starvation.
    par_blocks(
        policy,
        len,
        len,
        |k| k,
        |i| {
            if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
                let p = cp.ptr();
                let j = i | t_bit;
                // SAFETY: `i < len`, `j = i | t_bit < len`; distinct base
                // indices (bit `target` clear) produce disjoint {i, j} pairs,
                // so concurrent writes never alias.
                unsafe {
                    let a = *p.add(i);
                    let b = *p.add(j);
                    *p.add(i) = m[0][0] * a + m[0][1] * b;
                    *p.add(j) = m[1][0] * a + m[1][1] * b;
                }
            }
        },
    );
}

/// Diagonal 2×2 fast path: only the two diagonal entries matter, each
/// amplitude scaled in place (no pairing). Mirrors
/// `aos::apply_1q_diagonal_scalar`.
// `pub` (like `apply_1q_dense_scalar_f32`) so the `internal-bench`-gated
// integration test `tests/fp32_simd_scalar.rs` can call the forced scalar
// reference directly. Without that feature `kernels` is private, so the
// effective visibility is `pub(crate)`.
pub fn apply_1q_diag_scalar_f32(
    amps: &mut [Complex<f32>],
    target: u32,
    controls: &[u32],
    d0: Complex<f32>,
    d1: Complex<f32>,
) {
    let t_bit = 1usize << target;
    let ctrl_mask = control_mask(controls);
    let len = amps.len();
    let cp = ComplexF32Ptr(amps.as_mut_ptr());
    let policy = tuning::resolve_policy(
        GateClass::OneQDiag,
        tuning::pos_class(target, len.trailing_zeros()),
    );
    par_blocks(
        policy,
        len,
        len,
        |k| k,
        |i| {
            if (i & ctrl_mask) == ctrl_mask {
                let p = cp.ptr();
                let d = if i & t_bit == 0 { d0 } else { d1 };
                // SAFETY: i < len; each index written once, no aliasing.
                unsafe {
                    *p.add(i) = d * *p.add(i);
                }
            }
        },
    );
}

/// Scalar fallback for f32 2-qubit gate application.
///
/// Mirrors `aos::apply_2q_dense_scalar` with `Complex<f32>` substituted
/// for `Complex<f64>` and `ComplexF32Ptr` substituted for `ComplexPtr`.
/// Index algebra, MSB convention, SAFETY reasoning, and the
/// `par_blocks`/`par_units` call structure are identical.
///
/// **MSB convention (P0-06):** `targets[0]` is the *high* bit of the
/// matrix index `k`, `targets[1]` is the *low* bit. So matrix row 2
/// (binary `10`) corresponds to `(targets[0] = 1, targets[1] = 0)`.
/// This matches `Gate::Cnot` (`qubits = [control, target]`), whose
/// matrix swaps rows 2 ↔ 3.
///
/// Targets must be distinct; the caller (`apply_gate`) enforces this.
///
/// `pub` (like the 1q scalar references) so the `internal-bench`-gated
/// integration test `tests/fp32_simd_scalar.rs` can call this forced
/// scalar reference directly. Without that feature `kernels` is private,
/// so the effective visibility is `pub(crate)`.
pub fn apply_2q_dense_scalar_f32(
    amps: &mut [Complex<f32>],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex<f32>; 4]; 4],
) {
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let t_mask = t0_bit | t1_bit;
    let ctrl_mask = control_mask(controls);
    let len = amps.len();
    let cp = ComplexF32Ptr(amps.as_mut_ptr());
    let policy = tuning::resolve_policy(
        GateClass::TwoQDense,
        tuning::pos_class(targets[0].max(targets[1]), len.trailing_zeros()),
    );
    // Flat per-amplitude walk: each base index `i` (both target bits clear)
    // writes the disjoint quartet {i, i|t1_bit, i|t0_bit, i|t_mask}, so
    // concurrent tasks never alias.
    par_blocks(
        policy,
        len,
        len,
        |k| k,
        |i| {
            if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
                // MSB convention: matrix index k bit 1 → targets[0], bit 0 → targets[1].
                // So idx[k] sets t0_bit iff (k & 2) != 0, t1_bit iff (k & 1) != 0.
                let idx = [
                    i,          // k = 00
                    i | t1_bit, // k = 01
                    i | t0_bit, // k = 10
                    i | t_mask, // k = 11
                ];
                // SAFETY: all idx[] < len (base i has both target bits clear, OR-ing
                // in target bits keeps the result < len); distinct base indices produce
                // disjoint quartets — no aliasing across tasks. Read all 4 before writing.
                unsafe {
                    let p = cp.ptr();
                    let v0 = *p.add(idx[0]);
                    let v1 = *p.add(idx[1]);
                    let v2 = *p.add(idx[2]);
                    let v3 = *p.add(idx[3]);
                    *p.add(idx[0]) = m[0][0] * v0 + m[0][1] * v1 + m[0][2] * v2 + m[0][3] * v3;
                    *p.add(idx[1]) = m[1][0] * v0 + m[1][1] * v1 + m[1][2] * v2 + m[1][3] * v3;
                    *p.add(idx[2]) = m[2][0] * v0 + m[2][1] * v1 + m[2][2] * v2 + m[2][3] * v3;
                    *p.add(idx[3]) = m[3][0] * v0 + m[3][1] * v1 + m[3][2] * v2 + m[3][3] * v3;
                }
            }
        },
    );
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
///
/// Mirrors `aos::apply_3q_generic` with `Complex<f32>` substituted for
/// `Complex<f64>` and `ComplexF32Ptr` substituted for `ComplexPtr`.
/// Index algebra, SAFETY reasoning, and `par_blocks` call structure are
/// identical.
pub(crate) fn apply_3q_generic_f32(
    amps: &mut [Complex<f32>],
    targets: [u32; 3],
    controls: &[u32],
    m: &[[Complex<f32>; 8]; 8],
) {
    let t_bits = [
        1usize << targets[0],
        1usize << targets[1],
        1usize << targets[2],
    ];
    let t_mask = t_bits[0] | t_bits[1] | t_bits[2];
    let ctrl_mask = control_mask(controls);
    let len = amps.len();
    let cp = ComplexF32Ptr(amps.as_mut_ptr());
    let max_target = targets[0].max(targets[1]).max(targets[2]);
    let policy = tuning::resolve_policy(
        GateClass::ThreeQ,
        tuning::pos_class(max_target, len.trailing_zeros()),
    );
    // Flat per-amplitude walk: each base index `i` (all target bits clear) writes
    // the disjoint octet of 8 amplitudes indexed by idx[0..8]; concurrent tasks
    // never alias because distinct base indices produce disjoint octets.
    par_blocks(
        policy,
        len,
        len,
        |k| k,
        |i| {
            if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
                let mut idx = [0usize; 8];
                for (k, slot) in idx.iter_mut().enumerate() {
                    // MSB convention: k bit 2 → targets[0], bit 1 → targets[1], bit 0 → targets[2].
                    let bit_t0 = if k & 4 != 0 { t_bits[0] } else { 0 };
                    let bit_t1 = if k & 2 != 0 { t_bits[1] } else { 0 };
                    let bit_t2 = if k & 1 != 0 { t_bits[2] } else { 0 };
                    *slot = i | bit_t0 | bit_t1 | bit_t2;
                }
                // SAFETY: all idx[] < len (base `i` has all target bits clear; OR-ing
                // in target bits keeps each result < len); distinct base indices produce
                // disjoint octets — no aliasing across tasks. Read all 8 before writing.
                unsafe {
                    let p = cp.ptr();
                    let v0 = *p.add(idx[0]);
                    let v1 = *p.add(idx[1]);
                    let v2 = *p.add(idx[2]);
                    let v3 = *p.add(idx[3]);
                    let v4 = *p.add(idx[4]);
                    let v5 = *p.add(idx[5]);
                    let v6 = *p.add(idx[6]);
                    let v7 = *p.add(idx[7]);
                    let v = [v0, v1, v2, v3, v4, v5, v6, v7];
                    for r in 0..8 {
                        let mut acc = Complex::<f32>::new(0.0, 0.0);
                        for c in 0..8 {
                            acc += m[r][c] * v[c];
                        }
                        *p.add(idx[r]) = acc;
                    }
                }
            }
        },
    );
}

/// Dense k-qubit matvec on an f32 AoS slice (scalar path).
///
/// Mirrors `unitary_kq::apply_kq_scalar_aos` with `Complex<f32>` substituted
/// for `Complex<f64>` and `ComplexF32Ptr` substituted for `ComplexPtr`. The
/// index algebra (`targets_offsets_fixed`, `expand_with_fixed`, `base | offsets[m]`),
/// SAFETY reasoning, and `par_blocks(DEFAULT_POLICY, …)` call structure are
/// identical to the f64 source. The `vec!`-per-block allocation in the inner
/// closure is preserved intentionally (faithful mirror — no "optimization").
///
/// # Safety (parallel-write contract)
/// `par_blocks` hands each task a distinct `counter` value.  Two distinct
/// counters produce distinct `base` values that differ in at least one FREE
/// bit position, so `base_a | offsets[m] ≠ base_b | offsets[n]` for any m, n
/// — disjoint writes, no aliasing.
pub(crate) fn apply_kq_scalar_f32(
    amps: &mut [Complex<f32>],
    qubits: &[u32],
    k: u8,
    data: &[Complex<f32>],
) {
    use crate::kernels::expand_with_fixed;
    use crate::kernels::unitary_kq::targets_offsets_fixed;

    let dim = 1usize << k;
    let (offsets, fixed) = targets_offsets_fixed(qubits, k);
    let len = amps.len();
    let outer = len >> k; // 2^(n-k) outer blocks

    let p = ComplexF32Ptr(amps.as_mut_ptr());

    crate::kernels::par_blocks(
        crate::kernels::tuning::DEFAULT_POLICY,
        outer,
        len,
        |c| c,
        move |counter| {
            let base = expand_with_fixed(counter, &fixed);

            // Read the block of 2^k amplitudes into a local buffer.
            let mut inb = vec![Complex::<f32>::new(0.0, 0.0); dim];
            for (m, inb_m) in inb.iter_mut().enumerate() {
                // SAFETY: base|offsets[m] is within [0, len), distinct across
                // m (coverage invariant from targets_offsets_fixed), and
                // disjoint from other counters' blocks — no two parallel tasks
                // share an index. The pointer lives for the duration of
                // apply_kq_scalar_f32.
                *inb_m = unsafe { *p.ptr().add(base | offsets[m]) };
            }

            // Matvec: out[r] = Σ_c data[r*dim + c] * in[c].
            for r in 0..dim {
                let mut acc = Complex::<f32>::new(0.0, 0.0);
                for cc in 0..dim {
                    acc += data[r * dim + cc] * inb[cc];
                }
                // SAFETY: same disjointness guarantee as the read above.
                unsafe { *p.ptr().add(base | offsets[r]) = acc };
            }
        },
    );
}

/// Top-level f32 2q dispatch. Phase B adds an AVX-512 dense arm; for now
/// always the scalar dense kernel (correct for CNOT/CZ/SWAP/dense alike —
/// the f64 specializations are non-fused-path optimizations out of scope).
// `pub` (like `apply_1q_f32`) so the `internal-bench`-gated integration
// test can drive the dispatcher end-to-end; effective visibility is
// `pub(crate)` without the feature (private `kernels` module).
pub fn apply_2q_f32(
    amps: &mut [Complex<f32>],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex<f32>; 4]; 4],
) {
    // Generic dense 4×4 — SIMD where the contract holds, scalar otherwise.
    // Mirrors the f64 `aos::apply_2q` generic-dense AVX-512 arm (aos.rs
    // lines 1597-1613) exactly, substituting LANES_F32 = 8 for the f64
    // arm's LANES = 4: feature gate + `1 << t_lo ≥ LANES_F32` (low-target
    // bit aligned so the inner SIMD walk has ≥ LANES_F32 contiguous pairs)
    // + every control > t_hi (no control-bit toggling in the outer walk;
    // renormalisation subtraction safe). The CNOT/CZ/SWAP permutation and
    // diagonal-4x4 specializations the f64 arm routes first are OUT OF
    // SCOPE here — the scalar fallback handles every other shape.
    #[cfg(target_arch = "x86_64")]
    {
        const LANES_F32: usize = 8;
        let t_lo = targets[0].min(targets[1]);
        let t_hi = targets[0].max(targets[1]);
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << t_lo) >= LANES_F32
            && controls.iter().all(|&c| c > t_hi)
        {
            // SAFETY: feature gate, `1 << t_lo ≥ LANES_F32` (aligned block
            // stride), every control > t_hi (no control-bit toggling +
            // safe renorm), and the apply_gate-level distinct/in-range
            // qubit invariants — exactly the f64 generic-dense arm's
            // contract with LANES_F32 substituted for LANES.
            unsafe {
                apply_2q_dense_avx512_f32(amps, targets, controls, m);
            }
            return;
        }
    }
    apply_2q_dense_scalar_f32(amps, targets, controls, m);
}

/// Scalar f32 multi-qubit diagonal-phase application. Mirror of
/// `diagonal_phase::apply_diagonal_phase_scalar_aos`. Phase angle computed
/// in f64 (rule 9), rotation factors cast to f32 for the amplitude multiply.
pub(crate) fn apply_diagonal_phase_scalar_f32(
    amps: &mut [Complex<f32>],
    dp: &aleph_ir::DiagonalPhase,
) {
    use crate::kernels::diagonal_phase::phase_at;
    use crate::kernels::tuning::DEFAULT_POLICY;
    let len = amps.len();
    let p = ComplexF32Ptr(amps.as_mut_ptr());
    par_blocks(
        DEFAULT_POLICY,
        len,
        len,
        |k| k,
        move |k| {
            // SAFETY: each k in 0..len is distinct; par_blocks bodies write
            // disjoint indices, so no aliasing across tasks.
            let amp = unsafe { &mut *p.ptr().add(k) };
            let phi = phase_at(dp, k as u64); // f64 — rule 9: angle precision kept in f64
            let (s, co) = phi.sin_cos();
            let (s, co) = (s as f32, co as f32);
            let re = amp.re * co - amp.im * s;
            let im = amp.re * s + amp.im * co;
            amp.re = re;
            amp.im = im;
        },
    );
}

/// Packed-complex AVX-512 path for f32 AoS `apply_1q`. 8 complex pairs
/// per `__m512` (16 f32 lanes), interleaved as `(re_0, im_0, re_1, im_1,
/// …, re_7, im_7)`.
///
/// Single-precision mirror of `aos::apply_1q_avx512` (the f64 source) per
/// the P2-08 f64→f32 substitution rules. The two structural differences
/// from the f64 kernel are:
///
/// 1. **SIMD width.** A `__m512` holds 8 complex pairs (16 f32 lanes) vs
///    the f64 `__m512d`'s 4 complex pairs (8 f64 lanes), so
///    `LANES_F32 = 8`. The "2 scalars per complex" interleave factor is
///    IDENTICAL; only how many complex per vector load changes (4 → 8).
///    A `LANES_F32`-wide block is still 64 bytes = one cache line
///    (8 complex × 8 bytes), exactly as the f64 kernel's 4-complex block.
///
/// 2. **The adjacent (re, im) swap.** The f64 kernel swaps each 64-bit
///    complex's real/imag with `vpermilpd` imm `0x55`. For f32 each
///    complex is two **32-bit** lanes, so the adjacent-pair swap is
///    `vpermilps` imm `0xB1` (`0b10110001`), which maps lanes
///    `[0,1,2,3] → [1,0,3,2]` within each 128-bit group — swapping each
///    adjacent `(re, im)` pair. (Reusing `0x55` here would be wrong and
///    silently corrupt results.)
///
/// **Math.** For a 2×2 unitary `U = [[u00, u01], [u10, u11]]` and the
/// state pair `(z_0, z_1) = (state[i], state[i | t_bit])`, each output is
/// `new_z_r = u[r][0] * z_0 + u[r][1] * z_1`. Each complex multiply
/// `u_rk * z_k` is implemented as
/// `vfmaddsub(u_rk_re_bcast, z_k, u_rk_im_bcast × swap(z_k))`, where
/// `swap` swaps adjacent f32 (re, im) pairs via `vpermilps` imm `0xB1`.
/// `fmaddsub` alternates SUB / ADD across even / odd lanes, producing
/// `(re_out, im_out)` for each of the 8 packed complex.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1usize << target ≥ LANES_F32` so the inner block has at least
///   `LANES_F32` contiguous pairs (no in-block tail; outer step is
///   `2 * target_bit ≥ 2 * LANES_F32`, keeping `i1 + LANES_F32` inside
///   the outer extent).
/// * Every control's qubit index is strictly greater than `target`, so
///   the inner SIMD walk's `block | j` for `j ∈ [0, target_bit)` doesn't
///   toggle any control bit.
/// * Standard apply_gate invariants: `target` and `controls` are distinct
///   and in qubit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
// `pub` (like `aos::apply_1q_avx512`) so the `internal-bench`-gated
// integration test can reach this; effectively `pub(crate)` without the
// feature (the module is private).
pub unsafe fn apply_1q_dense_avx512_f32(
    amps: &mut [Complex<f32>],
    target: u32,
    controls: &[u32],
    m: &[[Complex<f32>; 2]; 2],
) {
    use core::arch::x86_64::*;

    const LANES_F32: usize = 8; // 8 complex pairs per __m512 (16 lanes f32)

    let target_bit = 1usize << target;
    let len = amps.len();
    let n_qubits = len.trailing_zeros();

    // Pin the # Safety contract as debug-only asserts (mirrors the f64
    // source): a release-mode violation would silently no-op
    // (target_bit < LANES_F32) or underflow outer_count (controls below
    // target) — both catastrophic.
    debug_assert!(
        target_bit >= LANES_F32,
        "target_bit < LANES_F32: dispatch contract violated"
    );
    debug_assert!(
        controls.iter().all(|&c| c > target),
        "control at-or-below target: dispatch contract violated"
    );

    // Broadcast U matrix entries — constant across all iterations.
    let m00r = _mm512_set1_ps(m[0][0].re);
    let m00i = _mm512_set1_ps(m[0][0].im);
    let m01r = _mm512_set1_ps(m[0][1].re);
    let m01i = _mm512_set1_ps(m[0][1].im);
    let m10r = _mm512_set1_ps(m[1][0].re);
    let m10i = _mm512_set1_ps(m[1][0].im);
    let m11r = _mm512_set1_ps(m[1][1].re);
    let m11i = _mm512_set1_ps(m[1][1].im);

    let bp = BlockPtrF32(amps.as_mut_ptr() as *mut f32);

    // One SIMD unit = the LANES_F32-wide amplitude pair
    // (i0, i0+target_bit). Flattening the outer block walk with the inner
    // j-walk (via `par_units`) keeps the parallel dimension at
    // `len / (2*LANES_F32)` regardless of `target` (P2-01
    // count-starvation fix).
    let pair = |i0: usize| {
        let amps_ptr = bp.ptr();
        let i1 = i0 + target_bit;

        // Each Complex<f32> is 8 bytes (re, im). `_mm512_loadu_ps` reads
        // 16 consecutive f32 = 8 complex starting at `amps[i0]`. The
        // per-complex factor of 2 (2 f32 per complex) is identical to the
        // f64 source's `i0 * 2`; only the vector width (8 complex) differs.
        // SAFETY: `i0 + LANES_F32 ≤ block + target_bit ≤ len` (outer block
        // stride is `2 * target_bit`); `i1 + LANES_F32 ≤ block + 2*target_bit ≤ len`.
        let z0 = _mm512_loadu_ps(amps_ptr.add(i0 * 2));
        let z1 = _mm512_loadu_ps(amps_ptr.add(i1 * 2));

        // vpermilps imm = 0xB1 = 0b10110001: within each 128-bit group,
        // lanes [0,1,2,3] → [1,0,3,2], i.e. each adjacent (re, im) pair
        // is swapped. After permute: (im_0, re_0, im_1, re_1, …).
        let z0_swap = _mm512_permute_ps::<0xB1>(z0);
        let z1_swap = _mm512_permute_ps::<0xB1>(z1);

        // U_ij × z_k = vfmaddsub(U_ij_re, z_k, U_ij_im × z_k_swap).
        // fmaddsub(a, b, c) = (a·b - c) on even lanes, (a·b + c) on odd.
        // Even lanes (re_out): m_re·z_re - m_im·z_im ✓
        // Odd lanes  (im_out): m_re·z_im + m_im·z_re ✓
        let t00 = _mm512_mul_ps(m00i, z0_swap);
        let prod00 = _mm512_fmaddsub_ps(m00r, z0, t00);
        let t01 = _mm512_mul_ps(m01i, z1_swap);
        let prod01 = _mm512_fmaddsub_ps(m01r, z1, t01);
        let new_z0 = _mm512_add_ps(prod00, prod01);

        let t10 = _mm512_mul_ps(m10i, z0_swap);
        let prod10 = _mm512_fmaddsub_ps(m10r, z0, t10);
        let t11 = _mm512_mul_ps(m11i, z1_swap);
        let prod11 = _mm512_fmaddsub_ps(m11r, z1, t11);
        let new_z1 = _mm512_add_ps(prod10, prod11);

        _mm512_storeu_ps(amps_ptr.add(i0 * 2), new_z0);
        _mm512_storeu_ps(amps_ptr.add(i1 * 2), new_z1);
    };

    // Caller guarantees `target_bit ≥ LANES_F32` (both powers of two), so
    // `units_per_block ≥ 1` is a power of two.
    let units_per_block = target_bit / LANES_F32;
    let policy =
        tuning::resolve_policy(GateClass::OneQGeneric, tuning::pos_class(target, n_qubits));

    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let outer_count = len / outer_step;
        crate::kernels::par_units(
            policy,
            outer_count,
            units_per_block,
            LANES_F32,
            len,
            |bk| bk * outer_step,
            pair,
        );
        return;
    }

    // Controlled SIMD path. Identical index algebra to the f64 source:
    // renormalise control positions (subtract `target + 1`) so
    // `expand_with_fixed` lays them out densely, then left-shift back.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    // Subtraction is safe: each control is distinct from target and
    // < n_qubits, all controls > target, so
    // `target + 1 + controls.len() ≤ n_qubits`.
    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    crate::kernels::par_units(
        policy,
        outer_count,
        units_per_block,
        LANES_F32,
        len,
        |bk| crate::kernels::expand_with_fixed(bk, &fixed_above) << (target + 1),
        pair,
    );
}

/// Packed-complex AVX-512 diagonal 1q kernel for f32 AoS. Each amplitude
/// is scaled by a broadcast diagonal entry — `d0` for amplitudes whose
/// `target` bit is clear, `d1` for those set — a per-amplitude complex
/// scale with no pairing/permute between the two halves.
///
/// Single-precision mirror of `aos::apply_1q_diagonal_avx512` (the f64
/// source) per the P2-08 f64→f32 substitution rules. The two structural
/// differences from the f64 kernel are:
///
/// 1. **SIMD width.** A `__m512` holds 8 complex pairs (16 f32 lanes) vs
///    the f64 `__m512d`'s 4 complex pairs (8 f64 lanes), so
///    `LANES_F32 = 8`. The "2 scalars per complex" interleave factor
///    (`i * 2`) is IDENTICAL; only how many complex per vector load
///    changes (4 → 8).
///
/// 2. **The adjacent (re, im) swap.** The f64 kernel swaps each 64-bit
///    complex's real/imag with `vpermilpd` imm `0x55`. For f32 each
///    complex is two **32-bit** lanes, so the adjacent-pair swap is
///    `vpermilps` imm `0xB1` (`0b10110001`), mapping lanes
///    `[0,1,2,3] → [1,0,3,2]` within each 128-bit group — swapping each
///    adjacent `(re, im)` pair. (Reusing `0x55` here would be wrong.)
///
/// **Math.** For a broadcast complex `d = (re, im)` and a packed state
/// vector `z`, the scale `d × z` is
/// `vfmaddsub(d_re_bcast, z, d_im_bcast × swap(z))`, where `swap` swaps
/// adjacent f32 (re, im) pairs via `vpermilps` imm `0xB1`. `fmaddsub`
/// alternates SUB / ADD across even / odd lanes, producing
/// `(re_out, im_out)` for each of the 8 packed complex. The 0-side block
/// is scaled by `d0`, the 1-side block by `d1`.
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1usize << target ≥ LANES_F32` so the inner SIMD walk has at least
///   `LANES_F32` contiguous pairs per sub-block.
/// * Every control's qubit index is strictly greater than `target`, so
///   the inner walk's `block | j` for `j ∈ [0, target_bit)` doesn't
///   toggle any control bit.
/// * Standard apply_gate invariants: `target` and `controls` are distinct
///   and in qubit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
// `pub` (like `aos::apply_1q_diagonal_avx512`) so the `internal-bench`-gated
// integration test can reach this; effectively `pub(crate)` without the
// feature (the module is private).
pub unsafe fn apply_1q_diag_avx512_f32(
    amps: &mut [Complex<f32>],
    target: u32,
    controls: &[u32],
    d0: Complex<f32>,
    d1: Complex<f32>,
) {
    use core::arch::x86_64::*;

    const LANES_F32: usize = 8; // 8 complex pairs per __m512 (16 lanes f32)

    let target_bit = 1usize << target;
    let len = amps.len();

    // Pin the # Safety contract as debug-only asserts (mirrors the f64
    // source): a release-mode violation would silently no-op
    // (target_bit < LANES_F32) or underflow outer_count (controls below
    // target).
    debug_assert!(
        target_bit >= LANES_F32,
        "target_bit < LANES_F32: dispatch contract violated"
    );
    debug_assert!(
        controls.iter().all(|&c| c > target),
        "control at-or-below target: dispatch contract violated"
    );

    // Broadcast the two diagonal entries; constant across the walk.
    let d0r = _mm512_set1_ps(d0.re);
    let d0i = _mm512_set1_ps(d0.im);
    let d1r = _mm512_set1_ps(d1.re);
    let d1i = _mm512_set1_ps(d1.im);

    let bp = BlockPtrF32(amps.as_mut_ptr() as *mut f32);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        // 0-side: amps[block .. block + target_bit] get * d0.
        let mut j = 0usize;
        while j + LANES_F32 <= target_bit {
            let i0 = block | j;
            // Each Complex<f32> is 8 bytes (re, im). `_mm512_loadu_ps`
            // reads 16 consecutive f32 = 8 complex starting at amps[i0];
            // the per-complex factor of 2 (`i0 * 2`) is identical to the
            // f64 source — only the vector width (8 complex) differs.
            // SAFETY: i0 + LANES_F32 ≤ block + target_bit ≤ len.
            let z = _mm512_loadu_ps(amps_ptr.add(i0 * 2));
            // vpermilps 0xB1: each adjacent (re, im) pair → (im, re).
            let zs = _mm512_permute_ps::<0xB1>(z);
            // t = d0_im * zs : per pair → (d0.im * im, d0.im * re, …).
            let t = _mm512_mul_ps(d0i, zs);
            // out = vfmaddsub(d0_re, z, t) :
            //   even lane = d0.re*re - d0.im*im = (d0 * z).re  ✓
            //   odd  lane = d0.re*im + d0.im*re = (d0 * z).im  ✓
            let out = _mm512_fmaddsub_ps(d0r, z, t);
            _mm512_storeu_ps(amps_ptr.add(i0 * 2), out);
            j += LANES_F32;
        }
        debug_assert_eq!(j, target_bit);

        // 1-side: amps[block + target_bit .. block + 2*target_bit] get * d1.
        let mut j = 0usize;
        while j + LANES_F32 <= target_bit {
            let i1 = block | target_bit | j;
            // SAFETY: i1 + LANES_F32 ≤ block + 2*target_bit ≤ len.
            let z = _mm512_loadu_ps(amps_ptr.add(i1 * 2));
            let zs = _mm512_permute_ps::<0xB1>(z);
            let t = _mm512_mul_ps(d1i, zs);
            let out = _mm512_fmaddsub_ps(d1r, z, t);
            _mm512_storeu_ps(amps_ptr.add(i1 * 2), out);
            j += LANES_F32;
        }
        debug_assert_eq!(j, target_bit);
    };

    let n_qubits = len.trailing_zeros();
    let policy = tuning::resolve_policy(GateClass::OneQDiag, tuning::pos_class(target, n_qubits));

    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let count = len / outer_step;
        crate::kernels::par_blocks(policy, count, len, |k| k * outer_step, outer_iter);
        return;
    }

    // Controlled SIMD path. Identical index algebra to the f64 source:
    // renormalise control positions (subtract `target + 1`) so
    // `expand_with_fixed` lays them out densely, then left-shift back.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    for &c in controls {
        fixed_above.push((c - target - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
    crate::kernels::par_blocks(
        policy,
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        outer_iter,
    );
}

/// Packed-complex AVX-512 generic 2q dense kernel for f32 AoS. The inner
/// walk steps by `LANES_F32 = 8` complex pairs along the low-target axis
/// (requires `1 << t_lo >= LANES_F32`); the outer walk enumerates quartet
/// base indices via [`expand_with_fixed`].
///
/// Single-precision mirror of `aos::apply_2q_avx512` (the f64 source) per
/// the P2-08 f64→f32 substitution rules. The renormalised outer-walk
/// index algebra (`expand_with_fixed`, the `<< (t_lo + 1)` shift, the
/// `(t_hi - t_lo - 1, false)` renormalisation, control renorm
/// `(c - t_lo - 1, true)`) is precision-independent and copied
/// byte-for-byte from the f64 source; only the SIMD width (8 complex pairs
/// per `__m512` vs 4 per `__m512d`), the intrinsic suffix (`_ps` vs `_pd`),
/// and the (re,im) swap permute (`0xB1` vs `0x55`) differ.
///
/// **Math.** For each quartet `(z00, z01, z10, z11)`, compute
/// `new_z_r = Σ_c m[r][c] * z_c`. Each `m[r][c] * z_c` is one
/// `vfmaddsub(m_re_bcast, z_c, m_im_bcast × vpermilps<0xB1>(z_c))` — the
/// same packed-complex idiom as `apply_1q_dense_avx512_f32`, replicated
/// across four loaded subspaces.
///
/// **Outer-walk (bit-disjointness invariant).** The kernel computes each
/// amplitude index as `i = block | offsets[k] | j`, so for the `|` to
/// behave as `+` (the only way the SAFETY bound holds), the three pieces
/// MUST occupy disjoint bit positions:
///
/// * `j` walks `[0, t_lo_bit)` in `LANES_F32` strides — bits `[0, t_lo)`.
/// * `offsets[k] ∈ {0, t_lo_bit, t_hi_bit, t_lo_bit | t_hi_bit}` — bits
///   exactly `{t_lo, t_hi}`.
/// * `block` MUST therefore use only bits strictly above `t_lo`, with bit
///   `t_hi` clear and every control bit set.
///
/// Achieved with the same renormalise-then-shift idiom as
/// `apply_1q_dense_avx512_f32`, extended to two reserved positions:
/// `expand_with_fixed` lays out `t_hi` and every control at *renormalised*
/// positions (each minus `t_lo + 1`) in the "above t_lo" subspace, and a
/// left-shift by `t_lo + 1` promotes the result to actual qubit positions.
/// `Controls.len() = 0` collapses naturally — no separate uncontrolled
/// branch needed.
///
/// [`expand_with_fixed`]: crate::kernels::expand_with_fixed
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports AVX-512F.
/// * `1 << min(targets) >= LANES_F32` (= 8) — inner SIMD walk has ≥
///   `LANES_F32` contiguous pairs per sub-block.
/// * Every external control's qubit index is strictly greater than
///   `max(targets)`, so the outer-walk's bit-expansion never toggles a
///   control bit and the renormalisation subtraction `c - t_lo - 1` never
///   underflows.
/// * Distinct targets/controls, all in qubit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn apply_2q_dense_avx512_f32(
    amps: &mut [Complex<f32>],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex<f32>; 4]; 4],
) {
    use core::arch::x86_64::*;

    const LANES_F32: usize = 8; // 8 complex pairs per __m512 (16 lanes f32)

    let t_lo = targets[0].min(targets[1]);
    let t_hi = targets[0].max(targets[1]);
    let t_lo_bit = 1usize << t_lo;
    let t_hi_bit = 1usize << t_hi;
    let t_mask = t_lo_bit | t_hi_bit;
    let len = amps.len();

    debug_assert!(
        t_lo_bit >= LANES_F32,
        "t_lo_bit < LANES_F32: dispatch contract violated"
    );
    debug_assert!(
        controls.iter().all(|&c| c > t_hi),
        "control at-or-below t_hi: dispatch contract violated"
    );

    // Index permutation: targets[0] is MSB of matrix index k, targets[1]
    // is LSB. Identical to the f64 source (precision-independent).
    let (offset_k1, offset_k2) = if targets[0] < targets[1] {
        // targets[0]=t_lo, targets[1]=t_hi → k bit 1 (low) selects t_hi_bit
        (t_hi_bit, t_lo_bit)
    } else {
        (t_lo_bit, t_hi_bit)
    };
    let offsets = [0usize, offset_k1, offset_k2, t_mask];

    // Broadcast all 16 matrix cells.
    let mut m_re = [_mm512_setzero_ps(); 16];
    let mut m_im = [_mm512_setzero_ps(); 16];
    for r in 0..4 {
        for c in 0..4 {
            m_re[r * 4 + c] = _mm512_set1_ps(m[r][c].re);
            m_im[r * 4 + c] = _mm512_set1_ps(m[r][c].im);
        }
    }

    let bp = BlockPtrF32(amps.as_mut_ptr() as *mut f32);

    let outer_iter = |block: usize| {
        let amps_ptr = bp.ptr();
        let mut j = 0usize;
        while j + LANES_F32 <= t_lo_bit {
            // Load 4 sub-blocks, each LANES_F32 complex pairs. Base index
            // for k=0 is `block | j` (no target bits set).
            let mut z = [_mm512_setzero_ps(); 4];
            let mut zs = [_mm512_setzero_ps(); 4];
            for k in 0..4 {
                let i_k = block | offsets[k] | j;
                // SAFETY: bit-disjointness invariant (see doc-comment
                // "Outer-walk" section): `block` ⊆ bits ≥ t_lo+1 (t_hi
                // clear, every control set), `offsets[k]` ⊆ {t_lo, t_hi},
                // `j` ⊆ [0, t_lo). The three are pairwise bit-disjoint, so
                //   i_k = block + offsets[k] + j ≤ len - LANES_F32.
                // The per-complex factor of 2 (2 f32 per complex) is
                // identical to the f64 source's `i_k * 2`.
                z[k] = _mm512_loadu_ps(amps_ptr.add(i_k * 2));
                zs[k] = _mm512_permute_ps::<0xB1>(z[k]);
            }

            // Compute each output row: new_z[r] = Σ_c m[r][c] * z[c].
            let mut new_z = [_mm512_setzero_ps(); 4];
            for r in 0..4 {
                let t0 = _mm512_mul_ps(m_im[r * 4], zs[0]);
                let mut p = _mm512_fmaddsub_ps(m_re[r * 4], z[0], t0);
                let t1 = _mm512_mul_ps(m_im[r * 4 + 1], zs[1]);
                p = _mm512_add_ps(p, _mm512_fmaddsub_ps(m_re[r * 4 + 1], z[1], t1));
                let t2 = _mm512_mul_ps(m_im[r * 4 + 2], zs[2]);
                p = _mm512_add_ps(p, _mm512_fmaddsub_ps(m_re[r * 4 + 2], z[2], t2));
                let t3 = _mm512_mul_ps(m_im[r * 4 + 3], zs[3]);
                p = _mm512_add_ps(p, _mm512_fmaddsub_ps(m_re[r * 4 + 3], z[3], t3));
                new_z[r] = p;
            }

            // Store back into the same 4 sub-blocks.
            for k in 0..4 {
                let i_k = block | offsets[k] | j;
                // SAFETY: same bit-disjointness invariant as the load
                // above ⇒ i_k + LANES_F32 ≤ len.
                _mm512_storeu_ps(amps_ptr.add(i_k * 2), new_z[k]);
            }

            j += LANES_F32;
        }
        debug_assert_eq!(j, t_lo_bit);
    };

    // Outer-walk: reserve bits `[0, t_lo]` for the inner SIMD walk by
    // renormalising every "fixed" position (t_hi and each external
    // control) — subtract `t_lo + 1` — then left-shift `expand_with_fixed`'s
    // result by `t_lo + 1`. Identical to the f64 source.
    //
    // Subtraction `t_hi - t_lo - 1` is safe (`t_hi > t_lo` by min/max of
    // distinct targets). `c - t_lo - 1` is safe because the dispatch
    // contract requires every control `c > t_hi > t_lo`.
    let mut fixed_above: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed_above.push((t_hi - t_lo - 1, false));
    for &c in controls {
        fixed_above.push((c - t_lo - 1, true));
    }
    fixed_above.sort_unstable_by_key(|&(pos, _)| pos);

    let n_qubits = len.trailing_zeros();
    let policy = tuning::resolve_policy(GateClass::TwoQDense, tuning::pos_class(t_hi, n_qubits));
    let outer_count = 1usize << (n_qubits - t_lo - 2 - controls.len() as u32);
    crate::kernels::par_blocks(
        policy,
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (t_lo + 1),
        outer_iter,
    );
}

/// Top-level f32 1q dispatch: diagonal fast path, then the AVX-512 dense
/// arm (when supported and the dispatch contract holds), else the generic
/// scalar dense kernel.
///
/// `pub` (like the scalar reference) so the `internal-bench`-gated
/// integration test can drive the dispatcher.
pub fn apply_1q_f32(
    amps: &mut [Complex<f32>],
    target: u32,
    controls: &[u32],
    m: &[[Complex<f32>; 2]; 2],
) {
    if is_diagonal_2x2_f32(m) {
        #[cfg(target_arch = "x86_64")]
        {
            const LANES_F32: usize = 8;
            if std::is_x86_feature_detected!("avx512f")
                && (1usize << target) >= LANES_F32
                && controls.iter().all(|&c| c > target)
            {
                // SAFETY: feature gate + target_bit ≥ LANES_F32 (aligned
                // block stride) + every control > target (no control-bit
                // toggling in the inner SIMD walk) + apply_gate-level
                // qubit-range + distinct invariants.
                unsafe {
                    apply_1q_diag_avx512_f32(amps, target, controls, m[0][0], m[1][1]);
                }
                return;
            }
        }
        apply_1q_diag_scalar_f32(amps, target, controls, m[0][0], m[1][1]);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        const LANES_F32: usize = 8;
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << target) >= LANES_F32
            && controls.iter().all(|&c| c > target)
        {
            // SAFETY: feature detection gates the call; the kernel's
            // bounds + alignment invariants follow from
            // `1usize << target ≥ LANES_F32` (LANES_F32-aligned block
            // stride), `c > target` for every control (no control-bit
            // toggling in the inner SIMD walk), and the apply_gate-level
            // qubit-range + duplicate-qubit checks.
            unsafe {
                apply_1q_dense_avx512_f32(amps, target, controls, m);
            }
            return;
        }
    }
    apply_1q_dense_scalar_f32(amps, target, controls, m);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }

    #[test]
    fn x_gate_swaps_q0() {
        // |0⟩ → X → |1⟩
        let mut amps = vec![c(1.0, 0.0), c(0.0, 0.0)];
        let x = [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]];
        apply_1q_dense_scalar_f32(&mut amps, 0, &[], &x);
        assert!((amps[0].norm() - 0.0).abs() < 1e-6);
        assert!((amps[1].norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hadamard_on_zero_is_plus() {
        // H|0⟩ = (|0⟩ + |1⟩) / √2
        let h = 1.0f32 / 2.0f32.sqrt();
        let mut amps = vec![c(1.0, 0.0), c(0.0, 0.0)];
        let hm = [[c(h, 0.0), c(h, 0.0)], [c(h, 0.0), c(-h, 0.0)]];
        apply_1q_dense_scalar_f32(&mut amps, 0, &[], &hm);
        assert!((amps[0].re - h).abs() < 1e-6);
        assert!((amps[1].re - h).abs() < 1e-6);
    }

    #[test]
    fn diag_dispatch_phase_gate() {
        // S gate = diag(1, i) on |+> ; check q0 amplitude picks up i on |1>.
        let mut amps = vec![c(0.5, 0.0), c(0.5, 0.0)];
        let s = [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]];
        apply_1q_f32(&mut amps, 0, &[], &s);
        assert!((amps[0].re - 0.5).abs() < 1e-6 && amps[0].im.abs() < 1e-6);
        assert!(amps[1].re.abs() < 1e-6 && (amps[1].im - 0.5).abs() < 1e-6);
    }

    #[test]
    fn toffoli_flips_target_when_controls_set() {
        // 8-amp state; index = (q0<<2)|(q1<<1)|q2 (MSB convention, qubits[0]=q0).
        //
        // Index adjustment: with the kernel's MSB convention, `targets[0]` is the
        // HIGH bit of the matrix-index k (bit 2), `targets[1]` the middle (bit 1),
        // and `targets[2]` the LOW bit (bit 0).  With `targets=[0,1,2]` the mapping
        // from matrix-index k to amplitude-index is:
        //   idx[k] = (k&4 → bit targets[0]=0) | (k&2 → bit targets[1]=1) | (k&1 → bit targets[2]=2)
        //         = (k>>2)<<0 | ((k>>1)&1)<<1 | (k&1)<<2
        // which is a bit-reversal of k.  So matrix k=6 (110₂) maps to amp 3 (011₂),
        // not 6.  To get k=n mapping directly to amp n (identity) we need
        // `targets=[2,1,0]` — highest qubit-index in targets[0] (MSB of k).
        let zero = c(0.0, 0.0);
        let mut amps = vec![zero; 8];
        amps[6] = c(1.0, 0.0); // |110>
        let mut t = [[zero; 8]; 8];
        for (i, row) in t.iter_mut().enumerate() {
            row[i] = c(1.0, 0.0);
        }
        t[6][6] = zero;
        t[7][7] = zero;
        t[6][7] = c(1.0, 0.0);
        t[7][6] = c(1.0, 0.0);
        // targets=[2,1,0] so that matrix-index k maps directly to amplitude-index k
        // (no bit reversal), matching the 6↔7 swap in the matrix above.
        apply_3q_generic_f32(&mut amps, [2, 1, 0], &[], &t);
        assert!(amps[6].norm() < 1e-6);
        assert!((amps[7].re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn kq_k2_swap_matrix() {
        // 2-qubit SWAP as a 4x4 UnitaryKq on a 2-qubit state |01>.
        //
        // Convention check: with qubits=[0,1] the kernel sorts targets ascending
        // (already [0,1]) and maps matrix index m to amp index via
        //   offsets[m] = (bit p of m) << q[k-1-p]
        // For k=2, q=[0,1]: offsets[0]=0, offsets[1]=1<<q[0]=1, offsets[2]=1<<q[1]=2,
        // offsets[3]=3. So matrix index 1 (|01>) maps to amp[1] and index 2 (|10>)
        // to amp[2]. SWAP swaps |01><->|10>, i.e. row 1 col 2 and row 2 col 1 are 1.
        // Starting from amps=[0,1,0,0] (amp[1]=1 = |01>), the result is amps[2]=1.
        let zero = c(0.0, 0.0);
        let one = c(1.0, 0.0);
        let mut amps = vec![zero, one, zero, zero]; // |01>
                                                    // SWAP swaps indices 1<->2.
        let data = vec![
            one, zero, zero, zero, zero, zero, one, zero, zero, one, zero, zero, zero, zero, zero,
            one,
        ];
        apply_kq_scalar_f32(&mut amps, &[0, 1], 2, &data);
        assert!((amps[2].re - 1.0).abs() < 1e-6);
        assert!(amps[1].norm() < 1e-6);
    }

    #[test]
    fn diagonal_phase_global_rotation() {
        // DiagonalPhase with a single empty-conds term (global phase π/2):
        // every amplitude is rotated by π/2, i.e. multiplied by i.
        // Starting from |0> (re=1, im=0), the result should be (re≈0, im≈1).
        use aleph_ir::{DiagonalPhase, PhaseTerm};
        use smallvec::smallvec;
        let dp = DiagonalPhase {
            n_qubits: 1,
            terms: vec![PhaseTerm {
                conds: smallvec![], // empty conds = global phase (fires for all indices)
                angle: std::f64::consts::FRAC_PI_2,
            }],
        };
        let mut amps = vec![c(1.0, 0.0), c(0.0, 0.0)];
        apply_diagonal_phase_scalar_f32(&mut amps, &dp);
        // e^{iπ/2} = i on |0>.
        assert!(amps[0].re.abs() < 1e-6 && (amps[0].im - 1.0).abs() < 1e-6);
        // |1> (amplitude 0) is also rotated by i (global phase), but starts at 0 — stays 0.
        assert!(amps[1].re.abs() < 1e-6 && amps[1].im.abs() < 1e-6);
    }

    #[test]
    fn cnot_makes_bell_from_plus_zero() {
        // (H⊗I)|00> then CNOT(control=qubits[0], target=qubits[1]) → Bell.
        //
        // MSB convention: targets[0] is the HIGH bit of the matrix index k.
        // To place the control on qubit 1 (bit 1 of the address, which is the
        // address-MSB for a 2-qubit state) we pass targets=[1, 0]: targets[0]=1
        // (bit-1 of address = matrix-k bit-1), targets[1]=0 (bit-0 of address).
        // With targets=[1,0]: t0_bit=2, t1_bit=1, so idx=[i, i|1, i|2, i|3].
        // Input (H⊗I)|00> = (|00>+|10>)/√2 → amps=[h, 0, h, 0] (index 2 = |10>
        // in MSB ordering has bit-1 set, matching targets[0]=1 = control set).
        // CX maps |10>→|11>: result = (|00>+|11>)/√2 → amps=[h, 0, 0, h].
        let h = 1.0f32 / 2.0f32.sqrt();
        let mut amps = vec![c(h, 0.0), c(0.0, 0.0), c(h, 0.0), c(0.0, 0.0)];
        let zero = c(0.0, 0.0);
        let one = c(1.0, 0.0);
        let cx = [
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
            [zero, zero, one, zero],
        ];
        // targets=[1,0]: targets[0]=1 is the MSB of the matrix index (= control qubit).
        apply_2q_f32(&mut amps, [1, 0], &[], &cx);
        assert!((amps[0].re - h).abs() < 1e-6);
        assert!((amps[3].re - h).abs() < 1e-6);
        assert!(amps[1].norm() < 1e-6 && amps[2].norm() < 1e-6);
    }
}
