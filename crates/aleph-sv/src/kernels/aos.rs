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
// Allow dead_code: caller in apply_1q dispatch lands in Task 5.
#[allow(dead_code)]
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

/// Apply a 2-qubit matrix to `targets = [t0, t1]` (with external
/// `controls`) in place.
///
/// **MSB convention (P0-06):** `targets[0]` is the *high* bit of the
/// matrix index `k`, `targets[1]` is the *low* bit. So matrix row 2
/// (binary `10`) corresponds to `(targets[0] = 1, targets[1] = 0)`.
/// This matches `Gate::Cnot` (`qubits = [control, target]`), whose
/// matrix swaps rows 2 ↔ 3.
///
/// Targets must be distinct; the caller (`apply_gate`) enforces this.
pub(crate) fn apply_2q(
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
