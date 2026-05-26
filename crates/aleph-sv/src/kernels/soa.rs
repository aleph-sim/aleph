//! SoA gate application kernels — paired `Vec<f64>` storage.
//!
//! Convention identical to the AoS path (ADR 0004 / P0-06 spec §6):
//! `target` / `targets[0]` is the MSB of the matrix-index group;
//! `controls` are external (no row in the matrix). Real-arithmetic
//! expansion of `m * (re + i·im)` runs entirely on f64 pairs — the
//! compiler can vectorise the inner loop, and P1-03 will land an
//! explicit AVX2 specialisation on top.

use aleph_core::Complex;

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

/// Apply a 2-qubit matrix to `targets = [t0, t1]` (with external
/// `controls`) in place over paired SoA storage. MSB convention:
/// `targets[0]` is the high bit of the matrix index, `targets[1]` is
/// the low bit (matches `aos::apply_2q`).
pub(crate) fn apply_2q(
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

/// Apply a 1-qubit matrix to `target` (with external `controls`) in
/// place over a paired `(re, im)` SoA state. See the `aos.rs` analogue
/// for the index-pair convention.
///
/// Branch-free iteration (P1-02), two paths:
///
/// * **Uncontrolled** (`controls.is_empty()`): pure nested block/pair
///   — outer iterates blocks of size `2 * target_bit`, inner walks
///   unit-stride over `target_bit` pairs.  Zero helper overhead;
///   inner loop is exactly the shape `_mm256_loadu_pd` consumes
///   (P1-03 AVX2 lands here unchanged).
///
/// * **Controlled** (`!controls.is_empty()`): sort `(target, controls)`
///   ONCE into a stack-only `fixed` array, then iterate
///   `0..(1 << (n - 1 - |controls|))` and reconstruct each `i0` via
///   `expand_with_fixed(k, &fixed)`.  Hoisting the sort matters: an
///   earlier P1-02 draft kept `SmallVec::push` + `sort_unstable_by_key`
///   inside the hot loop and measured 245 ms → 333 ms on QFT-20
///   (1.36× regression vs P1-01).  Pre-sort + walk drops it back to
///   the expected speedup band.
pub(crate) fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    debug_assert_eq!(re.len(), im.len());
    debug_assert!(
        re.len().is_power_of_two(),
        "apply_1q: state length must be a power of two, got {}",
        re.len()
    );
    let target_bit = 1usize << target;
    let len = re.len();

    // Hoist matrix entries — let the compiler keep them in registers
    // across the inner block instead of refetching from `m` each iter.
    let m00_re = m[0][0].re;
    let m00_im = m[0][0].im;
    let m01_re = m[0][1].re;
    let m01_im = m[0][1].im;
    let m10_re = m[1][0].re;
    let m10_im = m[1][0].im;
    let m11_re = m[1][1].re;
    let m11_im = m[1][1].im;

    // Inline closure: apply the 2×2 matrix to the (re,im) pair at
    // (i0, i1). Reused by both paths so the 8-mul/4-add formula has
    // a single source of truth.
    let apply_pair = |re: &mut [f64], im: &mut [f64], i0: usize, i1: usize| {
        let a0_re = re[i0];
        let a0_im = im[i0];
        let a1_re = re[i1];
        let a1_im = im[i1];

        re[i0] = m00_re * a0_re - m00_im * a0_im
            + m01_re * a1_re
            - m01_im * a1_im;
        im[i0] = m00_re * a0_im
            + m00_im * a0_re
            + m01_re * a1_im
            + m01_im * a1_re;
        re[i1] = m10_re * a0_re - m10_im * a0_im
            + m11_re * a1_re
            - m11_im * a1_im;
        im[i1] = m10_re * a0_im
            + m10_im * a0_re
            + m11_re * a1_im
            + m11_im * a1_re;
    };

    if controls.is_empty() {
        // Fast path: pure nested block/pair, no helper.
        let outer_step = target_bit << 1;
        let mut block = 0usize;
        while block < len {
            for j in 0..target_bit {
                let i0 = block | j;
                let i1 = i0 | target_bit;
                apply_pair(re, im, i0, i1);
            }
            block += outer_step;
        }
    } else {
        // Controlled path: pre-sort fixed positions OUTSIDE the loop.
        let mut fixed: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
        fixed.push((target, false));
        for &c in controls {
            fixed.push((c, true));
        }
        fixed.sort_unstable_by_key(|(pos, _)| *pos);
        let n_qubits = len.trailing_zeros();
        let outer_count = 1usize << (n_qubits - fixed.len() as u32);
        for k in 0..outer_count {
            let i0 = super::expand_with_fixed(k, &fixed);
            let i1 = i0 | target_bit;
            apply_pair(re, im, i0, i1);
        }
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

        /// AoS / SoA equivalence on `apply_1q`: for any 1q gate, any
        /// target qubit, any set of 0-2 distinct external controls,
        /// and any normalised state, applying through both kernels
        /// yields matching amplitudes within 1e-12.
        ///
        /// The controls set is built deterministically from the
        /// strategy seed: take qubits `[0..5)` minus `q`, sort by a
        /// seed-keyed pseudo-random key, take the first `n_ctrls`.
        /// Distinctness from `q` and among each other is structural.
        #[test]
        fn apply_1q_soa_matches_aos(
            gate in arb_1q_gate(),
            q in 0u32..5,
            n_ctrls in 0usize..=2,
            ctrl_seed in any::<u32>(),
            amps in arb_state_vector(5),
        ) {
            let m = match gate.matrix().unwrap() {
                GateMatrix::M2x2(m) => m,
                _ => unreachable!("arb_1q_gate yields 1q gates"),
            };
            // Build a deterministic control set: candidates are 0..5 \ {q},
            // shuffled by a seed-keyed hash, truncated to n_ctrls, sorted.
            let mut ctrls: Vec<u32> = (0u32..5).filter(|c| *c != q).collect();
            ctrls.sort_by_key(|c| ctrl_seed.wrapping_mul(*c + 7));
            ctrls.truncate(n_ctrls);
            ctrls.sort_unstable();

            let re: Vec<f64> = amps.iter().map(|c| c.re).collect();
            let im: Vec<f64> = amps.iter().map(|c| c.im).collect();
            // AoS reference
            let mut aos_state = amps.clone();
            aos::apply_1q(&mut aos_state, q, &ctrls, &m);
            // SoA candidate
            let mut soa_re = re.clone();
            let mut soa_im = im.clone();
            apply_1q(&mut soa_re, &mut soa_im, q, &ctrls, &m);
            let soa_state = aos_from(&soa_re, &soa_im);
            for (a, b) in aos_state.iter().zip(soa_state.iter()) {
                prop_assert!(
                    (a - b).norm() < 1e-12,
                    "ctrls={ctrls:?} q={q}: aos {a} vs soa {b}",
                );
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
}
