//! SoA gate application kernels — paired `Vec<f64>` storage.
//!
//! Convention identical to the AoS path (ADR 0004 / P0-06 spec §6):
//! `target` / `targets[0]` is the MSB of the matrix-index group;
//! `controls` are external (no row in the matrix). Real-arithmetic
//! expansion of `m * (re + i·im)` runs entirely on f64 pairs — the
//! compiler can vectorise the inner loop, and P1-03 will land an
//! explicit AVX2 specialisation on top.

use aleph_core::Complex;

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
pub(crate) fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    debug_assert_eq!(re.len(), im.len());
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
            re[i] = m[0][0].re * a0_re - m[0][0].im * a0_im
                + m[0][1].re * a1_re
                - m[0][1].im * a1_im;
            im[i] = m[0][0].re * a0_im
                + m[0][0].im * a0_re
                + m[0][1].re * a1_im
                + m[0][1].im * a1_re;
            // row 1
            re[j] = m[1][0].re * a0_re - m[1][0].im * a0_im
                + m[1][1].re * a1_re
                - m[1][1].im * a1_im;
            im[j] = m[1][0].re * a0_im
                + m[1][0].im * a0_re
                + m[1][1].re * a1_im
                + m[1][1].im * a1_re;
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
}
