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

/// Dispatch helper for SoA CNOT specialisations.  Placeholder routing
/// to the scalar kernel; AVX-512 tiers land in Task 13 (mirror of AoS
/// Tasks 6-8).
fn dispatch_cnot_soa(re: &mut [f64], im: &mut [f64], control: u32, target: u32, controls: &[u32]) {
    apply_2q_cnot_scalar(re, im, control, target, controls);
}

/// Dispatch helper for SoA SWAP.  Placeholder routing to the scalar
/// kernel; AVX-512 tiers land in Task 13 (mirror of AoS Task 9).
fn dispatch_swap_soa(re: &mut [f64], im: &mut [f64], targets: [u32; 2], controls: &[u32]) {
    apply_2q_swap_scalar(re, im, targets, controls);
}

/// Dispatch helper for the diagonal-4x4 branch (catches CZ,
/// controlled-Phase, Rzz, user diagonals).  Placeholder routing to
/// the scalar CZ kernel (when the matrix matches the CZ signature)
/// or scalar general-diagonal kernel; AVX-512 tiers land in Task 13
/// (mirror of AoS Tasks 10-11).
fn dispatch_diagonal_or_cz_soa(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 2],
    controls: &[u32],
    d: [Complex; 4],
    is_cz: bool,
) {
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
