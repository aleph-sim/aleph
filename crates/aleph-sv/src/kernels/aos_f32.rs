//! Single-precision (`Complex<f32>`) AoS gate kernels (P2-08).
//!
//! Scalar f32 kernels cover every gate type (correctness on any circuit
//! and on non-AVX-512 hosts); f32 AVX-512 kernels accelerate the fused
//! hot-types the optimized pipeline emits. Mirrors `kernels::aos` per the
//! f64→f32 substitution rules in the P2-08 plan; the FP64 path is untouched.

#![allow(dead_code)]

use aleph_core::Complex;

use crate::kernels::tuning::{self, GateClass};
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
pub(crate) fn apply_1q_dense_scalar_f32(
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
pub(crate) fn apply_1q_diag_scalar_f32(
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
        GateClass::OneQGeneric,
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

/// Top-level f32 1q dispatch: diagonal fast path, else generic dense.
/// (Phase B adds AVX-512 arms guarded by `is_x86_feature_detected`.)
pub(crate) fn apply_1q_f32(
    amps: &mut [Complex<f32>],
    target: u32,
    controls: &[u32],
    m: &[[Complex<f32>; 2]; 2],
) {
    if is_diagonal_2x2_f32(m) {
        apply_1q_diag_scalar_f32(amps, target, controls, m[0][0], m[1][1]);
        return;
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
}
