//! Single-precision (`Complex<f32>`) AoS gate kernels (P2-08).
//!
//! Scalar f32 kernels cover every gate type (correctness on any circuit
//! and on non-AVX-512 hosts); f32 AVX-512 kernels accelerate the fused
//! hot-types the optimized pipeline emits. Mirrors `kernels::aos` per the
//! f64→f32 substitution rules in the P2-08 plan; the FP64 path is untouched.

#![allow(dead_code)]

use aleph_core::Complex;

use crate::kernels::tuning::{self, GateClass};
use crate::kernels::{control_mask, par_blocks, ComplexF32Ptr};

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
}
