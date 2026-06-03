//! Diagonal-phase kernel: ψ[x] *= exp(i·phase(x)) in one streaming pass.

use aleph_core::Complex;
use aleph_ir::DiagonalPhase;

/// Evaluate the real phase for amplitude index `x`. Hot inner helper —
/// kept tiny so it inlines into the per-amplitude loop.
#[inline(always)]
pub(crate) fn phase_at(dp: &DiagonalPhase, x: u64) -> f64 {
    let mut phi = 0.0;
    for t in &dp.terms {
        let mut all = true;
        for &m in &t.conds {
            if (m & x).count_ones() & 1 == 0 {
                all = false;
                break;
            }
        }
        if all {
            phi += t.angle;
        }
    }
    phi
}

/// Scalar, rayon-parallel application over an AoS amplitude slice.
pub(crate) fn apply_diagonal_phase_scalar_aos(amps: &mut [Complex], dp: &DiagonalPhase) {
    use crate::kernels::tuning::DEFAULT_POLICY;
    use crate::kernels::{par_blocks, ComplexPtr};
    let len = amps.len();
    let p = ComplexPtr(amps.as_mut_ptr());
    par_blocks(
        DEFAULT_POLICY,
        len,
        len,
        |k| k,
        move |k| {
            // SAFETY: each k in 0..len is a distinct index; par_blocks calls
            // body on disjoint indices, so these per-element writes never
            // alias across rayon tasks. `p` (ComplexPtr) is Send+Sync.
            let amp = unsafe { &mut *p.ptr().add(k) };
            let phi = phase_at(dp, k as u64);
            let (s, co) = phi.sin_cos();
            let re = amp.re * co - amp.im * s;
            let im = amp.re * s + amp.im * co;
            amp.re = re;
            amp.im = im;
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_ir::PhaseTerm;
    use smallvec::smallvec;

    #[test]
    fn applies_controlled_phase_to_amplitudes() {
        // cp(π/2) on (0,1): multiply ψ[11] by i, others unchanged.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![0b01, 0b10],
                angle: std::f64::consts::FRAC_PI_2,
            }],
        };
        let mut amps = vec![Complex::new(1.0, 0.0); 4];
        apply_diagonal_phase_scalar_aos(&mut amps, &dp);
        for (x, amp) in amps.iter().enumerate().take(3) {
            assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12, "x={x}");
        }
        assert!((amps[3] - Complex::new(0.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn global_phase_applies_to_all() {
        // empty-conds term = global phase π → multiply everything by -1.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![],
                angle: std::f64::consts::PI,
            }],
        };
        let mut amps = vec![Complex::new(1.0, 0.0); 4];
        apply_diagonal_phase_scalar_aos(&mut amps, &dp);
        for (x, amp) in amps.iter().enumerate() {
            assert!((amp - Complex::new(-1.0, 0.0)).norm() < 1e-12, "x={x}");
        }
    }
}
