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

/// Scalar, rayon-parallel application over SoA (split re/im) arrays.
pub(crate) fn apply_diagonal_phase_scalar_soa(re: &mut [f64], im: &mut [f64], dp: &DiagonalPhase) {
    use crate::kernels::tuning::DEFAULT_POLICY;
    use crate::kernels::{par_blocks, BlockPtr};
    let len = re.len();
    debug_assert_eq!(len, im.len());
    let rp = BlockPtr(re.as_mut_ptr());
    let ip = BlockPtr(im.as_mut_ptr());
    par_blocks(
        DEFAULT_POLICY,
        len,
        len,
        |k| k,
        move |k| {
            // SAFETY: each k in 0..len is a distinct index; par_blocks calls
            // body on disjoint indices, so writes never alias across rayon
            // tasks. rp/ip point into two separate buffers and never alias
            // each other. Both BlockPtrs are Send+Sync.
            let r = unsafe { &mut *rp.ptr().add(k) };
            let i = unsafe { &mut *ip.ptr().add(k) };
            let phi = phase_at(dp, k as u64);
            let (s, co) = phi.sin_cos();
            let nr = *r * co - *i * s;
            let ni = *r * s + *i * co;
            *r = nr;
            *i = ni;
        },
    );
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

    #[test]
    fn applies_controlled_phase_soa() {
        use super::apply_diagonal_phase_scalar_soa;
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![0b01, 0b10],
                angle: std::f64::consts::FRAC_PI_2,
            }],
        };
        let mut re = vec![1.0f64; 4];
        let mut im = vec![0.0f64; 4];
        apply_diagonal_phase_scalar_soa(&mut re, &mut im, &dp);
        for x in 0..3 {
            assert!((re[x] - 1.0).abs() < 1e-12 && im[x].abs() < 1e-12, "x={x}");
        }
        // ψ[11] = e^{iπ/2} = i
        assert!(re[3].abs() < 1e-12 && (im[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn soa_matches_aos_on_random_terms() {
        // Cross-check the two scalar kernels agree on a multi-term diagonal.
        let dp = DiagonalPhase {
            n_qubits: 3,
            terms: vec![
                PhaseTerm {
                    conds: smallvec![0b001],
                    angle: 0.37,
                },
                PhaseTerm {
                    conds: smallvec![0b110],
                    angle: -1.2,
                },
                PhaseTerm {
                    conds: smallvec![0b010, 0b100],
                    angle: 2.1,
                },
                PhaseTerm {
                    conds: smallvec![],
                    angle: 0.5,
                }, // global
            ],
        };
        // seed an arbitrary non-uniform state
        let aos: Vec<Complex> = (0..8)
            .map(|k| Complex::new(0.1 * k as f64 + 1.0, 0.2 - 0.05 * k as f64))
            .collect();
        let mut aos_out = aos.clone();
        super::apply_diagonal_phase_scalar_aos(&mut aos_out, &dp);
        let mut re: Vec<f64> = aos.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = aos.iter().map(|c| c.im).collect();
        super::apply_diagonal_phase_scalar_soa(&mut re, &mut im, &dp);
        for k in 0..8 {
            assert!((re[k] - aos_out[k].re).abs() < 1e-12, "re k={k}");
            assert!((im[k] - aos_out[k].im).abs() < 1e-12, "im k={k}");
        }
    }
}
