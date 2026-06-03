//! `Fp32SvBackend` — opt-in single-precision CPU state-vector backend (P2-08).
//! Mirror of [`crate::NaiveSvBackend`] over `Complex<f32>`. The FP64 backend
//! is the oracle reference and is untouched; this is a large-n performance
//! mode (~2× less memory traffic) at f32 accuracy (~1e-6).

use aleph_backend::{Backend, BackendError};
use aleph_core::{AlignedBuf, Complex, GateError, GateInstance, GateMatrix, PauliString};
use rand::{rngs::StdRng, SeedableRng};

use crate::fp32_state::Fp32CpuState;

/// Soft qubit cap. `Complex<f32>` is 8 B/amp: 2^29 × 8 B = 4 GiB. The
/// project software cap (28, matching the f64 backends) still bounds us.
pub(crate) const MAX_FP32_QUBITS: u32 = 28;

/// Opt-in single-precision CPU state-vector backend.
pub struct Fp32SvBackend {
    pub(crate) rng: StdRng,
}

impl Fp32SvBackend {
    /// Construct with an entropy-seeded RNG.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Construct with an explicit seed; runs are reproducible across
    /// processes and machines for a given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Default for Fp32SvBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Narrow an f64 gate matrix entry to f32. Gate matrices are materialised in
/// f64 (angles keep full precision); only the state is single-precision.
#[inline]
fn narrow(z: Complex) -> Complex<f32> {
    Complex::<f32>::new(z.re as f32, z.im as f32)
}

impl Backend for Fp32SvBackend {
    type State = Fp32CpuState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_FP32_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_FP32_QUBITS,
            });
        }
        let dim = 1usize << num_qubits;
        let mut amps = AlignedBuf::<Complex<f32>>::zeroed_state(dim);
        amps[0] = Complex::<f32>::new(1.0, 0.0);
        Ok(Fp32CpuState { num_qubits, amps })
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let n = state.num_qubits;
        let expected = gate.gate.arity();
        let got = gate.qubits.len();
        if expected != got {
            return Err(BackendError::ArityMismatch {
                kind: gate.gate.name(),
                expected,
                got,
            });
        }
        let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for &q in gate.qubits.iter().chain(gate.controls.iter()) {
            if q >= n {
                return Err(BackendError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
            if seen.contains(&q) {
                return Err(BackendError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        // UnitaryKq has no fixed-size GateMatrix; the kernel reads `data`
        // directly. Intercept before matrix() so `Unrepresentable` never fires.
        if let aleph_core::Gate::UnitaryKq { k, data } = &gate.gate {
            let data_f32: Vec<Complex<f32>> = data.iter().copied().map(narrow).collect();
            crate::kernels::aos_f32::apply_kq_scalar_f32(
                &mut state.amps,
                &gate.qubits,
                *k,
                &data_f32,
            );
            return Ok(());
        }
        let matrix = gate.gate.matrix().map_err(|e| match e {
            GateError::SymbolicParam => BackendError::SymbolicParam,
            GateError::NonFiniteParam => BackendError::NonFiniteParam {
                kind: gate.gate.name(),
            },
            GateError::Unrepresentable => BackendError::UnsupportedGate {
                kind: gate.gate.name(),
            },
        })?;
        // Unitarity check on the f64 matrix (defense-in-depth, mirrors the
        // f64 backend) before narrowing to f32.
        let deviation = crate::validation::unitarity_deviation(&matrix);
        if !deviation.is_finite() || deviation > aleph_core::AMPLITUDE_TOL {
            return Err(BackendError::NonUnitaryMatrix { deviation });
        }
        match matrix {
            GateMatrix::M2x2(m) => {
                let mf = [
                    [narrow(m[0][0]), narrow(m[0][1])],
                    [narrow(m[1][0]), narrow(m[1][1])],
                ];
                crate::kernels::aos_f32::apply_1q_f32(
                    &mut state.amps,
                    gate.qubits[0],
                    &gate.controls,
                    &mf,
                );
            }
            GateMatrix::M4x4(m) => {
                let mut mf = [[Complex::<f32>::new(0.0, 0.0); 4]; 4];
                for r in 0..4 {
                    for cc in 0..4 {
                        mf[r][cc] = narrow(m[r][cc]);
                    }
                }
                let t = [gate.qubits[0], gate.qubits[1]];
                crate::kernels::aos_f32::apply_2q_f32(&mut state.amps, t, &gate.controls, &mf);
            }
            GateMatrix::M8x8(m) => {
                let mut mf = [[Complex::<f32>::new(0.0, 0.0); 8]; 8];
                for r in 0..8 {
                    for cc in 0..8 {
                        mf[r][cc] = narrow(m[r][cc]);
                    }
                }
                let t = [gate.qubits[0], gate.qubits[1], gate.qubits[2]];
                crate::kernels::aos_f32::apply_3q_generic_f32(
                    &mut state.amps,
                    t,
                    &gate.controls,
                    &mf,
                );
            }
        }
        Ok(())
    }

    fn apply_diagonal_phase(
        &mut self,
        state: &mut Self::State,
        dp: &aleph_ir::DiagonalPhase,
    ) -> Result<(), BackendError> {
        crate::kernels::aos_f32::apply_diagonal_phase_scalar_f32(&mut state.amps, dp);
        Ok(())
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        crate::fp32_measure::measure_impl_f32(&mut self.rng, state, qubit)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        crate::fp32_measure::sample_impl_f32(&mut self.rng, state, shots)
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        crate::fp32_measure::expectation_value_impl_f32(state, pauli)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        crate::fp32_measure::probabilities_impl_f32(state, qubits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn allocate_initialises_zero_state() {
        let mut b = Fp32SvBackend::with_seed(1);
        let s = b.allocate(3).unwrap();
        assert_eq!(s.amplitudes()[0], Complex::<f32>::new(1.0, 0.0));
        assert_eq!(s.amplitudes().len(), 8);
    }

    #[test]
    fn allocate_rejects_too_many_qubits() {
        let mut b = Fp32SvBackend::with_seed(1);
        let err = b.allocate(MAX_FP32_QUBITS + 1).unwrap_err();
        assert_eq!(
            err,
            BackendError::TooManyQubits {
                requested: MAX_FP32_QUBITS + 1,
                limit: MAX_FP32_QUBITS,
            }
        );
    }

    #[test]
    fn bell_state_via_h_cnot() {
        let mut b = Fp32SvBackend::with_seed(1);
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        let a = s.amplitudes();
        let h = 1.0f32 / 2.0f32.sqrt();
        assert!((a[0].re - h).abs() < 1e-6);
        assert!((a[3].re - h).abs() < 1e-6);
        assert!(a[1].norm() < 1e-6 && a[2].norm() < 1e-6);
    }
}
