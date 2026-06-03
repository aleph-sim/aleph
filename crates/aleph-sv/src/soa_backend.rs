//! `SoaSvBackend` — struct-of-arrays CPU state-vector backend.
//!
//! Mirrors `NaiveSvBackend` (P0-09) but stores amplitudes as paired
//! `AlignedBuf<f64>` (64-byte-aligned, P2-02) for SIMD-friendly memory
//! access. Same validation
//! discipline; same unitarity guard; dispatches to `kernels::soa`
//! rather than `kernels::aos`.

use aleph_backend::{Backend, BackendError};
use aleph_core::{AlignedBuf, GateError, GateInstance, GateMatrix, PauliString};
use rand::{rngs::StdRng, SeedableRng};

use crate::soa_state::SoaState;

/// Soft cap on qubits — matches `MAX_NAIVE_QUBITS = 28`
/// (`2^28 × 16 B = 4 GiB`, same as the AoS layout).
pub(crate) const MAX_SOA_QUBITS: u32 = 28;

/// SoA single-threaded CPU state-vector backend.
pub struct SoaSvBackend {
    pub(crate) rng: StdRng,
}

impl SoaSvBackend {
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

impl Default for SoaSvBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for SoaSvBackend {
    type State = SoaState;

    fn allocate(&mut self, num_qubits: u32) -> Result<SoaState, BackendError> {
        if num_qubits > MAX_SOA_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_SOA_QUBITS,
            });
        }
        let dim = 1usize << num_qubits;
        let mut re = AlignedBuf::<f64>::zeroed_state(dim);
        re[0] = 1.0;
        Ok(SoaState {
            num_qubits,
            re,
            im: AlignedBuf::<f64>::zeroed_state(dim),
        })
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
        let matrix = gate.gate.matrix().map_err(|e| match e {
            GateError::SymbolicParam => BackendError::SymbolicParam,
            GateError::NonFiniteParam => BackendError::NonFiniteParam {
                kind: gate.gate.name(),
            },
        })?;
        let deviation = crate::validation::unitarity_deviation(&matrix);
        if !deviation.is_finite() || deviation > aleph_core::AMPLITUDE_TOL {
            return Err(BackendError::NonUnitaryMatrix { deviation });
        }
        match matrix {
            GateMatrix::M2x2(m) => {
                let t = gate.qubits[0];
                crate::kernels::soa::apply_1q(&mut state.re, &mut state.im, t, &gate.controls, &m);
            }
            GateMatrix::M4x4(m) => {
                let t = [gate.qubits[0], gate.qubits[1]];
                crate::kernels::soa::apply_2q(&mut state.re, &mut state.im, t, &gate.controls, &m);
            }
            GateMatrix::M8x8(m) => {
                let t = [gate.qubits[0], gate.qubits[1], gate.qubits[2]];
                crate::kernels::soa::apply_3q(&mut state.re, &mut state.im, t, &gate.controls, &m);
            }
        }
        Ok(())
    }

    fn apply_diagonal_phase(
        &mut self,
        state: &mut Self::State,
        dp: &aleph_ir::DiagonalPhase,
    ) -> Result<(), BackendError> {
        crate::kernels::diagonal_phase::apply_diagonal_phase_scalar_soa(
            &mut state.re,
            &mut state.im,
            dp,
        );
        Ok(())
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        crate::measure_soa::measure_impl_soa(&mut self.rng, state, qubit)
    }
    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        crate::measure_soa::sample_impl_soa(&mut self.rng, state, shots)
    }
    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        crate::measure_soa::expectation_value_impl_soa(state, pauli)
    }
    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        crate::measure_soa::probabilities_impl_soa(state, qubits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Complex, Gate, Pauli, PauliString};
    use smallvec::smallvec;

    #[test]
    fn allocated_soa_state_is_cache_line_aligned() {
        let mut b = SoaSvBackend::new();
        // n=1 (small alloc): system allocator wouldn't incidentally 64-align —
        // AlignedBuf::zeroed is what forces it.
        let s1 = b.allocate(1).unwrap();
        assert_eq!(s1.re().as_ptr() as usize % aleph_core::CACHE_LINE, 0);
        assert_eq!(s1.im().as_ptr() as usize % aleph_core::CACHE_LINE, 0);
        // n=20 (large alloc) sanity check.
        let s20 = b.allocate(20).unwrap();
        assert_eq!(s20.re().as_ptr() as usize % aleph_core::CACHE_LINE, 0);
        assert_eq!(s20.im().as_ptr() as usize % aleph_core::CACHE_LINE, 0);
    }

    #[test]
    fn allocate_initialises_zero_ket() {
        let mut b = SoaSvBackend::with_seed(0);
        let s = b.allocate(3).unwrap();
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.re().len(), 8);
        assert_eq!(s.im().len(), 8);
        assert_eq!(s.re()[0], 1.0);
        assert!(s.re()[1..].iter().all(|&x| x == 0.0));
        assert!(s.im().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn allocate_rejects_too_many_qubits() {
        let mut b = SoaSvBackend::with_seed(0);
        let err = b.allocate(MAX_SOA_QUBITS + 1).unwrap_err();
        assert_eq!(
            err,
            BackendError::TooManyQubits {
                requested: MAX_SOA_QUBITS + 1,
                limit: MAX_SOA_QUBITS,
            }
        );
    }

    #[test]
    fn apply_gate_h_on_zero() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s.re()[0] - inv).abs() < 1e-12);
        assert!((s.re()[1] - inv).abs() < 1e-12);
    }

    #[test]
    fn apply_gate_cnot_creates_bell() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s.re()[0] - inv).abs() < 1e-12);
        assert!((s.re()[3] - inv).abs() < 1e-12);
        assert!(s.re()[1].abs() < 1e-12);
        assert!(s.re()[2].abs() < 1e-12);
    }

    #[test]
    fn apply_gate_arity_mismatch_rejected() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        let bad = GateInstance {
            gate: Gate::Cnot,
            qubits: smallvec![0u32],
            controls: smallvec![],
        };
        let err = b.apply_gate(&mut s, &bad).unwrap_err();
        assert_eq!(
            err,
            BackendError::ArityMismatch {
                kind: "Cnot",
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn apply_gate_non_unitary_user_matrix_rejected() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let two = Complex::new(2.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let mat = Gate::Unitary1q(Box::new([[two, zero], [zero, two]]));
        let bad = GateInstance::new(mat, smallvec![0u32]);
        let err = b.apply_gate(&mut s, &bad).unwrap_err();
        assert!(matches!(err, BackendError::NonUnitaryMatrix { .. }));
    }

    #[test]
    fn sample_bell_state_only_returns_00_or_11() {
        let mut b = SoaSvBackend::with_seed(7);
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        let shots = b.sample(&s, 1000).unwrap();
        assert!(shots.iter().all(|&v| v == 0 || v == 3));
        let zeros = shots.iter().filter(|&&v| v == 0).count();
        let threes = shots.iter().filter(|&&v| v == 3).count();
        assert!(
            zeros > 100 && threes > 100,
            "zeros={zeros}, threes={threes}"
        );
    }

    #[test]
    fn expectation_z_chain_on_ghz_is_plus_one() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(4).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        for t in 1u32..4 {
            b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0u32, t]))
                .unwrap();
        }
        let z_all = PauliString::new(
            1.0,
            vec![(0, Pauli::Z), (1, Pauli::Z), (2, Pauli::Z), (3, Pauli::Z)],
        )
        .unwrap();
        let ev = b.expectation_value(&s, &z_all).unwrap();
        assert!((ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn measure_rejects_nan_amplitude_state() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        s.re[1] = f64::NAN;
        let err = b.measure(&mut s, 0).unwrap_err();
        assert!(matches!(err, BackendError::InvalidState { .. }));
    }

    #[test]
    fn sample_rejects_unnormalised_state() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        s.re[0] = 2.0;
        s.re[1] = 0.0;
        let err = b.sample(&s, 10).unwrap_err();
        assert!(matches!(err, BackendError::InvalidState { .. }));
    }

    #[test]
    fn probabilities_plus_state_uniform_marginal() {
        let mut b = SoaSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let p = b.probabilities(&s, &[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }
}
