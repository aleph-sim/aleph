//! `NaiveSvBackend` — the naive CPU state-vector backend.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateInstance, PauliString};
use rand::{rngs::StdRng, SeedableRng};

use crate::state::CpuState;

/// Soft cap on qubits. 2^28 × 16 bytes = 4 GiB, comfortable on a
/// 16 GiB development machine. Acceptance target is 20 qubits.
pub(crate) const MAX_NAIVE_QUBITS: u32 = 28;

/// Naive single-threaded CPU state-vector backend.
pub struct NaiveSvBackend {
    // `rng` is consumed by `measure` / `sample` (lands in P0-09 Tasks 12–13).
    #[allow(dead_code)]
    pub(crate) rng: StdRng,
}

impl NaiveSvBackend {
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

impl Default for NaiveSvBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for NaiveSvBackend {
    type State = CpuState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_NAIVE_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_NAIVE_QUBITS,
            });
        }
        let dim = 1usize << num_qubits;
        let mut amps = vec![Complex::new(0.0, 0.0); dim];
        amps[0] = Complex::new(1.0, 0.0);
        Ok(CpuState { num_qubits, amps })
    }

    fn apply_gate(
        &mut self,
        _state: &mut Self::State,
        _gate: &GateInstance,
    ) -> Result<(), BackendError> {
        unimplemented!("apply_gate lands in P0-09 Task 11")
    }

    fn measure(&mut self, _state: &mut Self::State, _qubit: u32) -> Result<bool, BackendError> {
        unimplemented!("measure lands in P0-09 Task 12")
    }

    fn sample(&mut self, _state: &Self::State, _shots: u32) -> Result<Vec<u64>, BackendError> {
        unimplemented!("sample lands in P0-09 Task 13")
    }

    fn expectation_value(
        &mut self,
        _state: &Self::State,
        _pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        unimplemented!("expectation_value lands in P0-09 Task 15")
    }

    fn probabilities(
        &mut self,
        _state: &Self::State,
        _qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        unimplemented!("probabilities lands in P0-09 Task 14")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_initialises_zero_state() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(3).unwrap();
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.amplitudes().len(), 8);
        assert_eq!(s.amplitudes()[0], Complex::new(1.0, 0.0));
        for a in &s.amplitudes()[1..] {
            assert_eq!(*a, Complex::new(0.0, 0.0));
        }
    }

    #[test]
    fn allocate_rejects_too_many_qubits() {
        let mut b = NaiveSvBackend::with_seed(0);
        let err = b.allocate(MAX_NAIVE_QUBITS + 1).unwrap_err();
        assert_eq!(
            err,
            BackendError::TooManyQubits {
                requested: MAX_NAIVE_QUBITS + 1,
                limit: MAX_NAIVE_QUBITS,
            }
        );
    }

    #[test]
    fn allocate_zero_qubits_yields_unit_amplitude() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(0).unwrap();
        assert_eq!(s.amplitudes(), &[Complex::new(1.0, 0.0)]);
    }
}
