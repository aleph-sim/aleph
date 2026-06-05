//! `MpsBackend`: the `aleph_backend::Backend` impl over `MpsState`.

use aleph_backend::{Backend, BackendError};
use aleph_core::{GateInstance, PauliString};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::mps::MAX_PROB_QUBITS;
use crate::{MpsError, MpsState};

/// MPS backend with a configurable max bond dimension χ.
pub struct MpsBackend {
    rng: StdRng,
    max_bond: usize,
}

/// Maximum qubit count `allocate` accepts. MPS memory scales as O(n·χ²), so
/// this guards only pathological allocations; practical limits are set by χ.
const MAX_QUBITS: u32 = 1024;

/// Default bond dimension used when none is specified.
const DEFAULT_MAX_BOND: usize = 128;

impl MpsBackend {
    /// Entropy-seeded RNG, default bond dimension χ = 128.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
            max_bond: DEFAULT_MAX_BOND,
        }
    }

    /// Explicit seed; reproducible for a given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            max_bond: DEFAULT_MAX_BOND,
        }
    }

    /// Override the maximum bond dimension χ (clamped to at least 1).
    pub fn with_max_bond(mut self, chi: usize) -> Self {
        self.max_bond = chi.max(1);
        self
    }
}

impl Default for MpsBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn map_mps_err(e: MpsError) -> BackendError {
    match e {
        MpsError::QubitOutOfRange { qubit, num_qubits } => {
            BackendError::QubitOutOfRange { qubit, num_qubits }
        }
        MpsError::UnsupportedGate { kind } => BackendError::UnsupportedGate { kind },
        MpsError::ExternalControls { kind } => BackendError::UnsupportedGate { kind },
        MpsError::NonFiniteParam { kind } => BackendError::NonFiniteParam { kind },
        MpsError::NonNearestNeighbor { .. } => BackendError::InvalidState {
            reason: "non-adjacent 2q gate requires a SWAP network (see P3-06)",
        },
        MpsError::DegenerateMeasurement { qubit, probability } => {
            BackendError::DegenerateMeasurement { qubit, probability }
        }
    }
}

impl Backend for MpsBackend {
    type State = MpsState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_QUBITS,
            });
        }
        Ok(MpsState::new(num_qubits as usize, self.max_bond))
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        match gate.gate.arity() {
            1 => {
                let m = crate::gate::matrix_2x2(gate).map_err(map_mps_err)?;
                let q = gate.qubits[0];
                if q as usize >= state.num_qubits() {
                    return Err(BackendError::QubitOutOfRange {
                        qubit: q,
                        num_qubits: state.num_qubits() as u32,
                    });
                }
                state.apply_1q(q as usize, &m);
                Ok(())
            }
            2 => {
                let m = crate::gate::matrix_4x4(gate).map_err(map_mps_err)?;
                for &q in &gate.qubits {
                    if q as usize >= state.num_qubits() {
                        return Err(BackendError::QubitOutOfRange {
                            qubit: q,
                            num_qubits: state.num_qubits() as u32,
                        });
                    }
                }
                state.apply_2q(gate, &m).map_err(map_mps_err)
            }
            _ => Err(BackendError::UnsupportedGate {
                kind: gate.gate.name(),
            }),
        }
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        state
            .measure(qubit as usize, &mut self.rng)
            .map_err(map_mps_err)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        let n = state.num_qubits();
        // One shot packs one bitstring into a u64 (qubit q → bit q), so n ≤ 64.
        if n > 64 {
            return Err(BackendError::TooManyQubits {
                requested: n as u32,
                limit: 64,
            });
        }
        Ok(state.sample(shots, &mut self.rng))
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        state.expectation(pauli).map_err(map_mps_err)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        let mut seen = Vec::new();
        for &q in qubits {
            if seen.contains(&q) {
                return Err(BackendError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        if qubits.len() > MAX_PROB_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: qubits.len() as u32,
                limit: MAX_PROB_QUBITS as u32,
            });
        }
        state.probabilities(qubits).map_err(map_mps_err)
    }
}

#[cfg(test)]
mod tests {
    use super::MpsBackend;
    use aleph_backend::{Backend, BackendError};
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    #[test]
    fn bell_sample_correlated() {
        let mut be = MpsBackend::with_seed(0);
        let mut s = be.allocate(2).unwrap();
        be.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        be.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        for sh in be.sample(&s, 200).unwrap() {
            assert!(sh == 0b00 || sh == 0b11);
        }
    }

    #[test]
    fn rejects_three_qubit_gate() {
        let mut be = MpsBackend::with_seed(0);
        let mut s = be.allocate(3).unwrap();
        let err = be
            .apply_gate(
                &mut s,
                &GateInstance::new(Gate::Toffoli, smallvec![0u32, 1u32, 2u32]),
            )
            .unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedGate { .. }));
    }

    #[test]
    fn rejects_non_adjacent() {
        let mut be = MpsBackend::with_seed(0);
        let mut s = be.allocate(3).unwrap();
        let err = be
            .apply_gate(
                &mut s,
                &GateInstance::new(Gate::Cnot, smallvec![0u32, 2u32]),
            )
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidState { .. }));
    }

    #[test]
    fn probabilities_duplicate_rejected() {
        let mut be = MpsBackend::with_seed(0);
        let s = be.allocate(2).unwrap();
        assert!(matches!(
            be.probabilities(&s, &[0, 0]),
            Err(BackendError::DuplicateQubit { qubit: 0 })
        ));
    }
}
