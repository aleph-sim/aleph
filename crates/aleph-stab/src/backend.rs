//! `StabilizerBackend`: the `aleph_backend::Backend` implementation over
//! the CHP [`Tableau`]. Clifford circuits run end-to-end through the same
//! driver as the state-vector backends; non-Clifford gates are rejected.

use aleph_backend::{Backend, BackendError};
use aleph_core::{GateInstance, PauliString};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::{apply_gate, StabError, Tableau};

/// Stabilizer (Aaronson-Gottesman) backend. Simulates Clifford circuits
/// in O(n²) memory; rejects non-Clifford gates.
pub struct StabilizerBackend {
    rng: StdRng,
}

impl StabilizerBackend {
    /// Entropy-seeded RNG (for the random-measurement branch).
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Explicit seed; reproducible for a given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Default for StabilizerBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum qubit count `allocate` accepts (generous; stabilizer is
/// O(n²), so this guards only pathological allocations).
const MAX_QUBITS: u32 = 65_536;

fn map_stab_err(e: StabError) -> BackendError {
    match e {
        StabError::NonClifford { gate } => BackendError::UnsupportedGate { kind: gate },
        StabError::QubitOutOfRange { qubit, num_qubits } => {
            BackendError::QubitOutOfRange { qubit, num_qubits }
        }
        StabError::DuplicateQubit { qubit } => BackendError::DuplicateQubit { qubit },
    }
}

impl Backend for StabilizerBackend {
    type State = Tableau;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_QUBITS,
            });
        }
        Ok(Tableau::new(num_qubits as usize))
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        apply_gate(state, gate).map_err(map_stab_err)
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        state
            .measure(qubit as usize, &mut self.rng)
            .map_err(map_stab_err)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        let n = state.num_qubits();
        // One shot packs one bitstring into a u64 (qubit q → bit q, matching
        // the state-vector backends' `1 << qubit` convention), so n ≤ 64.
        if n > 64 {
            return Err(BackendError::TooManyQubits {
                requested: n as u32,
                limit: 64,
            });
        }
        let mut out = Vec::with_capacity(shots as usize);
        for _ in 0..shots {
            let mut t = state.clone();
            let mut bits = 0u64;
            for q in 0..n {
                if t.measure(q, &mut self.rng).map_err(map_stab_err)? {
                    bits |= 1u64 << q;
                }
            }
            out.push(bits);
        }
        Ok(out)
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        let n = state.num_qubits();
        let mut x_p = vec![false; n];
        let mut z_p = vec![false; n];
        for (q, p) in &pauli.terms {
            let qi = *q as usize;
            if qi >= n {
                return Err(BackendError::QubitOutOfRange {
                    qubit: *q,
                    num_qubits: n as u32,
                });
            }
            match p {
                aleph_core::Pauli::I => {}
                aleph_core::Pauli::X => x_p[qi] = true,
                aleph_core::Pauli::Z => z_p[qi] = true,
                aleph_core::Pauli::Y => {
                    x_p[qi] = true;
                    z_p[qi] = true;
                }
            }
        }
        let s = state.pauli_eigenvalue(&x_p, &z_p);
        Ok(pauli.coefficient * s as f64)
    }

    fn probabilities(
        &mut self,
        _state: &Self::State,
        _qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        Err(BackendError::UnsupportedInstruction {
            kind: "probabilities",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::StabilizerBackend;
    use aleph_backend::{Backend, BackendError};
    use aleph_core::{Gate, GateInstance};

    #[test]
    fn bell_apply_and_measure() {
        let mut be = StabilizerBackend::with_seed(0);
        let mut t = be.allocate(2).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32]))
            .unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![0u32, 1u32]))
            .unwrap();
        let b0 = be.measure(&mut t, 0).unwrap();
        let b1 = be.measure(&mut t, 1).unwrap();
        assert_eq!(b0, b1, "Bell correlation through the backend");
    }

    #[test]
    fn rejects_non_clifford() {
        let mut be = StabilizerBackend::with_seed(0);
        let mut t = be.allocate(1).unwrap();
        let err = be
            .apply_gate(&mut t, &GateInstance::new(Gate::T, vec![0u32]))
            .unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedGate { kind } if kind == "T"));
    }

    #[test]
    fn sample_ghz_is_all_zero_or_all_one() {
        // GHZ-4: every shot must be 0000 or 1111 (bits 0..4).
        let mut be = StabilizerBackend::with_seed(42);
        let mut t = be.allocate(4).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32]))
            .unwrap();
        for i in 0..3u32 {
            be.apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![i, i + 1]))
                .unwrap();
        }
        let shots = be.sample(&t, 200).unwrap();
        for s in shots {
            assert!(s == 0b0000 || s == 0b1111, "unexpected GHZ sample {s:04b}");
        }
    }

    #[test]
    fn sample_rejects_over_64_qubits() {
        let mut be = StabilizerBackend::with_seed(0);
        let t = be.allocate(65).unwrap();
        let err = be.sample(&t, 1).unwrap_err();
        assert!(matches!(
            err,
            BackendError::TooManyQubits {
                requested: 65,
                limit: 64
            }
        ));
    }

    #[test]
    fn expectation_value_bell() {
        use aleph_core::{Pauli, PauliString};
        let mut be = StabilizerBackend::with_seed(0);
        let mut t = be.allocate(2).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32]))
            .unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![0u32, 1u32]))
            .unwrap();
        let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        let xx = PauliString::new(1.0, vec![(0, Pauli::X), (1, Pauli::X)]).unwrap();
        let zi = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        assert!((be.expectation_value(&t, &zz).unwrap() - 1.0).abs() < 1e-12);
        assert!((be.expectation_value(&t, &xx).unwrap() - 1.0).abs() < 1e-12);
        assert!(be.expectation_value(&t, &zi).unwrap().abs() < 1e-12);
        // coefficient is honoured.
        let half_zz = PauliString::new(0.5, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        assert!((be.expectation_value(&t, &half_zz).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn expectation_value_qubit_out_of_range() {
        use aleph_core::{Pauli, PauliString};
        let mut be = StabilizerBackend::with_seed(0);
        let t = be.allocate(2).unwrap();
        let p = PauliString::new(1.0, vec![(5, Pauli::Z)]).unwrap();
        let err = be.expectation_value(&t, &p).unwrap_err();
        assert!(matches!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 2
            }
        ));
    }

    #[test]
    fn probabilities_unsupported() {
        let mut be = StabilizerBackend::with_seed(0);
        let t = be.allocate(2).unwrap();
        let err = be.probabilities(&t, &[0]).unwrap_err();
        assert!(
            matches!(err, BackendError::UnsupportedInstruction { kind } if kind == "probabilities")
        );
    }
}
