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
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let n = state.num_qubits;
        // Bounds + duplicate checks across qubits ∪ controls.
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
        // Materialise the matrix; map symbolic-param errors to BackendError.
        let matrix = gate
            .gate
            .matrix()
            .map_err(|_| BackendError::SymbolicParam)?;
        match matrix {
            aleph_core::GateMatrix::M2x2(m) => {
                let t = gate.qubits[0];
                crate::kernels::apply_1q(&mut state.amps, t, &gate.controls, &m);
            }
            aleph_core::GateMatrix::M4x4(m) => {
                let t = [gate.qubits[0], gate.qubits[1]];
                crate::kernels::apply_2q(&mut state.amps, t, &gate.controls, &m);
            }
            aleph_core::GateMatrix::M8x8(m) => {
                let t = [gate.qubits[0], gate.qubits[1], gate.qubits[2]];
                crate::kernels::apply_3q(&mut state.amps, t, &gate.controls, &m);
            }
        }
        Ok(())
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        crate::measure::measure_impl(&mut self.rng, state, qubit)
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

    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn apply_gate_x_on_q0() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let gate = GateInstance::new(Gate::X, smallvec![0u32]);
        b.apply_gate(&mut s, &gate).unwrap();
        assert_eq!(s.amplitudes()[0], Complex::new(0.0, 0.0));
        assert_eq!(s.amplitudes()[1], Complex::new(1.0, 0.0));
    }

    #[test]
    fn apply_gate_cnot_creates_bell() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        // H on q0, then CX(q0 → q1).
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
        let a = s.amplitudes();
        assert!((a[0].re - inv_s2).abs() < 1e-12);
        assert!((a[3].re - inv_s2).abs() < 1e-12);
        assert!(a[1].norm_sqr() < 1e-24);
        assert!(a[2].norm_sqr() < 1e-24);
    }

    #[test]
    fn apply_gate_external_control_matches_intrinsic_cnot() {
        // Path A: intrinsic CX.
        let mut b1 = NaiveSvBackend::with_seed(0);
        let mut s1 = b1.allocate(2).unwrap();
        b1.apply_gate(&mut s1, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b1.apply_gate(
            &mut s1,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        // Path B: X on q1 with external control = q0.
        let mut b2 = NaiveSvBackend::with_seed(0);
        let mut s2 = b2.allocate(2).unwrap();
        b2.apply_gate(&mut s2, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b2.apply_gate(
            &mut s2,
            &GateInstance::controlled(Gate::X, smallvec![1u32], smallvec![0u32]),
        )
        .unwrap();
        for (a, b) in s1.amplitudes().iter().zip(s2.amplitudes().iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn apply_gate_out_of_range() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let gate = GateInstance::new(Gate::X, smallvec![5u32]);
        let err = b.apply_gate(&mut s, &gate).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 1,
            }
        );
    }

    #[test]
    fn measure_zero_state_returns_false() {
        let mut b = NaiveSvBackend::with_seed(42);
        let mut s = b.allocate(2).unwrap();
        let outcome = b.measure(&mut s, 0).unwrap();
        assert!(!outcome);
        assert_eq!(s.amplitudes()[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn measure_plus_state_collapses_to_basis() {
        let mut b = NaiveSvBackend::with_seed(123);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let outcome = b.measure(&mut s, 0).unwrap();
        let a = s.amplitudes();
        if outcome {
            assert!((a[1].norm() - 1.0).abs() < 1e-12);
            assert!(a[0].norm() < 1e-12);
        } else {
            assert!((a[0].norm() - 1.0).abs() < 1e-12);
            assert!(a[1].norm() < 1e-12);
        }
    }

    #[test]
    fn measure_qubit_out_of_range() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let err = b.measure(&mut s, 5).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 1,
            }
        );
    }

    #[test]
    fn apply_gate_duplicate_qubit_via_controls() {
        // `GateInstance::controlled` panics on duplicate qubits in debug
        // builds, so build the bad instance directly via the public
        // fields to exercise the backend's release-build safety net.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        let gate = GateInstance {
            gate: Gate::X,
            qubits: smallvec![0u32],
            controls: smallvec![0u32],
        };
        let err = b.apply_gate(&mut s, &gate).unwrap_err();
        assert_eq!(err, BackendError::DuplicateQubit { qubit: 0 });
    }
}
