//! `NaiveSvBackend` — the naive CPU state-vector backend.

use aleph_backend::{Backend, BackendError};
use aleph_core::{AlignedBuf, Complex, GateError, GateInstance, GateMatrix, PauliString};
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
        let mut amps = AlignedBuf::<Complex>::zeroed(dim);
        amps[0] = Complex::new(1.0, 0.0);
        Ok(CpuState { num_qubits, amps })
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let n = state.num_qubits;
        // Arity check before any indexing — the public-fields constructor
        // path on `GateInstance` lets a release-build caller bypass the
        // debug_assert in `GateInstance::new`.
        let expected = gate.gate.arity();
        let got = gate.qubits.len();
        if expected != got {
            return Err(BackendError::ArityMismatch {
                kind: gate.gate.name(),
                expected,
                got,
            });
        }
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
        // Materialise the matrix; route GateError variants to the right BackendError.
        let matrix = gate.gate.matrix().map_err(|e| match e {
            GateError::SymbolicParam => BackendError::SymbolicParam,
            GateError::NonFiniteParam => BackendError::NonFiniteParam {
                kind: gate.gate.name(),
            },
        })?;
        // Unitarity check runs unconditionally as defense-in-depth: intrinsic
        // gates are unitary by construction within FP precision, user-supplied
        // matrices are not, and even intrinsic gates with pathological
        // parameters (e.g. `Rx(1e18)` where argument reduction loses precision)
        // can drift out of unitarity. Cost is constant (≤ 8×8 multiply) per
        // gate, negligible vs. the state-vector kernel itself.
        let deviation = crate::validation::unitarity_deviation(&matrix);
        if !deviation.is_finite() || deviation > aleph_core::AMPLITUDE_TOL {
            return Err(BackendError::NonUnitaryMatrix { deviation });
        }
        match matrix {
            GateMatrix::M2x2(m) => {
                let t = gate.qubits[0];
                crate::kernels::aos::apply_1q(&mut state.amps, t, &gate.controls, &m);
            }
            GateMatrix::M4x4(m) => {
                let t = [gate.qubits[0], gate.qubits[1]];
                crate::kernels::aos::apply_2q(&mut state.amps, t, &gate.controls, &m);
            }
            GateMatrix::M8x8(m) => {
                let t = [gate.qubits[0], gate.qubits[1], gate.qubits[2]];
                crate::kernels::aos::apply_3q(&mut state.amps, t, &gate.controls, &m);
            }
        }
        Ok(())
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        crate::measure::measure_impl(&mut self.rng, state, qubit)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        crate::measure::sample_impl(&mut self.rng, state, shots)
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        crate::measure::expectation_value_impl(state, pauli)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        crate::measure::probabilities_impl(state, qubits)
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
    fn sample_zero_state_only_returns_zero() {
        let mut b = NaiveSvBackend::with_seed(7);
        let s = b.allocate(3).unwrap();
        let shots = b.sample(&s, 100).unwrap();
        assert_eq!(shots.len(), 100);
        assert!(shots.iter().all(|&v| v == 0));
    }

    #[test]
    fn sample_bell_state_only_returns_00_or_11() {
        let mut b = NaiveSvBackend::with_seed(7);
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
    fn sample_zero_shots_returns_empty() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        assert!(b.sample(&s, 0).unwrap().is_empty());
    }

    #[test]
    fn probabilities_zero_state() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        assert_eq!(
            b.probabilities(&s, &[0, 1]).unwrap(),
            vec![1.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn probabilities_plus_state_uniform_marginal() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let p = b.probabilities(&s, &[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn probabilities_empty_subset_is_one() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(3).unwrap();
        assert_eq!(b.probabilities(&s, &[]).unwrap(), vec![1.0]);
    }

    #[test]
    fn probabilities_duplicate_qubit_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        let err = b.probabilities(&s, &[0, 0]).unwrap_err();
        assert_eq!(err, BackendError::DuplicateQubit { qubit: 0 });
    }

    #[test]
    fn probabilities_out_of_range_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        let err = b.probabilities(&s, &[5]).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 2,
            }
        );
    }

    use aleph_core::{Pauli, PauliString};

    #[test]
    fn expectation_z_on_zero_is_plus_one() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        let ev = b.expectation_value(&s, &z).unwrap();
        assert!((ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_x_on_plus_is_plus_one() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        let ev = b.expectation_value(&s, &x).unwrap();
        assert!((ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_x_on_zero_is_zero() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        let ev = b.expectation_value(&s, &x).unwrap();
        assert!(ev.abs() < 1e-12);
    }

    #[test]
    fn expectation_z_on_one_is_minus_one() {
        // X takes |0⟩ to |1⟩, and ⟨1|Z|1⟩ = -1.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![0u32]))
            .unwrap();
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        let ev = b.expectation_value(&s, &z).unwrap();
        assert!((ev - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn expectation_zz_sign_table() {
        // ⟨ψ|Z⊗Z|ψ⟩ on each computational basis state of 2 qubits.
        // qubit indexing matches sample/probabilities: q0 = LSB.
        let cases: &[(u32, f64)] = &[
            (0b00, 1.0),  // |00⟩ → (+1)(+1) = +1
            (0b01, -1.0), // |01⟩ → (-1)(+1) = -1
            (0b10, -1.0), // |10⟩ → (+1)(-1) = -1
            (0b11, 1.0),  // |11⟩ → (-1)(-1) = +1
        ];
        for &(basis, expected) in cases {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(2).unwrap();
            if basis & 0b01 != 0 {
                b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![0u32]))
                    .unwrap();
            }
            if basis & 0b10 != 0 {
                b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![1u32]))
                    .unwrap();
            }
            let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
            let ev = b.expectation_value(&s, &zz).unwrap();
            assert!(
                (ev - expected).abs() < 1e-12,
                "basis {basis:02b}: got {ev}, want {expected}"
            );
        }
    }

    #[test]
    fn expectation_y_on_zero_is_zero() {
        // Mixed-Pauli fallthrough: ⟨0|Y|0⟩ = 0 (Y has 0 diagonal in
        // the computational basis). Exercises the non-Z slow path.
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let y = PauliString::new(1.0, vec![(0, Pauli::Y)]).unwrap();
        let ev = b.expectation_value(&s, &y).unwrap();
        assert!(ev.abs() < 1e-12, "got {ev}");
    }

    #[test]
    fn expectation_z_chain_on_ghz_is_plus_one() {
        // |GHZ_n⟩ = (|0…0⟩ + |1…1⟩)/√2. ⟨GHZ|Z⊗…⊗Z|GHZ⟩ = +1 for
        // even n (popcount(0) and popcount(2^n−1) parities sum to even).
        // Use n = 4 so the answer is unambiguously +1.
        let mut b = NaiveSvBackend::with_seed(0);
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
        assert!((ev - 1.0).abs() < 1e-12, "got {ev}");
    }

    #[test]
    fn expectation_x_on_minus_is_minus_one() {
        // |−⟩ = HX|0⟩; ⟨−|X|−⟩ = -1.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![0u32]))
            .unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        let ev = b.expectation_value(&s, &x).unwrap();
        assert!((ev - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn expectation_identity_string_is_norm() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        let ev = b
            .expectation_value(&s, &PauliString::identity(2.5))
            .unwrap();
        assert!((ev - 2.5).abs() < 1e-12);
    }

    #[test]
    fn expectation_out_of_range_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let p = PauliString::new(1.0, vec![(5, Pauli::Z)]).unwrap();
        let err = b.expectation_value(&s, &p).unwrap_err();
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

    #[test]
    fn apply_gate_arity_mismatch_rejected() {
        // Direct field-literal construction bypasses `GateInstance::new`'s
        // debug_assert; the backend must reject in release mode.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        let bad = GateInstance {
            gate: Gate::Cnot,        // arity 2
            qubits: smallvec![0u32], // only 1 qubit supplied
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
    fn apply_gate_non_finite_param_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let bad = GateInstance::new(Gate::Rx(f64::NAN.into()), smallvec![0u32]);
        let err = b.apply_gate(&mut s, &bad).unwrap_err();
        assert_eq!(err, BackendError::NonFiniteParam { kind: "Rx" });
    }

    #[test]
    fn apply_gate_non_unitary_user_matrix_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        // 2× identity — not unitary.
        let two = Complex::new(2.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let mat = Gate::Unitary1q(Box::new([[two, zero], [zero, two]]));
        let bad = GateInstance::new(mat, smallvec![0u32]);
        let err = b.apply_gate(&mut s, &bad).unwrap_err();
        assert!(
            matches!(err, BackendError::NonUnitaryMatrix { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn apply_gate_legitimate_user_matrix_accepted() {
        // Identity 2×2 IS unitary.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let mat = Gate::Unitary1q(Box::new([[one, zero], [zero, one]]));
        let good = GateInstance::new(mat, smallvec![0u32]);
        b.apply_gate(&mut s, &good).unwrap();
    }

    #[test]
    fn expectation_value_rejects_mutated_duplicates() {
        // PauliString fields are pub — caller can construct invalid state.
        use aleph_core::Pauli as P;
        use aleph_core::PauliString as PS;
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let bad = PS {
            coefficient: 1.0,
            terms: vec![(0, P::X), (0, P::Z)],
        };
        let err = b.expectation_value(&s, &bad).unwrap_err();
        assert_eq!(err, BackendError::DuplicateQubit { qubit: 0 });
    }

    #[test]
    fn expectation_value_rejects_non_finite_coefficient() {
        use aleph_core::PauliString as PS;
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let bad = PS {
            coefficient: f64::NAN,
            terms: vec![],
        };
        let err = b.expectation_value(&s, &bad).unwrap_err();
        assert!(matches!(err, BackendError::InvalidPauliString { .. }));
    }

    #[test]
    fn measure_rejects_nan_amplitude_state() {
        // Put NaN in the bit-set branch (qubit 0 bit set) so p1 sums to NaN.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        s.amps[1] = Complex::new(f64::NAN, 0.0);
        let err = b.measure(&mut s, 0).unwrap_err();
        assert!(matches!(err, BackendError::InvalidState { .. }));
    }

    #[test]
    fn sample_rejects_nan_amplitude_state() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        s.amps[0] = Complex::new(f64::NAN, 0.0);
        let err = b.sample(&s, 10).unwrap_err();
        assert!(matches!(err, BackendError::InvalidState { .. }));
    }

    #[test]
    fn sample_rejects_unnormalised_state() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        // Total norm² = 4, well outside drift budget.
        s.amps[0] = Complex::new(2.0, 0.0);
        s.amps[1] = Complex::new(0.0, 0.0);
        let err = b.sample(&s, 10).unwrap_err();
        assert!(matches!(err, BackendError::InvalidState { .. }));
    }

    #[test]
    fn unitarity_check_rejects_nan_user_matrix() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let nan = Complex::new(f64::NAN, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        // NaN in one entry — must NOT pass the unitarity guard.
        let mat = Gate::Unitary1q(Box::new([[nan, zero], [zero, one]]));
        let bad = GateInstance::new(mat, smallvec![0u32]);
        let err = b.apply_gate(&mut s, &bad).unwrap_err();
        assert!(matches!(err, BackendError::NonUnitaryMatrix { .. }));
    }

    #[test]
    fn measure_rejects_nan_in_bit_clear_branch() {
        // Round-3 finding: round-2's p1-only guard left NaN in the
        // bit-clear branch unchecked. validate_state now walks every amp.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        // Bit 0 of index 0 is clear. Put NaN there; measure on qubit 0
        // should reject before consuming RNG or collapsing state.
        s.amps[0] = Complex::new(f64::NAN, 0.0);
        s.amps[1] = Complex::new(0.5, 0.0);
        s.amps[2] = Complex::new(0.5, 0.0);
        s.amps[3] = Complex::new(0.5, 0.0);
        let err = b.measure(&mut s, 0).unwrap_err();
        assert!(
            matches!(err, BackendError::InvalidState { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn measure_rejects_unnormalised_state() {
        // [0.9, 0.9] has total norm² = 1.62; sample rejects it, so
        // measure must reject it too (symmetry — round-3 finding #3).
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        s.amps[0] = Complex::new(0.9, 0.0);
        s.amps[1] = Complex::new(0.9, 0.0);
        let err = b.measure(&mut s, 0).unwrap_err();
        assert!(
            matches!(err, BackendError::InvalidState { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn probabilities_rejects_nan_state() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        s.amps[1] = Complex::new(f64::NAN, 0.0);
        let err = b.probabilities(&s, &[0]).unwrap_err();
        assert!(
            matches!(err, BackendError::InvalidState { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn expectation_value_rejects_nan_state() {
        use aleph_core::{Pauli, PauliString};
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        s.amps[1] = Complex::new(f64::NAN, 0.0);
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        let err = b.expectation_value(&s, &z).unwrap_err();
        assert!(
            matches!(err, BackendError::InvalidState { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn sample_rejects_state_with_mismatched_amps_len() {
        // P0-11 hardening: validate_state must reject states whose
        // amps.len() is not 2^num_qubits, otherwise AliasTable::draw's
        // power-of-two precondition could silently bias release builds.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap(); // num_qubits=2 → expects 4 amps
                                            // Replace the buffer with a length-5 one to break the pow2 invariant;
                                            // AlignedBuf is fixed-size so we reconstruct rather than push.
        let zero = Complex::new(0.0, 0.0);
        s.amps = AlignedBuf::from_slice(&[zero, zero, zero, zero, zero]); // 5 amps; not pow2
        let err = b.sample(&s, 10).unwrap_err();
        assert!(
            matches!(err, BackendError::InvalidState { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn measure_clamps_p_against_drifted_norm() {
        // Construct a state whose total norm² is slightly > 1.0 to mimic
        // FP drift. measure must NOT produce NaN amplitudes.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        // Hand-build amps: ~|0⟩ with a tiny over-normalisation.
        s.amps[0] = Complex::new(1.0 + 1e-15, 0.0);
        s.amps[1] = Complex::new(0.0, 0.0);
        let outcome = b.measure(&mut s, 0).unwrap();
        assert!(!outcome);
        assert!(s
            .amplitudes()
            .iter()
            .all(|a| a.re.is_finite() && a.im.is_finite()));
    }

    #[test]
    fn allocated_state_is_cache_line_aligned() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(20).unwrap();
        assert_eq!(s.amplitudes().as_ptr() as usize % aleph_core::CACHE_LINE, 0);
    }

    use aleph_test::gate::arb_1q_gate;
    use proptest::prelude::*;

    fn run_program(ops: &[(Gate, u32)], n: u32) -> CpuState {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(n).unwrap();
        for (g, q) in ops {
            b.apply_gate(&mut s, &GateInstance::new(g.clone(), smallvec![*q]))
                .unwrap();
        }
        s
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn normalisation_invariant(
            ops in proptest::collection::vec(
                (arb_1q_gate(), 0u32..4u32),
                0..30,
            )
        ) {
            let s = run_program(&ops, 4);
            let total: f64 = s.amplitudes().iter().map(|a| a.norm_sqr()).sum();
            prop_assert!((total - 1.0).abs() < 1e-10, "norm² = {total}");
        }

        #[test]
        fn h_is_involution(q in 0u32..4u32) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(4).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            prop_assert!((s.amplitudes()[0].re - 1.0).abs() < 1e-12);
            for a in &s.amplitudes()[1..] {
                prop_assert!(a.norm() < 1e-12);
            }
        }

        #[test]
        fn cnot_is_involution(c in 0u32..3u32, t in 0u32..3u32) {
            prop_assume!(c != t);
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![c])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![c, t])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![c, t])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn rx_then_rx_negative_returns_identity(q in 0u32..3u32, theta in -3.0_f64..3.0_f64) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Rx(theta.into()), smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Rx((-theta).into()), smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-10);
            }
        }

        #[test]
        fn ry_then_ry_negative_returns_identity(q in 0u32..3u32, theta in -3.0_f64..3.0_f64) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Ry(theta.into()), smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Ry((-theta).into()), smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-10);
            }
        }

        #[test]
        fn rz_then_rz_negative_returns_identity(q in 0u32..3u32, theta in -3.0_f64..3.0_f64) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Rz(theta.into()), smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Rz((-theta).into()), smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-10);
            }
        }

        #[test]
        fn s_then_sdg_returns_identity(q in 0u32..3u32) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::S, smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Sdg, smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn t_then_tdg_returns_identity(q in 0u32..3u32) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::T, smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Tdg, smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn iswap_then_iswapdg_returns_identity(a in 0u32..3u32, b_idx in 0u32..3u32) {
            prop_assume!(a != b_idx);
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![a])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Iswap, smallvec![a, b_idx])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::IswapDg, smallvec![a, b_idx])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn x_squared_is_identity(q in 0u32..3u32) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn y_squared_is_identity(q in 0u32..3u32) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Y, smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Y, smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn z_squared_is_identity(q in 0u32..3u32) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Z, smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Z, smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn swap_is_involution(a in 0u32..3u32, b_idx in 0u32..3u32) {
            prop_assume!(a != b_idx);
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![a])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Swap, smallvec![a, b_idx])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Swap, smallvec![a, b_idx])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn intrinsic_cnot_matches_external_control(
            c in 0u32..3u32,
            t in 0u32..3u32,
            preamble_q in 0u32..3u32,
        ) {
            prop_assume!(c != t);
            let mut b1 = NaiveSvBackend::with_seed(0);
            let mut s1 = b1.allocate(3).unwrap();
            b1.apply_gate(&mut s1, &GateInstance::new(Gate::H, smallvec![preamble_q])).unwrap();
            b1.apply_gate(&mut s1, &GateInstance::new(Gate::Cnot, smallvec![c, t])).unwrap();
            let mut b2 = NaiveSvBackend::with_seed(0);
            let mut s2 = b2.allocate(3).unwrap();
            b2.apply_gate(&mut s2, &GateInstance::new(Gate::H, smallvec![preamble_q])).unwrap();
            b2.apply_gate(
                &mut s2,
                &GateInstance::controlled(Gate::X, smallvec![t], smallvec![c]),
            ).unwrap();
            for (a, b) in s1.amplitudes().iter().zip(s2.amplitudes().iter()) {
                prop_assert!((a - b).norm() < 1e-12);
            }
        }

        /// Diagonal 1q gates (Z, S, Sdg, T, Tdg, Rz(θ)) only rotate
        /// phases; they MUST leave |aᵢ| invariant for every basis
        /// state.  The existing reversibility proptests verify a
        /// stronger property — but this targets magnitudes directly
        /// and would surface a single-direction bug (e.g. a Z kernel
        /// that accidentally scales an amplitude).  BACKLOG P0-05
        /// "diagonal gates leave magnitudes unchanged" AC.
        #[test]
        fn diagonal_gate_preserves_magnitudes(
            op in aleph_test::gate::arb_diagonal_1q_gate(),
            q in 0u32..4u32,
        ) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(4).unwrap();
            // Non-trivial preamble so the state isn't |0…0⟩.
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0, 1])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![2])).unwrap();
            let before: Vec<f64> = s.amplitudes().iter().map(|a| a.norm()).collect();
            b.apply_gate(&mut s, &GateInstance::new(op, smallvec![q])).unwrap();
            let after: Vec<f64> = s.amplitudes().iter().map(|a| a.norm()).collect();
            for (b_mag, a_mag) in before.iter().zip(after.iter()) {
                prop_assert!((b_mag - a_mag).abs() < 1e-12, "|a| changed: {b_mag} → {a_mag}");
            }
        }

        /// P1-06: diagonal-1q gates routed through `apply_gate` →
        /// `kernels::aos::apply_1q` → diagonal fast path must preserve
        /// state-vector norm to 1 ± 1e-12.  *Weaker* than the
        /// component-wise magnitude check above — a kernel that
        /// swaps two equal-magnitude amplitudes would pass this test
        /// but fail the magnitude one.  Kept as a defense-in-depth
        /// check that catches a kernel that breaks unitarity globally
        /// while leaving local magnitudes intact (e.g. a bug in the
        /// AVX-512 store address calculation that overlaps two amp
        /// indices, drifting total probability without changing
        /// per-amp norms).
        #[test]
        fn p1_06_diagonal_fast_path_preserves_norm(
            op in aleph_test::gate::arb_diagonal_1q_gate(),
            q in 0u32..4u32,
        ) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(4).unwrap();
            // Non-trivial preamble.
            for qq in 0..4u32 {
                b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![qq])).unwrap();
            }
            b.apply_gate(&mut s, &GateInstance::new(op, smallvec![q])).unwrap();
            let norm_sq: f64 = s.amplitudes().iter().map(|a| a.norm_sqr()).sum();
            prop_assert!(
                (norm_sq - 1.0).abs() < 1e-12,
                "norm drifted to {norm_sq}",
            );
        }
    }
}
