//! `aleph-backend`: the `Backend` trait, shared `BackendError`, and a
//! `run<B>` driver. Backend implementations live in `aleph-sv` (naive
//! CPU state vector), `aleph-mps`, `aleph-stab`, etc.
//!
//! See `docs/superpowers/specs/2026-05-24-p0-09-backend-naive-sv-design.md`.

/// Errors common to every backend.
///
/// Backends share one concrete error type rather than an associated
/// `type Error` so that the `run<B>` driver and downstream code (CLI,
/// Python bindings) don't have to be generic over an open-ended error.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BackendError {
    #[error("qubit {qubit} out of range for {num_qubits}-qubit state")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },

    #[error("duplicate qubit {qubit} in gate or query")]
    DuplicateQubit { qubit: u32 },

    #[error("gate `{kind}` expects {expected} qubits, got {got}")]
    ArityMismatch {
        kind: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("gate `{kind}` is not supported by this backend")]
    UnsupportedGate { kind: &'static str },

    #[error("IR instruction `{kind}` is not supported by this backend")]
    UnsupportedInstruction { kind: &'static str },

    #[error("backend requires concrete parameters; got symbolic")]
    SymbolicParam,

    #[error("gate `{kind}` has a non-finite (NaN or infinite) parameter")]
    NonFiniteParam { kind: &'static str },

    #[error("user-supplied matrix is not unitary (max deviation = {deviation:e})")]
    NonUnitaryMatrix { deviation: f64 },

    #[error("cannot run an empty circuit")]
    EmptyCircuit,

    #[error("measurement of qubit {qubit} on degenerate branch (p = {probability:e})")]
    DegenerateMeasurement { qubit: u32, probability: f64 },

    #[error("requested {requested} qubits exceeds backend limit of {limit}")]
    TooManyQubits { requested: u32, limit: u32 },

    #[error("Pauli string violates its invariants: {reason}")]
    InvalidPauliString { reason: &'static str },

    #[error("backend state is invalid: {reason}")]
    InvalidState { reason: &'static str },

    #[error("optimization pipeline failed: {0}")]
    Optimization(#[from] PassError),
}

use aleph_core::{GateInstance, PauliString};
use aleph_ir::passes::PassError;
use aleph_ir::Circuit;

/// A simulation backend.
///
/// Backends own no state vector; they construct and return one through
/// `allocate`, then mutate it in place via `apply_gate` / `measure`.
/// Query methods (`sample`, `expectation_value`, `probabilities`) take
/// `&Self::State` and do not mutate the state.
pub trait Backend {
    /// Backend-specific representation (state vector, MPS tensors,
    /// stabilizer tableau, …).
    type State;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError>;

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError>;

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError>;

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError>;

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError>;

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError>;
}

/// Run `circuit` on `backend`, returning the final backend state.
///
/// Iterates instructions in order, dispatching `Instruction::Gate` to
/// `Backend::apply_gate`. Non-gate instructions are handled inline:
///
/// * `Measure { qubit, .. }` calls `Backend::measure` and **discards**
///   the outcome. Use [`run_with_outcomes`] if you need to keep the
///   measurement record (e.g. for shot-based oracle comparison).
/// * `Reset(q)` is rejected as
///   [`BackendError::UnsupportedInstruction`] `{ kind: "reset" }`
///   because the naive backend doesn't yet express mid-circuit reset
///   declaratively. P0-13+ may revisit.
/// * `Barrier(_)` is a no-op (semantic-only).
///
/// Returns [`BackendError::EmptyCircuit`] only when the circuit declares
/// zero qubits **and** has zero instructions — the truly-degenerate
/// input.
pub fn run<B: Backend>(backend: &mut B, circuit: &Circuit) -> Result<B::State, BackendError> {
    let (state, _outcomes) = run_with_outcomes(backend, circuit)?;
    Ok(state)
}

/// One recorded measurement outcome from `run_with_outcomes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementRecord {
    /// Index of the `Instruction::Measure` within `circuit.instructions()`.
    pub instruction_index: usize,
    pub qubit: u32,
    pub clbit: u32,
    pub outcome: bool,
}

/// Run `circuit` on `backend` AND return every measurement outcome.
///
/// Same semantics as [`run`], but preserves the bool returned by each
/// `Backend::measure` call. Use this driver when downstream code needs
/// to inspect mid-circuit outcomes (postselection, oracle comparison
/// against shot-based references like Qiskit Aer's `meas_level=2`).
///
/// **Ordering contract:** the returned `Vec<MeasurementRecord>` is in
/// the same order as the corresponding `Instruction::Measure` entries
/// in `circuit.instructions()`. In particular,
/// `outcomes[i].instruction_index` is strictly increasing and equals
/// the position of the i-th measurement instruction within the
/// circuit. Downstream consumers (oracle harness, postselection logic)
/// may rely on this ordering.
pub fn run_with_outcomes<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<(B::State, Vec<MeasurementRecord>), BackendError> {
    if circuit.num_qubits() == 0 && circuit.is_empty() {
        return Err(BackendError::EmptyCircuit);
    }
    let mut state = backend.allocate(circuit.num_qubits())?;
    let mut outcomes = Vec::new();
    for (idx, inst) in circuit.instructions().iter().enumerate() {
        match inst {
            aleph_ir::Instruction::Gate(g) => backend.apply_gate(&mut state, g)?,
            aleph_ir::Instruction::Measure { qubit, clbit } => {
                let outcome = backend.measure(&mut state, *qubit)?;
                outcomes.push(MeasurementRecord {
                    instruction_index: idx,
                    qubit: *qubit,
                    clbit: *clbit,
                    outcome,
                });
            }
            aleph_ir::Instruction::Reset(_) => {
                return Err(BackendError::UnsupportedInstruction { kind: "reset" });
            }
            aleph_ir::Instruction::Barrier(_) => {}
        }
    }
    Ok((state, outcomes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_render() {
        let e = BackendError::QubitOutOfRange {
            qubit: 7,
            num_qubits: 3,
        };
        assert_eq!(e.to_string(), "qubit 7 out of range for 3-qubit state");

        let e = BackendError::DegenerateMeasurement {
            qubit: 0,
            probability: 1e-301,
        };
        assert!(e.to_string().contains("p = 1e-301"));
    }
}
