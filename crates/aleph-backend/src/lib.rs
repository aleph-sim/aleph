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

    #[error("circuit declares {circuit} qubits but state has {state}")]
    QubitCountMismatch { circuit: u32, state: u32 },

    #[error("gate `{kind}` is not supported by this backend")]
    UnsupportedGate { kind: &'static str },

    #[error("backend requires concrete parameters; got symbolic")]
    SymbolicParam,

    #[error("cannot run an empty circuit")]
    EmptyCircuit,

    #[error("measurement of qubit {qubit} on degenerate branch (p = {probability:e})")]
    DegenerateMeasurement { qubit: u32, probability: f64 },

    #[error("requested {requested} qubits exceeds backend limit of {limit}")]
    TooManyQubits { requested: u32, limit: u32 },
}

use aleph_core::{GateInstance, PauliString};
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
/// * `Measure { qubit, .. }` calls `Backend::measure` (and discards the
///   outcome — `run` is a state-producing driver, not a sampling one).
/// * `Reset(q)` is currently rejected as `UnsupportedGate { kind: "reset" }`
///   because the naive backend deals with mid-circuit reset via
///   measure-and-conditional-X, which the IR does not yet express
///   declaratively. P0-13+ may revisit.
/// * `Barrier(_)` is a no-op (semantic-only).
///
/// Returns `EmptyCircuit` only when the circuit declares zero qubits
/// **and** has zero instructions — the truly-degenerate input.
pub fn run<B: Backend>(backend: &mut B, circuit: &Circuit) -> Result<B::State, BackendError> {
    if circuit.num_qubits() == 0 && circuit.is_empty() {
        return Err(BackendError::EmptyCircuit);
    }
    let mut state = backend.allocate(circuit.num_qubits())?;
    for inst in circuit.instructions() {
        match inst {
            aleph_ir::Instruction::Gate(g) => backend.apply_gate(&mut state, g)?,
            aleph_ir::Instruction::Measure { qubit, .. } => {
                let _ = backend.measure(&mut state, *qubit)?;
            }
            aleph_ir::Instruction::Reset(_) => {
                return Err(BackendError::UnsupportedGate { kind: "reset" });
            }
            aleph_ir::Instruction::Barrier(_) => {}
        }
    }
    Ok(state)
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
