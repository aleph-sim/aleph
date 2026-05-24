//! Error type for circuit construction.

/// Errors returned by `Circuit` builder methods. All variants are
/// recoverable — `Circuit::new` cannot fail.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CircuitError {
    #[error("qubit {qubit} out of range (circuit has {num_qubits} qubits)")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },

    #[error("clbit {clbit} out of range (circuit has {num_clbits} clbits)")]
    ClbitOutOfRange { clbit: u32, num_clbits: u32 },

    #[error("duplicate qubit index {qubit} in instruction")]
    DuplicateQubit { qubit: u32 },

    #[error("gate {gate} has arity {expected} but {got} qubits supplied")]
    ArityMismatch {
        gate: &'static str,
        expected: usize,
        got: usize,
    },
}
