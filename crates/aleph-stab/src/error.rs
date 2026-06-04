//! Error type for the stabilizer backend.

/// Errors from applying a gate to a [`crate::Tableau`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StabError {
    /// A non-Clifford gate (T, Rz, Toffoli, arbitrary unitary, …) was
    /// dispatched to the stabilizer backend, which can only simulate
    /// Clifford circuits.
    #[error("non-Clifford gate {gate} cannot run on the stabilizer backend")]
    NonClifford { gate: &'static str },

    /// A gate referenced a qubit index ≥ the tableau's qubit count.
    #[error("qubit {qubit} out of range (tableau has {num_qubits} qubits)")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },
}
