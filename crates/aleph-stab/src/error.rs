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

    /// A 2-qubit gate referenced the same qubit for both operands (e.g.
    /// `CNOT(a, a)`); the column-major kernels require two distinct columns.
    #[error("2-qubit gate referenced qubit {qubit} twice")]
    DuplicateQubit { qubit: u32 },

    /// A circuit instruction the stabilizer engine does not handle (e.g. a
    /// post-optimization `DiagonalPhase`/`TiledBlock` fused block). QEC
    /// circuits are unoptimized gate lists, so these never appear there.
    #[error("unsupported instruction for the stabilizer engine: {what}")]
    Unsupported { what: &'static str },
}
