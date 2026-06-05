//! `aleph-mps`: Matrix Product State (MPS) backend.
//!
//! Mixed-canonical MPS with fixed bond-dimension χ truncation. Handles 1q
//! gates and nearest-neighbor 2q gates; non-adjacent 2q gates (SWAP networks)
//! are P3-06, error-bounded truncation is P3-05.
//!
//! See `docs/superpowers/specs/2026-06-05-p3-04-mps-basic-chain-design.md`.

mod mps;
mod tensor;
pub use mps::MpsState;

mod gate;

mod backend;
pub use backend::MpsBackend;

/// Errors raised by the MPS state operations, before mapping to the shared
/// `aleph_backend::BackendError`.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MpsError {
    #[error("qubit {qubit} out of range for {num_qubits}-qubit state")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },

    #[error("gate `{kind}` is not supported by the MPS backend")]
    UnsupportedGate { kind: &'static str },

    #[error("2q gate on non-adjacent qubits {a} and {b}; the basic MPS chain only supports nearest-neighbor 2q gates (SWAP networks are P3-06)")]
    NonNearestNeighbor { a: u32, b: u32 },

    #[error("gate `{kind}` carries external controls, which the MPS backend does not support")]
    ExternalControls { kind: &'static str },

    #[error("gate `{kind}` has a non-finite (NaN or infinite) parameter")]
    NonFiniteParam { kind: &'static str },

    #[error("measurement of qubit {qubit} on a degenerate branch (p = {probability:e})")]
    DegenerateMeasurement { qubit: u32, probability: f64 },
}

#[cfg(test)]
mod tests {
    use super::MpsError;
    #[test]
    fn error_messages_render() {
        let e = MpsError::NonNearestNeighbor { a: 0, b: 3 };
        assert!(e.to_string().contains("non-adjacent"));
    }
}
