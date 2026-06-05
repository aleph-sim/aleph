//! `aleph-mps`: Matrix Product State (MPS) backend.
//!
//! Mixed-canonical MPS with fixed bond-dimension χ truncation. Handles 1q
//! gates and nearest-neighbor 2q gates; non-adjacent 2q gates (SWAP networks)
//! are P3-06, error-bounded truncation is P3-05.
//!
//! See `docs/superpowers/specs/2026-06-05-p3-04-mps-basic-chain-design.md`.
//!
//! Truncation is configurable via [`TruncationPolicy`]: a fixed bond dimension
//! or an error-bounded mode that keeps the discarded weight per bond below `ε`.
//! The truncating SVD uses `faer` (`nalgebra`'s complex SVD and Hermitian
//! eigensolver both return orthonormal-but-incorrect vectors on some complex
//! matrices, which silently corrupted entangled two-site blocks).
//!
//! # Usage
//!
//! ```no_run
//! use aleph_backend::{run, Backend};
//! use aleph_core::{Gate, GateInstance, Param};
//! use aleph_ir::Circuit;
//! use aleph_mps::MpsBackend;
//!
//! // Build a circuit.
//! let mut c = Circuit::new(4, 0);
//! c.add_gate(GateInstance::new(Gate::H, vec![0])).unwrap();
//! c.add_gate(GateInstance::new(Gate::Cnot, vec![0, 1])).unwrap();
//! c.add_gate(GateInstance::new(Gate::Ry(Param::Concrete(0.5)), vec![2])).unwrap();
//!
//! // Run with bond dimension χ = 64 (exact for low-entanglement states).
//! let mut backend = MpsBackend::with_seed(0).with_max_bond(64);
//! let state = run(&mut backend, &c).unwrap();
//!
//! // Reconstruct the full statevector (only feasible for small n).
//! let amps = state.dense_statevector();
//!
//! // Check accumulated truncation error.
//! let err = state.truncation_error();
//!
//! // Sample bitstrings.
//! let shots = backend.sample(&state, 1000).unwrap();
//! ```

mod mps;
mod tensor;
pub use mps::MpsState;
pub use tensor::TruncationPolicy;

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

    #[error("SVD failed to converge during 2q-gate truncation")]
    SvdFailed,
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
