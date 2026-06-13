//! `aleph_sv::noise` — Monte-Carlo quantum-jump noise driver.
//!
//! Noise is a runtime [`NoiseModel`] config, never IR (ADR 0014). The
//! noiseless `run()` path and the `Backend` trait are untouched; this is a
//! separate `run_noisy` entry point operating on `CpuState`.

mod apply;
mod error;
mod model;

pub use error::{
    amplitude_damping_error, bit_flip_error, depolarizing_error, pauli_error, phase_damping_error,
    phase_flip_error, KrausChannel, PauliChannel, QuantumError, ReadoutError,
};
pub use model::NoiseModel;

use aleph_backend::BackendError;

/// Per-basis-state shot histogram of length `2^num_qubits`. `counts[i]` is the
/// number of shots whose final (readout-perturbed) bitstring was basis state
/// `|i⟩`. The Python layer (P4.6-05) maps this to a bitstring→count dict.
pub type Counts = Vec<u64>;

/// Errors raised by the noise driver, on top of backend failures.
#[derive(Debug, thiserror::Error)]
pub enum NoiseError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// v1 supports terminal measurement only; mid-circuit measure/reset under
    /// noise is a documented v1.1 follow-up (spec §3 "Measurement & reset").
    #[error("mid-circuit {kind} is not supported under noise in v1 (terminal measurement only)")]
    MidCircuit { kind: &'static str },
}
