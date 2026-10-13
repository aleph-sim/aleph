//! `aleph_sv::noise` — Monte-Carlo quantum-jump noise driver.
//!
//! Noise is a runtime [`NoiseModel`] config, never IR (ADR 0014). The
//! noiseless `run()` path and the `Backend` trait are untouched; this is a
//! separate `run_noisy` entry point operating on `CpuState`.
//!
//! # Example
//! ```no_run
//! use aleph_sv::noise::{depolarizing_error, run_noisy, NoiseModel};
//! # let circuit = aleph_parser::parse(
//! #     "OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; h q[0];").unwrap();
//! let mut nm = NoiseModel::new();
//! // Attach by aleph's internal Gate::name() — "H", not the QASM "h".
//! // (The P4.6-05 Python API maps Aer mnemonics like "h"/"cx" to these.)
//! nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), &["H"]);
//! let counts = run_noisy(&circuit, &nm, 100_000, 7).unwrap();
//! # assert_eq!(counts.len(), 2);
//! ```

mod apply;
mod error;
mod execution;
mod model;
pub use execution::{
    plan_noisy, run_noisy_sparse, run_noisy_with_options, NoiseOptions, NoisePlan, SparseCounts,
};

pub use error::{
    amplitude_damping_error, bit_flip_error, depolarizing_error, pauli_error, phase_damping_error,
    phase_flip_error, KrausChannel, PauliChannel, QuantumError, ReadoutError,
};
pub use model::NoiseModel;

use std::collections::HashMap;

use aleph_backend::BackendError;

use aleph_backend::Backend;
use aleph_ir::{Circuit, Instruction};

use crate::NaiveSvBackend;

/// Per-basis-state shot histogram of length `2^num_qubits`. `counts[i]` is the
/// number of shots whose final (readout-perturbed) bitstring was basis state
/// `|i⟩`. The Python layer (P4.6-05) maps this to a bitstring→count dict.
pub type Counts = Vec<u64>;

/// Errors raised by the noise driver, on top of backend failures.
#[derive(Debug, thiserror::Error)]
pub enum NoiseError {
    /// Requested output and one trajectory cannot fit the declared budget.
    #[error("noisy run requires at least {required} buffer bytes, budget is {budget}; use sparse counts or increase NoiseOptions::memory_budget_bytes")]
    MemoryBudget { required: usize, budget: usize },
    /// Invalid budget/concurrency or overflowing allocation estimate.
    #[error("invalid noisy execution configuration: {reason}")]
    Configuration { reason: &'static str },
    /// Malformed channel attached to an executed gate.
    #[error("invalid noise channel: {reason}")]
    InvalidChannel { reason: &'static str },
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// v1 supports terminal measurement only; mid-circuit measure/reset under
    /// noise is a documented v1.1 follow-up (spec §3 "Measurement & reset").
    #[error("mid-circuit {kind} is not supported under noise in v1 (terminal measurement only)")]
    MidCircuit { kind: &'static str },
    /// An instruction the v1 noise driver does not support (e.g. an
    /// IR-optimization artifact like `TiledBlock`). Run noise on the
    /// un-optimized circuit.
    #[error("instruction {kind} is not supported under noise in v1")]
    Unsupported { kind: &'static str },
}

/// Run noisy trajectories with the default 1 GiB incremental buffer budget.
///
/// Returns the legacy dense histogram of length `2^num_qubits`. Each chunk
/// reuses a state, resetting it before each absolute-index seeded shot.
/// Use [`run_noisy_with_options`] for an explicit budget, or
/// [`run_noisy_sparse`] when a dense result would dominate memory.
/// The budget excludes caller-owned inputs, allocator overhead and Rayon runtime.
pub fn run_noisy(
    circuit: &Circuit,
    noise_model: &NoiseModel,
    shots: u32,
    seed: u64,
) -> Result<Counts, NoiseError> {
    run_noisy_with_options(circuit, noise_model, shots, seed, NoiseOptions::default())
}

/// Execute one trajectory in a reusable state. Gate validation and channel
/// boundaries remain on the original path; no noiseless fusion is introduced.
fn run_trajectory(
    circuit: &Circuit,
    resolved: &[Vec<&QuantumError>],
    readout: &HashMap<u32, ReadoutError>,
    backend: &mut NaiveSvBackend,
    state: &mut crate::CpuState,
    scratch: &mut Vec<f64>,
) -> Result<u64, NoiseError> {
    let n = state.num_qubits;
    for (inst, errs) in circuit.instructions().iter().zip(resolved.iter()) {
        match inst {
            Instruction::Gate(gi) => {
                backend.apply_gate(state, gi)?;
                for err in errs.iter().copied() {
                    apply::apply_channel_with_scratch(
                        &mut state.amps,
                        n,
                        err,
                        &gi.qubits,
                        &mut backend.rng,
                        scratch,
                    );
                }
            }
            Instruction::DiagonalPhase(dp) => backend.apply_diagonal_phase(state, dp)?,
            Instruction::Barrier(_) => {}
            Instruction::Measure { .. } => return Err(NoiseError::MidCircuit { kind: "measure" }),
            Instruction::Reset(_) => return Err(NoiseError::MidCircuit { kind: "reset" }),
            Instruction::TiledBlock(_) => {
                return Err(NoiseError::Unsupported {
                    kind: "tiled-block",
                })
            }
        }
    }
    let idx = backend.sample(state, 1)?[0];
    Ok(apply::apply_readout(idx, n, readout, &mut backend.rng))
}
