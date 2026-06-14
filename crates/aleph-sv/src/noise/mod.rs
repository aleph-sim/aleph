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
mod model;

pub use error::{
    amplitude_damping_error, bit_flip_error, depolarizing_error, pauli_error, phase_damping_error,
    phase_flip_error, KrausChannel, PauliChannel, QuantumError, ReadoutError,
};
pub use model::NoiseModel;

use std::collections::HashMap;

use aleph_backend::BackendError;

use aleph_backend::Backend;
use aleph_ir::{Circuit, Instruction};
use rayon::prelude::*;

use crate::NaiveSvBackend;

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
    /// An instruction the v1 noise driver does not support (e.g. an
    /// IR-optimization artifact like `TiledBlock`). Run noise on the
    /// un-optimized circuit.
    #[error("instruction {kind} is not supported under noise in v1")]
    Unsupported { kind: &'static str },
}

/// Run `circuit` under `noise_model` for `shots` Monte-Carlo trajectories and
/// return a per-basis-state histogram (length `2^num_qubits`).
///
/// Each shot owns a fresh `CpuState` and a `StdRng` seeded `shot_seed(seed,
/// shot)`, so counts are reproducible regardless of rayon scheduling. v1
/// supports terminal measurement only: mid-circuit `Measure`/`Reset` raise
/// [`NoiseError::MidCircuit`]. `Barrier` is a no-op; `DiagonalPhase` is applied
/// via the backend.
pub fn run_noisy(
    circuit: &Circuit,
    noise_model: &NoiseModel,
    shots: u32,
    seed: u64,
) -> Result<Counts, NoiseError> {
    let n = circuit.num_qubits();
    let dim = 1usize
        .checked_shl(n)
        .ok_or(NoiseError::Backend(BackendError::InvalidState {
            reason: "num_qubits exceeds platform usize::BITS",
        }))?;

    // Resolve attached channels ONCE per gate (not per shot): errors_for
    // allocates a lookup key, so hoisting it out of the hot shot loop avoids
    // shots×gates allocations. The borrowed &QuantumError refs live for the
    // whole run_noisy call and are shared read-only across rayon shots.
    let resolved: Vec<Vec<&QuantumError>> = circuit
        .instructions()
        .iter()
        .map(|inst| match inst {
            Instruction::Gate(gi) => noise_model.errors_for(gi.gate.name(), &gi.qubits),
            _ => Vec::new(),
        })
        .collect();

    // Per-shot trajectory → final readout-perturbed basis-state index.
    let outcomes: Result<Vec<u64>, NoiseError> = (0..shots)
        .into_par_iter()
        .map(|shot| {
            run_one_shot(
                circuit,
                &resolved,
                noise_model.readout_map(),
                n,
                apply::shot_seed(seed, shot as u64),
            )
        })
        .collect();
    let outcomes = outcomes?;

    let mut hist = vec![0u64; dim];
    for idx in outcomes {
        hist[idx as usize] += 1;
    }
    Ok(hist)
}

/// One Monte-Carlo trajectory: apply each gate then its (pre-resolved) attached
/// channels, then sample a terminal Z-basis outcome and apply readout error.
fn run_one_shot(
    circuit: &Circuit,
    resolved: &[Vec<&QuantumError>],
    readout: &HashMap<u32, ReadoutError>,
    n: u32,
    seed: u64,
) -> Result<u64, NoiseError> {
    let mut backend = NaiveSvBackend::with_seed(seed);
    let mut state = backend.allocate(n)?;
    for (inst, errs) in circuit.instructions().iter().zip(resolved.iter()) {
        let errs: &Vec<&QuantumError> = errs;
        match inst {
            Instruction::Gate(gi) => {
                // TODO(perf): apply_gate re-runs the unitarity check on every
                // gate every shot; the circuit is immutable across shots, so a
                // future pre-validated/cached-matrix path could hoist it out of
                // the shot loop. Negligible vs the kernel + Kraus passes for v1.
                backend.apply_gate(&mut state, gi)?;
                for err in errs.iter().copied() {
                    apply::apply_channel(&mut state.amps, n, err, &gi.qubits, &mut backend.rng);
                }
            }
            Instruction::DiagonalPhase(dp) => {
                backend.apply_diagonal_phase(&mut state, dp)?;
            }
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
    // Terminal Z-basis sample: one draw from |amps|² via the backend's rng.
    let idx = backend.sample(&state, 1)?[0];
    Ok(apply::apply_readout(idx, n, readout, &mut backend.rng))
}
