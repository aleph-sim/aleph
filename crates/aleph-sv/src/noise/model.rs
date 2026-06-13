//! `NoiseModel` — Aer-style attachment of channels to gates and qubits.

use std::collections::HashMap;

use smallvec::SmallVec;

use super::error::{QuantumError, ReadoutError};

/// Maps `(gate, qubits) → channels` (Aer-style). Consumed by `run_noisy`;
/// never an IR instruction.
#[derive(Debug, Clone, Default)]
pub struct NoiseModel {
    /// Channels attached to a specific (gate-name, qubit-tuple).
    // filled in by Task 6 (errors_for/add_*)
    #[allow(dead_code)]
    specific: HashMap<(String, SmallVec<[u32; 2]>), Vec<QuantumError>>,
    /// Channels attached to a gate name on whichever qubits it acts on.
    // filled in by Task 6 (errors_for/add_*)
    #[allow(dead_code)]
    all_qubit: HashMap<String, Vec<QuantumError>>,
    /// Per-qubit readout error.
    // filled in by Task 6 (errors_for/add_*)
    #[allow(dead_code)]
    readout: HashMap<u32, ReadoutError>,
}

impl NoiseModel {
    /// A model with no errors. Task 7's `run_noisy` guarantees that running
    /// under an empty model reproduces the noiseless distribution.
    pub fn new() -> Self {
        Self::default()
    }
}
