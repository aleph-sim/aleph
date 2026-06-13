//! `NoiseModel` — Aer-style attachment of channels to gates and qubits.

use std::collections::HashMap;

use smallvec::SmallVec;

use super::error::{QuantumError, ReadoutError};

/// Maps `(gate, qubits) → channels` (Aer-style). Consumed by `run_noisy`;
/// never an IR instruction.
#[derive(Debug, Clone, Default)]
pub struct NoiseModel {
    /// Channels attached to a specific (gate-name, qubit-tuple).
    specific: HashMap<(String, SmallVec<[u32; 2]>), Vec<QuantumError>>,
    /// Channels attached to a gate name on whichever qubits it acts on.
    all_qubit: HashMap<String, Vec<QuantumError>>,
    /// Per-qubit readout error.
    readout: HashMap<u32, ReadoutError>,
}

impl NoiseModel {
    /// A model with no errors. Task 7's `run_noisy` guarantees that running
    /// under an empty model reproduces the noiseless distribution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach `err` to `gate_names` on the specific `qubits` tuple (Aer's
    /// `add_quantum_error`). Applied after the gate, in insertion order.
    pub fn add_quantum_error(&mut self, err: QuantumError, gate_names: &[&str], qubits: &[u32]) {
        let key_qubits: SmallVec<[u32; 2]> = SmallVec::from_slice(qubits);
        for name in gate_names {
            self.specific
                .entry(((*name).to_string(), key_qubits.clone()))
                .or_default()
                .push(err.clone());
        }
    }

    /// Attach `err` to `gate_names` on whichever qubits the gate acts on
    /// (Aer's `add_all_qubit_quantum_error`).
    pub fn add_all_qubit_quantum_error(&mut self, err: QuantumError, gate_names: &[&str]) {
        for name in gate_names {
            self.all_qubit
                .entry((*name).to_string())
                .or_default()
                .push(err.clone());
        }
    }

    /// Attach a per-qubit readout error.
    pub fn add_readout_error(&mut self, err: ReadoutError, qubit: u32) {
        self.readout.insert(qubit, err);
    }

    /// Errors that fire after a gate named `gate_name` acting on `qubits`:
    /// the all-qubit list first, then the qubit-specific list (Aer order).
    pub fn errors_for(&self, gate_name: &str, qubits: &[u32]) -> Vec<&QuantumError> {
        let mut out: Vec<&QuantumError> = Vec::new();
        if let Some(list) = self.all_qubit.get(gate_name) {
            out.extend(list.iter());
        }
        let key = (
            gate_name.to_string(),
            SmallVec::<[u32; 2]>::from_slice(qubits),
        );
        if let Some(list) = self.specific.get(&key) {
            out.extend(list.iter());
        }
        out
    }

    /// The readout error for `qubit`, if any.
    pub fn readout_error(&self, qubit: u32) -> Option<&ReadoutError> {
        self.readout.get(&qubit)
    }

    /// Full readout-error map; the driver calls `.is_empty()` to skip the
    /// per-qubit loop in the common no-readout case, and indexes it per qubit.
    // used by run_noisy in Task 7
    #[allow(dead_code)]
    pub(crate) fn readout_map(&self) -> &HashMap<u32, ReadoutError> {
        &self.readout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::error::{depolarizing_error, ReadoutError};

    #[test]
    fn all_qubit_and_specific_concatenate_in_order() {
        let mut nm = NoiseModel::new();
        nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), &["h"]);
        nm.add_quantum_error(depolarizing_error(0.02, 1), &["h"], &[0]);
        // all-qubit list first, then qubit-specific, per Aer order.
        let errs = nm.errors_for("h", &[0]);
        assert_eq!(errs.len(), 2);
        // On a qubit with no specific attachment, only the all-qubit error fires.
        assert_eq!(nm.errors_for("h", &[1]).len(), 1);
        // A gate with no attachment yields nothing.
        assert_eq!(nm.errors_for("x", &[0]).len(), 0);
        // add_quantum_error fans out across multiple gate names...
        nm.add_quantum_error(depolarizing_error(0.03, 2), &["cx", "cz"], &[0, 1]);
        assert_eq!(nm.errors_for("cx", &[0, 1]).len(), 1);
        assert_eq!(nm.errors_for("cz", &[0, 1]).len(), 1);
        // ...and the qubit-tuple key is order-sensitive ([0,1] != [1,0]).
        assert_eq!(nm.errors_for("cx", &[1, 0]).len(), 0);
    }

    #[test]
    fn readout_round_trips() {
        let mut nm = NoiseModel::new();
        nm.add_readout_error(ReadoutError::new([[0.98, 0.02], [0.03, 0.97]]), 0);
        assert!(nm.readout_error(0).is_some());
        assert!(nm.readout_error(1).is_none());
    }
}
