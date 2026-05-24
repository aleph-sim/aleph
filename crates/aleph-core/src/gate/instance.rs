//! `GateInstance` — a `Gate` placed on concrete qubit indices, with
//! optional generic external controls.

use smallvec::SmallVec;

use crate::gate::Gate;

/// A gate placed on concrete qubit indices.
///
/// `qubits` holds the gate's target qubits in spec-defined order
/// (e.g. `[control, target]` for `Cnot`). `controls` carries
/// generic external controls applied on top of the underlying gate
/// (e.g. lowered from OpenQASM `ctrl @` modifiers); Phase 0 backends
/// may refuse non-empty `controls`.
#[derive(Debug, Clone)]
pub struct GateInstance {
    pub gate: Gate,
    pub qubits: SmallVec<[u32; 4]>,
    pub controls: SmallVec<[u32; 2]>,
}

impl GateInstance {
    /// Construct an instance with no generic controls.
    pub fn new(gate: Gate, qubits: impl Into<SmallVec<[u32; 4]>>) -> Self {
        Self {
            gate,
            qubits: qubits.into(),
            controls: SmallVec::new(),
        }
    }

    /// Construct an instance with generic external controls.
    pub fn controlled(
        gate: Gate,
        qubits: impl Into<SmallVec<[u32; 4]>>,
        controls: impl Into<SmallVec<[u32; 2]>>,
    ) -> Self {
        Self {
            gate,
            qubits: qubits.into(),
            controls: controls.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn new_has_no_controls() {
        let inst = GateInstance::new(Gate::H, smallvec![0u32]);
        assert_eq!(inst.qubits.as_slice(), &[0]);
        assert!(inst.controls.is_empty());
        assert_eq!(inst.gate, Gate::H);
    }

    #[test]
    fn new_accepts_vec() {
        let inst = GateInstance::new(Gate::Cnot, vec![0u32, 1u32]);
        assert_eq!(inst.qubits.as_slice(), &[0, 1]);
    }

    #[test]
    fn controlled_carries_controls() {
        let inst = GateInstance::controlled(
            Gate::X,
            smallvec![3u32],
            smallvec![0u32, 1u32],
        );
        assert_eq!(inst.qubits.as_slice(), &[3]);
        assert_eq!(inst.controls.as_slice(), &[0, 1]);
    }
}
