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
///
/// Construction goes through [`GateInstance::new`] or
/// [`GateInstance::controlled`], both of which `debug_assert` that
/// `qubits.len() == gate.arity()`. Fields are `pub` for ergonomics
/// (IR passes pattern-match on them); callers who mutate fields
/// directly are responsible for preserving the arity invariant.
#[derive(Debug, Clone)]
pub struct GateInstance {
    pub gate: Gate,
    pub qubits: SmallVec<[u32; 4]>,
    pub controls: SmallVec<[u32; 2]>,
}

impl GateInstance {
    /// Construct an instance with no generic controls.
    ///
    /// In debug builds, panics if `qubits.len() != gate.arity()`.
    pub fn new(gate: Gate, qubits: impl Into<SmallVec<[u32; 4]>>) -> Self {
        let qubits = qubits.into();
        debug_assert_eq!(
            qubits.len(),
            gate.arity(),
            "GateInstance::new: qubits.len() ({}) != gate.arity() ({}) for {:?}",
            qubits.len(),
            gate.arity(),
            gate
        );
        Self {
            gate,
            qubits,
            controls: SmallVec::new(),
        }
    }

    /// Construct an instance with generic external controls.
    ///
    /// In debug builds, panics if `qubits.len() != gate.arity()`.
    pub fn controlled(
        gate: Gate,
        qubits: impl Into<SmallVec<[u32; 4]>>,
        controls: impl Into<SmallVec<[u32; 2]>>,
    ) -> Self {
        let qubits = qubits.into();
        debug_assert_eq!(
            qubits.len(),
            gate.arity(),
            "GateInstance::controlled: qubits.len() ({}) != gate.arity() ({}) for {:?}",
            qubits.len(),
            gate.arity(),
            gate
        );
        Self {
            gate,
            qubits,
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
        let inst = GateInstance::controlled(Gate::X, smallvec![3u32], smallvec![0u32, 1u32]);
        assert_eq!(inst.qubits.as_slice(), &[3]);
        assert_eq!(inst.controls.as_slice(), &[0, 1]);
    }

    // `debug_assert_eq!` is a no-op in release builds, so the
    // following `#[should_panic]` tests only make sense when
    // `debug_assertions` is on. Without this cfg-gate, `cargo test
    // --release` would report them as `test did not panic as
    // expected` — a false failure that masks real regressions.

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "qubits.len() (1) != gate.arity() (2)")]
    fn new_rejects_arity_mismatch_in_debug() {
        // Cnot is 2-qubit but only one qubit supplied.
        let _ = GateInstance::new(Gate::Cnot, smallvec![0u32]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "qubits.len() (3) != gate.arity() (1)")]
    fn controlled_rejects_arity_mismatch_in_debug() {
        let _ = GateInstance::controlled(Gate::H, smallvec![0u32, 1u32, 2u32], smallvec![]);
    }
}
