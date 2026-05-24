//! `GateInstance` — a `Gate` placed on concrete qubit indices, with
//! optional generic external controls.

use smallvec::SmallVec;

use crate::gate::Gate;

/// Debug-only check: every index appears at most once across the two
/// slices. Returns `Ok(())` or an `Err(msg)` describing the offending
/// index. Lives behind a separate function so the cold-path assert
/// doesn't bloat the hot constructor.
#[cfg(debug_assertions)]
fn check_qubit_uniqueness(qubits: &[u32], controls: &[u32]) -> Result<(), String> {
    // Small N (≤ 6 for any Phase-0 Gate); a quadratic scan beats a
    // HashSet allocation. Inline both lists into one logical sequence.
    let total = qubits.len() + controls.len();
    for i in 0..total {
        let qi = if i < qubits.len() {
            qubits[i]
        } else {
            controls[i - qubits.len()]
        };
        for j in (i + 1)..total {
            let qj = if j < qubits.len() {
                qubits[j]
            } else {
                controls[j - qubits.len()]
            };
            if qi == qj {
                return Err(format!(
                    "qubit index {qi} appears more than once in qubits={qubits:?} controls={controls:?}"
                ));
            }
        }
    }
    Ok(())
}

/// A gate placed on concrete qubit indices.
///
/// `qubits` holds the gate's target qubits in spec-defined order
/// (e.g. `[control, target]` for `Cnot`). `controls` carries
/// generic external controls applied on top of the underlying gate
/// (e.g. lowered from OpenQASM `ctrl @` modifiers); Phase 0 backends
/// may refuse non-empty `controls`.
///
/// Construction goes through [`GateInstance::new`] or
/// [`GateInstance::controlled`], both of which `debug_assert`:
/// - `qubits.len() == gate.arity()`
/// - every index in `qubits ∪ controls` appears at most once
///   (no qubit used as target twice or as both target and control)
///
/// Fields are `pub` for ergonomics (IR passes pattern-match on them);
/// callers who mutate fields directly are responsible for preserving
/// both invariants.
#[derive(Debug, Clone)]
pub struct GateInstance {
    pub gate: Gate,
    pub qubits: SmallVec<[u32; 4]>,
    pub controls: SmallVec<[u32; 2]>,
}

impl GateInstance {
    /// Construct an instance with no generic controls.
    ///
    /// In debug builds, panics if `qubits.len() != gate.arity()` or
    /// if `qubits` contains duplicate indices.
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
        #[cfg(debug_assertions)]
        if let Err(msg) = check_qubit_uniqueness(&qubits, &[]) {
            panic!("GateInstance::new: {msg}");
        }
        Self {
            gate,
            qubits,
            controls: SmallVec::new(),
        }
    }

    /// Construct an instance with generic external controls.
    ///
    /// In debug builds, panics if `qubits.len() != gate.arity()` or if
    /// any index appears more than once across `qubits ∪ controls`.
    pub fn controlled(
        gate: Gate,
        qubits: impl Into<SmallVec<[u32; 4]>>,
        controls: impl Into<SmallVec<[u32; 2]>>,
    ) -> Self {
        let qubits = qubits.into();
        let controls = controls.into();
        debug_assert_eq!(
            qubits.len(),
            gate.arity(),
            "GateInstance::controlled: qubits.len() ({}) != gate.arity() ({}) for {:?}",
            qubits.len(),
            gate.arity(),
            gate
        );
        #[cfg(debug_assertions)]
        if let Err(msg) = check_qubit_uniqueness(&qubits, &controls) {
            panic!("GateInstance::controlled: {msg}");
        }
        Self {
            gate,
            qubits,
            controls,
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

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "qubit index 1 appears more than once")]
    fn new_rejects_duplicate_target_qubits() {
        // Cnot has arity 2; duplicate target index is illegal.
        let _ = GateInstance::new(Gate::Cnot, smallvec![1u32, 1u32]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "qubit index 1 appears more than once")]
    fn controlled_rejects_qubits_controls_overlap() {
        // Cnot on qubits [0,1] with external control 1 — qubit 1
        // appears in both lists.
        let _ = GateInstance::controlled(Gate::Cnot, smallvec![0u32, 1u32], smallvec![1u32]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "qubit index 3 appears more than once")]
    fn controlled_rejects_duplicate_controls() {
        let _ = GateInstance::controlled(
            Gate::H,
            smallvec![0u32],
            smallvec![3u32, 3u32],
        );
    }

    #[test]
    fn controlled_disjoint_indices_ok() {
        // Sanity: a valid layout still constructs cleanly.
        let inst = GateInstance::controlled(
            Gate::Cnot,
            smallvec![0u32, 1u32],
            smallvec![2u32, 3u32],
        );
        assert_eq!(inst.qubits.as_slice(), &[0, 1]);
        assert_eq!(inst.controls.as_slice(), &[2, 3]);
    }
}
