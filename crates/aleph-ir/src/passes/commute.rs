//! `gates_commute` — a sound, conservative predicate answering whether
//! two gate instances commute (`A·B == B·A` as operators), so a later
//! pass may safely reorder them. First-match-wins rule table; when
//! unsure it returns `false` (a false negative only forgoes an
//! optimisation, a false positive would corrupt state). See
//! `docs/superpowers/specs/2026-05-30-p1-13-commutation-design.md`.

use aleph_core::GateInstance;

/// True iff `a` and `b` provably commute as operators. Conservative:
/// returns `false` whenever commutation is not established by a rule.
/// Symmetric in its arguments.
pub fn gates_commute(a: &GateInstance, b: &GateInstance) -> bool {
    // Rule 1: disjoint support — operators on different qubits commute.
    if !supports_overlap(a, b) {
        return true;
    }
    false
}

/// Whether `a` and `b` touch any common qubit (targets ∪ controls).
fn supports_overlap(a: &GateInstance, b: &GateInstance) -> bool {
    let in_a = |q: &u32| a.qubits.contains(q) || a.controls.contains(q);
    b.qubits.iter().any(in_a) || b.controls.iter().any(in_a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};

    fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
        GateInstance::new(gate, qubits.to_vec())
    }

    #[test]
    fn disjoint_support_commutes() {
        // H(0) and X(1) act on different qubits.
        assert!(gates_commute(&g(Gate::H, &[0]), &g(Gate::X, &[1])));
        // CNOT(0,1) and Z(2): q2 disjoint from {0,1}.
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[2])));
    }

    #[test]
    fn overlapping_non_commuting_is_false_for_now() {
        // X(0) and Z(0) overlap and (until rule table is built) must not
        // be falsely reported as commuting.
        assert!(!gates_commute(&g(Gate::X, &[0]), &g(Gate::Z, &[0])));
    }
}
