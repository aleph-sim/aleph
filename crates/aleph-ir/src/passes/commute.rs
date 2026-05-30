//! `gates_commute` — a sound, conservative predicate answering whether
//! two gate instances commute (`A·B == B·A` as operators), so a later
//! pass may safely reorder them. First-match-wins rule table; when
//! unsure it returns `false` (a false negative only forgoes an
//! optimisation, a false positive would corrupt state). See
//! `docs/superpowers/specs/2026-05-30-p1-13-commutation-design.md`.

use aleph_core::{Gate, GateInstance};

/// True iff `a` and `b` provably commute as operators. Conservative:
/// returns `false` whenever commutation is not established by a rule.
/// Symmetric in its arguments.
pub fn gates_commute(a: &GateInstance, b: &GateInstance) -> bool {
    // Rule 1: disjoint support — operators on different qubits commute.
    if !supports_overlap(a, b) {
        return true;
    }
    // Rule 2: both diagonal — diagonal matrices (incl. controlled-diagonal,
    // which is still diagonal) always commute, on any qubits.
    if a.gate.is_diagonal() && b.gate.is_diagonal() {
        return true;
    }
    // Rule 3: structurally identical — an operator commutes with itself.
    if instances_identical(a, b) {
        return true;
    }
    // Rule 4: CNOT control/target relations (symmetric over arg order).
    if cnot_commutes_with_1q(a, b) || cnot_commutes_with_1q(b, a) {
        return true;
    }
    false
}

/// Whether `a` and `b` touch any common qubit (targets ∪ controls).
fn supports_overlap(a: &GateInstance, b: &GateInstance) -> bool {
    let in_a = |q: &u32| a.qubits.contains(q) || a.controls.contains(q);
    b.qubits.iter().any(in_a) || b.controls.iter().any(in_a)
}

/// Order-independent equality of two external-control lists.
fn controls_eq_set(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_unstable();
    y.sort_unstable();
    x == y
}

/// Same gate, same target qubits (positional), same controls (as a set).
fn instances_identical(a: &GateInstance, b: &GateInstance) -> bool {
    a.gate == b.gate && a.qubits == b.qubits && controls_eq_set(&a.controls, &b.controls)
}

/// True iff `cnot` is a bare `Cnot(c, t)` and `other` is a bare single-
/// qubit gate that commutes with it by the control/target relations:
/// a gate that commutes with X (`X`, `Rx`) on the target `t`, or any
/// diagonal gate on the control `c`. Both instances must have no
/// external controls.
fn cnot_commutes_with_1q(cnot: &GateInstance, other: &GateInstance) -> bool {
    if cnot.gate != Gate::Cnot || !cnot.controls.is_empty() {
        return false;
    }
    if other.gate.arity() != 1 || !other.controls.is_empty() {
        return false;
    }
    let control = cnot.qubits[0];
    let target = cnot.qubits[1];
    let q = other.qubits[0];
    if q == target {
        // Commutes with X on the target: gates that are functions of I and X.
        matches!(other.gate, Gate::X | Gate::Rx(_))
    } else if q == control {
        // Any diagonal gate on the control passes through CNOT.
        other.gate.is_diagonal()
    } else {
        // `q` overlaps the CNOT support only via {control, target}; if it
        // is neither, supports do not overlap (handled by rule 1).
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
        GateInstance::new(gate, qubits.to_vec())
    }

    fn rz(theta: f64, q: u32) -> GateInstance {
        GateInstance::new(Gate::Rz(aleph_core::Param::Concrete(theta)), [q].to_vec())
    }
    fn rx(theta: f64, q: u32) -> GateInstance {
        GateInstance::new(Gate::Rx(aleph_core::Param::Concrete(theta)), [q].to_vec())
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

    #[test]
    fn both_diagonal_commute() {
        // Z·Rz, S·T, Phase·Cz, Rz·CRz, Cz·Ccz — all diagonal, all commute.
        assert!(gates_commute(&g(Gate::Z, &[0]), &rz(0.4, 0)));
        assert!(gates_commute(&g(Gate::S, &[0]), &g(Gate::T, &[0])));
        assert!(gates_commute(
            &GateInstance::new(Gate::Phase(aleph_core::Param::Concrete(0.3)), [0u32].to_vec()),
            &g(Gate::Cz, &[0, 1])
        ));
        assert!(gates_commute(
            &rz(0.2, 1),
            &GateInstance::new(Gate::CRz(aleph_core::Param::Concrete(0.5)), [0u32, 1u32].to_vec())
        ));
        assert!(gates_commute(&g(Gate::Cz, &[0, 1]), &g(Gate::Ccz, &[0, 1, 2])));
    }

    #[test]
    fn controlled_diagonal_is_diagonal_commutes() {
        // Externally-controlled Z is still diagonal; commutes with Rz.
        let cz_ext = GateInstance::controlled(Gate::Z, smallvec![1u32], smallvec![0u32]);
        assert!(gates_commute(&cz_ext, &rz(0.7, 1)));
    }

    #[test]
    fn structurally_identical_commute() {
        assert!(gates_commute(&g(Gate::H, &[0]), &g(Gate::H, &[0])));
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Cnot, &[0, 1])));
        // Identically-controlled X (controls compared as a set).
        let a = GateInstance::controlled(Gate::X, smallvec![3u32], smallvec![0u32, 1u32]);
        let b = GateInstance::controlled(Gate::X, smallvec![3u32], smallvec![1u32, 0u32]);
        assert!(gates_commute(&a, &b));
    }

    #[test]
    fn cnot_commutes_with_x_or_rx_on_target() {
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[1])));
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &rx(0.9, 1)));
    }

    #[test]
    fn cnot_commutes_with_diagonal_on_control() {
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[0])));
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &rz(0.5, 0)));
    }

    #[test]
    fn cnot_does_not_commute_with_z_target_or_x_control() {
        assert!(!gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[1]))); // Z on target
        assert!(!gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[0]))); // X on control
        assert!(!gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Y, &[1]))); // Y on target (deferred)
    }

    #[test]
    fn externally_controlled_cnot_skips_rule_4() {
        // A CNOT carrying an external control is not a bare CNOT; rule 4
        // must not fire. ctrl-CNOT(c=2; q=[0,1]) vs X(1): no rule applies → false.
        let cc = GateInstance::controlled(Gate::Cnot, smallvec![0u32, 1u32], smallvec![2u32]);
        assert!(!gates_commute(&cc, &g(Gate::X, &[1])));
    }
}
