//! Maps `aleph_core::GateInstance` onto [`Tableau`] operations and
//! rejects non-Clifford gates. Uses `Gate::is_clifford()` as the
//! single source of truth for the Clifford set.

use crate::{StabError, Tableau};
use aleph_core::{Gate, GateInstance};

/// Apply one IR gate to the tableau.
///
/// Returns [`StabError::NonClifford`] for any gate outside the Clifford
/// group, and [`StabError::QubitOutOfRange`] for out-of-range indices
/// (surfaced by the underlying `Tableau` methods).
///
/// External `controls` (generic `ctrl @` modifiers) are not supported by
/// the stabilizer backend in P3-01 and are rejected as non-Clifford if
/// present (a controlled-Clifford is not necessarily Clifford, and the
/// IR's `is_clifford()` describes the base gate only).
pub fn apply_gate(t: &mut Tableau, inst: &GateInstance) -> Result<(), StabError> {
    if !inst.controls.is_empty() {
        return Err(StabError::NonClifford {
            gate: gate_name(&inst.gate),
        });
    }
    let q = &inst.qubits;
    match &inst.gate {
        Gate::H => t.h(q[0] as usize),
        Gate::S => t.s(q[0] as usize),
        Gate::Sdg => t.sdg(q[0] as usize),
        Gate::X => t.x_gate(q[0] as usize),
        Gate::Y => t.y_gate(q[0] as usize),
        Gate::Z => t.z_gate(q[0] as usize),
        Gate::Cnot => t.cnot(q[0] as usize, q[1] as usize),
        Gate::Cz => t.cz(q[0] as usize, q[1] as usize),
        Gate::Swap => t.swap(q[0] as usize, q[1] as usize),
        Gate::Iswap => t.iswap(q[0] as usize, q[1] as usize),
        Gate::IswapDg => t.iswap_dg(q[0] as usize, q[1] as usize),
        other => {
            debug_assert!(
                !other.is_clifford(),
                "Clifford gate {other:?} not dispatched"
            );
            Err(StabError::NonClifford {
                gate: gate_name(other),
            })
        }
    }
}

/// Static name for error messages (no allocation). Exhaustive over all
/// `Gate` variants — the compiler enforces completeness.
fn gate_name(g: &Gate) -> &'static str {
    match g {
        Gate::H => "H",
        Gate::X => "X",
        Gate::Y => "Y",
        Gate::Z => "Z",
        Gate::S => "S",
        Gate::Sdg => "Sdg",
        Gate::T => "T",
        Gate::Tdg => "Tdg",
        Gate::Rx(_) => "Rx",
        Gate::Ry(_) => "Ry",
        Gate::Rz(_) => "Rz",
        Gate::Phase(_) => "Phase",
        Gate::U3(..) => "U3",
        Gate::Cnot => "Cnot",
        Gate::Cz => "Cz",
        Gate::Swap => "Swap",
        Gate::Iswap => "Iswap",
        Gate::IswapDg => "IswapDg",
        Gate::CRx(_) => "CRx",
        Gate::CRy(_) => "CRy",
        Gate::CRz(_) => "CRz",
        Gate::Toffoli => "Toffoli",
        Gate::Ccz => "Ccz",
        Gate::Unitary1q(_) => "Unitary1q",
        Gate::Unitary1qDiag(_) => "Unitary1qDiag",
        Gate::Unitary2q(_) => "Unitary2q",
        Gate::UnitaryKq { .. } => "UnitaryKq",
    }
}

#[cfg(test)]
mod tests {
    use crate::{apply_gate, Tableau};
    use aleph_core::{Gate, GateInstance};

    #[test]
    fn dispatch_bell() {
        let mut t = Tableau::new(2);
        apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32])).unwrap();
        apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![0u32, 1u32])).unwrap();
        assert_eq!(t.stabilizers().len(), 2);
    }

    #[test]
    fn rejects_non_clifford() {
        let mut t = Tableau::new(1);
        let err = apply_gate(&mut t, &GateInstance::new(Gate::T, vec![0u32])).unwrap_err();
        assert!(matches!(err, crate::StabError::NonClifford { .. }));
    }
}
