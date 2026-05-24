//! `Instruction` — one step of a `Circuit`. Either applies a gate,
//! measures, resets, or marks a barrier.

use aleph_core::GateInstance;
use smallvec::SmallVec;

/// One step in a circuit.
///
/// Qubit ordering inside `Gate` follows the per-variant convention
/// pinned in `aleph_core::Gate` (e.g. `[control, target]` for `Cnot`).
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Apply a gate to concrete qubits.
    Gate(GateInstance),
    /// Mid-circuit or terminal measurement of `qubit` into `clbit`.
    Measure { qubit: u32, clbit: u32 },
    /// Reset `qubit` to `|0⟩`.
    Reset(u32),
    /// Barrier forbidding optimization passes from crossing this point
    /// for the listed qubits. Inline 8 — larger barriers spill to heap.
    Barrier(SmallVec<[u32; 8]>),
}

impl Instruction {
    /// All qubits touched by this instruction (gate qubits ∪ external
    /// controls for `Gate`; the contained qubit(s) for the rest).
    pub fn used_qubits(&self) -> SmallVec<[u32; 6]> {
        let mut out: SmallVec<[u32; 6]> = SmallVec::new();
        match self {
            Instruction::Gate(g) => {
                out.extend(g.qubits.iter().copied());
                out.extend(g.controls.iter().copied());
            }
            Instruction::Measure { qubit, .. } => out.push(*qubit),
            Instruction::Reset(q) => out.push(*q),
            Instruction::Barrier(qs) => out.extend(qs.iter().copied()),
        }
        out
    }

    /// All classical bits touched by this instruction. Only `Measure`
    /// touches a clbit in Phase 0.
    pub fn used_clbits(&self) -> SmallVec<[u32; 2]> {
        let mut out: SmallVec<[u32; 2]> = SmallVec::new();
        if let Instruction::Measure { clbit, .. } = self {
            out.push(*clbit);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn used_qubits_gate_targets_and_controls() {
        let inst = Instruction::Gate(GateInstance::controlled(
            Gate::X,
            smallvec![3u32],
            smallvec![0u32, 1u32],
        ));
        let mut q = inst.used_qubits().to_vec();
        q.sort();
        assert_eq!(q, vec![0, 1, 3]);
    }

    #[test]
    fn used_qubits_measure() {
        let inst = Instruction::Measure { qubit: 7, clbit: 2 };
        assert_eq!(inst.used_qubits().as_slice(), &[7]);
    }

    #[test]
    fn used_qubits_reset() {
        let inst = Instruction::Reset(4);
        assert_eq!(inst.used_qubits().as_slice(), &[4]);
    }

    #[test]
    fn used_qubits_barrier() {
        let inst = Instruction::Barrier(smallvec![0u32, 2u32, 5u32]);
        assert_eq!(inst.used_qubits().as_slice(), &[0, 2, 5]);
    }

    #[test]
    fn used_clbits_only_for_measure() {
        assert_eq!(
            Instruction::Measure { qubit: 0, clbit: 3 }
                .used_clbits()
                .as_slice(),
            &[3]
        );
        assert!(Instruction::Reset(0).used_clbits().is_empty());
        assert!(Instruction::Barrier(smallvec![0u32])
            .used_clbits()
            .is_empty());
        assert!(
            Instruction::Gate(GateInstance::new(Gate::H, smallvec![0u32]))
                .used_clbits()
                .is_empty()
        );
    }
}
