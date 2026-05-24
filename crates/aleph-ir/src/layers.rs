//! Layer-extraction helper used by `Circuit::layers()`.
//!
//! Algorithm: greedy left-to-right pass tracking the most recent
//! (layer, instruction index) per qubit and per clbit. Two
//! instructions can share a layer iff their used-qubit sets are
//! disjoint OR the intersection consists only of qubits where both
//! are diagonal `Gate` instructions. Clbit writes never commute.
//!
//! Complexity: O(Σ arity(inst)) — each instruction does O(1) work
//! per touched qubit/clbit via fixed-size index arrays.

use crate::{Circuit, Instruction};

/// Whether two instructions can share a layer assuming they touch a
/// common qubit. Returns `true` only when both are `Gate(g)` with
/// `g.gate.is_diagonal() == true` — the diagonal-with-diagonal
/// commutation rule.
pub(crate) fn commute_on_qubit(a: &Instruction, b: &Instruction) -> bool {
    match (a, b) {
        (Instruction::Gate(ga), Instruction::Gate(gb)) => {
            ga.gate.is_diagonal() && gb.gate.is_diagonal()
        }
        _ => false,
    }
}

/// Group instruction indices into layers of (logically) parallel
/// instructions.
pub(crate) fn extract_layers(circuit: &Circuit) -> Vec<Vec<usize>> {
    let nq = circuit.num_qubits as usize;
    let nc = circuit.num_clbits as usize;

    let mut last_for_qubit: Vec<Option<(usize, usize)>> = vec![None; nq];
    let mut last_for_clbit: Vec<Option<usize>> = vec![None; nc];

    let mut layers: Vec<Vec<usize>> = Vec::new();
    // Monotonicity: each new instruction is placed at a layer >= the
    // previous instruction's layer. Without this, a later
    // dep-free instruction could backfill into a layer earlier than
    // the immediately preceding one, breaking the spec invariant that
    // flattening `layers` in order reproduces `0..len`.
    let mut prev_assigned_layer: usize = 0;

    for (i, inst) in circuit.instructions().iter().enumerate() {
        let qubits = inst.used_qubits();
        let clbits = inst.used_clbits();

        let mut earliest: usize = prev_assigned_layer;

        for &q in &qubits {
            if let Some((prev_layer, prev_idx)) = last_for_qubit[q as usize] {
                let prev_inst = &circuit.instructions()[prev_idx];
                if commute_on_qubit(prev_inst, inst) {
                    earliest = earliest.max(prev_layer);
                } else {
                    earliest = earliest.max(prev_layer + 1);
                }
            }
        }
        for &c in &clbits {
            if let Some(prev_layer) = last_for_clbit[c as usize] {
                earliest = earliest.max(prev_layer + 1);
            }
        }

        if earliest == layers.len() {
            layers.push(vec![i]);
        } else {
            layers[earliest].push(i);
        }

        for &q in &qubits {
            last_for_qubit[q as usize] = Some((earliest, i));
        }
        for &c in &clbits {
            last_for_clbit[c as usize] = Some(earliest);
        }
        prev_assigned_layer = earliest;
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Gate;

    #[test]
    fn empty_circuit_yields_no_layers() {
        let c = Circuit::new(2, 0);
        assert!(extract_layers(&c).is_empty());
    }

    #[test]
    fn single_h_yields_one_layer() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0]]);
    }

    #[test]
    fn two_disjoint_h_share_one_layer() {
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.h(1).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0, 1]]);
    }

    #[test]
    fn h_then_x_on_same_qubit_serializes() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.x(0).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0], vec![1]]);
    }

    #[test]
    fn two_diagonal_on_same_qubit_share_layer() {
        let mut c = Circuit::new(1, 0);
        c.z(0).unwrap();
        c.phase(0.5, 0).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0, 1]]);
    }

    #[test]
    fn three_diagonals_in_a_row_share_layer() {
        let mut c = Circuit::new(1, 0);
        c.z(0).unwrap();
        c.phase(0.3, 0).unwrap();
        c.z(0).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn diagonal_then_nondiagonal_breaks_layer() {
        let mut c = Circuit::new(1, 0);
        c.z(0).unwrap();
        c.phase(0.5, 0).unwrap();
        c.x(0).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn cnot_and_disjoint_h_share_layer() {
        let mut c = Circuit::new(3, 0);
        c.cnot(0, 1).unwrap();
        c.h(2).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0, 1]]);
    }

    #[test]
    fn measure_then_measure_same_qubit_serializes() {
        let mut c = Circuit::new(1, 2);
        c.measure(0, 0).unwrap();
        c.measure(0, 1).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0], vec![1]]);
    }

    #[test]
    fn measure_then_measure_same_clbit_serializes() {
        let mut c = Circuit::new(2, 1);
        c.measure(0, 0).unwrap();
        c.measure(1, 0).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0], vec![1]]);
    }

    #[test]
    fn barrier_blocks_subsequent_gates_on_listed_qubits() {
        let mut c = Circuit::new(3, 0);
        c.h(0).unwrap();
        c.h(1).unwrap();
        c.barrier([0u32, 1u32]).unwrap();
        c.h(0).unwrap();
        c.h(1).unwrap();
        assert_eq!(extract_layers(&c), vec![vec![0, 1], vec![2], vec![3, 4]]);
    }

    #[test]
    fn commute_on_qubit_predicate() {
        use aleph_core::{GateInstance, Param};
        use smallvec::smallvec;

        let z = Instruction::Gate(GateInstance::new(Gate::Z, smallvec![0u32]));
        let phase = Instruction::Gate(GateInstance::new(
            Gate::Phase(Param::Concrete(0.5)),
            smallvec![0u32],
        ));
        let h = Instruction::Gate(GateInstance::new(Gate::H, smallvec![0u32]));
        let m = Instruction::Measure { qubit: 0, clbit: 0 };
        let r = Instruction::Reset(0);
        let b = Instruction::Barrier(smallvec![0u32]);

        assert!(commute_on_qubit(&z, &phase));
        assert!(!commute_on_qubit(&z, &h));
        assert!(!commute_on_qubit(&z, &m));
        assert!(!commute_on_qubit(&m, &r));
        assert!(!commute_on_qubit(&b, &z));
    }
}
