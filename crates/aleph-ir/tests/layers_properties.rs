//! Property tests for `Circuit::layers()`. They use only the public
//! API and run against random circuits drawn from a broad mix of
//! Instruction variants so the §6 algorithm's distinguishing branches
//! (clbit collision, parametric-diagonal commutation, controlled-gate
//! qubit union, Reset/Barrier non-commutation) are all exercised.

use aleph_ir::{Circuit, Instruction};
use aleph_test::circuit::{arb_circuit_full, OpKind};
use proptest::prelude::*;

fn touched_qubits(inst: &Instruction) -> Vec<u32> {
    inst.used_qubits().to_vec()
}

fn touched_clbits(inst: &Instruction) -> Vec<u32> {
    inst.used_clbits().to_vec()
}

/// Two instructions sharing one or more qubits commute iff both are
/// `Gate(g)` with `g.gate.is_diagonal() == true`. Matches the §6
/// rule that this property test enforces.
fn pair_can_share_layer(a: &Instruction, b: &Instruction) -> bool {
    matches!(
        (a, b),
        (Instruction::Gate(ga), Instruction::Gate(gb))
            if ga.gate.is_diagonal() && gb.gate.is_diagonal()
    )
}

proptest! {
    #[test]
    fn layers_flatten_to_0_to_len(c in arb_circuit_full(4, 2, 16)) {
        let layers = c.layers();
        let flat: Vec<usize> = layers.into_iter().flatten().collect();
        prop_assert_eq!(flat, (0..c.len()).collect::<Vec<_>>());
    }

    #[test]
    fn within_layer_no_non_commuting_overlap(c in arb_circuit_full(4, 2, 16)) {
        for layer in c.layers() {
            for i in 0..layer.len() {
                for j in (i + 1)..layer.len() {
                    let a = &c.instructions()[layer[i]];
                    let b = &c.instructions()[layer[j]];
                    let qa: std::collections::HashSet<u32> =
                        touched_qubits(a).into_iter().collect();
                    let qb: std::collections::HashSet<u32> =
                        touched_qubits(b).into_iter().collect();
                    let shared_q: Vec<u32> = qa.intersection(&qb).copied().collect();
                    if !shared_q.is_empty() {
                        prop_assert!(
                            pair_can_share_layer(a, b),
                            "layer contains pair {:?} and {:?} sharing qubits {:?} but neither commutes",
                            a, b, shared_q
                        );
                    }
                    // Clbit collisions: two writes to the same clbit
                    // must never share a layer.
                    let ca: std::collections::HashSet<u32> =
                        touched_clbits(a).into_iter().collect();
                    let cb: std::collections::HashSet<u32> =
                        touched_clbits(b).into_iter().collect();
                    let shared_c: Vec<u32> = ca.intersection(&cb).copied().collect();
                    prop_assert!(
                        shared_c.is_empty(),
                        "layer contains pair {:?} and {:?} writing to shared clbits {:?}",
                        a, b, shared_c
                    );
                }
            }
        }
    }

    #[test]
    fn same_qubit_succession_respects_commutation(c in arb_circuit_full(3, 2, 12)) {
        let layers = c.layers();
        let mut layer_of = vec![0usize; c.len()];
        for (li, l) in layers.iter().enumerate() {
            for &idx in l {
                layer_of[idx] = li;
            }
        }
        for q in 0..c.num_qubits() {
            let on_q: Vec<usize> = c.instructions().iter().enumerate()
                .filter(|(_, inst)| touched_qubits(inst).contains(&q))
                .map(|(i, _)| i)
                .collect();
            for w in on_q.windows(2) {
                let (i, j) = (w[0], w[1]);
                let a = &c.instructions()[i];
                let b = &c.instructions()[j];
                if pair_can_share_layer(a, b) {
                    prop_assert!(layer_of[j] >= layer_of[i]);
                } else {
                    prop_assert!(
                        layer_of[j] > layer_of[i],
                        "non-commuting pair on qubit {q} ended up in same layer"
                    );
                }
            }
        }
    }

    #[test]
    fn same_clbit_writes_serialize(c in arb_circuit_full(3, 1, 12)) {
        let layers = c.layers();
        let mut layer_of = vec![0usize; c.len()];
        for (li, l) in layers.iter().enumerate() {
            for &idx in l {
                layer_of[idx] = li;
            }
        }
        for cl in 0..c.num_clbits() {
            let on_c: Vec<usize> = c.instructions().iter().enumerate()
                .filter(|(_, inst)| touched_clbits(inst).contains(&cl))
                .map(|(i, _)| i)
                .collect();
            for w in on_c.windows(2) {
                let (i, j) = (w[0], w[1]);
                prop_assert!(
                    layer_of[j] > layer_of[i],
                    "two writes to clbit {cl} (insts {i}, {j}) share layer {}",
                    layer_of[i]
                );
            }
        }
    }
}

/// Defensive guard: every `OpKind` variant must survive a round trip
/// through `apply`. If a variant ever rejects on a known-valid input,
/// the dispatch above would silently drop it — this test catches that
/// by asserting the exact count of appended instructions.
///
/// Covers the full union vocabulary exposed by
/// `aleph_test::circuit::OpKind`, including the parser-emittable
/// extras (Sdg, Tdg, U3) that the IR test's previous local OpKind
/// didn't include.
#[test]
fn every_op_kind_is_constructible() {
    let ops: [OpKind; 23] = [
        OpKind::H(0),
        OpKind::X(0),
        OpKind::Y(0),
        OpKind::Z(0),
        OpKind::S(0),
        OpKind::T(0),
        OpKind::Sdg(0),
        OpKind::Tdg(0),
        OpKind::Rx(0.1, 0),
        OpKind::Ry(0.2, 0),
        OpKind::Rz(0.3, 0),
        OpKind::Phase(0.4, 0),
        OpKind::U3(0.5, 0.6, 0.7, 0),
        OpKind::Cnot(0, 1),
        OpKind::Cz(0, 1),
        OpKind::Swap(0, 1),
        OpKind::Toffoli(0, 1, 2),
        OpKind::Ccz(0, 1, 2),
        OpKind::Controlled1q(3, 2),
        OpKind::Measure(0, 0),
        OpKind::Reset(1),
        OpKind::Barrier1(2),
        OpKind::Barrier2(0, 1),
    ];
    let expected = ops.len();
    let mut c = Circuit::new(4, 2);
    for op in ops {
        op.apply(&mut c);
    }
    assert_eq!(
        c.len(),
        expected,
        "some OpKind variants were silently rejected by the IR",
    );
}
