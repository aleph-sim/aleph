//! Property tests for `Circuit::layers()`. They use only the public
//! API and run against random circuits.

use aleph_ir::{Circuit, Instruction};
use proptest::prelude::*;

fn arb_circuit(nq: u32, n_ops: usize) -> impl Strategy<Value = Circuit> {
    let single = (0u8..4u8, 0u32..nq).prop_map(|(t, q)| (t, q, q));
    let two = (0u32..nq, 0u32..nq)
        .prop_filter("distinct cnot qubits", |(c, t)| c != t)
        .prop_map(|(c, t)| (4u8, c, t));
    let any_op = prop_oneof![single, two];

    proptest::collection::vec(any_op, 0..=n_ops).prop_map(move |ops| {
        let mut c = Circuit::new(nq, 0);
        for (tag, a, b) in ops {
            let _ = match tag {
                0 => c.h(a),
                1 => c.z(a),
                2 => c.s(a),
                3 => c.x(a),
                4 => c.cnot(a, b),
                _ => unreachable!(),
            };
        }
        c
    })
}

fn touched_qubits(inst: &Instruction) -> Vec<u32> {
    inst.used_qubits().to_vec()
}

proptest! {
    #[test]
    fn layers_flatten_to_0_to_len(c in arb_circuit(4, 16)) {
        let layers = c.layers();
        let flat: Vec<usize> = layers.into_iter().flatten().collect();
        prop_assert_eq!(flat, (0..c.len()).collect::<Vec<_>>());
    }

    #[test]
    fn within_layer_no_non_commuting_overlap(c in arb_circuit(4, 16)) {
        for layer in c.layers() {
            for i in 0..layer.len() {
                for j in (i + 1)..layer.len() {
                    let a = &c.instructions()[layer[i]];
                    let b = &c.instructions()[layer[j]];
                    let qa: std::collections::HashSet<u32> =
                        touched_qubits(a).into_iter().collect();
                    let qb: std::collections::HashSet<u32> =
                        touched_qubits(b).into_iter().collect();
                    let shared: Vec<u32> = qa.intersection(&qb).copied().collect();
                    if shared.is_empty() {
                        continue;
                    }
                    let ok = matches!((a, b),
                        (Instruction::Gate(ga), Instruction::Gate(gb))
                            if ga.gate.is_diagonal() && gb.gate.is_diagonal());
                    prop_assert!(
                        ok,
                        "layer contains pair {:?} and {:?} sharing qubits {:?} but neither commutes",
                        a, b, shared
                    );
                }
            }
        }
    }

    #[test]
    fn same_qubit_succession_respects_commutation(c in arb_circuit(3, 12)) {
        let layers = c.layers();
        let mut layer_of = vec![0usize; c.len()];
        for (li, l) in layers.iter().enumerate() {
            for &idx in l {
                layer_of[idx] = li;
            }
        }
        for q in 0..c.num_qubits {
            let on_q: Vec<usize> = c.instructions().iter().enumerate()
                .filter(|(_, inst)| touched_qubits(inst).contains(&q))
                .map(|(i, _)| i)
                .collect();
            for w in on_q.windows(2) {
                let (i, j) = (w[0], w[1]);
                let a = &c.instructions()[i];
                let b = &c.instructions()[j];
                let both_diagonal = matches!((a, b),
                    (Instruction::Gate(ga), Instruction::Gate(gb))
                        if ga.gate.is_diagonal() && gb.gate.is_diagonal());
                if both_diagonal {
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
}
