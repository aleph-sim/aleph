//! Property tests for `Circuit::layers()`. They use only the public
//! API and run against random circuits drawn from a broad mix of
//! Instruction variants so the §6 algorithm's distinguishing branches
//! (clbit collision, parametric-diagonal commutation, controlled-gate
//! qubit union, Reset/Barrier non-commutation) are all exercised.

use aleph_core::{Gate, GateInstance};
use aleph_ir::{Circuit, Instruction};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use smallvec::smallvec;

/// One synthesised operation. Apply via [`OpKind::apply`] — the
/// strategy never picks an OOB index, but the IR's validation still
/// runs so any rejected op is silently dropped (it would surface as a
/// bug in the strategy if it happened).
#[derive(Debug, Clone)]
enum OpKind {
    H(u32),
    X(u32),
    Y(u32),
    Z(u32),
    S(u32),
    T(u32),
    Rx(f64, u32),
    Ry(f64, u32),
    Rz(f64, u32),
    Phase(f64, u32),
    Cnot(u32, u32),
    Cz(u32, u32),
    Swap(u32, u32),
    Toffoli(u32, u32, u32),
    Ccz(u32, u32, u32),
    /// Generic controlled construction — a 1q Pauli with one external
    /// control. Exercises the `GateInstance::controlled` path that
    /// pure builder methods don't reach.
    Controlled1q(u32, u32),
    Measure(u32, u32),
    Reset(u32),
    /// Barrier covering one or two distinct qubits (never empty,
    /// never duplicate — those error paths are unit-tested).
    Barrier1(u32),
    Barrier2(u32, u32),
}

impl OpKind {
    fn apply(self, c: &mut Circuit) {
        let _ = match self {
            OpKind::H(q) => c.h(q),
            OpKind::X(q) => c.x(q),
            OpKind::Y(q) => c.y(q),
            OpKind::Z(q) => c.z(q),
            OpKind::S(q) => c.s(q),
            OpKind::T(q) => c.t(q),
            OpKind::Rx(t, q) => c.rx(t, q),
            OpKind::Ry(t, q) => c.ry(t, q),
            OpKind::Rz(t, q) => c.rz(t, q),
            OpKind::Phase(t, q) => c.phase(t, q),
            OpKind::Cnot(a, b) => c.cnot(a, b),
            OpKind::Cz(a, b) => c.cz(a, b),
            OpKind::Swap(a, b) => c.swap(a, b),
            OpKind::Toffoli(a, b, t) => c.ccx(a, b, t),
            OpKind::Ccz(a, b, t) => c.add_gate(GateInstance::new(Gate::Ccz, smallvec![a, b, t])),
            OpKind::Controlled1q(target, ctrl) => c.add_gate(GateInstance::controlled(
                Gate::X,
                smallvec![target],
                smallvec![ctrl],
            )),
            OpKind::Measure(q, cl) => c.measure(q, cl),
            OpKind::Reset(q) => c.reset(q),
            OpKind::Barrier1(q) => c.barrier([q]),
            OpKind::Barrier2(a, b) => c.barrier([a, b]),
        };
    }
}

fn distinct_pair(nq: u32) -> impl Strategy<Value = (u32, u32)> {
    (0u32..nq, 0u32..nq).prop_filter("distinct", |(a, b)| a != b)
}

fn distinct_triple(nq: u32) -> impl Strategy<Value = (u32, u32, u32)> {
    (0u32..nq, 0u32..nq, 0u32..nq).prop_filter("distinct", |(a, b, c)| a != b && a != c && b != c)
}

fn arb_op(nq: u32, nc: u32) -> BoxedStrategy<OpKind> {
    // angles bounded — proptest is happiest with finite floats
    let angle = -10.0_f64..10.0_f64;

    let single = prop_oneof![
        (0u32..nq).prop_map(OpKind::H),
        (0u32..nq).prop_map(OpKind::X),
        (0u32..nq).prop_map(OpKind::Y),
        (0u32..nq).prop_map(OpKind::Z),
        (0u32..nq).prop_map(OpKind::S),
        (0u32..nq).prop_map(OpKind::T),
    ];
    let parametric = prop_oneof![
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rx(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Ry(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rz(t, q)),
        (angle, 0u32..nq).prop_map(|(t, q)| OpKind::Phase(t, q)),
    ];
    let two_q = prop_oneof![
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cnot(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cz(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Swap(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Controlled1q(a, b)),
    ];
    let three_q = prop_oneof![
        distinct_triple(nq).prop_map(|(a, b, t)| OpKind::Toffoli(a, b, t)),
        distinct_triple(nq).prop_map(|(a, b, t)| OpKind::Ccz(a, b, t)),
    ];
    let non_gate = prop_oneof![
        (0u32..nq).prop_map(OpKind::Reset),
        (0u32..nq).prop_map(OpKind::Barrier1),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Barrier2(a, b)),
    ];

    // Weight gates heavier than non-gate so a typical generated
    // circuit looks gate-dominated, but every variant is exercised
    // within a default proptest budget. Measurement is only included
    // when `nc >= 1` — `0u32..0` is an empty range and would panic.
    if nc == 0 {
        prop_oneof![
            4 => single,
            3 => parametric,
            3 => two_q,
            2 => three_q,
            2 => non_gate,
        ]
        .boxed()
    } else {
        let measurement = (0u32..nq, 0u32..nc).prop_map(|(q, cl)| OpKind::Measure(q, cl));
        prop_oneof![
            4 => single,
            3 => parametric,
            3 => two_q,
            2 => three_q,
            2 => non_gate,
            // bumped from 1 (~6.7% of ops) to 4 (~21% of ops) so
            // `same_clbit_writes_serialize` actually sees collisions.
            4 => measurement,
        ]
        .boxed()
    }
}

fn arb_circuit(nq: u32, nc: u32, n_ops: usize) -> impl Strategy<Value = Circuit> {
    proptest::collection::vec(arb_op(nq, nc), 0..=n_ops).prop_map(move |ops| {
        let mut c = Circuit::new(nq, nc);
        for op in ops {
            op.apply(&mut c);
        }
        c
    })
}

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
    fn layers_flatten_to_0_to_len(c in arb_circuit(4, 2, 16)) {
        let layers = c.layers();
        let flat: Vec<usize> = layers.into_iter().flatten().collect();
        prop_assert_eq!(flat, (0..c.len()).collect::<Vec<_>>());
    }

    #[test]
    fn within_layer_no_non_commuting_overlap(c in arb_circuit(4, 2, 16)) {
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
    fn same_qubit_succession_respects_commutation(c in arb_circuit(3, 2, 12)) {
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
    fn same_clbit_writes_serialize(c in arb_circuit(3, 1, 12)) {
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
#[test]
fn every_op_kind_is_constructible() {
    let ops: [OpKind; 20] = [
        OpKind::H(0),
        OpKind::X(0),
        OpKind::Y(0),
        OpKind::Z(0),
        OpKind::S(0),
        OpKind::T(0),
        OpKind::Rx(0.1, 0),
        OpKind::Ry(0.2, 0),
        OpKind::Rz(0.3, 0),
        OpKind::Phase(0.4, 0),
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
