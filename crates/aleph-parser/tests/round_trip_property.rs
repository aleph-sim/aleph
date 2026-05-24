//! Property test: random `Circuit` (restricted to emitter-supported
//! variants) round-trips through `emit → parse → compare`.

use aleph_ir::Circuit;
use aleph_parser::{emit, parse};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

#[derive(Debug, Clone)]
enum OpKind {
    H(u32),
    X(u32),
    Y(u32),
    Z(u32),
    S(u32),
    T(u32),
    Sdg(u32),
    Tdg(u32),
    Rx(f64, u32),
    Ry(f64, u32),
    Rz(f64, u32),
    Phase(f64, u32),
    U3(f64, f64, f64, u32),
    Cnot(u32, u32),
    Cz(u32, u32),
    Swap(u32, u32),
    Toffoli(u32, u32, u32),
    Measure(u32, u32),
    Reset(u32),
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
            OpKind::Sdg(q) => c.sdg(q),
            OpKind::Tdg(q) => c.tdg(q),
            OpKind::Rx(t, q) => c.rx(t, q),
            OpKind::Ry(t, q) => c.ry(t, q),
            OpKind::Rz(t, q) => c.rz(t, q),
            OpKind::Phase(t, q) => c.phase(t, q),
            OpKind::U3(a, b, d, q) => c.u3(a, b, d, q),
            OpKind::Cnot(a, b) => c.cnot(a, b),
            OpKind::Cz(a, b) => c.cz(a, b),
            OpKind::Swap(a, b) => c.swap(a, b),
            OpKind::Toffoli(a, b, t) => c.ccx(a, b, t),
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
    let angle = -10.0_f64..10.0_f64;

    let single = prop_oneof![
        (0u32..nq).prop_map(OpKind::H),
        (0u32..nq).prop_map(OpKind::X),
        (0u32..nq).prop_map(OpKind::Y),
        (0u32..nq).prop_map(OpKind::Z),
        (0u32..nq).prop_map(OpKind::S),
        (0u32..nq).prop_map(OpKind::T),
        (0u32..nq).prop_map(OpKind::Sdg),
        (0u32..nq).prop_map(OpKind::Tdg),
    ];
    let parametric = prop_oneof![
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rx(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Ry(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rz(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Phase(t, q)),
        (angle.clone(), angle.clone(), angle.clone(), 0u32..nq)
            .prop_map(|(a, b, c, q)| OpKind::U3(a, b, c, q)),
    ];
    let two_q = prop_oneof![
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cnot(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cz(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Swap(a, b)),
    ];
    let three_q = distinct_triple(nq).prop_map(|(a, b, t)| OpKind::Toffoli(a, b, t));
    let non_gate = prop_oneof![
        (0u32..nq).prop_map(OpKind::Reset),
        (0u32..nq).prop_map(OpKind::Barrier1),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Barrier2(a, b)),
    ];

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
            3 => measurement,
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

proptest! {
    #[test]
    fn parse_emit_roundtrip(c in arb_circuit(4, 2, 12)) {
        let out = match emit(&c) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        let c2 = parse(&out).map_err(|e| TestCaseError::fail(format!(
            "re-parse failed.\nemitted:\n{out}\nerror:\n{}",
            e.render()
        )))?;
        prop_assert_eq!(c.len(), c2.len(), "instruction count mismatch");
        prop_assert_eq!(c.num_qubits(), c2.num_qubits());
        prop_assert_eq!(c.num_clbits(), c2.num_clbits());
        for (i, (a, b)) in c.instructions().iter().zip(c2.instructions().iter()).enumerate() {
            prop_assert_eq!(format!("{a:?}"), format!("{b:?}"), "instr {} differs", i);
        }
    }
}
