//! `OpKind` union enum + circuit strategies.  See spec §4.3 and
//! the plan's §"Spec amendment" — the parser and IR tests
//! intentionally curate divergent vocabularies; this module
//! exports the union plus two `arb_op_*` / `arb_circuit_*`
//! strategies so neither test loses coverage.

use aleph_core::{Gate, GateInstance};
use aleph_ir::Circuit;
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use smallvec::smallvec;

/// Union of the operation vocabularies the parser and IR tests
/// each curate locally.  Some variants are emitter-supported
/// (`Sdg`, `Tdg`, `U3`); some exercise the IR's
/// non-builder-method paths (`Ccz`, `Controlled1q`).  Each
/// `arb_op_*` strategy in this module selects the appropriate
/// subset for its consumer.
#[derive(Debug, Clone)]
pub enum OpKind {
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
    Ccz(u32, u32, u32),
    /// Generic controlled-1q construction.  Exercises the
    /// `GateInstance::controlled` path that pure builder methods
    /// don't reach.
    Controlled1q(u32, u32),
    Measure(u32, u32),
    Reset(u32),
    Barrier1(u32),
    Barrier2(u32, u32),
}

impl OpKind {
    /// Apply this op to `c`.  Returns nothing — `Circuit`'s builder
    /// methods may reject invalid combinations, but our strategies
    /// never generate them; a silent drop here would indicate a
    /// bug in the strategy, not in the IR.
    pub fn apply(self, c: &mut Circuit) {
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

    /// If this op is a gate (not a measurement, reset, or
    /// barrier), return the corresponding `GateInstance`.  Used by
    /// backend-level proptests that want to skip non-gate variants
    /// instead of filtering them out at strategy-construction
    /// time.
    pub fn as_gate_instance(&self) -> Option<GateInstance> {
        match *self {
            OpKind::H(q) => Some(GateInstance::new(Gate::H, smallvec![q])),
            OpKind::X(q) => Some(GateInstance::new(Gate::X, smallvec![q])),
            OpKind::Y(q) => Some(GateInstance::new(Gate::Y, smallvec![q])),
            OpKind::Z(q) => Some(GateInstance::new(Gate::Z, smallvec![q])),
            OpKind::S(q) => Some(GateInstance::new(Gate::S, smallvec![q])),
            OpKind::T(q) => Some(GateInstance::new(Gate::T, smallvec![q])),
            OpKind::Sdg(q) => Some(GateInstance::new(Gate::Sdg, smallvec![q])),
            OpKind::Tdg(q) => Some(GateInstance::new(Gate::Tdg, smallvec![q])),
            OpKind::Rx(t, q) => Some(GateInstance::new(Gate::Rx(t.into()), smallvec![q])),
            OpKind::Ry(t, q) => Some(GateInstance::new(Gate::Ry(t.into()), smallvec![q])),
            OpKind::Rz(t, q) => Some(GateInstance::new(Gate::Rz(t.into()), smallvec![q])),
            OpKind::Phase(t, q) => Some(GateInstance::new(Gate::Phase(t.into()), smallvec![q])),
            OpKind::U3(a, b, d, q) => Some(GateInstance::new(
                Gate::U3(a.into(), b.into(), d.into()),
                smallvec![q],
            )),
            OpKind::Cnot(a, b) => Some(GateInstance::new(Gate::Cnot, smallvec![a, b])),
            OpKind::Cz(a, b) => Some(GateInstance::new(Gate::Cz, smallvec![a, b])),
            OpKind::Swap(a, b) => Some(GateInstance::new(Gate::Swap, smallvec![a, b])),
            OpKind::Toffoli(a, b, t) => Some(GateInstance::new(Gate::Toffoli, smallvec![a, b, t])),
            OpKind::Ccz(a, b, t) => Some(GateInstance::new(Gate::Ccz, smallvec![a, b, t])),
            OpKind::Controlled1q(target, ctrl) => Some(GateInstance::controlled(
                Gate::X,
                smallvec![target],
                smallvec![ctrl],
            )),
            OpKind::Measure(_, _)
            | OpKind::Reset(_)
            | OpKind::Barrier1(_)
            | OpKind::Barrier2(_, _) => None,
        }
    }
}

/// Distinct unordered pair `(a, b)` with `a, b ∈ [0, nq)`.
pub fn distinct_pair(nq: u32) -> impl Strategy<Value = (u32, u32)> {
    (0u32..nq, 0u32..nq).prop_filter("distinct", |(a, b)| a != b)
}

/// Distinct unordered triple `(a, b, c)` with all three in `[0, nq)`.
pub fn distinct_triple(nq: u32) -> impl Strategy<Value = (u32, u32, u32)> {
    (0u32..nq, 0u32..nq, 0u32..nq).prop_filter("distinct", |(a, b, c)| a != b && a != c && b != c)
}

/// `OpKind` vocabulary restricted to variants the emitter can
/// serialise.  Excludes `Ccz` and `Controlled1q` (which the
/// builder constructs but the emitter doesn't yet round-trip).
///
/// Used by `aleph-parser/tests/round_trip_property.rs`.
pub fn arb_op_emittable(nq: u32, nc: u32) -> BoxedStrategy<OpKind> {
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

/// Random emitter-compatible `Circuit`.  Replaces the inline
/// `arb_circuit(...)` previously duplicated in
/// `aleph-parser/tests/round_trip_property.rs`.
pub fn arb_circuit_emittable(nq: u32, nc: u32, n_ops: usize) -> impl Strategy<Value = Circuit> {
    proptest::collection::vec(arb_op_emittable(nq, nc), 0..=n_ops).prop_map(move |ops| {
        let mut c = Circuit::new(nq, nc);
        for op in ops {
            op.apply(&mut c);
        }
        c
    })
}

/// `OpKind` vocabulary that exercises the IR's layer-algorithm
/// paths.  Excludes `Sdg`, `Tdg`, `U3` (they overlap with cases
/// the parser test covers); adds `Ccz` and `Controlled1q` (they
/// reach the `add_gate` / `GateInstance::controlled` paths the
/// pure builder methods bypass).
///
/// Used by `aleph-ir/tests/layers_properties.rs`.
pub fn arb_op_full(nq: u32, nc: u32) -> BoxedStrategy<OpKind> {
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
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Phase(t, q)),
    ];
    let two_q = prop_oneof![
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cnot(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cz(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Swap(a, b)),
        distinct_pair(nq).prop_map(|(t, c)| OpKind::Controlled1q(t, c)),
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

/// Random `Circuit` exercising the IR's broader op vocabulary
/// (including `Ccz` and `Controlled1q`).  Replaces the inline
/// `arb_circuit(...)` previously duplicated in
/// `aleph-ir/tests/layers_properties.rs`.
pub fn arb_circuit_full(nq: u32, nc: u32, n_ops: usize) -> impl Strategy<Value = Circuit> {
    proptest::collection::vec(arb_op_full(nq, nc), 0..=n_ops).prop_map(move |ops| {
        let mut c = Circuit::new(nq, nc);
        for op in ops {
            op.apply(&mut c);
        }
        c
    })
}
