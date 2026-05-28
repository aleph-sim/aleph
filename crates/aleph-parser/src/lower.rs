//! AST → IR lowering. Resolves register names to flat qubit/clbit
//! indices, expands whole-register `barrier`/`measure` forms, builds
//! `GateInstance`s by mapping OpenQASM gate names to `aleph_core::Gate`
//! variants, and delegates per-instruction validation to the IR.

use std::collections::HashMap;

use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::{Circuit, CircuitError, Instruction, MAX_CLBITS, MAX_QUBITS};

use crate::ast::{
    BarrierStmt, Decl, GateStmt, IndexedRef, MeasureStmt, Position, Program, RegOrIdx, ResetStmt,
    Stmt,
};
use crate::error::{ParseError, ParseErrorKind};

struct RegisterMap {
    /// `name → (base flat index, size)`.
    qregs: HashMap<String, (u32, u32)>,
    cregs: HashMap<String, (u32, u32)>,
    total_qubits: u32,
    total_clbits: u32,
}

/// Internal error type for `RegisterMap` mutations. The caller maps
/// these to `ParseErrorKind`.
enum RegError {
    Duplicate(String),
    TooManyQubits(u32),
    TooManyClbits(u32),
}

impl RegisterMap {
    fn new() -> Self {
        Self {
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            total_qubits: 0,
            total_clbits: 0,
        }
    }

    /// Errors for `add_qreg` / `add_creg`. `RegisterMap` errors are
    /// internally typed (not `CircuitError`) so the caller can map
    /// duplicate-register errors to `ParseErrorKind::DuplicateRegister`.
    fn add_qreg(&mut self, name: String, size: u32) -> Result<(), RegError> {
        if self.qregs.contains_key(&name) || self.cregs.contains_key(&name) {
            return Err(RegError::Duplicate(name));
        }
        let new_total = self.total_qubits.saturating_add(size);
        if new_total > MAX_QUBITS {
            return Err(RegError::TooManyQubits(new_total));
        }
        let base = self.total_qubits;
        self.qregs.insert(name, (base, size));
        self.total_qubits = new_total;
        Ok(())
    }

    fn add_creg(&mut self, name: String, size: u32) -> Result<(), RegError> {
        if self.qregs.contains_key(&name) || self.cregs.contains_key(&name) {
            return Err(RegError::Duplicate(name));
        }
        let new_total = self.total_clbits.saturating_add(size);
        if new_total > MAX_CLBITS {
            return Err(RegError::TooManyClbits(new_total));
        }
        let base = self.total_clbits;
        self.cregs.insert(name, (base, size));
        self.total_clbits = new_total;
        Ok(())
    }

    fn resolve_qubit(&self, name: &str, index: u32) -> Result<u32, ParseErrorKind> {
        match self.qregs.get(name) {
            None => Err(ParseErrorKind::UnknownRegister {
                name: name.to_string(),
            }),
            Some(&(base, size)) if index < size => Ok(base + index),
            Some(&(_, size)) => Err(ParseErrorKind::IndexOutOfBounds {
                register: name.to_string(),
                index,
                size,
            }),
        }
    }

    fn resolve_clbit(&self, name: &str, index: u32) -> Result<u32, ParseErrorKind> {
        match self.cregs.get(name) {
            None => Err(ParseErrorKind::UnknownRegister {
                name: name.to_string(),
            }),
            Some(&(base, size)) if index < size => Ok(base + index),
            Some(&(_, size)) => Err(ParseErrorKind::IndexOutOfBounds {
                register: name.to_string(),
                index,
                size,
            }),
        }
    }

    fn qreg_size(&self, name: &str) -> Option<(u32, u32)> {
        self.qregs.get(name).copied()
    }

    fn creg_size(&self, name: &str) -> Option<(u32, u32)> {
        self.cregs.get(name).copied()
    }
}

fn nth_line(source: &str, line: u32) -> String {
    source
        .lines()
        .nth(line.saturating_sub(1) as usize)
        .unwrap_or("")
        .to_string()
}

fn perr(source: &str, pos: Position, kind: ParseErrorKind) -> ParseError {
    ParseError::new(pos.line, pos.col, nth_line(source, pos.line), kind)
}

/// Public entry: ast::Program + the original source string → Circuit.
pub fn lower(program: Program, source: &str) -> Result<Circuit, ParseError> {
    for inc in &program.includes {
        if inc.path != "stdgates.inc" {
            return Err(perr(
                source,
                inc.pos,
                ParseErrorKind::UnsupportedFeature {
                    feature: "non-stdgates include",
                },
            ));
        }
    }

    let mut regs = RegisterMap::new();
    for d in &program.decls {
        match d {
            Decl::Qreg { pos, name, size } => regs.add_qreg(name.clone(), *size).map_err(|e| {
                let kind = match e {
                    RegError::Duplicate(n) => ParseErrorKind::DuplicateRegister { name: n },
                    RegError::TooManyQubits(requested) => ParseErrorKind::TooManyQubits {
                        requested,
                        max: MAX_QUBITS,
                    },
                    RegError::TooManyClbits(requested) => ParseErrorKind::TooManyClbits {
                        requested,
                        max: MAX_CLBITS,
                    },
                };
                perr(source, *pos, kind)
            })?,
            Decl::Creg { pos, name, size } => regs.add_creg(name.clone(), *size).map_err(|e| {
                let kind = match e {
                    RegError::Duplicate(n) => ParseErrorKind::DuplicateRegister { name: n },
                    RegError::TooManyQubits(requested) => ParseErrorKind::TooManyQubits {
                        requested,
                        max: MAX_QUBITS,
                    },
                    RegError::TooManyClbits(requested) => ParseErrorKind::TooManyClbits {
                        requested,
                        max: MAX_CLBITS,
                    },
                };
                perr(source, *pos, kind)
            })?,
        }
    }

    let mut circuit = Circuit::try_new(regs.total_qubits, regs.total_clbits).map_err(|e| {
        let kind = match e {
            CircuitError::TooManyQubits { requested, max } => {
                ParseErrorKind::TooManyQubits { requested, max }
            }
            CircuitError::TooManyClbits { requested, max } => {
                ParseErrorKind::TooManyClbits { requested, max }
            }
            other => ParseErrorKind::IrRejected(other),
        };
        ParseError::new(1, 1, nth_line(source, 1), kind)
    })?;
    circuit = circuit.with_generated_from("openqasm:3.0");

    for s in program.stmts {
        match s {
            Stmt::Gate(g) => lower_gate(&mut circuit, &regs, &g, source)?,
            Stmt::Barrier(b) => lower_barrier(&mut circuit, &regs, &b, source)?,
            Stmt::Measure(m) => lower_measure(&mut circuit, &regs, &m, source)?,
            Stmt::Reset(r) => lower_reset(&mut circuit, &regs, &r, source)?,
        }
    }

    Ok(circuit)
}

fn resolve_indexed(regs: &RegisterMap, r: &IndexedRef, source: &str) -> Result<u32, ParseError> {
    regs.resolve_qubit(&r.name, r.index)
        .map_err(|kind| perr(source, r.pos, kind))
}

fn lower_gate(
    circuit: &mut Circuit,
    regs: &RegisterMap,
    g: &GateStmt,
    source: &str,
) -> Result<(), ParseError> {
    let (shape, expected_params) = resolve_gate_name(&g.name).ok_or_else(|| {
        perr(
            source,
            g.pos,
            ParseErrorKind::UnknownGate {
                name: g.name.clone(),
            },
        )
    })?;
    if g.params.len() != expected_params {
        return Err(perr(
            source,
            g.pos,
            ParseErrorKind::UnexpectedToken {
                expected: "param count match",
                found: format!("{} params (expected {})", g.params.len(), expected_params),
            },
        ));
    }
    let gate = build_gate_variant(&shape, &g.params);
    let qubits: smallvec::SmallVec<[u32; 4]> = g
        .args
        .iter()
        .map(|r| resolve_indexed(regs, r, source))
        .collect::<Result<_, _>>()?;
    // Pre-check qubit uniqueness so we surface a structured ParseError
    // instead of tripping GateInstance::new's debug_assert in debug
    // builds (which would panic the host). validate_gate also catches
    // this in release, but only after the panic-prone constructor runs.
    let mut seen: smallvec::SmallVec<[u32; 4]> = smallvec::SmallVec::new();
    for &q in &qubits {
        if seen.contains(&q) {
            return Err(perr(
                source,
                g.pos,
                ParseErrorKind::IrRejected(aleph_ir::CircuitError::DuplicateQubit { qubit: q }),
            ));
        }
        seen.push(q);
    }
    let inst = GateInstance::new(gate, qubits);
    circuit
        .add_gate(inst)
        .map_err(|e| perr(source, g.pos, ParseErrorKind::IrRejected(e)))?;
    Ok(())
}

fn resolve_gate_name(name: &str) -> Option<(GateShape, usize)> {
    use GateShape as G;
    Some(match name {
        "h" => (G::H, 0),
        "x" => (G::X, 0),
        "y" => (G::Y, 0),
        "z" => (G::Z, 0),
        "s" => (G::S, 0),
        "sdg" => (G::Sdg, 0),
        "t" => (G::T, 0),
        "tdg" => (G::Tdg, 0),
        "rx" => (G::Rx, 1),
        "ry" => (G::Ry, 1),
        "rz" => (G::Rz, 1),
        "p" => (G::Phase, 1),
        "u3" => (G::U3, 3),
        "cx" => (G::Cnot, 0),
        "cz" => (G::Cz, 0),
        "swap" => (G::Swap, 0),
        "ccx" => (G::Toffoli, 0),
        // `ccz` is not in stdgates.inc but is a natural extension:
        // `ccz q[a], q[b], q[c]` lowers to `Gate::Ccz` which sign-flips
        // the amplitude where all three qubits are |1⟩. The P1-08 dispatch
        // routes this through the specialised CCZ kernel.
        "ccz" => (G::Ccz, 0),
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy)]
enum GateShape {
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    Rx,
    Ry,
    Rz,
    Phase,
    U3,
    Cnot,
    Cz,
    Swap,
    Toffoli,
    Ccz,
}

fn build_gate_variant(shape: &GateShape, params: &[f64]) -> Gate {
    use GateShape as G;
    match shape {
        G::H => Gate::H,
        G::X => Gate::X,
        G::Y => Gate::Y,
        G::Z => Gate::Z,
        G::S => Gate::S,
        G::Sdg => Gate::Sdg,
        G::T => Gate::T,
        G::Tdg => Gate::Tdg,
        G::Rx => Gate::Rx(Param::Concrete(params[0])),
        G::Ry => Gate::Ry(Param::Concrete(params[0])),
        G::Rz => Gate::Rz(Param::Concrete(params[0])),
        G::Phase => Gate::Phase(Param::Concrete(params[0])),
        G::U3 => Gate::U3(
            Param::Concrete(params[0]),
            Param::Concrete(params[1]),
            Param::Concrete(params[2]),
        ),
        G::Cnot => Gate::Cnot,
        G::Cz => Gate::Cz,
        G::Swap => Gate::Swap,
        G::Toffoli => Gate::Toffoli,
        G::Ccz => Gate::Ccz,
    }
}

fn lower_barrier(
    circuit: &mut Circuit,
    regs: &RegisterMap,
    b: &BarrierStmt,
    source: &str,
) -> Result<(), ParseError> {
    let mut qubits: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
    for arg in &b.args {
        match arg {
            RegOrIdx::Indexed(r) => qubits.push(resolve_indexed(regs, r, source)?),
            RegOrIdx::Whole { pos, name } => match regs.qreg_size(name) {
                None => {
                    return Err(perr(
                        source,
                        *pos,
                        ParseErrorKind::UnknownRegister { name: name.clone() },
                    ));
                }
                Some((base, size)) => {
                    for i in 0..size {
                        qubits.push(base + i);
                    }
                }
            },
        }
    }
    circuit
        .add_instruction(Instruction::Barrier(qubits))
        .map_err(|e| perr(source, b.pos, ParseErrorKind::IrRejected(e)))?;
    Ok(())
}

fn lower_measure(
    circuit: &mut Circuit,
    regs: &RegisterMap,
    m: &MeasureStmt,
    source: &str,
) -> Result<(), ParseError> {
    match (&m.source, &m.target) {
        (RegOrIdx::Indexed(qr), RegOrIdx::Indexed(cr)) => {
            let q = resolve_indexed(regs, qr, source)?;
            let c = regs
                .resolve_clbit(&cr.name, cr.index)
                .map_err(|kind| perr(source, cr.pos, kind))?;
            circuit
                .add_instruction(Instruction::Measure { qubit: q, clbit: c })
                .map_err(|e| perr(source, m.pos, ParseErrorKind::IrRejected(e)))?;
            Ok(())
        }
        (
            RegOrIdx::Whole {
                pos: qpos,
                name: qname,
            },
            RegOrIdx::Whole {
                pos: cpos,
                name: cname,
            },
        ) => {
            let (qbase, qsize) = regs.qreg_size(qname).ok_or_else(|| {
                perr(
                    source,
                    *qpos,
                    ParseErrorKind::UnknownRegister {
                        name: qname.clone(),
                    },
                )
            })?;
            let (cbase, csize) = regs.creg_size(cname).ok_or_else(|| {
                perr(
                    source,
                    *cpos,
                    ParseErrorKind::UnknownRegister {
                        name: cname.clone(),
                    },
                )
            })?;
            if qsize != csize {
                return Err(perr(
                    source,
                    m.pos,
                    ParseErrorKind::SizeMismatch {
                        lhs: qname.clone(),
                        lhs_size: qsize,
                        rhs: cname.clone(),
                        rhs_size: csize,
                    },
                ));
            }
            for i in 0..qsize {
                circuit
                    .add_instruction(Instruction::Measure {
                        qubit: qbase + i,
                        clbit: cbase + i,
                    })
                    .map_err(|e| perr(source, m.pos, ParseErrorKind::IrRejected(e)))?;
            }
            Ok(())
        }
        _ => Err(perr(
            source,
            m.pos,
            ParseErrorKind::UnsupportedFeature {
                feature: "mixed whole-register / indexed measure",
            },
        )),
    }
}

fn lower_reset(
    circuit: &mut Circuit,
    regs: &RegisterMap,
    r: &ResetStmt,
    source: &str,
) -> Result<(), ParseError> {
    let q = resolve_indexed(regs, &r.target, source)?;
    circuit
        .add_instruction(Instruction::Reset(q))
        .map_err(|e| perr(source, r.pos, ParseErrorKind::IrRejected(e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Span;
    use crate::parser::program;

    fn parse_and_lower(src: &str) -> Result<Circuit, ParseError> {
        let (_, prog) = program(Span::new(src)).unwrap();
        lower(prog, src)
    }

    #[test]
    fn lowers_single_qubit_gate() {
        let c = parse_and_lower("qubit[1] q; h q[0];").unwrap();
        assert_eq!(c.num_qubits(), 1);
        assert_eq!(c.len(), 1);
        match &c.instructions()[0] {
            Instruction::Gate(g) => assert_eq!(g.gate, Gate::H),
            _ => panic!(),
        }
    }

    #[test]
    fn lowers_cnot_with_two_registers() {
        let c = parse_and_lower("qubit[2] q; qubit[1] aux; cx q[1], aux[0];").unwrap();
        assert_eq!(c.num_qubits(), 3);
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert_eq!(g.gate, Gate::Cnot);
                assert_eq!(g.qubits.as_slice(), &[1, 2]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn lowers_rx_with_pi_expression() {
        let c = parse_and_lower("qubit[1] q; rx(pi/2) q[0];").unwrap();
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert!(matches!(g.gate, Gate::Rx(Param::Concrete(theta))
                    if (theta - std::f64::consts::FRAC_PI_2).abs() < 1e-15));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rejects_unknown_gate() {
        let err = parse_and_lower("qubit[1] q; hadamard q[0];").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnknownGate { .. }));
    }

    #[test]
    fn rejects_undeclared_register() {
        let err = parse_and_lower("qubit[1] q; h r[0];").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnknownRegister { .. }));
    }

    #[test]
    fn rejects_oob_index() {
        let err = parse_and_lower("qubit[2] q; h q[5];").unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::IndexOutOfBounds {
                index: 5,
                size: 2,
                ..
            }
        ));
    }

    #[test]
    fn rejects_too_many_qubits_in_decl() {
        let big = MAX_QUBITS + 1;
        let src = format!("qubit[{big}] q;");
        let err = parse_and_lower(&src).unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::TooManyQubits {
                requested,
                max
            } if requested == big && max == MAX_QUBITS
        ));
    }

    #[test]
    fn sets_generated_from_metadata() {
        let c = parse_and_lower("qubit[1] q; h q[0];").unwrap();
        assert_eq!(c.metadata().generated_from.as_deref(), Some("openqasm:3.0"));
    }

    #[test]
    fn lowers_indexed_barrier() {
        let c = parse_and_lower("qubit[3] q; barrier q[0], q[2];").unwrap();
        match &c.instructions()[0] {
            Instruction::Barrier(qs) => assert_eq!(qs.as_slice(), &[0, 2]),
            _ => panic!(),
        }
    }

    #[test]
    fn lowers_whole_register_barrier() {
        let c = parse_and_lower("qubit[3] q; barrier q;").unwrap();
        match &c.instructions()[0] {
            Instruction::Barrier(qs) => assert_eq!(qs.as_slice(), &[0, 1, 2]),
            _ => panic!(),
        }
    }

    #[test]
    fn lowers_indexed_measure() {
        let c = parse_and_lower("qubit[1] q; bit[1] c; measure q[0] -> c[0];").unwrap();
        match &c.instructions()[0] {
            Instruction::Measure { qubit: 0, clbit: 0 } => {}
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn lowers_whole_register_measure() {
        let c = parse_and_lower("qubit[2] q; bit[2] c; measure q -> c;").unwrap();
        assert_eq!(c.len(), 2);
        assert!(matches!(
            c.instructions()[0],
            Instruction::Measure { qubit: 0, clbit: 0 }
        ));
        assert!(matches!(
            c.instructions()[1],
            Instruction::Measure { qubit: 1, clbit: 1 }
        ));
    }

    #[test]
    fn rejects_size_mismatch_in_whole_measure() {
        let err = parse_and_lower("qubit[2] q; bit[3] c; measure q -> c;").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::SizeMismatch { .. }));
    }

    #[test]
    fn lowers_reset() {
        let c = parse_and_lower("qubit[2] q; reset q[1];").unwrap();
        assert!(matches!(c.instructions()[0], Instruction::Reset(1)));
    }

    #[test]
    fn rejects_oob_clbit_in_measure() {
        let err = parse_and_lower("qubit[1] q; bit[1] c; measure q[0] -> c[5];").unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::IndexOutOfBounds {
                index: 5,
                size: 1,
                ..
            }
        ));
    }

    #[test]
    fn rejects_duplicate_qubit_in_gate() {
        // `cx q[0], q[0];` would trip GateInstance::new's debug_assert
        // and panic in debug builds without the pre-check in lower_gate.
        let err = parse_and_lower("qubit[2] q; cx q[0], q[0];").unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::IrRejected(aleph_ir::CircuitError::DuplicateQubit { qubit: 0 })
        ));
    }

    #[test]
    fn rejects_duplicate_qreg_name() {
        let err = parse_and_lower("qubit[1] q; qubit[2] q;").unwrap_err();
        match err.kind {
            ParseErrorKind::DuplicateRegister { name } => assert_eq!(name, "q"),
            other => panic!("expected DuplicateRegister, got {other:?}"),
        }
    }

    #[test]
    fn rejects_qreg_creg_name_collision() {
        let err = parse_and_lower("qubit[1] x; bit[1] x;").unwrap_err();
        match err.kind {
            ParseErrorKind::DuplicateRegister { name } => assert_eq!(name, "x"),
            other => panic!("expected DuplicateRegister, got {other:?}"),
        }
    }

    #[test]
    fn cname_position_for_whole_register_measure() {
        // Caret should point at the offending classical-register name
        // `bogus`, not at the `measure` keyword.
        let err = parse_and_lower("qubit[2] q;\nbit[2] c;\nmeasure q -> bogus;").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnknownRegister { .. }));
        assert_eq!(err.line, 3);
        // The `measure` keyword starts at column 1; `bogus` starts at column 14.
        // We don't assert exact col 14 because get_utf8_column is 1-based and
        // depends on the leading-whitespace handling — just assert >= 10.
        assert!(
            err.col >= 10,
            "col={}, expected to point near `bogus`",
            err.col
        );
    }
}
