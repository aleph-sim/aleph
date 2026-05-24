//! `Circuit → String` emitter. Normalized output (see spec § 10): one
//! qubit register `q`, one clbit register `c`, one instruction per
//! line, no comments, expressions evaluated to `f64` literals.

use std::fmt::Write;

use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::{Circuit, Instruction};

use crate::error::EmitError;

/// Emit a `Circuit` as an OpenQASM 3 source string.
pub fn emit(circuit: &Circuit) -> Result<String, EmitError> {
    let mut out = String::new();
    out.push_str("OPENQASM 3.0;\n");
    out.push_str("include \"stdgates.inc\";\n\n");
    writeln!(out, "qubit[{}] q;", circuit.num_qubits()).unwrap();
    if circuit.num_clbits() > 0 {
        writeln!(out, "bit[{}] c;", circuit.num_clbits()).unwrap();
    }
    out.push('\n');
    for inst in circuit.instructions() {
        emit_instruction(&mut out, inst)?;
        out.push('\n');
    }
    Ok(out)
}

fn emit_instruction(out: &mut String, inst: &Instruction) -> Result<(), EmitError> {
    match inst {
        Instruction::Gate(g) => emit_gate(out, g),
        Instruction::Measure { qubit, clbit } => {
            write!(out, "measure q[{qubit}] -> c[{clbit}];").unwrap();
            Ok(())
        }
        Instruction::Reset(q) => {
            write!(out, "reset q[{q}];").unwrap();
            Ok(())
        }
        Instruction::Barrier(qs) => {
            out.push_str("barrier ");
            for (i, q) in qs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write!(out, "q[{q}]").unwrap();
            }
            out.push(';');
            Ok(())
        }
    }
}

fn emit_gate(out: &mut String, g: &GateInstance) -> Result<(), EmitError> {
    if !g.controls.is_empty() {
        return Err(EmitError::ExternalControls {
            count: g.controls.len(),
        });
    }
    let (name, params) = match &g.gate {
        Gate::H => ("h", vec![]),
        Gate::X => ("x", vec![]),
        Gate::Y => ("y", vec![]),
        Gate::Z => ("z", vec![]),
        Gate::S => ("s", vec![]),
        Gate::Sdg => ("sdg", vec![]),
        Gate::T => ("t", vec![]),
        Gate::Tdg => ("tdg", vec![]),
        Gate::Rx(p) => ("rx", vec![extract_concrete(p)?]),
        Gate::Ry(p) => ("ry", vec![extract_concrete(p)?]),
        Gate::Rz(p) => ("rz", vec![extract_concrete(p)?]),
        Gate::Phase(p) => ("p", vec![extract_concrete(p)?]),
        Gate::U3(a, b, c) => (
            "u3",
            vec![
                extract_concrete(a)?,
                extract_concrete(b)?,
                extract_concrete(c)?,
            ],
        ),
        Gate::Cnot => ("cx", vec![]),
        Gate::Cz => ("cz", vec![]),
        Gate::Swap => ("swap", vec![]),
        Gate::Iswap => return Err(EmitError::UnsupportedGate { name: "Iswap" }),
        Gate::IswapDg => return Err(EmitError::UnsupportedGate { name: "IswapDg" }),
        Gate::CRx(_) => return Err(EmitError::UnsupportedGate { name: "CRx" }),
        Gate::CRy(_) => return Err(EmitError::UnsupportedGate { name: "CRy" }),
        Gate::CRz(_) => return Err(EmitError::UnsupportedGate { name: "CRz" }),
        Gate::Toffoli => ("ccx", vec![]),
        Gate::Ccz => return Err(EmitError::UnsupportedGate { name: "Ccz" }),
        Gate::Unitary1q(_) => return Err(EmitError::UnsupportedGate { name: "Unitary1q" }),
        Gate::Unitary2q(_) => return Err(EmitError::UnsupportedGate { name: "Unitary2q" }),
    };
    out.push_str(name);
    if !params.is_empty() {
        out.push('(');
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write!(out, "{p}").unwrap();
        }
        out.push(')');
    }
    out.push(' ');
    for (i, q) in g.qubits.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write!(out, "q[{q}]").unwrap();
    }
    out.push(';');
    Ok(())
}

fn extract_concrete(p: &Param) -> Result<f64, EmitError> {
    match p {
        Param::Concrete(x) => Ok(*x),
        Param::Symbolic(_) => Err(EmitError::Symbolic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn emits_bell_pair() {
        let c = parse(
            "qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];",
        )
        .unwrap();
        let out = emit(&c).unwrap();
        assert!(out.starts_with("OPENQASM 3.0;\n"));
        assert!(out.contains("h q[0];"));
        assert!(out.contains("cx q[0], q[1];"));
        assert!(out.contains("measure q[0] -> c[0];"));
    }

    #[test]
    fn emits_parametric_gate_with_shortest_float() {
        let c = parse("qubit[1] q; rx(0.5) q[0];").unwrap();
        let out = emit(&c).unwrap();
        assert!(out.contains("rx(0.5) q[0];"));
    }

    #[test]
    fn emits_no_clbit_decl_when_zero() {
        let c = parse("qubit[1] q; h q[0];").unwrap();
        let out = emit(&c).unwrap();
        // No `bit[N] c;` line for zero clbits. The `qubit[1] q;` line
        // contains a substring "bit[" but is on a different line.
        assert!(!out.lines().any(|l| l.starts_with("bit[")));
    }

    #[test]
    fn round_trip_through_parse_emit_parse_preserves_instructions() {
        let src = "qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];";
        let c1 = parse(src).unwrap();
        let out = emit(&c1).unwrap();
        let c2 = parse(&out).unwrap();
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.instructions().iter().zip(c2.instructions().iter()) {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
    }
}
