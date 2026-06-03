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
        // DiagonalPhase is a post-optimization IR node; the parser never
        // produces it. Reject emission: a circuit containing this variant
        // should be lowered to gates before serialisation.
        Instruction::DiagonalPhase(_) => Err(EmitError::UnsupportedGate {
            name: "DiagonalPhase",
        }),
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
        g @ Gate::Iswap
        | g @ Gate::IswapDg
        | g @ Gate::CRx(_)
        | g @ Gate::CRy(_)
        | g @ Gate::CRz(_)
        | g @ Gate::Unitary1q(_)
        | g @ Gate::Unitary1qDiag(_)
        | g @ Gate::Unitary2q(_) => return Err(EmitError::UnsupportedGate { name: g.name() }),
        Gate::Toffoli => ("ccx", vec![]),
        Gate::Ccz => ("ccz", vec![]),
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
        Param::Concrete(x) if x.is_finite() => Ok(*x),
        Param::Concrete(x) => Err(EmitError::NonFiniteParam { value: *x }),
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
    fn rejects_nan_param() {
        use aleph_core::{Gate, GateInstance, Param};
        use smallvec::smallvec;
        let mut c = aleph_ir::Circuit::new(1, 0);
        c.add_gate(GateInstance::new(
            Gate::Rx(Param::Concrete(f64::NAN)),
            smallvec![0u32],
        ))
        .unwrap();
        let err = emit(&c).unwrap_err();
        assert!(matches!(err, EmitError::NonFiniteParam { value } if value.is_nan()));
    }

    #[test]
    fn rejects_inf_param() {
        use aleph_core::{Gate, GateInstance, Param};
        use smallvec::smallvec;
        let mut c = aleph_ir::Circuit::new(1, 0);
        c.add_gate(GateInstance::new(
            Gate::Rx(Param::Concrete(f64::INFINITY)),
            smallvec![0u32],
        ))
        .unwrap();
        let err = emit(&c).unwrap_err();
        assert!(matches!(err, EmitError::NonFiniteParam { value } if value.is_infinite()));
    }

    #[test]
    fn unitary_1q_diag_emits_unsupported() {
        use aleph_core::{Complex, Gate, GateInstance};
        use smallvec::smallvec;
        let g = Gate::Unitary1qDiag(Box::new([Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]));
        let mut c = aleph_ir::Circuit::new(1, 0);
        c.add_gate(GateInstance::new(g, smallvec![0u32])).unwrap();
        let err = emit(&c).unwrap_err();
        match err {
            EmitError::UnsupportedGate { name } => assert_eq!(name, "Unitary1qDiag"),
            other => panic!("expected UnsupportedGate(Unitary1qDiag), got {other:?}"),
        }
    }

    #[test]
    fn diagonal_phase_emits_unsupported() {
        // `DiagonalPhase` is a post-optimization IR node with no OpenQASM
        // surface syntax; emit must refuse it cleanly (not panic). It only
        // ever exists after `FuseDiagonalRuns`, never round-tripped.
        use aleph_ir::{DiagonalPhase, Instruction, PhaseTerm};
        use smallvec::smallvec;
        let dp = DiagonalPhase {
            n_qubits: 1,
            terms: vec![PhaseTerm {
                conds: smallvec![0b1u64],
                angle: std::f64::consts::FRAC_PI_4,
            }],
        };
        let mut c = aleph_ir::Circuit::new(1, 0);
        c.add_instruction(Instruction::DiagonalPhase(Box::new(dp)))
            .unwrap();
        let err = emit(&c).unwrap_err();
        match err {
            EmitError::UnsupportedGate { name } => assert_eq!(name, "DiagonalPhase"),
            other => panic!("expected UnsupportedGate(DiagonalPhase), got {other:?}"),
        }
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
