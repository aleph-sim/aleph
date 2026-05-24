//! nom combinators that build [`crate::ast::Program`].
//!
//! Errors are surfaced as nom errors at this layer; the top-level
//! `parse` function (Task 13) converts them into `ParseError`.

use nom::IResult;
use nom::Parser;
use nom::bytes::complete::tag;
use nom::character::complete::char as ch;
use nom::combinator::opt;
use nom::multi::many0;

use crate::ast::{
    BarrierStmt, Decl, GateStmt, IndexedRef, Include, MeasureStmt, Position, Program, RegOrIdx,
    ResetStmt, Stmt,
};
use crate::expr::expr as parse_expr;
use crate::lexer::{Span, ident, skip_ws, string_literal, uint};

/// Capture the 1-based (line, col) of the *current* position in `input`.
pub fn pos_of(input: &Span<'_>) -> Position {
    Position {
        line: input.location_line(),
        col: input.get_utf8_column() as u32,
    }
}

/// Parse the full `Program`.
pub fn program(input: Span<'_>) -> IResult<Span<'_>, Program> {
    let (input, header_version) = opt(header).parse(input)?;
    let (input, includes) = many0(include_stmt).parse(input)?;
    let (input, decls) = many0(decl).parse(input)?;
    let (input, stmts) = many0(stmt).parse(input)?;
    let (input, _) = skip_ws(input)?;
    Ok((
        input,
        Program {
            header_version,
            includes,
            decls,
            stmts,
        },
    ))
}

fn qreg_decl(input: Span<'_>) -> IResult<Span<'_>, Decl> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("qubit").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("[").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, size) = uint(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("]").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, name) = ident(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    Ok((
        input,
        Decl::Qreg {
            pos: p,
            name: name.fragment().to_string(),
            size,
        },
    ))
}

fn creg_decl(input: Span<'_>) -> IResult<Span<'_>, Decl> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("bit").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("[").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, size) = uint(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("]").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, name) = ident(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    Ok((
        input,
        Decl::Creg {
            pos: p,
            name: name.fragment().to_string(),
            size,
        },
    ))
}

fn decl(input: Span<'_>) -> IResult<Span<'_>, Decl> {
    nom::branch::alt((qreg_decl, creg_decl)).parse(input)
}

fn indexed_ref(input: Span<'_>) -> IResult<Span<'_>, IndexedRef> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, name) = ident(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("[").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, index) = uint(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("]").parse(input)?;
    Ok((
        input,
        IndexedRef {
            pos: p,
            name: name.fragment().to_string(),
            index,
        },
    ))
}

fn reg_or_idx(input: Span<'_>) -> IResult<Span<'_>, RegOrIdx> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, name) = ident(input)?;
    // Optional `[index]` follows.
    let after_name = input;
    let (input, _) = skip_ws(input)?;
    if let Ok((input, _)) = tag::<_, _, nom::error::Error<Span<'_>>>("[").parse(input) {
        let (input, _) = skip_ws(input)?;
        let (input, index) = uint(input)?;
        let (input, _) = skip_ws(input)?;
        let (input, _) = tag("]").parse(input)?;
        Ok((
            input,
            RegOrIdx::Indexed(IndexedRef {
                pos: p,
                name: name.fragment().to_string(),
                index,
            }),
        ))
    } else {
        Ok((
            after_name,
            RegOrIdx::Whole {
                pos: p,
                name: name.fragment().to_string(),
            },
        ))
    }
}

/// Parse an expression and unwrap the Result<f64, String> payload —
/// on Err, return a nom Failure so the top-level error converter can
/// surface a positioned error. (Spec § 9: bad expressions like
/// `1/0` are rejected; exact error text is best-effort.)
fn expr_value(input: Span<'_>) -> IResult<Span<'_>, f64> {
    let saved = input;
    let (input, v) = parse_expr(input)?;
    match v {
        Ok(x) => Ok((input, x)),
        Err(_) => Err(nom::Err::Failure(nom::error::Error::new(
            saved,
            nom::error::ErrorKind::Float,
        ))),
    }
}

fn gate_params(input: Span<'_>) -> IResult<Span<'_>, Vec<f64>> {
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("(").parse(input)?;
    let (mut input, first) = expr_value(input)?;
    let mut values = vec![first];
    loop {
        let saved = input;
        let (next, _) = skip_ws(input)?;
        if let Ok((next, _)) = tag::<_, _, nom::error::Error<Span<'_>>>(",").parse(next) {
            let (next, v) = expr_value(next)?;
            values.push(v);
            input = next;
        } else {
            input = saved;
            break;
        }
    }
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(")").parse(input)?;
    Ok((input, values))
}

fn gate_stmt(input: Span<'_>) -> IResult<Span<'_>, GateStmt> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, name) = ident(input)?;
    let (input, params) = opt(gate_params).parse(input)?;
    let (input, first) = indexed_ref(input)?;
    let (mut input, mut args) = (input, vec![first]);
    loop {
        let saved = input;
        let (next, _) = skip_ws(input)?;
        if let Ok((next, _)) = tag::<_, _, nom::error::Error<Span<'_>>>(",").parse(next) {
            let (next, r) = indexed_ref(next)?;
            args.push(r);
            input = next;
        } else {
            input = saved;
            break;
        }
    }
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    Ok((
        input,
        GateStmt {
            pos: p,
            name: name.fragment().to_string(),
            params: params.unwrap_or_default(),
            args,
        },
    ))
}

fn barrier_stmt(input: Span<'_>) -> IResult<Span<'_>, BarrierStmt> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("barrier").parse(input)?;
    // Require at least one whitespace/comment after the keyword to
    // avoid matching `barrierfoo` etc.
    let (input, _) = nom::character::complete::multispace1.parse(input)?;
    let (input, first) = reg_or_idx(input)?;
    let (mut input, mut args) = (input, vec![first]);
    loop {
        let saved = input;
        let (next, _) = skip_ws(input)?;
        if let Ok((next, _)) = tag::<_, _, nom::error::Error<Span<'_>>>(",").parse(next) {
            let (next, r) = reg_or_idx(next)?;
            args.push(r);
            input = next;
        } else {
            input = saved;
            break;
        }
    }
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    Ok((input, BarrierStmt { pos: p, args }))
}

fn measure_stmt(input: Span<'_>) -> IResult<Span<'_>, MeasureStmt> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("measure").parse(input)?;
    let (input, _) = nom::character::complete::multispace1.parse(input)?;
    let (input, source) = reg_or_idx(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("->").parse(input)?;
    let (input, target) = reg_or_idx(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    Ok((
        input,
        MeasureStmt {
            pos: p,
            source,
            target,
        },
    ))
}

fn reset_stmt(input: Span<'_>) -> IResult<Span<'_>, ResetStmt> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("reset").parse(input)?;
    let (input, _) = nom::character::complete::multispace1.parse(input)?;
    let (input, target) = indexed_ref(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    Ok((input, ResetStmt { pos: p, target }))
}

fn stmt(input: Span<'_>) -> IResult<Span<'_>, Stmt> {
    let (input, _) = skip_ws(input)?;
    nom::branch::alt((
        nom::combinator::map(barrier_stmt, Stmt::Barrier),
        nom::combinator::map(measure_stmt, Stmt::Measure),
        nom::combinator::map(reset_stmt, Stmt::Reset),
        nom::combinator::map(gate_stmt, Stmt::Gate),
    ))
    .parse(input)
}

fn header(input: Span<'_>) -> IResult<Span<'_>, String> {
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag("OPENQASM").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, major) = uint(input)?;
    let (input, minor) = opt(|i| {
        let (i, _) = ch('.').parse(i)?;
        uint(i)
    })
    .parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    let version = if let Some(m) = minor {
        format!("{major}.{m}")
    } else {
        format!("{major}")
    };
    Ok((input, version))
}

fn include_stmt(input: Span<'_>) -> IResult<Span<'_>, Include> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("include").parse(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, path) = string_literal(input)?;
    let (input, _) = skip_ws(input)?;
    let (input, _) = tag(";").parse(input)?;
    Ok((
        input,
        Include {
            pos: p,
            path: path.fragment().to_string(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(s: &str) -> Span<'_> {
        Span::new(s)
    }

    #[test]
    fn header_with_version() {
        let (_, prog) = program(sp("OPENQASM 3.0;")).unwrap();
        assert_eq!(prog.header_version.as_deref(), Some("3.0"));
    }

    #[test]
    fn header_then_include() {
        let src = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";
        let (_, prog) = program(sp(src)).unwrap();
        assert_eq!(prog.header_version.as_deref(), Some("3.0"));
        assert_eq!(prog.includes.len(), 1);
        assert_eq!(prog.includes[0].path, "stdgates.inc");
        assert_eq!(prog.includes[0].pos.line, 2);
    }

    #[test]
    fn no_header_is_ok() {
        let (_, prog) = program(sp("")).unwrap();
        assert!(prog.header_version.is_none());
        assert!(prog.includes.is_empty());
    }

    #[test]
    fn parses_qubit_decl() {
        let (_, prog) = program(sp("qubit[3] q;")).unwrap();
        assert_eq!(prog.decls.len(), 1);
        match &prog.decls[0] {
            Decl::Qreg { name, size, .. } => {
                assert_eq!(name, "q");
                assert_eq!(*size, 3);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_bit_decl() {
        let (_, prog) = program(sp("bit[2] c;")).unwrap();
        match &prog.decls[0] {
            Decl::Creg { name, size, .. } => {
                assert_eq!(name, "c");
                assert_eq!(*size, 2);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_multiple_decls() {
        let src = "qubit[2] q;\nqubit[3] aux;\nbit[5] c;";
        let (_, prog) = program(sp(src)).unwrap();
        assert_eq!(prog.decls.len(), 3);
    }

    #[test]
    fn parses_header_includes_decls() {
        let src = r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;"#;
        let (_, prog) = program(sp(src)).unwrap();
        assert_eq!(prog.header_version.as_deref(), Some("3.0"));
        assert_eq!(prog.includes.len(), 1);
        assert_eq!(prog.decls.len(), 2);
    }

    #[test]
    fn parses_single_qubit_gate() {
        let (_, prog) = program(sp("qubit[1] q; h q[0];")).unwrap();
        assert_eq!(prog.stmts.len(), 1);
        match &prog.stmts[0] {
            Stmt::Gate(g) => {
                assert_eq!(g.name, "h");
                assert_eq!(g.params.len(), 0);
                assert_eq!(g.args.len(), 1);
                assert_eq!(g.args[0].index, 0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_two_qubit_gate() {
        let (_, prog) = program(sp("qubit[2] q; cx q[0], q[1];")).unwrap();
        match &prog.stmts[0] {
            Stmt::Gate(g) => {
                assert_eq!(g.name, "cx");
                assert_eq!(g.args.len(), 2);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_parametric_gate() {
        let (_, prog) = program(sp("qubit[1] q; rx(pi/2) q[0];")).unwrap();
        match &prog.stmts[0] {
            Stmt::Gate(g) => {
                assert_eq!(g.name, "rx");
                assert_eq!(g.params.len(), 1);
                assert!((g.params[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_u3_with_three_params() {
        let (_, prog) = program(sp("qubit[1] q; u3(0.1, 0.2, 0.3) q[0];")).unwrap();
        match &prog.stmts[0] {
            Stmt::Gate(g) => {
                assert_eq!(g.params, vec![0.1, 0.2, 0.3]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_indexed_barrier() {
        let (_, prog) = program(sp("qubit[2] q; barrier q[0], q[1];")).unwrap();
        match &prog.stmts[0] {
            Stmt::Barrier(b) => {
                assert_eq!(b.args.len(), 2);
                match &b.args[0] {
                    RegOrIdx::Indexed(r) => assert_eq!(r.index, 0),
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_whole_register_barrier() {
        let (_, prog) = program(sp("qubit[2] q; barrier q;")).unwrap();
        match &prog.stmts[0] {
            Stmt::Barrier(b) => {
                assert_eq!(b.args.len(), 1);
                match &b.args[0] {
                    RegOrIdx::Whole { name, .. } => assert_eq!(name, "q"),
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_indexed_measure() {
        let (_, prog) = program(sp("qubit[1] q; bit[1] c; measure q[0] -> c[0];")).unwrap();
        match &prog.stmts[0] {
            Stmt::Measure(m) => {
                assert!(matches!(&m.source, RegOrIdx::Indexed(_)));
                assert!(matches!(&m.target, RegOrIdx::Indexed(_)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_whole_register_measure() {
        let (_, prog) = program(sp("qubit[2] q; bit[2] c; measure q -> c;")).unwrap();
        match &prog.stmts[0] {
            Stmt::Measure(m) => {
                assert!(matches!(&m.source, RegOrIdx::Whole { .. }));
                assert!(matches!(&m.target, RegOrIdx::Whole { .. }));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_reset() {
        let (_, prog) = program(sp("qubit[1] q; reset q[0];")).unwrap();
        match &prog.stmts[0] {
            Stmt::Reset(r) => assert_eq!(r.target.index, 0),
            _ => panic!(),
        }
    }

    #[test]
    fn parses_mixed_program() {
        let src = r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
barrier q;
measure q -> c;
"#;
        let (_, prog) = program(sp(src)).unwrap();
        assert_eq!(prog.stmts.len(), 4);
    }
}
