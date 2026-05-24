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

use crate::ast::{Decl, Include, Position, Program};
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
    let (input, _) = skip_ws(input)?;
    Ok((
        input,
        Program {
            header_version,
            includes,
            decls,
            stmts: Vec::new(),
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
}
