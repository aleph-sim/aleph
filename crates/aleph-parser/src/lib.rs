//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>` (Task 15)
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod ast;
mod error;
mod expr;
mod lexer;
mod lower;
mod parser;

pub use error::{EmitError, ParseError, ParseErrorKind};

use lexer::Span;

/// Parse an OpenQASM 3.0 source string into an `aleph_ir::Circuit`.
pub fn parse(source: &str) -> Result<aleph_ir::Circuit, ParseError> {
    let input = Span::new(source);
    let (rest, program) = parser::program(input).map_err(|e| match e {
        nom::Err::Error(err) | nom::Err::Failure(err) => nom_error_to_parse_error(source, err),
        nom::Err::Incomplete(_) => ParseError::new(
            1,
            1,
            nth_line(source, 1),
            ParseErrorKind::UnexpectedToken {
                expected: "more input",
                found: "end of source".to_string(),
            },
        ),
    })?;
    // Reject any trailing non-whitespace input.
    let rest_str = rest.fragment().trim();
    if !rest_str.is_empty() {
        let line = rest.location_line();
        let col = rest.get_utf8_column() as u32;
        return Err(ParseError::new(
            line,
            col,
            nth_line(source, line),
            ParseErrorKind::UnexpectedToken {
                expected: "end of input",
                found: rest_str.chars().take(20).collect(),
            },
        ));
    }
    lower::lower(program, source)
}

fn nom_error_to_parse_error(source: &str, err: nom::error::Error<Span<'_>>) -> ParseError {
    let line = err.input.location_line();
    let col = err.input.get_utf8_column() as u32;
    let snippet = nth_line(source, line);
    let found: String = err
        .input
        .fragment()
        .chars()
        .take(20)
        .collect::<String>()
        .trim_end()
        .to_string();
    ParseError::new(
        line,
        col,
        snippet,
        ParseErrorKind::UnexpectedToken {
            expected: nom_kind_label(err.code),
            found: if found.is_empty() {
                "end of source".to_string()
            } else {
                found
            },
        },
    )
}

fn nom_kind_label(kind: nom::error::ErrorKind) -> &'static str {
    use nom::error::ErrorKind as K;
    match kind {
        K::Tag => "keyword or punctuation",
        K::Digit => "integer literal",
        K::Float => "numeric literal",
        K::AlphaNumeric | K::Alpha => "identifier",
        _ => "token",
    }
}

fn nth_line(source: &str, line: u32) -> String {
    source
        .lines()
        .nth(line.saturating_sub(1) as usize)
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bell_pair_source() {
        let src = r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let c = parse(src).unwrap();
        assert_eq!(c.num_qubits(), 2);
        assert_eq!(c.num_clbits(), 2);
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn parses_with_pi_expression() {
        let src = "qubit[1] q; rx(pi/2) q[0];";
        let c = parse(src).unwrap();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn rejects_trailing_junk() {
        let src = "qubit[1] q; h q[0]; @@@";
        let err = parse(src).unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::UnexpectedToken { .. }
        ));
    }

    #[test]
    fn empty_source_yields_zero_qubit_circuit() {
        let c = parse("").unwrap();
        assert_eq!(c.num_qubits(), 0);
        assert!(c.is_empty());
    }
}
