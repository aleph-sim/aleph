//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>` (Task 15)
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod ast;
mod emit;
mod error;
mod expr;
mod lexer;
mod lower;
mod parser;

pub use error::{EmitError, ParseError, ParseErrorKind};

/// Emit an `aleph_ir::Circuit` as OpenQASM 3 source (normalized).
pub fn emit(circuit: &aleph_ir::Circuit) -> Result<String, EmitError> {
    emit::emit(circuit)
}

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
        let snippet = nth_line(source, line);
        // Detect interleaved register declarations.
        if rest_str.starts_with("qubit") || rest_str.starts_with("bit") {
            return Err(ParseError::new(
                line,
                col,
                snippet,
                ParseErrorKind::UnsupportedFeature {
                    feature: "register declaration after gate statement",
                },
            ));
        }
        // Detect other unsupported keywords by inspecting the snippet.
        if let Some(feature) = sniff_unsupported_feature(&snippet, col) {
            return Err(ParseError::new(
                line,
                col,
                snippet,
                ParseErrorKind::UnsupportedFeature { feature },
            ));
        }
        return Err(ParseError::new(
            line,
            col,
            snippet,
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
    // Best-effort detection of known unsupported features by inspecting
    // the source line at the failure position. Cheaper than refactoring
    // the parser to emit `UnsupportedFeature` at each construct site.
    if let Some(feature) = sniff_unsupported_feature(&snippet, col) {
        return ParseError::new(
            line,
            col,
            snippet,
            ParseErrorKind::UnsupportedFeature { feature },
        );
    }
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

/// Recognise common out-of-scope OpenQASM constructs by their leading
/// keyword. Returns the spec-§2.2 feature label so the caller can
/// surface `ParseErrorKind::UnsupportedFeature` instead of a generic
/// `UnexpectedToken`. `col` is the 1-based column of the failure
/// position within `line`.
fn sniff_unsupported_feature(line: &str, col: u32) -> Option<&'static str> {
    // Look at the source from the failure column onward — that's where
    // the parser tripped, so any unsupported keyword starts there.
    let at_col = line.get(col.saturating_sub(1) as usize..).unwrap_or("");
    let scan = at_col.trim_start();
    let starts = [
        ("if(", "classical control flow (if)"),
        ("if ", "classical control flow (if)"),
        ("gphase", "global phase (gphase)"),
        ("gate ", "custom gate definition"),
        ("def ", "subroutine definition (def)"),
        ("box ", "box block"),
        ("box{", "box block"),
        ("delay ", "delay statement"),
        ("delay[", "delay statement"),
        ("U(", "U(...) gate — use u3 instead"),
    ];
    for (prefix, feature) in starts {
        if scan.starts_with(prefix) {
            return Some(feature);
        }
    }
    None
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
        assert!(matches!(err.kind, ParseErrorKind::UnexpectedToken { .. }));
    }

    #[test]
    fn empty_source_yields_zero_qubit_circuit() {
        let c = parse("").unwrap();
        assert_eq!(c.num_qubits(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn header_missing_semicolon_points_at_header() {
        let err = parse("OPENQASM 3.0\nqubit[1] q;").unwrap_err();
        // Without `cut`, this surfaced as `error at 1:1: expected end
        // of input, found OPENQASM 3.0`. With `cut`, the failure lands
        // at the missing-`;` position (after whitespace consumption
        // that may move us to line 2, col 1 — i.e., the start of the
        // `qubit` token where `;` was expected). Either way the
        // diagnostic is no longer pinned to (1, 1) with the misleading
        // "found OPENQASM 3.0" snippet.
        let pos_ok = (err.line, err.col) != (1, 1);
        let msg_ok = match &err.kind {
            ParseErrorKind::UnexpectedToken { found, .. } => !found.contains("OPENQASM"),
            _ => true,
        };
        assert!(
            pos_ok || msg_ok,
            "error is still pinned to (1,1) with OPENQASM in message: {err:?}"
        );
    }

    #[test]
    fn rejects_openqasm_v2_with_clear_message() {
        let err = parse("OPENQASM 2.0;").unwrap_err();
        // Old behaviour: parser accepted it silently and failed later.
        // New behaviour: positioned Failure at the major version.
        assert_eq!(err.line, 1);
    }

    #[test]
    fn detects_interleaved_register_decl() {
        let err = parse("qubit[1] q; h q[0]; qubit[1] aux;").unwrap_err();
        match err.kind {
            ParseErrorKind::UnsupportedFeature { feature } => {
                assert!(feature.contains("declaration after"));
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    #[test]
    fn detects_unsupported_if_keyword() {
        let err = parse("qubit[1] q; bit[1] c; if(c == 0) x q[0];").unwrap_err();
        match err.kind {
            ParseErrorKind::UnsupportedFeature { feature } => {
                assert!(feature.contains("if"));
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    #[test]
    fn detects_unsupported_gate_definition() {
        let err = parse("gate foo a, b { x a; }").unwrap_err();
        match err.kind {
            ParseErrorKind::UnsupportedFeature { feature } => {
                assert!(feature.contains("custom gate") || feature.contains("gate "));
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }
}
