//! Public error types for the parser.
//!
//! `ParseError` carries an absolute 1-based line/column and a private
//! snippet of the offending source line so [`ParseError::render`] can
//! produce a rustc-style three-line block.

use thiserror::Error;

/// Failure during `parse(&str)`.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("error at {line}:{col}: {kind}")]
pub struct ParseError {
    pub line: u32,
    pub col: u32,
    pub kind: ParseErrorKind,
    snippet: String,
}

impl ParseError {
    /// Construct a `ParseError`. The `snippet` should be the full
    /// source line containing the error (no trailing newline).
    pub(crate) fn new(
        line: u32,
        col: u32,
        snippet: impl Into<String>,
        kind: ParseErrorKind,
    ) -> Self {
        Self {
            line,
            col,
            kind,
            snippet: snippet.into(),
        }
    }

    /// Render the error as a three-line block: a header (`error at L:C: <kind>`),
    /// the offending source line, and a `^`-underlined caret pointing at the
    /// column. Newline-terminated.
    pub fn render(&self) -> String {
        let mut out = format!("error at {}:{}: {}\n", self.line, self.col, self.kind);
        out.push_str(&self.snippet);
        out.push('\n');
        // `col` is 1-based; the caret needs col-1 spaces.
        let pad = self.col.saturating_sub(1) as usize;
        out.push_str(&" ".repeat(pad));
        out.push_str("^\n");
        out
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseErrorKind {
    #[error("unexpected token: expected {expected}, found `{found}`")]
    UnexpectedToken {
        expected: &'static str,
        found: String,
    },

    #[error("unknown gate `{name}`")]
    UnknownGate { name: String },

    #[error("undeclared register `{name}`")]
    UnknownRegister { name: String },

    #[error("register `{name}` is declared more than once")]
    DuplicateRegister { name: String },

    #[error("index {index} out of bounds for register `{register}` of size {size}")]
    IndexOutOfBounds {
        register: String,
        index: u32,
        size: u32,
    },

    #[error("size mismatch: `{lhs}` has {lhs_size} but `{rhs}` has {rhs_size}")]
    SizeMismatch {
        lhs: String,
        lhs_size: u32,
        rhs: String,
        rhs_size: u32,
    },

    #[error("bad expression: {0}")]
    BadExpression(String),

    #[error("unsupported feature: {feature}")]
    UnsupportedFeature { feature: &'static str },

    #[error("too many qubits: declared {requested}, max {max}")]
    TooManyQubits { requested: u32, max: u32 },

    #[error("too many clbits: declared {requested}, max {max}")]
    TooManyClbits { requested: u32, max: u32 },

    #[error("IR rejected this program: {0}")]
    IrRejected(aleph_ir::CircuitError),
}

/// Failure during `emit(&Circuit)`.
///
/// `Eq` is not derived because `NonFiniteParam` carries an `f64`.
#[derive(Debug, Error, PartialEq)]
pub enum EmitError {
    #[error("gate `{name}` has no OpenQASM 3 standard-subset representation")]
    UnsupportedGate { name: &'static str },

    #[error("symbolic parameter cannot be emitted (only Param::Concrete supported)")]
    Symbolic,

    #[error("non-finite parameter ({value}) cannot be emitted as OpenQASM literal")]
    NonFiniteParam { value: f64 },

    #[error("external controls (count = {count}) cannot be emitted in the standard subset")]
    ExternalControls { count: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_three_lines_with_caret_at_col() {
        let e = ParseError::new(
            3,
            5,
            "  x q[42];",
            ParseErrorKind::UnknownGate {
                name: "x".to_string(),
            },
        );
        let out = e.render();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "error at 3:5: unknown gate `x`");
        assert_eq!(lines[1], "  x q[42];");
        assert_eq!(lines[2], "    ^"); // 4 spaces + caret (col=5, pad=4)
    }

    #[test]
    fn render_caret_at_col_1() {
        let e = ParseError::new(
            1,
            1,
            "bad",
            ParseErrorKind::UnexpectedToken {
                expected: "keyword",
                found: "bad".to_string(),
            },
        );
        let out = e.render();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[2], "^");
    }

    #[test]
    fn ir_rejected_kind_displays_inner_error() {
        let kind = ParseErrorKind::IrRejected(aleph_ir::CircuitError::QubitOutOfRange {
            qubit: 5,
            num_qubits: 2,
        });
        let msg = format!("{kind}");
        assert!(msg.contains("qubit 5 out of range"));
    }
}
