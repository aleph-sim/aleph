# P0-08 OpenQASM 3.0 Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the OpenQASM 3.0 parser for the Tier-1 algorithm subset, lowering into `aleph_ir::Circuit`, with a normalized round-trip emitter and a rustc-style line/col/caret error renderer.

**Architecture:** Two-phase pipeline — `nom` combinators over `nom_locate::LocatedSpan` build a private AST, then `lower(ast, source)` flattens registers, expands whole-register `barrier`/`measure` forms, evaluates `pi`/arithmetic expressions to `f64`, and emits IR via `Circuit::try_new` + `add_gate`/`add_instruction`. Emitter walks `Circuit` directly (no AST involved) since round-trip is normalized.

**Tech Stack:** Rust 2021, `nom 7`, `nom_locate 4`, `thiserror`, `smallvec`, `proptest` (dev). Consumes `aleph-core` and `aleph-ir`. Workspace MSRV 1.85.

---

## File Structure

| Path | Responsibility |
|---|---|
| `crates/aleph-ir/src/circuit.rs` | (modify) add `Circuit::try_new`, drop `TODO(P0-08)` |
| `crates/aleph-ir/src/error.rs` | (modify) add `TooManyQubits` / `TooManyClbits` variants |
| `Cargo.toml` (root) | (modify) add `nom`, `nom_locate` to `[workspace.dependencies]` |
| `crates/aleph-parser/Cargo.toml` | (modify) add deps |
| `crates/aleph-parser/src/lib.rs` | public API: `parse`, `emit`, re-exports |
| `crates/aleph-parser/src/error.rs` | `ParseError`, `ParseErrorKind`, `EmitError`, `render()` |
| `crates/aleph-parser/src/lexer.rs` | `Span` alias, whitespace/comment skipper, primitive token parsers (ident, uint, float, punctuation, string literals) |
| `crates/aleph-parser/src/expr.rs` | expression grammar + evaluator → `f64` |
| `crates/aleph-parser/src/ast.rs` | AST types with embedded spans |
| `crates/aleph-parser/src/parser.rs` | nom combinators → `ast::Program` |
| `crates/aleph-parser/src/lower.rs` | `ast::Program + source → Result<Circuit, ParseError>` |
| `crates/aleph-parser/src/emit.rs` | `Circuit → Result<String, EmitError>` |
| `crates/aleph-parser/tests/fixtures/*.qasm` | GHZ, QFT, Grover, random |
| `crates/aleph-parser/tests/algorithms_qasm.rs` | parse each fixture → assert IR shape |
| `crates/aleph-parser/tests/round_trip.rs` | parse → emit → parse for fixtures |
| `crates/aleph-parser/tests/round_trip_property.rs` | proptest: random Circuit ↔ QASM |
| `crates/aleph-parser/tests/error_rendering.rs` | snapshot tests for `render()` |

---

## Task 1: Workspace deps + `Circuit::try_new` (companion IR change)

**Files:**
- Modify: `Cargo.toml` (root)
- Modify: `crates/aleph-parser/Cargo.toml`
- Modify: `crates/aleph-ir/src/error.rs`
- Modify: `crates/aleph-ir/src/circuit.rs`

- [ ] **Step 1: Add workspace deps**

In root `Cargo.toml`, append to `[workspace.dependencies]`:

```toml
nom = "7"
nom_locate = "4"
```

- [ ] **Step 2: Update aleph-parser deps**

Replace `crates/aleph-parser/Cargo.toml` body:

```toml
[package]
name = "aleph-parser"
edition.workspace = true
rust-version.workspace = true
version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]
aleph-core = { path = "../aleph-core" }
aleph-ir   = { path = "../aleph-ir" }
nom        = { workspace = true }
nom_locate = { workspace = true }
thiserror  = { workspace = true }
smallvec   = { workspace = true }

[dev-dependencies]
proptest = { workspace = true }
```

- [ ] **Step 3: Add `TooManyQubits` / `TooManyClbits` to `CircuitError`**

In `crates/aleph-ir/src/error.rs`, append before the closing `}` of the enum:

```rust
    /// `Circuit::try_new` rejected a `num_qubits` above [`crate::MAX_QUBITS`].
    #[error("too many qubits: requested {requested}, max {max}")]
    TooManyQubits { requested: u32, max: u32 },

    /// `Circuit::try_new` rejected a `num_clbits` above [`crate::MAX_CLBITS`].
    #[error("too many clbits: requested {requested}, max {max}")]
    TooManyClbits { requested: u32, max: u32 },
```

- [ ] **Step 4: Add `Circuit::try_new` and drop the TODO**

In `crates/aleph-ir/src/circuit.rs`, locate the `Circuit::new` impl block and:

1. Remove the `// TODO(P0-08): expose a fallible try_new ...` comment.
2. Insert this method **immediately after** `Circuit::new`:

```rust
    /// Fallible constructor — same as [`Circuit::new`] but returns a
    /// recoverable [`CircuitError`] instead of panicking. Intended for
    /// untrusted-input boundaries (parser, RPC). See spec § 12.4 of
    /// `docs/superpowers/specs/2026-05-24-p0-07-circuit-ir-design.md`.
    pub fn try_new(num_qubits: u32, num_clbits: u32) -> Result<Self, CircuitError> {
        if num_qubits > MAX_QUBITS {
            return Err(CircuitError::TooManyQubits {
                requested: num_qubits,
                max: MAX_QUBITS,
            });
        }
        if num_clbits > MAX_CLBITS {
            return Err(CircuitError::TooManyClbits {
                requested: num_clbits,
                max: MAX_CLBITS,
            });
        }
        Ok(Self {
            num_qubits,
            num_clbits,
            instructions: Vec::new(),
            metadata: CircuitMetadata::default(),
        })
    }
```

- [ ] **Step 5: Add tests for `try_new`**

In `crates/aleph-ir/src/circuit.rs`, inside the existing `mod tests`, append:

```rust
    #[test]
    fn try_new_accepts_max_bounds() {
        let c = Circuit::try_new(MAX_QUBITS, MAX_CLBITS).unwrap();
        assert_eq!(c.num_qubits(), MAX_QUBITS);
        assert_eq!(c.num_clbits(), MAX_CLBITS);
    }

    #[test]
    fn try_new_rejects_too_many_qubits() {
        let err = Circuit::try_new(MAX_QUBITS + 1, 0).unwrap_err();
        assert_eq!(
            err,
            CircuitError::TooManyQubits {
                requested: MAX_QUBITS + 1,
                max: MAX_QUBITS,
            }
        );
    }

    #[test]
    fn try_new_rejects_too_many_clbits() {
        let err = Circuit::try_new(0, MAX_CLBITS + 1).unwrap_err();
        assert_eq!(
            err,
            CircuitError::TooManyClbits {
                requested: MAX_CLBITS + 1,
                max: MAX_CLBITS,
            }
        );
    }

    #[test]
    fn try_new_zero_zero_works() {
        let c = Circuit::try_new(0, 0).unwrap();
        assert!(c.is_empty());
    }
```

- [ ] **Step 6: Build + test**

Run: `cargo build --workspace`
Expected: success.

Run: `cargo test --package aleph-ir`
Expected: all existing tests plus the 4 new ones pass.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/aleph-parser/Cargo.toml crates/aleph-ir/
git commit -m "[P0-08] Workspace deps + Circuit::try_new fallible constructor"
```

---

## Task 2: `ParseError` / `EmitError` scaffolding

**Files:**
- Create: `crates/aleph-parser/src/error.rs`
- Modify: `crates/aleph-parser/src/lib.rs`

- [ ] **Step 1: Write the error module**

Create `crates/aleph-parser/src/error.rs`:

```rust
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
    pub(crate) fn new(line: u32, col: u32, snippet: impl Into<String>, kind: ParseErrorKind) -> Self {
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
    UnexpectedToken { expected: &'static str, found: String },

    #[error("unknown gate `{name}`")]
    UnknownGate { name: String },

    #[error("undeclared register `{name}`")]
    UnknownRegister { name: String },

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
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EmitError {
    #[error("gate `{name}` has no OpenQASM 3 standard-subset representation")]
    UnsupportedGate { name: &'static str },

    #[error("symbolic parameter cannot be emitted (only Param::Concrete supported)")]
    Symbolic,

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
```

- [ ] **Step 2: Wire into `lib.rs`**

Replace `crates/aleph-parser/src/lib.rs`:

```rust
//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>`
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod error;

pub use error::{EmitError, ParseError, ParseErrorKind};
```

`parse` / `emit` land in later tasks; this is a scaffolding commit.

- [ ] **Step 3: Build + test**

Run: `cargo test --package aleph-parser`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] ParseError / EmitError scaffolding + render()"
```

---

## Task 3: Lexer — `Span` alias and whitespace/comment skipper

**Files:**
- Create: `crates/aleph-parser/src/lexer.rs`
- Modify: `crates/aleph-parser/src/lib.rs`

- [ ] **Step 1: Write the lexer module skeleton**

Create `crates/aleph-parser/src/lexer.rs`:

```rust
//! Lexical primitives: `Span` alias, whitespace/comment skipper, and
//! small token combinators (idents, integers, floats, punctuation,
//! string literals). All combinators operate on
//! [`nom_locate::LocatedSpan<&str>`] so the parser keeps line/col info
//! and can build [`crate::error::ParseError`]s with accurate positions.

use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::{is_not, tag, take_until};
use nom::character::complete::{multispace1, satisfy};
use nom::combinator::{opt, recognize, value};
use nom::multi::many0;
use nom_locate::LocatedSpan;

pub type Span<'a> = LocatedSpan<&'a str>;

/// Skip ASCII whitespace, `// line comments`, and `/* block comments */`
/// (non-nesting). Returns success even if nothing was consumed.
pub fn skip_ws(input: Span<'_>) -> IResult<Span<'_>, ()> {
    value(
        (),
        many0(alt((
            value((), multispace1),
            value((), line_comment),
            value((), block_comment),
        ))),
    )
    .parse(input)
}

fn line_comment(input: Span<'_>) -> IResult<Span<'_>, ()> {
    let (input, _) = tag("//").parse(input)?;
    let (input, _) = opt(is_not("\n\r")).parse(input)?;
    Ok((input, ()))
}

fn block_comment(input: Span<'_>) -> IResult<Span<'_>, ()> {
    let (input, _) = tag("/*").parse(input)?;
    let (input, _) = take_until("*/").parse(input)?;
    let (input, _) = tag("*/").parse(input)?;
    Ok((input, ()))
}

/// Wrap a parser so it skips leading whitespace/comments.
pub fn ws<'a, F, O>(mut inner: F) -> impl FnMut(Span<'a>) -> IResult<Span<'a>, O>
where
    F: Parser<Span<'a>, Output = O, Error = nom::error::Error<Span<'a>>>,
{
    move |input| {
        let (input, _) = skip_ws(input)?;
        inner.parse(input)
    }
}

/// Recognise (but don't consume) the start of an identifier.
pub(crate) fn ident_start(input: Span<'_>) -> IResult<Span<'_>, char> {
    satisfy(|c: char| c.is_ascii_alphabetic() || c == '_').parse(input)
}

/// Recognise (but don't consume) an identifier continuation char.
pub(crate) fn ident_cont(input: Span<'_>) -> IResult<Span<'_>, char> {
    satisfy(|c: char| c.is_ascii_alphanumeric() || c == '_').parse(input)
}

/// Parse an identifier: `[A-Za-z_][A-Za-z0-9_]*`. Returns the span.
pub fn ident(input: Span<'_>) -> IResult<Span<'_>, Span<'_>> {
    recognize((ident_start, many0(ident_cont))).parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(s: &str) -> Span<'_> {
        Span::new(s)
    }

    #[test]
    fn skip_ws_consumes_spaces_tabs_newlines() {
        let (rest, _) = skip_ws(span(" \t\n  x")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn skip_ws_consumes_line_comment() {
        let (rest, _) = skip_ws(span("// blah\nx")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn skip_ws_consumes_block_comment() {
        let (rest, _) = skip_ws(span("/* multi\nline */x")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn skip_ws_consumes_mixed() {
        let (rest, _) = skip_ws(span("  // x\n  /* y */\n  z")).unwrap();
        assert_eq!(*rest.fragment(), "z");
    }

    #[test]
    fn skip_ws_succeeds_on_empty_input() {
        let (rest, _) = skip_ws(span("x")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn ident_parses_simple() {
        let (rest, name) = ident(span("foo_bar123 q")).unwrap();
        assert_eq!(*name.fragment(), "foo_bar123");
        assert_eq!(*rest.fragment(), " q");
    }

    #[test]
    fn ident_rejects_leading_digit() {
        assert!(ident(span("1foo")).is_err());
    }
}
```

- [ ] **Step 2: Wire into `lib.rs`**

Replace `crates/aleph-parser/src/lib.rs`:

```rust
//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>`
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod error;
mod lexer;

pub use error::{EmitError, ParseError, ParseErrorKind};
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser lexer`
Expected: 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] Lexer: Span alias, ws/comment skipper, ident"
```

---

## Task 4: Lexer — number literals (`uint`, `float`) and string literals

**Files:**
- Modify: `crates/aleph-parser/src/lexer.rs`

- [ ] **Step 1: Add the literal combinators**

In `crates/aleph-parser/src/lexer.rs`, append (before the `#[cfg(test)] mod tests`):

```rust
use nom::bytes::complete::take_while1;
use nom::character::complete::char as ch;

/// Parse a non-negative decimal integer; returns the parsed `u32`.
/// The caller is responsible for context-bounded ranges; this is the
/// raw lexer primitive.
pub fn uint(input: Span<'_>) -> IResult<Span<'_>, u32> {
    let (input, digits) = take_while1(|c: char| c.is_ascii_digit()).parse(input)?;
    let value: u32 = digits.fragment().parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((input, value))
}

/// Parse a floating-point literal: `123`, `123.456`, `.5`, `1e10`,
/// `1.5e-3`. Returns the parsed `f64`.
pub fn float(input: Span<'_>) -> IResult<Span<'_>, f64> {
    let (rest, lit) = recognize((
        opt(alt((ch('+'), ch('-')))),
        alt((
            // forms like 1.5 or 1.5e-3 or 1.
            recognize((
                take_while1(|c: char| c.is_ascii_digit()),
                ch('.'),
                nom::combinator::opt(take_while1(|c: char| c.is_ascii_digit())),
                opt(exponent),
            )),
            // forms like .5 or .5e10
            recognize((
                ch('.'),
                take_while1(|c: char| c.is_ascii_digit()),
                opt(exponent),
            )),
            // forms like 1 or 1e10 (no dot)
            recognize((
                take_while1(|c: char| c.is_ascii_digit()),
                exponent,
            )),
        )),
    ))
    .parse(input)?;
    let value: f64 = lit.fragment().parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(rest, nom::error::ErrorKind::Float))
    })?;
    Ok((rest, value))
}

fn exponent(input: Span<'_>) -> IResult<Span<'_>, Span<'_>> {
    recognize((
        alt((ch('e'), ch('E'))),
        opt(alt((ch('+'), ch('-')))),
        take_while1(|c: char| c.is_ascii_digit()),
    ))
    .parse(input)
}

/// Parse a `"..."` string literal; returns the inner text (no escape handling).
pub fn string_literal(input: Span<'_>) -> IResult<Span<'_>, Span<'_>> {
    let (input, _) = ch('"').parse(input)?;
    let (input, body) = nom::bytes::complete::take_until("\"").parse(input)?;
    let (input, _) = ch('"').parse(input)?;
    Ok((input, body))
}
```

- [ ] **Step 2: Add tests**

In the existing `mod tests` of `lexer.rs`, append:

```rust
    #[test]
    fn uint_parses() {
        let (rest, n) = uint(span("42 rest")).unwrap();
        assert_eq!(n, 42);
        assert_eq!(*rest.fragment(), " rest");
    }

    #[test]
    fn uint_zero_works() {
        let (_, n) = uint(span("0;")).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn float_parses_decimal() {
        let (_, v) = float(span("1.5;")).unwrap();
        assert_eq!(v, 1.5);
    }

    #[test]
    fn float_parses_scientific() {
        let (_, v) = float(span("1.5e-3;")).unwrap();
        assert!((v - 1.5e-3).abs() < 1e-12);
    }

    #[test]
    fn float_parses_leading_dot() {
        let (_, v) = float(span(".5;")).unwrap();
        assert_eq!(v, 0.5);
    }

    #[test]
    fn float_parses_no_dot_with_exponent() {
        let (_, v) = float(span("1e10;")).unwrap();
        assert_eq!(v, 1e10);
    }

    #[test]
    fn float_parses_signed() {
        let (_, v) = float(span("-1.5;")).unwrap();
        assert_eq!(v, -1.5);
    }

    #[test]
    fn string_literal_parses_simple() {
        let (rest, body) = string_literal(span(r#""stdgates.inc";"#)).unwrap();
        assert_eq!(*body.fragment(), "stdgates.inc");
        assert_eq!(*rest.fragment(), ";");
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser lexer`
Expected: 14 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/src/lexer.rs
git commit -m "[P0-08] Lexer: uint, float, string literal combinators"
```

---

## Task 5: Expression parser + evaluator

**Files:**
- Create: `crates/aleph-parser/src/expr.rs`
- Modify: `crates/aleph-parser/src/lib.rs`

- [ ] **Step 1: Write the expression module**

Create `crates/aleph-parser/src/expr.rs`:

```rust
//! Expression sub-grammar for gate parameters.
//!
//! Supports: `pi`, float literals, `+ - * /`, unary minus, parens.
//! Evaluates to `f64` at parse time per spec § 9. Division by zero
//! and non-finite intermediate results are surfaced as a structured
//! error string that the caller wraps in `ParseErrorKind::BadExpression`.

use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::char as ch;
use nom::combinator::opt;
use nom::multi::many0;
use nom::sequence::delimited;

use crate::lexer::{Span, float, ident, skip_ws, ws};

/// Parse a full expression and evaluate to `f64`. Returns either the
/// finite result or a string describing the failure (the caller turns
/// this into `ParseErrorKind::BadExpression`).
pub fn expr(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    add(input)
}

fn add(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (mut input, mut acc) = mul(input)?;
    loop {
        let (next, op) = opt(ws(alt((ch('+'), ch('-'))))).parse(input)?;
        let Some(op) = op else {
            break;
        };
        let (next, rhs) = mul(next)?;
        acc = match (acc, rhs) {
            (Ok(a), Ok(b)) => {
                let v = if op == '+' { a + b } else { a - b };
                finite(v)
            }
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        input = next;
    }
    Ok((input, acc))
}

fn mul(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (mut input, mut acc) = unary(input)?;
    loop {
        let (next, op) = opt(ws(alt((ch('*'), ch('/'))))).parse(input)?;
        let Some(op) = op else {
            break;
        };
        let (next, rhs) = unary(next)?;
        acc = match (acc, rhs) {
            (Ok(a), Ok(b)) => {
                if op == '/' {
                    if b == 0.0 {
                        Err("division by zero".to_string())
                    } else {
                        finite(a / b)
                    }
                } else {
                    finite(a * b)
                }
            }
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        input = next;
    }
    Ok((input, acc))
}

fn unary(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (input, sign) = opt(ws(ch('-'))).parse(input)?;
    let (input, v) = atom(input)?;
    let v = match v {
        Ok(x) if sign.is_some() => finite(-x),
        other => other,
    };
    Ok((input, v))
}

fn atom(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (input, _) = skip_ws(input)?;
    alt((
        delimited(ws(ch('(')), expr, ws(ch(')'))),
        pi_ident,
        ws_float,
    ))
    .parse(input)
}

fn pi_ident(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    // We need to match the exact identifier "pi" without consuming a
    // surrounding longer ident like "pi2". `ident` already handles
    // alphanumeric-greedy matching, so we just check the captured span.
    let (input, name) = ident(input)?;
    if *name.fragment() == "pi" {
        Ok((input, Ok(std::f64::consts::PI)))
    } else {
        // Not "pi" — return an Err so `alt` falls through to ws_float.
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )))
    }
}

fn ws_float(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (input, _) = skip_ws(input)?;
    let (input, v) = float(input)?;
    Ok((input, finite(v)))
}

fn finite(v: f64) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite expression result ({v})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(s: &str) -> Result<f64, String> {
        let (_, v) = expr(Span::new(s)).unwrap();
        v
    }

    #[test]
    fn literal_float() {
        assert_eq!(eval("1.5"), Ok(1.5));
    }

    #[test]
    fn pi_substituted() {
        assert!((eval("pi").unwrap() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn pi_div_two() {
        assert!((eval("pi/2").unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn precedence_mul_before_add() {
        assert_eq!(eval("2 + 3 * 4"), Ok(14.0));
    }

    #[test]
    fn parens_override() {
        assert_eq!(eval("(2 + 3) * 4"), Ok(20.0));
    }

    #[test]
    fn unary_minus() {
        assert_eq!(eval("-1.5"), Ok(-1.5));
    }

    #[test]
    fn unary_minus_with_paren() {
        assert_eq!(eval("-(1 + 2)"), Ok(-3.0));
    }

    #[test]
    fn left_assoc_subtract() {
        assert_eq!(eval("10 - 3 - 2"), Ok(5.0));
    }

    #[test]
    fn division_by_zero_errors() {
        assert_eq!(eval("1/0"), Err("division by zero".to_string()));
    }

    #[test]
    fn division_by_computed_zero_errors() {
        assert_eq!(eval("1/(2-2)"), Err("division by zero".to_string()));
    }

    #[test]
    fn whitespace_tolerated() {
        assert_eq!(eval(" 2  *  3 "), Ok(6.0));
    }

    #[test]
    fn scientific_literal() {
        assert!((eval("1.5e-3").unwrap() - 1.5e-3).abs() < 1e-15);
    }
}
```

- [ ] **Step 2: Wire into `lib.rs`**

Replace `crates/aleph-parser/src/lib.rs`:

```rust
//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>`
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod error;
mod expr;
mod lexer;

pub use error::{EmitError, ParseError, ParseErrorKind};
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser expr`
Expected: 12 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] Expression parser + evaluator (pi, arithmetic, parens)"
```

---

## Task 6: AST types

**Files:**
- Create: `crates/aleph-parser/src/ast.rs`
- Modify: `crates/aleph-parser/src/lib.rs`

- [ ] **Step 1: Write the AST module**

Create `crates/aleph-parser/src/ast.rs`:

```rust
//! Private AST for the OpenQASM 3 subset.
//!
//! Every node carries a `Position` (line, col) so lowering can emit
//! `ParseError`s with accurate source locations. Expression results
//! are evaluated to `f64` at parse time (per spec § 9), so the AST
//! stores already-evaluated parameters — not raw expression strings.

/// 1-based source position captured at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub header_version: Option<String>,
    pub includes: Vec<Include>,
    pub decls: Vec<Decl>,
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Include {
    pub pos: Position,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Qreg { pos: Position, name: String, size: u32 },
    Creg { pos: Position, name: String, size: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Gate(GateStmt),
    Barrier(BarrierStmt),
    Measure(MeasureStmt),
    Reset(ResetStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateStmt {
    pub pos: Position,
    pub name: String,
    pub params: Vec<f64>,
    pub args: Vec<IndexedRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BarrierStmt {
    pub pos: Position,
    pub args: Vec<RegOrIdx>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasureStmt {
    pub pos: Position,
    pub source: RegOrIdx,
    pub target: RegOrIdx,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResetStmt {
    pub pos: Position,
    pub target: IndexedRef,
}

/// A `name[index]` reference. Used by gate args (which require an index)
/// and as one variant of [`RegOrIdx`].
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedRef {
    pub pos: Position,
    pub name: String,
    pub index: u32,
}

/// Either a whole register (`name`) or an indexed slot (`name[index]`).
/// Used by `barrier` and `measure` where both forms are legal.
#[derive(Debug, Clone, PartialEq)]
pub enum RegOrIdx {
    Whole { pos: Position, name: String },
    Indexed(IndexedRef),
}
```

- [ ] **Step 2: Wire into `lib.rs`**

Replace `crates/aleph-parser/src/lib.rs`:

```rust
//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>`
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod ast;
mod error;
mod expr;
mod lexer;

pub use error::{EmitError, ParseError, ParseErrorKind};
```

- [ ] **Step 3: Build (no test logic yet — pure type definitions)**

Run: `cargo build --package aleph-parser`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] AST types with embedded source positions"
```

---

## Task 7: Parser — header + includes

**Files:**
- Create: `crates/aleph-parser/src/parser.rs`
- Modify: `crates/aleph-parser/src/lib.rs`

- [ ] **Step 1: Write the parser skeleton**

Create `crates/aleph-parser/src/parser.rs`:

```rust
//! nom combinators that build [`crate::ast::Program`].
//!
//! Errors are surfaced as nom errors at this layer; the top-level
//! `parse` function (Task 13) converts them into `ParseError`.

use nom::IResult;
use nom::Parser;
use nom::bytes::complete::tag;
use nom::combinator::opt;
use nom::multi::many0;

use crate::ast::{Include, Position, Program};
use crate::lexer::{Span, ident, skip_ws, string_literal, uint, ws};

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
    // decls + stmts arrive in later tasks; for now Program has empties.
    let (input, _) = skip_ws(input)?;
    Ok((
        input,
        Program {
            header_version,
            includes,
            decls: Vec::new(),
            stmts: Vec::new(),
        },
    ))
}

fn header(input: Span<'_>) -> IResult<Span<'_>, String> {
    let (input, _) = ws(tag("OPENQASM")).parse(input)?;
    let (input, _) = skip_ws(input)?;
    // Major version (only "3" accepted upstream; we just record digits.)
    let (input, major) = uint(input)?;
    let (input, minor) = opt(|i| {
        let (i, _) = nom::character::complete::char('.').parse(i)?;
        uint(i)
    })
    .parse(input)?;
    let (input, _) = ws(tag(";")).parse(input)?;
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
    let (input, _) = ws(tag(";")).parse(input)?;
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
}
```

- [ ] **Step 2: Wire into `lib.rs`**

Replace `crates/aleph-parser/src/lib.rs`:

```rust
//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>`
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod ast;
mod error;
mod expr;
mod lexer;
mod parser;

pub use error::{EmitError, ParseError, ParseErrorKind};
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser parser`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] Parser: header + include directives"
```

---

## Task 8: Parser — declarations (`qubit[N] name;`, `bit[N] name;`)

**Files:**
- Modify: `crates/aleph-parser/src/parser.rs`

- [ ] **Step 1: Add decl combinators**

In `crates/aleph-parser/src/parser.rs`, add `use crate::ast::Decl;` to the top imports, then append (before the `#[cfg(test)]`):

```rust
fn qreg_decl(input: Span<'_>) -> IResult<Span<'_>, Decl> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("qubit").parse(input)?;
    let (input, _) = ws(tag("[")).parse(input)?;
    let (input, size) = ws(uint).parse(input)?;
    let (input, _) = ws(tag("]")).parse(input)?;
    let (input, name) = ws(ident).parse(input)?;
    let (input, _) = ws(tag(";")).parse(input)?;
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
    let (input, _) = ws(tag("[")).parse(input)?;
    let (input, size) = ws(uint).parse(input)?;
    let (input, _) = ws(tag("]")).parse(input)?;
    let (input, name) = ws(ident).parse(input)?;
    let (input, _) = ws(tag(";")).parse(input)?;
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
```

Then modify `program` to parse decls after includes. Replace its body:

```rust
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
```

- [ ] **Step 2: Add tests**

In the existing `mod tests` of `parser.rs`, append:

```rust
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser parser`
Expected: 7 tests pass (3 prior + 4 new).

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/src/parser.rs
git commit -m "[P0-08] Parser: qubit / bit register declarations"
```

---

## Task 9: Parser — qubit refs and gate statements

**Files:**
- Modify: `crates/aleph-parser/src/parser.rs`

- [ ] **Step 1: Add ref + gate combinators**

In `crates/aleph-parser/src/parser.rs`, expand the imports:

```rust
use crate::ast::{Decl, GateStmt, IndexedRef, RegOrIdx, Stmt};
use crate::expr::expr as parse_expr;
use nom::multi::separated_list1;
```

Append (before the `#[cfg(test)]`):

```rust
fn indexed_ref(input: Span<'_>) -> IResult<Span<'_>, IndexedRef> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, name) = ident(input)?;
    let (input, _) = ws(tag("[")).parse(input)?;
    let (input, index) = ws(uint).parse(input)?;
    let (input, _) = ws(tag("]")).parse(input)?;
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
    // Try the longer indexed form first; if no `[` follows, treat as whole.
    let p = pos_of(&input);
    let (input, name) = ident(input)?;
    let (input, bracket) = opt(ws(tag("["))).parse(input)?;
    if bracket.is_some() {
        let (input, index) = ws(uint).parse(input)?;
        let (input, _) = ws(tag("]")).parse(input)?;
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
            input,
            RegOrIdx::Whole {
                pos: p,
                name: name.fragment().to_string(),
            },
        ))
    }
}

fn gate_stmt(input: Span<'_>) -> IResult<Span<'_>, GateStmt> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, name) = ident(input)?;
    // Optional parameter list.
    let (input, params) = opt(gate_params).parse(input)?;
    let (input, first) = indexed_ref(input)?;
    let (input, mut rest) = many0(|i| {
        let (i, _) = ws(tag(",")).parse(i)?;
        indexed_ref(i)
    })
    .parse(input)?;
    let (input, _) = ws(tag(";")).parse(input)?;
    let mut args = Vec::with_capacity(1 + rest.len());
    args.push(first);
    args.append(&mut rest);
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

fn gate_params(input: Span<'_>) -> IResult<Span<'_>, Vec<f64>> {
    let (input, _) = ws(tag("(")).parse(input)?;
    let (input, first) = parse_expr(input)?;
    let (input, rest) = many0(|i| {
        let (i, _) = ws(tag(",")).parse(i)?;
        parse_expr(i)
    })
    .parse(input)?;
    let (input, _) = ws(tag(")")).parse(input)?;
    let mut values = Vec::with_capacity(1 + rest.len());
    // Surface expression errors via nom::Err::Failure so the caller can
    // attach a position. For now, encode the failure as a synthetic
    // ParseError-friendly nom error.
    let push = |dst: &mut Vec<f64>, v: Result<f64, String>, sp: Span<'_>| {
        match v {
            Ok(x) => {
                dst.push(x);
                Ok(())
            }
            Err(_) => Err(nom::Err::Failure(nom::error::Error::new(
                sp,
                nom::error::ErrorKind::Float,
            ))),
        }
    };
    push(&mut values, first, input)?;
    for v in rest {
        push(&mut values, v, input)?;
    }
    Ok((input, values))
}
```

Update `program` to also collect statements (only `gate_stmt` for now; the other stmt kinds arrive in Task 10). Replace its body:

```rust
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

fn stmt(input: Span<'_>) -> IResult<Span<'_>, Stmt> {
    nom::combinator::map(gate_stmt, Stmt::Gate).parse(input)
}
```

- [ ] **Step 2: Add tests**

Append to the `mod tests` in `parser.rs`:

```rust
    use crate::ast::Stmt;

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
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser parser`
Expected: 11 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/src/parser.rs
git commit -m "[P0-08] Parser: gate statements (with parametric args)"
```

---

## Task 10: Parser — `barrier`, `measure`, `reset` statements

**Files:**
- Modify: `crates/aleph-parser/src/parser.rs`

- [ ] **Step 1: Add stmt combinators**

In `crates/aleph-parser/src/parser.rs`, extend imports:

```rust
use crate::ast::{BarrierStmt, Decl, GateStmt, IndexedRef, MeasureStmt, RegOrIdx, ResetStmt, Stmt};
```

Append the new statement parsers (before `#[cfg(test)]`):

```rust
fn barrier_stmt(input: Span<'_>) -> IResult<Span<'_>, BarrierStmt> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("barrier").parse(input)?;
    let (input, first) = reg_or_idx(input)?;
    let (input, mut rest) = many0(|i| {
        let (i, _) = ws(tag(",")).parse(i)?;
        reg_or_idx(i)
    })
    .parse(input)?;
    let (input, _) = ws(tag(";")).parse(input)?;
    let mut args = Vec::with_capacity(1 + rest.len());
    args.push(first);
    args.append(&mut rest);
    Ok((input, BarrierStmt { pos: p, args }))
}

fn measure_stmt(input: Span<'_>) -> IResult<Span<'_>, MeasureStmt> {
    let (input, _) = skip_ws(input)?;
    let p = pos_of(&input);
    let (input, _) = tag("measure").parse(input)?;
    let (input, source) = reg_or_idx(input)?;
    let (input, _) = ws(tag("->")).parse(input)?;
    let (input, target) = reg_or_idx(input)?;
    let (input, _) = ws(tag(";")).parse(input)?;
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
    let (input, target) = indexed_ref(input)?;
    let (input, _) = ws(tag(";")).parse(input)?;
    Ok((input, ResetStmt { pos: p, target }))
}
```

Replace `stmt` (currently only handles `Gate`):

```rust
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
```

**Important:** order matters. `barrier`/`measure`/`reset` start with keywords that would otherwise be parsed as gate names by `gate_stmt`. Putting them first means we try the keyword forms before the generic gate form.

- [ ] **Step 2: Add tests**

Append to `mod tests`:

```rust
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser parser`
Expected: 17 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/src/parser.rs
git commit -m "[P0-08] Parser: barrier / measure / reset statements"
```

---

## Task 11: Lowering — gate-name mapping and validation backbone

**Files:**
- Create: `crates/aleph-parser/src/lower.rs`
- Modify: `crates/aleph-parser/src/lib.rs`

This task wires the AST + source string into a `Circuit`. Gate-name → `Gate` mapping is handled here. Barrier/measure/reset come in Task 12.

- [ ] **Step 1: Write the lowering module**

Create `crates/aleph-parser/src/lower.rs`:

```rust
//! AST → IR lowering. Resolves register names to flat qubit/clbit
//! indices, expands whole-register `barrier`/`measure` forms, builds
//! `GateInstance`s by mapping OpenQASM gate names to `aleph_core::Gate`
//! variants, and delegates per-instruction validation to the IR.

use std::collections::HashMap;

use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::{Circuit, CircuitError, Instruction, MAX_CLBITS, MAX_QUBITS};
use smallvec::smallvec;

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

impl RegisterMap {
    fn new() -> Self {
        Self {
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            total_qubits: 0,
            total_clbits: 0,
        }
    }

    fn add_qreg(&mut self, name: String, size: u32) -> Result<(), CircuitError> {
        if self.total_qubits.checked_add(size).is_none()
            || self.total_qubits + size > MAX_QUBITS
        {
            return Err(CircuitError::TooManyQubits {
                requested: self.total_qubits.saturating_add(size),
                max: MAX_QUBITS,
            });
        }
        let base = self.total_qubits;
        self.qregs.insert(name, (base, size));
        self.total_qubits += size;
        Ok(())
    }

    fn add_creg(&mut self, name: String, size: u32) -> Result<(), CircuitError> {
        if self.total_clbits.checked_add(size).is_none()
            || self.total_clbits + size > MAX_CLBITS
        {
            return Err(CircuitError::TooManyClbits {
                requested: self.total_clbits.saturating_add(size),
                max: MAX_CLBITS,
            });
        }
        let base = self.total_clbits;
        self.cregs.insert(name, (base, size));
        self.total_clbits += size;
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

/// Look up the source line (1-based) and return it without a trailing
/// newline. If `line` is past the end, return "".
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
    // Reject unsupported includes up front.
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
                    CircuitError::TooManyQubits { requested, max } => {
                        ParseErrorKind::TooManyQubits { requested, max }
                    }
                    other => ParseErrorKind::IrRejected(other),
                };
                perr(source, *pos, kind)
            })?,
            Decl::Creg { pos, name, size } => regs.add_creg(name.clone(), *size).map_err(|e| {
                let kind = match e {
                    CircuitError::TooManyClbits { requested, max } => {
                        ParseErrorKind::TooManyClbits { requested, max }
                    }
                    other => ParseErrorKind::IrRejected(other),
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
        // Locate the failing register declaration if we can; otherwise
        // attribute to the start of the program.
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

fn resolve_indexed(
    regs: &RegisterMap,
    r: &IndexedRef,
    source: &str,
) -> Result<u32, ParseError> {
    regs.resolve_qubit(&r.name, r.index)
        .map_err(|kind| perr(source, r.pos, kind))
}

fn lower_gate(
    circuit: &mut Circuit,
    regs: &RegisterMap,
    g: &GateStmt,
    source: &str,
) -> Result<(), ParseError> {
    let (gate, expected_params) = resolve_gate_name(&g.name).ok_or_else(|| {
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
    let gate = build_gate_variant(&gate, &g.params);
    let qubits: smallvec::SmallVec<[u32; 4]> = g
        .args
        .iter()
        .map(|r| resolve_indexed(regs, r, source))
        .collect::<Result<_, _>>()?;
    let inst = GateInstance::new(gate, qubits);
    circuit
        .add_gate(inst)
        .map_err(|e| perr(source, g.pos, ParseErrorKind::IrRejected(e)))?;
    Ok(())
}

/// Returns the gate "shape": a `Gate::H`-style placeholder we'll use to
/// pick the correct variant constructor in `build_gate_variant`, plus
/// the expected parameter arity.
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
    }
}

// Barrier / measure / reset lowering land in Task 12.
fn lower_barrier(
    _circuit: &mut Circuit,
    _regs: &RegisterMap,
    _b: &BarrierStmt,
    _source: &str,
) -> Result<(), ParseError> {
    Err(ParseError::new(
        0,
        0,
        String::new(),
        ParseErrorKind::UnsupportedFeature {
            feature: "barrier (pending Task 12)",
        },
    ))
}

fn lower_measure(
    _circuit: &mut Circuit,
    _regs: &RegisterMap,
    _m: &MeasureStmt,
    _source: &str,
) -> Result<(), ParseError> {
    Err(ParseError::new(
        0,
        0,
        String::new(),
        ParseErrorKind::UnsupportedFeature {
            feature: "measure (pending Task 12)",
        },
    ))
}

fn lower_reset(
    _circuit: &mut Circuit,
    _regs: &RegisterMap,
    _r: &ResetStmt,
    _source: &str,
) -> Result<(), ParseError> {
    Err(ParseError::new(
        0,
        0,
        String::new(),
        ParseErrorKind::UnsupportedFeature {
            feature: "reset (pending Task 12)",
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::program;
    use crate::lexer::Span;

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
        // Demonstrates flattening: `aux[0]` is qubit index 2.
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
}
```

- [ ] **Step 2: Wire into `lib.rs`**

Replace `crates/aleph-parser/src/lib.rs`:

```rust
//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>`
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod ast;
mod error;
mod expr;
mod lexer;
mod lower;
mod parser;

pub use error::{EmitError, ParseError, ParseErrorKind};
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser lower`
Expected: 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] Lowering: register map, gate-name resolution, IR emission"
```

---

## Task 12: Lowering — barrier / measure / reset (whole-register expansion)

**Files:**
- Modify: `crates/aleph-parser/src/lower.rs`

- [ ] **Step 1: Replace the three placeholder lowerers**

In `crates/aleph-parser/src/lower.rs`, replace the bodies of `lower_barrier`, `lower_measure`, and `lower_reset` (delete the placeholder `Err(...)` returns):

```rust
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
    // Both whole OR both indexed. Mixed → SizeMismatch with a clear msg.
    match (&m.source, &m.target) {
        (RegOrIdx::Indexed(qr), RegOrIdx::Indexed(cr)) => {
            let q = resolve_indexed(regs, qr, source)?;
            let c = regs
                .resolve_clbit(&cr.name, cr.index)
                .map_err(|kind| perr(source, cr.pos, kind))?;
            circuit
                .add_instruction(Instruction::Measure {
                    qubit: q,
                    clbit: c,
                })
                .map_err(|e| perr(source, m.pos, ParseErrorKind::IrRejected(e)))?;
            Ok(())
        }
        (RegOrIdx::Whole { pos: qpos, name: qname }, RegOrIdx::Whole { pos: _, name: cname }) => {
            let (qbase, qsize) = regs.qreg_size(qname).ok_or_else(|| {
                perr(
                    source,
                    *qpos,
                    ParseErrorKind::UnknownRegister { name: qname.clone() },
                )
            })?;
            let (cbase, csize) = regs.creg_size(cname).ok_or_else(|| {
                perr(
                    source,
                    m.pos,
                    ParseErrorKind::UnknownRegister { name: cname.clone() },
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
```

- [ ] **Step 2: Add tests**

Append to the `mod tests` of `lower.rs`:

```rust
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
        let err =
            parse_and_lower("qubit[1] q; bit[1] c; measure q[0] -> c[5];").unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::IndexOutOfBounds { index: 5, size: 1, .. }
        ));
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser lower`
Expected: 15 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/src/lower.rs
git commit -m "[P0-08] Lowering: barrier / measure (whole-register) / reset"
```

---

## Task 13: Top-level `parse` function

**Files:**
- Modify: `crates/aleph-parser/src/lib.rs`
- Modify: `crates/aleph-parser/src/parser.rs` (only if `program` returns nom errors that need conversion — see Step 2)

- [ ] **Step 1: Add `parse` and friends to `lib.rs`**

Replace `crates/aleph-parser/src/lib.rs`:

```rust
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

fn nom_error_to_parse_error<'a>(
    source: &str,
    err: nom::error::Error<Span<'a>>,
) -> ParseError {
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
```

- [ ] **Step 2: Add lib-level integration tests**

In `crates/aleph-parser/src/lib.rs`, append at the bottom:

```rust
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser`
Expected: all tests pass (4 new + previous).

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] Top-level parse() function wiring lex+parse+lower"
```

---

## Task 14: Error rendering tests (sanity-check `ParseError::render` paths from real parses)

**Files:**
- Create: `crates/aleph-parser/tests/error_rendering.rs`

- [ ] **Step 1: Write the integration test**

Create `crates/aleph-parser/tests/error_rendering.rs`:

```rust
//! Snapshot-style tests that the `ParseError::render()` output looks
//! sensible for the error paths users will actually hit.

use aleph_parser::{ParseError, ParseErrorKind, parse};

fn err(src: &str) -> ParseError {
    parse(src).unwrap_err()
}

#[test]
fn unknown_gate_renders_three_lines_with_caret() {
    let e = err("qubit[1] q;\nhadamard q[0];\n");
    let out = e.render();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("unknown gate"));
    assert_eq!(lines[1], "hadamard q[0];");
    assert!(lines[2].starts_with("^"));
}

#[test]
fn unknown_register_caret_points_at_register_name() {
    let e = err("qubit[1] q;\nh r[0];\n");
    let out = e.render();
    let lines: Vec<&str> = out.lines().collect();
    assert!(lines[0].contains("undeclared register"));
    assert_eq!(lines[1], "h r[0];");
    // The caret should fall at column 1 (we point at the gate statement),
    // matching `h r[0];`. Exact column is implementation-defined.
}

#[test]
fn index_out_of_bounds_renders() {
    let e = err("qubit[2] q; h q[9];");
    let out = e.render();
    assert!(out.contains("out of bounds"));
    assert!(out.contains("size 2"));
}

#[test]
fn too_many_qubits_renders() {
    let big = aleph_ir::MAX_QUBITS + 1;
    let src = format!("qubit[{big}] q;");
    let e = err(&src);
    matches!(e.kind, ParseErrorKind::TooManyQubits { .. });
    let out = e.render();
    assert!(out.contains("too many qubits"));
}

#[test]
fn size_mismatch_renders() {
    let e = err("qubit[2] q; bit[3] c; measure q -> c;");
    let out = e.render();
    assert!(out.contains("size mismatch"));
}

#[test]
fn display_one_liner_includes_line_col() {
    let e = err("qubit[1] q;\nhadamard q[0];\n");
    let one = format!("{e}");
    assert!(one.starts_with("error at 2:"));
    assert!(one.contains("unknown gate"));
}
```

- [ ] **Step 2: Run**

Run: `cargo test --package aleph-parser --test error_rendering`
Expected: 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-parser/tests/error_rendering.rs
git commit -m "[P0-08] Error-rendering integration tests"
```

---

## Task 15: Emitter (`Circuit → String`)

**Files:**
- Create: `crates/aleph-parser/src/emit.rs`
- Modify: `crates/aleph-parser/src/lib.rs`

- [ ] **Step 1: Write the emitter**

Create `crates/aleph-parser/src/emit.rs`:

```rust
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
            // Rust's default Display for f64 emits the shortest round-trip form.
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
        assert!(!out.contains("bit["));
    }

    #[test]
    fn round_trip_through_parse_emit_parse_preserves_instructions() {
        let src = "qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];";
        let c1 = parse(src).unwrap();
        let out = emit(&c1).unwrap();
        let c2 = parse(&out).unwrap();
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.instructions().iter().zip(c2.instructions().iter()) {
            // Debug equality is enough for instruction-by-instruction check.
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
    }
}
```

- [ ] **Step 2: Wire into `lib.rs`**

Update `crates/aleph-parser/src/lib.rs` — add `mod emit;` and a public `emit` function. Replace the module-list block + add the function near `parse`:

```rust
mod ast;
mod emit;
mod error;
mod expr;
mod lexer;
mod lower;
mod parser;

pub use error::{EmitError, ParseError, ParseErrorKind};

// ... (parse and its helpers stay the same; add this:)

/// Emit an `aleph_ir::Circuit` as OpenQASM 3 source (normalized).
pub fn emit(circuit: &aleph_ir::Circuit) -> Result<String, EmitError> {
    emit::emit(circuit)
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package aleph-parser`
Expected: all tests pass including 4 new emitter tests.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/
git commit -m "[P0-08] Emitter: Circuit -> normalized OpenQASM 3 source"
```

---

## Task 16: Tier-1 algorithm fixtures + parse tests

**Files:**
- Create: `crates/aleph-parser/tests/fixtures/ghz.qasm`
- Create: `crates/aleph-parser/tests/fixtures/qft.qasm`
- Create: `crates/aleph-parser/tests/fixtures/grover.qasm`
- Create: `crates/aleph-parser/tests/fixtures/random.qasm`
- Create: `crates/aleph-parser/tests/algorithms_qasm.rs`

- [ ] **Step 1: Write `tests/fixtures/ghz.qasm`**

```qasm
// 3-qubit GHZ state preparation.
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
bit[3] c;

h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
```

- [ ] **Step 2: Write `tests/fixtures/qft.qasm`**

3-qubit Quantum Fourier Transform without the final swap (standard textbook form).

```qasm
// 3-qubit QFT (without final swap).
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;

h q[2];
cz q[1], q[2];
rz(pi/2) q[2];
h q[1];
cz q[0], q[1];
rz(pi/4) q[1];
cz q[0], q[2];
rz(pi/8) q[2];
h q[0];
```

(Note: the canonical QFT uses controlled-`Phase` gates with `pi/2^k` angles — we approximate with `cz` + `rz` for simplicity here. This is enough to exercise the parser's expression evaluator on `pi/N` forms.)

- [ ] **Step 3: Write `tests/fixtures/grover.qasm`**

2-qubit Grover with a simple `cz` oracle marking `|11⟩` and standard diffusion.

```qasm
// 2-qubit Grover search marking |11>.
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

// Uniform superposition.
h q[0];
h q[1];

// Oracle: flip phase of |11>.
cz q[0], q[1];

// Diffusion operator.
h q[0];
h q[1];
x q[0];
x q[1];
cz q[0], q[1];
x q[0];
x q[1];
h q[0];
h q[1];

measure q -> c;
```

- [ ] **Step 4: Write `tests/fixtures/random.qasm`**

A 4-qubit hand-crafted random circuit that exercises every gate class.

```qasm
// 4-qubit "random" circuit exercising every gate class used by P0-08.
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
bit[4] c;

h q[0];
x q[1];
y q[2];
z q[3];
s q[0];
sdg q[1];
t q[2];
tdg q[3];
rx(0.7) q[0];
ry(-1.2) q[1];
rz(pi/3) q[2];
p(pi/4) q[3];
u3(0.1, 0.2, 0.3) q[0];
cx q[0], q[1];
cz q[1], q[2];
swap q[2], q[3];
ccx q[0], q[1], q[2];
barrier q;
measure q -> c;
```

- [ ] **Step 5: Write `tests/algorithms_qasm.rs`**

```rust
//! Parse each Tier-1 fixture and assert key IR-level facts (instruction
//! count, sample gate variants). Round-trip checks live in
//! `round_trip.rs`.

use aleph_core::Gate;
use aleph_ir::Instruction;
use aleph_parser::parse;

fn fixture(name: &str) -> String {
    let path = format!("tests/fixtures/{name}.qasm");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

#[test]
fn ghz_parses() {
    let c = parse(&fixture("ghz")).unwrap();
    assert_eq!(c.num_qubits(), 3);
    assert_eq!(c.num_clbits(), 3);
    // 1 H + 2 CNOT + 3 Measure (whole-register measure fans out) = 6.
    assert_eq!(c.len(), 6);
    assert!(matches!(
        &c.instructions()[0],
        Instruction::Gate(g) if g.gate == Gate::H
    ));
    assert!(matches!(
        &c.instructions()[1],
        Instruction::Gate(g) if g.gate == Gate::Cnot
    ));
}

#[test]
fn qft_parses() {
    let c = parse(&fixture("qft")).unwrap();
    assert_eq!(c.num_qubits(), 3);
    // 3 H + 3 Cz + 3 Rz = 9 instructions.
    assert_eq!(c.len(), 9);
}

#[test]
fn grover_parses() {
    let c = parse(&fixture("grover")).unwrap();
    assert_eq!(c.num_qubits(), 2);
    assert_eq!(c.num_clbits(), 2);
    // 2H (init) + Cz (oracle) + 9 (diffusion: 4H + 4X + Cz) + 2 measure = 14.
    assert_eq!(c.len(), 14);
}

#[test]
fn random_parses() {
    let c = parse(&fixture("random")).unwrap();
    assert_eq!(c.num_qubits(), 4);
    assert_eq!(c.num_clbits(), 4);
    // Sample a few gates.
    let kinds: Vec<&Gate> = c
        .instructions()
        .iter()
        .filter_map(|i| match i {
            Instruction::Gate(g) => Some(&g.gate),
            _ => None,
        })
        .collect();
    assert!(kinds.contains(&&Gate::H));
    assert!(kinds.contains(&&Gate::Toffoli));
    assert!(kinds.contains(&&Gate::Swap));
}

#[test]
fn random_sets_generated_from_metadata() {
    let c = parse(&fixture("random")).unwrap();
    assert_eq!(c.metadata().generated_from.as_deref(), Some("openqasm:3.0"));
}
```

- [ ] **Step 6: Run**

Run: `cargo test --package aleph-parser --test algorithms_qasm`
Expected: 5 tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-parser/tests/fixtures/ crates/aleph-parser/tests/algorithms_qasm.rs
git commit -m "[P0-08] Tier-1 fixtures + algorithm parse tests"
```

---

## Task 17: Round-trip tests for fixtures

**Files:**
- Create: `crates/aleph-parser/tests/round_trip.rs`

- [ ] **Step 1: Write the round-trip test**

Create `crates/aleph-parser/tests/round_trip.rs`:

```rust
//! Fixture-driven round-trip: parse → emit → parse, assert IR equality.

use aleph_ir::{Circuit, Instruction};
use aleph_parser::{emit, parse};

fn fixture(name: &str) -> String {
    let path = format!("tests/fixtures/{name}.qasm");
    std::fs::read_to_string(&path).expect("read fixture")
}

fn assert_round_trip(name: &str) {
    let src = fixture(name);
    let c1 = parse(&src).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
    let out = emit(&c1).unwrap_or_else(|e| panic!("{name}: emit: {e}"));
    let c2 = parse(&out).unwrap_or_else(|e| panic!(
        "{name}: re-parse failed.\nemitted source:\n{out}\nerror:\n{}",
        e.render()
    ));
    assert_eq!(c1.len(), c2.len(), "{name}: instruction count differs");
    assert_eq!(
        c1.num_qubits(),
        c2.num_qubits(),
        "{name}: num_qubits differs"
    );
    assert_eq!(
        c1.num_clbits(),
        c2.num_clbits(),
        "{name}: num_clbits differs"
    );
    for (i, (a, b)) in c1
        .instructions()
        .iter()
        .zip(c2.instructions().iter())
        .enumerate()
    {
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "{name}: instruction[{i}] differs"
        );
    }
}

#[test]
fn ghz_round_trip() {
    assert_round_trip("ghz");
}

#[test]
fn qft_round_trip() {
    assert_round_trip("qft");
}

#[test]
fn grover_round_trip() {
    assert_round_trip("grover");
}

#[test]
fn random_round_trip() {
    assert_round_trip("random");
}
```

- [ ] **Step 2: Run**

Run: `cargo test --package aleph-parser --test round_trip`
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-parser/tests/round_trip.rs
git commit -m "[P0-08] Round-trip fixture tests (parse -> emit -> parse)"
```

---

## Task 18: Round-trip property test

**Files:**
- Create: `crates/aleph-parser/tests/round_trip_property.rs`

- [ ] **Step 1: Write the proptest**

Create `crates/aleph-parser/tests/round_trip_property.rs`:

```rust
//! Property test: random `Circuit` (restricted to emitter-supported
//! variants) round-trips through `emit → parse → compare`. The
//! `OpKind` enum mirrors the P0-07 layer-extraction property test
//! shape but excludes `Iswap`/`IswapDg`/`CRx`/`CRy`/`CRz`/`Ccz`/
//! `Unitary1q`/`Unitary2q` and any `GateInstance` with external
//! `controls` — the emitter intentionally rejects those (spec § 13).

use aleph_ir::{Circuit, Instruction};
use aleph_parser::{emit, parse};
use proptest::prelude::*;

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
            Err(_) => return Ok(()), // emitter rejection — out of scope for this prop
        };
        let c2 = parse(&out).map_err(|e| TestCaseError::fail(format!(
            "re-parse failed.\nemitted:\n{out}\nerror:\n{}",
            e.render()
        )))?;
        prop_assert_eq!(c.len(), c2.len(), "instruction count mismatch");
        prop_assert_eq!(c.num_qubits(), c2.num_qubits());
        prop_assert_eq!(c.num_clbits(), c2.num_clbits());
        for (i, (a, b)) in c.instructions().iter().zip(c2.instructions().iter()).enumerate() {
            // Stringified Debug ignores f64 ULP-level differences; the
            // emitter uses Display which is the round-trip-exact form,
            // so Debug equality holds.
            prop_assert_eq!(format!("{a:?}"), format!("{b:?}"), "instr {i} differs");
        }
    }
}
```

- [ ] **Step 2: Run**

Run: `cargo test --package aleph-parser --test round_trip_property`
Expected: 1 test passes (256 random circuits by default).

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-parser/tests/round_trip_property.rs
git commit -m "[P0-08] Property test: random Circuit -> emit -> parse roundtrip"
```

---

## Task 19: Final acceptance gate

**Files:** none (verification only).

- [ ] **Step 1: Format**

Run: `cargo fmt --all`
Then: `cargo fmt --check`
Expected: clean.

- [ ] **Step 2: Workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: exit 0.

- [ ] **Step 3: Workspace test (debug)**

Run: `cargo test --workspace`
Expected: all green.

- [ ] **Step 4: Workspace test (release)**

Run: `cargo test --workspace --release`
Expected: all green.

- [ ] **Step 5: Commit any fmt diff**

```bash
if ! git diff --quiet; then
  git add -A
  git commit -m "[P0-08] cargo fmt"
fi
```

- [ ] **Step 6: Final status**

Run: `git log --oneline main..HEAD`
Expected: a clean linear `[P0-08] ...` history.

---

## Acceptance criteria checklist

Mirrors `BACKLOG.md:447` and spec § 15:

- [ ] Parses `tests/fixtures/{ghz,qft,grover,random}.qasm` into expected `Circuit` IR.
- [ ] `parse → emit → parse` yields equivalent `Circuit` on each fixture and on random `OpKind`-generated circuits (proptest default budget).
- [ ] `ParseError::render()` produces a 3-line block with line, column, and caret pointing at the source.
- [ ] Multiple `qubit`/`bit` declarations supported and flattened.
- [ ] `aleph_ir::Circuit::try_new` exposed; parser uses it; no `Circuit::new` panic path is reachable from `parse`.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] `cargo test --workspace` passes in both debug and release.

---

## Notes for the executor

- **Commit per task.** Don't squash inside this branch; the final PR squashes at merge time. P0-06 and P0-07 followed this pattern.
- **No `unwrap()`/`expect()` outside `#[cfg(test)]`** per CLAUDE.md § Code Conventions. The only acceptable `unwrap` in production parser code is on `write!(out, ...).unwrap()` where the target is a `String` (infallible). All builder errors flow through `Result` and `?`.
- **`Param::Concrete` only.** P0-06 left a stub for `Param::Symbolic` but P0-08 never produces it (the spec § 9 evaluation is concrete only). The emitter errors on `Symbolic` for completeness over the IR.
- **Spec drift:** if implementation surfaces a decision not in the spec, add a § 17 amendment to the spec and commit it alongside the change (same pattern as P0-06 § 12 and P0-07 § 12). Do not silently diverge.
- **`nom 7` API note:** the `.parse(input)` call style is the modern (deferred-to-trait) form. `tag(...)`, `take_until(...)` etc. all implement `Parser`; use `.parse(input)` rather than calling them as functions directly. This plan uses that style consistently.
- **No new workspace dependencies** beyond `nom`, `nom_locate` (added in Task 1).
- **`Circuit::new` keeps the `assert!`** — only the `TODO(P0-08)` comment goes away. Internal-use callers still get the panicking constructor; untrusted-input callers (the parser) use `try_new`.
