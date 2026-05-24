//! `aleph-parser`: OpenQASM 3.0 (minimal subset) parser and emitter.
//!
//! Top-level API:
//! - [`parse`] — `&str → Result<aleph_ir::Circuit, ParseError>`
//! - [`emit`]  — `&aleph_ir::Circuit → Result<String, EmitError>`
//!
//! See `docs/superpowers/specs/2026-05-24-p0-08-openqasm-parser-design.md`.

mod error;

pub use error::{EmitError, ParseError, ParseErrorKind};
