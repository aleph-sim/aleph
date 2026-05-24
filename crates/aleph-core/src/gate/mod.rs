//! Quantum gate representation: `Gate`, `GateInstance`, `GateMatrix`,
//! `Param`. See `docs/superpowers/specs/2026-05-24-p0-06-gate-enum-design.md`.

mod error;
mod param;

pub use error::GateError;
pub use param::{Param, SymbolId};
