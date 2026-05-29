//! `aleph-ir`: Backend-agnostic circuit IR and (future) optimization passes.
//!
//! Phase 0 — provides `Circuit`, `Instruction`, `CircuitMetadata`,
//! `CircuitError`, and a layer-extraction helper. See
//! `docs/superpowers/specs/2026-05-24-p0-07-circuit-ir-design.md`.

mod circuit;
mod error;
mod instruction;
mod layers;

pub use circuit::{Circuit, CircuitMetadata, MAX_CLBITS, MAX_GATE_CONTROLS, MAX_QUBITS};
pub use error::CircuitError;
pub use instruction::Instruction;

pub mod passes;
