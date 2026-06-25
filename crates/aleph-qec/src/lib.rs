//! `aleph-qec` — quantum error correction: codes, noise, syndromes, detector error
//! models, and decoders.
//!
//! This crate is the home of the QEC-decoder track (see `docs/qec/ROADMAP.md`). It owns
//! everything that sits *above* a backend: the [`DetectorErrorModel`] that decoders consume,
//! the [`Syndrome`]/[`Correction`] data they exchange, the [`Decoder`] trait they implement,
//! and the [`LogicalErrorResult`] a Monte-Carlo experiment produces.
//!
//! It deliberately depends on no backend so it stays usable from the CLI, the GPU decoders,
//! and the experiment harness alike — keeping `aleph-core`/`aleph-ir` backend-agnostic as
//! CLAUDE.md requires.
//!
//! Phase Q0-01 scope: crate skeleton + core types + a Stim-compatible Detector Error Model
//! with text round-trip. Noise injection, DEM construction from circuits, and the actual
//! decoders arrive in later issues (Q0-02 onward).

mod decoder;
mod dem;
mod error;
mod syndrome;

pub use decoder::{Decoder, LogicalErrorResult};
pub use dem::{DemError, DetectorErrorModel};
pub use error::{Error, Result};
pub use syndrome::{Correction, Syndrome};
