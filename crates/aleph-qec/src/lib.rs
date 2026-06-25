//! `aleph-qec` — quantum error correction: codes, noise, syndromes, detector error
//! models, and decoders.
//!
//! This crate is the home of the QEC-decoder track (see `docs/qec/ROADMAP.md`). It owns
//! everything that sits *above* a backend: the [`DetectorErrorModel`] that decoders consume,
//! the [`Syndrome`]/[`Correction`] data they exchange, the [`Decoder`] trait they implement,
//! and the [`LogicalErrorResult`] a Monte-Carlo experiment produces.
//!
//! It builds on the stabilizer engine (`aleph-stab`) for error propagation and consumes the
//! backend-agnostic circuit IR (`aleph-ir`), keeping `aleph-core`/`aleph-ir` themselves free
//! of backend concerns as CLAUDE.md requires.
//!
//! Contents so far: core types + a Stim-compatible [`DetectorErrorModel`] with text round-trip
//! (Q0-01); the surface-code memory-Z experiment ([`SurfaceCode`], [`MemoryExperiment`]) and
//! a [`build_dem`] that derives a DEM from an [`AnnotatedCircuit`] by symbolic Pauli
//! propagation (Q0-03); and the logical-error-rate Monte-Carlo harness
//! ([`run_memory_experiment`], [`run_dem_experiment`]) with a baseline [`NullDecoder`] and an
//! external [`PyMatchingOracle`] (Q0-04); and the [`MatchingGraph`] that turns a graph-like DEM
//! into the weighted detector-plus-boundary graph matching decoders consume (Q1-01).

mod blossom;
mod builder;
mod decoder;
mod dem;
mod error;
mod experiment;
mod matching;
mod mwpm;
mod pymatching;
mod surface;
mod syndrome;

pub use builder::{build_dem, AnnotatedCircuit, ErrorMechanism};
pub use decoder::{Decoder, LogicalErrorResult, NullDecoder};
pub use dem::{DemError, DetectorErrorModel};
pub use error::{Error, Result};
pub use experiment::{run_dem_experiment, run_memory_experiment, PhenomenologicalNoise};
pub use matching::{MatchingEdge, MatchingGraph, NodeId};
pub use mwpm::MwpmDecoder;
pub use pymatching::PyMatchingOracle;
pub use surface::{MemoryExperiment, SurfaceCode};
pub use syndrome::{Correction, Syndrome};
