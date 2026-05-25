//! Shared proptest strategies for the aleph workspace.  Dev-only;
//! never depended on from production code.  See
//! `docs/superpowers/specs/2026-05-25-p0-05-proptest-infra-design.md`.

pub mod circuit;
pub mod gate;
pub mod pauli;
pub mod state;
