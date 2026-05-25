//! `aleph` CLI — public surface re-exported here so integration tests
//! and the binary share the same compiled units.  See
//! `docs/superpowers/specs/2026-05-25-p0-12-cli-design.md`.

pub mod cli;
pub mod exec;
pub mod output;
pub mod pauli;
