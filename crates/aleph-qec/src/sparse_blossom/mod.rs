//! Sparse Blossom: minimum-weight perfect matching by local, event-driven region growth on the
//! detector graph (Higgott & Gidney, arXiv:2303.15933), re-derived from Edmonds' primal-dual
//! blossom algorithm. See `docs/superpowers/specs/2026-09-24-sparse-blossom-design.md`.
//!
//! Module map: [`graph`] compiles the [`crate::MatchingGraph`] into a CSR with doubled integer
//! weights; [`state`] holds the per-shot mutable arenas and the event heap; [`flooder`] grows and
//! shrinks regions and detects collisions; [`matcher`] runs the alternating-tree operations and
//! resolves the final matching. [`SparseMatcher`] is the crate-facing entry point.

pub(crate) mod graph;
pub(crate) mod state;

#[allow(unused_imports)]
pub(crate) use graph::{CompiledGraph, WEIGHT_SCALE};
