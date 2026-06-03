//! `aleph-ir::passes` — IR-level optimisation passes.
//!
//! Each pass implements [`Pass`]. A [`PassPipeline`] runs an ordered
//! sequence of passes over a [`Circuit`], aggregating per-pass
//! [`PassStats`]. The default pipeline ships
//! [`cancel::CancelInversePairs`], [`dce::DeadCodeElim`],
//! [`fuse_diagonal::FuseDiagonalRuns`], [`fuse_1q::Fuse1qRuns`], and
//! [`fuse_2q::Fuse2q`] — in that pipeline order (cancellation precedes
//! DCE so DCE can clean up gates newly exposed as dead by cancellation;
//! diagonal fusion precedes `Fuse2q` so raw `cx`s are still absorbable;
//! see [`PassPipeline::default_pipeline`]). Later tickets add more
//! passes that plug in by being pushed onto the pipeline.
//!
//! This module also exports [`commute::gates_commute`], a sound,
//! conservative commutation predicate over `GateInstance` pairs that
//! future passes use to decide when gates may be reordered. It is a
//! free function, not a [`Pass`], and is not part of `default_pipeline`.

use crate::Circuit;
use thiserror::Error;

pub mod cancel;
pub mod commute;
pub mod dce;
pub mod fuse_1q;
pub mod fuse_2q;
pub mod fuse_diagonal;
pub mod fuse_kq;

pub use cancel::CancelInversePairs;
pub use commute::gates_commute;
pub use dce::DeadCodeElim;
pub use fuse_1q::Fuse1qRuns;
pub use fuse_2q::Fuse2q;
pub use fuse_diagonal::FuseDiagonalRuns;

/// Statistics emitted by a single pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassStats {
    pub gates_before: usize,
    pub gates_after: usize,
    /// Pass-defined unit. For `Fuse1qRuns` this is the number of
    /// runs of length ≥ 2 that were collapsed into a single fused
    /// gate.
    pub transformations: u64,
}

/// Error returned by a [`Pass`] when it cannot continue.
///
/// Single-variant for Phase 1 — `Fuse1qRuns` never errors in
/// practice. The signature reserves the surface for future passes
/// that can legitimately fail (e.g. a transpiler refusing a gate
/// set).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PassError {
    #[error("internal invariant violated: {0}")]
    InternalInvariant(&'static str),
}

/// One optimisation pass. Mutates the circuit in place.
pub trait Pass {
    /// Stable identifier, used in logs and error messages.
    fn name(&self) -> &'static str;
    /// Run the pass on `circuit`. The implementation is allowed to
    /// rebuild the instruction vector; it must preserve the
    /// semantic content (qubit/clbit counts, metadata).
    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError>;
}

/// Ordered sequence of passes.
pub struct PassPipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl PassPipeline {
    pub fn new(passes: Vec<Box<dyn Pass>>) -> Self {
        Self { passes }
    }

    /// Default pipeline. Currently
    /// `[CancelInversePairs, DeadCodeElim, FuseDiagonalRuns, Fuse1qRuns, Fuse2q]`;
    /// later passes are appended here as they ship.
    ///
    /// Cancellation runs **before** dead-code elimination because
    /// cancelling an inverse pair can expose newly-dead gates — e.g. a
    /// gate whose only entangling neighbours were a `Swap·Swap` pair
    /// becomes dead once that pair is deleted. DCE must run afterwards
    /// to remove them, so the pipeline reaches a fixpoint in a single
    /// `optimize()` (idempotence). Running DCE first instead would leave
    /// such gates for a second pass and violate that property.
    ///
    /// Cancellation also runs before fusion so exact inverse pairs
    /// (e.g. `Rz(θ)·Rz(−θ)`) are deleted rather than fused into an
    /// identity block that still executes.
    ///
    /// [`FuseDiagonalRuns`] runs after Cancel/DCE and **before**
    /// [`Fuse2q`]. It must precede `Fuse2q` so that raw `cx`s are still
    /// available for the diagonal pass to absorb into a `DiagonalPhase`:
    /// once `Fuse2q` has buried a `cx` inside a non-diagonal dense 4×4
    /// `Unitary2q` block, the diagonal pass can no longer recognise or
    /// fuse it. An emitted [`Instruction::DiagonalPhase`](crate::Instruction)
    /// is itself a hard run-breaker for `FuseDiagonalRuns` (it is never a
    /// run member), so re-running the pass over its *own* output is a
    /// no-op.
    ///
    /// Note: the *pipeline* is convergent and deterministic but not
    /// strictly idempotent in a single `optimize()`. `FuseDiagonalRuns`
    /// uses a conservative "whole-run-or-nothing" v1 rule — a
    /// {diagonal ∪ cx} run whose net GF(2) permutation is not the
    /// identity is re-emitted verbatim. `Fuse2q` running afterwards can
    /// then remove the lone unpaired `cx` (folding it into a `Unitary2q`),
    /// re-exposing a permutation-closed diagonal sub-run that the *next*
    /// `optimize()` fuses. The state vector is preserved exactly at every
    /// step; only the gate count can shrink for one extra round.
    /// `optimize()` reaches its fixpoint within two passes. (A future
    /// ticket may split runs at the last identity-permutation prefix,
    /// restoring strict one-pass idempotence.)
    pub fn default_pipeline() -> Self {
        Self::new(vec![
            Box::new(CancelInversePairs),
            Box::new(DeadCodeElim),
            Box::new(FuseDiagonalRuns),
            Box::new(Fuse1qRuns),
            Box::new(Fuse2q),
        ])
    }

    /// Run every pass in order. Aggregate stats:
    /// - `gates_before` from the first pass
    /// - `gates_after` from the last pass
    /// - `transformations` summed across all passes
    pub fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        if self.passes.is_empty() {
            let n = circuit.len();
            return Ok(PassStats {
                gates_before: n,
                gates_after: n,
                transformations: 0,
            });
        }
        let mut agg = PassStats::default();
        for (i, pass) in self.passes.iter().enumerate() {
            let stats = pass.run(circuit)?;
            if i == 0 {
                agg.gates_before = stats.gates_before;
            }
            agg.gates_after = stats.gates_after;
            agg.transformations = agg.transformations.saturating_add(stats.transformations);
        }
        Ok(agg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circuit;

    #[test]
    fn default_pipeline_includes_fuse_2q() {
        // Rx(0); CNOT(0,1) — only Fuse2q can fuse this into one Unitary2q.
        let mut c = Circuit::new(2, 0);
        c.rx(0.5, 0).unwrap();
        c.cnot(0, 1).unwrap();
        let stats = PassPipeline::default_pipeline().run(&mut c).unwrap();
        assert_eq!(stats.gates_before, 2);
        assert_eq!(stats.gates_after, 1);
        assert!(stats.transformations >= 1);
    }

    #[test]
    fn default_pipeline_cancels_inverse_pair_before_fusion() {
        // H(0); H(0); H(1) — the H·H pair must be removed by cancellation;
        // fusion alone would instead fuse it into an identity Unitary1q that
        // still executes. After the pipeline: only H(1) remains.
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.h(0).unwrap();
        c.h(1).unwrap();
        let stats = PassPipeline::default_pipeline().run(&mut c).unwrap();
        assert_eq!(stats.gates_before, 3);
        assert_eq!(stats.gates_after, 1);
        assert!(stats.transformations >= 1);
    }

    struct Noop;
    impl Pass for Noop {
        fn name(&self) -> &'static str {
            "Noop"
        }
        fn run(&self, c: &mut Circuit) -> Result<PassStats, PassError> {
            let n = c.len();
            Ok(PassStats {
                gates_before: n,
                gates_after: n,
                transformations: 0,
            })
        }
    }

    #[test]
    fn empty_pipeline_is_a_no_op() {
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        let stats = PassPipeline::new(vec![]).run(&mut c).unwrap();
        assert_eq!(stats.gates_before, 1);
        assert_eq!(stats.gates_after, 1);
        assert_eq!(stats.transformations, 0);
    }

    #[test]
    fn pipeline_runs_passes_in_order_and_aggregates() {
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.h(1).unwrap();
        let pipeline = PassPipeline::new(vec![Box::new(Noop), Box::new(Noop)]);
        let stats = pipeline.run(&mut c).unwrap();
        assert_eq!(stats.gates_before, 2);
        assert_eq!(stats.gates_after, 2);
        assert_eq!(stats.transformations, 0);
    }

    #[test]
    fn default_pipeline_fuses_diagonal_ladder_and_is_idempotent() {
        use smallvec::smallvec;
        // Builder-style controlled-Phase ladder between two H's collapses.
        let mut c = Circuit::new(3, 0);
        c.h(2).unwrap();
        for (t, k) in [(0u32, 2u32), (1, 2), (0, 1)] {
            c.add_gate(aleph_core::GateInstance::controlled(
                aleph_core::Gate::Phase(0.5.into()),
                smallvec![t],
                smallvec![k],
            ))
            .unwrap();
        }
        let mut a = c.clone();
        PassPipeline::default_pipeline().run(&mut a).unwrap();
        // a DiagonalPhase was produced
        assert!(
            a.instructions()
                .iter()
                .any(|i| matches!(i, crate::Instruction::DiagonalPhase(_))),
            "pipeline should produce a DiagonalPhase"
        );
        // idempotent: running the pipeline again changes nothing
        let mut b = a.clone();
        let s2 = PassPipeline::default_pipeline().run(&mut b).unwrap();
        assert_eq!(
            a.len(),
            b.len(),
            "second pipeline run must not change length"
        );
        assert_eq!(s2.transformations, 0, "second pipeline run is a no-op");
    }

    #[test]
    fn default_pipeline_removes_dead_gates() {
        // X(2) is dead (q2 unmeasured); only DeadCodeElim removes it.
        // CancelInversePairs is a no-op on this input, so the count is
        // unaffected by its now running ahead of DCE in the pipeline.
        let mut c = Circuit::new(3, 1);
        c.h(0).unwrap();
        c.x(2).unwrap();
        c.measure(0, 0).unwrap();
        let stats = PassPipeline::default_pipeline().run(&mut c).unwrap();
        assert_eq!(stats.gates_before, 3);
        // After DCE: H(0), Measure(0) (X(2) removed). Fusion leaves the single
        // H as-is. So 2 instructions remain.
        assert_eq!(stats.gates_after, 2);
        assert!(c
            .instructions()
            .iter()
            .all(|i| !i.used_qubits().contains(&2)));
    }
}
