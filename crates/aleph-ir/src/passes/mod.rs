//! `aleph-ir::passes` — IR-level optimisation passes.
//!
//! Each pass implements [`Pass`]. A [`PassPipeline`] runs an ordered
//! sequence of passes over a [`Circuit`], aggregating per-pass
//! [`PassStats`]. The default pipeline ships
//! [`relabel::RelabelQubits`], [`cancel::CancelInversePairs`],
//! [`dce::DeadCodeElim`], [`fuse_diagonal::FuseDiagonalRuns`],
//! [`fuse_1q::Fuse1qRuns`], [`fuse_2q::Fuse2q`], [`fuse_kq::FuseKq`], and
//! [`tile_block::TileBlock`] — in that pipeline order
//! (`RelabelQubits` runs FIRST, permuting qubit indices so high-traffic
//! qubits occupy low/cache-local bit positions and recording the
//! permutation `π` on the circuit for the run driver to undo; cancellation
//! precedes DCE so DCE can clean up gates newly exposed as dead by
//! cancellation; diagonal fusion precedes `Fuse2q` so raw `cx`s are still
//! absorbable; `FuseKq` runs before `TileBlock`, merging the dense 1q/2q
//! blocks into ≥3q `UnitaryKq` blocks; `TileBlock` runs last, grouping the
//! post-fusion low-target runs for the tile-major executor; see
//! [`PassPipeline::default_pipeline`]). Later tickets add more passes that
//! plug in by being pushed onto the pipeline.
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
pub mod relabel;
pub mod tile_block;

pub use cancel::CancelInversePairs;
pub use commute::gates_commute;
pub use dce::DeadCodeElim;
pub use fuse_1q::Fuse1qRuns;
pub use fuse_2q::Fuse2q;
pub use fuse_diagonal::FuseDiagonalRuns;
pub use fuse_kq::FuseKq;
pub use relabel::RelabelQubits;
pub use tile_block::TileBlock;

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
    /// `[RelabelQubits, CancelInversePairs, DeadCodeElim, FuseDiagonalRuns, Fuse1qRuns, Fuse2q, FuseKq, TileBlock]`;
    /// later passes are appended here as they ship.
    ///
    /// [`RelabelQubits`] runs **first**, before any fusion. It permutes
    /// qubit indices so the highest-traffic qubits land on low (cache-local)
    /// bit positions, maximising how many gates `TileBlock` can later confine
    /// to a tile, and records the permutation `π[logical] = physical` on the
    /// circuit ([`Circuit::qubit_permutation`](crate::Circuit)). The run
    /// driver un-permutes the final state back to logical order. The pass is
    /// conservative — it only commits a non-identity permutation when doing so
    /// strictly increases the tile-confinable gate count — so correctness
    /// never hinges on the heuristic, only the achieved speedup.
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
    ///
    /// [`FuseKq`] runs **second-to-last**, consuming the dense 1q/2q blocks the
    /// earlier passes produced (`Unitary1q`/`Unitary2q`) into ≥3q
    /// `UnitaryKq` blocks (≤ `max_qubits`, default 4) where the cost
    /// model says it pays. An emitted
    /// [`Gate::UnitaryKq`](aleph_core::Gate) is opaque to all earlier
    /// passes — they have already run by the time it exists — and is a
    /// hard fence for `FuseKq` itself, so re-running the pipeline over
    /// its own output does not re-fuse it. The convergence caveat above
    /// is unchanged: the pipeline remains convergent and deterministic
    /// (fixpoint within two passes), not strictly one-pass idempotent.
    ///
    /// [`TileBlock`] runs **last**, grouping maximal runs of post-fusion
    /// gates whose targets are all `< tile_bits` into
    /// [`Instruction::TiledBlock`](crate::Instruction) so the backend can
    /// apply them tile-major (one DRAM pass per run). At the small qubit
    /// counts used by existing oracle tests (`n ≤ 20`), `tile_bits = 15`
    /// means the entire state fits in a single tile; the executor's
    /// degenerate single-tile path runs, which is bit-exact with the
    /// unblocked path (validated in Task 4). `TiledBlock` is a hard fence
    /// for `TileBlock` itself, so a second pipeline pass is a no-op.
    pub fn default_pipeline() -> Self {
        Self::new(vec![
            Box::new(RelabelQubits::default()),
            Box::new(CancelInversePairs),
            Box::new(DeadCodeElim),
            Box::new(FuseDiagonalRuns),
            Box::new(Fuse1qRuns),
            Box::new(Fuse2q),
            Box::new(FuseKq::default()),
            Box::new(TileBlock::default()),
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
    fn default_pipeline_fuses_kq_block_and_converges() {
        use smallvec::smallvec;
        // A chain of 2q gates across qubits 0-1-2-3 fuses (after Fuse2q produces
        // Unitary2q blocks) into a dense >=3q UnitaryKq via FuseKq.
        let mut c = Circuit::new(4, 0);
        // ry on each (non-diagonal, gives Fuse2q something to absorb with the cx)
        for q in 0..4u32 {
            c.add_gate(aleph_core::GateInstance::new(
                aleph_core::Gate::Ry(0.5.into()),
                smallvec![q],
            ))
            .unwrap();
        }
        c.cnot(0, 1).unwrap();
        c.cnot(1, 2).unwrap();
        c.cnot(2, 3).unwrap();
        let mut a = c.clone();
        PassPipeline::default_pipeline().run(&mut a).unwrap();
        // At least one fused dense block (UnitaryKq) was produced.
        assert!(
            a.instructions().iter().any(|i| matches!(i, crate::Instruction::Gate(g) if matches!(g.gate, aleph_core::Gate::UnitaryKq{..}))),
            "pipeline should produce a UnitaryKq"
        );
        // Convergence: optimize again reaches a fixpoint (a 2nd run is a no-op
        // OR converges; mirror the existing convergence expectation).
        let mut b = a.clone();
        let s2 = PassPipeline::default_pipeline().run(&mut b).unwrap();
        assert_eq!(
            a.len(),
            b.len(),
            "2nd pipeline run must not change instruction count"
        );
        assert_eq!(s2.transformations, 0, "2nd pipeline run is a no-op");
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
