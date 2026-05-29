//! `aleph-ir::passes` — IR-level optimisation passes.
//!
//! Each pass implements [`Pass`]. A [`PassPipeline`] runs an ordered
//! sequence of passes over a [`Circuit`], aggregating per-pass
//! [`PassStats`]. Phase-1 ships only [`fuse_1q::Fuse1qRuns`]; later
//! tickets (P1-10/11/12/13) add more passes that plug in by being
//! pushed onto the pipeline.

use crate::Circuit;
use thiserror::Error;

pub mod dce;
pub mod fuse_1q;
pub mod fuse_2q;

pub use fuse_1q::Fuse1qRuns;
pub use fuse_2q::Fuse2q;

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

    /// Phase-1 default pipeline. Currently `[Fuse1qRuns, Fuse2q]`; later
    /// passes are appended here as they ship.
    pub fn default_pipeline() -> Self {
        Self::new(vec![Box::new(Fuse1qRuns), Box::new(Fuse2q)])
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
}
