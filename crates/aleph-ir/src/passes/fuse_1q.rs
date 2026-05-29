//! `Fuse1qRuns` — collapses runs of adjacent uncontrolled 1q gates
//! on the same qubit into a single fused gate. See
//! `docs/superpowers/specs/2026-05-29-p1-09-fuse-1q-design.md`.

use super::{Pass, PassError, PassStats};
use crate::Circuit;

pub struct Fuse1qRuns;

impl Pass for Fuse1qRuns {
    fn name(&self) -> &'static str {
        "Fuse1qRuns"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.len();
        // Stub — real implementation lands in Task 5.
        Ok(PassStats {
            gates_before: n,
            gates_after: n,
            transformations: 0,
        })
    }
}
