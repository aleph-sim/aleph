//! Naive distributed planner (filled in Task 3).
use super::{DistError, DistLayout, DistPlan, Router};
use crate::Circuit;
use aleph_core::GateInstance;
use smallvec::SmallVec;

/// Logical qubits of `g` that must be local for `g` to run without exchange.
pub fn required_local(_g: &GateInstance) -> SmallVec<[u32; 4]> {
    SmallVec::new()
}

/// Build a distributed plan for `circuit` under `layout`.
pub fn plan(_c: &Circuit, layout: DistLayout, _r: Router) -> Result<DistPlan, DistError> {
    Err(DistError::BadLayout {
        n: layout.n,
        g: layout.g,
    })
}
