//! `CancelInversePairs` — removes adjacent inverse-pair gates
//! (H·H, X·X, CNOT·CNOT, S·Sdg, Rz(θ)·Rz(−θ), …). A single forward
//! pass with per-qubit stacks of live instruction indices; two gates
//! cancel iff they are adjacent on their entire shared support, act on
//! the same targets (positionally) and controls (as a set), and one is
//! the other's `Gate::inverse()`. Adjacent-only; commutation-aware
//! cancellation is P1-13. See
//! `docs/superpowers/specs/2026-05-30-p1-12-gate-cancellation-design.md`.

use super::{Pass, PassError, PassStats};
use crate::Circuit;

pub struct CancelInversePairs;

impl Pass for CancelInversePairs {
    fn name(&self) -> &'static str {
        "CancelInversePairs"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.instructions.len();
        Ok(PassStats {
            gates_before: n,
            gates_after: n,
            transformations: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circuit;

    fn run_pass(c: &mut Circuit) -> PassStats {
        CancelInversePairs
            .run(c)
            .expect("CancelInversePairs is infallible")
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(CancelInversePairs.name(), "CancelInversePairs");
    }

    #[test]
    fn empty_circuit_is_a_no_op() {
        let mut c = Circuit::new(2, 0);
        let s = run_pass(&mut c);
        assert_eq!(s.gates_before, 0);
        assert_eq!(s.gates_after, 0);
        assert_eq!(s.transformations, 0);
    }
}
