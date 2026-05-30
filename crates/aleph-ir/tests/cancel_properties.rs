//! Property tests for the P1-12 `CancelInversePairs` pass (run standalone).
//!
//! 1. Idempotence — after one pass no adjacent inverse pairs remain, so a
//!    second pass changes nothing (fixed point in one sweep).
//! 2. Non-growth + count consistency — the pass never adds instructions and
//!    accounts for exactly two removed gates per cancellation event.
//!
//! These live as an integration test (not an inline `#[cfg(test)] mod`)
//! because `arb_circuit_emittable` comes from `aleph-test`, which itself
//! depends on `aleph-ir`; only the external-crate view of `Circuit`
//! unifies with the strategy's output, matching the sibling
//! `dce_properties.rs` / `fuse_*_properties.rs` pattern.

use aleph_ir::passes::{CancelInversePairs, Pass};
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    // Running the pass a second time changes nothing: after one pass
    // there are no adjacent inverse pairs left, so the fixed point is
    // reached in one sweep.
    #[test]
    fn idempotent(c in arb_circuit_emittable(4, 2, 24)) {
        let mut once = c.clone();
        CancelInversePairs.run(&mut once).unwrap();
        let len_once = once.instructions().len();

        let mut twice = once.clone();
        let s = CancelInversePairs.run(&mut twice).unwrap();
        prop_assert_eq!(s.transformations, 0);
        prop_assert_eq!(twice.instructions().len(), len_once);
    }

    // The pass never adds instructions and accounts for exactly two
    // removed gates per cancellation event.
    #[test]
    fn never_grows_and_counts_are_consistent(c in arb_circuit_emittable(4, 2, 24)) {
        let mut cc = c.clone();
        let s = CancelInversePairs.run(&mut cc).unwrap();
        prop_assert!(s.gates_after <= s.gates_before);
        prop_assert_eq!(s.gates_before - s.gates_after, (s.transformations as usize) * 2);
    }
}
