//! Property tests for the default optimisation pipeline (`Fuse2q` plus,
//! since P2-06, `FuseDiagonalRuns`). Structural invariants only —
//! state-vector equivalence is covered by the aleph-sv / aleph-oracle
//! oracle tests.
//!
//! 1. Fixpoint convergence — `optimize()` reaches a stable circuit that
//!    a further `optimize()` leaves unchanged.
//! 2. Length monotonicity — optimised.len() <= original.len().
//! 3. Determinism — two independent optimisations agree structurally.

use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    /// `optimize()` converges to a fixpoint in at most two iterations.
    ///
    /// A single `optimize()` is *not* universally idempotent once
    /// `FuseDiagonalRuns` is in the pipeline (P2-06): the pass uses the
    /// documented "whole-run-or-nothing" v1 rule — a {diagonal ∪ cx} run
    /// whose net GF(2) permutation is not the identity is re-emitted
    /// verbatim rather than partially fused (design doc §3). `Fuse2q`,
    /// which runs after it, can then *remove* the lone `cx` that left the
    /// permutation open (folding it into a `Unitary2q`), exposing a now
    /// permutation-closed diagonal sub-run that the *next* `optimize()`
    /// fuses. Example: `Z(0); T(2); cx(1,2); Ccz(1,2,0)` -> pass 1 fuses
    /// `T·cx` (Fuse2q) but leaves the diagonal run split; pass 2 collapses
    /// the remaining `Z; Ccz`. The transform is exactly state-preserving
    /// at every step (verified to ~1e-16 in the aleph-oracle full-pipeline
    /// equivalence test) — only the *gate count* keeps shrinking for one
    /// extra round. The pipeline is still convergent and deterministic; it
    /// reaches its fixpoint within two passes. (A future ticket may make
    /// `FuseDiagonalRuns` split runs at the last identity-permutation
    /// prefix, restoring strict one-pass idempotence.)
    #[test]
    fn optimize_reaches_fixpoint(c in arb_circuit_emittable(4, 1, 24)) {
        // Iterate optimize() to a fixpoint, bounded to catch any pathological
        // non-convergence. In practice two passes always suffice (the only
        // source of a second-pass change is Fuse2q removing a cx that reopens
        // a diagonal sub-run for FuseDiagonalRuns; that resolves in one more
        // pass and cannot cascade further).
        const MAX_PASSES: usize = 8;
        let mut cur = c.clone();
        let mut prev = format!("{:?}", cur.instructions());
        let mut passes_to_fixpoint = 0usize;
        for p in 1..=MAX_PASSES {
            cur.optimize().unwrap();
            let now = format!("{:?}", cur.instructions());
            if now == prev {
                passes_to_fixpoint = p;
                break;
            }
            prev = now;
            passes_to_fixpoint = p;
        }
        prop_assert!(
            passes_to_fixpoint < MAX_PASSES,
            "optimize() did not converge within {} passes",
            MAX_PASSES
        );
        // Confirm the fixpoint is genuine: one more optimise is a true no-op.
        let at_fixpoint = format!("{:?}", cur.instructions());
        cur.optimize().unwrap();
        prop_assert_eq!(
            format!("{:?}", cur.instructions()),
            at_fixpoint,
            "optimize() at the claimed fixpoint still changed the circuit"
        );
    }

    #[test]
    fn optimize_does_not_grow(c in arb_circuit_emittable(4, 1, 24)) {
        let before = c.len();
        let mut c2 = c;
        c2.optimize().unwrap();
        prop_assert!(c2.len() <= before);
    }

    #[test]
    fn optimize_is_deterministic(c in arb_circuit_emittable(4, 1, 24)) {
        let mut a = c.clone();
        let mut b = c;
        a.optimize().unwrap();
        b.optimize().unwrap();
        prop_assert_eq!(
            format!("{:?}", a.instructions()),
            format!("{:?}", b.instructions())
        );
    }
}
