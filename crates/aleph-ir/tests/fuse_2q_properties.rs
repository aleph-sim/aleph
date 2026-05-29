//! Property tests for the P1-10 `Fuse2q` pass (run via the default
//! pipeline). Structural invariants only — state-vector equivalence is
//! covered by the aleph-sv oracle test.
//!
//! 1. Idempotence — optimising an already-optimised circuit does not
//!    reduce the gate count further (greedy fusion is maximal in one pass).
//! 2. Length monotonicity — optimised.len() <= original.len().
//! 3. Determinism — two independent optimisations agree structurally.

use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn optimize_is_idempotent(c in arb_circuit_emittable(4, 1, 24)) {
        let mut once = c.clone();
        once.optimize().unwrap();
        let after_once = once.len();
        let mut twice = once.clone();
        twice.optimize().unwrap();
        prop_assert_eq!(twice.len(), after_once);
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
