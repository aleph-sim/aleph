//! Property tests for the P1-11 `DeadCodeElim` pass (run standalone).
//!
//! 1. Monotonicity — DCE never grows the circuit.
//! 2. Idempotence — a second DCE removes nothing more.
//! 3. Determinism — two independent runs agree structurally.

use aleph_ir::passes::{DeadCodeElim, Pass};
use aleph_ir::Circuit;
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

fn dce(c: &Circuit) -> Circuit {
    let mut out = c.clone();
    DeadCodeElim.run(&mut out).unwrap();
    out
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn dce_does_not_grow(c in arb_circuit_emittable(4, 1, 24)) {
        let before = c.len();
        prop_assert!(dce(&c).len() <= before);
    }

    #[test]
    fn dce_is_idempotent(c in arb_circuit_emittable(4, 1, 24)) {
        let once = dce(&c);
        let n1 = once.len();
        let twice = dce(&once);
        prop_assert_eq!(twice.len(), n1);
    }

    #[test]
    fn dce_is_deterministic(c in arb_circuit_emittable(4, 1, 24)) {
        let a = dce(&c);
        let b = dce(&c);
        prop_assert_eq!(
            format!("{:?}", a.instructions()),
            format!("{:?}", b.instructions())
        );
    }
}
