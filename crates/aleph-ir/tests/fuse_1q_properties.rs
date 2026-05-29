//! Property tests for the P1-09 fusion pass.
//!
//! Invariants:
//! 1. Determinism — running `optimize()` twice on independent clones
//!    yields the same `Vec<Instruction>`.
//! 2. Length monotonicity — `optimized.len() <= original.len()`.
//! 3. Per-qubit touch monotonicity — for every qubit q, the count of
//!    instructions touching q does not increase under fusion.

use aleph_ir::Circuit;
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

fn touches(c: &Circuit, q: u32) -> usize {
    c.instructions()
        .iter()
        .filter(|inst| inst.used_qubits().contains(&q))
        .count()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn fusion_is_deterministic(c in arb_circuit_emittable(4, 1, 24)) {
        let mut a = c.clone();
        let mut b = c.clone();
        a.optimize().unwrap();
        b.optimize().unwrap();
        prop_assert_eq!(a.instructions().len(), b.instructions().len());
        // Structural equality — Vec<Instruction> derives Debug; compare
        // via debug strings (Complex doesn't impl Eq; structural-debug
        // comparison is exact for deterministic outputs).
        prop_assert_eq!(format!("{:?}", a.instructions()), format!("{:?}", b.instructions()));
    }

    #[test]
    fn fusion_does_not_grow_circuit(c in arb_circuit_emittable(4, 1, 24)) {
        let before = c.len();
        let mut c2 = c;
        c2.optimize().unwrap();
        prop_assert!(c2.len() <= before);
    }

    #[test]
    fn fusion_does_not_grow_per_qubit_touches(c in arb_circuit_emittable(4, 1, 24)) {
        let before: Vec<usize> = (0..c.num_qubits()).map(|q| touches(&c, q)).collect();
        let mut c2 = c;
        c2.optimize().unwrap();
        for q in 0..c2.num_qubits() {
            prop_assert!(touches(&c2, q) <= before[q as usize]);
        }
    }
}
