//! Tableau well-formedness is preserved under arbitrary Clifford
//! evolution: the destabilizer/stabilizer pair stays symplectic.

use aleph_stab::Tableau;
use proptest::prelude::*;
use rand::SeedableRng;

// A random gate op over the 11-gate Clifford set on `n` qubits, encoded
// as (opcode, q0, q1). We apply via the tableau's public methods.
#[derive(Debug, Clone)]
enum Op {
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    Cnot(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Iswap(usize, usize),
    IswapDg(usize, usize),
}

fn op_strategy(n: usize) -> impl Strategy<Value = Op> {
    let q = 0..n;
    let q2 = (0..n, 0..n).prop_filter("distinct", |(a, b)| a != b);
    prop_oneof![
        q.clone().prop_map(Op::H),
        q.clone().prop_map(Op::S),
        q.clone().prop_map(Op::Sdg),
        q.clone().prop_map(Op::X),
        q.clone().prop_map(Op::Y),
        q.clone().prop_map(Op::Z),
        q2.clone().prop_map(|(a, b)| Op::Cnot(a, b)),
        q2.clone().prop_map(|(a, b)| Op::Cz(a, b)),
        q2.clone().prop_map(|(a, b)| Op::Swap(a, b)),
        q2.clone().prop_map(|(a, b)| Op::Iswap(a, b)),
        q2.prop_map(|(a, b)| Op::IswapDg(a, b)),
    ]
}

fn apply(t: &mut Tableau, op: &Op) {
    match *op {
        Op::H(a) => t.h(a),
        Op::S(a) => t.s(a),
        Op::Sdg(a) => t.sdg(a),
        Op::X(a) => t.x_gate(a),
        Op::Y(a) => t.y_gate(a),
        Op::Z(a) => t.z_gate(a),
        Op::Cnot(a, b) => t.cnot(a, b),
        Op::Cz(a, b) => t.cz(a, b),
        Op::Swap(a, b) => t.swap(a, b),
        Op::Iswap(a, b) => t.iswap(a, b),
        Op::IswapDg(a, b) => t.iswap_dg(a, b),
    }
    .unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn symplectic_invariant_preserved(
        ops in {
            let n = 6;
            proptest::collection::vec(op_strategy(n), 0..40)
        }
    ) {
        let n = 6;
        let mut t = Tableau::new(n);
        for op in &ops {
            apply(&mut t, op);
        }
        // destab i anticommutes with stab i; commutes with all other rows.
        for i in 0..n {
            prop_assert!(t.rows_anticommute(i, n + i), "destab {i} ⊥ stab {i} broken");
            for j in 0..n {
                if j != i {
                    prop_assert!(!t.rows_anticommute(i, n + j));
                    prop_assert!(!t.rows_anticommute(n + i, n + j));
                    prop_assert!(!t.rows_anticommute(i, j));
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Measurement leaves a well-formed tableau: after a random Clifford
    /// circuit and one measurement, the destabilizer/stabilizer structure
    /// is still symplectic.
    #[test]
    fn symplectic_invariant_preserved_after_measure(
        ops in {
            let n = 6;
            proptest::collection::vec(op_strategy(n), 0..40)
        },
        target in 0usize..6,
        seed in any::<u64>(),
    ) {
        let n = 6;
        let mut t = Tableau::new(n);
        for op in &ops {
            apply(&mut t, op);
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let _ = t.measure(target, &mut rng).unwrap();
        for i in 0..n {
            prop_assert!(t.rows_anticommute(i, n + i), "destab {i} ⊥ stab {i} broken after measure");
            for j in 0..n {
                if j != i {
                    prop_assert!(!t.rows_anticommute(i, n + j));
                    prop_assert!(!t.rows_anticommute(n + i, n + j));
                }
            }
        }
    }
}
