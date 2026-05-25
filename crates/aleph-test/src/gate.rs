//! Random `Gate` strategies.  See spec §4.2.

use aleph_core::Gate;
use proptest::prelude::*;

/// Random 1-qubit gate.  Vocabulary:
/// H, X, Y, Z, S, Sdg, T, Tdg, Rx(θ), Ry(θ), Rz(θ).
/// Rotation angles ∈ [-2π, 2π].
pub fn arb_1q_gate() -> impl Strategy<Value = Gate> {
    let tau = std::f64::consts::TAU;
    prop_oneof![
        Just(Gate::H),
        Just(Gate::X),
        Just(Gate::Y),
        Just(Gate::Z),
        Just(Gate::S),
        Just(Gate::Sdg),
        Just(Gate::T),
        Just(Gate::Tdg),
        (-tau..=tau).prop_map(|t| Gate::Rx(t.into())),
        (-tau..=tau).prop_map(|t| Gate::Ry(t.into())),
        (-tau..=tau).prop_map(|t| Gate::Rz(t.into())),
    ]
}

/// Random 2-qubit gate.  Vocabulary: Cnot, Cz, Swap, Iswap, IswapDg.
pub fn arb_2q_gate() -> impl Strategy<Value = Gate> {
    prop_oneof![
        Just(Gate::Cnot),
        Just(Gate::Cz),
        Just(Gate::Swap),
        Just(Gate::Iswap),
        Just(Gate::IswapDg),
    ]
}

/// Union of `arb_1q_gate` and `arb_2q_gate`, weighted ~70/30
/// toward 1-qubit (matches typical circuit density).
pub fn arb_gate() -> impl Strategy<Value = Gate> {
    prop_oneof![
        7 => arb_1q_gate(),
        3 => arb_2q_gate(),
    ]
}

/// Diagonal-only 1q subset for the
/// "leaves-magnitudes-unchanged" invariant.  Vocabulary:
/// Z, S, Sdg, T, Tdg, Rz(θ).
pub fn arb_diagonal_1q_gate() -> impl Strategy<Value = Gate> {
    let tau = std::f64::consts::TAU;
    prop_oneof![
        Just(Gate::Z),
        Just(Gate::S),
        Just(Gate::Sdg),
        Just(Gate::T),
        Just(Gate::Tdg),
        (-tau..=tau).prop_map(|t| Gate::Rz(t.into())),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn arb_1q_gate_arity_is_one(g in arb_1q_gate()) {
            prop_assert_eq!(g.arity(), 1);
        }

        #[test]
        fn arb_2q_gate_arity_is_two(g in arb_2q_gate()) {
            prop_assert_eq!(g.arity(), 2);
        }

        #[test]
        fn arb_gate_arity_is_one_or_two(g in arb_gate()) {
            let a = g.arity();
            prop_assert!(a == 1 || a == 2, "got arity {a}");
        }

        #[test]
        fn arb_diagonal_1q_gate_excludes_non_diagonal(g in arb_diagonal_1q_gate()) {
            use Gate::*;
            // The strategy emits only diagonal 1q variants.  Rx/Ry/H/X/Y
            // would be a strategy bug.
            prop_assert!(!matches!(g, Rx(_) | Ry(_) | H | X | Y), "got non-diagonal {g:?}");
            // Sanity-check the positive set.
            prop_assert!(matches!(g, Z | S | Sdg | T | Tdg | Rz(_)), "unexpected {g:?}");
        }
    }
}
