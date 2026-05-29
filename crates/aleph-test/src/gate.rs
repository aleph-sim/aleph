//! Random `Gate` strategies.  See spec §4.2.

use aleph_core::{Complex, Gate};
use proptest::prelude::*;

/// Random 1-qubit gate.  Vocabulary:
/// H, X, Y, Z, S, Sdg, T, Tdg, Rx(θ), Ry(θ), Rz(θ).
/// Rotation angles ∈ [-2π, 2π].
pub fn arb_1q_gate() -> impl Strategy<Value = Gate> {
    let tau = std::f64::consts::TAU;
    prop_oneof![
        1 => Just(Gate::H),
        3 => Just(Gate::X),
        3 => Just(Gate::Y),
        1 => Just(Gate::Z),
        1 => Just(Gate::S),
        1 => Just(Gate::Sdg),
        1 => Just(Gate::T),
        1 => Just(Gate::Tdg),
        1 => (-tau..=tau).prop_map(|t| Gate::Rx(t.into())),
        1 => (-tau..=tau).prop_map(|t| Gate::Ry(t.into())),
        1 => (-tau..=tau).prop_map(|t| Gate::Rz(t.into())),
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
/// Z, S, Sdg, T, Tdg, Rz(θ), Phase(θ).
pub fn arb_diagonal_1q_gate() -> impl Strategy<Value = Gate> {
    let tau = std::f64::consts::TAU;
    prop_oneof![
        Just(Gate::Z),
        Just(Gate::S),
        Just(Gate::Sdg),
        Just(Gate::T),
        Just(Gate::Tdg),
        (-tau..=tau).prop_map(|t| Gate::Rz(t.into())),
        (-tau..=tau).prop_map(|t| Gate::Phase(t.into())),
    ]
}

/// Random `[[Complex; 4]; 4]` matrix matching exactly the CnotHi
/// canonical pattern (rows 2↔3 swapped, all non-zero entries = +1+0i).
/// Used in property tests for the AoS/SoA CNOT detection + dispatch.
pub fn arb_cnot_hi_matrix() -> impl Strategy<Value = [[Complex; 4]; 4]> {
    Just({
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        m[0][0] = Complex::new(1.0, 0.0);
        m[1][1] = Complex::new(1.0, 0.0);
        m[2][3] = Complex::new(1.0, 0.0);
        m[3][2] = Complex::new(1.0, 0.0);
        m
    })
}

/// Random `[[Complex; 4]; 4]` matrix matching exactly the SWAP pattern.
pub fn arb_swap_matrix() -> impl Strategy<Value = [[Complex; 4]; 4]> {
    Just({
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        m[0][0] = Complex::new(1.0, 0.0);
        m[1][2] = Complex::new(1.0, 0.0);
        m[2][1] = Complex::new(1.0, 0.0);
        m[3][3] = Complex::new(1.0, 0.0);
        m
    })
}

/// Random diagonal 4×4 unitary: each `d[k] = e^{iθ_k}` for independent
/// `θ_k ∈ [-π, π]`.  Covers CZ (θ_3 ≈ π), generic 2q-diag, controlled-
/// Phase (θ_0 = θ_1 = θ_2 = 0).
pub fn arb_diagonal_4x4() -> impl Strategy<Value = [[Complex; 4]; 4]> {
    let pi = std::f64::consts::PI;
    ((-pi..=pi), (-pi..=pi), (-pi..=pi), (-pi..=pi)).prop_map(|(t0, t1, t2, t3)| {
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        m[0][0] = Complex::new(t0.cos(), t0.sin());
        m[1][1] = Complex::new(t1.cos(), t1.sin());
        m[2][2] = Complex::new(t2.cos(), t2.sin());
        m[3][3] = Complex::new(t3.cos(), t3.sin());
        m
    })
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
            // Sanity-check the positive set. `Unitary1qDiag` is
            // accepted for forward-compatibility with P1-10/11/12 work
            // that may extend the strategy to include fused diagonals;
            // the current strategy still emits only the named variants.
            prop_assert!(
                matches!(g, Z | S | Sdg | T | Tdg | Rz(_) | Phase(_) | Unitary1qDiag(_)),
                "unexpected {g:?}",
            );
        }
    }
}
