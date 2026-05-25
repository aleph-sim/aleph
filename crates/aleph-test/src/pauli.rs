//! Random `PauliString` strategies.  See spec §4.4.

use aleph_core::{Pauli, PauliString};
use proptest::prelude::*;

/// Random `PauliString` with terms on qubits in `[0, n)`.
///   `mix_xy = false` → Z-only strings (exercises the Z fast path).
///   `mix_xy = true`  → full {I, X, Y, Z} (mixed fallthrough).
/// Coefficient is `1.0`; compose with `.prop_flat_map` if a
/// random coefficient is needed.
pub fn arb_pauli_string(n: u32, mix_xy: bool) -> impl Strategy<Value = PauliString> {
    let alphabet: Vec<Pauli> = if mix_xy {
        vec![Pauli::I, Pauli::X, Pauli::Y, Pauli::Z]
    } else {
        vec![Pauli::I, Pauli::Z]
    };
    let dim = n as usize;
    proptest::collection::vec(proptest::sample::select(alphabet), dim..=dim).prop_map(move |body| {
        let terms: Vec<(u32, Pauli)> = body
            .into_iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p))
            .collect();
        // PauliString::new sorts, dedupes (no dupes possible here),
        // drops I, and rejects non-finite coefficient.
        PauliString::new(1.0, terms).expect("arb_pauli_string produced a valid PauliString")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn z_only_terms_have_only_z(ps in arb_pauli_string(5, false)) {
            // PauliString::new drops I terms; an all-I draw yields an
            // empty terms vec and the loop would assert nothing.
            // Reject those samples as proptest discards so the
            // assertion is never vacuously satisfied.
            prop_assume!(!ps.terms.is_empty());
            for (_, p) in &ps.terms {
                prop_assert_eq!(*p, Pauli::Z);
            }
        }

        #[test]
        fn mixed_terms_are_x_y_or_z(ps in arb_pauli_string(5, true)) {
            prop_assume!(!ps.terms.is_empty());
            for (_, p) in &ps.terms {
                prop_assert!(matches!(p, Pauli::X | Pauli::Y | Pauli::Z));
            }
        }

        #[test]
        fn coefficient_is_one(ps in arb_pauli_string(4, true)) {
            // Coefficient assertion is non-vacuous regardless of
            // terms (the field exists even when terms is empty).
            prop_assert_eq!(ps.coefficient, 1.0);
        }
    }
}
