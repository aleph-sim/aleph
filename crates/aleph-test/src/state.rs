//! Random normalised state vectors.  See spec §4.1.

use aleph_core::Complex;
use proptest::prelude::*;

/// Random normalised state vector of `n` qubits.  Output length is
/// `2^n`; total norm² lies within `validate_state`'s drift budget
/// (`√n · AMPLITUDE_TOL`).
///
/// Samples (re, im) ∈ [-1, 1] uniformly per amplitude then
/// renormalises.  Not uniformly distributed on the Bloch sphere —
/// intentional: pathological near-degenerate states are part of
/// the input space we want to surface.
pub fn arb_state_vector(n: u32) -> impl Strategy<Value = Vec<Complex>> {
    let dim = 1usize << n;
    proptest::collection::vec((-1.0_f64..=1.0, -1.0_f64..=1.0), dim..=dim).prop_map(|pairs| {
        let mut amps: Vec<Complex> = pairs
            .into_iter()
            .map(|(re, im)| Complex::new(re, im))
            .collect();
        let norm2: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        // All-zero is possible but vanishingly unlikely; bias to a
        // valid state by mapping the degenerate case to |0…0⟩.
        if norm2 < 1e-300 {
            amps[0] = Complex::new(1.0, 0.0);
            return amps;
        }
        let inv = norm2.sqrt().recip();
        for a in &mut amps {
            *a *= Complex::new(inv, 0.0);
        }
        amps
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a strategy that draws `(n, amps)` so the consumer
    /// proptest sees both `n` and the amplitude vector at once.
    /// `prop_flat_map` is the standard way to feed a runtime
    /// `n` into a derived strategy.
    fn arb_n_and_state() -> impl Strategy<Value = (u32, Vec<Complex>)> {
        (1u32..=6).prop_flat_map(|n| arb_state_vector(n).prop_map(move |amps| (n, amps)))
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn output_length_is_2_to_n((n, amps) in arb_n_and_state()) {
            prop_assert_eq!(amps.len(), 1usize << n);
        }

        #[test]
        fn output_is_normalised((_n, amps) in arb_n_and_state()) {
            let total: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            // Drift budget: √dim · AMPLITUDE_TOL.  Same as validate_state.
            let budget = (amps.len() as f64).sqrt() * 1e-10;
            prop_assert!((total - 1.0).abs() <= budget, "total = {total}, budget = {budget}");
        }
    }
}
