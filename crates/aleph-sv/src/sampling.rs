//! Vose's alias method for O(1)-per-draw discrete sampling.
//!
//! Used by [`crate::measure::sample_impl`]. Build cost is `O(n)` over
//! the `n = 2^num_qubits` probability table; per-draw cost is one
//! `rng.gen_range` + one `rng.gen::<f64>()` + one branch.
//!
//! Vose 1991, "A linear algorithm for generating random numbers with
//! a given distribution", Algorithm 3.

use rand::{rngs::StdRng, Rng};

pub(crate) struct AliasTable {
    /// `prob[i]` is the threshold at index `i`: draw `u ∈ [0,1)`;
    /// if `u < prob[i]` return `i`, else return `alias[i]`.
    prob: Vec<f64>,
    alias: Vec<u32>,
}

impl AliasTable {
    /// Build from a normalised probability vector. `p` must sum to
    /// `1 ± validate_state`'s drift budget; callers go through
    /// `crate::measure::validate_state` first.
    pub(crate) fn build(_p: &[f64]) -> Self {
        todo!("Task 3");
    }

    /// One draw. Consumes one `u32` and one `f64` of RNG output.
    pub(crate) fn draw(&self, _rng: &mut StdRng) -> u32 {
        todo!("Task 4");
    }
}
