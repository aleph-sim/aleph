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
    pub(crate) fn build(p: &[f64]) -> Self {
        // Vose 1991, "A linear algorithm for generating random numbers
        // with a given distribution", Algorithm 3.
        //
        // We require `p.len() >= 1`; the only caller (`sample_impl`) is
        // gated by `validate_state` which rejects empty states.
        let n = p.len();
        debug_assert!(n >= 1, "AliasTable::build requires non-empty p");
        let mut prob = vec![0.0_f64; n];
        let mut alias = vec![0u32; n];
        if n == 1 {
            prob[0] = 1.0;
            alias[0] = 0;
            return Self { prob, alias };
        }
        let n_f = n as f64;
        let mut scaled: Vec<f64> = p.iter().map(|q| n_f * q).collect();
        // Partition indices into `small` (scaled < 1) and `large`
        // (scaled ≥ 1). Allocate full-size to avoid reallocations.
        let mut small: Vec<u32> = Vec::with_capacity(n);
        let mut large: Vec<u32> = Vec::with_capacity(n);
        for (i, &s) in scaled.iter().enumerate() {
            if s < 1.0 {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }
        while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
            prob[s as usize] = scaled[s as usize];
            alias[s as usize] = l;
            // The "+ scaled[s]) - 1.0" grouping mirrors the Vose paper
            // and keeps round-off symmetric across the two stacks.
            let new_l = (scaled[l as usize] + scaled[s as usize]) - 1.0;
            scaled[l as usize] = new_l;
            if new_l < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }
        // Drain leftovers from either stack: due to FP drift the
        // remaining indices have scaled ≈ 1.0, so the table degenerates
        // to a self-pointing entry with probability 1.
        for i in large.drain(..).chain(small.drain(..)) {
            prob[i as usize] = 1.0;
            alias[i as usize] = i;
        }
        Self { prob, alias }
    }

    /// One draw. Consumes one `u32` and one `f64` of RNG output.
    pub(crate) fn draw(&self, _rng: &mut StdRng) -> u32 {
        todo!("Task 4");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn single_index_always_returns_0() {
        let t = AliasTable::build(&[1.0]);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            assert_eq!(t.draw(&mut rng), 0);
        }
    }

    #[test]
    fn degenerate_1_0_0_0_always_returns_0() {
        let t = AliasTable::build(&[1.0, 0.0, 0.0, 0.0]);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..1000 {
            assert_eq!(t.draw(&mut rng), 0);
        }
    }

    #[test]
    fn bell_only_returns_0_or_3() {
        // |Φ+⟩ has support on indices 0 and 3 only.
        let t = AliasTable::build(&[0.5, 0.0, 0.0, 0.5]);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..10_000 {
            let i = t.draw(&mut rng);
            assert!(i == 0 || i == 3, "got {i}");
        }
    }

    #[test]
    fn uniform_8_outcomes_within_5_sigma_at_1m_draws() {
        let p = [0.125_f64; 8];
        let t = AliasTable::build(&p);
        let mut rng = StdRng::seed_from_u64(0);
        const N: u64 = 1_000_000;
        let mut counts = [0u64; 8];
        for _ in 0..N {
            counts[t.draw(&mut rng) as usize] += 1;
        }
        // σ = √(N · p · (1-p)) = √(1e6 · 0.125 · 0.875) ≈ 330.7; 5σ ≈ 1654.
        let mean = (N as f64) * 0.125;
        for (i, c) in counts.iter().enumerate() {
            let dev = (*c as f64 - mean).abs();
            assert!(dev <= 1654.0, "outcome {i}: count {c} deviates by {dev} > 5σ");
        }
    }

    #[test]
    fn near_normalised_1_plus_1e_minus_15_builds_and_draws() {
        // Total ≈ 1 + 1e-15; well inside `validate_state`'s drift budget.
        // Build must not panic and `draw` must return a valid index.
        let p = [0.25, 0.25, 0.25, 0.25 + 1e-15];
        let t = AliasTable::build(&p);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            let i = t.draw(&mut rng);
            assert!(i < 4);
        }
    }
}
