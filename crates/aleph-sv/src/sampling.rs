//! Validated probability scanning, single-draw inverse CDF and Vose alias sampling.
//!
//! Used by [`crate::measure::sample_impl`]. Build cost is `O(n)` over
//! the `n = 2^num_qubits` probability table; per-draw cost is one
//! `rng.gen_range` + one `rng.gen::<f64>()` + one branch.
//!
//! Vose 1991, "A linear algorithm for generating random numbers with
//! a given distribution", Algorithm 3.

use aleph_backend::BackendError;
use rand::{rngs::StdRng, Rng};

/// Shared layout check, before allocating probability or alias buffers.
pub(crate) fn validate_shape(
    qubits: u32,
    len: usize,
    mismatch: &'static str,
) -> Result<(), BackendError> {
    if len == 0 {
        return Err(BackendError::InvalidState {
            reason: "empty state vector",
        });
    }
    let expected = 1usize
        .checked_shl(qubits)
        .ok_or(BackendError::InvalidState {
            reason: "num_qubits exceeds platform usize::BITS",
        })?;
    if len != expected {
        return Err(BackendError::InvalidState { reason: mismatch });
    }
    Ok(())
}

/// Full norm/finiteness validation with an optional probability sink.
/// The same pass validates both stored-probability and allocation-free query paths.
pub(crate) fn validate_norm(
    probabilities: impl Iterator<Item = f64>,
    tolerance: f64,
    norm_error: &'static str,
    mut visit: impl FnMut(f64),
) -> Result<f64, BackendError> {
    let mut total = 0.0;
    for p in probabilities {
        if !p.is_finite() {
            return Err(BackendError::InvalidState {
                reason: "non-finite amplitude norm²",
            });
        }
        total += p;
        visit(p);
    }
    if !total.is_finite() || (total - 1.0).abs() > tolerance {
        return Err(BackendError::InvalidState { reason: norm_error });
    }
    Ok(total)
}

/// Invert one cumulative sum over an already validated immutable distribution.
/// Strict comparison skips zero-mass bins even for a zero random variate.
fn cumulative_index(
    probabilities: impl Iterator<Item = f64>,
    target: f64,
) -> Result<u64, BackendError> {
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (i, p) in probabilities.enumerate() {
        if p > 0.0 {
            last_positive = Some(i as u64);
            cumulative += p;
            if target < cumulative {
                return Ok(i as u64);
            }
        }
    }
    // A multiply may round a variate just below one to the total norm.
    // Use the final positive bin, never a trailing zero-probability bin.
    last_positive.ok_or(BackendError::InvalidState {
        reason: "state has no positive probability",
    })
}

/// One 53-bit uniform draw, scaled by the actual (validated) norm.
/// Consumes one u64 of StdRng, as does AliasTable::draw, including forced states.
pub(crate) fn draw_cumulative(
    rng: &mut StdRng,
    probabilities: impl Iterator<Item = f64>,
    total: f64,
) -> Result<u64, BackendError> {
    cumulative_index(probabilities, rng.gen::<f64>() * total)
}

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
        // Pop both stacks in lockstep. The `while let (Some, Some) =
        // (small.pop(), large.pop())` shape would still pop `large`
        // when `small` is empty, leaking an entry with prob=0.0 and
        // alias=0 — index 0 would then receive that leaked draw and
        // skew uniform distributions. Guard the lengths explicitly.
        while !small.is_empty() && !large.is_empty() {
            let s = small.pop().unwrap();
            let l = large.pop().unwrap();
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
        // Drain leftovers from either stack. In the exactly-normalised
        // case both stacks are empty here; when validate_state has
        // accepted a slightly over- or under-normalised input within
        // its `√n · AMPLITUDE_TOL` drift budget, residual entries
        // hold `scaled[i]` within ≈ ±drift_budget of 1.0. Setting
        // `prob[i] = 1.0` truncates that residual mass (bounded per
        // index by ≤ drift_budget/n), which is well below the 5σ
        // distribution-oracle band and any reasonable downstream
        // tolerance — but is *not* a faithful redistribution of
        // residual mass, only a clean self-alias.
        for i in large.drain(..).chain(small.drain(..)) {
            prob[i as usize] = 1.0;
            alias[i as usize] = i;
        }
        Self { prob, alias }
    }

    /// One draw. Consumes one `u64` of RNG output.
    ///
    /// `n = self.prob.len()` must be a power of two (the only caller
    /// `sample_impl` feeds in `2^num_qubits`-length tables). With that
    /// invariant, splitting a single `u64` into a high-half index and
    /// a low-half threshold is bias-free — `bits & (n − 1)` is exactly
    /// uniform over `[0, n)` for power-of-two `n`. Two-RNG-call shape
    /// regressed n ≤ 10 sample benches by 22–61 % on this machine
    /// (StdRng/ChaCha20 dominates per-shot cost over alias-method
    /// algorithmic savings); a single `u64` draw closes the gap.
    pub(crate) fn draw(&self, rng: &mut StdRng) -> u32 {
        let n = self.prob.len();
        debug_assert!(
            n.is_power_of_two(),
            "AliasTable::draw requires a power-of-two table length, got {n}"
        );
        let bits: u64 = rng.gen();
        let i = ((bits >> 32) as usize) & (n - 1);
        // Low 32 bits as a uniform u ∈ [0, 1): divide by 2^32. The
        // result has 32-bit resolution, which is well below the
        // `prob[i]` precision the build loop produces, and matches the
        // accuracy contract of `f64` thresholds in the alias method.
        let u = (bits as u32 as f64) * (1.0_f64 / (1u64 << 32) as f64);
        if u < self.prob[i] {
            i as u32
        } else {
            self.alias[i]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn cumulative_boundaries_skip_zeros_and_keep_tiny_branches() {
        assert_eq!(
            cumulative_index([0.0, 1e-300, 1.0, 0.0].into_iter(), 0.0).unwrap(),
            1
        );
        assert_eq!(
            cumulative_index([0.0, 0.25, 0.0, 0.75].into_iter(), 0.25).unwrap(),
            3
        );
        assert_eq!(
            cumulative_index([0.0, 1.0, 0.0, 0.0].into_iter(), 1.0).unwrap(),
            1
        );
    }

    #[test]
    fn low_shot_validation_and_rng_contract_all_cpu_layouts() {
        use crate::{CpuState, Fp32CpuState, SoaState};
        use aleph_core::{AlignedBuf, Complex};
        let check = |amps: &[Complex], qubits: u32, valid: bool| {
            let a = CpuState {
                num_qubits: qubits,
                amps: AlignedBuf::from_slice(amps),
            };
            let s = SoaState {
                num_qubits: qubits,
                re: AlignedBuf::from_slice(&amps.iter().map(|a| a.re).collect::<Vec<_>>()),
                im: AlignedBuf::from_slice(&amps.iter().map(|a| a.im).collect::<Vec<_>>()),
            };
            let f = Fp32CpuState {
                num_qubits: qubits,
                amps: AlignedBuf::from_slice(
                    &amps
                        .iter()
                        .map(|a| Complex::<f32>::new(a.re as f32, a.im as f32))
                        .collect::<Vec<_>>(),
                ),
            };
            for shots in [0, 1, 2, 64] {
                let mut ra = StdRng::seed_from_u64(556);
                let mut rs = ra.clone();
                let mut rf = ra.clone();
                let aa = crate::measure::sample_impl(&mut ra, &a, shots);
                let ss = crate::measure_soa::sample_impl_soa(&mut rs, &s, shots);
                let ff = crate::fp32_measure::sample_impl_f32(&mut rf, &f, shots);
                assert_eq!(aa.is_ok(), valid);
                assert_eq!(ss.is_ok(), valid);
                assert_eq!(ff.is_ok(), valid);
                if valid {
                    assert_eq!(aa.unwrap(), ss.unwrap());
                    assert_eq!(ff.unwrap().len(), shots as usize);
                }
                let mut reference = StdRng::seed_from_u64(556);
                if valid {
                    for _ in 0..shots {
                        let _: u64 = reference.gen();
                    }
                }
                let expected: u64 = reference.gen();
                assert_eq!(ra.gen::<u64>(), expected);
                assert_eq!(rs.gen::<u64>(), expected);
                assert_eq!(rf.gen::<u64>(), expected);
            }
        };
        check(&[], 0, false);
        check(&[Complex::new(1.0, 0.0)], 1, false);
        check(&[Complex::new(1.0, 0.0); 2], usize::BITS, false);
        check(
            &[Complex::new(1.0, 0.0), Complex::new(f64::NAN, 0.0)],
            1,
            false,
        );
        check(
            &[Complex::new(1.0, 0.0), Complex::new(0.0, f64::INFINITY)],
            1,
            false,
        );
        check(&[Complex::new(0.25, 0.0); 2], 1, false);
        check(&[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)], 1, true);
        check(&[Complex::new(0.5, 0.0); 4], 2, true);
        check(
            &[
                Complex::new(0.3_f64.sqrt(), 0.0),
                Complex::new(0.7_f64.sqrt(), 0.0),
            ],
            1,
            true,
        );
    }

    #[test]
    fn multi_shot_mapping_remains_alias_bit_exact() {
        use crate::CpuState;
        use aleph_core::{AlignedBuf, Complex};
        let s = CpuState {
            num_qubits: 2,
            amps: AlignedBuf::from_slice(&[
                Complex::new(0.1_f64.sqrt(), 0.0),
                Complex::new(0.2_f64.sqrt(), 0.0),
                Complex::new(0.3_f64.sqrt(), 0.0),
                Complex::new(0.4_f64.sqrt(), 0.0),
            ]),
        };
        for shots in [2, 4, 32, 4096] {
            let mut actual = StdRng::seed_from_u64(556);
            let mut reference = actual.clone();
            let probabilities = crate::measure::validate_state(&s).unwrap();
            let table = AliasTable::build(&probabilities);
            let expected: Vec<_> = (0..shots)
                .map(|_| table.draw(&mut reference) as u64)
                .collect();
            assert_eq!(
                crate::measure::sample_impl(&mut actual, &s, shots).unwrap(),
                expected
            );
        }
    }

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
            assert!(
                dev <= 1654.0,
                "outcome {i}: count {c} deviates by {dev} > 5σ"
            );
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
