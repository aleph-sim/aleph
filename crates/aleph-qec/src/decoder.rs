//! The [`Decoder`] trait and the [`LogicalErrorResult`] a Monte-Carlo experiment produces.
//!
//! Concrete decoders (MWPM in Q1, Union-Find in Q2, GPU variants in Q3) implement
//! [`Decoder`]; the experiment harness (Q0-04) feeds them syndromes and tallies logical
//! errors into a [`LogicalErrorResult`].

use crate::error::Result;
use crate::syndrome::{Correction, Syndrome};

/// A QEC decoder: maps a measured [`Syndrome`] to a predicted [`Correction`].
///
/// Implementations are constructed against a fixed Detector Error Model (so the matching
/// graph / parity-check structure can be built once and reused across shots), which is why
/// `decode` takes only the per-shot syndrome.
pub trait Decoder {
    /// Predict the logical correction for a single syndrome.
    fn decode(&self, syndrome: &Syndrome) -> Correction;

    /// Predict corrections for a batch of syndromes at once.
    ///
    /// The default loops [`decode`](Self::decode) — correct for any in-process decoder. It is
    /// a separate, *fallible* method because some decoders (notably the external
    /// [`PyMatchingOracle`](crate::PyMatchingOracle), which shells out to PyMatching) decode a
    /// whole batch in one round-trip and can fail at the process boundary. The Monte-Carlo
    /// harness ([`run_dem_experiment`](crate::run_dem_experiment)) decodes through this method,
    /// so a batch decoder never pays one subprocess per shot.
    fn decode_batch(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
        Ok(syndromes.iter().map(|s| self.decode(s)).collect())
    }
}

/// A trivial decoder that always predicts *no* correction.
///
/// The Q0-04 baseline: it is the worst possible decoder (it ignores the syndrome entirely),
/// so its logical-error rate is just the rate at which the raw noise flips an observable. It
/// exists to exercise the harness end-to-end before a real decoder lands in Q1, and to anchor
/// the `p = 0 ⇒ rate = 0` invariant.
#[derive(Clone, Copy, Debug)]
pub struct NullDecoder {
    observables: usize,
}

impl NullDecoder {
    /// A `NullDecoder` over a model with `observables` logical observables.
    pub fn new(observables: usize) -> Self {
        NullDecoder { observables }
    }
}

impl Decoder for NullDecoder {
    fn decode(&self, _syndrome: &Syndrome) -> Correction {
        Correction::none(self.observables)
    }
}

/// Outcome of a logical-error-rate Monte-Carlo run.
///
/// A *logical error* is a shot where the decoder's predicted observable flips differ from the
/// true flips (XOR). Use [`Self::interval95`] for confidence bounds on independent,
/// identically distributed Bernoulli shots. `ci95` remains the legacy Wald half-width.
/// Weighted, stratified and correlated estimators require their own uncertainty model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogicalErrorResult {
    /// Total shots run.
    pub shots: u64,
    /// Shots where the decoder mispredicted at least one observable.
    pub logical_errors: u64,
    /// `logical_errors / shots` (0.0 when `shots == 0`).
    pub rate: f64,
    /// Legacy Wald half-width, retained unchanged for source/data compatibility.
    /// Invalid at zero/all failures and unreliable for rare events. Use [`Self::interval95`].
    pub ci95: f64,
}

impl LogicalErrorResult {
    /// Validate a direct-sampling tally before computing the legacy result fields.
    pub fn try_new(shots: u64, logical_errors: u64) -> Result<Self> {
        if logical_errors > shots {
            return Err(crate::Error::InvalidTally {
                shots,
                errors: logical_errors,
            });
        }
        Ok(Self::new(shots, logical_errors))
    }

    /// Two-sided, equal-tailed 95% Clopper–Pearson interval for direct Bernoulli shots.
    ///
    /// Zero shots return `[0, 1]`: there is no estimated probability or evidence.
    /// This is not a one-sided 95% upper limit. For zero failures its upper endpoint
    /// is `1 - 0.025^(1/shots)`. Coverage is at least 95% in exact arithmetic,
    /// generally conservative because binomial counts are discrete.
    ///
    /// Revalidates the public tally fields; cached `rate`/`ci95` are not trusted.
    /// Returns an error for invalid counts or an unvalidated numerical evaluation.
    ///
    /// ```
    /// # use aleph_qec::LogicalErrorResult;
    /// let result = LogicalErrorResult::try_new(1000, 0)?;
    /// let bounds = result.interval95()?;
    /// assert_eq!(bounds.lower, 0.0);
    /// assert!(bounds.upper > 0.003);
    /// # Ok::<(), aleph_qec::Error>(())
    /// ```
    pub fn interval95(&self) -> Result<BinomialInterval95> {
        BinomialInterval95::from_counts(self.shots, self.logical_errors)
    }

    /// Legacy infallible constructor, computing the unchanged Wald half-width.
    /// Invalid tallies retain their counts with NaN rate/width; new callers should
    /// use [`Self::try_new`] to receive an error, and [`Self::interval95`] for bounds.
    pub fn new(shots: u64, logical_errors: u64) -> Self {
        if logical_errors > shots {
            return Self {
                shots,
                logical_errors,
                rate: f64::NAN,
                ci95: f64::NAN,
            };
        }
        if shots == 0 {
            return LogicalErrorResult {
                shots: 0,
                logical_errors: 0,
                rate: 0.0,
                ci95: 0.0,
            };
        }
        let n = shots as f64;
        let rate = logical_errors as f64 / n;
        // Wald interval half-width: 1.96 * sqrt(p(1-p)/n).
        let ci95 = 1.96 * (rate * (1.0 - rate) / n).sqrt();
        LogicalErrorResult {
            shots,
            logical_errors,
            rate,
            ci95,
        }
    }
}

/// Explicit probability bounds for a two-sided 95% Clopper–Pearson interval.
/// Only direct independent Bernoulli counts are accepted by the constructor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinomialInterval95 {
    /// Lower probability bound.
    pub lower: f64,
    /// Upper probability bound.
    pub upper: f64,
}

impl BinomialInterval95 {
    /// Compute equal-tailed binomial bounds; zero trials produce the full `[0, 1]` range.
    pub fn from_counts(shots: u64, errors: u64) -> Result<Self> {
        use statrs::function::beta::checked_beta_reg;
        if errors > shots {
            return Err(crate::Error::InvalidTally { shots, errors });
        }
        if shots == 0 {
            return Ok(Self {
                lower: 0.0,
                upper: 1.0,
            });
        }
        let n = shots as f64;
        // Stable boundary expression even for u64::MAX; avoid subtracting near-one powers.
        let boundary = -(0.025_f64.ln() / n).exp_m1();
        if errors == 0 {
            return Ok(Self {
                lower: 0.0,
                upper: boundary,
            });
        }
        if errors == shots {
            return Ok(Self {
                // Do not round a nontrivial uncertainty range to the singleton {1}.
                lower: (1.0 - boundary).next_down().max(0.0),
                upper: 1.0,
            });
        }
        // Beyond 2^53, neighbouring integer beta shapes are not representable in f64.
        if shots >= (1_u64 << 53) {
            return Err(crate::Error::IntervalNumerics { shots, errors });
        }
        // Inversion of binomial tails via beta quantiles. NIST §7.2.4.1:
        // https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
        let quantile = |a: f64, b: f64, target: f64, lower_side: bool| -> Result<f64> {
            // Bound the solver's work and retain the conservative side of the bracket.
            // The upstream inverse has unbounded Newton loops for extreme shapes.
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..128 {
                let mid = lo + (hi - lo) * 0.5;
                if mid == lo || mid == hi {
                    break;
                }
                let cdf = checked_beta_reg(a, b, mid)
                    .map_err(|_| crate::Error::IntervalNumerics { shots, errors })?;
                if !cdf.is_finite() || !(0.0..=1.0).contains(&cdf) {
                    return Err(crate::Error::IntervalNumerics { shots, errors });
                }
                if cdf < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let value = if lower_side { lo } else { hi };
            let tail = checked_beta_reg(a, b, value)
                .map_err(|_| crate::Error::IntervalNumerics { shots, errors })?;
            if !value.is_finite()
                || !tail.is_finite()
                || !(0.0..=1.0).contains(&value)
                || (tail - target).abs() > 1e-7
            {
                return Err(crate::Error::IntervalNumerics { shots, errors });
            }
            Ok(value)
        };
        let lower = quantile(errors as f64, (shots - errors + 1) as f64, 0.025, true)?;
        let upper = quantile((errors + 1) as f64, (shots - errors) as f64, 0.975, false)?;
        let rate = errors as f64 / n;
        if lower > rate || upper < rate {
            return Err(crate::Error::IntervalNumerics { shots, errors });
        }
        Ok(Self { lower, upper })
    }

    /// Whether these marginal intervals overlap. This does not establish equivalence
    /// or account for paired shots, multiple comparisons or an equivalence margin.
    pub fn overlaps(self, other: Self) -> bool {
        self.lower <= other.upper && other.lower <= self.upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::DetectorErrorModel;

    #[test]
    fn decoder_is_object_safe_and_callable() {
        let m = DetectorErrorModel::parse("error(0.1) D0 L0\n").unwrap();
        let dec: Box<dyn Decoder> = Box::new(NullDecoder::new(m.observables));
        let s = Syndrome::from_bits(&[true]);
        assert_eq!(dec.decode(&s), Correction::none(1));
    }

    #[test]
    fn decode_batch_default_loops_decode() {
        let dec = NullDecoder::new(2);
        let batch = vec![
            Syndrome::from_bits(&[true, false]),
            Syndrome::from_bits(&[false, true, true]),
        ];
        let out = dec.decode_batch(&batch).unwrap();
        assert_eq!(out, vec![Correction::none(2), Correction::none(2)]);
    }

    #[test]
    fn logical_error_result_rate_and_ci() {
        let r = LogicalErrorResult::new(1000, 50);
        assert_eq!(r.rate, 0.05);
        assert!((r.ci95 - 1.96 * (0.05 * 0.95 / 1000.0f64).sqrt()).abs() < 1e-12);

        let z = LogicalErrorResult::new(0, 0);
        assert_eq!(z.rate, 0.0);
        assert_eq!(z.ci95, 0.0);
    }
}
