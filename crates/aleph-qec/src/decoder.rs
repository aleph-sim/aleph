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
/// true flips (XOR). The rate is the fraction of such shots, with a 95% confidence half-width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogicalErrorResult {
    /// Total shots run.
    pub shots: u64,
    /// Shots where the decoder mispredicted at least one observable.
    pub logical_errors: u64,
    /// `logical_errors / shots` (0.0 when `shots == 0`).
    pub rate: f64,
    /// 95% confidence half-width on `rate` (normal approximation; 0.0 when `shots == 0`).
    pub ci95: f64,
}

impl LogicalErrorResult {
    /// Build a result from raw tallies, computing the rate and a normal-approximation 95% CI.
    pub fn new(shots: u64, logical_errors: u64) -> Self {
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
