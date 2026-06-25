//! The [`Decoder`] trait and the [`LogicalErrorResult`] a Monte-Carlo experiment produces.
//!
//! Concrete decoders (MWPM in Q1, Union-Find in Q2, GPU variants in Q3) implement
//! [`Decoder`]; the experiment harness (Q0-04) feeds them syndromes and tallies logical
//! errors into a [`LogicalErrorResult`].

use crate::syndrome::{Correction, Syndrome};

/// A QEC decoder: maps a measured [`Syndrome`] to a predicted [`Correction`].
///
/// Implementations are constructed against a fixed Detector Error Model (so the matching
/// graph / parity-check structure can be built once and reused across shots), which is why
/// `decode` takes only the per-shot syndrome.
pub trait Decoder {
    /// Predict the logical correction for a single syndrome.
    fn decode(&self, syndrome: &Syndrome) -> Correction;
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

    /// A trivial decoder predicting no correction — the Q0-04 `NullDecoder` baseline, used
    /// here to exercise the trait object boundary.
    struct NullDecoder {
        observables: usize,
    }

    impl Decoder for NullDecoder {
        fn decode(&self, _syndrome: &Syndrome) -> Correction {
            Correction::none(self.observables)
        }
    }

    #[test]
    fn decoder_is_object_safe_and_callable() {
        let m = DetectorErrorModel::parse("error(0.1) D0 L0\n").unwrap();
        let dec: Box<dyn Decoder> = Box::new(NullDecoder {
            observables: m.observables,
        });
        let s = Syndrome::from_bits(&[true]);
        assert_eq!(dec.decode(&s), Correction::none(1));
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
