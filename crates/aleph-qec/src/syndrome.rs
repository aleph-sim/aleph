//! Syndrome and correction: the data a [`Decoder`](crate::Decoder) consumes and produces.

/// A measured syndrome: which detectors fired in one shot.
///
/// Stored sparsely as the sorted list of fired detector indices — syndromes are sparse at
/// realistic error rates, and decoders walk the fired set, not the full bit-vector.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Syndrome {
    /// Total number of detectors in the model (the bit-vector length this is a view of).
    pub detectors: usize,
    /// Indices of detectors that fired, sorted ascending and de-duplicated.
    pub fired: Vec<u32>,
}

impl Syndrome {
    /// Build a syndrome from an explicit list of fired detector indices.
    pub fn new(detectors: usize, mut fired: Vec<u32>) -> Self {
        fired.sort_unstable();
        fired.dedup();
        Syndrome { detectors, fired }
    }

    /// Build a syndrome from a dense bit-vector (`bits[d] == true` ⇔ detector `d` fired).
    pub fn from_bits(bits: &[bool]) -> Self {
        let fired = bits
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some(i as u32))
            .collect();
        Syndrome {
            detectors: bits.len(),
            fired,
        }
    }

    /// Number of fired detectors (the syndrome weight).
    pub fn weight(&self) -> usize {
        self.fired.len()
    }

    /// Whether detector `d` fired.
    pub fn is_fired(&self, d: u32) -> bool {
        self.fired.binary_search(&d).is_ok()
    }
}

/// A decoder's predicted correction: which logical observables it believes were flipped.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Correction {
    /// `observable_flips[o] == true` ⇔ the decoder predicts observable `o` is flipped.
    pub observable_flips: Vec<bool>,
}

impl Correction {
    /// A correction predicting no observable flips, for `observables` observables.
    pub fn none(observables: usize) -> Self {
        Correction {
            observable_flips: vec![false; observables],
        }
    }

    /// Build from an explicit per-observable flip vector.
    pub fn new(observable_flips: Vec<bool>) -> Self {
        Correction { observable_flips }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bits_collects_fired_indices() {
        let s = Syndrome::from_bits(&[false, true, false, true, true]);
        assert_eq!(s.detectors, 5);
        assert_eq!(s.fired, vec![1, 3, 4]);
        assert_eq!(s.weight(), 3);
        assert!(s.is_fired(3));
        assert!(!s.is_fired(2));
    }

    #[test]
    fn new_sorts_and_dedups() {
        let s = Syndrome::new(8, vec![5, 1, 5, 3]);
        assert_eq!(s.fired, vec![1, 3, 5]);
    }
}
