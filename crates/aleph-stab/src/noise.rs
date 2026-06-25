//! Pauli noise model for the stabilizer frame sampler ([`crate::sample_noisy`]).
//!
//! The stabilizer engine stays inside the Clifford group, so its noise is
//! Pauli-only by construction (depolarizing channels + classical measurement
//! flips). This is deliberately a separate, simpler type from the state-vector
//! [`NoiseModel`](../../aleph_sv/noise) (which carries general Kraus channels):
//! a depolarizing channel injected into a Pauli frame is a free random bit-flip,
//! whereas amplitude damping is not Clifford and has no frame representation.
//!
//! Channels here attach uniformly to gate arity and to measurements. Per-gate /
//! per-location attachment (Aer-style) can layer on later if a circuit needs it;
//! QEC memory experiments use the uniform form.

/// Pauli noise applied during stabilizer frame sampling.
///
/// `depol1`/`depol2` are depolarizing strengths applied *after* each 1- and
/// 2-qubit Clifford gate respectively: with probability `p` the channel applies
/// a uniformly random non-identity Pauli (one of 3 for 1q, one of 15 for 2q).
/// `measure_flip` is the probability that a measurement's classical bit is
/// reported flipped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PauliNoise {
    /// Depolarizing probability after each 1-qubit gate.
    pub depol1: f64,
    /// Depolarizing probability after each 2-qubit gate.
    pub depol2: f64,
    /// Classical bit-flip probability on each measurement.
    pub measure_flip: f64,
}

impl PauliNoise {
    /// No noise — `sample_noisy` then reproduces the noiseless distribution.
    pub fn none() -> Self {
        PauliNoise {
            depol1: 0.0,
            depol2: 0.0,
            measure_flip: 0.0,
        }
    }

    /// Depolarizing noise after 1q and 2q gates (no measurement error).
    ///
    /// # Panics (debug)
    /// If either probability is outside `[0, 1]`.
    pub fn depolarizing(depol1: f64, depol2: f64) -> Self {
        debug_assert!((0.0..=1.0).contains(&depol1), "depol1 must be in [0,1]");
        debug_assert!((0.0..=1.0).contains(&depol2), "depol2 must be in [0,1]");
        PauliNoise {
            depol1,
            depol2,
            measure_flip: 0.0,
        }
    }

    /// Set the measurement bit-flip probability (builder style).
    ///
    /// # Panics (debug)
    /// If `p` is outside `[0, 1]`.
    pub fn with_measure_flip(mut self, p: f64) -> Self {
        debug_assert!((0.0..=1.0).contains(&p), "measure_flip must be in [0,1]");
        self.measure_flip = p;
        self
    }

    /// Whether the model injects no errors at all.
    pub fn is_noiseless(&self) -> bool {
        self.depol1 == 0.0 && self.depol2 == 0.0 && self.measure_flip == 0.0
    }
}
