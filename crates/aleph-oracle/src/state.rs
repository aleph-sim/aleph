//! Adapter trait letting the harness pull amplitudes off any backend
//! state without forcing an amplitudes accessor onto the public
//! `Backend` trait.
//!
//! The trait returns an owned `Vec<Complex>` rather than a borrowed
//! slice so a struct-of-arrays backend (`aleph_sv::SoaState`) can
//! materialise its paired `(re, im)` buffers on demand via
//! `to_aos()`. AoS backends pay a one-buffer `.clone()` here; the
//! largest committed oracle fixture is `ghz_10` at 1024 amps × 16 B =
//! 16 KB, well below noise. Hot paths never call this trait.

use aleph_core::Complex;

pub trait HasAmplitudes {
    /// Owned snapshot of the amplitude vector. AoS backends clone the
    /// underlying buffer; SoA backends materialise via `to_aos()`.
    /// Oracle-path only — see module-level docs.
    fn amplitudes(&self) -> Vec<Complex>;
}

impl HasAmplitudes for aleph_sv::CpuState {
    fn amplitudes(&self) -> Vec<Complex> {
        // Clone the inherent read-only view.
        aleph_sv::CpuState::amplitudes(self).to_vec()
    }
}

impl HasAmplitudes for aleph_sv::SoaState {
    fn amplitudes(&self) -> Vec<Complex> {
        self.to_aos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_backend::Backend;
    use aleph_sv::NaiveSvBackend;

    #[test]
    fn fresh_one_qubit_state_is_zero_ket() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let amps = HasAmplitudes::amplitudes(&s);
        assert_eq!(amps.len(), 2);
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
    }

    #[test]
    fn fresh_one_qubit_soa_state_is_zero_ket() {
        use aleph_sv::SoaSvBackend;
        let mut b = SoaSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let amps = HasAmplitudes::amplitudes(&s);
        assert_eq!(amps.len(), 2);
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
    }
}
