//! Adapter trait letting the harness pull amplitudes off any backend
//! state without forcing an amplitudes accessor onto the public
//! `Backend` trait. Implemented for `aleph_sv::CpuState` here; future
//! backends add their own impls in this same module.

use aleph_core::Complex;

pub trait HasAmplitudes {
    fn amplitudes(&self) -> &[Complex];
}

impl HasAmplitudes for aleph_sv::CpuState {
    fn amplitudes(&self) -> &[Complex] {
        // Inherent method already exposes this read-only view.
        aleph_sv::CpuState::amplitudes(self)
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
        let amps = s.amplitudes();
        assert_eq!(amps.len(), 2);
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
    }
}
