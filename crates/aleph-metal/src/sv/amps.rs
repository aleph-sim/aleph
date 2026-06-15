//! Amplitude accessor seams for the Metal SV state.
//!
//! `AmpsF32` is the native single-precision readout (mirrors the CLI's
//! `AmpsF64`). `HasAmplitudes` (from `aleph-oracle`) widens to f64 so the
//! existing oracle harness can compare a Metal run against any other backend.

use aleph_core::Complex;

use super::state::MetalSvState;

/// Native single-precision amplitude readout for a GPU state vector.
pub trait AmpsF32 {
    /// Zero-copy view of the amplitude buffer (`amps[i]` = amplitude of `|i⟩`).
    fn amplitudes_f32(&self) -> &[Complex<f32>];
}

impl AmpsF32 for MetalSvState {
    fn amplitudes_f32(&self) -> &[Complex<f32>] {
        MetalSvState::amplitudes_f32(self)
    }
}

impl aleph_oracle::HasAmplitudes for MetalSvState {
    fn amplitudes(&self) -> Vec<Complex<f64>> {
        self.to_aos_f64()
    }
}
