//! `Fp32CpuState` — dense `AlignedBuf<Complex<f32>>` of size 2^n for the
//! opt-in single-precision CPU state-vector backend. Mirror of
//! [`crate::state::CpuState`] at half the bytes per amplitude (8 B vs 16).

use aleph_core::{AlignedBuf, Complex};

/// State vector held by [`crate::Fp32SvBackend`].
///
/// Array-of-structs `AlignedBuf<Complex<f32>>`; `amps[i]` is the amplitude
/// of `|i⟩` (bit `q` of `i` = qubit `q`), same MSB convention as
/// [`crate::state::CpuState`] (ADR 0004). 64-byte aligned.
#[derive(Debug, Clone)]
pub struct Fp32CpuState {
    pub(crate) num_qubits: u32,
    pub(crate) amps: AlignedBuf<Complex<f32>>,
}

impl Fp32CpuState {
    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Read-only view of the single-precision amplitude buffer.
    pub fn amplitudes(&self) -> &[Complex<f32>] {
        &self.amps
    }

    /// Widen to `Vec<Complex<f64>>` for oracle / interop comparison. The
    /// FP32→FP64 widening is exact; the 1e-5 oracle tolerance accounts for
    /// the single-precision accumulation error already in `amps`.
    pub fn to_aos_f64(&self) -> Vec<Complex<f64>> {
        self.amps
            .iter()
            .map(|a| Complex::<f64>::new(a.re as f64, a.im as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn getters_and_widen() {
        let s = Fp32CpuState {
            num_qubits: 1,
            amps: AlignedBuf::from_slice(&[
                Complex::<f32>::new(0.5, 0.0),
                Complex::<f32>::new(0.0, -0.25),
            ]),
        };
        assert_eq!(s.num_qubits(), 1);
        assert_eq!(s.amplitudes().len(), 2);
        let w = s.to_aos_f64();
        assert_eq!(w[0], Complex::<f64>::new(0.5, 0.0));
        assert_eq!(w[1], Complex::<f64>::new(0.0, -0.25));
    }
}
