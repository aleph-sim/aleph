//! `MetalSvState` — device-resident FP32 statevector. AoS
//! `DeviceBuffer<Complex<f32>>` of length 2^n; `amps[i]` is the amplitude of
//! `|i⟩` (MSB qubit convention, ADR 0004), matching every other SV backend.

use aleph_core::Complex;

use crate::DeviceBuffer;

/// State vector held by [`crate::MetalSvBackend`]. Unified-memory shared
/// storage: host views are zero-copy windows onto the same bytes the GPU sees.
pub struct MetalSvState {
    pub(crate) num_qubits: u32,
    pub(crate) amps: DeviceBuffer<Complex<f32>>,
}

impl core::fmt::Debug for MetalSvState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `DeviceBuffer` wraps a `metal::Buffer` which is not `Debug`; show the
        // qubit count and amplitude length as a compact summary instead.
        f.debug_struct("MetalSvState")
            .field("num_qubits", &self.num_qubits)
            .field("amps_len", &self.amps.len())
            .finish_non_exhaustive()
    }
}

impl MetalSvState {
    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Zero-copy read-only view of the single-precision amplitude buffer.
    ///
    /// The caller must ensure no GPU command is mid-write (every `apply_gate`
    /// waits before returning, so this holds at the public API boundary).
    pub fn amplitudes_f32(&self) -> &[Complex<f32>] {
        self.amps.as_slice()
    }

    /// Widen to `Vec<Complex<f64>>` for oracle / interop comparison. The
    /// FP32→FP64 widening is exact; the 1e-5 oracle tolerance accounts for the
    /// single-precision accumulation error already in the buffer.
    pub fn to_aos_f64(&self) -> Vec<Complex<f64>> {
        self.amps
            .as_slice()
            .iter()
            .map(|a| Complex::<f64>::new(a.re as f64, a.im as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;

    #[test]
    fn widen_and_view_round_trip() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping state test: no Metal device");
                return;
            }
        };
        let data = [
            Complex::<f32>::new(0.5, 0.0),
            Complex::<f32>::new(0.0, -0.25),
        ];
        let amps = DeviceBuffer::from_slice(&ctx, &data);
        let s = MetalSvState {
            num_qubits: 1,
            amps,
        };
        assert_eq!(s.num_qubits(), 1);
        assert_eq!(s.amplitudes_f32().len(), 2);
        let w = s.to_aos_f64();
        assert_eq!(w[0], Complex::<f64>::new(0.5, 0.0));
        assert_eq!(w[1], Complex::<f64>::new(0.0, -0.25));
    }
}
