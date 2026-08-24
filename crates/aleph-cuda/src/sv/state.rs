//! `CudaSvState` — a device-resident FP64 statevector.
//!
//! The amplitudes live in one `DeviceBuffer<f64>` of length `2 · 2^n`, storing
//! `[re0, im0, re1, im1, …]` — the exact byte layout the kernels read as
//! `cplx*`. `amps[i]` is the amplitude of `|i⟩` (MSB qubit convention, ADR
//! 0004), matching every other SV backend.

use aleph_core::Complex;

use crate::{CudaContext, DeviceBuffer, Error};

/// Largest state this backend will allocate. `n = 30` is 16 GiB of FP64
/// amplitudes (`2^30 · 16 B`); the shift `1 << target` with `target ≤ 29` stays
/// well inside `u32`. Larger states need the host-migration path (P5-04/05).
pub const MAX_CUDA_QUBITS: u32 = 30;

/// State vector held by [`crate::CudaSvBackend`].
///
/// Gates are launched asynchronously on the context's default stream; the first
/// host read (`to_host`/`amplitudes`) synchronizes via `DeviceBuffer::to_vec`,
/// so callers never observe a half-applied state. The state keeps a cloned
/// [`CudaContext`] so readout works through `&self` with no backend reference.
pub struct CudaSvState {
    pub(crate) num_qubits: u32,
    /// Interleaved `[re, im]` amplitudes, length `2 · 2^n`.
    pub(crate) amps: DeviceBuffer<f64>,
    pub(crate) ctx: CudaContext,
    /// Reusable device scratch for the `apply_kq` matrix, grown on demand to
    /// avoid a per-2q-gate allocation (a real pool is P5-04).
    pub(crate) mat_scratch: Option<DeviceBuffer<f64>>,
}

impl core::fmt::Debug for CudaSvState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaSvState")
            .field("num_qubits", &self.num_qubits)
            .field("amps_len", &self.amps.len())
            .finish_non_exhaustive()
    }
}

impl CudaSvState {
    /// Allocate `|0…0⟩`: `2 · 2^n` zeroed `f64`, then set `re` of amplitude 0
    /// to 1. `alloc_zeros` runs device-side (no `2^n` host buffer), and the
    /// `write(&[1.0])` reuses that allocation to set the single leading element.
    pub(crate) fn allocate(ctx: &CudaContext, num_qubits: u32) -> Result<Self, Error> {
        let n_amps = 1usize << num_qubits;
        let mut amps = DeviceBuffer::<f64>::zeros(ctx, 2 * n_amps)?;
        amps.write(ctx, &[1.0])?;
        Ok(Self {
            num_qubits,
            amps,
            ctx: ctx.clone(),
            mat_scratch: None,
        })
    }

    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Download the interleaved `[re, im]` amplitude buffer to the host,
    /// synchronizing the stream so all queued gates have completed.
    pub(crate) fn to_host(&self) -> Vec<f64> {
        self.amps
            .to_vec(&self.ctx)
            .expect("device->host amplitude copy")
    }

    /// The amplitudes as `Vec<Complex<f64>>` (one per basis state). The
    /// `HasAmplitudes` oracle hook and interop go through this.
    pub fn amplitudes_vec(&self) -> Vec<Complex<f64>> {
        let host = self.to_host();
        host.as_chunks::<2>()
            .0
            .iter()
            .map(|&[re, im]| Complex::new(re, im))
            .collect()
    }
}
