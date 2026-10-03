//! Distributed state vector on CUDA (Phase 6, P6-01a).
//!
//! Executes an `aleph_ir::dist::DistPlan`: `2^g` rank slices of `2^m`
//! amplitudes, local steps through the existing kernels (rank-specialised,
//! optionally fused per rank), exchanges through an [`Exchange`]
//! implementation. [`LocalExchange`] keeps every rank on one device (the
//! single-card development path); NCCL across devices is P6-01b.

use aleph_backend::{Backend, BackendError};
use aleph_core::Complex;

mod device_sv;
mod exchange;

pub use exchange::{Exchange, LocalExchange};

/// What the distributed layer needs from a single-device SV backend beyond
/// [`Backend`]: rank-slice allocation, device-to-device amplitude copies, and
/// host transfer for readout / tests.
pub trait DeviceSv: Backend {
    /// A rank slice of `m` qubits: |0…0⟩ if `rank == 0`, all-zero otherwise.
    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<Self::State, BackendError>;
    /// Copy `len` amplitudes `src[src_off..]` → `dst[dst_off..]` on the device.
    fn copy_amps(
        &mut self,
        src: &Self::State,
        src_off: usize,
        dst: &mut Self::State,
        dst_off: usize,
        len: usize,
    ) -> Result<(), BackendError>;
    /// All amplitudes as complex f64 (FP32 widens).
    fn download(&mut self, st: &Self::State) -> Result<Vec<Complex<f64>>, BackendError>;
    /// A fresh `m`-qubit slice holding `amps` (`amps.len() == 2^m`).
    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<Self::State, BackendError>;
}

/// Distributed-run failure.
#[derive(Debug, thiserror::Error)]
pub enum DistSvError {
    #[error(transparent)]
    Dist(#[from] aleph_ir::dist::DistError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}
