//! GPU (CUDA) FP64 state-vector backend (P5-02).
//!
//! `CudaSvBackend` compiles the gate kernels once via NVRTC and launches them
//! per gate; `CudaSvState` is the device-resident amplitude buffer. Correctness
//! is gated by an oracle test vs the CPU `NaiveSvBackend` (ADR 0004 convention).

mod backend;
// Custom diagonal-gate kernels (P5-06). `pub(crate)` so the cuStateVec backend
// embeds the same `DiagKernels` and diverts diagonal gates to it.
pub(crate) mod diag;
// FP32 / mixed-precision backend (P5.10-03).
mod fp32;
mod kernel;
// `pub(crate)` so the cuStateVec backend reuses the identical host-side readout
// (measure / sample / expectation / probabilities), keeping both GPU backends'
// measurement distributions byte-for-byte aligned with the CPU oracle.
// Out-of-core (host-memory paged) executor for n > MAX_CUDA_QUBITS (P5.10-02).
mod paged;
// FP32 out-of-core paged executor → n=32 single-GPU reach (P5.11-01).
mod paged_f32;
pub(crate) mod readout;
mod state;

pub use backend::CudaSvBackend;
pub use fp32::{CudaSvBackendF32, CudaSvStateF32, MAX_CUDA_QUBITS_F32};
pub use paged::{paged_pass_counts, PagedSvState};
pub use paged_f32::PagedSvStateF32;
pub use state::{CudaSvState, MAX_CUDA_QUBITS};

// The oracle harness pulls amplitudes through `HasAmplitudes`. Implemented here
// (the state's home crate) per the orphan rule; widening the device FP64 buffer
// to `Vec<Complex<f64>>` is exact.
impl aleph_oracle::HasAmplitudes for CudaSvState {
    fn amplitudes(&self) -> Vec<aleph_core::Complex> {
        self.amplitudes_vec()
    }
}

// Same for the FP32 state — amplitudes are widened f32 → `Complex<f64>` (the
// ~1e-5 FP32 error is what the oracle tolerance accommodates).
impl aleph_oracle::HasAmplitudes for CudaSvStateF32 {
    fn amplitudes(&self) -> Vec<aleph_core::Complex> {
        self.amplitudes_vec()
    }
}
