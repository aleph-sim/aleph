//! GPU (CUDA) FP64 state-vector backend (P5-02).
//!
//! `CudaSvBackend` compiles the gate kernels once via NVRTC and launches them
//! per gate; `CudaSvState` is the device-resident amplitude buffer. Correctness
//! is gated by an oracle test vs the CPU `NaiveSvBackend` (ADR 0004 convention).

mod backend;
// Custom diagonal-gate kernels (P5-06). `pub(crate)` so the cuStateVec backend
// embeds the same `DiagKernels` and diverts diagonal gates to it.
pub(crate) mod diag;
mod kernel;
// `pub(crate)` so the cuStateVec backend reuses the identical host-side readout
// (measure / sample / expectation / probabilities), keeping both GPU backends'
// measurement distributions byte-for-byte aligned with the CPU oracle.
pub(crate) mod readout;
mod state;

pub use backend::CudaSvBackend;
pub use state::{CudaSvState, MAX_CUDA_QUBITS};

// The oracle harness pulls amplitudes through `HasAmplitudes`. Implemented here
// (the state's home crate) per the orphan rule; widening the device FP64 buffer
// to `Vec<Complex<f64>>` is exact.
impl aleph_oracle::HasAmplitudes for CudaSvState {
    fn amplitudes(&self) -> Vec<aleph_core::Complex> {
        self.amplitudes_vec()
    }
}
