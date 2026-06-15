//! FP32 state-vector backend on Metal (P5.5-02). Device-resident statevector,
//! generic single-qubit GPU kernel, host-side readout.

mod amps;
mod kernel;
mod state;

pub use amps::AmpsF32;
pub use state::MetalSvState;
