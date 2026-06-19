//! MPS-on-Metal scaffold backend (P5.5-06): device-resident FP32 site tensors,
//! GPU kernels for 1q apply / two-site contraction / 2q gate-apply, and a
//! host-side `faer` truncated SVD per NN 2q gate.

mod backend;
mod kernel;
mod state;
mod svd;

pub use backend::MetalMpsBackend;
pub use state::MetalMpsState;
