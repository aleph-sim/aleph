//! MPS-on-Metal scaffold backend (P5.5-06): device-resident FP32 site tensors and
//! GPU kernels for 1q apply / two-site contraction / 2q gate-apply. The per-gate
//! two-site SVD is GPU-resident too since P5.7-03 (a one-sided Jacobi kernel), with
//! the f64 `faer` SVD kept as the CPU fallback.

mod backend;
mod gpu_jacobi;
mod jacobi;
mod kernel;
mod state;
mod svd;

pub use backend::MetalMpsBackend;
pub use state::MetalMpsState;
