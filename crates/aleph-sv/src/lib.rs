//! `aleph-sv`: naive single-threaded CPU state-vector backend.
//!
//! The reference implementation: simple, correct, and the yardstick
//! every other backend or future optimization is compared against. See
//! `docs/superpowers/specs/2026-05-24-p0-09-backend-naive-sv-design.md`.

pub mod noise;

mod backend;
mod fp32_backend;
mod fp32_measure;
mod fp32_state;
#[cfg(not(any(test, feature = "internal-bench")))]
mod kernels;
#[cfg(any(test, feature = "internal-bench"))]
pub mod kernels;
mod measure;
mod measure_soa;
mod perm;
mod sampling;
mod soa_backend;
mod soa_state;
mod state;
mod validation;

pub use backend::NaiveSvBackend;
pub use fp32_backend::Fp32SvBackend;
pub use fp32_state::Fp32CpuState;
pub use soa_backend::SoaSvBackend;
pub use soa_state::SoaState;
pub use state::CpuState;
