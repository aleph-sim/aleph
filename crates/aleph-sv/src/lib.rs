//! `aleph-sv`: naive single-threaded CPU state-vector backend.
//!
//! The reference implementation: simple, correct, and the yardstick
//! every other backend or future optimization is compared against. See
//! `docs/superpowers/specs/2026-05-24-p0-09-backend-naive-sv-design.md`.

mod backend;
mod kernels;
mod measure;
mod measure_soa;
mod sampling;
mod soa_backend;
mod soa_state;
mod state;
mod validation;

pub use backend::NaiveSvBackend;
pub use soa_backend::SoaSvBackend;
pub use soa_state::SoaState;
pub use state::CpuState;
