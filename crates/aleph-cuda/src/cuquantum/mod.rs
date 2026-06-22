//! NVIDIA cuStateVec (cuQuantum) integration (P5-03).
//!
//! [`CuStateVecBackend`] is an optional GPU state-vector [`aleph_backend::Backend`]
//! that delegates gate application to `custatevecApplyMatrix` while sharing the
//! hand-written CUDA backend's device state ([`crate::sv::CudaSvState`]) and
//! host-side readout. It is the performance reference we benchmark our own
//! kernels against and the max-performance path for users on NVIDIA hardware.
//!
//! Gated behind the `cuquantum` feature (a superset of `cuda`): `build.rs`
//! links `libcustatevec` only then, so the default `--features cuda` build — and
//! the CUDA-less CI runner — never need cuQuantum installed.
//!
//! The `State` type is `CudaSvState`, which already implements
//! `aleph_oracle::HasAmplitudes` (in [`crate::sv`]), so cuStateVec runs drop into
//! the same oracle harness as the hand-written backend.

mod backend;
mod sys;

pub use backend::CuStateVecBackend;
