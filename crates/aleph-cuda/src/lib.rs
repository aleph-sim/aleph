//! CUDA GPU foundation for aleph: a device context, its default stream, and
//! typed device buffers — the Phase 5 (P5-01) plumbing that later kernel and
//! backend tickets build on.
//!
//! Everything is gated on `cfg(all(target_os = "linux", feature = "cuda"))`.
//! Without the `cuda` feature (the default), or off Linux, this crate is
//! intentionally empty so that `cargo build --workspace`, the macOS/Metal
//! track, and the default Linux build are all unaffected. This mirrors how
//! `aleph-metal` stays empty without its `metal` feature.
//!
//! Bindings go through [`cudarc`](https://github.com/coreylowman/cudarc) with
//! `dynamic-loading`, so the GPU code path compiles even where no CUDA SDK is
//! installed (e.g. the self-hosted CI runner); `libcuda` is `dlopen`ed at
//! runtime on a real GPU host.

#[cfg(all(target_os = "linux", feature = "cuda"))]
mod buffer;
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod context;
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod sv;

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use buffer::{device_alloc_count, DeviceBuffer};
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use context::CudaContext;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use sv::{CudaSvBackend, CudaSvState, MAX_CUDA_QUBITS};

/// Errors from the CUDA foundation layer.
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No usable CUDA device at the requested ordinal (e.g. a headless runner
    /// with no GPU). Callers use this to skip GPU work instead of failing.
    #[error("no CUDA device available at ordinal {0}")]
    NoDevice(usize),
    /// NVRTC failed to compile the kernel source to PTX.
    #[error("CUDA kernel compilation failed: {0}")]
    Compile(String),
    /// Any error surfaced by the CUDA driver API via `cudarc`.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}
