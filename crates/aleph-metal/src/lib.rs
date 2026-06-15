//! Metal GPU foundation for aleph: device, command queue, host-visible
//! buffers, and runtime shader compilation.
//!
//! Everything is gated on `cfg(all(target_os = "macos", feature = "metal"))`.
//! On any other target, or without the `metal` feature, this crate is
//! intentionally empty so that default and Linux builds are unaffected. The
//! physics (statevector, gate kernels) lands in later Phase 5.5 tickets; this
//! crate is plumbing only.

#[cfg(all(target_os = "macos", feature = "metal"))]
mod buffer;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod context;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod sv;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use buffer::DeviceBuffer;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use context::MetalContext;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use sv::{AmpsF32, MetalSvBackend, MetalSvState};

/// Errors from the Metal foundation layer.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No system-default Metal device (e.g. a headless CI runner). Callers use
    /// this variant to skip GPU work gracefully rather than fail.
    #[error("no Metal device available on this system")]
    NoDevice,
    /// `new_library_with_source` rejected the MSL source.
    #[error("Metal shader compilation failed: {0}")]
    ShaderCompile(String),
    /// Building the compute pipeline (function lookup or PSO creation) failed.
    #[error("Metal pipeline creation failed: {0}")]
    PipelineCreation(String),
}
