//! `MetalContext` owns the long-lived GPU handles (device + command queue) and
//! turns Metal Shading Language source into a compute pipeline at runtime.
//!
//! Runtime compilation (`new_library_with_source`) is deliberate: it uses the
//! Metal framework's built-in compiler, so it needs no `xcrun`, no `metallib`
//! CLI, and no separately-installed Metal Toolchain component (see the design
//! spec). That keeps both local dev and headless CI working today.

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, Library};

use crate::Error;

/// Discover the default Metal device without queues, shaders, or backend construction.
pub fn device_available() -> bool {
    Device::system_default().is_some()
}

static SV_BACKEND_INITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Number of SV backend construction attempts in this process (benchmark/test hook).
#[doc(hidden)]
pub fn sv_backend_init_count() -> u64 {
    SV_BACKEND_INITS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn record_sv_backend_init() {
    SV_BACKEND_INITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// A Metal device plus its command queue.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
}

impl MetalContext {
    /// Acquire the system-default GPU and a command queue.
    ///
    /// Returns [`Error::NoDevice`] when no Metal device is present (e.g. a
    /// headless CI runner) so callers can skip GPU work instead of failing.
    pub fn new() -> Result<Self, Error> {
        let device = Device::system_default().ok_or(Error::NoDevice)?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    /// The underlying Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The command queue for submitting work.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Compile MSL `src` into a library using the runtime compiler.
    fn compile_library(&self, src: &str) -> Result<Library, Error> {
        let options = CompileOptions::new();
        self.device
            .new_library_with_source(src, &options)
            .map_err(Error::ShaderCompile)
    }

    /// Compile `src` and build a compute pipeline for the kernel named `entry`.
    pub fn make_compute_pipeline(
        &self,
        src: &str,
        entry: &str,
    ) -> Result<ComputePipelineState, Error> {
        let library = self.compile_library(src)?;
        let function = library
            .get_function(entry, None)
            .map_err(Error::PipelineCreation)?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(Error::PipelineCreation)
    }
}
