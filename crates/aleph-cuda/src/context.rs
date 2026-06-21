//! [`CudaContext`] owns the long-lived GPU handles: the device's primary
//! context and a default stream to schedule work on. Both are reference-counted
//! inside `cudarc`, so cloning the handles is cheap; this wrapper keeps them
//! together and gives the rest of aleph a small, stable surface.

use std::sync::Arc;

use cudarc::driver::{CudaContext as RawContext, CudaStream};

use crate::Error;

/// A CUDA device context plus its default stream.
///
/// Cheap to `clone` — both handles are reference-counted (`Arc`) inside
/// `cudarc`, so a clone shares the same device context and stream. The
/// state vector keeps a clone so its host-readout paths can copy device→host
/// without threading the backend's context through every call.
#[derive(Clone)]
pub struct CudaContext {
    ctx: Arc<RawContext>,
    stream: Arc<CudaStream>,
}

impl CudaContext {
    /// Acquire device `ordinal` (0 = first GPU) and its default stream.
    ///
    /// Returns [`Error::NoDevice`] when the driver reports the device is absent
    /// or invalid (e.g. a headless CI runner) so callers can skip GPU work
    /// rather than fail; any other driver failure surfaces as
    /// [`Error::Driver`].
    pub fn new(ordinal: usize) -> Result<Self, Error> {
        let ctx = RawContext::new(ordinal).map_err(|e| classify_init_error(e, ordinal))?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    /// The default stream for submitting copies and (later) kernel launches.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The underlying device context, for APIs that need it directly.
    pub fn raw(&self) -> &Arc<RawContext> {
        &self.ctx
    }
}

/// Map a context-creation `DriverError` to [`Error::NoDevice`] when it means
/// "there is no such GPU", so a GPU-less host skips gracefully instead of
/// treating it as a hard failure. Other codes stay [`Error::Driver`].
fn classify_init_error(e: cudarc::driver::DriverError, ordinal: usize) -> Error {
    use cudarc::driver::sys::CUresult;
    match e.0 {
        CUresult::CUDA_ERROR_NO_DEVICE | CUresult::CUDA_ERROR_INVALID_DEVICE => {
            Error::NoDevice(ordinal)
        }
        _ => Error::Driver(e),
    }
}
