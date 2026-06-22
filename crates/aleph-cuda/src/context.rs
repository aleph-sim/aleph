//! [`CudaContext`] owns the long-lived GPU handles: the device's primary
//! context and a default stream to schedule work on. Both are reference-counted
//! inside `cudarc`, so cloning the handles is cheap; this wrapper keeps them
//! together and gives the rest of aleph a small, stable surface.

use std::sync::Arc;

use cudarc::driver::{CudaContext as RawContext, CudaStream};

use crate::pool::MemPool;
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
    /// The device's stream-ordered memory pool, tuned to retain freed blocks
    /// for reuse (P5-04). `None` when the device has no pool support, in which
    /// case allocations fall back to synchronous `cuMemAlloc`.
    pool: Option<MemPool>,
}

impl CudaContext {
    /// Acquire device `ordinal` (0 = first GPU) and its default stream.
    ///
    /// On a pool-capable device the default memory pool is configured to retain
    /// freed blocks so repeated allocate/free (many small circuits) reuses
    /// memory instead of round-tripping the OS (P5-04).
    ///
    /// Returns [`Error::NoDevice`] when the driver reports the device is absent
    /// or invalid (e.g. a headless CI runner) so callers can skip GPU work
    /// rather than fail; any other driver failure surfaces as
    /// [`Error::Driver`].
    pub fn new(ordinal: usize) -> Result<Self, Error> {
        let ctx = RawContext::new(ordinal).map_err(|e| classify_init_error(e, ordinal))?;
        let stream = ctx.default_stream();
        // Only meaningful with the stream-ordered allocator; otherwise cudarc
        // uses synchronous alloc/free and there is no pool to tune.
        let pool = if ctx.has_async_alloc() {
            Some(MemPool::configure(ordinal)?)
        } else {
            None
        };
        Ok(Self { ctx, stream, pool })
    }

    /// The default stream for submitting copies and (later) kernel launches.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The underlying device context, for APIs that need it directly.
    pub fn raw(&self) -> &Arc<RawContext> {
        &self.ctx
    }

    /// Block until all work queued on the default stream completes. Needed
    /// before reading pool byte-counts so async frees have actually executed.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.stream.synchronize().map_err(Error::Driver)
    }

    /// Whether the retaining memory pool is active (device supports pools).
    pub fn pool_enabled(&self) -> bool {
        self.pool.is_some()
    }

    /// Bytes the memory pool currently holds reserved from the OS (live +
    /// cached-free). Flat in steady state once warm ⇒ frees are being reused.
    /// `None` when there is no pool.
    pub fn pool_reserved_bytes(&self) -> Option<u64> {
        self.pool.as_ref().and_then(|p| p.reserved_bytes().ok())
    }

    /// Bytes currently handed out by the pool (live allocations). `None` when
    /// there is no pool. Returns to ~0 once states drop and the stream syncs.
    pub fn pool_used_bytes(&self) -> Option<u64> {
        self.pool.as_ref().and_then(|p| p.used_bytes().ok())
    }

    /// Release pool memory cached above `min_keep` bytes back to the OS.
    /// Synchronize first so pending async frees are trimmable. No-op without a
    /// pool.
    pub fn trim_pool(&self, min_keep: usize) -> Result<(), Error> {
        match &self.pool {
            Some(p) => p.trim(min_keep),
            None => Ok(()),
        }
    }

    /// Override the pool's release threshold (bytes retained before the pool
    /// returns memory to the OS). Test/benchmark hook for the retain-vs-release
    /// A/B; `u64::MAX` retains all, `0` is the un-tuned default. No-op without a
    /// pool.
    #[doc(hidden)]
    pub fn set_pool_release_threshold(&self, bytes: u64) -> Result<(), Error> {
        match &self.pool {
            Some(p) => p.set_release_threshold(bytes),
            None => Ok(()),
        }
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
