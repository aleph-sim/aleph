//! [`CudaContext`] owns the long-lived GPU handles: the device's primary
//! context and a default stream to schedule work on. Both are reference-counted
//! inside `cudarc`, so cloning the handles is cheap; this wrapper keeps them
//! together and gives the rest of aleph a small, stable surface.

use std::sync::Arc;

use cudarc::driver::{CudaContext as RawContext, CudaStream};

use crate::pool::MemPool;
use crate::Error;

/// Discover a CUDA device without a context, NVRTC, or backend construction.
/// Missing drivers/devices and driver-query failures mean unavailable. Discovery
/// does not promise that subsequent backend initialization or compilation succeeds.
pub fn device_available() -> bool {
    // SAFETY: cudarc only opens/closes candidate driver libraries here; no device
    // pointers or caller-owned buffers are involved. Avoid its panicking loader
    // on hosts where the optional CUDA driver is not installed.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return false;
    }
    device_count().is_ok_and(|n| n > 0)
}

/// Fail clearly when discovery succeeded but the optional compiler is missing.
pub(crate) fn require_nvrtc() -> Result<(), Error> {
    // SAFETY: cudarc opens/closes compiler library candidates only; no pointers
    // or buffers are passed. Keep its panicking loader out of normal init errors.
    if unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        Ok(())
    } else {
        Err(Error::Compile(
            "NVRTC compiler library is unavailable; install the CUDA NVRTC runtime".into(),
        ))
    }
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
        // SAFETY: only checks whether cudarc can open a driver library. Return
        // a normal absent-device error instead of entering its panicking loader.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(Error::NoDevice(ordinal));
        }
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

    /// A clone of this context that submits work to `stream` instead of the
    /// default one. Shares the same device context and memory pool (both
    /// reference-counted), so it is cheap. The out-of-core overlapped paged
    /// executor (P5.11-02) uses this to run gate kernels on a dedicated
    /// non-blocking compute stream — distinct from the legacy default stream,
    /// which would otherwise serialise against the copy streams and defeat the
    /// gather/compute/scatter overlap.
    pub(crate) fn with_stream(&self, stream: Arc<CudaStream>) -> CudaContext {
        CudaContext {
            ctx: self.ctx.clone(),
            stream,
            pool: self.pool,
        }
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
/// Number of CUDA devices the driver reports (0 on a GPU-less host).
pub fn device_count() -> Result<usize, Error> {
    match RawContext::device_count() {
        Ok(n) => Ok(usize::try_from(n).unwrap_or(0)),
        Err(e) => Err(classify_init_error(e, 0)),
    }
}

fn classify_init_error(e: cudarc::driver::DriverError, ordinal: usize) -> Error {
    use cudarc::driver::sys::CUresult;
    match e.0 {
        CUresult::CUDA_ERROR_NO_DEVICE | CUresult::CUDA_ERROR_INVALID_DEVICE => {
            Error::NoDevice(ordinal)
        }
        _ => Error::Driver(e),
    }
}
