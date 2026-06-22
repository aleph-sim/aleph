//! Device memory pool (P5-04).
//!
//! `cudarc` already routes every `CudaSlice` allocation through CUDA's
//! **stream-ordered** allocator (`cuMemAllocAsync`) and frees through
//! `cuMemFreeAsync` on drop, *on devices that support memory pools* (CUDA 11.2+;
//! true for our sm_89 box). The catch: the device's default pool ships with a
//! **release threshold of 0**, so every async-freed block is handed straight
//! back to the OS at the next synchronization — and the next allocation of the
//! same size pays a fresh `cuMemAlloc` again. For a workload of many small
//! circuits (allocate `|0…0⟩`, run, read out, drop, repeat) that is a real
//! per-circuit `cudaMalloc`/`cudaFree` cost.
//!
//! Raising the pool's [`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`] to "retain
//! everything" turns the default pool into a caching pool: freed blocks stay
//! reserved and the next same-size allocation is a pool hit (no OS round-trip).
//! This is the allocator-pool deliverable for P5-04, and because *both* GPU
//! backends allocate through [`crate::CudaContext`], both get it for free with
//! no per-backend change.
//!
//! See NVIDIA's "Using the CUDA Stream-Ordered Memory Allocator" (part 1).

use core::ffi::c_void;

use cudarc::driver::sys;

use crate::Error;

/// Retain every freed block in the pool (never release to the OS until the
/// process exits or an explicit [`MemPool::trim`]). `u64::MAX` is the documented
/// "hold everything" sentinel for the release threshold.
const RETAIN_ALL: u64 = u64::MAX;

/// A handle to a device's default stream-ordered memory pool, configured to
/// retain freed blocks for reuse.
///
/// The handle refers to the **device default** pool, which is owned by the
/// driver for the device's lifetime — we never create or destroy it — so this
/// type is a plain `Copy` handle and cloning a [`crate::CudaContext`] that holds
/// one just copies the handle.
#[derive(Clone, Copy)]
pub(crate) struct MemPool {
    pool: sys::CUmemoryPool,
}

impl MemPool {
    /// Configure device `ordinal`'s default memory pool to retain freed blocks.
    ///
    /// The caller must only invoke this when the device supports memory pools
    /// (`CudaContext::has_async_alloc()`); otherwise allocations are synchronous
    /// `cuMemAlloc` and there is no pool to tune.
    pub(crate) fn configure(ordinal: usize) -> Result<Self, Error> {
        // Device handle for the ordinal (does not require a current context).
        let mut dev: sys::CUdevice = 0;
        // SAFETY: `dev` is a valid out-param; ordinal is the same one the context
        // was created with. Status mapped through `DriverError`.
        unsafe { sys::cuDeviceGet(&mut dev, ordinal as i32) }
            .result()
            .map_err(Error::Driver)?;

        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        // SAFETY: valid out-param + live device handle.
        unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, dev) }
            .result()
            .map_err(Error::Driver)?;

        let me = Self { pool };
        me.set_release_threshold(RETAIN_ALL)?;
        Ok(me)
    }

    /// Set the pool's release threshold (bytes the pool may keep reserved before
    /// returning memory to the OS at a sync point). `u64::MAX` retains all.
    pub(crate) fn set_release_threshold(&self, bytes: u64) -> Result<(), Error> {
        let mut value = bytes;
        // SAFETY: `value` is a live `u64` for the call; the attribute expects a
        // `cuuint64_t*`. Pool handle is valid for the device's lifetime.
        unsafe {
            sys::cuMemPoolSetAttribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut value as *mut u64 as *mut c_void,
            )
        }
        .result()
        .map_err(Error::Driver)
    }

    /// Read back a `u64`-valued pool attribute.
    fn get_u64_attr(&self, attr: sys::CUmemPool_attribute) -> Result<u64, Error> {
        let mut value: u64 = 0;
        // SAFETY: `value` is a live out `u64`; these attributes are `cuuint64_t`.
        unsafe {
            sys::cuMemPoolGetAttribute(self.pool, attr, &mut value as *mut u64 as *mut c_void)
        }
        .result()
        .map_err(Error::Driver)?;
        Ok(value)
    }

    /// Bytes currently reserved from the OS by the pool (allocated + cached-free).
    /// Stays flat in steady state once the pool is warm — the signal that frees
    /// are being reused rather than returned to the OS.
    pub(crate) fn reserved_bytes(&self) -> Result<u64, Error> {
        self.get_u64_attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
    }

    /// Bytes currently handed out (live allocations). Returns to ~0 once all
    /// states are dropped and the stream is synchronized — the leak signal.
    pub(crate) fn used_bytes(&self) -> Result<u64, Error> {
        self.get_u64_attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
    }

    /// Return cached-free memory above `min_keep` bytes to the OS. Frees must
    /// have completed (synchronize the stream first) for them to be trimmable.
    pub(crate) fn trim(&self, min_keep: usize) -> Result<(), Error> {
        // SAFETY: valid pool handle.
        unsafe { sys::cuMemPoolTrimTo(self.pool, min_keep) }
            .result()
            .map_err(Error::Driver)
    }
}
