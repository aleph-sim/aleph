//! [`DeviceBuffer<T>`] is a typed buffer in GPU global memory, backed by a
//! `cudarc` `CudaSlice<T>`. Unlike the Metal backend's unified-memory buffers,
//! CUDA device memory is not host-visible, so transfers are explicit:
//! [`DeviceBuffer::from_slice`] uploads (host→device) and
//! [`DeviceBuffer::to_vec`] downloads (device→host).
//!
//! This is P5-01 plumbing — enough to allocate, round-trip data, and free. The
//! pooled allocator and pinned/async transfers come in P5-04 / P5-05.

use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use crate::{CudaContext, Error};

/// Process-wide count of device allocations (every `from_slice`/`zeros`). The
/// later P5-04 pool will drive the per-gate slope of this to ~0; for now it is a
/// diagnostic hook, mirroring `aleph-metal`'s `device_alloc_count`.
static DEVICE_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the process-wide device-allocation counter.
pub fn device_alloc_count() -> u64 {
    DEVICE_ALLOC_COUNT.load(Ordering::Relaxed)
}

/// Process-wide count of bytes copied **device→host** (every `to_vec`). The
/// P5-05 lazy-transfer invariant — that readout copies back only small results,
/// never the full `2^n` state — is checked against this in `tests/transfer.rs`.
static DEVICE_DTOH_BYTES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the process-wide device→host byte counter.
pub fn device_dtoh_bytes() -> u64 {
    DEVICE_DTOH_BYTES.load(Ordering::Relaxed)
}

/// A `T`-typed buffer in CUDA global memory.
///
/// `T: DeviceRepr` (cudarc's plain-old-data bound for device transfer) keeps the
/// host↔device byte copy sound. The `CudaSlice` frees the device allocation on
/// drop (RAII), so there is no explicit free.
pub struct DeviceBuffer<T: DeviceRepr> {
    slice: CudaSlice<T>,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Upload `data` (host→device) into a fresh device allocation.
    pub fn from_slice(ctx: &CudaContext, data: &[T]) -> Result<Self, Error> {
        DEVICE_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // `clone_htod` allocates a `CudaSlice` of `data.len()` and enqueues the
        // copy on the stream; for a borrowed (non-pinned) host slice cudarc
        // stream-synchronizes the source, so `data` need not outlive this call.
        let slice = ctx.stream().clone_htod(data)?;
        Ok(Self { slice })
    }

    /// Download (device→host) into a fresh `Vec<T>`, synchronizing the stream so
    /// the returned data is complete.
    pub fn to_vec(&self, ctx: &CudaContext) -> Result<Vec<T>, Error> {
        let host = ctx.stream().clone_dtoh(&self.slice)?;
        ctx.stream().synchronize()?;
        DEVICE_DTOH_BYTES.fetch_add(
            (self.slice.len() * std::mem::size_of::<T>()) as u64,
            Ordering::Relaxed,
        );
        Ok(host)
    }

    /// Number of `T` elements.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// True when the buffer holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// The underlying `CudaSlice`, for binding into future kernel launches.
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Mutable view of the underlying `CudaSlice`, for binding as a writable
    /// kernel argument (`launch_builder().arg(buf.slice_mut())`).
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    /// Overwrite the first `data.len()` elements with `data`, **reusing** the
    /// existing device allocation whenever it is large enough — only growing
    /// (reallocating) when `data` exceeds the current capacity. The hot-path
    /// escape from a per-gate `cudaMalloc` for the small reusable matrix scratch
    /// (a proper pool is P5-04). Note `len()` continues to report the allocated
    /// length, which may exceed `data.len()` after a reuse.
    pub fn write(&mut self, ctx: &CudaContext, data: &[T]) -> Result<(), Error> {
        if data.len() <= self.slice.len() {
            ctx.stream().memcpy_htod(data, &mut self.slice)?;
        } else {
            DEVICE_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            self.slice = ctx.stream().clone_htod(data)?;
        }
        Ok(())
    }
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceBuffer<T> {
    /// Allocate `len` zero-initialized elements on the device.
    pub fn zeros(ctx: &CudaContext, len: usize) -> Result<Self, Error> {
        DEVICE_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        let slice = ctx.stream().alloc_zeros(len)?;
        Ok(Self { slice })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acceptance test for P5-01: allocate GPU memory, copy 1M floats up and
    /// back, and confirm the round-trip is bit-exact. Skips cleanly when no CUDA
    /// device is present so a GPU-less host (e.g. CI) is a pass, not a failure.
    #[test]
    fn round_trip_one_million_floats() {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping CUDA round-trip test: {e}");
                return;
            }
        };

        let n = 1 << 20; // 1,048,576 floats
        let host: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 7.0).collect();

        let dev = DeviceBuffer::from_slice(&ctx, &host).expect("host->device upload");
        assert_eq!(dev.len(), n);
        assert!(!dev.is_empty());

        let back = dev.to_vec(&ctx).expect("device->host download");
        assert_eq!(host, back, "round-trip must be bit-exact");

        // Zeroed allocation works and reads back as zeros.
        let z = DeviceBuffer::<f32>::zeros(&ctx, 4096).expect("alloc_zeros");
        assert_eq!(z.len(), 4096);
        assert!(z.to_vec(&ctx).unwrap().iter().all(|&x| x == 0.0));

        // Empty buffer is a valid (degenerate) allocation.
        let empty = DeviceBuffer::<f32>::from_slice(&ctx, &[]).expect("empty upload");
        assert!(empty.is_empty());
        assert_eq!(empty.to_vec(&ctx).unwrap(), Vec::<f32>::new());
    }
}
