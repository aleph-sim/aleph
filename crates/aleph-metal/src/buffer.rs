//! `DeviceBuffer<T>` is a typed, host-visible GPU buffer backed by an
//! `MTLBuffer` with `StorageModeShared`. On Apple Silicon shared storage is
//! unified memory, so the host views (`as_slice`/`as_mut_slice`) are zero-copy
//! windows onto the same bytes the GPU sees — no staging or blit needed.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;

use metal::{Buffer, BufferRef, MTLResourceOptions};

use crate::MetalContext;

/// A `T`-typed GPU buffer in shared (unified-memory) storage.
///
/// `T: bytemuck::Pod` bounds the host view to plain-old-data so the byte
/// reinterpret is sound and self-documenting.
pub struct DeviceBuffer<T: bytemuck::Pod> {
    buf: Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> DeviceBuffer<T> {
    /// Allocate a shared-storage buffer initialized from `data`.
    ///
    /// If `data` is empty, a minimal placeholder buffer is allocated and the
    /// result has `len() == 0`; host views (`as_slice`/`as_mut_slice`) return
    /// empty slices and no dereferencing of GPU memory occurs.
    pub fn from_slice(ctx: &MetalContext, data: &[T]) -> Self {
        // Metal buffer storage is page-aligned, so any `T` whose alignment is
        // at most a page is safely reinterpretable from `contents()`. This
        // rejects exotic over-aligned Pod types at compile time
        // (post-monomorphization).
        const {
            assert!(
                mem::align_of::<T>() <= 4096,
                "DeviceBuffer<T>: T alignment exceeds Metal's page-aligned guarantee (4096)"
            )
        };

        let options = MTLResourceOptions::StorageModeShared;
        let buf = if data.is_empty() {
            // Metal rejects zero-length buffers; allocate a minimal placeholder
            // and report len 0 so no host view ever dereferences it.
            ctx.device()
                .new_buffer(mem::size_of::<T>().max(1) as u64, options)
        } else {
            ctx.device().new_buffer_with_data(
                data.as_ptr() as *const c_void,
                mem::size_of_val(data) as u64,
                options,
            )
        };
        Self {
            buf,
            len: data.len(),
            _marker: PhantomData,
        }
    }

    /// Number of `T` elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the buffer holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The underlying Metal buffer, for binding into a command encoder.
    pub fn metal_buffer(&self) -> &BufferRef {
        &self.buf
    }

    /// A zero-copy host view of the buffer contents.
    ///
    /// The caller must ensure no in-flight GPU command is writing this buffer
    /// before reading the host view (e.g. after `commit()`, wait via
    /// `wait_until_completed`).
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `self.buf` uses StorageModeShared, so `contents()` maps the
        // buffer bytes into the CPU address space and is non-null for any
        // successfully allocated buffer (only Private/Memoryless storage return
        // NULL). It is valid for `len * size_of::<T>()` bytes for the buffer's
        // lifetime and is page-aligned; `from_slice`'s const-assert guarantees
        // `align_of::<T>() <= 4096`, so the `*const T` cast is aligned. `T:
        // Pod` => every bit pattern is a valid `T` with no padding/uninit. The
        // returned borrow is tied to `&self`, so it cannot outlive `self.buf`.
        unsafe { std::slice::from_raw_parts(self.buf.contents() as *const T, self.len) }
    }

    /// A zero-copy mutable host view of the buffer contents.
    ///
    /// The caller must ensure no in-flight GPU command is writing this buffer
    /// before mutating the host view (e.g. after `commit()`, wait via
    /// `wait_until_completed`).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as in `as_slice`, `contents()` is non-null (StorageModeShared),
        // valid for `len * size_of::<T>()` bytes, and page-aligned
        // (`align_of::<T>() <= 4096` by the const-assert); `T: Pod` makes every
        // bit pattern valid. Additionally, `&mut self` guarantees exclusive
        // access, so this mutable view cannot alias the shared `as_slice` view.
        unsafe { std::slice::from_raw_parts_mut(self.buf.contents() as *mut T, self.len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;

    #[test]
    fn round_trip_mutate_and_empty() {
        // Allocation needs a device; skip cleanly if none (headless CI).
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping DeviceBuffer test: no Metal device");
                return;
            }
        };

        let mut b = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(b.len(), 4);
        assert!(!b.is_empty());
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        b.as_mut_slice()[1] = 9.0;
        assert_eq!(b.as_slice()[1], 9.0);

        let _ = b.metal_buffer();

        let empty = DeviceBuffer::<f32>::from_slice(&ctx, &[]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.as_slice(), &[] as &[f32]);
    }
}
