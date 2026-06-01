//! `AlignedBuf<T>` — a fixed-size, cache-line-aligned, owned heap buffer.
//!
//! State vectors never resize, so a growable `Vec` is unnecessary; what we
//! want is a *guaranteed* 64-byte (cache-line) base alignment so that the
//! AoS SIMD units (`LANES = 4` complex = 64 bytes, in `aleph-sv`) sit on the
//! cache-line grid and parallel tasks never share a boundary line (P2-02).
//! It is also the allocation hook P2-03 (NUMA first-touch) will extend.
//!
//! Intended for `Copy`/POD element types (`f64`, `aleph_core::Complex`): the
//! buffer does NOT run element destructors on drop (it only frees the block).
//! `zeroed` relies on the all-zero bit pattern being a valid `T`, which holds
//! for `f64` and `Complex` (`#[repr(C)] { re: f64, im: f64 }`).

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::{fmt, mem, slice};
use std::alloc::{self, Layout};

/// Cache-line size on all targets we support (x86-64, aarch64). 64 bytes =
/// exactly `LANES = 4` complex amplitudes, the AoS SIMD unit width.
pub const CACHE_LINE: usize = 64;

/// Page granularity for first-touch chunking. Keeping chunk boundaries on a
/// page multiple stops a single page from being faulted by two workers (which
/// would split its amplitudes across two NUMA nodes). 4 KiB is the base page
/// on x86-64 and aarch64; this is a *granularity*, not an exactness claim —
/// under transparent huge pages a boundary may land inside a larger page, a
/// negligible placement nit that is never incorrect.
#[cfg(feature = "numa")]
// Used by `first_touch_chunk_len` and the parallel-init pass added in later P2-03 tasks.
#[allow(dead_code)]
const FIRST_TOUCH_PAGE: usize = 4096;

/// Contiguous chunk length, in elements, for the first-touch parallel init.
///
/// Splits `[0, len)` into roughly `n_threads` contiguous chunks so each rayon
/// worker faults one chunk's pages, then rounds the chunk up to a whole number
/// of [`FIRST_TOUCH_PAGE`] pages so no page is split between two workers. Pure
/// function of its inputs (no rayon / globals) so the partition is unit-testable
/// without NUMA hardware. Returns at least one page of elements for `len > 0`.
#[cfg(feature = "numa")]
// Called by the parallel-init pass added in later P2-03 tasks and by the test below.
#[allow(dead_code)]
fn first_touch_chunk_len(len: usize, n_threads: usize, elem_size: usize) -> usize {
    debug_assert!(len > 0 && n_threads > 0 && elem_size > 0);
    // Elements per page (≥ 1; an element larger than a page degenerates to
    // per-element granularity, which is the finest useful split).
    let per_page = (FIRST_TOUCH_PAGE / elem_size).max(1);
    let target = len.div_ceil(n_threads); // ~equal contiguous chunks
    let chunk = target.div_ceil(per_page) * per_page; // round up to whole pages
    chunk.max(per_page)
}

/// A fixed-size, `CACHE_LINE`-aligned, owned heap buffer of `T`.
///
/// See the module docs for the element-type contract (POD/`Copy`, no
/// destructors run). Construct with [`AlignedBuf::zeroed`] or
/// [`AlignedBuf::from_slice`]; access through `Deref`/`DerefMut` to `[T]`.
pub struct AlignedBuf<T> {
    /// Non-null and aligned-for-`T`. For `len == 0` this is `NonNull::dangling()`
    /// (a provenance-safe, never-dereferenced sentinel); for `len > 0` it is the
    /// 64-aligned base of an owned heap block.
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    /// Layout for `len` elements at 64-byte alignment.
    ///
    /// `len` is bounded by the caller (`dim = 2^n`, `n ≤ 28`), so
    /// `len * size_of::<T>() ≤ 2^32` never approaches `isize::MAX` on a
    /// 64-bit target and the constructor cannot fail in-domain. An
    /// out-of-domain `len` is treated as an unsatisfiable allocation.
    fn layout(len: usize) -> Layout {
        let size = len.saturating_mul(mem::size_of::<T>());
        match Layout::from_size_align(size, CACHE_LINE) {
            Ok(l) => l,
            // Unreachable for in-domain `len` (`dim ≤ 2^28`, so the size never
            // approaches `isize::MAX`): `from_size_align` only errors on
            // overflow/oversize, which cannot happen here. The dummy
            // `Layout::new::<u8>()` is never actually allocated — it only feeds
            // the diverging OOM handler so the `Err` arm has a return type.
            Err(_) => alloc::handle_alloc_error(Layout::new::<u8>()),
        }
    }

    /// Provenance-safe, never-dereferenced sentinel for `len == 0`.
    fn empty() -> Self {
        // `NonNull::dangling()` is non-null and aligned for `T` (its address is
        // `align_of::<T>()`). It is the canonical zero-length base: it is only
        // ever handed to `slice::from_raw_parts(_, 0)`, which is valid for a
        // non-null, aligned pointer at length 0. Note it is NOT 64-aligned, but
        // that is fine — a zero-length buffer owns no data and is never read.
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Allocate `len` elements, zero-initialised.
    ///
    /// The all-zero bit pattern must be a valid `T` (holds for `f64` /
    /// `Complex`).
    pub fn zeroed(len: usize) -> Self {
        const {
            assert!(
                mem::size_of::<T>() != 0,
                "AlignedBuf<T> requires a non-ZST T"
            )
        };
        const {
            assert!(
                !mem::needs_drop::<T>(),
                "AlignedBuf<T> does not run element destructors; T must not need Drop"
            )
        };
        if len == 0 {
            return Self::empty();
        }
        let layout = Self::layout(len);
        // SAFETY: `layout` has non-zero size (`len > 0`, `size_of::<T>() > 0`).
        let raw = unsafe { alloc::alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Allocate and copy the contents of `src`.
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Copy,
    {
        const {
            assert!(
                mem::size_of::<T>() != 0,
                "AlignedBuf<T> requires a non-ZST T"
            )
        };
        const {
            assert!(
                !mem::needs_drop::<T>(),
                "AlignedBuf<T> does not run element destructors; T must not need Drop"
            )
        };
        let len = src.len();
        if len == 0 {
            return Self::empty();
        }
        let layout = Self::layout(len);
        // SAFETY: non-zero size; we initialise all `len` slots immediately
        // below via `copy_nonoverlapping` before any read.
        let raw = unsafe { alloc::alloc(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        // SAFETY: `src` holds `len` `T`; `ptr` owns `len` aligned, allocated
        // (uninitialised) slots; the regions do not overlap (fresh alloc).
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), ptr.as_ptr(), len);
        }
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }
}

impl<T> Deref for AlignedBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` is non-null and points to `len` initialised `T`
        // (zeroed or copied) at 64-byte alignment when `len > 0`; for
        // `len == 0` the dangling sentinel (aligned for `T`, never read) is a
        // valid zero-length base.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: same invariants as `deref`; `&mut self` gives unique access.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return; // `empty()` never allocated.
        }
        let layout = Self::layout(self.len);
        // SAFETY: `ptr` came from `alloc`/`alloc_zeroed` with exactly this
        // layout; we free the block once. Element destructors are
        // intentionally not run — the POD element contract (see module docs)
        // is machine-enforced by the `!needs_drop::<T>()` const assert in the
        // constructors, so `T` provably has no destructor to run.
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T: Copy> Clone for AlignedBuf<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedBuf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

// SAFETY: `AlignedBuf<T>` owns a unique heap region of `T` with no interior
// mutability or shared ownership — identical sharing semantics to `Vec<T>`,
// so it is `Send`/`Sync` under the same bounds.
unsafe impl<T: Send> Send for AlignedBuf<T> {}
unsafe impl<T: Sync> Sync for AlignedBuf<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_is_cache_line_aligned() {
        // Only the allocating cases carry the 64-alignment guarantee; the
        // `len == 0` sentinel (`NonNull::dangling()`) is intentionally not
        // 64-aligned and is covered by `empty_buf_is_zero_len`.
        for n in [1usize, 4, 1000] {
            let buf = AlignedBuf::<f64>::zeroed(n);
            assert_eq!(
                buf.as_ptr() as usize % CACHE_LINE,
                0,
                "len {n} not 64-aligned"
            );
            assert_eq!(buf.len(), n);
        }
    }

    #[test]
    fn empty_buf_is_zero_len() {
        // The `len == 0` sentinel owns no data: it is zero-length and empty,
        // but makes no 64-alignment promise (it is `NonNull::dangling()`).
        let buf = AlignedBuf::<f64>::zeroed(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn zeroed_contents_are_zero() {
        let buf = AlignedBuf::<f64>::zeroed(64);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn from_slice_round_trips() {
        let src = [1.0_f64, -2.0, 3.5, 4.0, 5.0];
        let buf = AlignedBuf::from_slice(&src);
        assert_eq!(&*buf, &src);
        assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[test]
    fn from_empty_slice_is_zero_len() {
        let buf = AlignedBuf::<f64>::from_slice(&[]);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn deref_mut_writes_through() {
        let mut buf = AlignedBuf::<f64>::zeroed(4);
        buf[2] = 9.0;
        assert_eq!(buf[2], 9.0);
        assert_eq!(buf[0], 0.0);
    }

    #[test]
    fn clone_is_independent_copy() {
        let mut a = AlignedBuf::<f64>::from_slice(&[1.0, 2.0, 3.0]);
        let b = a.clone();
        assert_ne!(a.as_ptr(), b.as_ptr(), "clone must allocate a fresh block");
        a[0] = 99.0;
        assert_eq!(&*b, &[1.0, 2.0, 3.0]);
        assert_eq!(b.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[test]
    fn complex_zeroed_and_round_trip() {
        // Exercise the production element type end-to-end.
        let zeros = AlignedBuf::<crate::Complex>::zeroed(8);
        assert_eq!(zeros.as_ptr() as usize % CACHE_LINE, 0);
        assert_eq!(zeros.len(), 8);
        assert!(zeros.iter().all(|&z| z == crate::Complex::new(0.0, 0.0)));

        let src = [
            crate::Complex::new(1.0, -2.0),
            crate::Complex::new(3.0, 4.0),
        ];
        let buf = AlignedBuf::<crate::Complex>::from_slice(&src);
        assert_eq!(&*buf, &src);
        assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[cfg(feature = "numa")]
    #[test]
    fn first_touch_partition_covers_range_once() {
        // (len, n_threads, elem_size) — Complex is 16 B, f64 is 8 B.
        let cases = [
            (1usize, 1usize, 16usize),
            (1, 8, 16),
            (1000, 4, 16),
            (1 << 20, 8, 16),        // 1M Complex, 8 threads
            (1 << 20, 20, 8),        // 1M f64, 20 threads (Xeon core count)
            ((1 << 20) + 7, 6, 16),  // non-page-multiple len
            (300, 16, 8),            // many threads, tiny len
        ];
        for (len, nt, es) in cases {
            let chunk = first_touch_chunk_len(len, nt, es);
            assert!(chunk > 0, "chunk must be positive: len={len} nt={nt} es={es}");
            let per_page = (FIRST_TOUCH_PAGE / es).max(1);
            assert_eq!(chunk % per_page, 0, "chunk {chunk} not page-aligned (es={es})");

            // Reconstruct the chunks: contiguous, disjoint, exact cover of [0,len).
            let n_chunks = len.div_ceil(chunk);
            let mut prev_end = 0usize;
            let mut covered = 0usize;
            for c in 0..n_chunks {
                let start = c * chunk;
                let end = (start + chunk).min(len);
                assert_eq!(start, prev_end, "gap/overlap before chunk {c}");
                assert!(start < end, "empty chunk {c}");
                covered += end - start;
                prev_end = end;
            }
            assert_eq!(prev_end, len, "partition did not reach len={len}");
            assert_eq!(covered, len, "coverage {covered} != len {len}");
        }
    }
}
