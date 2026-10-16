//! Heap-only experiment memory probe, separate from uninstrumented Criterion timings.
//! Run one shot count per process; identical source builds on the pre-batching baseline.
use aleph_qec::{run_dem_experiment, DemError, DetectorErrorModel, NullDecoder};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counter;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: all requests are forwarded unchanged to System; counters never
// inspect/dereference allocations. Deallocation uses the original pointer/layout.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc's caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the valid layout and preserves zero initialization.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        // SAFETY: same pointer/layout passed by the allocator caller.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller guarantees a live allocation and valid nonzero new size.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        result
    }
}
fn allocated(size: usize) {
    ALLOCS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size, Relaxed);
    let current = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(current, Relaxed);
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

fn main() {
    let shots: u64 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "100000".into())
        .parse()
        .unwrap();
    let dem = DetectorErrorModel {
        detectors: 128,
        observables: 3,
        errors: (0..128)
            .map(|d| DemError::new(0.02, vec![d], vec![d % 3]))
            .collect(),
    };
    let decoder = NullDecoder::new(dem.observables);
    // Warm Rayon outside accounting. Keep thread count fixed via RAYON_NUM_THREADS.
    run_dem_experiment(&dem, 1, &decoder, 565).unwrap();
    let live = LIVE.load(Relaxed);
    let allocations = ALLOCS.load(Relaxed);
    PEAK.store(live, Relaxed);
    let result = run_dem_experiment(&dem, shots, &decoder, 565).unwrap();
    let peak = PEAK.load(Relaxed).saturating_sub(live);
    let count = ALLOCS.load(Relaxed) - allocations;
    println!("shots,logical_errors,additional_peak_heap_bytes,allocations");
    println!(
        "{},{},{},{}",
        result.shots, result.logical_errors, peak, count
    );
}
