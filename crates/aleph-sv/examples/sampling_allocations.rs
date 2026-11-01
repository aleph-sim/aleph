//! Public API allocation counter. Counts heap allocations/bytes, not RSS or timings.
//! Baseline and candidate run this identical evaluator in separate processes.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_sv::{Fp32SvBackend, NaiveSvBackend, SoaSvBackend};
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

fn measure<T>(label: &str, call: impl FnOnce() -> T) {
    let (allocs, bytes, live) = (
        ALLOCS.load(Relaxed),
        BYTES.load(Relaxed),
        LIVE.load(Relaxed),
    );
    PEAK.store(live, Relaxed);
    let value = call();
    let row = (
        ALLOCS.load(Relaxed) - allocs,
        BYTES.load(Relaxed) - bytes,
        PEAK.load(Relaxed).saturating_sub(live),
    );
    std::hint::black_box(&value);
    drop(value);
    println!("{label},{},{},{}", row.0, row.1, row.2);
}

fn sampler<B: Backend>(name: &str, mut backend: B) {
    for n in [4, 12, 20] {
        let mut state = backend.allocate(n).unwrap();
        for q in 0..n {
            backend
                .apply_gate(&mut state, &GateInstance::new(Gate::H, &[q][..]))
                .unwrap();
        }
        for shots in [0, 1, 2, 1024] {
            measure(&format!("{name}_n{n}_shots{shots}"), || {
                backend.sample(&state, shots).unwrap()
            });
        }
    }
}

fn main() {
    println!("case,allocations,allocated_bytes,additional_peak_heap_bytes");
    sampler("aos", NaiveSvBackend::with_seed(556));
    sampler("soa", SoaSvBackend::with_seed(556));
    sampler("f32", Fp32SvBackend::with_seed(556));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for (n, shots) in [(4, 1024), (16, 64), (20, 8)] {
        let mut circuit = aleph_ir::Circuit::new(n, 0);
        for q in 0..n {
            circuit
                .add_gate(GateInstance::new(Gate::H, &[q][..]))
                .unwrap();
        }
        let mut model = aleph_sv::noise::NoiseModel::new();
        model.add_all_qubit_quantum_error(aleph_sv::noise::depolarizing_error(0.05, 1), &["H"]);
        model.add_readout_error(
            aleph_sv::noise::ReadoutError::new([[0.97, 0.03], [0.07, 0.93]]),
            0,
        );
        pool.install(|| {
            aleph_sv::noise::run_noisy(&circuit, &model, shots, 556).unwrap();
        });
        measure(&format!("noisy_pauli_readout_n{n}_shots{shots}"), || {
            pool.install(|| aleph_sv::noise::run_noisy(&circuit, &model, shots, 556).unwrap())
        });
    }
}
