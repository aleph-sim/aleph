//! Public API allocation counter. Counts heap allocations/bytes, not RSS or timings.
//! Baseline and candidate run this identical evaluator in separate processes.

use aleph_core::{Gate, GateInstance};
use aleph_ir::Circuit;
use aleph_sv::noise::{self, NoiseModel, ReadoutError};
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

fn fixture(n: u32, mixed: bool) -> (Circuit, NoiseModel) {
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit
            .add_gate(GateInstance::new(Gate::H, &[q][..]))
            .unwrap();
        if mixed {
            circuit
                .add_gate(GateInstance::new(Gate::Ry(0.73.into()), &[q][..]))
                .unwrap();
        }
    }
    let mut model = NoiseModel::new();
    model.add_all_qubit_quantum_error(noise::depolarizing_error(0.05, 1), &["H"]);
    if mixed {
        model.add_all_qubit_quantum_error(noise::amplitude_damping_error(0.17), &["Ry"]);
        model.add_all_qubit_quantum_error(noise::phase_damping_error(0.23), &["Ry"]);
    }
    model.add_readout_error(ReadoutError::new([[0.97, 0.03], [0.07, 0.93]]), 0);
    (circuit, model)
}
fn main() {
    println!("case,allocations,allocated_bytes,additional_peak_heap_bytes");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (circuit, model) = fixture(20, true);
    pool.install(|| {
        use noise::{plan_noisy, run_noisy_sparse, run_noisy_with_options, NoiseOptions};
        let full = plan_noisy(&circuit, &model, 8, NoiseOptions::default(), false).unwrap();
        for workers in [1, 2, 4] {
            let options = NoiseOptions {
                memory_budget_bytes: full.fixed_bytes + workers * full.per_worker_bytes,
                max_workers: Some(workers),
            };
            let plan = plan_noisy(&circuit, &model, 8, options, false).unwrap();
            assert_eq!(plan.workers, workers);
            run_noisy_with_options(&circuit, &model, 8, 557, options).unwrap();
            measure(
                &format!("dense_workers{workers}_planned{}", plan.peak_buffer_bytes),
                || run_noisy_with_options(&circuit, &model, 8, 557, options).unwrap(),
            );
        }
        let sparse = plan_noisy(&circuit, &model, 8, NoiseOptions::default(), true).unwrap();
        let options = NoiseOptions {
            memory_budget_bytes: sparse.fixed_bytes + sparse.per_worker_bytes,
            max_workers: Some(1),
        };
        assert!(plan_noisy(&circuit, &model, 8, options, false).is_err());
        run_noisy_sparse(&circuit, &model, 8, 557, options).unwrap();
        measure(
            &format!("sparse_workers1_planned{}", options.memory_budget_bytes),
            || run_noisy_sparse(&circuit, &model, 8, 557, options).unwrap(),
        );
    });
}
