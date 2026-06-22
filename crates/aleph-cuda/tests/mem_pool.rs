//! Memory-pool tests (P5-04): the retaining stream-ordered pool must reuse
//! freed device blocks (so repeated allocation is a pool hit, not an OS
//! round-trip) and must not leak across many small circuits.
//!
//! The pool is the **device default** pool — global per device — so a probe
//! `CudaContext` observes the same reserved/used byte counts as the allocations
//! a `CudaSvBackend` makes on its own context.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; every test skips
//! cleanly when no CUDA device is present, so a GPU-less host (CI) is a pass.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::{run, Backend};
use aleph_cuda::{CudaContext, CudaSvBackend};
use aleph_ir::Circuit;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A probe context sharing the device default pool, or `None` to skip (no
/// device, or device without pool support).
fn probe() -> Option<CudaContext> {
    match CudaContext::new(0) {
        Ok(c) if c.pool_enabled() => Some(c),
        Ok(_) => {
            eprintln!("skipping: device has no memory-pool support");
            None
        }
        Err(e) => {
            eprintln!("skipping mem-pool test: {e}");
            None
        }
    }
}

fn small_random(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            c.ry(theta, q).unwrap();
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1).unwrap();
        }
    }
    c
}

/// Repeated allocate/free of the same-size state must reuse retained blocks,
/// so the pool's OS-reserved footprint is **bounded by a small constant
/// independent of how many cycles we run** — not linear in the cycle count
/// (which is what no pooling, or a leak, would produce).
///
/// We measure reserved bytes after a few cycles and after 10× as many: a
/// retaining pool holds them roughly equal (a couple of blocks of driver
/// pipelining headroom), whereas re-allocating from the OS each time, or
/// leaking, would scale the footprint with the cycle count.
#[test]
fn pool_reuses_freed_blocks_without_growth() {
    let Some(probe) = probe() else { return };
    let mut be = CudaSvBackend::with_seed(0).expect("backend");
    let n = 22; // 2^22 complex f64 = 64 MiB per state
    let one_state: u64 = (1u64 << n) * 16;

    let churn = |be: &mut CudaSvBackend, cycles: usize| {
        for _ in 0..cycles {
            let s = be.allocate(n).expect("allocate");
            drop(s);
            probe.synchronize().unwrap();
        }
    };

    churn(&mut be, 24);
    let reserved_small = probe.pool_reserved_bytes().unwrap();
    churn(&mut be, 240);
    let reserved_big = probe.pool_reserved_bytes().unwrap();

    assert!(reserved_small > 0, "pool should hold the warmed block");
    // 10× the churn must not mean ~10× the memory. Bounded ⇒ reuse.
    assert!(
        reserved_big <= reserved_small * 2,
        "pool reserved grew with cycle count ({reserved_small} -> {reserved_big} bytes): not reusing freed blocks"
    );
    // And the steady-state footprint is a small multiple of a single state, not
    // hundreds (which is what per-cycle OS allocation would leave reserved).
    assert!(
        reserved_big <= one_state * 8,
        "pool reserved {reserved_big} bytes ≫ a few states of {one_state}: not pooling"
    );
}

/// Stress test: run many small circuits, each allocating/running/dropping a
/// state. No leak — once everything is dropped and the stream is synchronized,
/// the pool reports ~no memory still in use.
#[test]
fn many_small_circuits_no_leak() {
    let Some(probe) = probe() else { return };
    let mut be = CudaSvBackend::with_seed(0).expect("backend");
    let mut rng = StdRng::seed_from_u64(0xA1E);

    for i in 0..2000u32 {
        let n = 6 + (i % 6); // 6..=11 qubits
        let circuit = small_random(&mut rng, n, 4);
        let state = run(&mut be, &circuit).expect("run");
        drop(state);
    }
    probe.synchronize().unwrap();

    let used = probe.pool_used_bytes().unwrap();
    // Largest state here is 2^11 complex f64 = 32 KiB; nothing should remain
    // live. Permit a small constant for any driver bookkeeping.
    assert!(
        used < (1 << 20),
        "after dropping all states, pool still reports {used} bytes in use (leak)"
    );

    // Reserved (cached-free) memory is reclaimable on demand.
    probe.trim_pool(0).unwrap();
}

/// A/B benchmark — retaining pool vs the un-tuned (release-to-OS) default.
/// `#[ignore]` (needs a GPU + a moment); run with
/// `cargo test -p aleph-cuda --features cuda --release -- --ignored --nocapture
/// pool_alloc_overhead`.
#[test]
#[ignore]
fn pool_alloc_overhead() {
    use std::time::Instant;

    let Some(probe) = probe() else { return };
    let mut be = CudaSvBackend::with_seed(0).expect("backend");
    let n: u32 = std::env::var("ALEPH_POOL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24); // 256 MiB per state
    let iters = 200;

    // Each iteration mimics a small circuit's lifecycle: allocate the state,
    // free it, synchronize (as a readout would). Threshold 0 returns the block
    // to the OS each sync; u64::MAX retains it for the next allocation.
    let bench = |be: &mut CudaSvBackend, probe: &CudaContext| -> f64 {
        // Warm once so the first measured iteration isn't a cold alloc.
        drop(be.allocate(n).unwrap());
        probe.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let s = be.allocate(n).unwrap();
            drop(s);
            probe.synchronize().unwrap();
        }
        t.elapsed().as_secs_f64() / iters as f64 * 1e6 // µs/iter
    };

    probe.set_pool_release_threshold(0).unwrap();
    probe.trim_pool(0).unwrap();
    let us_release = bench(&mut be, &probe);

    probe.set_pool_release_threshold(u64::MAX).unwrap();
    let us_retain = bench(&mut be, &probe);

    println!(
        "pool alloc+free+sync n={n} ({} MiB/state): release-to-OS {us_release:.1} µs/iter  retain {us_retain:.1} µs/iter  speedup {:.1}x",
        (1usize << n) * 16 / (1 << 20),
        us_release / us_retain
    );
}
