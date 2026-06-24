//! P5.11-02: overlapped vs synchronous paging — A/B throughput.
//!
//! Both paths stream the whole `2^n` state through PCIe once per gate; the
//! overlapped path runs gather/compute/scatter on separate streams against two
//! ping-pong buffers, so H2D overlaps D2H (PCIe is full-duplex) and compute hides
//! behind the transfers. This cannot beat the PCIe-vs-resident bandwidth floor,
//! but should recover a meaningful fraction. Reports wall time, per-gate cost,
//! effective host↔device throughput, and the overlap speedup at n=30/31.
//!
//! `#[ignore]` (run explicitly); gated on `cfg(all(target_os = "linux", feature
//! = "cuda"))`; skips cleanly with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;

/// GHZ-style: H on q0 then a CNOT ladder. `gates = n`; each streams the whole
/// state once, so it is a clean per-gate throughput probe.
fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A/B: synchronous vs overlapped paging at n=30/31, reporting the speedup and
/// the achieved (vs the box's ~12 GB/s sync) host↔device bandwidth.
#[test]
#[ignore]
fn overlap_vs_sync_reach() {
    let n = env_u32("ALEPH_PAGED_N", 30);
    let m = env_u32("ALEPH_PAGED_TILE", 26);
    let reps = env_u32("ALEPH_PAGED_REPS", 2);
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping overlap A/B: {e}");
            return;
        }
    };
    let circ = ghz(n);
    let gates = circ.instructions().len();

    // Each gate streams the whole state to the device and back: 2·(2·2^n·8 B).
    let bytes_per_gate = 2.0 * (2.0 * (1u64 << n) as f64 * 8.0);

    let mut best_sync = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = gpu.run_paged(&circ, m).expect("paged sync");
        std::hint::black_box(st.num_qubits());
        best_sync = best_sync.min(t.elapsed().as_secs_f64());
    }
    let mut best_ov = f64::INFINITY;
    let mut norm = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        let st = gpu.run_paged_overlapped(&circ, m).expect("paged overlap");
        norm = st.norm_sqr();
        best_ov = best_ov.min(t.elapsed().as_secs_f64());
    }

    let gbps = |secs: f64| (gates as f64 * bytes_per_gate) / secs / 1e9;
    println!("== P5.11-02 overlap A/B: n={n} ({} GiB FP64), tile m={m}, {gates} gates ==", (1u64 << n) >> 26);
    println!("sync    : {best_sync:.3}s  ({:.1} GB/s)", gbps(best_sync));
    println!(
        "overlap : {best_ov:.3}s  ({:.1} GB/s)  → {:.2}× speedup",
        gbps(best_ov),
        best_sync / best_ov
    );
    println!("overlap norm = {norm:.6} (want ≈ 1)");
    assert!((norm - 1.0).abs() < 1e-6, "norm drifted: {norm}");
}
