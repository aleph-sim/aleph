//! P5.11-02: overlapped vs synchronous paging — A/B throughput.
//!
//! Both paths stream the whole `2^n` state through PCIe once per gate; the
//! overlapped path ([`CudaSvBackend::run_paged_overlapped`]) runs gather (H2D),
//! compute, and scatter (D2H) on separate streams against a ring of device
//! buffers, so H2D overlaps D2H (PCIe is full-duplex) and compute hides behind
//! the transfers. The card's duplex ceiling is ~1.24× (microbench: 16.0 vs 12.9
//! GB/s serial), so this is a *recover-a-fraction* lever, not a 2× one.
//!
//! ## Measurement methodology (important)
//!
//! This `openwebgui.splynx.com` box has **time-varying PCIe/clock state** — the
//! same run can read 7 GB/s or 14 GB/s minutes apart. So a single-process A/B
//! (sync then overlap) is unreliable: it can catch sync in a fast window and
//! overlap in a slow one (it spuriously showed 0.65×). The valid measurement is
//! **interleaved, separate processes, on a verified-idle GPU** (`nvidia-smi`
//! util ≈ 0). This bench runs **one path per process**, selected by
//! `ALEPH_PAGED_SYNC`; drive the A/B from a shell loop, e.g.:
//!
//! ```text
//! for i in 1 2 3; do
//!   ALEPH_PAGED_SYNC=1 <bin> --ignored --nocapture paged_throughput  # sync
//!   <bin>             --ignored --nocapture paged_throughput          # overlap
//! done
//! ```
//!
//! Interleaved on the idle box, overlap is a stable **~1.10×** over sync at
//! n=28/30 (≈89% of the 1.24× duplex ceiling).
//!
//! Env: `ALEPH_PAGED_N`, `ALEPH_PAGED_TILE` (m), `ALEPH_PAGED_DEPTH` (ring),
//! `ALEPH_PAGED_SYNC` (measure the synchronous path instead of the overlapped).
//!
//! `#[ignore]` (run explicitly); gated on `cfg(all(target_os = "linux", feature
//! = "cuda"))`; skips cleanly with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;

/// GHZ: H on q0 then a CNOT ladder. `gates = n`; each streams the whole state
/// once. The top CNOTs straddle the tile boundary (hh=1,2 ⇒ shallow gates that
/// take the synchronous fallback); the rest are hh=0 (overlapped).
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

/// One path per process (see module docs): the synchronous paged path when
/// `ALEPH_PAGED_SYNC` is set, else the overlapped path. Reports wall time and
/// effective host↔device throughput for a GHZ circuit.
#[test]
#[ignore]
fn paged_throughput() {
    let n = env_u32("ALEPH_PAGED_N", 30);
    let m = env_u32("ALEPH_PAGED_TILE", 26);
    let depth = env_u32("ALEPH_PAGED_DEPTH", 4);
    let sync = std::env::var("ALEPH_PAGED_SYNC").is_ok();
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged throughput: {e}");
            return;
        }
    };
    let circ = ghz(n);
    let gates = circ.instructions().len();
    // Each gate streams the whole state to the device and back: 2·(2·2^n·8 B).
    let bytes = gates as f64 * 2.0 * (2.0 * (1u64 << n) as f64 * 8.0);

    let t = Instant::now();
    let (label, norm) = if sync {
        let st = gpu.run_paged(&circ, m).expect("sync");
        ("sync   ", st.norm_sqr())
    } else {
        let st = gpu.run_paged_overlapped(&circ, m, depth).expect("overlap");
        ("overlap", st.norm_sqr())
    };
    let secs = t.elapsed().as_secs_f64();
    println!(
        "P5.11-02 {label} n={n} m={m} gates={gates} depth={depth}: {secs:.3}s \
         ({:.1} GB/s) norm={norm:.4}",
        bytes / secs / 1e9,
    );
    assert!((norm - 1.0).abs() < 1e-6, "norm drifted: {norm}");
}
