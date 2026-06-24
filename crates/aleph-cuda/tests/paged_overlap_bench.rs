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

/// nsys-friendly profile target: a small multi-gate overlap run, overlap path
/// only (no sync baseline), so the CUDA trace is clean. Env: ALEPH_PAGED_N,
/// ALEPH_PAGED_TILE, ALEPH_PAGED_NG (gate count), ALEPH_PAGED_DEPTH.
#[test]
#[ignore]
fn overlap_profile() {
    let n = env_u32("ALEPH_PAGED_N", 26);
    let m = env_u32("ALEPH_PAGED_TILE", 22);
    let ng = env_u32("ALEPH_PAGED_NG", 4);
    let depth = env_u32("ALEPH_PAGED_DEPTH", 4);
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping profile: {e}");
            return;
        }
    };
    let mut circ = Circuit::new(n, 0);
    circ.h(0).unwrap();
    for _ in 0..ng {
        circ.cnot(n - 2, n - 1).unwrap(); // both high ⇒ hh=2 ⇒ sync fallback path
    }
    let t = Instant::now();
    let st = gpu.run_paged_overlapped(&circ, m, depth).expect("overlap");
    let secs = t.elapsed().as_secs_f64();
    let bytes = ng as f64 * 2.0 * (2.0 * (1u64 << n) as f64 * 8.0);
    println!(
        "profile n={n} m={m} ng={ng} depth={depth}: {secs:.3}s ({:.1} GB/s) norm={:.4}",
        bytes / secs / 1e9,
        st.norm_sqr()
    );
}

/// Diagnostic: a SINGLE gate (one H on q0) — no gate-boundary barrier, so this
/// is the pure within-gate gather/compute/scatter pipeline. Isolates whether the
/// overlap itself works (vs the per-gate pipeline drain in the full circuit).
#[test]
#[ignore]
fn overlap_single_gate() {
    let n = env_u32("ALEPH_PAGED_N", 30);
    let m = env_u32("ALEPH_PAGED_TILE", 26);
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping single-gate: {e}");
            return;
        }
    };
    let mut circ = Circuit::new(n, 0);
    circ.h(0).unwrap();
    let bytes = 2.0 * (2.0 * (1u64 << n) as f64 * 8.0);
    let gbps = |secs: f64| bytes / secs / 1e9;

    let t = Instant::now();
    gpu.run_paged(&circ, m).expect("sync");
    let s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    gpu.run_paged_overlapped(&circ, m, 4).expect("overlap");
    let o = t.elapsed().as_secs_f64();
    println!("== P5.11-02 single-gate (n={n}, m={m}) ==");
    println!("sync    : {s:.3}s ({:.1} GB/s)", gbps(s));
    println!("overlap : {o:.3}s ({:.1} GB/s) → {:.2}×", gbps(o), s / o);
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
    let gbps = |secs: f64| (gates as f64 * bytes_per_gate) / secs / 1e9;
    println!(
        "== P5.11-02 overlap A/B: n={n} ({} GiB FP64), tile m={m}, {gates} gates ==",
        (1u64 << n) >> 26
    );
    println!(
        "sync       : {best_sync:.3}s  ({:.1} GB/s)",
        gbps(best_sync)
    );

    // Sweep pipeline depth: 2 is too shallow (gather/scatter alternate); a deeper
    // ring lets H2D run ahead of D2H. Depths are capped by device memory.
    let depths: Vec<u32> = std::env::var("ALEPH_PAGED_DEPTHS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 3, 4]);
    for depth in depths {
        let mut best_ov = f64::INFINITY;
        let mut norm = 0.0;
        let mut ok = true;
        for _ in 0..reps {
            let t = Instant::now();
            match gpu.run_paged_overlapped(&circ, m, depth) {
                Ok(st) => {
                    norm = st.norm_sqr();
                    best_ov = best_ov.min(t.elapsed().as_secs_f64());
                }
                Err(e) => {
                    println!("overlap d={depth}: skipped ({e})");
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        println!(
            "overlap d={depth} : {best_ov:.3}s  ({:.1} GB/s)  → {:.2}× speedup  (norm {norm:.6})",
            gbps(best_ov),
            best_sync / best_ov
        );
        assert!((norm - 1.0).abs() < 1e-6, "norm drifted: {norm}");
    }
}
