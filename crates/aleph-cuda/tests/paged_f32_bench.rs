//! P5.11-01: out-of-core **FP32** paging throughput + n=32 reach.
//!
//! The FP32 paged path halves the host footprint of the FP64 one
//! ([`paged_bench`]): `2^32 · 8 B = 32 GiB` of pinned host instead of 64 GiB, so
//! n=32 fits this box's 62 GiB host RAM where FP64 cannot. Two measurements on
//! the RTX 4000 SFF Ada (20 GiB card):
//!
//! 1. **Throughput hit vs FP32 in-core**, at an n that fits the card both ways:
//!    the streaming-overhead ratio.
//! 2. **Reach**: n=32 — 32 GiB of FP32, larger than the card — runs **only** via
//!    paging. Reports wall time, per-gate cost, effective host↔device bandwidth,
//!    and a norm≈1 sanity check. This is the new single-GPU reach record.
//!
//! `#[ignore]` (run explicitly); gated on `cfg(all(target_os = "linux", feature
//! = "cuda"))`; skips cleanly with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_cuda::CudaSvBackendF32;
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

/// One layer of H on every qubit + a CNOT brick — `~2n` gates, a mix of 1q and
/// 2q (low/high) fan-outs, for the equal-n in-core-vs-paged comparison.
fn one_layer(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for q in (0..n - 1).step_by(2) {
        c.cnot(q, q + 1).unwrap();
    }
    for q in (1..n - 1).step_by(2) {
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

/// Equal-n: FP32 in-core vs FP32 paged, reporting the streaming overhead ratio.
#[test]
#[ignore]
fn paged_f32_vs_in_core_throughput() {
    let n = env_u32("ALEPH_PAGED_CMP_N", 28);
    let reps = env_u32("ALEPH_PAGED_REPS", 3);
    let mut gpu = match CudaSvBackendF32::new().map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged-f32 throughput: {e}");
            return;
        }
    };
    let circ = one_layer(n);
    let gates = circ.instructions().len();

    // In-core baseline (`run` synchronises the stream before returning).
    let mut best_core = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = gpu.run(&circ).expect("in-core run");
        std::hint::black_box(st.num_qubits());
        best_core = best_core.min(t.elapsed().as_secs_f64());
    }

    println!("== P5.11-01 paged-f32 vs in-core, n={n}, {gates} gates, best of {reps} ==");
    println!("in-core: {best_core:.4}s");
    for m in [n - 3, n - 5] {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let st = gpu.run_paged(&circ, m).expect("paged run");
            std::hint::black_box(st.num_qubits());
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "paged m={m} (h={}): {best:.4}s  ({:.2}× in-core)",
            n - m,
            best / best_core
        );
    }
}

/// Reach: n=32 runs out-of-core in FP32 (32 GiB pinned host, impossible in-core
/// on a 20 GiB card and impossible for the FP64 paged path on a 62 GiB box).
#[test]
#[ignore]
fn paged_f32_reach_n32() {
    let n = env_u32("ALEPH_PAGED_N", 32);
    let m = env_u32("ALEPH_PAGED_TILE", 27);
    let mut gpu = match CudaSvBackendF32::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged-f32 reach: {e}");
            return;
        }
    };
    let circ = ghz(n);
    let gates = circ.instructions().len();

    println!(
        "== P5.11-01 reach: n={n} ({} GiB FP32), tile m={m} ==",
        (1u64 << n) >> 27
    );
    let t = Instant::now();
    let st = gpu.run_paged(&circ, m).expect("paged n=32 run");
    let secs = t.elapsed().as_secs_f64();

    // Each gate streams the whole state to the device and back: 2·(2·2^n·4 B).
    let bytes_per_gate = 2.0 * (2.0 * (1u64 << n) as f64 * 4.0);
    let gbps = (gates as f64 * bytes_per_gate) / secs / 1e9;
    let norm = st.norm_sqr();
    println!(
        "total {secs:.2}s, {gates} gates, {:.3}s/gate",
        secs / gates as f64
    );
    println!("effective host↔device throughput: {gbps:.1} GB/s");
    println!("norm = {norm:.6} (want ≈ 1)");
    assert!(st.num_qubits() == n);
    // FP32 norm drifts more than FP64; loosen the sanity bound accordingly.
    assert!((norm - 1.0).abs() < 1e-3, "norm drifted: {norm}");
}
