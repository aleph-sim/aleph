//! P5.10-02: out-of-core paging throughput + reach.
//!
//! Two measurements on the RTX 4000 SFF Ada (20 GiB):
//!
//! 1. **Throughput hit vs in-core**, at an n that fits the card both ways: the
//!    same circuit run in-core (`MAX_CUDA_QUBITS` path) and paged (forced, small
//!    tile), so the ratio is the cost of streaming the state through PCIe instead
//!    of keeping it resident. Swept over a couple of tile sizes.
//! 2. **Reach**: n=31 — 32 GiB of FP64, larger than the card — runs **only** via
//!    paging. Reports wall time, per-gate cost, effective host↔device bandwidth,
//!    and a norm≈1 sanity check. (n=32 would need 64 GiB of pinned host RAM; this
//!    box has 62 GiB total, so 31 is the reachable ceiling here — the code path is
//!    n-agnostic.)
//!
//! `#[ignore]` (run explicitly); gated on `cfg(all(target_os = "linux", feature
//! = "cuda"))`; skips cleanly with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::{run, Backend};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;

/// GHZ-style: H on q0 then a CNOT ladder. `gates = n` (1 H + (n-1) CNOTs); each
/// streams the whole state once, so it is a clean per-gate throughput probe.
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

/// Equal-n: in-core vs paged, reporting the streaming overhead ratio.
#[test]
#[ignore]
fn paged_vs_in_core_throughput() {
    let n = env_u32("ALEPH_PAGED_CMP_N", 28);
    let reps = env_u32("ALEPH_PAGED_REPS", 3);
    let mut gpu = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged throughput: {e}");
            return;
        }
    };
    let circ = one_layer(n);
    let gates = circ.instructions().len();

    // In-core baseline (sample(1) forces the async stream to drain, timed in).
    let mut best_core = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = run(&mut gpu, &circ).expect("in-core run");
        let _ = gpu.sample(&st, 1).expect("sync");
        best_core = best_core.min(t.elapsed().as_secs_f64());
    }

    println!("== P5.10-02 paged vs in-core, n={n}, {gates} gates, best of {reps} ==");
    println!("in-core: {best_core:.4}s");
    // Paged at the same n, a few tile sizes (smaller m ⇒ more tiles ⇒ more copies).
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

/// Reach: n=31 runs out-of-core on the 20 GiB card (impossible in-core).
#[test]
#[ignore]
fn paged_reach_n31() {
    let n = env_u32("ALEPH_PAGED_N", 31);
    let m = env_u32("ALEPH_PAGED_TILE", 26);
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged reach: {e}");
            return;
        }
    };
    let circ = ghz(n);
    let gates = circ.instructions().len();

    println!(
        "== P5.10-02 reach: n={n} ({} GiB FP64), tile m={m} ==",
        (1u64 << n) >> 26
    );
    let t = Instant::now();
    let st = gpu.run_paged(&circ, m).expect("paged n=31 run");
    let secs = t.elapsed().as_secs_f64();

    // Each gate streams the whole state to the device and back: 2·(2·2^n·8 B).
    let bytes_per_gate = 2.0 * (2.0 * (1u64 << n) as f64 * 8.0);
    let gbps = (gates as f64 * bytes_per_gate) / secs / 1e9;
    let norm = st.norm_sqr();
    println!(
        "total {secs:.2}s, {gates} gates, {:.3}s/gate",
        secs / gates as f64
    );
    println!("effective host↔device throughput: {gbps:.1} GB/s");
    println!("norm = {norm:.6} (want ≈ 1)");
    assert!(st.num_qubits() == n);
    assert!((norm - 1.0).abs() < 1e-6, "norm drifted: {norm}");
}
