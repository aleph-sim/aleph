//! P5.10-03: FP32 vs FP64 throughput + reach.
//!
//! 1. **Throughput** at an n both hold in-core: the same circuit on the FP64
//!    `CudaSvBackend` vs the FP32 `CudaSvBackendF32`. FP32 halves the bytes moved
//!    per full-state sweep, so on the bandwidth-bound GPU SV it should run ~2×.
//! 2. **Reach**: FP32 n=31 (16 GiB) runs **in-core** on the 20 GiB card — one
//!    qubit past the FP64 ceiling (n=30 = 16 GiB; n=31 FP64 = 32 GiB > card).
//!
//! `#[ignore]`; gated on `cfg(all(target_os = "linux", feature = "cuda"))`.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::{run, Backend};
use aleph_cuda::{CudaSvBackend, CudaSvBackendF32};
use aleph_ir::Circuit;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn random_brickwall(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            match rng.gen_range(0..3) {
                0 => c.rx(theta, q),
                1 => c.ry(theta, q),
                _ => c.rz(theta, q),
            }
            .unwrap();
        }
        for q in (0..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in (1..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
    }
    c
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Equal-n FP32 vs FP64 throughput (per-gate, same circuit).
#[test]
#[ignore]
fn fp32_vs_fp64_throughput() {
    let n = env_u32("ALEPH_FP32_N", 28);
    let reps = env_u32("ALEPH_FP32_REPS", 5);
    let mut rng = StdRng::seed_from_u64(0x51003b);
    let circ = random_brickwall(&mut rng, n, 20);
    let gates = circ.instructions().len();

    let mut f64b = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 throughput: {e}");
            return;
        }
    };
    let mut f32b = match CudaSvBackendF32::new().map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 throughput: {e}");
            return;
        }
    };

    let mut best64 = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = run(&mut f64b, &circ).expect("fp64 run");
        let _ = f64b.sample(&st, 1).expect("fp64 sync");
        best64 = best64.min(t.elapsed().as_secs_f64());
    }
    let mut best32 = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = f32b.run(&circ).expect("fp32 run");
        std::hint::black_box(st.num_qubits());
        best32 = best32.min(t.elapsed().as_secs_f64());
    }

    let bytes64 = (2u64 << n) * 8; // 2·2^n f64
    let bytes32 = (2u64 << n) * 4; // 2·2^n f32
    println!("== P5.10-03 FP32 vs FP64, n={n}, {gates} gates, best of {reps} ==");
    println!(
        "state bytes: fp64 {} MiB, fp32 {} MiB (0.50×)",
        bytes64 >> 20,
        bytes32 >> 20
    );
    println!(
        "fp64: {best64:.4}s   fp32: {best32:.4}s   speedup {:.2}×",
        best64 / best32
    );
}

/// Reach: FP32 n=31 runs in-core (FP64 n=31 would need 32 GiB > 20 GiB card).
#[test]
#[ignore]
fn fp32_reach_n31() {
    let n = env_u32("ALEPH_FP32_REACH_N", 31);
    let mut f32b = match CudaSvBackendF32::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 reach: {e}");
            return;
        }
    };
    let circ = ghz(n);
    let gates = circ.instructions().len();
    let gib = (2u64 << n) * 4 / (1 << 30); // fp32 state size in GiB
    println!(
        "== P5.10-03 FP32 reach: n={n} ({gib} GiB in-core, FP64 would need {} GiB) ==",
        gib * 2
    );
    let t = Instant::now();
    let st = f32b.run(&circ).expect("fp32 n=31 run");
    let secs = t.elapsed().as_secs_f64();
    let norm = st.norm_sqr();
    println!(
        "total {secs:.3}s, {gates} gates, {:.4}s/gate, norm={norm:.6}",
        secs / gates as f64
    );
    assert!(st.num_qubits() == n);
    assert!((norm - 1.0).abs() < 1e-3, "fp32 norm drifted: {norm}");
}
