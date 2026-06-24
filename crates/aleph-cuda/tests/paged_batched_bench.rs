//! P5.11-03: multi-gate tile-residency paging throughput.
//!
//! A/B on the RTX 4000 SFF Ada (20 GiB): the **same locality-rich circuit** run
//! out-of-core two ways at a large `n` that only fits via paging —
//!
//! 1. **per-gate** (`run_paged`, P5.10-02): every gate streams the whole `2^n`
//!    state once, and
//! 2. **batched** (`run_paged_batched`, P5.11-03): each maximal run of low-only
//!    gates is applied to a tile while it is resident, collapsing the run into one
//!    full-state PCIe pass.
//!
//! Reports the full-state pass counts ([`paged_pass_counts`]) — the acceptance
//! metric is **≥2× fewer passes** — and the measured wall-time speedup. A
//! locality-rich circuit (1q layers + nearest-neighbour 2q on low qubits, an
//! occasional high-qubit entangler) is where the lever pays.
//!
//! `#[ignore]` (run explicitly); gated on `cfg(all(target_os = "linux", feature
//! = "cuda"))`; skips cleanly with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_cuda::{paged_pass_counts, CudaSvBackend};
use aleph_ir::Circuit;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Locality-rich: `depth` layers of 1q rotations on every low qubit (`< split`) +
/// a nearest-neighbour CNOT brick within the low block, with one entangler into
/// the high block per layer. Each layer is a long low-only run (folded into one
/// pass when batched) plus a single high CNOT (per-gate either way).
fn locality_rich(rng: &mut StdRng, n: u32, split: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..split {
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            match rng.gen_range(0..3) {
                0 => c.rx(theta, q),
                1 => c.ry(theta, q),
                _ => c.rz(theta, q),
            }
            .unwrap();
        }
        for q in (0..split.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in (1..split.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        if n > split {
            c.cnot(split - 1, split).unwrap();
        }
    }
    c
}

/// Per-gate vs batched residency paging at n=30/31 on a locality-rich circuit.
#[test]
#[ignore]
fn batched_vs_per_gate_throughput() {
    let n = env_u32("ALEPH_PAGED_N", 31);
    let m = env_u32("ALEPH_PAGED_TILE", n - 5);
    let depth = env_u32("ALEPH_PAGED_DEPTH", 8);
    let reps = env_u32("ALEPH_PAGED_REPS", 2);
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping batched throughput: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x51103b);
    // The low block is the tile (`split = m`), so every layer's low run folds into
    // a single resident pass and only the cross-block CNOT stays per-gate.
    let circ = locality_rich(&mut rng, n, m, depth);
    let gates = circ.instructions().len();
    let (pg_passes, b_passes) = paged_pass_counts(&circ, m);

    println!(
        "== P5.11-03 batched vs per-gate paging, n={n}, tile m={m} (h={}) ==",
        n - m
    );
    println!("{gates} gates, depth={depth}");
    println!(
        "full-state PCIe passes: per-gate={pg_passes}, batched={b_passes}  ({:.1}× fewer)",
        pg_passes as f64 / b_passes as f64
    );

    let mut best_pg = f64::INFINITY;
    let mut best_b = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = gpu.run_paged(&circ, m).expect("per-gate paged");
        std::hint::black_box(st.num_qubits());
        best_pg = best_pg.min(t.elapsed().as_secs_f64());

        let t = Instant::now();
        let st = gpu.run_paged_batched(&circ, m).expect("batched paged");
        let norm = st.norm_sqr();
        assert!((norm - 1.0).abs() < 1e-6, "batched norm drifted: {norm}");
        best_b = best_b.min(t.elapsed().as_secs_f64());
    }

    println!("per-gate: {best_pg:.3}s");
    println!("batched : {best_b:.3}s  ({:.2}× speedup)", best_pg / best_b);
}
