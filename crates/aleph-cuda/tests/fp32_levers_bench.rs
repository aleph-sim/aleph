//! P5.11-04: FP32 throughput levers — fusion/phase-poly vs per-gate, stacked on
//! the 2× precision win vs FP64.
//!
//! Same circuit, four arms at an n both precisions hold in-core (all via the
//! `Backend` trait, drained with a cheap GPU-resident `probabilities` so kernel
//! completion is timed in):
//! - **A** FP32 per-gate, **B** FP32 IR-fused (`fuse_for_gpu` → phase-poly + fused
//!   blocks), **C** FP64 per-gate, **D** FP64 IR-fused.
//!
//! Reports B/A (FP32 fusion beats per-gate — the acceptance criterion), and the
//! stacked C/B and A/C / D/B ratios (FP32+levers vs FP64). `#[ignore]`; gated on
//! `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::{run, Backend};
use aleph_cuda::{fuse_for_gpu, CudaSvBackend, CudaSvBackendF32};
use aleph_ir::Circuit;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn qft(n: u32) -> Circuit {
    use aleph_core::{Gate, GateInstance, Param};
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        c.h(j).unwrap();
        for (offset, k) in ((j + 1)..n).enumerate() {
            let theta = std::f64::consts::PI / (1u64 << (offset + 1)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                vec![k],
                vec![j],
            ))
            .unwrap();
        }
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

/// Time `run(circuit)` + a cheap GPU drain (forces kernel completion), best of `reps`.
fn timed<B: Backend>(b: &mut B, circ: &Circuit, reps: u32) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = run(b, circ).expect("run");
        let _ = b.probabilities(&st, &[0]).expect("drain");
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn fp32_levers_vs_per_gate_and_fp64() {
    let n = env_u32("ALEPH_FP32_LEVERS_N", 27);
    let reps = env_u32("ALEPH_FP32_REPS", 3);
    let mut rng = StdRng::seed_from_u64(0x51104b);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("qft", qft(n)),
        ("random(d20)", random_brickwall(&mut rng, n, 20)),
    ];

    let mut f64b = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 levers bench: {e}");
            return;
        }
    };
    let mut f32b = match CudaSvBackendF32::new().map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 levers bench: {e}");
            return;
        }
    };

    println!("== P5.11-04 FP32 levers, n={n}, best of {reps} ==");
    for (name, circ) in &workloads {
        let fused = fuse_for_gpu(circ);
        let a = timed(&mut f32b, circ, reps); // FP32 per-gate
        let b = timed(&mut f32b, &fused, reps); // FP32 fused
        let c = timed(&mut f64b, circ, reps); // FP64 per-gate
        let d = timed(&mut f64b, &fused, reps); // FP64 fused
        println!(
            "-- {name} ({} gates → {} fused) --",
            circ.instructions().len(),
            fused.instructions().len()
        );
        println!(
            "  FP32 per-gate {a:.4}s | FP32 fused {b:.4}s  → fusion {:.2}×",
            a / b
        );
        println!(
            "  FP64 per-gate {c:.4}s | FP64 fused {d:.4}s  → fusion {:.2}×",
            c / d
        );
        println!(
            "  stacked: FP32-fused vs FP64-per-gate {:.2}× | vs FP64-fused {:.2}× | per-gate FP32 vs FP64 {:.2}×",
            c / b,
            d / b,
            c / a
        );
    }
}
