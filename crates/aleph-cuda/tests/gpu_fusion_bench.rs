//! P5.9-01: feed the IR gate-fusion passes into the GPU run path.
//!
//! `Fuse1qRuns` + `Fuse2q` collapse adjacent gates into `Unitary1q` / `Unitary2q`
//! blocks — which the GPU backend already applies (they carry a `GateMatrix`,
//! unlike `UnitaryKq`). Fewer gates ⇒ fewer full-state passes, the dominant cost
//! at scale. This file proves the fused GPU run is correct (vs the unfused CPU
//! oracle) and measures the speedup vs the unfused GPU run.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly with
//! no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{fuse_for_gpu, CudaSvBackend};
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn qft(n: u32) -> Circuit {
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

fn vqe(n: u32, layers: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let mut t = 0.1_f64;
    for _ in 0..layers {
        for q in 0..n {
            c.ry(t, q).unwrap();
            t += 0.017;
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in 0..n {
            c.rz(t, q).unwrap();
            t += 0.013;
        }
    }
    c
}

fn qaoa(n: u32, p: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for _ in 0..p {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = (i.min(j), i.max(j));
            c.cnot(a, b).unwrap();
            c.rz(1.4, b).unwrap();
            c.cnot(a, b).unwrap();
        }
        for q in 0..n {
            c.rx(0.8, q).unwrap();
        }
    }
    c
}

/// Fuse via the public GPU-safe helper, returning `(fused, before, after)` gate
/// counts for the report line.
fn fuse(circuit: &Circuit) -> (Circuit, usize, usize) {
    let before = circuit.len();
    let fused = fuse_for_gpu(circuit);
    let after = fused.len();
    (fused, before, after)
}

fn workloads(n: u32, rng: &mut StdRng) -> Vec<(&'static str, Circuit)> {
    vec![
        ("qft", qft(n)),
        ("random(d20)", random_brickwall(rng, n, 20)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
    ]
}

/// Correctness: the fused circuit on the GPU must equal the unfused circuit on
/// the CPU oracle (proves the GPU applies `Unitary1q`/`Unitary2q` correctly).
#[test]
fn gpu_fused_matches_cpu_unfused() {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fusion oracle: {e}");
            return;
        }
    };
    let n = 11;
    let mut rng = StdRng::seed_from_u64(0x5901);
    for (name, circ) in workloads(n, &mut rng) {
        let (fused, before, after) = fuse(&circ);
        let mut cpu = NaiveSvBackend::with_seed(0);
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, &circ).expect("cpu"));
        let got = HasAmplitudes::amplitudes(&run(&mut gpu, &fused).expect("gpu fused"));
        assert_eq!(got.len(), want.len(), "{name}: len");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let d = (a - b).norm();
            assert!(
                d <= 1e-10,
                "{name} i={i}: |Δ|={d:.2e} (fused {after} vs {before} gates)"
            );
        }
    }
}

fn time_run(gpu: &mut CudaSvBackend, circ: &Circuit, reps: u32) -> f64 {
    let _ = run(gpu, circ).expect("warmup");
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = run(gpu, circ).expect("run");
        let _ = HasAmplitudes::amplitudes(&st);
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// Benchmark: unfused vs fused GPU run at scale.
#[test]
#[ignore]
fn gpu_fusion_bench() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut gpu = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fusion bench: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x5208);
    println!("== P5.9-01 GPU fusion (Fuse1q+Fuse2q), n={n}, best of {reps} ==");
    println!("workload       gates(before→after)  unfused(s)  fused(s)  speedup");
    for (name, circ) in workloads(n, &mut rng) {
        let (fused, before, after) = fuse(&circ);
        let unf = time_run(&mut gpu, &circ, reps);
        let fus = time_run(&mut gpu, &fused, reps);
        println!(
            "{name:<14} {before:>6}→{after:<6}        {unf:8.4}   {fus:7.4}  {:.2}×",
            unf / fus
        );
    }
}
