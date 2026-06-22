//! P5.9-02b: dense ≥3q `FuseKq` blocks on the GPU vs the P5.9-01 baseline.
//!
//! P5.9-01 fed `Fuse1qRuns`+`Fuse2q` into the GPU path (1q/2q blocks only).
//! P5.9-02a taught the kernels to apply dense `UnitaryKq` (k≤5); this bench
//! adds `FuseKq` to the GPU fusion pipeline and measures the extra speedup on
//! the dense-2q workloads (random / VQE / QAOA), sweeping `max_qubits ∈ {3,4,5}`
//! to pick the kernel's sweet spot.
//!
//! A/B knob: `fuse_for_gpu_with(c, None)` is the P5.9-01 pipeline; `Some(k)`
//! appends `FuseKq{max_qubits:k}`. Both are oracle-pinned elsewhere
//! (`gpu_fusion_bench::gpu_fused_matches_cpu_unfused`, `gpu_unitary_kq`).
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{fuse_for_gpu_with, CudaSvBackend};
use aleph_ir::{Circuit, Instruction};
use aleph_oracle::HasAmplitudes;
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

fn workloads(n: u32, rng: &mut StdRng) -> Vec<(&'static str, Circuit)> {
    vec![
        ("qft", qft(n)),
        ("random(d20)", random_brickwall(rng, n, 20)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
    ]
}

/// Largest `UnitaryKq` block in the fused circuit (0 if none), for the report.
fn max_block(c: &Circuit) -> u8 {
    let mut m = 0u8;
    for inst in c.instructions() {
        if let Instruction::Gate(g) = inst {
            if let Gate::UnitaryKq { k, .. } = &g.gate {
                m = m.max(*k);
            }
        }
    }
    m
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

/// Benchmark: P5.9-01 (Fuse1q+Fuse2q) vs P5.9-02b (… +FuseKq{max_k}) at scale.
#[test]
#[ignore]
fn gpu_fuse_kq_bench() {
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
            eprintln!("skipping FuseKq bench: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x590202b);
    let kqs = [3usize, 4, 5];
    println!("== P5.9-02b GPU FuseKq, n={n}, best of {reps} ==");
    println!("baseline = P5.9-01 (Fuse1q+Fuse2q); speedups are vs that baseline.");
    print!("workload       p5.9-01_gates  base(s)");
    for k in kqs {
        print!("   k={k}_gates k={k}(s) k={k}_spd");
    }
    println!();
    for (name, circ) in workloads(n, &mut rng) {
        let base = fuse_for_gpu_with(&circ, None);
        let base_t = time_run(&mut gpu, &base, reps);
        print!("{name:<14} {:>9}    {base_t:7.4}", base.len());
        for k in kqs {
            let fk = fuse_for_gpu_with(&circ, Some(k));
            let fk_t = time_run(&mut gpu, &fk, reps);
            print!(
                "   {:>7}(≤{}) {fk_t:6.4}  {:.2}×",
                fk.len(),
                max_block(&fk),
                base_t / fk_t
            );
        }
        println!();
    }
}
