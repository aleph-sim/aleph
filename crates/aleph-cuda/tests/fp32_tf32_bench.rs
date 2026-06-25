//! P5.11-05 A/B: TF32 tensor-core fused block vs the FP32 ALU warp-tiled kernel.
//!
//! P5.10-01 showed the wall past k=3 fusion is the O(4^k) dense matvec compute,
//! not register spill — `apply_kq_tiled_f32` removed the spill yet k=4,5 still
//! lose to the k=3 baseline. This times the SAME fused circuit on three paths at
//! each width k ∈ {4,5}:
//! `tiled` = `apply_kq_tiled_f32` (FP32 ALU; `with_tf32_kq(false)`) and `tf32` =
//! `apply_kq_tf32_k{4,5}` (tensor cores; default), both compared against the **FP32
//! tiled k=3 baseline** (the production sweet spot the exit metric must beat —
//! printed as `vs tiled-k3`).
//!
//! Exit metric (P5.11-05): tf32 k=4 and/or k=5 `vs tiled-k3` > 1.00× on the dense
//! cells ⇒ raise `MAX_FUSE_QUBITS`. Correctness is pinned in `fp32_tf32_oracle`.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_cuda::{fuse_for_gpu_with, CudaSvBackendF32};
use aleph_ir::{Circuit, Instruction};
use aleph_oracle::HasAmplitudes;
use rand::{rngs::StdRng, Rng, SeedableRng};

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
        ("random(d20)", random_brickwall(rng, n, 20)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
    ]
}

fn max_block(c: &Circuit) -> u8 {
    let mut m = 0u8;
    for inst in c.instructions() {
        if let Instruction::Gate(g) = inst {
            if let aleph_core::Gate::UnitaryKq { k, .. } = &g.gate {
                m = m.max(*k);
            }
        }
    }
    m
}

fn time_run(gpu: &mut CudaSvBackendF32, circ: &Circuit, reps: u32) -> f64 {
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

#[test]
#[ignore]
fn fp32_tf32_bench() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    // ALU = warp-tiled FP32 (tf32 off); TF32 = tensor-core kernels (default on).
    let mut alu = match CudaSvBackendF32::with_seed(0)
        .map(|b| b.with_qubit_cap(n).with_tf32_kq(false).with_tiled_min_k(2))
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tf32 bench: {e}");
            return;
        }
    };
    let mut tf32 = match CudaSvBackendF32::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tf32 bench: {e}");
            return;
        }
    };

    let mut rng = StdRng::seed_from_u64(0x51105);
    let kqs = [4usize, 5];
    println!("== P5.11-05 TF32 tensor-core vs FP32 tiled apply_kq, n={n}, best of {reps} ==");
    println!("spd = tiled(k)/tf32(k); vs tiled-k3 = tiled(k=3) / tf32(k).");
    print!("workload      tiled-k3(s)");
    for k in kqs {
        print!("   k={k}:tiled   tf32   spd  vs-tiled-k3");
    }
    println!();

    for (name, circ) in workloads(n, &mut rng) {
        // FP32 tiled k=3 is the production baseline this ticket must beat.
        let f3 = fuse_for_gpu_with(&circ, Some(3));
        let base_k3 = time_run(&mut alu, &f3, reps);
        print!("{name:<13} {base_k3:8.4}");
        for k in kqs {
            let fk = fuse_for_gpu_with(&circ, Some(k));
            let blk = max_block(&fk);
            let tiled_t = time_run(&mut alu, &fk, reps);
            let tf32_t = time_run(&mut tf32, &fk, reps);
            print!(
                "   {tiled_t:7.4} {tf32_t:7.4} {:4.2}× {:5.2}× (≤{blk})",
                tiled_t / tf32_t,
                base_k3 / tf32_t
            );
        }
        println!();
    }
    println!(
        "exit metric: tf32 k=4 and/or k=5 'vs-tiled-k3' > 1.00× on the dense cells \
         ⇒ raise MAX_FUSE_QUBITS to that width."
    );
}
