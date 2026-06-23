//! P5.10-01 A/B: warp-cooperative `apply_kq_tiled` vs generic `apply_kq`.
//!
//! P5.9-02b found the generic `apply_kq` only profits from fusion up to k=3;
//! k=4,5 regress because each thread does the whole O(4^k) matvec in `v[32]`/
//! `gidx[32]` thread-local arrays that spill to local memory at k=5 (768 B).
//! The register-tiled kernel (one amplitude per warp lane, matvec = intra-warp
//! shuffle reduction, matrix in shared memory) is meant to remove that wall.
//!
//! For each dense workload (random / VQE / QAOA) and fusion width k ∈ {3,4,5}
//! this times the SAME fused circuit twice: the generic kernel
//! (`with_tiled_kq(false)`) vs the tiled kernel forced on every block
//! (`with_tiled_min_k(2)`). The headline P5.10-01 exit metric is whether the
//! tiled k=4 / k=5 runs beat the **generic k=3 baseline** (the P5.9-02b sweet
//! spot) — printed as the `vs gen-k3` column. Both kernels are oracle-pinned in
//! `gpu_tiled_kq_oracle` and `gpu_unitary_kq`.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_cuda::{fuse_for_gpu_with, CudaSvBackend};
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

/// Largest `UnitaryKq` block in the fused circuit (0 if none), for the report.
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

/// A/B: tiled vs generic `apply_kq` at k=3,4,5, plus the headline tiled-k{4,5}
/// vs generic-k3 exit comparison.
#[test]
#[ignore]
fn gpu_tiled_kq_bench() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    // Generic = `apply_kq` for every k; tiled = `apply_kq_tiled` for every k≥2.
    let mut generic =
        match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n).with_tiled_kq(false)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tiled-kq bench: {e}");
                return;
            }
        };
    let mut tiled =
        match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n).with_tiled_min_k(2)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tiled-kq bench: {e}");
                return;
            }
        };

    let mut rng = StdRng::seed_from_u64(0x51001);
    let kqs = [3usize, 4, 5];
    println!("== P5.10-01 GPU tiled-kq vs generic apply_kq, n={n}, best of {reps} ==");
    println!("spd = generic(k)/tiled(k); vs gen-k3 = generic(k=3) / tiled(k).");
    print!("workload      gen-k3(s)");
    for k in kqs {
        print!("   k={k}:gen   tiled   spd  vs-gen-k3");
    }
    println!();

    for (name, circ) in workloads(n, &mut rng) {
        // Generic k=3 is the P5.9-02b production baseline this ticket must beat.
        let f3 = fuse_for_gpu_with(&circ, Some(3));
        let base_k3 = time_run(&mut generic, &f3, reps);
        print!("{name:<13} {base_k3:8.4}");
        for k in kqs {
            let fk = fuse_for_gpu_with(&circ, Some(k));
            let blk = max_block(&fk);
            let gen_t = time_run(&mut generic, &fk, reps);
            let tiled_t = time_run(&mut tiled, &fk, reps);
            print!(
                "   {gen_t:7.4} {tiled_t:7.4} {:4.2}× {:5.2}× (≤{blk})",
                gen_t / tiled_t,
                base_k3 / tiled_t
            );
        }
        println!();
    }
    println!(
        "exit metric: tiled k=4 and/or k=5 'vs-gen-k3' > 1.00× on the dense cells \
         ⇒ raise MAX_FUSE_QUBITS to that width."
    );
}
