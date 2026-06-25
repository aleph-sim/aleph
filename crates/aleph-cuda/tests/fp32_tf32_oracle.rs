//! P5.11-05: TF32 tensor-core fused-block correctness.
//!
//! `apply_kq_tf32_k{4,5}` recasts the dense k-qubit matvec as a batched GEMM on
//! the Ada TF32 tensor cores. TF32 truncates the mantissa to 10 bits, so the
//! acceptance budget is **1e-4** vs the FP32 ALU dense apply (the warp-tiled
//! `apply_kq_tiled_f32`, forced via `with_tf32_kq(false)`) — both run the SAME
//! IR-fused circuit, so any difference is purely the TF32 GEMM. A looser CPU-FP64
//! oracle check (1e-2) catches gross layout/index bugs.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_cuda::{fuse_for_gpu_with, CudaSvBackendF32};
use aleph_ir::{Circuit, Instruction};
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Acceptance budget: TF32 GEMM vs FP32 ALU dense apply (same fused circuit).
const TOL_FP32: f64 = 1e-4;
/// Looser sanity budget vs the exact FP64 CPU oracle (catches index/layout bugs).
const TOL_CPU: f64 = 1e-2;

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

/// Largest `UnitaryKq` block in the fused circuit (0 if none).
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

fn max_dev(a: &[aleph_core::Complex], b: &[aleph_core::Complex]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[test]
fn fp32_tf32_matches_fp32_dense_and_cpu() {
    let mut tf32 = match CudaSvBackendF32::with_seed(0) {
        Ok(b) => b, // tf32_kq on by default
        Err(e) => {
            eprintln!("skipping tf32 oracle: {e}");
            return;
        }
    };
    let mut alu = CudaSvBackendF32::with_seed(0)
        .expect("second fp32 backend")
        .with_tf32_kq(false); // force k=4/5 onto the warp-tiled FP32 ALU kernel
    let mut cpu = NaiveSvBackend::with_seed(0);

    let n = 12;
    let mut rng = StdRng::seed_from_u64(0x51105);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("random(d16)", random_brickwall(&mut rng, n, 16)),
        ("vqe(8L)", vqe(n, 8)),
    ];

    for (name, circ) in &workloads {
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu"));
        for k in [4usize, 5] {
            let fused = fuse_for_gpu_with(circ, Some(k));
            let blk = max_block(&fused);
            assert!(
                blk as usize >= k,
                "{name} k={k}: fusion produced no width-{k} block (max {blk}) — test vacuous"
            );
            let got_tf32 = HasAmplitudes::amplitudes(&run(&mut tf32, &fused).expect("tf32 run"));
            let got_alu = HasAmplitudes::amplitudes(&run(&mut alu, &fused).expect("alu run"));
            let d_fp32 = max_dev(&got_tf32, &got_alu);
            let d_cpu = max_dev(&got_tf32, &want);
            println!("tf32 {name} k={k} (≤{blk}): Δfp32-alu={d_fp32:.2e}, Δcpu={d_cpu:.2e}");
            assert!(
                d_fp32 <= TOL_FP32,
                "{name} k={k} tf32 vs fp32-alu {d_fp32:.2e}"
            );
            assert!(d_cpu <= TOL_CPU, "{name} k={k} tf32 vs cpu {d_cpu:.2e}");
        }
    }
}
