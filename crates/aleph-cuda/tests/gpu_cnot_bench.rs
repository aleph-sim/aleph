//! P5.9-04: `apply_cnot` permutation kernel vs the dense `apply_kq` 4×4 path.
//!
//! Baseline = `with_custom_2q(false)` (CNOT through `apply_kq`: a 4×4 matvec over
//! all 2^n amplitudes). Custom = `apply_cnot` (swap the target pair where
//! control=1: zero FLOPs, touches only the control=1 half). `cnot_only` isolates
//! the effect; the dense workloads show the realistic mix.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// CNOT-only brickwall (nearest-neighbour, both parities) over an H-spread state.
fn cnot_only(n: u32, layers: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for _ in 0..layers {
        for q in (0..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in (1..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
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

fn time(gpu: &mut CudaSvBackend, circ: &Circuit, reps: u32) -> f64 {
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
fn gpu_cnot_bench() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut rng = StdRng::seed_from_u64(0x0590_4b02);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("cnot_only(d40)", cnot_only(n, 40)),
        ("random(d20)", random_brickwall(&mut rng, n, 20)),
        ("qaoa(p4)", qaoa(n, 4)),
    ];
    let mut dense =
        match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n).with_custom_2q(false)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping cnot bench: {e}");
                return;
            }
        };
    let mut custom = CudaSvBackend::with_seed(0)
        .map(|b| b.with_qubit_cap(n))
        .expect("gpu");
    println!("== P5.9-04 GPU CNOT kernel, n={n}, best of {reps} ==");
    println!("workload         dense_4x4(s)  apply_cnot(s)  speedup");
    for (name, circ) in &workloads {
        let d = time(&mut dense, circ, reps);
        let c = time(&mut custom, circ, reps);
        println!("{name:<16} {d:9.4}     {c:9.4}    {:.2}×", d / c);
    }
}
