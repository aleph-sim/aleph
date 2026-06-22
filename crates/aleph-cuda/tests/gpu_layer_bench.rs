//! P5.9-03: disjoint-1q-layer batched dispatch vs per-gate.
//!
//! Baseline = per-gate `aleph_backend::run` (one `apply_1q` sweep per 1q gate).
//! Layered = `CudaSvBackend::run_layered`, folding each disjoint 1q sublayer
//! into `⌈count / batch⌉` `apply_1q_multi` sweeps. Sweeps batch width ∈ {2,3,4,5}
//! to find the kernel's sweet spot (the strided 2^m gather can over-cost large
//! batches, à la the P5.9-02b k-sweep). `pure1q` isolates the effect; the dense
//! workloads show the realistic mix.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Deep stack of disjoint 1q layers, no 2q gates — the maximal-win case.
fn pure1q(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            match rng.gen_range(0..2) {
                0 => c.rx(theta, q),
                _ => c.ry(theta, q),
            }
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

fn time<F: FnMut() -> Vec<aleph_core::Complex>>(mut f: F, reps: u32) -> f64 {
    let _ = f(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let _ = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn gpu_layer_bench() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut base = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping layer bench: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x0590_203b);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("pure1q(d40)", pure1q(&mut rng, n, 40)),
        ("random(d20)", random_brickwall(&mut rng, n, 20)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
    ];
    let batches = [2usize, 3, 4, 5];
    println!("== P5.9-03 GPU layer dispatch, n={n}, best of {reps} ==");
    println!("baseline = per-gate run; speedups are vs that baseline.");
    print!("workload       base(s)");
    for b in batches {
        print!("   b={b}(s) b={b}_spd");
    }
    println!();
    for (name, circ) in &workloads {
        let base_t = time(
            || HasAmplitudes::amplitudes(&run(&mut base, circ).expect("base")),
            reps,
        );
        print!("{name:<14} {base_t:7.4}");
        for b in batches {
            let mut g = CudaSvBackend::with_seed(0)
                .map(|x| x.with_qubit_cap(n).with_layer_batch(b))
                .expect("gpu");
            let t = time(
                || HasAmplitudes::amplitudes(&g.run_layered(circ).expect("layered")),
                reps,
            );
            print!("   {t:6.4}  {:.2}×", base_t / t);
        }
        println!();
    }
}
