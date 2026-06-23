//! P5.9-05 verdict harness: aleph with **all Phase-5.9 levers** vs cuStateVec,
//! on the same Tier-1 + Tier-2 suite as the P5-08 report. The Aer-GPU column
//! comes from `gpu_report_bench.py` (Aer is driven from Python).
//!
//! Two aleph configurations are timed, and the better one per workload is the
//! honest "aleph 5.9" number:
//!   - **layered-raw**  = `run_layered(circuit)` — disjoint-1q-layer batching
//!     (P5.9-03) + the CNOT permutation kernel (P5.9-04) + P5-06 diagonal kernels.
//!     CNOTs stay plain, so the permutation kernel fires.
//!   - **layered-fused** = `run_layered(fuse_for_gpu(circuit))` — additionally
//!     the IR fusion passes (P5.9-01/02): 1q/2q/≤3q dense blocks. Fuse2q absorbs
//!     CNOTs into dense blocks, so this trades the CNOT kernel for fewer passes.
//!
//! Which wins is workload-dependent; the report shows both.
//!
//! Exit metric (ROADMAP §7): aleph FP64 GPU SV within **1.5×** of Aer-GPU on
//! every cell, no Tier-1 cell worse than 2×.
//!
//! `#[ignore]`. Run:
//! ```bash
//! ALEPH_REPORT_N=28 cargo test -p aleph-cuda --features cuquantum --release \
//!   -- --ignored --nocapture gpu_report59
//! ALEPH_REPORT_N=28 /root/aervenv/bin/python tests/gpu_report_bench.py
//! ```

#![cfg(all(target_os = "linux", feature = "cuquantum"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{fuse_for_gpu, CuStateVecBackend, CudaSvBackend};
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 1..n {
        c.cnot(0, q).unwrap();
    }
    c
}

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

fn grover(n: u32, iters: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let mcz = |c: &mut Circuit| {
        let ctrls: Vec<u32> = (0..(n - 1).min(8)).collect();
        c.add_gate(GateInstance::controlled(Gate::Z, vec![n - 1], ctrls))
            .unwrap();
    };
    for _ in 0..iters {
        mcz(&mut c);
        for q in 0..n {
            c.h(q).unwrap();
            c.x(q).unwrap();
        }
        mcz(&mut c);
        for q in 0..n {
            c.x(q).unwrap();
            c.h(q).unwrap();
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

fn qpe(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let eig = n - 1;
    let m = n - 1;
    c.x(eig).unwrap();
    for q in 0..m {
        c.h(q).unwrap();
    }
    let theta = 2.0 * std::f64::consts::PI * 0.123;
    for k in 0..m {
        let ang = theta * (1u64 << k) as f64;
        c.add_gate(GateInstance::controlled(
            Gate::Phase(Param::Concrete(ang)),
            vec![eig],
            vec![k],
        ))
        .unwrap();
    }
    for j in (0..m).rev() {
        for (offset, k) in (0..j).rev().enumerate() {
            let ang = -std::f64::consts::PI / (1u64 << (offset + 1)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(ang)),
                vec![j],
                vec![k],
            ))
            .unwrap();
        }
        c.h(j).unwrap();
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
    let gamma = 0.7_f64;
    let beta = 0.4_f64;
    for _ in 0..p {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = (i.min(j), i.max(j));
            c.cnot(a, b).unwrap();
            c.rz(2.0 * gamma, b).unwrap();
            c.cnot(a, b).unwrap();
        }
        for q in 0..n {
            c.rx(2.0 * beta, q).unwrap();
        }
    }
    c
}

/// Best-of-`reps` wall-clock of `f` (which must force the final device→host sync).
fn best_of(mut f: impl FnMut(), reps: u32) -> f64 {
    f(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn time_cuq(backend: &mut CuStateVecBackend, circ: &Circuit, reps: u32) -> f64 {
    best_of(
        || {
            let st = run(backend, circ).expect("cuq run");
            let _ = HasAmplitudes::amplitudes(&st);
        },
        reps,
    )
}

#[test]
#[ignore]
fn gpu_report59() {
    let n: u32 = std::env::var("ALEPH_REPORT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let reps: u32 = std::env::var("ALEPH_REPORT_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let mut rng = StdRng::seed_from_u64(0x5208);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("grover(4it)", grover(n, 4)),
        ("random(d20)", random_brickwall(&mut rng, n, 20)),
        ("qpe", qpe(n)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
    ];

    let mut ours = match CudaSvBackend::with_seed(0).map(|b| b.with_qubit_cap(n)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_report59: {e}");
            return;
        }
    };
    let mut cuq = CuStateVecBackend::with_seed(0)
        .expect("cuStateVec")
        .with_qubit_cap(n);

    println!("== P5.9-05 verdict: aleph (all 5.9 levers) vs cuStateVec, n={n}, best of {reps} ==");
    println!("(Aer-GPU column from gpu_report_bench.py; exit gate = aleph59 ≤ 1.5× Aer)");
    println!("workload       raw(s)   fused(s)  aleph59(s)  cuStateVec(s)");
    for (name, circ) in &workloads {
        // Pre-fuse once (host-side IR pass, microseconds) — not GPU time.
        let fused = fuse_for_gpu(circ);
        let raw = best_of(
            || {
                let _ = HasAmplitudes::amplitudes(&ours.run_layered(circ).expect("raw"));
            },
            reps,
        );
        let fus = best_of(
            || {
                let _ = HasAmplitudes::amplitudes(&ours.run_layered(&fused).expect("fused"));
            },
            reps,
        );
        let best59 = raw.min(fus);
        let q = time_cuq(&mut cuq, circ, reps);
        println!("{name:<14} {raw:7.4}  {fus:7.4}   {best59:8.4}    {q:9.4}");
    }
}
