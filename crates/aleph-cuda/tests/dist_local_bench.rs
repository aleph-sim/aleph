//! P6-01a overhead: DistSvBackend + LocalExchange (R ranks on ONE GPU) vs the
//! single-GPU CudaSvBackend on the same circuit. On one card an exchange is a
//! device-to-device copy, so this measures the distributed machinery's cost
//! (specialise + per-rank launches + chunk swaps), not interconnect speed.
//! Run: cargo test --release -p aleph-cuda --features cuda --test dist_local_bench -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{
    fuse_for_gpu, CudaContext, CudaSvBackend, CudaSvBackendF32, DistSvBackend, LocalExchange,
};
use aleph_ir::dist::{plan, DistLayout, Router};
use aleph_ir::Circuit;

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let th = std::f64::consts::PI / 2f64.powi((j - k) as i32);
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(th)),
                vec![j],
                vec![k],
            ))
            .unwrap();
        }
    }
    for q in 0..n / 2 {
        c.swap(q, n - 1 - q).unwrap();
    }
    c
}

fn brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rx(0.3 + 0.17 * f64::from(q), q).unwrap();
            c.rz(0.7 * d as f64, q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

/// Untimed: the plan's communication for `c` at `g` (explains the overhead).
fn comm(label: &str, c: &Circuit, g: u32) {
    let p = plan(
        c,
        DistLayout::new(c.num_qubits(), g).unwrap(),
        Router::Lookahead,
    )
    .unwrap();
    let local_steps = p.steps.len() as u32 - p.stats.exchanges;
    println!(
        "comm: {label} R={} exchanges={} amps_moved_per_rank={} (of 2^{}) local_steps={local_steps}",
        1u32 << g,
        p.stats.exchanges,
        p.stats.amps_moved_per_rank,
        c.num_qubits() - g,
    );
}

/// Best wall time of `reps` runs of `f`, each closed by a sync on the device's
/// default stream. Every backend here submits to that stream (one primary
/// context), so the sync waits for all queued work in both arms alike, with no
/// host readback in the timed region.
fn best_of<T, F: FnMut() -> T>(sync: &CudaContext, reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let out = f();
        sync.synchronize().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
        drop(out);
        sync.synchronize().unwrap();
    }
    best
}

#[test]
#[ignore]
fn dist_local_overhead() {
    let Ok(sync) = CudaContext::new(0) else {
        return;
    };
    let Ok(mut single) = CudaSvBackend::with_seed(0) else {
        return;
    };
    let Ok(be) = CudaSvBackend::with_seed(0) else {
        return;
    };
    let mut d = DistSvBackend::new(be, LocalExchange::new());
    println!("| precision | circuit | n | single-GPU (s) | R=2 (s) | R=4 (s) | R=2 / single | R=4 / single |");
    println!("|---|---|---|---|---|---|---|---|");
    for (name, c) in [("QFT", qft(28)), ("random d=10", brickwall(28, 10))] {
        let fused = fuse_for_gpu(&c);
        for g in [1u32, 2] {
            comm(name, &c, g);
        }
        let t1 = best_of(&sync, 3, || run(&mut single, &fused).unwrap());
        let mut ts = Vec::new();
        for g in [1u32, 2] {
            ts.push(best_of(&sync, 3, || {
                d.run(&c, g, Router::Lookahead).unwrap()
            }));
        }
        println!(
            "| FP64 | {name} | 28 | {t1:.3} | {:.3} | {:.3} | {:.2}× | {:.2}× |",
            ts[0],
            ts[1],
            ts[0] / t1,
            ts[1] / t1
        );
    }
    let Ok(mut single32) = CudaSvBackendF32::with_seed(0) else {
        return;
    };
    let Ok(be32) = CudaSvBackendF32::with_seed(0) else {
        return;
    };
    let mut d32 = DistSvBackend::new(be32, LocalExchange::new());
    for (name, c) in [("QFT", qft(29)), ("random d=10", brickwall(29, 10))] {
        let fused = fuse_for_gpu(&c);
        for g in [1u32, 2] {
            comm(name, &c, g);
        }
        let t1 = best_of(&sync, 3, || run(&mut single32, &fused).unwrap());
        let mut ts = Vec::new();
        for g in [1u32, 2] {
            ts.push(best_of(&sync, 3, || {
                d32.run(&c, g, Router::Lookahead).unwrap()
            }));
        }
        println!(
            "| FP32 | {name} | 29 | {t1:.3} | {:.3} | {:.3} | {:.2}× | {:.2}× |",
            ts[0],
            ts[1],
            ts[0] / t1,
            ts[1] / t1
        );
    }
}
