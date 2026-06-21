//! Oracle tests for the CUDA FP64 state-vector backend (P5-02): every Tier-1
//! circuit must match the CPU `NaiveSvBackend` amplitude-for-amplitude. Both are
//! FP64, so the tolerance is the full-precision 1e-10 (not the FP32 1e-5 the
//! Metal backend is held to).
//!
//! The whole file is gated on `cfg(all(target_os = "linux", feature = "cuda"))`
//! and every test skips cleanly when no CUDA device is present, so a GPU-less
//! host (CI) is a pass.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

const TOL: f64 = 1e-10;

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

/// Multi-controlled Z (phase oracle / diffusion core): Z on the last qubit
/// controlled by all others — exercises the multi-control `ctrl_mask` path.
fn mcz(c: &mut Circuit, n: u32) {
    if n == 1 {
        c.z(0).unwrap();
        return;
    }
    c.add_gate(GateInstance::controlled(
        Gate::Z,
        vec![n - 1],
        (0..n - 1).collect::<Vec<_>>(),
    ))
    .unwrap();
}

fn grover_iter(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    mcz(&mut c, n);
    for q in 0..n {
        c.h(q).unwrap();
        c.x(q).unwrap();
    }
    mcz(&mut c, n);
    for q in 0..n {
        c.x(q).unwrap();
        c.h(q).unwrap();
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
        let mut q = 0;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
        let mut q = 1;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

/// Toffoli + Ccz exercise the dense 3-qubit (M8×8, kq k=3) kernel path. Uses
/// distinct per-qubit rotations (NOT a uniform `H` layer) so amplitudes are all
/// different — a uniform state would mask an operand-order bug in Toffoli (the
/// target/control swap is invisible when every amplitude is equal).
fn three_qubit_mix(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.ry(0.3 + 0.4 * q as f64, q).unwrap();
    }
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![0, 1, 2]))
        .unwrap();
    if n >= 4 {
        c.add_gate(GateInstance::new(Gate::Ccz, vec![1, 2, 3]))
            .unwrap();
    }
    c
}

/// Run `circuit` on GPU and CPU, assert amplitudes agree. Returns `false` if no
/// CUDA device (the caller then skips the whole suite).
fn check(name: &str, circuit: &Circuit) -> bool {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA oracle ({name}): {e}");
            return false;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let gpu_state = run(&mut gpu, circuit).expect("gpu run");
    let cpu_state = run(&mut cpu, circuit).expect("cpu run");
    let g = HasAmplitudes::amplitudes(&gpu_state);
    let c = HasAmplitudes::amplitudes(&cpu_state);
    assert_eq!(g.len(), c.len(), "{name}: length mismatch");
    for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
        let d = (a - b).norm();
        assert!(
            d <= TOL,
            "{name} i={i}: |Δ|={d:.3e} > {TOL:e}\n gpu={a} cpu={b}"
        );
    }
    true
}

#[test]
fn tier1_oracle_matches_cpu_sv() {
    let mut rng = StdRng::seed_from_u64(0x51c2);
    for n in 2..=12u32 {
        if !check(&format!("ghz n={n}"), &ghz(n)) {
            return; // no device → skip the whole suite
        }
        check(&format!("qft n={n}"), &qft(n));
        // Grover's diffusion uses an (n-1)-controlled Z; the IR caps controls
        // at 8, so the multi-control oracle path is exercised up to n=9.
        if n <= 9 {
            check(&format!("grover n={n}"), &grover_iter(n));
        }
        check(&format!("random n={n}"), &random_brickwall(&mut rng, n, 6));
    }
}

#[test]
fn three_qubit_kernel_matches_cpu_sv() {
    for n in 3..=10u32 {
        if !check(&format!("3q n={n}"), &three_qubit_mix(n)) {
            return;
        }
    }
}

/// Honest GPU-vs-CPU wall-clock at scale. `#[ignore]` (needs a big GPU + minutes);
/// run with `cargo test -p aleph-cuda --features cuda --release -- --ignored
/// --nocapture perf`. Prints the speedup for the PR's benchmark numbers.
#[test]
#[ignore]
fn perf_gpu_vs_cpu() {
    use std::time::Instant;

    let n: u32 = std::env::var("ALEPH_PERF_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let depth: u32 = 20;

    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping perf: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(1);
    let circuit = random_brickwall(&mut rng, n, depth);

    // GPU: time a full run (includes the final device->host sync via amplitudes).
    let t = Instant::now();
    let gs = run(&mut gpu, &circuit).expect("gpu run");
    let _ = HasAmplitudes::amplitudes(&gs);
    let gpu_secs = t.elapsed().as_secs_f64();

    let mut cpu = NaiveSvBackend::with_seed(0);
    let t = Instant::now();
    let _ = run(&mut cpu, &circuit).expect("cpu run");
    let cpu_secs = t.elapsed().as_secs_f64();

    println!(
        "perf n={n} depth={depth}: GPU {gpu_secs:.3}s  CPU {cpu_secs:.3}s  speedup {:.2}x",
        cpu_secs / gpu_secs
    );
}
