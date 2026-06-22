//! Oracle tests for the cuStateVec backend (P5-03). Acceptance criteria:
//! 1. correct vs the CPU `NaiveSvBackend` (the authoritative FP64 oracle), and
//! 2. equivalent to our own hand-written `CudaSvBackend`.
//!
//! Both legs are FP64, so the tolerance is the full-precision 1e-10.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuquantum"))`; every test
//! skips cleanly when no CUDA device / cuQuantum is present, so a host without a
//! GPU is a pass, not a failure.

#![cfg(all(target_os = "linux", feature = "cuquantum"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{CuStateVecBackend, CudaSvBackend};
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

/// Multi-controlled Z (Grover diffusion core): exercises the external
/// `controls` path of `custatevecApplyMatrix`.
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

/// Toffoli + Ccz exercise the dense 3-qubit (M8×8) path — the strongest check of
/// the MSB-first ↔ little-endian `targets` reversal. Distinct per-qubit
/// rotations make every amplitude different so an operand-order bug can't hide.
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

fn assert_amps_eq(name: &str, a: &[aleph_core::Complex], b: &[aleph_core::Complex]) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).norm();
        assert!(
            d <= TOL,
            "{name} i={i}: |Δ|={d:.3e} > {TOL:e}\n a={x} b={y}"
        );
    }
}

/// Run `circuit` on cuStateVec, the CPU SV, and our hand-written CUDA SV, then
/// assert all three agree. Returns `false` if no CUDA device (caller skips).
fn check(name: &str, circuit: &Circuit) -> bool {
    let mut cusv = match CuStateVecBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cuStateVec oracle ({name}): {e}");
            return false;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let mut gpu = CudaSvBackend::with_seed(0).expect("cuda sv backend");

    let cusv_state = run(&mut cusv, circuit).expect("cuStateVec run");
    let cpu_state = run(&mut cpu, circuit).expect("cpu run");
    let gpu_state = run(&mut gpu, circuit).expect("cuda sv run");

    let q = HasAmplitudes::amplitudes(&cusv_state);
    assert_amps_eq(
        &format!("{name} [vs cpu]"),
        &q,
        &HasAmplitudes::amplitudes(&cpu_state),
    );
    assert_amps_eq(
        &format!("{name} [vs cuda-sv]"),
        &q,
        &HasAmplitudes::amplitudes(&gpu_state),
    );
    true
}

#[test]
fn tier1_oracle_matches_cpu_and_cuda_sv() {
    let mut rng = StdRng::seed_from_u64(0x51c3);
    for n in 2..=12u32 {
        if !check(&format!("ghz n={n}"), &ghz(n)) {
            return; // no device → skip the whole suite
        }
        check(&format!("qft n={n}"), &qft(n));
        // The IR caps controls at 8, so the multi-control path runs up to n=9.
        if n <= 9 {
            check(&format!("grover n={n}"), &grover_iter(n));
        }
        check(&format!("random n={n}"), &random_brickwall(&mut rng, n, 6));
    }
}

#[test]
fn three_qubit_matches_cpu_and_cuda_sv() {
    for n in 3..=10u32 {
        if !check(&format!("3q n={n}"), &three_qubit_mix(n)) {
            return;
        }
    }
}

/// Honest cuStateVec-vs-CPU and cuStateVec-vs-our-kernels wall-clock at scale.
/// `#[ignore]` (needs a big GPU + minutes); run with
/// `cargo test -p aleph-cuda --features cuquantum --release -- --ignored
/// --nocapture perf`.
#[test]
#[ignore]
fn perf_cuquantum_vs_cpu_and_cuda_sv() {
    use std::time::Instant;

    let n: u32 = std::env::var("ALEPH_PERF_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let depth: u32 = 20;

    let mut cusv = match CuStateVecBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping perf: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(1);
    let circuit = random_brickwall(&mut rng, n, depth);

    let t = Instant::now();
    let cs = run(&mut cusv, &circuit).expect("cuStateVec run");
    let _ = HasAmplitudes::amplitudes(&cs);
    let cusv_secs = t.elapsed().as_secs_f64();

    let mut gpu = CudaSvBackend::with_seed(0).expect("cuda sv backend");
    let t = Instant::now();
    let gs = run(&mut gpu, &circuit).expect("cuda sv run");
    let _ = HasAmplitudes::amplitudes(&gs);
    let gpu_secs = t.elapsed().as_secs_f64();

    let mut cpu = NaiveSvBackend::with_seed(0);
    let t = Instant::now();
    let _ = run(&mut cpu, &circuit).expect("cpu run");
    let cpu_secs = t.elapsed().as_secs_f64();

    println!(
        "perf n={n} depth={depth}: cuStateVec {cusv_secs:.3}s  aleph-cuda-sv {gpu_secs:.3}s  CPU {cpu_secs:.3}s  | cuSV vs CPU {:.2}x  aleph-sv vs cuSV {:.2}x",
        cpu_secs / cusv_secs,
        gpu_secs / cusv_secs
    );
}
