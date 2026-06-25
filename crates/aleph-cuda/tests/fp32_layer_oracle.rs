//! P5.11-04: FP32 throughput-lever correctness.
//!
//! The FP32 backend's apply-side levers must be oracle-exact:
//! - `run_layered` (disjoint-1q-layer batching via `apply_1q_multi_f32`) must match
//!   the per-gate `run` and the CPU FP64 oracle within 1e-5, and
//! - an IR-**fused** circuit (`fuse_for_gpu` → `DiagonalPhase` for QFT/QPE phase
//!   ladders + `UnitaryKq` dense blocks) run through the [`Backend`] trait must
//!   match the CPU oracle within 1e-5 — exercising `apply_phase_poly_f32` and the
//!   warp-tiled `apply_kq_tiled_f32`.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{fuse_for_gpu, CudaSvBackendF32};
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

const TOL: f64 = 1e-5;

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

/// 1q-heavy brickwall: many disjoint single-qubit rotations per layer (the case
/// `run_layered` batches) plus a CNOT brick.
fn brickwall(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
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

fn max_dev(a: &[aleph_core::Complex], b: &[aleph_core::Complex]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[test]
fn fp32_run_layered_matches_per_gate_and_cpu() {
    let mut gpu = match CudaSvBackendF32::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 layer oracle: {e}");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let n = 10;
    let mut rng = StdRng::seed_from_u64(0x51104);
    let circ = brickwall(&mut rng, n, 12);

    let want = HasAmplitudes::amplitudes(&run(&mut cpu, &circ).expect("cpu"));
    let per_gate = gpu.run(&circ).expect("fp32 per-gate").amplitudes_vec();
    let layered = gpu
        .run_layered(&circ)
        .expect("fp32 layered")
        .amplitudes_vec();

    let d_pg = max_dev(&per_gate, &want);
    let d_l = max_dev(&layered, &want);
    let d_lp = max_dev(&layered, &per_gate);
    println!("fp32 brickwall: per-gate|Δcpu={d_pg:.2e}, layered|Δcpu={d_l:.2e}, layered|Δper-gate={d_lp:.2e}");
    assert!(d_l <= TOL, "layered vs cpu {d_l:.2e}");
    assert!(d_lp <= TOL, "layered vs per-gate {d_lp:.2e}");
}

/// IR-fused QFT (phase-poly + fused blocks) through the Backend trait must match
/// the CPU oracle — exercises `apply_phase_poly_f32` and `apply_kq_tiled_f32`.
#[test]
fn fp32_fused_matches_cpu() {
    let mut gpu = match CudaSvBackendF32::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 fused oracle: {e}");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let n = 10;
    let mut rng = StdRng::seed_from_u64(0x51104f);
    let workloads: Vec<(&str, Circuit)> =
        vec![("qft", qft(n)), ("brickwall", brickwall(&mut rng, n, 10))];

    for (name, circ) in &workloads {
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu"));
        let fused = fuse_for_gpu(circ);
        let got = HasAmplitudes::amplitudes(&run(&mut gpu, &fused).expect("fp32 fused run"));
        let d = max_dev(&got, &want);
        println!("fp32 fused {name}: max |Δ| = {d:.2e}");
        assert!(d <= TOL, "fused {name} vs cpu {d:.2e}");
    }
}
