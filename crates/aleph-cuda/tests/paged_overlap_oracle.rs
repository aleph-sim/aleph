//! P5.11-02: overlapped (double-buffered) paging correctness.
//!
//! `run_paged_overlapped` schedules the tile copies on dedicated H2D/D2H streams
//! against two ping-pong device buffers, with explicit CUDA events for ordering,
//! so gather/compute/scatter overlap. It must produce a state **identical** to
//! the synchronous `run_paged` (same kernels, only the scheduling differs — FP64,
//! so bit-for-bit) and equal to the CPU `NaiveSvBackend` at 1e-10. This forces
//! paging on at small n with a small tile (so `h ≥ 1` high bits and the cross-
//! tile gather + the gate-boundary host hazard are genuinely exercised), across
//! several tile sizes and gate fan-outs (1q, CNOT, controlled phase, generic 2q
//! — high-qubit counts hh = 0, 1, 2).
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
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

/// Overlapped paging must equal both the CPU oracle (1e-10) and the synchronous
/// paged path (bit-for-bit, FP64 — same kernels, only scheduling differs).
fn assert_overlap_matches(name: &str, circ: &Circuit, m: u32) {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping overlap oracle ({name}, m={m}): {e}");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu in-core"));
    let sync = gpu
        .run_paged(circ, m)
        .expect("gpu paged sync")
        .amplitudes_vec();
    let got = gpu
        .run_paged_overlapped(circ, m, 4)
        .expect("gpu paged overlap")
        .amplitudes_vec();

    assert_eq!(got.len(), want.len(), "{name} m={m}: len");
    for (i, ((a, b), s)) in got.iter().zip(want.iter()).zip(sync.iter()).enumerate() {
        let d_cpu = (a - b).norm();
        let d_sync = (a - s).norm();
        assert!(d_cpu <= 1e-10, "{name} m={m} i={i}: |Δ_cpu|={d_cpu:.2e}");
        assert!(d_sync <= 1e-12, "{name} m={m} i={i}: |Δ_sync|={d_sync:.2e}");
    }
}

/// Tile sizes span `h = n − m ∈ {1, 2, 3}` so single-tile groups (hh=0) and
/// multi-tile groups (hh=1,2) are hit, and the multi-gate pipeline (with its
/// gate-boundary host-hazard barrier) runs over many gates.
#[test]
fn overlap_matches_sync_and_cpu() {
    let n = 8;
    let mut rng = StdRng::seed_from_u64(0x511020);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("random(d12)", random_brickwall(&mut rng, n, 12)),
    ];
    for (name, circ) in &workloads {
        for m in [5u32, 6, 7] {
            assert_overlap_matches(name, circ, m);
        }
    }
}

/// Tiny split worst case: n=4, m=2, a high↔high CNOT (hh=2) across the pipeline.
#[test]
fn overlap_tiny_tiles_high_high_cnot() {
    let mut c = Circuit::new(4, 0);
    c.h(2).unwrap();
    c.cnot(2, 3).unwrap();
    c.h(0).unwrap();
    c.cnot(0, 3).unwrap();
    assert_overlap_matches("tiny_hh2", &c, 2);
}

/// Norm preservation through the overlapped pipeline at a larger n.
#[test]
fn overlap_preserves_norm() {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping overlap norm: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x511020a);
    let circ = random_brickwall(&mut rng, 12, 16);
    let st = gpu
        .run_paged_overlapped(&circ, 8, 3)
        .expect("paged overlap");
    let norm = st.norm_sqr();
    assert!((norm - 1.0).abs() < 1e-9, "norm={norm}");
}
