//! P5.11-01: out-of-core (host-memory paged) **FP32** SV correctness.
//!
//! `CudaSvBackendF32::run_paged` holds the state in pinned host memory as `f32`
//! and streams `2^m` FP32 tiles through the GPU, applying each gate by gathering
//! its co-resident tiles, remapping its qubits device-local, and reusing the
//! in-core FP32 kernels. This file forces paging on at **small n with a small
//! tile** (so the high/low split and cross-tile gather are genuinely exercised —
//! `m < n` ⇒ `h ≥ 1` high bits) and pins the paged state both:
//!
//! 1. **bit-for-bit against the FP32 in-core backend** (`CudaSvBackendF32::run`) —
//!    same scalar type, so the only difference is data movement, not arithmetic;
//!    tolerance 1e-5 absorbs the launch-order FP32 reassociation, and
//! 2. **against the exact FP64 CPU `NaiveSvBackend` within 1e-5** (the FP32
//!    accuracy ceiling).
//!
//! Across several tile sizes and gate fan-outs (1q, CNOT, controlled phase,
//! generic 2q — high-qubit counts hh = 0, 1, 2).
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackendF32;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// GHZ: H on q0 then a CNOT ladder — CNOTs span low↔high and high↔high tile
/// boundaries as the ladder climbs past `m`.
fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

/// Textbook QFT (H + controlled Phase). The controlled phases pair distant
/// qubits, so many are high↔high (hh=2) — the worst case for the tile gather.
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

/// Random brickwall: per-layer random 1q rotations + a CNOT brick. Dense,
/// entangling, and hits every tile-boundary combination.
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

/// Paged FP32 must match (a) the FP32 in-core backend and (b) the FP64 CPU
/// oracle, both within the 1e-5 FP32 tolerance.
fn assert_paged_f32_matches(name: &str, circ: &Circuit, m: u32) {
    let mut gpu = match CudaSvBackendF32::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged-f32 oracle ({name}, m={m}): {e}");
            return;
        }
    };

    // (b) FP64 CPU reference.
    let mut cpu = NaiveSvBackend::with_seed(0);
    let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu in-core"));

    // (a) FP32 in-core reference (same scalar type as the paged path).
    let in_core = gpu.run(circ).expect("fp32 in-core").amplitudes_vec();

    let got = gpu.run_paged(circ, m).expect("fp32 paged").amplitudes_vec();
    assert_eq!(got.len(), want.len(), "{name} m={m}: len");
    assert_eq!(got.len(), in_core.len(), "{name} m={m}: len vs in-core");

    for (i, ((a, c64), f32_ref)) in got.iter().zip(want.iter()).zip(in_core.iter()).enumerate() {
        let d_cpu = (a - c64).norm();
        let d_f32 = (a - f32_ref).norm();
        assert!(d_cpu <= 1e-5, "{name} m={m} i={i}: |Δ_cpu|={d_cpu:.2e}");
        assert!(d_f32 <= 1e-5, "{name} m={m} i={i}: |Δ_f32|={d_f32:.2e}");
    }
}

/// Every (circuit, tile size) pair must reproduce both references. The tile
/// sizes span `h = n - m ∈ {1, 2, 3}` high bits so single-tile groups (hh=0) and
/// multi-tile groups (hh=1,2) are all hit.
#[test]
fn paged_f32_matches_references() {
    let n = 8;
    let mut rng = StdRng::seed_from_u64(0x511012);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("random(d12)", random_brickwall(&mut rng, n, 12)),
    ];
    // m=5 → h=3 ; m=6 → h=2 ; m=7 → h=1. Each forces a different split.
    for (name, circ) in &workloads {
        for m in [5u32, 6, 7] {
            assert_paged_f32_matches(name, circ, m);
        }
    }
}

/// Smallest non-trivial split: n=4, m=2 (h=2 high bits, tile = 4 amplitudes).
/// A CNOT between the two high qubits (q2,q3) is the hh=2 cross-tile worst case.
#[test]
fn paged_f32_tiny_tiles_high_high_cnot() {
    let mut c = Circuit::new(4, 0);
    c.h(2).unwrap();
    c.cnot(2, 3).unwrap(); // both high (m=2): hh=2
    c.h(0).unwrap();
    c.cnot(0, 3).unwrap(); // low control, high target: hh=1
    assert_paged_f32_matches("tiny_hh2", &c, 2);
}

/// Norm preservation at a slightly larger n via FP32 paging (sanity for the
/// large-`n` `norm_sqr` path used by the reach bench). FP32 norm drifts more than
/// FP64, so the tolerance is 1e-4.
#[test]
fn paged_f32_preserves_norm() {
    let mut gpu = match CudaSvBackendF32::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged-f32 norm: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x511012a);
    let circ = random_brickwall(&mut rng, 12, 16);
    let st = gpu.run_paged(&circ, 8).expect("paged");
    let norm = st.norm_sqr();
    assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
}
