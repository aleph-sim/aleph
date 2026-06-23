//! P5.10-02: out-of-core (host-memory paged) SV correctness.
//!
//! `run_paged` holds the state in pinned host memory and streams `2^m` tiles
//! through the GPU, applying each gate by gathering its co-resident tiles,
//! remapping its qubits device-local, and reusing the in-core kernels. This file
//! forces paging on at **small n with a small tile** (so the high/low split and
//! the cross-tile gather are genuinely exercised — `m < n` ⇒ `h ≥ 1` high bits)
//! and pins the final state **bit-for-bit against the CPU `NaiveSvBackend`** at
//! 1e-10, across several tile sizes and gate fan-outs (1q, CNOT, controlled
//! phase, generic 2q — covering high-qubit counts hh = 0, 1, 2).
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

fn assert_paged_matches_cpu(name: &str, circ: &Circuit, m: u32) {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged oracle ({name}, m={m}): {e}");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu in-core"));
    let got = gpu.run_paged(circ, m).expect("gpu paged").amplitudes_vec();
    assert_eq!(got.len(), want.len(), "{name} m={m}: len");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).norm();
        assert!(d <= 1e-10, "{name} m={m} i={i}: |Δ|={d:.2e}");
    }
}

/// Every (circuit, tile size) pair must reproduce the CPU oracle exactly. The
/// tile sizes span `h = n - m ∈ {1, 2, 3}` high bits so single-tile groups
/// (hh=0) and multi-tile groups (hh=1,2) are all hit.
#[test]
fn paged_matches_cpu_in_core() {
    let n = 8;
    let mut rng = StdRng::seed_from_u64(0x51002);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("random(d12)", random_brickwall(&mut rng, n, 12)),
    ];
    // m=5 → h=3 ; m=6 → h=2 ; m=7 → h=1. Each forces a different split.
    for (name, circ) in &workloads {
        for m in [5u32, 6, 7] {
            assert_paged_matches_cpu(name, circ, m);
        }
    }
}

/// Smallest non-trivial split: n=4, m=2 (h=2 high bits, tile = 4 amplitudes).
/// A CNOT between the two high qubits (q2,q3) is the hh=2 cross-tile worst case
/// at the tiniest scale, where any indexing error is unmissable.
#[test]
fn paged_tiny_tiles_high_high_cnot() {
    let mut c = Circuit::new(4, 0);
    c.h(2).unwrap();
    c.cnot(2, 3).unwrap(); // both high (m=2): hh=2
    c.h(0).unwrap();
    c.cnot(0, 3).unwrap(); // low control, high target: hh=1
    assert_paged_matches_cpu("tiny_hh2", &c, 2);
}

/// Norm preservation at a slightly larger n via paging (sanity for the
/// large-`n` `norm_sqr` path used by the bench).
#[test]
fn paged_preserves_norm() {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping paged norm: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x510023a);
    let circ = random_brickwall(&mut rng, 12, 16);
    let st = gpu.run_paged(&circ, 8).expect("paged");
    let norm = st.norm_sqr();
    assert!((norm - 1.0).abs() < 1e-9, "norm={norm}");
}
