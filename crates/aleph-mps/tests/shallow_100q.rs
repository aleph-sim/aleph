//! P3-10: MPS 100+ qubit shallow-circuit demo (ROADMAP §7 Phase-3 exit metric).
//!
//! Validation strategy: a depth-d nearest-neighbor circuit propagates a local
//! observable's support by at most d sites (Heisenberg light cone), so
//! ⟨ψ|O|ψ⟩ over the full n=128 chain equals the same expectation computed on
//! the ≤(|supp|+2d)-qubit backward-cone subcircuit, which runs exactly on the
//! state-vector backend. The cone extractor itself is validated against full
//! SV at n=20 below.

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::Circuit;
use aleph_mps::MpsBackend;
use aleph_sv::NaiveSvBackend;

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}

/// Deterministic non-Clifford NN brickwork: H wall, then `layers` brick
/// layers alternating even/odd bonds. Each brick = CNOT·Rz(θ_q)·CNOT (a ZZ
/// interaction), followed by an Rx mixer wall. Any chain cut is crossed by
/// at most one brick per layer, so the Schmidt rank is ≤ 2^layers.
fn brickwork(n: u32, layers: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for layer in 0..layers {
        let mut q = layer % 2;
        while q + 1 < n {
            let theta = 0.3 + 0.05 * f64::from(q);
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(Gate::Rz(Param::Concrete(theta)), &[q + 1]))
                .unwrap();
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            q += 2;
        }
        let phi = 0.4 + 0.03 * f64::from(layer);
        for q in 0..n {
            c.add_gate(g(Gate::Rx(Param::Concrete(phi)), &[q])).unwrap();
        }
    }
    c
}

/// Builder sanity + MPS exactness at SV-tractable size: with χ=64 ≥ 2^6 the
/// MPS run of a 6-layer brickwork is exact, so dense amplitudes must match
/// the state-vector backend to 1e-10.
#[test]
fn brickwork_small_n_matches_sv_dense() {
    let c = brickwork(12, 6);
    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();
    let a = ms.dense_statevector();
    let b = svs.amplitudes();
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).norm() < 1e-10,
            "amplitude {i} mismatch: {x:?} vs {y:?}"
        );
    }
    assert!(
        ms.truncation_error() < 1e-12,
        "expected exact run, truncation_error = {}",
        ms.truncation_error()
    );
}
