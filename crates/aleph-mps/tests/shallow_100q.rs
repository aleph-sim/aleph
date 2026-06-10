//! P3-10: MPS 100+ qubit shallow-circuit demo (ROADMAP §7 Phase-3 exit metric).
//!
//! Validation strategy: a depth-d nearest-neighbor circuit propagates a local
//! observable's support by at most d sites (Heisenberg light cone), so
//! ⟨ψ|O|ψ⟩ over the full n=128 chain equals the same expectation computed on
//! the ≤(|supp|+2d)-qubit backward-cone subcircuit, which runs exactly on the
//! state-vector backend. The cone extractor itself is validated against full
//! SV at n=20 below.

use aleph_backend::{run, Backend};
use aleph_core::{Gate, GateInstance, Param, Pauli, PauliString};
use aleph_ir::{Circuit, Instruction};
use aleph_mps::MpsBackend;
use aleph_sv::NaiveSvBackend;
use std::collections::{BTreeMap, BTreeSet};

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

/// Backward light-cone slice of a gate-only circuit for an observable
/// supported on `support`. Walk instructions in reverse: keep a gate iff it
/// touches the current cone, growing the cone with the kept gate's qubits.
/// Kept qubits are remapped to a compact 0..k range (BTreeSet order keeps
/// the mapping deterministic and order-preserving).
fn light_cone_subcircuit(circuit: &Circuit, support: &[u32]) -> (Circuit, BTreeMap<u32, u32>) {
    let mut cone: BTreeSet<u32> = support.iter().copied().collect();
    let mut kept: Vec<GateInstance> = Vec::new();
    for inst in circuit.instructions().iter().rev() {
        let gate = match inst {
            Instruction::Gate(gi) => gi,
            other => panic!("cone extractor handles gate-only circuits, got {other:?}"),
        };
        let touches: Vec<u32> = gate
            .qubits
            .iter()
            .chain(gate.controls.iter())
            .copied()
            .collect();
        if touches.iter().any(|q| cone.contains(q)) {
            cone.extend(touches.iter().copied());
            kept.push(gate.clone());
        }
    }
    kept.reverse();
    let map: BTreeMap<u32, u32> = cone
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, u32::try_from(new).unwrap()))
        .collect();
    let mut sub = Circuit::new(u32::try_from(cone.len()).unwrap(), 0);
    for mut gate in kept {
        for q in gate.qubits.iter_mut() {
            *q = map[q];
        }
        for q in gate.controls.iter_mut() {
            *q = map[q];
        }
        sub.add_gate(gate).unwrap();
    }
    (sub, map)
}

/// Exact ⟨terms⟩ on `circuit`'s output state, computed on the light-cone
/// subcircuit with the state-vector backend.
fn cone_expectation(circuit: &Circuit, terms: &[(u32, Pauli)]) -> f64 {
    let support: Vec<u32> = terms.iter().map(|&(q, _)| q).collect();
    let (sub, map) = light_cone_subcircuit(circuit, &support);
    let remapped: Vec<(u32, Pauli)> = terms.iter().map(|&(q, p)| (map[&q], p)).collect();
    let ps = PauliString::new(1.0, remapped).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let st = run(&mut sv, &sub).unwrap();
    sv.expectation_value(&st, &ps).unwrap()
}

/// "Who validates the validator": at n=20 the full circuit is SV-tractable,
/// so every cone-based expectation must equal the full-SV expectation to
/// 1e-12. n=20 > 14 = max cone width for depth 6, so mid-chain observables
/// genuinely exercise cone truncation (asserted below).
#[test]
fn cone_extractor_matches_full_sv() {
    let n = 20u32;
    let c = brickwork(n, 6);
    let mut sv = NaiveSvBackend::with_seed(0);
    let full = run(&mut sv, &c).unwrap();

    let mut observables: Vec<Vec<(u32, Pauli)>> = (0..n).map(|q| vec![(q, Pauli::Z)]).collect();
    for q in 0..n - 1 {
        observables.push(vec![(q, Pauli::Z), (q + 1, Pauli::Z)]);
    }

    let mut saw_truncated_cone = false;
    for terms in observables {
        let ps = PauliString::new(1.0, terms.clone()).unwrap();
        let e_full = sv.expectation_value(&full, &ps).unwrap();
        let e_cone = cone_expectation(&c, &terms);
        assert!(
            (e_full - e_cone).abs() < 1e-12,
            "cone mismatch for {terms:?}: full {e_full} vs cone {e_cone}"
        );
        let (sub, _) =
            light_cone_subcircuit(&c, &terms.iter().map(|&(q, _)| q).collect::<Vec<_>>());
        if sub.num_qubits() < n {
            saw_truncated_cone = true;
        }
    }
    assert!(
        saw_truncated_cone,
        "no observable had a cone smaller than the full circuit; test is vacuous"
    );
}

/// ROADMAP §7 Phase-3 exit metric: "MPS handles 100+ qubit shallow circuits".
/// n=128, depth 6, χ=64 (exact: Schmidt rank ≤ 2^6). Local observables are
/// validated against the exact light-cone SV reference to 1e-10.
#[test]
fn mps_128q_shallow_demo() {
    const N: u32 = 128;
    const LAYERS: u32 = 6;
    const CHI: usize = 64;
    // Generous ceiling: catches an accidental complexity regression
    // (e.g. exponential blowup), not normal variance. Debug builds run
    // faer SVDs unoptimized, so they get a wider budget.
    let ceiling_secs: u64 = if cfg!(debug_assertions) { 900 } else { 120 };

    let c = brickwork(N, LAYERS);
    let t0 = std::time::Instant::now();
    let mut be = MpsBackend::with_seed(0).with_max_bond(CHI);
    let st = run(&mut be, &c).unwrap();
    let elapsed = t0.elapsed();
    eprintln!(
        "mps_128q_shallow_demo: n={N} layers={LAYERS} chi={CHI} run took {elapsed:?} \
         (truncation_error={} max_bond_reached={})",
        st.truncation_error(),
        st.max_bond_reached()
    );

    assert!(
        elapsed < std::time::Duration::from_secs(ceiling_secs),
        "n=128 shallow run exceeded {ceiling_secs}s budget: {elapsed:?}"
    );
    assert!(
        st.truncation_error() < 1e-12,
        "expected exact run (rank ≤ 2^6 = χ), truncation_error = {}",
        st.truncation_error()
    );
    assert!(
        st.max_bond_reached() <= CHI,
        "max_bond_reached {} exceeds χ {CHI}",
        st.max_bond_reached()
    );

    // Edges + middle of the chain.
    for i in [0u32, 1, 63, 64, 127] {
        let terms = vec![(i, Pauli::Z)];
        let e_mps = be
            .expectation_value(&st, &PauliString::new(1.0, terms.clone()).unwrap())
            .unwrap();
        let e_ref = cone_expectation(&c, &terms);
        assert!(
            (e_mps - e_ref).abs() < 1e-10,
            "<Z_{i}> mismatch: mps {e_mps} vs cone-SV {e_ref}"
        );
    }
    for i in [0u32, 63, 126] {
        let terms = vec![(i, Pauli::Z), (i + 1, Pauli::Z)];
        let e_mps = be
            .expectation_value(&st, &PauliString::new(1.0, terms.clone()).unwrap())
            .unwrap();
        let e_ref = cone_expectation(&c, &terms);
        assert!(
            (e_mps - e_ref).abs() < 1e-10,
            "<Z_{i} Z_{}> mismatch: mps {e_mps} vs cone-SV {e_ref}",
            i + 1
        );
    }
}
