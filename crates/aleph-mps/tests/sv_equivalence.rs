//! Oracle equivalence vs NaiveSvBackend + MPS invariants.
//! MPS dense reconstruction must match the SV amplitude vector (ADR-0004).

use aleph_backend::{run, Backend};
use aleph_core::{Complex, Gate, GateInstance, Param, Pauli, PauliString};
use aleph_mps::{MpsBackend, MpsState, TruncationPolicy};
use aleph_sv::NaiveSvBackend;

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}

fn mps_dense(circuit: &aleph_ir::Circuit, chi: usize) -> Vec<Complex> {
    let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
    let st: MpsState = run(&mut be, circuit).unwrap();
    st.dense_statevector()
}

fn sv_dense(circuit: &aleph_ir::Circuit) -> Vec<Complex> {
    let mut be = NaiveSvBackend::with_seed(0);
    let st = run(&mut be, circuit).unwrap();
    st.amplitudes().to_vec()
}

#[test]
fn bell_matches_sv() {
    let mut c = aleph_ir::Circuit::new(2, 0);
    c.add_gate(g(Gate::H, &[0])).unwrap();
    c.add_gate(g(Gate::Cnot, &[0, 1])).unwrap();
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10);
    }
}

#[test]
fn nn_chain_matches_sv() {
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    for q in 0..n {
        c.add_gate(g(Gate::Rz(Param::Concrete(0.3 + q as f64 * 0.1)), &[q]))
            .unwrap();
    }
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10);
    }
}

#[test]
fn expectation_matches_sv() {
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();
    let observables = [
        vec![(0u32, Pauli::Z), (1, Pauli::Z)],
        vec![(0, Pauli::X), (2, Pauli::X)],
        vec![(1, Pauli::Z)],
    ];
    for terms in observables {
        let p = PauliString::new(1.0, terms).unwrap();
        let em = mps.expectation_value(&ms, &p).unwrap();
        let es = sv.expectation_value(&svs, &p).unwrap();
        assert!(
            (em - es).abs() < 1e-10,
            "expectation mismatch: {em} vs {es}"
        );
    }
}

#[test]
fn probabilities_matches_sv() {
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();
    for subset in [vec![0u32], vec![0, 2], vec![1, 3, 0]] {
        let pm = mps.probabilities(&ms, &subset).unwrap();
        let ps = sv.probabilities(&svs, &subset).unwrap();
        assert_eq!(pm.len(), ps.len());
        for (x, y) in pm.iter().zip(ps.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }
}

#[test]
fn small_chi_weak_entanglement_near_exact() {
    // Shallow nearest-neighbor circuit keeps Schmidt rank low; χ=4 near-exact.
    let n = 6u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    let a = mps_dense(&c, 4);
    let b = sv_dense(&c);
    let mut err = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        err += (x - y).norm_sqr();
    }
    assert!(err.sqrt() < 1e-6, "L2 error {} too large", err.sqrt());
}

#[test]
fn vqe_h2_matches_sv_machine_precision() {
    // 4-qubit hardware-efficient ansatz: Ry layers + nearest-neighbor CNOT ladder.
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    let thetas = [0.31, 0.59, 0.27, 0.18, 0.44, 0.62, 0.11, 0.53];
    let mut ti = 0usize;
    for _layer in 0..2 {
        for q in 0..n {
            c.add_gate(g(
                Gate::Ry(Param::Concrete(thetas[ti % thetas.len()])),
                &[q],
            ))
            .unwrap();
            ti += 1;
        }
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
        }
    }
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10, "VQE-H2 MPS vs SV mismatch");
    }
}

#[test]
#[ignore = "50-qubit MPS run; minutes-scale, runs on CI nightly"]
fn qaoa50_nn_ring_runs_reasonably() {
    let n = 50u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for _p in 0..3 {
        // Cost layer: nearest-neighbor ZZ via CNOT–RZ–CNOT on (q, q+1).
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(Gate::Rz(Param::Concrete(0.7)), &[q + 1]))
                .unwrap();
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
        }
        // Mixer: RX on every qubit.
        for q in 0..n {
            c.add_gate(g(Gate::Rx(Param::Concrete(0.5)), &[q])).unwrap();
        }
    }
    let mut be = MpsBackend::with_seed(1).with_max_bond(64);
    let st = run(&mut be, &c).unwrap();
    // "Reasonable": bounded truncation, non-degenerate sampling.
    assert!(
        st.truncation_error() < 1e-1,
        "trunc_error {}",
        st.truncation_error()
    );
    let shots = be.sample(&st, 1000).unwrap();
    let distinct: std::collections::HashSet<u64> = shots.iter().copied().collect();
    assert!(distinct.len() > 1, "sampling produced a single bitstring");
}

fn mps_dense_policy(circuit: &aleph_ir::Circuit, policy: TruncationPolicy) -> (Vec<Complex>, f64) {
    let mut be = MpsBackend::with_seed(0).with_truncation(policy);
    let st: MpsState = run(&mut be, circuit).unwrap();
    (st.dense_statevector(), st.truncation_error())
}

#[test]
fn error_bounded_eps0_is_exact() {
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    for q in 0..n {
        c.add_gate(g(Gate::Rz(Param::Concrete(0.2 + q as f64 * 0.1)), &[q]))
            .unwrap();
    }
    let (a, err) = mps_dense_policy(
        &c,
        TruncationPolicy::ErrorBounded {
            epsilon: 0.0,
            max_bond: 64,
        },
    );
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10);
    }
    assert!(err < 1e-12, "ε=0 should discard nothing, got {err}");
}

#[test]
fn error_bounded_deviation_within_budget() {
    let n = 6u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for layer in 0..3 {
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(
                Gate::Rz(Param::Concrete(0.3 + layer as f64 * 0.1)),
                &[q + 1],
            ))
            .unwrap();
        }
    }
    let (a, err) = mps_dense_policy(
        &c,
        TruncationPolicy::ErrorBounded {
            epsilon: 1e-4,
            max_bond: 64,
        },
    );
    let b = sv_dense(&c);
    let l2: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm_sqr())
        .sum::<f64>()
        .sqrt();
    // Per-truncation 2-norm error is √discardedᵢ; by Cauchy–Schwarz the total
    // satisfies ‖Δψ‖ ≤ √(K · Σ discardedᵢ) over K truncations. This circuit has
    // K = 3 layers × 5 CNOTs = 15 ≤ 16, so the 4× factor is a sound bound. If
    // you enlarge the circuit (K > 16) raise the factor to √K accordingly —
    // do NOT just bump it to make a real regression pass.
    assert!(
        l2 <= 4.0 * err.sqrt() + 1e-9,
        "L2 {l2} vs √err {}",
        err.sqrt()
    );
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random nearest-neighbor 1q+2q circuit on 4 qubits, χ=64 (no truncation):
    /// MPS dense must equal SV dense to 1e-9, norm ≈ 1.
    #[test]
    fn random_nn_circuit_matches_sv(seq in prop::collection::vec(0u8..6, 0..30)) {
        let n = 4u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        let mut q = 0u32;
        for op in seq {
            q = (q + 1) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[q])).unwrap(); }
                1 => { c.add_gate(g(Gate::X, &[q])).unwrap(); }
                2 => { c.add_gate(g(Gate::S, &[q])).unwrap(); }
                3 => { c.add_gate(g(Gate::Y, &[q])).unwrap(); }
                _ => {
                    let lo = q.min(n - 2);
                    c.add_gate(g(Gate::Cnot, &[lo, lo + 1])).unwrap();
                }
            }
        }
        if c.is_empty() { return Ok(()); }
        let a = mps_dense(&c, 64);
        let b = sv_dense(&c);
        let mut norm = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            prop_assert!((x - y).norm() < 1e-9);
            norm += x.norm_sqr();
        }
        prop_assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn error_bounded_eps0_matches_sv_random(seq in prop::collection::vec(0u8..6, 0..24)) {
        let n = 4u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        let mut q = 0u32;
        for op in seq {
            q = (q + 1) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[q])).unwrap(); }
                1 => { c.add_gate(g(Gate::X, &[q])).unwrap(); }
                2 => { c.add_gate(g(Gate::S, &[q])).unwrap(); }
                3 => { c.add_gate(g(Gate::Y, &[q])).unwrap(); }
                _ => { let lo = q.min(n - 2); c.add_gate(g(Gate::Cnot, &[lo, lo + 1])).unwrap(); }
            }
        }
        if c.is_empty() { return Ok(()); }
        let (a, _) = mps_dense_policy(&c, TruncationPolicy::ErrorBounded { epsilon: 0.0, max_bond: 64 });
        let b = sv_dense(&c);
        for (x, y) in a.iter().zip(b.iter()) { prop_assert!((x - y).norm() < 1e-9); }
    }
}
