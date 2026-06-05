//! Oracle equivalence vs NaiveSvBackend + MPS invariants.
//! MPS dense reconstruction must match the SV amplitude vector (ADR-0004).

use aleph_backend::{run, Backend};
use aleph_core::{Complex, Gate, GateInstance, Param, Pauli, PauliString};
use aleph_mps::{MpsBackend, MpsState};
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
}
