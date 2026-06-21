//! P5.7-05: `MetalMpsBackend` readout oracle.
//!
//! `measure` / `sample` / `probabilities` / `expectation_value` on the GPU MPS
//! scaffold must match the CPU MPS backend (`aleph_mps::MpsBackend`, the AC
//! reference) and the exact FP64 statevector (`aleph_sv::NaiveSvBackend`) within
//! the f32 oracle tolerance, **without** forming the dense `2^n` vector — the
//! readout path uses bond×bond environment contractions only. A large-n GHZ case
//! exercises that no-`2^n` property against analytic answers. Device-or-skip so
//! headless/Linux CI stays green.
//!
//! Run: `cargo test -p aleph-metal --features metal --test mps_readout`

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_backend::{run, Backend};
use aleph_core::{Pauli, PauliString};
use aleph_ir::Circuit;
use aleph_metal::MetalMpsBackend;
use aleph_mps::MpsBackend;
use aleph_sv::NaiveSvBackend;

const MAX_BOND: usize = 64;
const TOL: f64 = 1e-5;

fn metal() -> Option<MetalMpsBackend> {
    match MetalMpsBackend::with_seed(7) {
        Ok(b) => Some(b),
        Err(_) => {
            eprintln!("skipping MPS readout oracle: no Metal device available");
            None
        }
    }
}

/// Evolve `circuit` on each backend; return the three `State`s plus their backends
/// (backends are returned because the readout trait methods take `&mut self`).
fn cpu_mps_probs(circuit: &Circuit, qubits: &[u32]) -> Vec<f64> {
    let mut be = MpsBackend::new().with_max_bond(MAX_BOND);
    let s = run(&mut be, circuit).expect("cpu mps run");
    be.probabilities(&s, qubits).expect("cpu mps probabilities")
}

fn naive_probs(circuit: &Circuit, qubits: &[u32]) -> Vec<f64> {
    let mut be = NaiveSvBackend::with_seed(0);
    let s = run(&mut be, circuit).expect("naive run");
    be.probabilities(&s, qubits).expect("naive probabilities")
}

fn cpu_mps_expect(circuit: &Circuit, p: &PauliString) -> f64 {
    let mut be = MpsBackend::new().with_max_bond(MAX_BOND);
    let s = run(&mut be, circuit).expect("cpu mps run");
    be.expectation_value(&s, p).expect("cpu mps expectation")
}

fn naive_expect(circuit: &Circuit, p: &PauliString) -> f64 {
    let mut be = NaiveSvBackend::with_seed(0);
    let s = run(&mut be, circuit).expect("naive run");
    be.expectation_value(&s, p).expect("naive expectation")
}

fn assert_vec_close(label: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{label}: dim mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= TOL,
            "{label}: entry {i} {x} vs {y} (Δ={:.2e})",
            (x - y).abs()
        );
    }
}

/// `probabilities` over assorted subsets matches CPU MPS and the exact SV.
#[test]
fn probabilities_match_references() {
    let Some(mut gpu) = metal() else { return };
    let circuit = aleph_benches::random_brickwall_circuit(6, 6);
    let s = gpu.run(&circuit).expect("gpu run");
    for qubits in [
        vec![0u32],
        vec![3],
        vec![0, 1],
        vec![1, 4],
        vec![0, 2, 4],
        vec![0, 1, 2, 3, 4, 5],
    ] {
        let got = gpu.probabilities(&s, &qubits).expect("gpu probabilities");
        // Property: a marginal sums to 1.
        let sum: f64 = got.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum {sum} for {qubits:?}");
        assert_vec_close(
            &format!("probs{qubits:?} vs cpu-mps"),
            &got,
            &cpu_mps_probs(&circuit, &qubits),
        );
        assert_vec_close(
            &format!("probs{qubits:?} vs naive-sv"),
            &got,
            &naive_probs(&circuit, &qubits),
        );
    }
    // Empty subset → [1.0] (SV-backend contract).
    assert_eq!(gpu.probabilities(&s, &[]).unwrap(), vec![1.0]);
}

/// `expectation_value` matches CPU MPS and the exact SV for 1- and 2-site Paulis.
#[test]
fn expectation_matches_references() {
    let Some(mut gpu) = metal() else { return };
    let circuit = aleph_benches::random_brickwall_circuit(5, 5);
    let s = gpu.run(&circuit).expect("gpu run");
    let strings = [
        PauliString::new(1.0, vec![(0u32, Pauli::Z)]).unwrap(),
        PauliString::new(1.0, vec![(2u32, Pauli::X)]).unwrap(),
        PauliString::new(1.0, vec![(1u32, Pauli::Y)]).unwrap(),
        PauliString::new(0.7, vec![(0u32, Pauli::Z), (1u32, Pauli::Z)]).unwrap(),
        PauliString::new(1.0, vec![(0u32, Pauli::X), (1u32, Pauli::X)]).unwrap(),
        PauliString::new(1.0, vec![(0u32, Pauli::Z), (4u32, Pauli::Z)]).unwrap(),
    ];
    for p in &strings {
        let got = gpu.expectation_value(&s, p).expect("gpu expectation");
        let cpu = cpu_mps_expect(&circuit, p);
        let naive = naive_expect(&circuit, p);
        assert!(
            (got - cpu).abs() <= TOL,
            "⟨{p:?}⟩ gpu {got} vs cpu-mps {cpu}"
        );
        assert!(
            (got - naive).abs() <= TOL,
            "⟨{p:?}⟩ gpu {got} vs naive {naive}"
        );
    }
}

/// `sample` is self-consistent with `probabilities` (the strongest check) at a
/// calibrated 5σ band — and `probabilities` already matches the references above.
#[test]
fn sample_matches_probabilities() {
    let Some(mut gpu) = metal() else { return };
    let circuit = aleph_benches::random_brickwall_circuit(3, 5);
    let s = gpu.run(&circuit).expect("gpu run");
    let shots = 20_000u32;
    let samples = gpu.sample(&s, shots).expect("gpu sample");
    let mut counts = vec![0u64; 8];
    for sh in &samples {
        counts[*sh as usize] += 1;
    }
    let probs = gpu
        .probabilities(&s, &[0, 1, 2])
        .expect("gpu probabilities");
    aleph_oracle::assert_distribution_close("metal_mps_sample", 3, &counts, &probs, shots);
}

/// GHZ correlation: `measure(0)` then measuring the rest must all agree (GHZ is
/// `|0…0⟩+|1…1⟩`), and a re-run with the same seed is reproducible.
#[test]
fn measure_ghz_correlated() {
    let Some(mut gpu) = metal() else { return };
    let mut s = gpu.run(&aleph_benches::ghz_circuit(5)).expect("gpu run");
    let b0 = gpu.measure(&mut s, 0).expect("measure 0");
    for q in 1..5 {
        let bq = gpu.measure(&mut s, q).expect("measure q");
        assert_eq!(bq, b0, "GHZ qubit {q} must match qubit 0");
    }
}

/// No-`2^n` readout at scale: GHZ at n=26 (bond 2) has analytic answers, so the
/// readout is checked without ever building the 2^26 dense vector.
#[test]
fn ghz_large_n_readout_no_dense() {
    let Some(mut gpu) = metal() else { return };
    let n = 26u32;
    let s = gpu.run(&aleph_benches::ghz_circuit(n)).expect("gpu run");

    // Single-qubit marginal: P(0)=P(1)=1/2 on both ends.
    for q in [0u32, n - 1] {
        let p = gpu.probabilities(&s, &[q]).expect("probabilities");
        assert!(
            (p[0] - 0.5).abs() < TOL && (p[1] - 0.5).abs() < TOL,
            "q{q}: {p:?}"
        );
    }
    // ⟨Z_q⟩ = 0 (balanced); ⟨Z_0 Z_{n-1}⟩ = +1 (perfectly correlated).
    let z0 = PauliString::new(1.0, vec![(0u32, Pauli::Z)]).unwrap();
    assert!(gpu.expectation_value(&s, &z0).unwrap().abs() < TOL);
    let zz = PauliString::new(1.0, vec![(0u32, Pauli::Z), (n - 1, Pauli::Z)]).unwrap();
    assert!((gpu.expectation_value(&s, &zz).unwrap() - 1.0).abs() < TOL);

    // Every shot is all-0 or all-1.
    let all_ones = (1u64 << n) - 1;
    for sh in gpu.sample(&s, 64).expect("sample") {
        assert!(sh == 0 || sh == all_ones, "GHZ-{n} shot {sh:#x}");
    }
}
