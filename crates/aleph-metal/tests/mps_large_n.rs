//! P5.8-01: large-n correctness for the Metal MPS backend, asserted with
//! **`2^n`-free invariants** — the dense-statevector oracle (`mps_oracle.rs`) caps
//! out at 28 qubits, which hides the large-χ / large-n regime where a GPU MPS can
//! win (`docs/perf/phase5.7-audit.md`). Readout is `2^n`-free since P5.7-05, so
//! correctness at n ≫ 28 is checked without ever forming the dense vector.
//!
//! Three invariants, none allocating `2^n`:
//! - **Analytic GHZ `Z`-string expectations.** The GHZ state
//!   `(|0…0⟩ + |1…1⟩)/√2` has `⟨Z_i⟩ = 0`, `⟨Z_i Z_j⟩ = +1`, `⟨X^⊗n⟩ = +1` and
//!   norm 1 — exact closed forms to compare the `2^n`-free readout against.
//! - **Norm = 1** on a bond-saturating circuit (the readout's stability under a
//!   real bond, not just the χ=2 GHZ state).
//! - **`run` vs `run_batched` agreement** on a higher-bond *non-truncating* circuit,
//!   compared via expectation values rather than a dense compare. (`run_batched` is
//!   exact-only and refuses truncation, so the cap is kept above the natural bond.)
//!
//! Device-or-skip so headless/Linux CI stays green.
//!
//! Run: `cargo test -p aleph-metal --features metal --test mps_large_n`

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_benches::{brickwall_ry_cnot_rz, ghz_circuit};
use aleph_core::{Pauli, PauliString};
use aleph_ir::Circuit;
use aleph_metal::{MetalMpsBackend, MetalMpsState};

/// Bond cap above every test circuit's natural entanglement (n ≤ 24 here keeps the
/// GHZ bond at 2; the brickwall cell stays under this), so nothing truncates and
/// `run_batched` (exact-only) is usable.
const MAX_BOND: usize = 2048;
/// FP32 tolerance for `2^n`-free expectations/norm: the transfer sweep widens f32
/// site data to f64, but the underlying state is fp32 and error accumulates over n.
const TOL: f64 = 2e-3;

fn z_string(qubits: &[u32]) -> PauliString {
    PauliString::new(1.0, qubits.iter().map(|&q| (q, Pauli::Z)).collect()).expect("z string")
}

fn x_string(qubits: &[u32]) -> PauliString {
    PauliString::new(1.0, qubits.iter().map(|&q| (q, Pauli::X)).collect()).expect("x string")
}

fn assert_expect(label: &str, st: &MetalMpsState, p: &PauliString, want: f64) {
    let got = st.expectation(p).expect("expectation");
    assert!(
        (got - want).abs() <= TOL,
        "{label}: ⟨P⟩={got:.5} want {want:.5} (|Δ|={:.2e} > {TOL:.0e})",
        (got - want).abs()
    );
}

/// GHZ analytic invariants, `2^n`-free. Run on both the gate-by-gate `run` and the
/// layer-batched `run_batched` so both execution paths are checked at large n.
fn check_ghz(n: u32) {
    let circuit = ghz_circuit(n);
    let mut gpu = MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required");

    for (path, st) in [
        ("run", gpu.run(&circuit).expect("ghz run")),
        (
            "run_batched",
            gpu.run_batched(&circuit).expect("ghz batched"),
        ),
    ] {
        let label = format!("ghz n{n} {path}");
        // Norm 1.
        assert!(
            (st.norm() - 1.0).abs() <= TOL,
            "{label}: norm={:.5} ≠ 1",
            st.norm()
        );
        // ⟨X^⊗n⟩ = +1 (the GHZ X-stabiliser).
        assert_expect(&label, &st, &x_string(&(0..n).collect::<Vec<_>>()), 1.0);
        // Per-qubit and pairwise Z: ⟨Z_i⟩ = 0, ⟨Z_i Z_j⟩ = +1.
        for i in 0..n {
            assert_expect(&label, &st, &z_string(&[i]), 0.0);
        }
        // A spread of pairs (not just adjacent) — all correlated in GHZ.
        for &(i, j) in &[(0u32, 1u32), (0, n / 2), (0, n - 1), (n / 2, n - 1)] {
            assert_expect(&label, &st, &z_string(&[i, j]), 1.0);
        }
    }
}

#[test]
fn ghz_z_string_expectations_large_n() {
    if MetalMpsBackend::new().is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    for &n in &[16u32, 20, 24] {
        check_ghz(n);
    }
}

/// Norm = 1 on a bond-*saturating* circuit at large n — the readout invariant under
/// a real (χ ≫ 2) bond, where the dense oracle is out of reach. Ignored by default:
/// this drives the GPU through the slow large-χ truncating regime (~5 min on the M4),
/// so it runs on the nightly-ignored schedule, not every `cargo test`.
#[test]
#[ignore = "large-χ GPU run takes minutes; runs on the nightly-ignored schedule"]
fn bond_saturating_norm_unit_large_n() {
    if MetalMpsBackend::new().is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    for &n in &[16u32, 20] {
        let circuit: Circuit = brickwall_ry_cnot_rz(n, n + 6);
        // Canonical `run` renormalises after each truncating split; norm stays 1.
        let mut gpu = MetalMpsBackend::with_max_bond(512).expect("Metal device required");
        let st = gpu.run(&circuit).expect("saturating run");
        let norm = st.norm();
        assert!(
            norm.is_finite() && (norm - 1.0).abs() <= TOL,
            "n{n}: norm={norm:.5} ≠ 1"
        );
    }
}

/// `run` vs `run_batched` agreement on a higher-bond *non-truncating* circuit,
/// compared `2^n`-free via expectation values (no dense compare). The cap stays
/// above the natural bond so the exact-only batched path does not refuse.
#[test]
fn run_vs_batched_agreement_large_n() {
    if MetalMpsBackend::new().is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    let n = 16u32;
    let circuit = brickwall_ry_cnot_rz(n, 10);
    let mut gpu = MetalMpsBackend::with_max_bond(MAX_BOND).expect("Metal device required");
    let seq = gpu.run(&circuit).expect("run");
    let bat = gpu.run_batched(&circuit).expect("run_batched");

    // A spread of observables: single-site Z across the chain, an X-string, and a
    // few Z-correlators. The two execution paths must agree on every one.
    let mut probes: Vec<PauliString> = (0..n).map(|i| z_string(&[i])).collect();
    probes.push(x_string(&(0..n).collect::<Vec<_>>()));
    probes.push(z_string(&[0, n / 2, n - 1]));
    for p in &probes {
        let a = seq.expectation(p).expect("seq exp");
        let b = bat.expectation(p).expect("bat exp");
        assert!(
            (a - b).abs() <= TOL,
            "run vs batched disagree on {p:?}: {a:.5} vs {b:.5}"
        );
    }
}
