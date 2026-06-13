//! Driver-level tests for `aleph_sv::noise::run_noisy`: determinism and the
//! noiseless guard (empty model reproduces the noiseless distribution).

use aleph_oracle::assert_distribution_close;
use aleph_parser::parse;
use aleph_sv::noise::{depolarizing_error, run_noisy, NoiseModel};

/// A 2-qubit Bell circuit (gate-only — terminal sampling does the measuring).
const BELL: &str = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"#;

#[test]
fn deterministic_same_seed_same_counts() {
    let circ = parse(BELL).unwrap();
    let mut nm = NoiseModel::new();
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), &["h"]);
    let a = run_noisy(&circ, &nm, 20_000, 7).unwrap();
    let b = run_noisy(&circ, &nm, 20_000, 7).unwrap();
    assert_eq!(a, b, "same seed must give identical counts");
    let c = run_noisy(&circ, &nm, 20_000, 8).unwrap();
    assert_ne!(a, c, "different seed should (almost surely) differ");
}

#[test]
fn empty_model_reproduces_noiseless_distribution() {
    let circ = parse(BELL).unwrap();
    let nm = NoiseModel::new(); // no errors
    let counts = run_noisy(&circ, &nm, 100_000, 1).unwrap();
    // Noiseless Bell: |00⟩ and |11⟩ each 0.5, |01⟩=|10⟩=0.
    let exact = vec![0.5, 0.0, 0.0, 0.5];
    assert_distribution_close("noise_empty_bell", 2, &counts, &exact, 100_000);
}
