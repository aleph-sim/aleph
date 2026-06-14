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

/// A 1-qubit `X` circuit: noiseless terminal state is |1⟩ with certainty.
const SINGLE_X: &str = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
x q[0];
"#;

#[test]
fn deterministic_same_seed_same_counts() {
    let circ = parse(BELL).unwrap();
    let mut nm = NoiseModel::new();
    // Attach by aleph's internal Gate::name() ("H"), NOT the QASM "h" — the
    // driver resolves channels via gi.gate.name(), so a lowercase key would
    // silently never fire (see noise_actually_fires_under_correct_gate_name).
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), &["H"]);
    let a = run_noisy(&circ, &nm, 20_000, 7).unwrap();
    let b = run_noisy(&circ, &nm, 20_000, 7).unwrap();
    assert_eq!(a, b, "same seed must give identical counts");
    let c = run_noisy(&circ, &nm, 20_000, 8).unwrap();
    assert_ne!(a, c, "different seed should (almost surely) differ");
}

#[test]
fn noise_actually_fires_under_correct_gate_name() {
    // Regression guard for the lowercase-key bug: a channel keyed by the
    // wrong gate name silently no-ops, so a "noisy" test can pass while
    // exercising none of the noise path. On a 1-qubit X circuit the noiseless
    // terminal state is |1⟩ with certainty; depolarizing on "X" injects X/Y
    // (which flip |1⟩→|0⟩) with measurable probability, so |0⟩ MUST appear.
    let circ = parse(SINGLE_X).unwrap();
    let mut nm = NoiseModel::new();
    nm.add_all_qubit_quantum_error(depolarizing_error(0.3, 1), &["X"]);
    let counts = run_noisy(&circ, &nm, 50_000, 11).unwrap();
    // hist index 0 == |0⟩. p(0) ≈ (2/3)·(3·0.3/4) = 0.15 → comfortably > 0.
    assert!(
        counts[0] > 1_000,
        "depolarizing on X must flip some |1⟩→|0⟩; got counts={counts:?} \
         (a lowercase \"x\" key would silently no-op and leave counts[0]==0)"
    );

    // And the same channel keyed by the WRONG (lowercase) name fires nothing:
    // the distribution must then be the noiseless all-|1⟩.
    let mut wrong = NoiseModel::new();
    wrong.add_all_qubit_quantum_error(depolarizing_error(0.3, 1), &["x"]);
    let no_fire = run_noisy(&circ, &wrong, 50_000, 11).unwrap();
    assert_eq!(no_fire[0], 0, "a non-matching gate-name key must not fire");
    assert_eq!(no_fire[1], 50_000, "noiseless X circuit is all |1⟩");
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
