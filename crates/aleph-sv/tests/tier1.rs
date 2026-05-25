//! Tier-1 algorithm integration tests. Each test parses OpenQASM 3.0
//! text, runs it through `aleph_backend::run` with `NaiveSvBackend`,
//! and checks the final state against analytic expectations.
//!
//! Oracle comparison against Qiskit lands in P0-10.

use aleph_backend::{run, run_with_outcomes, MeasurementRecord};
use aleph_parser::parse;
use aleph_sv::NaiveSvBackend;

const TOL: f64 = 1e-10;

fn ghz_qasm(n: u32) -> String {
    let mut out = format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{n}] q;\n");
    out.push_str("h q[0];\n");
    for i in 0..n - 1 {
        out.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
    }
    out
}

#[test]
fn ghz_2() {
    let circ = parse(&ghz_qasm(2)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[3].re - inv_s2).abs() < TOL);
    assert!(a[1].norm() < TOL);
    assert!(a[2].norm() < TOL);
}

#[test]
fn ghz_5() {
    let circ = parse(&ghz_qasm(5)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let n = 5;
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    let last = (1usize << n) - 1;
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[last].re - inv_s2).abs() < TOL);
    let mass: f64 = (1..last).map(|i| a[i].norm_sqr()).sum();
    assert!(mass < TOL);
}

#[test]
fn ghz_10() {
    let circ = parse(&ghz_qasm(10)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let n = 10;
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    let last = (1usize << n) - 1;
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[last].re - inv_s2).abs() < TOL);
    let mass: f64 = (1..last).map(|i| a[i].norm_sqr()).sum();
    assert!(mass < TOL);
}

#[test]
fn ghz_20_runs() {
    // Acceptance criterion: 20 qubits must run end-to-end.
    let circ = parse(&ghz_qasm(20)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let n = 20;
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    let last = (1usize << n) - 1;
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[last].re - inv_s2).abs() < TOL);
}

#[test]
#[ignore = "n=25 needs ~512 MiB state vector; ROADMAP Phase-0 exit criterion. Run with `cargo test --release -- --ignored ghz_25_runs`."]
fn ghz_25_runs() {
    // ROADMAP § 7 Phase-0 success metric: "25-qubit GHZ runs
    // end-to-end".  At n=25 the state vector is 2^25 × 16 B ≈
    // 512 MiB; default `cargo test` builds with `-O0` and several
    // checks (validate_state per primitive call, unitarity guard
    // per gate dispatch) make this take seconds.  Run in release.
    let circ = parse(&ghz_qasm(25)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let n = 25;
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    let last = (1usize << n) - 1;
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[last].re - inv_s2).abs() < TOL);
    // Quick sanity check that the "in between" amplitudes are zero
    // (they form ~33 M samples — check a stride rather than all).
    for i in (1..last).step_by(1usize << (n - 6)) {
        assert!(
            a[i].norm() < TOL,
            "ghz_25: amp[{i}] = {} should be ≈ 0",
            a[i].norm()
        );
    }
}

#[test]
fn qft_3_on_one_probabilities_are_uniform() {
    // QFT applied to any computational basis state |x⟩ yields a state
    // with uniform probability 1/N over basis states. Phases differ by
    // `2π·x·k/N` but those are global-phase-sensitive after composing
    // controlled-phase decompositions, so check probabilities only.
    //
    // QFT-3 circuit, with `cp(λ) c, t` expanded as
    //   rz(λ/2) t; cx c, t; rz(-λ/2) t; cx c, t; rz(λ/2) c;
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
x q[0];
h q[2];
// cp(pi/2) q[1], q[2]:
rz(pi/4) q[2];
cx q[1], q[2];
rz(-pi/4) q[2];
cx q[1], q[2];
rz(pi/4) q[1];
// cp(pi/4) q[0], q[2]:
rz(pi/8) q[2];
cx q[0], q[2];
rz(-pi/8) q[2];
cx q[0], q[2];
rz(pi/8) q[0];
h q[1];
// cp(pi/2) q[0], q[1]:
rz(pi/4) q[1];
cx q[0], q[1];
rz(-pi/4) q[1];
cx q[0], q[1];
rz(pi/4) q[0];
h q[0];
swap q[0], q[2];
"#;
    let circ = parse(src).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let expected = 1.0 / 8.0;
    for (k, a) in s.amplitudes().iter().enumerate() {
        let p = a.norm_sqr();
        assert!(
            (p - expected).abs() < 1e-10,
            "k={k}: p={p}, expected {expected}"
        );
    }
}

#[test]
fn grover_3_one_marked() {
    // 3-qubit Grover with marked state |111⟩. One iteration yields
    // P(|111⟩) ≈ 0.7812. `ccz` is expanded as `h ccx h` on the target.
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
h q[1];
h q[2];
// ccz q[0], q[1], q[2]:
h q[2];
ccx q[0], q[1], q[2];
h q[2];
h q[0];
h q[1];
h q[2];
x q[0];
x q[1];
x q[2];
// ccz q[0], q[1], q[2]:
h q[2];
ccx q[0], q[1], q[2];
h q[2];
x q[0];
x q[1];
x q[2];
h q[0];
h q[1];
h q[2];
"#;
    let circ = parse(src).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let p_marked = s.amplitudes()[7].norm_sqr();
    assert!(p_marked > 0.78, "p_marked = {p_marked}");
}

#[test]
fn run_with_outcomes_records_in_instruction_order() {
    // Bell + two measurements. Outcomes must always agree (perfect
    // correlation) and be recorded in instruction order with the right
    // qubit/clbit pairing.
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
    let circ = parse(src).unwrap();
    let mut b = NaiveSvBackend::with_seed(11);
    let (_state, outcomes) = run_with_outcomes(&mut b, &circ).unwrap();
    assert_eq!(outcomes.len(), 2);
    let first = outcomes[0];
    let second = outcomes[1];
    assert_eq!(first.qubit, 0);
    assert_eq!(first.clbit, 0);
    assert_eq!(second.qubit, 1);
    assert_eq!(second.clbit, 1);
    assert!(first.instruction_index < second.instruction_index);
    // Bell-state correlation: both outcomes equal.
    assert_eq!(first.outcome, second.outcome);
    // Records carry the expected struct shape.
    let _: &MeasurementRecord = &first;
}

#[test]
fn random_clifford_t_8q_is_deterministic_and_normalised() {
    // 8-qubit random-ish Clifford+T circuit (depth ~10). Determinism
    // check: same circuit + same input ⇒ same final state.
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[8] q;
h q[0]; h q[1]; h q[2]; h q[3];
t q[4]; t q[5]; t q[6]; t q[7];
cx q[0], q[4];
cx q[1], q[5];
cx q[2], q[6];
cx q[3], q[7];
s q[0]; s q[1];
h q[4]; h q[5];
cx q[4], q[0];
cx q[5], q[1];
t q[2]; t q[3];
cx q[6], q[2];
cx q[7], q[3];
h q[6]; h q[7];
"#;
    let circ = parse(src).unwrap();
    let mut b1 = NaiveSvBackend::with_seed(0);
    let mut b2 = NaiveSvBackend::with_seed(0);
    let s1 = run(&mut b1, &circ).unwrap();
    let s2 = run(&mut b2, &circ).unwrap();
    // Determinism.
    for (a, b) in s1.amplitudes().iter().zip(s2.amplitudes().iter()) {
        assert!((a - b).norm() < 1e-15);
    }
    // Normalisation.
    let total: f64 = s1.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((total - 1.0).abs() < 1e-10, "norm² = {total}");
}
