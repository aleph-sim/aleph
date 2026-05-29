//! P1-09 oracle test — fused circuit ≡ unfused circuit on
//! `NaiveSvBackend`. Property: state-vector entries match within
//! 1e-12 across random circuits.
//!
//! This is the **semantic anchor** for the gate-fusion pass: T8's
//! property tests (`fuse_1q_properties.rs`) only confirm structural
//! invariants (determinism, length non-growth, touch-count
//! monotonicity). T9 confirms that for every input circuit C,
//! running `C.optimize()` produces a state vector identical to
//! running `C` itself — modulo 1e-12 FP drift. If fusion gets the
//! matrix product order wrong, drops a gate, or breaks a fence,
//! this test catches it.
//!
//! Filtering rationale: `arb_circuit_emittable` with `nq` at least
//! 2 emits `Instruction::Reset` (rejected by `aleph_backend::run`
//! as `UnsupportedInstruction`) and `Instruction::Measure`
//! (RNG-dependent, irrelevant for unitary equivalence). We filter
//! both classes out before running. Fusion still has full coverage
//! of the gate path — the 1q runs the pass actually fuses are
//! unaffected by which non-gate instructions originally separated
//! them.

use aleph_backend::run;
use aleph_core::Complex;
use aleph_ir::{Circuit, Instruction};
use aleph_sv::NaiveSvBackend;
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

const TOL: f64 = 1e-12;

/// Build a gate-only twin of `c` by copying only `Instruction::Gate`
/// entries. `Reset` and `Measure` are dropped (see module doc).
fn gate_only(c: &Circuit) -> Circuit {
    let mut out = Circuit::new(c.num_qubits(), c.num_clbits());
    for inst in c.instructions() {
        if let Instruction::Gate(g) = inst {
            out.add_gate(g.clone())
                .expect("gate validated in source circuit must validate in clone");
        }
    }
    out
}

fn run_to_state(c: &Circuit) -> Vec<Complex> {
    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, c).expect("naive backend must execute gate-only IR");
    state.amplitudes().to_vec()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    /// For random circuits, optimising via `Circuit::optimize()`
    /// (currently the `Fuse1qRuns` pass) must preserve the output
    /// state vector to FP tolerance on the naive CPU backend.
    #[test]
    fn fused_matches_unfused_on_random_circuit(
        c in arb_circuit_emittable(5, 1, 32),
    ) {
        let unfused = gate_only(&c);
        let mut fused = unfused.clone();
        fused
            .optimize()
            .expect("Phase-1 optimize() cannot fail on a validated circuit");

        let unfused_state = run_to_state(&unfused);
        let fused_state = run_to_state(&fused);

        prop_assert_eq!(unfused_state.len(), fused_state.len());
        for (i, (a, b)) in unfused_state.iter().zip(fused_state.iter()).enumerate() {
            let diff = (*a - *b).norm();
            prop_assert!(
                diff < TOL,
                "amplitude[{}] diff {} >= {} (unfused={:?}, fused={:?})",
                i, diff, TOL, a, b
            );
        }
    }
}
