//! End-to-end oracle: `run_optimized` ≡ `run` on `NaiveSvBackend`, amplitudes
//! within 1e-12. Per-pass oracles already exist in aleph-sv; this pins the
//! whole pipeline+sim path — the exact gap that let P1-14 measure raw-vs-fused.
//!
//! This is the **semantic gate** for wiring `Circuit::optimize()` into the run
//! path: it proves that for every input circuit `C`,
//! `run_optimized(C) == run(C)` (the raw, gate-by-gate reference) to FP
//! tolerance, and that measurement outcomes and barrier fences survive the
//! optimization. If the pipeline drops a gate, gets a matrix-product order
//! wrong, fuses across a barrier, or reorders across a measurement, one of
//! these tests catches it.

use aleph_backend::{run, run_optimized, run_optimized_with_outcomes, run_with_outcomes};
use aleph_core::Complex;
use aleph_ir::{Circuit, Instruction};
use aleph_sv::NaiveSvBackend;

const TOL: f64 = 1e-12;

/// Build a gate-only twin of `c`, keeping `Gate` and `Barrier` and dropping
/// `Reset` (rejected by `aleph_backend::run`) and `Measure` (RNG-dependent,
/// irrelevant for a pure-amplitude compare). `Barrier` is preserved so the
/// pipeline's fencing path is exercised by the random-circuit oracle. Mirrors
/// the `gate_only` helper in `aleph-sv/tests/fuse_1q_oracle.rs`.
fn gate_only(c: &Circuit) -> Circuit {
    let mut out = Circuit::new(c.num_qubits(), c.num_clbits());
    for inst in c.instructions() {
        match inst {
            Instruction::Gate(g) => {
                out.add_gate(g.clone())
                    .expect("gate validated in source circuit must validate in clone");
            }
            Instruction::Barrier(_) => {
                out.add_instruction(inst.clone())
                    .expect("barrier validated in source circuit must validate in clone");
            }
            // Drop Reset (rejected by run) and Measure (RNG-dependent).
            _ => {}
        }
    }
    out
}

fn raw_state(c: &Circuit) -> Vec<Complex> {
    run(&mut NaiveSvBackend::with_seed(0), c)
        .expect("raw run executes gate-only IR")
        .amplitudes()
        .to_vec()
}

fn opt_state(c: &Circuit) -> Vec<Complex> {
    run_optimized(&mut NaiveSvBackend::with_seed(0), c)
        .expect("optimized run executes gate-only IR")
        .amplitudes()
        .to_vec()
}

fn assert_states_match(c: &Circuit, label: &str) {
    let a = raw_state(c);
    let b = opt_state(c);
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (*x - *y).norm();
        assert!(
            diff < TOL,
            "{label}: amp[{i}] diff {diff} >= {TOL} (raw={x:?}, opt={y:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 4 — amplitude equivalence
// ---------------------------------------------------------------------------

/// The committed shared QASM circuits (used by the EPYC perf harness) at small
/// n, where simulation is cheap. grover_n15 has ~47k gates but only n=15
/// (32768 amplitudes), so it runs in seconds.
#[test]
fn fixtures_optimized_equals_raw() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/qiskit-baseline/circuits/"
    );
    let names = [
        "ghz_n15",
        "qft_n15",
        "random_brickwall_n15_d20",
        "grover_n15_iters5",
    ];
    for name in names {
        let path = format!("{dir}{name}.qasm");
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse {path}: {e:?}"));
        assert_states_match(&circuit, name);
    }
}

mod property {
    use super::*;
    use aleph_test::circuit::arb_circuit_emittable;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// For random gate+barrier circuits, optimizing via the default
        /// pipeline (Cancel, DCE, Fuse1q, Fuse2q) before simulation must
        /// preserve the output state vector to FP tolerance.
        ///
        /// `arb_circuit_emittable` (the strategy the existing
        /// `fuse_*_oracle.rs` tests use) emits gates and barriers only — no
        /// `Measure` (RNG-dependent, irrelevant for a pure-amplitude compare)
        /// and no `Reset` (rejected by `aleph_backend::run`). Barriers are
        /// preserved so the pipeline's fencing path is exercised here.
        #[test]
        fn random_optimized_equals_raw(raw in arb_circuit_emittable(5, 2, 30)) {
            let c = gate_only(&raw);
            let a = raw_state(&c);
            let b = opt_state(&c);
            prop_assert_eq!(a.len(), b.len());
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let diff = (*x - *y).norm();
                prop_assert!(
                    diff < TOL,
                    "amp[{}] diff {} >= {} (raw={:?}, opt={:?})",
                    i, diff, TOL, x, y
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Task 5 — measurement outcomes + barrier non-crossing
// ---------------------------------------------------------------------------

/// A circuit with a fusible 1q run (`X·Z·X` on q0) feeding a CNOT, plus an `H`
/// on q2, then three measurements. Both drivers with the same seed must yield
/// identical `(qubit, clbit, outcome)` records. We compare on those three
/// fields only — NOT `instruction_index`, which legitimately shifts when
/// fusion reduces the gate count ahead of the measurements.
#[test]
fn measurement_outcomes_optimized_equals_raw() {
    let mut c = Circuit::new(3, 3);
    c.x(0).unwrap();
    c.z(0).unwrap();
    c.x(0).unwrap(); // X·Z·X — a fusible 1q run on q0.
    c.cnot(0, 1).unwrap();
    c.h(2).unwrap();
    c.add_instruction(Instruction::Measure { qubit: 0, clbit: 0 })
        .unwrap();
    c.add_instruction(Instruction::Measure { qubit: 1, clbit: 1 })
        .unwrap();
    c.add_instruction(Instruction::Measure { qubit: 2, clbit: 2 })
        .unwrap();

    let (_s_raw, raw) = run_with_outcomes(&mut NaiveSvBackend::with_seed(7), &c).unwrap();
    let (_s_opt, opt) = run_optimized_with_outcomes(&mut NaiveSvBackend::with_seed(7), &c).unwrap();

    let key = |r: &aleph_backend::MeasurementRecord| (r.qubit, r.clbit, r.outcome);
    let raw_keys: Vec<_> = raw.iter().map(key).collect();
    let opt_keys: Vec<_> = opt.iter().map(key).collect();
    assert_eq!(raw_keys, opt_keys, "outcomes diverged after optimization");
}

/// A `Barrier` between two `T` gates on q0 must block fusion across it; the
/// optimized result must equal the raw result regardless. (If the barrier were
/// ignored, `T·T = S` would still give the same unitary here — but the point
/// is the pipeline must not crash or miscompile across the fence, and the
/// state must match within tolerance.)
#[test]
fn barrier_respected_optimized_equals_raw() {
    let mut c = Circuit::new(1, 0);
    c.t(0).unwrap();
    c.add_instruction(Instruction::Barrier(smallvec::smallvec![0u32]))
        .unwrap();
    c.t(0).unwrap();
    assert_states_match(&c, "barrier_between_t_gates");
}
