//! Oracle: `FuseDiagonalRuns` preserves the EXACT statevector (global phase
//! included) at 1e-12, on a GENERIC (non-|0…0⟩) input state, validated through
//! BOTH state-vector backends (AoS `NaiveSvBackend` and SoA `SoaSvBackend`).
//!
//! Two QFT encodings are stressed:
//!   * the controlled-Phase (builder) form, and
//!   * the decomposed (`p` + `cx`) form — which exercises the cx-absorption
//!     path inside the fusion pass.
//!
//! Plus a `proptest!` over random {diagonal ∪ cnot} runs.
//!
//! Why a *generic* prefix? Per the P1-13 lesson, an equivalence oracle that
//! only checks |0…0⟩ can miss `cx`-conjugation bugs (a `Z` on the target of a
//! `cx` is invisible on the all-zero state). Every circuit therefore starts
//! with a non-diagonal entangling prefix (H + rx + ry on each qubit).
//! `FuseDiagonalRuns` will NOT touch that prefix (H/rx/ry are run-breakers),
//! so the diagonal run downstream acts on a fully generic, entangled state.

use aleph_backend::run;
use aleph_core::{Gate, GateInstance};
use aleph_ir::passes::{FuseDiagonalRuns, Pass};
use aleph_ir::Circuit;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};
use smallvec::smallvec;
use std::f64::consts::PI;

const TOL: f64 = 1e-12;

/// Non-diagonal entangling prefix so the diagonal run acts on a generic state.
fn generic_prefix(c: &mut Circuit, n: u32) {
    for q in 0..n {
        c.h(q).unwrap();
        c.rx(0.3 + 0.1 * q as f64, q).unwrap();
        c.ry(0.7 - 0.05 * q as f64, q).unwrap();
    }
}

/// Builder QFT (controlled-Phase form) appended to `c`. Mirrors
/// benches/src/lib.rs::qft_circuit: control = higher-index qubit k, target = j.
fn append_qft_builder(c: &mut Circuit, n: u32) {
    for j in 0..n {
        c.h(j).unwrap();
        for k in (j + 1)..n {
            let theta = PI / (1u64 << (k - j)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(theta.into()),
                smallvec![j],
                smallvec![k],
            ))
            .unwrap();
        }
    }
}

/// Decomposed QFT: each controlled-Phase(θ) on (control=k, target=j) lowered to
/// p(θ/2)@k ; cx(k,j) ; p(-θ/2)@j ; cx(k,j) ; p(θ/2)@j. Exercises the
/// cx-absorption path inside `FuseDiagonalRuns`.
fn append_qft_decomposed(c: &mut Circuit, n: u32) {
    for j in 0..n {
        c.h(j).unwrap();
        for k in (j + 1)..n {
            let theta = PI / (1u64 << (k - j)) as f64;
            let half = theta / 2.0;
            c.add_gate(GateInstance::new(Gate::Phase(half.into()), smallvec![k]))
                .unwrap();
            c.cnot(k, j).unwrap();
            c.add_gate(GateInstance::new(Gate::Phase((-half).into()), smallvec![j]))
                .unwrap();
            c.cnot(k, j).unwrap();
            c.add_gate(GateInstance::new(Gate::Phase(half.into()), smallvec![j]))
                .unwrap();
        }
    }
}

/// Run `c` through `NaiveSvBackend` (AoS), returning the full amplitude vector.
fn naive_amps(c: &Circuit) -> Vec<aleph_core::Complex> {
    let mut be = NaiveSvBackend::with_seed(0);
    run(&mut be, c).unwrap().amplitudes().to_vec()
}

/// Run `c` through `SoaSvBackend` (SoA), returning the full amplitude vector.
/// `SoaState` stores re/im split, so materialize via `to_aos()`.
fn soa_amps(c: &Circuit) -> Vec<aleph_core::Complex> {
    let mut be = SoaSvBackend::new();
    run(&mut be, c).unwrap().to_aos()
}

/// Assert fused ≡ unfused (global phase included) through BOTH backends.
/// Reference = unfused through `NaiveSvBackend`.
fn assert_fused_equiv(base: &Circuit) {
    let mut fused = base.clone();
    FuseDiagonalRuns.run(&mut fused).unwrap();

    // reference: unfused on naive (AoS)
    let aref = naive_amps(base);

    // fused on naive (AoS kernel)
    let aaos = naive_amps(&fused);
    assert_eq!(aref.len(), aaos.len());
    for (x, (u, f)) in aref.iter().zip(aaos.iter()).enumerate() {
        assert!(
            (*u - *f).norm() < TOL,
            "AoS amp {x}: unfused {u:?} vs fused {f:?} (diff {})",
            (*u - *f).norm()
        );
    }

    // fused on soa (SoA kernel)
    let asoa = soa_amps(&fused);
    assert_eq!(aref.len(), asoa.len());
    for (x, (u, f)) in aref.iter().zip(asoa.iter()).enumerate() {
        assert!(
            (*u - *f).norm() < TOL,
            "SoA amp {x}: unfused {u:?} vs fused {f:?} (diff {})",
            (*u - *f).norm()
        );
    }
}

#[test]
fn builder_qft_fused_equiv_generic_state() {
    for n in [3u32, 5, 8] {
        let mut c = Circuit::new(n, 0);
        generic_prefix(&mut c, n);
        append_qft_builder(&mut c, n);
        assert_fused_equiv(&c);
    }
}

#[test]
fn decomposed_qft_fused_equiv_generic_state() {
    for n in [3u32, 5, 7] {
        let mut c = Circuit::new(n, 0);
        generic_prefix(&mut c, n);
        append_qft_decomposed(&mut c, n);
        assert_fused_equiv(&c);
    }
}

#[test]
fn fusion_actually_fires_on_qft() {
    // Guard against a vacuous oracle: confirm the pass really produced a
    // DiagonalPhase on a builder QFT (otherwise equivalence is trivial).
    let n = 5u32;
    let mut c = Circuit::new(n, 0);
    generic_prefix(&mut c, n);
    append_qft_builder(&mut c, n);
    FuseDiagonalRuns.run(&mut c).unwrap();
    assert!(
        c.instructions()
            .iter()
            .any(|i| matches!(i, aleph_ir::Instruction::DiagonalPhase(_))),
        "expected at least one fused DiagonalPhase"
    );
}

#[test]
fn fusion_fires_on_decomposed_qft() {
    // The cx-absorption path must also collapse into a DiagonalPhase.
    let n = 5u32;
    let mut c = Circuit::new(n, 0);
    generic_prefix(&mut c, n);
    append_qft_decomposed(&mut c, n);
    FuseDiagonalRuns.run(&mut c).unwrap();
    assert!(
        c.instructions()
            .iter()
            .any(|i| matches!(i, aleph_ir::Instruction::DiagonalPhase(_))),
        "expected at least one fused DiagonalPhase on decomposed QFT"
    );
}

// ---------------------------------------------------------------------------
// Proptest: random {diagonal ∪ cnot} runs ≡ unfused, on a generic state.
// ---------------------------------------------------------------------------

use proptest::prelude::*;

/// One random op drawn from the diagonal-fusible alphabet plus `cnot`.
#[derive(Clone, Debug)]
enum RandOp {
    Phase(usize, f64),
    Rz(usize, f64),
    Z(usize),
    S(usize),
    T(usize),
    Cz(usize, usize),
    Cnot(usize, usize),
}

/// Strategy for `n` qubits: a single op. `Cz`/`Cnot` pick two DISTINCT qubits
/// (GateInstance debug-asserts uniqueness; a duplicate would be a test bug).
fn arb_op(n: usize) -> impl Strategy<Value = RandOp> {
    let q = 0..n;
    let ang = -PI..PI;
    // Distinct ordered pair: pick a, then an offset 1..n, wrap to avoid a==b.
    let pair = (0..n, 1..n).prop_map(move |(a, d)| (a, (a + d) % n));
    prop_oneof![
        (q.clone(), ang.clone()).prop_map(|(a, t)| RandOp::Phase(a, t)),
        (q.clone(), ang).prop_map(|(a, t)| RandOp::Rz(a, t)),
        q.clone().prop_map(RandOp::Z),
        q.clone().prop_map(RandOp::S),
        q.prop_map(RandOp::T),
        pair.clone().prop_map(|(a, b)| RandOp::Cz(a, b)),
        pair.prop_map(|(a, b)| RandOp::Cnot(a, b)),
    ]
}

/// `(n, ops)`: qubit count 2..=5 paired with a 4..20-op run valid for that `n`.
/// Using `prop_flat_map` lets `n` drive op generation, so every generated op
/// references a qubit `< n` (no post-hoc filtering).
fn arb_n_and_run() -> impl Strategy<Value = (usize, Vec<RandOp>)> {
    (2usize..=5)
        .prop_flat_map(|n| prop::collection::vec(arb_op(n), 4..20).prop_map(move |ops| (n, ops)))
}

fn apply_randop(c: &mut Circuit, op: &RandOp) {
    match *op {
        RandOp::Phase(a, t) => c
            .add_gate(GateInstance::new(
                Gate::Phase(t.into()),
                smallvec![a as u32],
            ))
            .map(|_| ())
            .unwrap(),
        RandOp::Rz(a, t) => c
            .add_gate(GateInstance::new(Gate::Rz(t.into()), smallvec![a as u32]))
            .map(|_| ())
            .unwrap(),
        RandOp::Z(a) => c.z(a as u32).map(|_| ()).unwrap(),
        RandOp::S(a) => c.s(a as u32).map(|_| ()).unwrap(),
        RandOp::T(a) => c.t(a as u32).map(|_| ()).unwrap(),
        RandOp::Cz(a, b) => c.cz(a as u32, b as u32).map(|_| ()).unwrap(),
        RandOp::Cnot(a, b) => c.cnot(a as u32, b as u32).map(|_| ()).unwrap(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random {Phase, Rz, Z, S, T, Cz, Cnot} run after a generic entangling
    /// prefix: fusing must preserve the exact state (global phase included)
    /// through both SV backends. The prefix guarantees a non-|0…0⟩ input so
    /// cx-conjugation bugs cannot hide on the all-zero state (P1-13 lesson).
    #[test]
    fn random_diagonal_cnot_run_fused_equiv((n, ops) in arb_n_and_run()) {
        let mut c = Circuit::new(n as u32, 0);
        generic_prefix(&mut c, n as u32);
        for op in &ops {
            apply_randop(&mut c, op);
        }

        let mut fused = c.clone();
        FuseDiagonalRuns.run(&mut fused).unwrap();

        let aref = naive_amps(&c);
        let aaos = naive_amps(&fused);
        let asoa = soa_amps(&fused);

        prop_assert_eq!(aref.len(), aaos.len());
        prop_assert_eq!(aref.len(), asoa.len());
        for (i, ((u, fa), fs)) in aref.iter().zip(aaos.iter()).zip(asoa.iter()).enumerate() {
            let d_aos = (*u - *fa).norm();
            let d_soa = (*u - *fs).norm();
            prop_assert!(
                d_aos < TOL,
                "AoS amp[{}] diff {} >= {} (unfused={:?}, fused={:?})",
                i, d_aos, TOL, u, fa
            );
            prop_assert!(
                d_soa < TOL,
                "SoA amp[{}] diff {} >= {} (unfused={:?}, fused={:?})",
                i, d_soa, TOL, u, fs
            );
        }
    }
}
