//! P2-07 oracle: `FuseKq` preserves the EXACT statevector (global phase
//! included) at 1e-12, on a GENERIC (non-|0…0⟩) input state, for fusible
//! VQE/QAOA/random-brickwall-shaped workloads, validated through BOTH
//! state-vector backends (AoS `NaiveSvBackend` and SoA `SoaSvBackend`).
//!
//! Plus a `proptest!` over random rotation/entangler runs.
//!
//! Why a *generic* prefix? Per the P1-13 lesson, an equivalence oracle that
//! only checks |0…0⟩ can miss cx-conjugation bugs (a `Z` on the target of a
//! `cx` is invisible on the all-zero state). Every circuit therefore starts
//! with a non-diagonal entangling prefix (H + rx + ry on each qubit) so the
//! fused blocks downstream act on a fully generic, entangled state.
//!
//! `FuseKq` is exercised STANDALONE here (it is not yet wired into
//! `Circuit::optimize()` — that is Task 10).

use aleph_backend::run;
use aleph_core::{Gate, GateInstance};
use aleph_ir::passes::{FuseKq, Pass};
use aleph_ir::{Circuit, Instruction};
use aleph_sv::{NaiveSvBackend, SoaSvBackend};
use smallvec::smallvec;
use std::f64::consts::PI;

const TOL: f64 = 1e-12;

/// Non-diagonal entangling prefix so the fused blocks act on a generic state.
fn generic_prefix(c: &mut Circuit, n: u32) {
    for q in 0..n {
        c.h(q).unwrap();
        c.rx(0.3 + 0.1 * q as f64, q).unwrap();
        c.ry(0.7 - 0.05 * q as f64, q).unwrap();
    }
}

/// QAOA-like: layers of RZ + nearest-neighbour CNOT entanglers + RX mixer.
fn qaoa_like(c: &mut Circuit, n: u32, layers: usize) {
    for l in 0..layers {
        for q in 0..n {
            c.rz(0.2 + 0.13 * (l as f64) + 0.05 * q as f64, q).unwrap();
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in 0..n {
            c.rx(0.4 - 0.03 * (l as f64), q).unwrap();
        }
    }
}

/// VQE-like (hardware-efficient ansatz): RY/RZ rotations + CZ ladder.
fn vqe_like(c: &mut Circuit, n: u32, layers: usize) {
    for l in 0..layers {
        for q in 0..n {
            c.ry(0.5 + 0.07 * (l as f64) + 0.02 * q as f64, q).unwrap();
            c.rz(0.3 - 0.04 * q as f64, q).unwrap();
        }
        for q in 0..n.saturating_sub(1) {
            c.cz(q, q + 1).unwrap();
        }
    }
}

/// Random brick-wall: alternating CNOT pairs + per-qubit rotations.
fn random_brickwall(c: &mut Circuit, n: u32, depth: usize) {
    for d in 0..depth {
        for q in 0..n {
            c.rz(((d as f64) + 0.37 * q as f64).cos(), q).unwrap();
        }
        let start = d % 2;
        let mut q = start as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
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
/// Reference = unfused through `NaiveSvBackend` (AoS).
fn assert_fused_equiv(base: &Circuit, max_qubits: usize) {
    let mut fused = base.clone();
    FuseKq { max_qubits }.run(&mut fused).unwrap();

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

/// True iff fusing `base` at `max_qubits` produces at least one `UnitaryKq`.
fn fusion_fires(base: &Circuit, max_qubits: usize) -> bool {
    let mut c = base.clone();
    FuseKq { max_qubits }.run(&mut c).unwrap();
    c.instructions()
        .iter()
        .any(|i| matches!(i, Instruction::Gate(g) if matches!(g.gate, Gate::UnitaryKq { .. })))
}

#[test]
fn qaoa_like_fused_equiv() {
    for n in [3u32, 5, 7] {
        let mut c = Circuit::new(n, 0);
        generic_prefix(&mut c, n);
        qaoa_like(&mut c, n, 3);
        assert_fused_equiv(&c, 5);
    }
}

#[test]
fn vqe_like_fused_equiv() {
    for n in [3u32, 5, 6] {
        let mut c = Circuit::new(n, 0);
        generic_prefix(&mut c, n);
        vqe_like(&mut c, n, 3);
        assert_fused_equiv(&c, 4);
    }
}

#[test]
fn random_brickwall_fused_equiv() {
    for n in [4u32, 6] {
        let mut c = Circuit::new(n, 0);
        generic_prefix(&mut c, n);
        random_brickwall(&mut c, n, 6);
        assert_fused_equiv(&c, 5);
    }
}

#[test]
fn fusion_actually_fires() {
    // Guard against a vacuous oracle: confirm the pass really produced a
    // UnitaryKq (otherwise equivalence is trivial).
    let mut c = Circuit::new(5, 0);
    generic_prefix(&mut c, 5);
    qaoa_like(&mut c, 5, 3);
    assert!(fusion_fires(&c, 5), "expected at least one UnitaryKq");
}

#[test]
fn max_qubits_variants_all_equiv() {
    // Same circuit fused at different caps must all equal the unfused result.
    for mk in [2usize, 3, 4, 5] {
        let mut c = Circuit::new(6, 0);
        generic_prefix(&mut c, 6);
        qaoa_like(&mut c, 6, 2);
        assert_fused_equiv(&c, mk);
    }
}

// ---------------------------------------------------------------------------
// Proptest: random rotation/entangler runs ≡ unfused, on a generic state.
// ---------------------------------------------------------------------------

use proptest::prelude::*;

/// One random op drawn from the fusible alphabet {Rz, Rx, Ry, Cnot, Cz}.
#[derive(Clone, Debug)]
enum RandOp {
    Rz(usize, f64),
    Rx(usize, f64),
    Ry(usize, f64),
    Cnot(usize, usize),
    Cz(usize, usize),
}

/// Strategy for `n` qubits: a single op. `Cnot`/`Cz` pick two DISTINCT qubits
/// (GateInstance debug-asserts uniqueness; a duplicate would be a test bug).
fn arb_op(n: usize) -> impl Strategy<Value = RandOp> {
    let q = 0..n;
    let ang = -PI..PI;
    // Distinct ordered pair: pick a, then an offset 1..n, wrap to avoid a==b.
    let pair = (0..n, 1..n).prop_map(move |(a, d)| (a, (a + d) % n));
    prop_oneof![
        (q.clone(), ang.clone()).prop_map(|(a, t)| RandOp::Rz(a, t)),
        (q.clone(), ang.clone()).prop_map(|(a, t)| RandOp::Rx(a, t)),
        (q, ang).prop_map(|(a, t)| RandOp::Ry(a, t)),
        pair.clone().prop_map(|(a, b)| RandOp::Cnot(a, b)),
        pair.prop_map(|(a, b)| RandOp::Cz(a, b)),
    ]
}

/// `(n, ops)`: qubit count 3..=5 paired with a 6..24-op run valid for that `n`.
/// `prop_flat_map` lets `n` drive op generation, so every generated op
/// references a qubit `< n` (no post-hoc filtering).
fn arb_n_and_run() -> impl Strategy<Value = (usize, Vec<RandOp>)> {
    (3usize..=5)
        .prop_flat_map(|n| prop::collection::vec(arb_op(n), 6..24).prop_map(move |ops| (n, ops)))
}

fn apply_randop(c: &mut Circuit, op: &RandOp) {
    match *op {
        RandOp::Rz(a, t) => c
            .add_gate(GateInstance::new(Gate::Rz(t.into()), smallvec![a as u32]))
            .map(|_| ())
            .unwrap(),
        RandOp::Rx(a, t) => c
            .add_gate(GateInstance::new(Gate::Rx(t.into()), smallvec![a as u32]))
            .map(|_| ())
            .unwrap(),
        RandOp::Ry(a, t) => c
            .add_gate(GateInstance::new(Gate::Ry(t.into()), smallvec![a as u32]))
            .map(|_| ())
            .unwrap(),
        RandOp::Cnot(a, b) => c.cnot(a as u32, b as u32).map(|_| ()).unwrap(),
        RandOp::Cz(a, b) => c.cz(a as u32, b as u32).map(|_| ()).unwrap(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random {Rz, Rx, Ry, Cnot, Cz} run after a generic entangling prefix:
    /// fusing must preserve the exact state (global phase included) through
    /// both SV backends. The prefix guarantees a non-|0…0⟩ input so
    /// cx-conjugation bugs cannot hide on the all-zero state (P1-13 lesson).
    #[test]
    fn random_run_fused_equiv((n, ops) in arb_n_and_run()) {
        let mut c = Circuit::new(n as u32, 0);
        generic_prefix(&mut c, n as u32);
        for op in &ops {
            apply_randop(&mut c, op);
        }

        let mut fused = c.clone();
        FuseKq { max_qubits: 4 }.run(&mut fused).unwrap();

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
