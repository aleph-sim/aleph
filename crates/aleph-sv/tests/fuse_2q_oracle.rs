//! P1-10 oracle — fused circuit ≡ unfused circuit on `NaiveSvBackend`,
//! state-vector entries within 1e-12. Complements aleph-ir's structural
//! property tests by checking the actual unitary.
//!
//! `arb_circuit_emittable`-based random coverage already runs through
//! `fuse_1q_oracle.rs` (which calls `optimize()` — now both passes). This
//! file pins the specific 2q-fusion shapes with hand-built circuits.

use aleph_backend::run;
use aleph_core::Complex;
use aleph_ir::passes::{Fuse2q, PassPipeline};
use aleph_ir::Circuit;
use aleph_sv::NaiveSvBackend;

const TOL: f64 = 1e-12;

fn run_to_state(c: &Circuit) -> Vec<Complex> {
    let mut backend = NaiveSvBackend::with_seed(0);
    run(&mut backend, c)
        .expect("naive backend executes gate-only IR")
        .amplitudes()
        .to_vec()
}

/// Run only `Fuse2q` (no `Fuse1qRuns` pre-pass) so we can verify the pass is
/// correct standalone, in particular when pre-1q gates on one qubit have an
/// earlier index than intervening ops on the other qubit.
fn run_fuse2q_only(c: &mut Circuit) {
    PassPipeline::new(vec![Box::new(Fuse2q)])
        .run(c)
        .expect("Fuse2q is infallible");
}

fn assert_fusion_preserves_state(build: impl Fn(&mut Circuit)) {
    let mut unfused = Circuit::new(3, 0);
    build(&mut unfused);
    let mut fused = unfused.clone();
    fused
        .optimize()
        .expect("optimize cannot fail on validated circuit");

    let su = run_to_state(&unfused);
    let sf = run_to_state(&fused);
    assert_eq!(su.len(), sf.len());
    for (i, (a, b)) in su.iter().zip(sf.iter()).enumerate() {
        let diff = (*a - *b).norm();
        assert!(
            diff < TOL,
            "amp[{i}] diff {diff} >= {TOL} (unfused={a:?}, fused={b:?})"
        );
    }
}

#[test]
fn pre_1q_absorption_preserves_state() {
    assert_fusion_preserves_state(|c| {
        c.rx(0.7, 0).unwrap();
        c.ry(0.3, 1).unwrap();
        c.cnot(0, 1).unwrap();
    });
}

#[test]
fn post_1q_absorption_preserves_state() {
    assert_fusion_preserves_state(|c| {
        c.cnot(0, 1).unwrap();
        c.rz(0.9, 1).unwrap();
        c.h(0).unwrap();
    });
}

#[test]
fn same_pair_merge_preserves_state() {
    assert_fusion_preserves_state(|c| {
        c.cnot(0, 1).unwrap();
        c.rz(0.4, 1).unwrap();
        c.cnot(0, 1).unwrap();
    });
}

#[test]
fn reversed_operand_merge_preserves_state() {
    assert_fusion_preserves_state(|c| {
        c.h(0).unwrap();
        c.h(1).unwrap();
        c.cnot(0, 1).unwrap();
        c.cz(1, 0).unwrap();
    });
}

#[test]
fn mixed_chain_preserves_state() {
    // pre + post + same-pair + a disjoint qubit + a fence.
    assert_fusion_preserves_state(|c| {
        c.h(0).unwrap();
        c.h(1).unwrap();
        c.h(2).unwrap();
        c.cnot(0, 1).unwrap();
        c.rz(0.5, 1).unwrap();
        c.cnot(0, 1).unwrap();
        c.rx(0.6, 2).unwrap();
        c.cnot(1, 2).unwrap();
        c.ry(0.2, 0).unwrap();
    });
}

/// Verify that `Fuse2q` alone (without `Fuse1qRuns`) produces the correct state
/// when a pre-1q gate on one qubit has a smaller instruction index than an
/// intervening op on the OTHER qubit of the future 2q gate.
///
/// Circuit (3 qubits, unitary-only):
///   Ry(0.5, 1) @ 0   ← pre-1q candidate for CNOT(0,1); its index is 0
///   H(0)       @ 1   ← on qubit 0 (also a CNOT qubit); index 1
///   Cz(0, 2)   @ 2   ← fence: intervening op on qubit 0; index 2
///   CNOT(0, 1) @ 3   ← 2q gate
///   Rx(0.3, 1) @ 4   ← post-1q on qubit 1
///
/// Before the first_index-min bug fix, Fuse2q would key the fused block at
/// min(0, 3) = 0, sorting it before H(0) and Cz(0,2), reordering ops that
/// share qubit 0 with the CNOT — wrong. After the fix it keys at 3, and the
/// state vector matches the unfused circuit within 1e-12.
///
/// Note: Cz(0,2) forces Ry(1) to be flushed as a standalone 1q before the
/// CNOT opens its block, so the CNOT block has len=1 here and no fusion
/// occurs. The test still exercises the standalone correctness of the pass.
#[test]
fn fuse2q_alone_preserves_state_across_fence() {
    let mut unfused = Circuit::new(3, 0);
    unfused.ry(0.5, 1).unwrap(); // @ 0: pre-1q on q1
    unfused.h(0).unwrap(); // @ 1: q0
    unfused.cz(0, 2).unwrap(); // @ 2: fence — uses q0, so flushes any open q0 block
    unfused.cnot(0, 1).unwrap(); // @ 3: 2q gate on (q0, q1)
    unfused.rx(0.3, 1).unwrap(); // @ 4: post-1q on q1

    let mut fused = unfused.clone();
    run_fuse2q_only(&mut fused);

    let su = run_to_state(&unfused);
    let sf = run_to_state(&fused);
    assert_eq!(su.len(), sf.len());
    for (i, (a, b)) in su.iter().zip(sf.iter()).enumerate() {
        let diff = (*a - *b).norm();
        assert!(
            diff < TOL,
            "amp[{i}] diff {diff} >= {TOL} (unfused={a:?}, fused={b:?})"
        );
    }
}
