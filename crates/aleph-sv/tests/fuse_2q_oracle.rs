//! P1-10 oracle — fused circuit ≡ unfused circuit on `NaiveSvBackend`,
//! state-vector entries within 1e-12. Complements aleph-ir's structural
//! property tests by checking the actual unitary.
//!
//! `arb_circuit_emittable`-based random coverage already runs through
//! `fuse_1q_oracle.rs` (which calls `optimize()` — now both passes). This
//! file pins the specific 2q-fusion shapes with hand-built circuits.

use aleph_backend::run;
use aleph_core::Complex;
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
