//! P6-02 oracle: distributed plan + CPU reference executor vs NaiveSvBackend.

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_ir::dist::{plan, DistLayout, DistStep, Router};
use aleph_ir::{Circuit, Instruction};
use aleph_sv::dist_ref::run_dist;
use aleph_sv::NaiveSvBackend;
use proptest::prelude::*;

const TOL: f64 = 1e-10;

fn reference(c: &Circuit) -> Vec<Complex> {
    let mut b = NaiveSvBackend::new();
    run(&mut b, c).unwrap().amplitudes().to_vec()
}

fn assert_close(a: &[Complex], b: &[Complex], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).norm() < TOL, "{what}: amp {i}: {x} vs {y}");
    }
}

fn check(c: &Circuit, g: u32, what: &str) {
    let p = plan(
        c,
        DistLayout::new(c.num_qubits(), g).unwrap(),
        Router::Naive,
    )
    .unwrap();
    assert_close(
        &run_dist(&p).unwrap(),
        &reference(c),
        &format!("{what} g={g}"),
    );
}

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let theta = std::f64::consts::PI / f64::from(1u32 << (j - k));
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                vec![j],
                vec![k],
            ))
            .unwrap();
        }
    }
    for q in 0..n / 2 {
        c.swap(q, n - 1 - q).unwrap();
    }
    c
}

fn brickwall(n: u32, depth: usize, seed: u64) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let mut s = seed;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64) * 6.283
    };
    for d in 0..depth {
        for q in 0..n {
            c.rx(rnd(), q).unwrap();
            c.rz(rnd(), q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

#[test]
fn ghz_all_layouts() {
    for g in 0..=3 {
        check(&ghz(8), g, "ghz8");
    }
}

#[test]
fn qft_all_layouts() {
    for g in 0..=3 {
        check(&qft(8), g, "qft8");
    }
}

#[test]
fn brickwall_all_layouts() {
    for g in 0..=3 {
        check(&brickwall(9, 6, 7), g, "brick9");
    }
}

#[test]
fn rz_on_top_qubit_is_rank_phase_not_global() {
    let mut c = Circuit::new(5, 0);
    for q in 0..5 {
        c.h(q).unwrap();
    }
    c.rz(0.9, 4).unwrap();
    c.cz(3, 4).unwrap();
    check(&c, 2, "rz-top");
}

#[test]
fn trailing_relabel_swap() {
    let mut c = ghz(6);
    c.swap(0, 5).unwrap();
    let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Naive).unwrap();
    assert_eq!(p.stats.relabels, 1);
    assert_close(&run_dist(&p).unwrap(), &reference(&c), "trailing swap");
}

#[test]
fn toffoli_and_controlled_with_global_controls() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.h(q).unwrap();
    }
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![5, 4, 0]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![5, 1, 2]))
        .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::H,
        vec![1u32],
        vec![5u32, 0u32],
    ))
    .unwrap();
    c.add_gate(GateInstance::new(Gate::Ccz, vec![4, 5, 0]))
        .unwrap();
    check(&c, 2, "toffoli");
}

#[test]
fn mutation_dropping_an_exchange_breaks_oracle() {
    let c = qft(6);
    let mut p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Naive).unwrap();
    // The *last* exchange: the first one acts on |0…0⟩, where swapping two
    // |0⟩ qubits is the identity, so dropping it would be a no-op mutation.
    let idx = p
        .steps
        .iter()
        .rposition(|s| matches!(s, DistStep::Exchange { .. }))
        .expect("qft6 on g=2 needs an exchange");
    p.steps.remove(idx);
    let bad = match run_dist(&p) {
        Err(_) => true,
        Ok(v) => v
            .iter()
            .zip(reference(&c))
            .any(|(a, b)| (a - b).norm() > 1e-6),
    };
    assert!(bad, "removing an exchange must be detected");
}

/// Strip non-unitary instructions so arbitrary generated circuits are plannable.
fn unitary_only(c: &Circuit) -> Circuit {
    let mut out = Circuit::new(c.num_qubits(), 0);
    for i in c.instructions() {
        if let Instruction::Gate(g) = i {
            out.add_gate(g.clone()).unwrap();
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn prop_random_circuits_match(
        c in aleph_test::circuit::arb_circuit_full(6, 2, 40),
        g in 0u32..=3,
    ) {
        let c = unitary_only(&c);
        let p = plan(&c, DistLayout::new(6, g).unwrap(), Router::Naive).unwrap();
        let got = run_dist(&p).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}
