//! Prints P6-02/P6-03 communication counts (naive vs lookahead router).
//! Run from the workspace root:
//! `cargo run --release -p aleph-sv --example dist_comm_counts`

use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::dist::{plan, DistLayout, Router};
use aleph_ir::{Circuit, Instruction};

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let th = std::f64::consts::PI / 2f64.powi((j - k) as i32);
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(th)),
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

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rx(0.3 + f64::from(q), q).unwrap();
            c.rz(0.7 * d as f64, q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

fn grover() -> Circuit {
    let src = std::fs::read_to_string("scripts/qiskit-baseline/circuits/grover_n20_iters5.qasm")
        .expect("run from the workspace root");
    let parsed = aleph_parser::parse(&src).expect("parse grover");
    let mut out = Circuit::new(parsed.num_qubits(), 0);
    for i in parsed.instructions() {
        if let Instruction::Gate(g) = i {
            out.add_gate(g.clone()).unwrap();
        }
    }
    out
}

fn main() {
    let cases: Vec<(&str, Circuit)> = vec![
        ("GHZ-32", ghz(32)),
        ("QFT-32", qft(32)),
        ("random-30 d=20", brickwall(30, 20)),
        ("Grover-20 (5 iters)", grover()),
    ];
    println!(
        "| circuit | g | naive exch | naive × slice | lookahead exch | lookahead × slice | reduction | la local swaps |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for (name, c) in &cases {
        for g in [2u32, 3] {
            let l = DistLayout::new(c.num_qubits(), g).unwrap();
            let slice = (1u64 << l.m()) as f64;
            let n = plan(c, l, Router::Naive).unwrap().stats;
            let a = plan(c, l, Router::Lookahead).unwrap().stats;
            let ns = n.amps_moved_per_rank as f64 / slice;
            let la = a.amps_moved_per_rank as f64 / slice;
            println!(
                "| {name} | {g} | {} | {ns:.1} | {} | {la:.1} | {:.2}× | {} |",
                n.exchanges,
                a.exchanges,
                ns / la.max(f64::MIN_POSITIVE),
                a.local_swaps
            );
        }
    }
}
