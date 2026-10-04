//! Shared circuits + CPU reference for the distributed-SV GPU tests.
#![allow(dead_code)] // each test binary uses a different subset

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_ir::{Circuit, Instruction};
use aleph_sv::NaiveSvBackend;

pub fn reference(c: &Circuit) -> Vec<Complex<f64>> {
    let mut b = NaiveSvBackend::with_seed(0);
    run(&mut b, c).unwrap().amplitudes().to_vec()
}

pub fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

pub fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let th = std::f64::consts::PI / f64::from(1u32 << (j - k));
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

pub fn brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rx(0.3 + 0.17 * f64::from(q), q).unwrap();
            c.rz(0.7 * d as f64 + 0.05 * f64::from(q), q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

pub fn grover8() -> Circuit {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/qiskit-baseline/circuits/grover_n8_iters13.qasm"
    ))
    .unwrap();
    let parsed = aleph_parser::parse(&src).unwrap();
    let mut c = Circuit::new(parsed.num_qubits(), 0);
    for i in parsed.instructions() {
        if let Instruction::Gate(g) = i {
            c.add_gate(g.clone()).unwrap();
        }
    }
    c
}

pub fn all_diag_on_globals(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let top = n - 1;
    c.rz(0.9, top).unwrap();
    c.add_gate(GateInstance::new(Gate::Cz, vec![top - 1, top]))
        .unwrap();
    c.add_gate(GateInstance::new(
        Gate::CRz(Param::Concrete(1.1)),
        vec![0, top],
    ))
    .unwrap();
    c.add_gate(GateInstance::new(Gate::Ccz, vec![top - 1, 1, top]))
        .unwrap();
    c.add_gate(GateInstance::controlled(Gate::T, vec![2u32], vec![top]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![top, top - 1, 0]))
        .unwrap();
    // Externally controlled diagonals touching globals: at g >= 1 the first
    // specialises to a `Unitary1qDiag` on q1 *with* local control q0; at g >= 2
    // the second leaves only a scalar phase gated on local control q0.
    c.add_gate(GateInstance::controlled(Gate::Cz, vec![1, top], vec![0]))
        .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::Cz,
        vec![top - 1, top],
        vec![0],
    ))
    .unwrap();
    c
}

pub fn cases() -> Vec<(&'static str, Circuit)> {
    vec![
        ("ghz10", ghz(10)),
        ("qft10", qft(10)),
        ("brick10", brickwall(10, 6)),
        ("grover8", grover8()),
        ("diag10", all_diag_on_globals(10)),
    ]
}

/// `brickwall` without the per-qubit `Rz` offset: the exact gate content of
/// the P6-01a overhead bench (`dist_local_bench`), which P6-05 compares on.
pub fn brickwall_bench(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rx(0.3 + 0.17 * f64::from(q), q).unwrap();
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

/// All-diagonal ladder whose `Ccz`s each touch the top qubit, so at g = 1
/// every one specialises to a 2-local diagonal (#529).
pub fn ccz_ladder(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for d in 0..depth {
        for q in 0..n - 2 {
            c.add_gate(GateInstance::new(Gate::Ccz, vec![q, q + 1, n - 1]))
                .unwrap();
            c.add_gate(GateInstance::new(Gate::Cz, vec![q, q + 1]))
                .unwrap();
            c.rz(0.1 * (d as f64 + 1.0), q).unwrap();
        }
    }
    c
}
