//! Shared circuits + CPU reference for the distributed-SV GPU tests.
#![allow(dead_code)] // each test binary uses a different subset

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use std::time::Instant;

use aleph_cuda::{CudaContext, GpuCostModel};
use aleph_ir::build_qaoa;
use aleph_ir::dist::{DistPlan, DistStep};
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

/// Grover on `n` qubits, the construction of `gpu_report_bench.rs`: H layer,
/// then `iters` × (oracle multi-controlled Z, diffusion H·X·MCZ·X·H). The MCZ
/// targets `n − 1` with controls `0..min(n−1, 8)` (the IR caps controls at 8).
/// Cost-model gates use a small fixed `iters` (e.g. 3): they measure model
/// accuracy per launch, not the algorithm, and π/4·√2^n iterations would run
/// for hours at n=28.
pub fn grover_iters(n: u32, iters: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let mcz = |c: &mut Circuit| {
        let ctrls: Vec<u32> = (0..(n - 1).min(8)).collect();
        c.add_gate(GateInstance::controlled(Gate::Z, vec![n - 1], ctrls))
            .unwrap();
    };
    for _ in 0..iters {
        mcz(&mut c);
        for q in 0..n {
            c.h(q).unwrap();
            c.x(q).unwrap();
        }
        mcz(&mut c);
        for q in 0..n {
            c.x(q).unwrap();
            c.h(q).unwrap();
        }
    }
    c
}

/// The plan with every Local step emptied: exchanges (and allocation) only.
pub fn comm_only(p: &DistPlan) -> DistPlan {
    let mut q = p.clone();
    for s in &mut q.steps {
        if let DistStep::Local(v) = s {
            v.clear();
        }
    }
    q
}

pub fn best_of(sync: &CudaContext, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        sync.synchronize().unwrap();
        let t = Instant::now();
        f();
        sync.synchronize().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// QAOA Max-Cut p=2 on a ring plus chords: the ring `(i, i+1 mod n)` plus one
/// chord `(i, i + n/2)` for each even `i < n/2` (7 chords at n=28; not regular).
pub fn qaoa_ring_chords(n: u32) -> Circuit {
    let mut edges: Vec<(u32, u32)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    edges.extend((0..n / 2).step_by(2).map(|i| (i, i + n / 2)));
    build_qaoa(n, &edges, &[0.4, 0.7], &[0.3, 0.5]).unwrap()
}

/// All-ranks model compute with the state-class walk (`GpuCostModel::all_ranks`).
pub fn all_ranks(model: &GpuCostModel, p: &DistPlan) -> f64 {
    model.all_ranks(p).unwrap()
}

/// `depth` Clifford layers: `H` on even qubits and `S` on odd ones, then
/// nearest-neighbour `CNOT`s starting at qubit `d % 2` (#538 state (f) and the
/// held-out Clifford brickwall).
pub fn clifford_layers(c: &mut Circuit, n: u32, depth: usize) {
    for d in 0..depth {
        for q in 0..n {
            if q % 2 == 0 {
                c.h(q).unwrap();
            } else {
                c.s(q).unwrap();
            }
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
}

pub fn clifford_brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    clifford_layers(&mut c, n, depth);
    c
}

/// #538 held-out HEA: `build_hea(n, 4, params)` with `params[i] = 0.1 + 0.07·i`.
pub fn hea_bench(n: u32) -> Circuit {
    let depth = 4;
    let params: Vec<f64> = (0..n as usize * (depth as usize + 1))
        .map(|i| 0.1 + 0.07 * i as f64)
        .collect();
    aleph_ir::build_hea(n, depth, &params).unwrap()
}

/// #538 held-out QAOA p=2 on a 3-regular-like graph: the ring `(i, i+1 mod n)`
/// plus `(i, i+7 mod n)` for every `i`; γ = [0.4, 0.7], β = [0.3, 0.5].
pub fn qaoa_ring_skip7(n: u32) -> Circuit {
    let mut edges: Vec<(u32, u32)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    edges.extend((0..n).map(|i| (i, (i + 7) % n)));
    build_qaoa(n, &edges, &[0.4, 0.7], &[0.3, 0.5]).unwrap()
}
