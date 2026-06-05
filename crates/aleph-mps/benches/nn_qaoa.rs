//! Wall-clock scaling of an MPS nearest-neighbor QAOA depth-3 circuit.
//! No ratio gate — P3-04 has no perf AC; this records the scaling curve.

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_mps::{MpsBackend, TruncationPolicy};
use criterion::{criterion_group, criterion_main, Criterion};

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}

fn qaoa_circuit(n: u32) -> aleph_ir::Circuit {
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for _ in 0..3 {
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(Gate::Rz(Param::Concrete(0.7)), &[q + 1]))
                .unwrap();
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
        }
        for q in 0..n {
            c.add_gate(g(Gate::Rx(Param::Concrete(0.5)), &[q])).unwrap();
        }
    }
    c
}

fn bench(cr: &mut Criterion) {
    let mut grp = cr.benchmark_group("nn_qaoa_chi64");
    for n in [10u32, 20, 30] {
        let c = qaoa_circuit(n);
        grp.bench_function(format!("n{n}"), |b| {
            b.iter(|| {
                let mut be = MpsBackend::with_seed(0).with_max_bond(64);
                run(&mut be, &c).unwrap()
            })
        });
    }
    grp.finish();
}

fn bench_policies(cr: &mut Criterion) {
    let mut grp = cr.benchmark_group("nn_qaoa_n20_policy");
    let c = qaoa_circuit(20);
    grp.bench_function("fixed_chi64", |b| {
        b.iter(|| {
            let mut be = MpsBackend::with_seed(0).with_max_bond(64);
            run(&mut be, &c).unwrap()
        })
    });
    grp.bench_function("error_1e-8", |b| {
        b.iter(|| {
            let mut be = MpsBackend::with_seed(0).with_truncation(TruncationPolicy::ErrorBounded {
                epsilon: 1e-8,
                max_bond: 64,
            });
            run(&mut be, &c).unwrap()
        })
    });
    grp.finish();
}

criterion_group!(benches, bench, bench_policies);
criterion_main!(benches);
