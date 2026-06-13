//! Wall-clock cost of a single non-adjacent 2q gate as a function of qubit
//! distance. Since P3-09 the lazy permutation router applies `(distance-1)`
//! nearest-neighbor SWAPs (no swap-back); compare against the pre-P3-09
//! always-swap-back baseline to see the improvement.

use aleph_backend::run;
use aleph_benches::g;
use aleph_core::Gate;
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, Criterion};

/// n-qubit circuit: H layer + NN entangling ladder, then a single CNOT(0, dist)
/// whose SWAP-network cost is being measured.
fn long_range_circuit(n: u32, dist: u32) -> aleph_ir::Circuit {
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, dist])).unwrap();
    c
}

fn bench(cr: &mut Criterion) {
    let mut grp = cr.benchmark_group("long_range_cnot_n12_chi32");
    let n = 12u32;
    for dist in [1u32, 4, 8, 11] {
        let c = long_range_circuit(n, dist);
        grp.bench_function(format!("dist{dist}"), |b| {
            b.iter(|| {
                let mut be = MpsBackend::with_seed(0).with_max_bond(32);
                run(&mut be, &c).unwrap()
            })
        });
    }
    grp.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
