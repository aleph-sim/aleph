//! P3-12: wall-clock win of routing user-level `Gate::Swap`s through the
//! lazy-permutation relabel instead of physical tensor work.
//!
//! Self-contained A/B on one logically identical permutation, applied to an
//! entangled register at moderate χ. `relabel` uses `Gate::Swap` (an O(1) map
//! update); `cnot_decomposed` writes the same SWAP as 3 CNOTs (what a user who
//! cannot relabel would pay), each running a truncated SVD. The relabel path
//! applies zero physical SWAPs from the swaps themselves (`relabels` rises,
//! `swaps_applied` only reflects CNOT routing); the decomposed path drags
//! tensors through every gate.

use aleph_backend::run;
use aleph_benches::g;
use aleph_core::Gate;
use aleph_mps::{MpsBackend, MpsState};
use criterion::{criterion_group, criterion_main, Criterion};

/// (a, b) pairs of a SWAP network: a register reversal via long-range swaps,
/// the routing-aware-compiler-output workload the ticket targets.
fn swap_pairs(n: u32) -> Vec<(u32, u32)> {
    (0..n / 2).map(|i| (i, n - 1 - i)).collect()
}

/// H layer + NN entangling ladder to grow the bond, then the SWAP network.
/// `physical` expands each SWAP into 3 CNOTs instead of one `Gate::Swap`.
fn swap_circuit(n: u32, physical: bool) -> aleph_ir::Circuit {
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    for (a, b) in swap_pairs(n) {
        if physical {
            // SWAP(a,b) = CNOT(a,b) · CNOT(b,a) · CNOT(a,b).
            c.add_gate(g(Gate::Cnot, &[a, b])).unwrap();
            c.add_gate(g(Gate::Cnot, &[b, a])).unwrap();
            c.add_gate(g(Gate::Cnot, &[a, b])).unwrap();
        } else {
            c.add_gate(g(Gate::Swap, &[a, b])).unwrap();
        }
    }
    c
}

fn bench(cr: &mut Criterion) {
    let n = 14u32;
    let chi = 32usize;
    let mut grp = cr.benchmark_group(format!("swap_dense_n{n}_chi{chi}"));

    let relabel = swap_circuit(n, false);
    let decomposed = swap_circuit(n, true);

    // Sanity: the relabel circuit really exercises the relabel path.
    let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
    let st: MpsState = run(&mut be, &relabel).unwrap();
    assert_eq!(st.relabels(), (n / 2) as u64);

    grp.bench_function("relabel", |b| {
        b.iter(|| {
            let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
            run::<MpsBackend>(&mut be, &relabel).unwrap()
        })
    });
    grp.bench_function("cnot_decomposed", |b| {
        b.iter(|| {
            let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
            run::<MpsBackend>(&mut be, &decomposed).unwrap()
        })
    });
    grp.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
