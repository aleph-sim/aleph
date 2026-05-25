//! Micro-bench for `NaiveSvBackend::expectation_value`. Quantifies
//! the Z-only fast path (P0-11) vs the copy-and-apply baseline
//! (P0-09). Save a baseline from `main` with `--save-baseline
//! pre-zfast` before the P0-11 merge and compare with `--baseline
//! pre-zfast` after.
//!
//! State: Hadamard wall on every qubit (|+⟩⊗ⁿ). Neither path
//! short-circuits on this state.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance, Pauli, PauliString};
use aleph_sv::{CpuState, NaiveSvBackend};
use criterion::{criterion_group, criterion_main, Criterion};
use smallvec::smallvec;

fn hadamard_wall(n: u32) -> CpuState {
    let mut b = NaiveSvBackend::with_seed(0);
    let mut s = b.allocate(n).unwrap();
    for q in 0..n {
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q]))
            .unwrap();
    }
    s
}

fn bench_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expectation");
    const N: u32 = 10;

    let z_chain = PauliString::new(1.0, (0..N).map(|q| (q, Pauli::Z)).collect()).unwrap();
    let x_chain = PauliString::new(1.0, (0..N).map(|q| (q, Pauli::X)).collect()).unwrap();
    let mixed_zx = PauliString::new(
        1.0,
        (0..N)
            .map(|q| (q, if q % 2 == 0 { Pauli::Z } else { Pauli::X }))
            .collect(),
    )
    .unwrap();

    let state = hadamard_wall(N);

    group.bench_function("exp_z_chain_n10", |bencher| {
        bencher.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let v = backend.expectation_value(&state, &z_chain).unwrap();
                criterion::black_box(v);
            },
        );
    });
    group.bench_function("exp_x_chain_n10", |bencher| {
        bencher.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let v = backend.expectation_value(&state, &x_chain).unwrap();
                criterion::black_box(v);
            },
        );
    });
    group.bench_function("exp_mixed_zx_n10", |bencher| {
        bencher.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let v = backend.expectation_value(&state, &mixed_zx).unwrap();
                criterion::black_box(v);
            },
        );
    });

    group.finish();
}

criterion_group!(benches, bench_expectation);
criterion_main!(benches);
