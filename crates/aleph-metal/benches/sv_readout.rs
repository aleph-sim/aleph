//! P5.6-06 host-readout bench for `MetalSvBackend`: `measure`, `expectation`
//! (mixed Pauli slow path), and `probabilities` over a large state. These are
//! the multi-pass O(2^n) host scans the ticket flags; this bench backs the
//! before/after parallelisation claim.
//!
//! Each timed op runs on a fresh uniform state prepared in the (untimed) setup
//! closure — `measure` collapses in place, so it needs a fresh state per sample;
//! the others reuse the same shape for a like-for-like comparison.
//!
//! Run: `cargo bench -p aleph-metal --features metal --bench sv_readout`

use aleph_backend::Backend;
use aleph_core::{Pauli, PauliString};
use aleph_ir::Circuit;
use aleph_metal::{MetalSvBackend, MetalSvState};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Uniform superposition over `n` qubits (H on every qubit) — a cheap state
/// whose readout cost is data-independent, so it isolates the scan time.
fn uniform_state(backend: &mut MetalSvBackend, n: u32) -> MetalSvState {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    backend.run(&c).unwrap()
}

fn bench_measure(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_readout/measure");
    for &n in &[20u32, 22] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let mut be = MetalSvBackend::with_seed(0)
                        .expect("Metal device required for sv_readout bench");
                    let s = uniform_state(&mut be, n);
                    (be, s)
                },
                |(mut be, mut s)| {
                    black_box(be.measure(&mut s, 0).unwrap());
                },
            );
        });
    }
    group.finish();
}

fn bench_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_readout/expectation_xyz");
    // Mixed Pauli ⇒ the slow path (clone + transform + dot product).
    for &n in &[20u32, 22] {
        let pauli =
            PauliString::new(1.0, vec![(0u32, Pauli::X), (1, Pauli::Y), (2, Pauli::Z)]).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let mut be = MetalSvBackend::with_seed(0)
                        .expect("Metal device required for sv_readout bench");
                    let s = uniform_state(&mut be, n);
                    (be, s)
                },
                |(mut be, s)| {
                    black_box(be.expectation_value(&s, &pauli).unwrap());
                },
            );
        });
    }
    group.finish();
}

fn bench_probabilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("sv_readout/probabilities");
    for &n in &[20u32, 22] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let mut be = MetalSvBackend::with_seed(0)
                        .expect("Metal device required for sv_readout bench");
                    let s = uniform_state(&mut be, n);
                    (be, s)
                },
                |(mut be, s)| {
                    black_box(be.probabilities(&s, &[0, 1, 2]).unwrap());
                },
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_measure,
    bench_expectation,
    bench_probabilities
);
criterion_main!(benches);
