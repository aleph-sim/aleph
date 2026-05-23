//! GHZ-state preparation benchmark for n ∈ {10, 15, 20, 25}.
//!
//! GHZ on n qubits = (|0…0⟩ + |1…1⟩) / √2. Two non-zero amplitudes,
//! `1/√2` each at index 0 and index `2^n − 1`. Today this measures the
//! cost of allocating + initialising a `Vec<Complex>` of length `2^n`;
//! once P0-09 lands the body becomes `backend.apply_circuit(&ghz(n))`
//! and the bench naturally tracks circuit-execution time instead.

use aleph_benches::zero_state;
use aleph_core::Complex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

const QUBIT_COUNTS: &[u32] = &[10, 15, 20, 25];

fn ghz_state(n_qubits: u32) -> Vec<Complex> {
    let mut amps = zero_state(n_qubits);
    // |0…0⟩ amplitude already set to 1 by zero_state; renormalise to 1/√2
    // and add |1…1⟩ at the highest index.
    amps[0] = Complex::new(SQRT_HALF, 0.0);
    let last = amps.len() - 1;
    amps[last] = Complex::new(SQRT_HALF, 0.0);
    amps
}

fn bench_ghz(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz/prepare");
    for &n in QUBIT_COUNTS {
        // Throughput in amplitudes — bencher.dev plots this as elements/s
        // once we have real backend execution, so it stays meaningful.
        group.throughput(Throughput::Elements(1u64 << n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(ghz_state(n)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ghz);
criterion_main!(benches);
