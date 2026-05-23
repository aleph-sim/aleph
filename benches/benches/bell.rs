//! Bell pair (n=2) benchmark.
//!
//! Final state: (|00⟩ + |11⟩) / √2. Today we only construct the target
//! amplitude vector by hand — once P0-09's naive backend lands, the
//! body becomes `backend.apply_circuit(&bell_circuit())`. The fixture
//! shape (criterion bench function, parameters, output naming) stays
//! identical, so bencher.dev sees a continuous timeline.

use aleph_core::Complex;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn bell_state() -> Vec<Complex> {
    let mut amps = vec![Complex::new(0.0, 0.0); 4];
    amps[0b00] = Complex::new(SQRT_HALF, 0.0);
    amps[0b11] = Complex::new(SQRT_HALF, 0.0);
    amps
}

fn bench_bell(c: &mut Criterion) {
    c.bench_function("bell/prepare", |b| {
        b.iter(|| {
            // black_box prevents the optimiser from folding the
            // entire computation away once it sees the return value
            // is unused.
            black_box(bell_state())
        });
    });
}

criterion_group!(benches, bench_bell);
criterion_main!(benches);
