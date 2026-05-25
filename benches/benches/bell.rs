//! Bell pair (n=2) end-to-end benchmark — drives `NaiveSvBackend`
//! through `aleph_backend::run(&bell_circuit())`.  The bench name
//! (`bell`) is stable so bencher.dev sees a continuous timeline
//! across the P0-09 → P1 backend evolution.

use aleph_backend::run;
use aleph_benches::bell_circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_bell(c: &mut Criterion) {
    let circuit = bell_circuit();
    c.bench_function("bell", |b| {
        b.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let state = run(&mut backend, &circuit).unwrap();
                black_box(state);
            },
        );
    });
}

criterion_group!(benches, bench_bell);
criterion_main!(benches);
