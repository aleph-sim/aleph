//! P1-09 fusion bench — measures gate-count reduction and the
//! optimisation pass wall-clock on the VQE HEA fixture.

use aleph_ir::bench_fixtures::vqe_hea;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_fuse_1q(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuse_1q");

    // Print the reduction ratio at startup so CI logs surface it even
    // without running the full bench timings.
    let probe = vqe_hea(12, 10);
    let before = probe.len();
    let mut tmp = probe;
    tmp.optimize().unwrap();
    let after = tmp.len();
    eprintln!(
        "vqe_hea(12,10): {} → {} instructions ({:.2}× reduction)",
        before,
        after,
        before as f64 / after as f64
    );

    group.bench_function("optimize_vqe_hea_n12_d10", |b| {
        b.iter_batched(
            || vqe_hea(12, 10),
            |mut circuit| {
                circuit.optimize().expect("optimize cannot fail");
                black_box(circuit);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_fuse_1q);
criterion_main!(benches);
