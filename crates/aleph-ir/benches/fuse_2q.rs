//! P1-10 benchmark: cost of running the default pipeline (Fuse1qRuns +
//! Fuse2q) on a QAOA circuit, plus a printed gate-count reduction beyond
//! P1-09 (the acceptance-criterion figure).
//!
//! Run with: `cargo bench -p aleph-ir --features bench-fixtures --bench fuse_2q`

use aleph_ir::bench_fixtures::qaoa;
use aleph_ir::passes::{Fuse1qRuns, Fuse2q, PassPipeline};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_fuse_2q(c: &mut Criterion) {
    // Report the AC figure once, outside the timing loop.
    let probe = qaoa(12, 10);
    let mut a = probe.clone();
    PassPipeline::new(vec![Box::new(Fuse1qRuns)])
        .run(&mut a)
        .unwrap();
    let mut b = probe.clone();
    PassPipeline::new(vec![Box::new(Fuse1qRuns), Box::new(Fuse2q)])
        .run(&mut b)
        .unwrap();
    eprintln!(
        "qaoa(12,10): after P1-09 = {} → after P1-10 = {} ({:.2}× reduction beyond P1-09)",
        a.len(),
        b.len(),
        a.len() as f64 / b.len() as f64
    );

    let mut group = c.benchmark_group("fuse_2q");
    group.bench_function("optimize_qaoa_n12_d10", |bch| {
        bch.iter_batched(
            || qaoa(12, 10),
            |mut circ| {
                PassPipeline::default_pipeline().run(&mut circ).unwrap();
                circ
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_fuse_2q);
criterion_main!(benches);
