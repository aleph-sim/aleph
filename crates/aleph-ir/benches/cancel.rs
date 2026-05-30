//! P1-12 benchmark: cost of `CancelInversePairs` on a redundancy-heavy
//! circuit, plus a printed gate-count reduction (the acceptance-criterion
//! figure).
//!
//! Run with:
//! `cargo bench -p aleph-ir --features bench-fixtures --bench cancel`

use aleph_ir::bench_fixtures::cancel_redundant;
use aleph_ir::passes::{CancelInversePairs, PassPipeline};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_cancel(c: &mut Criterion) {
    // Report the AC figure once, outside the timing loop.
    let probe = cancel_redundant(8, 200);
    let before = probe.len();
    let mut reduced = probe.clone();
    PassPipeline::new(vec![Box::new(CancelInversePairs)])
        .run(&mut reduced)
        .unwrap();
    eprintln!(
        "cancel_redundant(8,200): {} → {} ({:.2}× reduction)",
        before,
        reduced.len(),
        before as f64 / reduced.len() as f64
    );

    let mut group = c.benchmark_group("cancel");
    group.bench_function("cancel_redundant_n8_pairs200", |bch| {
        bch.iter_batched(
            || cancel_redundant(8, 200),
            |mut circ| {
                PassPipeline::new(vec![Box::new(CancelInversePairs)])
                    .run(&mut circ)
                    .unwrap();
                circ
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_cancel);
criterion_main!(benches);
