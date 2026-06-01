//! Init-cost microbench: `AlignedBuf::zeroed` (lazy `alloc_zeroed`) vs
//! `zeroed_first_touch` (eager parallel first-touch) at state-vector scale.
//! First-touch pulls page-fault cost out of the first gate and into
//! allocation; this isolates that upfront cost. The NUMA *locality* win is
//! measured end-to-end on 2-node hardware (scripts/numa-bench.sh), not here.
//!
//! Run: `cargo bench -p aleph-core --features numa --bench numa_first_touch`

use aleph_core::{AlignedBuf, Complex};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn alloc_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("numa_first_touch");
    for &n in &[22u32, 25] {
        let len = 1usize << n;
        group.bench_with_input(BenchmarkId::new("zeroed", n), &len, |b, &len| {
            b.iter(|| criterion::black_box(AlignedBuf::<Complex>::zeroed(len)));
        });
        group.bench_with_input(BenchmarkId::new("first_touch", n), &len, |b, &len| {
            b.iter(|| criterion::black_box(AlignedBuf::<Complex>::zeroed_first_touch(len)));
        });
    }
    group.finish();
}

criterion_group!(benches, alloc_init);
criterion_main!(benches);
