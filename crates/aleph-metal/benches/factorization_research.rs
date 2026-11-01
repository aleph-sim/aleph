//! P8-11 Criterion: pooled buffers, actual pack+factor command completion.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(target_os = "macos")]
fn bench(c: &mut Criterion) {
    use aleph_metal::factorization_research::Case;
    let mut group = c.benchmark_group("p8_11_pack_factor");
    group
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(2));
    for chi in [32usize, 64] {
        let mut case = Case::new(2 * chi, 2 * chi, "well").expect("Metal fixture");
        for candidate in [false, true] {
            group.bench_function(
                BenchmarkId::new(if candidate { "rounds" } else { "single_group" }, chi),
                |b| {
                    b.iter(|| {
                        case.bench_factor(candidate)
                            .expect("actual Metal execution")
                    })
                },
            );
        }
    }
    group.finish();
}
#[cfg(not(target_os = "macos"))]
fn bench(_: &mut Criterion) {
    panic!("requires Metal hardware");
}
criterion_group!(benches, bench);
criterion_main!(benches);
