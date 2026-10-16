//! End-to-end same-FP64/query comparisons, plus separate planner/reuse tiers.
use aleph_backend::{run, Backend};
use aleph_sv::{expectation_query::ExactExpectationQuery, NaiveSvBackend};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[path = "../examples/support/cone_workloads.rs"]
mod workloads;
fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("expectation_query");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    for (name, circuit, observable) in [
        (
            "local18",
            workloads::brickwork(18, 3, 0),
            workloads::observable(9),
        ),
        ("global16", workloads::global(16), workloads::observable(15)),
    ] {
        let plan = ExactExpectationQuery::compile(&circuit, &observable, 64 << 20).unwrap();
        let result = plan.execute().unwrap();
        let mut be = NaiveSvBackend::with_seed(0);
        let state = run(&mut be, &circuit).unwrap();
        let expected = be.expectation_value(&state, &observable.terms[0]).unwrap();
        assert!((result.value - expected).abs() < 1e-10);
        drop(state);
        group.bench_function(format!("{name}/full"), |b| {
            b.iter(|| {
                let mut be = NaiveSvBackend::with_seed(0);
                let state = run(&mut be, black_box(&circuit)).unwrap();
                black_box(be.expectation_value(&state, &observable.terms[0]).unwrap())
            })
        });
        group.bench_function(format!("{name}/query"), |b| {
            b.iter(|| {
                let plan =
                    ExactExpectationQuery::compile(black_box(&circuit), &observable, 64 << 20)
                        .unwrap();
                black_box(plan.execute().unwrap())
            })
        });
        group.bench_function(format!("{name}/plan"), |b| {
            b.iter(|| {
                black_box(
                    ExactExpectationQuery::compile(black_box(&circuit), &observable, 64 << 20)
                        .unwrap(),
                )
            })
        });
        group.bench_function(format!("{name}/reuse"), |b| {
            b.iter(|| black_box(plan.execute().unwrap()))
        });
    }
    group.finish();
}
criterion_group!(query, benches);
criterion_main!(query);
