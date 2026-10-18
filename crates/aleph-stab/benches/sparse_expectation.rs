use aleph_backend::{run, Backend};
use aleph_mps::MpsBackend;
use aleph_stab::{SparsePauliOptions, SparsePauliQuery};
use aleph_sv::NaiveSvBackend;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[path = "../examples/support/sparse_workloads.rs"]
mod workloads;
fn bench(c: &mut Criterion) {
    // Campaign-only extensive sweep lives in the example; keep ordinary CI benches bounded.
    for name in workloads::names().into_iter().filter(|n| {
        matches!(
            n.as_str(),
            "ghz18-rotation" | "product16-k4" | "product16-k12" | "local128"
        )
    }) {
        let (circuit, p, _) = workloads::fixture(&name).unwrap();
        for cones in [false, true] {
            c.bench_function(&format!("pauli/{name}/sparse-cones-{cones}"), |b| {
                b.iter(|| {
                    black_box(
                        SparsePauliQuery::compile(
                            &circuit,
                            &p,
                            SparsePauliOptions {
                                use_cones: cones,
                                ..Default::default()
                            },
                        )
                        .unwrap()
                        .execute()
                        .unwrap(),
                    )
                })
            });
        }
        if circuit.num_qubits() <= 20 {
            c.bench_function(&format!("pauli/{name}/sv"), |b| {
                b.iter(|| {
                    let mut be = NaiveSvBackend::with_seed(0);
                    let s = run(&mut be, &circuit).unwrap();
                    black_box(be.expectation_value(&s, &p.terms[0]).unwrap())
                })
            });
        }
        c.bench_function(&format!("pauli/{name}/mps"), |b| {
            b.iter(|| {
                let mut be = MpsBackend::with_seed(0).with_max_bond(128);
                let s = run(&mut be, &circuit).unwrap();
                assert!(s.truncation_error() < 1e-12);
                black_box(be.expectation_value(&s, &p.terms[0]).unwrap())
            })
        });
    }
}
fn config() -> Criterion {
    let c = Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(1));
    match std::env::var_os("ALEPH_BENCH_OUTPUT") {
        Some(p) => c.output_directory(&std::path::PathBuf::from(p)),
        None => c,
    }
}
criterion_group! {name=benches;config=config();targets=bench}
criterion_main!(benches);
