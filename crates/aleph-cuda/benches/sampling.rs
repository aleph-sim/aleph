//! Sampling-only Criterion A/B. Gates/JIT are outside the measured closure.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn bench(c: &mut criterion::Criterion) {
    use aleph_backend::{run, Backend};
    use aleph_cuda::{CudaSvBackend, CudaSvBackendF32, SamplingAlgorithm};
    let mut group = c.benchmark_group("cuda_sampling");
    group
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(1));
    for (n, shots) in [(18, 1024), (22, 1024), (22, 65536)] {
        let mut circuit = aleph_ir::Circuit::new(n, 0);
        for q in 0..n {
            circuit.h(q).unwrap();
        }
        for mode in [SamplingAlgorithm::Reference, SamplingAlgorithm::Auto] {
            macro_rules! measure {
                ($kind:ident,$precision:literal) => {{
                    let mut be = $kind::with_seed(555).unwrap().with_sampling_algorithm(mode);
                    let state = run(&mut be, &circuit).unwrap();
                    be.sample(&state, shots).unwrap();
                    group.bench_function(format!("{}_{mode:?}_n{n}_s{shots}", $precision), |b| {
                        b.iter(|| {
                            criterion::black_box(
                                be.sample(criterion::black_box(&state), shots).unwrap(),
                            )
                        });
                    });
                }};
            }
            measure!(CudaSvBackend, "f64");
            measure!(CudaSvBackendF32, "f32");
        }
    }
    group.finish();
}
#[cfg(all(target_os = "linux", feature = "cuda"))]
criterion::criterion_group!(benches, bench);
#[cfg(all(target_os = "linux", feature = "cuda"))]
criterion::criterion_main!(benches);
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {}
