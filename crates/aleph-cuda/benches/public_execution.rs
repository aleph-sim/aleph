//! Criterion execution A/B for the same reusable plan used by public frontends.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn bench(c: &mut criterion::Criterion) {
    use aleph_cuda::{
        CudaExecutionOptions as Options, CudaExecutionPlan as Plan, CudaSvBackend, CudaSvBackendF32,
    };
    let mut group = c.benchmark_group("cuda_public_plan");
    group
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(1));
    for family in ["brickwall", "qft"] {
        let mut circuit = aleph_ir::Circuit::new(22, 0);
        match family {
            "qft" => {
                for q in 0..22 {
                    circuit.h(q).unwrap();
                    for j in q + 1..22 {
                        circuit
                            .add_gate(aleph_core::GateInstance::controlled(
                                aleph_core::Gate::Phase(aleph_core::Param::Concrete(
                                    std::f64::consts::PI / (2f64.powi((j - q) as i32)),
                                )),
                                vec![q],
                                vec![j],
                            ))
                            .unwrap();
                    }
                }
            }
            _ => {
                for layer in 0..6 {
                    for q in 0..22 {
                        circuit.ry(0.13 + (q + layer) as f64 * 0.07, q).unwrap();
                        circuit.rz(0.23 + (q + layer) as f64 * 0.03, q).unwrap();
                    }
                    for q in 0..21 {
                        circuit.cnot(q, q + 1).unwrap();
                    }
                }
            }
        }
        for optimize in [false, true] {
            let plan = Plan::compile(&circuit, Options { optimize }).unwrap();
            macro_rules! measure {
                ($backend:ty,$precision:literal) => {{
                    let mut be = <$backend>::with_seed(552).unwrap();
                    group.bench_function(
                        format!("{family}_{}_optimized{optimize}", $precision),
                        |b| b.iter(|| criterion::black_box(plan.execute(&mut be).unwrap())),
                    );
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
