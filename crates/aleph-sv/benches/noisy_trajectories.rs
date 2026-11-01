//! Identical before/after public-driver workloads; state/model setup excluded.
use aleph_core::{Gate, GateInstance};
use aleph_ir::Circuit;
use aleph_sv::noise::{self, NoiseModel, ReadoutError};
use criterion::{criterion_group, criterion_main, Criterion};
use std::{path::PathBuf, time::Duration};

fn fixture(n: u32, mixed: bool) -> (Circuit, NoiseModel) {
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit
            .add_gate(GateInstance::new(Gate::H, &[q][..]))
            .unwrap();
        if mixed {
            circuit
                .add_gate(GateInstance::new(Gate::Ry(0.73.into()), &[q][..]))
                .unwrap();
        }
    }
    let mut model = NoiseModel::new();
    model.add_all_qubit_quantum_error(noise::depolarizing_error(0.05, 1), &["H"]);
    if mixed {
        model.add_all_qubit_quantum_error(noise::amplitude_damping_error(0.17), &["Ry"]);
        model.add_all_qubit_quantum_error(noise::phase_damping_error(0.23), &["Ry"]);
    }
    model.add_readout_error(ReadoutError::new([[0.97, 0.03], [0.07, 0.93]]), 0);
    (circuit, model)
}
fn bench(c: &mut Criterion) {
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut group = c.benchmark_group(format!("p8_trajectories/workers{workers}"));
        group
            .sample_size(10)
            .warm_up_time(Duration::from_millis(300))
            .measurement_time(Duration::from_millis(700));
        for mixed in [false, true] {
            for (n, shots) in [(4, 1024), (16, 64), (20, 8)] {
                let (circuit, model) = fixture(n, mixed);
                let kind = if mixed { "kraus" } else { "pauli" };
                group.bench_function(format!("{kind}_n{n}_shots{shots}"), |b| {
                    b.iter(|| {
                        pool.install(|| {
                            criterion::black_box(
                                noise::run_noisy(&circuit, &model, shots, 557).unwrap(),
                            )
                        })
                    });
                });
            }
        }
        group.finish();
    }
}
fn config() -> Criterion {
    let mut c = Criterion::default();
    if let Some(path) = std::env::var_os("ALEPH_BENCH_OUTPUT") {
        c = c.output_directory(&PathBuf::from(path));
    }
    c
}
criterion_group! {name=benches; config=config(); targets=bench}
criterion_main!(benches);
