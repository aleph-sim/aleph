//! Random brick-wall circuit benchmark for n=20, depth=20.
//!
//! Drives `NaiveSvBackend` through a brick-wall circuit: each layer
//! is a per-qubit `Rz` + `Rx` (deterministic angles function of
//! `(layer, q)`) followed by an alternating-pair CNOT layer.  Not a
//! Sycamore-style Haar-random SU(4) circuit — the goal is bandwidth
//! shape and gate count comparable to a real random layer, with
//! determinism so the bench is reproducible without bringing rand
//! into the dep tree.  See `RANDOM CIRCUIT.md` at the repo root for
//! the playbook this approximates.

use aleph_backend::run;
use aleph_benches::random_brickwall_circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const N_QUBITS: u32 = 20;
const DEPTHS: &[usize] = &[20];

fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("random");
    for &depth in DEPTHS {
        let circuit = random_brickwall_circuit(N_QUBITS, depth);
        // depth × 2^n = total per-amplitude updates (1q rotations
        // touch each amplitude once per layer; CNOT rearranges in
        // place).  Stable elements/s shape across P0/P1.
        let elements = (1u64 << N_QUBITS) * depth as u64;
        group.throughput(Throughput::Elements(elements));
        group.bench_with_input(
            BenchmarkId::new(format!("n{N_QUBITS}"), depth),
            &(N_QUBITS, depth),
            |b, _| {
                b.iter_with_setup(
                    || NaiveSvBackend::with_seed(0),
                    |mut backend| {
                        let state = run(&mut backend, &circuit).unwrap();
                        black_box(state);
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_random);
criterion_main!(benches);
