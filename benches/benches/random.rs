//! Random-circuit stub benchmark for n=20, depth=20.
//!
//! A real random-circuit benchmark (Sycamore-style) is the hardest test
//! for any state-vector backend (see `RANDOM CIRCUIT.md`). Until P0-09
//! lands a backend we instead benchmark a fixed deterministic workload
//! with similar memory traffic: `depth` passes through the amplitude
//! vector, each pass mixing pairs of neighbouring amplitudes — coarsely
//! mirrors the touch pattern of a layer of two-qubit gates. The
//! "random" label is preserved so bencher.dev's timeline stays
//! continuous when the body gets replaced with a real circuit later.

use aleph_benches::zero_state;
use aleph_core::Complex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const N_QUBITS: u32 = 20;
const DEPTHS: &[usize] = &[20];

fn random_workload(n_qubits: u32, depth: usize) -> Vec<Complex> {
    let mut amps = zero_state(n_qubits);
    // Seed an interesting initial pattern so subsequent passes have
    // non-trivial data to move around.
    amps[1] = Complex::new(0.5, 0.5);
    amps[2] = Complex::new(0.5, -0.5);

    for layer in 0..depth {
        // Alternate between adjacent-pair and stride-2-pair mixes per
        // layer; matches roughly the brick-wall pattern of a Sycamore
        // random circuit at the memory-traffic level.
        let stride = if layer % 2 == 0 { 1 } else { 2 };
        let mut i = 0;
        while i + stride < amps.len() {
            let a = amps[i];
            let b = amps[i + stride];
            amps[i] = a + b;
            amps[i + stride] = a - b;
            i += stride * 2;
        }
    }
    amps
}

fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("random/circuit");
    for &depth in DEPTHS {
        let elements = (1u64 << N_QUBITS) * depth as u64;
        group.throughput(Throughput::Elements(elements));
        group.bench_with_input(
            BenchmarkId::new(format!("n{N_QUBITS}"), depth),
            &(N_QUBITS, depth),
            |b, &(n, d)| {
                b.iter(|| black_box(random_workload(n, d)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_random);
criterion_main!(benches);
