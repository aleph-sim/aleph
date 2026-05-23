//! Random-circuit-shaped placeholder benchmark for n=20, depth=20.
//!
//! A real random-circuit benchmark (Sycamore-style) is the hardest
//! test for any state-vector backend — see `RANDOM CIRCUIT.md` at the
//! repo root. Until P0-09 lands a backend we instead benchmark a
//! brick-wall workload with matching memory traffic: `depth` passes
//! through the amplitude vector, each pass mixing every adjacent pair
//! exactly once (even layers pair (0,1),(2,3),...; odd layers offset
//! by 1, pair (1,2),(3,4),...). Every amplitude is touched once per
//! layer — same bandwidth profile as a real layer of two-qubit gates.
//!
//! **The state is intentionally NOT a valid quantum state.** Initial
//! amplitudes (1, 0.5+0.5i, 0.5-0.5i, 0, …) sum-of-squares = 2; the
//! brick-wall mix doubles amplitudes per layer without normalisation,
//! so by depth 20 they peak around 2^20 ≈ 1e6 — comfortably within
//! f64 range but obviously nonsense as physics. This bench measures
//! memory traffic, not quantum correctness. When P0-09 replaces the
//! body with a real circuit, both correctness and the bench name
//! (`random`) stay continuous.

use aleph_benches::zero_state;
use aleph_core::Complex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const N_QUBITS: u32 = 20;
const DEPTHS: &[usize] = &[20];

fn random_workload(n_qubits: u32, depth: usize) -> Vec<Complex> {
    let mut amps = zero_state(n_qubits);
    // Seed non-trivial values so subsequent layers have data to move.
    amps[1] = Complex::new(0.5, 0.5);
    amps[2] = Complex::new(0.5, -0.5);

    for layer in 0..depth {
        let offset = layer & 1;
        let mut i = offset;
        while i + 1 < amps.len() {
            let a = amps[i];
            let b = amps[i + 1];
            amps[i] = a + b;
            amps[i + 1] = a - b;
            i += 2;
        }
    }
    amps
}

fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("random");
    for &depth in DEPTHS {
        // depth × 2^N_QUBITS = total pair-updates touched.
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
