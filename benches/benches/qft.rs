//! QFT-style benchmark stub for n ∈ {10, 15, 20}.
//!
//! Until P0-09's naive backend lands there's no Circuit/Backend to run
//! the actual QFT against. We benchmark a placeholder that does work
//! proportional to the QFT's O(2^n · n) gate count: build the |0…0⟩
//! state and walk every amplitude once applying a phase rotation —
//! roughly matches the bandwidth profile of a real QFT pass through
//! the state vector. When P0-09 lands, swap the body for a real circuit
//! invocation; the bench name + parameter shape stays identical so the
//! bencher.dev timeline is continuous.

use aleph_benches::zero_state;
use aleph_core::Complex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const QUBIT_COUNTS: &[u32] = &[10, 15, 20];

fn qft_workload(n_qubits: u32) -> Vec<Complex> {
    let mut amps = zero_state(n_qubits);
    let n = amps.len();
    // n iterations of a per-amplitude phase application — same memory
    // traffic pattern as one sweep of QFT's controlled-phase ladder.
    for (idx, amp) in amps.iter_mut().enumerate() {
        let theta = (idx as f64) * std::f64::consts::TAU / (n as f64);
        let (sin, cos) = theta.sin_cos();
        *amp *= Complex::new(cos, sin);
    }
    amps
}

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft/sweep");
    for &n in QUBIT_COUNTS {
        group.throughput(Throughput::Elements(1u64 << n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(qft_workload(n)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qft);
criterion_main!(benches);
