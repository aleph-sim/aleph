//! QFT-shaped placeholder benchmark for n ∈ {10, 15, 20}.
//!
//! Real QFT cost is O(n_qubits · 2^n_qubits) — the controlled-phase
//! ladder runs n_qubits passes through the state vector. Until
//! P0-09's naive backend lands, the bench body does an O(n_qubits ·
//! 2^n_qubits) workload with similar memory-traffic shape: `n_qubits`
//! sweeps, each applying a per-amplitude phase rotation. Throughput
//! reported in (amplitudes · passes) = `n_qubits * 2^n_qubits` so the
//! bencher.dev timeline stays stable when the body gets swapped for a
//! real circuit invocation.
//!
//! Reference: Nielsen & Chuang § 5.1 (Quantum Fourier Transform);
//! aleph's own `QFT.md` playbook at the repo root.

use aleph_benches::zero_state;
use aleph_core::Complex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const QUBIT_COUNTS: &[u32] = &[10, 15, 20];

fn qft_workload(n_qubits: u32) -> Vec<Complex> {
    let mut amps = zero_state(n_qubits);
    let dim = amps.len();
    for _pass in 0..n_qubits {
        for (idx, amp) in amps.iter_mut().enumerate() {
            let theta = (idx as f64) * std::f64::consts::TAU / (dim as f64);
            let (sin, cos) = theta.sin_cos();
            *amp *= Complex::new(cos, sin);
        }
    }
    amps
}

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft");
    for &n in QUBIT_COUNTS {
        // Throughput is total per-amplitude ops: n_qubits passes
        // × 2^n_qubits amplitudes. Matches what a real QFT will
        // produce on the same input.
        group.throughput(Throughput::Elements(u64::from(n) * (1u64 << n)));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(qft_workload(n)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qft);
criterion_main!(benches);
