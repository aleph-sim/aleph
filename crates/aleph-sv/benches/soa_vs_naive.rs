//! Side-by-side criterion benches: `NaiveSvBackend` vs `SoaSvBackend`
//! on QFT and GHZ. Group / parameter shape produces bencher.dev
//! side-by-side bars (`qft/n10/naive` vs `qft/n10/soa`, …).
//!
//! BACKLOG P1-01 acceptance: `qft/n20/soa` ≥ 1.5× faster than
//! `qft/n20/naive` on the canonical bench server.

use aleph_backend::run;
use aleph_core::{Gate, GateInstance};
use aleph_ir::Circuit;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallvec::smallvec;

/// Build a QFT circuit on `n` qubits via the standard
/// H-then-controlled-phase decomposition. Layout-only port — no
/// approximations, no SWAP elimination at the end.
fn qft_circuit(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        c.add_gate(GateInstance::new(Gate::H, smallvec![j]))
            .unwrap();
        for k in (j + 1)..n {
            let angle = std::f64::consts::PI / (1u64 << (k - j)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(angle.into()),
                smallvec![j],
                smallvec![k],
            ))
            .unwrap();
        }
    }
    // Bit-reversal swaps to land in the natural basis.
    for i in 0..(n / 2) {
        c.add_gate(GateInstance::new(Gate::Swap, smallvec![i, n - 1 - i]))
            .unwrap();
    }
    c
}

/// `n`-qubit GHZ via H on q0 and CX chain.
fn ghz_circuit(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(GateInstance::new(Gate::H, smallvec![0]))
        .unwrap();
    for t in 1..n {
        c.add_gate(GateInstance::new(Gate::Cnot, smallvec![0, t]))
            .unwrap();
    }
    c
}

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft");
    for &n in &[10u32, 15, 20] {
        let circuit = qft_circuit(n);
        group.bench_with_input(BenchmarkId::new(format!("n{n}"), "naive"), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut backend| {
                    let _state = run(&mut backend, &circuit).unwrap();
                },
            );
        });
        group.bench_with_input(BenchmarkId::new(format!("n{n}"), "soa"), &n, |b, _| {
            b.iter_with_setup(
                || SoaSvBackend::with_seed(0),
                |mut backend| {
                    let _state = run(&mut backend, &circuit).unwrap();
                },
            );
        });
    }
    group.finish();
}

fn bench_ghz(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz");
    let n = 20u32;
    let circuit = ghz_circuit(n);
    group.bench_with_input(BenchmarkId::new(format!("n{n}"), "naive"), &n, |b, _| {
        b.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let _state = run(&mut backend, &circuit).unwrap();
            },
        );
    });
    group.bench_with_input(BenchmarkId::new(format!("n{n}"), "soa"), &n, |b, _| {
        b.iter_with_setup(
            || SoaSvBackend::with_seed(0),
            |mut backend| {
                let _state = run(&mut backend, &circuit).unwrap();
            },
        );
    });
    group.finish();
}

criterion_group!(benches, bench_qft, bench_ghz);
criterion_main!(benches);
