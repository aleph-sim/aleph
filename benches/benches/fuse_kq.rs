//! P2-07 k-qubit-fusion before/after benchmark + `max_qubits` sweep.
//!
//! Isolates the impact of [`FuseKq`] on fusible workloads (random brick-wall,
//! VQE-like, QAOA-like): the same circuit is optimized with the pre-P2-07
//! pipeline (`Cancel, DCE, FuseDiagonal, Fuse1q, Fuse2q`) and with the current
//! default (which appends `FuseKq`), then each optimized circuit is executed
//! through the AoS + AVX-512 `NaiveSvBackend`. The wall-clock delta is the
//! arithmetic-intensity win from collapsing chained 1q/2q blocks into dense
//! `UnitaryKq` blocks (one pass instead of many).
//!
//! Also sweeps `max_qubits ∈ {2,3,4,5}` to pick the `FuseKq::default()` cap.
//!
//! Gated behind `scaling-bench`. Run on a verified-idle box (CLAUDE.md):
//!   cargo bench -p aleph-benches --bench fuse_kq --features scaling-bench
//! The instruction-count reduction prints to stderr once at startup.

use aleph_backend::run;
use aleph_core::{Gate, GateInstance};
use aleph_ir::passes::{
    CancelInversePairs, DeadCodeElim, Fuse1qRuns, Fuse2q, FuseDiagonalRuns, FuseKq, PassPipeline,
};
use aleph_ir::Circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallvec::smallvec;
use std::hint::black_box;

/// QAOA-like: layers of RZ + nearest-neighbour CNOT + RX mixer.
fn qaoa_like(n: u32, layers: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for l in 0..layers {
        for q in 0..n {
            let _ = c.rz(0.2 + 0.13 * (l as f64) + 0.05 * q as f64, q);
        }
        for q in 0..n.saturating_sub(1) {
            let _ = c.cnot(q, q + 1);
        }
        for q in 0..n {
            let _ = c.rx(0.4 - 0.03 * (l as f64), q);
        }
    }
    c
}

/// VQE-like hardware-efficient ansatz: RY/RZ rotations + CZ ladder.
fn vqe_like(n: u32, layers: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for l in 0..layers {
        for q in 0..n {
            let _ = c.ry(0.5 + 0.07 * (l as f64) + 0.02 * q as f64, q);
            let _ = c.rz(0.3 - 0.04 * q as f64, q);
        }
        for q in 0..n.saturating_sub(1) {
            let _ = c.add_gate(GateInstance::new(Gate::Cz, smallvec![q, q + 1]));
        }
    }
    c
}

fn pipeline_without_kq() -> PassPipeline {
    PassPipeline::new(vec![
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        Box::new(FuseDiagonalRuns),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
    ])
}

fn pipeline_with_kq(max_qubits: usize) -> PassPipeline {
    PassPipeline::new(vec![
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        Box::new(FuseDiagonalRuns),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
        Box::new(FuseKq { max_qubits }),
    ])
}

fn optimized(base: &Circuit, pipeline: &PassPipeline) -> Circuit {
    let mut c = base.clone();
    pipeline.run(&mut c).expect("pipeline run");
    c
}

fn workloads(n: u32) -> Vec<(String, Circuit)> {
    vec![
        (
            "random".to_string(),
            aleph_benches::random_brickwall_circuit(n, 20),
        ),
        ("qaoa".to_string(), qaoa_like(n, 6)),
        ("vqe".to_string(), vqe_like(n, 6)),
    ]
}

fn report_pass_counts() {
    let without = pipeline_without_kq();
    let with = pipeline_with_kq(4);
    eprintln!("--- P2-07 instruction-count reduction (with vs without FuseKq, max_qubits=4) ---");
    for n in [22u32, 25] {
        for (label, base) in workloads(n) {
            let a = optimized(&base, &without).len();
            let b = optimized(&base, &with).len();
            let ratio = a as f64 / b.max(1) as f64;
            eprintln!("  n{n:<3} {label:<8} without={a:>5}  with={b:>5}  reduction={ratio:>5.2}x");
        }
    }
    eprintln!("--------------------------------------------------------------------------------");
}

fn bench_fuse_kq(c: &mut Criterion) {
    report_pass_counts();

    let without = pipeline_without_kq();
    let with4 = pipeline_with_kq(4);

    let mut group = c.benchmark_group("fuse_kq");
    group.sample_size(10); // n=25 is 512 MiB/run; sample_size is a floor (P2-05 note)

    // Before/after at n=22 and n=25 for each workload.
    for n in [22u32, 25] {
        for (label, base) in workloads(n) {
            let opt_without = optimized(&base, &without);
            let opt_with = optimized(&base, &with4);
            group.bench_with_input(
                BenchmarkId::new(format!("{label}_without"), n),
                &opt_without,
                |b, circ| {
                    b.iter(|| {
                        let mut be = NaiveSvBackend::new();
                        black_box(run(&mut be, black_box(circ)).unwrap());
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{label}_with"), n),
                &opt_with,
                |b, circ| {
                    b.iter(|| {
                        let mut be = NaiveSvBackend::new();
                        black_box(run(&mut be, black_box(circ)).unwrap());
                    })
                },
            );
        }
    }

    // max_qubits sweep on QAOA-25 (representative fusible workload).
    let sweep_base = qaoa_like(25, 6);
    for mk in [2usize, 3, 4, 5] {
        let opt = optimized(&sweep_base, &pipeline_with_kq(mk));
        group.bench_with_input(BenchmarkId::new("qaoa_sweep_maxk", mk), &opt, |b, circ| {
            b.iter(|| {
                let mut be = NaiveSvBackend::new();
                black_box(run(&mut be, black_box(circ)).unwrap());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fuse_kq);
criterion_main!(benches);
