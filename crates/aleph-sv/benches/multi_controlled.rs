//! P1-08 benchmarks: synthetic chains of Toffoli, CCZ, and MCX gates.
//!
//! Seven bench cases:
//!   * `toffoli_chain_n15` / `toffoli_chain_n20` — 100 Toffoli gates cycling
//!     over distinct (c0, c1, target) triples. n=15 → L2-resident on EPYC;
//!     n=20 → DRAM-bound.
//!   * `ccz_chain_n15` / `ccz_chain_n20` — same shape for CCZ (symmetric 3q
//!     diagonal gate; exercises the P1-08 CCZ dispatch path).
//!   * `mcx_k2_n20` / `mcx_k4_n20` / `mcx_k6_n20` — 100 repetitions of MCX
//!     with k=2/4/6 external controls on a 20-qubit state. Validates that
//!     P1-05's anti-diagonal kernel with many controls does not regress.
//!
//! The bench file is intentionally simple (single setup allocation per
//! iter_with_setup call) to match the pattern in `naive_sv.rs` and
//! `soa_vs_naive.rs`. Actual EPYC perf numbers are collected in T17.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use smallvec::smallvec;

// ---------------------------------------------------------------------------
// Toffoli chain
// ---------------------------------------------------------------------------

fn bench_toffoli_chain(c: &mut Criterion, n: u32, gates: usize) {
    let bench_name = format!("toffoli_chain_n{n}");
    c.bench_function(&bench_name, |b| {
        b.iter_with_setup(
            || {
                let mut backend = NaiveSvBackend::with_seed(0);
                let state = backend.allocate(n).unwrap();
                (backend, state)
            },
            |(mut backend, mut state)| {
                for i in 0..gates {
                    // Cycle (c0, c1, target) over distinct qubit triples.
                    // Skip degenerate cases where two qubits coincide.
                    let c0 = (i as u32) % n;
                    let c1 = ((i as u32) + 1) % n;
                    let t = ((i as u32) + 2) % n;
                    if c0 == c1 || c0 == t || c1 == t {
                        continue;
                    }
                    let gi = GateInstance::new(Gate::Toffoli, smallvec![c0, c1, t]);
                    backend.apply_gate(&mut state, &gi).unwrap();
                }
                criterion::black_box(&state);
            },
        );
    });
}

// ---------------------------------------------------------------------------
// CCZ chain
// ---------------------------------------------------------------------------

fn bench_ccz_chain(c: &mut Criterion, n: u32, gates: usize) {
    let bench_name = format!("ccz_chain_n{n}");
    c.bench_function(&bench_name, |b| {
        b.iter_with_setup(
            || {
                let mut backend = NaiveSvBackend::with_seed(0);
                let state = backend.allocate(n).unwrap();
                (backend, state)
            },
            |(mut backend, mut state)| {
                for i in 0..gates {
                    // CCZ is symmetric; qubits = [q0, q1, q2].
                    let q0 = (i as u32) % n;
                    let q1 = ((i as u32) + 1) % n;
                    let q2 = ((i as u32) + 2) % n;
                    if q0 == q1 || q0 == q2 || q1 == q2 {
                        continue;
                    }
                    let gi = GateInstance::new(Gate::Ccz, smallvec![q0, q1, q2]);
                    backend.apply_gate(&mut state, &gi).unwrap();
                }
                criterion::black_box(&state);
            },
        );
    });
}

// ---------------------------------------------------------------------------
// MCX (X with k external controls)
// ---------------------------------------------------------------------------

fn bench_mcx(c: &mut Criterion, n: u32, k: u32) {
    let bench_name = format!("mcx_k{k}_n{n}");
    c.bench_function(&bench_name, |b| {
        // Build the GateInstance once outside iter_with_setup — it is
        // immutable and cheap to clone; allocating it inside would add noise.
        // Controls = q[0..k], target = q[k].  All fit within n qubits.
        assert!(
            k < n,
            "bench_mcx: k={k} controls + 1 target requires n > k but got n={n}",
        );
        let controls: smallvec::SmallVec<[u32; 2]> = (0u32..k).collect();
        let target = k;
        // Gate::X is the 1-qubit base gate; external controls are the k
        // additional qubits.  Pattern mirrors the oracle test in
        // `aleph-oracle/tests/multi_controlled.rs` (multi_ctrl_mcx_k7_8q_oracle).
        let gi = GateInstance::controlled(Gate::X, smallvec![target], controls);

        b.iter_with_setup(
            || {
                let mut backend = NaiveSvBackend::with_seed(0);
                let state = backend.allocate(n).unwrap();
                (backend, state)
            },
            |(mut backend, mut state)| {
                for _ in 0..100 {
                    backend.apply_gate(&mut state, &gi).unwrap();
                }
                criterion::black_box(&state);
            },
        );
    });
}

// ---------------------------------------------------------------------------
// Criterion entry point
// ---------------------------------------------------------------------------

fn benches(c: &mut Criterion) {
    // Toffoli chains
    bench_toffoli_chain(c, 15, 100);
    bench_toffoli_chain(c, 20, 100);

    // CCZ chains
    bench_ccz_chain(c, 15, 100);
    bench_ccz_chain(c, 20, 100);

    // MCX at k=2,4,6 on n=20 (DRAM-bound; validates no anti-diagonal regression)
    bench_mcx(c, 20, 2);
    bench_mcx(c, 20, 4);
    bench_mcx(c, 20, 6);
}

criterion_group!(p1_08, benches);
criterion_main!(p1_08);
