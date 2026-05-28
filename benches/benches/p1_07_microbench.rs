//! P1-07 micro-bench: pins the BACKLOG-AC claim that the specialised
//! `Cnot` 2q path is ≥ 5× faster than the generic `Unitary2q` kernel
//! on AVX-512 hardware.
//!
//! Three benchmarks at n=20, 100 gates each, driven through
//! `NaiveSvBackend` via `aleph_backend::run`:
//!
//! 1. `cnot_specialized` — `Gate::Cnot` 100× → exercises the
//!    `Perm2qKind::CnotHi` fast path in `apply_2q`.
//! 2. `cnot_via_generic` — same CNOT topology, but the payload is a
//!    `Gate::Unitary2q` matrix with a `1e-12` off-diagonal
//!    perturbation. The perturbation magnitude squared (`1e-24`) is
//!    above `DIAGONAL_EPS_SQ = 1e-30`, so the matrix fails the
//!    diagonal pre-test and is rejected by `classify_2q_permutation`;
//!    dispatch lands on the generic-2q SIMD kernel.
//! 3. `dense_2q` — a fully dense, non-permutation, non-diagonal 2q
//!    matrix. Realistic upper bound on generic-2q cost.
//!
//! Local arm64 will not see the 5× ratio (no AVX-512); the target
//! number is for the EPYC bencher.dev runner.

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance};
use aleph_ir::Circuit;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use smallvec::smallvec;
use std::hint::black_box;

const N_QUBITS: u32 = 20;
const N_GATES: usize = 100;

/// CNOT pair `(control, target)` for gate index `i`. Walks through
/// adjacent qubit pairs so the memory-access stride varies — keeps
/// the bench out of the degenerate "one stride forever" cache pattern.
#[inline]
fn cnot_pair(i: usize) -> (u32, u32) {
    let c = (i as u32) % (N_QUBITS - 1);
    (c, c + 1)
}

/// Canonical CNOT matrix in MSB-first basis: `|c t⟩` with `c` = high
/// qubit, so the permutation is `[0, 1, 3, 2]`.
fn cnot_matrix() -> [[Complex; 4]; 4] {
    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
}

/// CNOT matrix with a `1e-12` real perturbation on a zero off-diagonal
/// entry. `1e-12² = 1e-24 > DIAGONAL_EPS_SQ = 1e-30` → fails diagonal
/// pre-test → `classify_2q_permutation` returns `None` → generic path.
fn cnot_matrix_perturbed() -> [[Complex; 4]; 4] {
    let mut m = cnot_matrix();
    m[0][1] = Complex::new(1e-12, 0.0);
    m
}

/// Dense non-diagonal, non-permutation 2q matrix. Built as the
/// Kronecker product `Ry(θ) ⊗ Ry(φ)` for irrational `θ = 0.7`,
/// `φ = 1.1` — guaranteed unitary (so the backend's normalisation
/// check passes), fully real, with every entry non-zero. Neither
/// the diagonal nor the permutation fast path can engage.
fn dense_2q_matrix() -> [[Complex; 4]; 4] {
    // Ry(θ) = [[ cos(θ/2), -sin(θ/2)], [ sin(θ/2),  cos(θ/2)]]
    let (ca, sa) = ((0.7_f64 / 2.0).cos(), (0.7_f64 / 2.0).sin());
    let (cb, sb) = ((1.1_f64 / 2.0).cos(), (1.1_f64 / 2.0).sin());
    let r = |x: f64| Complex::new(x, 0.0);
    // Kronecker product Ry(θ) ⊗ Ry(φ), row-major in MSB-first basis.
    [
        [r(ca * cb), r(-ca * sb), r(-sa * cb), r(sa * sb)],
        [r(ca * sb), r(ca * cb), r(-sa * sb), r(-sa * cb)],
        [r(sa * cb), r(-sa * sb), r(ca * cb), r(-ca * sb)],
        [r(sa * sb), r(sa * cb), r(ca * sb), r(ca * cb)],
    ]
}

fn build_cnot_circuit() -> Circuit {
    let mut c = Circuit::new(N_QUBITS, 0);
    for i in 0..N_GATES {
        let (ctrl, tgt) = cnot_pair(i);
        let _ = c.cnot(ctrl, tgt);
    }
    c
}

fn build_unitary2q_circuit(matrix: [[Complex; 4]; 4]) -> Circuit {
    let mut c = Circuit::new(N_QUBITS, 0);
    for i in 0..N_GATES {
        let (q0, q1) = cnot_pair(i);
        let _ = c.add_gate(GateInstance::new(
            Gate::Unitary2q(Box::new(matrix)),
            smallvec![q0, q1],
        ));
    }
    c
}

fn bench_cnot_n20_specialized(c: &mut Criterion) {
    let circuit = build_cnot_circuit();
    c.bench_function("p1_07/cnot_specialized", |b| {
        b.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let state = run(&mut backend, &circuit).unwrap();
                black_box(state);
            },
        );
    });
}

fn bench_cnot_n20_via_generic(c: &mut Criterion) {
    let circuit = build_unitary2q_circuit(cnot_matrix_perturbed());
    c.bench_function("p1_07/cnot_via_generic", |b| {
        b.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let state = run(&mut backend, &circuit).unwrap();
                black_box(state);
            },
        );
    });
}

fn bench_dense_2q_n20(c: &mut Criterion) {
    let circuit = build_unitary2q_circuit(dense_2q_matrix());
    c.bench_function("p1_07/dense_2q", |b| {
        b.iter_with_setup(
            || NaiveSvBackend::with_seed(0),
            |mut backend| {
                let state = run(&mut backend, &circuit).unwrap();
                black_box(state);
            },
        );
    });
}

criterion_group!(
    benches,
    bench_cnot_n20_specialized,
    bench_cnot_n20_via_generic,
    bench_dense_2q_n20,
);
criterion_main!(benches);
