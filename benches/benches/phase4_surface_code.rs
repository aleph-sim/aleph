//! P4-07 surface-code cycle timing on the stabilizer backend. One rotated
//! surface-code syndrome-extraction cycle per distance d ∈ {3,5,7,9,11}
//! (2d²−1 qubits, up to 241 at d=11). The aleph half of the
//! `docs/perf/surface_code.md` report row (baseline = Stim, timed separately).
//!
//!   cargo bench -p aleph-benches --bench phase4_surface_code
//!
//! Each timed iteration allocates a fresh tableau, applies the cycle gates, and
//! measures all ancillas — matching Stim's per-cycle `TableauSimulator().do()`
//! which likewise builds fresh state. Allocation is O(n²) and negligible
//! relative to the gate + measurement work, so it does not distort the row.
//!
//! Note: the backend is constructed INSIDE the timed routine rather than handed
//! over from an `iter_batched` setup closure — `iter_batched`'s setup and
//! routine closures would both need a mutable `be`, which cannot be shared
//! across them under the borrow checker. Plain `b.iter` with per-iteration
//! construction is simpler and equally correct: each iteration runs a full
//! fresh cycle.

use aleph_backend::Backend;
use aleph_benches::SurfaceCode;
use aleph_stab::StabilizerBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];

fn bench_surface(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface_code");
    group.sample_size(10);
    for &d in DISTANCES {
        let sc = SurfaceCode::new(d);
        let gates = sc.cycle_gates();
        let order = sc.ancilla_order();
        let n = sc.num_qubits as u32;
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| {
                let mut be = StabilizerBackend::with_seed(0);
                let mut t = be.allocate(n).unwrap();
                for g in &gates {
                    be.apply_gate(&mut t, g).unwrap();
                }
                let mut acc = false;
                for &a in &order {
                    acc ^= be.measure(&mut t, a).unwrap();
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_surface);
criterion_main!(benches);
