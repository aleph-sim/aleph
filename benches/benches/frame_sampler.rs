//! P4.6-02 AC-1: batched Pauli-frame sampling vs sequential per-shot sampling
//! of the surface-d11 syndrome cycle (120 ancillas, 1024 shots).
//!
//! Both start from the same post-cycle stabilizer state (gates applied once);
//! `sequential` clones + measures the ancillas once per shot (the per-shot CHP
//! `Tableau::measure` loop the pre-P4.6-02 `Backend::sample` was built on),
//! `batched` runs `sample_frames` (64 shots/pass, x/z rowsum work shared).
//! Target: batched ≥ 10× faster.

use aleph_backend::Backend;
use aleph_benches::SurfaceCode;
use aleph_stab::{StabilizerBackend, Tableau};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Post-cycle stabilizer state of the distance-`d` surface code (one syndrome
/// extraction cycle of gates applied to |0…0⟩) plus the ancilla index list.
fn cycle_state(d: usize) -> (Tableau, Vec<usize>) {
    let sc = SurfaceCode::new(d);
    let mut be = StabilizerBackend::with_seed(0);
    let mut t = be.allocate(sc.num_qubits as u32).unwrap();
    for g in sc.cycle_gates() {
        be.apply_gate(&mut t, &g).unwrap();
    }
    let ancillas = sc.ancilla_order().iter().map(|&a| a as usize).collect();
    (t, ancillas)
}

fn bench(cr: &mut Criterion) {
    let d = 11usize;
    let shots = 1024u32;
    let (state, ancillas) = cycle_state(d);

    let mut grp = cr.benchmark_group(format!("frame_sampler_d{d}_shots{shots}"));
    grp.sample_size(10);

    grp.bench_function("sequential", |b| {
        let mut rng = StdRng::seed_from_u64(1);
        b.iter(|| {
            let mut acc = 0u64;
            for _ in 0..shots {
                let mut t = state.clone();
                for &a in &ancillas {
                    acc ^= t.measure(a, &mut rng).unwrap() as u64;
                }
            }
            acc
        })
    });

    grp.bench_function("batched", |b| {
        let mut rng = StdRng::seed_from_u64(1);
        b.iter(|| state.sample_frames(&ancillas, shots, &mut rng))
    });

    grp.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
