//! Wide-bond MPS benchmark: a random brickwall whose central bond saturates
//! the χ cap, so the per-gate cost is dominated by the (2χ)×(2χ) SVD and the
//! theta gemm — the surfaces parallelized in P3-09 (AC-2).
//!
//! Saturation was measured empirically on this branch (Apple Silicon, release):
//!   n=20, χ=128: saturates at L1=16 layers (max_bond_reached=128)
//!   n=24, χ=256: saturates at L2=20 layers (max_bond_reached=256)
//!   n=24/χ=256 single-run wall time: ~4.2s (well within criterion sample_size(10) budget)
//!
//! Since P3-13 the `parallel` feature is a default, so this bench compiles
//! everywhere (including `cargo bench --workspace`). A runtime env guard
//! (`WIDE_BOND=1`) keeps its saturating sweep cells out of routine
//! push-to-main Bench runs on the shared EPYC runner. Run it explicitly:
//! `WIDE_BOND=1 RAYON_NUM_THREADS=1|2|4|8|16 cargo bench -p aleph-mps --bench wide_bond`
//! The t=1 row doubles as the sequential reference.
//!
//! The chi=512 cell (where parallelism starts to win; ~47 s/iter sequential
//! on EPYC) is additionally gated behind `WIDE_BOND_CHI512=1`.

use aleph_backend::run;
use aleph_benches::brickwall_ry_cnot_rz as brickwall;
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, Criterion};

/// Layers needed for n=20, χ=128 central bond to hit the cap. The shared
/// builder's saturation property is pinned by aleph-mps
/// `brickwall_saturates_bond_cap`.
const L1: u32 = 16;
/// Layers needed for n=24, χ=256 central bond to hit the cap.
const L2: u32 = 20;

fn bench(cr: &mut Criterion) {
    // Runtime gate: `parallel` is a default feature since P3-13, so this
    // bench now compiles under `cargo bench --workspace` — but its sweep
    // cells would tax every push-to-main Bench run on the shared EPYC
    // runner. Opt in explicitly:
    // WIDE_BOND=1 [WIDE_BOND_CHI512=1] RAYON_NUM_THREADS=N \
    //   cargo bench -p aleph-mps --bench wide_bond
    if std::env::var("WIDE_BOND").as_deref() != Ok("1") {
        eprintln!("wide_bond: skipped (set WIDE_BOND=1 to run the sweep)");
        return;
    }
    let mut grp = cr.benchmark_group("wide_bond_brickwall");
    grp.sample_size(10);
    for (n, chi, layers) in [(20u32, 128usize, L1), (24, 256, L2)] {
        let c = brickwall(n, layers);
        grp.bench_function(format!("n{n}_chi{chi}_d{layers}"), |b| {
            b.iter(|| {
                let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
                run(&mut be, &c).unwrap()
            })
        });
    }
    if std::env::var("WIDE_BOND_CHI512").as_deref() == Ok("1") {
        let c = brickwall(26, 24);
        grp.bench_function("n26_chi512_d24", |b| {
            b.iter(|| {
                let mut be = MpsBackend::with_seed(0).with_max_bond(512);
                run(&mut be, &c).unwrap()
            })
        });
    }
    grp.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
