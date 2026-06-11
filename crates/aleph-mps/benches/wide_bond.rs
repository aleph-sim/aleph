//! Wide-bond MPS benchmark: a random brickwall whose central bond saturates
//! the χ cap, so the per-gate cost is dominated by the (2χ)×(2χ) SVD and the
//! theta gemm — the surfaces parallelized in P3-09 (AC-2).
//!
//! Saturation was measured empirically on this branch (Apple Silicon, release):
//!   n=20, χ=128: saturates at L1=16 layers (max_bond_reached=128)
//!   n=24, χ=256: saturates at L2=20 layers (max_bond_reached=256)
//!   n=24/χ=256 single-run wall time: ~4.2s (well within criterion sample_size(10) budget)
//!
//! Thread sweep (requires the `parallel` feature):
//! `RAYON_NUM_THREADS=1|2|4|8|16 cargo bench -p aleph-mps --features parallel --bench wide_bond`
//! Default (no feature) runs sequential faer — the production configuration.
//!
//! The chi=512 cell (where parallelism starts to win; ~47 s/iter sequential
//! on EPYC) is gated behind `WIDE_BOND_CHI512=1` to keep CI bench runs fast.

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, Criterion};

/// Layers needed for n=20, χ=128 central bond to hit the cap.
const L1: u32 = 16;
/// Layers needed for n=24, χ=256 central bond to hit the cap.
const L2: u32 = 20;

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}

/// Brickwall of parameterized 2q blocks. Only alternating layers cross a
/// given bond cut, so the central bond reaches the χ cap after roughly
/// 2·log2(χ) layers; the remaining layers run at full bond dimension.
fn brickwall(n: u32, layers: u32) -> aleph_ir::Circuit {
    let mut c = aleph_ir::Circuit::new(n, 0);
    let mut t = 0.1f64;
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for layer in 0..layers {
        let mut q = layer % 2;
        while q + 1 < n {
            c.add_gate(g(Gate::Ry(Param::Concrete(t)), &[q])).unwrap();
            c.add_gate(g(Gate::Ry(Param::Concrete(t * 1.3 + 0.2)), &[q + 1]))
                .unwrap();
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(Gate::Rz(Param::Concrete(t * 0.7 + 0.1)), &[q + 1]))
                .unwrap();
            t += 0.37;
            q += 2;
        }
    }
    c
}

fn bench(cr: &mut Criterion) {
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
    if std::env::var_os("WIDE_BOND_CHI512").is_some() {
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
