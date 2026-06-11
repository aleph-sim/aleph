//! P4.5-01 parity bench: aleph-mps on the byte-identical QASM fixtures that
//! `scripts/mps-baseline/run.py` times through Aer matrix_product_state.
//! Sequential default build (no `parallel` feature) — default-vs-default.
//! χ per family matches the harness's CHI table; brickwork and long_range are
//! exact at their χ (no truncation on either side, fidelity equal by
//! construction); wide_bond saturates the cap (truncation caveat in
//! docs/perf/parity.md).

use aleph_backend::run;
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// (fixture stem, max bond) — keep in sync with scripts/mps-baseline/run.py CHI.
const WORKLOADS: &[(&str, usize)] = &[
    ("brickwork_n128_d6", 64),
    ("long_range_n12_dist4", 64),
    ("long_range_n12_dist8", 64),
    ("long_range_n12_dist11", 64),
    ("wide_bond_n26_d12", 256),
];

fn load(stem: &str) -> aleph_ir::Circuit {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two dirs deep from repo root")
        .join("scripts/mps-baseline/circuits")
        .join(format!("{stem}.qasm"));
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
    aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse {} failed: {e:?}", path.display()))
}

fn bench_parity(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_parity");
    group.sample_size(10);
    for &(stem, chi) in WORKLOADS {
        let circuit = load(stem);
        group.bench_function(stem, |b| {
            b.iter_with_setup(
                || MpsBackend::with_seed(0).with_max_bond(chi),
                |mut backend| {
                    let state = run(&mut backend, &circuit).unwrap();
                    black_box(state);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parity);
criterion_main!(benches);
