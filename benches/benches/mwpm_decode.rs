//! Q1-03 baseline + speedup bench: MWPM decode throughput on surface-code memory syndromes at
//! threshold density, for d ∈ {7, 9, 11, 13}.
//!
//! Reports decoded **syndromes/second** (criterion `Throughput::Elements`). The Q1-02 dense
//! all-pairs decoder is the baseline; Q1-03's localized path is compared with
//! `cargo bench --bench mwpm_decode -- --baseline q1-02` after saving a baseline.
//!
//! Run (idle box; see CLAUDE.md perf rules):
//!   cargo bench -p aleph-benches --bench mwpm_decode

use aleph_qec::{build_dem, Decoder, DetectorErrorModel, MwpmDecoder, SurfaceCode, Syndrome};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Sample `shots` syndromes from `dem` at the given seed (Bernoulli per mechanism — the same
/// generative model the Monte-Carlo harness uses).
fn sample(dem: &DetectorErrorModel, shots: usize, seed: u64) -> Vec<Syndrome> {
    let mut out = Vec::with_capacity(shots);
    for s in 0..shots as u64 {
        let mut z = seed.wrapping_add(s.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut next = || {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut det = vec![false; dem.detectors];
        for e in &dem.errors {
            if next() < e.prob {
                for &d in &e.dets {
                    det[d as usize] ^= true;
                }
            }
        }
        out.push(Syndrome::from_bits(&det));
    }
    out
}

fn bench(cr: &mut Criterion) {
    // Phenomenological error rate near the memory threshold (~3%), so syndromes are dense enough
    // to stress the matching (the regime Q1-03's locality must speed up).
    let p = 0.03;
    let shots = 256usize;

    let mut grp = cr.benchmark_group("mwpm_decode");
    for d in [7usize, 9, 11, 13] {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        let decoder = MwpmDecoder::new(&dem).unwrap();
        let syndromes = sample(&dem, shots, 0x5EED ^ d as u64);
        let avg_defects: f64 =
            syndromes.iter().map(|s| s.weight()).sum::<usize>() as f64 / shots as f64;

        grp.throughput(Throughput::Elements(shots as u64));
        let tag = format!("d{d}_det{}_avgdef{avg_defects:.0}", dem.detectors);
        // Q1-02 baseline: dense all-pairs matching.
        grp.bench_with_input(
            BenchmarkId::new("dense", &tag),
            &syndromes,
            |b, syndromes| {
                b.iter(|| {
                    for s in syndromes {
                        std::hint::black_box(decoder.decode_dense(std::hint::black_box(s)));
                    }
                });
            },
        );
        // Q1-03: localized matching.
        grp.bench_with_input(
            BenchmarkId::new("local", &tag),
            &syndromes,
            |b, syndromes| {
                b.iter(|| {
                    for s in syndromes {
                        std::hint::black_box(decoder.decode(std::hint::black_box(s)));
                    }
                });
            },
        );
    }
    grp.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
