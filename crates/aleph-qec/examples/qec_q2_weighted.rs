//! Q2-02 head-to-head: **weighted** Union-Find vs **unweighted** Union-Find (Q2-01) vs MWPM, on
//! shared surface-code memory DEMs across `d ∈ {3,5,7,9,11}`.
//!
//! Two things to show:
//!
//! * **Accuracy** — weighted cluster growth (edge length ∝ matching weight) recovers part of MWPM's
//!   edge-weight awareness, so its logical-error rate should sit *below* unweighted UF and *closer
//!   to* MWPM. The effect grows with weight heterogeneity, so the default model is **asymmetric**
//!   (`p_meas > p_data`): horizontal data edges and vertical measurement edges then carry distinctly
//!   different weights. (Pass equal `p_data p_meas` to see the smaller uniform-noise effect.)
//! * **Runtime** — weighted growth must stay within 2× of unweighted (the Q2-02 budget). We time
//!   single-thread decode throughput for both and report the ratio.
//!
//! All three decoders run through the same Monte-Carlo harness on the *same* seed (hence the same
//! sampled shots), so every rate is apples-to-apples.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aleph-qec --example qec_q2_weighted -- [p_data] [p_meas] [shots] [seed]
//! # defaults: p_data=0.02 p_meas=0.06 shots=300000 seed=2024
//! ```

use std::time::Instant;

use aleph_qec::{
    build_dem, run_dem_experiment, Decoder, DetectorErrorModel, MwpmDecoder, SurfaceCode, Syndrome,
    UnionFindDecoder,
};

const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p_data: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let p_meas: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.06);
    let shots: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!("# Q2-02 weighted UF: p_data={p_data} p_meas={p_meas} shots={shots} seed={seed}");
    println!(
        "d,detectors,shots,rate_uf,ci_uf,rate_wuf,ci_wuf,rate_mwpm,ci_mwpm,\
         improve_abs,improve_rel,closes_gap_frac,uf_syn_per_s,wuf_syn_per_s,wuf_over_uf"
    );

    for &d in DISTANCES {
        let dem = cell_dem(d, p_data, p_meas);

        // Same seed ⇒ identical sampled shots for all three decoders.
        let uf = run_dem_experiment(&dem, shots, &UnionFindDecoder::new(&dem).unwrap(), seed)
            .expect("uf cell");
        let wuf = run_dem_experiment(
            &dem,
            shots,
            &UnionFindDecoder::new_weighted(&dem).unwrap(),
            seed,
        )
        .expect("weighted-uf cell");
        let mw = run_dem_experiment(&dem, shots, &MwpmDecoder::new(&dem).unwrap(), seed)
            .expect("mwpm cell");

        let improve_abs = uf.rate - wuf.rate;
        let improve_rel = if uf.rate > 0.0 {
            improve_abs / uf.rate
        } else {
            0.0
        };
        // Fraction of the UF→MWPM accuracy gap that weighting closes.
        let gap = uf.rate - mw.rate;
        let closes = if gap.abs() > 1e-12 {
            improve_abs / gap
        } else {
            0.0
        };

        // Throughput: single-thread decode, best of three (mirrors the Q1-05 methodology).
        let syndromes = sample(&dem, 40_000, seed ^ 0xA5A5 ^ d as u64);
        let uf_s = throughput(&UnionFindDecoder::new(&dem).unwrap(), &syndromes);
        let wuf_s = throughput(&UnionFindDecoder::new_weighted(&dem).unwrap(), &syndromes);

        println!(
            "{d},{},{shots},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.0},{:.0},{:.3}",
            dem.detectors,
            uf.rate,
            uf.ci95,
            wuf.rate,
            wuf.ci95,
            mw.rate,
            mw.ci95,
            improve_abs,
            improve_rel,
            closes,
            uf_s,
            wuf_s,
            uf_s / wuf_s,
        );
        eprintln!(
            "  d={d}: uf={:.4e} wuf={:.4e} mwpm={:.4e}  Δ={:.2e} ({:.1}% rel, closes {:.0}% of UF→MWPM gap)  \
             slowdown wuf/uf={:.2}x",
            uf.rate,
            wuf.rate,
            mw.rate,
            improve_abs,
            improve_rel * 100.0,
            closes * 100.0,
            uf_s / wuf_s,
        );
    }
}

fn cell_dem(d: usize, p_data: f64, p_meas: f64) -> DetectorErrorModel {
    let exp = SurfaceCode::new(d).memory_z_experiment(d);
    build_dem(
        &exp.annotated,
        &exp.phenomenological_mechanisms(p_data, p_meas),
    )
    .unwrap()
}

/// Single-thread decoded syndromes/second, best of three timed runs.
fn throughput(dec: &dyn Decoder, syndromes: &[Syndrome]) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let mut sink = 0u64;
        for s in syndromes {
            sink ^= dec
                .decode(s)
                .observable_flips
                .first()
                .copied()
                .unwrap_or(false) as u64;
        }
        std::hint::black_box(sink);
        best = best.min(t.elapsed().as_secs_f64());
    }
    syndromes.len() as f64 / best
}

/// Sample `shots` syndromes from `dem` (Bernoulli per mechanism), deterministically per seed.
fn sample(dem: &DetectorErrorModel, shots: usize, seed: u64) -> Vec<Syndrome> {
    (0..shots as u64)
        .map(|s| {
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
            Syndrome::from_bits(&det)
        })
        .collect()
}
