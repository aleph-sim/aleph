//! Q2-03 speed/accuracy **Pareto** sweep: unweighted Union-Find (Q2-01) vs weighted Union-Find
//! (Q2-02) vs MWPM (Q1) on shared surface-code memory DEMs, measured on **both** axes — logical
//! error rate and single-thread decode throughput (syndromes/second) — across distances and noise
//! strengths.
//!
//! Unlike the Q2-02 example (which timed only the two UF variants), this also times MWPM, so every
//! cell carries an errors/second *and* a logical-error-rate number for all three decoders. That is
//! the data the Q2-03 report (`docs/perf/qec-q2-unionfind.md`) turns into a Pareto front and a
//! "when to use which / which goes to hardware" recommendation.
//!
//! Three series are emitted (column `series`), all apples-to-apples (same seed ⇒ same sampled
//! shots for all three decoders within a cell):
//!
//! * `perd-uniform` — `p_data = p_meas = 3 %`, sweeping `d ∈ {3,5,7,9,11}` (the standard model).
//! * `perd-asym`    — `p_data = 2 %, p_meas = 6 %`, same distances (heterogeneous edge weights,
//!   where weighted growth helps most).
//! * `psweep-d7`    — fixed `d = 7`, uniform `p ∈ {1,2,3,4,5} %` (the noise-strength axis: how the
//!   rate gap and the throughput gap move with physical error rate, on and off threshold).
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aleph-qec --example qec_q2_pareto -- [acc_shots] [tput_shots] [seed]
//! # defaults: acc_shots=200000 tput_shots=20000 seed=2024
//! ```

use std::time::Instant;

use aleph_qec::{
    build_dem, run_dem_experiment, Decoder, DetectorErrorModel, MwpmDecoder, SurfaceCode, Syndrome,
    UnionFindDecoder,
};

const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];
const P_SWEEP: &[f64] = &[0.01, 0.02, 0.03, 0.04, 0.05];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let acc_shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let tput_shots: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!(
        "# Q2-03 Pareto: acc_shots={acc_shots} tput_shots={tput_shots} seed={seed}\n\
         # decoders: uf (unweighted UF, Q2-01), wuf (weighted UF, Q2-02), mwpm (Q1)"
    );
    println!(
        "series,d,detectors,p_data,p_meas,acc_shots,avg_defects,\
         rate_uf,ci_uf,rate_wuf,ci_wuf,rate_mwpm,ci_mwpm,\
         uf_syn_per_s,wuf_syn_per_s,mwpm_syn_per_s"
    );

    // Standard uniform model, per distance.
    for &d in DISTANCES {
        cell("perd-uniform", d, 0.03, 0.03, acc_shots, tput_shots, seed);
    }
    // Asymmetric model (p_meas > p_data), per distance.
    for &d in DISTANCES {
        cell("perd-asym", d, 0.02, 0.06, acc_shots, tput_shots, seed);
    }
    // Noise-strength sweep at fixed mid distance, uniform model.
    for &p in P_SWEEP {
        cell("psweep-d7", 7, p, p, acc_shots, tput_shots, seed);
    }
}

/// Measure one (model, d) cell: accuracy for all three decoders on the same shots, plus
/// single-thread throughput for all three on a fresh sampled batch.
fn cell(
    series: &str,
    d: usize,
    p_data: f64,
    p_meas: f64,
    acc_shots: u64,
    tput_shots: usize,
    seed: u64,
) {
    let dem = cell_dem(d, p_data, p_meas);

    let uf = UnionFindDecoder::new(&dem).unwrap();
    let wuf = UnionFindDecoder::new_weighted(&dem).unwrap();
    let mw = MwpmDecoder::new(&dem).unwrap();

    // Accuracy — same seed ⇒ identical sampled shots for all three decoders.
    let r_uf = run_dem_experiment(&dem, acc_shots, &uf, seed).expect("uf acc");
    let r_wuf = run_dem_experiment(&dem, acc_shots, &wuf, seed).expect("wuf acc");
    let r_mw = run_dem_experiment(&dem, acc_shots, &mw, seed).expect("mwpm acc");

    // Throughput — one shared batch, single-thread decode, best of three (Q1-05 methodology).
    let syndromes = sample(&dem, tput_shots, seed ^ 0xA5A5 ^ d as u64);
    let avg_defects =
        syndromes.iter().map(|s| s.weight()).sum::<usize>() as f64 / syndromes.len() as f64;
    let uf_s = throughput(&uf, &syndromes);
    let wuf_s = throughput(&wuf, &syndromes);
    let mw_s = throughput(&mw, &syndromes);

    println!(
        "{series},{d},{},{p_data},{p_meas},{acc_shots},{:.2},\
         {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.0},{:.0},{:.0}",
        dem.detectors,
        avg_defects,
        r_uf.rate,
        r_uf.ci95,
        r_wuf.rate,
        r_wuf.ci95,
        r_mw.rate,
        r_mw.ci95,
        uf_s,
        wuf_s,
        mw_s,
    );
    eprintln!(
        "  [{series}] d={d} p=({p_data},{p_meas}) def={avg_defects:.1}: \
         rate uf={:.4e} wuf={:.4e} mwpm={:.4e} | syn/s uf={uf_s:.0} wuf={wuf_s:.0} mwpm={mw_s:.0} \
         (uf/mwpm={:.1}x)",
        r_uf.rate,
        r_wuf.rate,
        r_mw.rate,
        uf_s / mw_s,
    );
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
