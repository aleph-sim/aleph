//! Q6-22 (streaming LER) — emit finite memory-Z experiments as per-round detector streams + the
//! software sliding-window logical-error-rate baseline, so the on-board streaming decoder's LER can be
//! checked against the boundary-aware software `SlidingWindowDecoder` within Monte-Carlo CI.
//!
//! This complements `qec_q6_stream_cosim` (continuous stream → validity + throughput). Here each shot
//! is a **finite** memory experiment of `rounds` rounds with a true logical flip. The software baseline
//! decodes each with the full sliding-window decode (interior windows + a boundary-aware LAST window
//! that commits every remaining round with a real time boundary). The on-board decoder instead runs
//! interior windows + a zero-drain to commit the tail (Q6-20's steady-state-interior wrapper); comparing
//! the two LERs measures whether that drain-based finite handling matches the boundary-aware software —
//! i.e. whether the documented warm-up/drain caveat actually costs accuracy at the operating point.
//!
//! The round length lands on a window boundary: `rounds + 1` detector slices `= W + k·C`.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_stream_ler -- [d] [W] [C] [rounds] [shots] [seed] [p,..]
//!   # defaults: d=3 W=9 C=3 rounds=17 shots=4000 seed=2024 p=0.01,0.02,0.03,0.04,0.05
//!   cargo run --release -p aleph-qec --example qec_q6_stream_ler -- 3 9 3 17 4000 2024 > hw/cosim_stream_ler_d3.vec
//!
//! Output (stdout) — the `.vec` the streaming DMA driver reads in LER mode:
//!   # comment metadata (d/W/C/dpr/slices/...)
//!   P p=<p> shots=<E> sw_rate=<r> sw_ci=<c>        ← one block per p
//!   E <truth>                                       ← start of an experiment; <truth> = true logical
//!   <dpr detector bits>                             ← `slices` such lines, round 0 first

use aleph_qec::{
    build_dem, sample_shots, Decoder, LogicalErrorResult, SlidingWindowDecoder, SurfaceCode,
};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(17);
    let shots: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let seed: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let probs: Vec<f64> = a
        .get(7)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.01, 0.02, 0.03, 0.04, 0.05]);

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem_at =
        |p: f64| build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    let dets = dem_at(0.01).detectors;

    // Group detector ids by slice (round-major): the RTL round handshake. Assert a fixed dpr and that
    // the stream lands on a window boundary (slices = W + k*C) so the last round completes a window.
    let n_slices = det_round.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut by_round: Vec<Vec<usize>> = vec![Vec::new(); n_slices];
    for (dd, &r) in det_round.iter().enumerate() {
        by_round[r].push(dd);
    }
    let dpr = by_round[0].len();
    assert!(
        by_round.iter().all(|r| r.len() == dpr),
        "expected a fixed detectors-per-round for the streaming frame"
    );
    assert_eq!(dets, n_slices * dpr, "detector count must be slices*dpr");
    assert!(
        n_slices >= w && (n_slices - w).is_multiple_of(c),
        "slices ({n_slices}) must be W + k*C (W={w}, C={c}); pick rounds = W + k*C - 1"
    );

    println!("# Q6-22 streaming finite-experiment LER vectors — GENERATED, do not edit.");
    println!(
        "# d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={dets} observables=1 shots={shots} seed={seed}"
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_stream_ler -- {d} {w} {c} {rounds} {shots} {seed}"
    );

    eprintln!("# d={d} W={w} C={c} slices={n_slices} shots={shots} (software SlidingWindowDecoder baseline)");
    eprintln!("#   p      sw_rate      ci95     errors");

    for &p in &probs {
        let dem = dem_at(p);
        assert_eq!(dem.detectors, dets, "p must not change detector count");
        assert_eq!(
            dem.observables, 1,
            "streaming LER assumes a single logical observable"
        );

        let (syndromes, truths) = sample_shots(&dem, shots, seed);

        // Boundary-aware software baseline: the full sliding-window decode (interior + committing last).
        let sw = SlidingWindowDecoder::new(dem.clone(), det_round.clone(), w, c);
        let errs = syndromes
            .iter()
            .zip(&truths)
            .filter(|(syn, truth)| {
                sw.decode(syn)
                    .observable_flips
                    .first()
                    .copied()
                    .unwrap_or(false)
                    != truth.first().copied().unwrap_or(false)
            })
            .count() as u64;
        let res = LogicalErrorResult::new(shots, errs);

        eprintln!(
            "  {p:.3}  {:.4e}  {:.2e}  {}/{}",
            res.rate, res.ci95, errs, shots
        );
        println!(
            "P p={p} shots={shots} sw_rate={} sw_ci={}",
            res.rate, res.ci95
        );

        let mut line = String::with_capacity(dpr);
        for (syn, truth) in syndromes.iter().zip(&truths) {
            println!("E {}", u8::from(truth.first().copied().unwrap_or(false)));
            for round in &by_round {
                line.clear();
                for &dd in round {
                    line.push(if syn.is_fired(dd as u32) { '1' } else { '0' });
                }
                println!("{line}");
            }
        }
    }
}
