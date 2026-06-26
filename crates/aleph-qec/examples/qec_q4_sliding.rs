//! Q4-01 sliding-window streaming decoder: logical-error rate vs the full-batch decode as the window
//! `W` grows (commit `C = d` fixed), on a long memory-Z stream. Shows the rate converging to batch
//! within CI once the buffer `W − C` is an adequate (d-dependent) size, and reports the per-window
//! working set (bounded, independent of stream length).
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aleph-qec --example qec_q4_sliding -- [shots] [seed]
//! # defaults: shots=200000 seed=2024
//! ```

use aleph_qec::{
    build_dem, run_dem_experiment, SlidingWindowDecoder, SurfaceCode, UnionFindDecoder,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.03);

    eprintln!("# Q4-01 sliding window vs batch, memory-Z stream, p={p}, commit C=d");
    println!("d,rounds,commit,window,buffer,batch_rate,sw_rate,abs_delta,combined_ci,within_ci,window_dets");

    for &d in &[3usize, 5, 7] {
        let rounds = 6 * d; // long stream
        let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        let det_rounds = exp.detector_rounds();

        let batch = UnionFindDecoder::new(&dem).unwrap();
        let b = run_dem_experiment(&dem, shots, &batch, seed).expect("batch");

        let c = d;
        for w in [c + 1, c + d / 2 + 1, 2 * c, 3 * c, 4 * c] {
            if w > rounds + 1 {
                continue;
            }
            let sw = SlidingWindowDecoder::new(dem.clone(), det_rounds.clone(), w, c);
            let wdets = sw.max_window_detectors();
            let s = run_dem_experiment(&dem, shots, &sw, seed).expect("sliding");
            let delta = (s.rate - b.rate).abs();
            let cci = b.ci95 + s.ci95;
            let within = delta <= cci;
            println!(
                "{d},{rounds},{c},{w},{},{:.6},{:.6},{:.6},{:.6},{within},{wdets}",
                w - c,
                b.rate,
                s.rate,
                delta,
                cci,
            );
            eprintln!(
                "  d={d} W={w} buf={}: batch={:.4e} sw={:.4e} Δ={delta:.2e} ci={cci:.2e} within={within}",
                w - c,
                b.rate,
                s.rate
            );
        }
    }
}
