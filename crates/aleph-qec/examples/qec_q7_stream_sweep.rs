//! Q7-04 M9a — the (W, C) × seam × p sliding-window LER sweep that picks the streaming
//! configuration M9b bakes into RTL.
//!
//! For each physical error rate the same Monte-Carlo shots (same seed ⇒ same `sample_shots`
//! stream) are decoded by the batch `FixedRelayBp` (the reference — windowing cost, not sampling
//! noise, is what the comparison isolates) and by every (W, C, seam) sliding-window
//! configuration. Reported per cell: windowed LER ± CI, batch LER ± CI, within-CI flag, the
//! fraction of shots with ≥1 non-converged window (feeds Q7-07), and the fraction with a
//! non-zero final residual.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q7_stream_sweep -- [rounds] [shots] [seed]
//!   # defaults: rounds=12 shots=20000 seed=2024
//!
//! Decision rule (spec § 4-M9a): pick the smallest (W, C, seam) whose LER stays within the batch
//! CI at every p (or a documented, explicitly-accepted gap). Soft priors ship only on a clear win.

use aleph_qec::{
    sample_shots, BBCode, CircuitNoise, Correction, FixedRelayBp, LogicalErrorResult, SeamMode,
    SlidingWindowBp,
};
use rayon::prelude::*;

const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
const LEGS: usize = 6;
const ITERS: u32 = 10;
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// Circuit-level per-cycle rates around the relay-BP threshold (~0.3 %).
const PS: &[f64] = &[0.001, 0.003, 0.005];
/// The (W, C) grid from the design spec § 4-M9a.
const WC: &[(usize, usize)] = &[(3, 1), (4, 2), (6, 2), (6, 3)];

fn mispredicted(pred: &Correction, truth: &[bool], observables: usize) -> bool {
    (0..observables).any(|o| {
        pred.observable_flips.get(o).copied().unwrap_or(false)
            != truth.get(o).copied().unwrap_or(false)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let shots: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    let code = BBCode::gross();
    eprintln!(
        "# gross [[144,12,12]] circuit-level rounds={rounds}, shots={shots}, seed={seed}, \
         schedule={LEGS}x{ITERS}, word=Q{}.{}",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS
    );
    println!("p,W,C,seam,ler_win,ci_win,ler_batch,ci_batch,within_ci,nonconv_frac,resid_frac");

    for &p in PS {
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .expect("circuit-level DEM");
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let (syndromes, truths) = sample_shots(&dem, shots, seed);

        // Batch reference on the same shots.
        let batch = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
        let batch_errs = syndromes
            .par_iter()
            .zip(&truths)
            .filter(|(syn, truth)| mispredicted(&batch.decode_fixed(syn).0, truth, dem.observables))
            .count() as u64;
        let rb = LogicalErrorResult::new(shots, batch_errs);
        eprintln!("p={p}: batch LER {:.3e} ± {:.1e}", rb.rate, rb.ci95);

        for &(w, c) in WC {
            for seam in [SeamMode::ResidualOnly, SeamMode::SoftPriors] {
                let sw = SlidingWindowBp::new(dem.clone(), dr.clone(), w, c).with_seam(seam);
                let results: Vec<_> = syndromes
                    .par_iter()
                    .map(|syn| sw.decode_stream(syn))
                    .collect();
                let errs = results
                    .iter()
                    .zip(&truths)
                    .filter(|((corr, _), truth)| mispredicted(corr, truth, dem.observables))
                    .count() as u64;
                let rw = LogicalErrorResult::new(shots, errs);
                let nonconv = results.iter().filter(|(_, s)| s.nonconverged > 0).count();
                let resid = results.iter().filter(|(_, s)| s.residual > 0).count();
                let within = (rw.rate - rb.rate).abs() <= (rw.ci95 + rb.ci95);
                let seam_name = match seam {
                    SeamMode::ResidualOnly => "residual",
                    SeamMode::SoftPriors => "soft",
                };
                println!(
                    "{p},{w},{c},{seam_name},{:.6},{:.6},{:.6},{:.6},{},{:.4},{:.4}",
                    rw.rate,
                    rw.ci95,
                    rb.rate,
                    rb.ci95,
                    within as u8,
                    nonconv as f64 / shots as f64,
                    resid as f64 / shots as f64
                );
                eprintln!(
                    "p={p} W={w} C={c} {seam_name}: LER {:.3e} ± {:.1e} {} | nonconv {:.2}% | resid>0 {:.2}%",
                    rw.rate,
                    rw.ci95,
                    if within { "[within CI]" } else { "[DIFFERS]" },
                    nonconv as f64 / shots as f64 * 100.0,
                    resid as f64 / shots as f64 * 100.0
                );
            }
        }
    }
}
