//! Q7-07 — non-convergence rate, attributable fraction, and fallback evaluation.
//!
//! Relay-BP occasionally emits a hard decision violating the syndrome; the RTL flags it
//! (`valid_flag`, `hw/bp_relay_banked.sv:968`) and emits it anyway. This measures what that costs.
//!
//! A direct A/B on campaign LER is statistically hopeless — at p=0.003 the LER CI is ±1.13e-4 at
//! 10⁶ shots while the non-convergence rate is order 0.1 %, so a fallback's effect sits orders of
//! magnitude under the noise floor. So the measurement is conditional:
//!
//!   L1  r(p) = P(valid = 0), with CI.
//!   L2  A(p) = (# logical errors with valid=0) / (# logical errors) — the HARD CEILING on any
//!       fallback, since a fallback only ever acts on valid=0 shots. Also P(err | valid) both ways.
//!   L3  conditional rescue on the retained valid=0 corpus (see the `candidates` subcommand),
//!       propagated back as ΔLER(p) = r(p) · [P(err|v=0) − P(err|v=0, fallback)].
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q7_nonconv -- block  [rounds] [shots] [seed] [out_prefix]
//!   cargo run --release -p aleph-qec --example qec_q7_nonconv -- window [rounds] [shots] [seed]
//!   cargo run --release -p aleph-qec --example qec_q7_nonconv -- candidates <corpus-file>
//!   # block defaults:  rounds=1  shots=1000000 seed=2024 out_prefix=q7-07
//!   # window defaults: rounds=12 shots=20000   seed=2024

use aleph_qec::{sample_shots, BBCode, CircuitNoise, FixedRelayBp, LogicalErrorResult};
use rayon::prelude::*;

const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
const LEGS: usize = 6;
const ITERS: u32 = 10;
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// The Q7-06 on-silicon campaign points (rounds=1 vehicle).
const BLOCK_PS: &[f64] = &[0.003, 0.005, 0.007];
/// Shots per chunk — bounds peak memory; the stream is `shots` total across chunks.
const CHUNK: u64 = 1_000_000;
/// Level-3 corpus target: retained non-converged shots per operating point.
const CORPUS_TARGET: usize = 1000;

fn mispredicted(pred: &[bool], truth: &[bool], observables: usize) -> bool {
    (0..observables)
        .any(|o| pred.get(o).copied().unwrap_or(false) != truth.get(o).copied().unwrap_or(false))
}

/// Per-chunk seed: independent, deterministic, reproducible from `(seed, chunk)`.
fn chunk_seed(seed: u64, chunk: u64) -> u64 {
    seed ^ chunk.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[derive(Default, Clone, Copy)]
struct Counts {
    shots: u64,
    nonconv: u64,
    err_total: u64,
    err_nonconv: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("block") => {
            let rounds = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1usize);
            let shots = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000u64);
            let seed = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024u64);
            let prefix = args.get(4).cloned().unwrap_or_else(|| "q7-07".to_string());
            run_block(rounds, shots, seed, &prefix);
        }
        other => {
            eprintln!("unknown subcommand {other:?}");
            eprintln!("usage: qec_q7_nonconv -- block|window|candidates ...");
            std::process::exit(2);
        }
    }
}

fn run_block(rounds: usize, shots: u64, seed: u64, prefix: &str) {
    let code = BBCode::gross();
    eprintln!(
        "# Q7-07 block path: gross [[144,12,12]] circuit-level rounds={rounds} shots={shots} \
         seed={seed} schedule={LEGS}x{ITERS} word=Q{}.{}",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS
    );
    println!("p,shots,r,r_ci95,ler,ler_ci95,p_err_given_nonconv,p_err_given_conv,attributable,iters_mean,iters_p50,iters_p99,iters_max,retained");

    for &p in BLOCK_PS {
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .expect("circuit-level DEM");
        let fx = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);

        let mut c = Counts::default();
        let mut iters_hist: Vec<u32> = Vec::new();
        let mut corpus: Vec<(Vec<u32>, Vec<bool>, u64)> = Vec::new();

        let mut done: u64 = 0;
        let mut chunk_idx: u64 = 0;
        while done < shots {
            let n = CHUNK.min(shots - done);
            let cs = chunk_seed(seed, chunk_idx);
            let (syndromes, truths) = sample_shots(&dem, n, cs);
            let out: Vec<(bool, bool, u32)> = syndromes
                .par_iter()
                .zip(&truths)
                .map(|(syn, truth)| {
                    let (_ehat, flips, valid) = fx.decode_fixed_ehat(syn);
                    let (_conv, iters) = fx.iters_to_valid(syn);
                    (valid, mispredicted(&flips, truth, dem.observables), iters)
                })
                .collect();

            for (i, &(valid, err, iters)) in out.iter().enumerate() {
                c.shots += 1;
                iters_hist.push(iters);
                if err {
                    c.err_total += 1;
                }
                if !valid {
                    c.nonconv += 1;
                    if err {
                        c.err_nonconv += 1;
                    }
                    if corpus.len() < CORPUS_TARGET {
                        corpus.push((syndromes[i].fired.clone(), truths[i].clone(), cs));
                    }
                }
            }
            done += n;
            chunk_idx += 1;
            eprintln!(
                "p={p}: {done}/{shots} shots, nonconv {} ({:.4}%), corpus {}",
                c.nonconv,
                100.0 * c.nonconv as f64 / c.shots as f64,
                corpus.len()
            );
        }

        write_corpus(prefix, rounds, p, shots, seed, &corpus);
        report_block(p, &c, &mut iters_hist, corpus.len());
    }
}

fn report_block(p: f64, c: &Counts, iters: &mut [u32], retained: usize) {
    let r = LogicalErrorResult::new(c.shots, c.nonconv);
    let ler = LogicalErrorResult::new(c.shots, c.err_total);
    let conv_shots = c.shots - c.nonconv;
    let p_err_nc = if c.nonconv > 0 {
        c.err_nonconv as f64 / c.nonconv as f64
    } else {
        f64::NAN
    };
    let p_err_c = if conv_shots > 0 {
        (c.err_total - c.err_nonconv) as f64 / conv_shots as f64
    } else {
        f64::NAN
    };
    // The ceiling: a fallback only ever acts on valid=0 shots, so even a perfect one removes at
    // most this fraction of the logical errors.
    let attributable = if c.err_total > 0 {
        c.err_nonconv as f64 / c.err_total as f64
    } else {
        f64::NAN
    };
    iters.sort_unstable();
    let pct = |q: f64| -> u32 {
        if iters.is_empty() {
            return 0;
        }
        let i = ((iters.len() as f64 - 1.0) * q).round() as usize;
        iters[i]
    };
    let mean = iters.iter().map(|&x| x as f64).sum::<f64>() / iters.len().max(1) as f64;
    println!(
        "{p},{},{:.8},{:.8},{:.8},{:.8},{:.6},{:.8},{:.6},{:.2},{},{},{},{retained}",
        c.shots,
        r.rate,
        r.ci95,
        ler.rate,
        ler.ci95,
        p_err_nc,
        p_err_c,
        attributable,
        mean,
        pct(0.50),
        pct(0.99),
        iters.last().copied().unwrap_or(0)
    );
    eprintln!(
        "p={p}: r={:.4e} ±{:.1e} | LER={:.4e} | P(err|v=0)={p_err_nc:.4} P(err|v=1)={p_err_c:.4e} \
         | A={attributable:.4}",
        r.rate, r.ci95, ler.rate
    );
}

fn write_corpus(
    prefix: &str,
    rounds: usize,
    p: f64,
    shots: u64,
    seed: u64,
    corpus: &[(Vec<u32>, Vec<bool>, u64)],
) {
    use std::io::Write;
    let path = format!("{prefix}-p{:03.0}.corpus", p * 1000.0);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create corpus"));
    writeln!(
        f,
        "# mode=block rounds={rounds} p={p} shots={shots} seed={seed} retained={}",
        corpus.len()
    )
    .expect("write corpus");
    writeln!(f, "# dets;truth").expect("write corpus");
    for (dets, truth, _cs) in corpus {
        let d: Vec<String> = dets.iter().map(|x| x.to_string()).collect();
        let t: Vec<String> = truth.iter().map(|&b| u8::from(b).to_string()).collect();
        writeln!(f, "{};{}", d.join(" "), t.join(" ")).expect("write corpus");
    }
    eprintln!("# wrote {path} ({} retained shots)", corpus.len());
}
