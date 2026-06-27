//! Q4-02 parallel-window decoding + the backlog problem.
//!
//! Demonstrates that decoding a continuous syndrome stream in **two layers of independent windows**
//! (Skoric et al. arXiv:2209.08552; Tan et al. arXiv:2209.09219) breaks the sequential dependency of
//! sliding-window decoding, so throughput scales with worker count and the unprocessed-syndrome
//! **backlog** (Battistel et al. arXiv:2303.00054) stays bounded.
//!
//! Three blocks of output:
//!   1. **correctness** — parallel-window logical-error rate vs full-batch UF, within CI.
//!   2. **throughput** — sustained syndrome-bits/second of the parallel decoder (P cores) vs the
//!      sequential sliding decoder (1 core), on one long stream.
//!   3. **backlog** — a fluid queue simulation driven by the *measured* per-window service time: at a
//!      fixed arrival rate between the two service rates, the sequential backlog grows without bound
//!      while the parallel backlog drains to zero (no unbounded backlog).
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q4_parallel -- [rounds_mult] [shots] [seed]
//! # defaults: rounds_mult=24 (rounds = rounds_mult*d) shots=40000 seed=2024
//! ```

use std::time::Instant;

use aleph_qec::{
    build_dem, run_dem_experiment, ParallelWindowDecoder, SlidingWindowDecoder, SurfaceCode,
    Syndrome, UnionFindDecoder,
};

/// SplitMix64 — deterministic syndrome sampling for the throughput streams.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Sample a DEM-Bernoulli syndrome (each mechanism fires with its prob, XOR its detector support) —
/// the same generative model `run_dem_experiment` uses, so the timed streams are realistic.
fn sample_syndrome(dem: &aleph_qec::DetectorErrorModel, rng: &mut Rng) -> Syndrome {
    let mut lit = vec![false; dem.detectors];
    for e in &dem.errors {
        // 53-bit uniform in [0,1).
        let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        if u < e.prob {
            for &d in &e.dets {
                lit[d as usize] ^= true;
            }
        }
    }
    Syndrome::from_bits(&lit)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds_mult: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let shots: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p = 0.03;
    let threads = rayon::current_num_threads();

    eprintln!(
        "# Q4-02 parallel-window decoding, memory-Z p={p}, rounds={rounds_mult}*d, {threads} rayon threads"
    );

    // ---- block 1: correctness (parallel rate within CI of batch) ----
    println!("# correctness: parallel-window logical rate vs full-batch UF");
    println!("d,rounds,commit,buffer,batch_rate,par_rate,abs_delta,combined_ci,within_ci");
    for &d in &[3usize, 5, 7] {
        let rounds = rounds_mult * d;
        let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        let det_rounds = exp.detector_rounds();

        let batch = UnionFindDecoder::new(&dem).unwrap();
        let b = run_dem_experiment(&dem, shots, &batch, seed).expect("batch");

        let (c, buf) = (d, d);
        let pw = ParallelWindowDecoder::new(dem.clone(), det_rounds.clone(), c, buf);
        let s = run_dem_experiment(&dem, shots, &pw, seed).expect("parallel");

        let delta = (s.rate - b.rate).abs();
        let cci = b.ci95 + s.ci95;
        println!(
            "{d},{rounds},{c},{buf},{:.6},{:.6},{:.6},{:.6},{}",
            b.rate,
            s.rate,
            delta,
            cci,
            delta <= cci
        );
    }

    // ---- block 2 + 3: throughput and backlog ----
    println!();
    println!("# throughput + backlog (one long stream, {threads} cores)");
    println!(
        "d,rounds,windows,bits_per_round,seq_rounds_per_s,par_rounds_per_s,speedup,\
         seq_bits_per_s,par_bits_per_s,window_us,seq_backlog_at_lambda,par_backlog_at_lambda"
    );

    for &d in &[3usize, 5, 7] {
        let rounds = rounds_mult * d;
        let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        let det_rounds = exp.detector_rounds();
        let num_slices = det_rounds.iter().copied().max().unwrap() + 1;
        let bits_per_round = dem.detectors as f64 / num_slices as f64;

        let (c, buf) = (d, d);
        let par = ParallelWindowDecoder::new(dem.clone(), det_rounds.clone(), c, buf);
        let windows = par.num_windows();
        // Sliding decoder with a comparable window (W = C + 2B forward-only buffer of 2B).
        let seq = SlidingWindowDecoder::new(dem.clone(), det_rounds.clone(), c + 2 * buf, c);

        // Time several stream decodes; one decode processes `num_slices` rounds of syndromes.
        let n_streams = 8usize;
        let mut rng = Rng(seed ^ (d as u64).wrapping_mul(0x100));
        let syndromes: Vec<Syndrome> = (0..n_streams)
            .map(|_| sample_syndrome(&dem, &mut rng))
            .collect();

        let t0 = Instant::now();
        for s in &syndromes {
            std::hint::black_box(seq.decode_stream(s).unwrap());
        }
        let seq_secs = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        for s in &syndromes {
            std::hint::black_box(par.decode_stream(s).unwrap());
        }
        let par_secs = t1.elapsed().as_secs_f64();

        let total_rounds = (num_slices * n_streams) as f64;
        let seq_rps = total_rounds / seq_secs;
        let par_rps = total_rounds / par_secs;
        let speedup = par_rps / seq_rps;
        let seq_bps = seq_rps * bits_per_round;
        let par_bps = par_rps * bits_per_round;
        // Single-core per-window service time (sequential decode = sum of window latencies).
        let window_us = seq_secs / (windows * n_streams) as f64 * 1e6;

        // Backlog (fluid queue): pick an arrival rate between the two service rates, where the
        // sequential decoder cannot keep up but the parallel one can.
        let lambda_rounds = 0.5 * (seq_rps + par_rps); // rounds/s, seq_rps < lambda < par_rps
        let seq_backlog = simulate_backlog(lambda_rounds, seq_rps, num_slices);
        let par_backlog = simulate_backlog(lambda_rounds, par_rps, num_slices);

        println!(
            "{d},{rounds},{windows},{bits_per_round:.1},{seq_rps:.0},{par_rps:.0},{speedup:.2},\
             {seq_bps:.0},{par_bps:.0},{window_us:.1},{seq_backlog:.0},{par_backlog:.0}"
        );
        eprintln!(
            "  d={d}: {threads}-core parallel {par_rps:.0} rounds/s ({par_bps:.3e} bits/s), {speedup:.2}x \
             sequential; @ λ={lambda_rounds:.0} rounds/s seq backlog→{seq_backlog:.0} rounds, par backlog→{par_backlog:.0}"
        );
    }

    eprintln!(
        "# backlog: at an arrival rate the 1-core sliding decoder cannot sustain, the {threads}-core \
         parallel decoder keeps the queue bounded (drains to 0). Throughput ∝ cores ⇒ any fixed \
         arrival rate is met with enough workers — the standard fix for the backlog problem."
    );
    eprintln!(
        "# absolute gap: real-time superconducting needs ~1e6 rounds/s (1 µs/round); the per-window \
         graph rebuild keeps us far below that (Q1-03/Q4-01 note) — closing the constant is Q6 (FPGA)."
    );
}

/// Fluid backlog model: rounds arrive at `lambda` rounds/s and are served at `mu` rounds/s over a
/// horizon of `100 * num_slices` rounds, starting from an empty queue. Returns the backlog (rounds
/// still queued) at the end. If `mu >= lambda` the queue drains to 0; otherwise it grows linearly,
/// the hallmark of an unbounded backlog. (Work-conserving, single fluid server at rate `mu` — `mu`
/// already embeds the worker count via the measured `*_rounds_per_s`.)
fn simulate_backlog(lambda: f64, mu: f64, num_slices: usize) -> f64 {
    let horizon_rounds = (100 * num_slices) as f64;
    let total_time = horizon_rounds / lambda; // seconds to emit the horizon
    let steps = 1000;
    let dt = total_time / steps as f64;
    let mut backlog = 0.0f64; // rounds queued but not yet decoded
    for _ in 0..steps {
        backlog += lambda * dt; // arrivals
        backlog = (backlog - mu * dt).max(0.0); // service (work-conserving)
    }
    backlog
}
