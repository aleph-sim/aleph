//! Q7-02 — early-termination study for the fixed-point relay-BP decoder (the RTL `early_exit` mode).
//!
//! The RTL runs a *fixed* `LEGS×ITERS` schedule with no early exit, so its worst-case latency is its
//! every-case latency. Standard BP stops the moment the hard decision satisfies the syndrome. That
//! **changes the result** (first valid `ê` instead of the lowest-weight valid over the whole schedule),
//! so it is an algorithm change, not a microarchitecture one — this study answers the two questions that
//! gate shipping it:
//!
//!   1. **Does early-exit hurt LER?** Compares full-schedule vs first-valid LER on the circuit-level DEM,
//!      with Monte-Carlo CI and a within-CI flag.
//!   2. **How much average latency does it buy?** Reports the per-shot iteration-count distribution
//!      (mean / p50 / p99 / max, and the converged fraction) — iterations map directly to silicon cycles.
//!
//! Early-exit never changes worst-case latency (a non-converging shot still runs the full schedule), so
//! it is an average-case / throughput / energy lever, not a hard-real-time-deadline one.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q7_early -- [rounds] [shots] [seed]
//!   # defaults: rounds=1 shots=40000 seed=2024

use aleph_qec::{sample_shots, BBCode, CircuitNoise, FixedRelayBp};

const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
const LEGS: usize = 6;
const ITERS: u32 = 10;
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// Circuit-level per-cycle physical error rates around the ~0.3 % relay-BP threshold.
const PS: &[f64] = &[0.001, 0.002, 0.003, 0.005];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let shots: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    let code = BBCode::gross();
    let sched = LEGS as u32 * ITERS;

    eprintln!(
        "# circuit-level rounds={rounds}, shots={shots}, seed={seed}, schedule={LEGS}×{ITERS}={sched} iters, word=Q{}.{}",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS
    );
    println!("p,ler_full,ci_full,ler_early,ci_early,within_ci,converged_frac,iters_mean,iters_p50,iters_p99,iters_max");

    for &p in PS {
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .expect("circuit-level DEM");

        let full = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
        let early = full.clone().with_early_exit(true);

        let rf = aleph_qec::run_dem_experiment(&dem, shots, &full, seed).expect("full");
        let re = aleph_qec::run_dem_experiment(&dem, shots, &early, seed).expect("early");
        let within = (rf.rate - re.rate).abs() <= (rf.ci95 + re.ci95);

        // Per-shot iteration count to the first valid ê (or full schedule if none).
        let (syndromes, _truths) = sample_shots(&dem, shots, seed);
        let mut iters: Vec<u32> = Vec::with_capacity(syndromes.len());
        let mut converged = 0u64;
        for syn in &syndromes {
            let (ok, n) = full.iters_to_valid(syn);
            if ok {
                converged += 1;
            }
            iters.push(n);
        }
        iters.sort_unstable();
        let mean = iters.iter().map(|&x| x as f64).sum::<f64>() / iters.len() as f64;
        let pct = |q: f64| iters[((iters.len() as f64 * q) as usize).min(iters.len() - 1)];
        let conv_frac = converged as f64 / shots as f64;

        println!(
            "{p},{:.6},{:.6},{:.6},{:.6},{},{:.4},{mean:.2},{},{},{}",
            rf.rate,
            rf.ci95,
            re.rate,
            re.ci95,
            within as u8,
            conv_frac,
            pct(0.50),
            pct(0.99),
            iters[iters.len() - 1]
        );
        eprintln!(
            "p={p}: LER full {:.3e}±{:.1e} vs early {:.3e}±{:.1e} {} | converged {:.1}% | iters mean {mean:.1} p50 {} p99 {} max {} (of {sched})",
            rf.rate,
            rf.ci95,
            re.rate,
            re.ci95,
            if within { "[within CI]" } else { "[DIFFERS]" },
            conv_frac * 100.0,
            pct(0.50),
            pct(0.99),
            iters[iters.len() - 1]
        );
    }
}
