//! Q7-02 milestone M5 — relay-BP **leg/iteration budget** sweep (the cycle-count lever).
//!
//! M4 got the spatially-unrolled decoder to 301 cycles = `legs·iters·3 + 1` (4 legs × 25 iters), which
//! is 3.16 µs at the KV260's 95.2 MHz — still ~3× over the ~1 µs budget. The dominant term is the
//! schedule length `legs·iters`, so the cheapest cycles to cut are the ones the decode doesn't need.
//! This sweeps `(legs, iters_per_leg)` at the hardware word (Q5.3) and reports each schedule's LER
//! **relative to the full 4×25 relay-BP** plus the RTL cycles/latency it would cost — so the smallest
//! schedule still within Monte-Carlo CI of the full decoder is obvious.
//!
//! Relay-BP's disorder diversity lives in the *legs* (each leg reseeds γ and relays the messages), so
//! a priori cutting legs should hurt more than cutting iters — the sweep quantifies that trade.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q7_budget -- [shots] [seed]
//! # defaults: shots=20000 seed=2024
//! ```

use aleph_qec::{run_dem_experiment, BBCode, FixedRelayBp};

/// Hardware word: Q5.3 (M0 verdict).
const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
/// γ disorder range and seed — identical to the M0/M4 golden so the baseline matches the RTL exactly.
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// Candidate schedules `(legs, iters_per_leg)`. First entry is the full 4×25 baseline.
const SCHEDULES: &[(usize, u32)] = &[
    (4, 25), // baseline — 100 sweeps, 301 cyc
    (4, 20),
    (4, 15),
    (4, 12),
    (4, 10),
    (4, 8),
    (3, 20),
    (3, 15),
    (3, 12),
    (2, 25),
    (2, 20),
    (2, 15),
    (6, 10),
    (5, 12),
];

/// Physical error rates (measurable LER without floor-clearing shot counts).
const PS: &[f64] = &[0.03, 0.04, 0.05];

/// KV260 M4 routed Fmax (MHz) — turns the RTL cycle count into a wall-clock latency estimate.
const KV260_MHZ: f64 = 95.2;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);

    let code = BBCode::gross();
    eprintln!("# Q7-02 M5: relay-BP leg/iter budget sweep, gross [[144,12,12]], independent-Z code capacity");
    eprintln!(
        "# shots={shots} seed={seed}, word=Q{}.{} ; baseline = (4 legs × 25 iters)",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS
    );

    println!("p,legs,iters,sweeps,cycles,latency_us,rate,ci95,base_rate,ratio_to_base,within_ci");
    for &p in PS {
        let dem = code.code_capacity_dem(p);

        // Baseline: full 4×25 at the hardware word.
        let base = FixedRelayBp::with_budget(&dem, 4, 25, GAMMA, SEED, MSG_BITS, FRAC_BITS);
        let rb = run_dem_experiment(&dem, shots, &base, seed).expect("baseline");
        eprintln!(
            "p={p}: baseline (4×25) LER = {:.3e} ± {:.1e}",
            rb.rate, rb.ci95
        );

        for &(legs, iters) in SCHEDULES {
            let dec =
                FixedRelayBp::with_budget(&dem, legs, iters, GAMMA, SEED, MSG_BITS, FRAC_BITS);
            let r = run_dem_experiment(&dem, shots, &dec, seed).expect("budget");
            let sweeps = legs as u32 * iters;
            let cycles = sweeps * 3 + 1; // M4 schedule: S_CHECK+S_VAR+S_SAT per iter, + S_EMIT
            let latency_us = cycles as f64 / KV260_MHZ;
            let ratio = if rb.rate > 0.0 {
                r.rate / rb.rate
            } else {
                f64::NAN
            };
            let within = (r.rate - rb.rate).abs() <= (r.ci95 + rb.ci95);
            println!(
                "{p},{legs},{iters},{sweeps},{cycles},{latency_us:.3},{:.6},{:.6},{:.6},{ratio:.3},{}",
                r.rate, r.ci95, rb.rate, within as u8
            );
            eprintln!(
                "   ({legs} legs × {iters:>2}) = {sweeps:>3} sweeps, {cycles:>3} cyc, {latency_us:.2} µs  \
                 LER {:.3e} ({ratio:.2}× base){}",
                r.rate,
                if within { "  [within CI]" } else { "" }
            );
        }
    }

    eprintln!(
        "# Smallest (legs,iters) with within_ci=1 across all p is the M5 schedule → regenerate the"
    );
    eprintln!("# .svh (BP_LEGS/BP_ITERS) and the M4 RTL uses it unchanged for a direct cycle/latency cut.");
}
