//! Q7-06 software precursor — relay-BP **iteration-budget** sweep on the *circuit-level* DEM.
//!
//! The Q7-08 banking-scaling qualification (`docs/qec/asic-architecture.md` § 5) proved that banking
//! is a sublinear, floored latency lever: `cycles = LEGS·ITERS·(GC+GV+7) + (2·GV+GC+1)`, and the
//! `LEGS·ITERS·7` per-iteration pipeline tail is **banking-invariant**. So once banking is maxed out,
//! the only remaining lever on worst-case latency is **ITERS** — cutting the BP iteration budget cuts
//! the floor linearly. The spec (§ 5) states that "the minimum tolerable ITERS at a target LER is
//! exactly what Q7-06 qualifies"; this example is the *software* side of that number.
//!
//! The earlier `qec_q7_budget` sweep ran on the **code-capacity** DEM (one perfect round, independent
//! Z, p≈0.05) at the old 4-leg default. That is the wrong operating point for a latency-budget claim:
//! the ASIC decodes the **circuit-level** DEM (depth-7 syndrome extraction, depolarizing CNOT/idle/
//! init/measure noise, `rounds = d`) in the **sub-threshold** regime (p≈0.001–0.003), at the shipped
//! **6-leg × 10-iter** hardware schedule (`hw/bb_gross_tanner.svh`: `BP_LEGS=6, BP_ITERS=10`). This
//! sweep re-runs the budget study at that operating point with the hardware fixed-point word (Q5.3)
//! and translates each `(legs, iters)` schedule directly into worst-case cycles/latency at the two
//! banking geometries the spec cares about (64/192 shipped-stretch and 144/864 full-parallel).
//!
//! For each schedule it reports LER (vs the full 6×10 relay-BP), whether it is within Monte-Carlo CI
//! of that baseline, and the worst-case latency it would cost — so the smallest ITERS still within CI
//! is the min-ITERS the latency budget can assume, and whether 64/192 or 144/864 meets 1 µs at it.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q7_circuit_budget -- [shots] [seed]
//! # defaults: shots=20000 seed=2024 ; spec-quality wants shots ≥ 1e5 to resolve sub-threshold LER
//! ```

use aleph_qec::{run_dem_experiment, BBCode, CircuitNoise, FixedRelayBp};

/// Hardware word: Q5.3 (M0 verdict — narrowest width within CI of f64 relay-BP).
const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
/// γ disorder range and seed — identical to the M0/M4 software golden.
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// gross code [[144,12,12]] with `rounds = d = 12` — the circuit-level memory experiment.
const ROUNDS: usize = 12;

/// Circuit-level physical error rates: the sub-threshold operating regime (below the ~0.005–0.007
/// circuit-level threshold of this code/decoder), where the ASIC actually runs.
const PS: &[f64] = &[0.001, 0.002, 0.003];

/// Shipped hardware schedule (`bb_gross_tanner.svh`): 6 legs × 10 iters = 60 sweeps.
const BASE_LEGS: usize = 6;
const BASE_ITERS: u32 = 10;

/// Candidate `(legs, iters_per_leg)` schedules. First entry is the shipped 6×10 baseline. Relay-BP's
/// disorder diversity lives in the *legs* (each reseeds γ and relays the messages), so the primary
/// lever we want to cut is ITERS at fixed legs=6; a couple of leg cuts are included to confirm they
/// hurt more per sweep than iter cuts.
const SCHEDULES: &[(usize, u32)] = &[
    (6, 10), // baseline — 60 sweeps
    (6, 8),  // 48 sweeps
    (6, 6),  // 36 sweeps
    (6, 5),  // 30 sweeps
    (6, 4),  // 24 sweeps
    (6, 3),  // 18 sweeps
    (5, 8),  // 40 sweeps — leg cut
    (4, 10), // 40 sweeps — leg cut (old default legs)
    (4, 8),  // 32 sweeps
];

/// Banking geometries from the § 5 qualification table: (label, check groups GC, var groups GV).
/// Worst-case cycles = `sweeps·(GC+GV+7) + (2·GV+GC+1)`; wall-clock @ 600 MHz.
const BANKS: &[(&str, u32, u32)] = &[
    ("64/192", 3, 5),  // shipped-stretch (4× banks): 913 cyc at 60 sweeps
    ("144/864", 1, 1), // full-parallel: 544 cyc at 60 sweeps (the hard floor)
];
const ASIC_MHZ: f64 = 600.0;

/// Worst-case cycle count for `sweeps` message-passing sweeps at banking `(gc, gv)` (§ 5 model).
fn worst_cycles(sweeps: u32, gc: u32, gv: u32) -> u32 {
    sweeps * (gc + gv + 7) + (2 * gv + gc + 1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);

    let code = BBCode::gross();
    eprintln!(
        "# Q7-06 precursor: relay-BP iter-budget sweep on the CIRCUIT-LEVEL DEM, gross [[144,12,12]], \
         rounds={ROUNDS}, depth-7 syndrome extraction, uniform depolarizing noise"
    );
    eprintln!(
        "# shots={shots} seed={seed}, word=Q{}.{}, baseline = ({BASE_LEGS} legs × {BASE_ITERS} iters \
         = {} sweeps, the shipped bb_gross_tanner.svh schedule)",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS,
        BASE_LEGS as u32 * BASE_ITERS
    );
    eprintln!(
        "# worst-case cycles/latency per § 5 banking model: cycles = sweeps·(GC+GV+7) + (2·GV+GC+1), \
         @ {ASIC_MHZ} MHz"
    );

    // CSV header: schedule + LER + one (cycles,latency) pair per banking geometry.
    let bank_cols: Vec<String> = BANKS
        .iter()
        .flat_map(|(b, _, _)| [format!("cyc_{b}"), format!("us_{b}")])
        .collect();
    println!(
        "p,legs,iters,sweeps,rate,ci95,base_rate,ratio_to_base,within_ci,{}",
        bank_cols.join(",")
    );

    for &p in PS {
        let dem = code
            .circuit_level_dem(ROUNDS, CircuitNoise::uniform(p))
            .expect("circuit-level DEM");

        // Baseline: full shipped 6×10 at the hardware word.
        let base = FixedRelayBp::with_budget(
            &dem, BASE_LEGS, BASE_ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS,
        );
        let rb = run_dem_experiment(&dem, shots, &base, seed).expect("baseline");
        eprintln!(
            "p={p}: circuit-level baseline (6×10) LER = {:.3e} ± {:.1e}  ({} detectors, {} mechanisms)",
            rb.rate,
            rb.ci95,
            dem.detectors,
            dem.errors.len()
        );

        for &(legs, iters) in SCHEDULES {
            let dec =
                FixedRelayBp::with_budget(&dem, legs, iters, GAMMA, SEED, MSG_BITS, FRAC_BITS);
            let r = run_dem_experiment(&dem, shots, &dec, seed).expect("budget");
            let sweeps = legs as u32 * iters;
            let ratio = if rb.rate > 0.0 {
                r.rate / rb.rate
            } else {
                f64::NAN
            };
            let within = (r.rate - rb.rate).abs() <= (r.ci95 + rb.ci95);

            let bank_vals: Vec<String> = BANKS
                .iter()
                .flat_map(|(_, gc, gv)| {
                    let cyc = worst_cycles(sweeps, *gc, *gv);
                    [format!("{cyc}"), format!("{:.3}", cyc as f64 / ASIC_MHZ)]
                })
                .collect();
            println!(
                "{p},{legs},{iters},{sweeps},{:.6},{:.6},{:.6},{ratio:.3},{},{}",
                r.rate,
                r.ci95,
                rb.rate,
                within as u8,
                bank_vals.join(",")
            );

            let lat_644 = worst_cycles(sweeps, 3, 5) as f64 / ASIC_MHZ;
            let lat_full = worst_cycles(sweeps, 1, 1) as f64 / ASIC_MHZ;
            eprintln!(
                "   ({legs} legs × {iters:>2}) = {sweeps:>2} sweeps  LER {:.3e} ({ratio:.2}× base){}  \
                 → 64/192 {lat_644:.2} µs, 144/864 {lat_full:.2} µs",
                r.rate,
                if within { "  [within CI]" } else { "" }
            );
        }
    }

    eprintln!(
        "# Smallest ITERS with within_ci=1 across all p is the min-ITERS the latency budget can assume. \
         The 1 µs worst-case budget is met when that ITERS's 64/192 (or 144/864) latency column ≤ 1.00."
    );
}
