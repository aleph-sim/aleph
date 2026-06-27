//! Q5-04 circuit-level DEM for the bivariate-bicycle gross code.
//!
//! Unlike `qec_q5_bposd` (which decodes the *code-capacity* DEM: one perfect round, independent
//! `Z` noise), this drives the **circuit-level** DEM from [`BBCode::circuit_level_dem`]: a
//! `rounds`-cycle memory-X experiment under the Bravyi depth-7 syndrome-extraction schedule with
//! depolarizing CNOT/idle/init/measure noise. The DEM is a space-time hypergraph and is verified
//! edge-for-edge against Stim (`tests/bb_circuit_dem_stim_oracle.rs`).
//!
//! Two blocks:
//!   1. **gross code** ([[144,12,12]], rounds = d = 12): logical-error rate vs physical rate `p`
//!      for plain BP, BP+OSD, and **relay-BP+OSD** (Q5-05 — the strongest decoder here).
//!   2. **code-size comparison**: [[72,12,6]] (d=6, 6 rounds) vs [[144,12,12]] (d=12, 12 rounds)
//!      under relay-BP+OSD, reported **per cycle** (the fair threshold metric — a d=12 memory runs
//!      2× the rounds of d=6), to locate the circuit-level threshold crossing.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q5_circuit_dem -- [shots] [osd_order] [seed]
//! # defaults: shots=1000 osd_order=20 seed=2024
//! ```

use aleph_qec::{
    run_dem_experiment, BBCode, BpDecoder, CircuitNoise, OsdDecoder, RelayBpOsdDecoder,
    DEFAULT_MAX_ITER,
};

const ALPHA: f64 = 0.875; // normalised min-sum
const PS: [f64; 5] = [0.0005, 0.001, 0.0015, 0.002, 0.003];

fn bb(l: usize) -> BBCode {
    BBCode::new(l, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000);
    let order: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!(
        "# Q5-04 circuit-level DEM (depth-7 syndrome extraction), uniform noise, α={ALPHA}, osd_order={order}"
    );

    // ---- block 1: gross code, rounds = d = 12, BP vs BP+OSD ----
    let gross = bb(12);
    {
        let dem = gross
            .circuit_level_dem(12, CircuitNoise::uniform(0.003))
            .unwrap();
        eprintln!(
            "# gross circuit-level DEM (d=12 rounds): {} detectors, {} observables, {} mechanisms",
            dem.detectors,
            dem.observables,
            dem.errors.len()
        );
    }
    println!("# gross [[144,12,12]], rounds=12: logical rate vs p (BP / BP+OSD / relay-BP+OSD)");
    println!("p,shots,bp_rate,bp_ci,bposd_rate,bposd_ci,relayosd_rate,relayosd_ci");
    for &p in &PS {
        let dem = gross
            .circuit_level_dem(12, CircuitNoise::uniform(p))
            .unwrap();
        let bp = BpDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA);
        let osd = OsdDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA, order);
        let ro = RelayBpOsdDecoder::new(&dem, order);
        let rb = run_dem_experiment(&dem, shots, &bp, seed).unwrap();
        let rosd = run_dem_experiment(&dem, shots, &osd, seed).unwrap();
        let rr = run_dem_experiment(&dem, shots, &ro, seed).unwrap();
        println!(
            "{p},{shots},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            rb.rate, rb.ci95, rosd.rate, rosd.ci95, rr.rate, rr.ci95
        );
        eprintln!(
            "  p={p}: BP={:.3e}  BP+OSD={:.3e}  relay-BP+OSD={:.3e}",
            rb.rate, rosd.rate, rr.rate
        );
    }

    // ---- block 2: d=6 vs d=12 under relay-BP+OSD, per-cycle metric ----
    //
    // The fair threshold metric is the logical error rate **per cycle**: a d=12 memory runs 12
    // rounds vs d=6's 6, so the larger code is exposed ~2× longer. p_L,cycle ≈ p_L / rounds (for
    // small p_L). The threshold is the p where the d=12 per-cycle curve crosses below d=6's.
    println!();
    println!(
        "# code-size comparison (relay-BP+OSD), rounds = d: per-shot and per-cycle logical rate"
    );
    println!("l,n,d,p,shots,logical_rate,ci95,per_cycle_rate");
    for &(l, d) in &[(6usize, 6usize), (12, 12)] {
        let code = bb(l);
        for &p in &PS {
            let dem = code.circuit_level_dem(d, CircuitNoise::uniform(p)).unwrap();
            let ro = RelayBpOsdDecoder::new(&dem, order);
            let r = run_dem_experiment(&dem, shots, &ro, seed).unwrap();
            let per_cycle = r.rate / d as f64;
            println!(
                "{l},{},{d},{p},{shots},{:.6},{:.6},{:.6}",
                code.n(),
                r.rate,
                r.ci95,
                per_cycle
            );
            eprintln!(
                "  ℓ={l} d={d} n={} p={p}: {:.4e} ± {:.1e}  (per-cycle {:.4e})",
                code.n(),
                r.rate,
                r.ci95,
                per_cycle
            );
        }
    }
}
