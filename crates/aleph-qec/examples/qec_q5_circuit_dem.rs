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
//!      for plain BP vs BP+OSD.
//!   2. **code-size comparison**: [[72,12,6]] (d=6, 6 rounds) vs [[144,12,12]] (d=12, 12 rounds)
//!      under BP+OSD, to see where (if anywhere in range) the larger code wins.
//!
//! relay-BP (Q5-03) layers on the same DEM but is omitted here for runtime; the decoder comparison
//! is `qec_q5_bposd`/Q5-03's job — this example's point is the *circuit-level DEM* itself.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q5_circuit_dem -- [shots] [osd_order] [seed]
//! # defaults: shots=1000 osd_order=20 seed=2024
//! ```

use aleph_qec::{
    run_dem_experiment, BBCode, BpDecoder, CircuitNoise, OsdDecoder, DEFAULT_MAX_ITER,
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
    println!("# gross [[144,12,12]], rounds=12: logical rate vs p");
    println!("p,shots,bp_rate,bp_ci,bposd_rate,bposd_ci,improvement");
    for &p in &PS {
        let dem = gross
            .circuit_level_dem(12, CircuitNoise::uniform(p))
            .unwrap();
        let bp = BpDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA);
        let osd = OsdDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA, order);
        let rb = run_dem_experiment(&dem, shots, &bp, seed).unwrap();
        let ro = run_dem_experiment(&dem, shots, &osd, seed).unwrap();
        let imp = if ro.rate > 0.0 {
            rb.rate / ro.rate
        } else {
            f64::INFINITY
        };
        println!(
            "{p},{shots},{:.6},{:.6},{:.6},{:.6},{imp:.2}",
            rb.rate, rb.ci95, ro.rate, ro.ci95
        );
        eprintln!(
            "  p={p}: BP={:.3e}±{:.0e}  BP+OSD={:.3e}±{:.0e}  ({imp:.1}× better)",
            rb.rate, rb.ci95, ro.rate, ro.ci95
        );
    }

    // ---- block 2: d=6 vs d=12 under BP+OSD ----
    println!();
    println!("# code-size comparison (BP+OSD), rounds = d: logical rate vs p");
    println!("l,n,d,p,shots,logical_rate,ci95");
    for &(l, d) in &[(6usize, 6usize), (12, 12)] {
        let code = bb(l);
        for &p in &PS {
            let dem = code.circuit_level_dem(d, CircuitNoise::uniform(p)).unwrap();
            let osd = OsdDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA, order);
            let r = run_dem_experiment(&dem, shots, &osd, seed).unwrap();
            println!(
                "{l},{},{d},{p},{shots},{:.6},{:.6}",
                code.n(),
                r.rate,
                r.ci95
            );
            eprintln!(
                "  ℓ={l} d={d} n={} p={p}: {:.4e} ± {:.1e}",
                code.n(),
                r.rate,
                r.ci95
            );
        }
    }
}
