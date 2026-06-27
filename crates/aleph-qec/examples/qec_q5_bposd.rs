//! Q5-02 BP+OSD on bivariate-bicycle codes.
//!
//! Two blocks:
//!   1. **gross code**: logical-error rate vs physical rate for plain BP vs BP+OSD (both normalised
//!      min-sum `α=0.875`). OSD fixes the degenerate, wrong-coset failures BP leaves behind — the
//!      gain is largest at low `p` and grows with code size.
//!   2. **threshold**: a BB family with *known growing distance* — `[[72,12,6]]` (ℓ=6) and
//!      `[[144,12,12]]` (ℓ=12) (m=6, `A=x³+y+y²`, `B=y³+x+x²`; d from Bravyi et al.) — under
//!      independent-`Z` code-capacity noise, BP+OSD, whose logical-rate curves cross at a threshold.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q5_bposd -- [shots] [osd_order] [seed]
//! # defaults: shots=20000 osd_order=10 seed=2024
//! ```

use aleph_qec::{run_dem_experiment, BBCode, BpDecoder, OsdDecoder, DEFAULT_MAX_ITER};

const ALPHA: f64 = 0.875; // normalised min-sum (plain α=1 over-converges to wrong cosets on qLDPC)

fn bb(l: usize) -> BBCode {
    BBCode::new(l, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let order: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!("# Q5-02 BP+OSD, independent-Z code capacity, normalised min-sum α={ALPHA}, osd_order={order}");

    // ---- block 1: gross code, BP vs BP+OSD ----
    println!("# gross [[144,12,12]]: plain BP vs BP+OSD (α={ALPHA}), logical rate vs p");
    println!("p,shots,bp_rate,bp_ci,bposd_rate,bposd_ci,improvement");
    let gross = bb(12);
    for &p in &[0.01_f64, 0.02, 0.03, 0.04, 0.05, 0.06] {
        let dem = gross.code_capacity_dem(p);
        let bp = BpDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA);
        let osd = OsdDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA, order);
        let rb = run_dem_experiment(&dem, shots, &bp, seed).expect("bp");
        let ro = run_dem_experiment(&dem, shots, &osd, seed).expect("bposd");
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
            "  p={p}: BP={:.3e}±{:.0e}  BP+OSD={:.3e}±{:.0e}  ({imp:.2}× better)",
            rb.rate, rb.ci95, ro.rate, ro.ci95
        );
    }

    // ---- block 2: threshold over the known-growing-distance family ----
    println!();
    println!("# threshold: BB family [[72,12,6]] (ℓ=6) & [[144,12,12]] (ℓ=12), BP+OSD, rate vs p");
    println!("l,n,d,p,shots,logical_rate,ci95");
    let ps = [0.06_f64, 0.07, 0.08, 0.085, 0.09, 0.095, 0.10, 0.11];
    let mut rate6 = Vec::new();
    let mut rate12 = Vec::new();
    for &(l, d) in &[(6usize, 6usize), (12, 12)] {
        let code = bb(l);
        for &p in &ps {
            let dem = code.code_capacity_dem(p);
            let osd = OsdDecoder::with_params(&dem, DEFAULT_MAX_ITER, ALPHA, order);
            let r = run_dem_experiment(&dem, shots, &osd, seed).expect("bposd");
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
            if l == 6 {
                rate6.push(r.rate)
            } else {
                rate12.push(r.rate)
            }
        }
    }

    // Threshold = the p where the d=12 curve crosses above the d=6 curve (larger code stops helping).
    // Linearly interpolate the bracketing grid points where (rate12 − rate6) changes sign.
    let mut p_th = f64::NAN;
    for i in 1..ps.len() {
        let (d0, d1) = (rate12[i - 1] - rate6[i - 1], rate12[i] - rate6[i]);
        if d0 <= 0.0 && d1 > 0.0 {
            p_th = ps[i - 1] + (ps[i] - ps[i - 1]) * (-d0) / (d1 - d0);
            break;
        }
    }
    println!();
    println!("threshold_p_th,{p_th:.4}");
    eprintln!(
        "# threshold p_th ≈ {p_th:.4} (d=6/d=12 crossing). Single-Pauli code capacity; cf. surface \
         code ~0.109 same channel — a qLDPC code at rate 1/12 with a comparable code-capacity threshold."
    );
}
