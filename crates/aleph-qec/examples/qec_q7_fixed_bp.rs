//! Q7-02 milestone M0 — fixed-point relay-BP width sweep vs the f64 reference.
//!
//! The RTL/ASIC decoder carries integer fixed-point messages, not f64. This finds the **narrowest
//! message word** whose logical-error rate matches the f64 relay-BP decoder within Monte-Carlo CI on
//! the gross code `[[144,12,12]]` under independent-Z code capacity — the key RTL sizing number.
//!
//! For each candidate `(msg_bits, frac_bits)` it prints the fixed-point LER next to the f64 LER and
//! their ratio, so the smallest word that is "close enough" is obvious.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q7_fixed_bp -- [shots] [seed]
//! # defaults: shots=20000 seed=2024
//! ```

use aleph_qec::{run_dem_experiment, BBCode, FixedRelayBp, RelayBpDecoder};

/// Candidate fixed-point words: (message width bits, fractional bits).
const WIDTHS: &[(u32, u32)] = &[(6, 2), (7, 3), (8, 3), (8, 4), (10, 4), (12, 6)];

/// Physical error rates to sweep (measurable LER without needing floor-clearing shot counts).
const PS: &[f64] = &[0.02, 0.03, 0.04, 0.05];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);

    let code = BBCode::gross();
    eprintln!(
        "# Q7-02 M0: fixed-point relay-BP width sweep, gross [[144,12,12]], independent-Z code capacity"
    );
    eprintln!(
        "# shots={shots} seed={seed}; f64 = RelayBpDecoder::new (α=0.875, 4 legs, γ∈[-0.3,0.9])"
    );

    // CSV: one row per (p, width), plus the f64 baseline per p as width label "f64".
    println!("p,msg_bits,frac_bits,rate,ci95,f64_rate,ratio_to_f64");
    for &p in PS {
        let dem = code.code_capacity_dem(p);

        let f64_dec = RelayBpDecoder::new(&dem);
        let rf = run_dem_experiment(&dem, shots, &f64_dec, seed).expect("f64 relay");
        println!(
            "{p},f64,-,{:.6},{:.6},{:.6},1.00",
            rf.rate, rf.ci95, rf.rate
        );
        eprintln!(
            "p={p}: f64 relay-BP LER = {:.3e} ± {:.1e}",
            rf.rate, rf.ci95
        );

        for &(mb, fb) in WIDTHS {
            let fx = FixedRelayBp::new(&dem, mb, fb);
            let r = run_dem_experiment(&dem, shots, &fx, seed).expect("fixed relay");
            let ratio = if rf.rate > 0.0 {
                r.rate / rf.rate
            } else {
                f64::NAN
            };
            // Within CI of the f64 baseline?
            let within = (r.rate - rf.rate).abs() <= (r.ci95 + rf.ci95);
            println!(
                "{p},{mb},{fb},{:.6},{:.6},{:.6},{ratio:.3}",
                r.rate, r.ci95, rf.rate
            );
            eprintln!(
                "   ({mb:>2},{fb}) fixed LER = {:.3e} ± {:.1e}  ({ratio:.2}× f64){}",
                r.rate,
                r.ci95,
                if within { "  [within CI]" } else { "" }
            );
        }
    }

    eprintln!(
        "# The narrowest (msg_bits,frac_bits) whose LER stays within CI of f64 across all p is the \
         RTL message word to carry (M1+). Wider words waste area; narrower ones lose accuracy."
    );
}
