//! Q7-02 M5-followup — the **OSD-0 tail** for the fixed-point relay-BP hardware decoder.
//!
//! Relay-BP on a degenerate qLDPC code occasionally leaves a hard decision that does not satisfy
//! `H ê = s` (a guaranteed failure). [`FixedRelayBpOsd`] adds an **OSD-0** software escape (Fossorier–
//! Lin, reliability-ordered GF(2) most-reliable-basis) that turns exactly those failure shots into a
//! guaranteed syndrome-consistent low-weight error, leaving relay-BP's valid decisions untouched.
//!
//! OSD-0's Gauss–Jordan is data-dependent and variable-latency — deliberately NOT on the RTL datapath
//! (the reason Q7-02 chose relay-BP over BP+OSD). So it could only ever be a **rare slow-path escape**:
//! the RTL emits `valid_flag`, the PS runs the tail only on the `!valid_flag` shots. This example
//! measures the two numbers that decide whether that tail is worth building: the **tail-rate** (fraction
//! of shots the PS pays for) and the **LER delta**, sweeping the OSD **order** at both **code-capacity**
//! and **circuit-level** (depth-7 syndrome extraction, gate noise).
//!
//! **Finding (see `docs/perf/qec-q7-fixed-bp.md`):** OSD-**0** does *not* cut LER — neutral at code
//! capacity, slightly *worse* circuit-level (a valid but wrong-coset decode beats BP's invalid guess
//! less often than it loses). The LER win needs a large combination sweep (order ≈ 12, `2^12` GF(2)
//! solves/shot), and the fixed Q5.3 tail tracks the float tail at every order — so quantisation is not
//! the limiter, the OSD order is. Net: no hardware-tractable OSD tail helps → Q7-02 ships pure relay-BP.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q7_osd -- [mode] [shots] [seed] [rounds]
//! # mode = capacity (default) | circuit ; defaults: shots=20000 seed=2024 rounds=6 (circuit only)
//! ```

use aleph_qec::{
    run_dem_experiment, sample_shots, BBCode, CircuitNoise, Decoder, FixedRelayBp, FixedRelayBpOsd,
    LogicalErrorResult, RelayBpDecoder, RelayBpOsdDecoder,
};

/// Hardware word: Q5.3 (M0 verdict). The tail is orthogonal to the front-end schedule; the front-end
/// here is the canonical 4-leg × 25-iter relay-BP (`FixedRelayBp::new`), the strongest golden.
const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;

/// Code-capacity physical error rates (measurable LER without floor-clearing shot counts).
const PS_CAPACITY: &[f64] = &[0.03, 0.04, 0.05, 0.06];
/// Circuit-level rates run lower (the depth-7 gadget compounds per-gate error into far more detectors).
const PS_CIRCUIT: &[f64] = &[0.001, 0.002, 0.003, 0.004];

/// One measurement point: plain fixed relay-BP vs the OSD-0-tailed decoder over the same shots, plus the
/// tail-rate (how often OSD actually ran = the relay-BP failure fraction = the PS slow-path cost).
fn measure(
    dem: &aleph_qec::DetectorErrorModel,
    shots: u64,
    seed: u64,
    order: usize,
) -> (LogicalErrorResult, LogicalErrorResult, u64) {
    let plain = FixedRelayBp::new(dem, MSG_BITS, FRAC_BITS);
    let osd = FixedRelayBpOsd::new(dem, MSG_BITS, FRAC_BITS, order);
    let (syndromes, truths) = sample_shots(dem, shots, seed);
    let (mut plain_err, mut osd_err, mut tail) = (0u64, 0u64, 0u64);
    for (syn, truth) in syndromes.iter().zip(&truths) {
        if &plain.decode(syn).observable_flips != truth {
            plain_err += 1;
        }
        let (corr, ran) = osd.decode_fixed_osd(syn);
        if ran {
            tail += 1;
        }
        if &corr.observable_flips != truth {
            osd_err += 1;
        }
    }
    (
        LogicalErrorResult::new(shots, plain_err),
        LogicalErrorResult::new(shots, osd_err),
        tail,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("capacity");
    let shots: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let rounds: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    // OSD combination-sweep order (0 = OSD-0, the only hardware-tractable tail). The Q5-05 LER win
    // needed order 12 (2^12 sweep) — this arg lets us confirm how much sweep the win actually requires.
    let order: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    let code = BBCode::gross();
    let circuit = mode == "circuit";
    let ps: &[f64] = if circuit { PS_CIRCUIT } else { PS_CAPACITY };

    eprintln!(
        "# Q7-02 OSD-0 tail on fixed relay-BP (4×25, Q{}.{}), gross [[144,12,12]], mode={mode}{}",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS,
        if circuit {
            format!(" rounds={rounds}")
        } else {
            String::new()
        }
    );
    eprintln!("# shots={shots} seed={seed} osd_order={order} (2^{order} combination sweep); tail-rate = relay-BP failure fraction = PS slow-path cost");

    // Float relay-BP ± OSD as the reference: isolates whether an OSD-0 LER win exists at all (float),
    // and whether it survives the Q5.3 fixed-point quantisation of the LLR that OSD orders columns by.
    println!("mode,p,shots,fx_plain,fx_plain_ci,fx_osd,fx_osd_ci,tail_rate,fl_plain,fl_osd");
    for &p in ps {
        let dem = if circuit {
            code.circuit_level_dem(rounds, CircuitNoise::uniform(p))
                .expect("circuit-level DEM")
        } else {
            code.code_capacity_dem(p)
        };
        let (plain, osd, tail) = measure(&dem, shots, seed, order);
        let tail_rate = tail as f64 / shots as f64;

        // Float reference (same DEM/shots/seed via the harness): plain relay-BP and relay-BP+OSD.
        let fl_plain = run_dem_experiment(&dem, shots, &RelayBpDecoder::new(&dem), seed)
            .expect("float relay-BP");
        let fl_osd = run_dem_experiment(&dem, shots, &RelayBpOsdDecoder::new(&dem, order), seed)
            .expect("float relay-BP+OSD");

        println!(
            "{mode},{p},{shots},{:.6},{:.6},{:.6},{:.6},{:.5},{:.6},{:.6}",
            plain.rate, plain.ci95, osd.rate, osd.ci95, tail_rate, fl_plain.rate, fl_osd.rate
        );
        let verdict = |a: f64, b: f64, ci: f64| {
            if (a - b).abs() <= ci {
                "≈"
            } else if b < a {
                "OSD wins"
            } else {
                "OSD worse"
            }
        };
        eprintln!(
            "p={p}: FIXED plain {:.3e} → +OSD {:.3e} [{}]  tail {:.2}%  |  FLOAT plain {:.3e} → +OSD {:.3e} [{}]",
            plain.rate,
            osd.rate,
            verdict(plain.rate, osd.rate, plain.ci95 + osd.ci95),
            tail_rate * 100.0,
            fl_plain.rate,
            fl_osd.rate,
            verdict(fl_plain.rate, fl_osd.rate, fl_plain.ci95 + fl_osd.ci95),
        );
    }

    eprintln!("# Finding: OSD-0 (order 0) does NOT cut LER — neutral at code capacity, worse circuit-level.");
    eprintln!(
        "# The win needs a large combination sweep (order ~12 = 2^12 GF(2) solves/shot); and FIXED"
    );
    eprintln!("# Q5.3 tracks FLOAT at every order, so quantisation is not the limiter — the OSD order is.");
}
