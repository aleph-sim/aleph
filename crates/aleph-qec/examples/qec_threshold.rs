//! Q0-05 surface-code threshold sweep.
//!
//! Runs the memory-Z experiment across a grid of code distances `d` and physical error
//! probabilities `p`, decodes every shot with the external [`PyMatchingOracle`] (real MWPM),
//! and prints one CSV row per `(d, p)` cell. The crossing of the per-distance
//! logical-error-rate curves is the threshold; `scripts/qec_threshold_plot.py` plots it and
//! fits `p_th`.
//!
//! Usage:
//!
//! ```text
//! PYMATCHING_PYTHON=/path/to/venv/bin/python \
//! STIM_PYTHON=/path/to/venv/bin/python \
//!   cargo run --release -p aleph-qec --example qec_threshold -- <analytic|stim> <shots> <seed>
//! ```
//!
//! * `analytic` — decode our own [`build_dem`] model (the default path).
//! * `stim` — decode the DEM that Stim emits for the *same* circuit + noise (acceptance
//!   cross-check). Both should give the same threshold within CI because Q0-03 proved the two
//!   DEMs equal edge-for-edge.
//!
//! The noise is phenomenological with `p_data == p_meas == p` and `rounds == d`.

use std::io::Write;
use std::process::{Command, Stdio};

use aleph_qec::{
    build_dem, run_dem_experiment, run_memory_experiment, DetectorErrorModel,
    PhenomenologicalNoise, PyMatchingOracle, SurfaceCode,
};

/// Code distances swept.
const DISTANCES: &[usize] = &[3, 5, 7, 9];
/// Physical error probabilities, bracketing the phenomenological threshold (~3%).
const PROBS: &[f64] = &[0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let source = args.get(1).map(String::as_str).unwrap_or("analytic");
    let shots: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!(
        "# source={source} shots={shots} seed={seed} rounds=d noise=phenomenological(p_data=p_meas=p)"
    );
    println!("source,d,rounds,p,shots,logical_errors,rate,ci95");

    for &d in DISTANCES {
        let code = SurfaceCode::new(d);
        for &p in PROBS {
            let noise = PhenomenologicalNoise::uniform(p);
            let res = match source {
                "analytic" => run_memory_experiment(&code, &noise, d, shots, seed, |dem| {
                    Ok(PyMatchingOracle::new(dem))
                })
                .expect("analytic sweep cell"),
                "stim" => run_stim_cell(&code, d, p, shots, seed),
                other => panic!("unknown source `{other}` (expected analytic|stim)"),
            };
            println!(
                "{source},{d},{d},{p},{},{},{},{}",
                res.shots, res.logical_errors, res.rate, res.ci95
            );
            eprintln!(
                "  d={d} p={p:.3}: rate={:.4e} ± {:.1e} ({} errs / {} shots)",
                res.rate, res.ci95, res.logical_errors, res.shots
            );
        }
    }
}

/// Decode one `(d, p)` cell against the DEM that Stim emits for the same circuit + noise.
fn run_stim_cell(
    code: &SurfaceCode,
    d: usize,
    p: f64,
    shots: u64,
    seed: u64,
) -> aleph_qec::LogicalErrorResult {
    let exp = code.memory_z_experiment(d);
    // Sanity: our analytic DEM equals Stim's (Q0-03), so the threshold must match the analytic
    // sweep. We decode Stim's DEM here to demonstrate it end-to-end.
    let stim_text = stim_dem(&exp.stim_program(p, p)).expect("stim subprocess");
    let stim = DetectorErrorModel::parse(&stim_text).expect("parse stim dem");

    // Cross-check the edge counts line up with our analytic model (cheap guard against a
    // detector-ordering mismatch silently producing a different threshold).
    let analytic = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        stim.detectors, analytic.detectors,
        "d={d} p={p}: stim/analytic detector count mismatch"
    );
    assert_eq!(stim.observables, analytic.observables);

    let oracle = PyMatchingOracle::new(&stim);
    run_dem_experiment(&stim, shots, &oracle, seed).expect("stim sweep cell")
}

/// Run a Stim circuit through `stim.detector_error_model` and return the `.dem` text.
fn stim_dem(program: &str) -> Option<String> {
    let python = std::env::var("STIM_PYTHON")
        .or_else(|_| std::env::var("PYMATCHING_PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());
    let mut child = Command::new(&python)
        .args([
            "-c",
            "import stim,sys;print(stim.Circuit(sys.stdin.read())\
             .detector_error_model(decompose_errors=False))",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(program.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8(out.stdout).ok())?
}
