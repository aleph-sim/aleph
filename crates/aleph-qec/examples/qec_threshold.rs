//! Surface-code threshold sweep (Q0-05 harness; Q1-04 native decoder).
//!
//! Runs the memory-Z experiment across a grid of code distances `d` and physical error
//! probabilities `p`, decodes every shot, and prints one CSV row per `(d, p)` cell. The crossing
//! of the per-distance logical-error-rate curves is the threshold; `scripts/qec_threshold_plot.py`
//! plots it and fits `p_th`.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aleph-qec --example qec_threshold -- [decoder] [source] [shots] [seed] [noise] [basis]
//! # decoder ∈ {mwpm (default), uf, pymatching}; source ∈ {analytic (default), stim}
//! # noise   ∈ {phenom (default), circuit, circuit-si1000}; basis ∈ {z (default), x}
//! # e.g. native sweep:        cargo run --release -p aleph-qec --example qec_threshold -- mwpm analytic 200000 2024
//! # circuit-level sweep:      cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 200000 2024 circuit
//! # SI1000 sweep:             cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 200000 2024 circuit-si1000
//! # oracle sweep:             PYMATCHING_PYTHON=/tmp/pmvenv/bin/python cargo run ... -- pymatching analytic 200000 2024
//! ```
//!
//! * `decoder = mwpm` (default, Q1-04) — our native [`MwpmDecoder`]. No external dependency.
//! * `decoder = pymatching` — the external [`PyMatchingOracle`] (real MWPM), the reference the
//!   native curve is validated against. Needs `PYMATCHING_PYTHON`.
//! * `source = analytic` — decode our own [`build_dem`] model.
//! * `source = stim` — decode the DEM Stim emits for the same circuit + noise (needs `STIM_PYTHON`;
//!   equals the analytic DEM edge-for-edge per Q0-03). Phenomenological noise only.
//! * `noise = phenom` — phenomenological noise (`p_data == p_meas == p`); threshold ~3%.
//! * `noise = circuit` — full circuit-level noise ([`MemoryExperiment::circuit_level_dem`], uniform
//!   `CircuitNoise`): every CNOT / idle / prep / measurement is a fault site, so the DEM has hook
//!   errors and the threshold is substantially lower (~0.5–1%). The prob grid auto-switches.
//! * `noise = circuit-si1000` — circuit-level with the SI1000 superconducting-inspired rates
//!   ([`CircuitNoise::si1000`]: reset 2p, measure 5p, idle 2p); threshold ~0.3–0.6% in `p`.
//!
//! * `basis = z` (default) — memory-Z (detects `X` errors). `basis = x` — the memory-X mirror
//!   (X-stabilizers, detects `Z` errors); by code symmetry its threshold matches memory-Z.
//!
//! `rounds == d` in all models.

use std::io::Write;
use std::process::{Command, Stdio};

use aleph_qec::{
    build_dem, run_dem_experiment, CircuitNoise, DetectorErrorModel, MwpmDecoder, PyMatchingOracle,
    SurfaceCode, UnionFindDecoder,
};

/// Code distances swept.
const DISTANCES: &[usize] = &[3, 5, 7, 9];
/// Physical error probabilities, bracketing the phenomenological threshold (~3%).
const PROBS_PHENOM: &[f64] = &[0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050];
/// Lower grid bracketing the circuit-level threshold (~0.5–1%).
const PROBS_CIRCUIT: &[f64] = &[0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012];
/// Lowest grid bracketing the SI1000 threshold (~0.3–0.6%; reset/meas/idle rates are 2–5×p).
const PROBS_SI1000: &[f64] = &[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let decoder = args.get(1).map(String::as_str).unwrap_or("mwpm");
    let source = args.get(2).map(String::as_str).unwrap_or("analytic");
    let shots: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let noise = args.get(5).map(String::as_str).unwrap_or("phenom");
    let basis = args.get(6).map(String::as_str).unwrap_or("z");

    let probs: &[f64] = match noise {
        "phenom" => PROBS_PHENOM,
        "circuit" => PROBS_CIRCUIT,
        "circuit-si1000" => PROBS_SI1000,
        other => panic!("unknown noise `{other}` (expected phenom|circuit|circuit-si1000)"),
    };
    assert!(
        !(noise.starts_with("circuit") && source == "stim"),
        "circuit-level stim source not wired here; use analytic (DEM is Stim-verified in tests)"
    );
    assert!(
        matches!(basis, "z" | "x"),
        "unknown basis `{basis}` (expected z|x)"
    );
    assert!(
        !(basis == "x" && source == "stim"),
        "memory-X stim source not wired here; use analytic (DEM is Stim-verified in tests)"
    );

    eprintln!(
        "# decoder={decoder} source={source} shots={shots} seed={seed} \
         rounds=d noise={noise} basis={basis}"
    );
    println!("decoder,source,d,rounds,p,shots,logical_errors,rate,ci95");

    for &d in DISTANCES {
        let code = SurfaceCode::new(d);
        for &p in probs {
            let exp = match basis {
                "x" => code.memory_x_experiment(d),
                _ => code.memory_z_experiment(d),
            };
            let dem = match (noise, source) {
                ("circuit", _) => exp.circuit_level_dem(CircuitNoise::uniform(p)).unwrap(),
                ("circuit-si1000", _) => exp.circuit_level_dem(CircuitNoise::si1000(p)).unwrap(),
                ("phenom", "analytic") => {
                    build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap()
                }
                ("phenom", "stim") => stim_cell_dem(&code, d, p),
                _ => unreachable!("noise/source validated above"),
            };
            // Decode the same DEM with the chosen decoder. The native MWPM is built against the
            // very DEM the shots are drawn from — the whole point of the Q0-04 factory design.
            let res = match decoder {
                "mwpm" => {
                    let dec = MwpmDecoder::new(&dem).expect("graphlike dem");
                    run_dem_experiment(&dem, shots, &dec, seed)
                }
                "pymatching" => {
                    let dec = PyMatchingOracle::new(&dem);
                    run_dem_experiment(&dem, shots, &dec, seed)
                }
                "uf" | "unionfind" => {
                    let dec = UnionFindDecoder::new(&dem).expect("graphlike dem");
                    run_dem_experiment(&dem, shots, &dec, seed)
                }
                other => panic!("unknown decoder `{other}` (expected mwpm|uf|pymatching)"),
            }
            .expect("sweep cell");
            println!(
                "{decoder},{source},{d},{d},{p},{},{},{},{}",
                res.shots, res.logical_errors, res.rate, res.ci95
            );
            eprintln!(
                "  d={d} p={p:.3}: rate={:.4e} ± {:.1e} ({} errs / {} shots)",
                res.rate, res.ci95, res.logical_errors, res.shots
            );
        }
    }
}

/// Build the DEM that Stim emits for the `(d, p)` cell's circuit + noise, with a cheap sanity
/// check that it matches our analytic model's shape (Q0-03 proved them edge-for-edge equal).
fn stim_cell_dem(code: &SurfaceCode, d: usize, p: f64) -> DetectorErrorModel {
    let exp = code.memory_z_experiment(d);
    let stim_text = stim_dem(&exp.stim_program(p, p)).expect("stim subprocess");
    let stim = DetectorErrorModel::parse(&stim_text).expect("parse stim dem");
    let analytic = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        stim.detectors, analytic.detectors,
        "d={d} p={p}: stim/analytic detector count mismatch"
    );
    assert_eq!(stim.observables, analytic.observables);
    stim
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
