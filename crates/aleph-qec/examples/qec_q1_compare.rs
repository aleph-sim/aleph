//! Q1-05 head-to-head: native [`MwpmDecoder`] vs the [`PyMatchingOracle`] (real PyMatching) on
//! shared surface-code memory DEMs, across `d ∈ {3,5,7,9,11}`.
//!
//! Two measurements per distance, both on the *same* DEM and the *same* sampled syndromes so the
//! comparison is apples-to-apples:
//!
//! * **Accuracy** — the logical-error rate of each decoder, with a combined 95% CI and a
//!   within-CI verdict. Both are exact MWPM (up to equal-weight ties), so the rates should agree
//!   far inside the CI; the test is that they do.
//! * **Throughput** — decoded **syndromes/second**. The native decoder is timed in-process,
//!   single-threaded (its [`Decoder::decode`] core), the directly comparable number to
//!   PyMatching's single-threaded C++ `decode_batch`. PyMatching is timed two ways: its *core*
//!   matching time (reported by the Python driver itself) and the *end-to-end* time a Rust caller
//!   actually pays (subprocess spawn + interpreter import + DEM compile + serialisation + match).
//!
//! Usage:
//!
//! ```text
//! PYMATCHING_PYTHON=/path/to/venv/bin/python \
//!   cargo run --release -p aleph-qec --example qec_q1_compare -- [acc_shots] [thru_shots] [seed]
//! # defaults: acc_shots=100000 thru_shots=50000 seed=2024
//! ```
//!
//! Prints two CSV blocks (accuracy, then throughput) to stdout and a human-readable log to stderr.
//! The noise is phenomenological with `p_data == p_meas == p` and `rounds == d`, decoded near the
//! ~3% memory threshold so every distance keeps a statistically resolvable logical-error rate.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use aleph_qec::{
    build_dem, Decoder, DetectorErrorModel, MwpmDecoder, PyMatchingOracle, SurfaceCode, Syndrome,
};
use rayon::prelude::*;

/// Code distances compared (the Q1-05 acceptance set).
const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];
/// Near-threshold physical error rate: dense enough that every distance has a resolvable rate.
const P: f64 = 0.03;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let acc_shots: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let thru_shots: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!(
        "# Q1-05 compare: p={P} rounds=d acc_shots={acc_shots} thru_shots={thru_shots} seed={seed}"
    );

    // ---- Accuracy ----------------------------------------------------------------------------
    println!("section,d,detectors,p,shots,rate_mwpm,ci_mwpm,rate_pymatching,ci_pymatching,abs_delta,combined_ci,within_ci");
    let mut acc_rows = Vec::new();
    for &d in DISTANCES {
        let dem = cell_dem(d, P);
        let (syndromes, truths) = sample(&dem, acc_shots, seed ^ (d as u64).wrapping_mul(0x9E37));

        // Native: parallel decode (correctness identical to the single-thread path; rayon only
        // shortens the wall clock for this 10^5-shot accuracy sweep).
        let ours: Vec<_> = syndromes.par_iter().map(|s| dem_decode(&dem, s)).collect();
        let theirs = PyMatchingOracle::new(&dem)
            .decode_batch(&syndromes)
            .expect("PyMatching subprocess (set PYMATCHING_PYTHON)");

        let (eo, et) = logical_errors(&ours, &theirs, &truths);
        let (ro, rt) = (eo as f64 / acc_shots as f64, et as f64 / acc_shots as f64);
        let ci_o = ci95(ro, acc_shots);
        let ci_t = ci95(rt, acc_shots);
        // Combined CI for the difference of two independent proportions.
        let comb = 1.96 * ((ro * (1.0 - ro) + rt * (1.0 - rt)) / acc_shots as f64).sqrt();
        let within = (ro - rt).abs() <= comb + 1e-12;
        println!(
            "accuracy,{d},{},{P},{acc_shots},{ro},{ci_o},{rt},{ci_t},{},{comb},{within}",
            dem.detectors,
            (ro - rt).abs()
        );
        eprintln!(
            "  acc d={d}: aleph={ro:.4e}±{ci_o:.1e}  pymatching={rt:.4e}±{ci_t:.1e}  |Δ|={:.2e} ci={comb:.2e}  within_ci={within}",
            (ro - rt).abs()
        );
        acc_rows.push(within);
    }

    // ---- Throughput --------------------------------------------------------------------------
    println!("section,d,detectors,avg_defects,shots,mwpm_syn_per_s,pymatching_core_syn_per_s,pymatching_e2e_syn_per_s,speedup_core,speedup_e2e");
    for &d in DISTANCES {
        let dem = cell_dem(d, P);
        let (syndromes, _) = sample(&dem, thru_shots, seed ^ (d as u64).wrapping_mul(0x1357));
        let avg_def =
            syndromes.iter().map(|s| s.weight()).sum::<usize>() as f64 / thru_shots as f64;

        // Native single-thread core throughput (the directly comparable number). Best of three
        // runs, mirroring the PyMatching driver's best-of-five, so transient host load doesn't
        // bias one decoder against the other.
        let dec = MwpmDecoder::new(&dem).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut sink = 0u64;
            for s in &syndromes {
                sink ^= dec
                    .decode(s)
                    .observable_flips
                    .first()
                    .copied()
                    .unwrap_or(false) as u64;
            }
            std::hint::black_box(sink);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let mwpm_s = thru_shots as f64 / best;

        // PyMatching: end-to-end (full subprocess round-trip) and its self-reported core time.
        let t1 = Instant::now();
        let _ = PyMatchingOracle::new(&dem)
            .decode_batch(&syndromes)
            .expect("PyMatching subprocess");
        let pm_e2e_s = thru_shots as f64 / t1.elapsed().as_secs_f64();
        let core_secs =
            pymatching_core_seconds(&dem, &syndromes).expect("PyMatching timing driver");
        let pm_core_s = thru_shots as f64 / core_secs;

        println!(
            "throughput,{d},{},{avg_def:.1},{thru_shots},{mwpm_s:.1},{pm_core_s:.1},{pm_e2e_s:.1},{:.3},{:.3}",
            dem.detectors,
            mwpm_s / pm_core_s,
            mwpm_s / pm_e2e_s,
        );
        eprintln!(
            "  thru d={d} (det={}, avg_def={avg_def:.1}): aleph={mwpm_s:.0}/s  pymatching_core={pm_core_s:.0}/s  pymatching_e2e={pm_e2e_s:.0}/s",
            dem.detectors
        );
    }

    if acc_rows.iter().all(|&w| w) {
        eprintln!("# accuracy: all distances within combined CI ✓");
    } else {
        eprintln!("# accuracy: SOME DISTANCE EXCEEDED CI ✗");
        std::process::exit(1);
    }
}

/// The phenomenological memory-Z DEM for distance `d` at physical error `p` (rounds == d).
fn cell_dem(d: usize, p: f64) -> DetectorErrorModel {
    let exp = SurfaceCode::new(d).memory_z_experiment(d);
    build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap()
}

/// Decode one syndrome against a freshly-built decoder — used only inside the parallel accuracy
/// map, where building a decoder per shot would dominate. Callers that decode many shots build the
/// decoder once (see the throughput loop); here we amortise it via a thread-local.
fn dem_decode(dem: &DetectorErrorModel, s: &Syndrome) -> aleph_qec::Correction {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<(usize, MwpmDecoder)>> =
            const { std::cell::RefCell::new(None) };
    }
    // Key the thread-local decoder on detector count: every distance in one run has a distinct
    // detector count, so this rebuilds exactly once per distance per worker thread.
    CACHE.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.as_ref().map(|(k, _)| *k) != Some(dem.detectors) {
            *slot = Some((dem.detectors, MwpmDecoder::new(dem).unwrap()));
        }
        slot.as_ref().unwrap().1.decode(s)
    })
}

/// Sample `shots` syndromes (and the true `L0` flip) from `dem`, deterministically per seed — the
/// Bernoulli-per-mechanism model the Monte-Carlo harness uses.
fn sample(dem: &DetectorErrorModel, shots: usize, seed: u64) -> (Vec<Syndrome>, Vec<bool>) {
    (0..shots as u64)
        .into_par_iter()
        .map(|s| {
            let mut z = seed.wrapping_add(s.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut next = || {
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 11) as f64 / (1u64 << 53) as f64
            };
            let mut det = vec![false; dem.detectors];
            let mut obs = false;
            for e in &dem.errors {
                if next() < e.prob {
                    for &d in &e.dets {
                        det[d as usize] ^= true;
                    }
                    if e.obs.contains(&0) {
                        obs ^= true;
                    }
                }
            }
            (Syndrome::from_bits(&det), obs)
        })
        .unzip()
}

/// Count logical errors (predicted `L0` flip ≠ truth) for each decoder over the shared shots.
fn logical_errors(
    ours: &[aleph_qec::Correction],
    theirs: &[aleph_qec::Correction],
    truths: &[bool],
) -> (u64, u64) {
    let flip = |c: &aleph_qec::Correction| c.observable_flips.first().copied().unwrap_or(false);
    let mut eo = 0;
    let mut et = 0;
    for ((o, t), truth) in ours.iter().zip(theirs).zip(truths) {
        eo += (flip(o) != *truth) as u64;
        et += (flip(t) != *truth) as u64;
    }
    (eo, et)
}

/// 95% Wald half-width for a proportion.
fn ci95(rate: f64, n: usize) -> f64 {
    1.96 * (rate * (1.0 - rate) / n as f64).sqrt()
}

/// The Python driver that reports PyMatching's *core* `decode_batch` time, excluding subprocess
/// startup, DEM compilation, and serialisation. It loops the call a few times and reports the
/// fastest run (the cleanest estimate of steady-state matching throughput).
const TIMING_DRIVER: &str = r#"
import sys, time, numpy as np, stim, pymatching
buf = sys.stdin.buffer.read()
off = 0
def u64():
    global off
    v = int.from_bytes(buf[off:off+8], "little"); off += 8; return v
dem_len = u64(); dem_text = buf[off:off+dem_len].decode("utf-8"); off += dem_len
shots = u64(); ndet = u64()
synd = np.frombuffer(buf[off:off+shots*ndet], dtype=np.uint8).reshape(shots, ndet)
m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(dem_text))
best = float("inf")
for _ in range(5):
    t = time.perf_counter()
    m.decode_batch(synd)
    best = min(best, time.perf_counter() - t)
sys.stdout.write(repr(best))
"#;

/// Run the timing driver and return PyMatching's fastest core `decode_batch` seconds over the
/// supplied syndromes. Same wire format as the oracle's request, minus the unused observable count.
fn pymatching_core_seconds(dem: &DetectorErrorModel, syndromes: &[Syndrome]) -> Option<f64> {
    let python = std::env::var("PYMATCHING_PYTHON")
        .or_else(|_| std::env::var("STIM_PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());
    let dem_text = dem.to_dem_string();
    let shots = syndromes.len();

    let mut input = Vec::with_capacity(24 + dem_text.len() + shots * dem.detectors);
    input.extend_from_slice(&(dem_text.len() as u64).to_le_bytes());
    input.extend_from_slice(dem_text.as_bytes());
    input.extend_from_slice(&(shots as u64).to_le_bytes());
    input.extend_from_slice(&(dem.detectors as u64).to_le_bytes());
    let mut row = vec![0u8; dem.detectors];
    for s in syndromes {
        row.iter_mut().for_each(|b| *b = 0);
        for &d in &s.fired {
            if (d as usize) < dem.detectors {
                row[d as usize] = 1;
            }
        }
        input.extend_from_slice(&row);
    }

    let mut child = Command::new(&python)
        .args(["-c", TIMING_DRIVER])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let writer = std::thread::spawn(move || stdin.write_all(&input).and_then(|_| stdin.flush()));
    let mut out = String::new();
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    writer.join().ok()?.ok()?;
    child.wait().ok()?;
    out.trim().parse().ok()
}
