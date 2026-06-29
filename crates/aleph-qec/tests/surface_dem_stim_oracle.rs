//! Q0-03 oracle: the analytic detector-error-model from [`build_dem`] matches
//! Stim's `detector_error_model` for the *same* surface-code memory-Z circuit.
//!
//! We emit our circuit + phenomenological noise as a Stim program, let Stim
//! compute its DEM, parse it, and compare edge-for-edge (support → probability).
//! Because both DEMs describe the identical program, this isolates *our* DEM
//! builder (Pauli propagation + detector wiring + probability merge) against
//! Stim's, with no circuit-convention matching to get wrong.
//!
//! Requires python3 + stim; `#[ignore]`d (run on a stim-equipped box, e.g. the
//! CUDA box's `/root/stimvenv`):
//!
//!   STIM_PYTHON=/root/stimvenv/bin/python \
//!     cargo test -p aleph-qec --test surface_dem_stim_oracle -- --ignored
//!
//! Validated during development against stim 1.16.0 (d∈{3,5}, rounds∈{1,2,3}):
//! every edge matched to < 1e-9.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use aleph_qec::{build_dem, CircuitNoise, DetectorErrorModel, SurfaceCode};

/// Canonical edge map: (sorted detectors, sorted observables) -> probability.
fn edge_map(dem: &DetectorErrorModel) -> BTreeMap<(Vec<u32>, Vec<u32>), f64> {
    dem.errors
        .iter()
        .map(|e| ((e.dets.clone(), e.obs.clone()), e.prob))
        .collect()
}

/// Run a Stim circuit string through `stim.detector_error_model` and return the
/// `.dem` text, or `None` if python/stim is unavailable.
fn stim_dem(program: &str) -> Option<String> {
    let python = std::env::var("STIM_PYTHON").unwrap_or_else(|_| "python3".to_string());
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
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

#[test]
#[ignore = "requires python3 + stim; run with STIM_PYTHON set on a stim box"]
fn analytic_dem_matches_stim() {
    let (p_data, p_meas) = (0.013, 0.021);
    for d in [3usize, 5] {
        for rounds in [1usize, 2, 3] {
            let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
            let analytic = build_dem(
                &exp.annotated,
                &exp.phenomenological_mechanisms(p_data, p_meas),
            )
            .unwrap();

            let program = exp.stim_program(p_data, p_meas);
            let stim_text = match stim_dem(&program) {
                Some(t) => t,
                None => panic!("could not run stim (set STIM_PYTHON to a stim python)"),
            };
            let stim = DetectorErrorModel::parse(&stim_text).expect("parse stim dem");

            let (a, b) = (edge_map(&analytic), edge_map(&stim));
            assert!(!a.is_empty(), "d={d} r={rounds}: empty analytic DEM");
            assert_eq!(
                a.len(),
                b.len(),
                "d={d} r={rounds}: edge count {} vs stim {}",
                a.len(),
                b.len()
            );
            for (k, &pa) in &a {
                let pb = b
                    .get(k)
                    .unwrap_or_else(|| panic!("d={d} r={rounds}: edge {k:?} missing in stim"));
                assert!(
                    (pa - pb).abs() < 1e-9,
                    "d={d} r={rounds}: edge {k:?} prob {pa} vs stim {pb}"
                );
            }
        }
    }
}

/// Circuit-level oracle: our `X`-sector circuit-level DEM matches Stim's `detector_error_model` for
/// the same depth-(per-round) syndrome-extraction circuit + circuit-level noise, edge-for-edge. This
/// isolates the circuit-level mechanism enumeration (CNOT depolarizing / idle / prep / measurement)
/// and its Pauli propagation against Stim. `decompose_errors=False`: the surface-code circuit-level
/// X-sector DEM is graphlike (verified in the unit tests), so no hyperedge decomposition is needed
/// and the two edge sets compare directly.
///
///   STIM_PYTHON=/root/stimvenv/bin/python \
///     cargo test -p aleph-qec --test surface_dem_stim_oracle circuit_level -- --ignored
///
/// Validated against stim 1.16.0 (d∈{3,5}, rounds∈{1,2,3}): every edge matched to < 1e-9.
#[test]
#[ignore = "requires python3 + stim; run with STIM_PYTHON set on a stim box"]
fn circuit_level_dem_matches_stim() {
    let noise = CircuitNoise {
        p_cnot: 0.011,
        p_init: 0.007,
        p_meas: 0.013,
        p_idle: 0.005,
    };
    for d in [3usize, 5] {
        for rounds in [1usize, 2, 3] {
            let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
            let ours = exp.circuit_level_dem(noise).unwrap();

            let program = exp.stim_program_circuit_level(noise);
            let stim_text = match stim_dem(&program) {
                Some(t) => t,
                None => panic!("could not run stim (set STIM_PYTHON to a stim python)"),
            };
            let stim = DetectorErrorModel::parse(&stim_text).expect("parse stim dem");

            let (a, b) = (edge_map(&ours), edge_map(&stim));
            assert!(!a.is_empty(), "d={d} r={rounds}: empty circuit-level DEM");
            assert_eq!(
                a.len(),
                b.len(),
                "d={d} r={rounds}: edge count {} vs stim {}",
                a.len(),
                b.len()
            );
            for (k, &pa) in &a {
                let pb = b
                    .get(k)
                    .unwrap_or_else(|| panic!("d={d} r={rounds}: edge {k:?} missing in stim"));
                assert!(
                    (pa - pb).abs() < 1e-9,
                    "d={d} r={rounds}: edge {k:?} prob {pa} vs stim {pb}"
                );
            }
        }
    }
}
