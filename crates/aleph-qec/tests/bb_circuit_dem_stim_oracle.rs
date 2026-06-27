//! Q5-04 oracle: the analytic circuit-level detector-error-model from
//! [`BBCode::circuit_level_dem`] matches Stim's `detector_error_model` for the *same*
//! depth-7 bivariate-bicycle memory-X circuit.
//!
//! We emit our circuit + Z-sector circuit-level noise as a Stim program
//! ([`BBMemoryExperiment::stim_program`]), let Stim compute its DEM, parse it, and compare
//! edge-for-edge (support -> probability). Because both DEMs describe the identical program,
//! this isolates *our* DEM builder (Pauli propagation + detector wiring + probability merge)
//! against Stim's. It is also the determinism gate: Stim refuses to build a DEM if any detector
//! is non-deterministic in the noiseless circuit, so a clean build certifies the syndrome
//! schedule measures both stabiliser types without mutual disturbance.
//!
//! Requires python3 + stim; `#[ignore]`d. Run on a stim-equipped box:
//!
//!   STIM_PYTHON=/Users/ex/aleph/.venv/bin/python \
//!     cargo test -p aleph-qec --test bb_circuit_dem_stim_oracle -- --ignored
//!
//! Validated during development against stim 1.16.0 (ℓ∈{6}, rounds∈{1,2,3}): every edge < 1e-9.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use aleph_qec::{BBCode, CircuitNoise, DetectorErrorModel};

/// Canonical edge map: (sorted detectors, sorted observables) -> probability.
fn edge_map(dem: &DetectorErrorModel) -> BTreeMap<(Vec<u32>, Vec<u32>), f64> {
    dem.errors
        .iter()
        .map(|e| ((e.dets.clone(), e.obs.clone()), e.prob))
        .collect()
}

/// Run a Stim circuit string through `stim.detector_error_model` and return the `.dem` text, or
/// `None` if python/stim is unavailable.
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
fn circuit_level_dem_matches_stim() {
    // Smaller BB family member ([[72,12,6]], ℓ=6) keeps the Stim run fast while exercising the
    // full hypergraph schedule. The gross code is verified the same way, just slower.
    let code = BBCode::new(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)]);
    let noise = CircuitNoise {
        p_cnot: 0.006,
        p_init: 0.004,
        p_meas: 0.008,
        p_idle: 0.003,
    };
    for rounds in [1usize, 2, 3] {
        let exp = code.memory_x_experiment(rounds);
        let analytic = aleph_qec::build_dem(&exp.annotated, &exp.circuit_level_mechanisms(noise))
            .expect("analytic dem");

        let program = exp.stim_program(noise);
        let stim_text = match stim_dem(&program) {
            Some(t) => t,
            None => panic!("could not run stim (set STIM_PYTHON to a stim python)"),
        };
        let stim = DetectorErrorModel::parse(&stim_text).expect("parse stim dem");

        let (a, b) = (edge_map(&analytic), edge_map(&stim));
        assert!(!a.is_empty(), "rounds={rounds}: empty analytic DEM");
        assert_eq!(
            a.len(),
            b.len(),
            "rounds={rounds}: edge count {} vs stim {}",
            a.len(),
            b.len()
        );
        for (k, &pa) in &a {
            let pb = b
                .get(k)
                .unwrap_or_else(|| panic!("rounds={rounds}: edge {k:?} missing in stim"));
            assert!(
                (pa - pb).abs() < 1e-9,
                "rounds={rounds}: edge {k:?} prob {pa} vs stim {pb}"
            );
        }
    }
}
