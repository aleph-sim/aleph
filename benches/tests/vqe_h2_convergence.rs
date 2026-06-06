//! P4-04 acceptance gate (AC1): VQE converges to the H2 ground-state energy.
//! Pure-Rust rotosolve on the committed real H2 4-qubit Hamiltonian (the same
//! data file the Python driver loads) reaching the FCI energy within chemical
//! accuracy (1.6e-3 Ha). This gates convergence in the Rust CI without needing
//! maturin/Python. Mirrors scripts/vqe/rotosolve.py.

use aleph_backend::{expectation_pauli_sum, run};
use aleph_core::PauliSum;
use aleph_ir::build_hea;
use aleph_sv::NaiveSvBackend;
use std::f64::consts::PI;
use std::path::PathBuf;

const DEPTH: u32 = 4; // enough DOF for H2 4q; raise if convergence misses.
const CHEM_ACC: f64 = 1.6e-3;

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scripts/vqe/hamiltonians")
        .join(name)
}

fn energy(ham: &PauliSum, n: u32, depth: u32, thetas: &[f64]) -> f64 {
    let circuit = build_hea(n, depth, thetas).expect("build_hea");
    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, &circuit).expect("run");
    expectation_pauli_sum(&mut backend, &state, ham).expect("energy")
}

#[test]
fn vqe_h2_reaches_fci() {
    let n = 4u32;
    let ham = PauliSum::parse(
        &std::fs::read_to_string(data_path("vqe_n4.txt")).unwrap(),
        n,
    )
    .unwrap();
    let fci: f64 = std::fs::read_to_string(data_path("vqe_n4.fci"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();

    // Deterministic, non-trivial starting point (avoid the all-zero saddle).
    let p = (n * (DEPTH + 1)) as usize;
    let mut theta: Vec<f64> = (0..p).map(|i| 0.1 * (i as f64 + 1.0)).collect();

    // Rotosolve sweeps.
    let mut last = energy(&ham, n, DEPTH, &theta);
    for _ in 0..50 {
        for j in 0..p {
            let base = theta[j];
            // theta[j] == base already; measure energy at the current point.
            let e0 = energy(&ham, n, DEPTH, &theta);
            theta[j] = base + PI / 2.0;
            let ep = energy(&ham, n, DEPTH, &theta);
            theta[j] = base - PI / 2.0;
            let em = energy(&ham, n, DEPTH, &theta);
            theta[j] = base - PI / 2.0 - (2.0 * e0 - ep - em).atan2(ep - em);
        }
        let e = energy(&ham, n, DEPTH, &theta);
        if (e - last).abs() < 1e-9 {
            break;
        }
        last = e;
    }
    let final_e = energy(&ham, n, DEPTH, &theta);
    assert!(
        final_e <= fci + CHEM_ACC,
        "VQE energy {final_e:.6} did not reach FCI {fci:.6} within {CHEM_ACC} (gap {:.2e})",
        final_e - fci
    );
}
