//! Single-shot runner for peak-RSS and single-shot wall-time measurement.
//! Loads a QASM circuit and runs `NaiveSvBackend` exactly once via the
//! optimized pipeline (`run_optimized`: optimize + simulate), so the RSS /
//! timing path matches the `qiskit_baseline` bench. Not a criterion benchmark —
//! the point is a clean process whose Maximum RSS reflects one state-vector
//! simulation, and which prints the wall time of the `run_optimized` call
//! (excluding parse) as `elapsed_ms <ms>` on stdout, followed by
//! `xeb <value>` — the noiseless linear XEB (`2^n·Σp²−1`) of the final state.
//! This is the practical way to measure n≥28 where 10-sample criterion is
//! prohibitively slow; the qubit cap is raised to 32 so the large QFT corpus
//! (n=30 ⇒ 16 GiB) can run on a big-memory host. Usage:
//!   /usr/bin/time -v ./oneshot scripts/qiskit-baseline/circuits/qft_n30.qasm

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: oneshot <circuit.qasm>");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse {path}: {e:?}"));
    // Cap raised to 32 so n=30 (16 GiB) is permitted on a big-memory host;
    // ordinary hosts are unaffected (they simply won't be asked for n>28 here).
    let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
    let start = Instant::now();
    let state = run_optimized(&mut backend, &circuit).expect("simulation failed");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let amps = state.amplitudes();
    let xeb = aleph_benches::linear_xeb(amps);
    // Touch the result so the optimiser can't elide the work.
    black_box(amps.len());
    println!("elapsed_ms {elapsed_ms:.3}");
    println!("xeb {xeb:.6}");
}
