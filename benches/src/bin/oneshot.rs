//! Single-shot runner for peak-RSS measurement under `/usr/bin/time -v`.
//! Loads a QASM circuit and runs `NaiveSvBackend` exactly once. Not a
//! benchmark — the point is a clean process whose Maximum RSS reflects one
//! state-vector simulation. Usage:
//!   /usr/bin/time -v ./oneshot scripts/qiskit-baseline/circuits/qft_n25.qasm

use aleph_backend::run;
use aleph_sv::NaiveSvBackend;
use std::hint::black_box;

fn main() {
    let path = std::env::args().nth(1).expect("usage: oneshot <circuit.qasm>");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse {path}: {e:?}"));
    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, &circuit).expect("simulation failed");
    // Touch the result so the optimiser can't elide the work.
    black_box(state.amplitudes().len());
}
