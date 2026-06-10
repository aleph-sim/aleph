//! `run(circuit, *, shots, backend, seed)` → `RunResult` (counts +
//! optional statevector). Mirrors `aleph-cli/src/exec.rs` semantics:
//! execute once, then sample the final state `shots` times.
//!
//! SV runs through `run_optimized` (the optimized driver the Phase-4
//! benches time; its oracle is `aleph-backend/tests/run_optimized_oracle.rs`).
//! MPS/stabilizer run verbatim — they reject fused `DiagonalPhase`
//! instructions the optimizer emits.
// pyo3 0.22 proc-macro expansion emits trivial PyErr→PyErr .into() calls —
// removing the allow yields false positives.
#![allow(clippy::useless_conversion)]

use crate::circuit::PyCircuit;
use aleph_backend::{run, run_optimized, Backend};
use aleph_mps::MpsBackend;
use aleph_stab::StabilizerBackend;
use aleph_sv::NaiveSvBackend;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyComplex;
use std::collections::BTreeMap;

/// Result of `run()`: a shot histogram and, on the SV backend, the final
/// state vector.
#[pyclass(name = "RunResult")]
pub(crate) struct RunResult {
    counts: BTreeMap<String, u64>,
    amps: Option<Vec<aleph_core::Complex>>,
}

#[pymethods]
impl RunResult {
    /// Histogram of sampled bitstrings. Qubit 0 is the LEFTMOST character
    /// (aleph's MSB qubit-ordering convention, ADR 0004).
    fn counts(&self) -> BTreeMap<String, u64> {
        self.counts.clone()
    }

    /// Final state vector (list of complex), SV backend only.
    fn statevector<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyComplex>>> {
        match &self.amps {
            Some(amps) => Ok(amps
                .iter()
                .map(|c| PyComplex::from_doubles_bound(py, c.re, c.im))
                .collect()),
            None => Err(PyValueError::new_err(
                "statevector is only available on the \"sv\" backend",
            )),
        }
    }
}

fn counts_map(samples: &[u64], num_qubits: u32) -> BTreeMap<String, u64> {
    let width = num_qubits as usize;
    let mut hist = BTreeMap::new();
    for s in samples {
        *hist.entry(format!("{s:0width$b}")).or_insert(0) += 1;
    }
    hist
}

fn err<E: std::fmt::Display>(what: &str, e: E) -> PyErr {
    PyValueError::new_err(format!("{what}: {e}"))
}

/// Run `circuit` once on the chosen backend and sample `shots` shots from
/// the final state. `seed=None` uses OS entropy.
#[pyfunction]
#[pyo3(name = "run", signature = (circuit, *, shots = 1024, backend = "sv", seed = None))]
pub(crate) fn run_circuit(
    circuit: &PyCircuit,
    shots: u32,
    backend: &str,
    seed: Option<u64>,
) -> PyResult<RunResult> {
    let c = &circuit.inner;
    let n = c.num_qubits();
    match backend {
        "sv" => {
            let mut be = match seed {
                Some(s) => NaiveSvBackend::with_seed(s),
                None => NaiveSvBackend::new(),
            };
            let state = run_optimized(&mut be, c).map_err(|e| err("run sv", e))?;
            let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
            Ok(RunResult {
                counts: counts_map(&samples, n),
                amps: Some(state.amplitudes().to_vec()),
            })
        }
        "mps" => {
            let mut be = match seed {
                Some(s) => MpsBackend::with_seed(s),
                None => MpsBackend::new(),
            }
            .with_max_bond(128);
            let state = run(&mut be, c).map_err(|e| err("run mps", e))?;
            let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
            Ok(RunResult {
                counts: counts_map(&samples, n),
                amps: None,
            })
        }
        "stab" => {
            let mut be = match seed {
                Some(s) => StabilizerBackend::with_seed(s),
                None => StabilizerBackend::new(),
            };
            let state = run(&mut be, c).map_err(|e| err("run stab", e))?;
            let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
            Ok(RunResult {
                counts: counts_map(&samples, n),
                amps: None,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown backend {other:?} (expected \"sv\", \"mps\", or \"stab\")"
        ))),
    }
}
