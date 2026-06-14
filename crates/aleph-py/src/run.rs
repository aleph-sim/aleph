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
use crate::noise::PyNoiseModel;
use aleph_backend::{run, run_optimized, select_explained, Backend, BackendKind, BackendRequest};
use aleph_mps::MpsBackend;
use aleph_stab::StabilizerBackend;
use aleph_sv::NaiveSvBackend;
use numpy::{Complex64, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// Result of `run()`: a shot histogram and, on the SV backend, the final
/// state vector.
#[pyclass(name = "RunResult")]
pub(crate) struct RunResult {
    counts: BTreeMap<String, u64>,
    // Store the full CpuState rather than a copied Vec<Complex> to avoid a
    // 2^n × 16-byte allocation (up to 4 GiB at the n=28 cap).
    amps: Option<aleph_sv::CpuState>,
}

#[pymethods]
impl RunResult {
    /// Histogram of sampled bitstrings. Qubit 0 is the RIGHTMOST character
    /// (qubit 0 is the LSB of the amplitude index — ADR 0004; the leftmost
    /// character is qubit n-1), matching the CLI's |q_{n-1}…q_0⟩ output.
    fn counts(&self) -> BTreeMap<String, u64> {
        self.counts.clone()
    }

    /// Final state vector as a numpy `complex128` array of shape `(2**n,)`,
    /// SV backend only.
    ///
    /// One contiguous O(state) buffer (a copy of the backend's amplitude
    /// slice) rather than 2^n boxed Python complex objects — at n=25 the old
    /// list cost ~1.9 GiB of objects for a 512 MiB state. `aleph_core::Complex`
    /// is `num_complex::Complex<f64>` (== `numpy::Complex64`) stored
    /// contiguously, so `from_slice` is a single memcpy.
    fn statevector<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        match &self.amps {
            Some(state) => Ok(PyArray1::from_slice_bound(py, state.amplitudes())),
            None => Err(PyValueError::new_err(
                "statevector is only available on the \"sv\" backend",
            )),
        }
    }
}

fn counts_map(samples: &[u64], num_qubits: u32) -> BTreeMap<String, u64> {
    let width = num_qubits as usize;
    // Aggregate on the raw u64 outcome first; format only unique outcomes
    // (shots can vastly outnumber distinct bitstrings).
    let mut raw: BTreeMap<u64, u64> = BTreeMap::new();
    for s in samples {
        *raw.entry(*s).or_insert(0) += 1;
    }
    raw.into_iter()
        .map(|(s, count)| (format!("{s:0width$b}"), count))
        .collect()
}

/// Format a dense `run_noisy` histogram (index = basis state, qubit 0 = LSB)
/// into the same bitstring→count dict as `counts_map`, skipping zero bins.
fn hist_to_counts(hist: &[u64], num_qubits: u32) -> BTreeMap<String, u64> {
    let width = num_qubits as usize;
    hist.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (format!("{i:0width$b}"), c))
        .collect()
}

fn err<E: std::fmt::Display>(what: &str, e: E) -> PyErr {
    PyValueError::new_err(format!("{what}: {e}"))
}

/// Run `circuit` once on the chosen backend and sample `shots` shots from
/// the final state. `seed=None` uses OS entropy.
///
/// `backend` accepts the same names as the CLI's `--backend` (P4-12):
/// `"auto"` (default — pick from circuit structure: Clifford → stabilizer,
/// large nearest-neighbor + shallow → MPS, else state vector), `"statevector"`
/// (alias `"sv"`), `"stabilizer"` (alias `"stab"`), or `"mps"`.
///
/// **Breaking change vs v0.2:** the default was `"sv"`; it is now `"auto"`, so a
/// Clifford circuit routes to the stabilizer backend by default and
/// `RunResult.statevector()` raises unless you pass `backend="sv"`.
///
/// `Measure` instructions collapse the state once during execution; the
/// `shots` samples re-sample that single final state — they do NOT re-run
/// the circuit per shot (unlike Qiskit's per-shot execution model).
#[pyfunction]
#[pyo3(name = "run", signature = (circuit, *, shots = 1024, backend = "auto", seed = None, noise = None))]
pub(crate) fn run_circuit(
    py: Python<'_>,
    circuit: &PyCircuit,
    shots: u32,
    backend: &str,
    seed: Option<u64>,
    noise: Option<&PyNoiseModel>,
) -> PyResult<RunResult> {
    let c = &circuit.inner;
    let n = c.num_qubits();

    // One shared parse site with the CLI: canonical names + `sv`/`stab` aliases.
    let request = BackendRequest::from_user_str(backend).map_err(PyValueError::new_err)?;

    // Noisy path: Monte-Carlo trajectories via run_noisy on the UN-optimized
    // circuit (the optimizer emits TiledBlock/UnitaryKq that run_noisy rejects).
    // SV-only; unlike the noiseless path, each shot is an independent trajectory.
    // Mirrors the CLI: noise runs on SV, so `auto` and explicit `sv` are
    // accepted while an explicit `stab`/`mps` is rejected.
    if let Some(nm) = noise {
        if !request.allows_noise() {
            return Err(PyValueError::new_err(format!(
                "noise is only supported on the \"sv\" backend, got {backend:?}"
            )));
        }
        let model = &nm.inner;
        let seed = seed.unwrap_or_else(rand::random::<u64>);
        let counts = py.allow_threads(|| -> PyResult<BTreeMap<String, u64>> {
            let hist = aleph_sv::noise::run_noisy(c, model, shots, seed)
                .map_err(|e| err("run noisy", e))?;
            Ok(hist_to_counts(&hist, n))
        })?;
        return Ok(RunResult { counts, amps: None });
    }

    // Resolve `auto` through the same structural heuristic the CLI uses.
    let kind = match request {
        BackendRequest::Auto => select_explained(c).kind,
        BackendRequest::Fixed(k) => k,
    };

    // Each arm releases the GIL for execute+sample (minutes at n ≥ 25):
    // other Python threads — and Ctrl-C delivery — stay live during the run.
    match kind {
        BackendKind::Statevector => {
            let (counts, amps) = py.allow_threads(
                || -> PyResult<(BTreeMap<String, u64>, aleph_sv::CpuState)> {
                    let mut be = match seed {
                        Some(s) => NaiveSvBackend::with_seed(s),
                        None => NaiveSvBackend::new(),
                    };
                    let state = run_optimized(&mut be, c).map_err(|e| err("run sv", e))?;
                    // Sample before moving state into RunResult (sample takes &state).
                    let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                    Ok((counts_map(&samples, n), state))
                },
            )?;
            Ok(RunResult {
                counts,
                amps: Some(amps),
            })
        }
        BackendKind::Mps => {
            // MpsBackend already defaults to FixedBond(DEFAULT_MAX_BOND=128);
            // explicit .with_max_bond(128) is a drift hazard — omitted.
            let counts = py.allow_threads(|| -> PyResult<BTreeMap<String, u64>> {
                let mut be = match seed {
                    Some(s) => MpsBackend::with_seed(s),
                    None => MpsBackend::new(),
                };
                let state = run(&mut be, c).map_err(|e| err("run mps", e))?;
                let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                Ok(counts_map(&samples, n))
            })?;
            Ok(RunResult { counts, amps: None })
        }
        BackendKind::Stabilizer => {
            let counts = py.allow_threads(|| -> PyResult<BTreeMap<String, u64>> {
                let mut be = match seed {
                    Some(s) => StabilizerBackend::with_seed(s),
                    None => StabilizerBackend::new(),
                };
                let state = run(&mut be, c).map_err(|e| err("run stab", e))?;
                let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                Ok(counts_map(&samples, n))
            })?;
            Ok(RunResult { counts, amps: None })
        }
    }
}
