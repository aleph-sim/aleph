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
use aleph_backend::{
    run, run_optimized, select_explained_full, Backend, BackendKind, BackendRequest, Reach,
    SelectEnv, Selection,
};
use aleph_mps::MpsBackend;
use aleph_stab::StabilizerBackend;
use aleph_sv::NaiveSvBackend;
use numpy::{Complex64, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// Final-state amplitude holder. The CPU SV backend keeps its `CpuState` so
/// `statevector()` is a single zero-copy memcpy of its slice (no 2^n widening,
/// up to 4 GiB at the n=28 cap); the Metal GPU backend is FP32, so it
/// materializes the widened f64 buffer once.
enum AmpStore {
    Cpu(aleph_sv::CpuState),
    // Constructed by the Metal arm (macOS + `metal`) and the CUDA in-core arms
    // (Linux + `cuda`); the `statevector()` reader matches it in every build, so
    // silence the never-constructed warning only when both GPU paths are compiled out.
    #[cfg_attr(
        not(any(
            all(target_os = "macos", feature = "metal"),
            all(target_os = "linux", feature = "cuda")
        )),
        allow(dead_code)
    )]
    Owned(Vec<Complex64>),
}

/// Result of `run()`: a shot histogram and, on the SV/Metal backends, the final
/// state vector.
#[pyclass(name = "RunResult")]
pub(crate) struct RunResult {
    counts: BTreeMap<String, u64>,
    amps: Option<AmpStore>,
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
            Some(AmpStore::Cpu(state)) => Ok(PyArray1::from_slice_bound(py, state.amplitudes())),
            Some(AmpStore::Owned(v)) => Ok(PyArray1::from_slice_bound(py, v.as_slice())),
            None => Err(PyValueError::new_err(
                "statevector is only available on the \"sv\", \"metal\", and \
                 in-core \"cuda\"/\"cuda-f32\" backends",
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
/// (alias `"sv"`), `"stabilizer"` (alias `"stab"`), `"mps"`, `"metal"` (alias
/// `"gpu"`; FP32 Apple Silicon GPU, macOS `metal`-feature wheel only; `"auto"`
/// never picks it), or `"cuda"` / `"cuda-f32"` (NVIDIA GPU state vector, FP64 /
/// FP32, Linux `cuda`-feature wheel only). `RunResult.statevector()` is available
/// on `"sv"`, `"metal"`, and the in-core CUDA backends.
///
/// `precision` (`"auto"` default, `"f64"`, `"f32"`) steers an `auto` GPU pick:
/// with a CUDA wheel on a CUDA host, large dense circuits route to `cuda-f32` by
/// default (2× throughput, ~1e-5). On the CPU, `auto` stays FP64. Circuits past
/// the CUDA in-core cap (n>31 FP32 / n>30 FP64) need the CLI/Rust paged API —
/// Python raises rather than running them, as the paged state has no sampling.
///
/// **Breaking change vs v0.2:** the default was `"sv"`; it is now `"auto"`, so a
/// Clifford circuit routes to the stabilizer backend by default and
/// `RunResult.statevector()` raises unless you pass `backend="sv"`.
///
/// `Measure` instructions collapse the state once during execution; the
/// `shots` samples re-sample that single final state — they do NOT re-run
/// the circuit per shot (unlike Qiskit's per-shot execution model).
#[pyfunction]
#[pyo3(name = "run", signature = (circuit, *, shots = 1024, backend = "auto", precision = "auto", seed = None, noise = None))]
pub(crate) fn run_circuit(
    py: Python<'_>,
    circuit: &PyCircuit,
    shots: u32,
    backend: &str,
    precision: &str,
    seed: Option<u64>,
    noise: Option<&PyNoiseModel>,
) -> PyResult<RunResult> {
    let c = &circuit.inner;
    let n = c.num_qubits();

    // One shared parse site with the CLI: canonical names + `sv`/`stab` aliases.
    let request = BackendRequest::from_user_str(backend).map_err(PyValueError::new_err)?;
    let precision = parse_precision(precision)?;

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

    // Resolve `auto` through the same reach-aware heuristic the CLI uses (P5.11-06):
    // GPU availability is probed, and a precision preference steers the GPU pick.
    let env = SelectEnv {
        // Python `auto` has never GPU-probed Metal (it stays explicit-only via
        // backend="metal"); P5.11-06 adds CUDA auto-routing only, so Metal-auto
        // behaviour is unchanged here.
        metal_available: false,
        cuda_available: cuda_available(),
        precision,
    };
    let selection = match request {
        BackendRequest::Auto => select_explained_full(c, &env),
        BackendRequest::Fixed(k) => Selection {
            kind: k,
            reach: Reach::for_kind(k, n),
            reason: "explicit backend",
        },
    };
    let kind = selection.kind;

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
                amps: Some(AmpStore::Cpu(amps)),
            })
        }
        BackendKind::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                let (counts, amps) =
                    py.allow_threads(|| -> PyResult<(BTreeMap<String, u64>, Vec<Complex64>)> {
                        let mut be = match seed {
                            Some(s) => aleph_metal::MetalSvBackend::with_seed(s),
                            None => aleph_metal::MetalSvBackend::new(),
                        }
                        .map_err(|e| err("init metal backend", e))?;
                        let state = be.run_optimized(c).map_err(|e| err("run metal", e))?;
                        let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                        Ok((counts_map(&samples, n), state.to_aos_f64()))
                    })?;
                Ok(RunResult {
                    counts,
                    amps: Some(AmpStore::Owned(amps)),
                })
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err(PyValueError::new_err(
                    "the \"metal\" backend is not available in this build; \
                     build the wheel with `--features python,metal` on macOS \
                     (Apple Silicon)",
                ))
            }
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
        BackendKind::Cuda | BackendKind::CudaF32 => {
            run_cuda_py(py, c, kind, selection.reach, shots, seed, n)
        }
    }
}

/// CUDA GPU `run()` dispatch (`backend="cuda"` / `"cuda-f32"`, or an `auto` GPU
/// pick). In core: run the FP64/FP32 backend, sample, and return amplitudes like
/// the SV/Metal arms. Paged (`n` past the in-core cap) is rejected — the
/// out-of-core host state exposes no GPU sampling/amplitude readout to Python, so
/// it raises with guidance rather than silently degrading. Linux + `cuda` only.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn run_cuda_py(
    py: Python<'_>,
    c: &aleph_ir::Circuit,
    kind: BackendKind,
    reach: Reach,
    shots: u32,
    seed: Option<u64>,
    n: u32,
) -> PyResult<RunResult> {
    if let Reach::Paged { .. } = reach {
        return Err(PyValueError::new_err(format!(
            "n={n} exceeds the CUDA in-core cap; out-of-core paged runs are not \
             available from Python (the paged host state has no sampling/amplitude \
             readout) — use the `aleph` CLI or the Rust API"
        )));
    }
    let (counts, amps) =
        py.allow_threads(|| -> PyResult<(BTreeMap<String, u64>, Vec<Complex64>)> {
            // The FP32/FP64 in-core states both widen to Vec<Complex<f64>> via
            // amplitudes_vec(); run unoptimized (raw gates) as MPS/stab do.
            match kind {
                BackendKind::CudaF32 => {
                    let mut be = match seed {
                        Some(s) => aleph_cuda::CudaSvBackendF32::with_seed(s),
                        None => aleph_cuda::CudaSvBackendF32::new(),
                    }
                    .map_err(|e| err("init cuda-f32 backend", e))?;
                    let state = run(&mut be, c).map_err(|e| err("run cuda-f32", e))?;
                    let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                    Ok((counts_map(&samples, n), state.amplitudes_vec()))
                }
                BackendKind::Cuda => {
                    let mut be = match seed {
                        Some(s) => aleph_cuda::CudaSvBackend::with_seed(s),
                        None => aleph_cuda::CudaSvBackend::new(),
                    }
                    .map_err(|e| err("init cuda backend", e))?;
                    let state = run(&mut be, c).map_err(|e| err("run cuda", e))?;
                    let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                    Ok((counts_map(&samples, n), state.amplitudes_vec()))
                }
                _ => unreachable!("run_cuda_py is only called for CUDA kinds"),
            }
        })?;
    Ok(RunResult {
        counts,
        amps: Some(AmpStore::Owned(amps)),
    })
}

/// Stub when CUDA is not compiled in: `backend="cuda"`/`"cuda-f32"` (or an `auto`
/// CUDA pick, which can't happen without the probe) raises a clear build hint.
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn run_cuda_py(
    _py: Python<'_>,
    _c: &aleph_ir::Circuit,
    _kind: BackendKind,
    _reach: Reach,
    _shots: u32,
    _seed: Option<u64>,
    _n: u32,
) -> PyResult<RunResult> {
    Err(PyValueError::new_err(
        "the \"cuda\" / \"cuda-f32\" backend is not available in this build; \
         build the wheel with `--features python,cuda` on a Linux + NVIDIA host",
    ))
}

/// CUDA device probe for the `auto` heuristic (Linux + `cuda` only). Constructing
/// the FP32 backend compiles the kernels + acquires the device.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn cuda_available() -> bool {
    aleph_cuda::CudaSvBackendF32::new().is_ok()
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn cuda_available() -> bool {
    false
}

/// Parse the `precision=` kwarg into the backend policy's [`aleph_backend::Precision`].
fn parse_precision(s: &str) -> PyResult<aleph_backend::Precision> {
    match s {
        "auto" => Ok(aleph_backend::Precision::Auto),
        "f64" | "fp64" | "double" => Ok(aleph_backend::Precision::F64),
        "f32" | "fp32" | "single" => Ok(aleph_backend::Precision::F32),
        other => Err(PyValueError::new_err(format!(
            "unknown precision {other:?}; expected one of: auto, f64, f32"
        ))),
    }
}
