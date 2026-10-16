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
    run, run_optimized, select_with_probes, Backend, BackendKind, BackendRequest, Reach,
};
use aleph_mps::MpsBackend;
use aleph_stab::StabilizerBackend;
use aleph_sv::NaiveSvBackend;
use numpy::{Complex64, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// CPU state or a GPU-owned state materialized on the first explicit read.
/// GPU cache reads return independent NumPy arrays; successful materialization
/// releases the device state. Counts-only results retain it until result cleanup.
enum AmpStore {
    Cpu(aleph_sv::CpuState),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(crate::lazy_amplitudes::LazyAmplitudes<aleph_metal::MetalSvReadout>),
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    Cuda(crate::lazy_amplitudes::LazyAmplitudes<aleph_cuda::CudaSvReadout>),
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    CudaF32(crate::lazy_amplitudes::LazyAmplitudes<aleph_cuda::CudaSvReadoutF32>),
}

/// Result of `run()`: a shot histogram and, on the SV/Metal backends, the final
/// state vector.
#[pyclass(name = "RunResult")]
pub(crate) struct RunResult {
    counts: BTreeMap<String, u64>,
    amps: Option<AmpStore>,
    cuda_stats: BTreeMap<String, f64>,
}

#[pymethods]
impl RunResult {
    /// CUDA plan/stage diagnostics. Empty for other backends. Kernel counters are
    /// process-wide deltas; do not attribute them per call during concurrent runs.
    fn execution_stats(&self) -> BTreeMap<String, f64> {
        self.cuda_stats.clone()
    }

    /// Histogram of sampled bitstrings. Qubit 0 is the RIGHTMOST character
    /// (qubit 0 is the LSB of the amplitude index — ADR 0004; the leftmost
    /// character is qubit n-1), matching the CLI's |q_{n-1}…q_0⟩ output.
    fn counts(&self) -> BTreeMap<String, u64> {
        self.counts.clone()
    }

    /// Final state vector as a numpy `complex128` array of shape `(2**n,)`,
    /// CPU SV and in-core GPU backends only. GPU download/widening is lazy:
    /// the first read caches host amplitudes and releases the retained GPU state;
    /// subsequent reads copy the cache into independent arrays. GPU synchronization,
    /// download/widening and cache lookup run without the GIL; the final NumPy
    /// copy uses the same GIL-held path as CPU/previous GPU results. CUDA copy errors raise ValueError
    /// and leave the device state available for another attempt.
    ///
    /// One contiguous O(state) buffer (a copy of the backend's amplitude
    /// slice) rather than 2^n boxed Python complex objects — at n=25 the old
    /// list cost ~1.9 GiB of objects for a 512 MiB state. `aleph_core::Complex`
    /// is `num_complex::Complex<f64>` (== `numpy::Complex64`) stored
    /// contiguously, so `from_slice` is a single memcpy.
    fn statevector<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        match &self.amps {
            Some(AmpStore::Cpu(state)) => Ok(PyArray1::from_slice_bound(py, state.amplitudes())),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Some(AmpStore::Metal(lazy)) => {
                let host = py
                    .allow_threads(|| lazy.materialize(|s| Ok(s.to_aos_f64())))
                    .map_err(|e| err("materialize metal statevector", e))?;
                Ok(PyArray1::from_slice_bound(py, host.as_slice()))
            }
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Some(AmpStore::Cuda(lazy)) => {
                let host = py
                    .allow_threads(|| {
                        lazy.materialize(|s| s.try_amplitudes_vec().map_err(|e| e.to_string()))
                    })
                    .map_err(|e| err("materialize cuda statevector", e))?;
                aleph_cuda::memory::ensure_host_bytes(
                    (host.len() as u64)
                        .checked_mul(16)
                        .ok_or_else(|| PyValueError::new_err("host export size overflow"))?,
                )
                .map_err(PyValueError::new_err)?;
                Ok(PyArray1::from_slice_bound(py, host.as_slice()))
            }
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Some(AmpStore::CudaF32(lazy)) => {
                let host = py
                    .allow_threads(|| {
                        lazy.materialize(|s| s.try_amplitudes_vec().map_err(|e| e.to_string()))
                    })
                    .map_err(|e| err("materialize cuda-f32 statevector", e))?;
                aleph_cuda::memory::ensure_host_bytes(
                    (host.len() as u64)
                        .checked_mul(16)
                        .ok_or_else(|| PyValueError::new_err("host export size overflow"))?,
                )
                .map_err(PyValueError::new_err)?;
                Ok(PyArray1::from_slice_bound(py, host.as_slice()))
            }
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

/// Format sorted sparse counts into the public bitstring dictionary.
fn sparse_to_counts(counts: &[(u64, u64)], num_qubits: u32) -> BTreeMap<String, u64> {
    let width = num_qubits as usize;
    counts
        .iter()
        .map(|&(index, count)| (format!("{index:0width$b}"), count))
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
/// GPU results retain their device state until the first statevector read or
/// result destruction. Counts-only calls never materialize host amplitudes.
///
/// `precision` (`"auto"` default, `"f64"`, `"f32"`) steers an `auto` GPU pick:
/// with a CUDA wheel on a CUDA host, large dense circuits route to `cuda-f32` by
/// default (2× throughput, ~1e-5). On the CPU, `auto` stays FP64. Circuits past
/// device/output memory budget raise before large allocation. Counts require
/// sampling scratch in addition to the state. The Rust paged API is norm-only;
/// it cannot substitute for requested Python outputs.
///
/// **Breaking change vs v0.2:** the default was `"sv"`; it is now `"auto"`, so a
/// Clifford circuit routes to the stabilizer backend by default and
/// `RunResult.statevector()` raises unless you pass `backend="sv"`.
///
/// `noise_memory_budget_bytes` overrides the 1 GiB incremental noisy buffer
/// budget; `noise_max_workers` caps trajectory contexts within the Rayon pool.
/// Both require `noise=`. The budget excludes inputs, allocator/runtime overhead
/// and Python result conversion; it is not an RSS cap.
///
/// CUDA uses full-state-safe fusion and layered dispatch by default. Set
/// `cuda_optimize=False` for the raw A/B path. Both paths enforce ordinary FP32
/// arithmetic; TF32 is disabled and fusion width is 3, including on capable GPUs.
/// `RunResult.execution_stats()` exposes CUDA planning, launch and stage diagnostics.
///
/// `Measure` instructions collapse the state once during execution; the
/// `shots` samples re-sample that single final state — they do NOT re-run
/// the circuit per shot (unlike Qiskit's per-shot execution model).
#[pyfunction]
#[pyo3(name = "run", signature = (circuit, *, shots = 1024, backend = "auto", precision = "auto", seed = None, noise = None, noise_memory_budget_bytes = None, noise_max_workers = None, cuda_optimize = true))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_circuit(
    py: Python<'_>,
    circuit: &PyCircuit,
    shots: u32,
    backend: &str,
    precision: &str,
    seed: Option<u64>,
    noise: Option<&PyNoiseModel>,
    noise_memory_budget_bytes: Option<usize>,
    noise_max_workers: Option<usize>,
    cuda_optimize: bool,
) -> PyResult<RunResult> {
    if noise.is_some() && !cuda_optimize {
        return Err(PyValueError::new_err(
            "CUDA execution options cannot be combined with noise",
        ));
    }
    if noise.is_none() && (noise_memory_budget_bytes.is_some() || noise_max_workers.is_some()) {
        return Err(PyValueError::new_err(
            "noise_memory_budget_bytes and noise_max_workers require noise=",
        ));
    }
    let noise_options = aleph_sv::noise::NoiseOptions {
        memory_budget_bytes: noise_memory_budget_bytes
            .unwrap_or_else(|| aleph_sv::noise::NoiseOptions::default().memory_budget_bytes),
        max_workers: noise_max_workers,
    };
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
            let sparse = aleph_sv::noise::run_noisy_sparse(c, model, shots, seed, noise_options)
                .map_err(|e| {
                    let context = if matches!(e, aleph_sv::noise::NoiseError::MemoryBudget { .. }) {
                        "run noisy (adjust noise_memory_budget_bytes)"
                    } else {
                        "run noisy"
                    };
                    err(context, e)
                })?;
            Ok(sparse_to_counts(&sparse, n))
        })?;
        return Ok(RunResult {
            counts,
            amps: None,
            cuda_stats: Default::default(),
        });
    }

    // Metal remains explicit-only in Python. Discovery is lazy and never compiles
    // kernels; construction of the selected backend below surfaces init errors.
    let selection = select_with_probes(c, request, precision, || false, cuda_available);
    let kind = selection.kind;

    if !cuda_optimize && !matches!(kind, BackendKind::Cuda | BackendKind::CudaF32) {
        return Err(PyValueError::new_err(
            "cuda_optimize option require a CUDA backend",
        ));
    }
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
                cuda_stats: Default::default(),
                amps: Some(AmpStore::Cpu(amps)),
            })
        }
        BackendKind::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                let (counts, amps) =
                    py.allow_threads(|| -> PyResult<(BTreeMap<String, u64>, AmpStore)> {
                        let mut be = match seed {
                            Some(s) => aleph_metal::MetalSvBackend::with_seed(s),
                            None => aleph_metal::MetalSvBackend::new(),
                        }
                        .map_err(|e| err("init metal backend", e))?;
                        let state = be.run_optimized(c).map_err(|e| err("run metal", e))?;
                        let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                        Ok((
                            counts_map(&samples, n),
                            AmpStore::Metal(crate::lazy_amplitudes::LazyAmplitudes::new(
                                state.into_readout(),
                                (1u64 << n) * 8,
                            )),
                        ))
                    })?;
                Ok(RunResult {
                    counts,
                    cuda_stats: Default::default(),
                    amps: Some(amps),
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
            Ok(RunResult {
                counts,
                amps: None,
                cuda_stats: Default::default(),
            })
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
            Ok(RunResult {
                counts,
                amps: None,
                cuda_stats: Default::default(),
            })
        }
        BackendKind::Cuda | BackendKind::CudaF32 => {
            run_cuda_py(py, c, kind, selection.reach, shots, seed, n, cuda_optimize)
        }
    }
}

/// CUDA run with shared operation-aware device/host preflight. Counts retain
/// device amplitudes; host export is checked separately when explicitly requested.
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn run_cuda_py(
    py: Python<'_>,
    c: &aleph_ir::Circuit,
    kind: BackendKind,
    _reach: Reach,
    shots: u32,
    seed: Option<u64>,
    n: u32,
    optimize: bool,
) -> PyResult<RunResult> {
    let (counts, amps, cuda_stats) = py.allow_threads(|| -> PyResult<_> {
        let total = std::time::Instant::now();
        let options = aleph_cuda::CudaExecutionOptions { optimize };
        let views = aleph_cuda::CudaReadoutRequest {
            shots,
            ..Default::default()
        };
        macro_rules! execute {
            ($backend:ident,$store:ident,$bytes:expr) => {{
                let (mut be, plan) = aleph_cuda::prepare_cuda_execution(c, options, views, || {
                    let be = match seed {
                        Some(s) => aleph_cuda::$backend::with_seed(s),
                        None => aleph_cuda::$backend::new(),
                    }?;
                    Ok(be.with_qubit_cap(n))
                })
                .map_err(|e| err("prepare CUDA execution", e))?;
                let before = aleph_cuda::execution_counters();
                let start = std::time::Instant::now();
                let state = plan.execute(&mut be).map_err(|e| err("run CUDA plan", e))?;
                let execution_ms = start.elapsed().as_secs_f64() * 1000.;
                let after = aleph_cuda::execution_counters();
                let start = std::time::Instant::now();
                let samples = be.sample(&state, shots).map_err(|e| err("sample", e))?;
                let sampling_ms = start.elapsed().as_secs_f64() * 1000.;
                let start = std::time::Instant::now();
                let counts = counts_map(&samples, n);
                let counts_ms = start.elapsed().as_secs_f64() * 1000.;
                let mut stats = plan.diagnostics();
                for (key, value) in [
                    ("execution_ms", execution_ms),
                    ("sampling_ms", sampling_ms),
                    ("counts_ms", counts_ms),
                    ("kernel_launches_process_delta", (after.0 - before.0) as f64),
                    ("layer_launches_process_delta", (after.1 - before.1) as f64),
                    ("tf32_launches_process_delta", (after.2 - before.2) as f64),
                    ("total_native_ms", total.elapsed().as_secs_f64() * 1000.),
                ] {
                    stats.insert(key.into(), value);
                }
                Ok((
                    counts,
                    AmpStore::$store(crate::lazy_amplitudes::LazyAmplitudes::new(
                        state
                            .into_readout()
                            .map_err(|e| err("finish CUDA state", e))?,
                        (1u64 << n) * $bytes,
                    )),
                    stats,
                ))
            }};
        }
        match kind {
            BackendKind::Cuda => execute!(CudaSvBackend, Cuda, 16),
            BackendKind::CudaF32 => execute!(CudaSvBackendF32, CudaF32, 8),
            _ => unreachable!(),
        }
    })?;
    Ok(RunResult {
        counts,
        amps: Some(amps),
        cuda_stats,
    })
}

/// Stub when CUDA is not compiled in: `backend="cuda"`/`"cuda-f32"` (or an `auto`
/// CUDA pick, which can't happen without the probe) raises a clear build hint.
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
fn run_cuda_py(
    _py: Python<'_>,
    _c: &aleph_ir::Circuit,
    _kind: BackendKind,
    _reach: Reach,
    _shots: u32,
    _seed: Option<u64>,
    _n: u32,
    _optimize: bool,
) -> PyResult<RunResult> {
    Err(PyValueError::new_err(
        "the \"cuda\" / \"cuda-f32\" backend is not available in this build; \
         build the wheel with `--features python,cuda` on a Linux + NVIDIA host",
    ))
}

/// CUDA device discovery for auto selection (Linux + cuda only). No backend
/// is constructed and no kernels are compiled; initialization happens after selection.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn cuda_available() -> bool {
    aleph_cuda::device_available()
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

/// Internal measurement hook: SV construction attempts, not device discovery.
#[pyfunction]
pub(crate) fn _gpu_backend_init_count() -> u64 {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        aleph_cuda::sv_backend_init_count()
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        aleph_metal::sv_backend_init_count()
    }
    #[cfg(not(any(
        all(target_os = "linux", feature = "cuda"),
        all(target_os = "macos", feature = "metal")
    )))]
    {
        0
    }
}

/// Internal measurement hook: materializations, live device amplitude bytes,
/// cached host amplitude bytes, and CUDA device-to-host bytes (zero on Metal).
#[pyfunction]
pub(crate) fn _gpu_result_stats() -> (u64, u64, u64, u64) {
    #[cfg(any(
        all(target_os = "macos", feature = "metal"),
        all(target_os = "linux", feature = "cuda")
    ))]
    let (materializations, resident, cached) = crate::lazy_amplitudes::stats();
    #[cfg(not(any(
        all(target_os = "macos", feature = "metal"),
        all(target_os = "linux", feature = "cuda")
    )))]
    let (materializations, resident, cached) = (0, 0, 0);
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let dtoh = aleph_cuda::device_dtoh_bytes();
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    let dtoh = 0;
    (materializations, resident, cached, dtoh)
}
