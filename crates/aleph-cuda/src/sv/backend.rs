//! `CudaSvBackend` — the GPU state-vector [`Backend`]. Compiles the FP64 gate
//! kernels once at construction via NVRTC, then launches `apply_1q` / `apply_kq`
//! per gate on the device. Readout copies amplitudes back to the host.

use std::sync::Arc;

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateError, GateInstance, GateMatrix, PauliString};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use rand::{rngs::StdRng, SeedableRng};

use crate::sv::kernel::{Gate1qParams, GateKqParams, APPLY_1Q, APPLY_KQ, SV_KERNELS_SRC};
use crate::sv::readout;
use crate::sv::state::{CudaSvState, MAX_CUDA_QUBITS};
use crate::{CudaContext, DeviceBuffer, Error};

/// Threads per block. 256 is the standard sweet spot for memory-bound SV
/// kernels (NVIDIA cuStateVec, QuEST); occupancy is bandwidth- not block-bound.
const BLOCK: u32 = 256;

/// GPU state-vector backend (FP64).
pub struct CudaSvBackend {
    ctx: CudaContext,
    f_1q: CudaFunction,
    f_kq: CudaFunction,
    // Keeps the loaded module alive for the lifetime of the functions.
    _module: Arc<CudaModule>,
    rng: StdRng,
    qubit_cap: u32,
}

impl CudaSvBackend {
    /// Construct on device 0 with an entropy-seeded RNG. Returns
    /// [`Error::NoDevice`] on a GPU-less host so callers can skip cleanly.
    pub fn new() -> Result<Self, Error> {
        Self::build(StdRng::from_entropy())
    }

    /// Construct with an explicit seed; measurement/sampling are reproducible
    /// across processes for a given seed.
    pub fn with_seed(seed: u64) -> Result<Self, Error> {
        Self::build(StdRng::seed_from_u64(seed))
    }

    fn build(rng: StdRng) -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        // NVRTC compiles the CUDA C++ to PTX at runtime (mirrors a CPU JIT) —
        // no nvcc, no build-time CUDA SDK; the driver JITs PTX→sm at load.
        let ptx = compile_ptx(SV_KERNELS_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        let f_1q = module.load_function(APPLY_1Q)?;
        let f_kq = module.load_function(APPLY_KQ)?;
        Ok(Self {
            ctx,
            f_1q,
            f_kq,
            _module: module,
            rng,
            qubit_cap: MAX_CUDA_QUBITS,
        })
    }

    /// Override the qubit cap (default [`MAX_CUDA_QUBITS`]). For large-memory
    /// benchmarks on a GPU that can hold the state.
    pub fn with_qubit_cap(mut self, cap: u32) -> Self {
        self.qubit_cap = cap;
        self
    }

    /// Launch `apply_1q` over `2^(n-1)` amplitude pairs.
    fn launch_1q(&self, state: &mut CudaSvState, params: Gate1qParams) -> Result<(), Error> {
        let n_pairs: u64 = 1 << (state.num_qubits - 1);
        let cfg = launch_config(n_pairs);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: kernel signature is (cplx* amps, Gate1q g, u64 n_pairs); the
        // args below match in order and type, `amps` has 2·2^n f64 = 2^n cplx,
        // and the grid covers exactly `n_pairs` threads with an in-bounds guard.
        unsafe {
            stream
                .launch_builder(&self.f_1q)
                .arg(amps)
                .arg(&params)
                .arg(&n_pairs)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Upload `mat` (interleaved row-major `2^k×2^k`) to the reusable scratch and
    /// launch `apply_kq` over `2^(n-k)` groups.
    fn launch_kq(
        &self,
        state: &mut CudaSvState,
        params: GateKqParams,
        mat: &[f64],
    ) -> Result<(), Error> {
        match state.mat_scratch.as_mut() {
            Some(buf) => buf.write(&self.ctx, mat)?,
            None => state.mat_scratch = Some(DeviceBuffer::<f64>::from_slice(&self.ctx, mat)?),
        }
        let n_groups: u64 = 1 << (state.num_qubits - params.k);
        let cfg = launch_config(n_groups);
        let stream = self.ctx.stream();
        // Disjoint field borrows: `amps` (&mut) and `mat_scratch` (&) are
        // different fields of `state`, so both borrows coexist.
        let amps = state.amps.slice_mut();
        let mat_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set above")
            .slice();
        // SAFETY: kernel signature is (cplx* amps, const cplx* mat, GateKq g,
        // u64 n_groups); args match in order/type. `mat_dev` holds 2^k·2^k cplx,
        // `amps` holds 2^n cplx, and the grid covers `n_groups` with a guard.
        unsafe {
            stream
                .launch_builder(&self.f_kq)
                .arg(amps)
                .arg(mat_dev)
                .arg(&params)
                .arg(&n_groups)
                .launch(cfg)?;
        }
        Ok(())
    }
}

/// `((n + BLOCK - 1) / BLOCK)` blocks of `BLOCK` threads, ≥1 block.
fn launch_config(n_threads: u64) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `max_{i,j} |(U·U†)_{i,j} - δ_{i,j}|`, mirroring `aleph_sv::validation`
/// (crate-private there). NaN-propagating: any NaN entry yields NaN so the
/// caller's `is_finite` reject fires (ADR 0006).
fn unitarity_deviation(matrix: &GateMatrix) -> f64 {
    fn max_dev<const N: usize>(m: &[[Complex; N]; N]) -> f64 {
        let mut worst = 0.0_f64;
        for (i, row_i) in m.iter().enumerate() {
            for (j, row_j) in m.iter().enumerate() {
                let mut acc = Complex::new(0.0, 0.0);
                for (a, b) in row_i.iter().zip(row_j.iter()) {
                    acc += a * b.conj();
                }
                let want = if i == j { 1.0 } else { 0.0 };
                let dev = (acc - Complex::new(want, 0.0)).norm();
                if dev.is_nan() {
                    return f64::NAN;
                }
                if dev > worst {
                    worst = dev;
                }
            }
        }
        worst
    }
    match matrix {
        GateMatrix::M2x2(m) => max_dev::<2>(m),
        GateMatrix::M4x4(m) => max_dev::<4>(m),
        GateMatrix::M8x8(m) => max_dev::<8>(m),
    }
}

/// Control-qubit bitmask `Σ 1 << c`.
fn control_mask(controls: &[u32]) -> u32 {
    controls.iter().fold(0u32, |acc, &c| acc | (1u32 << c))
}

/// Build a `Gate1qParams` from a 2×2 matrix and external controls.
fn gate_1q_params(m: &[[Complex; 2]; 2], target: u32, controls: &[u32]) -> Gate1qParams {
    Gate1qParams {
        m: [
            m[0][0].re, m[0][0].im, m[0][1].re, m[0][1].im, m[1][0].re, m[1][0].im, m[1][1].re,
            m[1][1].im,
        ],
        target,
        t_bit: 1u32 << target,
        ctrl_mask: control_mask(controls),
        _pad: 0,
    }
}

/// Build a `GateKqParams` from the target qubits and external controls.
///
/// `gate.matrix()` lays multi-qubit operands out **MSB-first**: `qubits[0]` is
/// the most-significant matrix-index bit, `qubits[k-1]` the least — the same
/// operand order the CPU kernels use (verified by the CPU-SV oracle; a CNOT on
/// `[control, target]` must swap basis indices, not no-op). So matrix-index bit
/// `b` (0 = LSB) corresponds to `qubits[k-1-b]`: `qbit[b] = 1 << qubits[k-1-b]`.
/// `sorted` (ascending target positions, operand-order-independent) drives the
/// kernel's zero-bit insertion.
fn gate_kq_params(qubits: &[u32], controls: &[u32]) -> GateKqParams {
    let k = qubits.len();
    let mut qbit = [0u32; 5];
    let mut sorted = [0u32; 5];
    for (b, slot) in qbit.iter_mut().take(k).enumerate() {
        *slot = 1u32 << qubits[k - 1 - b];
    }
    sorted[..k].copy_from_slice(qubits);
    sorted[..k].sort_unstable();
    GateKqParams {
        k: k as u32,
        qbit,
        sorted,
        ctrl_mask: control_mask(controls),
    }
}

/// Row-major interleaved `[re, im]` of an `N×N` complex matrix.
fn flatten_matrix<const N: usize>(m: &[[Complex; N]; N]) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * N * N);
    for row in m {
        for z in row {
            out.push(z.re);
            out.push(z.im);
        }
    }
    out
}

/// Map a CUDA-layer error to a backend error. Launch/transfer failures on a
/// working GPU indicate an internal fault, not user input; richer plumbing is a
/// follow-up (the variant set is shared across all backends).
fn to_backend_err(_e: Error) -> BackendError {
    BackendError::InvalidState {
        reason: "CUDA backend failure (compile/launch/transfer)",
    }
}

impl Backend for CudaSvBackend {
    type State = CudaSvState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits == 0 {
            return Err(BackendError::InvalidState {
                reason: "zero-qubit state",
            });
        }
        if num_qubits > self.qubit_cap {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: self.qubit_cap,
            });
        }
        CudaSvState::allocate(&self.ctx, num_qubits).map_err(to_backend_err)
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let n = state.num_qubits;
        let expected = gate.gate.arity();
        let got = gate.qubits.len();
        if expected != got {
            return Err(BackendError::ArityMismatch {
                kind: gate.gate.name(),
                expected,
                got,
            });
        }
        let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for &q in gate.qubits.iter().chain(gate.controls.iter()) {
            if q >= n {
                return Err(BackendError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
            if seen.contains(&q) {
                return Err(BackendError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        // Fused k-qubit blocks (run_optimized) use a different operand order and
        // are out of scope for this first GPU backend — raw `run` never emits
        // them. Reject explicitly rather than mis-apply.
        if matches!(gate.gate, aleph_core::Gate::UnitaryKq { .. }) {
            return Err(BackendError::UnsupportedGate {
                kind: gate.gate.name(),
            });
        }
        let matrix = gate.gate.matrix().map_err(|e| match e {
            GateError::SymbolicParam => BackendError::SymbolicParam,
            GateError::NonFiniteParam => BackendError::NonFiniteParam {
                kind: gate.gate.name(),
            },
            GateError::Unrepresentable => BackendError::UnsupportedGate {
                kind: gate.gate.name(),
            },
        })?;
        let deviation = unitarity_deviation(&matrix);
        if !deviation.is_finite() || deviation > aleph_core::AMPLITUDE_TOL {
            return Err(BackendError::NonUnitaryMatrix { deviation });
        }
        match matrix {
            GateMatrix::M2x2(m) => {
                let params = gate_1q_params(&m, gate.qubits[0], &gate.controls);
                self.launch_1q(state, params).map_err(to_backend_err)
            }
            GateMatrix::M4x4(m) => {
                let params = gate_kq_params(&[gate.qubits[0], gate.qubits[1]], &gate.controls);
                self.launch_kq(state, params, &flatten_matrix(&m))
                    .map_err(to_backend_err)
            }
            GateMatrix::M8x8(m) => {
                let params = gate_kq_params(
                    &[gate.qubits[0], gate.qubits[1], gate.qubits[2]],
                    &gate.controls,
                );
                self.launch_kq(state, params, &flatten_matrix(&m))
                    .map_err(to_backend_err)
            }
        }
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        let mut amps = state.amplitudes_vec();
        let outcome = readout::measure(&mut self.rng, &mut amps, state.num_qubits, qubit)?;
        state.write_host(&complex_to_f64(&amps));
        Ok(outcome)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        let amps = state.amplitudes_vec();
        readout::sample(&mut self.rng, &amps, state.num_qubits, shots)
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        let amps = state.amplitudes_vec();
        readout::expectation_value(&amps, state.num_qubits, pauli)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        let amps = state.amplitudes_vec();
        readout::probabilities(&amps, state.num_qubits, qubits)
    }
}

/// Interleave `[re, im]` for upload back to the device.
fn complex_to_f64(amps: &[Complex]) -> Vec<f64> {
    let mut out = Vec::with_capacity(amps.len() * 2);
    for a in amps {
        out.push(a.re);
        out.push(a.im);
    }
    out
}
