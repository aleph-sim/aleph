//! `CudaSvBackend` — the GPU state-vector [`Backend`]. Compiles the FP64 gate
//! kernels once at construction via NVRTC, then launches `apply_1q` / `apply_kq`
//! per gate on the device. Readout copies amplitudes back to the host.

use std::sync::Arc;

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateInstance, GateMatrix, PauliString};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use rand::{rngs::StdRng, SeedableRng};

use crate::common::{control_mask, diagonal_of, flatten_matrix, validate_and_extract};
use crate::sv::diag::{diag_1q_params, diag_kq_params, DiagKernels};
use crate::sv::kernel::{Gate1qParams, GateKqParams, APPLY_1Q, APPLY_KQ, SV_KERNELS_SRC};
use crate::sv::readout::GpuReadout;
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
    /// GPU-resident readout (P5-05): measurement / sampling / expectation /
    /// probabilities reduce on the device so only small results cross PCIe.
    readout: GpuReadout,
    /// Custom diagonal-gate kernels (P5-06).
    diag: DiagKernels,
    /// When set (default), diagonal gates divert to [`Self::diag`]; when cleared
    /// they fall back to the dense `apply_1q` / `apply_kq` path. The A/B switch
    /// for the P5-06 benchmark and the dual-path oracle test.
    custom_diag: bool,
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
        let diag = DiagKernels::new(&ctx)?;
        let readout = GpuReadout::new(&ctx)?;
        Ok(Self {
            ctx,
            f_1q,
            f_kq,
            _module: module,
            rng,
            qubit_cap: MAX_CUDA_QUBITS,
            readout,
            diag,
            custom_diag: true,
        })
    }

    /// Override the qubit cap (default [`MAX_CUDA_QUBITS`]). For large-memory
    /// benchmarks on a GPU that can hold the state.
    pub fn with_qubit_cap(mut self, cap: u32) -> Self {
        self.qubit_cap = cap;
        self
    }

    /// Enable (default) or disable routing diagonal gates to the custom
    /// `apply_diag` kernels (P5-06). Disabling forces the dense `apply_1q` /
    /// `apply_kq` path — the baseline arm of the P5-06 A/B benchmark.
    pub fn with_custom_kernels(mut self, on: bool) -> Self {
        self.custom_diag = on;
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

    /// Apply a diagonal gate via the custom `apply_diag` kernels (P5-06). `diag`
    /// is the interleaved `[re, im]` diagonal from [`diagonal_of`]; `qubits` are
    /// the operands in `gate.matrix()` MSB-first order; `controls` are external
    /// controls. The 1q case (Z/S/T/Rz/Phase and their controlled forms) skips
    /// the scratch upload entirely.
    fn launch_diag(
        &self,
        state: &mut CudaSvState,
        diag: &[f64],
        qubits: &[u32],
        controls: &[u32],
    ) -> Result<(), Error> {
        let ctrl_mask = control_mask(controls);
        if qubits.len() == 1 {
            let params = diag_1q_params(diag, qubits[0], ctrl_mask);
            self.diag.launch_1q(&self.ctx, state, params)
        } else {
            let params = diag_kq_params(qubits, ctrl_mask);
            self.diag.launch_kq(&self.ctx, state, params, diag)
        }
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
        let matrix = validate_and_extract(state.num_qubits, gate)?;
        // P5-06: a diagonal gate is one coalesced in-place phase multiply —
        // divert it to the custom `apply_diag` kernels instead of the dense path.
        if self.custom_diag {
            if let Some(diag) = diagonal_of(&matrix) {
                return self
                    .launch_diag(state, &diag, &gate.qubits, &gate.controls)
                    .map_err(to_backend_err);
            }
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
        self.readout.measure(&mut self.rng, state, qubit)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        self.readout.sample(&mut self.rng, state, shots)
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        self.readout.expectation_value(state, pauli)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        self.readout.probabilities(state, qubits)
    }
}
