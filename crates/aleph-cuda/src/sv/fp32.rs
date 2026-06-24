//! FP32 / mixed-precision GPU state-vector backend (P5.10-03).
//!
//! Mirrors [`crate::CudaSvBackend`] but stores the `2^n` amplitudes as **float**
//! complex (`2 · 2^n` `f32`). The GPU SV is memory-bandwidth bound, so halving
//! the bytes per sweep buys ~2× throughput, and halving the footprint reaches
//! **n=31 in-core** on the 20 GiB card (16 GiB of FP32) — one qubit past the
//! FP64 ceiling. Accuracy drops to ~1e-5 (oracle-pinned against the FP64
//! backend), the same accuracy ceiling as the Metal FP32 track.
//!
//! Gate **matrices stay `f64`** in the per-gate uniforms — the structs are
//! byte-identical to the FP64 ones, so this backend reuses the exact same host
//! param builders ([`gate_1q_params`], [`gate_kq_params`], the diag builders);
//! the kernels cast each matrix coefficient to float at point of use. Only the
//! amplitude buffer is FP32, which is where all the bandwidth lives.
//!
//! Scope: a per-gate apply path + host-side amplitude readout — enough to run
//! Tier-1/Tier-2 circuits and pin them against FP64. It does not implement the
//! GPU-resident readout / fusion / paging of the FP64 backend (those are
//! orthogonal and reusable later); drive it via [`CudaSvBackendF32::run`].

use aleph_backend::BackendError;
use aleph_core::{Complex, Gate, GateInstance, GateMatrix};
use aleph_ir::{Circuit, Instruction};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::common::{
    control_mask, diagonal_of, flatten_kq, flatten_matrix, validate_and_extract, validate_kq,
};
use crate::sv::backend::{gate_1q_params, gate_kq_params, to_backend_err};
use crate::sv::diag::{diag_1q_params, diag_kq_params};
use crate::sv::kernel::{CnotParams, Gate1qParams, GateKqParams};
use crate::{CudaContext, DeviceBuffer, Error};

/// Largest FP32 state held in-core: `n = 31` is 16 GiB of FP32 amplitudes
/// (`2^31 · 8 B`), inside the 20 GiB card — one qubit past
/// [`crate::MAX_CUDA_QUBITS`] (FP64's 16 GiB at n=30).
pub const MAX_CUDA_QUBITS_F32: u32 = 31;

const BLOCK: u32 = 256;

const SV_F32_SRC: &str = include_str!("kernels_f32.cu");
const APPLY_1Q_F32: &str = "apply_1q_f32";
const APPLY_CNOT_F32: &str = "apply_cnot_f32";
const APPLY_KQ_F32: &str = "apply_kq_f32";
const APPLY_DIAG_1Q_F32: &str = "apply_diag_1q_f32";
const APPLY_DIAG_F32: &str = "apply_diag_f32";

/// FP32 device-resident state vector: `2 · 2^n` interleaved `[re, im]` `f32`.
pub struct CudaSvStateF32 {
    pub(crate) num_qubits: u32,
    pub(crate) amps: DeviceBuffer<f32>,
    pub(crate) ctx: CudaContext,
    /// Reusable f64 scratch for the `apply_kq_f32` / `apply_diag_f32` matrix —
    /// matrices stay double (cast to float in the kernel), so this is f64.
    pub(crate) mat_scratch: Option<DeviceBuffer<f64>>,
}

impl core::fmt::Debug for CudaSvStateF32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaSvStateF32")
            .field("num_qubits", &self.num_qubits)
            .finish_non_exhaustive()
    }
}

impl CudaSvStateF32 {
    pub(crate) fn allocate(ctx: &CudaContext, num_qubits: u32) -> Result<Self, Error> {
        let n_amps = 1usize << num_qubits;
        let mut amps = DeviceBuffer::<f32>::zeros(ctx, 2 * n_amps)?;
        amps.write(ctx, &[1.0f32])?;
        Ok(Self {
            num_qubits,
            amps,
            ctx: ctx.clone(),
            mat_scratch: None,
        })
    }

    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Amplitudes widened to `Vec<Complex<f64>>` (one per basis state) for oracle
    /// comparison. Allocates a `2^n` host buffer — small-`n` use only.
    pub fn amplitudes_vec(&self) -> Vec<Complex> {
        let host = self.amps.to_vec(&self.ctx).expect("device->host f32 copy");
        host.chunks_exact(2)
            .map(|c| Complex::new(c[0] as f64, c[1] as f64))
            .collect()
    }

    /// `Σ |amp|²` (f64 accumulation), a large-`n` sanity check.
    pub fn norm_sqr(&self) -> f64 {
        let host = self.amps.to_vec(&self.ctx).expect("device->host f32 copy");
        host.chunks_exact(2)
            .map(|c| (c[0] as f64) * (c[0] as f64) + (c[1] as f64) * (c[1] as f64))
            .sum()
    }
}

/// FP32 GPU state-vector backend (P5.10-03).
pub struct CudaSvBackendF32 {
    ctx: CudaContext,
    f_1q: CudaFunction,
    f_cnot: CudaFunction,
    f_kq: CudaFunction,
    f_diag_1q: CudaFunction,
    f_diag: CudaFunction,
    _module: Arc<CudaModule>,
    qubit_cap: u32,
}

impl CudaSvBackendF32 {
    /// Construct on device 0, compiling the FP32 kernels via NVRTC. Returns
    /// [`Error::NoDevice`] on a GPU-less host so callers can skip cleanly.
    pub fn new() -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        let ptx = compile_ptx(SV_F32_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        Ok(Self {
            f_1q: module.load_function(APPLY_1Q_F32)?,
            f_cnot: module.load_function(APPLY_CNOT_F32)?,
            f_kq: module.load_function(APPLY_KQ_F32)?,
            f_diag_1q: module.load_function(APPLY_DIAG_1Q_F32)?,
            f_diag: module.load_function(APPLY_DIAG_F32)?,
            _module: module,
            ctx,
            qubit_cap: MAX_CUDA_QUBITS_F32,
        })
    }

    /// Override the qubit cap (default [`MAX_CUDA_QUBITS_F32`]).
    pub fn with_qubit_cap(mut self, cap: u32) -> Self {
        self.qubit_cap = cap;
        self
    }

    /// The CUDA context (cloned handle). `pub(crate)` so the out-of-core paged
    /// executor in [`crate::sv::paged_f32`] can allocate the group buffer and
    /// drive the tile copies on the same device/stream.
    pub(crate) fn ctx(&self) -> CudaContext {
        self.ctx.clone()
    }

    /// Run `circuit` end-to-end on the FP32 backend, returning the final state.
    /// Supports plain gates + barriers (the Tier-1/Tier-2 set); diagonal-phase /
    /// tiled-block / measure / reset are rejected (pass an unfused circuit).
    pub fn run(&mut self, circuit: &Circuit) -> Result<CudaSvStateF32, BackendError> {
        let n = circuit.num_qubits();
        if n == 0 && circuit.is_empty() {
            return Err(BackendError::EmptyCircuit);
        }
        if n > self.qubit_cap {
            return Err(BackendError::TooManyQubits {
                requested: n,
                limit: self.qubit_cap,
            });
        }
        let mut state = CudaSvStateF32::allocate(&self.ctx, n).map_err(to_backend_err)?;
        for inst in circuit.instructions() {
            match inst {
                Instruction::Gate(g) => self.apply_gate(&mut state, g)?,
                Instruction::Barrier(_) => {}
                Instruction::Measure { .. } => {
                    return Err(BackendError::UnsupportedInstruction { kind: "measure" })
                }
                Instruction::Reset(_) => {
                    return Err(BackendError::UnsupportedInstruction { kind: "reset" })
                }
                Instruction::DiagonalPhase(_) => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "diagonal-phase",
                    })
                }
                Instruction::TiledBlock(_) => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "tiled-block",
                    })
                }
            }
        }
        self.ctx.synchronize().map_err(to_backend_err)?;
        Ok(state)
    }

    /// Apply one gate — the FP32 mirror of the FP64 [`crate::CudaSvBackend`]
    /// dispatch (CNOT permutation, fused `UnitaryKq`, diagonal fast path, then
    /// the dense 1q/2q/3q kernels), launching the `*_f32` kernels.
    pub(crate) fn apply_gate(
        &mut self,
        state: &mut CudaSvStateF32,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        // Plain CNOT → permutation kernel.
        if matches!(gate.gate, Gate::Cnot) && gate.controls.is_empty() && gate.qubits.len() == 2 {
            let (c, t) = (gate.qubits[0], gate.qubits[1]);
            if c == t {
                return Err(BackendError::DuplicateQubit { qubit: c });
            }
            if c >= state.num_qubits || t >= state.num_qubits {
                return Err(BackendError::QubitOutOfRange {
                    qubit: c.max(t),
                    num_qubits: state.num_qubits,
                });
            }
            return self.launch_cnot(state, c, t).map_err(to_backend_err);
        }
        // Fused k≤5 dense block (no fixed GateMatrix).
        if let Gate::UnitaryKq { k, data } = &gate.gate {
            validate_kq(
                state.num_qubits,
                *k,
                data.len(),
                &gate.qubits,
                &gate.controls,
            )?;
            let params = gate_kq_params(&gate.qubits, &gate.controls);
            return self
                .launch_kq(state, params, &flatten_kq(data))
                .map_err(to_backend_err);
        }
        let matrix = validate_and_extract(state.num_qubits, gate)?;
        // Diagonal fast path.
        if let Some(diag) = diagonal_of(&matrix) {
            let ctrl_mask = control_mask(&gate.controls);
            return if gate.qubits.len() == 1 {
                let p = diag_1q_params(&diag, gate.qubits[0], ctrl_mask);
                self.launch_diag_1q(state, p).map_err(to_backend_err)
            } else {
                let p = diag_kq_params(&gate.qubits, ctrl_mask);
                self.launch_diag(state, p, &diag).map_err(to_backend_err)
            };
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

    fn launch_1q(&self, state: &mut CudaSvStateF32, params: Gate1qParams) -> Result<(), Error> {
        let n_pairs: u64 = 1 << (state.num_qubits - 1);
        let cfg = launch_config(n_pairs);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: signature (cplxf* amps, Gate1q g, u64 n_pairs); `amps` holds
        // 2·2^n f32 = 2^n cplxf, grid covers n_pairs with a guard.
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

    fn launch_cnot(
        &self,
        state: &mut CudaSvStateF32,
        control: u32,
        target: u32,
    ) -> Result<(), Error> {
        let n_groups: u64 = 1 << (state.num_qubits - 2);
        let cfg = launch_config(n_groups);
        let stream = self.ctx.stream();
        let params = CnotParams {
            ctrl: control,
            targ: target,
            lo: control.min(target),
            hi: control.max(target),
        };
        let amps = state.amps.slice_mut();
        // SAFETY: signature (cplxf* amps, Cnot g, u64 n_groups); grid covers
        // 2^(n-2) control=1 pairs with a guard; control/target validated distinct.
        unsafe {
            stream
                .launch_builder(&self.f_cnot)
                .arg(amps)
                .arg(&params)
                .arg(&n_groups)
                .launch(cfg)?;
        }
        Ok(())
    }

    fn launch_kq(
        &self,
        state: &mut CudaSvStateF32,
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
        let amps = state.amps.slice_mut();
        let mat_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set above")
            .slice();
        // SAFETY: signature (cplxf* amps, const cplx* mat, GateKq g, u64 n_groups);
        // `mat_dev` holds 2^k·2^k f64 cplx, `amps` 2^n cplxf, grid covers n_groups.
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

    fn launch_diag_1q(
        &self,
        state: &mut CudaSvStateF32,
        params: crate::sv::diag::Diag1qParams,
    ) -> Result<(), Error> {
        let n_amps: u64 = 1 << state.num_qubits;
        let cfg = launch_config(n_amps);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: signature (cplxf* amps, Diag1q g, u64 n_amps); in-place per-amp
        // multiply, grid covers 2^n with a guard.
        unsafe {
            stream
                .launch_builder(&self.f_diag_1q)
                .arg(amps)
                .arg(&params)
                .arg(&n_amps)
                .launch(cfg)?;
        }
        Ok(())
    }

    fn launch_diag(
        &self,
        state: &mut CudaSvStateF32,
        params: crate::sv::diag::DiagKqParams,
        diag: &[f64],
    ) -> Result<(), Error> {
        match state.mat_scratch.as_mut() {
            Some(buf) => buf.write(&self.ctx, diag)?,
            None => state.mat_scratch = Some(DeviceBuffer::<f64>::from_slice(&self.ctx, diag)?),
        }
        let n_amps: u64 = 1 << state.num_qubits;
        let cfg = launch_config(n_amps);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        let diag_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set above")
            .slice();
        // SAFETY: signature (cplxf* amps, const cplx* diag, DiagK g, u64 n_amps);
        // `diag_dev` holds 2^k f64 cplx, grid covers 2^n with a guard.
        unsafe {
            stream
                .launch_builder(&self.f_diag)
                .arg(amps)
                .arg(diag_dev)
                .arg(&params)
                .arg(&n_amps)
                .launch(cfg)?;
        }
        Ok(())
    }
}

fn launch_config(n_threads: u64) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
