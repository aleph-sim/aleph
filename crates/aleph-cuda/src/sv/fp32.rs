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
//! P5.11-04 made it a **first-class [`Backend`]**: GPU-resident readout
//! ([`GpuReadoutF32`] — measure / sample / expectation / probabilities reduce on
//! the device) plus the FP64 backend's apply-side throughput levers ported to
//! FP32 (CNOT permutation, warp-tiled fused blocks `apply_kq_tiled_f32`,
//! disjoint-1q-layer batching `apply_1q_multi_f32` via [`Self::run_layered`], and
//! fused diagonal-phase polynomials `apply_phase_poly_f32`). IR fusion
//! ([`crate::fuse_for_gpu`]) is precision-agnostic and reused as-is, so its wins
//! **stack on the 2× precision win**. Drive it via the [`Backend`] trait
//! ([`aleph_backend::run`]), [`Self::run_layered`], or the per-gate [`Self::run`].

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, Gate, GateInstance, GateMatrix, PauliString};
use aleph_ir::{Circuit, DiagonalPhase, Instruction};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx, compile_ptx_with_opts, CompileOptions};
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;

use crate::common::{
    control_mask, diagonal_of, flatten_kq, flatten_matrix, validate_and_extract, validate_kq,
};
use crate::sv::backend::{gate_1q_params, gate_kq_params, to_backend_err};
use crate::sv::diag::{diag_1q_params, diag_kq_params};
use crate::sv::kernel::{
    CnotParams, Gate1qParams, GateKqParams, Multi1qParams, DEFAULT_LAYER_BATCH, MAX_LAYER_BATCH,
};
use crate::sv::readout_f32::GpuReadoutF32;
use crate::{CudaContext, DeviceBuffer, Error};

/// Largest FP32 state held in-core: `n = 31` is 16 GiB of FP32 amplitudes
/// (`2^31 · 8 B`), inside the 20 GiB card — one qubit past
/// [`crate::MAX_CUDA_QUBITS`] (FP64's 16 GiB at n=30).
pub const MAX_CUDA_QUBITS_F32: u32 = 31;

const BLOCK: u32 = 256;

const SV_F32_SRC: &str = include_str!("kernels_f32.cu");
const APPLY_1Q_F32: &str = "apply_1q_f32";
const APPLY_1Q_MULTI_F32: &str = "apply_1q_multi_f32";
const APPLY_CNOT_F32: &str = "apply_cnot_f32";
const APPLY_PHASE_POLY_F32: &str = "apply_phase_poly_f32";
const APPLY_KQ_F32: &str = "apply_kq_f32";
const APPLY_KQ_TILED_F32: &str = "apply_kq_tiled_f32";
const APPLY_DIAG_1Q_F32: &str = "apply_diag_1q_f32";
const APPLY_DIAG_F32: &str = "apply_diag_f32";

/// TF32 tensor-core fused-block kernels (P5.11-05), in their own NVRTC module —
/// they need `--gpu-architecture=sm_89` + the CUDA `mma.h` include path, which the
/// base FP32 module (NVRTC-default arch) does not set.
const SV_TF32_SRC: &str = include_str!("kernels_tf32.cu");
const APPLY_KQ_TF32_K4: &str = "apply_kq_tf32_k4";
const APPLY_KQ_TF32_K5: &str = "apply_kq_tf32_k5";
/// Warps per block for the TF32 kernels — must match `K4_WARPS` / `K5_WARPS` in
/// `kernels_tf32.cu` (each warp owns one 16-group WMMA tile).
const TF32_K4_WARPS: u32 = 4;
const TF32_K5_WARPS: u32 = 2;
/// Group-tile width — `WMMA_N` in the kernel (16 group vectors per WMMA pass).
const TF32_TILE: u64 = 16;

/// The loaded TF32 tensor-core kernels + their module (kept alive). Held in an
/// `Option` on the backend so a host that can't compile the WMMA module degrades
/// to the tiled path instead of failing construction.
struct Tf32Kernels {
    k4: CudaFunction,
    k5: CudaFunction,
    _module: Arc<CudaModule>,
}

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

/// FP32 GPU state-vector backend (P5.10-03; first-class `Backend` in P5.11-04).
///
/// Mirrors [`crate::CudaSvBackend`] at FP32: the same apply-side throughput levers
/// (CNOT permutation, warp-tiled fused blocks, disjoint-1q-layer batching, fused
/// diagonal-phase polynomials) and GPU-resident readout, stacked on the 2×
/// precision win. The config flags (`custom_2q` / `tiled_kq` / `tiled_min_k` /
/// `layer_batch` / `custom_diag`) match the FP64 backend so the A/B benches are
/// precision-symmetric.
pub struct CudaSvBackendF32 {
    ctx: CudaContext,
    f_1q: CudaFunction,
    f_1q_multi: CudaFunction,
    f_cnot: CudaFunction,
    f_phase_poly: CudaFunction,
    f_kq: CudaFunction,
    f_kq_tiled: CudaFunction,
    /// TF32 tensor-core fused-block kernels (P5.11-05). `None` if the separate
    /// `sm_89` + `mma.h` NVRTC module failed to compile/load (e.g. a non-Ada card
    /// or no CUDA include dir) — the backend stays fully usable, k=4/k=5 dense
    /// blocks just fall back to the warp-tiled FP32 path. So a TF32-unsupported
    /// host loses the lever, not the whole backend.
    tf32: Option<Tf32Kernels>,
    f_diag_1q: CudaFunction,
    f_diag: CudaFunction,
    _module: Arc<CudaModule>,
    rng: StdRng,
    qubit_cap: u32,
    /// GPU-resident readout (P5.11-04): measure / sample / expectation /
    /// probabilities reduce on the device so only small results cross PCIe.
    readout: GpuReadoutF32,
    /// Route diagonal gates to the custom `apply_diag_f32` kernels (default).
    custom_diag: bool,
    /// Disjoint-1q-layer batch width for [`Self::run_layered`], `1..=MAX_LAYER_BATCH`.
    layer_batch: usize,
    /// Route plain CNOTs to the `apply_cnot_f32` permutation kernel (default).
    custom_2q: bool,
    /// Route dense `k`-qubit blocks with `k >= tiled_min_k` to the warp-tiled
    /// `apply_kq_tiled_f32` kernel (default).
    tiled_kq: bool,
    /// Smallest `k` routed to `apply_kq_tiled_f32` (default 2; clamped `2..=6`).
    tiled_min_k: u32,
    /// Route dense k=4/k=5 blocks (no external controls) to the TF32 tensor-core
    /// `apply_kq_tf32_k{4,5}` kernels (P5.11-05, default on). Takes precedence over
    /// `tiled_kq` for those widths; k≤3 stays on the warp-tiled kernel.
    tf32_kq: bool,
}

impl CudaSvBackendF32 {
    /// Construct on device 0 with an entropy-seeded RNG, compiling the FP32 kernels
    /// via NVRTC. Returns [`Error::NoDevice`] on a GPU-less host so callers can
    /// skip cleanly.
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
        let ptx = compile_ptx(SV_F32_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        // Separate module: TF32 WMMA needs sm_89 + the CUDA mma.h include path.
        // Non-fatal: a host that can't build it keeps a fully working FP32 backend
        // (k=4/k=5 dense blocks fall back to the warp-tiled kernel).
        let tf32 = build_tf32_kernels(&ctx);
        let readout = GpuReadoutF32::new(&ctx)?;
        Ok(Self {
            f_1q: module.load_function(APPLY_1Q_F32)?,
            f_1q_multi: module.load_function(APPLY_1Q_MULTI_F32)?,
            f_cnot: module.load_function(APPLY_CNOT_F32)?,
            f_phase_poly: module.load_function(APPLY_PHASE_POLY_F32)?,
            f_kq: module.load_function(APPLY_KQ_F32)?,
            f_kq_tiled: module.load_function(APPLY_KQ_TILED_F32)?,
            tf32,
            f_diag_1q: module.load_function(APPLY_DIAG_1Q_F32)?,
            f_diag: module.load_function(APPLY_DIAG_F32)?,
            _module: module,
            ctx,
            rng,
            qubit_cap: MAX_CUDA_QUBITS_F32,
            readout,
            custom_diag: true,
            layer_batch: DEFAULT_LAYER_BATCH,
            custom_2q: true,
            tiled_kq: true,
            tiled_min_k: 2,
            tf32_kq: true,
        })
    }

    /// Override the qubit cap (default [`MAX_CUDA_QUBITS_F32`]).
    pub fn with_qubit_cap(mut self, cap: u32) -> Self {
        self.qubit_cap = cap;
        self
    }

    /// Enable (default) or disable routing plain CNOTs to `apply_cnot_f32`.
    pub fn with_custom_2q(mut self, on: bool) -> Self {
        self.custom_2q = on;
        self
    }

    /// Enable (default) or disable routing dense `k`-qubit blocks
    /// (`k >= tiled_min_k`) to `apply_kq_tiled_f32`.
    pub fn with_tiled_kq(mut self, on: bool) -> Self {
        self.tiled_kq = on;
        self
    }

    /// Override the smallest `k` routed to `apply_kq_tiled_f32` (clamped `2..=6`).
    pub fn with_tiled_min_k(mut self, k: u32) -> Self {
        self.tiled_min_k = k.clamp(2, 6);
        self
    }

    /// Enable (default) or disable routing dense k=4/k=5 blocks (no external
    /// controls) to the TF32 tensor-core kernels (P5.11-05). When off, those widths
    /// fall back to the warp-tiled / generic FP32 ALU path.
    pub fn with_tf32_kq(mut self, on: bool) -> Self {
        self.tf32_kq = on;
        self
    }

    /// Override the disjoint-1q-layer batch width for [`Self::run_layered`]
    /// (clamped `1..=MAX_LAYER_BATCH`); 1 reproduces per-gate dispatch.
    pub fn with_layer_batch(mut self, batch: usize) -> Self {
        self.layer_batch = batch.clamp(1, MAX_LAYER_BATCH);
        self
    }

    /// Enable (default) or disable routing diagonal gates to `apply_diag_f32`.
    pub fn with_custom_kernels(mut self, on: bool) -> Self {
        self.custom_diag = on;
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
        // Plain CNOT → permutation kernel (P5.9-04, gated by `custom_2q`).
        if self.custom_2q
            && matches!(gate.gate, Gate::Cnot)
            && gate.controls.is_empty()
            && gate.qubits.len() == 2
        {
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
        // Diagonal fast path (P5-06, gated by `custom_diag`).
        if self.custom_diag {
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
        // P5.11-05: dense k=4/k=5 blocks (no external controls) run the O(4^k)
        // matvec on the TF32 tensor cores as a batched GEMM — the compute the
        // warp-tiled kernel can't escape. Takes precedence over `tiled_kq`. Gated on
        // the module actually loading, so a TF32-unsupported host falls through.
        if self.tf32_kq && params.ctrl_mask == 0 && (params.k == 4 || params.k == 5) {
            if let Some(tf32) = self.tf32.as_ref() {
                return self.launch_kq_tf32(state, tf32, params);
            }
        }
        // P5.11-04: above tiled_min_k the generic `apply_kq_f32` spills its
        // v[32]/gidx[32] thread-local arrays — route those to the warp-cooperative
        // `apply_kq_tiled_f32` (mirrors the FP64 P5.10-01 routing).
        if self.tiled_kq && params.k >= self.tiled_min_k {
            return self.launch_kq_tiled(state, params);
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

    /// Launch the warp-cooperative `apply_kq_tiled_f32` kernel over all `2^n`
    /// amplitudes. The `2^k × 2^k` matrix must already be in `state.mat_scratch`
    /// (uploaded by [`Self::launch_kq`]). Shared memory holds the tile as **float**
    /// complex (`2^k·2^k · 8 B`; 8 KiB at k=5, inside the per-block limit).
    fn launch_kq_tiled(
        &self,
        state: &mut CudaSvStateF32,
        params: GateKqParams,
    ) -> Result<(), Error> {
        let n_amps: u64 = 1 << state.num_qubits;
        let dim: u32 = 1 << params.k;
        let shared_bytes = (dim as usize) * (dim as usize) * std::mem::size_of::<[f32; 2]>();
        let cfg = launch_config_shared(n_amps, shared_bytes as u32);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        let mat_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set by launch_kq")
            .slice();
        // SAFETY: signature (cplxf* amps, const cplx* mat, GateKq g, u64 n_amps);
        // `mat_dev` holds 2^k·2^k f64 cplx, `amps` 2^n cplxf, the grid covers
        // exactly n_amps with an in-bounds guard, and `shared_bytes` matches the
        // kernel's dim·dim cplxf dynamic-shared allocation. BLOCK is a multiple of
        // 32 and dim divides 32 (k≤5), so every 2^k group is warp-local.
        unsafe {
            stream
                .launch_builder(&self.f_kq_tiled)
                .arg(amps)
                .arg(mat_dev)
                .arg(&params)
                .arg(&n_amps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch a TF32 tensor-core fused-block kernel (P5.11-05) for a dense k=4/k=5
    /// block over all `2^(n-k)` groups. The `2^k × 2^k` matrix must already be in
    /// `state.mat_scratch` (uploaded by [`Self::launch_kq`]). Each warp owns one
    /// 16-group WMMA tile; the grid covers `⌈groups / 16⌉` tiles packed
    /// `K{4,5}_WARPS` to a block. Static shared memory holds the cast matrix + tiles
    /// (no dynamic shared bytes).
    fn launch_kq_tf32(
        &self,
        state: &mut CudaSvStateF32,
        tf32: &Tf32Kernels,
        params: GateKqParams,
    ) -> Result<(), Error> {
        let n_groups: u64 = 1 << (state.num_qubits - params.k);
        let tiles = n_groups.div_ceil(TF32_TILE);
        let (func, warps) = if params.k == 4 {
            (&tf32.k4, TF32_K4_WARPS)
        } else {
            (&tf32.k5, TF32_K5_WARPS)
        };
        let blocks = tiles.div_ceil(warps as u64).max(1) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (warps * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        let mat_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set by launch_kq")
            .slice();
        // SAFETY: signature (cplxf* amps, const cplx* mat, GateKq g, u64 n_groups);
        // `mat_dev` holds 2^k·2^k f64 cplx, `amps` 2^n cplxf. The grid covers every
        // 16-group tile (partial tiles zero-padded + guarded in-kernel by gid <
        // n_groups), `warps` matches the kernel's K{4,5}_WARPS, and k ∈ {4,5} so the
        // selected kernel's static shared sizing is valid.
        unsafe {
            stream
                .launch_builder(func)
                .arg(amps)
                .arg(mat_dev)
                .arg(&params)
                .arg(&n_groups)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch `apply_1q_multi_f32` for one batch of `m = params.m` disjoint 1q
    /// gates over `2^(n-m)` groups — one state sweep for the whole batch (P5.9-03).
    fn launch_1q_multi(
        &self,
        state: &mut CudaSvStateF32,
        params: Multi1qParams,
    ) -> Result<(), Error> {
        let n_groups: u64 = 1 << (state.num_qubits - params.m);
        let cfg = launch_config(n_groups);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: signature (cplxf* amps, Multi1q g, u64 n_groups); `amps` holds
        // 2^n cplxf, grid covers n_groups with a guard. `m ≤ 5` (enforced by the
        // `apply_1q_layer` chunking), so `num_qubits - m` never underflows.
        unsafe {
            stream
                .launch_builder(&self.f_1q_multi)
                .arg(amps)
                .arg(&params)
                .arg(&n_groups)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Apply a fused [`DiagonalPhase`] in one coalesced sweep (P5.9-06):
    /// `amps[x] *= exp(i·φ(x))`. Mirrors the FP64 `launch_phase_poly` — CSR-encode
    /// the terms, upload, launch `apply_phase_poly_f32`, then synchronise so the
    /// per-call upload buffers outlive the kernel.
    fn launch_phase_poly(
        &self,
        state: &mut CudaSvStateF32,
        dp: &DiagonalPhase,
    ) -> Result<(), Error> {
        let n_terms = dp.terms.len();
        if n_terms == 0 {
            return Ok(()); // empty polynomial ⇒ identity
        }
        let mut angles: Vec<f64> = Vec::with_capacity(n_terms);
        let mut conds: Vec<u64> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(n_terms + 1);
        offsets.push(0);
        for t in &dp.terms {
            angles.push(t.angle);
            conds.extend(t.conds.iter().copied());
            offsets.push(conds.len() as u32);
        }
        if conds.is_empty() {
            conds.push(0); // keep a valid (unindexed) device pointer
        }

        let angles_dev = DeviceBuffer::from_slice(&self.ctx, &angles)?;
        let conds_dev = DeviceBuffer::from_slice(&self.ctx, &conds)?;
        let offsets_dev = DeviceBuffer::from_slice(&self.ctx, &offsets)?;

        let n_amps: u64 = 1 << state.num_qubits;
        let n_terms_u = n_terms as u32;
        let cfg = launch_config(n_amps);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: signature (cplxf* amps, const double* angles, const ull* conds,
        // const unsigned* offsets, unsigned n_terms, ull n_amps). Args match;
        // `amps` holds 2^n cplxf; `angles`/`offsets` have n_terms/n_terms+1 elements
        // and `conds` is non-empty; the grid covers n_amps with a guard.
        unsafe {
            stream
                .launch_builder(&self.f_phase_poly)
                .arg(amps)
                .arg(angles_dev.slice())
                .arg(conds_dev.slice())
                .arg(offsets_dev.slice())
                .arg(&n_terms_u)
                .arg(&n_amps)
                .launch(cfg)?;
        }
        // Block so the upload buffers (dropped at scope end) outlive the kernel.
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Apply a run of mutually-disjoint single-qubit gates, chunked into batches of
    /// ≤ [`MAX_LAYER_BATCH`] and dispatched one sweep each via `apply_1q_multi_f32`.
    /// `gates` is `(qubit, 2×2 matrix)`; the caller guarantees pairwise-distinct
    /// qubits (so the gates commute and chunking is exact).
    fn apply_1q_layer(
        &self,
        state: &mut CudaSvStateF32,
        gates: &[(u32, [[Complex; 2]; 2])],
    ) -> Result<(), Error> {
        for chunk in gates.chunks(self.layer_batch) {
            let mut idx: Vec<usize> = (0..chunk.len()).collect();
            idx.sort_unstable_by_key(|&i| chunk[i].0);

            let mut params = Multi1qParams {
                mats: [0.0; 40],
                m: chunk.len() as u32,
                sorted: [0u32; 5],
                _pad: [0u32; 2],
            };
            for (j, &i) in idx.iter().enumerate() {
                let (q, m) = chunk[i];
                params.sorted[j] = q;
                let base = j * 8;
                params.mats[base] = m[0][0].re;
                params.mats[base + 1] = m[0][0].im;
                params.mats[base + 2] = m[0][1].re;
                params.mats[base + 3] = m[0][1].im;
                params.mats[base + 4] = m[1][0].re;
                params.mats[base + 5] = m[1][0].im;
                params.mats[base + 6] = m[1][1].re;
                params.mats[base + 7] = m[1][1].im;
            }
            self.launch_1q_multi(state, params)?;
        }
        Ok(())
    }

    /// Run `circuit` with **disjoint-1q-layer batching** (P5.9-03), the FP32 mirror
    /// of [`crate::CudaSvBackend::run_layered`]: consecutive plain 1q gates on
    /// distinct qubits apply in `⌈count / layer_batch⌉` sweeps via
    /// `apply_1q_multi_f32`. Every other instruction flushes the pending batch
    /// (preserving program order). Oracle-equal to per-gate [`Self::run`].
    #[allow(unused_assignments)]
    pub fn run_layered(&mut self, circuit: &Circuit) -> Result<CudaSvStateF32, BackendError> {
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
        let mut pending: Vec<(u32, [[Complex; 2]; 2])> = Vec::new();
        let mut mask: u64 = 0;

        macro_rules! flush {
            () => {
                if !pending.is_empty() {
                    self.apply_1q_layer(&mut state, &pending)
                        .map_err(to_backend_err)?;
                    pending.clear();
                    mask = 0;
                }
            };
        }

        for inst in circuit.instructions() {
            match inst {
                Instruction::Gate(g) => match batchable_1q(g) {
                    Some(m) => {
                        let bit = 1u64 << g.qubits[0];
                        if mask & bit != 0 {
                            flush!();
                        }
                        mask |= bit;
                        pending.push((g.qubits[0], m));
                    }
                    None => {
                        flush!();
                        self.apply_gate(&mut state, g)?;
                    }
                },
                Instruction::Barrier(_) => flush!(),
                Instruction::Measure { qubit, .. } => {
                    flush!();
                    let _ = self.readout.measure(&mut self.rng, &mut state, *qubit)?;
                }
                Instruction::Reset(_) => {
                    flush!();
                    return Err(BackendError::UnsupportedInstruction { kind: "reset" });
                }
                Instruction::DiagonalPhase(dp) => {
                    flush!();
                    self.launch_phase_poly(&mut state, dp)
                        .map_err(to_backend_err)?;
                }
                Instruction::TiledBlock(_) => {
                    flush!();
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "tiled-block",
                    });
                }
            }
        }
        flush!();
        self.ctx.synchronize().map_err(to_backend_err)?;
        Ok(state)
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

/// NVRTC options for the TF32 module: target the Ada tensor cores (`sm_89`) and
/// add the CUDA toolkit include dir so `<mma.h>` resolves. The include root is
/// taken from `CUDA_HOME` / `CUDA_PATH`, defaulting to `/usr/local/cuda` (the
/// Phase-5 box layout). This path is only exercised on a real CUDA host.
fn tf32_compile_opts() -> CompileOptions {
    let root = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    CompileOptions {
        arch: Some("sm_89"),
        include_paths: vec![format!("{root}/include")],
        ..Default::default()
    }
}

/// Compile + load the TF32 tensor-core module, returning `None` (with a one-line
/// warning) on any failure so [`CudaSvBackendF32::build`] never fails just because
/// the host can't build WMMA. Failure modes: a pre-Ada card the `sm_89` PTX won't
/// load on, or a missing CUDA include dir so `<mma.h>` doesn't resolve.
fn build_tf32_kernels(ctx: &CudaContext) -> Option<Tf32Kernels> {
    let ptx = match compile_ptx_with_opts(SV_TF32_SRC, tf32_compile_opts()) {
        Ok(ptx) => ptx,
        Err(e) => {
            eprintln!(
                "aleph-cuda: TF32 fused-block module unavailable ({e}); \
                 k=4/k=5 dense blocks use the warp-tiled FP32 path"
            );
            return None;
        }
    };
    let module = match ctx.raw().load_module(ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("aleph-cuda: TF32 module load failed ({e}); falling back to tiled");
            return None;
        }
    };
    match (
        module.load_function(APPLY_KQ_TF32_K4),
        module.load_function(APPLY_KQ_TF32_K5),
    ) {
        (Ok(k4), Ok(k5)) => Some(Tf32Kernels {
            k4,
            k5,
            _module: module,
        }),
        _ => {
            eprintln!("aleph-cuda: TF32 kernel entry points missing; falling back to tiled");
            None
        }
    }
}

fn launch_config(n_threads: u64) -> LaunchConfig {
    launch_config_shared(n_threads, 0)
}

/// Like [`launch_config`] but with `shared_bytes` of dynamic shared memory per
/// block (for `apply_kq_tiled_f32`'s shared-memory matrix tile).
fn launch_config_shared(n_threads: u64, shared_bytes: u32) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: shared_bytes,
    }
}

/// A plain single-qubit gate the layer batcher can fold into `apply_1q_multi_f32`:
/// exactly one operand, no external controls, and a concrete 2×2 matrix. Mirrors
/// the FP64 `batchable_1q`.
fn batchable_1q(g: &GateInstance) -> Option<[[Complex; 2]; 2]> {
    if g.qubits.len() != 1 || !g.controls.is_empty() {
        return None;
    }
    if matches!(g.gate, Gate::UnitaryKq { .. }) {
        return None;
    }
    match g.gate.matrix() {
        Ok(GateMatrix::M2x2(m)) => Some(m),
        _ => None,
    }
}

impl Backend for CudaSvBackendF32 {
    type State = CudaSvStateF32;

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
        CudaSvStateF32::allocate(&self.ctx, num_qubits).map_err(to_backend_err)
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        // Delegate to the inherent worker (shared with `run` / the paged executor).
        CudaSvBackendF32::apply_gate(self, state, gate)
    }

    /// Apply a fused diagonal phase polynomial (P5.9-06) — without this override the
    /// trait default rejects `DiagonalPhase`, so a `FuseDiagonalRuns`-fused circuit
    /// (QFT/QPE) could not run on the FP32 GPU backend.
    fn apply_diagonal_phase(
        &mut self,
        state: &mut Self::State,
        dp: &DiagonalPhase,
    ) -> Result<(), BackendError> {
        self.launch_phase_poly(state, dp).map_err(to_backend_err)
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
