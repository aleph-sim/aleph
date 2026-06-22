//! Custom diagonal-gate kernels (P5-06) and their loader.
//!
//! A diagonal gate multiplies each amplitude by one phase in a single coalesced,
//! in-place pass — no partner gather, no `2^k` block matvec. That is strictly
//! less work than the dense `apply_kq` kernel and than routing the gate through
//! cuStateVec's generic `custatevecApplyMatrix`, so both GPU backends embed a
//! [`DiagKernels`] and divert diagonal gates here. The win shows up on
//! diagonal-dominated circuits: QFT (controlled-Phase), QAOA / Trotter (Rz, ZZ),
//! phase oracles and Grover diffusion (multi-controlled Z).
//!
//! Operand/control conventions are identical to `kernels.cu` (ADR 0004), so the
//! same oracle suite pins this path.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::sv::state::CudaSvState;
use crate::{CudaContext, DeviceBuffer, Error};

/// Threads per block — same memory-bound sweet spot as the dense kernels.
const BLOCK: u32 = 256;

const DIAG_KERNELS_SRC: &str = include_str!("diag.cu");
const APPLY_DIAG_1Q: &str = "apply_diag_1q";
const APPLY_DIAG: &str = "apply_diag";

/// Per-gate uniform for `apply_diag_1q`. Matches the CUDA `Diag1q` struct: two
/// `cplx` (`d0`, `d1`) stored as interleaved `[re, im]` f64, then two `u32`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Diag1qParams {
    pub d0: [f64; 2],
    pub d1: [f64; 2],
    pub t_bit: u32,
    pub ctrl_mask: u32,
}

// SAFETY: `#[repr(C)]`, POD scalar fields only, no padding (2×16 + 2×4 = 40,
// 8-byte aligned), every bit pattern valid — cudarc's `DeviceRepr` contract.
unsafe impl DeviceRepr for Diag1qParams {}

// 2×cplx (32) + 2×u32 (8) = 40, matching CUDA's `Diag1q`.
const _: () = assert!(core::mem::size_of::<Diag1qParams>() == 40);

/// Per-gate uniform for `apply_diag`. Matches the CUDA `DiagK` struct: 7×`u32`,
/// 28 bytes, no padding. `qbit[j]` is the global state-index bit for matrix-index
/// bit `j` (MSB-first operand order, same mapping as `apply_kq`); slots `j >= k`
/// are zero and never read.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DiagKqParams {
    pub k: u32,
    pub qbit: [u32; 5],
    pub ctrl_mask: u32,
}

// SAFETY: same contract as `Diag1qParams` — `#[repr(C)]`, all-`u32` POD, no
// padding (7×4 = 28), every bit pattern valid.
unsafe impl DeviceRepr for DiagKqParams {}

// 1 + 5 + 1 = 7 u32 = 28 bytes, matching CUDA's `DiagK`.
const _: () = assert!(core::mem::size_of::<DiagKqParams>() == 28);

/// Compiled diagonal-gate kernels, shared by both GPU state-vector backends.
///
/// Owns its own NVRTC module (a few-ms one-time compile at backend construction);
/// the dense `CudaSvBackend` and the cuStateVec `CuStateVecBackend` each hold one
/// and dispatch diagonal gates through it.
pub(crate) struct DiagKernels {
    f_diag_1q: CudaFunction,
    f_diag: CudaFunction,
    // Keeps the loaded module alive for the lifetime of the functions.
    _module: Arc<CudaModule>,
}

impl DiagKernels {
    /// Compile `diag.cu` and load both entry points.
    pub(crate) fn new(ctx: &CudaContext) -> Result<Self, Error> {
        let ptx = compile_ptx(DIAG_KERNELS_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        let f_diag_1q = module.load_function(APPLY_DIAG_1Q)?;
        let f_diag = module.load_function(APPLY_DIAG)?;
        Ok(Self {
            f_diag_1q,
            f_diag,
            _module: module,
        })
    }

    /// Launch `apply_diag_1q` over all `2^n` amplitudes.
    pub(crate) fn launch_1q(
        &self,
        ctx: &CudaContext,
        state: &mut CudaSvState,
        params: Diag1qParams,
    ) -> Result<(), Error> {
        let n_amps: u64 = 1 << state.num_qubits;
        let cfg = launch_config(n_amps);
        let stream = ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: kernel signature is (cplx* amps, Diag1q g, u64 n_amps); args
        // match in order/type, `amps` holds 2^n cplx, grid covers `n_amps` with
        // an in-bounds guard, and the op is an in-place per-element multiply.
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

    /// Upload the `2^k`-entry diagonal (`diag`, interleaved `[re, im]`) to the
    /// reusable scratch and launch `apply_diag` over all `2^n` amplitudes.
    pub(crate) fn launch_kq(
        &self,
        ctx: &CudaContext,
        state: &mut CudaSvState,
        params: DiagKqParams,
        diag: &[f64],
    ) -> Result<(), Error> {
        match state.mat_scratch.as_mut() {
            Some(buf) => buf.write(ctx, diag)?,
            None => state.mat_scratch = Some(DeviceBuffer::<f64>::from_slice(ctx, diag)?),
        }
        let n_amps: u64 = 1 << state.num_qubits;
        let cfg = launch_config(n_amps);
        let stream = ctx.stream();
        // Disjoint field borrows: `amps` (&mut) and `mat_scratch` (&).
        let amps = state.amps.slice_mut();
        let diag_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set above")
            .slice();
        // SAFETY: kernel signature is (cplx* amps, const cplx* diag, DiagK g,
        // u64 n_amps); args match in order/type. `diag_dev` holds 2^k cplx,
        // `amps` holds 2^n cplx, grid covers `n_amps` with a guard.
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

/// `((n + BLOCK - 1) / BLOCK)` blocks of `BLOCK` threads, ≥1 block.
fn launch_config(n_threads: u64) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Build a [`Diag1qParams`] from a diagonal 2×2 matrix's diagonal entries
/// (`diag` = `[m00.re, m00.im, m11.re, m11.im]`), the target qubit and external
/// controls.
pub(crate) fn diag_1q_params(diag: &[f64], target: u32, ctrl_mask: u32) -> Diag1qParams {
    Diag1qParams {
        d0: [diag[0], diag[1]],
        d1: [diag[2], diag[3]],
        t_bit: 1u32 << target,
        ctrl_mask,
    }
}

/// Build a [`DiagKqParams`] for a `k`-qubit diagonal gate. `qubits` are the
/// operands in `gate.matrix()` MSB-first order, so matrix-index bit `b` maps to
/// `qubits[k-1-b]` — identical to the dense `gate_kq_params` mapping.
pub(crate) fn diag_kq_params(qubits: &[u32], ctrl_mask: u32) -> DiagKqParams {
    let k = qubits.len();
    let mut qbit = [0u32; 5];
    for (b, slot) in qbit.iter_mut().take(k).enumerate() {
        *slot = 1u32 << qubits[k - 1 - b];
    }
    DiagKqParams {
        k: k as u32,
        qbit,
        ctrl_mask,
    }
}
