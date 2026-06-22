//! `CuStateVecBackend` — a [`Backend`] that routes gate application through
//! NVIDIA cuStateVec (`custatevecApplyMatrix`) while reusing the hand-written
//! GPU backend's device-resident state ([`CudaSvState`]) and host-side readout
//! ([`crate::sv::readout`]). The state-vector buffer layout is identical
//! (interleaved `[re, im]` FP64, qubit `q` ↦ index bit `q`), so the two backends
//! are drop-in interchangeable and the oracle suite pins both.

use std::ptr;

use aleph_backend::{Backend, BackendError};
use aleph_core::{GateInstance, GateMatrix, PauliString};
use cudarc::driver::DevicePtrMut;
use rand::{rngs::StdRng, SeedableRng};

use crate::common::{
    control_mask, diagonal_of, flatten_kq, flatten_matrix, validate_and_extract, validate_kq,
};
use crate::cuquantum::sys;
use crate::sv::diag::{diag_1q_params, diag_kq_params, DiagKernels};
use crate::sv::readout::GpuReadout;
use crate::sv::{CudaSvState, MAX_CUDA_QUBITS};
use crate::{CudaContext, DeviceBuffer, Error};

/// GPU state-vector backend backed by NVIDIA cuStateVec (FP64).
///
/// Holds the long-lived `custatevecHandle_t` and a reusable device workspace
/// buffer; gates are dispatched on the same `cudarc` stream the state lives on.
pub struct CuStateVecBackend {
    ctx: CudaContext,
    handle: sys::CustatevecHandle,
    rng: StdRng,
    qubit_cap: u32,
    /// Reusable extra device scratch for `custatevecApplyMatrix`, grown on
    /// demand. Empty (`None`) for the small `nTargets ≤ 3` gates, which need 0
    /// bytes; kept generic so larger fused gates would still work.
    workspace: Option<DeviceBuffer<u8>>,
    /// GPU-resident readout (P5-05), shared with the hand-written backend — the
    /// state buffer layout is identical, so the same reduction kernels apply.
    readout: GpuReadout,
    /// Custom diagonal-gate kernels (P5-06). cuStateVec's generic
    /// `custatevecApplyMatrix` overpays for diagonal gates; this is the
    /// "niche cuQuantum misses" the issue targets.
    diag: DiagKernels,
    /// When set (default), diagonal gates divert to [`Self::diag`] instead of
    /// cuStateVec. Cleared (`with_custom_kernels(false)`) routes every gate
    /// through cuStateVec — the pure-cuQuantum baseline for the P5-06 A/B.
    custom_diag: bool,
}

impl CuStateVecBackend {
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
        let mut handle: sys::CustatevecHandle = ptr::null_mut();
        // SAFETY: `handle` is a valid out-pointer; cuStateVec writes the new
        // handle into it and returns a status we check.
        check(unsafe { sys::custatevecCreate(&mut handle) })?;
        // Bind cuStateVec to cudarc's stream so its async work orders correctly
        // with our allocations and device→host readback copies.
        let stream = ctx.stream().cu_stream() as sys::CudaStream;
        // SAFETY: `handle` was just created; `stream` is a live CUstream owned by
        // `ctx` for the backend's lifetime.
        if let Err(e) = check(unsafe { sys::custatevecSetStream(handle, stream) }) {
            // Don't leak the handle if stream binding fails.
            unsafe { sys::custatevecDestroy(handle) };
            return Err(e);
        }
        let diag = DiagKernels::new(&ctx)?;
        let readout = GpuReadout::new(&ctx)?;
        Ok(Self {
            ctx,
            handle,
            rng,
            qubit_cap: MAX_CUDA_QUBITS,
            workspace: None,
            readout,
            diag,
            custom_diag: true,
        })
    }

    /// Override the qubit cap (default [`MAX_CUDA_QUBITS`]).
    pub fn with_qubit_cap(mut self, cap: u32) -> Self {
        self.qubit_cap = cap;
        self
    }

    /// Enable (default) or disable diverting diagonal gates to the custom
    /// `apply_diag` kernels (P5-06). Disabling routes every gate through
    /// cuStateVec — the pure-cuQuantum baseline arm of the P5-06 A/B benchmark.
    pub fn with_custom_kernels(mut self, on: bool) -> Self {
        self.custom_diag = on;
        self
    }

    /// Apply a diagonal gate via the custom `apply_diag` kernels instead of
    /// `custatevecApplyMatrix` (P5-06). `diag` is the interleaved `[re, im]`
    /// diagonal from [`diagonal_of`]; `qubits` are the operands MSB-first;
    /// `controls` are external controls.
    fn apply_diag(
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

    /// Apply a validated dense matrix to `state` via `custatevecApplyMatrix`.
    ///
    /// `mat` is the row-major interleaved `[re, im]` buffer (host pointer — the
    /// API accepts host or device matrices). `targets` are the operand qubits in
    /// **little-endian** order (LSB of the matrix index first); since
    /// `gate.matrix()` lays operands out MSB-first, the caller passes the operand
    /// list reversed. `controls` are external control qubits (all fire on `1`).
    fn apply_matrix(
        &mut self,
        state: &mut CudaSvState,
        mat: &[f64],
        targets: &[i32],
        controls: &[i32],
    ) -> Result<(), Error> {
        let n_index_bits = state.num_qubits;
        let n_targets = targets.len() as u32;
        let n_controls = controls.len() as u32;

        // How much device scratch this apply needs (≈0 for nTargets ≤ 3).
        let mut ws_bytes: usize = 0;
        // SAFETY: all pointers are valid host pointers for the call's duration;
        // `mat` covers 2^nTargets · 2^nTargets complex doubles; status checked.
        check(unsafe {
            sys::custatevecApplyMatrixGetWorkspaceSize(
                self.handle,
                sys::CUDA_C_64F,
                n_index_bits,
                mat.as_ptr() as *const _,
                sys::CUDA_C_64F,
                sys::CUSTATEVEC_MATRIX_LAYOUT_ROW,
                0, // adjoint
                n_targets,
                n_controls,
                sys::CUSTATEVEC_COMPUTE_64F,
                &mut ws_bytes,
            )
        })?;

        // Clone the stream Arc so the `&mut self.workspace` and `&mut state.amps`
        // borrows below don't alias the `&self.ctx` borrow.
        let stream = self.ctx.stream().clone();

        // Ensure a workspace buffer large enough (only when the API asks for it).
        if ws_bytes > 0 {
            let need_alloc = self.workspace.as_ref().is_none_or(|b| b.len() < ws_bytes);
            if need_alloc {
                self.workspace = Some(DeviceBuffer::<u8>::zeros(&self.ctx, ws_bytes)?);
            }
        }
        let (ws_ptr, _ws_sync): (*mut core::ffi::c_void, _) = match self.workspace.as_mut() {
            Some(buf) if ws_bytes > 0 => {
                let (p, guard) = buf.slice_mut().device_ptr_mut(&stream);
                (p as usize as *mut _, Some(guard))
            }
            _ => (ptr::null_mut(), None),
        };

        // Device pointer to the state vector. The `SyncOnDrop` guard is held
        // until after the (async) apply is scheduled on the stream.
        let (sv_dptr, _sv_sync) = state.amps.slice_mut().device_ptr_mut(&stream);
        let sv_ptr = sv_dptr as usize as *mut core::ffi::c_void;

        // SAFETY: `sv_ptr` is a live device allocation of 2^n complex doubles in
        // the primary context cuStateVec shares; `mat`/`targets`/`controls` are
        // valid host arrays for the call; `ws_ptr` covers `ws_bytes`. Argument
        // order/types match the transcribed `custatevecApplyMatrix` prototype.
        check(unsafe {
            sys::custatevecApplyMatrix(
                self.handle,
                sv_ptr,
                sys::CUDA_C_64F,
                n_index_bits,
                mat.as_ptr() as *const _,
                sys::CUDA_C_64F,
                sys::CUSTATEVEC_MATRIX_LAYOUT_ROW,
                0, // adjoint
                targets.as_ptr(),
                n_targets,
                controls.as_ptr(),
                ptr::null(), // controlBitValues = null ⇒ all fire on 1
                n_controls,
                sys::CUSTATEVEC_COMPUTE_64F,
                ws_ptr,
                ws_bytes,
            )
        })
    }
}

/// Map a non-success cuStateVec status to [`Error::CuStateVec`].
fn check(status: sys::CustatevecStatus) -> Result<(), Error> {
    if status == sys::CUSTATEVEC_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::CuStateVec(status))
    }
}

/// Map a CUDA-layer error to a backend error (shared shape with the SV backend).
fn to_backend_err(_e: Error) -> BackendError {
    BackendError::InvalidState {
        reason: "cuStateVec backend failure (create/apply/transfer)",
    }
}

impl Drop for CuStateVecBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `handle` was created by `custatevecCreate` and not yet
            // destroyed; destroying releases the library context.
            unsafe { sys::custatevecDestroy(self.handle) };
        }
    }
}

impl Backend for CuStateVecBackend {
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
        // P5.9-02: fused `UnitaryKq` (k=4,5) has no fixed-size `GateMatrix`;
        // `custatevecApplyMatrix` already takes an arbitrary `2^k × 2^k` row-major
        // matrix, so feed it the raw slice. Operands are reversed to little-endian
        // like every other matrix apply (gate.matrix() / UnitaryKq are MSB-first).
        if let aleph_core::Gate::UnitaryKq { k, data } = &gate.gate {
            validate_kq(
                state.num_qubits,
                *k,
                data.len(),
                &gate.qubits,
                &gate.controls,
            )?;
            let targets: Vec<i32> = gate.qubits.iter().rev().map(|&q| q as i32).collect();
            let controls: Vec<i32> = gate.controls.iter().map(|&c| c as i32).collect();
            return self
                .apply_matrix(state, &flatten_kq(data), &targets, &controls)
                .map_err(to_backend_err);
        }
        let matrix = validate_and_extract(state.num_qubits, gate)?;
        // P5-06: divert diagonal gates to the custom kernel — the niche where a
        // bespoke phase-multiply beats cuStateVec's generic dense apply.
        if self.custom_diag {
            if let Some(diag) = diagonal_of(&matrix) {
                return self
                    .apply_diag(state, &diag, &gate.qubits, &gate.controls)
                    .map_err(to_backend_err);
            }
        }
        // cuStateVec target ordering is little-endian (targets[0] = LSB of the
        // matrix index), but `gate.matrix()` lays operands out MSB-first
        // (qubits[0] = MSB). Reversing the operand list reconciles the two so the
        // same row-major matrix acts on the same physical qubits.
        let targets: Vec<i32> = gate.qubits.iter().rev().map(|&q| q as i32).collect();
        let controls: Vec<i32> = gate.controls.iter().map(|&c| c as i32).collect();
        let flat = match &matrix {
            GateMatrix::M2x2(m) => flatten_matrix(m),
            GateMatrix::M4x4(m) => flatten_matrix(m),
            GateMatrix::M8x8(m) => flatten_matrix(m),
        };
        self.apply_matrix(state, &flat, &targets, &controls)
            .map_err(to_backend_err)
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
