//! Minimal hand-written FFI for NVIDIA cuStateVec (part of cuQuantum).
//!
//! We bind only the five entry points the backend needs — create / destroy /
//! set-stream / apply-matrix (+ its workspace query) — rather than pull in a
//! `bindgen` build dependency for a four-function surface. The signatures,
//! enum values, and `cudaDataType_t` constants are transcribed from
//! `custatevec.h` (cuStateVec 1.13) and `library_types.h` (CUDA 13.0); see the
//! per-item comments for the source.
//!
//! Linking is handled by `build.rs`, which emits `-lcustatevec` only when the
//! `cuquantum` feature is on. cuStateVec shares the CUDA **primary context** and
//! the same virtual address space as `cudarc` (which retains that primary
//! context via the driver API), so a device pointer minted by `cudarc` is a
//! valid `sv` argument here — the standard driver/runtime interop contract.

use core::ffi::c_void;

/// Opaque `custatevecHandle_t` (a `custatevecContext*`).
pub type CustatevecHandle = *mut c_void;

/// `cudaStream_t` — an opaque stream handle. `cudarc`'s `CUstream` is the same
/// underlying pointer, cast through `*mut c_void`.
pub type CudaStream = *mut c_void;

/// `custatevecStatus_t` return code; `0` is success.
pub type CustatevecStatus = i32;

/// `cudaDataType_t`.
pub type CudaDataType = i32;

/// `custatevecMatrixLayout_t`.
pub type CustatevecMatrixLayout = i32;

/// `custatevecComputeType_t`.
pub type CustatevecComputeType = i32;

/// `CUSTATEVEC_STATUS_SUCCESS`.
pub const CUSTATEVEC_STATUS_SUCCESS: CustatevecStatus = 0;

/// `CUDA_C_64F` — complex as a pair of `double` (`library_types.h`). Our
/// amplitude buffer is interleaved `[re, im]` f64, exactly this layout.
pub const CUDA_C_64F: CudaDataType = 5;

/// `CUSTATEVEC_MATRIX_LAYOUT_ROW` — gate matrices from `gate.matrix()` are
/// row-major.
pub const CUSTATEVEC_MATRIX_LAYOUT_ROW: CustatevecMatrixLayout = 1;

/// `CUSTATEVEC_COMPUTE_64F = 1 << 4` — full FP64 matrix multiply.
pub const CUSTATEVEC_COMPUTE_64F: CustatevecComputeType = 1 << 4;

extern "C" {
    /// `custatevecCreate(custatevecHandle_t* handle)`.
    pub fn custatevecCreate(handle: *mut CustatevecHandle) -> CustatevecStatus;

    /// `custatevecDestroy(custatevecHandle_t handle)`.
    pub fn custatevecDestroy(handle: CustatevecHandle) -> CustatevecStatus;

    /// `custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId)` —
    /// binds cuStateVec work to `cudarc`'s stream so device alloc / H2D copies /
    /// gate applies stay correctly ordered.
    pub fn custatevecSetStream(handle: CustatevecHandle, stream: CudaStream) -> CustatevecStatus;

    /// `custatevecApplyMatrixGetWorkspaceSize(...)` — bytes of extra device
    /// scratch `custatevecApplyMatrix` needs for these parameters (≈0 for the
    /// small `nTargets ≤ 3` gates we apply).
    #[allow(clippy::too_many_arguments)]
    pub fn custatevecApplyMatrixGetWorkspaceSize(
        handle: CustatevecHandle,
        sv_data_type: CudaDataType,
        n_index_bits: u32,
        matrix: *const c_void,
        matrix_data_type: CudaDataType,
        layout: CustatevecMatrixLayout,
        adjoint: i32,
        n_targets: u32,
        n_controls: u32,
        compute_type: CustatevecComputeType,
        extra_workspace_size_in_bytes: *mut usize,
    ) -> CustatevecStatus;

    /// `custatevecApplyMatrix(...)` — apply a dense gate matrix to the device
    /// state vector `sv` in place. `targets`/`controls` are host arrays of bit
    /// positions (little-endian: `targets[0]` is the LSB of the matrix index);
    /// `control_bit_values = null` means all controls fire on `1`.
    #[allow(clippy::too_many_arguments)]
    pub fn custatevecApplyMatrix(
        handle: CustatevecHandle,
        sv: *mut c_void,
        sv_data_type: CudaDataType,
        n_index_bits: u32,
        matrix: *const c_void,
        matrix_data_type: CudaDataType,
        layout: CustatevecMatrixLayout,
        adjoint: i32,
        targets: *const i32,
        n_targets: u32,
        controls: *const i32,
        control_bit_values: *const i32,
        n_controls: u32,
        compute_type: CustatevecComputeType,
        extra_workspace: *mut c_void,
        extra_workspace_size_in_bytes: usize,
    ) -> CustatevecStatus;
}
