//! `DeviceSv` for the FP64 and FP32 CUDA state-vector backends. Rank slices
//! are ordinary `CudaSvState` / `CudaSvStateF32` with `num_qubits = m`, so the
//! existing kernels apply unchanged (the same trick `run_paged` uses).

use std::sync::Arc;

use aleph_backend::BackendError;
use aleph_core::Complex;
use cudarc::driver::{CudaStream, CudaView, CudaViewMut};

use super::DeviceSv;
use crate::sv::to_backend_err;
use crate::{CudaSvBackend, CudaSvBackendF32, CudaSvState, CudaSvStateF32, DeviceBuffer};

/// `2usize << m` (scalar count of an m-qubit slice) must not overflow.
const MAX_SHIFT_QUBITS: u32 = usize::BITS - 2;

/// Reject `m` above the backend cap (and above what `usize` can address)
/// before any `1 << m` is formed.
fn check_qubits(m: u32, cap: u32) -> Result<(), BackendError> {
    let limit = cap.min(MAX_SHIFT_QUBITS);
    if m > limit {
        return Err(BackendError::TooManyQubits {
            requested: m,
            limit,
        });
    }
    Ok(())
}

fn range_err() -> BackendError {
    BackendError::InvalidState {
        reason: "dist: amplitude copy out of range",
    }
}

/// Scalar range `[2*off, 2*(off+len))` if it lies within `scalars` (overflow-safe).
fn scalar_range(
    off: usize,
    len: usize,
    scalars: usize,
) -> Result<std::ops::Range<usize>, BackendError> {
    let s0 = off.checked_mul(2).ok_or_else(range_err)?;
    let l = len.checked_mul(2).ok_or_else(range_err)?;
    match s0.checked_add(l) {
        Some(e) if e <= scalars => Ok(s0..e),
        _ => Err(range_err()),
    }
}

impl DeviceSv for CudaSvBackend {
    type Scalar = f64;

    fn ordinal(&self) -> usize {
        self.ctx().raw().ordinal()
    }

    fn stream(&self) -> Arc<CudaStream> {
        self.ctx().stream().clone()
    }

    fn amps_view(
        st: &CudaSvState,
        off: usize,
        len: usize,
    ) -> Result<CudaView<'_, f64>, BackendError> {
        let r = scalar_range(off, len, st.amps.len())?;
        Ok(st.amps.slice().slice(r))
    }

    fn amps_view_mut(
        st: &mut CudaSvState,
        off: usize,
        len: usize,
    ) -> Result<CudaViewMut<'_, f64>, BackendError> {
        let r = scalar_range(off, len, st.amps.len())?;
        Ok(st.amps.slice_mut().slice_mut(r))
    }

    fn branch_norms(&mut self, st: &CudaSvState, qbit: u64) -> Result<(f64, f64), BackendError> {
        self.raw_branch(st, qbit)
    }
    fn max_qubits(&self) -> u32 {
        self.qubit_cap()
    }

    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<CudaSvState, BackendError> {
        check_qubits(m, self.qubit_cap())?;
        let ctx = self.ctx();
        if rank == 0 {
            return CudaSvState::allocate(&ctx, m).map_err(to_backend_err);
        }
        let amps = DeviceBuffer::<f64>::zeros(&ctx, 2usize << m).map_err(to_backend_err)?;
        Ok(CudaSvState {
            num_qubits: m,
            amps,
            ctx,
            mat_scratch: None,
        })
    }

    fn copy_amps(
        &mut self,
        src: &CudaSvState,
        src_off: usize,
        dst: &mut CudaSvState,
        dst_off: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        let view = Self::amps_view(src, src_off, len)?;
        let mut out = Self::amps_view_mut(dst, dst_off, len)?;
        self.ctx()
            .stream()
            .memcpy_dtod(&view, &mut out)
            .map_err(|e| to_backend_err(e.into()))
    }

    fn download(&mut self, st: &CudaSvState) -> Result<Vec<Complex<f64>>, BackendError> {
        let host = st.amps.to_vec(&st.ctx).map_err(to_backend_err)?;
        Ok(host
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[re, im]| Complex::new(re, im))
            .collect())
    }

    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<CudaSvState, BackendError> {
        check_qubits(m, self.qubit_cap())?;
        if amps.len() != 1usize << m {
            return Err(range_err());
        }
        let ctx = self.ctx();
        let flat: Vec<f64> = amps.iter().flat_map(|a| [a.re, a.im]).collect();
        let buf = DeviceBuffer::<f64>::from_slice(&ctx, &flat).map_err(to_backend_err)?;
        Ok(CudaSvState {
            num_qubits: m,
            amps: buf,
            ctx,
            mat_scratch: None,
        })
    }
}

impl DeviceSv for CudaSvBackendF32 {
    type Scalar = f32;

    fn ordinal(&self) -> usize {
        self.ctx().raw().ordinal()
    }

    fn stream(&self) -> Arc<CudaStream> {
        self.ctx().stream().clone()
    }

    fn amps_view(
        st: &CudaSvStateF32,
        off: usize,
        len: usize,
    ) -> Result<CudaView<'_, f32>, BackendError> {
        let r = scalar_range(off, len, st.amps.len())?;
        Ok(st.amps.slice().slice(r))
    }

    fn amps_view_mut(
        st: &mut CudaSvStateF32,
        off: usize,
        len: usize,
    ) -> Result<CudaViewMut<'_, f32>, BackendError> {
        let r = scalar_range(off, len, st.amps.len())?;
        Ok(st.amps.slice_mut().slice_mut(r))
    }

    fn branch_norms(&mut self, st: &CudaSvStateF32, qbit: u64) -> Result<(f64, f64), BackendError> {
        self.raw_branch(st, qbit)
    }
    fn max_qubits(&self) -> u32 {
        self.qubit_cap()
    }

    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<CudaSvStateF32, BackendError> {
        check_qubits(m, self.qubit_cap())?;
        let ctx = self.ctx();
        if rank == 0 {
            return CudaSvStateF32::allocate(&ctx, m).map_err(to_backend_err);
        }
        let amps = DeviceBuffer::<f32>::zeros(&ctx, 2usize << m).map_err(to_backend_err)?;
        Ok(CudaSvStateF32 {
            num_qubits: m,
            amps,
            ctx,
            mat_scratch: None,
        })
    }

    fn copy_amps(
        &mut self,
        src: &CudaSvStateF32,
        src_off: usize,
        dst: &mut CudaSvStateF32,
        dst_off: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        let view = Self::amps_view(src, src_off, len)?;
        let mut out = Self::amps_view_mut(dst, dst_off, len)?;
        self.ctx()
            .stream()
            .memcpy_dtod(&view, &mut out)
            .map_err(|e| to_backend_err(e.into()))
    }

    fn download(&mut self, st: &CudaSvStateF32) -> Result<Vec<Complex<f64>>, BackendError> {
        let host = st.amps.to_vec(&st.ctx).map_err(to_backend_err)?;
        Ok(host
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[re, im]| Complex::new(f64::from(re), f64::from(im)))
            .collect())
    }

    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<CudaSvStateF32, BackendError> {
        check_qubits(m, self.qubit_cap())?;
        if amps.len() != 1usize << m {
            return Err(range_err());
        }
        let ctx = self.ctx();
        let flat: Vec<f32> = amps
            .iter()
            .flat_map(|a| [a.re as f32, a.im as f32])
            .collect();
        let buf = DeviceBuffer::<f32>::from_slice(&ctx, &flat).map_err(to_backend_err)?;
        Ok(CudaSvStateF32 {
            num_qubits: m,
            amps: buf,
            ctx,
            mat_scratch: None,
        })
    }
}
