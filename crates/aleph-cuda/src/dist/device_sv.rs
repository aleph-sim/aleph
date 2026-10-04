//! `DeviceSv` for the FP64 and FP32 CUDA state-vector backends. Rank slices
//! are ordinary `CudaSvState` / `CudaSvStateF32` with `num_qubits = m`, so the
//! existing kernels apply unchanged (the same trick `run_paged` uses).

use aleph_backend::BackendError;
use aleph_core::Complex;

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

impl DeviceSv for CudaSvBackend {
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
        let (s0, d0, l) = (2 * src_off, 2 * dst_off, 2 * len);
        if s0 + l > src.amps.len() || d0 + l > dst.amps.len() {
            return Err(range_err());
        }
        let view = src.amps.slice().slice(s0..s0 + l);
        let mut out = dst.amps.slice_mut().slice_mut(d0..d0 + l);
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
        let (s0, d0, l) = (2 * src_off, 2 * dst_off, 2 * len);
        if s0 + l > src.amps.len() || d0 + l > dst.amps.len() {
            return Err(range_err());
        }
        let view = src.amps.slice().slice(s0..s0 + l);
        let mut out = dst.amps.slice_mut().slice_mut(d0..d0 + l);
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
