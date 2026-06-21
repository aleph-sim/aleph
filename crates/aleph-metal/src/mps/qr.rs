//! GPU Householder thin-QR (P5.8-04): host dispatch around `mps_qr.metal`, used to
//! move the orthogonality centre on-device (replacing the host f64 SVD in
//! [`super::canonical`]). A single threadgroup factors one `m × n` block into
//! `Q` (`m × size`, orthonormal) and `R` (`size × n`, upper-triangular), `size =
//! min(m, n)`, so `A = Q·R`.

use aleph_core::Complex;
use metal::{ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::kernel::QrMeta;
use crate::{DeviceBuffer, MetalContext};

/// Pooled device buffers for the standalone QR (isolation-test harness): the
/// input/working `a` (`m×n`), the orthogonal `q` (`m×size`), the triangular `r`
/// (`size×n`). Production uses the fused `super::move_gpu` encoders instead.
#[cfg(test)]
pub(crate) struct QrScratch {
    a: DeviceBuffer<Complex<f32>>,
    q: DeviceBuffer<Complex<f32>>,
    r: DeviceBuffer<Complex<f32>>,
}

impl QrScratch {
    pub(crate) fn new(ctx: &MetalContext) -> Self {
        Self {
            a: DeviceBuffer::with_capacity(ctx, 0),
            q: DeviceBuffer::with_capacity(ctx, 0),
            r: DeviceBuffer::with_capacity(ctx, 0),
        }
    }

    /// Q factor (`m × size`, column-major), current after [`gpu_qr`].
    pub(crate) fn q(&self) -> &[Complex<f32>] {
        self.q.as_slice()
    }
    /// R factor (`size × n`, column-major), current after [`gpu_qr`].
    pub(crate) fn r(&self) -> &[Complex<f32>] {
        self.r.as_slice()
    }
}

/// Run the GPU Householder QR on a **column-major** `m × n` block `a_cm`
/// (`a_cm[r + c*m]`) into the pooled `scratch`. After it returns, `scratch.q()` is
/// `Q` (column-major `m × size`) and `scratch.r()` is `R` (column-major `size × n`),
/// `size = min(m, n)`, with `A = Q·R`. Own `commit`/`wait`; drives the canonical
/// centre move (P5.8-04), replacing the host f64 SVD. Isolation-test harness for the
/// QR kernel; production encodes it fused via `super::move_gpu`.
#[cfg(test)]
pub(crate) fn gpu_qr(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    scratch: &mut QrScratch,
    a_cm: &[Complex<f32>],
    m: usize,
    n: usize,
) {
    debug_assert_eq!(a_cm.len(), m * n);
    let size = m.min(n);
    scratch.a.write(ctx, a_cm);
    scratch.q.ensure_capacity(ctx, m * size);
    scratch.r.ensure_capacity(ctx, size * n);
    let meta = QrMeta {
        m: m as u32,
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let threads = pipeline.max_total_threads_per_threadgroup().min(256);
    let cmd = ctx.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(scratch.a.metal_buffer()), 0);
    enc.set_buffer(1, Some(scratch.q.metal_buffer()), 0);
    enc.set_buffer(2, Some(scratch.r.metal_buffer()), 0);
    enc.set_bytes(
        3,
        std::mem::size_of::<QrMeta>() as u64,
        &meta as *const QrMeta as *const c_void,
    );
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

#[cfg(test)]
mod tests {
    use super::super::kernel::{MPS_QR_ENTRY, MPS_QR_SRC};
    use super::*;

    /// Deterministic FP32 complex block, **column-major** `m × n`.
    fn block(m: usize, n: usize) -> Vec<Complex<f32>> {
        let mut v = vec![Complex::<f32>::new(0.0, 0.0); m * n];
        for r in 0..m {
            for c in 0..n {
                v[r + c * m] = Complex::<f32>::new(
                    (((r * 7 + c * 3) % 11) as f32) * 0.37 - 1.1,
                    (((r * 5 + c) % 7) as f32) * 0.23 - 0.6,
                );
            }
        }
        v
    }

    #[test]
    fn householder_qr_reconstructs_and_is_orthonormal() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping GPU QR test: no Metal device");
                return;
            }
        };
        let pipeline = ctx
            .make_compute_pipeline(MPS_QR_SRC, MPS_QR_ENTRY)
            .expect("qr kernel compiles");
        let mut scratch = QrScratch::new(&ctx);

        for &(m, n) in &[
            (4usize, 4usize),
            (8, 5),
            (5, 8),
            (6, 6),
            (16, 4),
            (4, 16),
            (32, 12),
        ] {
            let a = block(m, n);
            let size = m.min(n);
            gpu_qr(&ctx, &pipeline, &mut scratch, &a, m, n);
            let q = &scratch.q()[..m * size];
            let r = &scratch.r()[..size * n];

            // A = Q·R: A[r,c] = Σ_t Q[r+t*m]·R[t+c*size].
            for rr in 0..m {
                for c in 0..n {
                    let mut acc = Complex::<f32>::new(0.0, 0.0);
                    for t in 0..size {
                        acc += q[rr + t * m] * r[t + c * size];
                    }
                    let e = a[rr + c * m];
                    let d = ((acc.re - e.re).powi(2) + (acc.im - e.im).powi(2)).sqrt();
                    assert!(d < 1e-3, "{m}x{n} QR recon ({rr},{c}) |Δ|={d:.2e}");
                }
            }
            // QᴴQ = I: Σ_r conj(Q[r+p*m])·Q[r+q*m] = δ_pq.
            for p in 0..size {
                for qq in 0..size {
                    let mut acc = Complex::<f32>::new(0.0, 0.0);
                    for rr in 0..m {
                        acc += q[rr + p * m].conj() * q[rr + qq * m];
                    }
                    let want = if p == qq { 1.0 } else { 0.0 };
                    let d = ((acc.re - want).powi(2) + acc.im.powi(2)).sqrt();
                    assert!(d < 2e-3, "{m}x{n} QᴴQ ({p},{qq})={acc:?} want {want}");
                }
            }
            // R upper-triangular: R[t,c]=0 for t>c.
            for t in 0..size {
                for c in 0..t.min(n) {
                    let z = r[t + c * size];
                    assert!(
                        z.re.abs() < 1e-4 && z.im.abs() < 1e-4,
                        "{m}x{n} R[{t},{c}] nonzero"
                    );
                }
            }
        }
    }
}
