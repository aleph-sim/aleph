//! GPU-resident one-sided Jacobi thin SVD (P5.7-02): host dispatch around the
//! `mps_jacobi.metal` kernel. A single threadgroup factors one two-site block on
//! the GPU, so the SVD no longer round-trips to a CPU library — the lever the
//! Phase-5.5/5.6 reports named as the MPS bottleneck.
//!
//! The kernel requires `m >= n`; a wide block (`rows < cols`) is factored via its
//! adjoint `Aᴴ` (tall) with the U/V roles swapped, exactly like the CPU reference
//! [`super::jacobi::jacobi_thin_svd`]. This module is the standalone, oracle-checked
//! kernel home; wiring it into [`super::backend::MetalMpsBackend`] is P5.7-03.

// The dispatch entry points are exercised by this module's on-device tests and
// land their production caller in P5.7-03 (backend wiring); until then they are
// unused in a non-test build. Remove when the backend calls `gpu_thin_svd`.
#![allow(dead_code)]

use aleph_core::Complex;
use metal::{ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::kernel::JacobiMeta;
use crate::{DeviceBuffer, MetalContext};

/// Thin-SVD factors of a `rows × cols` block, `k = min(rows, cols)`:
/// - `u`: column-major `rows × k` (`u[i + t*rows]` = `U[i][t]`),
/// - `sigma`: the `k` singular values (kernel order; **not** sorted),
/// - `v`: column-major `cols × k` (`v[j + t*cols]` = `V[j][t]`),
///
/// so that `A[i][j] = Σ_t U[i][t]·σ_t·conj(V[j][t])`.
pub(crate) struct GpuThinSvd {
    pub u: Vec<Complex<f32>>,
    pub sigma: Vec<f32>,
    pub v: Vec<Complex<f32>>,
}

/// Largest power of two `≤ x` (and `≥ 1`); the Jacobi reduction halves the
/// threadgroup size, so the dispatched count must be a power of two.
fn pow2_floor(x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    let high_bit = usize::BITS - 1 - x.leading_zeros();
    1usize << high_bit
}

/// Run the Metal Jacobi kernel on a tall/square column-major block `a_cm`
/// (`m × n`, `m ≥ n`), returning `(U(m×n col-major, in place), V(n×n col-major),
/// σ(n))`. `V` is seeded to the identity here. All device work completes before
/// return (the unified-memory read-back is then current).
fn dispatch_tall(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    a_cm: &[Complex<f32>],
    m: usize,
    n: usize,
) -> (Vec<Complex<f32>>, Vec<Complex<f32>>, Vec<f32>) {
    debug_assert!(m >= n && n >= 1);
    debug_assert_eq!(a_cm.len(), m * n);

    let mut a_buf = DeviceBuffer::from_slice(ctx, a_cm);
    // V = I_n, column-major.
    let mut v_host = vec![Complex::<f32>::new(0.0, 0.0); n * n];
    for t in 0..n {
        v_host[t + t * n] = Complex::<f32>::new(1.0, 0.0);
    }
    let mut v_buf = DeviceBuffer::from_slice(ctx, &v_host);
    let mut sig_buf = DeviceBuffer::from_slice(ctx, &vec![0.0f32; n]);

    let meta = JacobiMeta {
        m: m as u32,
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
    };
    // One threadgroup factors the block; threads stride the row dimension. The
    // count must be a power of two (reduction) and ≤ the kernel's `red[]` length.
    let cap = pipeline.max_total_threads_per_threadgroup().min(256) as usize;
    let threads = pow2_floor(cap.min(m).max(1)) as u64;

    let cmd = ctx.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a_buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(v_buf.metal_buffer()), 0);
    enc.set_buffer(2, Some(sig_buf.metal_buffer()), 0);
    enc.set_bytes(
        3,
        std::mem::size_of::<JacobiMeta>() as u64,
        &meta as *const JacobiMeta as *const c_void,
    );
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    (
        a_buf.as_mut_slice().to_vec(),
        v_buf.as_mut_slice().to_vec(),
        sig_buf.as_mut_slice().to_vec(),
    )
}

/// Thin one-sided Jacobi SVD of the row-major `rows × cols` block `theta` (FP32),
/// run on the GPU. Wide blocks (`rows < cols`) are factored as `Aᴴ` and the U/V
/// roles swapped. The returned factors follow [`GpuThinSvd`]'s column-major layout
/// with `k = min(rows, cols)`.
pub(crate) fn gpu_thin_svd(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    theta: &[Complex<f32>],
    rows: usize,
    cols: usize,
) -> GpuThinSvd {
    debug_assert_eq!(theta.len(), rows * cols);
    if rows >= cols {
        // m=rows, n=cols. Column-major A[i + t*m] = theta[i*cols + t].
        let (m, n) = (rows, cols);
        let mut a_cm = vec![Complex::<f32>::new(0.0, 0.0); m * n];
        for i in 0..m {
            for t in 0..n {
                a_cm[i + t * m] = theta[i * cols + t];
            }
        }
        let (u, v, sigma) = dispatch_tall(ctx, pipeline, &a_cm, m, n);
        GpuThinSvd { u, sigma, v }
    } else {
        // Wide: factor Aᴴ (cols×rows, tall). Aᴴ = Ũ Σ Ṽᴴ ⇒ A = Ṽ Σ Ũᴴ, so the
        // caller's U = Ṽ (rows×k) and V = Ũ (cols×k), k = rows.
        let (m, n) = (cols, rows);
        let mut ah_cm = vec![Complex::<f32>::new(0.0, 0.0); m * n];
        for i in 0..m {
            for t in 0..n {
                // (Aᴴ)[i][t] = conj(A[t][i]) = conj(theta[t*cols + i]).
                ah_cm[i + t * m] = theta[t * cols + i].conj();
            }
        }
        let (u_tilde, v_tilde, sigma) = dispatch_tall(ctx, pipeline, &ah_cm, m, n);
        GpuThinSvd {
            u: v_tilde, // Ṽ : rows×k
            sigma,
            v: u_tilde, // Ũ : cols×k
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernel::{MPS_JACOBI_ENTRY, MPS_JACOBI_SRC};
    use super::*;
    use faer::Mat;

    /// Deterministic FP32 complex test block, row-major `rows × cols`.
    fn test_block(rows: usize, cols: usize) -> Vec<Complex<f32>> {
        let mut v = vec![Complex::<f32>::new(0.0, 0.0); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                v[i * cols + j] = Complex::<f32>::new(
                    (((i * 7 + j * 3) % 11) as f32) * 0.37 - 1.1,
                    (((i * 5 + j) % 7) as f32) * 0.23 - 0.6,
                );
            }
        }
        v
    }

    /// faer FP64 reference singular values (descending) for the same block.
    fn faer_sigma(theta: &[Complex<f32>], rows: usize, cols: usize) -> Vec<f64> {
        let a = Mat::<aleph_core::Complex>::from_fn(rows, cols, |i, j| {
            let z = theta[i * cols + j];
            aleph_core::Complex::new(z.re as f64, z.im as f64)
        });
        let svd = a.thin_svd().unwrap();
        (0..rows.min(cols)).map(|t| svd.S()[t].re).collect()
    }

    /// Reconstruct `A[i][j] = Σ_t U[i][t]·σ_t·conj(V[j][t])` and compare to the
    /// input; cross-check σ against faer. FP32 Jacobi ⇒ ~1e-4 tolerance.
    #[test]
    fn gpu_jacobi_matches_reference() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping GPU Jacobi test: no Metal device");
                return;
            }
        };
        let pipeline = ctx
            .make_compute_pipeline(MPS_JACOBI_SRC, MPS_JACOBI_ENTRY)
            .expect("jacobi kernel compiles");

        for (rows, cols) in [
            (4usize, 4usize),
            (8, 5),
            (5, 8),
            (6, 6),
            (16, 4),
            (4, 16),
            (2, 2),
            (1, 1),
            (32, 12),
        ] {
            let theta = test_block(rows, cols);
            let k = rows.min(cols);
            let svd = gpu_thin_svd(&ctx, &pipeline, &theta, rows, cols);
            assert_eq!(svd.sigma.len(), k);
            assert_eq!(svd.u.len(), rows * k);
            assert_eq!(svd.v.len(), cols * k);

            // Reconstruction.
            for i in 0..rows {
                for j in 0..cols {
                    let mut acc = Complex::<f32>::new(0.0, 0.0);
                    for t in 0..k {
                        let u = svd.u[i + t * rows];
                        let vc = svd.v[j + t * cols].conj();
                        acc += u * svd.sigma[t] * vc;
                    }
                    let e = theta[i * cols + j];
                    let d = ((acc.re - e.re).powi(2) + (acc.im - e.im).powi(2)).sqrt();
                    assert!(d < 1e-4, "{rows}x{cols} reconstruct ({i},{j}) |Δ|={d:.2e}");
                }
            }

            // Singular values vs faer (sort the kernel's σ descending first).
            let mut got: Vec<f64> = svd.sigma.iter().map(|&s| s as f64).collect();
            got.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let want = faer_sigma(&theta, rows, cols);
            for t in 0..k {
                let d = (got[t] - want[t]).abs();
                assert!(
                    d < 1e-4,
                    "{rows}x{cols} σ[{t}] gpu {} vs faer {} (Δ={d:.2e})",
                    got[t],
                    want[t]
                );
            }
        }
    }

    /// Bell two-site block: σ = {1/√2, 1/√2}, reconstruct exactly.
    #[test]
    fn gpu_jacobi_bell_block() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping GPU Jacobi Bell test: no Metal device");
                return;
            }
        };
        let pipeline = ctx
            .make_compute_pipeline(MPS_JACOBI_SRC, MPS_JACOBI_ENTRY)
            .expect("jacobi kernel compiles");
        let r = (0.5f32).sqrt();
        // 2×2 diag(r, r), row-major.
        let theta = vec![
            Complex::<f32>::new(r, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(r, 0.0),
        ];
        let svd = gpu_thin_svd(&ctx, &pipeline, &theta, 2, 2);
        for &s in &svd.sigma {
            assert!((s - r).abs() < 1e-5, "σ should be 1/√2, got {s}");
        }
    }

    #[test]
    fn pow2_floor_values() {
        assert_eq!(pow2_floor(0), 1);
        assert_eq!(pow2_floor(1), 1);
        assert_eq!(pow2_floor(2), 2);
        assert_eq!(pow2_floor(3), 2);
        assert_eq!(pow2_floor(255), 128);
        assert_eq!(pow2_floor(256), 256);
        assert_eq!(pow2_floor(300), 256);
    }
}
