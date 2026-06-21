//! GPU-resident one-sided Jacobi thin SVD (P5.7-02): host dispatch around the
//! `mps_jacobi.metal` kernel. A single threadgroup factors one two-site block on
//! the GPU, so the SVD no longer round-trips to a CPU library — the lever the
//! Phase-5.5/5.6 reports named as the MPS bottleneck.
//!
//! The kernel requires `m >= n`; a wide block (`rows < cols`) is factored via its
//! adjoint `Aᴴ` (tall) with the U/V roles swapped, exactly like the CPU reference
//! [`super::jacobi::jacobi_thin_svd`]. This module is the standalone, oracle-checked
//! kernel home; wiring it into [`super::backend::MetalMpsBackend`] is P5.7-03.

use aleph_core::Complex;
use metal::{ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::kernel::{JacobiBlockMeta, JacobiMeta};
use super::svd::{truncation_plan, SplitResult};
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

/// GPU-resident truncated SVD split of the row-major Θ′ block → the two new site
/// tensors, matching the [`super::svd::svd_split`] contract (so the backend's
/// truncation guard is unchanged). Runs the Jacobi kernel, sorts σ descending,
/// applies the same `truncation_plan`, and builds `site_i = U[:, :χ]` /
/// `site_j = diag(σ_kept)·Vᴴ` directly in f32.
///
/// Returns `None` if the kernel produced a non-finite singular value, so the
/// caller can fall back to the f64 CPU SVD — a single-precision Gram-free Jacobi is
/// accurate for the well-conditioned exact-only scaffold, but the f64 path stays
/// the safety net.
pub(crate) fn gpu_svd_split(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    theta: &[Complex<f32>],
    rows: usize,
    cols: usize,
    max_bond: usize,
    renormalize: bool,
) -> Option<SplitResult> {
    let svd = gpu_thin_svd(ctx, pipeline, theta, rows, cols);
    // Accuracy guard: the single-precision one-sided Jacobi can mis-converge on an
    // ill-conditioned block (clustered/large-spread singular values), leaving U not
    // quite isometric — a small per-gate error that compounds the state norm over a
    // deep circuit. Verify ‖Θ′ − UΣVᴴ‖_F is small; if not, degrade to the f64 faer
    // SVD (as for a non-finite result). Well-conditioned blocks pass at ~1e-4.
    if recon_residual(theta, &svd.u, &svd.sigma, &svd.v, rows, cols) > RECON_TOL {
        return None;
    }
    split_from_thin(
        &svd.u,
        &svd.sigma,
        &svd.v,
        rows,
        cols,
        max_bond,
        renormalize,
    )
}

/// Relative Frobenius residual `‖Θ − UΣVᴴ‖_F / ‖Θ‖_F` of a thin SVD, in f64.
/// `u`/`v` are column-major (rows×k / cols×k), `sigma` length `k = min(rows,cols)`.
fn recon_residual(
    theta: &[Complex<f32>],
    u: &[Complex<f32>],
    sigma: &[f32],
    v: &[Complex<f32>],
    rows: usize,
    cols: usize,
) -> f64 {
    let k = rows.min(cols);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = Complex::<f64>::new(0.0, 0.0);
            for t in 0..k {
                let uu = u[i + t * rows];
                let vv = v[j + t * cols];
                let uf = Complex::<f64>::new(uu.re as f64, uu.im as f64);
                let vf = Complex::<f64>::new(vv.re as f64, -vv.im as f64); // conj(V)
                acc += uf * (sigma[t] as f64) * vf;
            }
            let e = theta[i * cols + j];
            let dr = acc.re - e.re as f64;
            let di = acc.im - e.im as f64;
            num += dr * dr + di * di;
            den += (e.re as f64) * (e.re as f64) + (e.im as f64) * (e.im as f64);
        }
    }
    if den > 0.0 {
        (num / den).sqrt()
    } else {
        0.0
    }
}

/// Relative-residual ceiling for accepting a GPU Jacobi split. Converged f32
/// blocks reconstruct to ~1e-4; a mis-converged block lands far above this and is
/// re-factored on the f64 CPU path.
const RECON_TOL: f64 = 1e-3;

/// Build the truncated two-site split from thin-SVD factors in [`GpuThinSvd`]'s
/// column-major layout (`u`: rows×k, `v`: cols×k, `sigma`: k, `k = min(rows,cols)`),
/// shared by the single-block [`gpu_svd_split`] and the batched
/// [`gpu_svd_split_batch`]. Sorts σ descending, applies the same `truncation_plan`,
/// and emits `site_i = U[:, :χ]` / `site_j = diag(σ_kept)·Vᴴ` in f32.
///
/// Returns `None` if any singular value is non-finite, so the caller can fall back
/// to the f64 CPU SVD (matching the single-block degrade path).
///
/// When `renormalize` is true (the canonical, truncating path — P5.7-07), the kept
/// σ are rescaled by `1/√Σ_kept σ²` so a truncated split preserves the (centre-on-
/// bond) state norm; with `false` (the exact `run_batched` path) the σ are kept
/// verbatim and the caller refuses any non-negligible `trunc_rel`.
fn split_from_thin(
    u: &[Complex<f32>],
    sigma: &[f32],
    v: &[Complex<f32>],
    rows: usize,
    cols: usize,
    max_bond: usize,
    renormalize: bool,
) -> Option<SplitResult> {
    let k = rows.min(cols);
    // σ descending; carry the permutation so U/V columns follow.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap());
    let sigmas_desc: Vec<f64> = order.iter().map(|&t| sigma[t] as f64).collect();
    if !sigmas_desc.iter().all(|s| s.is_finite()) {
        return None; // degrade to the CPU SVD
    }

    let (chi, discarded) = truncation_plan(&sigmas_desc, max_bond);
    let total: f64 = sigmas_desc.iter().map(|s| s * s).sum();
    let trunc_rel = if total > 0.0 { discarded / total } else { 0.0 };
    // Renormalise the kept block to unit weight ONLY on a real truncation. When
    // nothing is dropped (`trunc_rel` ≈ 0), rescaling by 1/√Σσ² would inject the
    // GPU σ's f32 error into the state norm on *every* gate and compound across the
    // circuit — so leave σ verbatim there (the split is exact). On a genuine
    // truncation, scale = 1/√Σ_kept σ² restores the centre-on-bond norm (P5.7-07).
    let scale = if renormalize && trunc_rel > 1e-12 {
        let kept: f64 = sigmas_desc[..chi].iter().map(|s| s * s).sum();
        if kept > 0.0 {
            (1.0 / kept).sqrt() as f32
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Site i ← U[:, kept]: row-major (rows, chi). U is column-major: u[i + t*rows].
    let mut site_i = vec![Complex::<f32>::new(0.0, 0.0); rows * chi];
    for row in 0..rows {
        for (t_new, &t_old) in order.iter().take(chi).enumerate() {
            site_i[row * chi + t_new] = u[row + t_old * rows];
        }
    }
    // Site j ← diag(σ_kept·scale)·Vᴴ: row-major (chi, cols), entry = σ_t·scale·conj(V[col,t]).
    // V is column-major: v[col + t*cols].
    let mut site_j = vec![Complex::<f32>::new(0.0, 0.0); chi * cols];
    for (t_new, &t_old) in order.iter().take(chi).enumerate() {
        let s = sigma[t_old] * scale;
        for col in 0..cols {
            site_j[t_new * cols + col] = v[col + t_old * cols].conj().scale(s);
        }
    }
    Some((chi, site_i, site_j, trunc_rel))
}

/// One block of a batched SVD: a row-major `rows × cols` Θ′ slice. The blocks of
/// a brickwall layer act on disjoint site pairs, so they are independent.
pub(crate) struct BatchBlock<'a> {
    pub theta: &'a [Complex<f32>],
    pub rows: usize,
    pub cols: usize,
}

/// GPU-resident **batched** truncated SVD split (P5.7-04): factor every block of a
/// brickwall layer in a *single* kernel launch (one threadgroup per block, one
/// `commit`/`wait` for the whole layer) instead of one dispatch + sync per gate.
///
/// Each block is packed (column-major, tall orientation — wide blocks factored as
/// `Aᴴ`) into shared `A`/`V`/`sig` buffers with per-block offsets; the kernel keys
/// off `threadgroup_position_in_grid`. Returns one entry per input block in order:
/// `Some(split)` on success, or `None` for a block whose GPU σ went non-finite, so
/// the caller falls back to the f64 CPU SVD for just that block (matching the
/// single-block degrade path).
pub(crate) fn gpu_svd_split_batch(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    blocks: &[BatchBlock<'_>],
    max_bond: usize,
) -> Vec<Option<SplitResult>> {
    if blocks.is_empty() {
        return Vec::new();
    }

    // --- Pack every block (tall orientation) into the shared buffers ---
    let mut a_host: Vec<Complex<f32>> = Vec::new();
    let mut v_host: Vec<Complex<f32>> = Vec::new();
    let mut sig_host: Vec<f32> = Vec::new();
    let mut metas: Vec<JacobiBlockMeta> = Vec::with_capacity(blocks.len());
    // Per-block (m, n, wide) so the read-back maps the factors back correctly.
    let mut shapes: Vec<(usize, usize, bool)> = Vec::with_capacity(blocks.len());
    let mut max_m = 1usize;

    for blk in blocks {
        let (rows, cols) = (blk.rows, blk.cols);
        debug_assert_eq!(blk.theta.len(), rows * cols);
        let wide = rows < cols;
        // Tall dims: m ≥ n. Tall keeps A as-is; wide factors Aᴴ.
        let (m, n) = if wide { (cols, rows) } else { (rows, cols) };
        let a_off = a_host.len();
        let v_off = v_host.len();
        let sig_off = sig_host.len();

        // A: column-major m×n. Tall: A[i + t*m] = θ[i*cols + t]. Wide (Aᴴ):
        // (Aᴴ)[i + t*m] = conj(θ[t*cols + i]).
        a_host.resize(a_off + m * n, Complex::<f32>::new(0.0, 0.0));
        let a_slot = &mut a_host[a_off..a_off + m * n];
        if wide {
            for i in 0..m {
                for t in 0..n {
                    a_slot[i + t * m] = blk.theta[t * cols + i].conj();
                }
            }
        } else {
            for i in 0..m {
                for t in 0..n {
                    a_slot[i + t * m] = blk.theta[i * cols + t];
                }
            }
        }

        // V seeded to I_n (column-major n×n).
        v_host.resize(v_off + n * n, Complex::<f32>::new(0.0, 0.0));
        for t in 0..n {
            v_host[v_off + t + t * n] = Complex::<f32>::new(1.0, 0.0);
        }
        // σ: n slots.
        sig_host.resize(sig_off + n, 0.0);

        metas.push(JacobiBlockMeta {
            m: m as u32,
            n: n as u32,
            a_off: a_off as u32,
            v_off: v_off as u32,
            sig_off: sig_off as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        });
        shapes.push((m, n, wide));
        max_m = max_m.max(m);
    }

    let mut a_buf = DeviceBuffer::from_slice(ctx, &a_host);
    let mut v_buf = DeviceBuffer::from_slice(ctx, &v_host);
    let sig_buf = DeviceBuffer::from_slice(ctx, &sig_host);
    let meta_buf = DeviceBuffer::from_slice(ctx, &metas);

    // One threadgroup per block; all share the same threadgroup size (≤ cap, a
    // power of two, sized to the largest block's rows). Threads beyond a block's
    // row count contribute zero to the reductions, so a uniform size is correct.
    let cap = pipeline.max_total_threads_per_threadgroup().min(256) as usize;
    let threads = pow2_floor(cap.min(max_m).max(1)) as u64;

    let cmd = ctx.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a_buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(v_buf.metal_buffer()), 0);
    enc.set_buffer(2, Some(sig_buf.metal_buffer()), 0);
    enc.set_buffer(3, Some(meta_buf.metal_buffer()), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(blocks.len() as u64, 1, 1),
        MTLSize::new(threads, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // --- Read back, build each split from its slice of the packed factors ---
    let a_out = a_buf.as_mut_slice();
    let v_out = v_buf.as_mut_slice();
    let sig_out = sig_buf.as_slice();
    let mut out = Vec::with_capacity(blocks.len());
    for (blk, (meta, &(m, n, wide))) in blocks.iter().zip(metas.iter().zip(shapes.iter())) {
        let (rows, cols) = (blk.rows, blk.cols);
        let a_off = meta.a_off as usize;
        let v_off = meta.v_off as usize;
        let sig_off = meta.sig_off as usize;
        let a_slice = &a_out[a_off..a_off + m * n];
        let v_slice = &v_out[v_off..v_off + n * n];
        let sig_slice = &sig_out[sig_off..sig_off + n];
        // Tall: A→U (rows×k), V→V (cols×k). Wide (factored Aᴴ): the original
        // U = Ṽ (= V-of-Aᴴ, rows×k) and V = Ũ (= A-of-Aᴴ, cols×k); k = n.
        let (u_b, v_b) = if wide {
            (v_slice, a_slice)
        } else {
            (a_slice, v_slice)
        };
        // Same accuracy guard as the single-block path: a mis-converged f32 block
        // degrades to the f64 CPU SVD (caller maps `None` → `svd_split`). Batched
        // path is exact-only (run_batched): `renormalize = false`.
        let split = if recon_residual(blk.theta, u_b, sig_slice, v_b, rows, cols) > RECON_TOL {
            None
        } else {
            split_from_thin(u_b, sig_slice, v_b, rows, cols, max_bond, false)
        };
        out.push(split);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::kernel::{MPS_JACOBI_BATCHED_ENTRY, MPS_JACOBI_ENTRY, MPS_JACOBI_SRC};
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

    /// P5.7-04: a batch of mixed-shape blocks factored in one dispatch must (a)
    /// reconstruct each Θ = U·(ΣVᴴ) and (b) agree, block-for-block, with the
    /// single-block path's χ. Tall, square, wide, and degenerate (1×1, 2×2) shapes
    /// in the same launch exercise the per-block offset/orientation packing.
    #[test]
    fn gpu_jacobi_batch_matches_single() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping batched Jacobi test: no Metal device");
                return;
            }
        };
        let single = ctx
            .make_compute_pipeline(MPS_JACOBI_SRC, MPS_JACOBI_ENTRY)
            .expect("single jacobi kernel compiles");
        let batched = ctx
            .make_compute_pipeline(MPS_JACOBI_SRC, MPS_JACOBI_BATCHED_ENTRY)
            .expect("batched jacobi kernel compiles");

        let shapes = [
            (4usize, 4usize),
            (8, 5),
            (5, 8),
            (6, 6),
            (2, 2),
            (1, 1),
            (16, 4),
            (4, 16),
            (32, 12),
        ];
        let data: Vec<(Vec<Complex<f32>>, usize, usize)> = shapes
            .iter()
            .map(|&(r, c)| (test_block(r, c), r, c))
            .collect();
        let batch: Vec<BatchBlock> = data
            .iter()
            .map(|(t, r, c)| BatchBlock {
                theta: t,
                rows: *r,
                cols: *c,
            })
            .collect();
        let results = gpu_svd_split_batch(&ctx, &batched, &batch, 64);
        assert_eq!(results.len(), shapes.len());

        for ((theta, rows, cols), res) in data.iter().zip(results.iter()) {
            let (chi, si, sj, trunc) = res.as_ref().expect("finite batched split");
            assert!(
                *trunc < 1e-5,
                "{rows}x{cols} unexpected truncation {trunc:.2e}"
            );
            // Reconstruct Θ = site_i · site_j (= U·diag(σ)·Vᴴ).
            for r in 0..*rows {
                for c in 0..*cols {
                    let mut acc = Complex::<f32>::new(0.0, 0.0);
                    for t in 0..*chi {
                        acc += si[r * chi + t] * sj[t * cols + c];
                    }
                    let e = theta[r * cols + c];
                    let d = ((acc.re - e.re).powi(2) + (acc.im - e.im).powi(2)).sqrt();
                    assert!(d < 1e-3, "{rows}x{cols} reconstruct ({r},{c}) |Δ|={d:.2e}");
                }
            }
            // χ must match the single-block path on the same block.
            let (single_chi, ..) =
                gpu_svd_split(&ctx, &single, theta, *rows, *cols, 64, false).expect("single split");
            assert_eq!(*chi, single_chi, "{rows}x{cols} χ batched vs single");
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
