//! P8-11 isolated matrix experiment. Explicit feature only; no production routing.
//! Round-robin column pairs follow the circle-method edge colouring of K_n:
//! disjoint pairs per round, n-1 dependent rounds per sweep (pad odd n with a bye).
use super::*;
use crate::mps::{kernel::FinalizeOut, move_gpu, svd};
use faer::Mat;
use metal::{ComputeCommandEncoderRef, MTLCommandBufferStatus};
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy)]
#[repr(C)]
struct RoundMeta {
    m: u32,
    n: u32,
    round: u32,
    padded: u32,
}

/// Standalone stage timings (host encode + commit/wait included, milliseconds).
#[derive(Debug, Serialize)]
pub struct Receipt {
    pub rows: usize,
    pub cols: usize,
    pub fixture: String,
    pub candidate: bool,
    pub pack_ms: f64,
    pub factor_ms: f64,
    pub sweeps: usize,
    pub dispatches: usize,
    pub finalize_guard_ms: f64,
    pub sort_install_ms: f64,
    pub cpu_factor_ms: f64,
    pub cpu_split_ms: f64,
    pub qr_ms: f64,
    pub cpu_qr_ms: f64,
    pub qr_reconstruction: f64,
    pub qr_isometry: f64,
    pub reconstruction: f64,
    pub retained_isometry: f64,
    pub discarded_error: f64,
    pub norm_error: f64,
    pub observable_error: f64,
    pub guard_accept: bool,
    pub fallback_ms: f64,
    pub numerical_accept: bool,
}

/// Reusable fixture, pipelines and scratch for a bounded research matrix.
pub struct Case {
    ctx: MetalContext,
    rows: usize,
    cols: usize,
    fixture: String,
    input: Vec<Complex<f32>>,
    theta: DeviceBuffer<Complex<f32>>,
    scratch: JacobiScratch,
    pack: ComputePipelineState,
    reference: ComputePipelineState,
    round: ComputePipelineState,
    normalize: ComputePipelineState,
    finalize: ComputePipelineState,
    no_guard: ComputePipelineState,
    qr: ComputePipelineState,
    changes: DeviceBuffer<u32>,
    left: DeviceBuffer<Complex<f32>>,
    right: DeviceBuffer<Complex<f32>>,
    result: DeviceBuffer<FinalizeOut>,
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
fn bytes<T>(enc: &ComputeCommandEncoderRef, index: u64, value: &T) {
    enc.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const c_void,
    );
}
fn finish(cmd: &CommandBufferRef) -> Result<(), String> {
    cmd.commit();
    cmd.wait_until_completed();
    if cmd.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("Metal command status {:?}", cmd.status()));
    }
    Ok(())
}

/// Deterministic complex matrix with a known singular spectrum before FP32
/// rounding. Unitary butterfly row/column rotations densify the diagonal.
fn fixture(rows: usize, cols: usize, kind: &str) -> Vec<Complex<f32>> {
    let mut a = vec![Complex::new(0f64, 0.0); rows * cols];
    let k = rows.min(cols);
    for t in 0..k {
        let x = t as f64 / (k - 1).max(1) as f64;
        let s = match kind {
            "clustered" => 1.0 - 0.001 * x,
            "ill" => 10f64.powf(-7.0 * x),
            "rank_deficient" => {
                if t < k / 2 {
                    1.0 - 0.5 * x
                } else {
                    0.0
                }
            }
            _ => 1.0 - 0.75 * x,
        };
        a[t * cols + t] = Complex::new(s, 0.0);
    }
    for axis in 0..2 {
        let len = if axis == 0 { rows } else { cols };
        for stride in (0..len.ilog2()).map(|p| 1usize << p) {
            for start in (0..len).step_by(2 * stride) {
                for offset in 0..stride {
                    let p = start + offset;
                    let q = p + stride;
                    if q >= len {
                        continue;
                    }
                    let angle = 0.2 + 0.7 * ((p + 3 * q + axis) as f64).sin();
                    let (sn, cs) = angle.sin_cos();
                    let phase = Complex::new(0.6, 0.8);
                    for t in 0..if axis == 0 { cols } else { rows } {
                        let (i, j) = if axis == 0 {
                            (p * cols + t, q * cols + t)
                        } else {
                            (t * cols + p, t * cols + q)
                        };
                        let x = a[i];
                        let y = a[j];
                        a[i] = x * cs + y * phase * sn;
                        a[j] = -x * phase.conj() * sn + y * cs;
                    }
                }
            }
        }
    }
    let norm = a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    a.into_iter()
        .map(|z| Complex::new((z.re / norm) as f32, (z.im / norm) as f32))
        .collect()
}

impl Case {
    /// Acquire actual Metal hardware; never silently skip a research run.
    pub fn new(rows: usize, cols: usize, kind: &str) -> Result<Self, String> {
        if rows < 2
            || cols < 2
            || rows > 1024
            || cols > 1024
            || !["well", "clustered", "ill", "rank_deficient"].contains(&kind)
        {
            return Err("unsupported bounded fixture".into());
        }
        let ctx = MetalContext::new().map_err(|e| e.to_string())?;
        let pipeline = |src: &str, name: &str| {
            ctx.make_compute_pipeline(src, name)
                .map_err(|e| e.to_string())
        };
        let pack = pipeline(include_str!("../shaders/mps_pack.metal"), "pack_theta")?;
        let reference = pipeline(include_str!("../shaders/mps_jacobi.metal"), "jacobi_svd")?;
        let round = pipeline(
            include_str!("../shaders/mps_jacobi_rounds.metal"),
            "jacobi_round",
        )?;
        let normalize = pipeline(
            include_str!("../shaders/mps_jacobi_rounds.metal"),
            "jacobi_normalize",
        )?;
        let finalize_src = include_str!("../shaders/mps_finalize.metal");
        let finalize = pipeline(finalize_src, "finalize_split")?;
        // Counterfactual ONLY to attribute the guard cost. Never used for
        // accepted outputs or a production path; the actual guard always runs.
        let start = finalize_src
            .find("    // --- Accuracy guard")
            .ok_or("guard marker missing")?;
        let end = finalize_src
            .find("    // Pad to a power")
            .ok_or("sort marker missing")?;
        let no_guard_src = format!("{}{}", &finalize_src[..start], &finalize_src[end..]);
        let no_guard = pipeline(&no_guard_src, "finalize_split")?;
        let qr = pipeline(include_str!("../shaders/mps_qr.metal"), "householder_qr")?;
        let input = fixture(rows, cols, kind);
        let k = rows.min(cols);
        let mut theta = DeviceBuffer::with_capacity(&ctx, input.len());
        theta.write(&ctx, &input);
        let scratch = JacobiScratch::new(&ctx);
        let changes = DeviceBuffer::from_slice(&ctx, &[0u32]);
        let left = DeviceBuffer::with_capacity(&ctx, rows * k);
        let right = DeviceBuffer::with_capacity(&ctx, k * cols);
        let mut result = DeviceBuffer::with_capacity(&ctx, 1);
        result.ensure_capacity(&ctx, 1);
        Ok(Self {
            ctx,
            rows,
            cols,
            fixture: kind.into(),
            input,
            theta,
            scratch,
            pack,
            reference,
            round,
            normalize,
            finalize,
            no_guard,
            qr,
            changes,
            left,
            right,
            result,
        })
    }

    fn pack(&mut self) -> Result<(), String> {
        let (rows, cols) = (self.rows, self.cols);
        let wide = rows < cols;
        self.scratch
            .prepare(&self.ctx, rows.max(cols), rows.min(cols));
        let meta = PackMeta {
            rows: rows as u32,
            cols: cols as u32,
            wide: wide as u32,
            _pad0: 0,
        };
        let cmd = self.ctx.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pack);
        enc.set_buffer(0, Some(self.theta.metal_buffer()), 0);
        enc.set_buffer(1, Some(self.scratch.a.metal_buffer()), 0);
        bytes(enc, 2, &meta);
        let count = (rows * cols) as u64;
        enc.dispatch_threads(
            MTLSize::new(count, 1, 1),
            MTLSize::new(
                self.pack.max_total_threads_per_threadgroup().min(count),
                1,
                1,
            ),
        );
        enc.end_encoding();
        finish(cmd)
    }

    fn factor(&mut self, candidate: bool) -> Result<(usize, usize), String> {
        let (m, n) = (self.rows.max(self.cols), self.rows.min(self.cols));
        let threads = pow2_floor(m.min(256)) as u64;
        if !candidate {
            let meta = JacobiMeta {
                m: m as u32,
                n: n as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let cmd = self.ctx.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.reference);
            enc.set_buffer(0, Some(self.scratch.a.metal_buffer()), 0);
            enc.set_buffer(1, Some(self.scratch.v.metal_buffer()), 0);
            enc.set_buffer(2, Some(self.scratch.sig.metal_buffer()), 0);
            bytes(enc, 3, &meta);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
            enc.end_encoding();
            finish(cmd)?;
            return Ok((0, 1)); // reference internally converges; no sweep claim
        }
        let padded = n + n % 2;
        let mut sweeps = 0;
        for sweep in 0..60 {
            self.changes.fill_zero();
            let cmd = self.ctx.queue().new_command_buffer();
            for round in 0..padded - 1 {
                let meta = RoundMeta {
                    m: m as u32,
                    n: n as u32,
                    round: round as u32,
                    padded: padded as u32,
                };
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.round);
                enc.set_buffer(0, Some(self.scratch.a.metal_buffer()), 0);
                enc.set_buffer(1, Some(self.scratch.v.metal_buffer()), 0);
                enc.set_buffer(2, Some(self.changes.metal_buffer()), 0);
                bytes(enc, 3, &meta);
                enc.dispatch_thread_groups(
                    MTLSize::new((padded / 2) as u64, 1, 1),
                    MTLSize::new(threads, 1, 1),
                );
                enc.end_encoding();
            }
            finish(cmd)?;
            sweeps = sweep + 1;
            if self.changes.as_slice()[0] == 0 {
                break;
            }
        }
        let cmd = self.ctx.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.normalize);
        enc.set_buffer(0, Some(self.scratch.a.metal_buffer()), 0);
        enc.set_buffer(1, Some(self.scratch.sig.metal_buffer()), 0);
        bytes(
            enc,
            2,
            &RoundMeta {
                m: m as u32,
                n: n as u32,
                round: 0,
                padded: padded as u32,
            },
        );
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(threads, 1, 1));
        enc.end_encoding();
        finish(cmd)?;
        Ok((sweeps, sweeps * (padded - 1) + 1))
    }

    /// Timing-only factorization for Criterion; scratch reset/pack included in
    /// both paths, as are host dispatch and completion waits. No shader compile.
    pub fn bench_factor(&mut self, candidate: bool) -> Result<(), String> {
        self.pack()?;
        self.factor(candidate)?;
        Ok(())
    }

    fn finalize(&self, guard: bool) -> Result<(), String> {
        let cmd = self.ctx.queue().new_command_buffer();
        encode_finalize(
            cmd,
            if guard {
                &self.finalize
            } else {
                &self.no_guard
            },
            &self.scratch,
            &self.theta,
            &self.left,
            &self.right,
            &self.result,
            self.rows,
            self.cols,
            self.rows < self.cols,
            self.rows.min(self.cols) / 2,
            true,
        );
        finish(cmd)
    }

    /// Run one instrumented case, measuring and checking the actual guard,
    /// CPU fallback, complete reconstruction and truncated canonical factors.
    pub fn run(&mut self, candidate: bool) -> Result<Receipt, String> {
        let k = self.rows.min(self.cols);
        let cap = k / 2;
        let start = Instant::now();
        self.pack()?;
        let pack_ms = ms(start);
        let start = Instant::now();
        let (sweeps, dispatches) = self.factor(candidate)?;
        let factor_ms = ms(start);
        let start = Instant::now();
        self.finalize(false)?;
        let sort_install_ms = ms(start);
        let start = Instant::now();
        self.finalize(true)?;
        let finalize_guard_ms = ms(start);
        let guard_accept = self.result.as_slice()[0].accept != 0;
        let a = Mat::from_fn(self.rows, self.cols, |i, j| {
            let z = self.input[i * self.cols + j];
            Complex::new(z.re as f64, z.im as f64)
        });
        let start = Instant::now();
        let cpu = svd::factor(a.as_ref(), k).map_err(|e| e.to_string())?;
        let cpu_factor_ms = ms(start);
        let start = Instant::now();
        let cpu_split = svd::svd_split(&self.input, self.rows, self.cols, cap, true)
            .map_err(|e| e.to_string())?;
        let cpu_split_ms = ms(start);
        let fallback_ms = if guard_accept { 0.0 } else { cpu_split_ms };
        let (u, v) = if self.rows < self.cols {
            (&self.scratch.v, &self.scratch.a)
        } else {
            (&self.scratch.a, &self.scratch.v)
        };
        let umat = Mat::from_fn(self.rows, k, |i, t| {
            let z = u.as_slice()[i + t * self.rows];
            Complex::new(z.re as f64, z.im as f64)
        });
        let vmat = Mat::from_fn(self.cols, k, |i, t| {
            let z = v.as_slice()[i + t * self.cols];
            Complex::new(z.re as f64, z.im as f64)
        });
        let sig = self.scratch.sig.as_slice();
        let us = Mat::from_fn(self.rows, k, |i, t| umat[(i, t)] * sig[t] as f64);
        let reconstructed = &us * vmat.adjoint();
        let reconstruction = relative(&reconstructed, &a);
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&x, &y| sig[y].total_cmp(&sig[x]));
        let sig64: Vec<_> = order.iter().map(|&i| sig[i] as f64).collect();
        let (keep, discard) = svd::truncation_plan(&sig64, cap);
        let total = sig64.iter().map(|s| s * s).sum::<f64>();
        let trunc = if total > 0.0 { discard / total } else { 0.0 };
        let kept = Mat::from_fn(self.rows, keep, |i, t| umat[(i, order[t])]);
        let gram = kept.adjoint() * &kept;
        let retained_isometry = (0..keep)
            .flat_map(|i| (0..keep).map(move |j| (i, j)))
            .map(|(i, j)| (gram[(i, j)] - Complex::new(if i == j { 1.0 } else { 0.0 }, 0.0)).norm())
            .fold(0.0, f64::max);
        // Reconstruct the *truncated normalized* bipartite state from the
        // prototype factors, even when the production guard declined them.
        let scale = if trunc > 1e-12 {
            1.0 / (total - discard).sqrt()
        } else {
            1.0
        };
        let kept_us = Mat::from_fn(self.rows, keep, |i, t| {
            umat[(i, order[t])] * sig64[t] * scale
        });
        let kept_v = Mat::from_fn(self.cols, keep, |i, t| vmat[(i, order[t])]);
        let truncated = &kept_us * kept_v.adjoint();
        let cpu_truncated = Mat::from_fn(self.rows, cpu_split.0, |i, t| {
            let z = cpu_split.1[i * cpu_split.0 + t];
            Complex::new(z.re as f64, z.im as f64)
        }) * Mat::from_fn(cpu_split.0, self.cols, |t, j| {
            let z = cpu_split.2[t * self.cols + j];
            Complex::new(z.re as f64, z.im as f64)
        });
        let norm_error = (norm2(&truncated) - norm2(&cpu_truncated)).abs();
        let observable_error = (observable(&truncated) - observable(&cpu_truncated)).abs();
        let discarded_error = (trunc - cpu_split.3).abs();
        // CPU QR is not a stand-in for SVD; compare the actual faer QR path.
        let start = Instant::now();
        let _cpu_qr = a.qr();
        let cpu_qr_ms = ms(start);
        let qr_a = DeviceBuffer::from_slice(
            &self.ctx,
            &(0..self.cols)
                .flat_map(|j| (0..self.rows).map(move |i| (i, j)))
                .map(|(i, j)| self.input[i * self.cols + j])
                .collect::<Vec<_>>(),
        );
        let mut q = DeviceBuffer::with_capacity(&self.ctx, self.rows * k);
        q.ensure_capacity(&self.ctx, self.rows * k);
        let mut r = DeviceBuffer::with_capacity(&self.ctx, k * self.cols);
        r.ensure_capacity(&self.ctx, k * self.cols);
        let start = Instant::now();
        let cmd = self.ctx.queue().new_command_buffer();
        move_gpu::encode_qr(cmd, &self.qr, &qr_a, &q, &r, self.rows, self.cols);
        finish(cmd)?;
        let qr_ms = ms(start);
        let qm = Mat::from_fn(self.rows, k, |i, t| {
            let z = q.as_slice()[i + t * self.rows];
            Complex::new(z.re as f64, z.im as f64)
        });
        let rm = Mat::from_fn(k, self.cols, |t, j| {
            let z = r.as_slice()[t + j * k];
            Complex::new(z.re as f64, z.im as f64)
        });
        let qr_reconstruction = relative(&(&qm * &rm), &a);
        let qgram = qm.adjoint() * &qm;
        let qr_isometry = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| {
                (qgram[(i, j)] - Complex::new(if i == j { 1.0 } else { 0.0 }, 0.0)).norm()
            })
            .fold(0.0, f64::max);
        let _cpu_spectrum = cpu.sigma; // force complete CPU factorization above
        let metrics = [
            reconstruction,
            retained_isometry,
            discarded_error,
            norm_error,
            observable_error,
        ];
        let numerical_accept = metrics.iter().all(|x| x.is_finite())
            && reconstruction <= 1e-3
            && retained_isometry <= 1e-4
            && discarded_error <= 1e-5
            && norm_error <= 1e-5
            && observable_error <= 1e-5;
        Ok(Receipt {
            rows: self.rows,
            cols: self.cols,
            fixture: self.fixture.clone(),
            candidate,
            pack_ms,
            factor_ms,
            sweeps,
            dispatches,
            finalize_guard_ms,
            sort_install_ms,
            cpu_factor_ms,
            cpu_split_ms,
            qr_ms,
            cpu_qr_ms,
            qr_reconstruction,
            qr_isometry,
            reconstruction,
            retained_isometry,
            discarded_error,
            norm_error,
            observable_error,
            guard_accept,
            fallback_ms,
            numerical_accept,
        })
    }
}
fn norm2(a: &Mat<Complex<f64>>) -> f64 {
    (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| a[(i, j)].norm_sqr()))
        .sum()
}
fn relative(a: &Mat<Complex<f64>>, b: &Mat<Complex<f64>>) -> f64 {
    let d = (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| (a[(i, j)] - b[(i, j)]).norm_sqr()))
        .sum::<f64>();
    (d / norm2(b).max(f64::MIN_POSITIVE)).sqrt()
}
fn observable(a: &Mat<Complex<f64>>) -> f64 {
    let signed = (0..a.nrows())
        .flat_map(|i| {
            (0..a.ncols()).map(move |j| a[(i, j)].norm_sqr() * if i % 2 == 0 { 1.0 } else { -1.0 })
        })
        .sum::<f64>();
    signed / norm2(a).max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    #[test]
    fn actual_metal_candidate_preserves_small_hard_fixtures() {
        if std::env::var_os("ALEPH_TEST_FACTOR_RESEARCH").is_none() {
            return;
        }
        for (m, n) in [(16, 16), (16, 8), (8, 16), (9, 7)] {
            for kind in ["well", "clustered", "ill", "rank_deficient"] {
                let mut case = super::Case::new(m, n, kind).unwrap();
                let receipt = case.run(true).unwrap();
                assert!(
                    receipt.reconstruction.is_finite() && receipt.reconstruction < 1e-3,
                    "{receipt:?}"
                );
                // Strict canonical/truncation acceptance remains in each receipt;
                // no performance claim is allowed to override a rejected case.
            }
        }
    }

    #[test]
    fn circle_rounds_cover_each_pair_once_without_column_races() {
        for n in 2..36 {
            let padded = n + n % 2;
            let mut seen = std::collections::BTreeSet::new();
            for round in 0..padded - 1 {
                let mut used = std::collections::BTreeSet::new();
                for pair in 0..padded / 2 {
                    let map = |x| {
                        if x == padded - 1 {
                            x
                        } else {
                            (x + round) % (padded - 1)
                        }
                    };
                    let (p, q) = (map(pair), map(padded - 1 - pair));
                    if p >= n || q >= n {
                        continue;
                    }
                    assert!(used.insert(p) && used.insert(q));
                    assert!(seen.insert((p.min(q), p.max(q))));
                }
            }
            assert_eq!(seen.len(), n * (n - 1) / 2);
        }
    }
}
