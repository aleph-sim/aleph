//! Fully GPU-resident, **fused** canonical centre moves (P5.8-04). Each centre step
//! (pack → Householder QR → install Q → absorb R) is *encoded* onto the gate's
//! command buffer, so a whole move sweep + the gate split share **one
//! `commit`/`wait` per gate** — no per-move host round-trip. Move dimensions are
//! deterministic (`size = min(left·2, mid)`), so all buffers are pre-sized on the
//! host before encoding; per-step "ping-pong" buffers keep every intermediate
//! resident while the single command buffer is in flight.

use aleph_core::Complex;
use metal::{CommandBufferRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::kernel::{PackMeta, QrInstallMeta, QrMeta};
use super::state::SiteTensor;
use crate::{DeviceBuffer, MetalContext};

/// QR `n` (= the contracted bond) above which the single-threadgroup QR kernel's
/// `betas[]` cache overflows; such a move falls back to the host f64 SVD.
const QR_MAX_N: usize = 1024;
/// Central bond below which a centre move stays on the host f64 SVD: for small
/// blocks the single-threadgroup GPU QR + install + absorb cannot out-parallelise
/// faer (even fused with no per-move sync), so the host path is faster. The fused
/// GPU move pays off only once the bond is large enough to amortise its kernels.
const FUSED_MIN_BOND: usize = 96;

/// One planned centre step's geometry (deterministic — no GPU needed to compute).
#[derive(Clone, Copy)]
pub(crate) struct StepDims {
    /// Site finalised by this step (its `keep` buffer is installed here).
    pub site: usize,
    /// Neighbour absorbed into (right: `site+1`; left: `site-1`).
    pub nbr: usize,
    pub qr_m: usize,
    pub qr_n: usize,
    pub size: usize,
    /// Finalised-site dims `(left, right)`.
    pub keep_l: usize,
    pub keep_r: usize,
    /// Absorbed-(center)-site dims `(left, right)`.
    pub ctr_l: usize,
    pub ctr_r: usize,
    /// Per-kernel scalars: for right `(mid, nr)`; for left `(li, ri, pl)`.
    pub p0: usize,
    pub p1: usize,
    pub p2: usize,
}

/// A planned, deterministic centre-move sweep. `None` from [`plan_canonical_moves`]
/// means a block is too large for the GPU QR — the caller uses the host SVD fallback.
pub(crate) struct MovePlan {
    /// +1 right sweep, -1 left sweep.
    pub dir: i8,
    pub steps: Vec<StepDims>,
    /// Post-move gate dims.
    pub li: usize,
    pub ci: usize,
    pub ri: usize,
}

/// Compute the deterministic move sweep from `center` to `target` (gate site `i`).
/// Returns `None` if any block exceeds the GPU QR size cap (host SVD fallback).
pub(crate) fn plan_canonical_moves(
    sites: &[SiteTensor],
    center: usize,
    target: usize,
) -> Option<MovePlan> {
    let i = target;
    if center == i {
        return Some(MovePlan {
            dir: 0,
            steps: Vec::new(),
            li: sites[i].left,
            ci: sites[i].right,
            ri: sites[i + 1].right,
        });
    }
    let mut steps = Vec::new();
    if center < i {
        // RIGHT sweep: steps at j = center .. i-1.
        let mut lj = sites[center].left;
        for j in center..i {
            let mid = sites[j].right;
            let rows = lj * 2;
            let size = rows.min(mid);
            let nr = sites[j + 1].right;
            if mid > QR_MAX_N {
                return None;
            }
            steps.push(StepDims {
                site: j,
                nbr: j + 1,
                qr_m: rows,
                qr_n: mid,
                size,
                keep_l: lj,
                keep_r: size,
                ctr_l: size,
                ctr_r: nr,
                p0: mid,
                p1: nr,
                p2: 0,
            });
            lj = size;
        }
        let last = steps.last().unwrap();
        let ci = sites[i].right;
        if ci < FUSED_MIN_BOND {
            return None; // tiny blocks: host f64 SVD move is faster
        }
        Some(MovePlan {
            dir: 1,
            li: last.size,
            ci,
            ri: sites[i + 1].right,
            steps,
        })
    } else {
        // LEFT sweep: steps at j = center, center-1, .., i+1.
        let mut rj = sites[center].right;
        let mut jj = center as isize;
        while jj > i as isize {
            let j = jj as usize;
            let lj = sites[j].left;
            let cols = 2 * rj;
            let size = cols.min(lj);
            let pl = sites[j - 1].left;
            if lj > QR_MAX_N {
                return None;
            }
            steps.push(StepDims {
                site: j,
                nbr: j - 1,
                qr_m: cols,
                qr_n: lj,
                size,
                keep_l: size,
                keep_r: rj,
                ctr_l: pl,
                ctr_r: size,
                p0: lj,
                p1: rj,
                p2: pl,
            });
            rj = size;
            jj -= 1;
        }
        let last = steps.last().unwrap();
        let ci = last.size;
        if ci < FUSED_MIN_BOND {
            return None; // tiny blocks: host f64 SVD move is faster
        }
        Some(MovePlan {
            dir: -1,
            li: sites[i].left,
            ci,
            ri: last.keep_r,
            steps,
        })
    }
}

/// Pipelines the fused move encoder needs (borrowed from the backend).
pub(crate) struct MovePipelines<'a> {
    pub pack_left: &'a ComputePipelineState,
    pub pack_gr_adj: &'a ComputePipelineState,
    pub qr: &'a ComputePipelineState,
    pub install_q_right: &'a ComputePipelineState,
    pub absorb_right: &'a ComputePipelineState,
    pub install_q_left: &'a ComputePipelineState,
    pub absorb_left: &'a ComputePipelineState,
}

/// Size the per-step buffers (must precede encoding — separates the `&mut` sizing
/// phase from the shared-borrow encode phase).
pub(crate) fn size_planned_moves(scratch: &mut MoveScratch, ctx: &MetalContext, plan: &MovePlan) {
    scratch.ensure_steps(ctx, plan.steps.len());
    for (k, s) in plan.steps.iter().enumerate() {
        let st = &mut scratch.steps[k];
        st.a.ensure_capacity(ctx, s.qr_m * s.qr_n);
        st.q.ensure_capacity(ctx, s.qr_m * s.size);
        st.r.ensure_capacity(ctx, s.size * s.qr_n);
        st.keep.ensure_capacity(ctx, s.keep_l * 2 * s.keep_r);
        st.center.ensure_capacity(ctx, s.ctr_l * 2 * s.ctr_r);
    }
}

/// Encode every planned move step onto `cmd` (shared-borrow phase; sizes were set by
/// [`size_planned_moves`]). `sites` is read for the original neighbour tensors.
pub(crate) fn encode_planned_moves(
    cmd: &CommandBufferRef,
    pls: &MovePipelines,
    scratch: &MoveScratch,
    sites: &[SiteTensor],
    plan: &MovePlan,
) {
    for (k, s) in plan.steps.iter().enumerate() {
        let step = &scratch.steps[k];
        // Input site buffer: the original site for k=0, else the prior step's center.
        let in_buf = if k == 0 {
            &sites[s.site].buf
        } else {
            &scratch.steps[k - 1].center
        };
        if plan.dir > 0 {
            // RIGHT: pack grouped-left → QR → install Q → absorb R into the neighbour.
            encode_pack_grouped_left(cmd, pls.pack_left, in_buf, &step.a, s.qr_m, s.qr_n);
            encode_qr(cmd, pls.qr, &step.a, &step.q, &step.r, s.qr_m, s.qr_n);
            encode_install_q_right(
                cmd,
                pls.install_q_right,
                &step.q,
                &step.keep,
                s.qr_m,
                s.size,
            );
            encode_absorb_right(
                cmd,
                pls.absorb_right,
                &step.r,
                &sites[s.nbr].buf,
                &step.center,
                s.p0, // mid
                s.size,
                s.p1, // nr
            );
        } else {
            // LEFT: pack grouped-right-adjoint → QR → install Qᴴ → absorb Rᴴ left.
            encode_pack_gr_adj(cmd, pls.pack_gr_adj, in_buf, &step.a, s.qr_m, s.p0, s.p1);
            encode_qr(cmd, pls.qr, &step.a, &step.q, &step.r, s.qr_m, s.qr_n);
            encode_install_q_left(
                cmd,
                pls.install_q_left,
                &step.q,
                &step.keep,
                s.qr_m,
                s.size,
                s.p1,
            );
            encode_absorb_left(
                cmd,
                pls.absorb_left,
                &step.r,
                &sites[s.nbr].buf,
                &step.center,
                s.p2 * 2, // rows = pl·2
                s.p0,     // mid = li
                s.size,
            );
        }
    }
}

/// One centre-step's resident buffers: the packed QR input `a`, the QR factors
/// `q`/`r`, the finalised canonical site `keep`, and the absorbed neighbour `center`
/// (the next step's input). All pooled (grown on demand).
pub(crate) struct MoveStep {
    pub a: DeviceBuffer<Complex<f32>>,
    pub q: DeviceBuffer<Complex<f32>>,
    pub r: DeviceBuffer<Complex<f32>>,
    pub keep: DeviceBuffer<Complex<f32>>,
    pub center: DeviceBuffer<Complex<f32>>,
}

impl MoveStep {
    fn new(ctx: &MetalContext) -> Self {
        Self {
            a: DeviceBuffer::with_capacity(ctx, 0),
            q: DeviceBuffer::with_capacity(ctx, 0),
            r: DeviceBuffer::with_capacity(ctx, 0),
            keep: DeviceBuffer::with_capacity(ctx, 0),
            center: DeviceBuffer::with_capacity(ctx, 0),
        }
    }
}

/// A growing pool of [`MoveStep`]s, indexed by sweep position. Reused across gates.
pub(crate) struct MoveScratch {
    pub steps: Vec<MoveStep>,
}

impl MoveScratch {
    pub(crate) fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Ensure at least `n` step slots exist.
    pub(crate) fn ensure_steps(&mut self, ctx: &MetalContext, n: usize) {
        while self.steps.len() < n {
            self.steps.push(MoveStep::new(ctx));
        }
    }
}

fn dispatch_grid(pl: &ComputePipelineState, count: u64) -> (MTLSize, MTLSize) {
    let tg = pl.max_total_threads_per_threadgroup().min(count.max(1));
    (MTLSize::new(count.max(1), 1, 1), MTLSize::new(tg, 1, 1))
}

fn set_bytes_at<T>(enc: &metal::ComputeCommandEncoderRef, idx: u64, v: &T) {
    enc.set_bytes(
        idx,
        std::mem::size_of::<T>() as u64,
        v as *const T as *const c_void,
    );
}

/// Encode the column-major pack of a grouped-left view `src` (row-major `rows × mid`)
/// into `dst` (reuses the `pack_theta` kernel, `wide = 0`).
pub(crate) fn encode_pack_grouped_left(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    src: &DeviceBuffer<Complex<f32>>,
    dst: &DeviceBuffer<Complex<f32>>,
    rows: usize,
    mid: usize,
) {
    let meta = PackMeta {
        rows: rows as u32,
        cols: mid as u32,
        wide: 0,
        _pad0: 0,
    };
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pl);
    enc.set_buffer(0, Some(src.metal_buffer()), 0);
    enc.set_buffer(1, Some(dst.metal_buffer()), 0);
    set_bytes_at(enc, 2, &meta);
    let (g, t) = dispatch_grid(pl, (rows * mid) as u64);
    enc.dispatch_threads(g, t);
    enc.end_encoding();
}

/// Encode the grouped-right-adjoint pack of `src` (site row-major `(li,2,ri)`) into
/// `dst` (column-major `2ri × li`) for a left move.
pub(crate) fn encode_pack_gr_adj(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    src: &DeviceBuffer<Complex<f32>>,
    dst: &DeviceBuffer<Complex<f32>>,
    cols: usize,
    li: usize,
    ri: usize,
) {
    let meta = install_meta(cols, li, 0, 0, ri);
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pl);
    enc.set_buffer(0, Some(src.metal_buffer()), 0);
    enc.set_buffer(1, Some(dst.metal_buffer()), 0);
    set_bytes_at(enc, 2, &meta);
    let (g, t) = dispatch_grid(pl, (cols * li) as u64);
    enc.dispatch_threads(g, t);
    enc.end_encoding();
}

/// Encode the Householder QR of `a` (column-major `m × n`) into `q`/`r`.
pub(crate) fn encode_qr(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    a: &DeviceBuffer<Complex<f32>>,
    q: &DeviceBuffer<Complex<f32>>,
    r: &DeviceBuffer<Complex<f32>>,
    m: usize,
    n: usize,
) {
    let meta = QrMeta {
        m: m as u32,
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pl);
    enc.set_buffer(0, Some(a.metal_buffer()), 0);
    enc.set_buffer(1, Some(q.metal_buffer()), 0);
    enc.set_buffer(2, Some(r.metal_buffer()), 0);
    set_bytes_at(enc, 3, &meta);
    let threads = pl.max_total_threads_per_threadgroup().min(256);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
}

fn install_meta(rows: usize, mid: usize, size: usize, nbr: usize, phys: usize) -> QrInstallMeta {
    QrInstallMeta {
        rows: rows as u32,
        mid: mid as u32,
        size: size as u32,
        nbr: nbr as u32,
        phys: phys as u32,
        _f0: 0,
        _f1: 0,
        _f2: 0,
    }
}

fn encode_install(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    b0: &DeviceBuffer<Complex<f32>>,
    b1: &DeviceBuffer<Complex<f32>>,
    meta: &QrInstallMeta,
    count: u64,
) {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pl);
    enc.set_buffer(0, Some(b0.metal_buffer()), 0);
    enc.set_buffer(1, Some(b1.metal_buffer()), 0);
    set_bytes_at(enc, 2, meta);
    let (g, t) = dispatch_grid(pl, count);
    enc.dispatch_threads(g, t);
    enc.end_encoding();
}

fn encode_install3(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    b0: &DeviceBuffer<Complex<f32>>,
    b1: &DeviceBuffer<Complex<f32>>,
    b2: &DeviceBuffer<Complex<f32>>,
    meta: &QrInstallMeta,
    count: u64,
) {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pl);
    enc.set_buffer(0, Some(b0.metal_buffer()), 0);
    enc.set_buffer(1, Some(b1.metal_buffer()), 0);
    enc.set_buffer(2, Some(b2.metal_buffer()), 0);
    set_bytes_at(enc, 3, meta);
    let (g, t) = dispatch_grid(pl, count);
    enc.dispatch_threads(g, t);
    enc.end_encoding();
}

/// RIGHT move: install Q (`rows × size`) → site `(li,2,size)`.
pub(crate) fn encode_install_q_right(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    q: &DeviceBuffer<Complex<f32>>,
    dst: &DeviceBuffer<Complex<f32>>,
    rows: usize,
    size: usize,
) {
    let meta = install_meta(rows, 0, size, 0, 0);
    encode_install(cmd, pl, q, dst, &meta, (rows * size) as u64);
}

/// RIGHT move: absorb R into the neighbour → site `(size,2,nr)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_absorb_right(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    r: &DeviceBuffer<Complex<f32>>,
    nbr: &DeviceBuffer<Complex<f32>>,
    dst: &DeviceBuffer<Complex<f32>>,
    mid: usize,
    size: usize,
    nr: usize,
) {
    let meta = install_meta(0, mid, size, nr, 0);
    encode_install3(cmd, pl, r, nbr, dst, &meta, (size * 2 * nr) as u64);
}

/// LEFT move: install Qᴴ (`q` column-major `cols × size`) → site `(size,2,ri)`.
pub(crate) fn encode_install_q_left(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    q: &DeviceBuffer<Complex<f32>>,
    dst: &DeviceBuffer<Complex<f32>>,
    cols: usize,
    size: usize,
    ri: usize,
) {
    let meta = install_meta(cols, 0, size, 0, ri);
    encode_install(cmd, pl, q, dst, &meta, (size * 2 * ri) as u64);
}

/// LEFT move: absorb Rᴴ into the left neighbour (grouped-left `a_h`, `rows = pl·2`) →
/// site `(pl,2,size)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_absorb_left(
    cmd: &CommandBufferRef,
    pl: &ComputePipelineState,
    r: &DeviceBuffer<Complex<f32>>,
    a_h: &DeviceBuffer<Complex<f32>>,
    dst: &DeviceBuffer<Complex<f32>>,
    rows: usize,
    mid: usize,
    size: usize,
) {
    let meta = install_meta(rows, mid, size, 0, 0);
    encode_install3(cmd, pl, r, a_h, dst, &meta, (rows * size) as u64);
}
