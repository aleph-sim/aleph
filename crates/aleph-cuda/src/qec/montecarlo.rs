//! Host driver for the Q3-03 on-device Monte-Carlo threshold harness.
//!
//! [`CudaThreshold`] fuses three device-resident stages — DEM-Bernoulli noisy-syndrome generation
//! (`montecarlo.cu`), the GPU Union-Find decode (`uf.cu`'s `uf_decode`, reused unmodified), and
//! logical-error scoring (`montecarlo.cu`) — so a whole threshold cell runs **without leaving the
//! GPU**: syndromes are sampled, decoded and scored on the device, and only the final logical-error
//! count is copied back. This removes the per-shot PCIe round-trip the standalone GPU decoders
//! (Q3-01/Q3-02) pay, which is the point of the ticket.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use aleph_qec::{DetectorErrorModel, UnionFindDecoder};

use crate::{CudaContext, DeviceBuffer, Error};

const UF_SRC: &str = include_str!("uf.cu");
const MC_SRC: &str = include_str!("montecarlo.cu");
const BLOCK: u32 = 128;

/// Device-memory budget for the per-shot working set (UF scratch + I/O), in bytes; larger sweeps
/// are tiled into chunks accumulating into one device counter.
const WORKSET_BUDGET: usize = 2 << 30;

/// Result of one Monte-Carlo cell.
#[derive(Clone, Copy, Debug)]
pub struct CellResult {
    /// Shots run.
    pub shots: u64,
    /// Shots whose decode mispredicted the logical observable(s).
    pub logical_errors: u64,
}

impl CellResult {
    /// Logical-error rate.
    pub fn rate(&self) -> f64 {
        if self.shots == 0 {
            0.0
        } else {
            self.logical_errors as f64 / self.shots as f64
        }
    }
    /// 95% confidence half-width (normal approximation).
    pub fn ci95(&self) -> f64 {
        if self.shots == 0 {
            return 0.0;
        }
        let n = self.shots as f64;
        let r = self.rate();
        1.96 * (r * (1.0 - r) / n).sqrt()
    }
}

/// On-device Monte-Carlo decoder-accuracy harness for a fixed [`DetectorErrorModel`], using the GPU
/// Union-Find decoder. Sample → decode → score, all on the GPU.
pub struct CudaThreshold {
    ctx: CudaContext,
    f_sample: CudaFunction,
    f_decode: CudaFunction,
    f_reduce: CudaFunction,
    _mc_module: Arc<CudaModule>,
    _uf_module: Arc<CudaModule>,

    // UF graph scalars + buffers (same layout as `CudaUnionFind`).
    n_nodes: u32,
    n_edges: u32,
    n_detectors: u32,
    weighted: u32,
    words_per_shot: u32,
    low_mask: u64,
    adj_off: DeviceBuffer<u32>,
    adj_edges: DeviceBuffer<u32>,
    edge_a: DeviceBuffer<u32>,
    edge_b: DeviceBuffer<u32>,
    edge_obs: DeviceBuffer<u64>,
    edge_len: DeviceBuffer<u32>,

    // DEM sampling arrays.
    n_mech: u32,
    mech_prob: DeviceBuffer<f64>,
    det_off: DeviceBuffer<u32>,
    det_idx: DeviceBuffer<u32>,
    mech_obs: DeviceBuffer<u64>,

    scratch: RefCell<Scratch>,
}

#[derive(Default)]
struct Scratch {
    capacity: usize,
    syn_words: Option<DeviceBuffer<u32>>,
    truth: Option<DeviceBuffer<u64>>,
    out_mask: Option<DeviceBuffer<u64>>,
    parent: Option<DeviceBuffer<u32>>,
    sz: Option<DeviceBuffer<u32>>,
    acc: Option<DeviceBuffer<u32>>,
    parity: Option<DeviceBuffer<u8>>,
    btouch: Option<DeviceBuffer<u8>>,
    grown: Option<DeviceBuffer<u8>>,
    syn_bit: Option<DeviceBuffer<u8>>,
    visited: Option<DeviceBuffer<u8>>,
    parent_edge: Option<DeviceBuffer<u32>>,
    parent_node: Option<DeviceBuffer<u32>>,
    order: Option<DeviceBuffer<u32>>,
}

impl CudaThreshold {
    /// Compile the kernels on device 0 and upload `dem` (both as a UF decoder graph and as DEM
    /// sampling arrays). Returns [`Error::NoDevice`] on a GPU-less host, or propagates a non-graphlike
    /// DEM error from the UF build.
    pub fn new(dem: &DetectorErrorModel) -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        let mc_ptx = compile_ptx(MC_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let uf_ptx = compile_ptx(UF_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let mc_module = ctx.raw().load_module(mc_ptx)?;
        let uf_module = ctx.raw().load_module(uf_ptx)?;
        let f_sample = mc_module.load_function("dem_sample")?;
        let f_reduce = mc_module.load_function("mispredict_reduce")?;
        let f_decode = uf_module.load_function("uf_decode")?;

        // UF graph (the CPU decoder's own arrays — identical to `CudaUnionFind`).
        let uf = UnionFindDecoder::new(dem).map_err(|e| Error::Compile(e.to_string()))?;
        let g = uf.graph();
        let adj_off = DeviceBuffer::from_slice(&ctx, g.adj_off)?;
        let adj_edges = DeviceBuffer::from_slice(&ctx, g.adj_edges)?;
        let edge_a = DeviceBuffer::from_slice(&ctx, g.edge_a)?;
        let edge_b = DeviceBuffer::from_slice(&ctx, g.edge_b)?;
        let edge_obs = DeviceBuffer::from_slice(&ctx, g.edge_obs)?;
        let edge_len = DeviceBuffer::from_slice(&ctx, g.edge_len)?;
        let words_per_shot = (g.num_detectors as u32).div_ceil(32).max(1);
        let low_mask = if g.num_observables >= 64 {
            u64::MAX
        } else {
            (1u64 << g.num_observables) - 1
        };

        // DEM sampling arrays: per-mechanism probability, detector CSR, observable mask.
        let n_mech = dem.errors.len();
        let mut mech_prob = vec![0.0f64; n_mech];
        let mut det_off = vec![0u32; n_mech + 1];
        let mut det_idx: Vec<u32> = Vec::new();
        let mut mech_obs = vec![0u64; n_mech];
        for (m, e) in dem.errors.iter().enumerate() {
            mech_prob[m] = e.prob;
            for &d in &e.dets {
                if (d as usize) < dem.detectors {
                    det_idx.push(d);
                }
            }
            det_off[m + 1] = det_idx.len() as u32;
            mech_obs[m] = e
                .obs
                .iter()
                .filter(|&&o| o < 64)
                .fold(0u64, |a, &o| a | (1u64 << o));
        }

        Ok(Self {
            f_sample,
            f_decode,
            f_reduce,
            _mc_module: mc_module,
            _uf_module: uf_module,
            n_nodes: g.n_nodes as u32,
            n_edges: g.edge_a.len() as u32,
            n_detectors: g.num_detectors as u32,
            weighted: g.weighted as u32,
            words_per_shot,
            low_mask,
            adj_off,
            adj_edges,
            edge_a,
            edge_b,
            edge_obs,
            edge_len,
            n_mech: n_mech as u32,
            mech_prob: DeviceBuffer::from_slice(&ctx, &mech_prob)?,
            det_off: DeviceBuffer::from_slice(&ctx, &det_off)?,
            det_idx: DeviceBuffer::from_slice(&ctx, &det_idx)?,
            mech_obs: DeviceBuffer::from_slice(&ctx, &mech_obs)?,
            ctx,
            scratch: RefCell::new(Scratch::default()),
        })
    }

    /// Run `shots` Monte-Carlo trials (sample + decode + score on the GPU) and return the cell
    /// result. Only the logical-error count crosses the PCIe bus.
    pub fn run(&self, shots: u64, seed: u64) -> Result<CellResult, Error> {
        if shots == 0 {
            return Ok(CellResult {
                shots: 0,
                logical_errors: 0,
            });
        }
        let per_shot = self.per_shot_bytes();
        let tile = ((WORKSET_BUDGET / per_shot.max(1)) as u64).clamp(1, shots);

        let mut counter = DeviceBuffer::<u64>::zeros(&self.ctx, 1)?;
        let mut base = 0u64;
        while base < shots {
            let cur = (shots - base).min(tile);
            self.run_chunk(cur as usize, seed, base, &mut counter)?;
            base += cur;
        }
        let errors = counter.to_vec(&self.ctx)?[0];
        Ok(CellResult {
            shots,
            logical_errors: errors,
        })
    }

    /// Block until queued kernels finish (for timing).
    pub fn synchronize(&self) -> Result<(), Error> {
        self.ctx.synchronize()
    }

    fn per_shot_bytes(&self) -> usize {
        let n = self.n_nodes as usize;
        let e = self.n_edges as usize;
        // UF scratch (see CudaUnionFind) + syn_words + truth(u64) + out_mask(u64).
        n * (5 * 4 + 4) + e * (4 + 1) + self.words_per_shot as usize * 4 + 16
    }

    fn run_chunk(
        &self,
        cur: usize,
        seed: u64,
        shot_base: u64,
        counter: &mut DeviceBuffer<u64>,
    ) -> Result<(), Error> {
        self.ensure_scratch(cur)?;
        let mut sc = self.scratch.borrow_mut();
        let cur_u32 = cur as u32;
        let cfg = launch_config(cur as u64);
        let stream = self.ctx.stream();

        let Scratch {
            syn_words,
            truth,
            out_mask,
            parent,
            sz,
            acc,
            parity,
            btouch,
            grown,
            syn_bit,
            visited,
            parent_edge,
            parent_node,
            order,
            ..
        } = &mut *sc;

        // Stage 1 — sample noisy syndromes + truth on the device.
        // SAFETY: args match `dem_sample`'s C signature; grid covers `cur` with an in-kernel guard;
        // each thread writes only its own `li`-strided syn_words slice and `truth[li]`.
        unsafe {
            stream
                .launch_builder(&self.f_sample)
                .arg(self.mech_prob.slice())
                .arg(self.det_off.slice())
                .arg(self.det_idx.slice())
                .arg(self.mech_obs.slice())
                .arg(&self.n_mech)
                .arg(&self.words_per_shot)
                .arg(&cur_u32)
                .arg(&seed)
                .arg(&shot_base)
                .arg(syn_words.as_mut().unwrap().slice_mut())
                .arg(truth.as_mut().unwrap().slice_mut())
                .launch(cfg)?;
        }

        // Stage 2 — decode the device-resident syndromes with `uf_decode` (reused unmodified).
        // SAFETY: args match `uf_decode`'s C signature exactly (see uf.cu / CudaUnionFind); grid
        // covers `cur`; per-thread scratch slices never alias; graph buffers are read-only.
        unsafe {
            stream
                .launch_builder(&self.f_decode)
                .arg(self.adj_off.slice())
                .arg(self.adj_edges.slice())
                .arg(self.edge_a.slice())
                .arg(self.edge_b.slice())
                .arg(self.edge_obs.slice())
                .arg(self.edge_len.slice())
                .arg(&self.n_nodes)
                .arg(&self.n_edges)
                .arg(&self.n_detectors)
                .arg(&self.weighted)
                .arg(syn_words.as_ref().unwrap().slice())
                .arg(&self.words_per_shot)
                .arg(&cur_u32)
                .arg(out_mask.as_mut().unwrap().slice_mut())
                .arg(parent.as_mut().unwrap().slice_mut())
                .arg(sz.as_mut().unwrap().slice_mut())
                .arg(acc.as_mut().unwrap().slice_mut())
                .arg(parity.as_mut().unwrap().slice_mut())
                .arg(btouch.as_mut().unwrap().slice_mut())
                .arg(grown.as_mut().unwrap().slice_mut())
                .arg(syn_bit.as_mut().unwrap().slice_mut())
                .arg(visited.as_mut().unwrap().slice_mut())
                .arg(parent_edge.as_mut().unwrap().slice_mut())
                .arg(parent_node.as_mut().unwrap().slice_mut())
                .arg(order.as_mut().unwrap().slice_mut())
                .launch(cfg)?;
        }

        // Stage 3 — score: count mispredictions into the device counter.
        // SAFETY: args match `mispredict_reduce`; grid covers `cur`; atomicAdd into a single counter.
        unsafe {
            stream
                .launch_builder(&self.f_reduce)
                .arg(out_mask.as_ref().unwrap().slice())
                .arg(truth.as_ref().unwrap().slice())
                .arg(&self.low_mask)
                .arg(&cur_u32)
                .arg(counter.slice_mut())
                .launch(cfg)?;
        }
        Ok(())
    }

    fn ensure_scratch(&self, cur: usize) -> Result<(), Error> {
        let mut sc = self.scratch.borrow_mut();
        if sc.capacity >= cur && sc.parent.is_some() {
            return Ok(());
        }
        let n = self.n_nodes as usize;
        let e = self.n_edges as usize;
        let wps = self.words_per_shot as usize;
        sc.syn_words = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * wps)?);
        sc.truth = Some(DeviceBuffer::<u64>::zeros(&self.ctx, cur)?);
        sc.out_mask = Some(DeviceBuffer::<u64>::zeros(&self.ctx, cur)?);
        sc.parent = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * n)?);
        sc.sz = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * n)?);
        sc.acc = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * e)?);
        sc.parity = Some(DeviceBuffer::<u8>::zeros(&self.ctx, cur * n)?);
        sc.btouch = Some(DeviceBuffer::<u8>::zeros(&self.ctx, cur * n)?);
        sc.grown = Some(DeviceBuffer::<u8>::zeros(&self.ctx, cur * e)?);
        sc.syn_bit = Some(DeviceBuffer::<u8>::zeros(&self.ctx, cur * n)?);
        sc.visited = Some(DeviceBuffer::<u8>::zeros(&self.ctx, cur * n)?);
        sc.parent_edge = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * n)?);
        sc.parent_node = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * n)?);
        sc.order = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * n)?);
        sc.capacity = cur;
        Ok(())
    }
}

fn launch_config(n_threads: u64) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
