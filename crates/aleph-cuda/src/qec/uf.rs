//! Host driver for the GPU Union-Find decoder (Q3-01).

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use aleph_qec::{DecoderGraph, Syndrome};

use crate::{CudaContext, DeviceBuffer, Error};

const UF_SRC: &str = include_str!("uf.cu");
const BLOCK: u32 = 128;

/// Device-memory budget for per-shot scratch, in bytes. A batch larger than this fits-in-budget is
/// decoded in tiles. 2 GiB leaves ample room on the 20 GiB card for the (tiny) graph + I/O buffers.
const SCRATCH_BUDGET: usize = 2 << 30;

/// GPU Union-Find decoder for a fixed matching graph. Compiles the kernel once, holds the
/// device-resident graph, and decodes batches of syndromes one GPU thread per shot —
/// bit-identical to the CPU [`UnionFindDecoder`](aleph_qec::UnionFindDecoder).
pub struct CudaUnionFind {
    ctx: CudaContext,
    f_decode: CudaFunction,
    _module: Arc<CudaModule>,

    // Graph scalars.
    n_nodes: u32,
    n_edges: u32,
    n_detectors: u32,
    num_observables: usize,
    weighted: u32,
    words_per_shot: u32,

    // Device-resident, read-only matching graph.
    adj_off: DeviceBuffer<u32>,
    adj_edges: DeviceBuffer<u32>,
    edge_a: DeviceBuffer<u32>,
    edge_b: DeviceBuffer<u32>,
    edge_obs: DeviceBuffer<u64>,
    edge_len: DeviceBuffer<u32>,

    // Reusable per-shot scratch + I/O, grown on demand (interior-mutable so decode is `&self`).
    scratch: RefCell<Scratch>,
}

/// Reusable device buffers sized to the largest tile seen so far.
#[derive(Default)]
struct Scratch {
    capacity: usize,
    syn_words: Option<DeviceBuffer<u32>>,
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

impl CudaUnionFind {
    /// Compile the kernel on device 0 and upload `graph`. The graph arrays are the CPU decoder's
    /// own ([`UnionFindDecoder::graph`](aleph_qec::UnionFindDecoder::graph)), so the GPU decodes the
    /// identical layout. Returns [`Error::NoDevice`] on a GPU-less host.
    pub fn new(graph: &DecoderGraph) -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        let ptx = compile_ptx(UF_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        let f_decode = module.load_function("uf_decode")?;

        let adj_off = DeviceBuffer::from_slice(&ctx, graph.adj_off)?;
        let adj_edges = DeviceBuffer::from_slice(&ctx, graph.adj_edges)?;
        let edge_a = DeviceBuffer::from_slice(&ctx, graph.edge_a)?;
        let edge_b = DeviceBuffer::from_slice(&ctx, graph.edge_b)?;
        let edge_obs = DeviceBuffer::from_slice(&ctx, graph.edge_obs)?;
        let edge_len = DeviceBuffer::from_slice(&ctx, graph.edge_len)?;

        let words_per_shot = (graph.num_detectors as u32).div_ceil(32).max(1);

        Ok(Self {
            ctx,
            f_decode,
            _module: module,
            n_nodes: graph.n_nodes as u32,
            n_edges: graph.edge_a.len() as u32,
            n_detectors: graph.num_detectors as u32,
            num_observables: graph.num_observables,
            weighted: graph.weighted as u32,
            words_per_shot,
            adj_off,
            adj_edges,
            edge_a,
            edge_b,
            edge_obs,
            edge_len,
            scratch: RefCell::new(Scratch::default()),
        })
    }

    /// Number of logical observables (width of each correction).
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Pack a batch of syndromes into the per-shot detector-bit words the kernel reads.
    pub fn pack(&self, syndromes: &[Syndrome]) -> Vec<u32> {
        let wps = self.words_per_shot as usize;
        let nd = self.n_detectors;
        let mut words = vec![0u32; syndromes.len() * wps];
        for (s, syn) in syndromes.iter().enumerate() {
            let base = s * wps;
            for &d in &syn.fired {
                if d < nd {
                    words[base + (d >> 5) as usize] |= 1u32 << (d & 31);
                }
            }
        }
        words
    }

    /// Decode a batch of syndromes, returning one observable-flip bitmask per shot (bit `o` set ⇔
    /// observable `o` flips). Use [`mask_to_flips`] to expand a mask to a `Vec<bool>`.
    pub fn decode(&self, syndromes: &[Syndrome]) -> Result<Vec<u64>, Error> {
        let packed = self.pack(syndromes);
        self.decode_packed(&packed, syndromes.len())
    }

    /// Decode `n_shots` syndromes given pre-packed detector words (`words_per_shot` u32 per shot).
    pub fn decode_packed(&self, packed: &[u32], n_shots: usize) -> Result<Vec<u64>, Error> {
        let mut out = vec![0u64; n_shots];
        if n_shots == 0 {
            return Ok(out);
        }
        let per_shot = self.per_shot_scratch_bytes();
        let tile = (SCRATCH_BUDGET / per_shot.max(1)).clamp(1, n_shots);
        let wps = self.words_per_shot as usize;

        for start in (0..n_shots).step_by(tile) {
            let cur = (n_shots - start).min(tile);
            self.decode_tile(
                &packed[start * wps..(start + cur) * wps],
                cur,
                &mut out[start..start + cur],
            )?;
        }
        Ok(out)
    }

    /// Per-shot scratch footprint in bytes (node + edge arrays).
    fn per_shot_scratch_bytes(&self) -> usize {
        let n = self.n_nodes as usize;
        let e = self.n_edges as usize;
        // node: parent,sz,parent_edge,parent_node,order (u32 ×5) + parity,btouch,syn,visited (u8 ×4)
        // edge: acc (u32) + grown (u8); plus out_mask (u64) and syn_words per shot.
        n * (5 * 4 + 4) + e * (4 + 1) + 8 + wps_bytes(self.words_per_shot)
    }

    /// Decode one tile (`cur` shots) whose packed words are `packed`, writing masks into `out`.
    fn decode_tile(&self, packed: &[u32], cur: usize, out: &mut [u64]) -> Result<(), Error> {
        self.ensure_scratch(cur)?;
        let mut sc = self.scratch.borrow_mut();
        // Upload this tile's syndromes (reusing the buffer when large enough).
        sc.syn_words.as_mut().unwrap().write(&self.ctx, packed)?;

        let cur_u32 = cur as u32;
        let cfg = launch_config(cur as u64);
        let stream = self.ctx.stream();

        // Destructure so we can borrow several scratch buffers mutably at once.
        let Scratch {
            syn_words,
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

        // SAFETY: arg order/types match `uf_decode`'s C signature exactly (see uf.cu). The grid
        // covers `cur` shots with an in-kernel `shot >= n_shots` guard; each thread touches only its
        // own `shot * stride` scratch slice and `out_mask[shot]`, so writes never alias across
        // threads. Graph buffers are read-only. Scratch capacity ≥ cur·stride is ensured above.
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

        let masks = out_mask.as_ref().unwrap().to_vec(&self.ctx)?;
        out.copy_from_slice(&masks[..cur]);
        Ok(())
    }

    /// Ensure all scratch buffers hold at least `cur` shots' worth of elements.
    fn ensure_scratch(&self, cur: usize) -> Result<(), Error> {
        let mut sc = self.scratch.borrow_mut();
        if sc.capacity >= cur && sc.parent.is_some() {
            return Ok(());
        }
        let n = self.n_nodes as usize;
        let e = self.n_edges as usize;
        let wps = self.words_per_shot as usize;
        sc.syn_words = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * wps)?);
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

    /// Block until queued kernels finish (for timing).
    pub fn synchronize(&self) -> Result<(), Error> {
        self.ctx.synchronize()
    }
}

fn wps_bytes(words_per_shot: u32) -> usize {
    words_per_shot as usize * 4
}

/// Expand an observable-flip bitmask to `num_observables` booleans (bit `o` → `flips[o]`).
pub fn mask_to_flips(mask: u64, num_observables: usize) -> Vec<bool> {
    (0..num_observables).map(|o| (mask >> o) & 1 == 1).collect()
}

fn launch_config(n_threads: u64) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
