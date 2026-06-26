//! Host driver for the GPU min-sum belief-propagation decoder (Q3-02).

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use aleph_qec::{Syndrome, TannerGraph};

use crate::{CudaContext, DeviceBuffer, Error};

const BP_SRC: &str = include_str!("bp.cu");
const BLOCK: u32 = 128;

/// Device-memory budget for per-shot scratch, in bytes (batches larger than this fit are tiled).
const SCRATCH_BUDGET: usize = 2 << 30;

/// GPU min-sum belief-propagation decoder for a fixed Tanner graph. Compiles the kernel once, holds
/// the device-resident graph, and decodes batches of syndromes one GPU thread per shot —
/// numerically identical to the CPU [`BpDecoder`](aleph_qec::BpDecoder).
pub struct CudaBp {
    ctx: CudaContext,
    f_decode: CudaFunction,
    _module: Arc<CudaModule>,

    n_vars: u32,
    n_edges: u32,
    n_checks: u32,
    num_observables: usize,
    max_iter: u32,
    alpha: f64,
    words_per_shot: u32,

    lambda: DeviceBuffer<f64>,
    obs: DeviceBuffer<u64>,
    var_off: DeviceBuffer<u32>,
    edge_check: DeviceBuffer<u32>,
    edge_var: DeviceBuffer<u32>,
    check_off: DeviceBuffer<u32>,
    check_edges: DeviceBuffer<u32>,

    scratch: RefCell<Scratch>,
}

#[derive(Default)]
struct Scratch {
    capacity: usize,
    syn_words: Option<DeviceBuffer<u32>>,
    out_mask: Option<DeviceBuffer<u64>>,
    m_vc: Option<DeviceBuffer<f64>>,
    e_cv: Option<DeviceBuffer<f64>>,
    ehat: Option<DeviceBuffer<u8>>,
    s: Option<DeviceBuffer<u8>>,
}

impl CudaBp {
    /// Compile the kernel on device 0 and upload `tanner` (the CPU decoder's own flattened graph, so
    /// the schedules are identical). Returns [`Error::NoDevice`] on a GPU-less host.
    pub fn new(tanner: &TannerGraph) -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        let ptx = compile_ptx(BP_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        let f_decode = module.load_function("bp_decode")?;

        let lambda = DeviceBuffer::from_slice(&ctx, tanner.lambda)?;
        let obs = DeviceBuffer::from_slice(&ctx, tanner.obs)?;
        let var_off = DeviceBuffer::from_slice(&ctx, tanner.var_off)?;
        let edge_check = DeviceBuffer::from_slice(&ctx, tanner.edge_check)?;
        let edge_var = DeviceBuffer::from_slice(&ctx, tanner.edge_var)?;
        let check_off = DeviceBuffer::from_slice(&ctx, tanner.check_off)?;
        let check_edges = DeviceBuffer::from_slice(&ctx, tanner.check_edges)?;

        let words_per_shot = (tanner.num_detectors as u32).div_ceil(32).max(1);

        Ok(Self {
            ctx,
            f_decode,
            _module: module,
            n_vars: tanner.n_vars as u32,
            n_edges: tanner.n_edges as u32,
            n_checks: tanner.num_detectors as u32,
            num_observables: tanner.num_observables,
            max_iter: tanner.max_iter,
            alpha: tanner.alpha,
            words_per_shot,
            lambda,
            obs,
            var_off,
            edge_check,
            edge_var,
            check_off,
            check_edges,
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
        let nd = self.n_checks;
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

    /// Decode a batch of syndromes, returning one observable-flip bitmask per shot.
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
        let per_shot = self.per_shot_bytes();
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

    fn per_shot_bytes(&self) -> usize {
        let e = self.n_edges as usize;
        let v = self.n_vars as usize;
        let c = self.n_checks as usize;
        // m_vc + e_cv (f64 ×2 per edge) + ehat (u8/var) + s (u8/check) + out_mask + syn_words.
        e * 16 + v + c + 8 + self.words_per_shot as usize * 4
    }

    fn decode_tile(&self, packed: &[u32], cur: usize, out: &mut [u64]) -> Result<(), Error> {
        self.ensure_scratch(cur)?;
        let mut sc = self.scratch.borrow_mut();
        sc.syn_words.as_mut().unwrap().write(&self.ctx, packed)?;

        let cur_u32 = cur as u32;
        let cfg = launch_config(cur as u64);
        let stream = self.ctx.stream();

        let Scratch {
            syn_words,
            out_mask,
            m_vc,
            e_cv,
            ehat,
            s,
            ..
        } = &mut *sc;

        // SAFETY: arg order/types match `bp_decode`'s C signature exactly (see bp.cu). The grid
        // covers `cur` shots with an in-kernel `shot >= n_shots` guard; each thread touches only its
        // own `shot * stride` scratch slice and `out_mask[shot]`, so writes never alias. Graph
        // buffers are read-only; scratch capacity ≥ cur·stride is ensured above.
        unsafe {
            stream
                .launch_builder(&self.f_decode)
                .arg(self.lambda.slice())
                .arg(self.obs.slice())
                .arg(self.var_off.slice())
                .arg(self.edge_check.slice())
                .arg(self.edge_var.slice())
                .arg(self.check_off.slice())
                .arg(self.check_edges.slice())
                .arg(&self.n_vars)
                .arg(&self.n_edges)
                .arg(&self.n_checks)
                .arg(&self.max_iter)
                .arg(&self.alpha)
                .arg(syn_words.as_ref().unwrap().slice())
                .arg(&self.words_per_shot)
                .arg(&cur_u32)
                .arg(out_mask.as_mut().unwrap().slice_mut())
                .arg(m_vc.as_mut().unwrap().slice_mut())
                .arg(e_cv.as_mut().unwrap().slice_mut())
                .arg(ehat.as_mut().unwrap().slice_mut())
                .arg(s.as_mut().unwrap().slice_mut())
                .launch(cfg)?;
        }

        let masks = out_mask.as_ref().unwrap().to_vec(&self.ctx)?;
        out.copy_from_slice(&masks[..cur]);
        Ok(())
    }

    fn ensure_scratch(&self, cur: usize) -> Result<(), Error> {
        let mut sc = self.scratch.borrow_mut();
        if sc.capacity >= cur && sc.m_vc.is_some() {
            return Ok(());
        }
        let e = self.n_edges as usize;
        let v = self.n_vars as usize;
        let c = self.n_checks as usize;
        let wps = self.words_per_shot as usize;
        sc.syn_words = Some(DeviceBuffer::<u32>::zeros(&self.ctx, cur * wps)?);
        sc.out_mask = Some(DeviceBuffer::<u64>::zeros(&self.ctx, cur)?);
        sc.m_vc = Some(DeviceBuffer::<f64>::zeros(&self.ctx, cur * e)?);
        sc.e_cv = Some(DeviceBuffer::<f64>::zeros(&self.ctx, cur * e)?);
        sc.ehat = Some(DeviceBuffer::<u8>::zeros(&self.ctx, cur * v)?);
        sc.s = Some(DeviceBuffer::<u8>::zeros(&self.ctx, cur * c)?);
        sc.capacity = cur;
        Ok(())
    }

    /// Block until queued kernels finish (for timing).
    pub fn synchronize(&self) -> Result<(), Error> {
        self.ctx.synchronize()
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
