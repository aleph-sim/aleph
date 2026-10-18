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

/// Scratch placement only; all variants use the identical per-shot FP64 schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpScratchLayout {
    /// Original `shot * elements + element` reference, with no cross-shot contiguity.
    ShotMajor,
    /// `element * tile_shots + shot`, contiguous across adjacent GPU lanes.
    EdgeMajor,
    /// Edge-major inside 32-shot tiles; pads the final tile and budgets that padding.
    WarpTiled,
}

/// GPU min-sum belief-propagation decoder for a fixed Tanner graph. Compiles the kernel once, holds
/// the device-resident graph, and by default decodes with one GPU thread per shot —
/// numerically identical to the CPU [`BpDecoder`](aleph_qec::BpDecoder).
/// [`Self::warp_cooperative`] is an explicit alternative parallel node schedule.
pub struct CudaBp {
    ctx: CudaContext,
    layout: BpScratchLayout,
    cooperative: bool,
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
        Self::with_scratch_layout(tanner, BpScratchLayout::ShotMajor)
    }

    /// Compile with an explicit scratch layout for same-decoder A/B comparisons.
    /// Arithmetic order, syndrome packing and observable outputs are identical.
    pub fn with_scratch_layout(
        tanner: &TannerGraph,
        layout: BpScratchLayout,
    ) -> Result<Self, Error> {
        Self::build(tanner, layout, false)
    }

    /// Experimental one-warp-per-shot BP. Nodes execute in parallel; each node keeps
    /// its serial FP64 reduction order. This is opt-in and uses unpadded shot-major scratch.
    /// CSR bounds and unique edge ownership are validated before device access: malformed
    /// graphs return a memory-capacity error instead of allowing concurrent conflicting writes.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), aleph_cuda::Error> {
    /// let dem = aleph_qec::BBCode::gross().code_capacity_dem(0.001);
    /// let cpu = aleph_qec::BpDecoder::new(&dem);
    /// let gpu = aleph_cuda::CudaBp::warp_cooperative(&cpu.tanner())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn warp_cooperative(tanner: &TannerGraph) -> Result<Self, Error> {
        validate_cooperative_graph(tanner)?;
        Self::build(tanner, BpScratchLayout::ShotMajor, true)
    }

    fn build(
        tanner: &TannerGraph,
        layout: BpScratchLayout,
        cooperative: bool,
    ) -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        let mut source = match layout {
            BpScratchLayout::ShotMajor => BP_SRC.to_owned(),
            BpScratchLayout::EdgeMajor => format!("#define BP_EDGE_MAJOR\n{BP_SRC}"),
            BpScratchLayout::WarpTiled => format!("#define BP_WARP_TILED\n{BP_SRC}"),
        };
        if cooperative {
            source.insert_str(0, "#define BP_COOPERATIVE\n");
        }
        let ptx = compile_ptx(source).map_err(|e| Error::Compile(e.to_string()))?;
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
            layout,
            cooperative,
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

    /// Scratch layout selected when the decoder was compiled.
    pub fn scratch_layout(&self) -> BpScratchLayout {
        self.layout
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
    /// As before, extra trailing words are ignored; insufficient input is rejected before upload.
    pub fn decode_packed(&self, packed: &[u32], n_shots: usize) -> Result<Vec<u64>, Error> {
        let expected = n_shots
            .checked_mul(self.words_per_shot as usize)
            .ok_or_else(|| Error::Memory("BP packed length overflow".into()))?;
        if packed.len() < expected {
            return Err(Error::Memory(
                "BP packed detector word count mismatch".into(),
            ));
        }
        let mut out = vec![0u64; n_shots];
        if n_shots == 0 {
            return Ok(out);
        }
        let per_shot = self.per_shot_bytes();
        let tile = tile_capacity(per_shot, n_shots, self.layout)?;
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

    /// Device scratch payload per shot, including packed input and returned mask.
    /// Shot/edge-major are unpadded; warp tiles round retained capacity up to 32 shots.
    pub fn per_shot_bytes(&self) -> usize {
        let e = self.n_edges as usize;
        let v = self.n_vars as usize;
        let c = self.n_checks as usize;
        // m_vc + e_cv (f64 ×2 per edge) + ehat (u8/var) + s (u8/check) + out_mask + syn_words.
        e * 16 + v + c + 8 + self.words_per_shot as usize * 4
    }

    /// Allocated device scratch payload, including retained capacity and any warp padding.
    pub fn retained_scratch_bytes(&self) -> usize {
        self.scratch.borrow().capacity * self.per_shot_bytes()
    }

    fn decode_tile(&self, packed: &[u32], cur: usize, out: &mut [u64]) -> Result<(), Error> {
        self.ensure_scratch(cur)?;
        let mut sc = self.scratch.borrow_mut();
        sc.syn_words.as_mut().unwrap().write(&self.ctx, packed)?;

        let cur_u32 = cur as u32;
        let cfg = launch_config(cur as u64 * if self.cooperative { 32 } else { 1 });
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
        // covers `cur` shots with an in-kernel `shot >= n_shots` guard. Serial threads touch only
        // their own shot-major slice, edge-major column or warp-tile lane and output mask.
        // Serial layouts have disjoint per-shot writes. In cooperative mode a full warp owns
        // each shot; independent node updates write disjoint edges and __syncwarp fences
        // each phase. Grid threads = cur*32 rounded to full blocks; the shot guard is warp-uniform.
        // Graph buffers are read-only and scratch covers every shot (including padding).
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
        let cur = if self.layout == BpScratchLayout::WarpTiled {
            cur.div_ceil(32) * 32
        } else {
            cur
        };
        // Release retained buffers before growing: old+new scratch must not double the budget.
        *sc = Scratch::default();
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

// A check owns each e_cv edge; a variable owns each m_vc edge. This is what makes
// parallel node updates race-free. Public TannerGraph fields must not bypass that proof.
fn validate_cooperative_graph(t: &TannerGraph) -> Result<(), Error> {
    let invalid = || Error::Memory("invalid cooperative BP graph bounds/edge ownership".into());
    if t.n_vars > u32::MAX as usize
        || t.n_edges > u32::MAX as usize
        || t.num_detectors > u32::MAX as usize
        || t.num_observables > 64
        || t.lambda.len() != t.n_vars
        || t.obs.len() != t.n_vars
        || t.var_off.len() != t.n_vars + 1
        || t.check_off.len() != t.num_detectors + 1
        || t.edge_var.len() != t.n_edges
        || t.edge_check.len() != t.n_edges
        || t.check_edges.len() != t.n_edges
    {
        return Err(invalid());
    }
    let bytes =
        t.n_edges * 16 + t.n_vars + t.num_detectors + 8 + t.num_detectors.div_ceil(32).max(1) * 4;
    if bytes > SCRATCH_BUDGET {
        return Err(invalid());
    }
    for offsets in [t.var_off, t.check_off] {
        if offsets.first() != Some(&0)
            || offsets.last().copied() != Some(t.n_edges as u32)
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(invalid());
        }
    }
    for (v, range) in t.var_off.windows(2).enumerate() {
        if t.edge_var[range[0] as usize..range[1] as usize]
            .iter()
            .any(|&owner| owner as usize != v)
        {
            return Err(invalid());
        }
    }
    let mut seen = vec![false; t.n_edges];
    for (c, range) in t.check_off.windows(2).enumerate() {
        for &edge in &t.check_edges[range[0] as usize..range[1] as usize] {
            let e = edge as usize;
            if e >= t.n_edges || seen[e] || t.edge_check[e] as usize != c {
                return Err(invalid());
            }
            seen[e] = true;
        }
    }
    Ok(())
}

fn tile_capacity(per_shot: usize, shots: usize, layout: BpScratchLayout) -> Result<usize, Error> {
    if per_shot == 0 || per_shot > SCRATCH_BUDGET {
        return Err(Error::Memory("one BP shot exceeds scratch budget".into()));
    }
    let mut capacity = (SCRATCH_BUDGET / per_shot).min(u32::MAX as usize);
    if layout == BpScratchLayout::WarpTiled {
        capacity = capacity / 32 * 32;
    }
    if capacity == 0 {
        return Err(Error::Memory(
            "one padded BP tile exceeds scratch budget".into(),
        ));
    }
    Ok(capacity.min(shots))
}

fn launch_config(n_threads: u64) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_qec::{BpDecoder, DemError, DetectorErrorModel, Syndrome};

    #[test]
    fn scratch_budget_has_no_padding_or_oversized_single_tile() {
        assert_eq!(
            tile_capacity(123, 0, BpScratchLayout::ShotMajor).unwrap(),
            0
        );
        assert_eq!(
            tile_capacity(123, 33, BpScratchLayout::ShotMajor).unwrap(),
            33
        );
        assert_eq!(
            tile_capacity(123, usize::MAX, BpScratchLayout::ShotMajor).unwrap(),
            SCRATCH_BUDGET / 123
        );
        assert_eq!(
            tile_capacity(SCRATCH_BUDGET, 2, BpScratchLayout::ShotMajor).unwrap(),
            1
        );
        assert!(tile_capacity(SCRATCH_BUDGET + 1, 1, BpScratchLayout::ShotMajor).is_err());
        assert!(tile_capacity(0, 1, BpScratchLayout::ShotMajor).is_err());
        assert_eq!(
            tile_capacity(123, 33, BpScratchLayout::WarpTiled).unwrap(),
            33
        );
        let max = tile_capacity(123, usize::MAX, BpScratchLayout::WarpTiled).unwrap();
        assert_eq!(max % 32, 0);
        assert!(max * 123 <= SCRATCH_BUDGET);
        assert!(tile_capacity(SCRATCH_BUDGET, 1, BpScratchLayout::WarpTiled).is_err());
        // Cooperative global thread indices can exceed u32 even though shot indices fit.
        let threads = u64::from(u32::MAX) * 32;
        let config = launch_config(threads);
        let launched = u64::from(config.grid_dim.0) * u64::from(config.block_dim.0);
        assert!(launched >= threads && launched - threads < u64::from(BLOCK));
        assert!(config.grid_dim.0 <= i32::MAX as u32);
        assert_eq!(config.block_dim.0 % 32, 0);
    }

    #[test]
    fn cooperative_graph_ownership_is_checked_before_device_access() {
        let dem = DetectorErrorModel {
            detectors: 2,
            observables: 1,
            errors: vec![
                DemError::new(0.1, vec![0], vec![0]),
                DemError::new(0.2, vec![1], vec![]),
            ],
        };
        let cpu = BpDecoder::new(&dem);
        assert!(validate_cooperative_graph(&cpu.tanner()).is_ok());
        let mut bad = cpu.tanner();
        bad.check_edges = &[0, 0];
        assert!(matches!(
            CudaBp::warp_cooperative(&bad),
            Err(Error::Memory(_))
        ));
        bad = cpu.tanner();
        bad.edge_var = &[1, 0];
        assert!(validate_cooperative_graph(&bad).is_err());
        bad = cpu.tanner();
        bad.check_edges = &[0, 2];
        assert!(validate_cooperative_graph(&bad).is_err());
        bad = cpu.tanner();
        bad.var_off = &[0, 2, 1];
        assert!(validate_cooperative_graph(&bad).is_err());
        bad = cpu.tanner();
        bad.check_off = &[0, 3, 2];
        assert!(validate_cooperative_graph(&bad).is_err());
        bad = cpu.tanner();
        bad.lambda = &[];
        assert!(validate_cooperative_graph(&bad).is_err());
        bad = cpu.tanner();
        bad.num_observables = 65;
        assert!(validate_cooperative_graph(&bad).is_err());
    }

    #[test]
    fn layouts_match_hard_decisions_and_convergence_across_reused_tile_sizes() {
        // Degree-one checks produce infinite messages; isolated variables/checks are intentional.
        let models = [
            DetectorErrorModel {
                detectors: 3,
                observables: 2,
                errors: vec![
                    DemError::new(0.1, vec![0], vec![0]),
                    DemError::new(0.2, vec![1], vec![1]),
                    DemError::new(0.3, vec![], vec![0]),
                ],
            },
            DetectorErrorModel {
                detectors: 3,
                observables: 2,
                errors: vec![
                    DemError::new(0.1, vec![0, 1, 2], vec![0]),
                    DemError::new(0.2, vec![0, 1], vec![1]),
                    DemError::new(0.3, vec![1, 2], vec![0, 1]),
                    DemError::new(0.5, vec![], vec![]),
                ],
            },
            DetectorErrorModel {
                detectors: 0,
                observables: 0,
                errors: vec![],
            },
            aleph_qec::BBCode::gross().code_capacity_dem(0.03),
        ];
        for model in models {
            for iterations in [1, 2, 64] {
                let cpu = BpDecoder::with_params(&model, iterations, 0.875);
                let graph = cpu.tanner();
                for (layout, cooperative) in [
                    (BpScratchLayout::ShotMajor, false),
                    (BpScratchLayout::EdgeMajor, false),
                    (BpScratchLayout::WarpTiled, false),
                    (BpScratchLayout::ShotMajor, true),
                ] {
                    let built = if cooperative {
                        CudaBp::warp_cooperative(&graph)
                    } else {
                        CudaBp::with_scratch_layout(&graph, layout)
                    };
                    let gpu = match built {
                        Ok(gpu) => gpu,
                        Err(e) => {
                            assert!(
                                std::env::var_os("ALEPH_REQUIRE_CUDA").is_none(),
                                "required GPU: {e}"
                            );
                            if matches!(e, Error::NoDevice(_)) {
                                eprintln!("GPU hard-decision oracle unavailable: {e}");
                                return;
                            }
                            panic!("GPU construction failed: {e}");
                        }
                    };
                    for n in [1usize, 31, 32, 33, 129, 7, 0, 257, 3] {
                        let syndromes: Vec<_> = (0..n)
                            .map(|shot| {
                                Syndrome::from_bits(
                                    &(0..model.detectors)
                                        .map(|c| (shot >> (c % usize::BITS as usize)) & 1 != 0)
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect();
                        let full = gpu.decode(&syndromes).unwrap();
                        if n == 0 {
                            continue;
                        }
                        let scratch = gpu.scratch.borrow();
                        let hard = scratch.ehat.as_ref().unwrap().to_vec(&gpu.ctx).unwrap();
                        for (shot, syndrome) in syndromes.iter().enumerate() {
                            let actual: Vec<_> = (0..graph.n_vars)
                                .map(|v| {
                                    hard[match layout {
                                        BpScratchLayout::ShotMajor => shot * graph.n_vars + v,
                                        BpScratchLayout::EdgeMajor => v * n + shot,
                                        BpScratchLayout::WarpTiled => {
                                            shot / 32 * graph.n_vars * 32 + v * 32 + shot % 32
                                        }
                                    }]
                                })
                                .collect();
                            let expected = cpu.decode_bp_soft(syndrome);
                            assert_eq!(actual, expected.ehat, "{layout:?} size={n} shot={shot}");
                            let valid = (0..model.detectors).all(|c| {
                                let parity = graph.check_edges
                                    [graph.check_off[c] as usize..graph.check_off[c + 1] as usize]
                                    .iter()
                                    .fold(0, |p, &e| {
                                        p ^ actual[graph.edge_var[e as usize] as usize]
                                    });
                                parity == u8::from(syndrome.fired.contains(&(c as u32)))
                            });
                            assert_eq!(valid, expected.converged);
                        }
                        drop(scratch);
                        if n == 257 {
                            // Exercise actual multi-tile calls without allocating a 2 GiB batch.
                            let packed = gpu.pack(&syndromes);
                            let wps = gpu.words_per_shot as usize;
                            let mut tiled = vec![0; n];
                            for start in (0..n).step_by(17) {
                                let end = (start + 17).min(n);
                                gpu.decode_tile(
                                    &packed[start * wps..end * wps],
                                    end - start,
                                    &mut tiled[start..end],
                                )
                                .unwrap();
                            }
                            assert_eq!(tiled, full, "small tiles changed output: {layout:?}");
                        }
                    }
                    assert!(gpu.decode_packed(&[], 1).is_err());
                    assert!(gpu.decode_packed(&[0], 0).unwrap().is_empty());
                    let prefix = vec![0; gpu.words_per_shot as usize];
                    let mut extended = prefix.clone();
                    extended.push(42);
                    assert_eq!(
                        gpu.decode_packed(&extended, 1).unwrap(),
                        gpu.decode_packed(&prefix, 1).unwrap()
                    );
                }
            }
        }
    }
}
