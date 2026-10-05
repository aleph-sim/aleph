//! Distributed state vector on CUDA (Phase 6, P6-01a).
//!
//! Executes an `aleph_ir::dist::DistPlan`: `2^g` rank slices of `2^m`
//! amplitudes, local steps through the existing kernels (rank-specialised,
//! optionally fused per rank), exchanges through an [`Exchange`]
//! implementation. Ranks sit on one or more devices in contiguous blocks
//! (P6-01b). [`LocalExchange`] handles every pair whose two devices share a
//! GPU (the single-card path); `NcclExchange` (feature `nccl`) moves chunks
//! between GPUs.

use aleph_backend::{Backend, BackendError};
use std::sync::Arc;

use aleph_core::Complex;
use cudarc::driver::{CudaStream, CudaView, CudaViewMut, DeviceRepr};

pub mod cost;
mod device_sv;
mod exchange;
#[cfg(feature = "nccl")]
mod nccl;
#[cfg_attr(not(feature = "nccl"), allow(dead_code))] // only NcclExchange schedules
mod schedule;

use exchange::rank_device;
pub use exchange::{Exchange, LocalExchange};
#[cfg(feature = "nccl")]
pub use nccl::NcclExchange;

/// What the distributed layer needs from a single-device SV backend beyond
/// [`Backend`]: rank-slice allocation, device-to-device amplitude copies, and
/// host transfer for readout / tests.
pub trait DeviceSv: Backend {
    /// Largest rank slice, in qubits, this backend accepts. `alloc_rank` and
    /// `upload` reject a larger `m` with `TooManyQubits` (spec §4: the
    /// per-rank state must fit the device).
    fn max_qubits(&self) -> u32;
    /// A rank slice of `m` qubits: |0…0⟩ if `rank == 0`, all-zero otherwise.
    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<Self::State, BackendError>;
    /// Copy `len` amplitudes `src[src_off..]` → `dst[dst_off..]` on the device.
    fn copy_amps(
        &mut self,
        src: &Self::State,
        src_off: usize,
        dst: &mut Self::State,
        dst_off: usize,
        len: usize,
    ) -> Result<(), BackendError>;
    /// All amplitudes as complex f64 (FP32 widens).
    fn download(&mut self, st: &Self::State) -> Result<Vec<Complex<f64>>, BackendError>;
    /// A fresh `m`-qubit slice holding `amps` (`amps.len() == 2^m`).
    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<Self::State, BackendError>;
    /// Device scalar of the interleaved (re, im) amplitude buffer: f64 or f32.
    type Scalar: DeviceRepr + Copy + 'static;
    /// CUDA ordinal this backend launches on.
    fn ordinal(&self) -> usize;
    /// The stream every kernel and copy of this backend is ordered on (NCCL
    /// comms bind to it, so exchanges need no host barrier).
    fn stream(&self) -> Arc<CudaStream>;
    /// Interleaved scalars `[2*off, 2*(off+len))` of `st` (bounds-checked).
    fn amps_view(
        st: &Self::State,
        off: usize,
        len: usize,
    ) -> Result<CudaView<'_, Self::Scalar>, BackendError>;
    /// Mutable form of [`Self::amps_view`].
    fn amps_view_mut(
        st: &mut Self::State,
        off: usize,
        len: usize,
    ) -> Result<CudaViewMut<'_, Self::Scalar>, BackendError>;
    /// `(Σ|a|², Σ_{i & qbit ≠ 0} |a|²)` on the device, with no normalisation
    /// check (rank slices are not normalised). `qbit = 0` gives the total alone.
    fn branch_norms(&mut self, st: &Self::State, qbit: u64) -> Result<(f64, f64), BackendError>;
}

/// Distributed-run failure.
#[derive(Debug, thiserror::Error)]
pub enum DistSvError {
    #[error(transparent)]
    Dist(#[from] aleph_ir::dist::DistError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}
use aleph_ir::dist::{
    compile, plan as dist_plan, specialize, CostModel, DistError, DistLayout, DistPlan, DistStep,
    Router,
};
use aleph_ir::{Circuit, Instruction};

/// A distributed state: `2^g` rank slices plus the plan's final
/// logical→physical map (needed to read amplitudes in logical order).
pub struct DistSvState<B: DeviceSv> {
    pub layout: DistLayout,
    pub final_map: Vec<u32>,
    ranks: Vec<B::State>,
}

/// Executes `DistPlan`s on one or more `B` devices with exchange transport `X`.
pub struct DistSvBackend<B: DeviceSv, X: Exchange<B>> {
    devs: Vec<B>,
    x: X,
    fuse: bool,
}

impl<B: DeviceSv, X: Exchange<B>> DistSvBackend<B, X> {
    /// One device; per-rank fusion on (`fuse_for_gpu`).
    pub fn new(be: B, x: X) -> Self {
        Self {
            devs: vec![be],
            x,
            fuse: true,
        }
    }

    /// One backend per device. Ranks go to devices in contiguous blocks
    /// (rank r → device r / (R/D)), so the top log2(D) rank bits name the
    /// device and an exchange on lower global bits stays on-device.
    pub fn multi(devs: Vec<B>, x: X) -> Result<Self, DistSvError> {
        if devs.is_empty() || !devs.len().is_power_of_two() {
            return Err(BackendError::InvalidState {
                reason: "dist: device count must be a power of two >= 1",
            }
            .into());
        }
        Ok(Self {
            devs,
            x,
            fuse: true,
        })
    }

    /// Number of devices.
    pub fn devices(&self) -> usize {
        self.devs.len()
    }

    /// Toggle per-rank fusion (off = apply specialised gates one by one).
    pub fn with_fusion(mut self, fuse: bool) -> Self {
        self.fuse = fuse;
        self
    }

    /// Whether per-rank fusion is on. A [`crate::GpuCostModel`] passed to
    /// [`Self::run_compiled`] should carry the same `fuse` value, or it
    /// prices a different program than the one that runs.
    pub fn fusion(&self) -> bool {
        self.fuse
    }

    /// Plan `c` over `2^g` ranks with `router`, then execute.
    pub fn run(
        &mut self,
        c: &Circuit,
        g: u32,
        router: Router,
    ) -> Result<DistSvState<B>, DistSvError> {
        let layout = DistLayout::new(c.num_qubits(), g)?;
        let p = dist_plan(c, layout, router)?;
        self.run_plan(&p)
    }

    /// Compile `c` over `2^g` ranks (P6-05: the cheapest of every router x
    /// placement candidate under `cost`), then execute.
    ///
    /// `cost` only *chooses* among valid plans, so a mis-calibrated model
    /// costs speed, never correctness. Use the preset matching the backend's
    /// precision (`rtx4000_fp64()` for `CudaSvBackend`, `rtx4000_fp32()` for
    /// `CudaSvBackendF32`). Build a `GpuCostModel` with
    /// `fuse: self.fusion()`; its constants are valid near the slice size
    /// they were calibrated at (`KindTimes::m_ref`).
    pub fn run_compiled(
        &mut self,
        c: &Circuit,
        g: u32,
        cost: &dyn CostModel,
    ) -> Result<DistSvState<B>, DistSvError> {
        let layout = DistLayout::new(c.num_qubits(), g)?;
        let p = compile(c, layout, cost)?;
        self.run_plan(&p)
    }

    /// Execute a plan from |0…0⟩.
    pub fn run_plan(&mut self, p: &DistPlan) -> Result<DistSvState<B>, DistSvError> {
        let l = p.layout;
        let (m, nr, nd) = (l.m(), l.ranks(), self.devs.len());
        if nd as u64 > u64::from(nr) {
            return Err(BackendError::InvalidState {
                reason: "dist: more devices than ranks",
            }
            .into());
        }
        check_final_map(&p.final_map, l.n)?;
        let mut ranks = Vec::with_capacity(nr as usize);
        for r in 0..nr {
            ranks.push(self.devs[rank_device(r, nr, nd)].alloc_rank(m, r)?);
        }
        // Launches are asynchronous on each device's stream, so this serial
        // host loop still overlaps devices; exchanges are ordered on the same
        // streams, so no host barrier is needed between steps.
        for step in &p.steps {
            match step {
                DistStep::Local(instrs) => {
                    for (r, st) in ranks.iter_mut().enumerate() {
                        let local = self.rank_program(instrs, l, r as u32)?;
                        let be = &mut self.devs[rank_device(r as u32, nr, nd)];
                        for i in local.instructions() {
                            apply_one(be, st, i)?;
                        }
                    }
                }
                DistStep::Exchange { global_bits } => {
                    self.x
                        .exchange(&mut self.devs, &mut ranks, l, global_bits)?;
                }
            }
        }
        Ok(DistSvState {
            layout: l,
            final_map: p.final_map.clone(),
            ranks,
        })
    }

    /// Kernel launches rank 0 issues over the whole plan (after specialise +
    /// fusion): the "passes" term of the P6 time model. Bench support.
    #[doc(hidden)]
    pub fn rank_pass_count(&self, p: &DistPlan) -> Result<usize, DistSvError> {
        let mut n = 0;
        for step in &p.steps {
            if let DistStep::Local(instrs) = step {
                n += self.rank_program(instrs, p.layout, 0)?.instructions().len();
            }
        }
        Ok(n)
    }

    /// Rank `r`'s `m`-qubit program for one `Local` step (see [`rank_circuit`]).
    fn rank_program(
        &self,
        instrs: &[Instruction],
        l: DistLayout,
        r: u32,
    ) -> Result<Circuit, DistSvError> {
        Ok(rank_circuit(instrs, l, r, self.fuse)?)
    }

    /// Full state in **logical** qubit order (host gather; tests / small n).
    pub fn amplitudes(&mut self, st: &DistSvState<B>) -> Result<Vec<Complex<f64>>, DistSvError> {
        let m = st.layout.m();
        let size = 1usize << m;
        let mut phys: Vec<Complex<f64>> = Vec::with_capacity(size << st.layout.g);
        let (nr, nd) = (st.layout.ranks(), self.devs.len());
        for (r, rs) in st.ranks.iter().enumerate() {
            phys.extend(self.devs[rank_device(r as u32, nr, nd)].download(rs)?);
        }
        let mut out = vec![Complex::new(0.0, 0.0); phys.len()];
        for (x, slot) in out.iter_mut().enumerate() {
            let mut p = 0usize;
            for (lq, &pq) in st.final_map.iter().enumerate() {
                p |= ((x >> lq) & 1) << pq;
            }
            *slot = phys[p];
        }
        Ok(out)
    }

    /// Σ|a|²: per-rank device reduction, host sum of R scalars (spec §3.4).
    pub fn norm_sqr(&mut self, st: &DistSvState<B>) -> Result<f64, DistSvError> {
        let (nr, nd) = (st.layout.ranks(), self.devs.len());
        let mut s = 0.0;
        for (r, rs) in st.ranks.iter().enumerate() {
            s += self.devs[rank_device(r as u32, nr, nd)]
                .branch_norms(rs, 0)?
                .0;
        }
        Ok(s)
    }

    /// P(logical qubit `q` = 1). At physical position `final_map[q]`, a local
    /// bit is a per-rank masked reduction; a global bit sums whole ranks with
    /// that rank bit set.
    pub fn prob_one(&mut self, st: &DistSvState<B>, q: u32) -> Result<f64, DistSvError> {
        let Some(&p) = st.final_map.get(q as usize) else {
            return Err(BackendError::QubitOutOfRange {
                qubit: q,
                num_qubits: st.layout.n,
            }
            .into());
        };
        let (m, nr, nd) = (st.layout.m(), st.layout.ranks(), self.devs.len());
        let mut s = 0.0;
        for (r, rs) in st.ranks.iter().enumerate() {
            let be = &mut self.devs[rank_device(r as u32, nr, nd)];
            if p < m {
                s += be.branch_norms(rs, 1u64 << p)?.1;
            } else if (r as u32 >> (p - m)) & 1 == 1 {
                s += be.branch_norms(rs, 0)?.0;
            }
        }
        Ok(s)
    }
}

/// `final_map` must be a permutation of 0..n (a corrupt plan would read the
/// wrong amplitudes or index out of bounds).
fn check_final_map(map: &[u32], n: u32) -> Result<(), DistSvError> {
    if map.len() != n as usize {
        return Err(BackendError::InvalidState {
            reason: "dist: final_map length != n",
        }
        .into());
    }
    let mut seen = vec![false; n as usize];
    for &p in map {
        let Some(s) = seen.get_mut(p as usize) else {
            return Err(BackendError::InvalidState {
                reason: "dist: final_map out of range",
            }
            .into());
        };
        if std::mem::replace(s, true) {
            return Err(BackendError::InvalidState {
                reason: "dist: final_map not a permutation",
            }
            .into());
        }
    }
    Ok(())
}

fn apply_one<B: DeviceSv>(
    be: &mut B,
    st: &mut B::State,
    i: &Instruction,
) -> Result<(), DistSvError> {
    match i {
        Instruction::Gate(g) => be.apply_gate(st, g)?,
        Instruction::DiagonalPhase(dp) => be.apply_diagonal_phase(st, dp)?,
        Instruction::TiledBlock(tb) => be.apply_tiled_block(st, tb)?,
        Instruction::Barrier(_) => {}
        Instruction::Measure { .. } | Instruction::Reset(_) => {
            return Err(BackendError::UnsupportedInstruction {
                kind: "measure/reset in dist",
            }
            .into())
        }
    }
    Ok(())
}

/// Rank `r`'s `m`-qubit program for one `Local` step: specialised, then
/// (optionally) fused — fusion runs *after* `specialize` so it never sees a
/// global qubit (spec §3.1). Shared by execution and the cost model, so the
/// model prices exactly what runs.
pub(crate) fn rank_circuit(
    instrs: &[Instruction],
    l: DistLayout,
    r: u32,
    fuse: bool,
) -> Result<Circuit, DistError> {
    let mut c = Circuit::new(l.m(), 0);
    for i in instrs {
        if let Some(s) = specialize(i, l, r)? {
            c.add_instruction(s).map_err(|_| DistError::Unsupported {
                kind: "internal: specialised instruction rejected by Circuit",
            })?;
        }
    }
    Ok(if fuse { crate::fuse_for_gpu(&c) } else { c })
}
