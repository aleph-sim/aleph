//! Distributed state vector on CUDA (Phase 6, P6-01a).
//!
//! Executes an `aleph_ir::dist::DistPlan`: `2^g` rank slices of `2^m`
//! amplitudes, local steps through the existing kernels (rank-specialised,
//! optionally fused per rank), exchanges through an [`Exchange`]
//! implementation. [`LocalExchange`] keeps every rank on one device (the
//! single-card development path); NCCL across devices is P6-01b.

use aleph_backend::{Backend, BackendError};
use std::sync::Arc;

use aleph_core::Complex;
use cudarc::driver::{CudaStream, CudaView, CudaViewMut, DeviceRepr};

mod device_sv;
mod exchange;

pub use exchange::{Exchange, LocalExchange};

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
use aleph_ir::dist::{plan as dist_plan, specialize, DistLayout, DistPlan, DistStep, Router};
use aleph_ir::{Circuit, Instruction};

/// A distributed state: `2^g` rank slices plus the plan's final
/// logical→physical map (needed to read amplitudes in logical order).
pub struct DistSvState<B: DeviceSv> {
    pub layout: DistLayout,
    pub final_map: Vec<u32>,
    ranks: Vec<B::State>,
}

/// Executes `DistPlan`s on `B` (one device) with exchange transport `X`.
pub struct DistSvBackend<B: DeviceSv, X: Exchange<B>> {
    be: B,
    x: X,
    fuse: bool,
}

impl<B: DeviceSv, X: Exchange<B>> DistSvBackend<B, X> {
    /// Per-rank fusion on (`fuse_for_gpu`).
    pub fn new(be: B, x: X) -> Self {
        Self { be, x, fuse: true }
    }

    /// Toggle per-rank fusion (off = apply specialised gates one by one).
    pub fn with_fusion(mut self, fuse: bool) -> Self {
        self.fuse = fuse;
        self
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

    /// Execute a plan from |0…0⟩.
    pub fn run_plan(&mut self, p: &DistPlan) -> Result<DistSvState<B>, DistSvError> {
        let l = p.layout;
        let m = l.m();
        let mut ranks = Vec::with_capacity(l.ranks() as usize);
        for r in 0..l.ranks() {
            ranks.push(self.be.alloc_rank(m, r)?);
        }
        for step in &p.steps {
            match step {
                DistStep::Local(instrs) => {
                    for (r, st) in ranks.iter_mut().enumerate() {
                        let local = self.rank_program(instrs, l, r as u32)?;
                        for i in local.instructions() {
                            apply_one(&mut self.be, st, i)?;
                        }
                    }
                }
                DistStep::Exchange { global_bits } => {
                    self.x.exchange(&mut self.be, &mut ranks, l, global_bits)?;
                }
            }
        }
        Ok(DistSvState {
            layout: l,
            final_map: p.final_map.clone(),
            ranks,
        })
    }

    /// Rank `r`'s `m`-qubit program for one `Local` step: specialised, then
    /// (optionally) fused — fusion runs *after* `specialize` so it never sees
    /// a global qubit (spec §3.1).
    fn rank_program(
        &self,
        instrs: &[Instruction],
        l: DistLayout,
        r: u32,
    ) -> Result<Circuit, DistSvError> {
        let mut c = Circuit::new(l.m(), 0);
        for i in instrs {
            if let Some(s) = specialize(i, l, r)? {
                c.add_instruction(s)
                    .map_err(|_| BackendError::InvalidState {
                        reason: "dist: specialised instruction rejected by Circuit",
                    })?;
            }
        }
        Ok(if self.fuse {
            crate::fuse_for_gpu(&c)
        } else {
            c
        })
    }

    /// Full state in **logical** qubit order (host gather; tests / small n).
    pub fn amplitudes(&mut self, st: &DistSvState<B>) -> Result<Vec<Complex<f64>>, DistSvError> {
        let m = st.layout.m();
        let size = 1usize << m;
        let mut phys: Vec<Complex<f64>> = Vec::with_capacity(size << st.layout.g);
        for r in &st.ranks {
            phys.extend(self.be.download(r)?);
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

    /// Σ|a|² over all ranks (host reduction of per-rank downloads).
    pub fn norm_sqr(&mut self, st: &DistSvState<B>) -> Result<f64, DistSvError> {
        let mut s = 0.0;
        for r in &st.ranks {
            s += self
                .be
                .download(r)?
                .iter()
                .map(|a| a.norm_sqr())
                .sum::<f64>();
        }
        Ok(s)
    }
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
