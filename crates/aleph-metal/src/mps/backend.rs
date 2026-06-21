//! `MetalMpsBackend` — a scaffold GPU MPS backend (P5.5-06).
//!
//! Runs nearest-neighbour circuits with the hot per-gate tensor work on the GPU
//! (1q apply, two-site contraction, 2q gate-apply, and — since P5.7-03 — the
//! truncated SVD via a one-sided Jacobi kernel; `faer` is the CPU fallback).
//! Non-adjacent 2q gates are routed with a physical SWAP network (P5.7-06);
//! external controls and ≥3q gates return `UnsupportedInstruction`.
//! Correctness is gated by the oracle (`tests/mps_oracle.rs`) vs the CPU MPS.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, Gate, GateError, GateInstance, GateMatrix, PauliString};
use aleph_ir::{Circuit, Instruction};
use metal::{CommandBufferRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::gpu_jacobi::{gpu_svd_split, gpu_svd_split_batch, BatchBlock};
use super::kernel::{
    Apply2qMeta, ContractMeta, Mps1q, MPS_1Q_ENTRY, MPS_1Q_SRC, MPS_APPLY2Q_ENTRY, MPS_APPLY2Q_SRC,
    MPS_CONTRACT_ENTRY, MPS_CONTRACT_SRC, MPS_JACOBI_BATCHED_ENTRY, MPS_JACOBI_ENTRY,
    MPS_JACOBI_SRC,
};
use super::readout;
use super::state::{MetalMpsState, SiteTensor};
use super::svd::svd_split;
use crate::{DeviceBuffer, Error, MetalContext};
use rand::{rngs::StdRng, SeedableRng};

/// Qubit cap for the scaffold. The MPS form scales past the SV 28-qubit ceiling,
/// but the scaffold's dense oracle readout is small-n only, so we keep the
/// project-wide cap to avoid surprising 2^n allocations in tests.
pub(crate) const MAX_MPS_QUBITS: u32 = 28;

/// Default maximum bond dimension χ. Generous enough that the small NN Tier-1
/// test circuits never truncate (so the oracle compare is exact-to-fp32).
const DEFAULT_MAX_BOND: usize = 128;

/// Relative Schmidt weight (`Σ_{j≥χ} σ_j² / Σ σ_j²`) above which a two-site split
/// is treated as a real truncation and refused. This naive (non-canonical)
/// scaffold has no orthogonality centre to absorb the renormalization, so any
/// non-negligible drop would silently corrupt the state (P5.6-02). The floor sits
/// far above the null-direction pruning residue (≈1e-14, see `svd::truncation_plan`)
/// and far below the loss from dropping any genuine singular value.
const MPS_TRUNC_TOL: f64 = 1e-10;

/// Opt-in single-precision Metal GPU MPS backend (scaffold).
pub struct MetalMpsBackend {
    ctx: MetalContext,
    pipeline_1q: ComputePipelineState,
    pipeline_contract: ComputePipelineState,
    pipeline_apply2q: ComputePipelineState,
    /// GPU-resident one-sided Jacobi thin-SVD (P5.7-03). Replaces the host faer
    /// SVD on the per-gate two-site split; faer remains the CPU fallback.
    pipeline_jacobi: ComputePipelineState,
    /// Batched one-sided Jacobi (P5.7-04): one threadgroup per block, a whole
    /// brickwall layer's independent splits in a single dispatch. Drives
    /// [`run_batched`](Self::run_batched).
    pipeline_jacobi_batched: ComputePipelineState,
    /// Reused 4×4 (16-entry) row-major f32 gate-matrix scratch for the 2q apply.
    mat_scratch: DeviceBuffer<Complex<f32>>,
    max_bond: usize,
    /// RNG for the stochastic readout ops (`measure`/`sample`, P5.7-05). Seeded
    /// via [`with_seed`](Self::with_seed) for reproducibility.
    rng: StdRng,
    /// Cumulative time (ns) in the GPU contract+apply dispatches vs the host SVD
    /// split, summed over every NN 2q gate. Drives the AC #2 round-trip-cost doc.
    gpu_ns: u128,
    svd_ns: u128,
    /// Cumulative relative Schmidt weight discarded across every two-site split
    /// (P5.6-02). Stays ≈0 on the exact-only path; a split above `MPS_TRUNC_TOL`
    /// is refused, but its weight is still recorded here for diagnostics.
    trunc_error: f64,
}

impl MetalMpsBackend {
    /// Construct with the system-default Metal device. Returns
    /// [`BackendError::InvalidState`] when no device is present (headless CI) or a
    /// shader/pipeline build fails.
    pub fn new() -> Result<Self, BackendError> {
        Self::build(DEFAULT_MAX_BOND, 0)
    }

    /// Construct with an explicit RNG seed for the stochastic readout ops
    /// (`measure`/`sample`, P5.7-05). Reproducible for a fixed seed.
    pub fn with_seed(seed: u64) -> Result<Self, BackendError> {
        Self::build(DEFAULT_MAX_BOND, seed)
    }

    /// Construct with an explicit maximum bond dimension χ.
    pub fn with_max_bond(max_bond: usize) -> Result<Self, BackendError> {
        Self::build(max_bond.max(1), 0)
    }

    /// Execute `circuit` gate-by-gate on the GPU. Thin wrapper over
    /// [`aleph_backend::run`]; the verbatim path the oracle checks.
    pub fn run(&mut self, circuit: &Circuit) -> Result<MetalMpsState, BackendError> {
        aleph_backend::run(self, circuit)
    }

    /// Optimize `circuit` with the default IR pipeline, then execute it. Fused
    /// blocks become `UnitaryKq`/tiled ops, which the scaffold does not support,
    /// so prefer [`run`](Self::run) for NN circuits; provided for API parity.
    pub fn run_optimized(&mut self, circuit: &Circuit) -> Result<MetalMpsState, BackendError> {
        aleph_backend::run_optimized(self, circuit)
    }

    fn build(max_bond: usize, seed: u64) -> Result<Self, BackendError> {
        let ctx = MetalContext::new().map_err(map_metal_err)?;
        let pipeline_1q = ctx
            .make_compute_pipeline(MPS_1Q_SRC, MPS_1Q_ENTRY)
            .map_err(map_metal_err)?;
        let pipeline_contract = ctx
            .make_compute_pipeline(MPS_CONTRACT_SRC, MPS_CONTRACT_ENTRY)
            .map_err(map_metal_err)?;
        let pipeline_apply2q = ctx
            .make_compute_pipeline(MPS_APPLY2Q_SRC, MPS_APPLY2Q_ENTRY)
            .map_err(map_metal_err)?;
        let pipeline_jacobi = ctx
            .make_compute_pipeline(MPS_JACOBI_SRC, MPS_JACOBI_ENTRY)
            .map_err(map_metal_err)?;
        let pipeline_jacobi_batched = ctx
            .make_compute_pipeline(MPS_JACOBI_SRC, MPS_JACOBI_BATCHED_ENTRY)
            .map_err(map_metal_err)?;
        let mat_scratch = DeviceBuffer::from_slice(&ctx, &[Complex::<f32>::new(0.0, 0.0); 16]);
        Ok(Self {
            ctx,
            pipeline_1q,
            pipeline_contract,
            pipeline_apply2q,
            pipeline_jacobi,
            pipeline_jacobi_batched,
            mat_scratch,
            max_bond,
            rng: StdRng::seed_from_u64(seed),
            gpu_ns: 0,
            svd_ns: 0,
            trunc_error: 0.0,
        })
    }

    /// Cumulative `(gpu_ns, svd_ns)` across all NN 2q gates so far: time in the
    /// GPU contract+apply dispatches vs the host SVD split + factor upload. Used
    /// to document the CPU-SVD round-trip cost (P5.5-06 AC #2).
    pub fn timing_ns(&self) -> (u128, u128) {
        (self.gpu_ns, self.svd_ns)
    }

    /// Reset the cumulative timing counters.
    pub fn reset_timing(&mut self) {
        self.gpu_ns = 0;
        self.svd_ns = 0;
    }

    /// Cumulative relative Schmidt weight discarded by two-site splits so far
    /// (`Σ_{j≥χ} σ_j² / Σ σ_j²`, summed over NN 2q gates). ≈0 on the exact path;
    /// non-zero only if a split approached/crossed the truncation tolerance.
    /// Surfaced so callers can inspect compression loss rather than have it
    /// silently dropped (P5.6-02).
    pub fn trunc_error(&self) -> f64 {
        self.trunc_error
    }

    /// Reset the cumulative truncation-weight accumulator.
    pub fn reset_trunc_error(&mut self) {
        self.trunc_error = 0.0;
    }

    /// Dispatch one kernel over `threads` and block until the GPU finishes so the
    /// unified-memory buffers are current for the next gate or the host SVD.
    fn dispatch_1q_site(&self, buf: &DeviceBuffer<Complex<f32>>, g: &Mps1q, threads: u64) {
        let cmd = self.ctx.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline_1q);
        enc.set_buffer(0, Some(buf.metal_buffer()), 0);
        enc.set_bytes(
            1,
            std::mem::size_of::<Mps1q>() as u64,
            g as *const Mps1q as *const c_void,
        );
        let tg = self
            .pipeline_1q
            .max_total_threads_per_threadgroup()
            .min(threads);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Θ = A · B into `theta` (must be pre-sized `rows·cols`). Grid = `threads`.
    fn dispatch_contract(
        &self,
        a: &DeviceBuffer<Complex<f32>>,
        b: &DeviceBuffer<Complex<f32>>,
        theta: &DeviceBuffer<Complex<f32>>,
        meta: &ContractMeta,
        threads: u64,
    ) {
        let cmd = self.ctx.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline_contract);
        enc.set_buffer(0, Some(a.metal_buffer()), 0);
        enc.set_buffer(1, Some(b.metal_buffer()), 0);
        enc.set_buffer(2, Some(theta.metal_buffer()), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<ContractMeta>() as u64,
            meta as *const ContractMeta as *const c_void,
        );
        let tg = self
            .pipeline_contract
            .max_total_threads_per_threadgroup()
            .min(threads);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Θ' = U·Θ in place; the caller must have written the 4×4 into `mat_scratch`.
    fn dispatch_apply2q(
        &self,
        theta: &DeviceBuffer<Complex<f32>>,
        meta: &Apply2qMeta,
        threads: u64,
    ) {
        let cmd = self.ctx.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline_apply2q);
        enc.set_buffer(0, Some(theta.metal_buffer()), 0);
        enc.set_buffer(1, Some(self.mat_scratch.metal_buffer()), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<Apply2qMeta>() as u64,
            meta as *const Apply2qMeta as *const c_void,
        );
        let tg = self
            .pipeline_apply2q
            .max_total_threads_per_threadgroup()
            .min(threads);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Encode (no commit/wait) Θ = A · B into `theta` onto an existing command
    /// buffer — the batched-layer (P5.7-04) counterpart to [`dispatch_contract`].
    /// Metal's default resource hazard tracking serializes a later encoder that
    /// reads `theta` after this one writes it, so a whole layer's contract +
    /// gate-apply encoders share one `commit`/`wait`.
    fn encode_contract(
        &self,
        cmd: &CommandBufferRef,
        a: &DeviceBuffer<Complex<f32>>,
        b: &DeviceBuffer<Complex<f32>>,
        theta: &DeviceBuffer<Complex<f32>>,
        meta: &ContractMeta,
        threads: u64,
    ) {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline_contract);
        enc.set_buffer(0, Some(a.metal_buffer()), 0);
        enc.set_buffer(1, Some(b.metal_buffer()), 0);
        enc.set_buffer(2, Some(theta.metal_buffer()), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<ContractMeta>() as u64,
            meta as *const ContractMeta as *const c_void,
        );
        let tg = self
            .pipeline_contract
            .max_total_threads_per_threadgroup()
            .min(threads);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
    }

    /// Encode (no commit/wait) Θ' = U·Θ in place onto an existing command buffer,
    /// reading the 4×4 from a caller-owned per-gate `mat` buffer (not the shared
    /// scratch, so a layer's gate-applies don't clobber each other). The
    /// batched-layer counterpart to [`dispatch_apply2q`].
    fn encode_apply2q(
        &self,
        cmd: &CommandBufferRef,
        theta: &DeviceBuffer<Complex<f32>>,
        mat: &DeviceBuffer<Complex<f32>>,
        meta: &Apply2qMeta,
        threads: u64,
    ) {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline_apply2q);
        enc.set_buffer(0, Some(theta.metal_buffer()), 0);
        enc.set_buffer(1, Some(mat.metal_buffer()), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<Apply2qMeta>() as u64,
            meta as *const Apply2qMeta as *const c_void,
        );
        let tg = self
            .pipeline_apply2q
            .max_total_threads_per_threadgroup()
            .min(threads);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
    }

    /// Apply a 1q gate to site `q` (in place on its tensor). Mirrors the CPU MPS
    /// `apply_1q`: no SVD, no bond change.
    fn apply_1q(&self, state: &MetalMpsState, q: usize, m: &[[Complex<f64>; 2]; 2]) {
        let site = &state.sites[q];
        let g = Mps1q {
            m: [
                narrow(m[0][0]),
                narrow(m[0][1]),
                narrow(m[1][0]),
                narrow(m[1][1]),
            ],
            right: site.right as u32,
            _pad: 0,
        };
        let threads = (site.left * site.right) as u64;
        self.dispatch_1q_site(&site.buf, &g, threads);
    }

    /// Apply a NN 2q gate `m` (4×4, `qa`=MSB, `qb`=LSB) to adjacent sites:
    /// GPU contract → GPU gate-apply → host truncated-SVD split. `qa.abs_diff(qb)`
    /// must be 1 (enforced by the caller).
    fn apply_2q_nn(
        &mut self,
        state: &mut MetalMpsState,
        qa: usize,
        qb: usize,
        m: &[[Complex<f64>; 4]; 4],
    ) -> Result<(), BackendError> {
        let i = qa.min(qb);
        let j = i + 1;
        // `i_is_msb`: the left site holds the matrix-MSB qubit (qa) iff qa is the
        // lower index. Selects the physical→matrix-index map in the apply kernel.
        let i_is_msb = qa == i;
        let li = state.sites[i].left;
        let ci = state.sites[i].right;
        debug_assert_eq!(
            ci, state.sites[j].left,
            "MPS bond mismatch at sites {i},{j}"
        );
        let ri = state.sites[j].right;
        let rows = li * 2;
        let cols = 2 * ri;

        // --- GPU phase: contract + gate-apply ---
        let gpu_start = std::time::Instant::now();
        // Θ on the GPU (fresh shared buffer, zero-initialised then overwritten).
        let theta =
            DeviceBuffer::from_slice(&self.ctx, &vec![Complex::<f32>::new(0.0, 0.0); rows * cols]);
        let cmeta = ContractMeta {
            c: ci as u32,
            ri: ri as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.dispatch_contract(
            &state.sites[i].buf,
            &state.sites[j].buf,
            &theta,
            &cmeta,
            (rows * cols) as u64,
        );

        // Upload the 4×4 (qa-MSB/qb-LSB row-major) into the scratch, then apply.
        {
            let scratch = self.mat_scratch.as_mut_slice();
            #[allow(clippy::needless_range_loop)]
            for r in 0..4 {
                for c in 0..4 {
                    scratch[r * 4 + c] = narrow(m[r][c]);
                }
            }
        }
        let ameta = Apply2qMeta {
            ri: ri as u32,
            i_is_msb: i_is_msb as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.dispatch_apply2q(&theta, &ameta, (li * ri) as u64);
        self.gpu_ns += gpu_start.elapsed().as_nanos();

        // --- SVD split: GPU-resident (P5.7-03), CPU faer as the fallback ---
        // The one-sided Jacobi kernel factorizes Θ' on the GPU, eliminating the
        // CPU round-trip that dominated per-gate time (the documented ~94.5%). The
        // f64 faer `svd_split` stays the safety net for a non-finite GPU result.
        // The slice is current: dispatch_apply2q waited.
        let svd_start = std::time::Instant::now();
        let (chi, site_i_data, site_j_data, trunc) = match gpu_svd_split(
            &self.ctx,
            &self.pipeline_jacobi,
            theta.as_slice(),
            rows,
            cols,
            self.max_bond,
        ) {
            Some(split) => split,
            None => svd_split(theta.as_slice(), rows, cols, self.max_bond)?,
        };
        // Record the dropped weight first, then refuse if it's a real truncation:
        // this naive scaffold has no orthogonality centre to renormalize against,
        // so applying a truncated split would silently corrupt the state. Refusing
        // before mutating `state.sites` leaves the pre-gate MPS intact (P5.6-02).
        self.trunc_error += trunc;
        if trunc > MPS_TRUNC_TOL {
            self.svd_ns += svd_start.elapsed().as_nanos();
            return Err(BackendError::MpsTruncationUnsupported {
                max_bond: self.max_bond,
                trunc_error: trunc,
            });
        }
        state.sites[i] = SiteTensor::from_host(&self.ctx, li, chi, &site_i_data);
        state.sites[j] = SiteTensor::from_host(&self.ctx, chi, ri, &site_j_data);
        self.svd_ns += svd_start.elapsed().as_nanos();
        Ok(())
    }

    /// Apply a 2q gate `m` to a **non-adjacent** pair (`qa.abs_diff(qb) ≥ 2`) by
    /// routing with a physical SWAP network (P5.7-06). The qubit on the higher
    /// site is walked down to sit beside the lower one via adjacent SWAPs (each a
    /// NN 2q gate through the same contract+apply+SVD path), the gate is applied on
    /// the now-adjacent pair with its original MSB/LSB orientation, then the SWAPs
    /// are unwound — so the scaffold's site≡qubit-order invariant is restored and
    /// the readout/dense paths stay correct. CPU MPS does the same (P3-06); lazy
    /// permutation tracking (P3-12) is avoided here precisely to keep that invariant.
    fn apply_2q_routed(
        &mut self,
        state: &mut MetalMpsState,
        qa: usize,
        qb: usize,
        m: &[[Complex<f64>; 4]; 4],
    ) -> Result<(), BackendError> {
        let i = qa.min(qb);
        let j = qa.max(qb);
        debug_assert!(j - i >= 2, "apply_2q_routed requires non-adjacent qubits");
        let swap = swap_matrix();

        // Walk the qubit at site j down to site i+1: swap pairs (m, m+1) for
        // m = j-1, j-2, …, i+1. After this, sites i and i+1 hold the two targets,
        // in the same relative order as qubit indices (lower index on site i).
        for s in (i + 1..j).rev() {
            self.apply_2q_nn(state, s, s + 1, &swap)?;
        }

        // Apply the gate on the adjacent pair, preserving orientation: the lower
        // qubit index sits on site i, so the (site_of_qa, site_of_qb) order tracks
        // (qa, qb), and `apply_2q_nn`'s own min/i_is_msb logic picks the map.
        let (pa, pb) = if qa < qb { (i, i + 1) } else { (i + 1, i) };
        self.apply_2q_nn(state, pa, pb, m)?;

        // Unwind the SWAP network (reverse order) to restore site≡qubit ordering.
        for s in (i + 1)..j {
            self.apply_2q_nn(state, s, s + 1, &swap)?;
        }
        Ok(())
    }

    /// Validate `gate` against the scaffold's limits and return its dense matrix.
    /// Shared by [`Backend::apply_gate`] (gate-by-gate) and the layered
    /// [`run_batched`](Self::run_batched) scheduler so the two paths reject the
    /// same inputs identically. `num_qubits` is the chain length.
    fn gate_matrix(
        &self,
        num_qubits: u32,
        gate: &GateInstance,
    ) -> Result<GateMatrix, BackendError> {
        let expected = gate.gate.arity();
        let got = gate.qubits.len();
        if expected != got {
            return Err(BackendError::ArityMismatch {
                kind: gate.gate.name(),
                expected,
                got,
            });
        }
        let mut seen: Vec<u32> = Vec::new();
        for &q in gate.qubits.iter().chain(gate.controls.iter()) {
            if q >= num_qubits {
                return Err(BackendError::QubitOutOfRange {
                    qubit: q,
                    num_qubits,
                });
            }
            if seen.contains(&q) {
                return Err(BackendError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        // Scaffold limits: no external controls, no fused dense blocks.
        if !gate.controls.is_empty() {
            return Err(BackendError::UnsupportedInstruction {
                kind: "externally-controlled gate (MPS scaffold)",
            });
        }
        if matches!(gate.gate, Gate::UnitaryKq { .. }) {
            return Err(BackendError::UnsupportedInstruction {
                kind: "fused UnitaryKq block (MPS scaffold; use run, not run_optimized)",
            });
        }
        gate.gate.matrix().map_err(|e| match e {
            GateError::SymbolicParam => BackendError::SymbolicParam,
            GateError::NonFiniteParam => BackendError::NonFiniteParam {
                kind: gate.gate.name(),
            },
            GateError::Unrepresentable => BackendError::UnsupportedGate {
                kind: gate.gate.name(),
            },
        })
    }

    /// Execute `circuit` with **layer-batched** two-site SVDs (P5.7-04): adjacent
    /// nearest-neighbour 2q gates that act on disjoint site pairs are grouped into
    /// a layer and their splits factored in a *single* batched-Jacobi dispatch
    /// (one `commit`/`wait` for the contract+apply phase, one for the split),
    /// instead of one dispatch + GPU sync per gate as in [`run`](Self::run).
    ///
    /// Semantics match the gate-by-gate path: gates in a layer touch disjoint
    /// qubits, so they commute and batching is exact; a 1q gate, a barrier, or any
    /// site conflict flushes the pending layer first, preserving program order.
    /// Truncation is still refused, not applied (P5.6-02). The dense oracle checks
    /// this path against the CPU MPS, same as `run`.
    pub fn run_batched(&mut self, circuit: &Circuit) -> Result<MetalMpsState, BackendError> {
        if circuit.num_qubits() == 0 && circuit.is_empty() {
            return Err(BackendError::EmptyCircuit);
        }
        let n = circuit.num_qubits();
        if n > MAX_MPS_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: n,
                limit: MAX_MPS_QUBITS,
            });
        }
        let mut state = MetalMpsState::product(&self.ctx, n);
        // The pending layer of disjoint NN 2q gates, and which sites it occupies.
        let mut layer: Vec<LayerGate> = Vec::new();
        let mut busy = vec![false; n as usize];
        let reset = |layer: &mut Vec<LayerGate>, busy: &mut [bool]| {
            layer.clear();
            busy.iter_mut().for_each(|b| *b = false);
        };

        for inst in circuit.instructions() {
            match inst {
                Instruction::Gate(g) => match self.gate_matrix(n, g)? {
                    GateMatrix::M2x2(m) => {
                        // 1q gates apply in place (no SVD); flush so the pending
                        // layer's reads of this site happen in program order.
                        self.flush_layer(&mut state, &layer)?;
                        reset(&mut layer, &mut busy);
                        self.apply_1q(&state, g.qubits[0] as usize, &m);
                    }
                    GateMatrix::M4x4(m) => {
                        let (qa, qb) = (g.qubits[0], g.qubits[1]);
                        if qa.abs_diff(qb) != 1 {
                            // Non-NN: flush the pending layer, then route via SWAPs
                            // (gate-by-gate; the SWAP network can't join a batch).
                            self.flush_layer(&mut state, &layer)?;
                            reset(&mut layer, &mut busy);
                            self.apply_2q_routed(&mut state, qa as usize, qb as usize, &m)?;
                            continue;
                        }
                        let i = qa.min(qb) as usize;
                        // A site conflict with the pending layer ⇒ flush first.
                        if busy[i] || busy[i + 1] {
                            self.flush_layer(&mut state, &layer)?;
                            reset(&mut layer, &mut busy);
                        }
                        busy[i] = true;
                        busy[i + 1] = true;
                        layer.push(LayerGate { qa, qb, m });
                    }
                    GateMatrix::M8x8(_) => {
                        return Err(BackendError::UnsupportedInstruction {
                            kind: "3q gate (MPS scaffold)",
                        });
                    }
                },
                // A barrier forbids reordering across it: flush the pending layer.
                Instruction::Barrier(_) => {
                    self.flush_layer(&mut state, &layer)?;
                    reset(&mut layer, &mut busy);
                }
                Instruction::Reset(_) => {
                    return Err(BackendError::UnsupportedInstruction { kind: "reset" })
                }
                Instruction::Measure { .. } => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "measure (MPS scaffold)",
                    })
                }
                Instruction::DiagonalPhase(_) => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "fused DiagonalPhase block (MPS scaffold; run un-optimized)",
                    })
                }
                Instruction::TiledBlock(_) => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "fused TiledBlock (MPS scaffold)",
                    })
                }
            }
        }
        self.flush_layer(&mut state, &layer)?;
        Ok(state)
    }

    /// Apply one layer of disjoint NN 2q gates with a single batched SVD dispatch.
    /// Phase A contracts + gate-applies every block onto one command buffer (one
    /// `commit`/`wait`); phase B factors all blocks in one batched-Jacobi dispatch;
    /// phase C refuses the layer if *any* block would truncate (before mutating
    /// `state`, so a refusal leaves the pre-layer MPS intact — P5.6-02), else
    /// installs the new site tensors. A no-op for an empty layer.
    fn flush_layer(
        &mut self,
        state: &mut MetalMpsState,
        layer: &[LayerGate],
    ) -> Result<(), BackendError> {
        if layer.is_empty() {
            return Ok(());
        }

        // --- Phase A: contract + gate-apply for the whole layer, one cmd buffer ---
        let gpu_start = std::time::Instant::now();
        let cmd = self.ctx.queue().new_command_buffer();
        let mut blks: Vec<LayerBlock> = Vec::with_capacity(layer.len());
        let mut thetas: Vec<DeviceBuffer<Complex<f32>>> = Vec::with_capacity(layer.len());
        // Per-gate 4×4 buffers; must outlive the GPU work, so kept until the wait.
        let mut mats: Vec<DeviceBuffer<Complex<f32>>> = Vec::with_capacity(layer.len());
        for gate in layer {
            let i = gate.qa.min(gate.qb) as usize;
            let j = i + 1;
            let i_is_msb = (gate.qa as usize) == i;
            let li = state.sites[i].left;
            let ci = state.sites[i].right;
            debug_assert_eq!(
                ci, state.sites[j].left,
                "MPS bond mismatch at sites {i},{j}"
            );
            let ri = state.sites[j].right;
            let rows = li * 2;
            let cols = 2 * ri;

            let theta = DeviceBuffer::from_slice(
                &self.ctx,
                &vec![Complex::<f32>::new(0.0, 0.0); rows * cols],
            );
            let cmeta = ContractMeta {
                c: ci as u32,
                ri: ri as u32,
                _pad0: 0,
                _pad1: 0,
            };
            self.encode_contract(
                cmd,
                &state.sites[i].buf,
                &state.sites[j].buf,
                &theta,
                &cmeta,
                (rows * cols) as u64,
            );

            let mut mat_data = [Complex::<f32>::new(0.0, 0.0); 16];
            #[allow(clippy::needless_range_loop)]
            for r in 0..4 {
                for c in 0..4 {
                    mat_data[r * 4 + c] = narrow(gate.m[r][c]);
                }
            }
            let mat = DeviceBuffer::from_slice(&self.ctx, &mat_data);
            let ameta = Apply2qMeta {
                ri: ri as u32,
                i_is_msb: i_is_msb as u32,
                _pad0: 0,
                _pad1: 0,
            };
            self.encode_apply2q(cmd, &theta, &mat, &ameta, (li * ri) as u64);

            blks.push(LayerBlock {
                i,
                li,
                ri,
                rows,
                cols,
            });
            thetas.push(theta);
            mats.push(mat);
        }
        cmd.commit();
        cmd.wait_until_completed();
        self.gpu_ns += gpu_start.elapsed().as_nanos();

        // --- Phase B: one batched-Jacobi dispatch for every block in the layer ---
        let svd_start = std::time::Instant::now();
        let batch: Vec<BatchBlock> = thetas
            .iter()
            .zip(blks.iter())
            .map(|(t, b)| BatchBlock {
                theta: t.as_slice(),
                rows: b.rows,
                cols: b.cols,
            })
            .collect();
        let splits = gpu_svd_split_batch(
            &self.ctx,
            &self.pipeline_jacobi_batched,
            &batch,
            self.max_bond,
        );

        // --- Phase C: resolve each split, refuse on truncation, then install ---
        let mut resolved: Vec<(LayerBlock, SplitOut)> = Vec::with_capacity(blks.len());
        for (idx, split) in splits.into_iter().enumerate() {
            let b = blks[idx];
            // GPU split, or the f64 CPU faer fallback for a non-finite GPU result
            // (the slice is current: the phase-A wait completed).
            let (chi, si, sj, trunc) = match split {
                Some(s) => s,
                None => svd_split(batch[idx].theta, b.rows, b.cols, self.max_bond)?,
            };
            resolved.push((b, (chi, si, sj, trunc)));
        }
        // Record every block's dropped weight, then refuse the layer if any block
        // is a real truncation — before mutating `state` (P5.6-02).
        for (_, (_, _, _, trunc)) in &resolved {
            self.trunc_error += trunc;
        }
        if let Some((_, (_, _, _, trunc))) = resolved.iter().find(|(_, s)| s.3 > MPS_TRUNC_TOL) {
            self.svd_ns += svd_start.elapsed().as_nanos();
            return Err(BackendError::MpsTruncationUnsupported {
                max_bond: self.max_bond,
                trunc_error: *trunc,
            });
        }
        for (b, (chi, si, sj, _)) in resolved {
            state.sites[b.i] = SiteTensor::from_host(&self.ctx, b.li, chi, &si);
            state.sites[b.i + 1] = SiteTensor::from_host(&self.ctx, chi, b.ri, &sj);
        }
        self.svd_ns += svd_start.elapsed().as_nanos();
        Ok(())
    }
}

/// One pending NN 2q gate in a batched layer: the qubit pair (for the
/// physical→matrix-index map) and its dense 4×4.
struct LayerGate {
    qa: u32,
    qb: u32,
    m: [[Complex<f64>; 4]; 4],
}

/// Book-keeping for one block of a flushed layer: left site index `i` and the
/// bond dims needed to rebuild the two site tensors after the batched split.
#[derive(Clone, Copy)]
struct LayerBlock {
    i: usize,
    li: usize,
    ri: usize,
    rows: usize,
    cols: usize,
}

/// `(chi, site_i_data, site_j_data, trunc_rel)` — the per-block split result, as
/// produced by the batched SVD or the CPU fallback.
type SplitOut = (usize, Vec<Complex<f32>>, Vec<Complex<f32>>, f64);

/// The 4×4 SWAP gate (exchanges `|01⟩ ↔ |10⟩`), symmetric in its two qubits, used
/// by the [`apply_2q_routed`](MetalMpsBackend::apply_2q_routed) SWAP network.
fn swap_matrix() -> [[Complex<f64>; 4]; 4] {
    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    [[o, z, z, z], [z, z, o, z], [z, o, z, z], [z, z, z, o]]
}

/// Narrow an f64 gate-matrix entry to f32 (matrices kept in f64; state is f32).
#[inline]
fn narrow(z: Complex<f64>) -> Complex<f32> {
    Complex::<f32>::new(z.re as f32, z.im as f32)
}

/// Map a foundation `Error` into the shared `BackendError` (all as `InvalidState`
/// so callers fail explicitly rather than silently returning a wrong result).
fn map_metal_err(e: Error) -> BackendError {
    match e {
        Error::NoDevice => BackendError::InvalidState {
            reason: "no Metal device available",
        },
        Error::ShaderCompile(_) => BackendError::InvalidState {
            reason: "Metal shader compilation failed",
        },
        Error::PipelineCreation(_) => BackendError::InvalidState {
            reason: "Metal pipeline creation failed",
        },
    }
}

impl Backend for MetalMpsBackend {
    type State = MetalMpsState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_MPS_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_MPS_QUBITS,
            });
        }
        Ok(MetalMpsState::product(&self.ctx, num_qubits))
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let matrix = self.gate_matrix(state.num_qubits, gate)?;
        match matrix {
            GateMatrix::M2x2(m) => {
                self.apply_1q(state, gate.qubits[0] as usize, &m);
                Ok(())
            }
            GateMatrix::M4x4(m) => {
                let qa = gate.qubits[0];
                let qb = gate.qubits[1];
                if qa.abs_diff(qb) == 1 {
                    self.apply_2q_nn(state, qa as usize, qb as usize, &m)
                } else {
                    // Non-adjacent: route with a physical SWAP network (P5.7-06).
                    self.apply_2q_routed(state, qa as usize, qb as usize, &m)
                }
            }
            GateMatrix::M8x8(_) => Err(BackendError::UnsupportedInstruction {
                kind: "3q gate (MPS scaffold)",
            }),
        }
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        readout::measure(state, qubit as usize, &mut self.rng)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        let n = state.num_qubits();
        // One shot packs one bitstring into a u64 (qubit q → bit q), so n ≤ 64.
        if n > 64 {
            return Err(BackendError::TooManyQubits {
                requested: n,
                limit: 64,
            });
        }
        Ok(readout::sample(state, shots, &mut self.rng))
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        readout::expectation(state, pauli)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        readout::probabilities(state, qubits)
    }
}
