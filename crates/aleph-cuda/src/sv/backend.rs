//! `CudaSvBackend` — the GPU state-vector [`Backend`]. Compiles the FP64 gate
//! kernels once at construction via NVRTC, then launches `apply_1q` / `apply_kq`
//! per gate on the device. Readout copies amplitudes back to the host.

use std::sync::Arc;

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, Gate, GateInstance, GateMatrix, PauliString};
use aleph_ir::{Circuit, DiagonalPhase, Instruction};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use rand::{rngs::StdRng, SeedableRng};

use crate::common::{
    control_mask, diagonal_of, flatten_kq, flatten_matrix, validate_and_extract, validate_kq,
};
use crate::sv::diag::{diag_1q_params, diag_kq_params, DiagKernels};
use crate::sv::kernel::{
    CnotParams, Gate1qParams, GateKqParams, Multi1qParams, APPLY_1Q, APPLY_1Q_MULTI, APPLY_CNOT,
    APPLY_KQ, APPLY_KQ_TILED, APPLY_PHASE_POLY, DEFAULT_LAYER_BATCH, MAX_LAYER_BATCH,
    SV_KERNELS_SRC,
};
use crate::sv::readout::GpuReadout;
use crate::sv::state::{CudaSvState, MAX_CUDA_QUBITS};
use crate::{CudaContext, DeviceBuffer, Error};

/// Threads per block. 256 is the standard sweet spot for memory-bound SV
/// kernels (NVIDIA cuStateVec, QuEST); occupancy is bandwidth- not block-bound.
const BLOCK: u32 = 256;

/// GPU state-vector backend (FP64).
pub struct CudaSvBackend {
    ctx: CudaContext,
    f_1q: CudaFunction,
    f_1q_multi: CudaFunction,
    f_cnot: CudaFunction,
    f_phase_poly: CudaFunction,
    f_kq: CudaFunction,
    f_kq_tiled: CudaFunction,
    // Keeps the loaded module alive for the lifetime of the functions.
    _module: Arc<CudaModule>,
    rng: StdRng,
    qubit_cap: u32,
    /// GPU-resident readout (P5-05): measurement / sampling / expectation /
    /// probabilities reduce on the device so only small results cross PCIe.
    readout: GpuReadout,
    /// Custom diagonal-gate kernels (P5-06).
    diag: DiagKernels,
    /// When set (default), diagonal gates divert to [`Self::diag`]; when cleared
    /// they fall back to the dense `apply_1q` / `apply_kq` path. The A/B switch
    /// for the P5-06 benchmark and the dual-path oracle test.
    custom_diag: bool,
    /// Disjoint-1q-layer batch width for [`Self::run_layered`] (P5.9-03),
    /// clamped to `1..=MAX_LAYER_BATCH`. Tunable for the A/B bench; 1 reproduces
    /// per-gate dispatch.
    layer_batch: usize,
    /// When set (default), a plain CNOT routes to the `apply_cnot` permutation
    /// kernel (P5.9-04) instead of the dense `apply_kq` 4×4 matvec. Cleared for
    /// the A/B baseline.
    custom_2q: bool,
    /// When set (default), a dense `k`-qubit block with `k >= tiled_min_k` routes
    /// to the warp-cooperative `apply_kq_tiled` kernel (P5.10-01) instead of the
    /// generic `apply_kq` (which spills `v[32]`/`gidx[32]` to local memory at
    /// k=4,5). Cleared for the A/B baseline.
    tiled_kq: bool,
    /// Smallest `k` routed to `apply_kq_tiled` when [`Self::tiled_kq`] is set
    /// (default 4 — the regime where generic `apply_kq` regresses; k≤3 keeps the
    /// proven generic path). Lower it to 2 to force the tiled kernel across all
    /// dense blocks for the P5.10-01 A/B benchmark.
    tiled_min_k: u32,
}

impl CudaSvBackend {
    /// Construct on device 0 with an entropy-seeded RNG. Returns
    /// [`Error::NoDevice`] on a GPU-less host so callers can skip cleanly.
    pub fn new() -> Result<Self, Error> {
        Self::build(StdRng::from_entropy())
    }

    /// Construct with an explicit seed; measurement/sampling are reproducible
    /// across processes for a given seed.
    pub fn with_seed(seed: u64) -> Result<Self, Error> {
        Self::build(StdRng::seed_from_u64(seed))
    }

    fn build(rng: StdRng) -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        // NVRTC compiles the CUDA C++ to PTX at runtime (mirrors a CPU JIT) —
        // no nvcc, no build-time CUDA SDK; the driver JITs PTX→sm at load.
        let ptx = compile_ptx(SV_KERNELS_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        let f_1q = module.load_function(APPLY_1Q)?;
        let f_1q_multi = module.load_function(APPLY_1Q_MULTI)?;
        let f_cnot = module.load_function(APPLY_CNOT)?;
        let f_phase_poly = module.load_function(APPLY_PHASE_POLY)?;
        let f_kq = module.load_function(APPLY_KQ)?;
        let f_kq_tiled = module.load_function(APPLY_KQ_TILED)?;
        let diag = DiagKernels::new(&ctx)?;
        let readout = GpuReadout::new(&ctx)?;
        Ok(Self {
            ctx,
            f_1q,
            f_1q_multi,
            f_cnot,
            f_phase_poly,
            f_kq,
            f_kq_tiled,
            _module: module,
            rng,
            qubit_cap: MAX_CUDA_QUBITS,
            readout,
            diag,
            custom_diag: true,
            layer_batch: DEFAULT_LAYER_BATCH,
            custom_2q: true,
            tiled_kq: true,
            tiled_min_k: 4,
        })
    }

    /// Enable (default) or disable routing plain CNOTs to the `apply_cnot`
    /// permutation kernel (P5.9-04). Disabling forces the dense `apply_kq` 4×4
    /// path — the baseline arm of the P5.9-04 A/B benchmark.
    pub fn with_custom_2q(mut self, on: bool) -> Self {
        self.custom_2q = on;
        self
    }

    /// Enable (default) or disable routing dense `k`-qubit blocks with
    /// `k >= tiled_min_k` to the warp-cooperative `apply_kq_tiled` kernel
    /// (P5.10-01). Disabling forces the generic `apply_kq` for every `k` — the
    /// baseline arm of the P5.10-01 A/B benchmark.
    pub fn with_tiled_kq(mut self, on: bool) -> Self {
        self.tiled_kq = on;
        self
    }

    /// Override the smallest `k` routed to `apply_kq_tiled` (clamped to `2..=5`).
    /// Default 4; set to 2 to force the tiled kernel across all dense blocks for
    /// the P5.10-01 A/B benchmark, or 6 to disable it without touching
    /// [`Self::tiled_kq`].
    pub fn with_tiled_min_k(mut self, k: u32) -> Self {
        self.tiled_min_k = k.clamp(2, 6);
        self
    }

    /// Override the disjoint-1q-layer batch width for [`Self::run_layered`]
    /// (clamped to `1..=MAX_LAYER_BATCH`). 1 reproduces per-gate dispatch — the
    /// baseline arm of the P5.9-03 A/B benchmark.
    pub fn with_layer_batch(mut self, batch: usize) -> Self {
        self.layer_batch = batch.clamp(1, MAX_LAYER_BATCH);
        self
    }

    /// Override the qubit cap (default [`MAX_CUDA_QUBITS`]). For large-memory
    /// benchmarks on a GPU that can hold the state.
    pub fn with_qubit_cap(mut self, cap: u32) -> Self {
        self.qubit_cap = cap;
        self
    }

    /// Enable (default) or disable routing diagonal gates to the custom
    /// `apply_diag` kernels (P5-06). Disabling forces the dense `apply_1q` /
    /// `apply_kq` path — the baseline arm of the P5-06 A/B benchmark.
    pub fn with_custom_kernels(mut self, on: bool) -> Self {
        self.custom_diag = on;
        self
    }

    /// Launch `apply_1q` over `2^(n-1)` amplitude pairs.
    fn launch_1q(&self, state: &mut CudaSvState, params: Gate1qParams) -> Result<(), Error> {
        let n_pairs: u64 = 1 << (state.num_qubits - 1);
        let cfg = launch_config(n_pairs);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: kernel signature is (cplx* amps, Gate1q g, u64 n_pairs); the
        // args below match in order and type, `amps` has 2·2^n f64 = 2^n cplx,
        // and the grid covers exactly `n_pairs` threads with an in-bounds guard.
        unsafe {
            stream
                .launch_builder(&self.f_1q)
                .arg(amps)
                .arg(&params)
                .arg(&n_pairs)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Upload `mat` (interleaved row-major `2^k×2^k`) to the reusable scratch and
    /// launch `apply_kq` over `2^(n-k)` groups.
    fn launch_kq(
        &self,
        state: &mut CudaSvState,
        params: GateKqParams,
        mat: &[f64],
    ) -> Result<(), Error> {
        match state.mat_scratch.as_mut() {
            Some(buf) => buf.write(&self.ctx, mat)?,
            None => state.mat_scratch = Some(DeviceBuffer::<f64>::from_slice(&self.ctx, mat)?),
        }
        // P5.10-01: above tiled_min_k the generic `apply_kq` regresses (its
        // v[32]/gidx[32] thread-local arrays spill); route those to the
        // warp-cooperative `apply_kq_tiled` kernel instead.
        if self.tiled_kq && params.k >= self.tiled_min_k {
            return self.launch_kq_tiled(state, params);
        }
        let n_groups: u64 = 1 << (state.num_qubits - params.k);
        let cfg = launch_config(n_groups);
        let stream = self.ctx.stream();
        // Disjoint field borrows: `amps` (&mut) and `mat_scratch` (&) are
        // different fields of `state`, so both borrows coexist.
        let amps = state.amps.slice_mut();
        let mat_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set above")
            .slice();
        // SAFETY: kernel signature is (cplx* amps, const cplx* mat, GateKq g,
        // u64 n_groups); args match in order/type. `mat_dev` holds 2^k·2^k cplx,
        // `amps` holds 2^n cplx, and the grid covers `n_groups` with a guard.
        unsafe {
            stream
                .launch_builder(&self.f_kq)
                .arg(amps)
                .arg(mat_dev)
                .arg(&params)
                .arg(&n_groups)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the warp-cooperative `apply_kq_tiled` kernel (P5.10-01) over all
    /// `2^n` amplitudes (one thread each). The `2^k × 2^k` matrix must already be
    /// uploaded to `state.mat_scratch` by the caller ([`Self::launch_kq`]). Unlike
    /// `apply_kq`, this keeps each amplitude in a register and does the per-block
    /// matvec as an intra-warp shuffle reduction, so k=4,5 blocks don't spill.
    fn launch_kq_tiled(&self, state: &mut CudaSvState, params: GateKqParams) -> Result<(), Error> {
        let n_amps: u64 = 1 << state.num_qubits;
        // Shared memory holds the whole 2^k×2^k matrix: dim*dim cplx = 16 bytes
        // each. At k=5 that is 32·32·16 = 16 KiB, well inside the per-block limit.
        let dim: u32 = 1 << params.k;
        let shared_bytes = (dim as usize) * (dim as usize) * std::mem::size_of::<[f64; 2]>();
        let cfg = launch_config_shared(n_amps, shared_bytes as u32);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        let mat_dev = state
            .mat_scratch
            .as_ref()
            .expect("scratch set by launch_kq")
            .slice();
        // SAFETY: kernel signature is (cplx* amps, const cplx* mat, GateKq g,
        // u64 n_amps); args match in order/type. `mat_dev` holds 2^k·2^k cplx,
        // `amps` holds 2^n cplx, the grid covers exactly `n_amps` threads with an
        // in-bounds guard, and `shared_bytes` matches the kernel's `dim*dim` cplx
        // dynamic-shared allocation. BLOCK is a multiple of 32 and dim divides 32
        // (k≤5), so every 2^k group is warp-local (shuffles stay intra-warp).
        unsafe {
            stream
                .launch_builder(&self.f_kq_tiled)
                .arg(amps)
                .arg(mat_dev)
                .arg(&params)
                .arg(&n_amps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch `apply_cnot` over the `2^(n-2)` control=1 amplitude pairs (P5.9-04).
    fn launch_cnot(&self, state: &mut CudaSvState, control: u32, target: u32) -> Result<(), Error> {
        let n_groups: u64 = 1 << (state.num_qubits - 2);
        let cfg = launch_config(n_groups);
        let stream = self.ctx.stream();
        let params = CnotParams {
            ctrl: control,
            targ: target,
            lo: control.min(target),
            hi: control.max(target),
        };
        let amps = state.amps.slice_mut();
        // SAFETY: kernel signature is (cplx* amps, Cnot g, u64 n_groups); args
        // match in order/type, `amps` holds 2^n cplx, and the grid covers exactly
        // `n_groups = 2^(n-2)` threads with an in-bounds guard. `control`/`target`
        // are validated `< num_qubits` and distinct by the caller.
        unsafe {
            stream
                .launch_builder(&self.f_cnot)
                .arg(amps)
                .arg(&params)
                .arg(&n_groups)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Apply a fused [`DiagonalPhase`] in one coalesced sweep (P5.9-06):
    /// `amps[x] *= exp(i·φ(x))`. Flattens the terms to CSR host arrays
    /// (`angles` / `conds` / `offsets`), uploads them, launches `apply_phase_poly`
    /// over all `2^n` amplitudes, then synchronises so the per-call upload buffers
    /// outlive the kernel. `DiagonalPhase` instructions are rare (one per fused
    /// cphase ladder), so the upload + sync is amortised over the whole sweep.
    fn launch_phase_poly(&self, state: &mut CudaSvState, dp: &DiagonalPhase) -> Result<(), Error> {
        let n_terms = dp.terms.len();
        if n_terms == 0 {
            return Ok(()); // empty polynomial ⇒ identity
        }
        // CSR encode: term `t` owns `conds[offsets[t]..offsets[t+1]]`.
        let mut angles: Vec<f64> = Vec::with_capacity(n_terms);
        let mut conds: Vec<u64> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(n_terms + 1);
        offsets.push(0);
        for t in &dp.terms {
            angles.push(t.angle);
            conds.extend(t.conds.iter().copied());
            offsets.push(conds.len() as u32);
        }
        // Every term may be a global phase (empty conds) ⇒ `conds` empty. Keep a
        // valid (non-null) device pointer; the kernel never indexes it then.
        if conds.is_empty() {
            conds.push(0);
        }

        let angles_dev = DeviceBuffer::from_slice(&self.ctx, &angles)?;
        let conds_dev = DeviceBuffer::from_slice(&self.ctx, &conds)?;
        let offsets_dev = DeviceBuffer::from_slice(&self.ctx, &offsets)?;

        let n_amps: u64 = 1 << state.num_qubits;
        let n_terms_u = n_terms as u32;
        let cfg = launch_config(n_amps);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: signature is (cplx* amps, const double* angles, const ull* conds,
        // const unsigned* offsets, unsigned n_terms, ull n_amps). Args match in
        // order/type; `amps` holds 2^n cplx; `angles`/`offsets` have `n_terms`/
        // `n_terms+1` elements and `conds` is non-empty; the grid covers `n_amps`
        // with an in-bounds guard.
        unsafe {
            stream
                .launch_builder(&self.f_phase_poly)
                .arg(amps)
                .arg(angles_dev.slice())
                .arg(conds_dev.slice())
                .arg(offsets_dev.slice())
                .arg(&n_terms_u)
                .arg(&n_amps)
                .launch(cfg)?;
        }
        // Block until the kernel finishes so the upload buffers (dropped at end of
        // scope) are not freed out from under it.
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Apply a diagonal gate via the custom `apply_diag` kernels (P5-06). `diag`
    /// is the interleaved `[re, im]` diagonal from [`diagonal_of`]; `qubits` are
    /// the operands in `gate.matrix()` MSB-first order; `controls` are external
    /// controls. The 1q case (Z/S/T/Rz/Phase and their controlled forms) skips
    /// the scratch upload entirely.
    fn launch_diag(
        &self,
        state: &mut CudaSvState,
        diag: &[f64],
        qubits: &[u32],
        controls: &[u32],
    ) -> Result<(), Error> {
        let ctrl_mask = control_mask(controls);
        if qubits.len() == 1 {
            let params = diag_1q_params(diag, qubits[0], ctrl_mask);
            self.diag.launch_1q(&self.ctx, state, params)
        } else {
            let params = diag_kq_params(qubits, ctrl_mask);
            self.diag.launch_kq(&self.ctx, state, params, diag)
        }
    }

    /// Launch `apply_1q_multi` for one batch of `m = params.m` disjoint 1q gates
    /// over `2^(n-m)` groups — one state sweep for the whole batch (P5.9-03).
    fn launch_1q_multi(&self, state: &mut CudaSvState, params: Multi1qParams) -> Result<(), Error> {
        let n_groups: u64 = 1 << (state.num_qubits - params.m);
        let cfg = launch_config(n_groups);
        let stream = self.ctx.stream();
        let amps = state.amps.slice_mut();
        // SAFETY: kernel signature is (cplx* amps, Multi1q g, u64 n_groups); args
        // match in order/type, `amps` holds 2^n cplx, and the grid covers exactly
        // `n_groups` threads with an in-bounds guard. `m ≤ 5` (enforced by the
        // batching in `apply_1q_layer`), so `num_qubits - m` never underflows for
        // an `m`-qubit-or-larger state.
        unsafe {
            stream
                .launch_builder(&self.f_1q_multi)
                .arg(amps)
                .arg(&params)
                .arg(&n_groups)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Apply a run of mutually-disjoint single-qubit gates, chunked into batches
    /// of ≤ [`MAX_LAYER_BATCH`] and dispatched one sweep each via
    /// `apply_1q_multi`. `gates` is `(qubit, 2×2 matrix)`; the caller guarantees
    /// the qubits are pairwise distinct (so the gates commute and any chunking /
    /// per-chunk reordering is exact).
    fn apply_1q_layer(
        &self,
        state: &mut CudaSvState,
        gates: &[(u32, [[Complex; 2]; 2])],
    ) -> Result<(), Error> {
        for chunk in gates.chunks(self.layer_batch) {
            // Sort the chunk's qubits ascending: the kernel inserts zero bits at
            // ascending positions and maps local bit j ↔ sorted[j], so `mats[j]`
            // must be the gate on the j-th smallest qubit.
            let mut idx: Vec<usize> = (0..chunk.len()).collect();
            idx.sort_unstable_by_key(|&i| chunk[i].0);

            let mut params = Multi1qParams {
                mats: [0.0; 40],
                m: chunk.len() as u32,
                sorted: [0u32; 5],
                _pad: [0u32; 2],
            };
            for (j, &i) in idx.iter().enumerate() {
                let (q, m) = chunk[i];
                params.sorted[j] = q;
                let base = j * 8;
                params.mats[base] = m[0][0].re;
                params.mats[base + 1] = m[0][0].im;
                params.mats[base + 2] = m[0][1].re;
                params.mats[base + 3] = m[0][1].im;
                params.mats[base + 4] = m[1][0].re;
                params.mats[base + 5] = m[1][0].im;
                params.mats[base + 6] = m[1][1].re;
                params.mats[base + 7] = m[1][1].im;
            }
            self.launch_1q_multi(state, params)?;
        }
        Ok(())
    }

    /// Run `circuit` with **disjoint-1q-layer batching** (P5.9-03): consecutive
    /// plain single-qubit gates on distinct qubits are applied in
    /// `⌈count / MAX_LAYER_BATCH⌉` state sweeps via `apply_1q_multi` instead of
    /// one sweep each. Every other instruction flushes the pending batch (so
    /// program order is preserved) and routes through the normal per-gate path.
    ///
    /// Oracle-equal to per-gate [`aleph_backend::run`] (pinned at 1e-10 in
    /// `tests/gpu_layer_oracle.rs`); the A/B speedup is in `tests/gpu_layer_bench.rs`.
    // The `flush!` macro resets `mask` after every batch; the final flush at
    // end-of-circuit makes that last reset dead, which is correct, not a bug.
    #[allow(unused_assignments)]
    pub fn run_layered(&mut self, circuit: &Circuit) -> Result<CudaSvState, BackendError> {
        if circuit.num_qubits() == 0 && circuit.is_empty() {
            return Err(BackendError::EmptyCircuit);
        }
        let mut state = self.allocate(circuit.num_qubits())?;
        // Pending disjoint 1q batch + a mask of the qubits it covers (num_qubits
        // ≤ 30 ≤ 64, so a u64 mask is always wide enough).
        let mut pending: Vec<(u32, [[Complex; 2]; 2])> = Vec::new();
        let mut mask: u64 = 0;

        macro_rules! flush {
            () => {
                if !pending.is_empty() {
                    self.apply_1q_layer(&mut state, &pending)
                        .map_err(to_backend_err)?;
                    pending.clear();
                    mask = 0;
                }
            };
        }

        for inst in circuit.instructions() {
            match inst {
                Instruction::Gate(g) => match batchable_1q(g) {
                    // A collision (qubit already pending) forces a flush first, so
                    // two gates on the same qubit never reorder.
                    Some(m) => {
                        let bit = 1u64 << g.qubits[0];
                        if mask & bit != 0 {
                            flush!();
                        }
                        mask |= bit;
                        pending.push((g.qubits[0], m));
                    }
                    None => {
                        flush!();
                        self.apply_gate(&mut state, g)?;
                    }
                },
                Instruction::Barrier(_) => flush!(),
                Instruction::Measure { qubit, .. } => {
                    flush!();
                    let _ = self.measure(&mut state, *qubit)?;
                }
                Instruction::Reset(_) => {
                    flush!();
                    return Err(BackendError::UnsupportedInstruction { kind: "reset" });
                }
                Instruction::DiagonalPhase(dp) => {
                    flush!();
                    self.apply_diagonal_phase(&mut state, dp)?;
                }
                Instruction::TiledBlock(tb) => {
                    flush!();
                    self.apply_tiled_block(&mut state, tb)?;
                }
            }
        }
        flush!();
        Ok(state)
    }
}

/// A plain single-qubit gate the layer batcher can fold into `apply_1q_multi`:
/// exactly one operand, no external controls, and a concrete 2×2 matrix.
/// Diagonal gates qualify too — inside `run_layered` the dense butterfly is
/// still correct, and batching the whole 1q sublayer beats per-gate dispatch
/// even when some members are diagonal.
fn batchable_1q(g: &GateInstance) -> Option<[[Complex; 2]; 2]> {
    if g.qubits.len() != 1 || !g.controls.is_empty() {
        return None;
    }
    // UnitaryKq is multi-qubit; everything else with arity 1 resolves to M2x2.
    if matches!(g.gate, Gate::UnitaryKq { .. }) {
        return None;
    }
    match g.gate.matrix() {
        Ok(GateMatrix::M2x2(m)) => Some(m),
        _ => None,
    }
}

/// `((n + BLOCK - 1) / BLOCK)` blocks of `BLOCK` threads, ≥1 block.
fn launch_config(n_threads: u64) -> LaunchConfig {
    launch_config_shared(n_threads, 0)
}

/// Like [`launch_config`] but with `shared_bytes` of dynamic shared memory per
/// block (for `apply_kq_tiled`'s shared-memory matrix tile, P5.10-01).
fn launch_config_shared(n_threads: u64, shared_bytes: u32) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: shared_bytes,
    }
}

/// Build a `Gate1qParams` from a 2×2 matrix and external controls.
fn gate_1q_params(m: &[[Complex; 2]; 2], target: u32, controls: &[u32]) -> Gate1qParams {
    Gate1qParams {
        m: [
            m[0][0].re, m[0][0].im, m[0][1].re, m[0][1].im, m[1][0].re, m[1][0].im, m[1][1].re,
            m[1][1].im,
        ],
        target,
        t_bit: 1u32 << target,
        ctrl_mask: control_mask(controls),
        _pad: 0,
    }
}

/// Build a `GateKqParams` from the target qubits and external controls.
///
/// `gate.matrix()` lays multi-qubit operands out **MSB-first**: `qubits[0]` is
/// the most-significant matrix-index bit, `qubits[k-1]` the least — the same
/// operand order the CPU kernels use (verified by the CPU-SV oracle; a CNOT on
/// `[control, target]` must swap basis indices, not no-op). So matrix-index bit
/// `b` (0 = LSB) corresponds to `qubits[k-1-b]`: `qbit[b] = 1 << qubits[k-1-b]`.
/// `sorted` (ascending target positions, operand-order-independent) drives the
/// kernel's zero-bit insertion.
fn gate_kq_params(qubits: &[u32], controls: &[u32]) -> GateKqParams {
    let k = qubits.len();
    let mut qbit = [0u32; 5];
    let mut sorted = [0u32; 5];
    for (b, slot) in qbit.iter_mut().take(k).enumerate() {
        *slot = 1u32 << qubits[k - 1 - b];
    }
    sorted[..k].copy_from_slice(qubits);
    sorted[..k].sort_unstable();
    GateKqParams {
        k: k as u32,
        qbit,
        sorted,
        ctrl_mask: control_mask(controls),
    }
}

/// Map a CUDA-layer error to a backend error. Launch/transfer failures on a
/// working GPU indicate an internal fault, not user input; richer plumbing is a
/// follow-up (the variant set is shared across all backends).
fn to_backend_err(_e: Error) -> BackendError {
    BackendError::InvalidState {
        reason: "CUDA backend failure (compile/launch/transfer)",
    }
}

impl Backend for CudaSvBackend {
    type State = CudaSvState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits == 0 {
            return Err(BackendError::InvalidState {
                reason: "zero-qubit state",
            });
        }
        if num_qubits > self.qubit_cap {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: self.qubit_cap,
            });
        }
        CudaSvState::allocate(&self.ctx, num_qubits).map_err(to_backend_err)
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        // P5.9-04: a plain CNOT is a permutation (swap the target pair where the
        // control is 1) — route it to `apply_cnot`, which touches only the
        // control=1 half of the state with zero FLOPs, instead of the dense
        // `apply_kq` 4×4 matvec over all 2^n amplitudes.
        if self.custom_2q
            && matches!(gate.gate, Gate::Cnot)
            && gate.controls.is_empty()
            && gate.qubits.len() == 2
        {
            let (c, t) = (gate.qubits[0], gate.qubits[1]);
            if c == t {
                return Err(BackendError::DuplicateQubit { qubit: c });
            }
            if c >= state.num_qubits {
                return Err(BackendError::QubitOutOfRange {
                    qubit: c,
                    num_qubits: state.num_qubits,
                });
            }
            if t >= state.num_qubits {
                return Err(BackendError::QubitOutOfRange {
                    qubit: t,
                    num_qubits: state.num_qubits,
                });
            }
            return self.launch_cnot(state, c, t).map_err(to_backend_err);
        }
        // P5.9-02: a fused `UnitaryKq` (k=4,5) has no fixed-size `GateMatrix`
        // (the enum stops at 8×8), so `validate_and_extract` would reject it. The
        // `apply_kq` kernel already handles k≤5; feed it the raw row-major slice.
        // qubits are MSB-first (qubits[0] = matrix-index MSB), the same order
        // `gate_kq_params` expects — identical to the `Unitary2q` path below.
        if let aleph_core::Gate::UnitaryKq { k, data } = &gate.gate {
            validate_kq(
                state.num_qubits,
                *k,
                data.len(),
                &gate.qubits,
                &gate.controls,
            )?;
            let params = gate_kq_params(&gate.qubits, &gate.controls);
            return self
                .launch_kq(state, params, &flatten_kq(data))
                .map_err(to_backend_err);
        }
        let matrix = validate_and_extract(state.num_qubits, gate)?;
        // P5-06: a diagonal gate is one coalesced in-place phase multiply —
        // divert it to the custom `apply_diag` kernels instead of the dense path.
        if self.custom_diag {
            if let Some(diag) = diagonal_of(&matrix) {
                return self
                    .launch_diag(state, &diag, &gate.qubits, &gate.controls)
                    .map_err(to_backend_err);
            }
        }
        match matrix {
            GateMatrix::M2x2(m) => {
                let params = gate_1q_params(&m, gate.qubits[0], &gate.controls);
                self.launch_1q(state, params).map_err(to_backend_err)
            }
            GateMatrix::M4x4(m) => {
                let params = gate_kq_params(&[gate.qubits[0], gate.qubits[1]], &gate.controls);
                self.launch_kq(state, params, &flatten_matrix(&m))
                    .map_err(to_backend_err)
            }
            GateMatrix::M8x8(m) => {
                let params = gate_kq_params(
                    &[gate.qubits[0], gate.qubits[1], gate.qubits[2]],
                    &gate.controls,
                );
                self.launch_kq(state, params, &flatten_matrix(&m))
                    .map_err(to_backend_err)
            }
        }
    }

    /// Apply a fused diagonal phase polynomial (P5.9-06) — the GPU analogue of
    /// the CPU SV path. Without this override the trait default rejects
    /// `DiagonalPhase`, so a `FuseDiagonalRuns`-fused circuit (QFT/QPE) could not
    /// run on the GPU.
    fn apply_diagonal_phase(
        &mut self,
        state: &mut Self::State,
        dp: &DiagonalPhase,
    ) -> Result<(), BackendError> {
        self.launch_phase_poly(state, dp).map_err(to_backend_err)
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        self.readout.measure(&mut self.rng, state, qubit)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        self.readout.sample(&mut self.rng, state, shots)
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        self.readout.expectation_value(state, pauli)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        self.readout.probabilities(state, qubits)
    }
}
