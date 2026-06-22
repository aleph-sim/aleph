//! GPU-resident readout (P5-05) shared by both CUDA state-vector backends.
//!
//! Measurement, sampling, expectation, and marginal probabilities all reduce the
//! device-resident state **on the GPU** and copy back only the small result — a
//! scalar, a `2^k` marginal vector, or `shots` indices — instead of downloading
//! the full `2^n` amplitudes (and, for measurement, re-uploading them). The
//! state vector stays on the device across the whole circuit; the only PCIe
//! crossings are the initial `|0…0⟩` setup and these final results.
//!
//! The algorithms mirror the former host implementation exactly (same degenerate
//! threshold, norm-drift budget, and inverse-CDF `partition_point` semantics) so
//! the distribution and expectation oracle tests agree with the CPU backend; the
//! cheap validation (qubit range / duplicate / Pauli checks) stays on the host
//! where it costs nothing.

use std::sync::Arc;

use aleph_backend::BackendError;
use aleph_core::{Pauli, PauliString};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use rand::{rngs::StdRng, Rng};

use crate::sv::state::CudaSvState;
use crate::{CudaContext, DeviceBuffer, Error};

/// Threads per block; must match `BLOCK` in `readout.cu` (the reduction kernels
/// size their shared memory to it).
const BLOCK: u32 = 256;

/// See `aleph_sv::measure::DEGENERATE_BRANCH_THRESHOLD`.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;

/// CUDA C++ source for the readout reduction kernels (compiled once via NVRTC).
const READOUT_SRC: &str = include_str!("readout.cu");

/// Compiled GPU readout kernels plus a small reusable accumulator. Both backends
/// hold one and delegate their `Backend` readout methods to it.
pub(crate) struct GpuReadout {
    ctx: CudaContext,
    f_branch: CudaFunction,
    f_expect: CudaFunction,
    f_collapse: CudaFunction,
    f_marginal: CudaFunction,
    f_final: CudaFunction,
    f_abs2: CudaFunction,
    f_scan: CudaFunction,
    f_search: CudaFunction,
    _module: Arc<CudaModule>,
    /// Two-element device accumulator (`[out0, out1]`) holding the final scalar
    /// reduction result.
    acc: DeviceBuffer<f64>,
    /// Per-block subtotals for the two-pass reductions (`2 · n_blocks` f64),
    /// grown on demand. No global atomics, so no cross-block contention.
    partials: Option<DeviceBuffer<f64>>,
}

impl GpuReadout {
    /// Compile the readout kernels on `ctx`'s device.
    pub(crate) fn new(ctx: &CudaContext) -> Result<Self, Error> {
        let ptx = compile_ptx(READOUT_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        let f_branch = module.load_function("reduce_abs2_branch")?;
        let f_expect = module.load_function("expect_pauli")?;
        let f_final = module.load_function("final_reduce2")?;
        let f_collapse = module.load_function("collapse")?;
        let f_marginal = module.load_function("marginal")?;
        let f_abs2 = module.load_function("abs2_into")?;
        let f_scan = module.load_function("scan_step")?;
        let f_search = module.load_function("sample_search")?;
        let acc = DeviceBuffer::<f64>::zeros(ctx, 2)?;
        Ok(Self {
            ctx: ctx.clone(),
            f_branch,
            f_expect,
            f_final,
            f_collapse,
            f_marginal,
            f_abs2,
            f_scan,
            f_search,
            _module: module,
            acc,
            partials: None,
        })
    }

    /// Number of `BLOCK`-sized blocks covering `n` elements.
    fn n_blocks(n: u64) -> u64 {
        n.div_ceil(BLOCK as u64).max(1)
    }

    /// Ensure `partials` holds at least `2 · n_blocks` f64.
    fn ensure_partials(&mut self, n_blocks: u64) -> Result<(), Error> {
        let need = (2 * n_blocks) as usize;
        let ok = self.partials.as_ref().is_some_and(|b| b.len() >= need);
        if !ok {
            self.partials = Some(DeviceBuffer::<f64>::zeros(&self.ctx, need)?);
        }
        Ok(())
    }

    /// Second pass: collapse the per-block pairs in `partials` (for `n_blocks`
    /// blocks) to the scalar `(out[0], out[1])`, returned after the only small
    /// device→host copy (16 bytes).
    fn finalize(&mut self, n_blocks: u64) -> Result<(f64, f64), Error> {
        self.acc.write(&self.ctx, &[0.0, 0.0])?;
        let stream = self.ctx.stream().clone();
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let partials = self.partials.as_ref().expect("partials set").slice();
        let acc = self.acc.slice_mut();
        // SAFETY: kernel (const double* partials, u64 m, double* out[2]); a single
        // block grid-strides over `n_blocks` pairs; `acc` has 2 f64.
        unsafe {
            stream
                .launch_builder(&self.f_final)
                .arg(partials)
                .arg(&n_blocks)
                .arg(acc)
                .launch(cfg)?;
        }
        let out = self.acc.to_vec(&self.ctx)?;
        Ok((out[0], out[1]))
    }

    /// `(Σ|aᵢ|², Σ_{i&qbit≠0}|aᵢ|²)`. Pass `qbit = 0` for the total alone.
    fn reduce_branch(&mut self, state: &CudaSvState, qbit: u64) -> Result<(f64, f64), Error> {
        let n: u64 = 1 << state.num_qubits;
        let nb = Self::n_blocks(n);
        self.ensure_partials(nb)?;
        let stream = self.ctx.stream().clone();
        let cfg = launch_cfg(n);
        {
            let amps = state.amps.slice();
            let partials = self.partials.as_mut().expect("partials set").slice_mut();
            // SAFETY: kernel (const cplx*, double* partials, u64 N, u64 qbit); args
            // match; `partials` holds 2·n_blocks f64, grid covers N.
            unsafe {
                stream
                    .launch_builder(&self.f_branch)
                    .arg(amps)
                    .arg(partials)
                    .arg(&n)
                    .arg(&qbit)
                    .launch(cfg)?;
            }
        }
        self.finalize(nb)
    }

    /// Validate finiteness + normalization from a GPU-computed total norm².
    fn check_norm(&mut self, state: &CudaSvState) -> Result<f64, BackendError> {
        let (total, _) = self.reduce_branch(state, 0).map_err(to_be)?;
        if !total.is_finite() {
            return Err(BackendError::InvalidState {
                reason: "non-finite state norm²",
            });
        }
        let n = (1u64 << state.num_qubits) as f64;
        let drift = n.sqrt() * aleph_core::AMPLITUDE_TOL;
        if (total - 1.0).abs() > drift {
            return Err(BackendError::InvalidState {
                reason: "state norm² deviates from 1 beyond drift budget",
            });
        }
        Ok(total)
    }

    /// Collapse onto the measured branch of `qubit`, returning the outcome. The
    /// branch probabilities are reduced on the GPU and the collapse runs in
    /// place on the device — the state never crosses PCIe.
    pub(crate) fn measure(
        &mut self,
        rng: &mut StdRng,
        state: &mut CudaSvState,
        qubit: u32,
    ) -> Result<bool, BackendError> {
        let nq = state.num_qubits;
        if qubit >= nq {
            return Err(BackendError::QubitOutOfRange {
                qubit,
                num_qubits: nq,
            });
        }
        let qbit = 1u64 << qubit;
        let (total, p1) = self.reduce_branch(state, qbit).map_err(to_be)?;
        if !total.is_finite() {
            return Err(BackendError::InvalidState {
                reason: "non-finite state norm²",
            });
        }
        let n = (1u64 << nq) as f64;
        let drift = n.sqrt() * aleph_core::AMPLITUDE_TOL;
        if (total - 1.0).abs() > drift {
            return Err(BackendError::InvalidState {
                reason: "state norm² deviates from 1 beyond drift budget",
            });
        }
        let p1 = p1.clamp(0.0, 1.0);
        let p0 = (total - p1).clamp(0.0, 1.0);
        let one_degen = p1 < DEGENERATE_BRANCH_THRESHOLD;
        let zero_degen = p0 < DEGENERATE_BRANCH_THRESHOLD;
        let (outcome, p) = match (zero_degen, one_degen) {
            (true, true) => {
                return Err(BackendError::DegenerateMeasurement {
                    qubit,
                    probability: p1.max(p0),
                });
            }
            (true, false) => (true, p1),
            (false, true) => (false, p0),
            (false, false) => {
                let outcome = rng.gen::<f64>() < p1;
                (outcome, if outcome { p1 } else { p0 })
            }
        };
        let scale = 1.0 / p.sqrt();
        self.collapse(state, qbit, outcome, scale).map_err(to_be)?;
        Ok(outcome)
    }

    /// In-place GPU collapse: keep the `outcome` branch (scaled), zero the other.
    fn collapse(
        &self,
        state: &mut CudaSvState,
        qbit: u64,
        outcome: bool,
        scale: f64,
    ) -> Result<(), Error> {
        let n: u64 = 1 << state.num_qubits;
        let stream = self.ctx.stream().clone();
        let cfg = launch_cfg(n);
        let oc: u32 = outcome as u32;
        let amps = state.amps.slice_mut();
        // SAFETY: kernel (cplx* amps, u64 N, u64 qbit, u32 outcome, f64 scale);
        // args match. In-place over 2^n amplitudes, grid covers N.
        unsafe {
            stream
                .launch_builder(&self.f_collapse)
                .arg(amps)
                .arg(&n)
                .arg(&qbit)
                .arg(&oc)
                .arg(&scale)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `⟨ψ| c·P |ψ⟩` reduced on the GPU; only the scalar crosses PCIe.
    pub(crate) fn expectation_value(
        &mut self,
        state: &CudaSvState,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        if !pauli.coefficient.is_finite() {
            return Err(BackendError::InvalidPauliString {
                reason: "non-finite coefficient",
            });
        }
        let nq = state.num_qubits;
        let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for (q, _) in &pauli.terms {
            if *q >= nq {
                return Err(BackendError::QubitOutOfRange {
                    qubit: *q,
                    num_qubits: nq,
                });
            }
            if seen.contains(q) {
                return Err(BackendError::DuplicateQubit { qubit: *q });
            }
            seen.push(*q);
        }
        self.check_norm(state)?;

        // P|i⟩ = i^numY · (-1)^popcount(i & sign_mask) |i ⊕ flip⟩.
        let mut flip: u64 = 0;
        let mut sign_mask: u64 = 0;
        let mut num_y: u32 = 0;
        for (q, p) in &pauli.terms {
            match p {
                Pauli::I => {}
                Pauli::X => flip |= 1 << q,
                Pauli::Z => sign_mask |= 1 << q,
                Pauli::Y => {
                    flip |= 1 << q;
                    sign_mask |= 1 << q;
                    num_y += 1;
                }
            }
        }

        let n: u64 = 1 << nq;
        let nb = Self::n_blocks(n);
        self.ensure_partials(nb).map_err(to_be)?;
        let stream = self.ctx.stream().clone();
        let cfg = launch_cfg(n);
        {
            let amps = state.amps.slice();
            let partials = self.partials.as_mut().expect("partials set").slice_mut();
            // SAFETY: kernel (const cplx*, double* partials, u64 N, u64 flip, u64 sign).
            unsafe {
                stream
                    .launch_builder(&self.f_expect)
                    .arg(amps)
                    .arg(partials)
                    .arg(&n)
                    .arg(&flip)
                    .arg(&sign_mask)
                    .launch(cfg)
                    .map_err(|e| to_be(Error::Driver(e)))?;
            }
        }
        let (sre, sim) = self.finalize(nb).map_err(to_be)?;
        // Re(i^numY · S): fold the global phase i^numY in on the host.
        let real = match num_y & 3 {
            0 => sre,
            1 => -sim,
            2 => -sre,
            _ => sim,
        };
        Ok(pauli.coefficient * real)
    }

    /// Marginal probabilities over `qubits` (length `2^|qubits|`, `qubits[0]` =
    /// LSB), accumulated on the GPU; only the small marginal crosses PCIe.
    pub(crate) fn probabilities(
        &mut self,
        state: &CudaSvState,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        let nq = state.num_qubits;
        let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for &q in qubits {
            if q >= nq {
                return Err(BackendError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: nq,
                });
            }
            if seen.contains(&q) {
                return Err(BackendError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        self.check_norm(state)?;
        if qubits.is_empty() {
            return Ok(vec![1.0]);
        }
        let k = qubits.len();
        let out_dim = 1usize << k;
        let n: u64 = 1 << nq;
        let positions = DeviceBuffer::<u32>::from_slice(&self.ctx, qubits).map_err(to_be)?;
        let mut bins = DeviceBuffer::<f64>::zeros(&self.ctx, out_dim).map_err(to_be)?;
        let stream = self.ctx.stream().clone();
        let cfg = launch_cfg(n);
        let kk = k as u32;
        {
            let amps = state.amps.slice();
            let out = bins.slice_mut();
            let pos = positions.slice();
            // SAFETY: kernel (const cplx*, double* out, u64 N, const u32* pos, u32 k);
            // `out` has 2^k entries, `pos` has k entries, grid covers N.
            unsafe {
                stream
                    .launch_builder(&self.f_marginal)
                    .arg(amps)
                    .arg(out)
                    .arg(&n)
                    .arg(pos)
                    .arg(&kk)
                    .launch(cfg)
                    .map_err(|e| to_be(Error::Driver(e)))?;
            }
        }
        let mut v = bins.to_vec(&self.ctx).map_err(to_be)?;
        v.truncate(out_dim);
        Ok(v)
    }

    /// Draw `shots` basis-state samples via an inverse-CDF built entirely on the
    /// GPU (|a|² → Hillis-Steele inclusive scan → per-shot upper-bound search);
    /// only the `shots` indices cross PCIe.
    pub(crate) fn sample(
        &mut self,
        rng: &mut StdRng,
        state: &CudaSvState,
        shots: u32,
    ) -> Result<Vec<u64>, BackendError> {
        let total = self.check_norm(state)?;
        if shots == 0 {
            return Ok(Vec::new());
        }
        let nq = state.num_qubits;
        let n: u64 = 1 << nq;
        let stream = self.ctx.stream().clone();
        let cfg = launch_cfg(n);

        // probs in `a`; ping-pong scan between `a` and `b` into the CDF.
        let mut a = DeviceBuffer::<f64>::zeros(&self.ctx, n as usize).map_err(to_be)?;
        let mut b = DeviceBuffer::<f64>::zeros(&self.ctx, n as usize).map_err(to_be)?;
        {
            let amps = state.amps.slice();
            let probs = a.slice_mut();
            // SAFETY: kernel (const cplx*, double* probs, u64 N); both length 2^n.
            unsafe {
                stream
                    .launch_builder(&self.f_abs2)
                    .arg(amps)
                    .arg(probs)
                    .arg(&n)
                    .launch(cfg)
                    .map_err(|e| to_be(Error::Driver(e)))?;
            }
        }
        let mut src_is_a = true;
        let mut d: u64 = 1;
        while d < n {
            // SAFETY: kernel (const double* in, double* out, u64 N, u64 d); `in`
            // and `out` are distinct buffers of length 2^n, grid covers N.
            if src_is_a {
                let (inb, outb) = (a.slice(), b.slice_mut());
                unsafe {
                    stream
                        .launch_builder(&self.f_scan)
                        .arg(inb)
                        .arg(outb)
                        .arg(&n)
                        .arg(&d)
                        .launch(cfg)
                        .map_err(|e| to_be(Error::Driver(e)))?;
                }
            } else {
                let (inb, outb) = (b.slice(), a.slice_mut());
                unsafe {
                    stream
                        .launch_builder(&self.f_scan)
                        .arg(inb)
                        .arg(outb)
                        .arg(&n)
                        .arg(&d)
                        .launch(cfg)
                        .map_err(|e| to_be(Error::Driver(e)))?;
                }
            }
            src_is_a = !src_is_a;
            d <<= 1;
        }

        // Targets r·total (r ∈ [0,1)); upper-bound search matches the CPU
        // sampler's `partition_point(|&c| c <= r)`.
        let targets: Vec<f64> = (0..shots).map(|_| rng.gen::<f64>() * total).collect();
        let targets_dev = DeviceBuffer::<f64>::from_slice(&self.ctx, &targets).map_err(to_be)?;
        let mut out_dev = DeviceBuffer::<u64>::zeros(&self.ctx, shots as usize).map_err(to_be)?;
        let shots_u: u64 = shots as u64;
        let scfg = launch_cfg(shots_u);
        {
            let cdf = if src_is_a { a.slice() } else { b.slice() };
            let tgt = targets_dev.slice();
            let out = out_dev.slice_mut();
            // SAFETY: kernel (const double* cdf, u64 N, const double* targets,
            // u64 shots, u64* out); cdf len 2^n, targets/out len `shots`.
            unsafe {
                stream
                    .launch_builder(&self.f_search)
                    .arg(cdf)
                    .arg(&n)
                    .arg(tgt)
                    .arg(&shots_u)
                    .arg(out)
                    .launch(scfg)
                    .map_err(|e| to_be(Error::Driver(e)))?;
            }
        }
        let mut v = out_dev.to_vec(&self.ctx).map_err(to_be)?;
        v.truncate(shots as usize);
        Ok(v)
    }
}

/// `⌈n / BLOCK⌉` blocks of `BLOCK` threads (≥1), one thread per element.
fn launch_cfg(n: u64) -> LaunchConfig {
    let blocks = n.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Map a CUDA-layer error to a backend error (same shape as the backends use).
fn to_be(_e: Error) -> BackendError {
    BackendError::InvalidState {
        reason: "CUDA readout failure (launch/transfer)",
    }
}
