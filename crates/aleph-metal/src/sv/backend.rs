//! `MetalSvBackend` — device-resident FP32 state-vector backend (P5.5-02).
//! Mirrors `Fp32SvBackend` over a Metal GPU buffer. The f64 `NaiveSvBackend`
//! remains the oracle reference; this is a GPU mode at f32 accuracy.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateError, GateInstance, GateMatrix, PauliString, AMPLITUDE_TOL};
use aleph_ir::Circuit;
use metal::{ComputePipelineState, MTLSize};
use rand::{rngs::StdRng, SeedableRng};
use std::ffi::c_void;

use super::kernel::{
    DiagMeta, DiagTermDesc, Gate1q, GateKqMeta, SV_1Q_ENTRY, SV_1Q_SRC, SV_DIAG_ENTRY, SV_DIAG_SRC,
    SV_KQ_ENTRY, SV_KQ_SRC,
};
use super::state::{MetalSvState, BATCH_CAP};
use crate::{DeviceBuffer, Error, MetalContext};

/// Soft qubit cap — the project-wide 28-qubit software limit, matching the CPU
/// backends. At 8 B/amp the memory ceiling is higher, but 28 binds first.
pub(crate) const MAX_METAL_QUBITS: u32 = 28;

/// Opt-in single-precision Metal GPU state-vector backend.
pub struct MetalSvBackend {
    ctx: MetalContext,
    // Used by the apply_gate dispatch (Task 5).
    pipeline_1q: ComputePipelineState,
    pipeline_kq: ComputePipelineState,
    pipeline_diag: ComputePipelineState,
    // Used by the host-side readout / measure / sample in Task 6.
    rng: StdRng,
}

impl MetalSvBackend {
    /// Construct with an entropy-seeded RNG.
    ///
    /// Acquires the system-default Metal device and compiles+caches the gate
    /// pipelines once. Returns [`BackendError::InvalidState`] when no device is
    /// present (headless CI) or a shader/pipeline build fails — unlike the
    /// infallible CPU `new`, GPU acquisition can fail, so this returns `Result`.
    pub fn new() -> Result<Self, BackendError> {
        Self::build(StdRng::from_entropy())
    }

    /// Construct with an explicit seed; host-side `measure`/`sample` are then
    /// reproducible across processes and machines for a given seed.
    pub fn with_seed(seed: u64) -> Result<Self, BackendError> {
        Self::build(StdRng::seed_from_u64(seed))
    }

    /// Execute `circuit` gate-by-gate (unfused) on the GPU, returning the
    /// device-resident final state. Thin wrapper over [`aleph_backend::run`];
    /// the verbatim reference path with no IR optimization.
    ///
    /// Gates batch into one command buffer (P5.6-04); the trailing `sync` flushes
    /// that batch so the returned state is fully computed — one wait per circuit,
    /// not per gate. (The state also self-syncs on first read, so this is just
    /// belt-and-suspenders for a caller that forwards it without reading.)
    pub fn run(&mut self, circuit: &Circuit) -> Result<MetalSvState, BackendError> {
        let state = aleph_backend::run(self, circuit)?;
        state.sync();
        Ok(state)
    }

    /// Optimize `circuit` with the default IR pipeline (cancellation, DCE,
    /// 1q/2q/kq fusion) and execute it on the GPU. The fused dense blocks ride
    /// the `apply_kq` kernel, cutting the number of GPU dispatches; any qubit
    /// permutation the pipeline introduces is settled via `unpermute_state`, so
    /// the returned state is in logical qubit order and directly comparable to
    /// [`run`](Self::run). Thin wrapper over [`aleph_backend::run_optimized`].
    ///
    /// The trailing `sync` flushes the final gate batch (see [`run`](Self::run)).
    pub fn run_optimized(&mut self, circuit: &Circuit) -> Result<MetalSvState, BackendError> {
        let state = aleph_backend::run_optimized(self, circuit)?;
        state.sync();
        Ok(state)
    }

    fn build(rng: StdRng) -> Result<Self, BackendError> {
        let ctx = MetalContext::new().map_err(map_metal_err)?;
        let pipeline_1q = ctx
            .make_compute_pipeline(SV_1Q_SRC, SV_1Q_ENTRY)
            .map_err(map_metal_err)?;
        let pipeline_kq = ctx
            .make_compute_pipeline(SV_KQ_SRC, SV_KQ_ENTRY)
            .map_err(map_metal_err)?;
        let pipeline_diag = ctx
            .make_compute_pipeline(SV_DIAG_SRC, SV_DIAG_ENTRY)
            .map_err(map_metal_err)?;
        Ok(Self {
            ctx,
            pipeline_1q,
            pipeline_kq,
            pipeline_diag,
            rng,
        })
    }

    /// Encode one 1q-kernel dispatch over `2^(n-1)` pairs into the state's open
    /// batch — no commit/wait (P5.6-04). Metal's automatic hazard tracking on the
    /// shared `amps` buffer serialises this dispatch after the previous gate, so
    /// the un-waited batch is still applied in program order.
    fn encode_1q(&self, state: &MetalSvState, g: &Gate1q) {
        let pairs = 1u64 << (state.num_qubits - 1); // num_qubits ≥ 1 here
        state.encode(self.ctx.queue(), |encoder| {
            encoder.set_compute_pipeline_state(&self.pipeline_1q);
            encoder.set_buffer(0, Some(state.amps.metal_buffer()), 0);
            // Metal copies the uniform block synchronously inside set_bytes, so
            // the stack-local `g` need not outlive this dispatch even un-waited.
            encoder.set_bytes(
                1,
                std::mem::size_of::<Gate1q>() as u64,
                g as *const Gate1q as *const c_void,
            );
            let tg = self
                .pipeline_1q
                .max_total_threads_per_threadgroup()
                .min(pairs);
            encoder.dispatch_threads(MTLSize::new(pairs, 1, 1), MTLSize::new(tg, 1, 1));
        });
        self.maybe_flush(state);
    }

    /// Encode one generic dense-kq dispatch over `2^(n-k)` groups into the open
    /// batch. The row-major f32 `matrix` (length `4^k`) is uploaded into its own
    /// slot of the state's reusable kq arena (distinct byte offset per dispatch),
    /// so batched kq dispatches never race over a shared region yet pay no
    /// per-gate buffer allocation (P5.6-04 AC #1).
    fn encode_kq(&self, state: &MetalSvState, meta: &GateKqMeta, matrix: &[Complex<f32>]) {
        let groups = 1u64 << (state.num_qubits - meta.k); // num_qubits >= k
        let mat_offset = state.kq_upload(&self.ctx, matrix);
        state.encode(self.ctx.queue(), |encoder| {
            encoder.set_compute_pipeline_state(&self.pipeline_kq);
            encoder.set_buffer(0, Some(state.amps.metal_buffer()), 0);
            encoder.set_buffer(1, Some(state.kq_pool_buffer().metal_buffer()), mat_offset);
            encoder.set_bytes(
                2,
                std::mem::size_of::<GateKqMeta>() as u64,
                meta as *const GateKqMeta as *const c_void,
            );
            let tg = self
                .pipeline_kq
                .max_total_threads_per_threadgroup()
                .min(groups);
            encoder.dispatch_threads(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        });
        self.maybe_flush(state);
    }

    /// Bound the open batch: once `BATCH_CAP` dispatches are queued, drain so the
    /// command buffer and the live per-dispatch kq buffers stay bounded on deep
    /// circuits. Correctness is unaffected — `sync` only adds a wait.
    fn maybe_flush(&self, state: &MetalSvState) {
        if state.batch_len() >= BATCH_CAP {
            state.sync();
        }
    }

    /// Build the [`GateKqMeta`] for a dense gate on `targets` (logical/MSB order,
    /// `targets[0]` is the matrix-index MSB) with external `controls`. `targets`
    /// has at most 5 entries (k ≤ 5).
    fn kq_meta(targets: &[u32], controls: &[u32]) -> GateKqMeta {
        let k = targets.len();
        let mut sorted = [0u32; 5];
        let mut tbit = [0u32; 5];
        for (j, &q) in targets.iter().enumerate() {
            tbit[j] = 1u32 << q; // logical/MSB order for matrix-index bits
        }
        // sorted[] = ascending target positions for the kernel's zero-bit
        // insertion. Plain `[u32; 5]` copy+sort — no extra dependency.
        let mut s = [0u32; 5];
        s[..k].copy_from_slice(targets);
        s[..k].sort_unstable();
        sorted[..k].copy_from_slice(&s[..k]);
        let ctrl_mask = controls.iter().fold(0u32, |acc, &c| acc | (1u32 << c));
        GateKqMeta {
            k: k as u32,
            sorted,
            tbit,
            ctrl_mask,
        }
    }
}

/// Narrow an f64 gate-matrix entry to f32. Matrices are materialised in f64
/// (angles keep full precision); only the state is single-precision.
#[inline]
fn narrow(z: Complex<f64>) -> Complex<f32> {
    Complex::<f32>::new(z.re as f32, z.im as f32)
}

/// Max deviation of `M†M` from the identity for an N×N matrix. NaN-disciplined
/// (ADR 0006): any NaN entry returns NaN rather than being swallowed by
/// `dev > max_dev` (NaN comparisons are always false). Covers the 2×2/4×4/8×8
/// `GateMatrix` arms uniformly.
#[allow(clippy::needless_range_loop)]
fn unitarity_deviation_square<const N: usize>(m: &[[Complex<f64>; N]; N]) -> f64 {
    let mut max_dev = 0.0_f64;
    for r in 0..N {
        for c in 0..N {
            let mut acc = Complex::<f64>::new(0.0, 0.0);
            for k in 0..N {
                acc += m[k][r].conj() * m[k][c];
            }
            let target = if r == c {
                Complex::<f64>::new(1.0, 0.0)
            } else {
                Complex::<f64>::new(0.0, 0.0)
            };
            let dev = (acc - target).norm();
            if dev.is_nan() {
                return f64::NAN;
            }
            if dev > max_dev {
                max_dev = dev;
            }
        }
    }
    max_dev
}

/// Max deviation of `M†M` from the identity for a row-major `dim×dim` matrix
/// stored flat (the `UnitaryKq` representation: `data[r*dim + c] = m[r][c]`,
/// `dim = 2^k`). Same NaN discipline as `unitarity_deviation_square` (ADR 0006):
/// any NaN entry returns NaN rather than being swallowed by `dev > max_dev`.
/// Used for the variable-size `UnitaryKq` arm, which has no const-`N` form.
fn unitarity_deviation_flat(data: &[Complex<f64>], dim: usize) -> f64 {
    let mut max_dev = 0.0_f64;
    for r in 0..dim {
        for c in 0..dim {
            let mut acc = Complex::<f64>::new(0.0, 0.0);
            for k in 0..dim {
                // (M†M)[r,c] = Σ_k conj(M[k,r]) · M[k,c]
                acc += data[k * dim + r].conj() * data[k * dim + c];
            }
            let target = if r == c {
                Complex::<f64>::new(1.0, 0.0)
            } else {
                Complex::<f64>::new(0.0, 0.0)
            };
            let dev = (acc - target).norm();
            if dev.is_nan() {
                return f64::NAN;
            }
            if dev > max_dev {
                max_dev = dev;
            }
        }
    }
    max_dev
}

/// Max deviation of `M†M` from the identity for a 2×2 — local reimplementation
/// of `aleph-sv`'s `validation::unitarity_deviation` (which is `pub(crate)`),
/// restricted to the 2×2 case this backend handles.
///
/// NaN discipline (ADR 0006): if any entry is NaN, returns NaN — it does NOT
/// rely on `if dev > max_dev` to propagate it, because NaN comparisons always
/// return false and `max_dev` would silently stay 0.
fn unitarity_deviation_2x2(m: &[[Complex<f64>; 2]; 2]) -> f64 {
    unitarity_deviation_square(m)
}

/// Map a foundation `Error` into the shared `BackendError`. Device/compile
/// failures all surface as `InvalidState` so callers fail explicitly rather
/// than silently returning a wrong (CPU-fallback) result.
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

impl Backend for MetalSvBackend {
    type State = MetalSvState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_METAL_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_METAL_QUBITS,
            });
        }
        let dim = 1usize << num_qubits;
        let mut host = vec![Complex::<f32>::new(0.0, 0.0); dim];
        host[0] = Complex::<f32>::new(1.0, 0.0);
        let amps = DeviceBuffer::from_slice(&self.ctx, &host);
        Ok(MetalSvState::new(num_qubits, amps))
    }

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let n = state.num_qubits;
        let expected = gate.gate.arity();
        let got = gate.qubits.len();
        if expected != got {
            return Err(BackendError::ArityMismatch {
                kind: gate.gate.name(),
                expected,
                got,
            });
        }
        // Range + duplicate validation over targets ∪ controls.
        let mut seen: Vec<u32> = Vec::new();
        for &q in gate.qubits.iter().chain(gate.controls.iter()) {
            if q >= n {
                return Err(BackendError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
            if seen.contains(&q) {
                return Err(BackendError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        // UnitaryKq carries a raw 2^k x 2^k matrix and has no GateMatrix form;
        // dispatch it directly (mirrors the CPU FP32 backend). 2 <= k <= 5, and
        // 4^k = data.len(). `k` ignored here — kq_meta re-derives it from
        // gate.qubits.len().
        if let aleph_core::Gate::UnitaryKq { k: _, data } = &gate.gate {
            // `Gate::UnitaryKq` stores its raw 2^k×2^k matrix without validation,
            // so a non-unitary or NaN block would otherwise be applied silently.
            // Guard before narrowing, mirroring the fixed-size arms and ADR-0006
            // NaN discipline (the `dim` is re-derived from the target count, the
            // same source `kq_meta` uses, not the unvalidated `k` field).
            let dim = 1usize << gate.qubits.len();
            let deviation = unitarity_deviation_flat(data, dim);
            if !deviation.is_finite() || deviation > AMPLITUDE_TOL {
                return Err(BackendError::NonUnitaryMatrix { deviation });
            }
            let meta = Self::kq_meta(&gate.qubits, &gate.controls);
            let matrix: Vec<Complex<f32>> = data.iter().map(|z| narrow(*z)).collect();
            self.encode_kq(state, &meta, &matrix);
            return Ok(());
        }
        // Fixed-size GateMatrix gates (M2x2/M4x4/M8x8) below. (UnitaryKq was
        // handled just above.)
        let matrix = gate.gate.matrix().map_err(|e| match e {
            GateError::SymbolicParam => BackendError::SymbolicParam,
            GateError::NonFiniteParam => BackendError::NonFiniteParam {
                kind: gate.gate.name(),
            },
            GateError::Unrepresentable => BackendError::UnsupportedGate {
                kind: gate.gate.name(),
            },
        })?;
        match matrix {
            GateMatrix::M2x2(m) => {
                let deviation = unitarity_deviation_2x2(&m);
                if !deviation.is_finite() || deviation > AMPLITUDE_TOL {
                    return Err(BackendError::NonUnitaryMatrix { deviation });
                }
                let target = gate.qubits[0];
                let ctrl_mask = gate.controls.iter().fold(0u32, |acc, &c| acc | (1u32 << c));
                let g = Gate1q {
                    m: [
                        narrow(m[0][0]),
                        narrow(m[0][1]),
                        narrow(m[1][0]),
                        narrow(m[1][1]),
                    ],
                    target,
                    t_bit: 1u32 << target,
                    ctrl_mask,
                    _pad: 0,
                };
                self.encode_1q(state, &g);
            }
            GateMatrix::M4x4(m) => {
                // Defense-in-depth unitarity guard (catches a non-unitary or
                // NaN Unitary2q) before narrowing — mirrors the CPU FP32 path.
                let deviation = unitarity_deviation_square(&m);
                if !deviation.is_finite() || deviation > AMPLITUDE_TOL {
                    return Err(BackendError::NonUnitaryMatrix { deviation });
                }
                let meta = Self::kq_meta(&[gate.qubits[0], gate.qubits[1]], &gate.controls);
                let mut matrix = vec![Complex::<f32>::new(0.0, 0.0); 16];
                #[allow(clippy::needless_range_loop)]
                for r in 0..4 {
                    for c in 0..4 {
                        matrix[r * 4 + c] = narrow(m[r][c]);
                    }
                }
                self.encode_kq(state, &meta, &matrix);
            }
            GateMatrix::M8x8(m) => {
                let deviation = unitarity_deviation_square(&m);
                if !deviation.is_finite() || deviation > AMPLITUDE_TOL {
                    return Err(BackendError::NonUnitaryMatrix { deviation });
                }
                let meta = Self::kq_meta(
                    &[gate.qubits[0], gate.qubits[1], gate.qubits[2]],
                    &gate.controls,
                );
                let mut matrix = vec![Complex::<f32>::new(0.0, 0.0); 64];
                #[allow(clippy::needless_range_loop)]
                for r in 0..8 {
                    for c in 0..8 {
                        matrix[r * 8 + c] = narrow(m[r][c]);
                    }
                }
                self.encode_kq(state, &meta, &matrix);
            }
        }
        Ok(())
    }

    fn unpermute_state(
        &mut self,
        state: &mut Self::State,
        perm: &[u32],
    ) -> Result<(), BackendError> {
        // Host-side single gather: physical-order amplitudes -> logical order.
        // Drain the pending gate batch first so the slice reflects every prior
        // dispatch (P5.6-04). `bit_permute_buf` is the shared SV/MPS helper, so
        // the gather matches what the run_optimized driver expects after
        // RelabelQubits.
        state.sync();
        let gathered = aleph_core::bit_permute_buf(state.amps.as_slice(), perm);
        state.amps = DeviceBuffer::from_slice(&self.ctx, &gathered);
        Ok(())
    }

    fn apply_diagonal_phase(
        &mut self,
        state: &mut Self::State,
        dp: &aleph_ir::DiagonalPhase,
    ) -> Result<(), BackendError> {
        // Empty operator: nothing to do.
        if dp.terms.is_empty() {
            return Ok(());
        }
        // Masks must fit u32; the 28-qubit cap guarantees it, but assert the
        // invariant the kernel relies on rather than silently truncating.
        debug_assert!(
            dp.n_qubits <= MAX_METAL_QUBITS,
            "DiagonalPhase n_qubits {} exceeds Metal cap {}",
            dp.n_qubits,
            MAX_METAL_QUBITS
        );
        let mut cond_masks: Vec<u32> = Vec::new();
        let mut descs: Vec<DiagTermDesc> = Vec::with_capacity(dp.terms.len());
        for term in &dp.terms {
            let cond_offset = cond_masks.len() as u32;
            for &m in &term.conds {
                cond_masks.push(m as u32);
            }
            descs.push(DiagTermDesc {
                cond_offset,
                n_conds: term.conds.len() as u32,
                angle: term.angle as f32, // f64→f32; within 1e-5 for Tier-1 angles
                _pad: 0,
            });
        }
        // Encode into the open batch (P5.6-04) — the diagonal phase joins the same
        // command buffer as the surrounding gates, hazard-tracked on `amps` so it
        // runs in order with no extra sync. cm_buf/desc_buf must outlive the batch
        // (the GPU reads them after commit), so they're retained on the state.
        let cm_buf = DeviceBuffer::from_slice(&self.ctx, &cond_masks);
        let desc_buf = DeviceBuffer::from_slice(&self.ctx, &descs);
        let meta = DiagMeta {
            n_terms: descs.len() as u32,
        };
        let n_amps = 1u64 << state.num_qubits;
        state.encode(self.ctx.queue(), |encoder| {
            encoder.set_compute_pipeline_state(&self.pipeline_diag);
            encoder.set_buffer(0, Some(state.amps.metal_buffer()), 0);
            encoder.set_buffer(1, Some(cm_buf.metal_buffer()), 0);
            encoder.set_buffer(2, Some(desc_buf.metal_buffer()), 0);
            encoder.set_bytes(
                3,
                std::mem::size_of::<DiagMeta>() as u64,
                &meta as *const DiagMeta as *const c_void,
            );
            let tg = self
                .pipeline_diag
                .max_total_threads_per_threadgroup()
                .min(n_amps);
            encoder.dispatch_threads(MTLSize::new(n_amps, 1, 1), MTLSize::new(tg, 1, 1));
        });
        state.retain_aux(cm_buf);
        state.retain_aux(desc_buf);
        self.maybe_flush(state);
        Ok(())
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        super::readout::measure(&mut self.rng, state, qubit)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        super::readout::sample(&mut self.rng, state, shots)
    }

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        super::readout::expectation_value(state, pauli)
    }

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        super::readout::probabilities(state, qubits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Gate, Pauli};

    /// Construct a backend or skip (headless CI has no device).
    fn backend_or_skip() -> Option<MetalSvBackend> {
        match MetalSvBackend::with_seed(1) {
            Ok(b) => Some(b),
            Err(_) => {
                eprintln!("skipping Metal SV test: no Metal device");
                None
            }
        }
    }

    /// Build a `GateInstance` from a gate and target-qubit slice. `Vec<u32>`
    /// converts to `SmallVec<[u32; 4]>` via `Into`, so no extra dep is needed.
    fn gate(g: Gate, qubits: &[u32]) -> GateInstance {
        GateInstance::new(g, qubits.to_vec())
    }

    #[test]
    fn allocate_initialises_zero_state() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let s = b.allocate(3).unwrap();
        let a = s.amplitudes_f32();
        assert_eq!(a.len(), 8);
        assert_eq!(a[0], Complex::<f32>::new(1.0, 0.0));
        assert!(a[1..].iter().all(|z| z.re == 0.0 && z.im == 0.0));
    }

    #[test]
    fn allocate_rejects_too_many_qubits() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let err = b.allocate(MAX_METAL_QUBITS + 1).unwrap_err();
        assert_eq!(
            err,
            BackendError::TooManyQubits {
                requested: MAX_METAL_QUBITS + 1,
                limit: MAX_METAL_QUBITS,
            }
        );
    }

    #[test]
    fn h_on_zero_is_uniform_superposition() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &gate(Gate::H, &[0])).unwrap();
        let a = s.amplitudes_f32();
        let h = 1.0f32 / 2.0f32.sqrt();
        assert!((a[0].re - h).abs() < 1e-6, "got {:?}", a[0]);
        assert!((a[1].re - h).abs() < 1e-6, "got {:?}", a[1]);
        assert!(a[0].im.abs() < 1e-6 && a[1].im.abs() < 1e-6);
    }

    #[test]
    fn x_on_zero_flips_to_one() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap();
        let a = s.amplitudes_f32();
        assert!(a[0].norm() < 1e-6);
        assert!((a[1].re - 1.0).abs() < 1e-6);
    }

    /// Target a non-LSB qubit so the bit-insertion index math is exercised.
    #[test]
    fn x_on_qubit_1_of_two() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[1])).unwrap();
        // |00> -> |10>, i.e. index 0b10 = 2 (bit 1 = qubit 1).
        let a = s.amplitudes_f32();
        assert!((a[2].re - 1.0).abs() < 1e-6, "amps = {a:?}");
        assert!(a[0].norm() < 1e-6 && a[1].norm() < 1e-6 && a[3].norm() < 1e-6);
    }

    /// |10> --CNOT(0,1)--> |11>. qubits = [control, target] = [0,1].
    #[test]
    fn cnot_flips_target_when_control_set() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap(); // |10>
        b.apply_gate(&mut s, &gate(Gate::Cnot, &[0, 1])).unwrap();
        let a = s.amplitudes_f32();
        // |11> = index 0b11 = 3.
        assert!((a[3].re - 1.0).abs() < 1e-6, "amps = {a:?}");
        assert!(a[0].norm() < 1e-6 && a[1].norm() < 1e-6 && a[2].norm() < 1e-6);
    }

    /// CZ on |11> flips the sign; on |10> leaves it.
    #[test]
    fn cz_phases_eleven() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[1])).unwrap(); // |11>
        b.apply_gate(&mut s, &gate(Gate::Cz, &[0, 1])).unwrap();
        let a = s.amplitudes_f32();
        // CZ|11> = -|11>: amplitude is purely real -1 (check im too so a phase
        // bug like (-0.9 + 0.4i) fails clearly, not just on the real part).
        assert!(
            (a[3].re + 1.0).abs() < 1e-6 && a[3].im.abs() < 1e-6,
            "amps = {a:?}"
        );
    }

    /// SWAP(0,1) on |q0=1,q1=0> -> |q0=0,q1=1>.
    ///
    /// Qubit `q` maps to bit `q` of the state index. X on qubit 0 sets bit 0,
    /// giving index 1 (|q1 q0⟩ = |01⟩). After SWAP(0,1), qubit 0=0 and
    /// qubit 1=1, so index = 0b10 = 2. This matches the CPU kq kernel's SWAP
    /// behavior (see `apply_kq_swap_matches_manual` in aleph-sv).
    #[test]
    fn swap_exchanges_basis() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap(); // bit 0 set → index 1
        b.apply_gate(&mut s, &gate(Gate::Swap, &[0, 1])).unwrap();
        let a = s.amplitudes_f32();
        // After SWAP: qubit 0=0, qubit 1=1 → index 2.
        assert!((a[2].re - 1.0).abs() < 1e-6, "amps = {a:?}");
        assert!(a[0].norm() < 1e-6 && a[1].norm() < 1e-6 && a[3].norm() < 1e-6);
    }

    /// Pure-helper test (no Metal device): `kq_meta` must sort the insertion
    /// positions ascending while keeping `tbit` in logical/MSB order, and OR the
    /// control mask. Runs even on a device-less macOS box so the index logic is
    /// covered without a GPU.
    #[test]
    fn kq_meta_sorts_positions_keeps_logical_tbit() {
        // targets [2, 0]: q[0]=2 is the matrix MSB. Controls [3].
        let m = MetalSvBackend::kq_meta(&[2, 0], &[3]);
        assert_eq!(m.k, 2);
        assert_eq!(m.tbit[0], 1 << 2); // logical order: targets[0] first
        assert_eq!(m.tbit[1], 1 << 0);
        assert_eq!(m.sorted[0], 0); // ascending for zero-bit insertion
        assert_eq!(m.sorted[1], 2);
        assert_eq!(m.ctrl_mask, 1 << 3);
        // Unused slots (j >= k) stay zero.
        assert_eq!(m.tbit[2], 0);
        assert_eq!(m.sorted[2], 0);
    }

    #[test]
    fn out_of_range_qubit_is_rejected() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        let err = b.apply_gate(&mut s, &gate(Gate::X, &[3])).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 3,
                num_qubits: 1
            }
        );
    }

    /// (a) Direct unit test of the pure helper — no Metal device required.
    ///
    /// Without Fix 1 (`dev.is_nan()` early-return), this test would fail:
    /// `NaN > max_dev` is always false, so `max_dev` would remain `0.0` and
    /// `is_nan()` would return false. The test therefore directly proves the bug
    /// was present and the fix closes it.
    #[test]
    fn unitarity_deviation_nan_entry_returns_nan() {
        // A NaN entry must propagate to NaN, not be swallowed by `dev > max_dev`
        // (ADR 0006). Without the is_nan guard this returns 0.0 and the matrix
        // would wrongly pass the unitarity check.
        let m = [
            [
                Complex::<f64>::new(f64::NAN, 0.0),
                Complex::<f64>::new(0.0, 0.0),
            ],
            [Complex::<f64>::new(0.0, 0.0), Complex::<f64>::new(1.0, 0.0)],
        ];
        assert!(super::unitarity_deviation_2x2(&m).is_nan());
    }

    /// (b) Integration test: `apply_gate` must reject a `Gate::Unitary1q` whose
    /// matrix contains a NaN entry as `BackendError::NonUnitaryMatrix`.
    ///
    /// `Gate::Unitary1q` stores the raw matrix without a finiteness check, and
    /// `Gate::matrix()` copies it out verbatim, so the NaN flows all the way to
    /// `unitarity_deviation_2x2` in `apply_gate`. The explicit `!deviation.is_finite()`
    /// guard in `apply_gate` (which fires when `unitarity_deviation_2x2` returns NaN)
    /// should catch it and return `Err(BackendError::NonUnitaryMatrix { .. })`.
    #[test]
    fn nan_matrix_is_rejected_as_non_unitary() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        let nan_matrix = Box::new([
            [
                Complex::<f64>::new(f64::NAN, 0.0),
                Complex::<f64>::new(0.0, 0.0),
            ],
            [Complex::<f64>::new(0.0, 0.0), Complex::<f64>::new(1.0, 0.0)],
        ]);
        let g = gate(Gate::Unitary1q(nan_matrix), &[0]);
        let err = b.apply_gate(&mut s, &g).unwrap_err();
        assert!(
            matches!(err, BackendError::NonUnitaryMatrix { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn probabilities_plus_state_uniform() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &gate(Gate::H, &[0])).unwrap();
        let p = b.probabilities(&s, &[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-5 && (p[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn sample_collapsed_state_is_deterministic() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap();
        let shots = b.sample(&s, 256).unwrap();
        assert!(shots.iter().all(|&v| v == 1));
    }

    #[test]
    fn measure_plus_state_collapses_to_basis() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &gate(Gate::H, &[0])).unwrap();
        let outcome = b.measure(&mut s, 0).unwrap();
        let a = s.amplitudes_f32();
        if outcome {
            assert!((a[1].norm() - 1.0).abs() < 1e-5 && a[0].norm() < 1e-5);
        } else {
            assert!((a[0].norm() - 1.0).abs() < 1e-5 && a[1].norm() < 1e-5);
        }
    }

    #[test]
    fn expectation_z_on_zero_is_plus_one() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let s = b.allocate(1).unwrap();
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        assert!((b.expectation_value(&s, &z).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn expectation_x_on_plus_is_plus_one() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &gate(Gate::H, &[0])).unwrap();
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        assert!((b.expectation_value(&s, &x).unwrap() - 1.0).abs() < 1e-5);
    }

    /// Toffoli(0,1,2) on |110> -> |111>.
    #[test]
    fn toffoli_flips_when_both_controls_set() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(3).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[1])).unwrap(); // |110>
        b.apply_gate(&mut s, &gate(Gate::Toffoli, &[0, 1, 2]))
            .unwrap();
        let a = s.amplitudes_f32();
        assert!((a[7].re - 1.0).abs() < 1e-6, "amps = {a:?}"); // |111> = 7
    }

    /// apply_diagonal_phase on a known DiagonalPhase must rotate each amplitude
    /// by exp(i * phase_at(x)). Build a uniform |++> and compare to the f64
    /// reference computed via DiagonalPhase::phase_at.
    #[test]
    fn diagonal_phase_matches_phase_at() {
        use aleph_ir::diagonal_phase::{DiagonalPhase, PhaseTerm};
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &gate(Gate::H, &[0])).unwrap();
        b.apply_gate(&mut s, &gate(Gate::H, &[1])).unwrap(); // |++>, all amps 0.5

        // Term 1: fires when bit0 parity odd -> angle 0.7.
        // Term 2: fires when bits {0,1} BOTH parity-odd -> angle -0.4.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![
                PhaseTerm {
                    conds: vec![0b01u64].into(),
                    angle: 0.7,
                },
                PhaseTerm {
                    conds: vec![0b01u64, 0b10u64].into(),
                    angle: -0.4,
                },
            ],
        };
        b.apply_diagonal_phase(&mut s, &dp).unwrap();

        let a = s.amplitudes_f32();
        for x in 0u64..4 {
            let phi = dp.phase_at(x);
            let expect = aleph_core::Complex::<f64>::new(0.5 * phi.cos(), 0.5 * phi.sin());
            let got =
                aleph_core::Complex::<f64>::new(a[x as usize].re as f64, a[x as usize].im as f64);
            assert!(
                (got - expect).norm() < 1e-5,
                "x={x} got {got:?} expect {expect:?}"
            );
        }
    }

    /// CCZ on |111> applies -1.
    #[test]
    fn ccz_phases_all_ones() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(3).unwrap();
        for q in 0..3 {
            b.apply_gate(&mut s, &gate(Gate::X, &[q])).unwrap();
        }
        b.apply_gate(&mut s, &gate(Gate::Ccz, &[0, 1, 2])).unwrap();
        let a = s.amplitudes_f32();
        assert!(
            (a[7].re + 1.0).abs() < 1e-6 && a[7].im.abs() < 1e-6,
            "amps = {a:?}"
        );
    }

    /// A 2-control MCX (X with two controls) via the shipped 1q ctrl_mask path:
    /// |110> with X on q2 controlled by q0,q1 -> |111>. This exercises the 1q
    /// path (NOT the UnitaryKq path added here) — a regression guard that the new
    /// dispatch code left multi-controlled 1q gates working.
    #[test]
    fn mcx_two_controls_via_1q_path() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(3).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[1])).unwrap(); // |110>
        let mcx = GateInstance::controlled(Gate::X, vec![2u32], vec![0u32, 1u32]);
        b.apply_gate(&mut s, &mcx).unwrap();
        let a = s.amplitudes_f32();
        assert!((a[7].re - 1.0).abs() < 1e-6, "amps = {a:?}");
    }

    /// `run` executes a circuit gate-by-gate and yields the expected state.
    /// GHZ(3): H(0), CNOT(0,1), CNOT(1,2) -> (|000> + |111>)/sqrt(2),
    /// i.e. amps[0] = amps[7] = 1/sqrt(2), all others ~0.
    #[test]
    fn run_executes_ghz_circuit() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut c = Circuit::new(3, 0);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.cnot(1, 2).unwrap();
        let s = b.run(&c).unwrap();
        let a = s.amplitudes_f32();
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
        assert!((a[0].re - inv_sqrt2).abs() < 1e-5, "amps = {a:?}");
        assert!((a[7].re - inv_sqrt2).abs() < 1e-5, "amps = {a:?}");
        for (i, z) in a.iter().enumerate() {
            if i != 0 && i != 7 {
                assert!(z.norm() < 1e-5, "amps[{i}] = {z:?}");
            }
        }
    }

    /// unpermute_state applies a single bit-permutation gather. perm[logical] =
    /// physical, so perm=[1,0] maps logical qubit 0 -> physical qubit 1. A
    /// physical state with qubit 1 set (X on qubit 1 -> physical index 2) gathers
    /// to the logical state with qubit 0 set -> logical index 1.
    #[test]
    fn unpermute_swaps_bit_order() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[1])).unwrap(); // physical index 2
        b.unpermute_state(&mut s, &[1, 0]).unwrap();
        let a = s.amplitudes_f32();
        assert!((a[1].re - 1.0).abs() < 1e-6, "amps = {a:?}"); // logical index 1
        assert!(
            a[0].norm() < 1e-6 && a[2].norm() < 1e-6 && a[3].norm() < 1e-6,
            "amps = {a:?}"
        );
    }

    /// A dense UnitaryKq (k=2) equal to SWAP. State after X(0) is index 1
    /// (qubit 0 = bit 0); SWAP(0,1) moves it to index 2 (qubit 1 = bit 1).
    #[test]
    fn unitary_kq_k2_swap() {
        use aleph_core::Complex as C;
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &gate(Gate::X, &[0])).unwrap(); // index 1
                                                             // Row-major 4x4 SWAP: swaps basis 01 <-> 10.
        let z = C::new(0.0, 0.0);
        let o = C::new(1.0, 0.0);
        let data: Vec<C> = vec![
            o, z, z, z, //
            z, z, o, z, //
            z, o, z, z, //
            z, z, z, o, //
        ];
        let g = gate(
            Gate::UnitaryKq {
                k: 2,
                data: data.into_boxed_slice(),
            },
            &[0, 1],
        );
        b.apply_gate(&mut s, &g).unwrap();
        let a = s.amplitudes_f32();
        assert!((a[2].re - 1.0).abs() < 1e-6, "amps = {a:?}"); // index 2
        assert!(a[0].norm() < 1e-6 && a[1].norm() < 1e-6 && a[3].norm() < 1e-6);
    }

    /// AC #2: a `UnitaryKq` carrying a NaN entry must be rejected as
    /// `NonUnitaryMatrix`, not silently applied. `Gate::UnitaryKq` stores its
    /// raw matrix without validation, so the NaN reaches the apply-path guard.
    #[test]
    fn unitary_kq_nan_is_rejected_as_non_unitary() {
        use aleph_core::Complex as C;
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        let z = C::new(0.0, 0.0);
        let o = C::new(1.0, 0.0);
        // Identity 4x4 except a NaN in the top-left — a NaN comparison must not
        // be swallowed (ADR 0006).
        let data: Vec<C> = vec![
            C::new(f64::NAN, 0.0),
            z,
            z,
            z, //
            z,
            o,
            z,
            z, //
            z,
            z,
            o,
            z, //
            z,
            z,
            z,
            o, //
        ];
        let g = gate(
            Gate::UnitaryKq {
                k: 2,
                data: data.into_boxed_slice(),
            },
            &[0, 1],
        );
        let err = b.apply_gate(&mut s, &g).unwrap_err();
        assert!(
            matches!(err, BackendError::NonUnitaryMatrix { .. }),
            "got {err:?}"
        );
    }

    /// AC #2: a finite-but-non-unitary `UnitaryKq` (a scaled identity, deviation
    /// well above tolerance) must also be rejected.
    #[test]
    fn unitary_kq_non_unitary_is_rejected() {
        use aleph_core::Complex as C;
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        let z = C::new(0.0, 0.0);
        let two = C::new(2.0, 0.0);
        let data: Vec<C> = vec![
            two, z, z, z, //
            z, two, z, z, //
            z, z, two, z, //
            z, z, z, two, //
        ];
        let g = gate(
            Gate::UnitaryKq {
                k: 2,
                data: data.into_boxed_slice(),
            },
            &[0, 1],
        );
        let err = b.apply_gate(&mut s, &g).unwrap_err();
        assert!(
            matches!(err, BackendError::NonUnitaryMatrix { .. }),
            "got {err:?}"
        );
    }

    /// Assert two FP32 amplitude buffers agree elementwise within `tol`.
    fn assert_amps_close(fused: &[Complex<f32>], unfused: &[Complex<f32>], tol: f32) {
        assert_eq!(fused.len(), unfused.len(), "length mismatch");
        for (i, (f, u)) in fused.iter().zip(unfused.iter()).enumerate() {
            assert!(
                (*f - *u).norm() < tol,
                "amp[{i}] fused={f:?} unfused={u:?} (tol {tol})"
            );
        }
    }

    /// AC #1: the fused path (run_optimized -> apply_kq dense blocks) must match
    /// the unfused path (run, gate-by-gate) within 1e-5 on the GPU. GHZ.
    #[test]
    fn fused_matches_unfused_ghz() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let c = aleph_benches::ghz_circuit(8);
        let unfused = b.run(&c).unwrap().amplitudes_f32().to_vec();
        let fused = b.run_optimized(&c).unwrap().amplitudes_f32().to_vec();
        assert_amps_close(&fused, &unfused, 1e-5);
    }

    /// AC #1: QFT — the canonical fusion beneficiary (H + cphase ladder).
    #[test]
    fn fused_matches_unfused_qft() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let c = aleph_benches::qft_circuit(8);
        let unfused = b.run(&c).unwrap().amplitudes_f32().to_vec();
        let fused = b.run_optimized(&c).unwrap().amplitudes_f32().to_vec();
        assert_amps_close(&fused, &unfused, 1e-5);
    }

    /// AC #1: random brickwall — dense 2q gates fuse into UnitaryKq blocks.
    /// The builder is deterministic (angles from cos, no RNG), so both runs see
    /// the same circuit.
    #[test]
    fn fused_matches_unfused_random() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let c = aleph_benches::random_brickwall_circuit(8, 8);
        let unfused = b.run(&c).unwrap().amplitudes_f32().to_vec();
        let fused = b.run_optimized(&c).unwrap().amplitudes_f32().to_vec();
        assert_amps_close(&fused, &unfused, 1e-5);
    }

    /// AC #1: Grover — parsed from the measurement-free n=4 baseline fixture
    /// (grover_n4_iters3.qasm has zero `measure` ops, so amplitude comparison is
    /// well-defined). Path anchored to CARGO_MANIFEST_DIR so it is independent of
    /// the test's working directory.
    #[test]
    fn fused_matches_unfused_grover() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let src = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../scripts/qiskit-baseline/circuits/grover_n4_iters3.qasm"
        ));
        let c = aleph_parser::parse(src).expect("parse grover_n4");
        let unfused = b.run(&c).unwrap().amplitudes_f32().to_vec();
        let fused = b.run_optimized(&c).unwrap().amplitudes_f32().to_vec();
        assert_amps_close(&fused, &unfused, 1e-5);
    }

    /// P5.6-04: a circuit deeper than `BATCH_CAP` must stay correct across the
    /// forced mid-run flush and the kq-arena slot reuse it triggers. Builds well
    /// over 2·BATCH_CAP gates (H + CNOT, so the kq path is exercised) and checks
    /// the batched GPU `run` against the exact FP64 CPU `NaiveSvBackend` within
    /// the f32 oracle tolerance.
    #[test]
    fn deep_circuit_survives_batch_flush() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let reps = (2 * BATCH_CAP) / 5 + 4; // 5 gates/rep ⇒ well over 2·BATCH_CAP
        let mut c = Circuit::new(3, 0);
        for _ in 0..reps {
            c.h(0).unwrap();
            c.h(1).unwrap();
            c.h(2).unwrap();
            c.cnot(0, 1).unwrap();
            c.cnot(1, 2).unwrap();
        }
        let gpu = b.run(&c).unwrap().to_aos_f64();

        let mut cpu = aleph_sv::NaiveSvBackend::with_seed(0);
        let cpu_state = aleph_backend::run(&mut cpu, &c).unwrap();
        let reference = aleph_oracle::HasAmplitudes::amplitudes(&cpu_state);

        assert_eq!(gpu.len(), reference.len());
        for (i, (g, r)) in gpu.iter().zip(reference.iter()).enumerate() {
            let d = ((g.re - r.re).powi(2) + (g.im - r.im).powi(2)).sqrt();
            assert!(d <= 1e-5, "amp {i}: |Δ|={d:.3e}\n  gpu {g:?}\n  ref {r:?}");
        }
    }
}
