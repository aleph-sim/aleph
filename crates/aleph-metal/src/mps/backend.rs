//! `MetalMpsBackend` — a scaffold GPU MPS backend (P5.5-06).
//!
//! Runs nearest-neighbour circuits with the hot per-gate tensor work on the GPU
//! (1q apply, two-site contraction, 2q gate-apply) and the truncated SVD on the
//! CPU via `faer` (the GPU has no SVD). NN-only: non-adjacent 2q gates, external
//! controls, and ≥3q gates return `UnsupportedInstruction` (no SWAP router yet).
//! Correctness is gated by the oracle (`tests/mps_oracle.rs`) vs the CPU MPS.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, Gate, GateError, GateInstance, GateMatrix, PauliString};
use aleph_ir::Circuit;
use metal::{ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::kernel::{
    Apply2qMeta, ContractMeta, Mps1q, MPS_1Q_ENTRY, MPS_1Q_SRC, MPS_APPLY2Q_ENTRY, MPS_APPLY2Q_SRC,
    MPS_CONTRACT_ENTRY, MPS_CONTRACT_SRC,
};
use super::state::{MetalMpsState, SiteTensor};
use super::svd::svd_split;
use crate::{DeviceBuffer, Error, MetalContext};

/// Qubit cap for the scaffold. The MPS form scales past the SV 28-qubit ceiling,
/// but the scaffold's dense oracle readout is small-n only, so we keep the
/// project-wide cap to avoid surprising 2^n allocations in tests.
pub(crate) const MAX_MPS_QUBITS: u32 = 28;

/// Default maximum bond dimension χ. Generous enough that the small NN Tier-1
/// test circuits never truncate (so the oracle compare is exact-to-fp32).
const DEFAULT_MAX_BOND: usize = 128;

/// Opt-in single-precision Metal GPU MPS backend (scaffold).
pub struct MetalMpsBackend {
    ctx: MetalContext,
    pipeline_1q: ComputePipelineState,
    pipeline_contract: ComputePipelineState,
    pipeline_apply2q: ComputePipelineState,
    /// Reused 4×4 (16-entry) row-major f32 gate-matrix scratch for the 2q apply.
    mat_scratch: DeviceBuffer<Complex<f32>>,
    max_bond: usize,
    /// Cumulative time (ns) in the GPU contract+apply dispatches vs the host SVD
    /// split, summed over every NN 2q gate. Drives the AC #2 round-trip-cost doc.
    gpu_ns: u128,
    svd_ns: u128,
}

impl MetalMpsBackend {
    /// Construct with the system-default Metal device. Returns
    /// [`BackendError::InvalidState`] when no device is present (headless CI) or a
    /// shader/pipeline build fails.
    pub fn new() -> Result<Self, BackendError> {
        Self::build(DEFAULT_MAX_BOND)
    }

    /// Construct with an explicit seed. The scaffold has no stochastic ops
    /// (measure/sample are unsupported), so the seed is currently unused; the
    /// constructor exists for API parity with the other backends.
    pub fn with_seed(_seed: u64) -> Result<Self, BackendError> {
        Self::build(DEFAULT_MAX_BOND)
    }

    /// Construct with an explicit maximum bond dimension χ.
    pub fn with_max_bond(max_bond: usize) -> Result<Self, BackendError> {
        Self::build(max_bond.max(1))
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

    fn build(max_bond: usize) -> Result<Self, BackendError> {
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
        let mat_scratch = DeviceBuffer::from_slice(&ctx, &[Complex::<f32>::new(0.0, 0.0); 16]);
        Ok(Self {
            ctx,
            pipeline_1q,
            pipeline_contract,
            pipeline_apply2q,
            mat_scratch,
            max_bond,
            gpu_ns: 0,
            svd_ns: 0,
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

        // --- Host phase: truncated SVD split (the documented CPU round-trip) ---
        // Θ' is in unified memory so the read is zero-copy, but the SVD runs
        // single-threaded on the CPU while the GPU idles, then the two factor
        // tensors are uploaded into fresh buffers. The slice is current:
        // dispatch_apply2q waited.
        let svd_start = std::time::Instant::now();
        let (chi, site_i_data, site_j_data, _discarded) =
            svd_split(theta.as_slice(), rows, cols, self.max_bond)?;
        state.sites[i] = SiteTensor::from_host(&self.ctx, li, chi, &site_i_data);
        state.sites[j] = SiteTensor::from_host(&self.ctx, chi, ri, &site_j_data);
        self.svd_ns += svd_start.elapsed().as_nanos();
        Ok(())
    }
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
                self.apply_1q(state, gate.qubits[0] as usize, &m);
                Ok(())
            }
            GateMatrix::M4x4(m) => {
                let qa = gate.qubits[0];
                let qb = gate.qubits[1];
                if qa.abs_diff(qb) != 1 {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "non-nearest-neighbour 2q gate (MPS scaffold; no SWAP router)",
                    });
                }
                self.apply_2q_nn(state, qa as usize, qb as usize, &m)
            }
            GateMatrix::M8x8(_) => Err(BackendError::UnsupportedInstruction {
                kind: "3q gate (MPS scaffold)",
            }),
        }
    }

    fn measure(&mut self, _state: &mut Self::State, _qubit: u32) -> Result<bool, BackendError> {
        Err(BackendError::UnsupportedInstruction {
            kind: "measure (MPS scaffold)",
        })
    }

    fn sample(&mut self, _state: &Self::State, _shots: u32) -> Result<Vec<u64>, BackendError> {
        Err(BackendError::UnsupportedInstruction {
            kind: "sample (MPS scaffold)",
        })
    }

    fn expectation_value(
        &mut self,
        _state: &Self::State,
        _pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        Err(BackendError::UnsupportedInstruction {
            kind: "expectation_value (MPS scaffold)",
        })
    }

    fn probabilities(
        &mut self,
        _state: &Self::State,
        _qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        Err(BackendError::UnsupportedInstruction {
            kind: "probabilities (MPS scaffold)",
        })
    }
}
