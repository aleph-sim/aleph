//! Shared full-final-state CUDA planning for public frontends and benchmarks (#552).
use crate::{CudaSvBackend, CudaSvBackendF32, Error};
use aleph_backend::{Backend, BackendError, BackendKind};
use aleph_core::Gate;
use aleph_ir::{Circuit, Instruction};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

/// Public CUDA execution policy. Ordinary FP32 arithmetic is enforced;
/// experimental TF32 remains in the separate legacy Rust kernel/helper APIs.
#[derive(Clone, Copy, Debug)]
pub struct CudaExecutionOptions {
    /// GPU-safe fusion and disjoint single-qubit batching; false is the raw A/B path.
    pub optimize: bool,
}
impl Default for CudaExecutionOptions {
    fn default() -> Self {
        Self { optimize: true }
    }
}
/// Requested readout for operation-aware preflight before backend/state construction.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaReadoutRequest {
    /// Samples requested from the final state.
    pub shots: u32,
    /// Whether Pauli expectations will be computed.
    pub expectation: bool,
    /// Whether amplitudes will be materialized immediately.
    pub export: bool,
}

static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static LAYERS: AtomicU64 = AtomicU64::new(0);
static TF32: AtomicU64 = AtomicU64::new(0);
/// Process-wide successful execution-kernel launch counters (exclude readout).
/// Concurrent callers must not interpret differences as per-run measurements.
pub fn execution_counters() -> (u64, u64, u64) {
    (
        LAUNCHES.load(Ordering::Relaxed),
        LAYERS.load(Ordering::Relaxed),
        TF32.load(Ordering::Relaxed),
    )
}
pub(crate) fn record_launch() {
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}
pub(crate) fn record_layer() {
    LAYERS.fetch_add(1, Ordering::Relaxed);
}
pub(crate) fn record_tf32() {
    TF32.fetch_add(1, Ordering::Relaxed);
}

/// Inspectable reusable full-state plan; no measurement-distribution DCE or relabeling.
pub struct CudaExecutionPlan {
    circuit: Circuit,
    options: CudaExecutionOptions,
    /// Number of instructions before optimization.
    pub input_instructions: usize,
    /// Dense fusion width (3 for the strict public precision policy).
    pub fusion_width: usize,
    /// Whether the backend's optional TF32 module loaded, populated by shared
    /// preparation. GPU-independent compile leaves this false (unknown). Public
    /// plans enforce ordinary FP32; experimental TF32 helpers remain separate.
    pub tf32_available: bool,
    /// Host fusion/clone time; validation is covered by preparation overhead.
    pub planning_ms: f64,
    /// Backend/JIT construction time (set by shared preparation).
    pub initialization_ms: f64,
    /// Capacity checks and preparation overhead excluding planning/JIT.
    pub preflight_ms: f64,
}
impl CudaExecutionPlan {
    /// Compile without a GPU. Public plans always use width 3 and ordinary FP32;
    /// TF32's separate empirical error budget is not a public precision contract.
    pub fn compile(c: &Circuit, options: CudaExecutionOptions) -> Result<Self, BackendError> {
        // Gate/controlled-kernel masks use u32 even on devices whose memory
        // could hold a larger state. Capacity alone cannot authorize that IR.
        if c.num_qubits() > 32 {
            return Err(BackendError::DeviceFailure(format!(
            "CUDA memory plan needs {}-qubit state/sampling; CUDA gate masks support at most 32 qubits",c.num_qubits())));
        }
        if c.qubit_permutation().is_some() {
            return Err(BackendError::UnsupportedInstruction {
                kind: "CUDA plan cannot consume CPU-relabeled IR",
            });
        }
        for i in c.instructions() {
            match i {
                Instruction::Gate(g) => {
                    if let Gate::UnitaryKq { k, data } = &g.gate {
                        crate::common::validate_kq(
                            c.num_qubits(),
                            *k,
                            data.len(),
                            &g.qubits,
                            &g.controls,
                        )?;
                        if data.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
                            return Err(BackendError::InvalidState {
                                reason: "non-finite UnitaryKq matrix",
                            });
                        }
                    }
                }
                Instruction::Reset(_) => {
                    return Err(BackendError::UnsupportedInstruction { kind: "reset" })
                }
                Instruction::TiledBlock(_) => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "CUDA plan cannot consume CPU tiled blocks",
                    })
                }
                _ => {}
            }
        }
        let start = Instant::now();
        let width = crate::MAX_FUSE_QUBITS;
        let circuit = if options.optimize {
            crate::fusion::fuse_full_state(c, width)
                .map_err(|e| BackendError::DeviceFailure(e.to_string()))?
        } else {
            c.clone()
        };
        Ok(Self {
            circuit,
            options,
            input_instructions: c.len(),
            fusion_width: width,
            tf32_available: false,
            planning_ms: start.elapsed().as_secs_f64() * 1000.,
            initialization_ms: 0.,
            preflight_ms: 0.,
        })
    }
    /// Optimized instructions, also used for memory accounting.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    /// Counts of dense fused 1q/2q/Kq and diagonal-polynomial instructions.
    pub fn instruction_kinds(&self) -> (usize, usize, usize, usize) {
        let mut kinds = (0, 0, 0, 0);
        for i in self.circuit.instructions() {
            match i {
                Instruction::Gate(g) => match &g.gate {
                    Gate::Unitary1q(_) => kinds.0 += 1,
                    Gate::Unitary2q(_) => kinds.1 += 1,
                    Gate::UnitaryKq { .. } => kinds.2 += 1,
                    _ => {}
                },
                Instruction::DiagonalPhase(_) => kinds.3 += 1,
                _ => {}
            }
        }
        kinds
    }
    /// Numeric diagnostics; counters measured separately are process-wide deltas.
    pub fn diagnostics(&self) -> std::collections::BTreeMap<String, f64> {
        let (a, b, c, d) = self.instruction_kinds();
        [
            ("input_instructions", self.input_instructions as f64),
            ("output_instructions", self.circuit.len() as f64),
            ("fused_1q", a as f64),
            ("fused_2q", b as f64),
            ("fused_kq", c as f64),
            ("diagonal_polynomials", d as f64),
            ("fusion_width", self.fusion_width as f64),
            ("tf32_available", self.tf32_available as u8 as f64),
            ("tf32_effective", 0.),
            ("optimized", self.options.optimize as u8 as f64),
            ("planning_ms", self.planning_ms),
            ("initialization_ms", self.initialization_ms),
            ("preflight_ms", self.preflight_ms),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }
    /// Execute on a fresh |0> state and synchronize. The plan can be reused; RNG
    /// consumption at retained measurements is unchanged by planning.
    pub fn execute<B: CudaPlanBackend>(&self, backend: &mut B) -> Result<B::State, BackendError> {
        // Do not inherit the legacy FP32 backend's default-enabled TF32 switch,
        // including manually fused width-4/5 inputs in the raw comparison path.
        backend.set_tf32(false);
        let state = if self.options.optimize {
            backend.execute_layered(&self.circuit)?
        } else {
            aleph_backend::run(backend, &self.circuit)?
        };
        backend.synchronize()?;
        Ok(state)
    }
}
/// CUDA SV execution hooks used by the shared preparation boundary.
pub trait CudaPlanBackend: Backend {
    /// Concrete backend for operation-aware memory accounting.
    fn kind() -> BackendKind;
    /// Whether the optional TF32 module really loaded.
    fn tf32_available(&self) -> bool;
    /// Set the explicit arithmetic policy.
    fn set_tf32(&mut self, on: bool);
    /// Execute supported layered dispatch.
    fn execute_layered(&mut self, c: &Circuit) -> Result<Self::State, BackendError>;
    /// Finish queued execution and surface asynchronous driver errors.
    fn synchronize(&self) -> Result<(), BackendError>;
}
macro_rules! backend_impl {
    ($backend:ty,$kind:expr,$available:expr,$set:expr) => {
        impl CudaPlanBackend for $backend {
            fn kind() -> BackendKind {
                $kind
            }
            fn tf32_available(&self) -> bool {
                ($available)(self)
            }
            fn set_tf32(&mut self, on: bool) {
                ($set)(self, on)
            }
            fn execute_layered(&mut self, c: &Circuit) -> Result<Self::State, BackendError> {
                self.run_layered(c)
            }
            fn synchronize(&self) -> Result<(), BackendError> {
                self.ctx()
                    .synchronize()
                    .map_err(|e| BackendError::DeviceFailure(e.to_string()))
            }
        }
    };
}
backend_impl!(
    CudaSvBackend,
    BackendKind::Cuda,
    |_: &CudaSvBackend| false,
    |_: &mut CudaSvBackend, _: bool| {}
);
backend_impl!(
    CudaSvBackendF32,
    BackendKind::CudaF32,
    |b: &CudaSvBackendF32| b.tf32_available(),
    |b: &mut CudaSvBackendF32, on: bool| b.set_tf32_policy(on)
);

/// Preflight the actual candidate plan before backend construction, then resolve
/// optional-module diagnostics before allocating a state. The returned
/// backend owns the primary context after the preflight guard drops.
pub fn prepare_cuda_execution<B: CudaPlanBackend>(
    c: &Circuit,
    options: CudaExecutionOptions,
    views: CudaReadoutRequest,
    create: impl FnOnce() -> Result<B, Error>,
) -> Result<(B, CudaExecutionPlan), BackendError> {
    let preparation_start = Instant::now();
    let mut plan = CudaExecutionPlan::compile(c, options)?;
    let _guard = crate::memory::plan_frontend(
        plan.circuit(),
        B::kind(),
        views.shots.into(),
        views.expectation,
        views.export,
    )
    .map_err(BackendError::DeviceFailure)?;
    let init_start = Instant::now();
    let backend = create().map_err(|e| BackendError::DeviceFailure(e.to_string()))?;
    let initialization_ms = init_start.elapsed().as_secs_f64() * 1000.;
    plan.tf32_available = backend.tf32_available();
    plan.initialization_ms = initialization_ms;
    plan.preflight_ms =
        (preparation_start.elapsed().as_secs_f64() * 1000. - plan.planning_ms - initialization_ms)
            .max(0.);
    Ok((backend, plan))
}
