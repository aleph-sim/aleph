//! `MetalSvBackend` — device-resident FP32 state-vector backend (P5.5-02).
//! Mirrors `Fp32SvBackend` over a Metal GPU buffer. The f64 `NaiveSvBackend`
//! remains the oracle reference; this is a GPU mode at f32 accuracy.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateInstance, PauliString};
use metal::ComputePipelineState;
use rand::{rngs::StdRng, SeedableRng};

use super::kernel::{SV_1Q_ENTRY, SV_1Q_SRC};
use super::state::MetalSvState;
use crate::{DeviceBuffer, Error, MetalContext};

/// Soft qubit cap — the project-wide 28-qubit software limit, matching the CPU
/// backends. At 8 B/amp the memory ceiling is higher, but 28 binds first.
pub(crate) const MAX_METAL_QUBITS: u32 = 28;

/// Opt-in single-precision Metal GPU state-vector backend.
pub struct MetalSvBackend {
    ctx: MetalContext,
    // Used by the apply_gate dispatch in Task 5.
    #[allow(dead_code)]
    pipeline_1q: ComputePipelineState,
    // Used by the host-side readout / measure / sample in Task 6.
    #[allow(dead_code)]
    rng: StdRng,
}

impl MetalSvBackend {
    /// Construct with an entropy-seeded RNG.
    ///
    /// Acquires the system-default Metal device and compiles+caches the 1q
    /// pipeline once. Returns [`BackendError::InvalidState`] when no device is
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

    fn build(rng: StdRng) -> Result<Self, BackendError> {
        let ctx = MetalContext::new().map_err(map_metal_err)?;
        let pipeline_1q = ctx
            .make_compute_pipeline(SV_1Q_SRC, SV_1Q_ENTRY)
            .map_err(map_metal_err)?;
        Ok(Self {
            ctx,
            pipeline_1q,
            rng,
        })
    }
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
        Ok(MetalSvState { num_qubits, amps })
    }

    fn apply_gate(
        &mut self,
        _state: &mut Self::State,
        _gate: &GateInstance,
    ) -> Result<(), BackendError> {
        // Filled in Task 5.
        Err(BackendError::UnsupportedInstruction { kind: "apply_gate" })
    }

    fn measure(&mut self, _state: &mut Self::State, _qubit: u32) -> Result<bool, BackendError> {
        // Filled in Task 6.
        Err(BackendError::UnsupportedInstruction { kind: "measure" })
    }

    fn sample(&mut self, _state: &Self::State, _shots: u32) -> Result<Vec<u64>, BackendError> {
        // Filled in Task 6.
        Err(BackendError::UnsupportedInstruction { kind: "sample" })
    }

    fn expectation_value(
        &mut self,
        _state: &Self::State,
        _pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        // Filled in Task 6.
        Err(BackendError::UnsupportedInstruction {
            kind: "expectation_value",
        })
    }

    fn probabilities(
        &mut self,
        _state: &Self::State,
        _qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        // Filled in Task 6.
        Err(BackendError::UnsupportedInstruction {
            kind: "probabilities",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
