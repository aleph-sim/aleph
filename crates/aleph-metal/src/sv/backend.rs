//! `MetalSvBackend` — device-resident FP32 state-vector backend (P5.5-02).
//! Mirrors `Fp32SvBackend` over a Metal GPU buffer. The f64 `NaiveSvBackend`
//! remains the oracle reference; this is a GPU mode at f32 accuracy.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateError, GateInstance, GateMatrix, PauliString, AMPLITUDE_TOL};
use metal::{ComputePipelineState, MTLSize};
use rand::{rngs::StdRng, SeedableRng};
use std::ffi::c_void;

use super::kernel::{Gate1q, SV_1Q_ENTRY, SV_1Q_SRC};
use super::state::MetalSvState;
use crate::{DeviceBuffer, Error, MetalContext};

/// Soft qubit cap — the project-wide 28-qubit software limit, matching the CPU
/// backends. At 8 B/amp the memory ceiling is higher, but 28 binds first.
pub(crate) const MAX_METAL_QUBITS: u32 = 28;

/// Opt-in single-precision Metal GPU state-vector backend.
pub struct MetalSvBackend {
    ctx: MetalContext,
    // Used by the apply_gate dispatch (Task 5).
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

    /// Encode and run one 1q-kernel dispatch over `2^(n-1)` pairs, then block
    /// until the GPU finishes so the unified-memory buffer is current for any
    /// subsequent host read or gate.
    fn dispatch_1q(&self, state: &MetalSvState, g: &Gate1q) {
        let pairs = 1u64 << (state.num_qubits - 1); // num_qubits ≥ 1 here
        let cmd = self.ctx.queue().new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_1q);
        encoder.set_buffer(0, Some(state.amps.metal_buffer()), 0);
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
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

/// Narrow an f64 gate-matrix entry to f32. Matrices are materialised in f64
/// (angles keep full precision); only the state is single-precision.
#[inline]
fn narrow(z: Complex<f64>) -> Complex<f32> {
    Complex::<f32>::new(z.re as f32, z.im as f32)
}

/// Max deviation of `M†M` from the identity for a 2×2 — local reimplementation
/// of `aleph-sv`'s `validation::unitarity_deviation` (which is `pub(crate)`),
/// restricted to the 2×2 case this backend handles.
fn unitarity_deviation_2x2(m: &[[Complex<f64>; 2]; 2]) -> f64 {
    // (M†M)[r][c] = Σ_k conj(M[k][r]) * M[k][c]; compare to δ_rc.
    let mut max_dev = 0.0_f64;
    #[allow(clippy::needless_range_loop)]
    for r in 0..2 {
        for c in 0..2 {
            let mut acc = Complex::<f64>::new(0.0, 0.0);
            for k in 0..2 {
                acc += m[k][r].conj() * m[k][c];
            }
            let target = if r == c {
                Complex::<f64>::new(1.0, 0.0)
            } else {
                Complex::<f64>::new(0.0, 0.0)
            };
            let dev = (acc - target).norm();
            if dev > max_dev {
                max_dev = dev;
            }
        }
    }
    max_dev
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
        // This ticket: 1q gates only. `UnitaryKq` has no fixed GateMatrix;
        // intercept it (and any non-1q arity) as unsupported before matrix().
        if matches!(gate.gate, aleph_core::Gate::UnitaryKq { .. }) || expected != 1 {
            return Err(BackendError::UnsupportedGate {
                kind: gate.gate.name(),
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
        let m = match matrix {
            GateMatrix::M2x2(m) => m,
            // 4x4 / 8x8 are 2q/3q gates — not this ticket.
            _ => {
                return Err(BackendError::UnsupportedGate {
                    kind: gate.gate.name(),
                })
            }
        };
        // Unitarity check on the f64 matrix (defense-in-depth, mirrors the CPU
        // FP32 backend) before narrowing.
        let deviation = unitarity_deviation_2x2(&m);
        if !deviation.is_finite() || deviation > AMPLITUDE_TOL {
            return Err(BackendError::NonUnitaryMatrix { deviation });
        }
        let target = gate.qubits[0];
        let ctrl_mask = gate.controls.iter().fold(0u32, |acc, &c| acc | (1u32 << c));
        let g = Gate1q {
            m: [narrow(m[0][0]), narrow(m[0][1]), narrow(m[1][0]), narrow(m[1][1])],
            target,
            t_bit: 1u32 << target,
            ctrl_mask,
            _pad: 0,
        };
        self.dispatch_1q(state, &g);
        Ok(())
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
    use aleph_core::Gate;

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

    #[test]
    fn two_qubit_gate_is_unsupported() {
        let Some(mut b) = backend_or_skip() else {
            return;
        };
        let mut s = b.allocate(2).unwrap();
        let err = b.apply_gate(&mut s, &gate(Gate::Cnot, &[0, 1])).unwrap_err();
        assert_eq!(err, BackendError::UnsupportedGate { kind: "Cnot" });
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
}
