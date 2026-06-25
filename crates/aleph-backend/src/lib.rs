//! `aleph-backend`: the `Backend` trait, shared `BackendError`, and a
//! `run<B>` driver. Backend implementations live in `aleph-sv` (naive
//! CPU state vector), `aleph-mps`, `aleph-stab`, etc.
//!
//! See `docs/superpowers/specs/2026-05-24-p0-09-backend-naive-sv-design.md`.

pub mod select;
pub use select::{
    analyze, select_backend, select_explained, select_explained_env, select_explained_full,
    select_from, select_from_env, select_from_full, BackendKind, BackendRequest, CircuitFeatures,
    Precision, Reach, SelectEnv, Selection, GPU_PREFER_N, MAX_CUDA_QUBITS, MAX_CUDA_QUBITS_F32,
    MPS_DEPTH_THRESHOLD, SV_EXACT_CAP,
};

/// Errors common to every backend.
///
/// Backends share one concrete error type rather than an associated
/// `type Error` so that the `run<B>` driver and downstream code (CLI,
/// Python bindings) don't have to be generic over an open-ended error.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BackendError {
    #[error("qubit {qubit} out of range for {num_qubits}-qubit state")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },

    #[error("duplicate qubit {qubit} in gate or query")]
    DuplicateQubit { qubit: u32 },

    #[error("gate `{kind}` expects {expected} qubits, got {got}")]
    ArityMismatch {
        kind: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("gate `{kind}` is not supported by this backend")]
    UnsupportedGate { kind: &'static str },

    #[error("IR instruction `{kind}` is not supported by this backend")]
    UnsupportedInstruction { kind: &'static str },

    #[error("backend requires concrete parameters; got symbolic")]
    SymbolicParam,

    #[error("gate `{kind}` has a non-finite (NaN or infinite) parameter")]
    NonFiniteParam { kind: &'static str },

    #[error("user-supplied matrix is not unitary (max deviation = {deviation:e})")]
    NonUnitaryMatrix { deviation: f64 },

    #[error("cannot run an empty circuit")]
    EmptyCircuit,

    #[error("measurement of qubit {qubit} on degenerate branch (p = {probability:e})")]
    DegenerateMeasurement { qubit: u32, probability: f64 },

    #[error("requested {requested} qubits exceeds backend limit of {limit}")]
    TooManyQubits { requested: u32, limit: u32 },

    #[error("Pauli string violates its invariants: {reason}")]
    InvalidPauliString { reason: &'static str },

    #[error("backend state is invalid: {reason}")]
    InvalidState { reason: &'static str },

    #[error("MPS backend cannot truncate (bond cap {max_bond}); applying this gate would drop Schmidt weight {trunc_error:e}. Raise max_bond or use a statevector backend")]
    MpsTruncationUnsupported { max_bond: usize, trunc_error: f64 },

    #[error("optimization pipeline failed: {0}")]
    Optimization(#[from] PassError),
}

use aleph_core::{GateInstance, PauliString};
use aleph_ir::passes::PassError;
use aleph_ir::Circuit;

/// A simulation backend.
///
/// Backends own no state vector; they construct and return one through
/// `allocate`, then mutate it in place via `apply_gate` / `measure`.
/// Query methods (`sample`, `expectation_value`, `probabilities`) take
/// `&Self::State` and do not mutate the state.
pub trait Backend {
    /// Backend-specific representation (state vector, MPS tensors,
    /// stabilizer tableau, …).
    type State;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError>;

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError>;

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError>;

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError>;

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError>;

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError>;

    /// Apply a fused multi-qubit diagonal (`Instruction::DiagonalPhase`).
    ///
    /// Default implementation rejects it as unsupported, so backends that
    /// never see optimized circuits (MPS/stabilizer, for now) need not
    /// implement it. State-vector backends override this.
    fn apply_diagonal_phase(
        &mut self,
        _state: &mut Self::State,
        _dp: &aleph_ir::DiagonalPhase,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedInstruction {
            kind: "diagonal_phase",
        })
    }

    /// Apply a cache-tile-confinable run (`Instruction::TiledBlock`).
    ///
    /// Default implementation replays each gate via `apply_gate` in order
    /// — semantically identical to executing the gates individually, just
    /// without the tile-major cache benefit. State-vector backends with a
    /// tiled fast path override this; others (SoA, FP32, MPS) inherit the
    /// correct replay.
    fn apply_tiled_block(
        &mut self,
        state: &mut Self::State,
        block: &aleph_ir::TiledBlock,
    ) -> Result<(), BackendError> {
        for gate in &block.gates {
            self.apply_gate(state, gate)?;
        }
        Ok(())
    }

    /// Reorder a physical-bit-order state into logical order per `perm`
    /// (`perm[logical] = physical`), undoing a `RelabelQubits` permutation.
    /// Called by `run_optimized_with_outcomes` exactly once, only when the
    /// optimized circuit carries a permutation. Default errors so a backend
    /// that can't un-permute never silently returns a physically-ordered
    /// (wrong) state; state-vector backends override it.
    fn unpermute_state(
        &mut self,
        _state: &mut Self::State,
        _perm: &[u32],
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedInstruction {
            kind: "unpermute_state",
        })
    }
}

/// Run `circuit` on `backend`, returning the final backend state.
///
/// Iterates instructions in order, dispatching `Instruction::Gate` to
/// `Backend::apply_gate`. Non-gate instructions are handled inline:
///
/// * `Measure { qubit, .. }` calls `Backend::measure` and **discards**
///   the outcome. Use [`run_with_outcomes`] if you need to keep the
///   measurement record (e.g. for shot-based oracle comparison).
/// * `Reset(q)` is rejected as
///   [`BackendError::UnsupportedInstruction`] `{ kind: "reset" }`
///   because the naive backend doesn't yet express mid-circuit reset
///   declaratively. P0-13+ may revisit.
/// * `Barrier(_)` is a no-op (semantic-only).
///
/// Returns [`BackendError::EmptyCircuit`] only when the circuit declares
/// zero qubits **and** has zero instructions — the truly-degenerate
/// input.
pub fn run<B: Backend>(backend: &mut B, circuit: &Circuit) -> Result<B::State, BackendError> {
    let (state, _outcomes) = run_with_outcomes(backend, circuit)?;
    Ok(state)
}

/// One recorded measurement outcome from `run_with_outcomes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementRecord {
    /// Index of the `Instruction::Measure` within `circuit.instructions()`.
    pub instruction_index: usize,
    pub qubit: u32,
    pub clbit: u32,
    pub outcome: bool,
}

/// Run `circuit` on `backend` AND return every measurement outcome.
///
/// Same semantics as [`run`], but preserves the bool returned by each
/// `Backend::measure` call. Use this driver when downstream code needs
/// to inspect mid-circuit outcomes (postselection, oracle comparison
/// against shot-based references like Qiskit Aer's `meas_level=2`).
///
/// **Ordering contract:** the returned `Vec<MeasurementRecord>` is in
/// the same order as the corresponding `Instruction::Measure` entries
/// in `circuit.instructions()`. In particular,
/// `outcomes[i].instruction_index` is strictly increasing and equals
/// the position of the i-th measurement instruction within the
/// circuit. Downstream consumers (oracle harness, postselection logic)
/// may rely on this ordering.
pub fn run_with_outcomes<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<(B::State, Vec<MeasurementRecord>), BackendError> {
    if circuit.num_qubits() == 0 && circuit.is_empty() {
        return Err(BackendError::EmptyCircuit);
    }
    let mut state = backend.allocate(circuit.num_qubits())?;
    let mut outcomes = Vec::new();
    for (idx, inst) in circuit.instructions().iter().enumerate() {
        match inst {
            aleph_ir::Instruction::Gate(g) => backend.apply_gate(&mut state, g)?,
            aleph_ir::Instruction::Measure { qubit, clbit } => {
                let outcome = backend.measure(&mut state, *qubit)?;
                outcomes.push(MeasurementRecord {
                    instruction_index: idx,
                    qubit: *qubit,
                    clbit: *clbit,
                    outcome,
                });
            }
            aleph_ir::Instruction::Reset(_) => {
                return Err(BackendError::UnsupportedInstruction { kind: "reset" });
            }
            aleph_ir::Instruction::Barrier(_) => {}
            aleph_ir::Instruction::DiagonalPhase(dp) => {
                backend.apply_diagonal_phase(&mut state, dp)?;
            }
            aleph_ir::Instruction::TiledBlock(tb) => {
                backend.apply_tiled_block(&mut state, tb)?;
            }
        }
    }
    Ok((state, outcomes))
}

/// Optimize `circuit` with the default IR pipeline, then simulate.
///
/// Unlike [`run`], which executes the circuit verbatim (the raw reference
/// path used by oracle tests), this first runs `Circuit::optimize`
/// (`PassPipeline::default_pipeline`: cancellation, DCE, and 1q/2q fusion).
/// Semantics are preserved — see the end-to-end oracle in
/// `tests/run_optimized_oracle.rs` — and the win is far fewer state-vector
/// passes (QFT collapses ~5x: 970->190 gates at n=20).
///
/// The optimization runs on a clone, so the caller's `circuit` is untouched.
pub fn run_optimized<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<B::State, BackendError> {
    let (state, _outcomes) = run_optimized_with_outcomes(backend, circuit)?;
    Ok(state)
}

/// [`run_optimized`] preserving measurement outcomes.
///
/// The outcomes are in the same order, and `outcome`/`qubit`/`clbit` match what
/// [`run_with_outcomes`] returns for the same circuit and seed (the optimization
/// pipeline must not reorder gates across `Measure`/`Barrier`; that invariant is
/// pinned by `tests/run_optimized_oracle.rs`). **Note:** each record's
/// `instruction_index` refers to the *optimized* circuit, so it can be smaller
/// than the index of the same measurement in the caller's input circuit (fusion
/// and cancellation drop earlier gates). Compare on `(qubit, clbit, outcome)`,
/// not on the absolute `instruction_index`, when relating outcomes back to the
/// pre-optimization circuit.
///
/// **Relabelling transparency:** the default pipeline's `RelabelQubits` pass may
/// permute qubit indices for cache locality (`perm[logical] = physical`), in
/// which case the optimized circuit carries a permutation
/// ([`Circuit::qubit_permutation`]), the simulated state ends in *physical*-bit
/// order, and `Measure` outcomes are recorded against *physical* qubits. This
/// driver makes that invisible: it maps each outcome's `qubit` back to its
/// logical index and applies a single final gather via
/// [`Backend::unpermute_state`] so the returned state is logical-order — exactly
/// as if no relabelling had occurred. A backend that doesn't override
/// `unpermute_state` surfaces [`BackendError::UnsupportedInstruction`] rather
/// than silently returning a physically-ordered (wrong) state.
pub fn run_optimized_with_outcomes<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<(B::State, Vec<MeasurementRecord>), BackendError> {
    let mut optimized = circuit.clone();
    optimized.optimize()?; // PassError -> BackendError via #[from]
    let perm = optimized.qubit_permutation().map(|p| p.to_vec());
    let (mut state, mut outcomes) = run_with_outcomes(backend, &optimized)?;
    if let Some(perm) = perm {
        // RelabelQubits rewrote Measure qubits to physical; report them logical.
        // logical_of[physical] = logical.
        let logical_of = invert_perm(&perm);
        for rec in &mut outcomes {
            rec.qubit = logical_of[rec.qubit as usize];
        }
        // Single final gather: physical-order state → logical order.
        backend.unpermute_state(&mut state, &perm)?;
    }
    Ok((state, outcomes))
}

/// `inv[perm[l]] = l` — invert a qubit permutation (`perm[logical] =
/// physical` ⟹ `inv[physical] = logical`).
fn invert_perm(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (logical, &physical) in perm.iter().enumerate() {
        inv[physical as usize] = logical as u32;
    }
    inv
}

/// Energy of a state under a Pauli-sum observable: `Σ_i ⟨ψ|c_i P_i|ψ⟩`.
///
/// Thin loop over [`Backend::expectation_value`] (each [`aleph_core::PauliString`]
/// carries its coefficient). This is the VQE energy primitive.
pub fn expectation_pauli_sum<B: Backend>(
    backend: &mut B,
    state: &B::State,
    hamiltonian: &aleph_core::PauliSum,
) -> Result<f64, BackendError> {
    let mut energy = 0.0;
    for term in &hamiltonian.terms {
        energy += backend.expectation_value(state, term)?;
    }
    Ok(energy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_render() {
        let e = BackendError::QubitOutOfRange {
            qubit: 7,
            num_qubits: 3,
        };
        assert_eq!(e.to_string(), "qubit 7 out of range for 3-qubit state");

        let e = BackendError::DegenerateMeasurement {
            qubit: 0,
            probability: 1e-301,
        };
        assert!(e.to_string().contains("p = 1e-301"));
    }

    /// Minimal stub backend that does NOT override `apply_diagonal_phase`.
    /// Used to verify the default trait-method path surfaces the correct error.
    struct StubBackend;

    impl Backend for StubBackend {
        type State = ();

        fn allocate(&mut self, _num_qubits: u32) -> Result<Self::State, BackendError> {
            Ok(())
        }

        fn apply_gate(
            &mut self,
            _state: &mut Self::State,
            _gate: &aleph_core::GateInstance,
        ) -> Result<(), BackendError> {
            Ok(())
        }

        fn measure(&mut self, _state: &mut Self::State, _qubit: u32) -> Result<bool, BackendError> {
            Ok(false)
        }

        fn sample(&mut self, _state: &Self::State, _shots: u32) -> Result<Vec<u64>, BackendError> {
            Ok(vec![])
        }

        fn expectation_value(
            &mut self,
            _state: &Self::State,
            _pauli: &aleph_core::PauliString,
        ) -> Result<f64, BackendError> {
            Ok(0.0)
        }

        fn probabilities(
            &mut self,
            _state: &Self::State,
            qubits: &[u32],
        ) -> Result<Vec<f64>, BackendError> {
            Ok(vec![0.0; 1 << qubits.len()])
        }
    }

    /// The default `apply_tiled_block` must call `apply_gate` for each gate
    /// in the block, in order.
    ///
    /// Uses a recording variant of `StubBackend` that logs `apply_gate` calls
    /// (by gate kind name). Two gates H/CNOT wrapped in a `TiledBlock` must
    /// produce exactly the same two recorded calls as running them individually.
    ///
    /// We use the recording approach rather than `NaiveSvBackend` because
    /// `aleph-sv` re-exports `Backend` from its own copy of `aleph-backend`,
    /// causing a trait-version mismatch when `aleph-backend` is being compiled
    /// as the crate under test.
    #[test]
    fn default_apply_tiled_block_replays_gates_in_order() {
        use aleph_core::{Gate, GateInstance};
        use aleph_ir::TiledBlock;
        use smallvec::smallvec;
        use std::cell::RefCell;
        use std::rc::Rc;

        // A backend that records the name of each gate passed to apply_gate.
        struct RecordingBackend {
            log: Rc<RefCell<Vec<&'static str>>>,
        }

        impl Backend for RecordingBackend {
            type State = ();

            fn allocate(&mut self, _n: u32) -> Result<(), BackendError> {
                Ok(())
            }

            fn apply_gate(
                &mut self,
                _state: &mut (),
                gate: &GateInstance,
            ) -> Result<(), BackendError> {
                self.log.borrow_mut().push(gate.gate.name());
                Ok(())
            }

            fn measure(&mut self, _state: &mut (), _qubit: u32) -> Result<bool, BackendError> {
                Ok(false)
            }

            fn sample(&mut self, _state: &(), _shots: u32) -> Result<Vec<u64>, BackendError> {
                Ok(vec![])
            }

            fn expectation_value(
                &mut self,
                _state: &(),
                _pauli: &aleph_core::PauliString,
            ) -> Result<f64, BackendError> {
                Ok(0.0)
            }

            fn probabilities(
                &mut self,
                _state: &(),
                qubits: &[u32],
            ) -> Result<Vec<f64>, BackendError> {
                Ok(vec![0.0; 1 << qubits.len()])
            }
        }

        let h = GateInstance::new(Gate::H, smallvec![0u32]);
        let cnot = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);

        let tb = TiledBlock {
            gates: vec![h.clone(), cnot.clone()],
            tile_bits: 3,
        };

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut backend = RecordingBackend {
            log: Rc::clone(&log),
        };
        let mut state = ();
        backend
            .apply_tiled_block(&mut state, &tb)
            .expect("apply_tiled_block must succeed");

        assert_eq!(
            *log.borrow(),
            vec![h.gate.name(), cnot.gate.name()],
            "default replay must call apply_gate for each gate in order"
        );
    }

    /// `run` on a circuit containing a `DiagonalPhase` instruction must
    /// propagate `UnsupportedInstruction { kind: "diagonal_phase" }` when
    /// the backend uses the default (non-overriding) implementation.
    #[test]
    fn default_apply_diagonal_phase_returns_unsupported() {
        use aleph_ir::{Circuit, DiagonalPhase, Instruction, PhaseTerm};
        use smallvec::smallvec;

        // Build a 1-qubit circuit with a single DiagonalPhase instruction:
        // P(π/4) on qubit 0 — fires when bit 0 is set.
        let dp = DiagonalPhase {
            n_qubits: 1,
            terms: vec![PhaseTerm {
                conds: smallvec![0b1u64],
                angle: std::f64::consts::FRAC_PI_4,
            }],
        };
        let mut circuit = Circuit::new(1, 0);
        circuit
            .add_instruction(Instruction::DiagonalPhase(Box::new(dp)))
            .expect("valid 1-qubit DiagonalPhase");

        let mut stub = StubBackend;
        let result = run(&mut stub, &circuit);
        assert_eq!(
            result,
            Err(BackendError::UnsupportedInstruction {
                kind: "diagonal_phase"
            }),
            "default apply_diagonal_phase must return UnsupportedInstruction"
        );
    }
}
