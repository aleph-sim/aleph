//! Error type for circuit construction.

/// Errors returned by `Circuit` builder methods. All variants are
/// recoverable. `Circuit::new` itself panics on bounds violations
/// (programmer error at construction); see `MAX_QUBITS` / `MAX_CLBITS`
/// in `crate::circuit`.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CircuitError {
    #[error("qubit {qubit} out of range (circuit has {num_qubits} qubits)")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },

    #[error("clbit {clbit} out of range (circuit has {num_clbits} clbits)")]
    ClbitOutOfRange { clbit: u32, num_clbits: u32 },

    /// Same qubit index appears more than once in a gate's `qubits ∪
    /// controls` set, or twice inside a `Barrier`. A gate with
    /// `control == target` (e.g. `Cnot(0, 0)`) is ill-defined; the IR
    /// rejects it independent of the `GateInstance` debug-assert.
    #[error("duplicate qubit index {qubit} in instruction")]
    DuplicateQubit { qubit: u32 },

    #[error("gate {gate} has arity {expected} but {got} qubits supplied")]
    ArityMismatch {
        gate: &'static str,
        expected: usize,
        got: usize,
    },

    /// `Instruction::Barrier(empty)` has no semantic content and would
    /// silently be a no-op for layer extraction. Reject at construction.
    #[error("barrier must cover at least one qubit")]
    EmptyBarrier,

    /// A `GateInstance` was constructed with more external controls than
    /// the IR is willing to validate in O(N²) uniqueness-check time.
    /// Bounded by [`crate::MAX_GATE_CONTROLS`]; existing Phase-0 gate
    /// shapes use at most 2 controls.
    #[error("gate {gate} has {controls} external controls but max is {max}")]
    TooManyControls {
        gate: &'static str,
        controls: usize,
        max: usize,
    },

    /// `Circuit::try_new` rejected a `num_qubits` above [`crate::MAX_QUBITS`].
    #[error("too many qubits: requested {requested}, max {max}")]
    TooManyQubits { requested: u32, max: u32 },

    /// `Circuit::try_new` rejected a `num_clbits` above [`crate::MAX_CLBITS`].
    #[error("too many clbits: requested {requested}, max {max}")]
    TooManyClbits { requested: u32, max: u32 },
}
