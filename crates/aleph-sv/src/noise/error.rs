//! Noise channel data types and Aer-named constructors.
//!
//! A [`QuantumError`] is a CPTP map attached to a gate. v1 splits into two
//! application strategies: [`PauliChannel`] (state-independent weights — the
//! quantum-jump fast path) and [`KrausChannel`] (general 1q operators applied
//! via `pᵢ=‖Kᵢ|ψ〉‖²`). See the P4.6-03 design spec §3.

use aleph_core::{Complex, Pauli};
use smallvec::SmallVec;

/// A probabilistic mixture of (multi-qubit) Pauli operators. `Σ probs = 1`.
///
/// `terms[i] = (prob, paulis)` where `paulis[j]` is the Pauli applied to the
/// channel's local qubit `j` (so `paulis.len() == arity`). State-independent:
/// the branch is sampled directly from `prob`, no norm computation.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliChannel {
    pub arity: u8,
    pub terms: Vec<(f64, SmallVec<[Pauli; 2]>)>,
}

/// A general single-qubit CPTP map given by its Kraus operators.
/// `Σ Kᵢ† Kᵢ = I`. Applied by quantum-jump (compute `pᵢ`, sample, renormalize).
/// v1: single-qubit only (amplitude/phase damping); 2q noise in v1 is depolarizing, handled as a Pauli channel.
#[derive(Debug, Clone, PartialEq)]
pub struct KrausChannel {
    pub kraus: Vec<[[Complex; 2]; 2]>,
}

/// A CPTP error map attached to a gate.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumError {
    Pauli(PauliChannel),
    Kraus(KrausChannel),
}

impl QuantumError {
    /// Number of qubits this error acts on.
    pub fn arity(&self) -> usize {
        match self {
            QuantumError::Pauli(p) => p.arity as usize,
            QuantumError::Kraus(_) => 1,
        }
    }
}

/// Per-qubit classical readout error: a 2×2 row-stochastic matrix.
/// `m[t][o]` = P(measure outcome `o` | true value `t`). Rows sum to 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadoutError {
    pub m: [[f64; 2]; 2],
}

// ---------------------------------------------------------------------------
// Aer-compatible constructor stubs — bodies filled in by Task 2.
// ---------------------------------------------------------------------------

/// Depolarizing channel on `num_qubits` qubits with total error probability `p`.
/// Stub — implemented in Task 2.
pub fn depolarizing_error(p: f64, num_qubits: u8) -> QuantumError {
    let _ = (p, num_qubits);
    unimplemented!("Task 2")
}

/// Amplitude damping channel with decay parameter `gamma` (0 ≤ γ ≤ 1).
/// Stub — implemented in Task 2.
pub fn amplitude_damping_error(gamma: f64) -> QuantumError {
    let _ = gamma;
    unimplemented!("Task 2")
}

/// Phase damping (dephasing) channel with parameter `lam` (0 ≤ λ ≤ 1).
/// Stub — implemented in Task 2.
pub fn phase_damping_error(lam: f64) -> QuantumError {
    let _ = lam;
    unimplemented!("Task 2")
}

/// Bit-flip channel: identity with probability `1-p`, X with probability `p`.
/// Stub — implemented in Task 2.
pub fn bit_flip_error(p: f64) -> QuantumError {
    let _ = p;
    unimplemented!("Task 2")
}

/// Phase-flip channel: identity with probability `1-p`, Z with probability `p`.
/// Stub — implemented in Task 2.
pub fn phase_flip_error(p: f64) -> QuantumError {
    let _ = p;
    unimplemented!("Task 2")
}

/// Explicit Pauli channel from a list of `("IX", prob)` string/weight pairs.
/// Stub — implemented in Task 2.
pub fn pauli_error(terms: &[(&str, f64)]) -> QuantumError {
    let _ = terms;
    unimplemented!("Task 2")
}
