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
// Aer-compatible constructors
// ---------------------------------------------------------------------------

/// All single-qubit Paulis in fixed order I, X, Y, Z.
const PAULI1: [Pauli; 4] = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

/// Depolarizing channel on `num_qubits` qubits with total error probability `p`.
///
/// Matches Qiskit Aer's `depolarizing_error(p, num_qubits)`: the error is a
/// uniform mixture over the `d²` Pauli operators (`d = 2^num_qubits`).
/// The identity carries weight `1 - p·(d²-1)/d²` and each of the other `d²-1`
/// Paulis carries weight `p/d²`. For 1q: I has `1 - 3p/4`, X/Y/Z have `p/4`.
/// For 2q: II has `1 - 15p/16`, each other pair has `p/16`.
///
/// Panics if `num_qubits ∉ {1, 2}` or `p ∉ [0, 1]`.
pub fn depolarizing_error(p: f64, num_qubits: u8) -> QuantumError {
    assert!((0.0..=1.0).contains(&p), "depolarizing p must be in [0,1]");
    assert!(
        num_qubits == 1 || num_qubits == 2,
        "v1 depolarizing is 1q or 2q"
    );
    let d2 = 1usize << (2 * num_qubits as u32); // d² = 4^num_qubits
    let off = p / d2 as f64;
    let mut terms = Vec::with_capacity(d2);
    if num_qubits == 1 {
        for pl in PAULI1 {
            let w = if pl == Pauli::I {
                1.0 - p * 3.0 / 4.0
            } else {
                off
            };
            terms.push((w, SmallVec::from_slice(&[pl])));
        }
    } else {
        for a in PAULI1 {
            for b in PAULI1 {
                let is_ii = a == Pauli::I && b == Pauli::I;
                let w = if is_ii { 1.0 - p * 15.0 / 16.0 } else { off };
                terms.push((w, SmallVec::from_slice(&[a, b])));
            }
        }
    }
    QuantumError::Pauli(PauliChannel {
        arity: num_qubits,
        terms,
    })
}

/// Amplitude damping channel with decay parameter `gamma` (0 ≤ γ ≤ 1).
///
/// Kraus operators: K₀ = diag(1, √(1-γ)), K₁ = √γ·|0⟩⟨1|.
/// Satisfies Σ Kᵢ†Kᵢ = I (CPTP).
pub fn amplitude_damping_error(gamma: f64) -> QuantumError {
    assert!((0.0..=1.0).contains(&gamma), "gamma must be in [0,1]");
    let z = Complex::new(0.0, 0.0);
    // K₀ = [[1, 0], [0, √(1-γ)]]
    let k0 = [
        [Complex::new(1.0, 0.0), z],
        [z, Complex::new((1.0 - gamma).sqrt(), 0.0)],
    ];
    // K₁ = [[0, √γ], [0, 0]]
    let k1 = [[z, Complex::new(gamma.sqrt(), 0.0)], [z, z]];
    QuantumError::Kraus(KrausChannel {
        kraus: vec![k0, k1],
    })
}

/// Phase damping (dephasing) channel with parameter `lam` (0 ≤ λ ≤ 1).
///
/// Kraus operators: K₀ = diag(1, √(1-λ)), K₁ = diag(0, √λ).
/// Satisfies Σ Kᵢ†Kᵢ = I (CPTP).
pub fn phase_damping_error(lam: f64) -> QuantumError {
    assert!((0.0..=1.0).contains(&lam), "lambda must be in [0,1]");
    let z = Complex::new(0.0, 0.0);
    // K₀ = [[1, 0], [0, √(1-λ)]]
    let k0 = [
        [Complex::new(1.0, 0.0), z],
        [z, Complex::new((1.0 - lam).sqrt(), 0.0)],
    ];
    // K₁ = [[0, 0], [0, √λ]]
    let k1 = [[z, z], [z, Complex::new(lam.sqrt(), 0.0)]];
    QuantumError::Kraus(KrausChannel {
        kraus: vec![k0, k1],
    })
}

/// Bit-flip channel: identity with probability `1-p`, X with probability `p`.
pub fn bit_flip_error(p: f64) -> QuantumError {
    single_pauli_flip(Pauli::X, p)
}

/// Phase-flip channel: identity with probability `1-p`, Z with probability `p`.
pub fn phase_flip_error(p: f64) -> QuantumError {
    single_pauli_flip(Pauli::Z, p)
}

fn single_pauli_flip(pl: Pauli, p: f64) -> QuantumError {
    assert!((0.0..=1.0).contains(&p), "flip p must be in [0,1]");
    QuantumError::Pauli(PauliChannel {
        arity: 1,
        terms: vec![
            (1.0 - p, SmallVec::from_slice(&[Pauli::I])),
            (p, SmallVec::from_slice(&[pl])),
        ],
    })
}

/// Build a single-qubit Pauli channel from `(label, prob)` pairs; each label is
/// one of "I","X","Y","Z". Weights are renormalized to sum to 1 (mirrors Aer's
/// `pauli_error`). v1 is 1q only — for 2-qubit Pauli noise use
/// [`depolarizing_error`]`(p, 2)`.
pub fn pauli_error(terms: &[(&str, f64)]) -> QuantumError {
    let total: f64 = terms.iter().map(|(_, p)| *p).sum();
    assert!(total > 0.0, "pauli_error weights must sum to > 0");
    let parse = |s: &str| match s {
        "I" => Pauli::I,
        "X" => Pauli::X,
        "Y" => Pauli::Y,
        "Z" => Pauli::Z,
        other => panic!(
            "pauli_error label {other:?} is not a single-qubit Pauli (I/X/Y/Z); \
             pauli_error is 1q-only — use depolarizing_error(p, 2) for 2-qubit noise"
        ),
    };
    let terms = terms
        .iter()
        .map(|(s, p)| (p / total, SmallVec::from_slice(&[parse(s)])))
        .collect();
    QuantumError::Pauli(PauliChannel { arity: 1, terms })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Σ over a Pauli channel's weights must equal 1 (CPTP for a Pauli mix).
    fn pauli_weight_sum(c: &PauliChannel) -> f64 {
        c.terms.iter().map(|(p, _)| *p).sum()
    }

    /// Σ Kᵢ† Kᵢ for a 1q Kraus set, as a 2×2 matrix.
    fn kraus_completeness(c: &KrausChannel) -> [[Complex; 2]; 2] {
        let mut acc = [[Complex::new(0.0, 0.0); 2]; 2];
        for k in &c.kraus {
            // (K† K)[r][c] = Σ_s conj(K[s][r]) * K[s][c]
            for r in 0..2 {
                for col in 0..2 {
                    let sum: Complex = k.iter().map(|row| row[r].conj() * row[col]).sum();
                    acc[r][col] += sum;
                }
            }
        }
        acc
    }

    fn assert_is_identity(m: [[Complex; 2]; 2]) {
        let eye = [
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        ];
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (m[r][c] - eye[r][c]).norm() < 1e-12,
                    "ΣK†K[{r}][{c}] = {:?}, expected I",
                    m[r][c]
                );
            }
        }
    }

    #[test]
    fn depolarizing_1q_weights_sum_to_one_and_match_aer() {
        let QuantumError::Pauli(c) = depolarizing_error(0.1, 1) else {
            panic!("1q depolarizing must be a Pauli channel");
        };
        assert!((pauli_weight_sum(&c) - 1.0).abs() < 1e-12);
        // Aer parameterization: I weight = 1 - 3p/4, each of X,Y,Z = p/4.
        let i_weight = c.terms.iter().find(|(_, p)| p[0] == Pauli::I).unwrap().0;
        assert!((i_weight - (1.0 - 3.0 * 0.1 / 4.0)).abs() < 1e-12);
        for pl in [Pauli::X, Pauli::Y, Pauli::Z] {
            let w = c.terms.iter().find(|(_, p)| p[0] == pl).unwrap().0;
            assert!((w - 0.1 / 4.0).abs() < 1e-12, "{pl:?} weight {w}");
        }
    }

    #[test]
    fn depolarizing_2q_weights_sum_to_one_and_match_aer() {
        let QuantumError::Pauli(c) = depolarizing_error(0.2, 2) else {
            panic!("2q depolarizing must be a Pauli channel");
        };
        assert_eq!(c.arity, 2);
        assert_eq!(c.terms.len(), 16); // 4×4 Paulis
        assert!((pauli_weight_sum(&c) - 1.0).abs() < 1e-12);
        // I⊗I weight = 1 - 15p/16; every other = p/16.
        let ii = c
            .terms
            .iter()
            .find(|(_, p)| p[0] == Pauli::I && p[1] == Pauli::I)
            .unwrap()
            .0;
        assert!((ii - (1.0 - 15.0 * 0.2 / 16.0)).abs() < 1e-12);
        for (w, p) in &c.terms {
            if !(p[0] == Pauli::I && p[1] == Pauli::I) {
                assert!((w - 0.2 / 16.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn amplitude_damping_is_cptp() {
        let QuantumError::Kraus(c) = amplitude_damping_error(0.3) else {
            panic!("amplitude damping must be a general Kraus channel");
        };
        assert_eq!(c.kraus.len(), 2);
        assert_is_identity(kraus_completeness(&c));
    }

    #[test]
    fn phase_damping_is_cptp() {
        let QuantumError::Kraus(c) = phase_damping_error(0.4) else {
            panic!("phase damping must be a general Kraus channel");
        };
        assert_is_identity(kraus_completeness(&c));
    }

    #[test]
    fn bit_flip_is_pauli_mix() {
        let QuantumError::Pauli(c) = bit_flip_error(0.25) else {
            panic!();
        };
        let i = c.terms.iter().find(|(_, p)| p[0] == Pauli::I).unwrap().0;
        let x = c.terms.iter().find(|(_, p)| p[0] == Pauli::X).unwrap().0;
        assert!((i - 0.75).abs() < 1e-12);
        assert!((x - 0.25).abs() < 1e-12);
    }

    #[test]
    fn pauli_error_normalizes_input() {
        let QuantumError::Pauli(c) = pauli_error(&[("X", 2.0), ("I", 8.0)]) else {
            panic!();
        };
        assert!((pauli_weight_sum(&c) - 1.0).abs() < 1e-12);
        let x = c.terms.iter().find(|(_, p)| p[0] == Pauli::X).unwrap().0;
        let i = c.terms.iter().find(|(_, p)| p[0] == Pauli::I).unwrap().0;
        assert!((x - 0.2).abs() < 1e-12, "X weight {x}");
        assert!((i - 0.8).abs() < 1e-12, "I weight {i}");
    }

    #[test]
    fn phase_flip_is_pauli_mix() {
        let QuantumError::Pauli(c) = phase_flip_error(0.25) else {
            panic!();
        };
        let i = c.terms.iter().find(|(_, p)| p[0] == Pauli::I).unwrap().0;
        let z = c.terms.iter().find(|(_, p)| p[0] == Pauli::Z).unwrap().0;
        assert!((i - 0.75).abs() < 1e-12);
        assert!((z - 0.25).abs() < 1e-12);
    }
}
