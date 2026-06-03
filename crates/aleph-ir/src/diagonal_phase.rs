//! Symbolic multi-qubit diagonal operator produced by `FuseDiagonalRuns`.
//!
//! The amplitude at basis index `x` is multiplied by
//! `exp(i * Σ_t angle_t * [∀ m ∈ conds_t: parity(m & x) == 1])`,
//! where `parity(v) = v.count_ones() & 1`. An empty `conds` is a
//! vacuously-true (global-phase) term. Masks are `u64`; the producer
//! asserts `n_qubits <= 64`.

use smallvec::SmallVec;

/// A single term in a [`DiagonalPhase`]: contributes `angle` (radians) to the
/// accumulated phase whenever **all** parity conditions are satisfied.
///
/// # Condition semantics
///
/// For a mask `m` and basis index `x`, the condition fires when
/// `(m & x).count_ones() & 1 == 1` — i.e. when the parity of the
/// bits selected by `m` is odd. An empty `conds` list is vacuously
/// true (global phase).
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTerm {
    /// AND of these parity-conditions. Empty == global phase.
    pub conds: SmallVec<[u64; 2]>,
    /// Phase contribution in radians when all conditions are satisfied.
    pub angle: f64,
}

/// Symbolic representation of a fused multi-qubit diagonal operator.
///
/// The phase applied to amplitude `x` is
/// `Σ_t angle_t * [∀ m ∈ conds_t: parity(m & x) == 1]`.
///
/// # Example
///
/// ```
/// use aleph_ir::diagonal_phase::{DiagonalPhase, PhaseTerm};
/// use smallvec::smallvec;
///
/// // P(θ) on qubit 1: fires when bit 1 is set.
/// let dp = DiagonalPhase {
///     n_qubits: 2,
///     terms: vec![PhaseTerm { conds: smallvec![0b10], angle: 0.5 }],
/// };
/// assert_eq!(dp.phase_at(0b10), 0.5);
/// assert_eq!(dp.phase_at(0b00), 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalPhase {
    /// Number of qubits in scope. Must be ≤ 64.
    pub n_qubits: u32,
    /// Ordered list of phase terms.
    pub terms: Vec<PhaseTerm>,
}

impl DiagonalPhase {
    /// Real phase (radians) applied to amplitude index `x`.
    ///
    /// Iterates all terms; for each term whose conditions are all
    /// satisfied, accumulates the term's angle.
    pub fn phase_at(&self, x: u64) -> f64 {
        let mut phi = 0.0;
        for t in &self.terms {
            if t.conds.iter().all(|&m| (m & x).count_ones() & 1 == 1) {
                phi += t.angle;
            }
        }
        phi
    }

    /// Union of all qubit indices referenced by any condition mask.
    ///
    /// Returns a bitmask where bit `k` is set if qubit `k` appears in
    /// at least one condition across all terms.
    pub fn support_mask(&self) -> u64 {
        self.terms
            .iter()
            .flat_map(|t| t.conds.iter())
            .fold(0u64, |acc, &m| acc | m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn phase_at_single_bit_term() {
        // p(θ) on qubit 1: fires when bit 1 set.
        let dp = DiagonalPhase {
            n_qubits: 3,
            terms: vec![PhaseTerm {
                conds: smallvec![0b010],
                angle: 0.5,
            }],
        };
        assert_eq!(dp.phase_at(0b000), 0.0);
        assert_eq!(dp.phase_at(0b010), 0.5);
        assert_eq!(dp.phase_at(0b011), 0.5);
        assert_eq!(dp.phase_at(0b101), 0.0);
    }

    #[test]
    fn phase_at_and_of_two_conds_is_controlled() {
        // controlled-Phase(θ), ctrl 0, tgt 1: fires only when both set.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![0b01, 0b10],
                angle: 0.7,
            }],
        };
        assert_eq!(dp.phase_at(0b00), 0.0);
        assert_eq!(dp.phase_at(0b01), 0.0);
        assert_eq!(dp.phase_at(0b10), 0.0);
        assert_eq!(dp.phase_at(0b11), 0.7);
    }

    #[test]
    fn empty_conds_is_global_phase() {
        let dp = DiagonalPhase {
            n_qubits: 1,
            terms: vec![PhaseTerm {
                conds: smallvec![],
                angle: 1.1,
            }],
        };
        assert_eq!(dp.phase_at(0), 1.1);
        assert_eq!(dp.phase_at(1), 1.1);
    }

    #[test]
    fn support_mask_unions_all_conds() {
        let dp = DiagonalPhase {
            n_qubits: 4,
            terms: vec![
                PhaseTerm {
                    conds: smallvec![0b0011],
                    angle: 0.1,
                },
                PhaseTerm {
                    conds: smallvec![0b1000, 0b0010],
                    angle: 0.2,
                },
            ],
        };
        assert_eq!(dp.support_mask(), 0b1011);
    }
}
