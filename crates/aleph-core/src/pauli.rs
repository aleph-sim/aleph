//! Pauli operators and Pauli strings — used by `Backend::expectation_value`.
//!
//! `Pauli` is the four single-qubit Pauli matrices `{I, X, Y, Z}`.
//! `PauliString` is a tensor product over named qubits with a real
//! coefficient, e.g. `0.5 · X₀ ⊗ Z₂`. Qubits not listed in `terms` are
//! implicit identity. `terms` is kept sorted by qubit and deduplicated.

use crate::Complex;

/// Single-qubit Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    /// 2×2 matrix in basis `|0⟩, |1⟩`.
    pub fn matrix(self) -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let i = Complex::new(0.0, 1.0);
        let neg_o = Complex::new(-1.0, 0.0);
        let neg_i = Complex::new(0.0, -1.0);
        match self {
            Pauli::I => [[o, z], [z, o]],
            Pauli::X => [[z, o], [o, z]],
            Pauli::Y => [[z, neg_i], [i, z]],
            Pauli::Z => [[o, z], [z, neg_o]],
        }
    }
}

/// A Pauli tensor-product with a real coefficient.
///
/// `terms` is sorted ascending by qubit index and contains no
/// duplicates. Construct via [`PauliString::new`] to enforce these
/// invariants, or [`PauliString::identity`] for the empty string.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliString {
    pub coefficient: f64,
    pub terms: Vec<(u32, Pauli)>,
}

impl PauliString {
    pub fn new(coefficient: f64, mut terms: Vec<(u32, Pauli)>) -> Result<Self, PauliError> {
        if !coefficient.is_finite() {
            return Err(PauliError::NonFiniteCoefficient);
        }
        terms.sort_by_key(|(q, _)| *q);
        for w in terms.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(PauliError::DuplicateQubit { qubit: w[0].0 });
            }
        }
        terms.retain(|(_, p)| *p != Pauli::I);
        Ok(Self { coefficient, terms })
    }

    pub fn identity(coefficient: f64) -> Self {
        Self {
            coefficient,
            terms: Vec::new(),
        }
    }
}

/// Errors from constructing a [`PauliString`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PauliError {
    #[error("duplicate qubit {qubit} in Pauli string")]
    DuplicateQubit { qubit: u32 },
    #[error("non-finite coefficient")]
    NonFiniteCoefficient,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_are_correct() {
        let m_x = Pauli::X.matrix();
        assert_eq!(m_x[0][1], Complex::new(1.0, 0.0));
        assert_eq!(m_x[1][0], Complex::new(1.0, 0.0));
        assert_eq!(m_x[0][0], Complex::new(0.0, 0.0));

        let m_y = Pauli::Y.matrix();
        assert_eq!(m_y[0][1], Complex::new(0.0, -1.0));
        assert_eq!(m_y[1][0], Complex::new(0.0, 1.0));

        let m_z = Pauli::Z.matrix();
        assert_eq!(m_z[0][0], Complex::new(1.0, 0.0));
        assert_eq!(m_z[1][1], Complex::new(-1.0, 0.0));
        assert_eq!(m_z[0][1], Complex::new(0.0, 0.0));

        let m_i = Pauli::I.matrix();
        assert_eq!(m_i[0][0], Complex::new(1.0, 0.0));
        assert_eq!(m_i[1][1], Complex::new(1.0, 0.0));
    }

    #[test]
    fn new_sorts_and_drops_identity() {
        let p = PauliString::new(1.0, vec![(2, Pauli::Z), (0, Pauli::X), (1, Pauli::I)]).unwrap();
        assert_eq!(p.terms, vec![(0, Pauli::X), (2, Pauli::Z)]);
    }

    #[test]
    fn new_rejects_duplicates() {
        let err = PauliString::new(1.0, vec![(0, Pauli::X), (0, Pauli::Z)]).unwrap_err();
        assert_eq!(err, PauliError::DuplicateQubit { qubit: 0 });
    }

    #[test]
    fn new_rejects_non_finite_coefficient() {
        let err = PauliString::new(f64::NAN, vec![]).unwrap_err();
        assert_eq!(err, PauliError::NonFiniteCoefficient);
        let err = PauliString::new(f64::INFINITY, vec![]).unwrap_err();
        assert_eq!(err, PauliError::NonFiniteCoefficient);
    }

    #[test]
    fn identity_is_empty() {
        let p = PauliString::identity(0.5);
        assert!(p.terms.is_empty());
        assert_eq!(p.coefficient, 0.5);
    }
}
