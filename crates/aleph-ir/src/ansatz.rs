//! Hardware-efficient ansatz (HEA) builder for VQE.
//!
//! `build_hea` produces the canonical "Ry + linear-CNOT" ansatz: `depth`
//! entangling layers, each a full Ry rotation layer followed by a linear CNOT
//! chain, then a final Ry rotation layer. Ry-only ⇒ real amplitudes, which is
//! sufficient for the real-symmetric H₂ Hamiltonian's real ground state.
//!
//! Distinct from the `bench_fixtures::vqe_hea` fixture (which uses a 5-rotation
//! Ry·Rz·Ry·Rz·Ry per-qubit block tuned for fusion-pass tests).

use crate::{Circuit, CircuitError};
use aleph_core::{Pauli, PauliString, PauliSum};

/// Error building a HEA circuit.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AnsatzError {
    #[error("expected {expected} params (n_qubits*(depth+1)), got {got}")]
    ParamCount { expected: usize, got: usize },
    #[error("gammas has {gammas} angles but betas has {betas} (must match = QAOA depth p)")]
    LayerMismatch { gammas: usize, betas: usize },
    #[error("edge ({i},{j}) out of range for {n} qubits")]
    EdgeOutOfRange { i: u32, j: u32, n: u32 },
    #[error(transparent)]
    Circuit(#[from] CircuitError),
    #[error(transparent)]
    Pauli(#[from] aleph_core::PauliError),
}

/// Build the Ry + linear-CNOT hardware-efficient ansatz.
///
/// `params.len()` must equal `n_qubits * (depth + 1)`: one Ry angle per qubit
/// for each of the `depth` entangling layers plus a final rotation layer.
/// Angle `params[layer * n_qubits + q]` drives qubit `q` in rotation layer
/// `layer` (`layer` in `0..=depth`).
pub fn build_hea(n_qubits: u32, depth: u32, params: &[f64]) -> Result<Circuit, AnsatzError> {
    let n = n_qubits as usize;
    let expected = n * (depth as usize + 1);
    if params.len() != expected {
        return Err(AnsatzError::ParamCount {
            expected,
            got: params.len(),
        });
    }
    let mut c = Circuit::new(n_qubits, 0);
    let row = |layer: usize| &params[layer * n..(layer + 1) * n];
    for layer in 0..depth as usize {
        let angles = row(layer);
        for q in 0..n_qubits {
            c.ry(angles[q as usize], q)?;
        }
        for q in 0..n_qubits.saturating_sub(1) {
            c.cnot(q, q + 1)?;
        }
    }
    let last = row(depth as usize);
    for q in 0..n_qubits {
        c.ry(last[q as usize], q)?;
    }
    Ok(c)
}

/// Build the QAOA Max-Cut ansatz for a graph.
///
/// `H` on every qubit, then `p = gammas.len()` layers, each a cost layer
/// (`RZZ(2γ_l)` per edge, decomposed `CNOT·Rz·CNOT`) followed by a mixer
/// (`Rx(2β_l)` on every qubit). `gammas.len()` must equal `betas.len()`.
/// `edges` endpoints must be `< n_qubits`. Farhi-Goldstone-Gutmann 2014.
pub fn build_qaoa(
    n_qubits: u32,
    edges: &[(u32, u32)],
    gammas: &[f64],
    betas: &[f64],
) -> Result<Circuit, AnsatzError> {
    if gammas.len() != betas.len() {
        return Err(AnsatzError::LayerMismatch {
            gammas: gammas.len(),
            betas: betas.len(),
        });
    }
    let mut c = Circuit::new(n_qubits, 0);
    for q in 0..n_qubits {
        c.h(q)?;
    }
    for (&gamma, &beta) in gammas.iter().zip(betas.iter()) {
        for &(i, j) in edges {
            c.cnot(i, j)?;
            c.rz(2.0 * gamma, j)?;
            c.cnot(i, j)?;
        }
        for q in 0..n_qubits {
            c.rx(2.0 * beta, q)?;
        }
    }
    Ok(c)
}

/// Max-Cut cost Hamiltonian `H_C = Σ_{(i,j)∈E} ½(I − Z_iZ_j)`.
///
/// `⟨ψ|H_C|ψ⟩` is the expected number of cut edges; its maximum over basis
/// states is the max-cut. Returned as a [`PauliSum`]: one identity term `½·|E|`
/// plus one `−½ Z_iZ_j` per edge.
pub fn maxcut_pauli_sum(n_qubits: u32, edges: &[(u32, u32)]) -> Result<PauliSum, AnsatzError> {
    let mut terms = Vec::with_capacity(edges.len() + 1);
    terms.push(PauliString::identity(0.5 * edges.len() as f64));
    for &(i, j) in edges {
        if i >= n_qubits || j >= n_qubits {
            return Err(AnsatzError::EdgeOutOfRange { i, j, n: n_qubits });
        }
        let ps = PauliString::new(-0.5, vec![(i, Pauli::Z), (j, Pauli::Z)])
            .map_err(AnsatzError::Pauli)?;
        terms.push(ps);
    }
    Ok(PauliSum { terms })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn param_count_validated() {
        assert!(matches!(
            build_hea(2, 1, &[0.0; 3]),
            Err(AnsatzError::ParamCount {
                expected: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn shape_is_ry_cnot_ry() {
        // n=2, depth=1 -> [Ry,Ry, CNOT, Ry,Ry] = 4 Ry gates + 1 CNOT, 5 instrs.
        let c = build_hea(2, 1, &[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(c.instructions().len(), 5);
        assert!(c
            .instructions()
            .iter()
            .all(|i| matches!(i, Instruction::Gate(_))));
    }

    #[test]
    fn qaoa_layer_mismatch() {
        assert!(matches!(
            build_qaoa(3, &[(0, 1)], &[0.1, 0.2], &[0.3]),
            Err(AnsatzError::LayerMismatch {
                gammas: 2,
                betas: 1
            })
        ));
    }

    #[test]
    fn qaoa_shape_p1_triangle() {
        // n=3, edges (0,1),(1,2),(0,2), p=1:
        // H*3 + per edge [CNOT, Rz, CNOT]=3*3=9 + Rx*3 = 3+9+3 = 15 instructions.
        let c = build_qaoa(3, &[(0, 1), (1, 2), (0, 2)], &[0.5], &[0.4]).unwrap();
        assert_eq!(c.instructions().len(), 15);
    }

    #[test]
    fn maxcut_hamiltonian_structure() {
        // 2 nodes, 1 edge: H_C = 0.5*I - 0.5*Z0Z1.
        let h = maxcut_pauli_sum(2, &[(0, 1)]).unwrap();
        assert_eq!(h.terms.len(), 2);
        // identity term coeff 0.5*|E| = 0.5
        let id = h.terms.iter().find(|t| t.terms.is_empty()).unwrap();
        assert_eq!(id.coefficient, 0.5);
        // ZZ term coeff -0.5
        let zz = h.terms.iter().find(|t| t.terms.len() == 2).unwrap();
        assert_eq!(zz.coefficient, -0.5);
        assert_eq!(zz.terms, vec![(0, Pauli::Z), (1, Pauli::Z)]);
    }
}
