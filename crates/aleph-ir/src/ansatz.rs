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

/// Error building a HEA circuit.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AnsatzError {
    #[error("expected {expected} params (n_qubits*(depth+1)), got {got}")]
    ParamCount { expected: usize, got: usize },
    #[error(transparent)]
    Circuit(#[from] CircuitError),
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
}
