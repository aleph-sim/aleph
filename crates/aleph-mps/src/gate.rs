//! Extract dense unitary matrices from `GateInstance` for the MPS backend.

use crate::MpsError;
use aleph_core::{Complex, GateInstance, GateMatrix};

/// Extract a 1q gate's 2×2 matrix. Rejects external controls and non-1q gates.
pub(crate) fn matrix_2x2(g: &GateInstance) -> Result<[[Complex; 2]; 2], MpsError> {
    if !g.controls.is_empty() {
        return Err(MpsError::ExternalControls {
            kind: g.gate.name(),
        });
    }
    match g.gate.matrix() {
        Ok(GateMatrix::M2x2(m)) => {
            if m.iter()
                .flatten()
                .any(|c| !c.re.is_finite() || !c.im.is_finite())
            {
                return Err(MpsError::NonFiniteParam {
                    kind: g.gate.name(),
                });
            }
            Ok(m)
        }
        _ => Err(MpsError::UnsupportedGate {
            kind: g.gate.name(),
        }),
    }
}

/// Extract a 2q gate's 4×4 matrix. Rejects external controls and non-2q gates.
pub(crate) fn matrix_4x4(g: &GateInstance) -> Result<[[Complex; 4]; 4], MpsError> {
    if !g.controls.is_empty() {
        return Err(MpsError::ExternalControls {
            kind: g.gate.name(),
        });
    }
    match g.gate.matrix() {
        Ok(GateMatrix::M4x4(m)) => {
            if m.iter()
                .flatten()
                .any(|c| !c.re.is_finite() || !c.im.is_finite())
            {
                return Err(MpsError::NonFiniteParam {
                    kind: g.gate.name(),
                });
            }
            Ok(m)
        }
        _ => Err(MpsError::UnsupportedGate {
            kind: g.gate.name(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn x_matrix() {
        let g = GateInstance::new(Gate::X, smallvec![0u32]);
        let m = matrix_2x2(&g).unwrap();
        assert!((m[0][1].re - 1.0).abs() < 1e-12);
        assert!((m[1][0].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_controls() {
        let g = GateInstance::controlled(Gate::X, smallvec![1u32], smallvec![0u32]);
        assert!(matches!(
            matrix_2x2(&g),
            Err(MpsError::ExternalControls { .. })
        ));
    }

    #[test]
    fn cnot_matrix_shape() {
        // CNOT is a permutation matrix: each row has exactly one entry ≈ 1.
        let g = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);
        let m = matrix_4x4(&g).unwrap();
        for row in &m {
            let ones: usize = row
                .iter()
                .filter(|c| (c.norm() - 1.0).abs() < 1e-10)
                .count();
            let zeros: usize = row.iter().filter(|c| c.norm() < 1e-10).count();
            assert_eq!(ones, 1, "each row must have exactly one entry ≈ 1");
            assert_eq!(zeros, 3, "each row must have exactly three entries ≈ 0");
        }
    }
}
