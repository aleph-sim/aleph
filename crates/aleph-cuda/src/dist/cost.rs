//! P6-05 GPU cost model (spec §6.2). `classify` mirrors the decision order of
//! `CudaSvBackend::apply_gate` (sv/backend.rs) and `apply_diagonal_phase`, so
//! each instruction is priced as the kernel it actually launches.

use aleph_core::Gate;
use aleph_ir::dist::DistError;
use aleph_ir::Instruction;

use crate::common::diagonal_of;

/// The CUDA kernel an instruction of a fused rank program launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// `apply_1q` (dense, non-diagonal 1-qubit).
    Dense1,
    /// `apply_kq_tiled`, k = 2 (dense 2q incl. `Swap`, controlled `Cnot`).
    Dense2,
    /// `apply_kq_tiled`, k = 3 (`UnitaryKq` k=3, `Toffoli`, dense 3q).
    Dense3,
    /// `apply_diag_1q` (numerically diagonal 1-qubit).
    Diag1,
    /// `apply_diag`, k = 2 or 3 (numerically diagonal 2q/3q).
    DiagK,
    /// `apply_cnot` (bare `Cnot`, no external controls).
    Cnot,
    /// `apply_phase_poly` with `terms` phase terms.
    PhasePoly { terms: usize },
    /// No kernel (barrier, empty `DiagonalPhase`).
    Free,
}

/// Kernel kind `instr` launches on `CudaSvBackend` / `CudaSvBackendF32`.
///
/// The classification assumes the backend's defaults (`custom_2q` and
/// `custom_diag` on), as `DistSvBackend` runs them.
pub fn classify(instr: &Instruction) -> Result<KernelKind, DistError> {
    let g = match instr {
        Instruction::Gate(g) => g,
        Instruction::DiagonalPhase(dp) => {
            return Ok(if dp.terms.is_empty() {
                KernelKind::Free
            } else {
                KernelKind::PhasePoly {
                    terms: dp.terms.len(),
                }
            })
        }
        Instruction::Barrier(_) => return Ok(KernelKind::Free),
        _ => {
            return Err(DistError::Unsupported {
                kind: "cost: non-unitary instruction",
            })
        }
    };
    // 1. bare Cnot → apply_cnot (backend.rs:665-687)
    if matches!(g.gate, Gate::Cnot) && g.controls.is_empty() {
        return Ok(KernelKind::Cnot);
    }
    // 2. UnitaryKq → apply_kq_tiled before any diagonal test (backend.rs:693-705)
    if let Gate::UnitaryKq { k, .. } = &g.gate {
        return match k {
            1 => Ok(KernelKind::Dense1),
            2 => Ok(KernelKind::Dense2),
            3 => Ok(KernelKind::Dense3),
            _ => Err(DistError::Unsupported {
                kind: "cost: UnitaryKq k > 3 (unreachable from fuse_for_gpu)",
            }),
        };
    }
    // 3–5. matrix: numerically diagonal → apply_diag(_1q), else dense by arity
    let m = g.gate.matrix()?;
    let diag = diagonal_of(&m).is_some();
    Ok(match (g.qubits.len(), diag) {
        (1, true) => KernelKind::Diag1,
        (_, true) => KernelKind::DiagK,
        (1, false) => KernelKind::Dense1,
        (2, false) => KernelKind::Dense2,
        _ => KernelKind::Dense3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Complex, GateInstance, Param};
    use aleph_ir::{DiagonalPhase, PhaseTerm};
    use smallvec::smallvec;

    fn g(gate: Gate, q: &[u32]) -> Instruction {
        Instruction::Gate(GateInstance::new(gate, q.to_vec()))
    }

    #[test]
    fn classify_mirrors_apply_gate() {
        use KernelKind::*;
        let p = Param::Concrete(0.3);
        assert_eq!(classify(&g(Gate::H, &[0])).unwrap(), Dense1);
        assert_eq!(classify(&g(Gate::Rx(p), &[0])).unwrap(), Dense1);
        assert_eq!(classify(&g(Gate::Rz(p), &[0])).unwrap(), Diag1);
        assert_eq!(classify(&g(Gate::T, &[0])).unwrap(), Diag1);
        assert_eq!(classify(&g(Gate::Cnot, &[0, 1])).unwrap(), Cnot);
        let ccx = Instruction::Gate(GateInstance::controlled(
            Gate::Cnot,
            vec![0u32, 1],
            vec![2u32],
        ));
        assert_eq!(classify(&ccx).unwrap(), Dense2);
        assert_eq!(classify(&g(Gate::Swap, &[0, 1])).unwrap(), Dense2);
        assert_eq!(classify(&g(Gate::Iswap, &[0, 1])).unwrap(), Dense2);
        assert_eq!(classify(&g(Gate::Cz, &[0, 1])).unwrap(), DiagK);
        assert_eq!(classify(&g(Gate::CRz(p), &[0, 1])).unwrap(), DiagK);
        let d2 = Gate::Unitary2qDiag(Box::new([Complex::new(1.0, 0.0); 4]));
        assert_eq!(classify(&g(d2, &[0, 1])).unwrap(), DiagK);
        // Numerically diagonal dense Unitary2q → apply_diag (diagonal_of is numeric).
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = Complex::new(1.0, 0.0);
        }
        assert_eq!(
            classify(&g(Gate::Unitary2q(Box::new(m)), &[0, 1])).unwrap(),
            DiagK
        );
        assert_eq!(classify(&g(Gate::Toffoli, &[0, 1, 2])).unwrap(), Dense3);
        assert_eq!(classify(&g(Gate::Ccz, &[0, 1, 2])).unwrap(), DiagK);
        // UnitaryKq is dispatched before diagonal_of: dense even if diagonal.
        let mut id8 = vec![Complex::new(0.0, 0.0); 64];
        for i in 0..8 {
            id8[i * 8 + i] = Complex::new(1.0, 0.0);
        }
        let kq = Gate::UnitaryKq {
            k: 3,
            data: id8.into_boxed_slice(),
        };
        assert_eq!(classify(&g(kq, &[0, 1, 2])).unwrap(), Dense3);
    }

    #[test]
    fn phase_poly_and_free() {
        let dp = |terms: Vec<PhaseTerm>| {
            Instruction::DiagonalPhase(Box::new(DiagonalPhase { n_qubits: 3, terms }))
        };
        assert_eq!(classify(&dp(vec![])).unwrap(), KernelKind::Free);
        let t = PhaseTerm {
            conds: smallvec![0b011],
            angle: 0.2,
        };
        assert_eq!(
            classify(&dp(vec![t.clone(), t])).unwrap(),
            KernelKind::PhasePoly { terms: 2 }
        );
        assert_eq!(
            classify(&Instruction::Barrier(smallvec![0, 1])).unwrap(),
            KernelKind::Free
        );
    }

    #[test]
    fn measure_is_rejected() {
        assert!(classify(&Instruction::Measure { qubit: 0, clbit: 0 }).is_err());
    }
}
