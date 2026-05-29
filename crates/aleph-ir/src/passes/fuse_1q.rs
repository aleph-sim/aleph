//! `Fuse1qRuns` — collapses runs of adjacent uncontrolled 1q gates
//! on the same qubit into a single fused gate. See
//! `docs/superpowers/specs/2026-05-29-p1-09-fuse-1q-design.md`.

use super::{Pass, PassError, PassStats};
use crate::Circuit;

pub struct Fuse1qRuns;

impl Pass for Fuse1qRuns {
    fn name(&self) -> &'static str {
        "Fuse1qRuns"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.len();
        // Stub — real implementation lands in Task 5.
        Ok(PassStats {
            gates_before: n,
            gates_after: n,
            transformations: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;
    use aleph_core::Complex;
    use aleph_core::gate::Gate;

    fn run_pass(c: &mut Circuit) -> PassStats {
        Fuse1qRuns.run(c).expect("Fuse1qRuns is infallible in tests")
    }

    #[test]
    fn empty_circuit_no_op() {
        let mut c = Circuit::new(1, 0);
        let stats = run_pass(&mut c);
        assert_eq!(stats.gates_before, 0);
        assert_eq!(stats.gates_after, 0);
        assert_eq!(stats.transformations, 0);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn single_h_short_circuits() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 0);
        assert_eq!(c.instructions().len(), 1);
        match &c.instructions()[0] {
            Instruction::Gate(g) => assert!(matches!(g.gate, Gate::H)),
            other => panic!("expected H, got {other:?}"),
        }
    }

    #[test]
    fn diag_only_run_emits_unitary_1q_diag() {
        // S · T · Z on q0 — all diagonal. Fused diag = Z * T * S =
        // diag(1, -1) * diag(1, e^{iπ/4}) * diag(1, i)  (matrix product
        // applied right-to-left as Z · T · S where S is applied first).
        // d[0] = 1
        // d[1] = (-1) * e^{iπ/4} * i = -i * e^{iπ/4}
        let mut c = Circuit::new(1, 0);
        c.s(0).unwrap();
        c.t(0).unwrap();
        c.z(0).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        match &c.instructions()[0] {
            Instruction::Gate(g) => match &g.gate {
                Gate::Unitary1qDiag(d) => {
                    assert!((d[0] - Complex::new(1.0, 0.0)).norm() < 1e-14);
                    let expected_d1 = Complex::new(0.0, -1.0)
                        * Complex::from_polar(1.0, std::f64::consts::FRAC_PI_4);
                    assert!((d[1] - expected_d1).norm() < 1e-12);
                }
                other => panic!("expected Unitary1qDiag, got {other:?}"),
            },
            other => panic!("expected Gate instruction, got {other:?}"),
        }
        assert_eq!(g_qubit(&c.instructions()[0]), 0);
    }

    #[test]
    fn mixed_run_emits_unitary_1q() {
        // H · S on q0 — mixed (H is non-diagonal).
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.s(0).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        match &c.instructions()[0] {
            Instruction::Gate(g) => assert!(matches!(g.gate, Gate::Unitary1q(_))),
            other => panic!("expected Unitary1q, got {other:?}"),
        }
    }

    fn g_qubit(inst: &Instruction) -> u32 {
        match inst {
            Instruction::Gate(g) => g.qubits[0],
            _ => panic!("not a gate"),
        }
    }
}
