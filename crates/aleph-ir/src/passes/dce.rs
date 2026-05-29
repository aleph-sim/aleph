//! `DeadCodeElim` — removes gate instructions whose effect cannot reach
//! any measured qubit. Reverse-walk data-flow liveness seeded by
//! measurements. See `docs/superpowers/specs/2026-05-29-p1-11-dce-design.md`.

use super::{Pass, PassError, PassStats};
use crate::Circuit;
use crate::Instruction;
use std::collections::HashSet;

pub struct DeadCodeElim;

impl Pass for DeadCodeElim {
    fn name(&self) -> &'static str {
        "DeadCodeElim"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let input: &[Instruction] = &circuit.instructions;
        let gates_before = input.len();

        // Observable guard: with no measurement the full state vector is the
        // observable, so removing any gate would change it. DCE is only valid
        // when the observable is the measurement distribution. No-op here.
        if !input
            .iter()
            .any(|i| matches!(i, Instruction::Measure { .. }))
        {
            return Ok(PassStats {
                gates_before,
                gates_after: gates_before,
                transformations: 0,
            });
        }

        let mut live: HashSet<u32> = HashSet::new();
        let mut kept_rev: Vec<Instruction> = Vec::with_capacity(input.len());
        let mut transformations: u64 = 0;

        // Reverse walk: a qubit is live if its pre-instruction state can still
        // affect a retained measurement. Measurements seed liveness; a gate
        // touching any live qubit is kept and entangles all its qubits into
        // the live set; Reset severs backward liveness; Barrier is
        // conservative (keeps its qubits live so fenced gates are never cut).
        for inst in input.iter().rev() {
            match inst {
                Instruction::Measure { qubit, .. } => {
                    live.insert(*qubit);
                    kept_rev.push(inst.clone());
                }
                Instruction::Reset(q) => {
                    kept_rev.push(inst.clone());
                    live.remove(q);
                }
                Instruction::Barrier(qs) => {
                    for q in qs {
                        live.insert(*q);
                    }
                    kept_rev.push(inst.clone());
                }
                Instruction::Gate(_) => {
                    let touched = inst.used_qubits();
                    if touched.iter().any(|q| live.contains(q)) {
                        live.extend(touched.iter().copied());
                        kept_rev.push(inst.clone());
                    } else {
                        // No data-flow path to any measurement → dead.
                        transformations += 1;
                    }
                }
            }
        }

        kept_rev.reverse();
        let gates_after = kept_rev.len();
        circuit.instructions = kept_rev;
        Ok(PassStats {
            gates_before,
            gates_after,
            transformations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    fn run_pass(c: &mut Circuit) -> PassStats {
        DeadCodeElim
            .run(c)
            .expect("DeadCodeElim is infallible in tests")
    }

    fn touches(c: &Circuit, q: u32) -> bool {
        c.instructions()
            .iter()
            .any(|i| i.used_qubits().contains(&q))
    }

    #[test]
    fn dead_branch_on_unmeasured_qubit_removed() {
        // H(0); CNOT(0,1); X(2); Measure(0); Measure(1)
        // q2 is unmeasured and unentangled with {0,1} → X(2) is dead.
        let mut c = Circuit::new(3, 2);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.x(2).unwrap();
        c.measure(0, 0).unwrap();
        c.measure(1, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1, "exactly the X(2) is dead");
        assert!(!touches(&c, 2), "q2 gate should be gone");
        // H, CNOT, Measure(0), Measure(1) remain = 4.
        assert_eq!(c.instructions().len(), 4);
    }

    #[test]
    fn reset_severs_backward_liveness() {
        // H(0); Reset(0); X(0); Measure(0) → H(0) is wiped by the reset.
        let mut c = Circuit::new(1, 1);
        c.h(0).unwrap();
        c.reset(0).unwrap();
        c.x(0).unwrap();
        c.measure(0, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1, "H(0) before the reset is dead");
        // Reset(0), X(0), Measure(0) remain = 3; no H.
        assert_eq!(c.instructions().len(), 3);
        assert!(matches!(c.instructions()[0], Instruction::Reset(0)));
    }

    #[test]
    fn barrier_keeps_its_qubits_live() {
        // X(2); Barrier([2]); Measure(0) → X(2) KEPT (barrier made q2 live).
        let mut c = Circuit::new(3, 1);
        c.x(2).unwrap();
        c.barrier([2u32]).unwrap();
        c.measure(0, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(
            s.transformations, 0,
            "barrier makes q2 live, nothing removed"
        );
        assert!(touches(&c, 2));
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn entanglement_keeps_unmeasured_gate_alive() {
        // X(2); CNOT(2,0); Measure(0) → CNOT touches live q0 → both kept;
        // X(2) then touches now-live q2 → kept.
        let mut c = Circuit::new(3, 1);
        c.x(2).unwrap();
        c.cnot(2, 0).unwrap();
        c.measure(0, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn no_measurement_circuit_is_unchanged() {
        // No Measure → full state is the observable → DCE is a no-op.
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.x(1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn fully_measured_circuit_loses_nothing() {
        // Every qubit measured → no false positives.
        let mut c = Circuit::new(2, 2);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.measure(0, 0).unwrap();
        c.measure(1, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 4);
    }

    #[test]
    fn mid_circuit_measure_then_dead_gate() {
        // H(0); Measure(0); X(1) — only q0 measured; X(1) reaches no measure.
        let mut c = Circuit::new(2, 1);
        c.h(0).unwrap();
        c.measure(0, 0).unwrap();
        c.x(1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1, "X(1) is dead");
        assert!(!touches(&c, 1));
        assert_eq!(c.instructions().len(), 2);
    }
}
