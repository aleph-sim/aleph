//! `CancelInversePairs` — removes adjacent inverse-pair gates
//! (H·H, X·X, CNOT·CNOT, S·Sdg, Rz(θ)·Rz(−θ), …). A single forward
//! pass with per-qubit stacks of live instruction indices; two gates
//! cancel iff they are adjacent on their entire shared support, act on
//! the same targets (positionally) and controls (as a set), and one is
//! the other's `Gate::inverse()`. Adjacent-only; commutation-aware
//! cancellation is P1-13. See
//! `docs/superpowers/specs/2026-05-30-p1-12-gate-cancellation-design.md`.

use super::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction};
use std::collections::HashMap;

pub struct CancelInversePairs;

/// Order-independent equality of two external-control lists. Controls
/// are semantically a set: equal length plus equal sorted contents.
fn controls_match(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a: smallvec::SmallVec<[u32; 2]> = a.iter().copied().collect();
    let mut b: smallvec::SmallVec<[u32; 2]> = b.iter().copied().collect();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

impl Pass for CancelInversePairs {
    fn name(&self) -> &'static str {
        "CancelInversePairs"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let input: &[Instruction] = &circuit.instructions;
        let gates_before = input.len();

        let mut result: Vec<Instruction> = Vec::with_capacity(input.len());
        let mut removed: Vec<bool> = Vec::with_capacity(input.len());
        // Per qubit: stack of indices into `result` of still-live
        // instructions touching it. Top = most recent.
        let mut live: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut transformations: u64 = 0;

        for inst in input.iter() {
            // Non-gate instructions are hard barriers: push and never pop,
            // so no gate cancels across a Measure/Reset/Barrier.
            let gate = match inst {
                Instruction::Gate(g) => g,
                _ => {
                    let k = result.len();
                    result.push(inst.clone());
                    removed.push(false);
                    for q in inst.used_qubits() {
                        live.entry(q).or_default().push(k);
                    }
                    continue;
                }
            };

            let support = inst.used_qubits();

            // Candidate predecessor: the single live instruction that is the
            // current top on EVERY qubit of `support`. If the tops disagree
            // (something intervened on part of the support) or any support
            // qubit has no live predecessor, there is no candidate.
            let mut cand: Option<usize> = None;
            let mut shared = true;
            for &q in &support {
                match live.get(&q).and_then(|s| s.last().copied()) {
                    Some(top) => match cand {
                        None => cand = Some(top),
                        Some(c) if c == top => {}
                        Some(_) => {
                            shared = false;
                            break;
                        }
                    },
                    None => {
                        shared = false;
                        break;
                    }
                }
            }

            let cancels = shared
                && match cand {
                    Some(i) => match &result[i] {
                        // Same targets positionally, same controls as a set,
                        // and inverse gate. Target/control equality also
                        // guarantees `prev` touches exactly `support`, so
                        // popping it from every support stack is complete.
                        Instruction::Gate(prev) => {
                            prev.qubits == gate.qubits
                                && controls_match(&prev.controls, &gate.controls)
                                && prev.gate == gate.gate.inverse()
                        }
                        _ => false,
                    },
                    None => false,
                };

            if cancels {
                let i = cand.expect("cancels implies a candidate");
                removed[i] = true;
                for &q in &support {
                    // `i` is the top of every support stack (candidate
                    // condition), so this pops exactly `result[i]`.
                    live.get_mut(&q).expect("support qubit has a stack").pop();
                }
                transformations += 1;
                // `gate` itself is dropped (not pushed).
            } else {
                let k = result.len();
                result.push(inst.clone());
                removed.push(false);
                for &q in &support {
                    live.entry(q).or_default().push(k);
                }
            }
        }

        let kept: Vec<Instruction> = result
            .into_iter()
            .zip(removed)
            .filter_map(|(inst, dead)| if dead { None } else { Some(inst) })
            .collect();
        let gates_after = kept.len();
        circuit.instructions = kept;

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
    use crate::Circuit;

    fn run_pass(c: &mut Circuit) -> PassStats {
        CancelInversePairs
            .run(c)
            .expect("CancelInversePairs is infallible")
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(CancelInversePairs.name(), "CancelInversePairs");
    }

    #[test]
    fn empty_circuit_is_a_no_op() {
        let mut c = Circuit::new(2, 0);
        let s = run_pass(&mut c);
        assert_eq!(s.gates_before, 0);
        assert_eq!(s.gates_after, 0);
        assert_eq!(s.transformations, 0);
    }

    #[test]
    fn h_h_cancels_to_empty() {
        // H(0); H(0) → ∅
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.h(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.gates_before, 2);
        assert_eq!(s.gates_after, 0);
        assert_eq!(s.transformations, 1);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn h_on_different_qubits_does_not_cancel() {
        // H(0); H(1) → unchanged (different support).
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.h(1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 2);
    }
}
