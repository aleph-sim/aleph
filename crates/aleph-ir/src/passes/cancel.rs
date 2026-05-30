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
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

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

    #[test]
    fn s_sdg_cancels() {
        // S(0); Sdg(0) → ∅  (adjoint pair, not self-inverse).
        let mut c = Circuit::new(1, 0);
        c.s(0).unwrap();
        c.sdg(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn t_tdg_cancels_both_orders() {
        for build in [
            |c: &mut Circuit| {
                c.t(0).unwrap();
                c.tdg(0).unwrap();
            },
            |c: &mut Circuit| {
                c.tdg(0).unwrap();
                c.t(0).unwrap();
            },
        ] {
            let mut c = Circuit::new(1, 0);
            build(&mut c);
            let s = run_pass(&mut c);
            assert_eq!(s.transformations, 1);
            assert!(c.instructions().is_empty());
        }
    }

    #[test]
    fn rz_theta_rz_neg_theta_cancels() {
        // Rz(0.3); Rz(-0.3) → ∅  (exact f64 negation).
        let mut c = Circuit::new(1, 0);
        c.rz(0.3, 0).unwrap();
        c.rz(-0.3, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn rz_same_sign_does_not_cancel() {
        // Rz(0.3); Rz(0.3) is Rz(0.6), NOT identity → kept.
        let mut c = Circuit::new(1, 0);
        c.rz(0.3, 0).unwrap();
        c.rz(0.3, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 2);
    }

    #[test]
    fn rz_near_but_unequal_angle_does_not_cancel() {
        // Cancellation requires exact -θ; a 1e-12 mismatch must NOT cancel
        // (that is fusion/tolerance territory, not exact-inverse deletion).
        let mut c = Circuit::new(1, 0);
        c.rz(0.3, 0).unwrap();
        c.rz(-0.3 + 1e-12, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 2);
    }

    #[test]
    fn cnot_cnot_same_qubits_cancels() {
        // CNOT(0,1); CNOT(0,1) → ∅
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn cnot_reversed_roles_does_not_cancel() {
        // CNOT(0,1); CNOT(1,0) are different operations → kept.
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.cnot(1, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 2);
    }

    #[test]
    fn cz_swap_toffoli_cancel() {
        // CZ(0,1)·CZ(0,1), SWAP(0,1)·SWAP(0,1), Toffoli·Toffoli all self-inverse.
        let mut c = Circuit::new(3, 0);
        c.cz(0, 1).unwrap();
        c.cz(0, 1).unwrap();
        c.swap(0, 1).unwrap();
        c.swap(0, 1).unwrap();
        c.ccx(0, 1, 2).unwrap();
        c.ccx(0, 1, 2).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 3);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn iswap_iswapdg_cancels() {
        // Iswap(0,1); IswapDg(0,1) → ∅  (adjoint pair via add_gate).
        let mut c = Circuit::new(2, 0);
        c.add_gate(GateInstance::new(Gate::Iswap, smallvec![0u32, 1u32]))
            .unwrap();
        c.add_gate(GateInstance::new(Gate::IswapDg, smallvec![0u32, 1u32]))
            .unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn controlled_x_same_controls_cancels_control_order_independent() {
        // ctrl-X target=3 controls={0,1}; then controls={1,0} → cancels
        // (controls compared as a set).
        let mut c = Circuit::new(4, 0);
        c.add_gate(GateInstance::controlled(
            Gate::X,
            smallvec![3u32],
            smallvec![0u32, 1u32],
        ))
        .unwrap();
        c.add_gate(GateInstance::controlled(
            Gate::X,
            smallvec![3u32],
            smallvec![1u32, 0u32],
        ))
        .unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn controlled_x_different_controls_does_not_cancel() {
        // Same target, different control set → not the same operation → kept.
        let mut c = Circuit::new(4, 0);
        c.add_gate(GateInstance::controlled(
            Gate::X,
            smallvec![3u32],
            smallvec![0u32],
        ))
        .unwrap();
        c.add_gate(GateInstance::controlled(
            Gate::X,
            smallvec![3u32],
            smallvec![1u32],
        ))
        .unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 2);
    }

    #[test]
    fn symmetric_gate_reordered_qubits_not_cancelled_v1() {
        // Cz(0,1) and Cz(1,0) ARE the same operation, but v1 compares targets
        // positionally and conservatively does NOT cancel them. Documents the
        // deferral (symmetric-gate normalisation is future work).
        let mut c = Circuit::new(2, 0);
        c.cz(0, 1).unwrap();
        c.cz(1, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 2);
    }

    #[test]
    fn barrier_blocks_cancellation() {
        // H(0); Barrier([0]); H(0) → kept (barrier severs adjacency).
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.barrier([0u32]).unwrap();
        c.h(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn measure_blocks_cancellation() {
        // X(0); Measure(0,0); X(0) → kept.
        let mut c = Circuit::new(1, 1);
        c.x(0).unwrap();
        c.measure(0, 0).unwrap();
        c.x(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn reset_blocks_cancellation() {
        // X(0); Reset(0); X(0) → kept.
        let mut c = Circuit::new(1, 0);
        c.x(0).unwrap();
        c.reset(0).unwrap();
        c.x(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn intervening_gate_on_partial_support_blocks_cancellation() {
        // CNOT(0,1); X(1); CNOT(0,1) → NOT cancelled: X(1) intervenes on
        // qubit 1, so the two CNOTs are not adjacent on their full support.
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.x(1).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn nested_single_qubit_cancellation() {
        // X H H X (all on qubit 0): inner H·H cancels, then the X·X become
        // adjacent and cancel → ∅.
        let mut c = Circuit::new(1, 0);
        c.x(0).unwrap();
        c.h(0).unwrap();
        c.h(0).unwrap();
        c.x(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 2);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn nested_through_two_qubit_gate() {
        // CNOT(0,1); X(0); X(0); CNOT(0,1) → inner X·X cancels, then the two
        // CNOTs share the top on both qubits and cancel → ∅.
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.x(0).unwrap();
        c.x(0).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 2);
        assert!(c.instructions().is_empty());
    }

    #[test]
    fn partial_cancellation_keeps_survivors_in_order() {
        // H(0); X(0); X(0); Z(1) → X·X cancels; H(0) and Z(1) survive in order.
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.x(0).unwrap();
        c.x(0).unwrap();
        c.z(1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert_eq!(c.instructions().len(), 2);
        assert!(matches!(
            &c.instructions()[0],
            Instruction::Gate(g) if g.gate == aleph_core::Gate::H
        ));
        assert!(matches!(
            &c.instructions()[1],
            Instruction::Gate(g) if g.gate == aleph_core::Gate::Z
        ));
    }
}
