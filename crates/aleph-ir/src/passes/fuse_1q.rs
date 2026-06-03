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
        use crate::Instruction;
        use aleph_core::gate::{Gate, GateInstance, GateMatrix};
        use aleph_core::Complex;
        use smallvec::smallvec;
        use std::collections::HashMap;

        // Build output to a separate Vec without mutating circuit.instructions
        // until success — leaves the circuit intact on any future Err path.
        // (Currently Fuse1qRuns is infallible, but later passes may not be.)
        let input: &[Instruction] = &circuit.instructions;
        let gates_before = input.len();

        struct Pending {
            start_index: usize,
            matrix: [[Complex; 2]; 2],
            diag_only: bool,
            run_len: usize,
        }

        let mut pending: HashMap<u32, Pending> = HashMap::new();
        let mut output: Vec<Instruction> = Vec::with_capacity(input.len());
        let mut transformations: u64 = 0;

        // Flush pending run on `q` into `output`. Returns whether
        // anything was flushed.
        fn flush(
            q: u32,
            pending: &mut HashMap<u32, Pending>,
            input: &[Instruction],
            output: &mut Vec<Instruction>,
            transformations: &mut u64,
        ) {
            let Some(p) = pending.remove(&q) else { return };
            if p.run_len == 1 {
                // Re-emit the original instruction verbatim.
                output.push(input[p.start_index].clone());
                return;
            }
            *transformations += 1;
            let gate = if p.diag_only {
                Gate::Unitary1qDiag(Box::new([p.matrix[0][0], p.matrix[1][1]]))
            } else {
                Gate::Unitary1q(Box::new(p.matrix))
            };
            output.push(Instruction::Gate(GateInstance::new(gate, smallvec![q])));
        }

        // Left-multiply `lhs` onto `acc`: acc := lhs · acc.
        // Quantum convention applies G₁ first, so for a run [A, B, C]
        // the fused matrix is M = C · B · A: each newly-seen gate
        // left-multiplies the accumulated product.
        fn left_mul(acc: &mut [[Complex; 2]; 2], lhs: &[[Complex; 2]; 2]) {
            let a = lhs;
            let b = *acc;
            acc[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
            acc[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
            acc[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
            acc[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
        }

        for (idx, inst) in input.iter().enumerate() {
            match inst {
                Instruction::Gate(g) => {
                    let fusible =
                        g.gate.arity() == 1 && g.controls.is_empty() && g.qubits.len() == 1;
                    if fusible {
                        let q = g.qubits[0];
                        let m2 = match g.gate.matrix() {
                            Ok(GateMatrix::M2x2(m)) => m,
                            // arity==1 must yield M2x2; symbolic params can't
                            // appear here (only Phase 4 surfaces those).
                            // Treat any other result as fence-this-qubit
                            // and re-emit verbatim — conservative, never
                            // mis-fuses.
                            _ => {
                                flush(q, &mut pending, input, &mut output, &mut transformations);
                                output.push(inst.clone());
                                continue;
                            }
                        };
                        let g_is_diag = g.gate.is_diagonal();
                        if let Some(p) = pending.get_mut(&q) {
                            left_mul(&mut p.matrix, &m2);
                            p.diag_only &= g_is_diag;
                            p.run_len += 1;
                        } else {
                            pending.insert(
                                q,
                                Pending {
                                    start_index: idx,
                                    matrix: m2,
                                    diag_only: g_is_diag,
                                    run_len: 1,
                                },
                            );
                        }
                    } else {
                        // Multi-qubit or controlled — fence every qubit
                        // it touches.
                        for q in inst.used_qubits() {
                            flush(q, &mut pending, input, &mut output, &mut transformations);
                        }
                        output.push(inst.clone());
                    }
                }
                Instruction::Barrier(qs) => {
                    for q in qs {
                        flush(*q, &mut pending, input, &mut output, &mut transformations);
                    }
                    output.push(inst.clone());
                }
                Instruction::Measure { qubit, .. } => {
                    flush(
                        *qubit,
                        &mut pending,
                        input,
                        &mut output,
                        &mut transformations,
                    );
                    output.push(inst.clone());
                }
                Instruction::Reset(qubit) => {
                    flush(
                        *qubit,
                        &mut pending,
                        input,
                        &mut output,
                        &mut transformations,
                    );
                    output.push(inst.clone());
                }
                // DiagonalPhase is an opaque fused diagonal operator. Treat it
                // as a fence: flush any pending 1q runs on all its qubits and
                // re-emit verbatim without attempting to fuse through it.
                Instruction::DiagonalPhase(_) => {
                    for q in inst.used_qubits() {
                        flush(q, &mut pending, input, &mut output, &mut transformations);
                    }
                    output.push(inst.clone());
                }
            }
        }

        // Final flush — stable order so the pass is deterministic.
        let mut leftover: Vec<u32> = pending.keys().copied().collect();
        leftover.sort_unstable();
        for q in leftover {
            flush(q, &mut pending, input, &mut output, &mut transformations);
        }

        let gates_after = output.len();
        circuit.instructions = output;
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
    use aleph_core::gate::Gate;
    use aleph_core::Complex;

    fn run_pass(c: &mut Circuit) -> PassStats {
        Fuse1qRuns
            .run(c)
            .expect("Fuse1qRuns is infallible in tests")
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

    #[test]
    fn cnot_fences_both_qubits() {
        // H[0]; CNOT(0,1); H[0]  → H[0], CNOT, H[0]  (two length-1 runs).
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.h(0).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
        match &c.instructions()[0] {
            Instruction::Gate(g) => assert!(matches!(g.gate, Gate::H)),
            _ => panic!(),
        }
        match &c.instructions()[1] {
            Instruction::Gate(g) => assert!(matches!(g.gate, Gate::Cnot)),
            _ => panic!(),
        }
        match &c.instructions()[2] {
            Instruction::Gate(g) => assert!(matches!(g.gate, Gate::H)),
            _ => panic!(),
        }
    }

    #[test]
    fn barrier_fences_listed_qubits_only() {
        // H[0]; H[1]; Barrier([0]); H[0]; H[1]
        // Barrier listing q0 only fences q0; q1's run-of-2 across the
        // barrier fuses normally.
        //
        // Walk: H(0) → pending[0]; H(1) → pending[1]; Barrier([0]) →
        // flush(0) re-emits H(0), push Barrier; H(0) → pending[0];
        // H(1) → pending[1].matrix = H·H, len=2. End: final flush
        // sorted: flush(0) re-emits the second H(0); flush(1) emits
        // one Unitary1q (H·H). Output length = 4 (H, Barrier, H,
        // Unitary1q). transformations = 1.
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.h(1).unwrap();
        c.add_instruction(Instruction::Barrier(smallvec::smallvec![0u32]))
            .unwrap();
        c.h(0).unwrap();
        c.h(1).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 1);
        assert_eq!(c.instructions().len(), 4);
    }

    #[test]
    fn measure_fences_target_qubit() {
        // H[0]; Measure(0,0); H[0]  → H[0]; Measure(0,0); H[0]
        let mut c = Circuit::new(1, 1);
        c.h(0).unwrap();
        c.add_instruction(Instruction::Measure { qubit: 0, clbit: 0 })
            .unwrap();
        c.h(0).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn reset_fences_target_qubit() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.add_instruction(Instruction::Reset(0)).unwrap();
        c.h(0).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn controlled_1q_does_not_fuse() {
        // H[0]; controlled-H(target=0, control=1); H[0]
        // Middle is a controlled-1q (arity==1 but controls non-empty),
        // so it fences q0 (and q1 — though q1 is otherwise idle).
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.add_instruction(Instruction::Gate(
            aleph_core::gate::GateInstance::controlled(
                Gate::H,
                smallvec::smallvec![0u32],
                smallvec::smallvec![1u32],
            ),
        ))
        .unwrap();
        c.h(0).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
    }

    #[test]
    fn per_qubit_runs_are_independent() {
        // q0: H, S         (mixed run, fuses to Unitary1q)
        // q1: T, S, Z      (diag run, fuses to Unitary1qDiag)
        // Interleaved IR order: H(0), T(1), S(0), S(1), Z(1).
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.t(1).unwrap();
        c.s(0).unwrap();
        c.s(1).unwrap();
        c.z(1).unwrap();
        let stats = run_pass(&mut c);
        assert_eq!(stats.transformations, 2);
        assert_eq!(c.instructions().len(), 2);
        // Final flush is in qubit-id order: q0 first, then q1.
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert_eq!(g.qubits[0], 0);
                assert!(matches!(g.gate, Gate::Unitary1q(_)));
            }
            _ => panic!(),
        }
        match &c.instructions()[1] {
            Instruction::Gate(g) => {
                assert_eq!(g.qubits[0], 1);
                assert!(matches!(g.gate, Gate::Unitary1qDiag(_)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rz_h_rz_fused_matrix_matches_hand_computation() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let a = FRAC_PI_4;
        let b = FRAC_PI_2;
        let mut c = Circuit::new(1, 0);
        c.rz(a, 0).unwrap();
        c.h(0).unwrap();
        c.rz(b, 0).unwrap();
        run_pass(&mut c);
        assert_eq!(c.instructions().len(), 1);
        let m = match &c.instructions()[0] {
            Instruction::Gate(g) => match &g.gate {
                Gate::Unitary1q(m) => **m,
                other => panic!("expected Unitary1q, got {other:?}"),
            },
            _ => panic!(),
        };
        // Hand-computed reference: Rz(b) · H · Rz(a).
        let rz = |t: f64| -> [[Complex; 2]; 2] {
            let h = t / 2.0;
            [
                [Complex::from_polar(1.0, -h), Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::from_polar(1.0, h)],
            ]
        };
        let h_mat: [[Complex; 2]; 2] = {
            let r = 1.0 / 2.0_f64.sqrt();
            [
                [Complex::new(r, 0.0), Complex::new(r, 0.0)],
                [Complex::new(r, 0.0), Complex::new(-r, 0.0)],
            ]
        };
        let mul = |x: &[[Complex; 2]; 2], y: &[[Complex; 2]; 2]| -> [[Complex; 2]; 2] {
            let mut out = [[Complex::new(0.0, 0.0); 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        out[i][j] += x[i][k] * y[k][j];
                    }
                }
            }
            out
        };
        let inner = mul(&h_mat, &rz(a));
        let expected = mul(&rz(b), &inner);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (m[i][j] - expected[i][j]).norm() < 1e-14,
                    "m[{i}][{j}]={:?} expected={:?}",
                    m[i][j],
                    expected[i][j]
                );
            }
        }
    }
}
