//! `Fuse2q` — fuses 2-qubit gates with adjacent 1-qubit neighbours and
//! adjacent same-pair 2-qubit gates into a single 4×4 `Gate::Unitary2q`.
//! See `docs/superpowers/specs/2026-05-29-p1-10-fuse-2q-design.md`.
//!
//! 4×4 operand convention (P0-06 / verified vs `apply_2q_dense_scalar`,
//! kernels/aos.rs:1208): for operands `[qubits[0], qubits[1]]`, matrix
//! index `k` has bit 1 (MSB) = qubits[0], bit 0 (LSB) = qubits[1].

use super::{Pass, PassError, PassStats};
use crate::Circuit;
use aleph_core::Complex;

type M2 = [[Complex; 2]; 2];
type M4 = [[Complex; 4]; 4];

/// 4×4 product `a · b`.
fn mul4(a: &M4, b: &M4) -> M4 {
    let zero = Complex::new(0.0, 0.0);
    let mut out = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut acc = zero;
            for k in 0..4 {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// 2×2 left-multiply: `acc := lhs · acc` (quantum order — later gate
/// left-multiplies the accumulated product).
fn left_mul2(acc: &mut M2, lhs: &M2) {
    let a = lhs;
    let b = *acc;
    acc[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    acc[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    acc[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    acc[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
}

fn ident4() -> M4 {
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let mut m = [[zero; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = one;
    }
    m
}

/// Lift a 1q matrix `m2` acting on qubit `q` into the block's 4×4, where
/// `block_qubits = [hi, lo]` (`block_qubits[0]` is the MSB matrix bit).
fn lift_1q(m2: &M2, q: u32, block_qubits: &[u32; 2]) -> M4 {
    let zero = Complex::new(0.0, 0.0);
    let on_high = q == block_qubits[0]; // else q == block_qubits[1] (low)
    let mut out = [[zero; 4]; 4];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let (hr, lr) = ((r >> 1) & 1, r & 1);
            let (hc, lc) = ((c >> 1) & 1, c & 1);
            *cell = if on_high {
                if lr == lc { m2[hr][hc] } else { zero }
            } else if hr == hc {
                m2[lr][lc]
            } else {
                zero
            };
        }
    }
    out
}

/// Re-express a 4×4 given in operand order `(a, b)` into order `(b, a)`
/// by swapping the two qubit bits of every index. Self-inverse.
fn reorder_swap(m: &M4) -> M4 {
    let zero = Complex::new(0.0, 0.0);
    let sw = |i: usize| ((i & 1) << 1) | ((i >> 1) & 1);
    let mut out = [[zero; 4]; 4];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = m[sw(r)][sw(c)];
        }
    }
    out
}

pub struct Fuse2q;

impl Pass for Fuse2q {
    fn name(&self) -> &'static str {
        "Fuse2q"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        use crate::Instruction;
        use aleph_core::gate::{Gate, GateInstance, GateMatrix};
        use smallvec::{smallvec, SmallVec};
        use std::collections::HashMap;

        let input: &[Instruction] = &circuit.instructions;
        let gates_before = input.len();

        // Accumulated unitary for a block: 1q (M2) or 2q (M4).
        enum Acc {
            One(M2),
            Two(M4),
        }
        struct Block {
            qubits: SmallVec<[u32; 2]>, // [hi, lo] for 2q; [q] for 1q
            acc: Acc,
            first_index: usize,
            len: usize,
            diag_only: bool, // meaningful only for 1q blocks
        }

        let mut blocks: Vec<Option<Block>> = Vec::new();
        let mut open: HashMap<u32, usize> = HashMap::new();
        // Each entry is (original_position, instruction). A fused block sorts
        // at its `first_index`; an inline gate sorts at its own loop index.
        // The stable sort at the end restores program order without any
        // reordering across disjoint qubits.
        let mut output: Vec<(usize, Instruction)> = Vec::with_capacity(input.len());
        let mut transformations: u64 = 0;

        fn emit_block(
            b: &Block,
            input: &[Instruction],
            output: &mut Vec<(usize, Instruction)>,
            transformations: &mut u64,
        ) {
            if b.len == 1 {
                // Verbatim re-emit — preserves named/specialised gates
                // (Cnot, Cz, …) and the original 1q gate.
                output.push((b.first_index, input[b.first_index].clone()));
                return;
            }
            *transformations += 1;
            match &b.acc {
                Acc::One(m2) => {
                    let gate = if b.diag_only {
                        Gate::Unitary1qDiag(Box::new([m2[0][0], m2[1][1]]))
                    } else {
                        Gate::Unitary1q(Box::new(*m2))
                    };
                    output.push((
                        b.first_index,
                        Instruction::Gate(GateInstance::new(gate, smallvec![b.qubits[0]])),
                    ));
                }
                Acc::Two(m4) => {
                    output.push((
                        b.first_index,
                        Instruction::Gate(GateInstance::new(
                            Gate::Unitary2q(Box::new(*m4)),
                            smallvec![b.qubits[0], b.qubits[1]],
                        )),
                    ));
                }
            }
        }

        fn flush_block(
            id: usize,
            blocks: &mut [Option<Block>],
            open: &mut HashMap<u32, usize>,
            input: &[Instruction],
            output: &mut Vec<(usize, Instruction)>,
            transformations: &mut u64,
        ) {
            let Some(b) = blocks[id].take() else { return };
            for q in &b.qubits {
                if open.get(q) == Some(&id) {
                    open.remove(q);
                }
            }
            emit_block(&b, input, output, transformations);
        }

        fn flush_qubit(
            q: u32,
            blocks: &mut [Option<Block>],
            open: &mut HashMap<u32, usize>,
            input: &[Instruction],
            output: &mut Vec<(usize, Instruction)>,
            transformations: &mut u64,
        ) {
            if let Some(&id) = open.get(&q) {
                flush_block(id, blocks, open, input, output, transformations);
            }
        }

        for (idx, inst) in input.iter().enumerate() {
            match inst {
                // ---- fusable 1q gate ----
                Instruction::Gate(g)
                    if g.gate.arity() == 1 && g.controls.is_empty() && g.qubits.len() == 1 =>
                {
                    let q = g.qubits[0];
                    let m2 = match g.gate.matrix() {
                        Ok(GateMatrix::M2x2(m)) => m,
                        _ => {
                            // Symbolic / unexpected — fence and re-emit.
                            flush_qubit(q, &mut blocks, &mut open, input, &mut output, &mut transformations);
                            output.push((idx, inst.clone()));
                            continue;
                        }
                    };
                    let g_diag = g.gate.is_diagonal();
                    if let Some(&id) = open.get(&q) {
                        let b = blocks[id].as_mut().unwrap();
                        match &mut b.acc {
                            Acc::One(acc) => {
                                left_mul2(acc, &m2);
                                b.diag_only &= g_diag;
                            }
                            Acc::Two(acc) => {
                                let bq = [b.qubits[0], b.qubits[1]];
                                let lifted = lift_1q(&m2, q, &bq);
                                *acc = mul4(&lifted, acc);
                            }
                        }
                        b.len += 1;
                    } else {
                        blocks.push(Some(Block {
                            qubits: smallvec![q],
                            acc: Acc::One(m2),
                            first_index: idx,
                            len: 1,
                            diag_only: g_diag,
                        }));
                        open.insert(q, blocks.len() - 1);
                    }
                }

                // ---- fusable 2q gate ----
                Instruction::Gate(g)
                    if g.gate.arity() == 2 && g.controls.is_empty() && g.qubits.len() == 2 =>
                {
                    let q0 = g.qubits[0];
                    let q1 = g.qubits[1];
                    let m4 = match g.gate.matrix() {
                        Ok(GateMatrix::M4x4(m)) => m,
                        _ => {
                            for q in [q0, q1] {
                                flush_qubit(q, &mut blocks, &mut open, input, &mut output, &mut transformations);
                            }
                            output.push((idx, inst.clone()));
                            continue;
                        }
                    };

                    // Same-pair merge: both qubits already in the SAME open
                    // 2q block (which therefore has qubit set {q0,q1}).
                    let ba = open.get(&q0).copied();
                    let bb = open.get(&q1).copied();
                    if let (Some(ida), Some(idb)) = (ba, bb) {
                        if ida == idb {
                            let b = blocks[ida].as_mut().unwrap();
                            debug_assert_eq!(b.qubits.len(), 2);
                            let oriented = if b.qubits[0] == q0 && b.qubits[1] == q1 {
                                m4
                            } else {
                                reorder_swap(&m4)
                            };
                            if let Acc::Two(acc) = &mut b.acc {
                                *acc = mul4(&oriented, acc);
                            } else {
                                return Err(PassError::InternalInvariant(
                                    "2q block did not hold a 4×4",
                                ));
                            }
                            b.len += 1;
                            continue;
                        }
                    }

                    // Otherwise open a fresh 2q block on [q0, q1].
                    // First flush any open *2q* block on q0/q1 — a 2q block
                    // can't be a pre-1q for a different 2q gate.
                    for q in [q0, q1] {
                        if let Some(&id) = open.get(&q) {
                            if blocks[id].as_ref().unwrap().qubits.len() == 2 {
                                flush_block(id, &mut blocks, &mut open, input, &mut output, &mut transformations);
                            }
                        }
                    }
                    // Fold any remaining (1q) open blocks on q0/q1 as pre-1q.
                    let mut pre = ident4();
                    let mut first_index = idx;
                    let mut len = 1usize;
                    for q in [q0, q1] {
                        if let Some(&id) = open.get(&q) {
                            let b = blocks[id].take().unwrap();
                            open.remove(&q);
                            if let Acc::One(m2) = b.acc {
                                let lifted = lift_1q(&m2, q, &[q0, q1]);
                                pre = mul4(&lifted, &pre);
                            }
                            first_index = first_index.min(b.first_index);
                            len += b.len;
                        }
                    }
                    let mat = mul4(&m4, &pre);
                    blocks.push(Some(Block {
                        qubits: smallvec![q0, q1],
                        acc: Acc::Two(mat),
                        first_index,
                        len,
                        diag_only: false,
                    }));
                    let id = blocks.len() - 1;
                    open.insert(q0, id);
                    open.insert(q1, id);
                }

                // ---- non-fusable gate (arity ≥ 3, controlled, etc.) ----
                Instruction::Gate(_) => {
                    for q in inst.used_qubits() {
                        flush_qubit(q, &mut blocks, &mut open, input, &mut output, &mut transformations);
                    }
                    output.push((idx, inst.clone()));
                }
                Instruction::Barrier(qs) => {
                    for q in qs {
                        flush_qubit(*q, &mut blocks, &mut open, input, &mut output, &mut transformations);
                    }
                    output.push((idx, inst.clone()));
                }
                Instruction::Measure { qubit, .. } => {
                    flush_qubit(*qubit, &mut blocks, &mut open, input, &mut output, &mut transformations);
                    output.push((idx, inst.clone()));
                }
                Instruction::Reset(qubit) => {
                    flush_qubit(*qubit, &mut blocks, &mut open, input, &mut output, &mut transformations);
                    output.push((idx, inst.clone()));
                }
            }
        }

        // Flush any remaining open blocks (order irrelevant — the stable sort
        // below restores program order by each item's original position).
        let remaining: Vec<usize> = (0..blocks.len()).filter(|&i| blocks[i].is_some()).collect();
        for id in remaining {
            flush_block(id, &mut blocks, &mut open, input, &mut output, &mut transformations);
        }

        // Emit in original program order: each fused block sorts at its first
        // gate's index, each inline gate at its own index. All positions are
        // distinct (every input index is consumed exactly once), and any two
        // items sharing a qubit have monotonically increasing positions, so this
        // preserves all dependencies while never reordering across the program.
        output.sort_by_key(|(pos, _)| *pos);
        let gates_after = output.len();
        circuit.instructions = output.into_iter().map(|(_, inst)| inst).collect();
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
    use aleph_core::gate::{Gate, GateMatrix};

    fn run_pass(c: &mut Circuit) -> PassStats {
        Fuse2q.run(c).expect("Fuse2q is infallible in tests")
    }

    fn the_gate(c: &Circuit, i: usize) -> &Gate {
        match &c.instructions()[i] {
            Instruction::Gate(g) => &g.gate,
            other => panic!("expected gate at {i}, got {other:?}"),
        }
    }

    // ---- helper unit tests (lift_1q / reorder_swap / mul4) ----

    #[test]
    fn lift_1q_low_bit_matches_kron_identity() {
        // m2 on the LSB qubit (block_qubits[1]); high bit untouched.
        let m2: M2 = [
            [Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)],
            [Complex::new(5.0, 0.0), Complex::new(7.0, 0.0)],
        ];
        let out = lift_1q(&m2, 1, &[0, 1]); // q=1 == block_qubits[1] (lo)
        // out[k_r][k_c] = m2[lo_r][lo_c] * δ(hi_r,hi_c)
        for (r, row) in out.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let (hr, lr) = ((r >> 1) & 1, r & 1);
                let (hc, lc) = ((c >> 1) & 1, c & 1);
                let want = if hr == hc { m2[lr][lc] } else { Complex::new(0.0, 0.0) };
                assert!((val - want).norm() < 1e-15, "lo r{r} c{c}");
            }
        }
    }

    #[test]
    fn lift_1q_high_bit_matches_kron_identity() {
        let m2: M2 = [
            [Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)],
            [Complex::new(5.0, 0.0), Complex::new(7.0, 0.0)],
        ];
        let out = lift_1q(&m2, 0, &[0, 1]); // q=0 == block_qubits[0] (hi)
        for (r, row) in out.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let (hr, lr) = ((r >> 1) & 1, r & 1);
                let (hc, lc) = ((c >> 1) & 1, c & 1);
                let want = if lr == lc { m2[hr][hc] } else { Complex::new(0.0, 0.0) };
                assert!((val - want).norm() < 1e-15, "hi r{r} c{c}");
            }
        }
    }

    #[test]
    fn reorder_swap_is_involution() {
        let mut m: M4 = [[Complex::new(0.0, 0.0); 4]; 4];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = Complex::new((r * 4 + c) as f64, (r + c) as f64);
            }
        }
        let twice = reorder_swap(&reorder_swap(&m));
        for (r, (trow, mrow)) in twice.iter().zip(m.iter()).enumerate() {
            for (c, (&tv, &mv)) in trow.iter().zip(mrow.iter()).enumerate() {
                assert!((tv - mv).norm() < 1e-15, "r{r} c{c}");
            }
        }
    }

    // ---- pass behaviour: empty / passthrough / 1q runs / fences ----

    #[test]
    fn empty_circuit_no_op() {
        let mut c = Circuit::new(2, 0);
        let s = run_pass(&mut c);
        assert_eq!((s.gates_before, s.gates_after, s.transformations), (0, 0, 0));
    }

    #[test]
    fn single_1q_is_verbatim() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 1);
        assert!(matches!(the_gate(&c, 0), Gate::H));
    }

    #[test]
    fn lone_cnot_is_verbatim_not_unitary2q() {
        // A 2q gate with nothing to fold must stay a named gate so the
        // specialised CNOT kernel is preserved.
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 1);
        assert!(matches!(the_gate(&c, 0), Gate::Cnot));
    }

    #[test]
    fn one_q_run_fuses() {
        // H · S on q0 → one fused 1q gate (Fuse2q subsumes 1q-run fusion).
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.s(0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        assert!(matches!(the_gate(&c, 0), Gate::Unitary1q(_)));
    }

    #[test]
    fn measure_fences() {
        let mut c = Circuit::new(2, 1);
        c.cnot(0, 1).unwrap();
        c.add_instruction(Instruction::Measure { qubit: 0, clbit: 0 }).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        // CNOT, Measure, CNOT — both CNOTs verbatim (fenced by measure).
        assert_eq!(c.instructions().len(), 3);
        assert!(matches!(the_gate(&c, 0), Gate::Cnot));
        assert!(matches!(the_gate(&c, 2), Gate::Cnot));
    }

    // ---- pre-1q / post-1q absorption ----

    #[test]
    fn pre_1q_absorbed_into_cnot() {
        // Rx(θ,0); CNOT(0,1) → one Unitary2q.
        let mut c = Circuit::new(2, 0);
        c.rx(0.5, 0).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        assert!(matches!(the_gate(&c, 0), Gate::Unitary2q(_)));
    }

    #[test]
    fn pre_1q_fused_matrix_matches_hand_computation() {
        use aleph_core::gate::Param;
        // Rx(0.7, q0); CNOT(0,1) → Unitary2q whose 4×4 equals
        // mul4(&cnot, &lift_1q(&rx, 0, &[0,1])).
        // This distinguishes correct compose order (cnot · pre) from
        // wrong order (pre · cnot) since mul4 is non-commutative here.
        let theta = 0.7_f64;
        let mut c = Circuit::new(2, 0);
        c.rx(theta, 0).unwrap();
        c.cnot(0, 1).unwrap();
        run_pass(&mut c);
        assert_eq!(c.instructions().len(), 1);
        let fused = match the_gate(&c, 0) {
            Gate::Unitary2q(m) => **m,
            other => panic!("expected Unitary2q, got {other:?}"),
        };
        let rx_mat = match Gate::Rx(Param::Concrete(theta)).matrix().unwrap() {
            GateMatrix::M2x2(m) => m,
            _ => unreachable!(),
        };
        let cnot = match Gate::Cnot.matrix().unwrap() {
            GateMatrix::M4x4(m) => m,
            _ => unreachable!(),
        };
        let lifted = lift_1q(&rx_mat, 0, &[0, 1]);
        let want = mul4(&cnot, &lifted);
        for (k, (frow, wrow)) in fused.iter().zip(want.iter()).enumerate() {
            for (j, (&fv, &wv)) in frow.iter().zip(wrow.iter()).enumerate() {
                assert!(
                    (fv - wv).norm() < 1e-12,
                    "fused[{k}][{j}] = {:?}, want {:?}",
                    fv,
                    wv,
                );
            }
        }
    }

    #[test]
    fn post_1q_absorbed_into_cnot() {
        // CNOT(0,1); Rz(θ,1) → one Unitary2q.
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.rz(0.5, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        assert!(matches!(the_gate(&c, 0), Gate::Unitary2q(_)));
    }

    // ---- same-pair 2q·2q merge + reversed operand order ----

    #[test]
    fn same_pair_cnot_cnot_merges() {
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        assert!(matches!(the_gate(&c, 0), Gate::Unitary2q(_)));
    }

    #[test]
    fn same_pair_reversed_operands_merge() {
        // CNOT(0,1); CNOT(1,0) — operand order reversed; reorder_swap path.
        // CNOT is asymmetric, so reorder_swap(CNOT) ≠ CNOT, which means
        // this test fails if reorder_swap is a no-op.
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.cnot(1, 0).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        // Fused = CNOT(1,0) · CNOT(0,1) expressed in block's [0,1] convention.
        // CNOT(1,0) in [0,1] convention = reorder_swap(CNOT_matrix).
        let fused = match the_gate(&c, 0) {
            Gate::Unitary2q(m) => **m,
            other => panic!("expected Unitary2q, got {other:?}"),
        };
        let cnot = match Gate::Cnot.matrix().unwrap() {
            GateMatrix::M4x4(m) => m,
            _ => unreachable!(),
        };
        // second gate (CNOT(1,0)) left-multiplies: reorder_swap gives it in
        // [0,1] convention, then multiply by the first block acc (CNOT(0,1)).
        let want = mul4(&reorder_swap(&cnot), &cnot);
        for (r, (frow, wrow)) in fused.iter().zip(want.iter()).enumerate() {
            for (c2, (&fv, &wv)) in frow.iter().zip(wrow.iter()).enumerate() {
                assert!((fv - wv).norm() < 1e-12, "r{r} c{c2}");
            }
        }
    }

    #[test]
    fn disjoint_op_does_not_block_same_pair_merge() {
        // CNOT(0,1); H(2); CNOT(0,1) → two CNOTs merge; H(2) preserved.
        let mut c = Circuit::new(3, 0);
        c.cnot(0, 1).unwrap();
        c.h(2).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        // One Unitary2q + the H(2) = 2 instructions.
        assert_eq!(c.instructions().len(), 2);
        let kinds: Vec<&str> = c.instructions().iter().map(|i| match i {
            Instruction::Gate(g) => g.gate.name(),
            _ => "non-gate",
        }).collect();
        assert!(kinds.contains(&"Unitary2q"));
        assert!(kinds.contains(&"H"));
    }

    #[test]
    fn shared_qubit_1q_folds_then_merges() {
        // CNOT(0,1); H(0); CNOT(0,1) → H folds (post-1q), 2nd CNOT merges
        // → one Unitary2q.
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert_eq!(c.instructions().len(), 1);
        assert!(matches!(the_gate(&c, 0), Gate::Unitary2q(_)));
    }

    #[test]
    fn toffoli_fences_fusion() {
        // CNOT(0,1); Toffoli(0,1,2); CNOT(0,1) → no merge (arity-3 fence).
        let mut c = Circuit::new(3, 0);
        c.cnot(0, 1).unwrap();
        c.ccx(0, 1, 2).unwrap();
        c.cnot(0, 1).unwrap();
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 0);
        assert_eq!(c.instructions().len(), 3);
        assert!(matches!(the_gate(&c, 1), Gate::Toffoli));
    }
}
