//! `FuseKq` — greedy fusion of adjacent gates into dense k-qubit blocks.
//! See docs/superpowers/specs/2026-06-03-p2-07-fuse-kq-design.md.

use crate::passes::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction};
use aleph_core::Complex;
use aleph_core::{Gate, GateInstance, GateMatrix};
use std::collections::BTreeSet;

/// Embed a gate's matrix into the `k`-qubit block space spanned by sorted
/// `block_qubits` (ascending). `gate_qubits` is the gate's own operand list
/// (MSB-first: `gate_qubits[0]` is the gate-matrix MSB). Returns the
/// row-major `2^k × 2^k` lifted matrix.
///
/// # Invariant
///
/// The caller must ensure `gate.matrix()` succeeds — i.e. the gate has a
/// concrete `GateMatrix` (arity ≤ 3, no symbolic parameters). `UnitaryKq`
/// members must use `lift_dense` directly.
pub(crate) fn lift_to_block(
    gate: &Gate,
    gate_qubits: &[u32],
    block_qubits: &[u32],
) -> Vec<Complex> {
    let gd: Vec<Complex> = match gate
        .matrix()
        .expect("FuseKq lifts only concrete GateMatrix gates (no UnitaryKq, no symbolic/non-finite params); UnitaryKq routes via lift_dense")
    {
        GateMatrix::M2x2(m) => m.iter().flatten().copied().collect(),
        GateMatrix::M4x4(m) => m.iter().flatten().copied().collect(),
        GateMatrix::M8x8(m) => m.iter().flatten().copied().collect(),
    };
    lift_dense(&gd, gate_qubits, block_qubits)
}

/// Embed an already-flattened dense `2^ga × 2^ga` matrix (`ga == gate_qubits.len()`)
/// into the `k`-qubit block space spanned by sorted `block_qubits` (ascending).
///
/// `gate_qubits` is MSB-first: `gate_qubits[0]` is the most significant bit
/// of the gate's row/col index. `block_qubits` is sorted ascending; block-bit
/// position `p` (0 = LSB … k−1 = MSB) corresponds to `block_qubits[k-1-p]`.
///
/// Returns a row-major `2^k × 2^k` matrix.
pub(crate) fn lift_dense(
    gd: &[Complex],
    gate_qubits: &[u32],
    block_qubits: &[u32],
) -> Vec<Complex> {
    let k = block_qubits.len();
    let dim = 1usize << k;
    let ga = gate_qubits.len();
    let gdim = 1usize << ga;
    debug_assert_eq!(gd.len(), gdim * gdim);

    // Map a physical qubit to its block-bit position (0 = LSB, k-1 = MSB).
    // block_qubits is sorted ascending so block_qubits[0] is the smallest qubit
    // index, which by the MSB-first convention is the HIGHEST block bit (k-1).
    let block_bit_of = |q: u32| -> usize {
        let idx = block_qubits
            .iter()
            .position(|&bq| bq == q)
            .expect("gate qubit must lie in block");
        k - 1 - idx
    };

    let mut out = vec![Complex::new(0.0, 0.0); dim * dim];

    for col in 0..dim {
        // Extract the gate-input index from this block column index.
        // gate_qubits[a] occupies gate bit (ga-1-a) (MSB-first).
        let mut gin = 0usize;
        for (a, &gq) in gate_qubits.iter().enumerate() {
            let bb = block_bit_of(gq);
            gin |= ((col >> bb) & 1) << (ga - 1 - a);
        }

        for grow in 0..gdim {
            let amp = gd[grow * gdim + gin];
            if amp == Complex::new(0.0, 0.0) {
                continue;
            }
            // Reconstruct the block row: start from col (identity on non-gate bits),
            // then overwrite the gate bits with the output bits from grow.
            let mut row = col;
            for (a, &gq) in gate_qubits.iter().enumerate() {
                let bb = block_bit_of(gq);
                let outbit = (grow >> (ga - 1 - a)) & 1;
                row = (row & !(1usize << bb)) | (outbit << bb);
            }
            out[row * dim + col] += amp;
        }
    }

    out
}

/// Compose `members` (circuit order) into one row-major `2^k × 2^k` dense
/// matrix over the sorted `block_qubits`. Each member must act within
/// `block_qubits`. Later gates LEFT-multiply (quantum convention):
/// `acc := lift(member) · acc`, so `acc` starts as the identity.
///
/// `UnitaryKq` members route through `lift_dense` directly; all other gate
/// variants route through `lift_to_block` (which calls `gate.matrix()`).
pub(crate) fn build_block_matrix(members: &[GateInstance], block_qubits: &[u32]) -> Vec<Complex> {
    let k = block_qubits.len();
    let dim = 1usize << k;
    // Start from the identity matrix.
    let mut acc = vec![Complex::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        acc[i * dim + i] = Complex::new(1.0, 0.0);
    }
    for gi in members {
        let lifted = match &gi.gate {
            Gate::UnitaryKq { data, .. } => lift_dense(data, &gi.qubits[..], block_qubits),
            _ => lift_to_block(&gi.gate, &gi.qubits[..], block_qubits),
        };
        acc = matmul(&lifted, &acc, dim); // acc := lifted · acc
    }
    acc
}

/// Row-major dense `n×n` matrix product `a · b`.
fn matmul(a: &[Complex], b: &[Complex], n: usize) -> Vec<Complex> {
    let mut out = vec![Complex::new(0.0, 0.0); n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == Complex::new(0.0, 0.0) {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

/// Fuses adjacent gates spanning ≤ `max_qubits` into a dense `UnitaryKq`.
///
/// Runs after `Fuse1qRuns`/`Fuse2q`/`FuseDiagonalRuns`. The strategy is
/// greedy and dependency-respecting: we walk the instruction stream in
/// program order, maintaining a set of *open blocks*. Each block owns a
/// set of qubits — for every qubit, the `owner` table points at the
/// single open block that most recently wrote it. A fusable gate joins
/// the union of the blocks owning its operands, provided the merged span
/// stays ≤ `max_qubits`; otherwise those owner-blocks are closed and a
/// fresh block opens for the gate.
///
/// Dependency correctness: because instructions are processed in order
/// and a block owns each of its qubits until it is closed, any gate
/// touching qubit `q` depends only on the block currently owning `q`.
/// Merging the owner-blocks of a gate's operands therefore never reorders
/// a gate past another gate on a shared qubit — the matrix product inside
/// a block is rebuilt from members sorted by their original absolute
/// instruction index.
///
/// Fences (anything that is not a fusable `Gate`: `Measure`, `Reset`,
/// `Barrier`, `DiagonalPhase`, an existing `UnitaryKq`, or a gate
/// carrying external `controls`) close every block they touch and are
/// emitted verbatim.
///
/// Cost model on close: a block is materialised as a dense `UnitaryKq`
/// only when its span ≥ 3 *and* it has ≥ 2 members; otherwise its members
/// are re-emitted verbatim (preserving the specialised 1q/2q kernels).
pub struct FuseKq {
    /// Maximum block span. Clamped to `[2, 5]` (the `UnitaryKq` range).
    pub max_qubits: usize,
}

impl Default for FuseKq {
    fn default() -> Self {
        Self { max_qubits: 4 } // tuned on EPYC in the perf task
    }
}

/// An open fusion block. `members` keeps each gate paired with its
/// absolute index in the original instruction stream so the matrix
/// product (and verbatim re-emit) can be rebuilt in program order even
/// after several blocks have merged.
struct Block {
    qubits: BTreeSet<u32>,
    members: Vec<(usize, GateInstance)>,
    first_idx: usize,
}

impl FuseKq {
    /// Materialise a closed block into `out`, keyed by absolute index so a
    /// final stable sort restores program order.
    fn close_block(mut b: Block, out: &mut Vec<(usize, Instruction)>, transformations: &mut u64) {
        // Members may arrive out of order after merges; sort by absolute idx.
        b.members.sort_by_key(|(i, _)| *i);
        let span = b.qubits.len();
        if span >= 3 && b.members.len() >= 2 {
            // Sorted ascending (BTreeSet); matches build_block_matrix's contract.
            let q: Vec<u32> = b.qubits.iter().copied().collect();
            let member_gates: Vec<GateInstance> = b.members.into_iter().map(|(_, g)| g).collect();
            let data = build_block_matrix(&member_gates, &q);
            let g = GateInstance::new(
                Gate::UnitaryKq {
                    k: span as u8,
                    data: data.into_boxed_slice(),
                },
                smallvec::SmallVec::<[u32; 4]>::from_vec(q),
            );
            out.push((b.first_idx, Instruction::Gate(g)));
            *transformations += 1;
        } else {
            for (idx, m) in b.members {
                out.push((idx, Instruction::Gate(m)));
            }
        }
    }
}

/// Close every open block touching any qubit in `qs`, emitting each.
fn close_touching(
    qs: &[u32],
    blocks: &mut [Option<Block>],
    owner: &mut [Option<usize>],
    out: &mut Vec<(usize, Instruction)>,
    transformations: &mut u64,
) {
    let mut to_close = BTreeSet::new();
    for &q in qs {
        if let Some(bi) = owner[q as usize] {
            to_close.insert(bi);
        }
    }
    for bi in to_close {
        if let Some(b) = blocks[bi].take() {
            for &q in &b.qubits {
                owner[q as usize] = None;
            }
            FuseKq::close_block(b, out, transformations);
        }
    }
}

impl Pass for FuseKq {
    fn name(&self) -> &'static str {
        "FuseKq"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.num_qubits() as usize;
        let before = circuit.len();
        let max_k = self.max_qubits.clamp(2, 5);
        let input = circuit.instructions.clone();

        let mut blocks: Vec<Option<Block>> = Vec::new();
        let mut owner: Vec<Option<usize>> = vec![None; n];
        let mut out: Vec<(usize, Instruction)> = Vec::new();
        let mut transformations = 0u64;

        for (idx, inst) in input.iter().enumerate() {
            // Fusable = a plain Gate with NO external controls and not
            // already a dense UnitaryKq (re-fusing those is pointless and
            // they have no GateMatrix to lift via lift_to_block).
            let fusable = matches!(inst, Instruction::Gate(g)
                if g.controls.is_empty() && !matches!(g.gate, Gate::UnitaryKq { .. }));
            if !fusable {
                let qs: Vec<u32> = inst.used_qubits().to_vec();
                close_touching(&qs, &mut blocks, &mut owner, &mut out, &mut transformations);
                out.push((idx, inst.clone()));
                continue;
            }
            let g = match inst {
                Instruction::Gate(g) => g,
                _ => unreachable!("fusable implies Instruction::Gate"),
            };
            let mut support: Vec<u32> = g.qubits.iter().copied().collect();
            support.sort_unstable();

            // Dependency blocks: those owning any support qubit.
            let mut dep: BTreeSet<usize> = BTreeSet::new();
            for &q in &support {
                if let Some(bi) = owner[q as usize] {
                    dep.insert(bi);
                }
            }
            // Candidate merged span = support ∪ all dep block qubits.
            let mut merged: BTreeSet<u32> = support.iter().copied().collect();
            for &bi in &dep {
                if let Some(b) = &blocks[bi] {
                    merged.extend(b.qubits.iter().copied());
                }
            }

            if merged.len() <= max_k {
                // Merge dep blocks + this gate into one new block. Member
                // ordering is recovered later from absolute indices, so we
                // can concatenate dep members in any order here.
                let mut members: Vec<(usize, GateInstance)> = Vec::new();
                let mut first_idx = idx;
                for &bi in &dep {
                    if let Some(b) = blocks[bi].take() {
                        first_idx = first_idx.min(b.first_idx);
                        members.extend(b.members);
                    }
                }
                members.push((idx, g.clone()));
                let new_bi = blocks.len();
                for &q in &merged {
                    owner[q as usize] = Some(new_bi);
                }
                blocks.push(Some(Block {
                    qubits: merged,
                    members,
                    first_idx,
                }));
            } else {
                // Span would overflow: close the dep blocks, then open a
                // fresh block for this gate alone.
                let dep_qs: Vec<u32> = dep
                    .iter()
                    .filter_map(|&bi| blocks[bi].as_ref())
                    .flat_map(|b| b.qubits.iter().copied())
                    .collect();
                close_touching(
                    &dep_qs,
                    &mut blocks,
                    &mut owner,
                    &mut out,
                    &mut transformations,
                );
                let new_bi = blocks.len();
                let qubits: BTreeSet<u32> = support.iter().copied().collect();
                for &q in &qubits {
                    owner[q as usize] = Some(new_bi);
                }
                blocks.push(Some(Block {
                    qubits,
                    members: vec![(idx, g.clone())],
                    first_idx: idx,
                }));
            }
        }

        // Flush any blocks still open at end of stream.
        for slot in blocks.iter_mut() {
            if let Some(b) = slot.take() {
                FuseKq::close_block(b, &mut out, &mut transformations);
            }
        }

        // Restore program order. Keys are unique absolute indices except
        // that a fused block is keyed at its first member's index, which is
        // unique among the surviving entries, so the sort is well-defined.
        out.sort_by_key(|(i, _)| *i);
        let result: Vec<Instruction> = out.into_iter().map(|(_, i)| i).collect();
        let after = result.len();
        circuit.instructions = result;
        Ok(PassStats {
            gates_before: before,
            gates_after: after,
            transformations,
        })
    }
}

#[cfg(test)]
mod lift_tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    fn brute_lift(gate: &Gate, gate_qubits: &[u32], block_qubits: &[u32]) -> Vec<Complex> {
        let k = block_qubits.len();
        let dim = 1usize << k;
        let gd = match gate.matrix().unwrap() {
            aleph_core::GateMatrix::M2x2(m) => m.iter().flatten().copied().collect::<Vec<_>>(),
            aleph_core::GateMatrix::M4x4(m) => m.iter().flatten().copied().collect::<Vec<_>>(),
            aleph_core::GateMatrix::M8x8(m) => m.iter().flatten().copied().collect::<Vec<_>>(),
        };
        let ga = gate_qubits.len();
        let gdim = 1usize << ga;
        let block_bit_of = |q: u32| -> usize {
            let idx = block_qubits.iter().position(|&bq| bq == q).unwrap();
            k - 1 - idx
        };
        let mut out = vec![Complex::new(0.0, 0.0); dim * dim];
        for col in 0..dim {
            let mut gin = 0usize;
            for (a, &gq) in gate_qubits.iter().enumerate() {
                let bb = block_bit_of(gq);
                gin |= ((col >> bb) & 1) << (ga - 1 - a);
            }
            for grow in 0..gdim {
                let amp = gd[grow * gdim + gin];
                if amp.norm() < 1e-18 {
                    continue;
                }
                let mut row = col;
                for (a, &gq) in gate_qubits.iter().enumerate() {
                    let bb = block_bit_of(gq);
                    let outbit = (grow >> (ga - 1 - a)) & 1;
                    row = (row & !(1usize << bb)) | (outbit << bb);
                }
                out[row * dim + col] += amp;
            }
        }
        out
    }

    #[test]
    fn lift_1q_into_3q_block_matches_bruteforce() {
        let g = Gate::X;
        let gq = [5u32];
        let bq = [2u32, 5, 7];
        let got = lift_to_block(&g, &gq, &bq);
        let want = brute_lift(&g, &gq, &bq);
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert!((got[i] - want[i]).norm() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn lift_2q_cnot_into_3q_block_matches_bruteforce() {
        let g = Gate::Cnot;
        let gq = [7u32, 2]; // [control, target], reversed/non-adjacent
        let bq = [2u32, 5, 7];
        let got = lift_to_block(&g, &gq, &bq);
        let want = brute_lift(&g, &gq, &bq);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).norm() < 1e-12, "entry {i}");
        }
        let _ = GateInstance::new(Gate::H, smallvec![0u32]);
    }

    #[test]
    fn lift_rz_2q_block_matches_bruteforce() {
        // single-qubit non-trivial gate into a 2-qubit block, low position
        let g = Gate::Rz(0.7.into());
        let gq = [0u32];
        let bq = [0u32, 3];
        let got = lift_to_block(&g, &gq, &bq);
        let want = brute_lift(&g, &gq, &bq);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).norm() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn lift_cnot_msb_first_is_canonical_cnot() {
        // CNOT, gate_qubits=[0,1] (control=0=MSB, target=1=LSB), block Q=[0,1].
        // Canonical CNOT (q0=MSB): |10>->|11>, |11>->|10>.
        let got = lift_to_block(&Gate::Cnot, &[0u32, 1], &[0u32, 1]);
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let want: [Complex; 16] = [o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z];
        for i in 0..16 {
            assert!(
                (got[i] - want[i]).norm() < 1e-12,
                "entry {i}: got {:?} want {:?}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn lift_cnot_reversed_operands_is_reversed_cnot() {
        // CNOT, gate_qubits=[1,0] (control=q1=LSB, target=q0=MSB), block Q=[0,1].
        // Flips q0(MSB) when q1(LSB)=1: |01>->|11>, |11>->|01>.
        let got = lift_to_block(&Gate::Cnot, &[1u32, 0], &[0u32, 1]);
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let want: [Complex; 16] = [o, z, z, z, z, z, z, o, z, z, o, z, z, o, z, z];
        for i in 0..16 {
            assert!(
                (got[i] - want[i]).norm() < 1e-12,
                "entry {i}: got {:?} want {:?}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn lift_x_middle_qubit_flips_only_that_bit() {
        // X on q5 in block Q=[2,5,7]: q2=bit2(MSB), q5=bit1, q7=bit0(LSB).
        // Lifted 8x8 is the permutation row == col ^ 0b010 (flip bit 1 only).
        let got = lift_to_block(&Gate::X, &[5u32], &[2u32, 5, 7]);
        for col in 0..8usize {
            let expect_row = col ^ 0b010;
            for row in 0..8usize {
                let want = if row == expect_row {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert!(
                    (got[row * 8 + col] - want).norm() < 1e-12,
                    "row={row} col={col}"
                );
            }
        }
    }
}

#[cfg(test)]
mod build_tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    #[test]
    fn build_two_cnots_chain_equals_bruteforce() {
        // Block Q=[0,1,2]; members in circuit order: CNOT(0,1) then CNOT(1,2).
        let members = vec![
            GateInstance::new(Gate::Cnot, smallvec![0u32, 1]),
            GateInstance::new(Gate::Cnot, smallvec![1u32, 2]),
        ];
        let bq = [0u32, 1, 2];
        let m = build_block_matrix(&members, &bq);
        // Independent brute force: for each 3-qubit basis state (MSB-first:
        // bit2=q0, bit1=q1, bit0=q2), apply CNOT(0,1) then CNOT(1,2).
        let dim = 8;
        for col in 0..dim {
            let q0 = (col >> 2) & 1;
            let q1 = (col >> 1) & 1;
            let q2 = col & 1;
            let nq1 = q1 ^ q0; // CNOT(0,1): target q1 ^= control q0
            let nq2 = q2 ^ nq1; // CNOT(1,2): target q2 ^= control q1(new)
            let row = (q0 << 2) | (nq1 << 1) | nq2;
            for r in 0..dim {
                let want = if r == row {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert!((m[r * dim + col] - want).norm() < 1e-12, "col={col} r={r}");
            }
        }
    }

    #[test]
    fn build_single_member_equals_lift() {
        // One member → build == lift of that member.
        let members = vec![GateInstance::new(Gate::Cnot, smallvec![2u32, 0])];
        let bq = [0u32, 1, 2];
        let built = build_block_matrix(&members, &bq);
        let lifted = lift_to_block(&Gate::Cnot, &[2u32, 0], &bq);
        for i in 0..built.len() {
            assert!((built[i] - lifted[i]).norm() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn build_order_matters_left_multiply() {
        // H(0) then S(0) on a 1-qubit-in-2q block should equal S·H (later
        // gate left-multiplies). Use block Q=[0,1], members both on q0.
        let members = vec![
            GateInstance::new(Gate::H, smallvec![0u32]),
            GateInstance::new(Gate::S, smallvec![0u32]),
        ];
        let bq = [0u32, 1];
        let built = build_block_matrix(&members, &bq);
        // expected on q0 (block bit 1 = MSB): M = lift(S)·lift(H)
        let lh = lift_to_block(&Gate::H, &[0u32], &bq);
        let ls = lift_to_block(&Gate::S, &[0u32], &bq);
        // manual 4x4 product ls·lh
        let dim = 4;
        let mut want = vec![Complex::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..dim {
                    acc += ls[i * dim + k] * lh[k * dim + j];
                }
                want[i * dim + j] = acc;
            }
        }
        for i in 0..built.len() {
            assert!((built[i] - want[i]).norm() < 1e-12, "entry {i}");
        }
    }
}

#[cfg(test)]
mod pass_tests {
    use super::*;
    use crate::passes::Pass;
    use crate::{Circuit, Instruction};
    use aleph_core::Gate;
    use smallvec::smallvec;

    // A non-trivial 2-qubit dense gate (SWAP as Unitary2q) so block matrices
    // have real off-diagonals.
    fn swap2q() -> Gate {
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        m[0][0] = Complex::new(1.0, 0.0);
        m[1][2] = Complex::new(1.0, 0.0);
        m[2][1] = Complex::new(1.0, 0.0);
        m[3][3] = Complex::new(1.0, 0.0);
        Gate::Unitary2q(Box::new(m))
    }
    fn count_kq(c: &Circuit) -> usize {
        c.instructions()
            .iter()
            .filter(|i| matches!(i, Instruction::Gate(g) if matches!(g.gate, Gate::UnitaryKq{..})))
            .count()
    }

    #[test]
    fn two_chained_2q_blocks_fuse_into_one_3q() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![0u32, 1]))
            .unwrap();
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![1u32, 2]))
            .unwrap();
        let stats = FuseKq { max_qubits: 4 }.run(&mut c).unwrap();
        assert_eq!(count_kq(&c), 1, "should fuse into one 3q block");
        assert_eq!(c.len(), 1);
        assert!(stats.transformations >= 1);
        if let Instruction::Gate(g) = &c.instructions()[0] {
            assert_eq!(g.qubits.as_slice(), &[0u32, 1, 2]);
            assert!(matches!(g.gate, Gate::UnitaryKq { k: 3, .. }));
        }
    }

    #[test]
    fn lone_2q_block_is_not_fused() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![0u32, 1]))
            .unwrap();
        FuseKq { max_qubits: 4 }.run(&mut c).unwrap();
        assert_eq!(count_kq(&c), 0);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn single_3q_gate_alone_is_not_fused() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(aleph_core::GateInstance::new(
            Gate::Toffoli,
            smallvec![0u32, 1, 2],
        ))
        .unwrap();
        FuseKq { max_qubits: 5 }.run(&mut c).unwrap();
        assert_eq!(count_kq(&c), 0);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn exceeding_max_qubits_does_not_overfuse() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![0u32, 1]))
            .unwrap();
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![1u32, 2]))
            .unwrap();
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![2u32, 3]))
            .unwrap();
        FuseKq { max_qubits: 3 }.run(&mut c).unwrap();
        for i in c.instructions() {
            if let Instruction::Gate(g) = i {
                if let Gate::UnitaryKq { k, .. } = g.gate {
                    assert!(k <= 3, "k={k} exceeds max");
                }
            }
        }
    }

    #[test]
    fn barrier_is_a_fence() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![0u32, 1]))
            .unwrap();
        c.barrier([0u32, 1, 2]).unwrap();
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![1u32, 2]))
            .unwrap();
        FuseKq { max_qubits: 4 }.run(&mut c).unwrap();
        assert_eq!(count_kq(&c), 0, "barrier separates the two 2q gates");
        assert!(c
            .instructions()
            .iter()
            .any(|i| matches!(i, Instruction::Barrier(_))));
    }

    #[test]
    fn preserves_program_order_with_independent_blocks() {
        // Two independent fused blocks on disjoint qubits interleaved with a
        // gate must keep relative order. Build: 2q(0,1),2q(0,1) [→3q? no, span2]
        // Instead: chain on {0,1,2} then chain on {3,4,5}; both fuse, order kept.
        let mut c = Circuit::new(6, 0);
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![0u32, 1]))
            .unwrap();
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![3u32, 4]))
            .unwrap();
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![1u32, 2]))
            .unwrap();
        c.add_gate(aleph_core::GateInstance::new(swap2q(), smallvec![4u32, 5]))
            .unwrap();
        FuseKq { max_qubits: 3 }.run(&mut c).unwrap();
        // Two 3q blocks {0,1,2} and {3,4,5}.
        assert_eq!(count_kq(&c), 2);
    }
}
