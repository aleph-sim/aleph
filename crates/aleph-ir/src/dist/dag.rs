//! Commutation-respecting dependency DAG for the P6-05 reordering scheduler.
//!
//! Each instruction gets an [`Act`] per qubit: `Z` (block-diagonal in that
//! qubit's computational basis: diagonal gates, controls), `X` (block-diagonal
//! in its Hadamard basis: `X`/`Rx`, CNOT/Toffoli/CRx targets) or `Other`.
//! Two instructions commute if, on every shared qubit, they have the same
//! `Z` or the same `X` type. Proof: each operator commutes with that qubit's
//! basis projectors, so both are block-diagonal over a common product basis of the shared
//! qubits, with blocks acting on disjoint unshared qubits. This is a subset of
//! `passes::commute::gates_commute`.
//!
//! Per qubit, consecutive same-type `Z`/`X` instructions form one *block*
//! (`Other` is always a block of one). An instruction is ready once every
//! member of the *previous* block on each of its qubits is scheduled. Counters
//! per block instead of explicit edges keep build and scheduling O(Σ arity)
//! even on long Z/X alternations.

use aleph_core::{Gate, GateInstance};
use smallvec::SmallVec;

use super::DistError;
use crate::{Circuit, Instruction};

/// How an instruction acts on one qubit (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Z,
    X,
    Other,
}

/// Per-qubit actions of `instr`. Errors on instructions the distributed
/// planner rejects (`Measure`, `Reset`, `TiledBlock`).
pub fn actions(instr: &Instruction) -> Result<SmallVec<[(u32, Act); 6]>, DistError> {
    let mut out: SmallVec<[(u32, Act); 6]> = SmallVec::new();
    match instr {
        Instruction::Gate(g) => gate_actions(g, &mut out),
        Instruction::DiagonalPhase(dp) => {
            let mut mask = 0u64;
            for t in &dp.terms {
                for &c in &t.conds {
                    mask |= c;
                }
            }
            while mask != 0 {
                out.push((mask.trailing_zeros(), Act::Z));
                mask &= mask - 1;
            }
        }
        Instruction::Barrier(qs) => out.extend(qs.iter().map(|&q| (q, Act::Other))),
        Instruction::Measure { .. } => return Err(DistError::Unsupported { kind: "measure" }),
        Instruction::Reset(_) => return Err(DistError::Unsupported { kind: "reset" }),
        Instruction::TiledBlock(_) => {
            return Err(DistError::Unsupported {
                kind: "tiled_block",
            })
        }
    }
    Ok(out)
}

fn gate_actions(g: &GateInstance, out: &mut SmallVec<[(u32, Act); 6]>) {
    out.extend(g.controls.iter().map(|&c| (c, Act::Z)));
    let q = &g.qubits;
    if g.gate.is_diagonal() {
        out.extend(q.iter().map(|&p| (p, Act::Z)));
        return;
    }
    match &g.gate {
        Gate::X | Gate::Rx(_) => out.push((q[0], Act::X)),
        Gate::Cnot | Gate::CRx(_) => out.extend([(q[0], Act::Z), (q[1], Act::X)]),
        Gate::CRy(_) => out.extend([(q[0], Act::Z), (q[1], Act::Other)]),
        Gate::Toffoli => out.extend([(q[0], Act::Z), (q[1], Act::Z), (q[2], Act::X)]),
        _ => out.extend(q.iter().map(|&p| (p, Act::Other))),
    }
}

/// Dependency DAG over a circuit's instructions (indices = positions).
pub struct Dag {
    /// Per instruction: the block it belongs to on each of its qubits.
    block_of: Vec<SmallVec<[u32; 6]>>,
    /// Per block: members not yet scheduled.
    remaining: Vec<u32>,
    /// Per block: instructions of the *next* block on the same qubit.
    waiters: Vec<Vec<u32>>,
    /// Per instruction: previous blocks it still waits on.
    pending: Vec<u32>,
    done: Vec<bool>,
}

impl Dag {
    /// Build the DAG for `c`; errors on instructions the planner rejects.
    pub fn build(c: &Circuit) -> Result<Self, DistError> {
        let n = c.num_qubits() as usize;
        let len = c.instructions().len();
        // Per qubit: (current block, its type, previous block).
        let mut cur: Vec<Option<(u32, Act, Option<u32>)>> = vec![None; n];
        let mut block_of = vec![SmallVec::new(); len];
        let mut remaining: Vec<u32> = Vec::new();
        let mut waiters: Vec<Vec<u32>> = Vec::new();
        let mut pending = vec![0u32; len];
        for (i, instr) in c.instructions().iter().enumerate() {
            for (q, act) in actions(instr)? {
                let slot = cur.get_mut(q as usize).ok_or(DistError::Unsupported {
                    kind: "internal: qubit out of range",
                })?;
                let (block, prev) = match *slot {
                    Some((b, t, prev)) if t == act && act != Act::Other => {
                        remaining[b as usize] += 1;
                        (b, prev)
                    }
                    other => {
                        let b = remaining.len() as u32;
                        remaining.push(1);
                        waiters.push(Vec::new());
                        let prev = other.map(|(pb, _, _)| pb);
                        *slot = Some((b, act, prev));
                        (b, prev)
                    }
                };
                if let Some(pb) = prev {
                    waiters[pb as usize].push(i as u32);
                    pending[i] += 1;
                }
                block_of[i].push(block);
            }
        }
        Ok(Self {
            block_of,
            remaining,
            waiters,
            pending,
            done: vec![false; len],
        })
    }

    pub fn initial_ready(&self) -> Vec<usize> {
        (0..self.len()).filter(|&i| self.pending[i] == 0).collect()
    }

    pub fn complete(&mut self, i: usize, ready: &mut Vec<usize>) -> Result<(), DistError> {
        if self.done.get(i).copied().unwrap_or(true) || self.pending[i] != 0 {
            return Err(DistError::Unsupported {
                kind: "internal: completing a non-ready DAG node",
            });
        }
        self.done[i] = true;
        for &b in &self.block_of[i] {
            let r = &mut self.remaining[b as usize];
            *r -= 1;
            if *r == 0 {
                for &w in &self.waiters[b as usize] {
                    let p = &mut self.pending[w as usize];
                    *p -= 1;
                    if *p == 0 {
                        ready.push(w as usize);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Param;
    use smallvec::smallvec;

    fn acts(i: Instruction) -> Vec<(u32, Act)> {
        let mut v = actions(&i).unwrap().to_vec();
        v.sort_by_key(|a| a.0);
        v
    }

    fn g(gate: Gate, q: &[u32]) -> Instruction {
        Instruction::Gate(GateInstance::new(gate, q.to_vec()))
    }

    #[test]
    fn classification_table() {
        use Act::*;
        let p = Param::Concrete(0.3);
        assert_eq!(acts(g(Gate::Rz(p), &[2])), vec![(2, Z)]);
        assert_eq!(acts(g(Gate::Cz, &[0, 3])), vec![(0, Z), (3, Z)]);
        assert_eq!(acts(g(Gate::Ccz, &[0, 1, 2])), vec![(0, Z), (1, Z), (2, Z)]);
        assert_eq!(acts(g(Gate::X, &[1])), vec![(1, X)]);
        assert_eq!(acts(g(Gate::Rx(p), &[1])), vec![(1, X)]);
        assert_eq!(acts(g(Gate::Cnot, &[2, 0])), vec![(0, X), (2, Z)]);
        assert_eq!(acts(g(Gate::CRx(p), &[0, 1])), vec![(0, Z), (1, X)]);
        assert_eq!(acts(g(Gate::CRy(p), &[0, 1])), vec![(0, Z), (1, Other)]);
        assert_eq!(
            acts(g(Gate::Toffoli, &[0, 1, 2])),
            vec![(0, Z), (1, Z), (2, X)]
        );
        assert_eq!(acts(g(Gate::H, &[0])), vec![(0, Other)]);
        assert_eq!(acts(g(Gate::Ry(p), &[0])), vec![(0, Other)]);
        assert_eq!(acts(g(Gate::Swap, &[0, 1])), vec![(0, Other), (1, Other)]);
        assert_eq!(acts(g(Gate::Iswap, &[0, 1])), vec![(0, Other), (1, Other)]);
        // External controls are Z; the target keeps its own class.
        let ch = Instruction::Gate(GateInstance::controlled(Gate::H, vec![1u32], vec![3u32]));
        assert_eq!(acts(ch), vec![(1, Other), (3, Z)]);
        let cx = Instruction::Gate(GateInstance::controlled(Gate::X, vec![1u32], vec![0u32]));
        assert_eq!(acts(cx), vec![(0, Z), (1, X)]);
        // DiagonalPhase: Z on every qubit of every cond mask.
        let dp = Instruction::DiagonalPhase(Box::new(crate::DiagonalPhase {
            n_qubits: 4,
            terms: vec![
                crate::PhaseTerm {
                    conds: smallvec![0b0101],
                    angle: 0.2,
                },
                crate::PhaseTerm {
                    conds: smallvec![0b1000, 0b0001],
                    angle: 0.1,
                },
            ],
        }));
        assert_eq!(acts(dp), vec![(0, Z), (2, Z), (3, Z)]);
        assert_eq!(
            acts(Instruction::Barrier(smallvec![1, 2])),
            vec![(1, Other), (2, Other)]
        );
        assert!(matches!(
            actions(&Instruction::Measure { qubit: 0, clbit: 0 }),
            Err(DistError::Unsupported { kind: "measure" })
        ));
    }

    /// Schedule in ascending-ready order, recording each wave of ready sets.
    fn waves(c: &Circuit) -> Vec<Vec<usize>> {
        let mut d = Dag::build(c).unwrap();
        let mut ready = d.initial_ready();
        let mut out = Vec::new();
        while !ready.is_empty() {
            ready.sort_unstable();
            out.push(ready.clone());
            let mut next = Vec::new();
            for i in std::mem::take(&mut ready) {
                d.complete(i, &mut next).unwrap();
            }
            ready = next;
        }
        out
    }

    #[test]
    fn same_type_run_is_unordered() {
        let mut c = Circuit::new(2, 0);
        c.rz(0.1, 0).unwrap();
        c.cz(0, 1).unwrap();
        c.t(0).unwrap();
        assert_eq!(waves(&c), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn x_then_z_is_ordered_and_joiners_wait_on_the_previous_block() {
        let mut c = Circuit::new(1, 0);
        c.rx(0.1, 0).unwrap();
        c.rz(0.2, 0).unwrap();
        c.rz(0.3, 0).unwrap();
        assert_eq!(waves(&c), vec![vec![0], vec![1, 2]]);
    }

    #[test]
    fn cnots_sharing_a_target_commute_but_control_target_swap_does_not() {
        let mut c = Circuit::new(3, 0);
        c.cnot(0, 2).unwrap();
        c.cnot(1, 2).unwrap();
        assert_eq!(waves(&c), vec![vec![0, 1]]);
        let mut d = Circuit::new(2, 0);
        d.cnot(0, 1).unwrap();
        d.cnot(1, 0).unwrap();
        assert_eq!(waves(&d), vec![vec![0], vec![1]]);
    }

    #[test]
    fn disjoint_gates_are_independent() {
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.h(1).unwrap();
        c.h(0).unwrap();
        assert_eq!(waves(&c), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn barrier_fences() {
        let mut c = Circuit::new(1, 0);
        c.rz(0.1, 0).unwrap();
        c.barrier([0u32]).unwrap();
        c.rz(0.2, 0).unwrap();
        assert_eq!(waves(&c), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn swap_relabel_is_a_fence_on_its_qubits() {
        let mut c = Circuit::new(3, 0);
        c.rz(0.1, 0).unwrap();
        c.swap(0, 1).unwrap();
        c.rz(0.2, 1).unwrap();
        c.h(2).unwrap();
        assert_eq!(waves(&c), vec![vec![0, 3], vec![1], vec![2]]);
    }

    #[test]
    fn complete_rejects_non_ready() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.h(0).unwrap();
        let mut d = Dag::build(&c).unwrap();
        let mut r = Vec::new();
        assert!(d.complete(1, &mut r).is_err());
        d.complete(0, &mut r).unwrap();
        assert!(d.complete(0, &mut r).is_err());
    }

    #[test]
    fn measure_rejected_by_build() {
        let mut c = Circuit::new(1, 1);
        c.measure(0, 0).unwrap();
        assert!(matches!(
            Dag::build(&c),
            Err(DistError::Unsupported { kind: "measure" })
        ));
    }
}
