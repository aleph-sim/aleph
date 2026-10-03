//! Distributed planner: logical→physical tracking, exchange insertion.
//!
//! Naive router: one global qubit per exchange, on demand, swapped with the
//! top local physical slot `m-1` (so exchanged chunks are contiguous halves).

use aleph_core::{Gate, GateInstance};
use smallvec::{smallvec, SmallVec};

use super::{CommStats, DistError, DistLayout, DistPlan, DistStep, Router};
use crate::{Circuit, DiagonalPhase, Instruction, PhaseTerm};

/// Logical qubits of `g` that must be local for it to run without an exchange.
///
/// Diagonal gates need none (a global qubit only selects a phase); controls —
/// external or in-qubit (`Cnot`/`CRx`/`CRy`/`Toffoli`) — need none; every
/// other qubit is a non-diagonal target and must be local.
pub fn required_local(g: &GateInstance) -> SmallVec<[u32; 4]> {
    if g.gate.is_diagonal() {
        return SmallVec::new();
    }
    match &g.gate {
        Gate::Cnot | Gate::CRx(_) | Gate::CRy(_) => smallvec![g.qubits[1]],
        Gate::Toffoli => smallvec![g.qubits[2]],
        _ => g.qubits.clone(),
    }
}

/// Bidirectional logical↔physical qubit map.
struct Map {
    l2p: Vec<u32>,
    p2l: Vec<u32>,
}

impl Map {
    fn new(n: u32) -> Self {
        Self {
            l2p: (0..n).collect(),
            p2l: (0..n).collect(),
        }
    }

    /// Swap the logical occupants of physical slots `a` and `b`.
    fn swap_phys(&mut self, a: u32, b: u32) {
        let (la, lb) = (self.p2l[a as usize], self.p2l[b as usize]);
        self.p2l.swap(a as usize, b as usize);
        self.l2p[la as usize] = b;
        self.l2p[lb as usize] = a;
    }

    fn remap_mask(&self, mask: u64) -> u64 {
        let mut out = 0u64;
        let mut m = mask;
        while m != 0 {
            let l = m.trailing_zeros();
            out |= 1u64 << self.l2p[l as usize];
            m &= m - 1;
        }
        out
    }
}

/// Build a distributed plan for `circuit` under `layout`.
///
/// `Local` instructions in the result use physical qubit indices; run them
/// through [`super::specialize`] per rank.
pub fn plan(circuit: &Circuit, layout: DistLayout, router: Router) -> Result<DistPlan, DistError> {
    let Router::Naive = router;
    if circuit.num_qubits() != layout.n {
        return Err(DistError::QubitCountMismatch {
            circuit: circuit.num_qubits(),
            layout: layout.n,
        });
    }
    let m = layout.m();
    let mut map = Map::new(layout.n);
    let mut steps: Vec<DistStep> = Vec::new();
    let mut cur: Vec<Instruction> = Vec::new();
    let mut stats = CommStats::default();

    for instr in circuit.instructions() {
        match instr {
            Instruction::Barrier(_) => {}
            Instruction::Measure { .. } => return Err(DistError::Unsupported { kind: "measure" }),
            Instruction::Reset(_) => return Err(DistError::Unsupported { kind: "reset" }),
            Instruction::TiledBlock(_) => {
                return Err(DistError::Unsupported {
                    kind: "tiled_block",
                })
            }
            Instruction::DiagonalPhase(dp) => {
                let terms = dp
                    .terms
                    .iter()
                    .map(|t| PhaseTerm {
                        conds: t.conds.iter().map(|&c| map.remap_mask(c)).collect(),
                        angle: t.angle,
                    })
                    .collect();
                cur.push(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
                    n_qubits: layout.n,
                    terms,
                })));
            }
            Instruction::Gate(g) => {
                if matches!(g.gate, Gate::Swap) && g.controls.is_empty() {
                    let a = map.l2p[g.qubits[0] as usize];
                    let b = map.l2p[g.qubits[1] as usize];
                    map.swap_phys(a, b);
                    stats.relabels += 1;
                    continue;
                }
                let req = required_local(g);
                if req.len() > m as usize {
                    return Err(DistError::TooFewLocalQubits { need: req.len(), m });
                }
                for &q in &req {
                    if !layout.is_global(map.l2p[q as usize]) {
                        continue;
                    }
                    let top = m - 1;
                    // Never evict a qubit this same gate needs local.
                    if req.contains(&map.p2l[top as usize]) {
                        let free = (0..top)
                            .rev()
                            .find(|&p| !req.contains(&map.p2l[p as usize]))
                            .ok_or(DistError::TooFewLocalQubits { need: req.len(), m })?;
                        cur.push(Instruction::Gate(GateInstance::new(
                            Gate::Swap,
                            vec![top, free],
                        )));
                        map.swap_phys(top, free);
                        stats.local_swaps += 1;
                    }
                    if !cur.is_empty() {
                        steps.push(DistStep::Local(std::mem::take(&mut cur)));
                    }
                    let gbit = map.l2p[q as usize];
                    steps.push(DistStep::Exchange {
                        global_bits: smallvec![gbit],
                    });
                    map.swap_phys(gbit, top);
                    stats.exchanges += 1;
                    // Saturate: n=64 with g=1 moves 2^62 amps per exchange.
                    stats.amps_moved_per_rank =
                        stats.amps_moved_per_rank.saturating_add(1u64 << (m - 1));
                }
                cur.push(Instruction::Gate(GateInstance {
                    gate: g.gate.clone(),
                    qubits: g.qubits.iter().map(|&l| map.l2p[l as usize]).collect(),
                    controls: g.controls.iter().map(|&l| map.l2p[l as usize]).collect(),
                }));
            }
        }
    }
    if !cur.is_empty() {
        steps.push(DistStep::Local(cur));
    }
    if steps.is_empty() {
        steps.push(DistStep::Local(Vec::new()));
    }
    Ok(DistPlan {
        layout,
        steps,
        final_map: map.l2p,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{DistStep, Router};
    use crate::Instruction;
    use aleph_core::{Gate, GateInstance, Param};

    fn exchanges(p: &DistPlan) -> usize {
        p.steps
            .iter()
            .filter(|s| matches!(s, DistStep::Exchange { .. }))
            .count()
    }

    #[test]
    fn required_local_rules() {
        assert!(required_local(&GateInstance::new(Gate::Cz, vec![0, 1])).is_empty());
        assert_eq!(
            required_local(&GateInstance::new(Gate::Cnot, vec![0, 1])).as_slice(),
            &[1]
        );
        assert_eq!(
            required_local(&GateInstance::new(Gate::Toffoli, vec![0, 1, 2])).as_slice(),
            &[2]
        );
        assert_eq!(
            required_local(&GateInstance::new(Gate::H, vec![3])).as_slice(),
            &[3]
        );
        assert!(
            required_local(&GateInstance::new(Gate::Rz(Param::Concrete(0.1)), vec![3])).is_empty()
        );
        assert_eq!(
            required_local(&GateInstance::new(Gate::Iswap, vec![0, 1])).as_slice(),
            &[0, 1]
        );
    }

    #[test]
    fn local_only_circuit_has_no_exchange() {
        let mut c = Circuit::new(6, 0);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.rz(0.3, 5).unwrap();
        c.cz(2, 5).unwrap();
        let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Naive).unwrap();
        assert_eq!(exchanges(&p), 0);
        assert_eq!(p.stats.exchanges, 0);
        assert_eq!(p.final_map, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn h_on_global_triggers_one_exchange() {
        let mut c = Circuit::new(6, 0);
        c.h(5).unwrap();
        let l = DistLayout::new(6, 2).unwrap(); // m = 4
        let p = plan(&c, l, Router::Naive).unwrap();
        assert_eq!(p.stats.exchanges, 1);
        assert_eq!(p.stats.amps_moved_per_rank, 1 << 3);
        // logical 5 now at physical 3 (top local), logical 3 at physical 5
        assert_eq!(p.final_map[5], 3);
        assert_eq!(p.final_map[3], 5);
        let DistStep::Local(last) = p.steps.last().unwrap() else {
            panic!()
        };
        let Instruction::Gate(g) = &last[0] else {
            panic!()
        };
        assert_eq!(g.qubits.as_slice(), &[3]);
    }

    #[test]
    fn swap_is_free_relabel() {
        let mut c = Circuit::new(6, 0);
        c.swap(0, 5).unwrap();
        let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Naive).unwrap();
        assert_eq!(p.stats.exchanges, 0);
        assert_eq!(p.stats.relabels, 1);
        assert_eq!(p.final_map[0], 5);
        assert_eq!(p.final_map[5], 0);
    }

    #[test]
    fn exchange_does_not_evict_required_qubit() {
        // Iswap on logical (3, 5): 3 sits at top local (m-1 = 3), 5 is global.
        let mut c = Circuit::new(6, 0);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![3, 5]))
            .unwrap();
        let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Naive).unwrap();
        assert_eq!(p.stats.local_swaps, 1);
        assert_eq!(p.stats.exchanges, 1);
        assert!(p.final_map[3] < 4 && p.final_map[5] < 4);
    }

    #[test]
    fn too_few_local_qubits() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![0, 2]))
            .unwrap();
        let r = plan(&c, DistLayout::new(3, 2).unwrap(), Router::Naive);
        assert!(matches!(
            r,
            Err(DistError::TooFewLocalQubits { need: 2, m: 1 })
        ));
    }

    #[test]
    fn huge_slice_traffic_saturates_instead_of_overflowing() {
        // n=64, g=1 -> m=63: each exchange moves 2^62 amps; six of them overflow u64.
        let mut c = Circuit::new(64, 0);
        for _ in 0..3 {
            c.h(63).unwrap();
            c.h(62).unwrap();
        }
        let p = plan(&c, DistLayout::new(64, 1).unwrap(), Router::Naive).unwrap();
        assert!(p.stats.exchanges >= 4);
        assert_eq!(p.stats.amps_moved_per_rank, u64::MAX);
    }

    #[test]
    fn single_rank_is_one_local_step() {
        let mut c = Circuit::new(4, 0);
        c.h(3).unwrap();
        c.cnot(3, 0).unwrap();
        let p = plan(&c, DistLayout::new(4, 0).unwrap(), Router::Naive).unwrap();
        assert_eq!(p.steps.len(), 1);
        assert_eq!(p.stats.exchanges, 0);
    }

    #[test]
    fn measure_rejected_and_qubit_mismatch() {
        let mut c = Circuit::new(4, 1);
        c.h(0).unwrap();
        c.add_instruction(Instruction::Measure { qubit: 0, clbit: 0 })
            .unwrap();
        assert!(matches!(
            plan(&c, DistLayout::new(4, 1).unwrap(), Router::Naive),
            Err(DistError::Unsupported { kind: "measure" })
        ));
        assert!(matches!(
            plan(
                &Circuit::new(5, 0),
                DistLayout::new(4, 1).unwrap(),
                Router::Naive
            ),
            Err(DistError::QubitCountMismatch { .. })
        ));
    }
}
