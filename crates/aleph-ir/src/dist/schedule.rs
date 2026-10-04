//! P6-05 reordering list scheduler. Runs every ready instruction whose
//! required-local qubits are local (lowest original index first, which keeps
//! neighbours together for fusion). When every ready gate is blocked, it
//! brings in the lowest-index blocked gate's missing qubits, plus prefetches
//! by P6-03's rule (needed before the victim it displaces), and evicts by
//! farthest next local need (Belady). Same `DistPlan` contract as `plan.rs`.

use std::collections::BTreeSet;

use aleph_core::Gate;
use smallvec::SmallVec;

use super::dag::Dag;
use super::next_use::NextUse;
use super::plan::{emit_exchange, required_local, to_physical, Map};
use super::{CommStats, DistError, DistLayout, DistPlan, DistStep};
use crate::{Circuit, Instruction};

pub(crate) fn schedule(
    circuit: &Circuit,
    layout: DistLayout,
    max_k: u32,
    mut map: Map,
) -> Result<DistPlan, DistError> {
    let m = layout.m();
    let instrs = circuit.instructions();
    let mut dag = Dag::build(circuit)?;
    let reqs: Vec<SmallVec<[u32; 4]>> = instrs
        .iter()
        .map(|i| match i {
            Instruction::Gate(g) if !is_relabel(g) => required_local(g),
            _ => SmallVec::new(),
        })
        .collect();
    if let Some(r) = reqs.iter().find(|r| r.len() > m as usize) {
        return Err(DistError::TooFewLocalQubits { need: r.len(), m });
    }
    let mut nu = NextUse::build(circuit);
    let mut done = vec![false; instrs.len()];
    let mut ready: BTreeSet<usize> = dag.initial_ready().into_iter().collect();
    let mut fresh: Vec<usize> = Vec::new();
    let mut steps: Vec<DistStep> = Vec::new();
    let mut cur: Vec<Instruction> = Vec::new();
    let mut stats = CommStats::default();
    let max_k = max_k.clamp(1, layout.g.max(1)) as usize;
    // Prefetch window over *original* indices [seed, seed + 4n): it may hold
    // fewer than 4n still-unscheduled instructions.
    let horizon = 4 * layout.n as usize;

    while let Some(&seed) = ready.iter().next() {
        let runnable = ready.iter().copied().find(|&i| {
            reqs[i]
                .iter()
                .all(|&q| !layout.is_global(map.l2p[q as usize]))
        });
        if let Some(i) = runnable {
            ready.remove(&i);
            emit(&instrs[i], layout, &mut map, &mut nu, &mut cur, &mut stats);
            done[i] = true;
            dag.complete(i, &mut fresh)?;
            ready.extend(fresh.drain(..));
            continue;
        }
        // Every ready gate is blocked: one exchange for the lowest-index one.
        // Progress: afterwards seed's missing qubits are local and none of its
        // required qubits were victims, so the next iteration finds seed
        // runnable and the loop terminates.
        let need = &reqs[seed];
        let missing: SmallVec<[u32; 4]> = need
            .iter()
            .copied()
            .filter(|&q| layout.is_global(map.l2p[q as usize]))
            .collect();
        let mut victims: Vec<(usize, u32, u32)> = (0..m)
            .map(|p| map.p2l[p as usize])
            .filter(|l| !need.contains(l))
            .map(|l| (nu.next_unscheduled(l, &done), map.l2p[l as usize], l))
            .collect();
        // Farthest next need first; ties prefer higher physical slots.
        victims.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
        if victims.len() < missing.len() {
            return Err(DistError::TooFewLocalQubits {
                need: need.len(),
                m,
            });
        }
        let mut bring = missing;
        let limit = seed.saturating_add(horizon);
        let mut cands: Vec<(usize, u32)> = (m..layout.n)
            .map(|p| map.p2l[p as usize])
            .filter(|l| !bring.contains(l))
            .map(|l| (nu.next_unscheduled(l, &done), l))
            .filter(|&(t, _)| t < limit)
            .collect();
        cands.sort_unstable();
        for (t, l) in cands {
            if bring.len() >= max_k || bring.len() >= victims.len() {
                break;
            }
            if t < victims[bring.len()].0 {
                bring.push(l);
            } else {
                break;
            }
        }
        let chosen: SmallVec<[u32; 4]> = victims[..bring.len()].iter().map(|v| v.2).collect();
        emit_exchange(
            &bring, &chosen, layout, &mut map, &mut cur, &mut steps, &mut stats,
        )?;
    }
    if done.iter().any(|d| !d) {
        return Err(DistError::Unsupported {
            kind: "internal: DAG left instructions unscheduled",
        });
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

fn is_relabel(g: &aleph_core::GateInstance) -> bool {
    matches!(g.gate, Gate::Swap) && g.controls.is_empty()
}

/// Emit one runnable instruction: relabel `Swap`s update the map, barriers
/// vanish, everything else is lowered to physical qubits.
fn emit(
    instr: &Instruction,
    layout: DistLayout,
    map: &mut Map,
    nu: &mut NextUse,
    cur: &mut Vec<Instruction>,
    stats: &mut CommStats,
) {
    if let Instruction::Gate(g) = instr {
        if is_relabel(g) {
            let (a, b) = (map.l2p[g.qubits[0] as usize], map.l2p[g.qubits[1] as usize]);
            map.swap_phys(a, b);
            nu.relabel(g.qubits[0], g.qubits[1]);
            stats.relabels += 1;
            return;
        }
    }
    cur.extend(to_physical(instr, map, layout.n));
}

#[cfg(test)]
mod tests {
    use crate::dist::{plan, specialize, DistLayout, DistPlan, DistStep, Router};
    use crate::{Circuit, Instruction};
    use aleph_core::{Gate, GateInstance, Param};

    fn ghz(n: u32) -> Circuit {
        let mut c = Circuit::new(n, 0);
        c.h(0).unwrap();
        for q in 0..n - 1 {
            c.cnot(q, q + 1).unwrap();
        }
        c
    }

    fn qft(n: u32) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for j in (0..n).rev() {
            c.h(j).unwrap();
            for k in (0..j).rev() {
                let th = std::f64::consts::PI / f64::from(1u32 << (j - k));
                c.add_gate(GateInstance::controlled(
                    Gate::Phase(Param::Concrete(th)),
                    vec![j],
                    vec![k],
                ))
                .unwrap();
            }
        }
        for q in 0..n / 2 {
            c.swap(q, n - 1 - q).unwrap();
        }
        c
    }

    fn brick(n: u32, depth: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for d in 0..depth {
            for q in 0..n {
                c.rx(0.3 + f64::from(q), q).unwrap();
                c.rz(0.7 * d as f64, q).unwrap();
            }
            let mut q = (d % 2) as u32;
            while q + 1 < n {
                c.cnot(q, q + 1).unwrap();
                q += 2;
            }
        }
        c
    }

    /// Spec §7 plan invariants.
    fn check_invariants(c: &Circuit, p: &DistPlan) {
        let orig_gates = c
            .instructions()
            .iter()
            .filter(|i| match i {
                Instruction::Gate(g) => !(matches!(g.gate, Gate::Swap) && g.controls.is_empty()),
                Instruction::DiagonalPhase(_) => true,
                _ => false,
            })
            .count();
        let mut emitted = 0usize;
        for s in &p.steps {
            if let DistStep::Local(v) = s {
                for i in v {
                    emitted += 1;
                    for r in 0..p.layout.ranks() {
                        specialize(i, p.layout, r).unwrap();
                    }
                }
            }
        }
        assert_eq!(emitted, orig_gates + p.stats.local_swaps as usize);
        let mut seen = vec![false; p.final_map.len()];
        for &q in &p.final_map {
            assert!(!std::mem::replace(&mut seen[q as usize], true));
        }
    }

    fn exch(p: &DistPlan) -> u32 {
        p.stats.exchanges
    }

    #[test]
    fn local_only_circuit_has_no_exchange() {
        let mut c = Circuit::new(6, 0);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.rz(0.3, 5).unwrap();
        c.cz(2, 5).unwrap();
        let p = plan(
            &c,
            DistLayout::new(6, 2).unwrap(),
            Router::Reorder { max_k: 2 },
        )
        .unwrap();
        assert_eq!(exch(&p), 0);
        check_invariants(&c, &p);
    }

    #[test]
    fn invariants_on_standard_circuits() {
        for (c, g) in [
            (ghz(10), 2),
            (qft(10), 3),
            (brick(10, 6), 2),
            (brick(9, 4), 3),
        ] {
            for k in 1..=3 {
                let p = plan(
                    &c,
                    DistLayout::new(c.num_qubits(), g).unwrap(),
                    Router::Reorder { max_k: k },
                )
                .unwrap();
                check_invariants(&c, &p);
            }
        }
    }

    #[test]
    fn never_more_exchanges_than_lookahead_on_ghz_qft() {
        for c in [ghz(12), qft(12)] {
            for g in 1..=3 {
                let l = DistLayout::new(12, g).unwrap();
                let la = plan(&c, l, Router::Lookahead).unwrap();
                let re = plan(&c, l, Router::Reorder { max_k: g }).unwrap();
                assert!(
                    exch(&re) <= exch(&la),
                    "g={g}: {} > {}",
                    exch(&re),
                    exch(&la)
                );
            }
        }
    }

    #[test]
    fn missing_exceeding_max_k_still_exchanges() {
        // Iswap on logical 4,5: both global at n=6, g=2. One gate needs 2 bits.
        let mut c = Circuit::new(6, 0);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![4, 5]))
            .unwrap();
        let p = plan(
            &c,
            DistLayout::new(6, 2).unwrap(),
            Router::Reorder { max_k: 1 },
        )
        .unwrap();
        assert_eq!(exch(&p), 1);
        let bits = p
            .steps
            .iter()
            .find_map(|s| match s {
                DistStep::Exchange { global_bits } => Some(global_bits.len()),
                _ => None,
            })
            .unwrap();
        assert_eq!(bits, 2);
        check_invariants(&c, &p);
    }

    #[test]
    fn g0_never_exchanges() {
        let c = brick(6, 3);
        for k in [0u32, 1, 5] {
            let p = plan(
                &c,
                DistLayout::new(6, 0).unwrap(),
                Router::Reorder { max_k: k },
            )
            .unwrap();
            assert_eq!(exch(&p), 0);
            check_invariants(&c, &p);
        }
    }

    #[test]
    fn without_barrier_local_work_is_hoisted() {
        // n=4, g=1: logical 3 is global. rx(3) is blocked, h(0) runs first.
        let mut c = Circuit::new(4, 0);
        c.rx(0.3, 3).unwrap();
        c.h(0).unwrap();
        let p = plan(
            &c,
            DistLayout::new(4, 1).unwrap(),
            Router::Reorder { max_k: 1 },
        )
        .unwrap();
        assert!(matches!(p.steps[0], DistStep::Local(ref v) if v.len() == 1));
        assert!(matches!(p.steps[1], DistStep::Exchange { .. }));
    }

    #[test]
    fn barrier_fences_reordering() {
        let mut c = Circuit::new(4, 0);
        c.rx(0.3, 3).unwrap();
        c.barrier([0u32, 1, 2, 3]).unwrap();
        c.h(0).unwrap();
        let p = plan(
            &c,
            DistLayout::new(4, 1).unwrap(),
            Router::Reorder { max_k: 1 },
        )
        .unwrap();
        assert!(matches!(p.steps[0], DistStep::Exchange { .. }));
    }

    #[test]
    fn measure_is_rejected_not_panicking() {
        let mut c = Circuit::new(4, 1);
        c.h(0).unwrap();
        c.measure(0, 0).unwrap();
        assert!(plan(
            &c,
            DistLayout::new(4, 1).unwrap(),
            Router::Reorder { max_k: 1 }
        )
        .is_err());
    }
}
