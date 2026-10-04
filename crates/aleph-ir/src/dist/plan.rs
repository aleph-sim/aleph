//! Distributed planner: logical→physical tracking, exchange insertion.
//!
//! Naive router: one global qubit per exchange, on demand, swapped with the
//! top local physical slot `m-1` (so exchanged chunks are contiguous halves).

use aleph_core::{Gate, GateInstance};
use smallvec::{smallvec, SmallVec};

use super::next_use::NextUse;
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
pub(crate) struct Map {
    pub(crate) l2p: Vec<u32>,
    pub(crate) p2l: Vec<u32>,
}

impl Map {
    /// Map from `l2p[logical] = physical`; must be a permutation of `0..len`.
    pub(crate) fn from_l2p(l2p: &[u32]) -> Result<Self, DistError> {
        let n = l2p.len();
        let mut p2l = vec![u32::MAX; n];
        for (l, &p) in l2p.iter().enumerate() {
            let slot = p2l.get_mut(p as usize).ok_or(DistError::BadPlacement)?;
            if *slot != u32::MAX {
                return Err(DistError::BadPlacement);
            }
            *slot = l as u32;
        }
        Ok(Self {
            l2p: l2p.to_vec(),
            p2l,
        })
    }

    /// Swap the logical occupants of physical slots `a` and `b`.
    pub(crate) fn swap_phys(&mut self, a: u32, b: u32) {
        let (la, lb) = (self.p2l[a as usize], self.p2l[b as usize]);
        self.p2l.swap(a as usize, b as usize);
        self.l2p[la as usize] = b;
        self.l2p[lb as usize] = a;
    }

    pub(crate) fn remap_mask(&self, mask: u64) -> u64 {
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

/// Lower a logical `Gate` / `DiagonalPhase` onto physical qubits; `None` for
/// anything else (the callers handle barriers, relabels and errors).
pub(crate) fn to_physical(instr: &Instruction, map: &Map, n: u32) -> Option<Instruction> {
    match instr {
        Instruction::Gate(g) => Some(Instruction::Gate(GateInstance {
            gate: g.gate.clone(),
            qubits: g.qubits.iter().map(|&l| map.l2p[l as usize]).collect(),
            controls: g.controls.iter().map(|&l| map.l2p[l as usize]).collect(),
        })),
        Instruction::DiagonalPhase(dp) => {
            Some(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
                n_qubits: n,
                terms: dp
                    .terms
                    .iter()
                    .map(|t| PhaseTerm {
                        conds: t.conds.iter().map(|&c| map.remap_mask(c)).collect(),
                        angle: t.angle,
                    })
                    .collect(),
            })))
        }
        _ => None,
    }
}

/// Build a distributed plan for `circuit` under `layout`.
///
/// `Local` instructions in the result use physical qubit indices; run them
/// through [`super::specialize`] per rank.
pub fn plan(circuit: &Circuit, layout: DistLayout, router: Router) -> Result<DistPlan, DistError> {
    let id: Vec<u32> = (0..layout.n).collect();
    plan_from(circuit, layout, router, &id)
}

/// Like [`plan`], starting from `init[logical] = physical` instead of the
/// identity. Free, because every plan starts from |0…0⟩ (see [`DistPlan`]).
pub fn plan_from(
    circuit: &Circuit,
    layout: DistLayout,
    router: Router,
    init: &[u32],
) -> Result<DistPlan, DistError> {
    if circuit.num_qubits() != layout.n {
        return Err(DistError::QubitCountMismatch {
            circuit: circuit.num_qubits(),
            layout: layout.n,
        });
    }
    if init.len() != layout.n as usize {
        return Err(DistError::BadPlacement);
    }
    let map = Map::from_l2p(init)?;
    match router {
        Router::Reorder { max_k } => super::schedule::schedule(circuit, layout, max_k, map),
        Router::Naive | Router::Lookahead => plan_in_order(circuit, layout, router, map),
    }
}

fn plan_in_order(
    circuit: &Circuit,
    layout: DistLayout,
    router: Router,
    mut map: Map,
) -> Result<DistPlan, DistError> {
    let m = layout.m();
    let mut steps: Vec<DistStep> = Vec::new();
    let mut cur: Vec<Instruction> = Vec::new();
    let mut stats = CommStats::default();
    let mut next_use = match router {
        Router::Naive => None,
        Router::Lookahead => Some(NextUse::build(circuit)),
        Router::Reorder { .. } => None,
    };

    for (idx, instr) in circuit.instructions().iter().enumerate() {
        match instr {
            Instruction::Barrier(_) => {}
            Instruction::Measure { .. } => return Err(DistError::Unsupported { kind: "measure" }),
            Instruction::Reset(_) => return Err(DistError::Unsupported { kind: "reset" }),
            Instruction::TiledBlock(_) => {
                return Err(DistError::Unsupported {
                    kind: "tiled_block",
                })
            }
            Instruction::DiagonalPhase(_) => cur.extend(to_physical(instr, &map, layout.n)),
            Instruction::Gate(g) => {
                if matches!(g.gate, Gate::Swap) && g.controls.is_empty() {
                    let a = map.l2p[g.qubits[0] as usize];
                    let b = map.l2p[g.qubits[1] as usize];
                    map.swap_phys(a, b);
                    if let Some(nu) = next_use.as_mut() {
                        nu.relabel(g.qubits[0], g.qubits[1]);
                    }
                    stats.relabels += 1;
                    continue;
                }
                let req = required_local(g);
                if req.len() > m as usize {
                    return Err(DistError::TooFewLocalQubits { need: req.len(), m });
                }
                match next_use.as_mut() {
                    None => {
                        for &q in &req {
                            if layout.is_global(map.l2p[q as usize]) {
                                exchange_naive(
                                    q, &req, layout, &mut map, &mut cur, &mut steps, &mut stats,
                                )?;
                            }
                        }
                    }
                    Some(nu) => {
                        let missing: SmallVec<[u32; 4]> = req
                            .iter()
                            .copied()
                            .filter(|&q| layout.is_global(map.l2p[q as usize]))
                            .collect();
                        if !missing.is_empty() {
                            exchange_lookahead(
                                idx, &missing, &req, layout, nu, &mut map, &mut cur, &mut steps,
                                &mut stats,
                            )?;
                        }
                    }
                }
                cur.extend(to_physical(instr, &map, layout.n));
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

/// P6-02 naive exchange: bring logical `q` in by swapping it with the top
/// local slot, after moving a required occupant of that slot down.
#[allow(clippy::too_many_arguments)]
fn exchange_naive(
    q: u32,
    req: &[u32],
    layout: DistLayout,
    map: &mut Map,
    cur: &mut Vec<Instruction>,
    steps: &mut Vec<DistStep>,
    stats: &mut CommStats,
) -> Result<(), DistError> {
    let m = layout.m();
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
        steps.push(DistStep::Local(std::mem::take(cur)));
    }
    let gbit = map.l2p[q as usize];
    steps.push(DistStep::Exchange {
        global_bits: smallvec![gbit],
    });
    map.swap_phys(gbit, top);
    stats.exchanges += 1;
    // Saturate: n=64 with g=1 moves 2^62 amps per exchange.
    stats.amps_moved_per_rank = stats.amps_moved_per_rank.saturating_add(1u64 << (m - 1));
    Ok(())
}

/// P6-03 lookahead exchange: one k-bit exchange for all `missing` qubits plus
/// prefetches, evicting the local qubits whose next use is farthest away.
///
/// A k-bit exchange moves `(1 - 2^-k)` of a slice, vs `k/2` for k separate
/// single-bit exchanges, so batching pays whenever the prefetched qubit is
/// needed before the victim it displaces (Belady's farthest-next-use rule).
#[allow(clippy::too_many_arguments)]
fn exchange_lookahead(
    idx: usize,
    missing: &[u32],
    req: &[u32],
    layout: DistLayout,
    nu: &mut NextUse,
    map: &mut Map,
    cur: &mut Vec<Instruction>,
    steps: &mut Vec<DistStep>,
    stats: &mut CommStats,
) -> Result<(), DistError> {
    let m = layout.m();
    let after = idx + 1;
    // Victim candidates: local logical qubits the current gate does not need,
    // farthest next use first; ties prefer higher physical slots (already near
    // the top, so fewer local swaps).
    let locals: Vec<u32> = (0..m)
        .map(|p| map.p2l[p as usize])
        .filter(|l| !req.contains(l))
        .collect();
    let mut victims: Vec<(usize, u32, u32)> = locals
        .into_iter()
        .map(|l| (nu.next(l, after), map.l2p[l as usize], l))
        .collect();
    victims.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    if victims.len() < missing.len() {
        return Err(DistError::TooFewLocalQubits { need: req.len(), m });
    }
    let mut bring: SmallVec<[u32; 4]> = missing.iter().copied().collect();
    // Prefetch: other global qubits, soonest next use first, while each is
    // needed before the victim it would displace.
    let globals: Vec<u32> = (m..layout.n)
        .map(|p| map.p2l[p as usize])
        .filter(|l| !bring.contains(l))
        .collect();
    let mut cands: Vec<(usize, u32)> = globals
        .into_iter()
        .map(|l| (nu.next(l, after), l))
        .filter(|&(t, _)| t != usize::MAX)
        .collect();
    cands.sort_unstable();
    for (t, l) in cands {
        if bring.len() >= layout.g as usize || bring.len() >= victims.len() {
            break;
        }
        if t < victims[bring.len()].0 {
            bring.push(l);
        } else {
            break;
        }
    }
    let chosen: SmallVec<[u32; 4]> = victims[..bring.len()].iter().map(|v| v.2).collect();
    emit_exchange(&bring, &chosen, layout, map, cur, steps, stats)
}

/// Bring logical `bring` in by evicting logical `victims` (same length): park
/// the victims in the top `k` local slots with local `Swap`s, then swap those
/// slots with the bring-in qubits' global bits — the contiguity contract of
/// [`DistStep::Exchange`].
pub(crate) fn emit_exchange(
    bring: &[u32],
    victims: &[u32],
    layout: DistLayout,
    map: &mut Map,
    cur: &mut Vec<Instruction>,
    steps: &mut Vec<DistStep>,
    stats: &mut CommStats,
) -> Result<(), DistError> {
    let m = layout.m();
    let k = bring.len() as u32;
    if k == 0 || victims.len() != bring.len() || k > m {
        return Err(DistError::Unsupported {
            kind: "internal: bad exchange shape",
        });
    }
    let top_lo = m - k;
    let mut free_slots: SmallVec<[u32; 4]> = (top_lo..m)
        .filter(|&p| !victims.contains(&map.p2l[p as usize]))
        .collect();
    for &v in victims {
        let vp = map.l2p[v as usize];
        if vp >= top_lo {
            continue;
        }
        let Some(slot) = free_slots.pop() else {
            return Err(DistError::Unsupported {
                kind: "internal: no free top slot",
            });
        };
        cur.push(Instruction::Gate(GateInstance::new(
            Gate::Swap,
            vec![slot, vp],
        )));
        map.swap_phys(slot, vp);
        stats.local_swaps += 1;
    }
    if !cur.is_empty() {
        steps.push(DistStep::Local(std::mem::take(cur)));
    }
    let global_bits: SmallVec<[u32; 4]> = bring.iter().map(|&l| map.l2p[l as usize]).collect();
    for (j, &gb) in global_bits.iter().enumerate() {
        map.swap_phys(gb, top_lo + j as u32);
    }
    steps.push(DistStep::Exchange { global_bits });
    stats.exchanges += 1;
    let moved = ((1u64 << k) - 1) << (m - k);
    stats.amps_moved_per_rank = stats.amps_moved_per_rank.saturating_add(moved);
    Ok(())
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

    fn brick(n: u32, depth: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for d in 0..depth {
            for q in 0..n {
                c.rx(0.3 + f64::from(q), q).unwrap();
            }
            let mut q = (d % 2) as u32;
            while q + 1 < n {
                c.cnot(q, q + 1).unwrap();
                q += 2;
            }
        }
        c
    }

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
                c.add_gate(GateInstance::controlled(
                    Gate::Phase(Param::Concrete(0.1 * f64::from(j - k))),
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

    fn is_iswap(i: &Instruction) -> bool {
        matches!(i, Instruction::Gate(g) if matches!(g.gate, Gate::Iswap))
    }

    #[test]
    fn lookahead_batches_exchanges_on_brickwall() {
        let c = brick(16, 12);
        let l = DistLayout::new(16, 2).unwrap();
        let naive = plan(&c, l, Router::Naive).unwrap().stats;
        let la = plan(&c, l, Router::Lookahead).unwrap();
        // every exchange is k <= g bits wide and uses distinct global bits
        for s in &la.steps {
            if let DistStep::Exchange { global_bits } = s {
                assert!(!global_bits.is_empty() && global_bits.len() <= 2);
                assert!(global_bits.iter().all(|&b| b >= l.m()));
                let mut v = global_bits.to_vec();
                v.sort_unstable();
                v.dedup();
                assert_eq!(v.len(), global_bits.len());
            }
        }
        assert!(
            la.stats.amps_moved_per_rank * 2 <= naive.amps_moved_per_rank,
            "lookahead {:?} vs naive {:?}",
            la.stats,
            naive
        );
    }

    #[test]
    fn lookahead_no_worse_on_ghz_qft() {
        for c in [ghz(16), qft(16)] {
            for g in [1u32, 2, 3] {
                let l = DistLayout::new(16, g).unwrap();
                let n = plan(&c, l, Router::Naive).unwrap().stats;
                let a = plan(&c, l, Router::Lookahead).unwrap().stats;
                assert!(
                    a.amps_moved_per_rank <= n.amps_moved_per_rank,
                    "g={g}: {a:?} vs {n:?}"
                );
            }
        }
    }

    #[test]
    fn lookahead_never_evicts_required() {
        // Iswap on logical (3, 5) at n=6, g=2 (m=4): 3 sits in a top slot, 5 is
        // global, and the later H(4) makes a k=2 prefetch worthwhile, so the
        // top-2 slots must be cleared without evicting 3.
        let mut c = Circuit::new(6, 0);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![3, 5]))
            .unwrap();
        c.h(4).unwrap();
        let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Lookahead).unwrap();
        assert!(p.stats.local_swaps >= 1, "{:?}", p.stats);
        let first = p
            .steps
            .iter()
            .find_map(|s| match s {
                DistStep::Exchange { global_bits } => Some(global_bits.len()),
                _ => None,
            })
            .unwrap();
        assert_eq!(first, 2, "expected a k=2 prefetching exchange");
        let isw = p
            .steps
            .iter()
            .filter_map(|s| match s {
                DistStep::Local(v) => v.iter().find(|i| is_iswap(i)),
                _ => None,
            })
            .next()
            .unwrap();
        let Instruction::Gate(isw) = isw else {
            panic!()
        };
        assert!(isw.qubits.iter().all(|&p| p < 4), "{:?}", isw.qubits);
    }

    #[test]
    fn lookahead_tight_m_caps_prefetch() {
        // n=4, g=2, m=2: an Iswap needs both local slots, so no victim is free
        // for a prefetch even though H(3) follows; k must be exactly 1.
        let mut c = Circuit::new(4, 0);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![0, 2]))
            .unwrap();
        c.h(3).unwrap();
        let p = plan(&c, DistLayout::new(4, 2).unwrap(), Router::Lookahead).unwrap();
        let ks: Vec<usize> = p
            .steps
            .iter()
            .filter_map(|s| match s {
                DistStep::Exchange { global_bits } => Some(global_bits.len()),
                _ => None,
            })
            .collect();
        assert_eq!(ks.first(), Some(&1), "{ks:?}");
    }

    /// Deterministic random circuits mixing relabel `Swap`s with exchange-forcing gates.
    fn rand_circ(n: u32, len: usize, seed: u64) -> Circuit {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s >> 33
        };
        let mut c = Circuit::new(n, 0);
        for _ in 0..len {
            let a = (next() % u64::from(n)) as u32;
            let mut b = (next() % u64::from(n)) as u32;
            if b == a {
                b = (a + 1) % n;
            }
            match next() % 7 {
                0 => drop(c.h(a).unwrap()),
                1 => drop(c.rx(0.3, a).unwrap()),
                2 => drop(c.cnot(a, b).unwrap()),
                3 => drop(
                    c.add_gate(GateInstance::new(Gate::Iswap, vec![a, b]))
                        .unwrap(),
                ),
                4 => drop(c.swap(a, b).unwrap()),
                5 => drop(c.rz(0.7, a).unwrap()),
                _ => drop(c.cz(a, b).unwrap()),
            }
        }
        c
    }

    #[test]
    fn lookahead_bounded_regression_with_relabels() {
        // Next use must follow *data* across relabel Swaps; keyed by label it
        // mispredicts and lookahead moved up to 4.5x more than naive here.
        let mut worst = 0.0f64;
        for n in [6u32, 8, 10, 12] {
            for g in 1..=3u32 {
                for seed in 0..100u64 {
                    let c = rand_circ(n, 30, seed * 7919 + u64::from(n));
                    let l = DistLayout::new(n, g).unwrap();
                    let a = plan(&c, l, Router::Lookahead).unwrap().stats;
                    let b = plan(&c, l, Router::Naive).unwrap().stats;
                    if b.amps_moved_per_rank > 0 {
                        worst =
                            worst.max(a.amps_moved_per_rank as f64 / b.amps_moved_per_rank as f64);
                    }
                }
            }
        }
        assert!(worst <= 1.3, "worst lookahead/naive traffic ratio {worst}");
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

    #[test]
    fn plan_from_identity_equals_plan() {
        let c = brick(8, 6);
        let l = DistLayout::new(8, 2).unwrap();
        let id: Vec<u32> = (0..8).collect();
        for r in [Router::Naive, Router::Lookahead] {
            let a = plan(&c, l, r).unwrap();
            let b = plan_from(&c, l, r, &id).unwrap();
            assert_eq!(a.stats, b.stats);
            assert_eq!(a.final_map, b.final_map);
            assert_eq!(a.steps.len(), b.steps.len());
        }
    }

    #[test]
    fn plan_from_rejects_non_permutations() {
        let c = brick(4, 1);
        let l = DistLayout::new(4, 1).unwrap();
        for bad in [vec![0u32, 1, 2], vec![0, 1, 2, 2], vec![0, 1, 2, 4]] {
            assert_eq!(
                plan_from(&c, l, Router::Naive, &bad).unwrap_err(),
                DistError::BadPlacement
            );
        }
    }

    #[test]
    fn plan_from_starts_at_the_given_map() {
        // logical 0 starts global (physical 3): an H on it needs one exchange.
        let mut c = Circuit::new(4, 0);
        c.h(0).unwrap();
        let l = DistLayout::new(4, 1).unwrap();
        let p = plan_from(&c, l, Router::Naive, &[3, 0, 1, 2]).unwrap();
        assert_eq!(p.stats.exchanges, 1);
        assert!(p.final_map[0] < 3);
    }
}
