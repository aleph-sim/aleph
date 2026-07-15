//! Serial-gather conflict-free slot solver (M9c gather-crossbar fix, step 3.1).
//!
//! The streaming relay-BP core stores each message-passing edge in one of `P` physical
//! BRAM banks and gathers `P` messages per cycle over `STEPS` cycles into a row buffer.
//! No two edges of the same group may ever be read from the same physical bank in the same
//! step — otherwise two edges would need to read the same single-port BRAM in the same cycle,
//! which is undefined.
//!
//! This module is the pure-Rust solver that makes that true: [`plan_serial`] assigns each
//! edge a FIXED `(physical bank, intra-bank address)` storage slot (consistent across every
//! group it appears in) by folding the caller's logical banks onto `p` physical banks, and
//! then computes a per-group conflict-free `(step, bank)` read schedule. [`verify_layout`] is
//! the independent checker (also used as the emitter's gen-time guard) that the M9c crossbar
//! bug — a naive schedule that ignores real bank identity — cannot sneak back in.
//!
//! ## Residual-mux design (2026-07-15 correction)
//!
//! An earlier version of this solver forced `STEPS = ceil(max_group_size / p)` (the *ideal*,
//! perfectly-balanced step count) and PANICKED if a real group's edges concentrated more
//! heavily than that in one physical bank. Real data does this: a 164-edge group at `p=8` can
//! put more than `ceil(164/8)=21` edges in a single physical bank, because the *fold* that
//! assigns edges to physical banks is shared across every group an edge appears in (each edge
//! has one fixed storage slot) and cannot simultaneously balance every group.
//!
//! The corrected design accepts a small **residual output mux** downstream (a `STEPS`-way
//! select per tap, ROM-fed, in the RTL) and makes the solver **feasible-always**: `STEPS` is
//! *data-dependent*, defined as the true bottleneck — the max, over every group and every
//! physical bank, of how many of that group's edges live in that bank. That is always
//! achievable (a bank with `K` edges in a group just takes `K` serial steps to drain), so
//! [`plan_serial`] never panics on occupancy; the only remaining panic is the genuine
//! precondition failure `p == 0`. The balanced (rank-based) folding is kept because it still
//! minimizes `STEPS` (hence throughput) when the input happens to be balanced — but
//! correctness no longer depends on balance.
//!
//! Design context: `docs/superpowers/specs/2026-07-15-m9c-serial-gather-design.md` (see the
//! "DESIGN CORRECTION" section).

use std::collections::{HashMap, HashSet};

/// Fixed BRAM storage + per-group conflict-free read schedule for the serial-gather core.
#[derive(Debug)]
pub struct SerialLayout {
    /// Number of physical BRAM banks / read ports.
    pub p: usize,
    /// Number of read steps: the data-dependent bottleneck `max over groups g of (max over
    /// physical banks b of |{edges in group g stored in bank b}|)`. NOT a fixed `ceil(N/p)` —
    /// see the module-level "Residual-mux design" note. Always feasible; never causes a panic.
    pub steps: usize,
    /// `bank_of[e]` = edge `e`'s fixed physical BRAM bank (`0..p`).
    pub bank_of: Vec<usize>,
    /// `addr_of[e]` = edge `e`'s intra-bank address (unique within its physical bank).
    pub addr_of: Vec<usize>,
    /// `sched[g][t]` = the `(step, bank)` slot for group `g`'s `t`-th tap (in tap order), where
    /// `bank` is that tap's edge's fixed storage bank (`bank_of[edge]`) and `step` is that
    /// edge's 0-based occurrence index among group `g`'s taps that share `bank` (so within a
    /// group, taps sharing a bank get steps `0, 1, 2, …` in tap order). The RTL residual-mux
    /// select ROM is fed directly by `step` at the tap's fixed bank column.
    pub sched: Vec<Vec<(usize, usize)>>,
}

/// Ideal *lower-bound* step count for a perfectly balanced group of `max_group` edges spread
/// over `p` parallel ports: `ceil(max_group / p)`. This is NOT the authoritative step count —
/// real folds are not always balanced (an edge's storage bank is fixed across every group it
/// appears in, so no single fold can balance every group simultaneously). The authoritative,
/// data-dependent step count is [`SerialLayout::steps`], produced by [`plan_serial`]. This
/// function is kept as a reference/lower-bound helper (e.g. for reporting how far a real
/// layout's `steps` is above the ideal).
pub fn serial_steps(max_group: usize, p: usize) -> usize {
    assert!(p > 0, "serial_gather: p must be positive");
    max_group.div_ceil(p)
}

/// Assign fixed BRAM storage and a conflict-free per-group read schedule.
///
/// `edges[e] = (logical_bank, row)`; `groups[g]` = the edge ids read together in group `g`,
/// in tap order. Deterministic (sorts by edge id; no RNG). **Feasible-always**: `layout.steps`
/// is derived from the actual data (see [`SerialLayout::steps`]), so this never panics on
/// occupancy, however imbalanced the input. The only panic is the genuine precondition failure
/// `p == 0`.
pub fn plan_serial(edges: &[(usize, usize)], groups: &[Vec<usize>], p: usize) -> SerialLayout {
    assert!(p > 0, "serial_gather: p must be positive");

    // --- Storage folding: logical banks -> p physical banks, balanced round-robin by RANK
    // (the sorted position of a logical bank id among the distinct ids that actually appear),
    // not the raw id value. Using rank rather than `logical_bank % p` means sparse/clustered
    // logical-bank numbering (e.g. every id a multiple of p) still spreads evenly across
    // physical banks instead of aliasing them all onto bank 0 — see the
    // `sparse_logical_bank_ids_fold_across_banks` test for why raw-id folding is not safe to
    // assume (that test uses logical banks that are all multiples of p, where raw `lb % p`
    // collapses everything onto bank 0 but rank-based folding spreads them). This still helps
    // minimize `steps` when the input is reasonably balanced, but — per the module-level
    // "Residual-mux design" note — correctness no longer depends on it: however skewed a
    // group's occupancy ends up per bank, `plan_serial` still produces a valid schedule.
    let mut logical_banks: Vec<usize> = edges.iter().map(|&(lb, _)| lb).collect();
    logical_banks.sort_unstable();
    logical_banks.dedup();
    let rank_of: HashMap<usize, usize> = logical_banks
        .iter()
        .enumerate()
        .map(|(rank, &lb)| (lb, rank))
        .collect();

    // addr_of: a running per-physical-bank offset assigned in ascending edge-id order, so two
    // distinct edges can never alias the same (bank, addr) slot (requirement (b)).
    let mut bank_of = vec![0usize; edges.len()];
    let mut addr_of = vec![0usize; edges.len()];
    let mut next_addr = vec![0usize; p];
    // `_` = the edge's `row`: unused here because `addr_of` is a fresh running per-bank
    // counter assigned by this loop, not the caller-supplied row value.
    for (e, &(lb, _)) in edges.iter().enumerate() {
        let bank = rank_of[&lb] % p;
        bank_of[e] = bank;
        addr_of[e] = next_addr[bank];
        next_addr[bank] += 1;
    }

    // --- Schedule: per group, per-bank sequential steps; STEPS is the max across every group
    // and bank (the true, data-dependent bottleneck — see module docs).
    let mut sched: Vec<Vec<(usize, usize)>> = Vec::with_capacity(groups.len());
    let mut steps = 0usize;
    for group in groups {
        let (group_sched, group_max) = schedule_group(group, &bank_of);
        steps = steps.max(group_max);
        sched.push(group_sched);
    }

    SerialLayout {
        p,
        steps,
        bank_of,
        addr_of,
        sched,
    }
}

/// Compute one group's serial-gather schedule. For each tap, in tap order, pair it with its
/// edge's fixed physical bank and that tap's 0-based occurrence index among the group's taps
/// that share the bank so far (so a bank read `K` times in this group gets steps `0..K` in tap
/// order — always well-defined, never infeasible). Returns `(sched, group_max_per_bank)`,
/// where `group_max_per_bank` is this group's contribution to the layout-wide `STEPS`
/// bottleneck (`max` over all groups is taken by the caller).
fn schedule_group(group: &[usize], bank_of: &[usize]) -> (Vec<(usize, usize)>, usize) {
    let mut next_step: HashMap<usize, usize> = HashMap::new();
    let mut sched = Vec::with_capacity(group.len());
    let mut group_max = 0usize;
    for &e in group {
        let bank = bank_of[e];
        let step = *next_step.get(&bank).unwrap_or(&0);
        next_step.insert(bank, step + 1);
        sched.push((step, bank));
        group_max = group_max.max(step + 1);
    }
    (sched, group_max)
}

/// Check a [`SerialLayout`] against `groups`:
/// (a) within each group, all `(step, bank)` slots are distinct;
/// (b) `step < layout.steps` and `bank < p` for every slot (and `bank` matches the tap's
///     edge's real storage bank);
/// (c) `layout.steps` equals the max-per-bank bottleneck, re-derived independently from
///     `groups` + `layout.bank_of` (NOT the old `ceil(max_group/p)` formula — see module docs);
/// (d) no two distinct edges alias the same `(bank_of, addr_of)` storage slot;
/// (e) tap order is preserved (`sched[g]` parallels `groups[g]` index-for-index).
pub fn verify_layout(layout: &SerialLayout, groups: &[Vec<usize>]) -> Result<(), String> {
    if layout.sched.len() != groups.len() {
        return Err(format!(
            "sched has {} groups, expected {}",
            layout.sched.len(),
            groups.len()
        ));
    }

    // (c) steps == max over groups of max-per-bank occupancy, re-derived independently of
    // plan_serial's own bookkeeping (using only layout.bank_of + groups).
    let mut expected_steps = 0usize;
    for group in groups {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &e in group {
            if e >= layout.bank_of.len() {
                return Err(format!("edge {e} has no storage slot"));
            }
            *counts.entry(layout.bank_of[e]).or_insert(0) += 1;
        }
        if let Some(&m) = counts.values().max() {
            expected_steps = expected_steps.max(m);
        }
    }
    if layout.steps != expected_steps {
        return Err(format!(
            "steps {} != max-per-bank bottleneck {expected_steps}",
            layout.steps
        ));
    }

    for (g, group) in groups.iter().enumerate() {
        let sched_g = &layout.sched[g];
        // (e) tap order preserved: sched[g] must have one entry per tap, index-aligned with
        // groups[g] (checked below by zipping in order).
        if sched_g.len() != group.len() {
            return Err(format!(
                "group {g}: sched has {} taps, expected {}",
                sched_g.len(),
                group.len()
            ));
        }

        let mut seen_slots: HashSet<(usize, usize)> = HashSet::new();
        for (t, (&edge, &(step, bank))) in group.iter().zip(sched_g.iter()).enumerate() {
            if edge >= layout.bank_of.len() {
                return Err(format!(
                    "group {g} tap {t}: edge {edge} has no storage slot"
                ));
            }
            if step >= layout.steps {
                return Err(format!(
                    "group {g} tap {t}: step {step} >= layout.steps {}",
                    layout.steps
                ));
            }
            if bank >= layout.p {
                return Err(format!("group {g} tap {t}: bank {bank} >= p {}", layout.p));
            }
            let real_bank = layout.bank_of[edge];
            if bank != real_bank {
                return Err(format!(
                    "group {g} tap {t}: sched bank {bank} != edge {edge}'s storage bank \
                     {real_bank}"
                ));
            }
            // (a) distinct (step, bank) within the group == at most one read per bank per step.
            if !seen_slots.insert((step, bank)) {
                return Err(format!(
                    "group {g} tap {t}: duplicate (step,bank) ({step},{bank}) within group"
                ));
            }
        }
    }

    // (d) storage: no two distinct edges share (bank_of, addr_of).
    let mut seen_storage: HashMap<(usize, usize), usize> = HashMap::new();
    for e in 0..layout.bank_of.len() {
        let key = (layout.bank_of[e], layout.addr_of[e]);
        if let Some(&other) = seen_storage.get(&key) {
            return Err(format!(
                "edges {other} and {e} alias storage slot (bank {}, addr {})",
                key.0, key.1
            ));
        }
        seen_storage.insert(key, e);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_steps_ceils() {
        assert_eq!(serial_steps(0, 2), 0);
        assert_eq!(serial_steps(1, 2), 1);
        assert_eq!(serial_steps(4, 2), 2);
        assert_eq!(serial_steps(5, 2), 3);
        assert_eq!(serial_steps(7, 3), 3);
    }

    /// 12 edges, 6 logical banks (2 edges each), 3 groups of 4 edges, p=2. The small
    /// synthetic sanity case from the task brief. Balanced input, so `layout.steps` equals the
    /// ideal `serial_steps` lower bound here.
    #[test]
    fn small_synthetic_layout_verifies() {
        let edges: Vec<(usize, usize)> = vec![
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
        ];
        let groups = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
        let p = 2;
        let layout = plan_serial(&edges, &groups, p);
        assert_eq!(layout.p, p);
        assert_eq!(layout.steps, serial_steps(4, p));
        verify_layout(&layout, &groups).expect("layout must verify");
    }

    /// A group whose edges sit on alternating physical banks (0,0,1,1 in tap order). A *naive*
    /// scheduler that assigns `(step, bank) = (t/p, bank_of[edge])` purely from tap position —
    /// ignoring per-bank occupancy — puts the two bank-0 edges (taps 0 and 1) in the same step:
    /// a genuine hardware conflict (two reads from the same single-port BRAM in one cycle).
    /// `plan_serial`'s per-bank-sequential schedule must not do this; `verify_layout` must
    /// reject the naive one.
    #[test]
    fn naive_round_robin_would_collide() {
        let edges: Vec<(usize, usize)> = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
        let groups = vec![vec![0, 2, 1, 3]]; // tap order: bank0, bank0, bank1, bank1
        let p = 2;

        let layout = plan_serial(&edges, &groups, p);
        assert_eq!(layout.bank_of, vec![0, 1, 0, 1]); // edge id % 2 (contiguous ids => rank=id)
        verify_layout(&layout, &groups).expect("solver-produced schedule must be conflict-free");
        // Per-bank sequential steps: bank-0 taps (edges 0, 2) get steps 0, 1; bank-1 taps
        // (edges 1, 3) get steps 0, 1 — in tap order [0, 2, 1, 3].
        assert_eq!(layout.sched[0], vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        assert_eq!(layout.steps, 2);

        // Hand-roll a naive POSITIONAL schedule (step = tap_index / p, bank = the edge's real
        // storage bank) against the same real storage assignment, and confirm verify_layout
        // rejects it: taps 0 and 1 (edges 0 and 2) both live in bank 0 but the naive rule packs
        // them into the same step.
        let naive = SerialLayout {
            p: layout.p,
            steps: layout.steps,
            bank_of: layout.bank_of.clone(),
            addr_of: layout.addr_of.clone(),
            sched: vec![vec![(0, 0), (0, 0), (1, 1), (1, 1)]],
        };
        let err = verify_layout(&naive, &groups)
            .expect_err("naive positional schedule must collide (taps 0,1 share bank 0 in step 0)");
        assert!(err.contains("duplicate"), "unexpected error: {err}");
    }

    /// Logical banks `{0,2,4,6}` (all multiples of `p=2`): raw `lb % p` would alias all four
    /// edges onto physical bank 0 (forcing every occupied group onto a single bank — defeating
    /// the whole point of parallel gather). Rank-based folding — `rank(lb) % p`, where `rank`
    /// is the sorted position of `lb` among the distinct logical-bank ids that actually appear
    /// — must instead spread them across both physical banks (`rank(0,2,4,6) = 0,1,2,3` =>
    /// `rank % 2 = 0,1,0,1`). Every other test in this module uses CONTIGUOUS logical-bank ids,
    /// where `rank(lb) == lb`, so raw-id folding and rank folding are indistinguishable there —
    /// this is the one test that actually exercises rank-based folding.
    #[test]
    fn sparse_logical_bank_ids_fold_across_banks() {
        let edges: Vec<(usize, usize)> = vec![(0, 0), (2, 0), (4, 0), (6, 0)];
        let groups = vec![vec![0usize, 1, 2, 3]];
        let p = 2;

        let layout = plan_serial(&edges, &groups, p);
        verify_layout(&layout, &groups).expect("sparse-id layout must be conflict-free");

        let distinct_banks: std::collections::BTreeSet<_> = layout.bank_of.iter().collect();
        assert!(
            distinct_banks.len() > 1,
            "rank-based folding must spread multiples-of-p logical banks across >1 physical \
             bank, got {:?}",
            layout.bank_of
        );
        // Pin the exact rank-based assignment: if this regressed to raw `lb % p`, bank_of
        // would collapse to all-0s and this assert (not just the distinct-banks one above)
        // would catch it directly.
        assert_eq!(layout.bank_of, vec![0, 1, 0, 1]);
    }

    /// Group edges deliberately chosen (all-even logical-bank ids, so all fold onto physical
    /// bank 0 at p=2) so per-bank occupancy (5) exceeds the OLD balanced-only expectation
    /// `ceil(size/p) = 3`. Under the corrected residual-mux design this is no longer an
    /// infeasible case that panics: `plan_serial` must succeed, and `layout.steps` must equal
    /// the true max-per-bank bottleneck (5) — the actual scenario the M9c fit verdict hit on
    /// real data (a 164-edge group at p=8 concentrating beyond `ceil(164/8)=21` in one bank).
    #[test]
    fn imbalanced_group_concentrates_in_one_bank_and_succeeds() {
        let edges: Vec<(usize, usize)> = (0..9).map(|lb| (lb, 0)).collect();
        let groups = vec![vec![0, 2, 4, 6, 8]]; // 5 edges, all even id => all fold onto bank 0 at p=2
        let p = 2;

        // (i) must not panic.
        let layout = plan_serial(&edges, &groups, p);
        for &e in &groups[0] {
            assert_eq!(
                layout.bank_of[e], 0,
                "edge {e} expected to fold onto bank 0"
            );
        }

        // (ii) must pass verify_layout.
        verify_layout(&layout, &groups)
            .expect("imbalanced layout must still verify (residual-mux)");

        // (iii) layout.steps == the true max-per-bank count, which EXCEEDS the ideal balanced
        // lower bound `serial_steps(5, 2) = 3` — that old formula is no longer the authority.
        let ideal_lower_bound = serial_steps(5, p);
        assert_eq!(ideal_lower_bound, 3);
        assert_eq!(
            layout.steps, 5,
            "steps must equal the true max-per-bank bottleneck (5), not the balanced lower bound (3)"
        );

        // (iv) the five bank-0 edges get distinct steps 0..5 in tap order.
        let mut bank0_steps: Vec<usize> = layout.sched[0]
            .iter()
            .filter(|&&(_, bank)| bank == 0)
            .map(|&(step, _)| step)
            .collect();
        bank0_steps.sort_unstable();
        assert_eq!(bank0_steps, vec![0, 1, 2, 3, 4]);
    }

    /// The only remaining precondition failure: `p == 0` has no physical banks to fold onto.
    /// There is no longer an occupancy panic (see `imbalanced_group_concentrates_in_one_bank_and_succeeds`
    /// above) — this replaces the old `overloaded_bank_panics` test, which asserted a panic
    /// that the residual-mux design correction removed.
    #[test]
    #[should_panic(expected = "p must be positive")]
    fn p_zero_panics() {
        let edges: Vec<(usize, usize)> = vec![(0, 0)];
        let groups = vec![vec![0usize]];
        plan_serial(&edges, &groups, 0);
    }

    /// Deterministic xorshift PRNG (test-only; the solver itself has no RNG — see module docs).
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn shuffle<T>(v: &mut [T], state: &mut u64) {
        for i in (1..v.len()).rev() {
            let j = (xorshift(state) as usize) % (i + 1);
            v.swap(i, j);
        }
    }

    /// 100 random graphs across several `p` values, with deliberately UNBALANCED logical-bank
    /// assignment (no attempt to keep any group's per-bank occupancy under `ceil(size/p)`, the
    /// old feasibility ceiling). The whole point of the residual-mux design is that
    /// `plan_serial` is feasible-always regardless of skew: every output must pass
    /// `verify_layout`, and none may panic.
    #[test]
    fn stress_100_random_graphs_verify() {
        let ps = [1usize, 2, 3, 4, 5, 8];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;

        for iter in 0..100u64 {
            let p = ps[(xorshift(&mut state) as usize) % ps.len()];
            let num_edges = 1 + (xorshift(&mut state) as usize) % 64;
            let num_logical_banks = 1 + (xorshift(&mut state) as usize) % (p * 3 + 1);

            // Random, deliberately NOT balanced logical-bank assignment per edge: skew is the
            // point of this stress test now (an even/round-robin assignment would never
            // exercise the imbalanced-bank path the residual-mux design exists for).
            let edges: Vec<(usize, usize)> = (0..num_edges)
                .map(|i| ((xorshift(&mut state) as usize) % num_logical_banks, i))
                .collect();

            let num_groups = 1 + (xorshift(&mut state) as usize) % 4;
            let mut groups: Vec<Vec<usize>> = Vec::with_capacity(num_groups);
            for _ in 0..num_groups {
                // A group is a shuffled subset (no repeats) of distinct edge ids.
                let group_len = 1 + (xorshift(&mut state) as usize) % num_edges;
                let mut pool: Vec<usize> = (0..num_edges).collect();
                shuffle(&mut pool, &mut state);
                groups.push(pool.into_iter().take(group_len).collect());
            }

            // Must never panic, however skewed edges/groups end up.
            let layout = plan_serial(&edges, &groups, p);
            verify_layout(&layout, &groups)
                .unwrap_or_else(|e| panic!("iter {iter}: p={p} layout failed to verify: {e}"));
        }
    }
}
