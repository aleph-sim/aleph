//! Serial-gather conflict-free slot solver (M9c gather-crossbar fix, step 3.1).
//!
//! The streaming relay-BP core stores each message-passing edge in one of `P` physical
//! BRAM banks and gathers `P` messages per cycle over `ceil(N/P)` steps into a row buffer,
//! with no select mux/crossbar between banks and read ports. That only works if, for every
//! group of edges gathered together (one CHK/VAR/etc. tap list), no two edges in the group
//! ever land in the same physical bank in the same step — otherwise two edges would need to
//! read the same single-port BRAM in the same cycle, which is undefined.
//!
//! This module is the pure-Rust solver that makes that true: [`plan_serial`] assigns each
//! edge a FIXED `(physical bank, intra-bank address)` storage slot (consistent across every
//! group it appears in) by folding the caller's logical banks onto `p` physical banks, and
//! then computes a per-group conflict-free `(step, port)` read schedule. [`verify_layout`] is
//! the independent checker (also used as the emitter's gen-time guard) that the M9c crossbar
//! bug — a naive schedule that ignores real bank identity — cannot sneak back in.
//!
//! Design context: `docs/superpowers/specs/2026-07-15-m9c-serial-gather-design.md`.

use std::collections::{BTreeMap, HashMap, HashSet};

/// Fixed BRAM storage + per-group conflict-free read schedule for the serial-gather core.
pub struct SerialLayout {
    /// Number of physical BRAM banks / read ports.
    pub p: usize,
    /// Number of read steps (`ceil(max group size / p)`), shared by every group.
    pub steps: usize,
    /// `bank_of[e]` = edge `e`'s fixed physical BRAM bank (`0..p`).
    pub bank_of: Vec<usize>,
    /// `addr_of[e]` = edge `e`'s intra-bank address (unique within its physical bank).
    pub addr_of: Vec<usize>,
    /// `sched[g][t]` = the `(step, port)` at which group `g`'s `t`-th tap (in tap order) is
    /// read, i.e. `buffer[step * p + port]` holds that tap's message.
    pub sched: Vec<Vec<(usize, usize)>>,
}

/// Number of read steps needed to gather a group of `max_group` edges over `p` parallel
/// ports: `ceil(max_group / p)`.
pub fn serial_steps(max_group: usize, p: usize) -> usize {
    assert!(p > 0, "serial_gather: p must be positive");
    max_group.div_ceil(p)
}

/// Assign fixed BRAM storage and a conflict-free per-group read schedule.
///
/// `edges[e] = (logical_bank, row)`; `groups[g]` = the edge ids read together in group `g`,
/// in tap order. Deterministic (sorts by edge id; no RNG). Panics if `p == 0` or if no
/// conflict-free schedule exists within `serial_steps(max_group, p)` steps (should not happen
/// for reasonably balanced inputs; see module docs).
pub fn plan_serial(edges: &[(usize, usize)], groups: &[Vec<usize>], p: usize) -> SerialLayout {
    assert!(p > 0, "serial_gather: p must be positive");

    // --- Storage folding: logical banks -> p physical banks, balanced round-robin by RANK
    // (the sorted position of a logical bank id among the distinct ids that actually appear),
    // not the raw id value. Using rank rather than `logical_bank % p` means sparse/clustered
    // logical-bank numbering (e.g. every id a multiple of p) still spreads evenly across
    // physical banks instead of aliasing them all onto bank 0 — see the
    // `naive_round_robin_would_collide` test for why raw-id folding is not safe to assume.
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
    for (e, &(lb, _)) in edges.iter().enumerate() {
        let bank = rank_of[&lb] % p;
        bank_of[e] = bank;
        addr_of[e] = next_addr[bank];
        next_addr[bank] += 1;
    }

    // --- Schedule: one shared step count across all groups.
    let max_group = groups.iter().map(|g| g.len()).max().unwrap_or(0);
    let steps = serial_steps(max_group, p);

    let sched: Vec<Vec<(usize, usize)>> = groups
        .iter()
        .map(|group| schedule_group(group, &bank_of, p, steps))
        .collect();

    SerialLayout {
        p,
        steps,
        bank_of,
        addr_of,
        sched,
    }
}

/// First-fit-by-bank greedy: colour `group`'s taps by physical bank, then pack them into
/// `steps` steps of capacity `p` with at most one tap per bank per step. Banks are visited in
/// ascending order and, within a bank, taps in original tap order — deterministic, no RNG.
/// Panics if `steps` is insufficient for some bank's occurrence count in this group (should not
/// happen for balanced folding; see module docs).
fn schedule_group(
    group: &[usize],
    bank_of: &[usize],
    p: usize,
    steps: usize,
) -> Vec<(usize, usize)> {
    // Bucket tap positions by physical bank (BTreeMap => ascending bank order for determinism).
    let mut by_bank: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (t, &e) in group.iter().enumerate() {
        by_bank.entry(bank_of[e]).or_default().push(t);
    }

    let mut step_banks: Vec<HashSet<usize>> = vec![HashSet::new(); steps];
    let mut step_count: Vec<usize> = vec![0; steps];
    let mut result = vec![(usize::MAX, usize::MAX); group.len()];

    for (&bank, taps) in &by_bank {
        for &t in taps {
            let step = (0..steps)
                .find(|&s| step_count[s] < p && !step_banks[s].contains(&bank))
                .unwrap_or_else(|| {
                    panic!(
                        "serial_gather: infeasible schedule — bank {bank} needs more than \
                         {steps} steps at p={p} (group has {} taps)",
                        group.len()
                    )
                });
            let port = step_count[step];
            step_banks[step].insert(bank);
            step_count[step] += 1;
            result[t] = (step, port);
        }
    }

    result
}

/// Check a [`SerialLayout`] against `groups`:
/// (a) every group's reads have distinct `(step, port)` and hit distinct physical banks per
///     step; (b) no two distinct edges alias the same `(bank_of, addr_of)` storage slot;
/// (c) tap order is preserved (`sched[g][t]` corresponds to `groups[g][t]`);
/// (d) `steps == ceil(max group size / p)`.
pub fn verify_layout(layout: &SerialLayout, groups: &[Vec<usize>]) -> Result<(), String> {
    if layout.sched.len() != groups.len() {
        return Err(format!(
            "sched has {} groups, expected {}",
            layout.sched.len(),
            groups.len()
        ));
    }

    for (g, group) in groups.iter().enumerate() {
        let sched_g = &layout.sched[g];
        // (c) tap order preserved: sched[g] must have one entry per tap, index-aligned with
        // groups[g] (checked below by zipping in order).
        if sched_g.len() != group.len() {
            return Err(format!(
                "group {g}: sched has {} taps, expected {}",
                sched_g.len(),
                group.len()
            ));
        }

        let mut seen_slots: HashSet<(usize, usize)> = HashSet::new();
        let mut banks_per_step: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (t, (&edge, &(step, port))) in group.iter().zip(sched_g.iter()).enumerate() {
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
            if port >= layout.p {
                return Err(format!("group {g} tap {t}: port {port} >= p {}", layout.p));
            }
            if !seen_slots.insert((step, port)) {
                return Err(format!(
                    "group {g} tap {t}: duplicate (step,port) ({step},{port})"
                ));
            }
            let bank = layout.bank_of[edge];
            if !banks_per_step.entry(step).or_default().insert(bank) {
                return Err(format!(
                    "group {g} tap {t}: bank {bank} read twice in step {step} (edge {edge})"
                ));
            }
        }
    }

    // (b) storage: no two distinct edges share (bank_of, addr_of).
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

    // (d) steps == ceil(max group size / p).
    let max_group = groups.iter().map(|g| g.len()).max().unwrap_or(0);
    let expected_steps = serial_steps(max_group, layout.p);
    if layout.steps != expected_steps {
        return Err(format!(
            "steps {} != ceil(max_group={}/p={}) = {}",
            layout.steps, max_group, layout.p, expected_steps
        ));
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
    /// synthetic sanity case from the task brief.
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

    /// A group whose edges sit on alternating physical banks (0,0,1,1 in tap order). A
    /// *naive* scheduler that assigns `(step, port) = (t/p, t%p)` purely from tap position —
    /// ignoring which physical bank each edge actually lives in — puts the two bank-0 edges
    /// (taps 0 and 1) in the same step: a genuine hardware conflict (two reads from the same
    /// single-port BRAM in one cycle). `plan_serial`'s bank-aware schedule must not do this;
    /// `verify_layout` must reject the naive one.
    #[test]
    fn naive_round_robin_would_collide() {
        let edges: Vec<(usize, usize)> = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
        let groups = vec![vec![0, 2, 1, 3]]; // tap order: bank0, bank0, bank1, bank1
        let p = 2;

        let layout = plan_serial(&edges, &groups, p);
        assert_eq!(layout.bank_of, vec![0, 1, 0, 1]); // edge id % 2 (contiguous ids => rank=id)
        verify_layout(&layout, &groups).expect("solver-produced schedule must be conflict-free");

        // Hand-roll the naive positional schedule against the SAME real storage assignment
        // and confirm verify_layout correctly rejects it as a same-step bank collision.
        let naive = SerialLayout {
            p: layout.p,
            steps: layout.steps,
            bank_of: layout.bank_of.clone(),
            addr_of: layout.addr_of.clone(),
            sched: vec![vec![(0, 0), (0, 1), (1, 0), (1, 1)]],
        };
        let err = verify_layout(&naive, &groups).expect_err(
            "naive round-robin schedule must collide (taps 0,1 share bank 0 in step 0)",
        );
        assert!(err.contains("bank 0 read twice"), "unexpected error: {err}");
    }

    /// Group edges deliberately chosen (all-even logical-bank ids, so all fold onto physical
    /// bank 0 at p=2) so no schedule fits within `serial_steps`. `plan_serial` must panic
    /// rather than silently emit a colliding layout.
    #[test]
    #[should_panic(expected = "infeasible schedule")]
    fn overloaded_bank_panics() {
        let edges: Vec<(usize, usize)> = (0..9).map(|lb| (lb, 0)).collect();
        let groups = vec![vec![0, 2, 4, 6, 8]]; // 5 edges, all even id => all bank 0 at p=2
        plan_serial(&edges, &groups, 2);
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

    /// 100 random graphs across several `p` values. Every layout `plan_serial` produces must
    /// pass `verify_layout`, use exactly `ceil(max_group/p)` steps, and have no `(step,port)`
    /// collision. Graphs are built so every physical bank appears at most `steps = ceil(max
    /// group len / p)` times per group — via a shuffled per-residue multiset, NOT by
    /// round-robining tap position — which keeps the stress test itself feasible (the solver
    /// is only promised a schedule within `steps`, not `steps_cap` guesses that can be tighter
    /// than what the largest group actually needs) while still exercising the solver's
    /// bank-aware scheduling against a randomized mix of group sizes, orders, and bank
    /// distributions.
    #[test]
    fn stress_100_random_graphs_verify() {
        let ps = [1usize, 2, 3, 4, 5, 8];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;

        for iter in 0..100u64 {
            let p = ps[(xorshift(&mut state) as usize) % ps.len()];
            let num_groups = 1 + (xorshift(&mut state) as usize) % 4;

            // Decide every group's length FIRST, then derive `steps` from the largest one.
            // (Deriving `steps` from a separate random guess and only bounding the largest
            // group by it is the bug this comment is here to warn against: a smaller group in
            // the same iteration can still legitimately want up to `steps` copies of one
            // physical bank, since `steps` is shared across all groups in a layout.)
            let desired_len: Vec<usize> = (0..num_groups)
                .map(|_| 1 + (xorshift(&mut state) as usize) % (p * 4))
                .collect();
            let max_len = *desired_len.iter().max().unwrap();
            let steps = serial_steps(max_len, p);
            let num_logical_banks = p * steps;

            // One edge per logical bank id, ids contiguous 0..num_logical_banks => rank(lb) ==
            // lb, so bank_of[e] == e % p exactly (matches the residue bookkeeping below).
            let edges: Vec<(usize, usize)> = (0..num_logical_banks).map(|lb| (lb, lb)).collect();

            // Candidate logical-bank ids per residue class (physical bank), shuffled.
            let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); p];
            for lb in 0..num_logical_banks {
                candidates[lb % p].push(lb);
            }
            for c in &mut candidates {
                shuffle(c, &mut state);
            }

            // Base multiset: each residue (physical bank) appears exactly `steps` times, so
            // any subset of it used for one group has per-bank count <= steps.
            let mut base_slots: Vec<usize> = Vec::with_capacity(num_logical_banks);
            for r in 0..p {
                base_slots.extend(std::iter::repeat_n(r, steps));
            }

            let mut groups: Vec<Vec<usize>> = Vec::with_capacity(num_groups);
            for &group_len in &desired_len {
                let mut slots = base_slots.clone();
                shuffle(&mut slots, &mut state);

                let mut cursor = vec![0usize; p];
                let mut group = Vec::with_capacity(group_len);
                for &r in slots.iter().take(group_len) {
                    let lb = candidates[r][cursor[r]];
                    cursor[r] += 1;
                    group.push(lb);
                }
                groups.push(group);
            }

            let layout = plan_serial(&edges, &groups, p);
            assert_eq!(layout.steps, steps, "iter {iter}: p={p} steps mismatch");
            verify_layout(&layout, &groups)
                .unwrap_or_else(|e| panic!("iter {iter}: p={p} layout failed to verify: {e}"));
        }
    }
}
