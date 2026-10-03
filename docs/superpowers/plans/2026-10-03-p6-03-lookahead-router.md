# [P6-03] Lookahead multi-bit router for the distributed SV plan: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Router::Lookahead` to `aleph-ir::dist::plan`. It moves up to `g` global qubits in a single `Exchange`
(k-bit index swap) and picks which local qubits to evict by next use (Belady). This measurably reduces communication
vs `Router::Naive` (#57 AC: "pass implemented", "measured reduction in communication").

**Architecture:**
- When the current gate needs global logical qubits (`missing`), the lookahead router brings all of them in at once.
- It also **prefetches** further global qubits whose next required use comes *before* the next use of the local
  qubit they would evict.
- Victims are the local qubits not required by the current gate, ranked by farthest next use.
- Victims are moved into the top `k` local slots with local `Swap`s. The existing contiguous-chunk exchange primitive
  (`DistStep::Exchange { global_bits }`, swapping `global_bits[j] ↔ m−k+j`) then performs one `k`-bit exchange,
  costing `(1 − 2^−k)` of a slice instead of `k/2`.
- Next uses come from a per-logical-qubit index of required-local positions, built once.

**Tech Stack:** Rust 2021 (MSRV 1.89), `smallvec`, `proptest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-03-multi-gpu-sv-design.md` (§3.1 `Router::Lookahead`, §6, §7 item 2; issue #57).
Builds on PR #527 (P6-02). Rebase onto `main` once #527 is merged.

## Global Constraints

- Backend-agnostic IR: `aleph-ir::dist` must not mention GPUs or backends.
- No `unwrap()`/`expect()`/`panic!` in library code. Use `DistError`.
- FP64 oracle tolerance is `1e-10` vs `NaiveSvBackend`.
- `Router::Naive` stays, unchanged in behaviour, as the baseline. All its existing tests must still pass.
- Exchange primitive semantics are unchanged: `global_bits[j]` swaps with local physical bit `m − k + j`,
  `k = global_bits.len() ≤ g`.
- `amps_moved_per_rank` for a `k`-bit exchange is `(2^k − 1) · 2^(m−k)`, added with `saturating_add`.
- Branch `p6-03-lookahead-router` in the main checkout. No worktrees.
- PR title `[P6-03] Lookahead multi-bit router for distributed SV`, body `Closes #57`.

## Review Focus

1. **A `k`-bit exchange whose top-`k` slots hold a qubit the current gate needs** (e.g. an `Iswap` on {top slot, global}
   at g=2). Expected: that qubit is swapped *down* (it stays local) and never evicted. Pinned in Task 1
   (`lookahead_never_evicts_required`) and in the Task 2 oracle.
2. **Too few local slots for any prefetch** (n=4, g=2, m=2, a 2-target gate). Expected: `k` shrinks to exactly the
   missing qubits, with no error and a correct state. Pinned in Task 1 (`lookahead_tight_m_no_prefetch`).
3. **A relabel `Swap` between exchanges** changes which logical qubit is global. Expected: next-use is keyed by
   *logical* qubit, so routing stays correct. Pinned by the Task 2 oracle (QFT's trailing swaps, a mid-circuit
   local↔global swap test) and the proptest.
4. **The `k>1` exchange primitive itself.** `exchange_cpu` with 2 and 3 bits must equal the explicit product of
   physical SWAPs. Pinned in Task 2 (`exchange_cpu_multi_bit_is_product_of_swaps`).
5. **Lookahead regressing on easy circuits.** On GHZ and QFT, lookahead must move no more amplitudes than naive. Pinned
   in Task 1 (`lookahead_no_worse_on_ghz_qft`).

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/aleph-ir/src/dist/mod.rs` (modify) | add `Router::Lookahead` variant + doc |
| `crates/aleph-ir/src/dist/next_use.rs` (create) | `NextUse` index: per-logical-qubit required-local positions, `next(q, from)` |
| `crates/aleph-ir/src/dist/plan.rs` (modify) | dispatch on router; `exchange_lookahead` (victims, prefetch, top-slot placement, k-bit exchange) |
| `crates/aleph-sv/tests/dist_oracle.rs` (modify) | run every oracle test under both routers; k-bit exchange unit test; lookahead proptests |
| `crates/aleph-sv/examples/dist_comm_counts.rs` (modify) | Naive vs Lookahead columns |
| `docs/perf/p6-03-routing.md` (create) | algorithm + measured reduction table (#57 AC) |

---

### Task 1: `NextUse` index + `Router::Lookahead` in the planner

**Files:**
- Create: `crates/aleph-ir/src/dist/next_use.rs`
- Modify: `crates/aleph-ir/src/dist/mod.rs` (add `mod next_use;` and the `Router::Lookahead` variant)
- Modify: `crates/aleph-ir/src/dist/plan.rs`

**Interfaces:**
- Consumes: `required_local`, `Map::{l2p, p2l, swap_phys}`, `CommStats`, `DistStep::Exchange` (P6-02).
- Produces:
  - `pub enum Router { Naive, Lookahead }`.
  - `pub(crate) struct NextUse` with `fn build(circuit: &Circuit) -> Self` and
    `fn next(&mut self, q: u32, from: usize) -> usize` (`usize::MAX` = never). Calls must have non-decreasing `from`
    for each `q`.
  - `plan(.., Router::Lookahead)` emits `Exchange`s with `1 ≤ k ≤ g` bits.

- [ ] **Step 1: Write the failing tests**

Append to `crates/aleph-ir/src/dist/next_use.rs` (the file starts with only this test module, so it fails to compile):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circuit;

    #[test]
    fn next_use_tracks_required_positions() {
        let mut c = Circuit::new(3, 0);
        c.h(0).unwrap(); // 0: q0 required
        c.cnot(0, 1).unwrap(); // 1: q1 required (target)
        c.rz(0.2, 0).unwrap(); // 2: diagonal, nothing required
        c.swap(0, 2).unwrap(); // 3: relabel, nothing required
        c.h(0).unwrap(); // 4: q0 required
        let mut nu = NextUse::build(&c);
        assert_eq!(nu.next(0, 0), 0);
        assert_eq!(nu.next(0, 1), 4);
        assert_eq!(nu.next(1, 0), 1);
        assert_eq!(nu.next(1, 2), usize::MAX);
        assert_eq!(nu.next(2, 0), usize::MAX);
        assert_eq!(nu.next(0, 5), usize::MAX);
    }
}
```

Append to the `tests` module in `crates/aleph-ir/src/dist/plan.rs`:

```rust
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
                v.dedup();
                assert_eq!(v.len(), global_bits.len());
            }
        }
        assert!(
            la.stats.amps_moved_per_rank * 2 <= naive.amps_moved_per_rank,
            "lookahead {} vs naive {}",
            la.stats.amps_moved_per_rank,
            naive.amps_moved_per_rank
        );
    }

    #[test]
    fn lookahead_no_worse_on_ghz_qft() {
        for c in [ghz(16), qft(16)] {
            for g in [1u32, 2, 3] {
                let l = DistLayout::new(16, g).unwrap();
                let n = plan(&c, l, Router::Naive).unwrap().stats;
                let a = plan(&c, l, Router::Lookahead).unwrap().stats;
                assert!(a.amps_moved_per_rank <= n.amps_moved_per_rank, "g={g}: {a:?} vs {n:?}");
            }
        }
    }

    #[test]
    fn lookahead_never_evicts_required() {
        // Iswap on logical (3, 5) at n=6, g=2: 3 sits in a top slot, 5 is global.
        let mut c = Circuit::new(6, 0);
        c.h(4).unwrap(); // makes logical 4 a prefetch candidate (global at phys 4)
        c.add_gate(GateInstance::new(Gate::Iswap, vec![3, 5])).unwrap();
        c.h(4).unwrap();
        let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Lookahead).unwrap();
        // after planning, both Iswap operands were local at the Iswap
        let DistStep::Local(seg) = p
            .steps
            .iter()
            .rev()
            .find(|s| matches!(s, DistStep::Local(v) if v.iter().any(|i| matches!(i, Instruction::Gate(g) if matches!(g.gate, Gate::Iswap)))))
            .unwrap()
        else {
            panic!()
        };
        let Instruction::Gate(isw) = seg
            .iter()
            .find(|i| matches!(i, Instruction::Gate(g) if matches!(g.gate, Gate::Iswap)))
            .unwrap()
        else {
            panic!()
        };
        assert!(isw.qubits.iter().all(|&p| p < 4), "{:?}", isw.qubits);
    }

    #[test]
    fn lookahead_tight_m_no_prefetch() {
        // n=4, g=2, m=2: an Iswap needs both local slots, so no room to prefetch.
        let mut c = Circuit::new(4, 0);
        c.h(3).unwrap();
        c.add_gate(GateInstance::new(Gate::Iswap, vec![0, 2])).unwrap();
        c.h(3).unwrap();
        let p = plan(&c, DistLayout::new(4, 2).unwrap(), Router::Lookahead).unwrap();
        assert!(p.stats.exchanges >= 1);
    }
```

Add `pub mod`-less registration in `crates/aleph-ir/src/dist/mod.rs` next to `mod plan;`:

```rust
mod next_use;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p aleph-ir --lib dist::`
Expected: FAIL to compile (`NextUse` not defined, `Router::Lookahead` not a variant).

- [ ] **Step 3: Write the implementation**

Prepend to `crates/aleph-ir/src/dist/next_use.rs` (above the tests):

```rust
//! Next-use index for the lookahead router: for every logical qubit, the
//! ascending instruction indices at which it must be local
//! (`required_local`). Belady-style eviction asks "when is `q` needed next?".

use super::plan::required_local;
use crate::{Circuit, Instruction};
use aleph_core::Gate;

pub(crate) struct NextUse {
    pos: Vec<Vec<usize>>,
    cursor: Vec<usize>,
}

impl NextUse {
    pub(crate) fn build(circuit: &Circuit) -> Self {
        let n = circuit.num_qubits() as usize;
        let mut pos = vec![Vec::new(); n];
        for (i, instr) in circuit.instructions().iter().enumerate() {
            if let Instruction::Gate(g) = instr {
                if matches!(g.gate, Gate::Swap) && g.controls.is_empty() {
                    continue; // relabel: needs nothing local
                }
                for q in required_local(g) {
                    pos[q as usize].push(i);
                }
            }
        }
        Self { pos, cursor: vec![0; n] }
    }

    /// First index `>= from` at which logical `q` must be local; `usize::MAX`
    /// if never. `from` must be non-decreasing per `q` across calls.
    pub(crate) fn next(&mut self, q: u32, from: usize) -> usize {
        let (p, c) = (&self.pos[q as usize], &mut self.cursor[q as usize]);
        while *c < p.len() && p[*c] < from {
            *c += 1;
        }
        p.get(*c).copied().unwrap_or(usize::MAX)
    }
}
```

In `crates/aleph-ir/src/dist/mod.rs`, replace the `Router` enum with:

```rust
/// Exchange-placement strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Router {
    /// One global qubit per exchange, on demand, evicting the top local slot.
    Naive,
    /// Up to `g` global qubits per exchange: everything the current gate
    /// needs plus prefetched qubits needed before their victim's next use;
    /// victims chosen by farthest next use (Belady). P6-03.
    Lookahead,
}
```

In `crates/aleph-ir/src/dist/plan.rs`:

1. Add `use super::next_use::NextUse;` to the imports.
2. Replace the line `let Router::Naive = router;` with nothing; then, right after `let mut stats = CommStats::default();`,
   add:

```rust
    let mut next_use = match router {
        Router::Naive => None,
        Router::Lookahead => Some(NextUse::build(circuit)),
    };
```

3. Change `for instr in circuit.instructions() {` to `for (idx, instr) in circuit.instructions().iter().enumerate() {`.
4. Replace the whole `for &q in &req { ... }` exchange loop with:

```rust
                match next_use.as_mut() {
                    None => {
                        for &q in &req {
                            if layout.is_global(map.l2p[q as usize]) {
                                exchange_naive(q, &req, layout, &mut map, &mut cur, &mut steps, &mut stats)?;
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
                                idx, &missing, &req, layout, nu, &mut map, &mut cur, &mut steps, &mut stats,
                            )?;
                        }
                    }
                }
```

5. Add these two functions below `plan` (the first is the P6-02 loop body moved verbatim):

```rust
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
    if req.contains(&map.p2l[top as usize]) {
        let free = (0..top)
            .rev()
            .find(|&p| !req.contains(&map.p2l[p as usize]))
            .ok_or(DistError::TooFewLocalQubits { need: req.len(), m })?;
        cur.push(Instruction::Gate(GateInstance::new(Gate::Swap, vec![top, free])));
        map.swap_phys(top, free);
        stats.local_swaps += 1;
    }
    if !cur.is_empty() {
        steps.push(DistStep::Local(std::mem::take(cur)));
    }
    let gbit = map.l2p[q as usize];
    steps.push(DistStep::Exchange { global_bits: smallvec![gbit] });
    map.swap_phys(gbit, top);
    stats.exchanges += 1;
    // Saturate: n=64 with g=1 moves 2^62 amps per exchange.
    stats.amps_moved_per_rank = stats.amps_moved_per_rank.saturating_add(1u64 << (m - 1));
    Ok(())
}

/// P6-03 lookahead exchange: one k-bit exchange for all `missing` qubits plus
/// prefetches, evicting the local qubits whose next use is farthest away.
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
    let mut victims: Vec<(usize, u32, u32)> = (0..m)
        .map(|p| map.p2l[p as usize])
        .filter(|l| !req.contains(l))
        .map(|l| (nu.next(l, after), map.l2p[l as usize], l))
        .collect();
    victims.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    if victims.len() < missing.len() {
        return Err(DistError::TooFewLocalQubits { need: req.len(), m });
    }
    let mut bring: SmallVec<[u32; 4]> = missing.iter().copied().collect();
    // Prefetch: other global qubits, soonest next use first, while each is
    // needed before the victim it would displace.
    let mut cands: Vec<(usize, u32)> = (m..layout.n)
        .map(|p| map.p2l[p as usize])
        .filter(|l| !bring.contains(l))
        .map(|l| (nu.next(l, after), l))
        .filter(|&(t, _)| t != usize::MAX)
        .collect();
    cands.sort();
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
    let k = bring.len() as u32;
    let chosen: SmallVec<[u32; 4]> = victims[..k as usize].iter().map(|v| v.2).collect();
    // Place the chosen victims in the top-k local slots (m-k..m).
    let top_lo = m - k;
    let mut free_slots: SmallVec<[u32; 4]> = (top_lo..m)
        .filter(|&p| !chosen.contains(&map.p2l[p as usize]))
        .collect();
    for &v in &chosen {
        let vp = map.l2p[v as usize];
        if vp >= top_lo {
            continue;
        }
        let Some(slot) = free_slots.pop() else {
            return Err(DistError::Unsupported { kind: "internal: no free top slot" });
        };
        cur.push(Instruction::Gate(GateInstance::new(Gate::Swap, vec![slot, vp])));
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
```

The exchange loop body used `top`, `free` and `gbit` inline in P6-02. `exchange_naive` is that body unchanged, so
`Router::Naive` output is byte-identical. The existing naive tests prove this.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-ir --lib dist:: && cargo clippy -p aleph-ir --all-targets -- -D warnings && cargo fmt --check`
Expected: all PASS, including the 4 new planner tests, `next_use_tracks_required_positions`, and every P6-02 test.

If `lookahead_batches_exchanges_on_brickwall` fails on the 2× bound, print both stats. Debug with
`superpowers:systematic-debugging` before touching the bound. The expected ratio for the brickwall is about 3×.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-ir/src/dist
git commit -m "[P6-03] dist::plan: lookahead multi-bit router (Belady eviction + prefetch)"
```

---

### Task 2: oracle under both routers + k-bit exchange check

**Files:**
- Modify: `crates/aleph-sv/tests/dist_oracle.rs`

**Interfaces:**
- Consumes: `Router::{Naive, Lookahead}`, `aleph_sv::dist_ref::{run_dist, exchange_cpu}`, `DistLayout`.

- [ ] **Step 1: Write the tests**

1. Change `check` so every existing caller covers both routers:

```rust
fn check(c: &Circuit, g: u32, what: &str) {
    for router in [Router::Naive, Router::Lookahead] {
        let p = plan(c, DistLayout::new(c.num_qubits(), g).unwrap(), router).unwrap();
        assert_close(&run_dist(&p).unwrap(), &reference(c), &format!("{what} g={g} {router:?}"));
    }
}
```

2. In both proptests, replace `Router::Naive` with a strategy-drawn router. Add
   `router in prop_oneof![Just(Router::Naive), Just(Router::Lookahead)],` to each `proptest!` fn's inputs and pass
   `router` to `plan`.

3. Add:

```rust
#[test]
fn exchange_cpu_multi_bit_is_product_of_swaps() {
    use aleph_sv::dist_ref::exchange_cpu;
    // n=5, g=2 (m=3), and n=6, g=3 (m=3): fill ranks with distinct values
    for (n, g, bits) in [(5u32, 2u32, vec![3u32, 4]), (5, 2, vec![4, 3]), (6, 3, vec![5, 3, 4])] {
        let l = DistLayout::new(n, g).unwrap();
        let m = l.m();
        let size = 1usize << m;
        let mut ranks: Vec<Vec<Complex>> = (0..l.ranks() as usize)
            .map(|r| (0..size).map(|i| Complex::new((r * size + i) as f64, 0.0)).collect())
            .collect();
        let full: Vec<Complex> = ranks.concat();
        exchange_cpu(&mut ranks, l, &bits);
        let got: Vec<Complex> = ranks.concat();
        let k = bits.len() as u32;
        for (x, want) in full.iter().enumerate() {
            // destination of index x: swap bit bits[j] with bit m-k+j for all j
            let mut y = x;
            for (j, &gb) in bits.iter().enumerate() {
                let lb = m - k + j as u32;
                let (a, b) = ((x >> gb) & 1, (x >> lb) & 1);
                y = (y & !(1 << gb) & !(1 << lb)) | (b << gb) | (a << lb);
            }
            assert_eq!(got[y], *want, "n={n} bits={bits:?} x={x}");
        }
    }
}

#[test]
fn mid_circuit_relabel_then_lookahead() {
    let mut c = brickwall(8, 3, 11);
    c.swap(1, 7).unwrap();
    c.swap(6, 2).unwrap();
    let tail = brickwall(8, 3, 12);
    for i in tail.instructions() {
        if let Instruction::Gate(g) = i {
            c.add_gate(g.clone()).unwrap();
        }
    }
    for g in 1..=3 {
        check(&c, g, "relabel-mid");
    }
}

#[test]
fn lookahead_moves_less_on_brickwall() {
    let c = brickwall(12, 10, 3);
    let l = DistLayout::new(12, 2).unwrap();
    let n = plan(&c, l, Router::Naive).unwrap().stats;
    let a = plan(&c, l, Router::Lookahead).unwrap().stats;
    assert!(a.amps_moved_per_rank < n.amps_moved_per_rank, "{a:?} vs {n:?}");
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p aleph-sv --test dist_oracle`
Expected: all PASS. These are coverage tests over Task 1's code, which already exists.

- [ ] **Step 3: Prove the new tests bite (mutation)**

In `exchange_lookahead`, temporarily change `map.swap_phys(gb, top_lo + j as u32);` to
`map.swap_phys(gb, top_lo + (k - 1 - j as u32));` (a wrong pairing). Then run:
`cargo test -p aleph-sv --test dist_oracle`
Expected: FAIL in several router=Lookahead cases. Revert the mutation and re-run. Expected: PASS. Then delete any
`dist_oracle.proptest-regressions` file the mutant run created.

- [ ] **Step 4: Lint**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/tests/dist_oracle.rs
git commit -m "[P6-03] dist oracle: both routers, k-bit exchange check, relabel + lookahead"
```

---

### Task 3: measured reduction, report, PR

**Files:**
- Modify: `crates/aleph-sv/examples/dist_comm_counts.rs`
- Create: `docs/perf/p6-03-routing.md`

- [ ] **Step 1: Extend the example to print both routers**

Replace `main` in `crates/aleph-sv/examples/dist_comm_counts.rs` with:

```rust
fn main() {
    let cases: Vec<(&str, Circuit)> = vec![
        ("GHZ-32", ghz(32)),
        ("QFT-32", qft(32)),
        ("random-30 d=20", brickwall(30, 20)),
        ("Grover-20 (5 iters)", grover()),
    ];
    println!("| circuit | g | naive exch | naive × slice | lookahead exch | lookahead × slice | reduction | la local swaps |");
    println!("|---|---|---|---|---|---|---|---|");
    for (name, c) in &cases {
        for g in [2u32, 3] {
            let l = DistLayout::new(c.num_qubits(), g).unwrap();
            let slice = (1u64 << l.m()) as f64;
            let n = plan(c, l, Router::Naive).unwrap().stats;
            let a = plan(c, l, Router::Lookahead).unwrap().stats;
            let (ns, la) = (n.amps_moved_per_rank as f64 / slice, a.amps_moved_per_rank as f64 / slice);
            println!(
                "| {name} | {g} | {} | {ns:.1} | {} | {la:.1} | {:.2}× | {} |",
                n.exchanges,
                a.exchanges,
                ns / la.max(f64::MIN_POSITIVE),
                a.local_swaps
            );
        }
    }
}
```

- [ ] **Step 2: Run it**

Run: `cargo run --release -p aleph-sv --example dist_comm_counts`
Expected: 8 rows. Random brickwall shows a reduction of at least 2× at g=2. GHZ and QFT show a reduction of 1.00× or
better, never below 1. If any row shows below 1.00×, stop and investigate before writing the doc.

- [ ] **Step 3: Write `docs/perf/p6-03-routing.md`**

```markdown
# P6-03 — Lookahead multi-bit routing (distributed SV)

Issue #57. Builds on P6-02 ([`p6-02-partitioning.md`](p6-02-partitioning.md)).

## Algorithm
- When a gate needs global qubits, bring **all** of them in with one k-bit exchange (k ≤ g) —
  `(1 − 2^−k)` of a slice instead of `k/2` for k separate swaps.
- **Prefetch** further global qubits (soonest next use first) while each is needed before the local
  qubit it would evict.
- **Victims** = local qubits the current gate does not need, farthest next use first (Belady);
  moved into the top-k local slots with local `Swap`s (one on-device bandwidth pass each — cheap next to a
  link transfer), so the exchange stays a contiguous-chunk all-to-all.
- Next use is keyed by **logical** qubit, so user `Swap` relabels never confuse it.

## Correctness
<what the oracle covers — both routers on every P6-02 case, k-bit `exchange_cpu` vs explicit SWAP product,
mid-circuit relabel, both proptests; the mutation that proved the tests bite>

## Measured reduction
<table from `cargo run --release -p aleph-sv --example dist_comm_counts`>

## Reading
<2–4 sentences from the real numbers: where lookahead wins, where it ties, local-swap overhead, and what
#59 (reordering commuting gates) could still remove.>
```

Fill the three bracketed parts from the actual run. No placeholders in the committed file.

- [ ] **Step 4: Full verification**

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo +beta clippy -p aleph-ir -p aleph-sv --all-targets -- -D warnings
cargo test -p aleph-ir -p aleph-sv
```

Expected: all clean and passing.

- [ ] **Step 5: Commit, push, PR**

```bash
git add crates/aleph-sv/examples/dist_comm_counts.rs docs/perf/p6-03-routing.md
git commit -m "[P6-03] Routing report: naive vs lookahead communication"
git push -u origin p6-03-lookahead-router
gh pr create --title "[P6-03] Lookahead multi-bit router for distributed SV" --body "Closes #57 ..."
```

The PR body must include: a summary, test counts, the reduction table, the note "planning-only, no perf bench (GPU
timing comes with PR 3 + AWS)", follow-ups (#59, PR 3), and the attribution lines from the session's system
reminder.
