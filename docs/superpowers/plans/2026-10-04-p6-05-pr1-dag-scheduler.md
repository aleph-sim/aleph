# P6-05 PR 1: DAG scheduler + placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a commutation-respecting reordering scheduler (`Router::Reorder { max_k }`) and an initial global-qubit
placement to the distributed planner. They emit the existing `DistPlan`, are oracle-equivalent, and are measured by the
CPU comm-stats table.

**Architecture:**
- `dag.rs` builds a dependency DAG from per-qubit Z/X/Other action blocks, using block counters instead of explicit
  edges.
- `schedule.rs` list-schedules that DAG. It runs every runnable ready gate, and when none is runnable it emits one
  Belady-chosen k-bit exchange.
- `placement.rs` picks the initial global set.
- `plan.rs` is refactored so that both the in-order routers and the scheduler share `Map`, physical lowering, and
  exchange emission. A new `plan_from` accepts an initial map.

**Tech Stack:** Rust 2021 (MSRV 1.89), `smallvec`, `thiserror`, `proptest` (dev). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-p6-05-comm-aware-compiler-design.md` (§2, §3, §4, §5, §7; §8's CPU
comm-stats part). PR 2 (cost model) and PR 3 (`compile`, GPU bench, report) get their own plans.

## Global Constraints

- Branch: `p6-05-comm-aware-compiler`, in the main checkout `/Users/ex/GitHub/aleph`. **No git worktrees.**
- Library code has no `unwrap()`/`expect()`/`panic!` on input. Internal impossibilities return
  `DistError::Unsupported { kind: "internal: …" }` (the existing pattern in `specialize.rs`).
- `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, **and**
  `cargo +nightly clippy --workspace --all-targets -- -D warnings` must be clean. CI runs the beta channel.
- Amplitude oracle tolerance: `1e-10` (FP64).
- `aleph-ir` stays backend-agnostic: no `aleph-sv`/`aleph-cuda` imports in `aleph-ir/src`.
- Prepend rustup to PATH before cargo on this Mac: `export PATH="$(brew --prefix rustup)/bin:$PATH"`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn
  ```
- The PR body says `Refs #59`, **not** `Closes`, because PR 3 closes it. It ends with the Claude Code attribution lines.

## Review Focus

1. **A relabel `Swap` reordered relative to gates on other qubits** must still produce the oracle state. It is pinned by
   proptests that include `Swap` (Task 4), plus `swap_relabel_is_a_fence_on_its_qubits` (Task 2).
2. **A gate needing more global qubits than `max_k`** (e.g. `Iswap` on two globals with `max_k = 1`) must still get one
   exchange carrying all of them, not loop or error. Pinned by `missing_exceeding_max_k_still_exchanges` (Task 4).
3. **`g = 0`** (single rank) must produce zero exchanges with `Reorder` and with any placement. Pinned by
   `g0_never_exchanges` (Task 4).
4. **A non-identity initial map with the Naive router**: Naive's "top slot" eviction assumes nothing about identity.
   Pinned by the random-permutation placement proptest (Task 5).
5. **A `Barrier`** must fence reordering across it. Pinned by `barrier_fences_reordering` /
   `without_barrier_local_work_is_hoisted` (Task 4).

---

### Task 1: Refactor `plan.rs`: shared `Map`, lowering, exchange emission, `plan_from`

**Files:**
- Modify: `crates/aleph-ir/src/dist/plan.rs`
- Modify: `crates/aleph-ir/src/dist/mod.rs` (export `plan_from`; add `DistError::BadPlacement`; document the |0…0⟩
  invariant on `DistPlan`)
- Test: `crates/aleph-ir/src/dist/plan.rs` (unit), `crates/aleph-sv/tests/dist_oracle.rs`

**Interfaces:**
- Produces (all in `plan.rs`):
  - `pub(crate) struct Map { pub(crate) l2p: Vec<u32>, pub(crate) p2l: Vec<u32> }`
  - `impl Map { pub(crate) fn from_l2p(l2p: &[u32]) -> Result<Map, DistError>; pub(crate) fn swap_phys(&mut self, a: u32, b: u32); pub(crate) fn remap_mask(&self, mask: u64) -> u64 }`
  - `pub(crate) fn to_physical(instr: &Instruction, map: &Map, n: u32) -> Option<Instruction>` — `Some` for `Gate` and
    `DiagonalPhase`, `None` otherwise.
  - `pub(crate) fn emit_exchange(bring: &[u32], victims: &[u32], layout: DistLayout, map: &mut Map, cur: &mut Vec<Instruction>, steps: &mut Vec<DistStep>, stats: &mut CommStats) -> Result<(), DistError>`
  - `pub fn plan_from(circuit: &Circuit, layout: DistLayout, router: Router, init: &[u32]) -> Result<DistPlan, DistError>`
  - `DistError::BadPlacement`

- [ ] **Step 1: Write failing tests** (append to `plan.rs` `mod tests`)

```rust
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
```

- [ ] **Step 2: Run them, expect a compile failure** (`plan_from` / `BadPlacement` undefined)

Run: `cargo test -p aleph-ir --lib dist::plan`
Expected: error[E0425] cannot find function `plan_from`.

- [ ] **Step 3: Implement**

In `mod.rs`:
- Add the `DistError` variant (after `TooFewLocalQubits`):

```rust
    #[error("initial placement is not a permutation of 0..n")]
    BadPlacement,
```

- Change `pub use plan::{plan, required_local};` to `pub use plan::{plan, plan_from, required_local};`.
- Add to the `DistPlan` doc comment: `/// Every plan assumes the |0…0⟩ initial state. It is invariant under qubit
  permutations, so a plan may start from a non-identity map (see [`plan_from`]) at no cost.`

In `plan.rs`, replace `struct Map` / `impl Map` with:

```rust
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
        Instruction::DiagonalPhase(dp) => Some(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
            n_qubits: n,
            terms: dp
                .terms
                .iter()
                .map(|t| PhaseTerm {
                    conds: t.conds.iter().map(|&c| map.remap_mask(c)).collect(),
                    angle: t.angle,
                })
                .collect(),
        }))),
        _ => None,
    }
}
```

Split `plan` into `plan` → `plan_from` → private `plan_in_order`:

```rust
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
    plan_in_order(circuit, layout, router, map)
}
```

`plan_in_order(circuit, layout, router, mut map: Map)` is the old body of `plan`:
- Drop the qubit-count check (now in `plan_from`) and drop `let mut map = Map::new(layout.n);`.
- The `Instruction::DiagonalPhase(dp)` arm becomes `cur.extend(to_physical(instr, &map, layout.n));`.
- The final `cur.push(Instruction::Gate(GateInstance { … }))` becomes `cur.extend(to_physical(instr, &map, layout.n));`.
- Delete `Map::new`. `remap_mask` is now only used through `to_physical`.

Extract the tail of `exchange_lookahead`, from `let k = bring.len() as u32;` down to the stats update, into:

```rust
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
```

`exchange_lookahead` then ends with:

```rust
    let chosen: SmallVec<[u32; 4]> = victims[..bring.len()].iter().map(|v| v.2).collect();
    emit_exchange(&bring, &chosen, layout, map, cur, steps, stats)
```

Leave `exchange_naive` unchanged. Its single-bit top-slot logic is the P6-02 baseline.

- [ ] **Step 4: Add the oracle proptest for arbitrary initial maps** (append to `crates/aleph-sv/tests/dist_oracle.rs`)

Add `plan_from` to the `use aleph_ir::dist::{…}` line, then:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn prop_any_initial_map_matches(
        gates in prop::collection::vec(arb_any_gate(6), 1..30),
        g in 0u32..=2,
        init in Just((0..6u32).collect::<Vec<u32>>()).prop_shuffle(),
        router in prop_oneof![Just(Router::Naive), Just(Router::Lookahead)],
    ) {
        let mut c = h_layer(6);
        for gi in gates {
            c.add_gate(gi).unwrap();
        }
        let p = plan_from(&c, DistLayout::new(6, g).unwrap(), router, &init).unwrap();
        let got = run_dist(&p).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}
```

- [ ] **Step 5: Run everything that touches the planner**

Run: `cargo test -p aleph-ir --lib dist && cargo test -p aleph-sv --test dist_oracle`
Expected: all PASS, including the pre-existing P6-02/P6-03 tests (the refactor must not change any plan).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-ir/src/dist/plan.rs crates/aleph-ir/src/dist/mod.rs crates/aleph-sv/tests/dist_oracle.rs
git commit -m "[P6-05] plan_from + shared Map/lowering/exchange emission

Refactor so the upcoming reordering scheduler reuses the in-order
planner's map, physical lowering and exchange emission. plan_from
starts from any initial map: free because plans start from |0…0>.

<attribution lines>"
```

---

### Task 2: `dag.rs`: commutation-respecting DAG

**Files:**
- Create: `crates/aleph-ir/src/dist/dag.rs`
- Modify: `crates/aleph-ir/src/dist/mod.rs` (`pub mod dag;`, `pub use dag::{Act, Dag};`)

**Interfaces:**
- Produces:
  - `pub enum Act { Z, X, Other }` (derive `Debug, Clone, Copy, PartialEq, Eq`)
  - `pub fn actions(instr: &Instruction) -> Result<SmallVec<[(u32, Act); 6]>, DistError>`
  - `pub struct Dag` with `pub fn build(c: &Circuit) -> Result<Dag, DistError>`, `pub fn len(&self) -> usize`,
    `pub fn is_empty(&self) -> bool`, `pub fn initial_ready(&self) -> Vec<usize>`, and
    `pub fn complete(&mut self, i: usize, ready: &mut Vec<usize>) -> Result<(), DistError>`, which pushes newly ready
    indices onto `ready`.
  - These are public, not `pub(crate)`, because the Task 3 soundness proptest in `aleph-sv` drives `Dag` directly.

- [ ] **Step 1: Write the module with its tests first** (implementation stubbed as `todo!()` so the tests compile and
  fail)

Create `crates/aleph-ir/src/dist/dag.rs`:

```rust
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
    todo!()
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
    pub fn build(c: &Circuit) -> Result<Self, DistError> {
        todo!()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Instructions with no predecessor, ascending.
    pub fn initial_ready(&self) -> Vec<usize> {
        todo!()
    }

    /// Mark ready instruction `i` scheduled; push newly ready ones onto `ready`.
    pub fn complete(&mut self, i: usize, ready: &mut Vec<usize>) -> Result<(), DistError> {
        todo!()
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
        assert_eq!(acts(g(Gate::Toffoli, &[0, 1, 2])), vec![(0, Z), (1, Z), (2, X)]);
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
                crate::PhaseTerm { conds: smallvec![0b0101], angle: 0.2 },
                crate::PhaseTerm { conds: smallvec![0b1000, 0b0001], angle: 0.1 },
            ],
        }));
        assert_eq!(acts(dp), vec![(0, Z), (2, Z), (3, Z)]);
        assert_eq!(acts(Instruction::Barrier(smallvec![1, 2])), vec![(1, Other), (2, Other)]);
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
```

Add `pub mod dag;` and `pub use dag::{Act, Dag};` to `mod.rs`, next to the other `mod` lines. If
`c.barrier([0u32])` doesn't match `Circuit::barrier`'s signature (`crates/aleph-ir/src/circuit.rs:439`), adapt the call
to that signature.

- [ ] **Step 2: Run, expect panics at `todo!()`**

Run: `cargo test -p aleph-ir --lib dist::dag`
Expected: every test FAILS with "not yet implemented".

- [ ] **Step 3: Implement**

```rust
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
```

`build`, `initial_ready`, `complete`:

```rust
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
```

Why waiting on the previous block alone is enough: block b−1's members were themselves only readied after block b−2
completed, so b−1 completing implies b−2 completed.

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-ir --lib dist::dag`
Expected: 9 PASS.

- [ ] **Step 5: Commit** (`[P6-05] Commutation-respecting dependency DAG (Z/X/Other blocks)` + attribution)

---

### Task 3: DAG soundness proptest over random topological orders

**Files:**
- Modify: `crates/aleph-sv/tests/dist_oracle.rs`

**Interfaces:**
- Consumes: `aleph_ir::dist::Dag::{build, initial_ready, complete}` from Task 2, and `arb_any_gate`, `h_layer`,
  `reference`, `TOL`, which already exist in this file.

- [ ] **Step 1: Write the test**

Add `Dag` to the `use aleph_ir::dist::{…}` import and `use aleph_ir::{DiagonalPhase, PhaseTerm};`, then append:

```rust
/// `c` re-emitted in a pseudo-random topological order of its DAG.
fn random_topo(c: &Circuit, seed: u64) -> Circuit {
    let mut dag = Dag::build(c).unwrap();
    let mut ready = dag.initial_ready();
    let mut out = Circuit::new(c.num_qubits(), 0);
    let mut s = seed | 1;
    while !ready.is_empty() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let i = ready.swap_remove((s >> 33) as usize % ready.len());
        out.add_instruction(c.instructions()[i].clone()).unwrap();
        dag.complete(i, &mut ready).unwrap();
    }
    assert_eq!(out.len(), c.len(), "DAG must schedule every instruction");
    out
}

fn arb_dp(n: u32) -> impl Strategy<Value = Instruction> {
    prop::collection::vec(
        (prop::collection::vec(1u64..(1u64 << n), 1..3), -3.0f64..3.0),
        1..4,
    )
    .prop_map(move |terms| {
        Instruction::DiagonalPhase(Box::new(DiagonalPhase {
            n_qubits: n,
            terms: terms
                .into_iter()
                .map(|(conds, angle)| PhaseTerm {
                    conds: conds.into_iter().collect(),
                    angle,
                })
                .collect(),
        }))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    /// Spec §7: any order the DAG allows is the same operator; random orders
    /// test the DAG itself, not one scheduler's choice.
    #[test]
    fn prop_every_dag_order_is_equivalent(
        gates in prop::collection::vec(arb_any_gate(6), 1..40),
        dp in arb_dp(6),
        dp_at in 0usize..40,
        seed in any::<u64>(),
    ) {
        let mut c = h_layer(6);
        for (k, gi) in gates.into_iter().enumerate() {
            if k == dp_at {
                c.add_instruction(dp.clone()).unwrap();
            }
            c.add_gate(gi).unwrap();
        }
        let want = reference(&c);
        for k in 0..8u64 {
            let got = reference(&random_topo(&c, seed ^ (k.wrapping_mul(0x9E37_79B9))));
            for (x, y) in got.iter().zip(&want) {
                prop_assert!((x - y).norm() < TOL);
            }
        }
    }
}

/// Mutation check: an unsound DAG (here, treating CNOT target as Z) would be
/// caught. Swapping the two CNOTs of `cx(0,1); cx(1,0)` changes the state.
#[test]
fn reordering_non_commuting_cnots_is_detectable() {
    let mut c = h_layer(2);
    c.rx(0.7, 0).unwrap();
    c.cnot(0, 1).unwrap();
    c.cnot(1, 0).unwrap();
    let mut swapped = h_layer(2);
    swapped.rx(0.7, 0).unwrap();
    swapped.cnot(1, 0).unwrap();
    swapped.cnot(0, 1).unwrap();
    let (a, b) = (reference(&c), reference(&swapped));
    assert!(a.iter().zip(&b).any(|(x, y)| (x - y).norm() > 1e-6));
}
```

- [ ] **Step 2: Run**

Run: `cargo test -p aleph-sv --test dist_oracle prop_every_dag_order reordering_non_commuting`
Expected: PASS. If the proptest fails, the DAG is unsound. Fix `actions` in Task 2 (shrink the counterexample and add
it as a `dag.rs` unit test). Never loosen the tolerance.

- [ ] **Step 3: Commit** (`[P6-05] DAG soundness proptest over random topological orders` + attribution)

---

### Task 4: `schedule.rs` + `Router::Reorder`

**Files:**
- Create: `crates/aleph-ir/src/dist/schedule.rs`
- Modify: `crates/aleph-ir/src/dist/mod.rs` (`mod schedule;`; the `Router::Reorder` variant)
- Modify: `crates/aleph-ir/src/dist/plan.rs` (`plan_from` dispatches `Reorder`)
- Modify: `crates/aleph-ir/src/dist/next_use.rs` (`next_unscheduled`)
- Test: `crates/aleph-ir/src/dist/schedule.rs` (unit), `crates/aleph-sv/tests/dist_oracle.rs`

**Interfaces:**
- Consumes: `Map`, `to_physical`, `emit_exchange`, `required_local` (Task 1); `Dag` (Task 2);
  `NextUse::{build, relabel}`.
- Produces:
  - `Router::Reorder { max_k: u32 }`. `max_k` is clamped to `1..=max(g, 1)`. It caps *prefetch*; a gate's own missing
    qubits are always all brought in.
  - `pub(crate) fn schedule(circuit: &Circuit, layout: DistLayout, max_k: u32, map: Map) -> Result<DistPlan, DistError>`
  - `pub(crate) fn NextUse::next_unscheduled(&mut self, q: u32, done: &[bool]) -> usize`

- [ ] **Step 1: Add the variant and the failing unit tests**

In `mod.rs`, extend `Router`:

```rust
    /// P6-05: reorder within commutation (see [`dag`]): run every gate that
    /// is runnable locally, exchange only when every ready gate is blocked.
    /// `max_k` caps the bits per exchange beyond what the blocked gate itself
    /// needs (clamped to `1..=max(g, 1)`).
    Reorder { max_k: u32 },
```

and add `mod schedule;`. In `plan_from`, replace the last line with:

```rust
    match router {
        Router::Reorder { max_k } => super::schedule::schedule(circuit, layout, max_k, map),
        Router::Naive | Router::Lookahead => plan_in_order(circuit, layout, router, map),
    }
```

In `plan_in_order`, the `match router` that builds `next_use` gets the arm `Router::Reorder { .. } => None`. It is
unreachable, but kept exhaustive.

Create `schedule.rs` with a stub and the tests:

```rust
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
    map: Map,
) -> Result<DistPlan, DistError> {
    todo!()
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
        let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Reorder { max_k: 2 }).unwrap();
        assert_eq!(exch(&p), 0);
        check_invariants(&c, &p);
    }

    #[test]
    fn invariants_on_standard_circuits() {
        for (c, g) in [(ghz(10), 2), (qft(10), 3), (brick(10, 6), 2), (brick(9, 4), 3)] {
            for k in 1..=3 {
                let p = plan(&c, DistLayout::new(c.num_qubits(), g).unwrap(), Router::Reorder { max_k: k })
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
                assert!(exch(&re) <= exch(&la), "g={g}: {} > {}", exch(&re), exch(&la));
            }
        }
    }

    #[test]
    fn missing_exceeding_max_k_still_exchanges() {
        // Iswap on logical 4,5: both global at n=6, g=2. One gate needs 2 bits.
        let mut c = Circuit::new(6, 0);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![4, 5])).unwrap();
        let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Reorder { max_k: 1 }).unwrap();
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
            let p = plan(&c, DistLayout::new(6, 0).unwrap(), Router::Reorder { max_k: k }).unwrap();
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
        let p = plan(&c, DistLayout::new(4, 1).unwrap(), Router::Reorder { max_k: 1 }).unwrap();
        assert!(matches!(p.steps[0], DistStep::Local(ref v) if v.len() == 1));
        assert!(matches!(p.steps[1], DistStep::Exchange { .. }));
    }

    #[test]
    fn barrier_fences_reordering() {
        let mut c = Circuit::new(4, 0);
        c.rx(0.3, 3).unwrap();
        c.barrier([0u32, 1, 2, 3]).unwrap();
        c.h(0).unwrap();
        let p = plan(&c, DistLayout::new(4, 1).unwrap(), Router::Reorder { max_k: 1 }).unwrap();
        assert!(matches!(p.steps[0], DistStep::Exchange { .. }));
    }

    #[test]
    fn measure_is_rejected_not_panicking() {
        let mut c = Circuit::new(4, 1);
        c.h(0).unwrap();
        c.measure(0, 0).unwrap();
        assert!(plan(&c, DistLayout::new(4, 1).unwrap(), Router::Reorder { max_k: 1 }).is_err());
    }
}
```

Adapt `c.barrier(...)` to the real signature exactly as in Task 2.

- [ ] **Step 2: Run, expect `todo!()` failures**

Run: `cargo test -p aleph-ir --lib dist::schedule`
Expected: FAIL ("not yet implemented").

- [ ] **Step 3: Implement `NextUse::next_unscheduled`** (in `next_use.rs`)

```rust
    /// Smallest not-yet-`done` index at which the data now under label `q`
    /// must be local; `usize::MAX` if none. Used by the reordering scheduler,
    /// where "next" means next *unscheduled*, not next in circuit order.
    pub(crate) fn next_unscheduled(&mut self, q: u32, done: &[bool]) -> usize {
        let t = self.track[q as usize] as usize;
        let (p, c) = (&self.pos[t], &mut self.cursor[t]);
        while *c < p.len() && done[p[*c]] {
            *c += 1;
        }
        p.get(*c).copied().unwrap_or(usize::MAX)
    }
```

(`Reorder` never calls `next`, so the shared cursor is safe: entries before the cursor are always `done`.)

- [ ] **Step 4: Implement `schedule`**

```rust
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
            return Err(DistError::TooFewLocalQubits { need: need.len(), m });
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
        emit_exchange(&bring, &chosen, layout, &mut map, &mut cur, &mut steps, &mut stats)?;
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
```

Progress guarantee (put it in a comment above the exchange block): after the exchange, `seed`'s missing qubits are
local, and none of its required qubits were victims. So the next iteration finds `seed` runnable and the loop
terminates.

- [ ] **Step 5: Run unit tests**

Run: `cargo test -p aleph-ir --lib dist`
Expected: all PASS (P6-02/03 tests included).

`never_more_exchanges_than_lookahead_on_ghz_qft` is a heuristic pin (spec §7), not a theorem. If it fails, **stop and
report the counts to the user. Do not weaken or delete the assertion.** A loss there is a design finding for the spec.
It is not a test to bend.

- [ ] **Step 6: Extend the CPU oracle to `Reorder`** (in `crates/aleph-sv/tests/dist_oracle.rs`)

In `check`, change the router list to:

```rust
    for router in [
        Router::Naive,
        Router::Lookahead,
        Router::Reorder { max_k: 1 },
        Router::Reorder { max_k: 3 },
    ] {
```

In **every** `proptest!` that has a `router in prop_oneof![…]` strategy (`prop_random_circuits_match`,
`prop_all_gate_families_with_controls`, and Task 1's `prop_any_initial_map_matches`), use:

```rust
        router in prop_oneof![
            Just(Router::Naive),
            Just(Router::Lookahead),
            (1u32..=3).prop_map(|max_k| Router::Reorder { max_k }),
        ],
```

- [ ] **Step 7: Run the oracle**

Run: `cargo test -p aleph-sv --test dist_oracle`
Expected: all PASS.

- [ ] **Step 8: Commit** (`[P6-05] Reordering list scheduler: Router::Reorder` + attribution)

---

### Task 5: `placement.rs`: initial global-qubit choice

**Files:**
- Create: `crates/aleph-ir/src/dist/placement.rs`
- Modify: `crates/aleph-ir/src/dist/mod.rs` (`mod placement; pub use placement::initial_placement;`)
- Test: `placement.rs` (unit), `crates/aleph-sv/tests/dist_oracle.rs`

**Interfaces:**
- Consumes: `required_local` (Task 1).
- Produces: `pub fn initial_placement(circuit: &Circuit, layout: DistLayout) -> Vec<u32>`. Returns `l2p`, which can be
  passed straight to `plan_from`.

- [ ] **Step 1: Write the module with failing tests**

```rust
//! P6-05 initial placement: start the `g` qubits that are needed local *last*
//! as the global ones (ties: fewest local needs, then highest index). Free,
//! because plans start from |0…0⟩.

use aleph_core::Gate;

use super::plan::required_local;
use super::DistLayout;
use crate::{Circuit, Instruction};

/// `l2p[logical] = physical`: chosen globals at `m..n` (ascending logical),
/// the rest at `0..m` in logical order.
pub fn initial_placement(circuit: &Circuit, layout: DistLayout) -> Vec<u32> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{plan, plan_from, Router};

    #[test]
    fn unused_qubits_go_global() {
        let mut c = Circuit::new(6, 0);
        for q in 0..4 {
            c.h(q).unwrap();
        }
        let l = DistLayout::new(6, 2).unwrap();
        assert_eq!(initial_placement(&c, l), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn latest_first_need_goes_global() {
        let mut c = Circuit::new(6, 0);
        for q in [5u32, 4, 3, 0, 1, 2] {
            c.h(q).unwrap();
        }
        let l = DistLayout::new(6, 2).unwrap();
        // First needs: 5@0 4@1 3@2 0@3 1@4 2@5 → globals {1, 2}.
        assert_eq!(initial_placement(&c, l), vec![0, 4, 5, 1, 2, 3]);
        let id = plan(&c, l, Router::Naive).unwrap().stats.exchanges;
        let placed = plan_from(&c, l, Router::Naive, &initial_placement(&c, l))
            .unwrap()
            .stats
            .exchanges;
        assert!(placed <= id);
    }

    #[test]
    fn diagonal_and_control_uses_do_not_count() {
        let mut c = Circuit::new(4, 0);
        c.rz(0.1, 3).unwrap(); // diagonal: never needs local
        c.cnot(3, 0).unwrap(); // 3 is only a control
        c.h(1).unwrap();
        c.h(2).unwrap();
        let l = DistLayout::new(4, 1).unwrap();
        assert_eq!(initial_placement(&c, l), vec![0, 1, 2, 3]);
    }

    #[test]
    fn g0_is_identity() {
        let mut c = Circuit::new(3, 0);
        c.h(2).unwrap();
        assert_eq!(initial_placement(&c, DistLayout::new(3, 0).unwrap()), vec![0, 1, 2]);
    }

    #[test]
    fn relabel_swaps_are_followed() {
        // swap(0, 3) then H on label 3: the data needed is track 0's.
        let mut c = Circuit::new(4, 0);
        c.swap(0, 3).unwrap();
        c.h(3).unwrap();
        c.h(1).unwrap();
        c.h(2).unwrap();
        let l = DistLayout::new(4, 1).unwrap();
        // Track 3 (label 3 at t=0) is never needed: it goes global.
        assert_eq!(initial_placement(&c, l), vec![0, 1, 2, 3]);
    }
}
```

- [ ] **Step 2: Run, expect failures** — `cargo test -p aleph-ir --lib dist::placement` → FAIL (`todo!()`).

- [ ] **Step 3: Implement**

```rust
pub fn initial_placement(circuit: &Circuit, layout: DistLayout) -> Vec<u32> {
    let n = layout.n as usize;
    let mut first = vec![usize::MAX; n];
    let mut uses = vec![0u32; n];
    // track[label] = which t=0 data the label holds (relabel swaps move data).
    let mut track: Vec<usize> = (0..n).collect();
    for (i, instr) in circuit.instructions().iter().enumerate() {
        let Instruction::Gate(g) = instr else { continue };
        if matches!(g.gate, Gate::Swap) && g.controls.is_empty() {
            track.swap(g.qubits[0] as usize, g.qubits[1] as usize);
            continue;
        }
        for q in required_local(g) {
            let t = track[q as usize];
            first[t] = first[t].min(i);
            uses[t] += 1;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        first[b]
            .cmp(&first[a])
            .then(uses[a].cmp(&uses[b]))
            .then(b.cmp(&a))
    });
    let mut global: Vec<usize> = order[..layout.g as usize].to_vec();
    global.sort_unstable();
    let mut l2p = vec![0u32; n];
    let mut next_local = 0u32;
    for (q, slot) in l2p.iter_mut().enumerate() {
        if !global.contains(&q) {
            *slot = next_local;
            next_local += 1;
        }
    }
    for (j, &q) in global.iter().enumerate() {
        l2p[q] = layout.m() + j as u32;
    }
    l2p
}
```

- [ ] **Step 4: Run** — `cargo test -p aleph-ir --lib dist::placement` → 5 PASS.

- [ ] **Step 5: Oracle proptest with the chosen placement for every router** (append to `dist_oracle.rs`; add
  `initial_placement` to the import)

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn prop_initial_placement_matches(
        gates in prop::collection::vec(arb_any_gate(6), 1..30),
        g in 0u32..=3,
        router in prop_oneof![
            Just(Router::Naive),
            Just(Router::Lookahead),
            (1u32..=3).prop_map(|max_k| Router::Reorder { max_k }),
        ],
    ) {
        let mut c = h_layer(6);
        for gi in gates {
            c.add_gate(gi).unwrap();
        }
        let l = DistLayout::new(6, g).unwrap();
        let p = plan_from(&c, l, router, &initial_placement(&c, l)).unwrap();
        let got = run_dist(&p).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}
```

Note: `h_layer` puts an `H` (needs local) on every qubit first, so this mostly exercises the correctness plumbing. The
non-trivial placements are covered by Task 1's random-permutation proptest.

- [ ] **Step 6: Run** — `cargo test -p aleph-sv --test dist_oracle` → all PASS.

- [ ] **Step 7: Commit** (`[P6-05] Initial global-qubit placement` + attribution)

---

### Task 6: Comm-stats table, report stub, full verification, PR

**Files:**
- Modify: `crates/aleph-sv/examples/dist_comm_counts.rs`
- Create: `docs/perf/p6-05-compiler.md`

**Interfaces:**
- Consumes: `plan`, `plan_from`, `initial_placement`, `Router::Reorder`.

- [ ] **Step 1: Extend the example** (replace `main` in `dist_comm_counts.rs`; add `plan_from, initial_placement` to
  the import)

```rust
fn main() {
    let cases: Vec<(&str, Circuit)> = vec![
        ("GHZ-32", ghz(32)),
        ("QFT-32", qft(32)),
        ("random-30 d=20", brickwall(30, 20)),
        ("Grover-20 (5 iters)", grover()),
    ];
    println!("| circuit | g | strategy | exch | × slice | local swaps | vs lookahead |");
    println!("|---|---|---|---|---|---|---|");
    for (name, c) in &cases {
        for g in [2u32, 3] {
            let l = DistLayout::new(c.num_qubits(), g).unwrap();
            let slice = (1u64 << l.m()) as f64;
            let placed = initial_placement(c, l);
            let la = plan(c, l, Router::Lookahead).unwrap().stats;
            let la_s = la.amps_moved_per_rank as f64 / slice;
            let rows = [
                ("naive", plan(c, l, Router::Naive).unwrap().stats),
                ("lookahead", la),
                ("reorder", plan(c, l, Router::Reorder { max_k: g }).unwrap().stats),
                ("reorder k=1", plan(c, l, Router::Reorder { max_k: 1 }).unwrap().stats),
                (
                    "reorder+place",
                    plan_from(c, l, Router::Reorder { max_k: g }, &placed).unwrap().stats,
                ),
            ];
            for (s, st) in rows {
                let x = st.amps_moved_per_rank as f64 / slice;
                println!(
                    "| {name} | {g} | {s} | {} | {x:.1} | {} | {:.2}× |",
                    st.exchanges,
                    st.local_swaps,
                    la_s / x.max(f64::MIN_POSITIVE)
                );
            }
        }
    }
}
```

- [ ] **Step 2: Run it from the workspace root**

Run: `cargo run --release -p aleph-sv --example dist_comm_counts`
Expected: a 40-row markdown table. Save the output for Step 3.

- [ ] **Step 3: Write `docs/perf/p6-05-compiler.md`**

Use this skeleton. Paste in the Step 2 table, and write the reading **from the actual numbers**: state which circuits
reorder helps or hurts, and by how much. Do not claim GPU speedups; those come in PR 3.

```markdown
# P6-05: Communication-aware compiler for distributed SV

Spec: `docs/superpowers/specs/2026-10-04-p6-05-comm-aware-compiler-design.md`. Issue #59.

This report grows over three PRs.
- PR 1 (this one) adds the commutation DAG, the reordering scheduler (`Router::Reorder`) and initial placement, and
  reports CPU communication counts.
- PR 2 adds the calibrated GPU cost model.
- PR 3 adds `compile` and the predicted-time benchmark.

## 1. Communication counts (CPU, PR 1)

`cargo run --release -p aleph-sv --example dist_comm_counts`

<table from Step 2>

`× slice` is the amplitudes moved per rank divided by the slice size `2^m`. `vs lookahead` is lookahead's moved volume
divided by this row's (> 1 = less traffic).

**Reading.**
- <circuit-by-circuit, from the numbers>
- Comm volume is only half of the objective. Reordering also lengthens local segments, which matters for per-rank
  fusion, and that is measured in PR 3. `compile` keeps Lookahead as a candidate, so a row where reorder moves more
  data is not a regression of the final compiler.
```

- [ ] **Step 4: Full verification**

```bash
export PATH="$(brew --prefix rustup)/bin:$PATH"
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo +nightly clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep -E "FAILED|panicked|failed;" | grep -v " 0 failed"; echo "grep-exit=$?"
```

Expected: fmt/clippy silent. `grep-exit=1`, meaning no failure lines.

- [ ] **Step 5: Commit, push, open the PR**

```bash
git add crates/aleph-sv/examples/dist_comm_counts.rs docs/perf/p6-05-compiler.md
git commit -m "[P6-05] Comm-count table for Reorder/placement + report §1

<attribution lines>"
git push -u origin p6-05-comm-aware-compiler
gh pr create --title "[P6-05] Commutation DAG + reordering scheduler + placement (1/3)" --body "<body>"
```

PR body sections:
- **Summary**: links the spec and plan; this is PR 1/3, `Refs #59`.
- **Approach**: DAG, scheduler, placement, `plan_from`.
- **Tests**: list the unit tests and proptests, with the DAG random-topological-order proptest called out.
- **Comm-count table**: paste it.
- **Notes**: no GPU or perf claim in this PR. PR 2 (cost model) and PR 3 (compile + bench) follow.
- End with the Claude Code attribution lines from Global Constraints.
