# Sparse Blossom MWPM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-shot textbook blossom in `MwpmDecoder` with an event-driven, region-growing Sparse Blossom matcher that is weight-identical to `decode_dense`, ≥ 10× faster at d = 11, and scales to high distance.

**Architecture:** A compiled CSR detector graph (weights doubled so every event time is an integer); a per-shot `State` (node arrays reset via a touched list, region/tree arenas, one binary heap of lazily-validated events); a *flooder* that grows/shrinks regions and detects collisions; a *matcher* that runs Edmonds' tree operations on regions (augment, grow, blossom, shatter, degenerate implosion); a final recursive resolution that turns top-level matches into defect–defect edges carrying `(obs, weight)`. `MwpmDecoder` keeps `decode_dense`/`decode_local` as oracles behind lazily-built all-pairs tables and hands `decode` to the sparse matcher through a mutex pool of states.

**Tech Stack:** Rust 2021 (workspace MSRV 1.89), `std` only (no new deps), `proptest` (dev), criterion (`benches/`).

**Spec:** `docs/superpowers/specs/2026-09-24-sparse-blossom-design.md`

## Global Constraints

- No new crate dependencies (spec §1).
- No `unwrap()`/`expect()` in library code; `?` with `thiserror` types (CLAUDE.md). Decoding itself cannot fail.
- No float comparison gating correctness: all matching arithmetic on `i64` (CLAUDE.md / ADR 0006).
- Public API of `MwpmDecoder` (`new`, `from_graph`, `decode`, `decode_dense`, `with_locality_k`) and the Python decoder name `"mwpm"` unchanged (spec §1).
- Every task ends with `cargo fmt`, `cargo clippy -p aleph-qec --all-targets -- -D warnings` and `cargo +nightly clippy` (CI runs beta; memory `feedback-clippy-beta-lints`) green, and a commit on branch `q1-03b-sparse-blossom`.
- Debug `cargo test -p aleph-qec` is very slow (relay_window sims): run the crate's tests with `--release` or filter by name.
- Rust on this Mac: `export PATH=/opt/homebrew/opt/rustup/bin:$PATH` in every shell.
- Do not read PyMatching / Blossom V source (CLAUDE.md: never copy code).

## Review Focus

1. Zero-weight edges (a mechanism with `p ≈ 0.5` rounds to weight 0): two defects at distance 0 must still match with weight 0 and the shrink handler must not spin. Test pinned in Task 4 (`zero_weight_edge_matches_at_time_zero`).
2. Simultaneous events (many equal-weight edges, the surface-code regime): three mutually equidistant defects tie at the same time; the result must still equal the dense optimum. Pinned in Task 4 (`equilateral_triangle_with_boundary`).
3. Duplicate or out-of-range detector indices in a `Syndrome`: `defects_of` filters range; duplicates must not create two regions on one node. Pinned in Task 6 (`duplicate_defects_are_ignored`).
4. An odd component with no boundary (unmatchable defect): the heap drains with a live tree; result must be best-effort, no panic, no infinite loop. Pinned in Task 4 (`odd_component_without_boundary_is_best_effort`).
5. Re-entrancy across threads and shots (state pool reuse): decoding the same syndrome twice and from several threads gives identical output. Pinned in Task 6 (`pool_reuse_is_deterministic`).

---

### Task 1: Compiled CSR graph

**Files:**
- Create: `crates/aleph-qec/src/sparse_blossom/mod.rs`
- Create: `crates/aleph-qec/src/sparse_blossom/graph.rs`
- Modify: `crates/aleph-qec/src/lib.rs` (add `mod sparse_blossom;`)

**Interfaces:**
- Consumes: `crate::matching::MatchingGraph` (`edges()`, `num_detectors()`, `num_observables()`, `boundary()`; `MatchingEdge { a, b, weight: f64, observables: Vec<u32> }`).
- Produces: `pub(crate) const WEIGHT_SCALE: f64`; `pub(crate) struct CompiledGraph`; `CompiledGraph::from_matching_graph(&MatchingGraph) -> CompiledGraph`; `CompiledGraph::from_int_edges(num_nodes, &[(u32,u32,i64,u64)], &[(u32,i64,u64)]) -> CompiledGraph` (test constructor, raw integer weights); `fn num_nodes(&self) -> usize`; `fn num_observables(&self) -> usize`; `fn edges(&self, u: u32) -> &[Edge]`; `fn boundary(&self, u: u32) -> Option<(i64, u64)>`; `pub(crate) struct Edge { pub v: u32, pub w: i64, pub obs: u64 }`. All stored weights are **doubled** (`2 * round(w * WEIGHT_SCALE)`).

- [ ] **Step 1: Write the failing tests**

`crates/aleph-qec/src/sparse_blossom/graph.rs` (tests module at the bottom):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::{DemError, DetectorErrorModel};
    use crate::matching::MatchingGraph;

    fn graph(errors: Vec<DemError>) -> CompiledGraph {
        let dem = DetectorErrorModel { detectors: 3, observables: 2, errors };
        CompiledGraph::from_matching_graph(&MatchingGraph::from_dem(&dem).unwrap())
    }

    #[test]
    fn csr_holds_each_undirected_edge_on_both_endpoints_with_doubled_weight() {
        let g = graph(vec![DemError::new(0.1, vec![0, 1], vec![1])]);
        let w = 2 * ((0.9f64 / 0.1).ln() * WEIGHT_SCALE).round() as i64;
        assert_eq!(g.edges(0), &[Edge { v: 1, w, obs: 0b10 }]);
        assert_eq!(g.edges(1), &[Edge { v: 0, w, obs: 0b10 }]);
        assert!(g.edges(2).is_empty());
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_observables(), 2);
    }

    #[test]
    fn parallel_edges_keep_the_lightest_one() {
        // Same endpoints, different observables ⇒ MatchingGraph keeps both; the compiled graph
        // keeps the lighter (more probable) one, as the all-pairs Dijkstra implicitly did.
        let g = graph(vec![
            DemError::new(0.05, vec![0, 1], vec![0]),
            DemError::new(0.2, vec![0, 1], vec![1]),
        ]);
        assert_eq!(g.edges(0).len(), 1);
        assert_eq!(g.edges(0)[0].obs, 0b10);
    }

    #[test]
    fn boundary_edge_is_per_node_min_weight() {
        let g = graph(vec![
            DemError::new(0.05, vec![2], vec![0]),
            DemError::new(0.3, vec![2], vec![]),
            DemError::new(0.1, vec![0, 2], vec![]),
        ]);
        let wb = 2 * ((0.7f64 / 0.3).ln() * WEIGHT_SCALE).round() as i64;
        assert_eq!(g.boundary(2), Some((wb, 0)));
        assert_eq!(g.boundary(0), None);
    }

    #[test]
    fn from_int_edges_doubles_and_mirrors() {
        let g = CompiledGraph::from_int_edges(2, &[(0, 1, 3, 1)], &[(1, 5, 2)]);
        assert_eq!(g.edges(0), &[Edge { v: 1, w: 6, obs: 1 }]);
        assert_eq!(g.edges(1), &[Edge { v: 0, w: 6, obs: 1 }]);
        assert_eq!(g.boundary(1), Some((10, 2)));
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p aleph-qec sparse_blossom::graph`
Expected: compile error (module missing).

- [ ] **Step 3: Implement**

`crates/aleph-qec/src/sparse_blossom/mod.rs`:

```rust
//! Sparse Blossom: minimum-weight perfect matching by local, event-driven region growth on the
//! detector graph (Higgott & Gidney, arXiv:2303.15933), re-derived from Edmonds' primal-dual
//! blossom algorithm. See `docs/superpowers/specs/2026-09-24-sparse-blossom-design.md`.
//!
//! Module map: [`graph`] compiles the [`crate::MatchingGraph`] into a CSR with doubled integer
//! weights; [`state`] holds the per-shot mutable arenas and the event heap; [`flooder`] grows and
//! shrinks regions and detects collisions; [`matcher`] runs the alternating-tree operations and
//! resolves the final matching. [`SparseMatcher`] is the crate-facing entry point.

pub(crate) mod graph;

pub(crate) use graph::{CompiledGraph, WEIGHT_SCALE};
```

`crates/aleph-qec/src/sparse_blossom/graph.rs`:

```rust
//! Compiled detector graph for the sparse matcher: CSR adjacency with integer weights.
//!
//! Weights are `2 · round(w · WEIGHT_SCALE)`. Doubling makes every event time an integer: with
//! integer weights Edmonds' duals are half-integral (all outer regions grow at the same global
//! rate, so any two-region collision splits an integer gap in two), and doubling clears the
//! half. The matcher divides its total weight by 2 on output so it is comparable with the dense
//! path's `round(w · WEIGHT_SCALE)` sums.

use crate::matching::MatchingGraph;

/// Fixed-point scale for real edge weights `ln((1-p)/p)`; ~7 significant digits, far inside
/// `i64` for the largest path sums at the distances we target.
pub(crate) const WEIGHT_SCALE: f64 = (1u64 << 24) as f64;

/// One directed half of an undirected edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Edge {
    pub v: u32,
    pub w: i64,
    pub obs: u64,
}

/// CSR detector graph plus a per-node boundary edge.
#[derive(Clone, Debug)]
pub(crate) struct CompiledGraph {
    num_nodes: usize,
    num_observables: usize,
    offsets: Vec<u32>,
    adj: Vec<Edge>,
    /// Boundary edge per node: `(doubled weight, obs)`; weight `i64::MAX` when absent.
    boundary_w: Vec<i64>,
    boundary_obs: Vec<u64>,
}

impl CompiledGraph {
    /// Compile `g`: parallel detector–detector edges collapse to the lightest (ties by edge
    /// index), boundary edges to the lightest per node.
    pub(crate) fn from_matching_graph(g: &MatchingGraph) -> Self {
        let scaled = |w: f64| 2 * (w * WEIGHT_SCALE).round() as i64;
        let boundary = g.boundary();
        let edges: Vec<(u32, u32, i64, u64)> = g
            .edges()
            .iter()
            .filter(|e| e.b != boundary)
            .map(|e| (e.a as u32, e.b as u32, scaled(e.weight), obs_mask(&e.observables)))
            .collect();
        let bnd: Vec<(u32, i64, u64)> = g
            .edges()
            .iter()
            .filter(|e| e.b == boundary)
            .map(|e| (e.a as u32, scaled(e.weight), obs_mask(&e.observables)))
            .collect();
        Self::build(g.num_detectors(), g.num_observables(), &edges, &bnd)
    }

    /// Test constructor from raw integer weights (doubled internally).
    #[cfg(test)]
    pub(crate) fn from_int_edges(
        num_nodes: usize,
        edges: &[(u32, u32, i64, u64)],
        boundary: &[(u32, i64, u64)],
    ) -> Self {
        let e: Vec<_> = edges.iter().map(|&(a, b, w, o)| (a, b, 2 * w, o)).collect();
        let b: Vec<_> = boundary.iter().map(|&(a, w, o)| (a, 2 * w, o)).collect();
        Self::build(num_nodes, 64, &e, &b)
    }

    fn build(
        num_nodes: usize,
        num_observables: usize,
        edges: &[(u32, u32, i64, u64)],
        boundary: &[(u32, i64, u64)],
    ) -> Self {
        // Per node: (neighbour, weight, edge index, obs) sorted so the lightest parallel edge
        // comes first, then dedup by neighbour.
        let mut per: Vec<Vec<(u32, i64, usize, u64)>> = vec![Vec::new(); num_nodes];
        for (k, &(a, b, w, o)) in edges.iter().enumerate() {
            per[a as usize].push((b, w, k, o));
            per[b as usize].push((a, w, k, o));
        }
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut adj = Vec::new();
        offsets.push(0u32);
        for list in per.iter_mut() {
            list.sort_unstable_by_key(|&(v, w, k, _)| (v, w, k));
            list.dedup_by_key(|x| x.0);
            adj.extend(list.iter().map(|&(v, w, _, obs)| Edge { v, w, obs }));
            offsets.push(adj.len() as u32);
        }
        let mut boundary_w = vec![i64::MAX; num_nodes];
        let mut boundary_obs = vec![0u64; num_nodes];
        for &(a, w, o) in boundary {
            let a = a as usize;
            if w < boundary_w[a] {
                boundary_w[a] = w;
                boundary_obs[a] = o;
            }
        }
        CompiledGraph { num_nodes, num_observables, offsets, adj, boundary_w, boundary_obs }
    }

    pub(crate) fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub(crate) fn num_observables(&self) -> usize {
        self.num_observables
    }

    #[inline]
    pub(crate) fn edges(&self, u: u32) -> &[Edge] {
        let u = u as usize;
        &self.adj[self.offsets[u] as usize..self.offsets[u + 1] as usize]
    }

    #[inline]
    pub(crate) fn boundary(&self, u: u32) -> Option<(i64, u64)> {
        let w = self.boundary_w[u as usize];
        (w != i64::MAX).then_some((w, self.boundary_obs[u as usize]))
    }
}

fn obs_mask(observables: &[u32]) -> u64 {
    observables.iter().fold(0u64, |m, &o| m | (1u64 << o))
}
```

Add `mod sparse_blossom;` to `crates/aleph-qec/src/lib.rs` next to the other private modules (e.g. after `mod blossom;`). Until Task 6 wires it in, mark the module `#[allow(dead_code)]` at the `mod` line so clippy stays green.

- [ ] **Step 4: Run tests, fmt, clippy**

Run: `cargo test --release -p aleph-qec sparse_blossom::graph && cargo fmt && cargo clippy -p aleph-qec --all-targets -- -D warnings`
Expected: 4 passed; no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-qec/src/sparse_blossom crates/aleph-qec/src/lib.rs
git commit -m "[Q1-03b] sparse_blossom: compiled CSR graph with doubled integer weights"
```

---

### Task 2: Per-shot state, affine radii, compressed edges, event heap

**Files:**
- Create: `crates/aleph-qec/src/sparse_blossom/state.rs`
- Modify: `crates/aleph-qec/src/sparse_blossom/mod.rs` (add `pub(crate) mod state;`)

**Interfaces:**
- Produces (all `pub(crate)`): `type NodeId = u32; type RegionId = u32; type TreeId = u32; const NONE: u32; const BOUNDARY: u32; const NO_TIME: i64;` `struct Varying { y0: i64, slope: i8 }` with `frozen(v)`, `at(t)`, `with_slope(t, slope)`, `time_of(target, now) -> Option<i64>`; `struct CEdge { from: NodeId, to: NodeId, obs: u64, weight: i64 }` with `reversed()`, `then(next)`; `struct NodeState { region, source, dist, obs, arrival_z, wrapped, queued }`; `struct Region { radius, blossom_parent, tree, matched_to, match_edge, children: Vec<(RegionId, CEdge)>, shell: Vec<NodeId>, source, shrink_queued }`; `struct TreeNode { inner, outer, inner_to_outer, parent, parent_edge, children: Vec<TreeId>, alive }`; `struct Stats { events, pushes }`; `struct State { nodes, touched, regions, trees, heap, seq, now, active_trees, scratch, stats }` with `new(num_nodes)`, `reset()`, `push_event(t, kind, id)`, `pop_event() -> Option<(i64, u8, u32)>`; event kinds `EV_NODE: u8 = 0`, `EV_SHRINK: u8 = 1`.

- [ ] **Step 1: Write the failing tests** (tests module in `state.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varying_evaluates_and_changes_slope_continuously() {
        let r = Varying { y0: 0, slope: 1 };
        assert_eq!(r.at(7), 7);
        let f = r.with_slope(7, 0);
        assert_eq!((f.at(7), f.at(100)), (7, 7));
        let s = f.with_slope(10, -1);
        assert_eq!((s.at(10), s.at(13)), (7, 4));
        assert_eq!(s.time_of(0, 10), Some(17));
        assert_eq!(f.time_of(9, 10), None);
        assert_eq!(r.time_of(3, 5), None); // already past
    }

    #[test]
    fn cedge_reverses_and_chains() {
        let a = CEdge { from: 1, to: 2, obs: 0b01, weight: 3 };
        let b = CEdge { from: 2, to: 5, obs: 0b11, weight: 4 };
        assert_eq!(a.reversed(), CEdge { from: 2, to: 1, obs: 0b01, weight: 3 });
        assert_eq!(a.then(b), CEdge { from: 1, to: 5, obs: 0b10, weight: 7 });
    }

    #[test]
    fn heap_pops_in_time_then_insertion_order() {
        let mut st = State::new(4);
        st.push_event(5, EV_NODE, 1);
        st.push_event(2, EV_SHRINK, 9);
        st.push_event(5, EV_NODE, 0);
        assert_eq!(st.pop_event(), Some((2, EV_SHRINK, 9)));
        assert_eq!(st.pop_event(), Some((5, EV_NODE, 1)));
        assert_eq!(st.pop_event(), Some((5, EV_NODE, 0)));
        assert_eq!(st.pop_event(), None);
        assert_eq!(st.stats.pushes, 3);
    }

    #[test]
    fn reset_clears_touched_nodes_only() {
        let mut st = State::new(3);
        st.nodes[2].region = 7;
        st.touched.push(2);
        st.regions.push(Region::leaf(2));
        st.reset();
        assert_eq!(st.nodes[2].region, NONE);
        assert!(st.regions.is_empty() && st.touched.is_empty());
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p aleph-qec sparse_blossom::state`
Expected: compile error.

- [ ] **Step 3: Implement**

```rust
//! Per-shot mutable state of the sparse matcher: node records, region and tree arenas, and the
//! event heap. Everything here is plain data; the flooder and matcher own the logic.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub(crate) type NodeId = u32;
pub(crate) type RegionId = u32;
pub(crate) type TreeId = u32;
/// "No node / region / tree".
pub(crate) const NONE: u32 = u32::MAX;
/// Pseudo-region: matched to the boundary. Also the `to` of a boundary compressed edge.
pub(crate) const BOUNDARY: u32 = u32::MAX - 1;
/// "No event queued".
pub(crate) const NO_TIME: i64 = i64::MIN;
pub(crate) const EV_NODE: u8 = 0;
pub(crate) const EV_SHRINK: u8 = 1;

/// A radius that is affine in the global clock: `value(t) = y0 + slope · t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Varying {
    pub y0: i64,
    pub slope: i8,
}

impl Varying {
    pub(crate) fn frozen(v: i64) -> Self {
        Varying { y0: v, slope: 0 }
    }
    #[inline]
    pub(crate) fn at(self, t: i64) -> i64 {
        self.y0 + self.slope as i64 * t
    }
    /// Same value at `t`, new slope from `t` on.
    pub(crate) fn with_slope(self, t: i64, slope: i8) -> Self {
        Varying { y0: self.at(t) - slope as i64 * t, slope }
    }
    /// Time `≥ now` at which the value equals `target`, if the slope ever gets there.
    pub(crate) fn time_of(self, target: i64, now: i64) -> Option<i64> {
        let t = match self.slope {
            1 => target - self.y0,
            -1 => self.y0 - target,
            _ => return None,
        };
        (t >= now).then_some(t)
    }
}

/// A path between two defects (or a defect and the boundary), compressed to its endpoints, the
/// observable parity along it and its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub obs: u64,
    pub weight: i64,
}

impl CEdge {
    pub(crate) const EMPTY: CEdge = CEdge { from: NONE, to: NONE, obs: 0, weight: 0 };
    pub(crate) fn reversed(self) -> Self {
        CEdge { from: self.to, to: self.from, ..self }
    }
    /// Concatenate `self` (ending at `x`) with `next` (starting at `x`).
    pub(crate) fn then(self, next: CEdge) -> Self {
        debug_assert_eq!(self.to, next.from);
        CEdge { from: self.from, to: next.to, obs: self.obs ^ next.obs, weight: self.weight + next.weight }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeState {
    /// Bottom-level region owning this node, or `NONE`.
    pub region: RegionId,
    /// The defect whose growth reached this node.
    pub source: NodeId,
    /// Distance from `source` along the growth path.
    pub dist: i64,
    /// Observable parity from `source`.
    pub obs: u64,
    /// The owning region's own radius when it reached this node (release order under shrinking).
    pub arrival_z: i64,
    /// Sum of the frozen radii of the region chain below the top-level region.
    pub wrapped: i64,
    /// Time of the queued look-at event, or `NO_TIME`.
    pub queued: i64,
}

impl NodeState {
    pub(crate) const FREE: NodeState =
        NodeState { region: NONE, source: NONE, dist: 0, obs: 0, arrival_z: 0, wrapped: 0, queued: NO_TIME };
}

#[derive(Clone, Debug)]
pub(crate) struct Region {
    pub radius: Varying,
    pub blossom_parent: RegionId,
    /// Alternating-tree node this region belongs to (as inner or outer), or `NONE`.
    pub tree: TreeId,
    /// Matched partner region, `BOUNDARY`, or `NONE`.
    pub matched_to: RegionId,
    /// Match edge from this region's defect outward.
    pub match_edge: CEdge,
    /// Blossom children in cyclic order with the edge from each child to the next (last → first).
    pub children: Vec<(RegionId, CEdge)>,
    /// Nodes this region itself acquired, in arrival order.
    pub shell: Vec<NodeId>,
    /// The defect for a leaf region; `NONE` for a blossom.
    pub source: NodeId,
    /// Time of the queued shrink event, or `NO_TIME`.
    pub shrink_queued: i64,
}

impl Region {
    pub(crate) fn leaf(source: NodeId) -> Self {
        Region {
            radius: Varying { y0: 0, slope: 1 },
            blossom_parent: NONE,
            tree: NONE,
            matched_to: NONE,
            match_edge: CEdge::EMPTY,
            children: Vec::new(),
            shell: vec![source],
            source,
            shrink_queued: NO_TIME,
        }
    }
    pub(crate) fn is_blossom(&self) -> bool {
        !self.children.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TreeNode {
    /// `NONE` for a root.
    pub inner: RegionId,
    pub outer: RegionId,
    /// From the inner region's defect to the outer region's defect (the pair's match edge).
    pub inner_to_outer: CEdge,
    pub parent: TreeId,
    /// From the parent's outer defect to this node's inner defect.
    pub parent_edge: CEdge,
    pub children: Vec<TreeId>,
    pub alive: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Stats {
    pub events: u64,
    pub pushes: u64,
}

pub(crate) struct State {
    pub nodes: Vec<NodeState>,
    pub touched: Vec<NodeId>,
    pub regions: Vec<Region>,
    pub trees: Vec<TreeNode>,
    heap: BinaryHeap<Reverse<(i64, u64, u8, u32)>>,
    seq: u64,
    pub now: i64,
    pub active_trees: usize,
    /// Reusable scratch for subtree walks.
    pub scratch: Vec<NodeId>,
    pub stats: Stats,
}

impl State {
    pub(crate) fn new(num_nodes: usize) -> Self {
        State {
            nodes: vec![NodeState::FREE; num_nodes],
            touched: Vec::new(),
            regions: Vec::new(),
            trees: Vec::new(),
            heap: BinaryHeap::new(),
            seq: 0,
            now: 0,
            active_trees: 0,
            scratch: Vec::new(),
            stats: Stats::default(),
        }
    }

    /// Return to the pristine state, touching only what the last shot touched.
    pub(crate) fn reset(&mut self) {
        for &u in &self.touched {
            self.nodes[u as usize] = NodeState::FREE;
        }
        self.touched.clear();
        self.regions.clear();
        self.trees.clear();
        self.heap.clear();
        self.seq = 0;
        self.now = 0;
        self.active_trees = 0;
        self.stats = Stats::default();
    }

    pub(crate) fn push_event(&mut self, t: i64, kind: u8, id: u32) {
        self.seq += 1;
        self.stats.pushes += 1;
        self.heap.push(Reverse((t, self.seq, kind, id)));
    }

    pub(crate) fn pop_event(&mut self) -> Option<(i64, u8, u32)> {
        self.heap.pop().map(|Reverse((t, _, k, id))| (t, k, id))
    }
}
```

- [ ] **Step 4: Run tests, fmt, clippy**

Run: `cargo test --release -p aleph-qec sparse_blossom::state && cargo fmt && cargo clippy -p aleph-qec --all-targets -- -D warnings`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-qec/src/sparse_blossom
git commit -m "[Q1-03b] sparse_blossom: per-shot state, affine radii, compressed edges, event heap"
```

---

### Task 3: Flooder + tree-free matcher (augment, boundary, grow) + leaf resolution

**Files:**
- Create: `crates/aleph-qec/src/sparse_blossom/flooder.rs`
- Create: `crates/aleph-qec/src/sparse_blossom/matcher.rs`
- Modify: `crates/aleph-qec/src/sparse_blossom/mod.rs` (modules, `SparseMatcher`, `State::run`)

**Interfaces:**
- Consumes: Task 1 `CompiledGraph`, Task 2 `State` and types.
- Produces: `pub(crate) enum MEvent { HitRegion { a: RegionId, b: RegionId, e: CEdge }, HitBoundary { a: RegionId, e: CEdge }, Shatter { b: RegionId } }`; `impl State { fn run(&mut self, g: &CompiledGraph, defects: &[u32]) -> (u64, i64) }`; `pub(crate) struct SparseMatcher { graph: CompiledGraph, pool: Mutex<Vec<Box<State>>> }` with `new(graph)`, `decode(&self, defects: &[u32]) -> (u64 /*obs*/, i64 /*weight, undoubled*/)`, `graph(&self) -> &CompiledGraph`. Matcher methods `blossom` and `shatter` exist as `unimplemented!()` stubs in this task; the degenerate implosion path in the flooder is implemented here (it only emits an event).
- Also produces a test helper in `mod.rs` tests: `fn dense_optimum(g: &CompiledGraph, defects: &[u32]) -> (u64, i64)` (all-pairs Dijkstra on the compiled graph + `crate::blossom::max_weight_matching` on savings), used by every later task.

- [ ] **Step 1: Write the failing tests** (in `mod.rs`, `#[cfg(test)] mod tests`)

```rust
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// Exact reference on the compiled graph: all-pairs Dijkstra (boundary settled, never
    /// expanded) + the Q1-03 savings matching. Returns `(obs, weight)` in undoubled units.
    pub(crate) fn dense_optimum(g: &CompiledGraph, defects: &[u32]) -> (u64, i64) {
        let n = g.num_nodes();
        let inf = i64::MAX / 4;
        let dijkstra = |src: u32| -> (Vec<i64>, Vec<u64>, i64, u64) {
            let mut dist = vec![inf; n];
            let mut par = vec![0u64; n];
            let (mut bd, mut bp) = (inf, 0u64);
            dist[src as usize] = 0;
            let mut heap = BinaryHeap::new();
            heap.push(Reverse((0i64, src)));
            while let Some(Reverse((d, u))) = heap.pop() {
                if d > dist[u as usize] {
                    continue;
                }
                if let Some((wb, ob)) = g.boundary(u) {
                    if d + wb < bd {
                        bd = d + wb;
                        bp = par[u as usize] ^ ob;
                    }
                }
                for e in g.edges(u) {
                    let nd = d + e.w;
                    if nd < dist[e.v as usize] {
                        dist[e.v as usize] = nd;
                        par[e.v as usize] = par[u as usize] ^ e.obs;
                        heap.push(Reverse((nd, e.v)));
                    }
                }
            }
            (dist, par, bd, bp)
        };
        let rows: Vec<_> = defects.iter().map(|&d| dijkstra(d)).collect();
        let k = defects.len();
        let mut edges = Vec::new();
        for i in 0..k {
            for j in (i + 1)..k {
                let d = rows[i].0[defects[j] as usize];
                let s = rows[i].2.saturating_add(rows[j].2).saturating_sub(d);
                if d < inf && s > 0 {
                    edges.push((i, j, s));
                }
            }
        }
        let mate = crate::blossom::max_weight_matching(k, &edges, false);
        let (mut obs, mut w) = (0u64, 0i64);
        for i in 0..k {
            let m = mate[i];
            if m == usize::MAX {
                if rows[i].2 < inf {
                    obs ^= rows[i].3;
                    w += rows[i].2;
                }
            } else if i < m {
                obs ^= rows[i].1[defects[m] as usize];
                w += rows[i].0[defects[m] as usize];
            }
        }
        (obs, w / 2)
    }

    fn sparse(g: &CompiledGraph, defects: &[u32]) -> (u64, i64) {
        SparseMatcher::new(g.clone()).decode(defects)
    }

    #[test]
    fn two_defects_on_a_line_match_each_other() {
        let g = CompiledGraph::from_int_edges(3, &[(0, 1, 3, 0b1), (1, 2, 5, 0b10)], &[]);
        assert_eq!(sparse(&g, &[0, 1]), (0b1, 3));
        assert_eq!(sparse(&g, &[0, 2]), (0b11, 8));
    }

    #[test]
    fn lone_defect_matches_boundary() {
        let g = CompiledGraph::from_int_edges(2, &[(0, 1, 3, 0)], &[(1, 4, 0b1)]);
        assert_eq!(sparse(&g, &[1]), (0b1, 4));
        assert_eq!(sparse(&g, &[0]), (0b1, 7)); // grows through node 1 to its boundary
    }

    #[test]
    fn grow_then_augment_through_the_boundary() {
        // w01=3, w0b=1, w12=6: 0 hits its boundary first (t=2 doubled), 1 then hits the
        // boundary-matched 0 (augment via the unmatched region), 2 then hits the matched 1 (tree
        // grows: 1 inner, 0 outer), and 0 — already at its boundary distance — augments at once.
        // Optimum: 0–b (1) + 1–2 (6) = 7, beating 0–1 (3) + 2 unmatched.
        let g = CompiledGraph::from_int_edges(3, &[(0, 1, 3, 0b1), (1, 2, 6, 0b10)], &[(0, 1, 0b100)]);
        assert_eq!(sparse(&g, &[0, 1, 2]), dense_optimum(&g, &[0, 1, 2]));
        assert_eq!(sparse(&g, &[0, 1, 2]), (0b110, 7));
    }

    #[test]
    fn empty_syndrome_is_zero() {
        let g = CompiledGraph::from_int_edges(2, &[(0, 1, 3, 0)], &[]);
        assert_eq!(sparse(&g, &[]), (0, 0));
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p aleph-qec sparse_blossom::tests`
Expected: compile error (`SparseMatcher` missing).

- [ ] **Step 3: Implement the flooder**

`crates/aleph-qec/src/sparse_blossom/flooder.rs`:

```rust
//! The flooder: grows outer regions, shrinks inner ones, and turns geometry into matcher events.
//!
//! Every owned node `u` carries `total(u,t) = wrapped(u) + top(u).radius(t)`; `u` is covered
//! while `total ≥ dist(u)`. A node's *look-at* event is the earliest of, over its neighbours `v`:
//! arrival into an unowned `v` (own top growing), collision with `v`'s top region (combined slope
//! > 0), and the boundary (own top growing). Scheduling is lazy: a heap entry is pushed only if
//! earlier than what is queued, and a popped entry whose time is not the queued time is stale.
//! Shrinking regions release their own shell in reverse arrival order and, at radius 0, shatter
//! (blossoms) or implode (leaf regions: the parent and child outer regions meet through it).

use super::graph::CompiledGraph;
use super::state::*;

/// What the matcher has to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MEvent {
    /// `a` is growing (outer); `e` runs from `a`'s defect to `b`'s defect.
    HitRegion { a: RegionId, b: RegionId, e: CEdge },
    /// `a` is growing; `e.to == BOUNDARY`.
    HitBoundary { a: RegionId, e: CEdge },
    /// Inner blossom `b` shrank to radius 0.
    Shatter { b: RegionId },
}

#[derive(Clone, Copy)]
enum Next {
    Arrive(usize),
    Collide(usize),
    Boundary,
}

impl State {
    /// Top-level region containing `r`.
    #[inline]
    pub(crate) fn top(&self, mut r: RegionId) -> RegionId {
        loop {
            let p = self.regions[r as usize].blossom_parent;
            if p == NONE {
                return r;
            }
            r = p;
        }
    }

    /// `total(u, ·)` as an affine function of time, for an owned node.
    #[inline]
    fn total(&self, u: NodeId) -> (Varying, RegionId) {
        let n = &self.nodes[u as usize];
        let top = self.top(n.region);
        let r = self.regions[top as usize].radius;
        (Varying { y0: r.y0 + n.wrapped, slope: r.slope }, top)
    }

    /// Earliest event at `u` and what it is; `None` if nothing will ever happen at `u`.
    fn next_at(&self, g: &CompiledGraph, u: NodeId) -> Option<(i64, Next)> {
        let n = self.nodes[u as usize];
        if n.region == NONE {
            return None;
        }
        let (tu, top_u) = self.total(u);
        let mut best: Option<(i64, Next)> = None;
        let mut consider = |t: i64, what: Next| {
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, what));
            }
        };
        for (k, e) in g.edges(u).iter().enumerate() {
            let m = self.nodes[e.v as usize];
            if m.region == NONE {
                if tu.slope > 0 {
                    consider(n.dist + e.w - tu.y0, Next::Arrive(k));
                }
                continue;
            }
            let (tv, top_v) = self.total(e.v);
            if top_v == top_u {
                continue;
            }
            let cs = tu.slope + tv.slope;
            if cs <= 0 {
                continue;
            }
            let num = n.dist + e.w + m.dist - tu.y0 - tv.y0;
            let t = if cs == 2 {
                debug_assert_eq!(num & 1, 0, "collision at a half-integer time");
                num / 2
            } else {
                num
            };
            consider(t, Next::Collide(k));
        }
        if tu.slope > 0 {
            if let Some((wb, _)) = g.boundary(u) {
                consider(n.dist + wb - tu.y0, Next::Boundary);
            }
        }
        best.map(|(t, w)| (t.max(self.now), w))
    }

    /// Queue `u`'s look-at if it is earlier than what is queued.
    pub(crate) fn schedule_node(&mut self, g: &CompiledGraph, u: NodeId) {
        if let Some((t, _)) = self.next_at(g, u) {
            let q = self.nodes[u as usize].queued;
            if q == NO_TIME || t < q {
                self.nodes[u as usize].queued = t;
                self.push_event(t, EV_NODE, u);
            }
        }
    }

    /// Pop-time handler for a node event: execute what is due now (at most one matcher event),
    /// then reschedule.
    pub(crate) fn look_at_node(&mut self, g: &CompiledGraph, u: NodeId) -> Option<MEvent> {
        let Some((t, what)) = self.next_at(g, u) else { return None };
        if t > self.now {
            self.nodes[u as usize].queued = t;
            self.push_event(t, EV_NODE, u);
            return None;
        }
        let n = self.nodes[u as usize];
        let ev = match what {
            Next::Arrive(k) => {
                let e = g.edges(u)[k];
                self.arrive(g, u, e.v, e.w, e.obs);
                None
            }
            Next::Collide(k) => {
                let e = g.edges(u)[k];
                let m = self.nodes[e.v as usize];
                let (tu, top_u) = self.total(u);
                let top_v = self.top(m.region);
                let ce = CEdge { from: n.source, to: m.source, obs: n.obs ^ e.obs ^ m.obs, weight: n.dist + e.w + m.dist };
                // Orient so that `a` is the growing side.
                if tu.slope > 0 {
                    Some(MEvent::HitRegion { a: top_u, b: top_v, e: ce })
                } else {
                    Some(MEvent::HitRegion { a: top_v, b: top_u, e: ce.reversed() })
                }
            }
            Next::Boundary => {
                let (wb, ob) = g.boundary(u).unwrap_or((0, 0));
                let top_u = self.top(n.region);
                Some(MEvent::HitBoundary { a: top_u, e: CEdge { from: n.source, to: BOUNDARY, obs: n.obs ^ ob, weight: n.dist + wb } })
            }
        };
        self.schedule_node(g, u);
        ev
    }

    /// `u`'s top region takes the unowned node `v` across the edge `(w, obs)`.
    fn arrive(&mut self, g: &CompiledGraph, u: NodeId, v: NodeId, w: i64, obs: u64) {
        let n = self.nodes[u as usize];
        let top = self.top(n.region);
        let z = self.regions[top as usize].radius.at(self.now);
        debug_assert_eq!(n.wrapped + z, n.dist + w, "arrival before the region reached the node");
        self.nodes[v as usize] = NodeState { region: top, source: n.source, dist: n.dist + w, obs: n.obs ^ obs, arrival_z: z, wrapped: n.wrapped, queued: NO_TIME };
        self.regions[top as usize].shell.push(v);
        self.touched.push(v);
        self.schedule_node(g, v);
    }

    /// Un-own `v`; its owned neighbours may now grow into it.
    fn release(&mut self, g: &CompiledGraph, v: NodeId) {
        self.nodes[v as usize] = NodeState::FREE;
        for k in 0..g.edges(v).len() {
            let x = g.edges(v)[k].v;
            if self.nodes[x as usize].region != NONE {
                self.schedule_node(g, x);
            }
        }
    }

    /// Time of the next release / radius-0 event of a top-level shrinking region.
    fn next_shrink_time(&self, r: RegionId) -> Option<i64> {
        let reg = &self.regions[r as usize];
        if reg.slope() != -1 || reg.blossom_parent != NONE {
            return None;
        }
        let keep = usize::from(!reg.is_blossom()); // a leaf never releases its source
        let mut target = 0;
        if reg.shell.len() > keep {
            target = target.max(self.nodes[reg.shell[reg.shell.len() - 1] as usize].arrival_z);
        }
        reg.radius.time_of(target, self.now)
    }

    pub(crate) fn schedule_shrink(&mut self, r: RegionId) {
        if let Some(t) = self.next_shrink_time(r) {
            let q = self.regions[r as usize].shrink_queued;
            if q == NO_TIME || t < q {
                self.regions[r as usize].shrink_queued = t;
                self.push_event(t, EV_SHRINK, r);
            }
        }
    }

    /// Pop-time handler for a shrink event.
    pub(crate) fn shrink_step(&mut self, g: &CompiledGraph, r: RegionId) -> Option<MEvent> {
        let Some(t) = self.next_shrink_time(r) else { return None };
        if t > self.now {
            self.regions[r as usize].shrink_queued = t;
            self.push_event(t, EV_SHRINK, r);
            return None;
        }
        let z = self.regions[r as usize].radius.at(self.now);
        let keep = usize::from(!self.regions[r as usize].is_blossom());
        while self.regions[r as usize].shell.len() > keep {
            let last = *self.regions[r as usize].shell.last().unwrap_or(&NONE);
            if self.nodes[last as usize].arrival_z < z {
                break;
            }
            self.regions[r as usize].shell.pop();
            self.release(g, last);
        }
        if z == 0 {
            let reg = &self.regions[r as usize];
            if reg.is_blossom() {
                return Some(MEvent::Shatter { b: r });
            }
            // Degenerate implosion: the parent outer and the child outer meet through this
            // zero-radius inner region.
            let n = &self.trees[reg.tree as usize];
            debug_assert_eq!(n.inner, r);
            let p = &self.trees[n.parent as usize];
            return Some(MEvent::HitRegion { a: p.outer, b: n.outer, e: n.parent_edge.then(n.inner_to_outer) });
        }
        self.schedule_shrink(r);
        None
    }

    /// All nodes owned by `r` or any blossom descendant, into `self.scratch`.
    pub(crate) fn collect_subtree(&mut self, r: RegionId) {
        self.scratch.clear();
        let mut stack = vec![r];
        while let Some(x) = stack.pop() {
            let reg = &self.regions[x as usize];
            self.scratch.extend_from_slice(&reg.shell);
            stack.extend(reg.children.iter().map(|&(c, _)| c));
        }
    }

    /// Change a top-level region's slope at `now` and re-derive every affected event.
    pub(crate) fn set_slope(&mut self, g: &CompiledGraph, r: RegionId, slope: i8) {
        debug_assert_eq!(self.regions[r as usize].blossom_parent, NONE);
        let rad = self.regions[r as usize].radius;
        self.regions[r as usize].radius = rad.with_slope(self.now, slope);
        self.reschedule_region(g, r);
    }

    /// Re-derive events for every node of `r`'s subtree (after a slope or wrapping change).
    pub(crate) fn reschedule_region(&mut self, g: &CompiledGraph, r: RegionId) {
        self.collect_subtree(r);
        let nodes = std::mem::take(&mut self.scratch);
        for &u in &nodes {
            self.schedule_node(g, u);
        }
        self.scratch = nodes;
        if self.regions[r as usize].slope() < 0 {
            self.schedule_shrink(r);
        }
    }

    /// `wrapped += delta` for every node of `c`'s subtree.
    pub(crate) fn shift_wrapped(&mut self, c: RegionId, delta: i64) {
        self.collect_subtree(c);
        let nodes = std::mem::take(&mut self.scratch);
        for &u in &nodes {
            self.nodes[u as usize].wrapped += delta;
        }
        self.scratch = nodes;
    }
}

impl Region {
    #[inline]
    pub(crate) fn slope(&self) -> i8 {
        self.radius.slope
    }
}
```

- [ ] **Step 4: Implement the tree-free matcher** (`matcher.rs`)

```rust
//! The matcher: Edmonds' alternating-tree operations on regions, driven by flooder events, and
//! the final resolution of top-level matches into defect–defect edges.

use super::flooder::MEvent;
use super::graph::CompiledGraph;
use super::state::*;

impl State {
    pub(crate) fn handle(&mut self, g: &CompiledGraph, ev: MEvent) {
        self.stats.events += 1;
        match ev {
            MEvent::HitRegion { a, b, e } => {
                let ta = self.regions[a as usize].tree;
                let tb = self.regions[b as usize].tree;
                debug_assert!(ta != NONE && self.trees[ta as usize].outer == a, "a must be outer");
                if tb == NONE {
                    if self.regions[b as usize].matched_to == BOUNDARY {
                        self.regions[b as usize].matched_to = NONE;
                        self.augment(g, a, b, e, false);
                    } else {
                        self.grow(g, a, b, e);
                    }
                } else if self.root_of(a) == self.root_of(b) {
                    // `Region.tree` is a tree-NODE id; "same tree" means the same root.
                    self.blossom(g, a, b, e);
                } else {
                    debug_assert_eq!(self.trees[tb as usize].outer, b);
                    self.augment(g, a, b, e, true);
                }
            }
            MEvent::HitBoundary { a, e } => {
                self.regions[a as usize].matched_to = BOUNDARY;
                self.regions[a as usize].match_edge = e;
                self.flip_path_to_root(a);
                let root = self.root_of(a);
                self.dissolve_tree(g, root);
            }
            MEvent::Shatter { b } => self.shatter(g, b),
        }
    }

    pub(crate) fn set_match(&mut self, x: RegionId, y: RegionId, e: CEdge) {
        self.regions[x as usize].matched_to = y;
        self.regions[x as usize].match_edge = e;
        self.regions[y as usize].matched_to = x;
        self.regions[y as usize].match_edge = e.reversed();
    }

    fn root_of(&self, r: RegionId) -> TreeId {
        let mut n = self.regions[r as usize].tree;
        while self.trees[n as usize].parent != NONE {
            n = self.trees[n as usize].parent;
        }
        n
    }

    /// Re-pair the regions on the path from `x`'s tree node up to the root.
    fn flip_path_to_root(&mut self, x: RegionId) {
        let mut n = self.regions[x as usize].tree;
        while self.trees[n as usize].parent != NONE {
            let p = self.trees[n as usize].parent;
            let inner = self.trees[n as usize].inner;
            let pe = self.trees[n as usize].parent_edge;
            let po = self.trees[p as usize].outer;
            self.set_match(inner, po, pe.reversed());
            n = p;
        }
    }

    /// Freeze every region of the tree rooted at `root`, matching untouched pairs as they stand.
    fn dissolve_tree(&mut self, g: &CompiledGraph, root: TreeId) {
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            let (inner, outer, io) = {
                let t = &mut self.trees[n as usize];
                t.alive = false;
                stack.extend_from_slice(&t.children);
                (t.inner, t.outer, t.inner_to_outer)
            };
            if inner != NONE {
                if self.regions[inner as usize].matched_to == NONE {
                    self.set_match(inner, outer, io);
                }
                self.regions[inner as usize].tree = NONE;
                self.set_slope(g, inner, 0);
            }
            self.regions[outer as usize].tree = NONE;
            self.set_slope(g, outer, 0);
        }
        self.active_trees -= 1;
    }

    /// `a` (outer) meets `b` (outer of another tree, or a boundary-matched free region).
    fn augment(&mut self, g: &CompiledGraph, a: RegionId, b: RegionId, e: CEdge, b_in_tree: bool) {
        self.set_match(a, b, e);
        self.flip_path_to_root(a);
        let ra = self.root_of(a);
        if b_in_tree {
            self.flip_path_to_root(b);
            let rb = self.root_of(b);
            self.dissolve_tree(g, rb);
        }
        self.dissolve_tree(g, ra);
    }

    /// `a` (outer) meets the frozen matched region `m`: `(m inner, m' outer)` joins `a`'s tree.
    fn grow(&mut self, g: &CompiledGraph, a: RegionId, m: RegionId, e: CEdge) {
        let m2 = self.regions[m as usize].matched_to;
        let io = self.regions[m as usize].match_edge;
        debug_assert!(m2 != NONE && m2 != BOUNDARY);
        self.regions[m as usize].matched_to = NONE;
        self.regions[m2 as usize].matched_to = NONE;
        let parent = self.regions[a as usize].tree;
        let n = self.new_tree_node(m, m2, io, parent, e);
        self.regions[m as usize].tree = n;
        self.regions[m2 as usize].tree = n;
        self.set_slope(g, m2, 1);
        self.set_slope(g, m, -1);
    }

    pub(crate) fn new_tree_root(&mut self, outer: RegionId) -> TreeId {
        self.trees.push(TreeNode { inner: NONE, outer, inner_to_outer: CEdge::EMPTY, parent: NONE, parent_edge: CEdge::EMPTY, children: Vec::new(), alive: true });
        self.active_trees += 1;
        (self.trees.len() - 1) as TreeId
    }

    pub(crate) fn new_tree_node(&mut self, inner: RegionId, outer: RegionId, inner_to_outer: CEdge, parent: TreeId, parent_edge: CEdge) -> TreeId {
        self.trees.push(TreeNode { inner, outer, inner_to_outer, parent, parent_edge, children: Vec::new(), alive: true });
        let id = (self.trees.len() - 1) as TreeId;
        self.trees[parent as usize].children.push(id);
        id
    }

    fn blossom(&mut self, _g: &CompiledGraph, _a: RegionId, _b: RegionId, _e: CEdge) {
        unimplemented!("Task 4")
    }

    fn shatter(&mut self, _g: &CompiledGraph, _b: RegionId) {
        unimplemented!("Task 4")
    }

    /// Turn top-level matches into `(obs, doubled weight)`; blossoms are resolved recursively.
    pub(crate) fn resolve(&self) -> (u64, i64) {
        let (mut obs, mut w) = (0u64, 0i64);
        for r in 0..self.regions.len() as RegionId {
            let reg = &self.regions[r as usize];
            if reg.blossom_parent != NONE || reg.matched_to == NONE {
                continue;
            }
            let e = reg.match_edge;
            if reg.matched_to == BOUNDARY {
                obs ^= e.obs;
                w += e.weight;
                self.descend(r, e.from, &mut obs, &mut w);
            } else if r < reg.matched_to {
                obs ^= e.obs;
                w += e.weight;
                self.descend(r, e.from, &mut obs, &mut w);
                self.descend(reg.matched_to, e.to, &mut obs, &mut w);
            }
        }
        (obs, w)
    }

    /// Inside blossom `r`, `defect` is the endpoint of the outside match; pair up the rest.
    fn descend(&self, r: RegionId, defect: NodeId, obs: &mut u64, w: &mut i64) {
        let reg = &self.regions[r as usize];
        if !reg.is_blossom() {
            return;
        }
        let k = reg.children.len();
        let idx = self.child_index_containing(r, defect);
        let mut i = (idx + 1) % k;
        while i != idx {
            let (x, e) = reg.children[i];
            let (y, _) = reg.children[(i + 1) % k];
            *obs ^= e.obs;
            *w += e.weight;
            self.descend(x, e.from, obs, w);
            self.descend(y, e.to, obs, w);
            i = (i + 2) % k;
        }
        self.descend(reg.children[idx].0, defect, obs, w);
    }

    /// Index of the child of blossom `b` that contains `defect`.
    pub(crate) fn child_index_containing(&self, b: RegionId, defect: NodeId) -> usize {
        let mut r = self.nodes[defect as usize].region;
        while self.regions[r as usize].blossom_parent != b {
            r = self.regions[r as usize].blossom_parent;
            debug_assert_ne!(r, NONE, "defect is not inside the blossom");
        }
        self.regions[b as usize].children.iter().position(|&(c, _)| c == r).unwrap_or(0)
    }
}
```

- [ ] **Step 5: Wire `run` and `SparseMatcher` in `mod.rs`**

```rust
pub(crate) mod flooder;
pub(crate) mod graph;
pub(crate) mod matcher;
pub(crate) mod state;

use std::sync::Mutex;

pub(crate) use graph::{CompiledGraph, WEIGHT_SCALE};
use state::*;

impl State {
    /// Decode one syndrome: `(observable mask, total weight in undoubled units)`.
    pub(crate) fn run(&mut self, g: &CompiledGraph, defects: &[u32]) -> (u64, i64) {
        self.reset();
        let mut last = NONE;
        for &d in defects {
            if d == last || d as usize >= g.num_nodes() {
                continue;
            }
            last = d;
            let r = self.regions.len() as RegionId;
            self.regions.push(Region::leaf(d));
            self.nodes[d as usize] = NodeState { region: r, source: d, dist: 0, obs: 0, arrival_z: 0, wrapped: 0, queued: NO_TIME };
            self.touched.push(d);
            let t = self.new_tree_root(r);
            self.regions[r as usize].tree = t;
        }
        let roots: Vec<NodeId> = self.regions.iter().map(|r| r.source).collect();
        for d in roots {
            self.schedule_node(g, d);
        }
        while let Some((t, kind, id)) = self.pop_event() {
            debug_assert!(t >= self.now);
            let ev = if kind == EV_NODE {
                if self.nodes[id as usize].queued != t {
                    continue;
                }
                self.nodes[id as usize].queued = NO_TIME;
                self.now = t;
                self.look_at_node(g, id)
            } else {
                if self.regions[id as usize].shrink_queued != t {
                    continue;
                }
                self.regions[id as usize].shrink_queued = NO_TIME;
                self.now = t;
                self.shrink_step(g, id)
            };
            if let Some(ev) = ev {
                self.handle(g, ev);
            }
        }
        // `active_trees > 0` here means an odd component with no boundary: best effort.
        let (obs, w) = self.resolve();
        (obs, w / 2)
    }
}

/// Crate-facing sparse matcher: a compiled graph plus a pool of reusable per-shot states so
/// `decode(&self)` works from `rayon` without a lock held during the decode.
pub(crate) struct SparseMatcher {
    graph: CompiledGraph,
    pool: Mutex<Vec<Box<State>>>,
}

impl SparseMatcher {
    pub(crate) fn new(graph: CompiledGraph) -> Self {
        SparseMatcher { graph, pool: Mutex::new(Vec::new()) }
    }

    pub(crate) fn graph(&self) -> &CompiledGraph {
        &self.graph
    }

    /// `defects` must be ascending. Returns `(observable mask, weight)`.
    pub(crate) fn decode(&self, defects: &[u32]) -> (u64, i64) {
        let mut st = self
            .pool
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .unwrap_or_else(|| Box::new(State::new(self.graph.num_nodes())));
        let out = st.run(&self.graph, defects);
        self.pool.lock().unwrap_or_else(|p| p.into_inner()).push(st);
        out
    }
}

impl Clone for SparseMatcher {
    fn clone(&self) -> Self {
        SparseMatcher::new(self.graph.clone())
    }
}

impl std::fmt::Debug for SparseMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseMatcher").field("nodes", &self.graph.num_nodes()).finish()
    }
}
```

`Mutex::lock().unwrap_or_else(PoisonError::into_inner)` is the no-`unwrap` way to tolerate a poisoned pool (a panic in another thread's decode).

- [ ] **Step 6: Run tests, fmt, clippy**

Run: `cargo test --release -p aleph-qec sparse_blossom && cargo fmt && cargo clippy -p aleph-qec --all-targets -- -D warnings`
Expected: all Task 1–3 tests pass (12).

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-qec/src/sparse_blossom
git commit -m "[Q1-03b] sparse_blossom: flooder, augment/grow/boundary matcher, leaf resolution"
```

---

### Task 4: Blossoms — formation, shattering, degenerate implosion; edge cases

**Files:**
- Modify: `crates/aleph-qec/src/sparse_blossom/matcher.rs` (replace the two stubs)
- Modify: `crates/aleph-qec/src/sparse_blossom/mod.rs` (tests)

**Interfaces:**
- Consumes: everything from Tasks 1–3. `Region.children: Vec<(RegionId, CEdge)>` in cyclic order, edge from child `i` to child `i+1 mod k`.
- Produces: `fn blossom(&mut self, g, a, b, e)`, `fn shatter(&mut self, g, b)` (complete).

- [ ] **Step 1: Write the failing tests** (append to `mod.rs` tests)

```rust
    #[test]
    fn equilateral_triangle_with_boundary() {
        // Three mutually equidistant defects tie at t=2; one pair augments, the third grows the
        // tree, the inner region implodes, the blossom grows to the boundary at node 2.
        let g = CompiledGraph::from_int_edges(3, &[(0, 1, 4, 1), (1, 2, 4, 2), (0, 2, 4, 4)], &[(2, 10, 8)]);
        let d = [0, 1, 2];
        assert_eq!(sparse(&g, &d), dense_optimum(&g, &d));
        assert_eq!(sparse(&g, &d).1, 14);
    }

    #[test]
    fn blossom_is_shattered_when_it_becomes_inner() {
        // Triangle {0,1,2} (w=4) forms a blossom, matches defect 3 (w(0,3)=20), is grabbed by
        // defect 4 (w(1,4)=30) and shrinks to zero while 3 grows to its boundary (40).
        let g = CompiledGraph::from_int_edges(
            5,
            &[(0, 1, 4, 1), (1, 2, 4, 2), (0, 2, 4, 4), (0, 3, 20, 8), (1, 4, 30, 16)],
            &[(3, 40, 32)],
        );
        let d = [0, 1, 2, 3, 4];
        assert_eq!(sparse(&g, &d), dense_optimum(&g, &d));
        assert_eq!(sparse(&g, &d).1, 74);
    }

    #[test]
    fn zero_weight_edge_matches_at_time_zero() {
        let g = CompiledGraph::from_int_edges(3, &[(0, 1, 0, 1), (1, 2, 3, 2)], &[(2, 5, 4)]);
        assert_eq!(sparse(&g, &[0, 1]), (1, 0));
        assert_eq!(sparse(&g, &[0, 1, 2]), dense_optimum(&g, &[0, 1, 2]));
    }

    #[test]
    fn odd_component_without_boundary_is_best_effort() {
        let g = CompiledGraph::from_int_edges(3, &[(0, 1, 2, 1), (1, 2, 2, 2)], &[]);
        let (obs, w) = sparse(&g, &[0, 1, 2]);
        // One pair matched (weight 2), one defect left over; no panic, no hang.
        assert_eq!(w, 2);
        assert!(obs == 1 || obs == 2);
    }

    #[test]
    fn pentagon_with_chord_forms_and_uses_a_blossom() {
        // 5-cycle of weight 2 edges plus a boundary far from node 0: dense oracle decides.
        let g = CompiledGraph::from_int_edges(
            5,
            &[(0, 1, 2, 1), (1, 2, 2, 2), (2, 3, 2, 4), (3, 4, 2, 8), (4, 0, 2, 16)],
            &[(0, 9, 32)],
        );
        for d in [&[0u32, 1, 2, 3, 4][..], &[0, 1, 2][..], &[1, 2, 3, 4][..], &[0, 2, 4][..]] {
            assert_eq!(sparse(&g, d), dense_optimum(&g, d), "defects {d:?}");
        }
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p aleph-qec sparse_blossom::tests`
Expected: the first two and the pentagon panic at `unimplemented!("Task 4")`; zero-weight and odd-component may already pass.

- [ ] **Step 3: Implement `blossom`**

Replace the stub in `matcher.rs`:

```rust
    /// `a` and `b` are outer regions of the same tree: contract the odd cycle they close.
    fn blossom(&mut self, g: &CompiledGraph, a: RegionId, b: RegionId, e: CEdge) {
        let na = self.regions[a as usize].tree;
        let nb = self.regions[b as usize].tree;
        // Paths to the root; the LCA is the first common node.
        let up = |s: &Self, mut n: TreeId| {
            let mut p = vec![n];
            while s.trees[n as usize].parent != NONE {
                n = s.trees[n as usize].parent;
                p.push(n);
            }
            p
        };
        let pa = up(self, na);
        let pb = up(self, nb);
        let lca = *pa.iter().find(|n| pb.contains(n)).unwrap_or(&pa[pa.len() - 1]);
        let path_a: Vec<TreeId> = pa.iter().copied().take_while(|&n| n != lca).collect();
        let path_b: Vec<TreeId> = pb.iter().copied().take_while(|&n| n != lca).collect();

        // Cycle in order a, inner(a), outer(parent), …, outer(lca), inner(b side), …, b.
        let mut cycle: Vec<(RegionId, CEdge)> = Vec::with_capacity(2 * (path_a.len() + path_b.len()) + 1);
        for &n in &path_a {
            let t = &self.trees[n as usize];
            cycle.push((t.outer, t.inner_to_outer.reversed()));
            cycle.push((t.inner, t.parent_edge.reversed()));
        }
        let lca_outer = self.trees[lca as usize].outer;
        let first_down = path_b.last().copied();
        // From the LCA's outer region down the b side, or straight back to `a` if `b` is that region.
        cycle.push((lca_outer, first_down.map_or(e.reversed(), |n| self.trees[n as usize].parent_edge)));
        for (i, &n) in path_b.iter().enumerate().rev() {
            let t = &self.trees[n as usize];
            cycle.push((t.inner, t.inner_to_outer));
            let next = if i == 0 { e.reversed() } else { self.trees[path_b[i - 1] as usize].parent_edge };
            cycle.push((t.outer, next));
        }
        debug_assert_eq!(cycle.len() % 2, 1);
        debug_assert_eq!(cycle[0].0, if path_a.is_empty() { lca_outer } else { a });

        // The blossom region.
        let bid = self.regions.len() as RegionId;
        self.regions.push(Region {
            radius: Varying { y0: -self.now, slope: 1 },
            blossom_parent: NONE,
            tree: lca,
            matched_to: NONE,
            match_edge: CEdge::EMPTY,
            children: cycle.clone(),
            shell: Vec::new(),
            source: NONE,
            shrink_queued: NO_TIME,
        });
        for &(c, _) in &cycle {
            let rc = self.regions[c as usize].radius.at(self.now);
            self.regions[c as usize].radius = Varying::frozen(rc);
            self.regions[c as usize].blossom_parent = bid;
            self.regions[c as usize].tree = NONE;
            self.shift_wrapped(c, rc);
        }

        // Tree surgery: the LCA node's outer becomes the blossom; the path nodes die and their
        // other children re-hang on the LCA node.
        let mut moved: Vec<TreeId> = Vec::new();
        for &n in path_a.iter().chain(path_b.iter()) {
            let t = &mut self.trees[n as usize];
            t.alive = false;
            moved.extend(t.children.drain(..).filter(|c| !path_a.contains(c) && !path_b.contains(c)));
        }
        let top_a = path_a.last().copied();
        let top_b = path_b.last().copied();
        let lca_node = &mut self.trees[lca as usize];
        lca_node.outer = bid;
        lca_node.children.retain(|&c| Some(c) != top_a && Some(c) != top_b);
        lca_node.children.extend_from_slice(&moved);
        for c in moved {
            self.trees[c as usize].parent = lca;
        }
        self.regions[bid as usize].tree = lca;
        self.reschedule_region(g, bid);
    }
```

- [ ] **Step 4: Implement `shatter`**

```rust
    /// Inner blossom `b` reached radius 0: replace it in the tree by the alternating path of
    /// its children between the parent edge's child and the outer edge's child; match the rest.
    fn shatter(&mut self, g: &CompiledGraph, b: RegionId) {
        let n = self.regions[b as usize].tree;
        debug_assert_eq!(self.trees[n as usize].inner, b);
        debug_assert!(self.regions[b as usize].shell.is_empty());
        let (p, parent_edge, io, outer_o) = {
            let t = &self.trees[n as usize];
            (t.parent, t.parent_edge, t.inner_to_outer, t.outer)
        };
        let children = std::mem::take(&mut self.regions[b as usize].children);
        let k = children.len();
        let i_par = self.child_index_of(&children, parent_edge.to);
        let i_out = self.child_index_of(&children, io.from);
        for &(c, _) in &children {
            let rc = self.regions[c as usize].radius.y0;
            self.regions[c as usize].blossom_parent = NONE;
            self.regions[c as usize].tree = NONE;
            self.shift_wrapped(c, -rc);
        }
        // Direction with an even number of edges from i_par to i_out.
        let fwd = (i_out + k - i_par) % k;
        let forward = fwd % 2 == 0;
        let step = |i: usize| if forward { (i + 1) % k } else { (i + k - 1) % k };
        let edge = |i: usize, j: usize| -> CEdge {
            // Edge from child i to child j, adjacent in the chosen direction.
            if forward { children[i].1 } else { children[j].1.reversed() }
        };
        // Path i_par → … → i_out (odd number of children).
        let mut path = vec![i_par];
        while *path.last().unwrap_or(&i_out) != i_out {
            let last = *path.last().unwrap_or(&i_out);
            path.push(step(last));
        }
        // The rest, continuing past i_out until just before i_par (even number of children).
        let mut rest = Vec::new();
        let mut i = step(i_out);
        while i != i_par {
            rest.push(i);
            i = step(i);
        }
        debug_assert_eq!(path.len() % 2, 1);
        debug_assert_eq!(rest.len() % 2, 0);

        // Rebuild the tree chain in place of `n`.
        self.trees[n as usize].alive = false;
        self.trees[p as usize].children.retain(|&c| c != n);
        let old_children = std::mem::take(&mut self.trees[n as usize].children);
        let mut parent = p;
        let mut pe = parent_edge;
        let mut j = 0;
        while j + 1 < path.len() {
            let (inner, outer) = (children[path[j]].0, children[path[j + 1]].0);
            let node = self.new_tree_node(inner, outer, edge(path[j], path[j + 1]), parent, pe);
            self.regions[inner as usize].tree = node;
            self.regions[outer as usize].tree = node;
            self.set_slope(g, outer, 1);
            self.set_slope(g, inner, -1);
            parent = node;
            pe = edge(path[j + 1], path[j + 2]);
            j += 2;
        }
        let c_out = children[path[path.len() - 1]].0;
        let last = self.new_tree_node(c_out, outer_o, io, parent, pe);
        self.regions[c_out as usize].tree = last;
        self.regions[outer_o as usize].tree = last;
        self.trees[last as usize].children = old_children;
        for &c in &self.trees[last as usize].children.clone() {
            self.trees[c as usize].parent = last;
        }
        self.set_slope(g, c_out, -1);
        // Matched pairs along the rest of the cycle: frozen, re-derive their events.
        let mut i = 0;
        while i + 1 < rest.len() {
            let (x, y) = (children[rest[i]].0, children[rest[i + 1]].0);
            self.set_match(x, y, edge(rest[i], rest[i + 1]));
            self.reschedule_region(g, x);
            self.reschedule_region(g, y);
            i += 2;
        }
    }

    fn child_index_of(&self, children: &[(RegionId, CEdge)], defect: NodeId) -> usize {
        let mut r = self.nodes[defect as usize].region;
        loop {
            if let Some(i) = children.iter().position(|&(c, _)| c == r) {
                return i;
            }
            r = self.regions[r as usize].blossom_parent;
            debug_assert_ne!(r, NONE, "defect not inside the shattering blossom");
        }
    }
```

Note the `pe = edge(path[j + 1], path[j + 2])` read: when `j + 2 == path.len() - 1` it is the edge into `c_out`; the loop guard `j + 1 < path.len()` with odd `path.len()` guarantees `j + 2` is in range.

- [ ] **Step 5: Run tests, fmt, clippy**

Run: `cargo test --release -p aleph-qec sparse_blossom && cargo fmt && cargo clippy -p aleph-qec --all-targets -- -D warnings`
Expected: all pass, including the five new tests.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-qec/src/sparse_blossom
git commit -m "[Q1-03b] sparse_blossom: blossom formation, shattering, degenerate implosion"
```

---

### Task 5: Exhaustive oracle on random graphs (proptest)

**Files:**
- Create: `crates/aleph-qec/tests/sparse_blossom_oracle.rs`
- Modify: `crates/aleph-qec/src/sparse_blossom/mod.rs` — make `dense_optimum` and `from_int_edges` reachable from the integration test: add a `#[doc(hidden)] pub mod testing` in `lib.rs`? No — keep it hermetic by moving the oracle test *inside* the crate as a `proptest!` in `mod.rs` tests (integration tests cannot see `pub(crate)`). So: **Modify** `mod.rs` tests only; no new file.

**Interfaces:**
- Consumes: `CompiledGraph::from_int_edges`, `SparseMatcher::decode`, `dense_optimum`.

- [ ] **Step 1: Write the property test** (append to `mod.rs` tests)

```rust
    use proptest::prelude::*;

    /// Random connected sparse graph: `n` nodes on a random spanning tree plus `extra` random
    /// edges, integer weights in `1..=wmax`, each node a boundary edge with probability 1/3,
    /// and a random defect subset.
    fn graph_strategy() -> impl Strategy<Value = (CompiledGraph, Vec<u32>)> {
        (2usize..=12, 0usize..=10, 1i64..=9, any::<u64>()).prop_map(|(n, extra, wmax, seed)| {
            let mut z = seed;
            let mut next = move || {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                z
            };
            let mut edges = Vec::new();
            for v in 1..n as u32 {
                let u = (next() % v as u64) as u32;
                edges.push((u, v, 1 + (next() % wmax as u64) as i64, 1u64 << (next() % 8)));
            }
            for _ in 0..extra {
                let u = (next() % n as u64) as u32;
                let v = (next() % n as u64) as u32;
                if u != v {
                    edges.push((u.min(v), u.max(v), 1 + (next() % wmax as u64) as i64, 1u64 << (next() % 8)));
                }
            }
            let boundary: Vec<(u32, i64, u64)> = (0..n as u32)
                .filter(|_| next() % 3 == 0)
                .map(|u| (u, 1 + (next() % (2 * wmax) as u64) as i64, 1u64 << (8 + next() % 8)))
                .collect();
            let defects: Vec<u32> = (0..n as u32).filter(|_| next() % 2 == 0).collect();
            (CompiledGraph::from_int_edges(n, &edges, &boundary), defects)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(3000))]
        #[test]
        fn sparse_reaches_the_dense_optimum_weight((g, defects) in graph_strategy()) {
            // Guarantee a perfect matching exists: even defect count, or every node has a
            // boundary within reach (the spanning tree makes the graph connected).
            let (so, sw) = SparseMatcher::new(g.clone()).decode(&defects);
            let (dobs, dw) = dense_optimum(&g, &defects);
            let has_boundary = (0..g.num_nodes() as u32).any(|u| g.boundary(u).is_some());
            prop_assume!(defects.len() % 2 == 0 || has_boundary);
            prop_assert_eq!(sw, dw, "weight differs: sparse {} dense {}", sw, dw);
            // Ties (equal weight, different parity) are legitimate; count them loosely.
            if so != dobs {
                prop_assert!(sw == dw);
            }
        }
    }
```

- [ ] **Step 2: Run it**

Run: `cargo test --release -p aleph-qec sparse_blossom::tests::sparse_reaches -- --nocapture`
Expected: PASS over 3000 cases. If a case fails, proptest prints the minimal `(g, defects)`; fix the engine (not the test) and add the shrunk case as a named unit test in the Task 4 style.

- [ ] **Step 3: Same-thread determinism and repeat-decode test** (append)

```rust
    #[test]
    fn repeated_decodes_reuse_state_identically() {
        let g = CompiledGraph::from_int_edges(
            5,
            &[(0, 1, 4, 1), (1, 2, 4, 2), (0, 2, 4, 4), (0, 3, 20, 8), (1, 4, 30, 16)],
            &[(3, 40, 32)],
        );
        let m = SparseMatcher::new(g);
        let first = m.decode(&[0, 1, 2, 3, 4]);
        for _ in 0..50 {
            assert_eq!(m.decode(&[0, 1, 2, 3, 4]), first);
            assert_eq!(m.decode(&[1, 2]), (2, 4));
        }
    }
```

- [ ] **Step 4: Run, fmt, clippy, commit**

Run: `cargo test --release -p aleph-qec sparse_blossom && cargo fmt && cargo clippy -p aleph-qec --all-targets -- -D warnings`

```bash
git add crates/aleph-qec/src/sparse_blossom
git commit -m "[Q1-03b] sparse_blossom: exhaustive proptest oracle vs all-pairs blossom"
```

---

### Task 6: Wire into `MwpmDecoder` (lazy dense tables, sparse default, differential tests)

**Files:**
- Modify: `crates/aleph-qec/src/mwpm.rs` (struct, constructor, `decode`, tables, tests)
- Modify: `crates/aleph-qec/src/lib.rs` (drop the `#[allow(dead_code)]`)

**Interfaces:**
- Consumes: `SparseMatcher::{new, decode}`, `CompiledGraph::from_matching_graph`, `WEIGHT_SCALE` (moved here from `mwpm.rs`).
- Produces: `MwpmDecoder { num_detectors, num_observables, graph: MatchingGraph, sparse: SparseMatcher, dense: OnceLock<DenseTables>, locality_k }`; `struct DenseTables { stride, dist, parity }`; `fn tables(&self) -> &DenseTables`; `pub fn decode_dense(&self, &Syndrome) -> Correction`; `pub(crate) fn decode_dense_weighted(&self, &Syndrome) -> (Correction, i64)`; `pub(crate) fn decode_local(&self, &Syndrome) -> (Correction, i64)`; `pub(crate) fn decode_sparse(&self, &Syndrome) -> (Correction, i64)`; `impl Decoder` → `decode_sparse`.

- [ ] **Step 1: Write the failing tests** (replace the `local_matches_dense_weight_and_corrections` test and add three)

```rust
    /// The AC of #331: the sparse matcher reaches the *same minimum weight* as the dense
    /// matching on every shot, phenomenological and circuit-level, across p; corrections differ
    /// only on genuine ties.
    #[test]
    fn sparse_matches_dense_weight_and_corrections() {
        use crate::{build_dem, SurfaceCode};
        let mut total = 0usize;
        for d in [3usize, 5, 7, 9, 11] {
            for p in [0.01, 0.03, 0.06] {
                let exp = SurfaceCode::new(d).memory_z_experiment(d);
                let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
                let dec = MwpmDecoder::new(&dem).unwrap();
                let shots = if cfg!(debug_assertions) { 40 } else if d >= 11 { 1500 } else { 8000 };
                let (mut nonempty, mut ties) = (0usize, 0usize);
                for fired in sample_defects(&dem, shots, 0xD00D ^ (d as u64) << 8 ^ (p * 1000.0) as u64) {
                    let s = Syndrome::new(dem.detectors, fired);
                    if s.weight() > 0 {
                        nonempty += 1;
                    }
                    let (cs, ws) = dec.decode_sparse(&s);
                    let (cd, wd) = dec.decode_dense_weighted(&s);
                    assert_eq!(ws, wd, "d={d} p={p}: sparse weight {ws} != dense {wd} on {:?}", s.fired);
                    if cs != cd {
                        ties += 1;
                    }
                }
                total += shots;
                let rate = ties as f64 / nonempty.max(1) as f64;
                assert!(rate < 0.05, "d={d} p={p}: {ties}/{nonempty} tie disagreements");
            }
        }
        assert!(cfg!(debug_assertions) || total >= 100_000, "AC asks for 1e5 shots, ran {total}");
    }

    #[test]
    fn sparse_matches_dense_on_circuit_level_dems() {
        use crate::{CircuitNoise, SurfaceCode};
        for d in [3usize, 5] {
            let exp = SurfaceCode::new(d).memory_z_experiment(d);
            let dem = exp.circuit_level_dem(CircuitNoise::uniform(0.003)).unwrap();
            let dec = MwpmDecoder::new(&dem).unwrap();
            for fired in sample_defects(&dem, 2000, 77 + d as u64) {
                let s = Syndrome::new(dem.detectors, fired);
                assert_eq!(dec.decode_sparse(&s).1, dec.decode_dense_weighted(&s).1);
            }
        }
    }

    #[test]
    fn duplicate_defects_are_ignored() {
        let dem = DetectorErrorModel::parse("error(0.1) D0\nerror(0.1) D0 D1\nerror(0.1) D1 L0\n").unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();
        // `Syndrome::new` sorts and dedups; build the raw struct to bypass it.
        let s = Syndrome { fired: vec![1, 1, 7], detectors: 2 };
        assert_eq!(dec.decode(&s), Correction::new(vec![true]));
    }

    #[test]
    fn pool_reuse_is_deterministic() {
        use crate::{build_dem, SurfaceCode};
        use rayon::prelude::*;
        let exp = SurfaceCode::new(7).memory_z_experiment(7);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();
        let synds: Vec<Syndrome> = sample_defects(&dem, 2000, 5).into_iter().map(|f| Syndrome::new(dem.detectors, f)).collect();
        let serial: Vec<_> = synds.iter().map(|s| dec.decode_sparse(s)).collect();
        let parallel: Vec<_> = synds.par_iter().map(|s| dec.decode_sparse(s)).collect();
        assert_eq!(serial, parallel);
    }
```

`exp.circuit_level_dem(CircuitNoise::uniform(p))` is the constructor `qec_threshold.rs` uses. `Syndrome` has public `detectors` / `fired` fields and `Syndrome::new` sorts + dedups, so `defects_of` must dedup itself for the raw-struct case.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p aleph-qec mwpm::tests`
Expected: compile errors (`decode_sparse`, `decode_dense_weighted` missing).

- [ ] **Step 3: Restructure `MwpmDecoder`**

In `mwpm.rs`:

```rust
use std::sync::OnceLock;

use crate::sparse_blossom::{CompiledGraph, SparseMatcher, WEIGHT_SCALE};

/// All-pairs shortest-path tables for the dense / local oracle paths. `O(D²)`; built lazily.
#[derive(Clone, Debug)]
struct DenseTables {
    stride: usize,
    dist: Vec<i64>,
    parity: Vec<u64>,
}

impl DenseTables {
    fn build(graph: &MatchingGraph) -> Self {
        let num_detectors = graph.num_detectors();
        let boundary = graph.boundary();
        let stride = num_detectors + 1;
        let edge_w: Vec<i64> = graph.edges().iter().map(|e| (e.weight * WEIGHT_SCALE).round() as i64).collect();
        let edge_mask: Vec<u64> = graph.edges().iter().map(|e| e.observables.iter().fold(0u64, |m, &o| m | (1u64 << o))).collect();
        let mut dist = vec![INF; num_detectors * stride];
        let mut parity = vec![0u64; num_detectors * stride];
        for src in 0..num_detectors {
            dijkstra_from(graph, src, boundary, &edge_w, &edge_mask, &mut dist[src * stride..(src + 1) * stride], &mut parity[src * stride..(src + 1) * stride]);
        }
        DenseTables { stride, dist, parity }
    }
}

#[derive(Clone, Debug)]
pub struct MwpmDecoder {
    num_detectors: usize,
    num_observables: usize,
    graph: MatchingGraph,
    sparse: SparseMatcher,
    dense: OnceLock<DenseTables>,
    locality_k: usize,
}
```

`from_graph`:

```rust
    pub fn from_graph(graph: &MatchingGraph) -> Self {
        MwpmDecoder {
            num_detectors: graph.num_detectors(),
            num_observables: graph.num_observables(),
            graph: graph.clone(),
            sparse: SparseMatcher::new(CompiledGraph::from_matching_graph(graph)),
            dense: OnceLock::new(),
            locality_k: DEFAULT_LOCALITY_K,
        }
    }

    fn tables(&self) -> &DenseTables {
        self.dense.get_or_init(|| DenseTables::build(&self.graph))
    }
```

Replace every `self.dist[...]`, `self.parity[...]`, `self.stride` in the dense/local code with `let t = self.tables();` + `t.dist`, `t.parity`, `t.stride`. Add:

```rust
    /// Dense reference with its total weight (differential tests).
    pub(crate) fn decode_dense_weighted(&self, syndrome: &Syndrome) -> (Correction, i64) {
        self.decode_with(syndrome, |s, d| s.augmented_edges_dense(d))
    }

    /// Q1-03b production path: Sparse Blossom on the detector graph.
    pub(crate) fn decode_sparse(&self, syndrome: &Syndrome) -> (Correction, i64) {
        let defects = self.defects_of(syndrome);
        if defects.is_empty() {
            return (Correction::none(self.num_observables), 0);
        }
        let d32: Vec<u32> = defects.iter().map(|&d| d as u32).collect();
        let (acc, weight) = self.sparse.decode(&d32);
        let flips = (0..self.num_observables).map(|o| (acc >> o) & 1 == 1).collect();
        (Correction::new(flips), weight)
    }
```

`defects_of` must return ascending, deduplicated indices (`sort_unstable(); dedup()` if `Syndrome::new` does not already guarantee it). `impl Decoder for MwpmDecoder { fn decode(&self, s) -> Correction { self.decode_sparse(s).0 } }` with the doc comment updated. Delete the old `WEIGHT_SCALE` const here (import from `sparse_blossom`), keep `INF`. Update the module doc (the three-step description: step 2 now names the sparse path as production and the two others as oracles; the memory note about `O(D²)` says "built lazily for the oracles"). Rewrite the existing `profile_local_phases_d11` to also time `decode_sparse` and print `stats` (expose `SparseMatcher::last_stats()`? no — keep it simple: time only).

- [ ] **Step 4: Run the whole crate's tests, clippy on stable and nightly, fmt**

Run:
```bash
cargo test --release -p aleph-qec
cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy -p aleph-qec --all-targets -- -D warnings
```
Expected: all green, including `mwpm_threshold` and the four new tests. If `+nightly` is red on pre-existing `bp.rs`/`bivariate_bicycle.rs` lints (memory: they are), confirm the only errors are in those files.

- [ ] **Step 5: Python bindings still build**

Run: `cargo build -p aleph-py`
Expected: OK (only uses `MwpmDecoder::new` + `Decoder`).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-qec/src/mwpm.rs crates/aleph-qec/src/lib.rs
git commit -m "[Q1-03b] MwpmDecoder: Sparse Blossom is the decode path; dense tables lazy oracles"
```

---

### Task 7: Benchmark, PyMatching oracle, profile

**Files:**
- Modify: `benches/benches/mwpm_decode.rs`

- [ ] **Step 1: Extend the bench**

Replace the distance loop and add the sparse arm; dense skipped at d ≥ 15 (it is the AC baseline only at d = 11 and takes minutes beyond 13):

```rust
    for d in [7usize, 9, 11, 13, 15, 17] {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        let decoder = MwpmDecoder::new(&dem).unwrap();
        let syndromes = sample(&dem, shots, 0x5EED ^ d as u64);
        let avg_defects: f64 = syndromes.iter().map(|s| s.weight()).sum::<usize>() as f64 / shots as f64;
        grp.throughput(Throughput::Elements(shots as u64));
        if d >= 13 {
            grp.sample_size(10);
        }
        let tag = format!("d{d}_det{}_avgdef{avg_defects:.0}", dem.detectors);
        if d <= 13 {
            grp.bench_with_input(BenchmarkId::new("dense", &tag), &syndromes, |b, synds| {
                b.iter(|| for s in synds { std::hint::black_box(decoder.decode_dense(std::hint::black_box(s))); });
            });
        }
        grp.bench_with_input(BenchmarkId::new("sparse", &tag), &syndromes, |b, synds| {
            b.iter(|| for s in synds { std::hint::black_box(decoder.decode(std::hint::black_box(s))); });
        });
    }
```

Keep a "local" arm too (it needs `decode_local` public-to-the-bench: add `#[doc(hidden)] pub fn decode_local_pub(&self, s: &Syndrome) -> Correction` in `mwpm.rs`, documented as bench-only). Update the file's doc comment.

- [ ] **Step 2: Run the bench on an idle box**

Run: `uptime; pgrep -af "cargo bench|bencher run|Runner.Worker"; RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches --bench mwpm_decode 2>&1 | tee /private/tmp/claude-501/-Users-ex-GitHub-aleph/b3b9f375-1a35-4b8d-9217-dbfec62074e7/scratchpad/bench_sparse.txt`
Expected: `sparse/d11…` ≥ 10× `dense/d11…` throughput. Record all numbers.

- [ ] **Step 3: Run the PyMatching oracle** (needs a venv with pymatching; `scripts/qiskit-baseline/.venv` or create one in the scratchpad with `uv venv && uv pip install stim pymatching numpy`)

Run: `PYMATCHING_PYTHON=<venv>/bin/python cargo test --release -p aleph-qec --test mwpm_pymatching_oracle -- --ignored --nocapture`
Expected: both tests pass (LER within CI at d ∈ {3,…,11}; ≥ 99 % per-shot agreement at p = 0.006).

- [ ] **Step 4: Profile** — run `cargo test --release -p aleph-qec profile_local_phases_d11 -- --ignored --nocapture` and note µs/shot for local vs sparse. If sparse at d = 11 is > 2× PyMatching's ~18 µs, look at (in order) heap pushes per shot, `top()` walk depth, `collect_subtree` allocation; apply only what the numbers justify, re-run Task 5's proptest after each change.

- [ ] **Step 5: Commit**

```bash
git add benches/benches/mwpm_decode.rs crates/aleph-qec/src/mwpm.rs
git commit -m "[Q1-03b] bench: sparse arm, d up to 17; dense baseline kept to d=13"
```

---

### Task 8: Perf record, docs, PR

**Files:**
- Create: `docs/perf/q1-03b-sparse-blossom.md`
- Modify: `docs/perf/q1-03-localized-matching.md` (status line → superseded by Q1-03b), `crates/aleph-py/README.md` (the "Per core, pymatching is faster" paragraph and #331 pointer), `CHANGELOG.md` (Unreleased entry), `docs/qec/BACKLOG.md` if it lists Q1-03b.

- [ ] **Step 1: Write the perf record** with sections: What changed (one paragraph + the file map); Results (criterion table dense / local / sparse at d ∈ {7,…,17}, syndromes/s and ratio to dense; the AC line for d = 11); Correctness (proptest cases, differential shot counts per (d, p), circuit-level, PyMatching oracle result); Profile (µs/shot, events and heap pushes per shot at d = 11 vs PyMatching's figure from the README); Remaining gap / follow-ups; Reproduce (the exact commands from Task 7).

- [ ] **Step 2: Refresh the Python README table** only if the pymatching venv from Task 7 exists: `python scripts/python/bench_qec.py` per its header, paste the mwpm rows and rewrite the "Per core" paragraph to the measured ratio; otherwise change the paragraph to say the sparse rewrite landed (link the perf record) and that the table predates it.

- [ ] **Step 3: Final checks**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --release --workspace
```

- [ ] **Step 4: Commit and open the PR**

```bash
git add docs/perf/q1-03b-sparse-blossom.md docs/perf/q1-03-localized-matching.md crates/aleph-py/README.md CHANGELOG.md
git commit -m "[Q1-03b] docs: Sparse Blossom perf record, README and changelog"
git push -u origin q1-03b-sparse-blossom
gh pr create --title "[Q1-03b] Sparse Blossom: local region-growing MWPM" --body-file <body>
```

PR body: `Closes #331` (and `Closes #302`), summary of approach (spec link), the criterion table, the differential/proptest/oracle results, follow-ups. Then the `superpowers:requesting-code-review` step before merge.
