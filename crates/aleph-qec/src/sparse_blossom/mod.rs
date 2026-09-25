//! Sparse Blossom: minimum-weight perfect matching by local, event-driven region growth on the
//! detector graph (Higgott & Gidney, arXiv:2303.15933), re-derived from Edmonds' primal-dual
//! blossom algorithm. See `docs/superpowers/specs/2026-09-24-sparse-blossom-design.md`.
//!
//! Module map: [`graph`] compiles the [`crate::MatchingGraph`] into a CSR with doubled integer
//! weights; [`state`] holds the per-shot mutable arenas and the event heap; [`flooder`] grows and
//! shrinks regions and detects collisions; [`matcher`] runs the alternating-tree operations and
//! resolves the final matching; [`retrace`] recovers each matched pair's path for the
//! per-column error output. [`SparseMatcher`] is the crate-facing entry point.

pub(crate) mod flooder;
pub(crate) mod graph;
pub(crate) mod matcher;
pub(crate) mod retrace;
pub(crate) mod state;

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) use graph::{CompiledGraph, WEIGHT_SCALE};
use state::*;

/// Per-decoder id source: each `SparseMatcher` (including every `Clone`) gets a fresh id so
/// per-thread cache entries never collide across decoders that happen to run on the same thread.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A thread keeps at most this many decoders' `State`s cached; a program that creates many
/// short-lived decoders on one thread should not leak arenas without bound.
const CACHE_CAP: usize = 8;

thread_local! {
    /// One `State` arena per `(decoder id, thread)`, reused across `decode` calls on that
    /// thread. Linear search: with `CACHE_CAP` this small, a `Vec` beats a `HashMap`.
    ///
    /// Lifetime: a dropped `SparseMatcher`'s entry is *not* reclaimed proactively — it just
    /// sits here until evicted (FIFO by insertion order, i.e. oldest-inserted first, not an
    /// LRU) or the thread exits. Worst case per thread is `CACHE_CAP` live `State`s, even if
    /// only one decoder is still in use. Because eviction is FIFO rather than LRU, a thread that
    /// round-robins more than `CACHE_CAP` live decoders never gets a cache hit for any of them —
    /// every `decode` call evicts the oldest entry and allocates a fresh `State` instead of
    /// reusing one.
    static CACHE: RefCell<Vec<(u64, State)>> = const { RefCell::new(Vec::new()) };
}

impl State {
    /// Decode one syndrome: `(observable mask, total weight in undoubled units)`.
    pub(crate) fn run(&mut self, g: &CompiledGraph, defects: &[u32]) -> (u64, i64) {
        self.run_to_matching(g, defects);
        let (obs, w) = self.resolve();
        (obs, w / 2)
    }

    /// Like `run`, but also XORs every matched path's edges into `ehat` by column.
    pub(crate) fn run_edges(
        &mut self,
        g: &CompiledGraph,
        defects: &[u32],
        ehat: &mut [u8],
    ) -> (u64, i64) {
        self.run_to_matching(g, defects);
        // Collected first: `retrace` needs `&mut self` while `resolve_with` borrows `&self`.
        let mut matched: Vec<CEdge> = Vec::new();
        self.resolve_with(&mut |e: &CEdge| matched.push(*e));
        let (mut obs, mut w) = (0u64, 0i64);
        for e in &matched {
            obs ^= e.obs;
            w += e.weight;
            self.retrace(g, e, ehat);
        }
        (obs, w / 2)
    }

    /// Everything `run` does up to (not including) `resolve`: seed one region per defect, drain
    /// the event queue, and dissolve any leftover tree.
    fn run_to_matching(&mut self, g: &CompiledGraph, defects: &[u32]) {
        debug_assert!(
            defects.windows(2).all(|w| w[0] <= w[1]),
            "defects must be ascending"
        );
        self.reset();
        let mut last = NONE;
        for &d in defects {
            if d == last || d as usize >= g.num_nodes() {
                continue;
            }
            last = d;
            let r = self.regions.len() as RegionId;
            self.regions.push(Region::leaf(d));
            self.nodes[d as usize] = NodeState {
                region: r,
                source: d,
                dist: 0,
                obs: 0,
                arrival_z: 0,
                wrapped: 0,
                queued: NO_TIME,
            };
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
        // `active_trees > 0` here means an odd component with no boundary: best effort. Dissolve
        // whatever tree structure remains so `resolve` sees ordinary matches everywhere except
        // each leftover tree's one exposed root.
        if self.active_trees > 0 {
            self.dissolve_leftover_trees(g);
        }
    }
}

/// Crate-facing sparse matcher: a compiled graph plus an id into the thread-local `CACHE`, so
/// `decode(&self)` works from `rayon` without any lock — each thread keeps its own `State` per
/// decoder id, reused across calls on that thread. A global `Mutex<Vec<State>>` pool was tried
/// first but re-contends on every `pop`/`push`; at the ~1.5 µs decode times of small distances
/// (e.g. d=5) that contention dominated the decode itself and collapsed multi-thread throughput
/// below the single-thread number.
pub(crate) struct SparseMatcher {
    graph: CompiledGraph,
    id: u64,
}

impl SparseMatcher {
    pub(crate) fn new(graph: CompiledGraph) -> Self {
        SparseMatcher {
            graph,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// `defects` must be ascending. Returns `(observable mask, weight)`.
    pub(crate) fn decode(&self, defects: &[u32]) -> (u64, i64) {
        self.decode_with_stats(defects).0
    }

    /// Same as `decode`, but also returns the sparse matcher's per-shot event statistics
    /// (`State::stats`, reset every call inside `State::run`'s `self.reset()`). `decode` is a
    /// thin wrapper around this that drops the `Stats` half (the cache lookup/eviction lives in
    /// `with_state`, shared with `decode_errors`); profiling tests
    /// (`mwpm::tests::profile_local_phases_d11`) call this directly to report average
    /// events/pushes per shot.
    pub(crate) fn decode_with_stats(&self, defects: &[u32]) -> ((u64, i64), Stats) {
        self.with_state(|st| {
            let out = st.run(&self.graph, defects);
            (out, st.stats)
        })
    }

    /// `decode`, plus the matched paths XORed into `ehat` by representative DEM column (`ehat`
    /// must cover every column of the graph). Same matching, same `(obs, weight)` as `decode`.
    pub(crate) fn decode_errors(&self, defects: &[u32], ehat: &mut [u8]) -> (u64, i64) {
        self.with_state(|st| st.run_edges(&self.graph, defects, ehat))
    }

    /// Run `f` on this thread's cached `State` for this decoder (allocating and caching one on
    /// first use; evicting FIFO past `CACHE_CAP`). Falls back to an uncached `State` if the
    /// thread-local is unavailable — never panics.
    ///
    /// `CACHE.try_with` (rather than `.with`) keeps this panic-free even if a decode is somehow
    /// reached while this thread's locals are being torn down (e.g. called from another TLS
    /// destructor at thread exit) — `.with` panics in that case, `try_with` returns `Err`
    /// instead. `f` (a `State::run*` call) never touches `CACHE` — it's plain arithmetic over its
    /// own arenas — so the inner `try_borrow_mut` can never re-enter and can't actually fail on
    /// this path either. Both fallbacks compute on a fresh, uncached `State` rather than
    /// panicking, per the no-panic-in-library-code rule.
    fn with_state<R>(&self, mut f: impl FnMut(&mut State) -> R) -> R {
        // `FnMut`, not `FnOnce`: the cached path and the TLS-teardown fallback below each need
        // `f`, and only one of them ever runs; two sequential `&mut` borrows express that
        // without an `Option::take` dance.
        let n = self.graph.num_nodes();
        CACHE
            .try_with(|cache| {
                if let Ok(mut cache) = cache.try_borrow_mut() {
                    if let Some((_, st)) = cache.iter_mut().find(|(id, _)| *id == self.id) {
                        return f(st);
                    }
                    if cache.len() >= CACHE_CAP {
                        cache.remove(0); // evict the oldest entry
                    }
                    let mut st = State::new(n);
                    let out = f(&mut st);
                    cache.push((self.id, st));
                    out
                } else {
                    f(&mut State::new(n))
                }
            })
            .unwrap_or_else(|_| f(&mut State::new(n)))
    }
}

impl Clone for SparseMatcher {
    fn clone(&self) -> Self {
        SparseMatcher::new(self.graph.clone())
    }
}

impl std::fmt::Debug for SparseMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseMatcher")
            .field("nodes", &self.graph.num_nodes())
            .finish()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;
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
        let g =
            CompiledGraph::from_int_edges(3, &[(0, 1, 3, 0b1), (1, 2, 6, 0b10)], &[(0, 1, 0b100)]);
        assert_eq!(sparse(&g, &[0, 1, 2]), dense_optimum(&g, &[0, 1, 2]));
        assert_eq!(sparse(&g, &[0, 1, 2]), (0b110, 7));
    }

    #[test]
    fn empty_syndrome_is_zero() {
        let g = CompiledGraph::from_int_edges(2, &[(0, 1, 3, 0)], &[]);
        assert_eq!(sparse(&g, &[]), (0, 0));
    }

    #[test]
    fn equilateral_triangle_with_boundary() {
        // Three mutually equidistant defects tie at t=2: one pair augments, the third grows its
        // tree, and outer-outer collisions form a blossom that grows to the boundary at node 2 —
        // two ordinary outer-outer blossoms, not a degenerate implosion.
        let g = CompiledGraph::from_int_edges(
            3,
            &[(0, 1, 4, 1), (1, 2, 4, 2), (0, 2, 4, 4)],
            &[(2, 10, 8)],
        );
        let d = [0, 1, 2];
        assert_eq!(sparse(&g, &d), dense_optimum(&g, &d));
        assert_eq!(sparse(&g, &d).1, 14);
    }

    #[test]
    fn blossom_is_shattered_when_it_becomes_inner() {
        // Triangle {0,1,2} (w=4) forms a blossom, matches defect 3 (w(0,3)=20), is grabbed by
        // defect 4 (w(1,4)=30) and shrinks to zero while 3 grows to its boundary (40). This is
        // also the spec §7.1 "degenerate implosion" case: after the shatter, the inner leaf
        // regions shrink to radius 0 and implode into a nested blossom.
        let g = CompiledGraph::from_int_edges(
            5,
            &[
                (0, 1, 4, 1),
                (1, 2, 4, 2),
                (0, 2, 4, 4),
                (0, 3, 20, 8),
                (1, 4, 30, 16),
            ],
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
            &[
                (0, 1, 2, 1),
                (1, 2, 2, 2),
                (2, 3, 2, 4),
                (3, 4, 2, 8),
                (4, 0, 2, 16),
            ],
            &[(0, 9, 32)],
        );
        for d in [
            &[0u32, 1, 2, 3, 4][..],
            &[0, 1, 2][..],
            &[1, 2, 3, 4][..],
            &[0, 2, 4][..],
        ] {
            assert_eq!(sparse(&g, d), dense_optimum(&g, d), "defects {d:?}");
        }
    }

    #[test]
    fn blossom_shatter_pairs_the_remaining_cycle() {
        // Same triangle {0,1,2} (w=4) as `blossom_is_shattered_when_it_becomes_inner`, but the
        // parent edge (0,4) and the outer edge (0,3) both land on child 0: the tree path through
        // the blossom is a single child, so shattering must match the *other* two children (1,2)
        // along the remaining cycle instead of leaving one of them exposed.
        let g = CompiledGraph::from_int_edges(
            5,
            &[
                (0, 1, 4, 1),
                (1, 2, 4, 2),
                (0, 2, 4, 4),
                (0, 3, 20, 8),
                (0, 4, 30, 16),
            ],
            &[(3, 40, 32)],
        );
        let d = [0, 1, 2, 3, 4];
        assert_eq!(sparse(&g, &d), dense_optimum(&g, &d));
    }

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
                edges.push((
                    u,
                    v,
                    1 + (next() % wmax as u64) as i64,
                    1u64 << (next() % 8),
                ));
            }
            for _ in 0..extra {
                let u = (next() % n as u64) as u32;
                let v = (next() % n as u64) as u32;
                if u != v {
                    edges.push((
                        u.min(v),
                        u.max(v),
                        1 + (next() % wmax as u64) as i64,
                        1u64 << (next() % 8),
                    ));
                }
            }
            // Two sequential closures (`filter` then `map`) can't both hold `&mut next` live at
            // once, so pick-then-build in one pass instead of chaining adapters.
            let mut boundary: Vec<(u32, i64, u64)> = Vec::new();
            for u in 0..n as u32 {
                if next() % 3 == 0 {
                    boundary.push((
                        u,
                        1 + (next() % (2 * wmax) as u64) as i64,
                        1u64 << (8 + next() % 8),
                    ));
                }
            }
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
            let has_boundary = (0..g.num_nodes() as u32).any(|u| g.boundary(u).is_some());
            prop_assume!(defects.len() % 2 == 0 || has_boundary);
            let (_, sw) = SparseMatcher::new(g.clone()).decode(&defects);
            let (_, dw) = dense_optimum(&g, &defects);
            prop_assert_eq!(sw, dw, "weight differs: sparse {} dense {}", sw, dw);
            // Corrections may differ from the dense oracle only on genuine ties (equal-weight
            // matchings in different homology classes); weight equality above is the invariant,
            // and there is no tie-aware oracle to assert on the correction itself.
        }

        #[test]
        fn decode_errors_is_weight_and_syndrome_exact((g, defects) in graph_strategy()) {
            let has_boundary = (0..g.num_nodes() as u32).any(|u| g.boundary(u).is_some());
            prop_assume!(defects.len() % 2 == 0 || has_boundary);
            let m = SparseMatcher::new(g.clone());
            let (_, w) = m.decode(&defects);
            let mut ehat = vec![0u8; num_cols(&g)];
            let (_, we) = m.decode_errors(&defects, &mut ehat);
            prop_assert_eq!(we, w);
            // Σ weight of marked columns == matching weight, and parity at every node == defect.
            let mut sum = 0i64;
            let mut parity = vec![0u8; g.num_nodes()];
            for u in 0..g.num_nodes() as u32 {
                for (e, &c) in g.edges(u).iter().zip(g.edge_cols(u)) {
                    if u < e.v && ehat[c as usize] == 1 {
                        sum += e.w;
                        parity[u as usize] ^= 1;
                        parity[e.v as usize] ^= 1;
                    }
                }
                if let Some((wb, _)) = g.boundary(u) {
                    if ehat[g.boundary_col(u) as usize] == 1 {
                        sum += wb;
                        parity[u as usize] ^= 1;
                    }
                }
            }
            prop_assert_eq!(sum / 2, w, "marked-edge weight != matching weight");
            for u in 0..g.num_nodes() as u32 {
                prop_assert_eq!(parity[u as usize] == 1, defects.contains(&u), "node {} parity", u);
            }
        }
    }

    /// Column count of a `from_int_edges` graph (edges `0..E`, then boundary entries): one past
    /// the largest column any retained edge or boundary edge carries.
    fn num_cols(g: &CompiledGraph) -> usize {
        let nodes = 0..g.num_nodes() as u32;
        let edge = nodes
            .clone()
            .flat_map(|u| g.edge_cols(u).iter().map(|&c| c + 1));
        let bnd = nodes
            .filter(|&u| g.boundary(u).is_some())
            .map(|u| g.boundary_col(u) + 1);
        edge.chain(bnd).max().unwrap_or(0) as usize
    }

    #[test]
    fn decode_errors_marks_the_matched_paths() {
        // Chain 0-1-2-3 (columns 0,1,2), boundary at 3 (column 3 via from_int_edges numbering:
        // boundary columns follow the edge columns).
        let g = CompiledGraph::from_int_edges(
            4,
            &[(0, 1, 2, 1), (1, 2, 2, 2), (2, 3, 2, 4)],
            &[(3, 5, 8)],
        );
        let m = SparseMatcher::new(g);
        let mut ehat = vec![0u8; 4];
        let (obs, w) = m.decode_errors(&[0, 2], &mut ehat);
        assert_eq!((obs, w), (3, 4));
        assert_eq!(ehat, vec![1, 1, 0, 0]);
        let mut ehat = vec![0u8; 4];
        let (obs, w) = m.decode_errors(&[2], &mut ehat);
        assert_eq!((obs, w), (4 ^ 8, 7));
        assert_eq!(ehat, vec![0, 0, 1, 1]);
    }

    #[test]
    fn repeated_decodes_reuse_state_identically() {
        let g = CompiledGraph::from_int_edges(
            5,
            &[
                (0, 1, 4, 1),
                (1, 2, 4, 2),
                (0, 2, 4, 4),
                (0, 3, 20, 8),
                (1, 4, 30, 16),
            ],
            &[(3, 40, 32)],
        );
        let m = SparseMatcher::new(g);
        let first = m.decode(&[0, 1, 2, 3, 4]);
        for _ in 0..50 {
            assert_eq!(m.decode(&[0, 1, 2, 3, 4]), first);
            assert_eq!(m.decode(&[1, 2]), (2, 4));
        }
    }
}
