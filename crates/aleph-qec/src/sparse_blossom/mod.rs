//! Sparse Blossom: minimum-weight perfect matching by local, event-driven region growth on the
//! detector graph (Higgott & Gidney, arXiv:2303.15933), re-derived from Edmonds' primal-dual
//! blossom algorithm. See `docs/superpowers/specs/2026-09-24-sparse-blossom-design.md`.
//!
//! Module map: [`graph`] compiles the [`crate::MatchingGraph`] into a CSR with doubled integer
//! weights; [`state`] holds the per-shot mutable arenas and the event heap; [`flooder`] grows and
//! shrinks regions and detects collisions; [`matcher`] runs the alternating-tree operations and
//! resolves the final matching. [`SparseMatcher`] is the crate-facing entry point.

pub(crate) mod flooder;
pub(crate) mod graph;
pub(crate) mod matcher;
pub(crate) mod state;

use std::sync::Mutex;

#[allow(unused_imports)]
pub(crate) use graph::{CompiledGraph, WEIGHT_SCALE};
use state::*;

impl State {
    /// Decode one syndrome: `(observable mask, total weight in undoubled units)`.
    pub(crate) fn run(&mut self, g: &CompiledGraph, defects: &[u32]) -> (u64, i64) {
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
        let (obs, w) = self.resolve();
        (obs, w / 2)
    }
}

/// Crate-facing sparse matcher: a compiled graph plus a pool of reusable per-shot states so
/// `decode(&self)` works from `rayon` without a lock held during the decode.
pub(crate) struct SparseMatcher {
    graph: CompiledGraph,
    pool: Mutex<Vec<State>>,
}

impl SparseMatcher {
    pub(crate) fn new(graph: CompiledGraph) -> Self {
        SparseMatcher {
            graph,
            pool: Mutex::new(Vec::new()),
        }
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
            .unwrap_or_else(|| State::new(self.graph.num_nodes()));
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
        f.debug_struct("SparseMatcher")
            .field("nodes", &self.graph.num_nodes())
            .finish()
    }
}

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
        // Three mutually equidistant defects tie at t=2; one pair augments, the third grows the
        // tree, the inner region implodes, the blossom grows to the boundary at node 2.
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
        // defect 4 (w(1,4)=30) and shrinks to zero while 3 grows to its boundary (40).
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
}
