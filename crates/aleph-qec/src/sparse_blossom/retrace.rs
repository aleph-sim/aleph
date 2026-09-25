//! Shortest-path retrace of a matched pair, for the per-column error output. The matcher only
//! remembers *which* defects are paired and the tight path length; like PyMatching's
//! `decode_to_edges`, the actual edges are recovered afterwards by a Dijkstra from one defect
//! bounded at that length. Ties are broken by `(distance, node index)` and first-relaxation, so
//! the result is deterministic; on a genuine tie the retraced path may differ from the path the
//! region growth took, which is why `decode_errors`' observable parity can differ from
//! `decode`'s — never its weight, never `H ê = s`.
//!
//! Negative edges (mechanisms with p > 0.5) are not relaxed: an undirected negative edge is a
//! negative 2-cycle, on which a Dijkstra never terminates. A pair whose path needs one is left
//! unmarked (or retraced along a non-negative path, if one fits the bound).

use std::cmp::Reverse;

use super::graph::CompiledGraph;
use super::state::{CEdge, NodeId, State, BOUNDARY, NONE};

impl State {
    /// XOR the edges of a shortest path realising `e` (`e.from` → `e.to`, or `e.from` → any
    /// boundary edge when `e.to == BOUNDARY`) into `ehat` by representative column.
    pub(crate) fn retrace(&mut self, g: &CompiledGraph, e: &CEdge, ehat: &mut [u8]) {
        // Sized lazily so `decode`-only `State`s never allocate retrace scratch.
        let n = g.num_nodes();
        if self.rt_dist.len() < n {
            self.rt_dist.resize(n, i64::MAX);
            self.rt_pred.resize(n, (NONE, 0));
        }
        for &u in &self.rt_touched {
            self.rt_dist[u as usize] = i64::MAX;
            self.rt_pred[u as usize] = (NONE, 0);
        }
        self.rt_touched.clear();
        self.rt_heap.clear();

        let src = e.from;
        self.rt_dist[src as usize] = 0;
        self.rt_touched.push(src);
        self.rt_heap.push(Reverse((0, src)));
        let mut end: Option<(NodeId, Option<u32>)> = None; // (last node, boundary column)

        while let Some(Reverse((d, u))) = self.rt_heap.pop() {
            if d > self.rt_dist[u as usize] {
                continue;
            }
            if e.to == BOUNDARY {
                if let Some((wb, _)) = g.boundary(u) {
                    if d + wb == e.weight {
                        end = Some((u, Some(g.boundary_col(u))));
                        break;
                    }
                }
            } else if u == e.to {
                debug_assert!(
                    d == e.weight || g.has_negative_weight(),
                    "retrace reached the mate at distance {d}, not the tight {}",
                    e.weight
                );
                end = Some((u, None));
                break;
            }
            for (k, ne) in g.edges(u).iter().enumerate() {
                if ne.w < 0 {
                    continue; // negative 2-cycle: see the module doc
                }
                let nd = d + ne.w;
                if nd > e.weight {
                    continue; // bounded: nothing past the tight length can be on the path
                }
                let v = ne.v as usize;
                if nd < self.rt_dist[v] {
                    if self.rt_dist[v] == i64::MAX {
                        self.rt_touched.push(ne.v);
                    }
                    self.rt_dist[v] = nd;
                    self.rt_pred[v] = (u, g.edge_cols(u)[k]);
                    self.rt_heap.push(Reverse((nd, ne.v)));
                }
            }
        }

        // On a non-negative graph tightness guarantees `end`; the fallback keeps library code
        // panic-free when it isn't reached (a skipped negative edge, or a bug upstream — the
        // latter caught by the debug assertion in tests).
        debug_assert!(
            end.is_some() || g.has_negative_weight(),
            "retrace did not reach the mate within the tight weight"
        );
        let Some((mut v, bcol)) = end else { return };
        if let Some(c) = bcol {
            ehat[c as usize] ^= 1;
        }
        while v != src {
            let (p, col) = self.rt_pred[v as usize];
            ehat[col as usize] ^= 1;
            v = p;
        }
    }
}
