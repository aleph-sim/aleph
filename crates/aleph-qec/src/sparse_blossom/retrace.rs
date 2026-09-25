//! Shortest-path retrace of a matched pair, for the per-column error output. The matcher only
//! remembers *which* defects are paired and the tight path length; like PyMatching's
//! `decode_to_edges`, the actual edges are recovered afterwards by a Dijkstra from one defect
//! bounded at that length. Ties are broken by `(distance, node index)` and first-relaxation, so
//! the result is deterministic; on a genuine tie the retraced path may differ from the path the
//! region growth took, which is why `decode_errors`' observable parity can differ from
//! `decode`'s — never its weight, never `H ê = s`.

use std::cmp::Reverse;

use super::graph::CompiledGraph;
use super::state::{CEdge, NodeId, State, BOUNDARY, NONE};

impl State {
    /// XOR the edges of a shortest path realising `e` (`e.from` → `e.to`, or `e.from` → any
    /// boundary edge when `e.to == BOUNDARY`) into `ehat` by representative column.
    pub(crate) fn retrace(&mut self, g: &CompiledGraph, e: &CEdge, ehat: &mut [u8]) {
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
                debug_assert_eq!(
                    d, e.weight,
                    "retrace reached the mate at the wrong distance"
                );
                end = Some((u, None));
                break;
            }
            for (k, ne) in g.edges(u).iter().enumerate() {
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

        // Tightness guarantees `end`; the fallback keeps library code panic-free if it ever
        // isn't (a bug upstream, caught by the debug assertion in tests).
        debug_assert!(
            end.is_some(),
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
