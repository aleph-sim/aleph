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

/// One endpoint's view of an input edge while building the CSR: `(neighbour, doubled weight,
/// input edge index, obs, column)`.
type HalfEdge = (u32, i64, usize, u64, u32);

/// CSR detector graph plus a per-node boundary edge.
#[derive(Clone, Debug)]
pub(crate) struct CompiledGraph {
    num_nodes: usize,
    offsets: Vec<u32>,
    adj: Vec<Edge>,
    /// Representative DEM column of `adj[k]` (parallel to `adj`).
    adj_col: Vec<u32>,
    /// Boundary edge per node: `(doubled weight, obs)`; weight `i64::MAX` when absent.
    boundary_w: Vec<i64>,
    boundary_obs: Vec<u64>,
    /// Representative DEM column of each node's boundary edge (unused when absent).
    boundary_col: Vec<u32>,
}

impl CompiledGraph {
    /// Compile `g`: parallel detector–detector edges collapse to the lightest (ties by edge
    /// index), boundary edges to the lightest per node.
    pub(crate) fn from_matching_graph(g: &MatchingGraph) -> Self {
        let scaled = |w: f64| 2 * (w * WEIGHT_SCALE).round() as i64;
        let boundary = g.boundary();
        let edges: Vec<(u32, u32, i64, u64, u32)> = g
            .edges()
            .iter()
            .filter(|e| e.b != boundary)
            .map(|e| {
                (
                    e.a as u32,
                    e.b as u32,
                    scaled(e.weight),
                    obs_mask(&e.observables),
                    e.column,
                )
            })
            .collect();
        let bnd: Vec<(u32, i64, u64, u32)> = g
            .edges()
            .iter()
            .filter(|e| e.b == boundary)
            .map(|e| {
                (
                    e.a as u32,
                    scaled(e.weight),
                    obs_mask(&e.observables),
                    e.column,
                )
            })
            .collect();
        Self::build(g.num_detectors(), &edges, &bnd)
    }

    /// Test constructor from raw integer weights (doubled internally). Columns are numbered by
    /// position: edge `k` is column `k`, boundary entry `k` is column `edges.len() + k`.
    #[cfg(test)]
    pub(crate) fn from_int_edges(
        num_nodes: usize,
        edges: &[(u32, u32, i64, u64)],
        boundary: &[(u32, i64, u64)],
    ) -> Self {
        let e: Vec<_> = edges
            .iter()
            .enumerate()
            .map(|(k, &(a, b, w, o))| (a, b, 2 * w, o, k as u32))
            .collect();
        let b: Vec<_> = boundary
            .iter()
            .enumerate()
            .map(|(k, &(a, w, o))| (a, 2 * w, o, (edges.len() + k) as u32))
            .collect();
        Self::build(num_nodes, &e, &b)
    }

    fn build(
        num_nodes: usize,
        edges: &[(u32, u32, i64, u64, u32)],
        boundary: &[(u32, i64, u64, u32)],
    ) -> Self {
        // Per node: (neighbour, weight, edge index, obs, column) sorted so the lightest parallel
        // edge comes first, then dedup by neighbour.
        let mut per: Vec<Vec<HalfEdge>> = vec![Vec::new(); num_nodes];
        for (k, &(a, b, w, o, c)) in edges.iter().enumerate() {
            per[a as usize].push((b, w, k, o, c));
            per[b as usize].push((a, w, k, o, c));
        }
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut adj = Vec::new();
        let mut adj_col = Vec::new();
        offsets.push(0u32);
        for list in per.iter_mut() {
            list.sort_unstable_by_key(|&(v, w, k, _, _)| (v, w, k));
            list.dedup_by_key(|x| x.0);
            adj.extend(list.iter().map(|&(v, w, _, obs, _)| Edge { v, w, obs }));
            adj_col.extend(list.iter().map(|x| x.4));
            offsets.push(adj.len() as u32);
        }
        let mut boundary_w = vec![i64::MAX; num_nodes];
        let mut boundary_obs = vec![0u64; num_nodes];
        let mut boundary_col = vec![0u32; num_nodes];
        for &(a, w, o, c) in boundary {
            let a = a as usize;
            if w < boundary_w[a] {
                boundary_w[a] = w;
                boundary_obs[a] = o;
                boundary_col[a] = c;
            }
        }
        CompiledGraph {
            num_nodes,
            offsets,
            adj,
            adj_col,
            boundary_w,
            boundary_obs,
            boundary_col,
        }
    }

    pub(crate) fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    pub(crate) fn edges(&self, u: u32) -> &[Edge] {
        let u = u as usize;
        &self.adj[self.offsets[u] as usize..self.offsets[u + 1] as usize]
    }

    /// Representative columns of `edges(u)`, parallel to it.
    #[inline]
    pub(crate) fn edge_cols(&self, u: u32) -> &[u32] {
        let u = u as usize;
        &self.adj_col[self.offsets[u] as usize..self.offsets[u + 1] as usize]
    }

    #[inline]
    pub(crate) fn boundary(&self, u: u32) -> Option<(i64, u64)> {
        let w = self.boundary_w[u as usize];
        (w != i64::MAX).then_some((w, self.boundary_obs[u as usize]))
    }

    /// Representative column of `u`'s boundary edge (meaningful only when `boundary(u)` is `Some`).
    #[inline]
    pub(crate) fn boundary_col(&self, u: u32) -> u32 {
        self.boundary_col[u as usize]
    }
}

fn obs_mask(observables: &[u32]) -> u64 {
    observables.iter().fold(0u64, |m, &o| m | (1u64 << o))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::{DemError, DetectorErrorModel};
    use crate::matching::MatchingGraph;

    fn graph(errors: Vec<DemError>) -> CompiledGraph {
        let dem = DetectorErrorModel {
            detectors: 3,
            observables: 2,
            errors,
        };
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
