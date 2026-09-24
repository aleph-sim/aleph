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
            .map(|e| {
                (
                    e.a as u32,
                    e.b as u32,
                    scaled(e.weight),
                    obs_mask(&e.observables),
                )
            })
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
        CompiledGraph {
            num_nodes,
            num_observables,
            offsets,
            adj,
            boundary_w,
            boundary_obs,
        }
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
