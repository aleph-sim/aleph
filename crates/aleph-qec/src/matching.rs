//! Matching graph: the weighted graph that matching-based decoders operate on, built from a
//! graph-like Detector Error Model.
//!
//! Both MWPM (Q1) and Union-Find (Q2) decode by working over this graph. Its nodes are the
//! detectors of a [`DetectorErrorModel`] plus a single virtual **boundary** node; its edges are
//! the error mechanisms. A mechanism that flips one detector becomes an edge from that detector
//! to the boundary (the syndrome can be "explained" by an error reaching the code's edge); a
//! mechanism that flips two detectors becomes an edge between them. Each edge carries the
//! logical observables that mechanism flips, so a decoder can XOR them along the matched edges to
//! reconstruct the correction.
//!
//! # Weights
//!
//! An edge's weight is `w = ln((1 - p) / p)`, so a minimum-weight perfect matching is the most
//! likely set of error mechanisms consistent with the syndrome (the standard MWPM weighting;
//! Dennis et al. 2002, and PyMatching — Higgott, arXiv:2105.13082). The logarithm base only
//! scales every weight uniformly and so does not change which matching is minimal.
//!
//! # Parallel edges
//!
//! Two mechanisms with the *same* endpoints and the *same* observable set are parallel edges;
//! they are merged into one before weighting. Two independent mechanisms each fire the edge with
//! its own probability, and the edge is "on" iff an odd number fire, so their probabilities
//! combine by the XOR rule `p = p₁(1−p₂) + p₂(1−p₁) = p₁ + p₂ − 2p₁p₂` — the same combination
//! PyMatching uses, and the same one the DEM builder applies (`surface.rs`). Naive probability
//! addition is wrong here: it can push the merged probability past `0.5` and yield a *negative*
//! weight, whereas the XOR rule keeps `p ≤ 0.5` (hence `w ≥ 0`) whenever the inputs are `≤ 0.5`.
//!
//! # Graph-like only
//!
//! A mechanism flipping three or more detectors is a hyperedge; this builder rejects it with
//! [`Error::NonGraphlike`]. Surface-code memory DEMs (Q0-03) are graph-like by construction;
//! hypergraph decoding (qLDPC) is Phase Q5.

use std::collections::HashMap;

use crate::dem::DetectorErrorModel;
use crate::error::{Error, Result};

/// A node in the matching graph.
///
/// Detectors take indices `0..num_detectors`; the single virtual boundary node is index
/// `num_detectors` (see [`MatchingGraph::boundary`]).
pub type NodeId = usize;

/// One weighted edge of the matching graph.
///
/// `a < b` always; `b` may be the boundary node. `observables` is the sorted, parity-reduced set
/// of logical observables the underlying mechanism flips (used to reconstruct the correction).
#[derive(Clone, Debug, PartialEq)]
pub struct MatchingEdge {
    /// Lower-indexed endpoint.
    pub a: NodeId,
    /// Higher-indexed endpoint (possibly the boundary node).
    pub b: NodeId,
    /// Combined firing probability of this edge, in `(0, 1)`.
    pub prob: f64,
    /// Edge weight `ln((1 - prob) / prob)`. Non-negative when `prob ≤ 0.5`.
    pub weight: f64,
    /// Logical observables flipped when this edge is part of the error (sorted ascending).
    pub observables: Vec<u32>,
}

/// A weighted matching graph over detectors plus a virtual boundary node.
///
/// Build one with [`MatchingGraph::from_dem`]. The graph is immutable afterwards: a decoder
/// builds it once per DEM and reuses it across shots.
#[derive(Clone, Debug)]
pub struct MatchingGraph {
    num_detectors: usize,
    num_observables: usize,
    edges: Vec<MatchingEdge>,
    /// `adjacency[n]` = indices into [`edges`](Self::edges) incident to node `n`. Length is
    /// `num_detectors + 1` (the trailing entry is the boundary node).
    adjacency: Vec<Vec<usize>>,
}

impl MatchingGraph {
    /// Build a matching graph from a graph-like [`DetectorErrorModel`].
    ///
    /// Single-detector mechanisms become boundary edges, two-detector mechanisms become edges
    /// between detectors, parallel edges are merged (see the module docs), and mechanisms with no
    /// detector endpoints (undetectable / purely-logical noise) are dropped — they cannot
    /// participate in any matching. Within a mechanism, a detector or observable listed an even
    /// number of times cancels (Stim parity semantics).
    ///
    /// # Errors
    /// Returns [`Error::NonGraphlike`] if any mechanism flips three or more detectors.
    pub fn from_dem(dem: &DetectorErrorModel) -> Result<Self> {
        let boundary = dem.detectors;

        // Accumulate parallel edges' probabilities, keyed by (endpoints, observable set). A
        // separate `order` vector keeps edge order deterministic (first-seen) rather than
        // HashMap iteration order.
        let mut merged: HashMap<(NodeId, NodeId, Vec<u32>), f64> = HashMap::new();
        let mut order: Vec<(NodeId, NodeId, Vec<u32>)> = Vec::new();

        for e in &dem.errors {
            // A non-positive (or NaN) probability never fires; `!(p > 0.0)` is true for NaN, so
            // this also rejects NaN per the IEEE-754 discipline in CLAUDE.md / ADR 0006. A
            // probability of 1.0 (or more) is a deterministic fault with `ln(0) = -inf` weight,
            // which has no place in a matching; skip it too.
            if !(e.prob > 0.0 && e.prob < 1.0) {
                continue;
            }

            let dets = odd_parity(&e.dets);
            let obs = odd_parity(&e.obs);

            let (a, b) = match dets.len() {
                // No detector endpoints: undetectable noise (may flip an observable but can never
                // be matched). Drop it.
                0 => continue,
                1 => (dets[0] as NodeId, boundary),
                2 => (dets[0] as NodeId, dets[1] as NodeId),
                n => return Err(Error::NonGraphlike { dets: n }),
            };
            // `odd_parity` yields distinct, ascending detectors and `boundary` is the largest
            // index, so `a < b` already holds; assert the invariant rather than re-sort.
            debug_assert!(a < b, "endpoints must be distinct and ordered");

            let key = (a, b, obs);
            if let Some(p) = merged.get_mut(&key) {
                *p = xor_combine(*p, e.prob);
            } else {
                merged.insert(key.clone(), e.prob);
                order.push(key);
            }
        }

        let mut edges = Vec::with_capacity(order.len());
        let mut adjacency = vec![Vec::new(); dem.detectors + 1];
        for key in order {
            let prob = merged[&key];
            let (a, b, observables) = key;
            let idx = edges.len();
            adjacency[a].push(idx);
            adjacency[b].push(idx);
            edges.push(MatchingEdge {
                a,
                b,
                prob,
                weight: edge_weight(prob),
                observables,
            });
        }

        Ok(MatchingGraph {
            num_detectors: dem.detectors,
            num_observables: dem.observables,
            edges,
            adjacency,
        })
    }

    /// Number of detector nodes (their indices are `0..num_detectors()`).
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of logical observables the edges may flip.
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Total node count: detectors plus the one boundary node.
    pub fn num_nodes(&self) -> usize {
        self.num_detectors + 1
    }

    /// The virtual boundary node's index (`num_detectors()`).
    pub fn boundary(&self) -> NodeId {
        self.num_detectors
    }

    /// Whether `node` is the boundary node.
    pub fn is_boundary(&self, node: NodeId) -> bool {
        node == self.num_detectors
    }

    /// All edges, in deterministic first-seen order.
    pub fn edges(&self) -> &[MatchingEdge] {
        &self.edges
    }

    /// Indices (into [`edges`](Self::edges)) of the edges incident to `node`.
    ///
    /// # Panics
    /// If `node >= num_nodes()`.
    pub fn incident(&self, node: NodeId) -> &[usize] {
        &self.adjacency[node]
    }
}

/// Edge weight `ln((1 - p) / p)`.
///
/// For `p ∈ (0, 0.5]` this is `≥ 0`; smaller `p` ⇒ larger weight ⇒ a less-likely (more costly)
/// edge for the matching to use. Caller guarantees `p ∈ (0, 1)`.
fn edge_weight(p: f64) -> f64 {
    ((1.0 - p) / p).ln()
}

/// Combine two independent firing probabilities under XOR: `p₁ + p₂ − 2p₁p₂` — the probability
/// that an odd number of the two mechanisms fire. Keeps the result in `[0, 0.5]` when both
/// inputs are (so merged weights stay non-negative).
fn xor_combine(p1: f64, p2: f64) -> f64 {
    p1 + p2 - 2.0 * p1 * p2
}

/// Reduce a sorted index slice to the values appearing an *odd* number of times (Stim semantics:
/// flipping the same detector/observable twice cancels). Output is sorted and de-duplicated.
fn odd_parity(sorted: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(sorted.len());
    let mut i = 0;
    while i < sorted.len() {
        let v = sorted[i];
        let mut count = 0usize;
        while i < sorted.len() && sorted[i] == v {
            count += 1;
            i += 1;
        }
        if count % 2 == 1 {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::DemError;
    use proptest::prelude::*;

    /// `ln((1-p)/p)` to compare against in tests.
    fn w(p: f64) -> f64 {
        ((1.0 - p) / p).ln()
    }

    #[test]
    fn single_detector_error_connects_to_boundary() {
        let dem = DetectorErrorModel::parse("error(0.1) D0\ndetector D1\n").unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.num_detectors(), 2);
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.boundary(), 2);
        assert_eq!(g.edges().len(), 1);
        let e = &g.edges()[0];
        assert_eq!((e.a, e.b), (0, 2)); // detector 0 to boundary (node 2)
        assert!(g.is_boundary(e.b));
        assert!((e.weight - w(0.1)).abs() < 1e-12);
        assert!(e.observables.is_empty());
        // Incidence: boundary and detector 0 both see the edge; detector 1 is isolated.
        assert_eq!(g.incident(0), &[0]);
        assert_eq!(g.incident(2), &[0]);
        assert!(g.incident(1).is_empty());
    }

    #[test]
    fn two_detector_error_connects_detectors() {
        let dem = DetectorErrorModel::parse("error(0.2) D0 D1 L0\n").unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        let e = &g.edges()[0];
        assert_eq!((e.a, e.b), (0, 1));
        assert!(!g.is_boundary(e.b));
        assert_eq!(e.observables, vec![0]);
        assert!((e.weight - w(0.2)).abs() < 1e-12);
    }

    #[test]
    fn parallel_edges_merge_by_xor() {
        // Two identical D0 D1 mechanisms merge into one with p = 0.1 + 0.1 - 2*0.01 = 0.18.
        let dem = DetectorErrorModel::parse("error(0.1) D0 D1\nerror(0.1) D0 D1\n").unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.edges().len(), 1);
        let e = &g.edges()[0];
        assert!((e.prob - 0.18).abs() < 1e-12, "merged prob {}", e.prob);
        assert!((e.weight - w(0.18)).abs() < 1e-12);
    }

    #[test]
    fn parallel_edges_stay_below_half_and_keep_weight_nonnegative() {
        // Two p=0.4 mechanisms: naive addition gives 0.8 (negative weight!); XOR gives 0.48.
        let dem = DetectorErrorModel::parse("error(0.4) D0 D1\nerror(0.4) D0 D1\n").unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        let e = &g.edges()[0];
        assert!((e.prob - 0.48).abs() < 1e-12, "merged prob {}", e.prob);
        assert!(
            e.weight >= 0.0,
            "weight {} must stay non-negative",
            e.weight
        );
    }

    #[test]
    fn different_observables_are_distinct_edges() {
        // Same endpoints, different observable sets: not parallel, two edges.
        let dem = DetectorErrorModel::parse("error(0.1) D0 D1\nerror(0.1) D0 D1 L0\n").unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.edges().len(), 2);
        assert_eq!(g.edges()[0].observables, Vec::<u32>::new());
        assert_eq!(g.edges()[1].observables, vec![0]);
        // Both incident to detector 0.
        assert_eq!(g.incident(0).len(), 2);
    }

    #[test]
    fn hyperedge_is_rejected() {
        let dem = DetectorErrorModel::parse("error(0.1) D0 D1 D2\n").unwrap();
        match MatchingGraph::from_dem(&dem) {
            Err(Error::NonGraphlike { dets }) => assert_eq!(dets, 3),
            other => panic!("expected NonGraphlike, got {other:?}"),
        }
    }

    #[test]
    fn zero_detector_error_is_dropped() {
        // An observable-only (undetectable) mechanism contributes no edge, but the boundary node
        // still exists.
        let dem = DetectorErrorModel::parse("error(0.1) L0\ndetector D2\n").unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.edges().len(), 0);
        assert_eq!(g.num_nodes(), 4); // D0..D2 + boundary
    }

    #[test]
    fn duplicate_targets_cancel_by_parity() {
        // D0 D0 cancels -> 0 detectors -> dropped. D1 D1 D2 -> just D2 -> boundary edge.
        let dem = DetectorErrorModel {
            detectors: 3,
            observables: 0,
            errors: vec![
                DemError {
                    prob: 0.1,
                    dets: vec![0, 0],
                    obs: vec![],
                },
                DemError {
                    prob: 0.1,
                    dets: vec![1, 1, 2],
                    obs: vec![],
                },
            ],
        };
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.edges().len(), 1);
        let e = &g.edges()[0];
        assert_eq!((e.a, e.b), (2, 3)); // D2 to boundary
    }

    #[test]
    fn zero_and_one_probabilities_are_skipped() {
        let dem = DetectorErrorModel {
            detectors: 2,
            observables: 0,
            errors: vec![
                DemError {
                    prob: 0.0,
                    dets: vec![0],
                    obs: vec![],
                },
                DemError {
                    prob: 1.0,
                    dets: vec![1],
                    obs: vec![],
                },
                DemError {
                    prob: 0.1,
                    dets: vec![0, 1],
                    obs: vec![],
                },
            ],
        };
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.edges().len(), 1);
        assert_eq!((g.edges()[0].a, g.edges()[0].b), (0, 1));
    }

    #[test]
    fn repetition_code_hand_drawn_adjacency() {
        // A distance-3 repetition-code-style DEM (cf. the dem.rs parser test): two parity-check
        // detectors D0, D1 and a boundary. Three error mechanisms:
        //   error(0.125) D0       -> left boundary edge   (0, boundary=2)
        //   error(0.125) D0 D1    -> bulk edge            (0, 1)
        //   error(0.125) D1 L0    -> right boundary edge  (1, boundary=2), flips observable 0
        let dem =
            DetectorErrorModel::parse("error(0.125) D0\nerror(0.125) D0 D1\nerror(0.125) D1 L0\n")
                .unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.num_detectors(), 2);
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.edges().len(), 3);

        // Edge endpoints, in build order.
        let endpoints: Vec<(NodeId, NodeId)> = g.edges().iter().map(|e| (e.a, e.b)).collect();
        assert_eq!(endpoints, vec![(0, 2), (0, 1), (1, 2)]);

        // Only the right boundary edge flips the observable.
        assert_eq!(g.edges()[0].observables, Vec::<u32>::new());
        assert_eq!(g.edges()[1].observables, Vec::<u32>::new());
        assert_eq!(g.edges()[2].observables, vec![0]);

        // Adjacency: D0 touches edges 0,1; D1 touches edges 1,2; boundary touches edges 0,2.
        assert_eq!(g.incident(0), &[0, 1]);
        assert_eq!(g.incident(1), &[1, 2]);
        assert_eq!(g.incident(2), &[0, 2]);
    }

    #[test]
    fn surface_code_d3_d5_node_counts_and_graphlike() {
        use crate::SurfaceCode;
        for d in [3usize, 5] {
            let exp = SurfaceCode::new(d).memory_z_experiment(d);
            let mechs = exp.phenomenological_mechanisms(0.01, 0.01);
            let dem = crate::build_dem(&exp.annotated, &mechs).unwrap();
            let g = MatchingGraph::from_dem(&dem).expect("surface DEM is graph-like");
            assert_eq!(g.num_detectors(), dem.detectors);
            assert_eq!(g.num_nodes(), dem.detectors + 1);
            assert_eq!(g.num_observables(), 1);
            assert!(!g.edges().is_empty());
            // At least one edge reaches the boundary (spatial/temporal boundary detectors).
            assert!(
                g.edges().iter().any(|e| g.is_boundary(e.b)),
                "d={d}: expected boundary edges"
            );
            // At least one edge flips the logical observable.
            assert!(
                g.edges().iter().any(|e| !e.observables.is_empty()),
                "d={d}: expected an observable-flipping edge"
            );
        }
    }

    // ---- Property tests -------------------------------------------------------------------

    prop_compose! {
        // A graph-like DEM: every mechanism flips 0, 1, or 2 distinct detectors, p in (0, 0.5].
        fn arb_graphlike_dem()(detectors in 1usize..8, observables in 0usize..3)
            (errors in prop::collection::vec(arb_graphlike_error(detectors, observables), 0..12),
             detectors in Just(detectors), observables in Just(observables))
            -> DetectorErrorModel {
            DetectorErrorModel { detectors, observables, errors }
        }
    }

    fn arb_graphlike_error(
        detectors: usize,
        observables: usize,
    ) -> impl Strategy<Value = DemError> {
        let dets = prop::collection::hash_set(0u32..detectors as u32, 0..=2)
            .prop_map(|s| s.into_iter().collect::<Vec<_>>());
        let obs = if observables == 0 {
            Just(Vec::new()).boxed()
        } else {
            prop::collection::hash_set(0u32..observables as u32, 0..=1)
                .prop_map(|s| s.into_iter().collect::<Vec<_>>())
                .boxed()
        };
        (0.0001f64..=0.5, dets, obs).prop_map(|(p, d, o)| DemError::new(p, d, o))
    }

    proptest! {
        #[test]
        fn all_weights_nonnegative_for_p_le_half(dem in arb_graphlike_dem()) {
            let g = MatchingGraph::from_dem(&dem).expect("graph-like input never errors");
            for e in g.edges() {
                prop_assert!(e.weight >= 0.0, "weight {} < 0 for prob {}", e.weight, e.prob);
                prop_assert!(e.prob > 0.0 && e.prob <= 0.5 + 1e-12, "prob {} out of (0,0.5]", e.prob);
            }
        }

        #[test]
        fn endpoints_ordered_and_in_range(dem in arb_graphlike_dem()) {
            let g = MatchingGraph::from_dem(&dem).unwrap();
            let boundary = g.boundary();
            for e in g.edges() {
                prop_assert!(e.a < e.b);
                prop_assert!(e.b <= boundary);
            }
        }

        #[test]
        fn adjacency_consistent_with_edges(dem in arb_graphlike_dem()) {
            let g = MatchingGraph::from_dem(&dem).unwrap();
            // Every incident-list entry points back to an edge with that endpoint.
            for node in 0..g.num_nodes() {
                for &ei in g.incident(node) {
                    let e = &g.edges()[ei];
                    prop_assert!(e.a == node || e.b == node);
                }
            }
            // Every edge appears in exactly the incidence lists of its two endpoints.
            let total: usize = (0..g.num_nodes()).map(|n| g.incident(n).len()).sum();
            prop_assert_eq!(total, g.edges().len() * 2);
        }
    }
}
