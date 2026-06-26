//! [`MwpmDecoder`] — a from-scratch minimum-weight perfect matching decoder over the
//! [`MatchingGraph`] built from a DEM (Q1-02), with a localized matching reformulation (Q1-03).
//!
//! Decoding a syndrome is three steps:
//!
//! 1. **Distances.** Pre-compute, once per DEM, the shortest-path distance and the observable
//!    parity along that path between every detector and every other detector / the boundary, by
//!    running Dijkstra from each detector ([`MatchingGraph`] edge weights are `≥ 0`). The boundary
//!    node is never expanded, so a detector→detector distance never routes "through" the boundary
//!    (that would be two separate boundary matches, which the matching handles itself).
//! 2. **Matching.** Find the minimum-weight way to pair each fired detector (defect) with another
//!    defect or with the boundary, via Edmonds' blossom ([`crate::blossom`]). Two equivalent
//!    encodings are implemented:
//!     - [`decode_dense`](MwpmDecoder::decode_dense) — the Q1-02 reference: the complete graph of
//!       defect pairs plus a private boundary clone per defect (clones interconnected at cost 0),
//!       solved as a maximum-cardinality maximum-weight matching on `2n` nodes. `O(n²)` edges.
//!     - [`decode`](Decoder::decode) — the Q1-03 production path: drop the clones and solve a
//!       *non-perfect* maximum-weight matching on just the `n` defect nodes using *savings*
//!       weights `b_i + b_j − dist(i,j)` (an unmatched defect goes to the boundary). Half the
//!       nodes, no clone clique, and only positive-savings edges — exactly the same optimum, ~5×
//!       faster at d=11. See [`decode_local`](MwpmDecoder::decode_local).
//! 3. **Correction.** XOR the observable parity along every matched path. The result is the
//!    decoder's predicted logical-observable flip.
//!
//! This is the textbook MWPM decoder (Dennis et al. 2002; Higgott, PyMatching, arXiv:2105.13082).
//! It wraps nothing — the matching is our own blossom — which is the point of the exercise
//! (ROADMAP Phase B): the understanding it builds feeds Union-Find (Q2), GPU (Q3), and hardware.
//!
//! Both paths still run the textbook O(n·m) blossom on a candidate graph, so cost grows with the
//! defect count. Closing the remaining gap to the Q1-03 ≥10× target needs local region-growing
//! (Sparse Blossom; Higgott & Gidney, arXiv:2303.15933), a follow-up. The all-pairs pre-compute is
//! `O(D · E log D)` time and `O(D²)` memory; fine for the d ≤ 13 sizes measured here.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::blossom::max_weight_matching;
use crate::decoder::Decoder;
use crate::dem::DetectorErrorModel;
use crate::error::Result;
use crate::matching::MatchingGraph;
use crate::syndrome::{Correction, Syndrome};

/// Fixed-point scale for converting real edge weights `ln((1-p)/p)` to integers. Edmonds' blossom
/// compares dual-variable slacks for equality, which must be exact (CLAUDE.md / ADR 0006 forbids
/// float comparisons gating correctness), so the whole matching runs on scaled integers. `2^24`
/// keeps ~7 significant digits — finer than PyMatching's default discretisation — while staying
/// far inside `i64` for the largest path sums at the distances we target.
const WEIGHT_SCALE: f64 = (1u64 << 24) as f64;

/// Integer "infinity" for unreachable pairs. Kept well below `i64::MAX` so summing two of them
/// (or adding a finite distance) cannot overflow.
const INF: i64 = i64::MAX / 4;

/// Default neighbours-per-defect kept in the localized matching graph. The default keeps *every*
/// positive-savings neighbour (no cap), which is weight-exact — the savings prune alone never
/// drops an edge the optimum needs. A finite cap trades that guarantee for speed and is opt-in via
/// [`MwpmDecoder::with_locality_k`]; it must be validated weight-identical on the differential test
/// for the target distance (K = 12 was already too aggressive at d = 11).
const DEFAULT_LOCALITY_K: usize = usize::MAX;

/// A minimum-weight perfect matching decoder for a fixed [`DetectorErrorModel`].
#[derive(Clone, Debug)]
pub struct MwpmDecoder {
    num_detectors: usize,
    num_observables: usize,
    /// Target stride: detectors `0..num_detectors` plus the boundary at index `num_detectors`.
    stride: usize,
    /// `dist[src * stride + dst]` = scaled shortest-path distance from detector `src` to node
    /// `dst` (a detector, or the boundary at `num_detectors`); [`INF`] if unreachable.
    dist: Vec<i64>,
    /// `parity[src * stride + dst]` = observable-flip bitmask along that shortest path (bit `o`
    /// set ⇔ observable `o` flipped an odd number of times).
    parity: Vec<u64>,
    /// Neighbours-per-defect kept in the localized matching graph (see [`DEFAULT_LOCALITY_K`]).
    locality_k: usize,
}

impl MwpmDecoder {
    /// Build a decoder for `dem`.
    ///
    /// # Errors
    /// Propagates [`crate::Error::NonGraphlike`] if the DEM has a hyperedge (matching needs a
    /// graph-like DEM).
    pub fn new(dem: &DetectorErrorModel) -> Result<Self> {
        let graph = MatchingGraph::from_dem(dem)?;
        Ok(Self::from_graph(&graph))
    }

    /// Build a decoder directly from an already-constructed [`MatchingGraph`].
    pub fn from_graph(graph: &MatchingGraph) -> Self {
        let num_detectors = graph.num_detectors();
        let num_observables = graph.num_observables();
        let boundary = graph.boundary();
        let stride = num_detectors + 1;

        // Per-edge integer weight and observable bitmask, looked up by edge index during Dijkstra.
        let edge_w: Vec<i64> = graph
            .edges()
            .iter()
            .map(|e| (e.weight * WEIGHT_SCALE).round() as i64)
            .collect();
        let edge_mask: Vec<u64> = graph
            .edges()
            .iter()
            .map(|e| e.observables.iter().fold(0u64, |m, &o| m | (1u64 << o)))
            .collect();

        let mut dist = vec![INF; num_detectors * stride];
        let mut parity = vec![0u64; num_detectors * stride];
        for src in 0..num_detectors {
            dijkstra_from(
                graph,
                src,
                boundary,
                &edge_w,
                &edge_mask,
                &mut dist[src * stride..(src + 1) * stride],
                &mut parity[src * stride..(src + 1) * stride],
            );
        }

        MwpmDecoder {
            num_detectors,
            num_observables,
            stride,
            dist,
            parity,
            locality_k: DEFAULT_LOCALITY_K,
        }
    }

    /// Override the neighbours-per-defect cap of the localized matcher (default
    /// [`DEFAULT_LOCALITY_K`]). Larger values approach the dense matching (and its cost); smaller
    /// values are faster but risk missing a far optimal edge. Mainly for benchmarking/tuning.
    pub fn with_locality_k(mut self, k: usize) -> Self {
        self.locality_k = k.max(1);
        self
    }

    #[inline]
    fn boundary(&self) -> usize {
        self.num_detectors
    }

    /// Decode `syndrome` with the **dense** all-pairs matching of Q1-02 (every defect pair plus
    /// the full boundary-clone clique). Quadratic in the defect count; kept as the reference
    /// implementation the localized [`decode`](Decoder::decode) is differentially tested against,
    /// and as the benchmark baseline.
    pub fn decode_dense(&self, syndrome: &Syndrome) -> Correction {
        self.decode_with(syndrome, |s, d| s.augmented_edges_dense(d))
            .0
    }

    /// Defect indices that fired in `syndrome` (clamped to this model's detector range).
    fn defects_of(&self, syndrome: &Syndrome) -> Vec<usize> {
        syndrome
            .fired
            .iter()
            .map(|&d| d as usize)
            .filter(|&d| d < self.num_detectors)
            .collect()
    }

    /// Shared decode skeleton: build the augmented graph with `build_edges`, solve the
    /// minimum-weight perfect matching, and reconstruct the correction. Returns the correction and
    /// the total scaled weight of the matched paths (used by differential tests to certify that
    /// the localized and dense graphs reach the same optimum).
    fn decode_with(
        &self,
        syndrome: &Syndrome,
        build_edges: impl Fn(&Self, &[usize]) -> (Vec<(usize, usize, i64)>, i64),
    ) -> (Correction, i64) {
        let defects = self.defects_of(syndrome);
        let n = defects.len();
        if n == 0 {
            return (Correction::none(self.num_observables), 0);
        }
        let boundary = self.boundary();
        let (edges, maxw) = build_edges(self, &defects);

        // Minimum-weight perfect matching = maximum-weight (of offset weights) perfect matching.
        let transformed: Vec<(usize, usize, i64)> =
            edges.iter().map(|&(u, v, w)| (u, v, maxw - w)).collect();
        let mate = max_weight_matching(2 * n, &transformed, true);

        // Reconstruct the correction (XOR observable parity along matched paths) and total weight.
        let mut acc: u64 = 0;
        let mut weight = 0i64;
        for i in 0..n {
            let m = mate[i];
            if m == usize::MAX {
                continue; // unmatched (only if a defect is unreachable; best-effort skip)
            }
            if m == n + i {
                acc ^= self.parity[defects[i] * self.stride + boundary];
                weight += self.dist[defects[i] * self.stride + boundary];
            } else if m < n && i < m {
                acc ^= self.parity[defects[i] * self.stride + defects[m]];
                weight += self.dist[defects[i] * self.stride + defects[m]];
            }
            // m >= n && m != n+i cannot occur: defect i only has an edge to clone n+i.
        }
        let flips = (0..self.num_observables)
            .map(|o| (acc >> o) & 1 == 1)
            .collect();
        (Correction::new(flips), weight)
    }

    /// Dense Q1-02 augmented graph: all defect pairs + full boundary-clone clique. `O(n²)` edges.
    fn augmented_edges_dense(&self, defects: &[usize]) -> (Vec<(usize, usize, i64)>, i64) {
        let n = defects.len();
        let boundary = self.boundary();
        let mut edges: Vec<(usize, usize, i64)> = Vec::new();
        let mut maxw = 0i64;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let db = self.dist[defects[i] * self.stride + boundary];
            if db < INF {
                edges.push((i, n + i, db));
                maxw = maxw.max(db);
            }
            for j in (i + 1)..n {
                let dd = self.dist[defects[i] * self.stride + defects[j]];
                if dd < INF {
                    edges.push((i, j, dd));
                    maxw = maxw.max(dd);
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((n + i, n + j, 0));
            }
        }
        (edges, maxw)
    }

    /// Decode `syndrome` with the localized matching (Q1-03), returning the correction and the
    /// total scaled weight of the matched paths.
    ///
    /// Reformulated to drop the boundary clones entirely. "Matching with a boundary" is equivalent
    /// to a **non-perfect** maximum-weight matching on just the `n` defect nodes, where each edge
    /// carries the *savings* of pairing two defects instead of sending both to the boundary:
    ///
    /// ```text
    ///   savings(i,j) = b_i + b_j − dist(i,j)          (b = distance to boundary)
    ///   total cost   = Σ b_i − Σ_matched savings(i,j) (unmatched defects go to the boundary)
    /// ```
    ///
    /// Minimising cost ⇔ maximising total savings, so an unmatched defect simply pays its boundary
    /// cost. This halves the node count (`n`, not `2n`), needs no clone clique, and keeps only the
    /// positive-savings edges — exactly the boundary prune (`dist(i,j) < b_i + b_j`), which is
    /// weight-exact. A [`locality_k`](Self::locality_k) cap keeps each defect's most-beneficial
    /// neighbours; the differential test certifies the optimum is unchanged.
    fn decode_local(&self, syndrome: &Syndrome) -> (Correction, i64) {
        let defects = self.defects_of(syndrome);
        let n = defects.len();
        if n == 0 {
            return (Correction::none(self.num_observables), 0);
        }
        let boundary = self.boundary();
        let stride = self.stride;
        let b: Vec<i64> = defects
            .iter()
            .map(|&di| self.dist[di * stride + boundary])
            .collect();

        let edges = self.local_savings_edges(&defects, &b);

        // Non-perfect max-weight matching: defects with no worthwhile partner stay unmatched
        // (i.e. matched to the boundary).
        let mate = max_weight_matching(n, &edges, false);

        let mut acc: u64 = 0;
        let mut weight = 0i64;
        for i in 0..n {
            let m = mate[i];
            if m == usize::MAX {
                acc ^= self.parity[defects[i] * stride + boundary];
                weight += b[i];
            } else if i < m {
                acc ^= self.parity[defects[i] * stride + defects[m]];
                weight += self.dist[defects[i] * stride + defects[m]];
            }
        }
        let flips = (0..self.num_observables)
            .map(|o| (acc >> o) & 1 == 1)
            .collect();
        (Correction::new(flips), weight)
    }

    /// Positive-savings candidate edges for the defects (savings = `b_i + b_j − dist(i,j)`),
    /// capped to each defect's [`locality_k`](Self::locality_k) most beneficial neighbours.
    /// `b[i]` is defect `i`'s boundary distance.
    fn local_savings_edges(&self, defects: &[usize], b: &[i64]) -> Vec<(usize, usize, i64)> {
        let n = defects.len();
        let stride = self.stride;
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut scratch: Vec<(i64, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            scratch.clear();
            let row = defects[i] * stride;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = self.dist[row + defects[j]];
                // `saturating_sub` guards the unreachable-boundary (b == INF) case.
                let savings = b[i].saturating_add(b[j]).saturating_sub(d);
                if d < INF && savings > 0 {
                    scratch.push((savings, j));
                }
            }
            if scratch.len() > self.locality_k {
                // Keep the K *largest* savings: partition so the top-K are last, then take them.
                let cut = scratch.len() - self.locality_k;
                scratch.select_nth_unstable(cut);
                scratch.drain(..cut);
            }
            for &(_, j) in &scratch {
                pairs.push((i.min(j), i.max(j)));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
            .iter()
            .map(|&(i, j)| {
                (
                    i,
                    j,
                    b[i] + b[j] - self.dist[defects[i] * stride + defects[j]],
                )
            })
            .collect()
    }
}

impl Decoder for MwpmDecoder {
    /// Decode via the localized minimum-weight perfect matching (Q1-03).
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_local(syndrome).0
    }
}

/// Dijkstra from `src` over the matching graph, writing scaled distances and path observable
/// parities into `dist`/`parity` (length `stride`, indexed by node). The boundary node is settled
/// but never expanded, so detector→detector distances never pass through it.
fn dijkstra_from(
    graph: &MatchingGraph,
    src: usize,
    boundary: usize,
    edge_w: &[i64],
    edge_mask: &[u64],
    dist: &mut [i64],
    parity: &mut [u64],
) {
    for d in dist.iter_mut() {
        *d = INF;
    }
    for p in parity.iter_mut() {
        *p = 0;
    }
    dist[src] = 0;
    // Min-heap on (distance, node). `Reverse` turns the max-heap into a min-heap.
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0, src)));
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u] {
            continue; // stale entry
        }
        if u == boundary {
            continue; // settle the boundary but do not relax out of it
        }
        for &ei in graph.incident(u) {
            let e = &graph.edges()[ei];
            let v = if e.a == u { e.b } else { e.a };
            let nd = d + edge_w[ei];
            if nd < dist[v] {
                dist[v] = nd;
                parity[v] = parity[u] ^ edge_mask[ei];
                heap.push(Reverse((nd, v)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::{DemError, DetectorErrorModel};

    #[test]
    fn no_defects_no_correction() {
        let dem = DetectorErrorModel::parse("error(0.1) D0 D1 L0\n").unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();
        let s = Syndrome::new(2, vec![]);
        assert_eq!(dec.decode(&s), Correction::none(1));
    }

    #[test]
    fn single_defect_matches_boundary_and_applies_its_parity() {
        // Repetition-code-style DEM:
        //   D0 -- boundary           (no observable)
        //   D0 -- D1                 (no observable)
        //   D1 -- boundary, flips L0 (the only observable-carrying edge)
        // A lone defect at D1: cheapest explanation is the D1→boundary edge, which flips L0.
        let dem = DetectorErrorModel::parse("error(0.1) D0\nerror(0.1) D0 D1\nerror(0.1) D1 L0\n")
            .unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();

        let only_d1 = Syndrome::new(2, vec![1]);
        assert_eq!(dec.decode(&only_d1), Correction::new(vec![true]));

        // A lone defect at D0: cheapest is D0→boundary, no observable flip.
        let only_d0 = Syndrome::new(2, vec![0]);
        assert_eq!(dec.decode(&only_d0), Correction::new(vec![false]));
    }

    #[test]
    fn two_defects_match_each_other_via_bulk_edge() {
        // Both D0 and D1 fire: the single bulk edge D0–D1 (no observable) explains both at once,
        // cheaper than two separate boundary trips, so no observable flips.
        let dem = DetectorErrorModel::parse("error(0.1) D0\nerror(0.1) D0 D1\nerror(0.1) D1 L0\n")
            .unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();
        let both = Syndrome::new(2, vec![0, 1]);
        assert_eq!(dec.decode(&both), Correction::new(vec![false]));
    }

    #[test]
    fn prefers_cheaper_boundary_over_expensive_pairing() {
        // D0 and D1 each have a *cheap* boundary edge (high prob ⇒ low weight) but only an
        // *expensive* bulk edge between them (low prob ⇒ high weight). MWPM should send each to
        // the boundary independently. D1's boundary edge flips L0; D0's does not ⇒ net flip L0.
        let dem = DetectorErrorModel {
            detectors: 2,
            observables: 1,
            errors: vec![
                DemError::new(0.4, vec![0], vec![]),      // cheap D0→boundary
                DemError::new(0.4, vec![1], vec![0]),     // cheap D1→boundary, flips L0
                DemError::new(0.001, vec![0, 1], vec![]), // expensive D0–D1
            ],
        };
        let dec = MwpmDecoder::new(&dem).unwrap();
        let both = Syndrome::new(2, vec![0, 1]);
        assert_eq!(dec.decode(&both), Correction::new(vec![true]));
    }

    #[test]
    fn surface_code_d3_decodes_a_known_single_error() {
        // On a real d=3 memory DEM, injecting one error mechanism produces its detector support;
        // the decoder must recover that mechanism's observable flip (it is the unique cheapest
        // explanation for a single low-weight fault).
        use crate::{build_dem, SurfaceCode};
        let exp = SurfaceCode::new(3).memory_z_experiment(3);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();

        // Find an observable-flipping mechanism and feed exactly its detectors.
        let obs_mech = dem
            .errors
            .iter()
            .find(|e| !e.obs.is_empty() && !e.dets.is_empty())
            .expect("an observable-flipping edge exists");
        let s = Syndrome::new(dem.detectors, obs_mech.dets.clone());
        let corr = dec.decode(&s);
        assert!(
            corr.observable_flips[0],
            "decoder should recover the injected observable flip"
        );
    }

    #[test]
    fn integration_decoder_beats_null_at_low_noise() {
        // End-to-end through the Q0 harness (no external oracle): at low physical error the MWPM
        // decoder's logical error rate must be far below the do-nothing NullDecoder's.
        use crate::{build_dem, SurfaceCode};
        use crate::{run_dem_experiment, NullDecoder};
        let exp = SurfaceCode::new(3).memory_z_experiment(3);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).unwrap();

        let mwpm = MwpmDecoder::new(&dem).unwrap();
        let null = NullDecoder::new(dem.observables);
        let shots = 20_000;
        let r_mwpm = run_dem_experiment(&dem, shots, &mwpm, 1).unwrap();
        let r_null = run_dem_experiment(&dem, shots, &null, 1).unwrap();
        assert!(
            r_mwpm.rate < r_null.rate * 0.5,
            "MWPM rate {} should be well below NullDecoder rate {}",
            r_mwpm.rate,
            r_null.rate
        );
    }

    /// Total raw weight of a matching `mate` over the augmented `edges` (best edge per pair).
    fn matching_weight(n2: usize, edges: &[(usize, usize, i64)], mate: &[usize]) -> i64 {
        let mut w = vec![vec![i64::MIN; n2]; n2];
        for &(i, j, wt) in edges {
            w[i][j] = w[i][j].max(wt);
            w[j][i] = w[j][i].max(wt);
        }
        let mut total = 0;
        for v in 0..n2 {
            let u = mate[v];
            if u != usize::MAX && v < u {
                total += w[v][u];
            }
        }
        total
    }

    #[test]
    fn mwpm_weight_is_at_most_greedy() {
        // Property: the blossom matching is optimal, so its total weight ≤ any greedy matching's.
        // Greedy here: send every defect to the boundary independently (always a valid perfect
        // matching of the augmented graph — each defect to its clone, clones unused-pairs at 0).
        use crate::{build_dem, SurfaceCode};
        let exp = SurfaceCode::new(5).memory_z_experiment(5);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.04, 0.04)).unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();

        // Sample syndromes deterministically via the harness' sampler surrogate: just XOR random
        // mechanisms in. Use a simple LCG for reproducibility.
        let mut state = 0xC0FF_EE12_3456_789Au64;
        let mut bit = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f64 / (1u64 << 24) as f64
        };
        for _trial in 0..300 {
            let mut det = vec![false; dem.detectors];
            for e in &dem.errors {
                if bit() < e.prob {
                    for &d in &e.dets {
                        det[d as usize] ^= true;
                    }
                }
            }
            let defects: Vec<usize> = det
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| b.then_some(i))
                .collect();
            let n = defects.len();
            if n == 0 {
                continue;
            }
            let (edges, maxw) = dec.augmented_edges_dense(&defects);
            let transformed: Vec<(usize, usize, i64)> =
                edges.iter().map(|&(u, v, w)| (u, v, maxw - w)).collect();
            let mate = max_weight_matching(2 * n, &transformed, true);
            let mwpm_w = matching_weight(2 * n, &edges, &mate);

            // Greedy all-to-boundary matching weight.
            let mut greedy = vec![usize::MAX; 2 * n];
            for i in 0..n {
                greedy[i] = n + i;
                greedy[n + i] = i;
            }
            let greedy_w = matching_weight(2 * n, &edges, &greedy);
            assert!(
                mwpm_w <= greedy_w,
                "trial: MWPM weight {mwpm_w} exceeds greedy {greedy_w}"
            );
        }
    }

    /// Sample defect lists from a DEM via Bernoulli-per-mechanism (deterministic LCG).
    fn sample_defects(dem: &DetectorErrorModel, shots: usize, seed: u64) -> Vec<Vec<u32>> {
        let mut state = seed;
        let mut bit = |p: f64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f64 / (1u64 << 24) as f64) < p
        };
        (0..shots)
            .map(|_| {
                let mut det = vec![false; dem.detectors];
                for e in &dem.errors {
                    if bit(e.prob) {
                        for &d in &e.dets {
                            det[d as usize] ^= true;
                        }
                    }
                }
                det.iter()
                    .enumerate()
                    .filter_map(|(i, &b)| b.then_some(i as u32))
                    .collect()
            })
            .collect()
    }

    /// Localized matching reaches the *same minimum weight* as the dense matching on every shot
    /// (the rigorous optimality invariant), and the same correction except on genuine ties.
    #[test]
    fn local_matches_dense_weight_and_corrections() {
        use crate::{build_dem, SurfaceCode};
        for d in [5usize, 7, 9, 11] {
            let exp = SurfaceCode::new(d).memory_z_experiment(d);
            let dem =
                build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
            let dec = MwpmDecoder::new(&dem).unwrap();
            let shots = if d >= 11 { 400 } else { 3000 };
            let (mut nonempty, mut ties) = (0usize, 0usize);
            for fired in sample_defects(&dem, shots, 0xD00D ^ d as u64) {
                let s = Syndrome::new(dem.detectors, fired);
                if s.weight() > 0 {
                    nonempty += 1;
                }
                let (cl, wl) = dec.decode_local(&s);
                let (cd, wd) = dec.decode_with(&s, |s, d| s.augmented_edges_dense(d));
                // THE invariant: localized reaches the same minimum weight as dense on every shot.
                // This proves the locality prune/cap never dropped an edge the optimum needed.
                assert_eq!(wl, wd, "d={d}: localized weight {wl} != dense {wd}");
                // A correction difference can now only be a weight-tie (equal-weight matchings in
                // different homology classes), since the weights are provably equal above.
                if cl != cd {
                    ties += 1;
                }
            }
            // Sanity ceiling on the tie fraction (smaller codes tie more; d=5 ≈ 1.4%). Far below
            // this would be a bug; the weight equality above is the real correctness gate.
            let rate = ties as f64 / nonempty.max(1) as f64;
            assert!(
                rate < 0.05,
                "d={d}: {ties}/{nonempty} tie disagreements ({rate:.4}) exceeds sentinel"
            );
        }
    }

    #[test]
    #[ignore = "profiling only: run with --ignored --nocapture"]
    fn profile_local_phases_d11() {
        use crate::{build_dem, SurfaceCode};
        use std::time::Instant;
        let exp = SurfaceCode::new(11).memory_z_experiment(11);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();
        let shots = 512;
        let synds: Vec<Syndrome> = sample_defects(&dem, shots, 1)
            .into_iter()
            .map(|f| Syndrome::new(dem.detectors, f))
            .collect();

        let (mut t_build, mut t_blossom) = (0u128, 0u128);
        let (mut tot_edges, mut tot_n) = (0usize, 0usize);
        for s in &synds {
            let defects = dec.defects_of(s);
            let n = defects.len();
            if n == 0 {
                continue;
            }
            let b: Vec<i64> = defects
                .iter()
                .map(|&di| dec.dist[di * dec.stride + dec.boundary()])
                .collect();
            let t0 = Instant::now();
            let edges = dec.local_savings_edges(&defects, &b);
            t_build += t0.elapsed().as_nanos();
            tot_edges += edges.len();
            tot_n += n;
            let t1 = Instant::now();
            std::hint::black_box(max_weight_matching(n, &edges, false));
            t_blossom += t1.elapsed().as_nanos();
        }
        eprintln!(
            "d=11 over {shots} shots: avg n={}, avg edges={}, build={}us/shot, blossom={}us/shot",
            tot_n / shots,
            tot_edges / shots,
            t_build / 1000 / shots as u128,
            t_blossom / 1000 / shots as u128,
        );
    }

    #[test]
    fn p_zero_gives_zero_logical_errors() {
        use crate::run_dem_experiment;
        use crate::{build_dem, SurfaceCode};
        let exp = SurfaceCode::new(3).memory_z_experiment(2);
        // p=0 ⇒ no shot ever fires a detector ⇒ decoder always sees an empty syndrome.
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.0, 0.0)).unwrap();
        let dec = MwpmDecoder::new(&dem).unwrap();
        let res = run_dem_experiment(&dem, 5_000, &dec, 7).unwrap();
        assert_eq!(res.logical_errors, 0);
    }
}
