//! [`UnionFindDecoder`] — the almost-linear-time Union-Find / cluster-growth decoder of
//! Delfosse & Nickerson (arXiv:1709.06218), with Delfosse's peeling decoder (arXiv:1703.01517)
//! for the correction-extraction step.
//!
//! # The algorithm
//!
//! Two phases on the [`MatchingGraph`](crate::MatchingGraph) (detectors + one virtual boundary):
//!
//! 1. **Syndrome validation (cluster growth).** Seed one cluster per lit detector. A cluster is
//!    *neutral* when it holds an even number of defects **or** touches the boundary; otherwise it
//!    is *odd*. Every odd cluster grows by half an edge per round along its boundary edges; when an
//!    edge is grown from both ends it is *fully grown* and its endpoints' clusters fuse (Union-Find
//!    with union-by-size + path-halving). Repeat until no odd clusters remain. The fully-grown
//!    edges form an **erasure** that is guaranteed to support a correction for the syndrome.
//!
//! 2. **Peeling.** Build a spanning forest of the erasure (rooted at the boundary wherever a
//!    cluster touches it, so the boundary absorbs leftover parity), then peel leaves: a leaf whose
//!    vertex is still lit puts its pendant edge into the correction and toggles the parent's parity.
//!    The surviving edge set reproduces the syndrome exactly; XOR-ing those edges' observable masks
//!    gives the logical correction.
//!
//! Unlike MWPM this is unweighted (every edge grows at the same rate), so it is slightly less
//! accurate than [`MwpmDecoder`](crate::MwpmDecoder) but runs in near-linear time with integer-only
//! control flow over fixed-size arrays — the property that makes it the natural FPGA/ASIC target
//! (Q6/Q7). Weighted growth (closing part of the accuracy gap) is Q2-02.
//!
//! All per-decode scratch lives in a thread-local arena, reset by a generation counter so a decode
//! only touches the nodes/edges its clusters actually reach (not the whole graph) — decode is
//! `&self`, so the decoder is `Sync` and the Monte-Carlo harness can decode shots in parallel.

use std::cell::RefCell;

use crate::decoder::Decoder;
use crate::dem::DetectorErrorModel;
use crate::error::Result;
use crate::matching::MatchingGraph;
use crate::syndrome::{Correction, Syndrome};

/// Sentinel for "no parent" in the peeling forest (a tree root).
const NONE: u32 = u32::MAX;

/// A Union-Find (cluster-growth) decoder for a fixed [`DetectorErrorModel`].
///
/// Construct once from a DEM (or a prebuilt [`MatchingGraph`]); the graph is flattened into
/// CSR-style fixed arrays for a cache-friendly, hardware-shaped hot loop, then reused across shots.
#[derive(Clone, Debug)]
pub struct UnionFindDecoder {
    num_detectors: usize,
    num_observables: usize,
    /// Total nodes: detectors `0..num_detectors` plus the boundary at `num_detectors`.
    n_nodes: usize,
    /// CSR adjacency: edges incident to node `v` are `adj_edges[adj_off[v]..adj_off[v + 1]]`.
    adj_off: Vec<u32>,
    adj_edges: Vec<u32>,
    /// Endpoints of each edge (`edge_a[e] < edge_b[e]`; `b` may be the boundary).
    edge_a: Vec<u32>,
    edge_b: Vec<u32>,
    /// Observable-flip bitmask of each edge (bit `o` set ⇔ the edge flips observable `o`).
    edge_obs: Vec<u64>,
}

impl UnionFindDecoder {
    /// Build a decoder for `dem`.
    ///
    /// # Errors
    /// Propagates [`crate::Error::NonGraphlike`] if the DEM has a hyperedge (Union-Find, like
    /// MWPM, needs a graph-like DEM).
    pub fn new(dem: &DetectorErrorModel) -> Result<Self> {
        let graph = MatchingGraph::from_dem(dem)?;
        Ok(Self::from_graph(&graph))
    }

    /// Build a decoder directly from an already-constructed [`MatchingGraph`].
    pub fn from_graph(graph: &MatchingGraph) -> Self {
        let n_nodes = graph.num_nodes();
        let edges = graph.edges();

        let edge_a: Vec<u32> = edges.iter().map(|e| e.a as u32).collect();
        let edge_b: Vec<u32> = edges.iter().map(|e| e.b as u32).collect();
        let edge_obs: Vec<u64> = edges
            .iter()
            .map(|e| {
                e.observables
                    .iter()
                    .filter(|&&o| o < 64)
                    .fold(0u64, |m, &o| m | (1u64 << o))
            })
            .collect();

        // Flatten the adjacency lists into CSR. The index `v` is intrinsic here — `adj_off` is a
        // running prefix sum and `adj_edges` is scattered at per-node offsets — so the range loops
        // can't be plain slice iterations.
        #[allow(clippy::needless_range_loop)]
        let adj_off = {
            let mut off = vec![0u32; n_nodes + 1];
            for v in 0..n_nodes {
                off[v + 1] = off[v] + graph.incident(v).len() as u32;
            }
            off
        };
        let mut adj_edges = vec![0u32; adj_off[n_nodes] as usize];
        #[allow(clippy::needless_range_loop)]
        for v in 0..n_nodes {
            let start = adj_off[v] as usize;
            for (i, &e) in graph.incident(v).iter().enumerate() {
                adj_edges[start + i] = e as u32;
            }
        }

        UnionFindDecoder {
            num_detectors: graph.num_detectors(),
            num_observables: graph.num_observables(),
            n_nodes,
            adj_off,
            adj_edges,
            edge_a,
            edge_b,
            edge_obs,
        }
    }

    #[inline]
    fn boundary(&self) -> u32 {
        self.num_detectors as u32
    }

    /// Edges incident to node `v` (indices into the edge arrays).
    #[inline]
    fn incident(&self, v: u32) -> &[u32] {
        &self.adj_edges[self.adj_off[v as usize] as usize..self.adj_off[v as usize + 1] as usize]
    }

    /// The endpoint of edge `e` that is not `v`.
    #[inline]
    fn other(&self, e: u32, v: u32) -> u32 {
        let a = self.edge_a[e as usize];
        if a == v {
            self.edge_b[e as usize]
        } else {
            a
        }
    }

    /// Decode `syndrome`, returning the correction **and** the matched edge indices (the erasure
    /// edges the peeler put into the correction). The edge set is exposed for the syndrome-validity
    /// property test; [`decode`](Decoder::decode) just drops it.
    pub fn decode_edges(&self, syndrome: &Syndrome) -> (Correction, Vec<usize>) {
        let defects: Vec<u32> = syndrome
            .fired
            .iter()
            .copied()
            .filter(|&d| (d as usize) < self.num_detectors)
            .collect();

        if defects.is_empty() {
            return (Correction::none(self.num_observables), Vec::new());
        }

        SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            sc.begin(self.n_nodes, self.edge_a.len());
            self.grow_clusters(&mut sc, &defects);
            self.peel(&mut sc, &defects)
        })
    }

    /// Phase 1: grow odd clusters until all are neutral, accumulating the erasure.
    fn grow_clusters(&self, sc: &mut Scratch, defects: &[u32]) {
        let boundary = self.boundary();
        for &d in defects {
            sc.ensure(d, boundary);
            sc.parity[d as usize] = 1;
        }

        let mut odd: Vec<u32> = Vec::new();
        let mut to_fuse: Vec<u32> = Vec::new();
        let mut frontier: Vec<u32> = Vec::new();

        loop {
            sc.collect_odd_roots(defects, &mut odd, boundary);
            if odd.is_empty() {
                break;
            }
            to_fuse.clear();
            let mut grew = false;
            for &r in &odd {
                // Snapshot the cluster's vertices: growth calls `find` (which path-compresses the
                // scratch), so we can't hold a borrow of `verts[r]` across the inner loop.
                frontier.clear();
                frontier.extend_from_slice(&sc.verts[r as usize]);
                for &v in &frontier {
                    for &e in self.incident(v) {
                        let other = self.other(e, v);
                        // Only grow boundary edges (an endpoint outside this cluster). Internal
                        // edges (both endpoints already fused into `r`) are skipped.
                        if sc.find(other, boundary) == r {
                            continue;
                        }
                        let s = sc.support(e);
                        if s < 2 {
                            sc.support[e as usize] = s + 1;
                            grew = true;
                            if s + 1 == 2 {
                                to_fuse.push(e);
                                sc.erasure.push(e);
                            }
                        }
                    }
                }
            }
            for &e in &to_fuse {
                sc.union(self.edge_a[e as usize], self.edge_b[e as usize], boundary);
            }
            // Defensive: a connected component with odd parity and no boundary is unsatisfiable;
            // without this guard it would spin. (Cannot happen on a surface-code DEM, where every
            // detector reaches the boundary.)
            if !grew && to_fuse.is_empty() {
                break;
            }
        }
    }

    /// Phase 2: peel the erasure into a correction. Returns the correction and the chosen edges.
    fn peel(&self, sc: &mut Scratch, defects: &[u32]) -> (Correction, Vec<usize>) {
        let boundary = self.boundary();

        // Residual syndrome the peeler clears bottom-up: the lit detectors.
        for &d in defects {
            sc.syn[d as usize] = 1;
        }

        // Build a spanning forest over the erasure. Root every tree that reaches the boundary AT
        // the boundary, so leftover odd parity drains into it (the boundary is not a real detector,
        // so its residual is simply discarded). `order` is a BFS pre-order; peeling walks it in
        // reverse so children (leaves) are processed before their parents.
        sc.order.clear();
        // Root boundary-touching trees at the boundary so leftover parity drains into it. The
        // boundary is in the erasure iff one of its incident edges is fully grown.
        let boundary_in_erasure = self.incident(boundary).iter().any(|&e| sc.support(e) == 2);
        if boundary_in_erasure {
            self.bfs_tree(sc, boundary);
        }
        for &d in defects {
            // Each lit detector's component must be covered; BFS is a no-op if already visited.
            self.bfs_tree(sc, d);
        }

        let mut mask = 0u64;
        let mut chosen: Vec<usize> = Vec::new();
        for i in (0..sc.order.len()).rev() {
            let u = sc.order[i];
            let pe = sc.parent_edge[u as usize];
            if pe != NONE && sc.syn[u as usize] == 1 {
                mask ^= self.edge_obs[pe as usize];
                let p = sc.parent_node[u as usize];
                sc.syn[p as usize] ^= 1;
                sc.syn[u as usize] = 0;
                chosen.push(pe as usize);
            }
        }

        let flips = (0..self.num_observables)
            .map(|o| (mask >> o) & 1 == 1)
            .collect();
        (Correction::new(flips), chosen)
    }

    /// BFS a spanning tree of the erasure component containing `start`, appending newly-visited
    /// nodes to `sc.order` and recording each node's discovering (parent) edge. No-op if `start`
    /// is already visited or has no fully-grown incident edge.
    fn bfs_tree(&self, sc: &mut Scratch, start: u32) {
        if sc.is_visited(start) {
            return;
        }
        sc.mark_visited(start);
        sc.parent_edge[start as usize] = NONE;
        let head = sc.order.len();
        sc.order.push(start);
        let mut q = head;
        while q < sc.order.len() {
            let u = sc.order[q];
            q += 1;
            for idx in self.adj_off[u as usize]..self.adj_off[u as usize + 1] {
                let e = self.adj_edges[idx as usize];
                if sc.support(e) != 2 {
                    continue;
                }
                let w = self.other(e, u);
                if !sc.is_visited(w) {
                    sc.mark_visited(w);
                    sc.parent_edge[w as usize] = e;
                    sc.parent_node[w as usize] = u;
                    sc.order.push(w);
                }
            }
        }
    }
}

impl Decoder for UnionFindDecoder {
    /// Decode via cluster growth + peeling (Delfosse-Nickerson).
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_edges(syndrome).0
    }
}

thread_local! {
    /// Per-thread reusable scratch arena (reset by generation counter each decode).
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
}

/// Reusable per-decode working memory. Generation stamps (`gen`) make a fresh decode O(touched
/// nodes/edges) rather than O(graph): a node/edge is lazily initialised the first time this decode
/// reaches it, so clusters that never form cost nothing.
#[derive(Debug)]
struct Scratch {
    gen: u64,
    // --- Union-Find over nodes ---
    node_gen: Vec<u64>,
    parent: Vec<u32>,
    size: Vec<u32>,
    parity: Vec<u8>,
    boundary_touch: Vec<bool>,
    verts: Vec<Vec<u32>>,
    // --- edge growth ---
    edge_gen: Vec<u64>,
    support: Vec<u8>,
    erasure: Vec<u32>,
    // --- odd-root dedup within a growth round ---
    mark: Vec<u64>,
    mark_ctr: u64,
    // --- peeling ---
    visit_gen: Vec<u64>,
    parent_edge: Vec<u32>,
    parent_node: Vec<u32>,
    order: Vec<u32>,
    syn: Vec<u8>,
}

impl Scratch {
    const fn new() -> Self {
        Scratch {
            gen: 0,
            node_gen: Vec::new(),
            parent: Vec::new(),
            size: Vec::new(),
            parity: Vec::new(),
            boundary_touch: Vec::new(),
            verts: Vec::new(),
            edge_gen: Vec::new(),
            support: Vec::new(),
            erasure: Vec::new(),
            mark: Vec::new(),
            mark_ctr: 0,
            visit_gen: Vec::new(),
            parent_edge: Vec::new(),
            parent_node: Vec::new(),
            order: Vec::new(),
            syn: Vec::new(),
        }
    }

    /// Start a new decode: bump the generation and ensure the arrays are at least graph-sized.
    fn begin(&mut self, n_nodes: usize, n_edges: usize) {
        self.gen += 1;
        if self.node_gen.len() < n_nodes {
            self.node_gen.resize(n_nodes, 0);
            self.parent.resize(n_nodes, 0);
            self.size.resize(n_nodes, 0);
            self.parity.resize(n_nodes, 0);
            self.boundary_touch.resize(n_nodes, false);
            self.verts.resize_with(n_nodes, Vec::new);
            self.mark.resize(n_nodes, 0);
            self.visit_gen.resize(n_nodes, 0);
            self.parent_edge.resize(n_nodes, 0);
            self.parent_node.resize(n_nodes, 0);
            self.syn.resize(n_nodes, 0);
        }
        if self.edge_gen.len() < n_edges {
            self.edge_gen.resize(n_edges, 0);
            self.support.resize(n_edges, 0);
        }
        self.erasure.clear();
    }

    /// Lazily initialise node `v` as a fresh singleton cluster for this decode.
    #[inline]
    fn ensure(&mut self, v: u32, boundary: u32) {
        let i = v as usize;
        if self.node_gen[i] != self.gen {
            self.node_gen[i] = self.gen;
            self.parent[i] = v;
            self.size[i] = 1;
            self.parity[i] = 0;
            self.boundary_touch[i] = v == boundary;
            self.verts[i].clear();
            self.verts[i].push(v);
            self.syn[i] = 0;
        }
    }

    /// Find with path-halving; lazily initialises nodes it visits.
    #[inline]
    fn find(&mut self, v: u32, boundary: u32) -> u32 {
        self.ensure(v, boundary);
        let mut x = v;
        while self.parent[x as usize] != x {
            // Path halving: point x at its grandparent (both already ensured along the chain).
            let gp = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    /// Union the clusters of `a` and `b` (union by size). Combines parity, boundary-touch, and
    /// vertex lists into the surviving root.
    fn union(&mut self, a: u32, b: u32, boundary: u32) {
        let ra = self.find(a, boundary);
        let rb = self.find(b, boundary);
        if ra == rb {
            return;
        }
        // Attach the smaller tree under the larger root.
        let (big, small) = if self.size[ra as usize] >= self.size[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = big;
        self.size[big as usize] += self.size[small as usize];
        self.parity[big as usize] ^= self.parity[small as usize];
        self.boundary_touch[big as usize] |= self.boundary_touch[small as usize];
        let moved = std::mem::take(&mut self.verts[small as usize]);
        self.verts[big as usize].extend_from_slice(&moved);
    }

    /// Current support of edge `e` (0 ungrown, 1 half-grown, 2 fully grown), lazily initialised.
    #[inline]
    fn support(&mut self, e: u32) -> u8 {
        let i = e as usize;
        if self.edge_gen[i] != self.gen {
            self.edge_gen[i] = self.gen;
            self.support[i] = 0;
        }
        self.support[i]
    }

    /// Collect the distinct roots of the still-odd clusters (odd defect parity, boundary untouched).
    fn collect_odd_roots(&mut self, defects: &[u32], out: &mut Vec<u32>, boundary: u32) {
        out.clear();
        self.mark_ctr += 1;
        let ctr = self.mark_ctr;
        for &d in defects {
            let r = self.find(d, boundary);
            if self.parity[r as usize] == 1
                && !self.boundary_touch[r as usize]
                && self.mark[r as usize] != ctr
            {
                self.mark[r as usize] = ctr;
                out.push(r);
            }
        }
    }

    #[inline]
    fn is_visited(&self, v: u32) -> bool {
        self.visit_gen[v as usize] == self.gen
    }

    #[inline]
    fn mark_visited(&mut self, v: u32) {
        self.visit_gen[v as usize] = self.gen;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_dem;
    use crate::matching::MatchingGraph;
    use crate::surface::SurfaceCode;

    /// The surface-code memory-Z matching graph + UF decoder at distance `d`, phys error `p`.
    fn setup(d: usize, p: f64) -> (MatchingGraph, UnionFindDecoder) {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        let graph = MatchingGraph::from_dem(&dem).unwrap();
        let dec = UnionFindDecoder::from_graph(&graph);
        (graph, dec)
    }

    /// Detector-flip bit-vector produced by a set of edges (the boundary node is dropped — it is
    /// not a detector). This is the "boundary" of the edge set: what syndrome it would light up.
    fn detector_flips(graph: &MatchingGraph, edges: &[usize], num_detectors: usize) -> Vec<bool> {
        let mut bits = vec![false; num_detectors];
        for &e in edges {
            let ed = &graph.edges()[e];
            for endpoint in [ed.a, ed.b] {
                if endpoint < num_detectors {
                    bits[endpoint] ^= true;
                }
            }
        }
        bits
    }

    #[test]
    fn empty_syndrome_decodes_to_no_flip() {
        let (_g, dec) = setup(5, 0.05);
        let s = Syndrome::new(dec.num_detectors, vec![]);
        let (c, edges) = dec.decode_edges(&s);
        assert_eq!(c, Correction::none(dec.num_observables));
        assert!(edges.is_empty());
    }

    #[test]
    fn correction_reproduces_a_single_edge_syndrome() {
        // A single interior edge lights its two detectors; UF must return a correction whose
        // detector boundary is exactly those two detectors.
        let (graph, dec) = setup(5, 0.05);
        let e = graph
            .edges()
            .iter()
            .position(|ed| ed.b < dec.num_detectors) // a non-boundary edge
            .expect("an interior edge");
        let truth = detector_flips(&graph, &[e], dec.num_detectors);
        let s = Syndrome::from_bits(&truth);
        let (_c, chosen) = dec.decode_edges(&s);
        let got = detector_flips(&graph, &chosen, dec.num_detectors);
        assert_eq!(got, truth, "decoded edges must reproduce the syndrome");
    }

    /// Every decoded correction must reproduce the input syndrome (Q2-01 acceptance + property
    /// test). We draw the input syndrome from a *realisable* error (a random subset of edges) so it
    /// is guaranteed to have a consistent correction, then check the decoder finds one with the
    /// same detector boundary.
    #[test]
    fn corrections_are_syndrome_consistent() {
        for &d in &[3usize, 5, 7] {
            let (graph, dec) = setup(d, 0.10);
            let m = graph.edges().len();
            // Deterministic SplitMix64 stream — many random error patterns of varying weight.
            let mut z: u64 = 0xC0FFEE ^ (d as u64);
            let mut next = || {
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                z
            };
            for trial in 0..2000 {
                // Error density swept across trials so we cover sparse and dense syndromes.
                let q = 0.02 + 0.20 * (trial as f64 / 2000.0);
                let error: Vec<usize> = (0..m)
                    .filter(|_| ((next() >> 11) as f64 / (1u64 << 53) as f64) < q)
                    .collect();
                let truth = detector_flips(&graph, &error, dec.num_detectors);
                let s = Syndrome::from_bits(&truth);
                let (_c, chosen) = dec.decode_edges(&s);
                let got = detector_flips(&graph, &chosen, dec.num_detectors);
                assert_eq!(
                    got, truth,
                    "d={d} trial={trial}: decoded edges must reproduce the input syndrome"
                );
            }
        }
    }

    /// Arbitrary (not necessarily error-derived) random syndromes must also decode to something
    /// syndrome-consistent: in the surface code every detector reaches the boundary, so any
    /// detector pattern is realisable and the peeler must reproduce it exactly.
    #[test]
    fn arbitrary_syndromes_are_consistent() {
        let (graph, dec) = setup(5, 0.05);
        let nd = dec.num_detectors;
        let mut z: u64 = 0x1234_5678;
        let mut next = || {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            z
        };
        for _ in 0..1000 {
            let bits: Vec<bool> = (0..nd)
                .map(|_| next() & 1 == 0 && next() & 3 == 0)
                .collect();
            let s = Syndrome::from_bits(&bits);
            let (_c, chosen) = dec.decode_edges(&s);
            let got = detector_flips(&graph, &chosen, nd);
            assert_eq!(
                got, bits,
                "decoded edges must reproduce an arbitrary syndrome"
            );
        }
    }
}
