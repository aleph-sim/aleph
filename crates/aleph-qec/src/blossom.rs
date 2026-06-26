//! Maximum-weight matching in a general (non-bipartite) graph via Edmonds' blossom algorithm,
//! primal-dual form, O(V³).
//!
//! This is the engine the MWPM decoder ([`crate::mwpm`]) runs to find a minimum-weight *perfect*
//! matching of syndrome defects: negate the weights and demand maximum cardinality, and the
//! maximum-weight matching of a complete graph is the minimum-weight perfect matching.
//!
//! Re-implemented from the algorithm of Edmonds (1965, "Paths, trees, and flowers") in the
//! efficient primal-dual form of Galil (1986, "Efficient algorithms for finding maximum matching
//! in graphs", ACM Computing Surveys). Concepts only — no code was copied from any
//! implementation. Weights are integers so the dual-variable slacks are compared exactly (the
//! decoder scales its real weights to integers before calling in; CLAUDE.md / ADR 0006 forbids
//! float comparisons gating correctness).
//!
//! # Method, in brief
//!
//! Each vertex `v` holds a dual `dualvar[v]`; each (odd) blossom `b` holds a dual `dualvar[b]`.
//! An edge's *slack* is `dualvar[i] + dualvar[j] − 2·w(i,j) ≥ 0`; an edge is *tight* (usable)
//! when its slack is 0. The algorithm grows alternating trees from unmatched vertices over tight
//! edges, contracting odd cycles into blossoms, until an augmenting path is found (improving the
//! matching) or no tight edge exists — at which point the duals are adjusted by the largest step
//! that keeps every slack ≥ 0, exposing a new tight edge. It terminates with a matching that is
//! optimal by LP duality.

// The algorithm is irreducibly index-heavy (endpoints, blossom children, parent pointers); the
// clearest faithful transcription uses indexed loops and explicit sentinels rather than
// iterator combinators, so we opt out of the lints that would fight that style here.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;

/// Sentinel for "no vertex / no endpoint / no edge" — the algorithm's `-1`.
const NONE: i64 = -1;

thread_local! {
    /// One reusable solver per thread. Matching is called once per decoded shot, so reusing the
    /// solver's buffers (rather than allocating ~`O(n)` vectors per call in [`Blossom::load`])
    /// removes the dominant per-decode allocation cost. Thread-local keeps it `Sync`-free and
    /// correct under the parallel decode paths.
    static SOLVER: RefCell<Blossom> = RefCell::new(Blossom::empty());
}

/// Compute a maximum-weight matching of the graph on `n` vertices (`0..n`) with the given
/// weighted `edges` (`(i, j, weight)`, `i != j`).
///
/// Returns `mate`, where `mate[v]` is the vertex matched to `v`, or [`usize::MAX`] if `v` is
/// unmatched. The matching is symmetric (`mate[mate[v]] == v`).
///
/// If `maxcardinality` is `true`, the matching has the maximum possible number of edges, and
/// among those is of maximum weight. The decoder uses this so that — on a complete graph, where a
/// perfect matching always exists — the result is a maximum-weight *perfect* matching.
pub fn max_weight_matching(
    n: usize,
    edges: &[(usize, usize, i64)],
    maxcardinality: bool,
) -> Vec<usize> {
    SOLVER.with(|cell| {
        let mut state = cell.borrow_mut();
        state.load(n, edges, maxcardinality);
        state.run();
        state.extract_mates()
    })
}

/// Internal solver state. All the `-1`-bearing arrays are `i64`; pure counts/lengths are `usize`.
struct Blossom {
    n: usize,
    edges: Vec<(usize, usize, i64)>,
    maxcardinality: bool,

    /// `endpoint[p]` = the vertex at endpoint `p`; the edge of endpoint `p` is `p / 2`.
    endpoint: Vec<i64>,
    /// `neighbend[v]` = the endpoints *pointing away* from `v` along each incident edge (so
    /// `endpoint[neighbend[v][k]]` is the k-th neighbour of `v`).
    neighbend: Vec<Vec<i64>>,

    /// `mate[v]` = the endpoint of `v`'s matched edge that points at `v`'s partner, or `NONE`.
    mate: Vec<i64>,

    /// Top-level (S=1, T=2, free=0) label of each vertex/blossom. Bit 2 (`& 4`) is a scratch mark
    /// used by [`scan_blossom`].
    label: Vec<i64>,
    /// The endpoint through which a labelled vertex/blossom was reached.
    labelend: Vec<i64>,
    /// `inblossom[v]` = the outermost blossom currently containing vertex `v`.
    inblossom: Vec<i64>,
    /// `blossomparent[b]` = the blossom immediately containing `b`, or `NONE` if top-level.
    blossomparent: Vec<i64>,
    /// Sub-blossoms of each non-trivial blossom, in cyclic order.
    blossomchilds: Vec<Vec<i64>>,
    /// `blossombase[b]` = the (single) exposed/base vertex of blossom `b`, or `NONE` if recycled.
    blossombase: Vec<i64>,
    /// Endpoints of the edges connecting consecutive sub-blossoms.
    blossomendps: Vec<Vec<i64>>,
    /// Best (least-slack) edge from a top-level vertex/blossom to an S-blossom, or `NONE`.
    bestedge: Vec<i64>,
    /// For a non-trivial blossom, candidate least-slack edges to each other S-blossom.
    blossombestedges: Vec<Option<Vec<i64>>>,
    /// Stack of blossom ids in `n..2n` not currently in use.
    unusedblossoms: Vec<i64>,
    /// Dual variable of each vertex (`< n`) and blossom (`>= n`).
    dualvar: Vec<i64>,
    /// `allowedge[k]` = edge `k` has slack 0 (tight, usable), even before its endpoints are
    /// formally labelled.
    allowedge: Vec<bool>,
    /// Work list of S-vertices whose incident edges still need scanning.
    queue: Vec<i64>,
}

/// Resize `v` to at least `len` (growing with empty inners) and clear the first `len` inner
/// vectors so their capacity is reused. Entries beyond `len` are stale but never indexed.
fn reset_nested(v: &mut Vec<Vec<i64>>, len: usize) {
    if v.len() < len {
        v.resize_with(len, Vec::new);
    }
    for inner in v.iter_mut().take(len) {
        inner.clear();
    }
}

impl Blossom {
    /// An empty solver; call [`load`](Self::load) before [`run`](Self::run).
    fn empty() -> Self {
        Blossom {
            n: 0,
            edges: Vec::new(),
            maxcardinality: false,
            endpoint: Vec::new(),
            neighbend: Vec::new(),
            mate: Vec::new(),
            label: Vec::new(),
            labelend: Vec::new(),
            inblossom: Vec::new(),
            blossomparent: Vec::new(),
            blossomchilds: Vec::new(),
            blossombase: Vec::new(),
            blossomendps: Vec::new(),
            bestedge: Vec::new(),
            blossombestedges: Vec::new(),
            unusedblossoms: Vec::new(),
            dualvar: Vec::new(),
            allowedge: Vec::new(),
            queue: Vec::new(),
        }
    }

    /// (Re)initialise the solver for a new problem, reusing all buffer capacity from the previous
    /// call. Equivalent to a fresh solver but without the per-call allocation.
    fn load(&mut self, n: usize, edges: &[(usize, usize, i64)], maxcardinality: bool) {
        let m = edges.len();
        let nb = 2 * n; // vertices 0..n are trivial blossoms; n..2n are slots for real blossoms
        self.n = n;
        self.maxcardinality = maxcardinality;

        self.edges.clear();
        self.edges.extend_from_slice(edges);

        self.endpoint.clear();
        self.endpoint.resize(2 * m, 0);
        reset_nested(&mut self.neighbend, n);
        let mut maxweight = 0i64;
        for (k, &(i, j, w)) in edges.iter().enumerate() {
            self.endpoint[2 * k] = i as i64;
            self.endpoint[2 * k + 1] = j as i64;
            self.neighbend[i].push((2 * k + 1) as i64);
            self.neighbend[j].push((2 * k) as i64);
            maxweight = maxweight.max(w);
        }

        self.mate.clear();
        self.mate.resize(n, NONE);

        self.label.clear();
        self.label.resize(nb, 0);
        self.labelend.clear();
        self.labelend.resize(nb, NONE);
        self.inblossom.clear();
        self.inblossom.extend(0..n as i64);
        self.blossomparent.clear();
        self.blossomparent.resize(nb, NONE);
        reset_nested(&mut self.blossomchilds, nb);
        self.blossombase.clear();
        self.blossombase.extend(0..n as i64);
        self.blossombase.resize(nb, NONE);
        reset_nested(&mut self.blossomendps, nb);
        self.bestedge.clear();
        self.bestedge.resize(nb, NONE);
        if self.blossombestedges.len() < nb {
            self.blossombestedges.resize_with(nb, || None);
        }
        for be in self.blossombestedges.iter_mut().take(nb) {
            *be = None;
        }
        self.unusedblossoms.clear();
        self.unusedblossoms.extend(n as i64..nb as i64);
        self.dualvar.clear();
        self.dualvar.resize(nb, 0);
        for v in 0..n {
            self.dualvar[v] = maxweight;
        }
        self.allowedge.clear();
        self.allowedge.resize(m, false);
        self.queue.clear();
    }

    /// Slack of edge `k`: `dualvar[i] + dualvar[j] − 2·w`. Always `≥ 0` at a consistent state.
    fn slack(&self, k: usize) -> i64 {
        let (i, j, w) = self.edges[k];
        self.dualvar[i] + self.dualvar[j] - 2 * w
    }

    /// Leaf vertices of blossom `b` (just `b` itself if `b` is a trivial vertex blossom).
    fn blossom_leaves(&self, b: i64) -> Vec<i64> {
        let mut out = Vec::new();
        let mut stack = vec![b];
        while let Some(x) = stack.pop() {
            if (x as usize) < self.n {
                out.push(x);
            } else {
                for &c in &self.blossomchilds[x as usize] {
                    stack.push(c);
                }
            }
        }
        out
    }

    /// Label the top-level blossom of `w` with type `t` (1=S, 2=T), reached via endpoint `p`.
    fn assign_label(&mut self, w: i64, t: i64, p: i64) {
        let b = self.inblossom[w as usize];
        self.label[w as usize] = t;
        self.label[b as usize] = t;
        self.labelend[w as usize] = p;
        self.labelend[b as usize] = p;
        self.bestedge[w as usize] = NONE;
        self.bestedge[b as usize] = NONE;
        if t == 1 {
            // S-blossom: queue all its vertices for scanning. Trivial vertex-blossoms (the common
            // case, e.g. every exposed vertex at the start of a stage) are their own only leaf, so
            // queue directly and skip the `blossom_leaves` allocation.
            if (b as usize) < self.n {
                self.queue.push(b);
            } else {
                let leaves = self.blossom_leaves(b);
                self.queue.extend(leaves);
            }
        } else if t == 2 {
            // T-blossom: its base is matched; label the partner S.
            let base = self.blossombase[b as usize];
            let mate_end = self.mate[base as usize];
            self.assign_label(self.endpoint[mate_end as usize], 1, mate_end ^ 1);
        }
    }

    /// Walk back the alternating trees from the two S-endpoints of a tight edge `(v, w)`. Returns
    /// the base of the blossom to form if they share one, else `NONE` (an augmenting path exists).
    fn scan_blossom(&mut self, mut v: i64, mut w: i64) -> i64 {
        let mut path: Vec<i64> = Vec::new();
        let mut base = NONE;
        while v != NONE || w != NONE {
            let mut b = self.inblossom[v as usize];
            if self.label[b as usize] & 4 != 0 {
                base = self.blossombase[b as usize];
                break;
            }
            path.push(b);
            self.label[b as usize] |= 4;
            if self.labelend[b as usize] == NONE {
                v = NONE;
            } else {
                v = self.endpoint[self.labelend[b as usize] as usize];
                b = self.inblossom[v as usize];
                // b is a T-blossom; step through it to the next S.
                v = self.endpoint[self.labelend[b as usize] as usize];
            }
            if w != NONE {
                std::mem::swap(&mut v, &mut w);
            }
        }
        for b in path {
            self.label[b as usize] &= !4;
        }
        base
    }

    /// Contract the odd cycle through tight edge `k = (v, w)` that closes on `base` into a new
    /// blossom labelled S.
    fn add_blossom(&mut self, base: i64, k: usize) {
        let (vv, ww, _w) = self.edges[k];
        let (mut v, mut w) = (vv as i64, ww as i64);
        let bb = self.inblossom[base as usize];
        let mut bv = self.inblossom[v as usize];
        let mut bw = self.inblossom[w as usize];

        let b = self.unusedblossoms.pop().expect("a blossom slot is free");
        self.blossombase[b as usize] = base;
        self.blossomparent[b as usize] = NONE;
        self.blossomparent[bb as usize] = b;

        let mut path: Vec<i64> = Vec::new();
        let mut endps: Vec<i64> = Vec::new();

        // Trace v's branch up to the common base.
        while bv != bb {
            self.blossomparent[bv as usize] = b;
            path.push(bv);
            endps.push(self.labelend[bv as usize]);
            v = self.endpoint[self.labelend[bv as usize] as usize];
            bv = self.inblossom[v as usize];
        }
        path.push(bb);
        path.reverse();
        endps.reverse();
        endps.push((2 * k) as i64);

        // Trace w's branch up to the common base.
        while bw != bb {
            self.blossomparent[bw as usize] = b;
            path.push(bw);
            endps.push(self.labelend[bw as usize] ^ 1);
            w = self.endpoint[self.labelend[bw as usize] as usize];
            bw = self.inblossom[w as usize];
        }

        self.blossomchilds[b as usize] = path;
        self.blossomendps[b as usize] = endps;

        // The new blossom is an S-blossom, reached the way its base was.
        self.label[b as usize] = 1;
        self.labelend[b as usize] = self.labelend[bb as usize];
        self.dualvar[b as usize] = 0;

        // Relabel every vertex inside as belonging to b; queue former T-vertices (now S).
        let leaves = self.blossom_leaves(b);
        for leaf in leaves {
            if self.label[self.inblossom[leaf as usize] as usize] == 2 {
                self.queue.push(leaf);
            }
            self.inblossom[leaf as usize] = b;
        }

        // Compute the new blossom's best edges to other S-blossoms.
        let mut bestedgeto = vec![NONE; 2 * self.n];
        let childs = self.blossomchilds[b as usize].clone();
        for bv in childs {
            let nblists: Vec<Vec<i64>> = match &self.blossombestedges[bv as usize] {
                Some(list) => vec![list.clone()],
                None => self
                    .blossom_leaves(bv)
                    .into_iter()
                    .map(|leaf| {
                        self.neighbend[leaf as usize]
                            .iter()
                            .map(|&p| p / 2)
                            .collect()
                    })
                    .collect(),
            };
            for nblist in nblists {
                for kk in nblist {
                    let kk = kk as usize;
                    let (mut i, mut j, _) = self.edges[kk];
                    if self.inblossom[j] == b {
                        std::mem::swap(&mut i, &mut j);
                    }
                    let bj = self.inblossom[j];
                    if bj != b
                        && self.label[bj as usize] == 1
                        && (bestedgeto[bj as usize] == NONE
                            || self.slack(kk) < self.slack(bestedgeto[bj as usize] as usize))
                    {
                        bestedgeto[bj as usize] = kk as i64;
                    }
                }
            }
            self.blossombestedges[bv as usize] = None;
            self.bestedge[bv as usize] = NONE;
        }
        let best: Vec<i64> = bestedgeto.into_iter().filter(|&kk| kk != NONE).collect();
        self.bestedge[b as usize] = NONE;
        for &kk in &best {
            if self.bestedge[b as usize] == NONE
                || self.slack(kk as usize) < self.slack(self.bestedge[b as usize] as usize)
            {
                self.bestedge[b as usize] = kk;
            }
        }
        self.blossombestedges[b as usize] = Some(best);
    }

    /// Expand blossom `b` back into its sub-blossoms. `endstage` selects the simpler end-of-stage
    /// behaviour (no relabelling) vs the in-stage T-blossom expansion that relabels children.
    fn expand_blossom(&mut self, b: i64, endstage: bool) {
        let childs = self.blossomchilds[b as usize].clone();
        for &s in &childs {
            self.blossomparent[s as usize] = NONE;
            if (s as usize) < self.n {
                self.inblossom[s as usize] = s;
            } else if endstage && self.dualvar[s as usize] == 0 {
                self.expand_blossom(s, endstage);
            } else {
                for leaf in self.blossom_leaves(s) {
                    self.inblossom[leaf as usize] = s;
                }
            }
        }

        if !endstage && self.label[b as usize] == 2 {
            // In-stage T-blossom: relabel its children along the alternating path. `jj` is a
            // signed cursor over the cyclic child list (made negative for an odd start so that
            // `+jstep` walks it toward the base at index 0); array accesses use `rem_euclid`.
            let entry_vertex = self.endpoint[(self.labelend[b as usize] ^ 1) as usize];
            let entrychild = self.inblossom[entry_vertex as usize];
            let nchild = childs.len() as i64;
            let i = childs.iter().position(|&c| c == entrychild).unwrap() as i64;
            let (jstep, endptrick) = if i & 1 != 0 {
                (1i64, 0i64)
            } else {
                (-1i64, 1i64)
            };
            let mut jj = if i & 1 != 0 { i - nchild } else { i };
            let endps = self.blossomendps[b as usize].clone();
            let mut p = self.labelend[b as usize];
            while jj != 0 {
                self.label[self.endpoint[(p ^ 1) as usize] as usize] = 0;
                let idx = (jj - endptrick).rem_euclid(nchild) as usize;
                self.label[self.endpoint[(endps[idx] ^ endptrick ^ 1) as usize] as usize] = 0;
                let ep = self.endpoint[(p ^ 1) as usize];
                self.assign_label(ep, 2, p);
                self.allowedge[(endps[idx] / 2) as usize] = true;
                jj += jstep;
                let idx2 = (jj - endptrick).rem_euclid(nchild) as usize;
                p = endps[idx2] ^ endptrick;
                self.allowedge[(p / 2) as usize] = true;
                jj += jstep;
            }
            // Base sub-blossom (cursor back at 0) inherits the T-label.
            let bv = childs[jj.rem_euclid(nchild) as usize];
            self.label[self.endpoint[(p ^ 1) as usize] as usize] = 2;
            self.label[bv as usize] = 2;
            self.labelend[self.endpoint[(p ^ 1) as usize] as usize] = p;
            self.labelend[bv as usize] = p;
            self.bestedge[bv as usize] = NONE;
            jj += jstep;
            // The remaining sub-blossoms become free; re-examine any that still hold a label.
            while childs[jj.rem_euclid(nchild) as usize] != entrychild {
                let bv = childs[jj.rem_euclid(nchild) as usize];
                if self.label[bv as usize] == 1 {
                    jj += jstep;
                    continue;
                }
                let mut found = NONE;
                for v in self.blossom_leaves(bv) {
                    if self.label[v as usize] != 0 {
                        found = v;
                        break;
                    }
                }
                if found != NONE {
                    self.label[found as usize] = 0;
                    let base = self.blossombase[bv as usize];
                    self.label[self.endpoint[self.mate[base as usize] as usize] as usize] = 0;
                    self.assign_label(found, 2, self.labelend[found as usize]);
                }
                jj += jstep;
            }
        }

        // Recycle the slot.
        self.label[b as usize] = NONE;
        self.labelend[b as usize] = NONE;
        self.blossomchilds[b as usize].clear();
        self.blossomendps[b as usize].clear();
        self.blossombase[b as usize] = NONE;
        self.blossombestedges[b as usize] = None;
        self.bestedge[b as usize] = NONE;
        self.unusedblossoms.push(b);
    }

    /// Swap matched/unmatched edges along blossom `b` so that vertex `v` becomes its base.
    fn augment_blossom(&mut self, b: i64, v: i64) {
        // Descend to the sub-blossom of b that contains v.
        let mut t = v;
        while self.blossomparent[t as usize] != b {
            t = self.blossomparent[t as usize];
        }
        if (t as usize) >= self.n {
            self.augment_blossom(t, v);
        }
        let childs = self.blossomchilds[b as usize].clone();
        let endps = self.blossomendps[b as usize].clone();
        let nchild = childs.len() as i64;
        let i = childs.iter().position(|&c| c == t).unwrap() as i64;
        let (jstep, endptrick) = if i & 1 != 0 {
            (1i64, 0i64)
        } else {
            (-1i64, 1i64)
        };
        // Negative start for an odd index so `+jstep` walks the cyclic list toward index 0.
        let mut jj = if i & 1 != 0 { i - nchild } else { i };
        while jj != 0 {
            jj += jstep;
            let t = childs[jj.rem_euclid(nchild) as usize];
            let p = endps[(jj - endptrick).rem_euclid(nchild) as usize] ^ endptrick;
            if (t as usize) >= self.n {
                self.augment_blossom(t, self.endpoint[p as usize]);
            }
            jj += jstep;
            let t2 = childs[jj.rem_euclid(nchild) as usize];
            if (t2 as usize) >= self.n {
                self.augment_blossom(t2, self.endpoint[(p ^ 1) as usize]);
            }
            self.mate[self.endpoint[p as usize] as usize] = p ^ 1;
            self.mate[self.endpoint[(p ^ 1) as usize] as usize] = p;
        }
        // Rotate so that the chosen sub-blossom is first; b's base becomes v.
        let rot = i.rem_euclid(nchild) as usize;
        self.blossomchilds[b as usize].rotate_left(rot);
        self.blossomendps[b as usize].rotate_left(rot);
        self.blossombase[b as usize] = self.blossombase[self.blossomchilds[b as usize][0] as usize];
    }

    /// Augment the matching along the path exposed by tight edge `k = (v, w)`.
    fn augment_matching(&mut self, k: usize) {
        let (vv, ww, _w) = self.edges[k];
        for &(s_init, p_init) in &[(vv as i64, (2 * k + 1) as i64), (ww as i64, (2 * k) as i64)] {
            let mut s = s_init;
            let mut p = p_init;
            loop {
                let bs = self.inblossom[s as usize];
                if (bs as usize) >= self.n {
                    self.augment_blossom(bs, s);
                }
                self.mate[s as usize] = p;
                if self.labelend[bs as usize] == NONE {
                    break;
                }
                let t = self.endpoint[self.labelend[bs as usize] as usize];
                let bt = self.inblossom[t as usize];
                s = self.endpoint[self.labelend[bt as usize] as usize];
                let j = self.endpoint[(self.labelend[bt as usize] ^ 1) as usize];
                if (bt as usize) >= self.n {
                    self.augment_blossom(bt, j);
                }
                self.mate[j as usize] = self.labelend[bt as usize];
                p = self.labelend[bt as usize] ^ 1;
            }
        }
    }

    /// Run stages until no further augmenting path exists.
    fn run(&mut self) {
        let nb = 2 * self.n;
        for _ in 0..self.n {
            // Reset all labels and per-stage scratch.
            for b in 0..nb {
                self.label[b] = 0;
                self.bestedge[b] = NONE;
            }
            for b in self.n..nb {
                self.blossombestedges[b] = None;
            }
            for a in self.allowedge.iter_mut() {
                *a = false;
            }
            self.queue.clear();

            // Label every exposed vertex S to root an alternating tree.
            for v in 0..self.n {
                if self.mate[v] == NONE && self.label[self.inblossom[v] as usize] == 0 {
                    self.assign_label(v as i64, 1, NONE);
                }
            }

            let mut augmented = false;
            loop {
                // Scan S-vertices' edges for tight edges that grow the tree or augment.
                while !self.queue.is_empty() && !augmented {
                    let v = self.queue.pop().unwrap();
                    // `neighbend` is immutable after construction, so index it directly rather
                    // than cloning the (potentially large) neighbour list on every scan.
                    let deg = self.neighbend[v as usize].len();
                    for idx in 0..deg {
                        let p = self.neighbend[v as usize][idx];
                        let k = (p / 2) as usize;
                        let w = self.endpoint[p as usize];
                        if self.inblossom[v as usize] == self.inblossom[w as usize] {
                            continue; // internal edge
                        }
                        let kslack = self.slack(k);
                        if !self.allowedge[k] && kslack <= 0 {
                            self.allowedge[k] = true;
                        }
                        if self.allowedge[k] {
                            if self.label[self.inblossom[w as usize] as usize] == 0 {
                                self.assign_label(w, 2, p ^ 1);
                            } else if self.label[self.inblossom[w as usize] as usize] == 1 {
                                let base = self.scan_blossom(v, w);
                                if base != NONE {
                                    self.add_blossom(base, k);
                                } else {
                                    self.augment_matching(k);
                                    augmented = true;
                                    break;
                                }
                            } else if self.label[w as usize] == 0 {
                                // w is inside a T-blossom but itself unlabelled.
                                self.label[w as usize] = 2;
                                self.labelend[w as usize] = p ^ 1;
                            }
                        } else if self.label[self.inblossom[w as usize] as usize] == 1 {
                            let b = self.inblossom[v as usize];
                            if self.bestedge[b as usize] == NONE
                                || kslack < self.slack(self.bestedge[b as usize] as usize)
                            {
                                self.bestedge[b as usize] = k as i64;
                            }
                        } else if self.label[w as usize] == 0
                            && (self.bestedge[w as usize] == NONE
                                || kslack < self.slack(self.bestedge[w as usize] as usize))
                        {
                            self.bestedge[w as usize] = k as i64;
                        }
                    }
                }
                if augmented {
                    break;
                }

                // No tight edge usable; adjust duals by the largest feasible step.
                let mut deltatype = -1i32;
                let mut delta = 0i64;
                let mut deltaedge = NONE;
                let mut deltablossom = NONE;

                if !self.maxcardinality {
                    deltatype = 1;
                    delta = (0..self.n).map(|v| self.dualvar[v]).min().unwrap_or(0);
                }
                // delta2: least slack of an edge from an S-blossom to a free vertex.
                for v in 0..self.n {
                    if self.label[self.inblossom[v] as usize] == 0 && self.bestedge[v] != NONE {
                        let d = self.slack(self.bestedge[v] as usize);
                        if deltatype == -1 || d < delta {
                            delta = d;
                            deltatype = 2;
                            deltaedge = self.bestedge[v];
                        }
                    }
                }
                // delta3: half the least slack of an S–S edge.
                for b in 0..nb {
                    if self.blossomparent[b] == NONE
                        && self.label[b] == 1
                        && self.bestedge[b] != NONE
                    {
                        let d = self.slack(self.bestedge[b] as usize) / 2;
                        if deltatype == -1 || d < delta {
                            delta = d;
                            deltatype = 3;
                            deltaedge = self.bestedge[b];
                        }
                    }
                }
                // delta4: least dual of a top-level T-blossom.
                for b in self.n..nb {
                    if self.blossombase[b] != NONE
                        && self.blossomparent[b] == NONE
                        && self.label[b] == 2
                        && (deltatype == -1 || self.dualvar[b] < delta)
                    {
                        delta = self.dualvar[b];
                        deltatype = 4;
                        deltablossom = b as i64;
                    }
                }
                if deltatype == -1 {
                    // Only reachable with maxcardinality: no positive step exists; stop the stage.
                    deltatype = 1;
                    delta = (0..self.n)
                        .map(|v| self.dualvar[v])
                        .min()
                        .unwrap_or(0)
                        .max(0);
                }

                // Apply the dual adjustment.
                for v in 0..self.n {
                    match self.label[self.inblossom[v] as usize] {
                        1 => self.dualvar[v] -= delta,
                        2 => self.dualvar[v] += delta,
                        _ => {}
                    }
                }
                for b in self.n..nb {
                    if self.blossombase[b] != NONE && self.blossomparent[b] == NONE {
                        match self.label[b] {
                            1 => self.dualvar[b] += delta,
                            2 => self.dualvar[b] -= delta,
                            _ => {}
                        }
                    }
                }

                // Take the action that the chosen delta unlocked.
                match deltatype {
                    1 => break, // no further progress possible this stage
                    2 => {
                        let (i, j, _) = self.edges[deltaedge as usize];
                        self.allowedge[deltaedge as usize] = true;
                        let s = if self.label[self.inblossom[i] as usize] == 1 {
                            i
                        } else {
                            j
                        };
                        self.queue.push(s as i64);
                    }
                    3 => {
                        let (i, _j, _) = self.edges[deltaedge as usize];
                        self.allowedge[deltaedge as usize] = true;
                        self.queue.push(i as i64);
                    }
                    4 => self.expand_blossom(deltablossom, false),
                    _ => unreachable!(),
                }
            }

            if !augmented {
                break;
            }

            // End of stage: expand top-level S-blossoms whose dual has reached 0.
            for b in self.n..nb {
                if self.blossomparent[b] == NONE
                    && self.blossombase[b] != NONE
                    && self.label[b] == 1
                    && self.dualvar[b] == 0
                {
                    self.expand_blossom(b as i64, true);
                }
            }
        }
    }

    /// Convert the internal endpoint-based `mate` into vertex pairs.
    fn extract_mates(&self) -> Vec<usize> {
        (0..self.n)
            .map(|v| {
                if self.mate[v] == NONE {
                    usize::MAX
                } else {
                    self.endpoint[self.mate[v] as usize] as usize
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force minimum/maximum-weight matching by subset DP — the trusted oracle for small
    /// graphs. Returns the maximum total weight over *perfect* matchings (or `None` if `n` is odd
    /// / no perfect matching), considering only the given edges.
    fn brute_max_perfect(n: usize, edges: &[(usize, usize, i64)]) -> Option<i64> {
        if !n.is_multiple_of(2) {
            return None;
        }
        // w[i][j] = best single-edge weight between i and j (or i64::MIN if no edge).
        let mut w = vec![vec![i64::MIN; n]; n];
        for &(i, j, wt) in edges {
            w[i][j] = w[i][j].max(wt);
            w[j][i] = w[j][i].max(wt);
        }
        let full = 1usize << n;
        let mut dp = vec![i64::MIN; full];
        dp[0] = 0;
        for mask in 0..full {
            if dp[mask] == i64::MIN {
                continue;
            }
            // Lowest unmatched vertex.
            let i = (0..n).find(|&i| mask & (1 << i) == 0);
            let i = match i {
                Some(i) => i,
                None => continue,
            };
            for j in (i + 1)..n {
                if mask & (1 << j) == 0 && w[i][j] != i64::MIN {
                    let nm = mask | (1 << i) | (1 << j);
                    dp[nm] = dp[nm].max(dp[mask] + w[i][j]);
                }
            }
        }
        let r = dp[full - 1];
        if r == i64::MIN {
            None
        } else {
            Some(r)
        }
    }

    /// Total weight of the matching `mate` over the best edge available for each matched pair.
    fn matching_weight(n: usize, edges: &[(usize, usize, i64)], mate: &[usize]) -> i64 {
        let mut w = vec![vec![i64::MIN; n]; n];
        for &(i, j, wt) in edges {
            w[i][j] = w[i][j].max(wt);
            w[j][i] = w[j][i].max(wt);
        }
        let mut total = 0;
        for v in 0..n {
            let u = mate[v];
            if u != usize::MAX && v < u {
                total += w[v][u];
            }
        }
        total
    }

    #[test]
    fn empty_graph() {
        assert_eq!(max_weight_matching(0, &[], true), Vec::<usize>::new());
    }

    #[test]
    fn single_edge() {
        let mate = max_weight_matching(2, &[(0, 1, 5)], true);
        assert_eq!(mate, vec![1, 0]);
    }

    #[test]
    fn picks_heavier_of_two_disjoint() {
        // Triangle-free: 0-1 weight 1, 2-3 weight 1, 0-2 weight 10 -> max matching takes 0-2 and
        // leaves 1,3 (non-maxcardinality) ... with maxcardinality it must match all four.
        let edges = [(0, 1, 1), (2, 3, 1), (0, 2, 10)];
        let non_card = max_weight_matching(4, &edges, false);
        assert_eq!(matching_weight(4, &edges, &non_card), 10);
        let card = max_weight_matching(4, &edges, true);
        // Perfect matching must be {0-1, 2-3} (weight 2); {0-2} can't be completed.
        assert_eq!(matching_weight(4, &edges, &card), 2);
    }

    #[test]
    fn blossom_odd_cycle() {
        // A 5-cycle with a pendant forces an odd-cycle (blossom) contraction. Weighted so the
        // optimum perfect matching is non-trivial.
        let edges = [
            (0, 1, 8),
            (1, 2, 8),
            (2, 3, 8),
            (3, 4, 8),
            (4, 0, 8),
            (2, 5, 10),
        ];
        let mate = max_weight_matching(6, &edges, true);
        // Compare to brute force.
        let opt = brute_max_perfect(6, &edges).unwrap();
        assert_eq!(matching_weight(6, &edges, &mate), opt);
        // Every vertex matched (perfect).
        assert!(mate.iter().all(|&m| m != usize::MAX));
    }

    /// Brute-force maximum-weight (non-perfect) matching by subset DP: each vertex may be left
    /// unmatched. The oracle for `maxcardinality = false`.
    fn brute_max_nonperfect(n: usize, edges: &[(usize, usize, i64)]) -> i64 {
        let mut w = vec![vec![i64::MIN; n]; n];
        for &(i, j, wt) in edges {
            w[i][j] = w[i][j].max(wt);
            w[j][i] = w[j][i].max(wt);
        }
        let full = 1usize << n;
        let mut dp = vec![i64::MIN; full];
        dp[0] = 0;
        for mask in 0..full {
            if dp[mask] == i64::MIN {
                continue;
            }
            let i = match (0..n).find(|&i| mask & (1 << i) == 0) {
                Some(i) => i,
                None => continue,
            };
            // Leave i unmatched.
            let m1 = mask | (1 << i);
            dp[m1] = dp[m1].max(dp[mask]);
            // Or match i to some free j with a positive-or-any edge.
            for j in (i + 1)..n {
                if mask & (1 << j) == 0 && w[i][j] != i64::MIN {
                    let nm = mask | (1 << i) | (1 << j);
                    dp[nm] = dp[nm].max(dp[mask] + w[i][j]);
                }
            }
        }
        dp[full - 1]
    }

    #[test]
    fn nonperfect_matches_brute_force() {
        // Validate maxcardinality = false (the mode the savings-formulation decoder relies on)
        // against brute force, including negative edges (which must be left out).
        let mut state = 0x0BAD_F00D_1234_5678u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for n in [2usize, 4, 5, 7, 9, 11] {
            for _ in 0..200 {
                let mut edges = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        // Mix of positive and negative weights; skip ~1/4 of edges (sparse).
                        if next() % 4 != 0 {
                            edges.push((i, j, next() % 200 - 100));
                        }
                    }
                }
                let mate = max_weight_matching(n, &edges, false);
                for v in 0..n {
                    if mate[v] != usize::MAX {
                        assert_eq!(mate[mate[v]], v, "n={n}: asymmetric mate");
                    }
                }
                let got = matching_weight(n, &edges, &mate);
                let opt = brute_max_nonperfect(n, &edges);
                assert_eq!(got, opt, "n={n}: non-perfect weight {got} != optimum {opt}");
            }
        }
    }

    #[test]
    fn matches_brute_force_on_random_complete_graphs() {
        // Deterministic LCG so the test is reproducible without an RNG dependency.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for n in [2usize, 4, 6, 8, 10] {
            for _trial in 0..200 {
                let mut edges = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        let w = next() % 100; // 0..99
                        edges.push((i, j, w));
                    }
                }
                let mate = max_weight_matching(n, &edges, true);
                // Perfect (complete graph, even n).
                assert!(
                    mate.iter().all(|&m| m != usize::MAX),
                    "n={n}: not a perfect matching"
                );
                // Symmetric.
                for v in 0..n {
                    assert_eq!(mate[mate[v]], v, "n={n}: asymmetric mate");
                }
                let got = matching_weight(n, &edges, &mate);
                let opt = brute_max_perfect(n, &edges).unwrap();
                assert_eq!(got, opt, "n={n}: weight {got} != optimum {opt}");
            }
        }
    }

    #[test]
    fn min_weight_perfect_via_negation() {
        // Minimum-weight perfect matching = max-weight perfect matching of negated weights.
        let mut state = 0xdead_beef_cafe_babeu64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for n in [4usize, 6, 8] {
            for _ in 0..100 {
                let mut edges = Vec::new();
                let mut neg = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        let w = next() % 100;
                        edges.push((i, j, w));
                        neg.push((i, j, -w));
                    }
                }
                let mate = max_weight_matching(n, &neg, true);
                let got = matching_weight(n, &edges, &mate); // real weight of chosen matching
                                                             // Brute-force minimum over perfect matchings = -max(-w).
                let min_opt = -brute_max_perfect(n, &neg).unwrap();
                assert_eq!(got, min_opt, "n={n}: min-weight {got} != optimum {min_opt}");
            }
        }
    }
}
