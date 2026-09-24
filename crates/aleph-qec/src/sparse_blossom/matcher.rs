//! The matcher: Edmonds' alternating-tree operations on regions, driven by flooder events, and
//! the final resolution of top-level matches into defect–defect edges.

use super::flooder::MEvent;
use super::graph::CompiledGraph;
use super::state::*;

impl State {
    pub(crate) fn handle(&mut self, g: &CompiledGraph, ev: MEvent) {
        self.stats.events += 1;
        match ev {
            MEvent::HitRegion { a, b, e } => {
                let ta = self.regions[a as usize].tree;
                let tb = self.regions[b as usize].tree;
                debug_assert!(
                    ta != NONE && self.trees[ta as usize].outer == a,
                    "a must be outer"
                );
                if tb == NONE {
                    if self.regions[b as usize].matched_to == BOUNDARY {
                        self.regions[b as usize].matched_to = NONE;
                        self.augment(g, a, b, e, false);
                    } else {
                        self.grow(g, a, b, e);
                    }
                } else if self.root_of(a) == self.root_of(b) {
                    // `tree` is a per-tree-node id (set per node in `grow`, per root in `run`),
                    // so two distinct outer regions of the same tree always carry different
                    // `tree` values — "same tree" has to compare roots, not tree-node ids. This
                    // is also the branch a degenerate implosion (`shrink_step`'s
                    // `HitRegion { a: parent outer, b: child outer }`) takes: the parent and
                    // child outer are always in the same tree.
                    self.blossom(g, a, b, e);
                } else {
                    debug_assert_eq!(self.trees[tb as usize].outer, b);
                    self.augment(g, a, b, e, true);
                }
            }
            MEvent::HitBoundary { a, e } => {
                self.regions[a as usize].matched_to = BOUNDARY;
                self.regions[a as usize].match_edge = e;
                self.flip_path_to_root(a);
                let root = self.root_of(a);
                self.dissolve_tree(g, root);
            }
            MEvent::Shatter { b } => self.shatter(g, b),
        }
    }

    pub(crate) fn set_match(&mut self, x: RegionId, y: RegionId, e: CEdge) {
        self.regions[x as usize].matched_to = y;
        self.regions[x as usize].match_edge = e;
        self.regions[y as usize].matched_to = x;
        self.regions[y as usize].match_edge = e.reversed();
    }

    fn root_of(&self, r: RegionId) -> TreeId {
        let mut n = self.regions[r as usize].tree;
        while self.trees[n as usize].parent != NONE {
            n = self.trees[n as usize].parent;
        }
        n
    }

    /// Re-pair the regions on the path from `x`'s tree node up to the root.
    fn flip_path_to_root(&mut self, x: RegionId) {
        let mut n = self.regions[x as usize].tree;
        while self.trees[n as usize].parent != NONE {
            let p = self.trees[n as usize].parent;
            let inner = self.trees[n as usize].inner;
            let pe = self.trees[n as usize].parent_edge;
            let po = self.trees[p as usize].outer;
            self.set_match(inner, po, pe.reversed());
            n = p;
        }
    }

    /// Freeze every region of the tree rooted at `root`, matching untouched pairs as they stand.
    fn dissolve_tree(&mut self, g: &CompiledGraph, root: TreeId) {
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            let (inner, outer, io) = {
                let t = &mut self.trees[n as usize];
                t.alive = false;
                stack.extend_from_slice(&t.children);
                (t.inner, t.outer, t.inner_to_outer)
            };
            if inner != NONE {
                if self.regions[inner as usize].matched_to == NONE {
                    self.set_match(inner, outer, io);
                }
                self.regions[inner as usize].tree = NONE;
                self.set_slope(g, inner, 0);
            }
            self.regions[outer as usize].tree = NONE;
            self.set_slope(g, outer, 0);
        }
        self.active_trees -= 1;
    }

    /// `a` (outer) meets `b` (outer of another tree, or a boundary-matched free region).
    fn augment(&mut self, g: &CompiledGraph, a: RegionId, b: RegionId, e: CEdge, b_in_tree: bool) {
        self.set_match(a, b, e);
        self.flip_path_to_root(a);
        let ra = self.root_of(a);
        if b_in_tree {
            let rb = self.root_of(b);
            debug_assert_ne!(ra, rb, "augment: a and b must be in different trees");
            self.flip_path_to_root(b);
            self.dissolve_tree(g, rb);
        }
        self.dissolve_tree(g, ra);
    }

    /// `a` (outer) meets the frozen matched region `m`: `(m inner, m' outer)` joins `a`'s tree.
    fn grow(&mut self, g: &CompiledGraph, a: RegionId, m: RegionId, e: CEdge) {
        let m2 = self.regions[m as usize].matched_to;
        let io = self.regions[m as usize].match_edge;
        debug_assert!(m2 != NONE && m2 != BOUNDARY);
        self.regions[m as usize].matched_to = NONE;
        self.regions[m2 as usize].matched_to = NONE;
        let parent = self.regions[a as usize].tree;
        let n = self.new_tree_node(m, m2, io, parent, e);
        self.regions[m as usize].tree = n;
        self.regions[m2 as usize].tree = n;
        self.set_slope(g, m2, 1);
        self.set_slope(g, m, -1);
    }

    pub(crate) fn new_tree_root(&mut self, outer: RegionId) -> TreeId {
        self.trees.push(TreeNode {
            inner: NONE,
            outer,
            inner_to_outer: CEdge::EMPTY,
            parent: NONE,
            parent_edge: CEdge::EMPTY,
            children: Vec::new(),
            alive: true,
        });
        self.active_trees += 1;
        (self.trees.len() - 1) as TreeId
    }

    pub(crate) fn new_tree_node(
        &mut self,
        inner: RegionId,
        outer: RegionId,
        inner_to_outer: CEdge,
        parent: TreeId,
        parent_edge: CEdge,
    ) -> TreeId {
        self.trees.push(TreeNode {
            inner,
            outer,
            inner_to_outer,
            parent,
            parent_edge,
            children: Vec::new(),
            alive: true,
        });
        let id = (self.trees.len() - 1) as TreeId;
        self.trees[parent as usize].children.push(id);
        id
    }

    fn blossom(&mut self, _g: &CompiledGraph, _a: RegionId, _b: RegionId, _e: CEdge) {
        unimplemented!("Task 4")
    }

    fn shatter(&mut self, _g: &CompiledGraph, _b: RegionId) {
        unimplemented!("Task 4")
    }

    /// Turn top-level matches into `(obs, doubled weight)`; blossoms are resolved recursively.
    pub(crate) fn resolve(&self) -> (u64, i64) {
        let (mut obs, mut w) = (0u64, 0i64);
        for r in 0..self.regions.len() as RegionId {
            let reg = &self.regions[r as usize];
            if reg.blossom_parent != NONE || reg.matched_to == NONE {
                continue;
            }
            let e = reg.match_edge;
            if reg.matched_to == BOUNDARY {
                obs ^= e.obs;
                w += e.weight;
                self.descend(r, e.from, &mut obs, &mut w);
            } else if r < reg.matched_to {
                obs ^= e.obs;
                w += e.weight;
                self.descend(r, e.from, &mut obs, &mut w);
                self.descend(reg.matched_to, e.to, &mut obs, &mut w);
            }
        }
        (obs, w)
    }

    /// Inside blossom `r`, `defect` is the endpoint of the outside match; pair up the rest.
    fn descend(&self, r: RegionId, defect: NodeId, obs: &mut u64, w: &mut i64) {
        let reg = &self.regions[r as usize];
        if !reg.is_blossom() {
            return;
        }
        let k = reg.children.len();
        let idx = self.child_index_containing(r, defect);
        let mut i = (idx + 1) % k;
        while i != idx {
            let (x, e) = reg.children[i];
            let (y, _) = reg.children[(i + 1) % k];
            *obs ^= e.obs;
            *w += e.weight;
            self.descend(x, e.from, obs, w);
            self.descend(y, e.to, obs, w);
            i = (i + 2) % k;
        }
        self.descend(reg.children[idx].0, defect, obs, w);
    }

    /// Index of the child of blossom `b` that contains `defect`.
    pub(crate) fn child_index_containing(&self, b: RegionId, defect: NodeId) -> usize {
        let mut r = self.nodes[defect as usize].region;
        while self.regions[r as usize].blossom_parent != b {
            r = self.regions[r as usize].blossom_parent;
            debug_assert_ne!(r, NONE, "defect is not inside the blossom");
        }
        self.regions[b as usize]
            .children
            .iter()
            .position(|&(c, _)| c == r)
            .unwrap_or(0)
    }
}
