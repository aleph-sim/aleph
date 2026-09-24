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

    /// `a` and `b` are outer regions of the same tree: contract the odd cycle they close.
    fn blossom(&mut self, g: &CompiledGraph, a: RegionId, b: RegionId, e: CEdge) {
        let na = self.regions[a as usize].tree;
        let nb = self.regions[b as usize].tree;
        // Paths to the root; the LCA is the first common node.
        let up = |s: &Self, mut n: TreeId| {
            let mut p = vec![n];
            while s.trees[n as usize].parent != NONE {
                n = s.trees[n as usize].parent;
                p.push(n);
            }
            p
        };
        let pa = up(self, na);
        let pb = up(self, nb);
        let lca = *pa
            .iter()
            .find(|n| pb.contains(n))
            .unwrap_or(&pa[pa.len() - 1]);
        let path_a: Vec<TreeId> = pa.iter().copied().take_while(|&n| n != lca).collect();
        let path_b: Vec<TreeId> = pb.iter().copied().take_while(|&n| n != lca).collect();

        // Cycle in order a, inner(a), outer(parent), …, outer(lca), inner(b side), …, b.
        let mut cycle: Vec<(RegionId, CEdge)> =
            Vec::with_capacity(2 * (path_a.len() + path_b.len()) + 1);
        for &n in &path_a {
            let t = &self.trees[n as usize];
            cycle.push((t.outer, t.inner_to_outer.reversed()));
            cycle.push((t.inner, t.parent_edge.reversed()));
        }
        let lca_outer = self.trees[lca as usize].outer;
        let first_down = path_b.last().copied();
        // From the LCA's outer region down the b side, or straight back to `a` if `b` is that region.
        cycle.push((
            lca_outer,
            first_down.map_or(e.reversed(), |n| self.trees[n as usize].parent_edge),
        ));
        for (i, &n) in path_b.iter().enumerate().rev() {
            let t = &self.trees[n as usize];
            cycle.push((t.inner, t.inner_to_outer));
            let next = if i == 0 {
                e.reversed()
            } else {
                self.trees[path_b[i - 1] as usize].parent_edge
            };
            cycle.push((t.outer, next));
        }
        debug_assert_eq!(cycle.len() % 2, 1);
        debug_assert_eq!(cycle[0].0, if path_a.is_empty() { lca_outer } else { a });

        // The blossom region.
        let bid = self.regions.len() as RegionId;
        self.regions.push(Region {
            radius: Varying {
                y0: -self.now,
                slope: 1,
            },
            blossom_parent: NONE,
            tree: lca,
            matched_to: NONE,
            match_edge: CEdge::EMPTY,
            children: cycle.clone(),
            shell: Vec::new(),
            source: NONE,
            shrink_queued: NO_TIME,
        });
        for &(c, _) in &cycle {
            let rc = self.regions[c as usize].radius.at(self.now);
            self.regions[c as usize].radius = Varying::frozen(rc);
            self.regions[c as usize].blossom_parent = bid;
            self.regions[c as usize].tree = NONE;
            self.shift_wrapped(c, rc);
        }

        // Tree surgery: the LCA node's outer becomes the blossom; the path nodes die and their
        // other children re-hang on the LCA node.
        let mut moved: Vec<TreeId> = Vec::new();
        for &n in path_a.iter().chain(path_b.iter()) {
            let t = &mut self.trees[n as usize];
            t.alive = false;
            moved.extend(
                t.children
                    .drain(..)
                    .filter(|c| !path_a.contains(c) && !path_b.contains(c)),
            );
        }
        let top_a = path_a.last().copied();
        let top_b = path_b.last().copied();
        let lca_node = &mut self.trees[lca as usize];
        lca_node.outer = bid;
        lca_node
            .children
            .retain(|&c| Some(c) != top_a && Some(c) != top_b);
        lca_node.children.extend_from_slice(&moved);
        for c in moved {
            self.trees[c as usize].parent = lca;
        }
        self.regions[bid as usize].tree = lca;
        self.reschedule_region(g, bid);
    }

    /// Inner blossom `b` reached radius 0: replace it in the tree by the alternating path of
    /// its children between the parent edge's child and the outer edge's child; match the rest.
    fn shatter(&mut self, g: &CompiledGraph, b: RegionId) {
        let n = self.regions[b as usize].tree;
        debug_assert_eq!(self.trees[n as usize].inner, b);
        debug_assert!(self.regions[b as usize].shell.is_empty());
        let (p, parent_edge, io, outer_o) = {
            let t = &self.trees[n as usize];
            (t.parent, t.parent_edge, t.inner_to_outer, t.outer)
        };
        let children = std::mem::take(&mut self.regions[b as usize].children);
        let k = children.len();
        let i_par = self.child_index_of(&children, parent_edge.to);
        let i_out = self.child_index_of(&children, io.from);
        for &(c, _) in &children {
            let rc = self.regions[c as usize].radius.y0;
            self.regions[c as usize].blossom_parent = NONE;
            self.regions[c as usize].tree = NONE;
            self.shift_wrapped(c, -rc);
        }
        // Direction with an even number of edges from i_par to i_out.
        let fwd = (i_out + k - i_par) % k;
        let forward = fwd.is_multiple_of(2);
        let step = |i: usize| {
            if forward {
                (i + 1) % k
            } else {
                (i + k - 1) % k
            }
        };
        let edge = |i: usize, j: usize| -> CEdge {
            // Edge from child i to child j, adjacent in the chosen direction.
            if forward {
                children[i].1
            } else {
                children[j].1.reversed()
            }
        };
        // Path i_par → … → i_out (odd number of children).
        let mut path = vec![i_par];
        while *path.last().unwrap_or(&i_out) != i_out {
            let last = *path.last().unwrap_or(&i_out);
            path.push(step(last));
        }
        // The rest, continuing past i_out until just before i_par (even number of children).
        let mut rest = Vec::new();
        let mut i = step(i_out);
        while i != i_par {
            rest.push(i);
            i = step(i);
        }
        debug_assert_eq!(path.len() % 2, 1);
        debug_assert_eq!(rest.len() % 2, 0);

        // Rebuild the tree chain in place of `n`.
        self.trees[n as usize].alive = false;
        self.trees[p as usize].children.retain(|&c| c != n);
        let old_children = std::mem::take(&mut self.trees[n as usize].children);
        let mut parent = p;
        let mut pe = parent_edge;
        let mut j = 0;
        while j + 1 < path.len() {
            let (inner, outer) = (children[path[j]].0, children[path[j + 1]].0);
            let node = self.new_tree_node(inner, outer, edge(path[j], path[j + 1]), parent, pe);
            self.regions[inner as usize].tree = node;
            self.regions[outer as usize].tree = node;
            self.set_slope(g, outer, 1);
            self.set_slope(g, inner, -1);
            parent = node;
            pe = edge(path[j + 1], path[j + 2]);
            j += 2;
        }
        let c_out = children[path[path.len() - 1]].0;
        let last = self.new_tree_node(c_out, outer_o, io, parent, pe);
        self.regions[c_out as usize].tree = last;
        self.regions[outer_o as usize].tree = last;
        self.trees[last as usize].children = old_children;
        for &c in &self.trees[last as usize].children.clone() {
            self.trees[c as usize].parent = last;
        }
        self.set_slope(g, c_out, -1);
        // Matched pairs along the rest of the cycle: frozen, re-derive their events.
        let mut i = 0;
        while i + 1 < rest.len() {
            let (x, y) = (children[rest[i]].0, children[rest[i + 1]].0);
            self.set_match(x, y, edge(rest[i], rest[i + 1]));
            self.reschedule_region(g, x);
            self.reschedule_region(g, y);
            i += 2;
        }
        // `b` itself is now dead: its children were reparented above and it owns nothing (no
        // shell, no tree slot). Reset it explicitly rather than leaving stale radius/tree state
        // behind, so a debug print or a future reuse of this slot sees an inert region.
        self.regions[b as usize].radius = Varying::frozen(0);
        self.regions[b as usize].tree = NONE;
    }

    fn child_index_of(&self, children: &[(RegionId, CEdge)], defect: NodeId) -> usize {
        let mut r = self.nodes[defect as usize].region;
        loop {
            if let Some(i) = children.iter().position(|&(c, _)| c == r) {
                return i;
            }
            r = self.regions[r as usize].blossom_parent;
            debug_assert_ne!(r, NONE, "defect not inside the shattering blossom");
        }
    }

    /// After the event queue drains, any tree whose root never found an augmenting path (an odd
    /// component with no boundary) is dissolved like a completed one, except the root's own
    /// outer region is deliberately left unmatched — the one best-effort exposed vertex.
    pub(crate) fn dissolve_leftover_trees(&mut self, g: &CompiledGraph) {
        for n in 0..self.trees.len() as TreeId {
            if self.trees[n as usize].alive && self.trees[n as usize].parent == NONE {
                self.dissolve_tree(g, n);
            }
        }
    }

    /// A top-level region left unmatched by `dissolve_leftover_trees`: a bare leaf contributes
    /// nothing, but an exposed blossom's odd cycle still needs pairing down to one exposed child
    /// (picked arbitrarily as `children[0]`, recursing the same way for a nested blossom there).
    fn resolve_exposed(&self, r: RegionId, obs: &mut u64, w: &mut i64) {
        let reg = &self.regions[r as usize];
        if !reg.is_blossom() {
            return;
        }
        self.resolve_exposed(reg.children[0].0, obs, w);
        let k = reg.children.len();
        let mut i = 1;
        while i + 1 < k {
            let (x, e) = reg.children[i];
            let (y, _) = reg.children[i + 1];
            *obs ^= e.obs;
            *w += e.weight;
            self.descend(x, e.from, obs, w);
            self.descend(y, e.to, obs, w);
            i += 2;
        }
    }

    /// Turn top-level matches into `(obs, doubled weight)`; blossoms are resolved recursively.
    pub(crate) fn resolve(&self) -> (u64, i64) {
        let (mut obs, mut w) = (0u64, 0i64);
        for r in 0..self.regions.len() as RegionId {
            let reg = &self.regions[r as usize];
            if reg.blossom_parent != NONE {
                continue;
            }
            if reg.matched_to == NONE {
                self.resolve_exposed(r, &mut obs, &mut w);
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
