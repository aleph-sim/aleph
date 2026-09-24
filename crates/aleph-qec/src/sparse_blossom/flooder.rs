//! The flooder: grows outer regions, shrinks inner ones, and turns geometry into matcher events.
//!
//! Every owned node `u` carries `total(u,t) = wrapped(u) + top(u).radius(t)`; `u` is covered
//! while `total ≥ dist(u)`. A node's *look-at* event is the earliest of, over its neighbours `v`:
//! arrival into an unowned `v` (own top growing), collision with `v`'s top region (combined
//! slope greater than zero), and the boundary (own top growing). Scheduling is lazy: a heap entry
//! is pushed only if earlier than what is queued, and a popped entry whose time is not the queued
//! time is stale. Shrinking regions release their own shell in reverse arrival order and, at
//! radius 0, shatter (blossoms) or implode (leaf regions: the parent and child outer regions meet
//! through it).

use super::graph::CompiledGraph;
use super::state::*;

/// What the matcher has to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MEvent {
    /// `a` is growing (outer); `e` runs from `a`'s defect to `b`'s defect.
    HitRegion { a: RegionId, b: RegionId, e: CEdge },
    /// `a` is growing; `e.to == BOUNDARY`.
    HitBoundary { a: RegionId, e: CEdge },
    /// Inner blossom `b` shrank to radius 0.
    Shatter { b: RegionId },
}

#[derive(Clone, Copy)]
enum Next {
    Arrive(usize),
    Collide(usize),
    Boundary,
}

impl State {
    /// Top-level region containing `r`.
    #[inline]
    pub(crate) fn top(&self, mut r: RegionId) -> RegionId {
        loop {
            let p = self.regions[r as usize].blossom_parent;
            if p == NONE {
                return r;
            }
            r = p;
        }
    }

    /// `total(u, ·)` as an affine function of time, for an owned node.
    #[inline]
    fn total(&self, u: NodeId) -> (Varying, RegionId) {
        let n = &self.nodes[u as usize];
        let top = self.top(n.region);
        let r = self.regions[top as usize].radius;
        (
            Varying {
                y0: r.y0 + n.wrapped,
                slope: r.slope,
            },
            top,
        )
    }

    /// Earliest event at `u` and what it is; `None` if nothing will ever happen at `u`.
    fn next_at(&self, g: &CompiledGraph, u: NodeId) -> Option<(i64, Next)> {
        let n = self.nodes[u as usize];
        if n.region == NONE {
            return None;
        }
        let (tu, top_u) = self.total(u);
        let mut best: Option<(i64, Next)> = None;
        let mut consider = |t: i64, what: Next| {
            if best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, what));
            }
        };
        for (k, e) in g.edges(u).iter().enumerate() {
            let m = self.nodes[e.v as usize];
            if m.region == NONE {
                if tu.slope > 0 {
                    consider(n.dist + e.w - tu.y0, Next::Arrive(k));
                }
                continue;
            }
            let (tv, top_v) = self.total(e.v);
            if top_v == top_u {
                continue;
            }
            let cs = tu.slope + tv.slope;
            if cs <= 0 {
                continue;
            }
            let num = n.dist + e.w + m.dist - tu.y0 - tv.y0;
            let t = if cs == 2 {
                debug_assert_eq!(num & 1, 0, "collision at a half-integer time");
                num / 2
            } else {
                num
            };
            consider(t, Next::Collide(k));
        }
        if tu.slope > 0 {
            if let Some((wb, _)) = g.boundary(u) {
                consider(n.dist + wb - tu.y0, Next::Boundary);
            }
        }
        best.map(|(t, w)| (t.max(self.now), w))
    }

    /// Queue `u`'s look-at if it is earlier than what is queued.
    pub(crate) fn schedule_node(&mut self, g: &CompiledGraph, u: NodeId) {
        if let Some((t, _)) = self.next_at(g, u) {
            let q = self.nodes[u as usize].queued;
            if q == NO_TIME || t < q {
                self.nodes[u as usize].queued = t;
                self.push_event(t, EV_NODE, u);
            }
        }
    }

    /// Pop-time handler for a node event: execute what is due now (at most one matcher event),
    /// then reschedule.
    pub(crate) fn look_at_node(&mut self, g: &CompiledGraph, u: NodeId) -> Option<MEvent> {
        let (t, what) = self.next_at(g, u)?;
        if t > self.now {
            self.nodes[u as usize].queued = t;
            self.push_event(t, EV_NODE, u);
            return None;
        }
        let n = self.nodes[u as usize];
        let ev = match what {
            Next::Arrive(k) => {
                let e = g.edges(u)[k];
                self.arrive(g, u, e.v, e.w, e.obs);
                None
            }
            Next::Collide(k) => {
                let e = g.edges(u)[k];
                let m = self.nodes[e.v as usize];
                let (tu, top_u) = self.total(u);
                let top_v = self.top(m.region);
                let ce = CEdge {
                    from: n.source,
                    to: m.source,
                    obs: n.obs ^ e.obs ^ m.obs,
                    weight: n.dist + e.w + m.dist,
                };
                // Orient so that `a` is the growing side.
                if tu.slope > 0 {
                    Some(MEvent::HitRegion {
                        a: top_u,
                        b: top_v,
                        e: ce,
                    })
                } else {
                    Some(MEvent::HitRegion {
                        a: top_v,
                        b: top_u,
                        e: ce.reversed(),
                    })
                }
            }
            Next::Boundary => {
                let (wb, ob) = g.boundary(u).unwrap_or((0, 0));
                let top_u = self.top(n.region);
                Some(MEvent::HitBoundary {
                    a: top_u,
                    e: CEdge {
                        from: n.source,
                        to: BOUNDARY,
                        obs: n.obs ^ ob,
                        weight: n.dist + wb,
                    },
                })
            }
        };
        self.schedule_node(g, u);
        ev
    }

    /// `u`'s top region takes the unowned node `v` across the edge `(w, obs)`.
    fn arrive(&mut self, g: &CompiledGraph, u: NodeId, v: NodeId, w: i64, obs: u64) {
        let n = self.nodes[u as usize];
        let top = self.top(n.region);
        let z = self.regions[top as usize].radius.at(self.now);
        debug_assert_eq!(
            n.wrapped + z,
            n.dist + w,
            "arrival before the region reached the node"
        );
        self.nodes[v as usize] = NodeState {
            region: top,
            source: n.source,
            dist: n.dist + w,
            obs: n.obs ^ obs,
            arrival_z: z,
            wrapped: n.wrapped,
            queued: NO_TIME,
        };
        self.regions[top as usize].shell.push(v);
        self.touched.push(v);
        self.schedule_node(g, v);
    }

    /// Un-own `v`; its owned neighbours may now grow into it.
    fn release(&mut self, g: &CompiledGraph, v: NodeId) {
        self.nodes[v as usize] = NodeState::FREE;
        for k in 0..g.edges(v).len() {
            let x = g.edges(v)[k].v;
            if self.nodes[x as usize].region != NONE {
                self.schedule_node(g, x);
            }
        }
    }

    /// Time of the next release / radius-0 event of a top-level shrinking region.
    fn next_shrink_time(&self, r: RegionId) -> Option<i64> {
        let reg = &self.regions[r as usize];
        if reg.slope() != -1 || reg.blossom_parent != NONE {
            return None;
        }
        let keep = usize::from(!reg.is_blossom()); // a leaf never releases its source
        let mut target = 0;
        if reg.shell.len() > keep {
            target = target.max(self.nodes[reg.shell[reg.shell.len() - 1] as usize].arrival_z);
        }
        reg.radius.time_of(target, self.now)
    }

    pub(crate) fn schedule_shrink(&mut self, r: RegionId) {
        if let Some(t) = self.next_shrink_time(r) {
            let q = self.regions[r as usize].shrink_queued;
            if q == NO_TIME || t < q {
                self.regions[r as usize].shrink_queued = t;
                self.push_event(t, EV_SHRINK, r);
            }
        }
    }

    /// Pop-time handler for a shrink event.
    pub(crate) fn shrink_step(&mut self, g: &CompiledGraph, r: RegionId) -> Option<MEvent> {
        let t = self.next_shrink_time(r)?;
        if t > self.now {
            self.regions[r as usize].shrink_queued = t;
            self.push_event(t, EV_SHRINK, r);
            return None;
        }
        let z = self.regions[r as usize].radius.at(self.now);
        let keep = usize::from(!self.regions[r as usize].is_blossom());
        while self.regions[r as usize].shell.len() > keep {
            let last = *self.regions[r as usize].shell.last().unwrap_or(&NONE);
            if self.nodes[last as usize].arrival_z < z {
                break;
            }
            self.regions[r as usize].shell.pop();
            self.release(g, last);
        }
        if z == 0 {
            let reg = &self.regions[r as usize];
            if reg.is_blossom() {
                return Some(MEvent::Shatter { b: r });
            }
            // Degenerate implosion: the parent outer and the child outer meet through this
            // zero-radius inner region.
            let n = &self.trees[reg.tree as usize];
            debug_assert_eq!(n.inner, r);
            let p = &self.trees[n.parent as usize];
            return Some(MEvent::HitRegion {
                a: p.outer,
                b: n.outer,
                e: n.parent_edge.then(n.inner_to_outer),
            });
        }
        self.schedule_shrink(r);
        None
    }

    /// All nodes owned by `r` or any blossom descendant, into `self.scratch`.
    pub(crate) fn collect_subtree(&mut self, r: RegionId) {
        self.scratch.clear();
        let mut stack = vec![r];
        while let Some(x) = stack.pop() {
            let reg = &self.regions[x as usize];
            self.scratch.extend_from_slice(&reg.shell);
            stack.extend(reg.children.iter().map(|&(c, _)| c));
        }
    }

    /// Change a top-level region's slope at `now` and re-derive every affected event.
    pub(crate) fn set_slope(&mut self, g: &CompiledGraph, r: RegionId, slope: i8) {
        debug_assert_eq!(self.regions[r as usize].blossom_parent, NONE);
        let rad = self.regions[r as usize].radius;
        self.regions[r as usize].radius = rad.with_slope(self.now, slope);
        self.reschedule_region(g, r);
    }

    /// Re-derive events for every node of `r`'s subtree (after a slope or wrapping change).
    pub(crate) fn reschedule_region(&mut self, g: &CompiledGraph, r: RegionId) {
        self.collect_subtree(r);
        let nodes = std::mem::take(&mut self.scratch);
        for &u in &nodes {
            self.schedule_node(g, u);
        }
        self.scratch = nodes;
        if self.regions[r as usize].slope() < 0 {
            self.schedule_shrink(r);
        }
    }

    /// `wrapped += delta` for every node of `c`'s subtree.
    pub(crate) fn shift_wrapped(&mut self, c: RegionId, delta: i64) {
        self.collect_subtree(c);
        let nodes = std::mem::take(&mut self.scratch);
        for &u in &nodes {
            self.nodes[u as usize].wrapped += delta;
        }
        self.scratch = nodes;
    }
}

impl Region {
    #[inline]
    pub(crate) fn slope(&self) -> i8 {
        self.radius.slope
    }
}
