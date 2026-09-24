//! Per-shot mutable state of the sparse matcher: node records, region and tree arenas, and the
//! event heap. Everything here is plain data; the flooder and matcher own the logic.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub(crate) type NodeId = u32;
pub(crate) type RegionId = u32;
pub(crate) type TreeId = u32;
/// "No node / region / tree".
pub(crate) const NONE: u32 = u32::MAX;
/// Pseudo-region: matched to the boundary. Also the `to` of a boundary compressed edge.
pub(crate) const BOUNDARY: u32 = u32::MAX - 1;
/// "No event queued".
pub(crate) const NO_TIME: i64 = i64::MIN;
pub(crate) const EV_NODE: u8 = 0;
pub(crate) const EV_SHRINK: u8 = 1;

/// A radius that is affine in the global clock: `value(t) = y0 + slope · t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Varying {
    pub y0: i64,
    pub slope: i8,
}

impl Varying {
    pub(crate) fn frozen(v: i64) -> Self {
        Varying { y0: v, slope: 0 }
    }
    #[inline]
    pub(crate) fn at(self, t: i64) -> i64 {
        self.y0 + self.slope as i64 * t
    }
    /// Same value at `t`, new slope from `t` on.
    pub(crate) fn with_slope(self, t: i64, slope: i8) -> Self {
        Varying {
            y0: self.at(t) - slope as i64 * t,
            slope,
        }
    }
    /// Time `≥ now` at which the value equals `target`, if the slope ever gets there.
    pub(crate) fn time_of(self, target: i64, now: i64) -> Option<i64> {
        let t = match self.slope {
            1 => target - self.y0,
            -1 => self.y0 - target,
            _ => return None,
        };
        (t >= now).then_some(t)
    }
}

/// A path between two defects (or a defect and the boundary), compressed to its endpoints, the
/// observable parity along it and its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub obs: u64,
    pub weight: i64,
}

impl CEdge {
    pub(crate) const EMPTY: CEdge = CEdge {
        from: NONE,
        to: NONE,
        obs: 0,
        weight: 0,
    };
    pub(crate) fn reversed(self) -> Self {
        CEdge {
            from: self.to,
            to: self.from,
            ..self
        }
    }
    /// Concatenate `self` (ending at `x`) with `next` (starting at `x`).
    pub(crate) fn then(self, next: CEdge) -> Self {
        debug_assert_eq!(self.to, next.from);
        CEdge {
            from: self.from,
            to: next.to,
            obs: self.obs ^ next.obs,
            weight: self.weight + next.weight,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeState {
    /// Bottom-level region owning this node, or `NONE`.
    pub region: RegionId,
    /// The defect whose growth reached this node.
    pub source: NodeId,
    /// Distance from `source` along the growth path.
    pub dist: i64,
    /// Observable parity from `source`.
    pub obs: u64,
    /// The owning region's own radius when it reached this node (release order under shrinking).
    pub arrival_z: i64,
    /// Sum of the frozen radii of the region chain below the top-level region.
    pub wrapped: i64,
    /// Time of the queued look-at event, or `NO_TIME`.
    pub queued: i64,
}

impl NodeState {
    pub(crate) const FREE: NodeState = NodeState {
        region: NONE,
        source: NONE,
        dist: 0,
        obs: 0,
        arrival_z: 0,
        wrapped: 0,
        queued: NO_TIME,
    };
}

#[derive(Clone, Debug)]
pub(crate) struct Region {
    pub radius: Varying,
    pub blossom_parent: RegionId,
    /// Alternating-tree node this region belongs to (as inner or outer), or `NONE`.
    pub tree: TreeId,
    /// Matched partner region, `BOUNDARY`, or `NONE`.
    pub matched_to: RegionId,
    /// Match edge from this region's defect outward.
    pub match_edge: CEdge,
    /// Blossom children in cyclic order with the edge from each child to the next (last → first).
    pub children: Vec<(RegionId, CEdge)>,
    /// Nodes this region itself acquired, in arrival order.
    pub shell: Vec<NodeId>,
    /// The defect for a leaf region; `NONE` for a blossom.
    pub source: NodeId,
    /// Time of the queued shrink event, or `NO_TIME`.
    pub shrink_queued: i64,
}

impl Region {
    pub(crate) fn leaf(source: NodeId) -> Self {
        Region {
            radius: Varying { y0: 0, slope: 1 },
            blossom_parent: NONE,
            tree: NONE,
            matched_to: NONE,
            match_edge: CEdge::EMPTY,
            children: Vec::new(),
            shell: vec![source],
            source,
            shrink_queued: NO_TIME,
        }
    }
    pub(crate) fn is_blossom(&self) -> bool {
        !self.children.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TreeNode {
    /// `NONE` for a root.
    pub inner: RegionId,
    pub outer: RegionId,
    /// From the inner region's defect to the outer region's defect (the pair's match edge).
    pub inner_to_outer: CEdge,
    pub parent: TreeId,
    /// From the parent's outer defect to this node's inner defect.
    pub parent_edge: CEdge,
    pub children: Vec<TreeId>,
    pub alive: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Stats {
    pub events: u64,
    pub pushes: u64,
}

pub(crate) struct State {
    pub nodes: Vec<NodeState>,
    pub touched: Vec<NodeId>,
    pub regions: Vec<Region>,
    pub trees: Vec<TreeNode>,
    heap: BinaryHeap<Reverse<(i64, u64, u8, u32)>>,
    seq: u64,
    pub now: i64,
    pub active_trees: usize,
    /// Reusable scratch for subtree walks.
    pub scratch: Vec<NodeId>,
    pub stats: Stats,
}

impl State {
    pub(crate) fn new(num_nodes: usize) -> Self {
        State {
            nodes: vec![NodeState::FREE; num_nodes],
            touched: Vec::new(),
            regions: Vec::new(),
            trees: Vec::new(),
            heap: BinaryHeap::new(),
            seq: 0,
            now: 0,
            active_trees: 0,
            scratch: Vec::new(),
            stats: Stats::default(),
        }
    }

    /// Return to the pristine state, touching only what the last shot touched.
    pub(crate) fn reset(&mut self) {
        for &u in &self.touched {
            self.nodes[u as usize] = NodeState::FREE;
        }
        self.touched.clear();
        self.regions.clear();
        self.trees.clear();
        self.heap.clear();
        self.seq = 0;
        self.now = 0;
        self.active_trees = 0;
        self.stats = Stats::default();
    }

    pub(crate) fn push_event(&mut self, t: i64, kind: u8, id: u32) {
        self.seq += 1;
        self.stats.pushes += 1;
        self.heap.push(Reverse((t, self.seq, kind, id)));
    }

    pub(crate) fn pop_event(&mut self) -> Option<(i64, u8, u32)> {
        self.heap.pop().map(|Reverse((t, _, k, id))| (t, k, id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varying_evaluates_and_changes_slope_continuously() {
        let r = Varying { y0: 0, slope: 1 };
        assert_eq!(r.at(7), 7);
        let f = r.with_slope(7, 0);
        assert_eq!((f.at(7), f.at(100)), (7, 7));
        let s = f.with_slope(10, -1);
        assert_eq!((s.at(10), s.at(13)), (7, 4));
        assert_eq!(s.time_of(0, 10), Some(17));
        assert_eq!(f.time_of(9, 10), None);
        assert_eq!(r.time_of(3, 5), None); // already past
    }

    #[test]
    fn cedge_reverses_and_chains() {
        let a = CEdge {
            from: 1,
            to: 2,
            obs: 0b01,
            weight: 3,
        };
        let b = CEdge {
            from: 2,
            to: 5,
            obs: 0b11,
            weight: 4,
        };
        assert_eq!(
            a.reversed(),
            CEdge {
                from: 2,
                to: 1,
                obs: 0b01,
                weight: 3
            }
        );
        assert_eq!(
            a.then(b),
            CEdge {
                from: 1,
                to: 5,
                obs: 0b10,
                weight: 7
            }
        );
    }

    #[test]
    fn heap_pops_in_time_then_insertion_order() {
        let mut st = State::new(4);
        st.push_event(5, EV_NODE, 1);
        st.push_event(2, EV_SHRINK, 9);
        st.push_event(5, EV_NODE, 0);
        assert_eq!(st.pop_event(), Some((2, EV_SHRINK, 9)));
        assert_eq!(st.pop_event(), Some((5, EV_NODE, 1)));
        assert_eq!(st.pop_event(), Some((5, EV_NODE, 0)));
        assert_eq!(st.pop_event(), None);
        assert_eq!(st.stats.pushes, 3);
    }

    #[test]
    fn reset_clears_touched_nodes_only() {
        let mut st = State::new(3);
        st.nodes[2].region = 7;
        st.touched.push(2);
        st.regions.push(Region::leaf(2));
        st.reset();
        assert_eq!(st.nodes[2].region, NONE);
        assert!(st.regions.is_empty() && st.touched.is_empty());
    }
}
