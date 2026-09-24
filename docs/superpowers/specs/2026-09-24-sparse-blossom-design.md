# Sparse Blossom MWPM — design (Q1-03b, issue #331)

**Status:** draft for review. **Date:** 2026-09-24.
**Scope:** replace the per-shot matching engine of `MwpmDecoder` with a local, event-driven
region-growing implementation of Edmonds' primal-dual blossom algorithm on the detector graph —
the algorithm known as *Sparse Blossom* (Higgott & Gidney, arXiv:2303.15933, the core of
PyMatching v2). Re-derived from the paper and from Edmonds/Galil; no code from PyMatching or
Blossom V is read or copied (CLAUDE.md "never copy code").

## 1. Intent (what, for whom, success)

**Stated in the issue.** Q1-03 (savings reformulation) got the decoder to ~3.7× the Q1-02 dense
baseline and plateaued: the textbook blossom restarts its alternating-tree search on every
augmentation stage, `O(n·m)` per shot, and that is compute, not allocation. The goal is a matcher
whose work is proportional to the number of defects and the neighbourhood they explore, so that
(a) the Q1-03 acceptance criterion **≥ 10× over the dense baseline at d = 11, threshold density**
is met and (b) the decoder **scales to high distance** (d ≥ 13) instead of growing quadratically.

**Success criteria (acceptance, verbatim from #331):**

1. ≥ 10× faster than `decode_dense` on d = 11 phenomenological syndromes at p = 0.03.
2. Identical corrections to the existing decoder: differential against `decode_dense` on 10⁵
   random syndromes — **weight-identical on every shot**, corrections equal modulo genuine ties.
3. Criterion benchmark d ∈ {7, 9, 11, 13}, syndromes/second reported; sweep extended to
   d ∈ {15, 17} to show the scaling crossover.

**Assumptions I am making (correct me):**

- The sparse matcher becomes the default `Decoder::decode` path. `decode_dense` stays as the
  reference oracle; the Q1-03 savings path stays too (as a second, cheaper oracle) but is no
  longer public.
- The all-pairs Dijkstra tables (`O(D²)` memory) are no longer built in `MwpmDecoder::new`; they
  are built lazily on first use of a dense/local path. Otherwise the memory cost, not the matching,
  would cap the reachable distance.
- The public API (`MwpmDecoder::new`, `from_graph`, `decode`, `decode_dense`, `with_locality_k`)
  and the Python surface (`aleph.qec` decoder name `"mwpm"`) do not change. `with_locality_k`
  only affects the (now non-default) local path and is documented as such.
- Circuit-level DEMs (stim-generated, ~4× the edges per node of the phenomenological graph) are in
  scope for correctness tests from day one; the single-thread PyMatching comparison in the Python
  README is refreshed at the end but is *not* an acceptance gate (the AC is vs `decode_dense`).
- No new crate dependencies.

## 2. Approaches considered

**A. Sparse Blossom — event-driven region growth on the detector graph (recommended).**
Grow a "region" (a dual variable) around each defect directly on the sparse detector graph; keep
alternating trees, blossoms and duals *persistent* across augmentations; drive everything by a
priority queue of "when does the next thing happen" events. No all-pairs precompute, no candidate
edge list. Work per shot is ~linear in the number of defects times the explored neighbourhood.
This is what the issue asks for and the only approach that satisfies both the 10× *and* the
high-distance goal. It is also the largest: PyMatching's engine is several thousand lines of C++.
Risk: research-grade correctness. Mitigation: three independent oracles (§7) and weight-identity,
not just correction equality.

**B. Blossom-V-style persistent trees on the Q1-03 candidate graph.** Keep the all-pairs tables
and the savings edge build; replace only the blossom engine by one that grows all trees
simultaneously with per-tree priority queues and never restarts. Removes the `O(n·m)` restart,
but keeps the `O(n²)` edge build (58 µs of the 288 µs at d = 11 today, and growing) and the
`O(D²)` tables. Ceiling ~3–5× over Q1-03: reaches 10× over dense only marginally, and does
nothing for high distance. Rejected.

**C. Bounded local-Dijkstra candidate graph + existing blossom.** Replace the all-pairs edge
build with a per-defect region growth that stops after K neighbours, then run the textbook
blossom on that sparse graph. Not weight-exact (the K = 12 experiment already failed the
differential at d = 11), so it violates AC 2. Rejected.

## 3. Algorithm (as implemented, in our own terms)

### 3.1 The LP being solved

Defects `V`, shortest-path metric `d(·,·)` on the detector graph, boundary distance `b_v`. The
Q1-03 record shows MWPM-with-boundary is the max-weight *non-perfect* matching on savings
`s(i,j) = b_i + b_j − d(i,j)`. Its dual, after the substitution `y_v = b_v − y'_v`, is

```
maximise  Σ_v y_v + Σ_B z_B          (B over odd sets, z_B ≥ 0)
s.t.      y_u + y_v + Σ_{B ∋ u,v} z_B ≤ d(u,v)     for every defect pair
          y_v ≤ b_v                                  (the boundary has dual 0)
```

with `y_v` otherwise unrestricted in sign. A defect's `y_v` is its **region radius**; a blossom's
`z_B` is the blossom's radius. Complementary slackness: matched pairs are tight, a defect matched
to the boundary has `y_v = b_v`, a blossom with `z_B > 0` is fully matched inside plus one edge
out. Edmonds' primal-dual maintains feasibility and grows the objective until the matching is
perfect (every defect matched to a defect or to the boundary) — at which point it is optimal.

### 3.2 Regions and time

Global integer clock `t` (same `2^24` weight scale as today, `i64`). Each region's radius is
affine in time: `radius(t) = y0 + slope·t`, `slope ∈ {−1, 0, +1}`; **outer** tree regions grow
(+1), **inner** tree regions shrink (−1), matched regions are frozen (0). A blossom is a region
whose children are regions; children's radii are frozen when the blossom forms. The **total
radius** of a node `u` owned by bottom-level region `R` is the sum of radii along the chain
`R → parent blossom → … → top`; only the top varies, so `total(u,t) = wrapped(u) + top(u).radius(t)`
with `wrapped(u)` cached and updated when blossoms form or shatter.

Each detector node records, while owned: bottom-level `region`, `source` (the *defect* whose
growth reached it — always a defect, even inside blossoms), `dist` (distance from `source` along
the growth path), `obs` (observable parity from `source`), `radius_at_arrival` (total radius when
it was reached — the release order under shrinking), `wrapped`.

### 3.3 Events (flooder)

A node is *covered* when `total(u,t) ≥ dist(u)`. For an edge `(u,v)` of weight `w`:

- `v` unowned, `u` in a growing region: **arrival** at `total(u,t) = dist(u) + w`; `v` joins `u`'s
  top region with `source(v)=source(u)`, `dist(v)=dist(u)+w`, `obs(v)=obs(u)⊕obs(e)`.
- `u`, `v` owned by different top regions `A`, `B` with combined slope `> 0`: **collision** when
  `total(u,t) + total(v,t) = dist(u) + w + dist(v)` → matcher event `RegionHitRegion(A, B, e)`
  where the *compressed edge* `e = (source(u), source(v), obs(u)⊕obs(e)⊕obs(v), dist(u)+w+dist(v))`.
  Combined slope ≤ 0 (outer–inner, or two frozen regions) never produces an event; this is what
  keeps the "constant slack" pairs quiet.
- `u` has a boundary edge `w_b`: **boundary hit** at `total(u,t) = dist(u) + w_b` → matcher event
  `RegionHitBoundary(A, (source(u), boundary, obs(u)⊕obs_b, dist(u)+w_b))`.
- A shrinking region releases its most recently arrived node when `total = radius_at_arrival`
  (nodes are kept in arrival order, so release is a stack pop). A shrinking **blossom** whose own
  radius reaches 0 → `BlossomShatter(B)`. A shrinking **defect region** whose radius reaches 0
  does *not* go negative: its source node stays owned and the flooder emits a
  `RegionHitRegion(parent_outer, child_outer, e_parent ⊕ e_child)` through it ("degenerate
  implosion"). §3.5 argues why this is exactly Edmonds.

Scheduling is lazy: one tentative "look at node `u`" event per node and one "look at shrinking
region" per region, each with a tracker `{queued_at, desired}`. Whenever `u`'s neighbourhood or
its region's slope changes, `desired` is recomputed and a new heap entry pushed only if it is
earlier than what is queued; popped entries whose time ≠ `queued_at` are stale and dropped.
When a node event pops at its desired time, the flooder executes every neighbour event due *now*
(arrivals, collisions, boundary) and reschedules. Simultaneous events are ordered by
`(time, sequence)` for determinism; two regions reaching an unowned node at the same time
resolve as "first pop arrives, second pop sees a collision at `now`".

The queue is a binary heap of `(time, seq, NodeEvent | RegionEvent)`. A radix/bucket queue is a
later optimisation if profiling shows the heap on the critical path.

### 3.4 Alternating trees and blossoms (matcher)

Standard Edmonds on regions. Tree node = `(inner region, outer region)` pair (the root has only an
outer region), with parent/child compressed edges. On events:

- `RegionHitRegion(A outer in T, B outer in T′ ≠ T)` — **augment**: match `A–B` via `e`, flip
  the alternating paths from `A` and `B` to their roots into matched pairs, freeze every region of
  both trees (slope 0).
- `RegionHitRegion(A, B both outer in T)` — **blossom**: walk both to the lowest common ancestor,
  the two tree paths plus `e` form an odd cycle of regions; create a blossom region with those
  children (each with its frozen radius and the cycle edge to the next child), `radius = 0`,
  slope +1; it replaces the cycle as one outer tree node inheriting the children's other tree
  children. Update `wrapped` for every node owned by any descendant (`O(size)`).
- `RegionHitRegion(A outer in T, M frozen, matched to M′)` — **grow**: `(M inner, M′ outer)`
  becomes a child of `A`; `M` starts shrinking, `M′` growing; reschedule their nodes.
- `RegionHitBoundary(A outer in T)` — **augment to boundary**: `A` matches the boundary, the path
  to the root flips as above, the tree freezes.
- `BlossomShatter(B inner in T)` — `B`'s parent edge lands on child `p`, its child edge on child
  `q`. The odd cycle splits into the odd-length path `p … q` (which becomes alternating inner/outer
  tree nodes replacing `B`) and the even remainder (matched pairwise along the cycle edges, frozen).
  Children regain their frozen radii as top-level regions; `wrapped` updated.
- A `RegionHitRegion` where the two top regions are the same (a blossom's child hitting a sibling)
  cannot occur — the flooder skips same-top pairs.

Termination: no tree remains (queue empty or all regions frozen). If the queue drains while trees
remain (an odd component with no boundary), the remaining tree roots stay unmatched — best-effort,
as today, documented.

### 3.5 Why the dual stays feasible (the two non-obvious cases)

*Outer hitting inner in the same tree* never fires because the slack is constant. *Outer `A`
reclaiming territory from a shrinking inner `V` in its own tree*: `A`'s frontier advances at the
rate `V`'s shell recedes along geodesics through `V`'s source, so `A` re-takes a node exactly when
`V` releases it iff the node is on an `A–V` geodesic, and otherwise takes it strictly later. Hence
if the true `d(p,c)` between the parent and child outer regions of `V` is shorter than
`d(p,V)+d(V,c)`, the geodesic nodes are released in time (triangle inequality) and `p`, `c`
collide *before* `y_V` reaches 0 — a normal blossom. If not, they meet exactly at `y_V = 0` and the
degenerate implosion forms the blossom `{p, V, c}` with `z = 0`, which is the tight-edge blossom
Edmonds would form. So `y_v ≥ 0` for defects is a consequence, not an extra constraint; `z_B ≥ 0`
is enforced by shattering at 0. The boundary constraint `y_v ≤ b_v` is the boundary-hit event.

The metric subtlety (a region's `dist` is the geodesic *inside its own territory*, which can be
longer than the true shortest path) is handled by the same argument the paper gives: the first
collision between two regions occurs along a true shortest path, because any shorter route through
a third region's territory would have produced an earlier collision with that region. We do not
rely on the argument alone — §7 verifies weight-identity against the exact all-pairs oracle.

### 3.6 Output

The final matching is a set of top-level `(region, region | boundary, compressed edge)`. Blossoms
are resolved recursively: the child containing the outside edge's inner endpoint takes that edge;
the remaining even cycle is matched pairwise along its stored cycle edges. Every resulting edge
connects two *defects* (or a defect and the boundary) and carries `(obs, weight)`, so the
correction is the XOR of the `obs` fields and the total weight is the sum — the same two numbers
`decode_dense` produces, which is what makes the differential test exact.

## 4. Components and files

```
crates/aleph-qec/src/sparse_blossom/
  mod.rs        pub(crate) SparseMatcher: compile(graph) → CompiledGraph; decode(defects) → (obs, weight)
  graph.rs      CompiledGraph: CSR adjacency (neighbour u32, weight i64, obs u64), per-node boundary
                edge (min weight over parallel boundary edges), built once per DEM
  state.rs      per-shot mutable state: node arrays (D-sized, reset via a touched list), region
                arena, tree-node arena, event heap, sequence counter
  flooder.rs    §3.3: scheduling, arrival/collision/boundary/release, degenerate implosion
  matcher.rs    §3.4: augment, blossom, grow, boundary, shatter; final resolution (§3.6)
crates/aleph-qec/src/mwpm.rs
  MwpmDecoder gains `sparse: SparseMatcher` (compiled graph + a per-decoder id into a thread-local
  State cache so `decode(&self)` stays usable from rayon with no lock on the hot path); dense
  tables move behind a OnceLock; `decode` → sparse;
  `decode_dense` / `decode_local` become the oracles. The all-pairs Dijkstra moves with them.
benches/benches/mwpm_decode.rs   adds "sparse", extends d to {7,9,11,13,15,17}
docs/perf/q1-03b-sparse-blossom.md   the perf record (before/after, profile, verdict)
```

Parallel edges between the same pair with different observable sets: the compiled graph keeps
the minimum-weight one per pair (first by edge index on ties), which is what the all-pairs
Dijkstra implicitly did.

**Thread-local state cache.** `decode(&self)` is called from `rayon` (`run_dem_experiment`, the
Python batch path). A first pass used a `Mutex<Vec<Box<State>>>` pool (pop before, push after);
that re-contended badly once decode itself got down to ~1.5 µs at small distances (d=5), where the
lock's own overhead started to dominate the decode and multi-thread throughput fell *below*
single-thread. Replaced with a thread-local `Vec<(decoder id, State)>`: each `SparseMatcher` gets
a unique id at construction, and `decode` finds-or-allocates its entry in the calling thread's
cache — no lock, no cross-thread contention, one arena per (decoder, thread) pair reused across
calls. States are sized to `D` nodes and reset by walking the touched list, so per-shot cost is
independent of `D`.

**Integer types.** Node ids `u32`; region/tree ids `u32` arena indices; times and radii `i64`.
`INF`-free: unreachable pairs simply never collide.

## 5. Error handling

Constructor errors are unchanged (`NonGraphlike` from `MatchingGraph::from_dem`). Decoding cannot
fail: an unmatched tree root (odd component without a boundary) is skipped, mirroring today's
best-effort branch, and is covered by a unit test. Debug builds assert the dual invariants at
every matcher event (`slack ≥ 0` on the compressed edge, region slopes consistent with tree
role); release builds do not pay for them.

## 6. Performance plan

Measure first (CLAUDE.md): keep `profile_local_phases_d11` and add a sparse twin that reports
events processed, heap pushes, nodes touched per shot. Expected: tens of µs per shot at d = 11
(PyMatching is ~18 µs there on this Mac; a factor ≤ 2 of that is the realistic first landing,
i.e. ≥ 25× over dense). Optimisation levers, in order, only if the profile asks: radix heap;
`i32` weights/radii for cache density; region-size-aware iteration in blossom formation.

Benchmarks are run on an idle box (`uptime`, `pgrep` check) with `--baseline`, and the PR body
carries the criterion numbers for dense / local / sparse at every distance.

## 7. Testing

1. **Unit** (in `sparse_blossom/`): hand-built graphs — two defects on a line; one defect with a
   boundary; a triangle with equal weights (blossom); a five-cycle with a pendant boundary
   (blossom + boundary augment); the parent–inner–child line that forces a degenerate implosion;
   an odd component without boundary (best-effort unmatched); parallel edges with different
   observables (min-weight edge wins).
2. **Exhaustive oracle** (hermetic, `proptest`): random connected sparse graphs, ≤ 12 defects,
   random integer weights, optional boundary edges; the sparse result's total weight equals the
   all-pairs-Dijkstra + `max_weight_matching` optimum on every instance (this also cross-checks
   against the existing brute-force-validated blossom). Thousands of cases per run.
3. **Differential vs `decode_dense`** (hermetic, the AC): 10⁵ syndromes total across
   d ∈ {5, 7, 9, 11} phenomenological at p ∈ {0.01, 0.03, 0.06}, plus the circuit-level d = 5 and
   d = 7 DEMs already used by the sinter tests: **weight identical on every shot**; corrections
   identical except on ties, with the existing tie-rate sentinel. Runs in `--release` under a
   minute; the d = 11 slice is the slow part and is shot-capped in debug.
4. **PyMatching oracle** (`#[ignore]`, nightly): unchanged test, now exercising the sparse path
   — LER within CI at all distances and per-shot corrections ≥ 99 % equal in the sparse regime.
5. **Threshold regression** (`mwpm_threshold.rs`): unchanged.
6. **Property**: decoding the same syndrome twice, and across two threads, yields identical
   output (each thread's cached state is fully reset per shot).

## 8. Out of scope

Path reconstruction for per-edge corrections (we output observable flips only, as today);
a `num_neighbours`-style approximate mode; streaming / windowed matching; multi-threaded decoding
of a single shot; a Python API change.

## 9. Deliverables

One PR titled `[Q1-03b] Sparse Blossom: local region-growing MWPM` closing #331, with the perf
record, benchmark table, differential results, and the README throughput table refreshed.
