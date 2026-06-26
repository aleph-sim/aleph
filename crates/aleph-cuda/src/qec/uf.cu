// Q3-01 GPU Union-Find / cluster-growth decoder.
//
// Design: ONE THREAD PER SYNDROME SHOT. Each thread runs the full serial Delfosse-Nickerson
// decode (cluster growth + Delfosse peeling) on its own syndrome, against the shared read-only
// matching graph. Throughput comes from batch parallelism (thousands of independent shots in
// flight), not from parallelising a single decode — which keeps the result *bit-identical* to the
// CPU `UnionFindDecoder` (no atomic-merge ordering to diverge on). The CPU decoder is the oracle.
//
// Why this reproduces the CPU result exactly:
//   * Cluster growth is replicated edge-centrically. Per round, an ungrown edge whose two
//     endpoints lie in different clusters gains one unit of support for each endpoint whose cluster
//     is *odd* (odd defect parity AND not boundary-touching) — exactly the per-vertex visitation the
//     CPU performs, but enumerated over edges instead of cluster-vertex lists. All support deltas in
//     a round are computed against the round-start partition, then unions are applied; so the set of
//     fully-grown (erasure) edges is independent of cluster-vertex iteration order, hence identical.
//   * Peeling builds the spanning forest in the SAME order as the CPU: the boundary tree first (when
//     the boundary is in the erasure), then defects in ascending detector index; BFS follows the
//     identical CSR adjacency order. The reverse pre-order peel then selects the identical edges, so
//     the XOR-ed observable mask matches bit-for-bit.
//
// Scratch is per-thread, laid out as `arr[shot * stride + i]`: node arrays stride `n_nodes`, edge
// arrays stride `n_edges`. The graph arrays are shared (no shot index). The kernel re-initialises
// its scratch region at entry, so buffers may be reused across launches without a host memset.

typedef unsigned int u32;
typedef unsigned char u8;
typedef unsigned long long u64;

#define UF_NONE 0xffffffffu

// Union-Find `find` with path halving over this shot's `parent` slice.
__device__ __forceinline__ u32 uf_find(u32 *parent, u32 v) {
    while (parent[v] != v) {
        u32 gp = parent[parent[v]];
        parent[v] = gp;
        v = gp;
    }
    return v;
}

// Union by size (CPU tie-break: equal sizes ⇒ first root wins). Combines parity + boundary-touch.
__device__ __forceinline__ void uf_union(u32 *parent, u32 *sz, u8 *parity, u8 *btouch, u32 a,
                                          u32 b) {
    u32 ra = uf_find(parent, a);
    u32 rb = uf_find(parent, b);
    if (ra == rb) return;
    u32 big, small;
    if (sz[ra] >= sz[rb]) {
        big = ra;
        small = rb;
    } else {
        big = rb;
        small = ra;
    }
    parent[small] = big;
    sz[big] += sz[small];
    parity[big] ^= parity[small];
    btouch[big] |= btouch[small];
}

extern "C" __global__ void uf_decode(
    // --- shared read-only matching graph ---
    const u32 *adj_off,    // [n_nodes + 1]
    const u32 *adj_edges,  // [n_adj]
    const u32 *edge_a,     // [n_edges]
    const u32 *edge_b,     // [n_edges]
    const u64 *edge_obs,   // [n_edges]
    const u32 *edge_len,   // [n_edges] (weighted mode only)
    u32 n_nodes, u32 n_edges, u32 n_detectors, u32 weighted,
    // --- input syndromes (packed detector bits) ---
    const u32 *syn_words,  // [n_shots * words_per_shot]
    u32 words_per_shot, u32 n_shots,
    // --- output: one observable-flip bitmask per shot ---
    u64 *out_mask,  // [n_shots]
    // --- per-shot scratch (stride n_nodes / n_edges) ---
    u32 *parent, u32 *sz, u32 *acc, u8 *parity, u8 *btouch, u8 *grown, u8 *syn_bit, u8 *visited,
    u32 *parent_edge, u32 *parent_node, u32 *order) {
    u32 shot = blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= n_shots) return;

    // This shot's scratch slices.
    u32 *par = parent + (size_t)shot * n_nodes;
    u32 *size_ = sz + (size_t)shot * n_nodes;
    u8 *pty = parity + (size_t)shot * n_nodes;
    u8 *bt = btouch + (size_t)shot * n_nodes;
    u8 *syn = syn_bit + (size_t)shot * n_nodes;
    u8 *vis = visited + (size_t)shot * n_nodes;
    u32 *pedge = parent_edge + (size_t)shot * n_nodes;
    u32 *pnode = parent_node + (size_t)shot * n_nodes;
    u32 *ord = order + (size_t)shot * n_nodes;
    u32 *ea_acc = acc + (size_t)shot * n_edges;
    u8 *gr = grown + (size_t)shot * n_edges;
    const u32 *words = syn_words + (size_t)shot * words_per_shot;
    u32 boundary = n_detectors;

    // --- init scratch + seed defect parities from the packed syndrome ---
    for (u32 v = 0; v < n_nodes; ++v) {
        par[v] = v;
        size_[v] = 1;
        pty[v] = 0;
        bt[v] = (v == boundary) ? 1 : 0;
        syn[v] = 0;
        vis[v] = 0;
        pedge[v] = UF_NONE;
        pnode[v] = 0;
    }
    for (u32 e = 0; e < n_edges; ++e) {
        ea_acc[e] = 0;
        gr[e] = 0;
    }
    u32 n_defects = 0;
    for (u32 d = 0; d < n_detectors; ++d) {
        if ((words[d >> 5] >> (d & 31)) & 1u) {
            pty[d] = 1;
            syn[d] = 1;  // peeling residual seeded here too (CPU sets syn from the same defects)
            ++n_defects;
        }
    }
    if (n_defects == 0) {
        out_mask[shot] = 0;
        return;
    }

    // --- Phase 1: cluster growth ---------------------------------------------------------------
    // Safety bound on rounds (each round either completes ≥1 edge or terminates). n_edges is a hard
    // ceiling; the surface-code DEM converges in ~O(d) rounds.
    for (u32 iter = 0; iter <= n_edges; ++iter) {
        // Determine the jump `delta`. Unweighted: always 1 (synchronous half-edge growth). Weighted
        // (Q2-02 jump step): the fewest units that complete the next edge anywhere this round.
        u32 delta = 1;
        bool any = false;
        if (weighted) {
            delta = UF_NONE;
            for (u32 e = 0; e < n_edges; ++e) {
                if (gr[e]) continue;
                u32 ra = uf_find(par, edge_a[e]);
                u32 rb = uf_find(par, edge_b[e]);
                if (ra == rb) continue;
                u32 sides = (pty[ra] && !bt[ra]) + (pty[rb] && !bt[rb]);
                if (sides == 0) continue;
                any = true;
                u32 rem = edge_len[e] - ea_acc[e];
                u32 need = (rem + sides - 1) / sides;  // ceil(rem / sides)
                if (need < delta) delta = need;
            }
            if (!any) break;
        }

        // Apply growth to every boundary edge of an odd cluster, marking completions. Roots are read
        // against the round-start partition (no unions happen in this pass).
        // First pass: accumulate support / growth and flag newly-grown edges.
        bool grew_any = false;
        for (u32 e = 0; e < n_edges; ++e) {
            if (gr[e]) continue;
            u32 ra = uf_find(par, edge_a[e]);
            u32 rb = uf_find(par, edge_b[e]);
            if (ra == rb) continue;
            u32 sides = (pty[ra] && !bt[ra]) + (pty[rb] && !bt[rb]);
            if (sides == 0) continue;
            any = true;
            if (weighted) {
                ea_acc[e] += delta * sides;
                if (ea_acc[e] >= edge_len[e]) {
                    gr[e] = 1;
                    grew_any = true;
                }
            } else {
                u32 s = ea_acc[e] + sides;
                if (s > 2) s = 2;  // CPU caps support at 2
                ea_acc[e] = s;
                if (s >= 2) {
                    gr[e] = 1;
                    grew_any = true;
                }
            }
        }
        if (!any) break;  // no odd cluster has a growth edge ⇒ all neutral (or unsatisfiable)
        // Second pass: fuse the edges that completed this round.
        if (grew_any) {
            for (u32 e = 0; e < n_edges; ++e) {
                if (gr[e] && uf_find(par, edge_a[e]) != uf_find(par, edge_b[e])) {
                    uf_union(par, size_, pty, bt, edge_a[e], edge_b[e]);
                }
            }
        }
    }

    // --- Phase 2: peel the erasure into a correction -------------------------------------------
    // Build the spanning forest. Root the boundary tree first when the boundary is in the erasure
    // (so leftover parity drains into it), then each defect's component in ascending index order.
    u32 head = 0;  // write cursor into `ord`
    // Boundary in erasure iff one of its incident edges is fully grown.
    bool boundary_in_erasure = false;
    for (u32 i = adj_off[boundary]; i < adj_off[boundary + 1]; ++i) {
        if (gr[adj_edges[i]]) {
            boundary_in_erasure = true;
            break;
        }
    }

    // BFS one component (no-op if `start` already visited or has no grown incident edge).
    // Inlined for both root kinds: appends to `ord[head..]`, records parent edge/node.
#define UF_BFS(start)                                                                  \
    do {                                                                               \
        u32 s0 = (start);                                                              \
        if (!vis[s0]) {                                                                \
            vis[s0] = 1;                                                               \
            pedge[s0] = UF_NONE;                                                        \
            u32 qh = head;                                                             \
            ord[head++] = s0;                                                          \
            while (qh < head) {                                                        \
                u32 u = ord[qh++];                                                     \
                for (u32 i = adj_off[u]; i < adj_off[u + 1]; ++i) {                    \
                    u32 e = adj_edges[i];                                              \
                    if (!gr[e]) continue;                                              \
                    u32 w = (edge_a[e] == u) ? edge_b[e] : edge_a[e];                  \
                    if (!vis[w]) {                                                     \
                        vis[w] = 1;                                                    \
                        pedge[w] = e;                                                  \
                        pnode[w] = u;                                                  \
                        ord[head++] = w;                                              \
                    }                                                                  \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    } while (0)

    if (boundary_in_erasure) UF_BFS(boundary);
    for (u32 d = 0; d < n_detectors; ++d) {
        if (syn[d]) UF_BFS(d);  // syn[d] still flags the original defects here
    }
#undef UF_BFS

    // Reverse pre-order peel: a still-lit leaf pushes its pendant edge and toggles its parent.
    u64 mask = 0;
    for (u32 k = head; k-- > 0;) {
        u32 u = ord[k];
        u32 pe = pedge[u];
        if (pe != UF_NONE && syn[u]) {
            mask ^= edge_obs[pe];
            syn[pnode[u]] ^= 1;
            syn[u] = 0;
        }
    }
    out_mask[shot] = mask;
}
