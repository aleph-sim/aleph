#include <metal_stdlib>
using namespace metal;

// Uniform for the GPU-resident split finalize (P5.8-03). MUST match the Rust
// `FinalizeMeta` struct (mps/kernel.rs): six used uints + two pad => 32 bytes.
//   rows, cols   : the gated block Θ′ shape (so k = min(rows, cols))
//   wide         : 1 when rows < cols (Θ′ was factored as its adjoint Aᴴ)
//   max_bond     : bond cap χ_max
//   renormalize  : 1 on the canonical/truncating path (rescale kept σ), else 0
//   k            : min(rows, cols) = the singular-value count
struct FinalizeMeta {
    uint rows;
    uint cols;
    uint wide;
    uint max_bond;
    uint renormalize;
    uint k;
    uint _pad0;
    uint _pad1;
};

// Output scalars (MUST match Rust `FinalizeOut`, 16 bytes): the kept bond χ, an
// accept flag (0 ⇒ host falls back to the f64 CPU SVD), and the relative discarded
// Schmidt weight.
struct FinalizeOut {
    uint chi;
    uint accept;
    float trunc_rel;
    float _pad;
};

// Largest block the in-threadgroup sort handles: k ≤ 2·χ_max. Above this the kernel
// declines (accept = 0) and the host f64 path takes over. 2048 ⇒ χ_max ≤ 1024.
constant uint FIN_MAXK = 2048u;
// Relative Frobenius residual ceiling for accepting the f32 Jacobi factors (matches
// the host `RECON_TOL` in gpu_jacobi.rs). A mis-converged one-sided Jacobi leaves U
// not quite isometric; the residual catches it and the host f64 SVD takes over.
constant float FIN_RECON_TOL = 1e-3f;

inline float2 cconj(float2 z) { return float2(z.x, -z.y); }
inline float2 cmul2(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Finish one two-site split entirely on the GPU (P5.8-03): sort σ descending, pick
// χ (matching the host `truncation_plan`), and assemble the two new site tensors
// `site_i = U[:, kept]` / `site_j = scale·diag(σ_kept)·Vᴴ` — so U/V/σ are never read
// back to the host. One threadgroup; threads stride the work.
//
//   a   : Jacobi A buffer (col-major). Tall: = U (rows×k). Wide: = V (cols×k).
//   v   : Jacobi V buffer (col-major). Tall: = V (cols×k). Wide: = U (rows×k).
//   sig : σ (k, kernel order, unsorted).
// The U/V roles swap with `wide` exactly as the host read-back labeling did.
kernel void finalize_split(device const float2* a        [[buffer(0)]],
                           device const float2* v        [[buffer(1)]],
                           device const float*  sig      [[buffer(2)]],
                           device float2*       site_i   [[buffer(3)]],
                           device float2*       site_j   [[buffer(4)]],
                           device FinalizeOut*  out       [[buffer(5)]],
                           constant FinalizeMeta& g       [[buffer(6)]],
                           device const float2* theta    [[buffer(7)]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint tcnt [[threads_per_threadgroup]]) {
    uint rows = g.rows, cols = g.cols, k = g.k, maxb = max(g.max_bond, 1u);
    bool wide = g.wide != 0u;
    // Caller U (rows×k col-major) and V (cols×k col-major).
    device const float2* U = wide ? v : a;
    device const float2* V = wide ? a : v;

    threadgroup float keys[FIN_MAXK]; // -σ² (so an ascending sort ⇒ σ descending)
    threadgroup uint  idxs[FIN_MAXK]; // carried original column index
    threadgroup uint  sh_chi;
    threadgroup float sh_scale;
    threadgroup uint  sh_accept;

    // Oversized block: decline, the host f64 SVD finishes it.
    if (k > FIN_MAXK) {
        if (tid == 0u) { out->chi = 0u; out->accept = 0u; out->trunc_rel = 1.0f; }
        return;
    }

    // --- Accuracy guard (P5.8-03, on-device): relative Frobenius residual
    // ‖Θ − UΣVᴴ‖_F / ‖Θ‖_F over ALL k columns (independent of truncation). A
    // mis-converged f32 Jacobi lands far above FIN_RECON_TOL → decline → host f64
    // SVD. Threads stride the rows·cols entries; a per-thread partial is reduced in
    // threadgroup memory. Replaces the old O(χ³) HOST reconstruction.
    threadgroup float red_num[256];
    threadgroup float red_den[256];
    float pnum = 0.0f, pden = 0.0f;
    bool finite = true;
    for (uint e = tid; e < rows * cols; e += tcnt) {
        uint r = e / cols;
        uint c = e % cols;
        float2 acc = float2(0.0f, 0.0f);
        for (uint t = 0u; t < k; ++t) {
            float2 u = U[r + t * rows];
            float2 vc = cconj(V[c + t * cols]);
            acc += cmul2(u, vc) * sig[t];
        }
        float2 e0 = theta[r * cols + c];
        float dr = acc.x - e0.x;
        float di = acc.y - e0.y;
        pnum += dr * dr + di * di;
        pden += e0.x * e0.x + e0.y * e0.y;
    }
    for (uint t = tid; t < k; t += tcnt) {
        if (!isfinite(sig[t])) finite = false;
    }
    red_num[tid] = pnum;
    red_den[tid] = pden;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float num = 0.0f, den = 0.0f;
        for (uint t = 0u; t < tcnt; ++t) { num += red_num[t]; den += red_den[t]; }
        float resid = (den > 0.0f) ? sqrt(num / den) : 0.0f;
        sh_accept = (finite && resid <= FIN_RECON_TOL) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sh_accept == 0u) {
        if (tid == 0u) { out->chi = 0u; out->accept = 0u; out->trunc_rel = 1.0f; }
        return;
    }

    // Pad to a power of two with +∞ keys (sort to the tail = smallest σ).
    uint n = 1u;
    while (n < k) n <<= 1u;
    for (uint i = tid; i < n; i += tcnt) {
        if (i < k) {
            float s = sig[i];
            keys[i] = -(s * s);
            idxs[i] = i;
        } else {
            keys[i] = INFINITY;
            idxs[i] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort ascending on `keys` (= σ² descending). Standard network; each
    // thread strides the index space, a barrier between every comparison stage.
    for (uint kk = 2u; kk <= n; kk <<= 1u) {
        for (uint j = kk >> 1u; j > 0u; j >>= 1u) {
            for (uint i = tid; i < n; i += tcnt) {
                uint l = i ^ j;
                if (l > i) {
                    bool asc = ((i & kk) == 0u);
                    bool need = asc ? (keys[i] > keys[l]) : (keys[i] < keys[l]);
                    if (need) {
                        float fk = keys[i]; keys[i] = keys[l]; keys[l] = fk;
                        uint ik = idxs[i]; idxs[i] = idxs[l]; idxs[l] = ik;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // χ-selection (matches host `truncation_plan`), serially on thread 0.
    if (tid == 0u) {
        float smax = sqrt(max(-keys[0], 0.0f));
        float eps = 1e-7f * max(smax, FLT_MIN);
        uint significant = 0u;
        float total = 0.0f;
        for (uint t = 0u; t < k; ++t) {
            float s2 = -keys[t];
            total += s2;
            if (sqrt(max(s2, 0.0f)) > eps) significant += 1u;
        }
        uint chi = min(max(significant, 1u), maxb);
        float discarded = 0.0f;
        for (uint t = chi; t < k; ++t) discarded += -keys[t];
        float trunc_rel = (total > 0.0f) ? (discarded / total) : 0.0f;
        // Rescale kept σ ONLY on a real truncation (P5.7-07); else leave verbatim so
        // the f32 σ error is not injected into the norm on every exact gate.
        float scale = 1.0f;
        if (g.renormalize != 0u && trunc_rel > 1e-12f) {
            float kept = 0.0f;
            for (uint t = 0u; t < chi; ++t) kept += -keys[t];
            scale = (kept > 0.0f) ? rsqrt(kept) : 1.0f;
        }
        sh_chi = chi;
        sh_scale = scale;
        // Accepted (recon guard already passed above).
        out->chi = chi;
        out->accept = 1u;
        out->trunc_rel = trunc_rel;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chi = sh_chi;
    float scale = sh_scale;

    // site_i ← U[:, kept]: row-major (rows × chi). site_i[row*chi + t] = U[row + idx*rows].
    uint ni = rows * chi;
    for (uint e = tid; e < ni; e += tcnt) {
        uint row = e / chi;
        uint t = e % chi;
        uint col = idxs[t];
        site_i[e] = U[row + col * rows];
    }
    // site_j ← scale·diag(σ_kept)·Vᴴ: row-major (chi × cols).
    // site_j[t*cols + c] = scale·σ[idx]·conj(V[c + idx*cols]).
    uint nj = chi * cols;
    for (uint e = tid; e < nj; e += tcnt) {
        uint t = e / cols;
        uint c = e % cols;
        uint col = idxs[t];
        float s = sig[col] * scale;
        float2 vc = cconj(V[c + col * cols]);
        site_j[e] = float2(vc.x * s, vc.y * s);
    }
}
