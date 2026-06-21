#include <metal_stdlib>
using namespace metal;

// Uniform for the single-block one-sided Jacobi SVD kernel. MUST match the Rust
// `JacobiMeta` struct (mps/kernel.rs): two used uints + two pad => 16 bytes.
//   m : rows of A (the HOST guarantees m >= n; wide inputs are factored as Aᴴ)
//   n : cols of A (= number of singular values k)
struct JacobiMeta {
    uint m;
    uint n;
    uint _pad0;
    uint _pad1;
};

// Per-block descriptor for the batched kernel. MUST match the Rust
// `JacobiBlockMeta` struct (mps/kernel.rs): five used uints + three pad => 32
// bytes (the array stride the host packs into `buffer(3)`).
//   m, n   : block dims as in `JacobiMeta` (host guarantees m >= n per block)
//   a_off  : float2 offset of this block's A inside the packed `A` buffer
//   v_off  : float2 offset of this block's V inside the packed `V` buffer
//   sig_off: float  offset of this block's σ inside the packed `sig` buffer
struct JacobiBlockMeta {
    uint m;
    uint n;
    uint a_off;
    uint v_off;
    uint sig_off;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Threadgroup size cap. MUST be a power of two (the reduction halves `tcount`)
// and MUST equal the `red[]` length below. The host dispatches a power-of-two
// threadgroup size <= this. The same cap bounds both entry points so a single
// `red[]` allocation serves the single-block and batched kernels.
constant uint JACOBI_THREADS = 256u;
// Sweep cap mirroring the CPU reference. Well-conditioned small blocks converge
// in <=8; the cap only guards a pathological non-convergence.
constant uint JACOBI_MAX_SWEEPS = 60u;
// f32 relative off-diagonal threshold: a pair (p,q) is left alone once
// |⟨A_p,A_q⟩| <= TOL·‖A_p‖·‖A_q‖. Looser than the f64 CPU reference (1e-14)
// because the state is single precision; still far below the 1e-5 oracle bar.
constant float JACOBI_TOL = 1e-6f;

// One-sided Jacobi thin SVD of one COLUMN-MAJOR block A (m x n, m >= n):
// A[i + t*m] is row i, column t. Shared by the single-block and batched entry
// points — both bind A/V/sig device pointers already offset to their block and
// pass the threadgroup scratch the reduction needs. On return:
//   * A holds U (column-major, m x n): U[:,t] = A[:,t] after normalization,
//   * V (n x n column-major) holds the right singular vectors (caller seeds I),
//   * sig[t] = σ_t (the converged column norm).
//
// One threadgroup factors the whole block; threads stride the row dimension.
// The columns stay in device memory — only the per-pair 2x2 reduction and the
// broadcast rotation scalars live in the threadgroup scratch.
//
// Golub & Van Loan, *Matrix Computations* 4th ed. §8.5 (Jacobi SVD). The complex
// 2x2 column-Gram is real-symmetrized by a diag(1, e^{-iφ}) phase pre-rotation
// (φ = arg of the column inner product) before the standard real Jacobi angle.
// Structure mirrors the CPU reference in mps/jacobi.rs so the kernel is validated
// against it.
static void jacobi_block(device float2* A,
                         device float2* V,
                         device float*  sig,
                         uint m,
                         uint n,
                         uint tid,
                         uint tcount,
                         threadgroup float4* red,    // [JACOBI_THREADS] partials
                         threadgroup float*  params, // [6] cs,sn,es_re,es_im,ec_re,ec_im
                         threadgroup int*    p_rot,  // this pair rotates?
                         threadgroup int*    any_rot // any pair rotated this sweep?
) {
    for (uint sweep = 0u; sweep < JACOBI_MAX_SWEEPS; ++sweep) {
        if (tid == 0u) {
            *any_rot = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0u; p < n; ++p) {
            for (uint q = p + 1u; q < n; ++q) {
                // --- 2x2 column Gram reduced over the rows ---
                float app = 0.0f, bqq = 0.0f, gre = 0.0f, gim = 0.0f;
                for (uint i = tid; i < m; i += tcount) {
                    float2 a = A[i + p * m];
                    float2 b = A[i + q * m];
                    app += a.x * a.x + a.y * a.y;
                    bqq += b.x * b.x + b.y * b.y;
                    gre += a.x * b.x + a.y * b.y; // Re conj(a)·b
                    gim += a.x * b.y - a.y * b.x; // Im conj(a)·b
                }
                red[tid] = float4(app, bqq, gre, gim);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = tcount >> 1; s > 0u; s >>= 1) {
                    if (tid < s) {
                        red[tid] += red[tid + s];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (tid == 0u) {
                    float alpha = red[0].x, beta = red[0].y;
                    float grr = red[0].z, gii = red[0].w;
                    float gabs = sqrt(grr * grr + gii * gii);
                    float scale = sqrt(alpha) * sqrt(beta);
                    if (alpha > 0.0f && beta > 0.0f && gabs > JACOBI_TOL * scale) {
                        float inv = 1.0f / gabs;
                        float ephi_re = grr * inv;  //  cos φ
                        float ephi_im = -gii * inv; // -sin φ  ⇒ e^{-iφ}
                        float tau = (alpha - beta) / (2.0f * gabs);
                        float t = (tau >= 0.0f)
                                      ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                                      : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                        float cs = 1.0f / sqrt(1.0f + t * t);
                        float sn = t * cs;
                        params[0] = cs;
                        params[1] = sn;
                        params[2] = ephi_re * sn; // es_re
                        params[3] = ephi_im * sn; // es_im
                        params[4] = ephi_re * cs; // ec_re
                        params[5] = ephi_im * cs; // ec_im
                        *p_rot = 1;
                        *any_rot = 1;
                    } else {
                        *p_rot = 0;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (*p_rot == 1) {
                    float cs = params[0], sn = params[1];
                    float es_re = params[2], es_im = params[3];
                    float ec_re = params[4], ec_im = params[5];
                    // new_p = cs·col_p + (e^{-iφ}sn)·col_q
                    // new_q = -sn·col_p + (e^{-iφ}cs)·col_q
                    for (uint i = tid; i < m; i += tcount) {
                        float2 ap = A[i + p * m];
                        float2 aq = A[i + q * m];
                        A[i + p * m] = float2(cs * ap.x + (es_re * aq.x - es_im * aq.y),
                                              cs * ap.y + (es_re * aq.y + es_im * aq.x));
                        A[i + q * m] = float2(-sn * ap.x + (ec_re * aq.x - ec_im * aq.y),
                                              -sn * ap.y + (ec_re * aq.y + ec_im * aq.x));
                    }
                    for (uint i = tid; i < n; i += tcount) {
                        float2 vp = V[i + p * n];
                        float2 vq = V[i + q * n];
                        V[i + p * n] = float2(cs * vp.x + (es_re * vq.x - es_im * vq.y),
                                              cs * vp.y + (es_re * vq.y + es_im * vq.x));
                        V[i + q * n] = float2(-sn * vp.x + (ec_re * vq.x - ec_im * vq.y),
                                              -sn * vp.y + (ec_re * vq.y + ec_im * vq.x));
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (*any_rot == 0) {
            break; // uniform: *any_rot is threadgroup-shared
        }
    }

    // Finalize: σ_t = ‖A_:,t‖, then U_:,t = A_:,t / σ_t (in place).
    for (uint t = 0u; t < n; ++t) {
        float partial = 0.0f;
        for (uint i = tid; i < m; i += tcount) {
            float2 a = A[i + t * m];
            partial += a.x * a.x + a.y * a.y;
        }
        red[tid] = float4(partial, 0.0f, 0.0f, 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tcount >> 1; s > 0u; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sigma = sqrt(red[0].x);
        if (tid == 0u) {
            sig[t] = sigma;
        }
        float inv = (sigma > 0.0f) ? 1.0f / sigma : 0.0f;
        for (uint i = tid; i < m; i += tcount) {
            float2 a = A[i + t * m];
            A[i + t * m] = float2(a.x * inv, a.y * inv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Single-block entry point (P5.7-02): one threadgroup factors the whole block.
kernel void jacobi_svd(device float2*       A   [[buffer(0)]],
                       device float2*       V   [[buffer(1)]],
                       device float*        sig [[buffer(2)]],
                       constant JacobiMeta& g   [[buffer(3)]],
                       uint tid                 [[thread_position_in_threadgroup]],
                       uint tcount              [[threads_per_threadgroup]]) {
    threadgroup float4 red[JACOBI_THREADS]; // (α, β, γ_re, γ_im) partials
    threadgroup float  params[6];           // cs, sn, es_re, es_im, ec_re, ec_im
    threadgroup int    p_rot;               // this pair rotates?
    threadgroup int    any_rot;             // any pair rotated this sweep?
    jacobi_block(A, V, sig, g.m, g.n, tid, tcount, red, params, &p_rot, &any_rot);
}

// Batched entry point (P5.7-04): grid = num_blocks threadgroups, one per block.
// Each threadgroup reads its descriptor by `threadgroup_position_in_grid`, then
// factors its slice of the packed A/V/sig buffers. All blocks of a brickwall
// layer act on disjoint site pairs, so a single dispatch (one commit/wait)
// replaces the per-gate launch + sync that dominated the GPU-resident split.
kernel void jacobi_svd_batched(device float2*               A     [[buffer(0)]],
                               device float2*               V     [[buffer(1)]],
                               device float*                sig   [[buffer(2)]],
                               constant JacobiBlockMeta*    metas [[buffer(3)]],
                               uint tgid  [[threadgroup_position_in_grid]],
                               uint tid   [[thread_position_in_threadgroup]],
                               uint tcount [[threads_per_threadgroup]]) {
    threadgroup float4 red[JACOBI_THREADS];
    threadgroup float  params[6];
    threadgroup int    p_rot;
    threadgroup int    any_rot;
    JacobiBlockMeta bm = metas[tgid];
    jacobi_block(A + bm.a_off, V + bm.v_off, sig + bm.sig_off, bm.m, bm.n,
                 tid, tcount, red, params, &p_rot, &any_rot);
}
