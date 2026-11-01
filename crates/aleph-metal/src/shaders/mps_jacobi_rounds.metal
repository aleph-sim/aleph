#include <metal_stdlib>
using namespace metal;
// P8-11 research-only round-robin one-sided Jacobi. Same complex rotation
// as mps_jacobi.metal (Golub & Van Loan, Matrix Computations, §8.5).
// Each group owns disjoint columns in this round. Separate compute encoders
// provide device-buffer dependencies between rounds; no cross-group spin barrier.
struct RoundMeta { uint m; uint n; uint round; uint padded; };
constant uint JACOBI_THREADS = 256u;
constant float JACOBI_TOL = 1e-6f;
kernel void jacobi_round(device float2* A [[buffer(0)]],
 device float2* V [[buffer(1)]], device atomic_uint* changes [[buffer(2)]],
 constant RoundMeta& g [[buffer(3)]], uint pair [[threadgroup_position_in_grid]],
 uint tid [[thread_position_in_threadgroup]], uint tcount [[threads_per_threadgroup]]) {
 uint m=g.m, n=g.n;
 uint x=pair, y=g.padded-1u-pair;
 uint p=x==g.padded-1u ? x : (x+g.round)%(g.padded-1u);
 uint q=y==g.padded-1u ? y : (y+g.round)%(g.padded-1u);
 if(p>=n || q>=n) return; // uniform dummy-column bye for odd n
 threadgroup float4 red[JACOBI_THREADS];
 threadgroup float params[6];
 threadgroup int rotate;
 threadgroup int* p_rot=&rotate;
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
                        atomic_fetch_add_explicit(changes, 1u, memory_order_relaxed);
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
kernel void jacobi_normalize(device float2* A [[buffer(0)]],
 device float* sig [[buffer(1)]], constant RoundMeta& g [[buffer(2)]],
 uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
 uint tcount [[threads_per_threadgroup]]) {
 uint m=g.m;
 threadgroup float4 red[JACOBI_THREADS];
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
}
