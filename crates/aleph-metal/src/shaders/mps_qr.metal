#include <metal_stdlib>
using namespace metal;

// Uniform for the GPU Householder thin-QR kernel (P5.8-04). MUST match the Rust
// `QrMeta` struct (mps/kernel.rs): two used uints + two pad => 16 bytes.
//   m, n : the block A shape (column-major m×n). size = min(m, n).
struct QrMeta {
    uint m;
    uint n;
    uint _pad0;
    uint _pad1;
};

constant uint QR_THREADS = 256u;
// Largest n the kernel handles (β cache); n ≤ χ_max. 1024 ⇒ χ_max ≤ 1024.
constant uint QR_MAXN = 1024u;

inline float2 cmulq(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cconjq(float2 z) { return float2(z.x, -z.y); }

// Thin Householder QR of a column-major `m × n` block `A` (`A[r + c*m]`), one
// threadgroup. On return:
//   * `q`  = Q, column-major `m × size` (orthonormal columns, size = min(m,n)),
//   * `r`  = R, column-major `size × n` (upper-triangular; zeros below the diagonal),
// so A = Q·R. `a` is overwritten with the Householder reflectors (working storage).
//
// Golub & Van Loan §5.2 (Householder QR), complex variant: reflector
// H_j = I − β_j v_j v_jᴴ with v_j[j] chosen so H_j zeroes A[j+1:, j], β_j = 2/‖v_j‖².
// Factorisation is `size` sequential reflector steps (each a row-reduction to form
// the reflector, then a column-parallel apply); Q is then formed by applying the
// reflectors in reverse to I. f32 throughout; the host keeps an f64 SVD fallback.
kernel void householder_qr(device float2*      a   [[buffer(0)]],
                           device float2*      q   [[buffer(1)]],
                           device float2*      r   [[buffer(2)]],
                           constant QrMeta&    g   [[buffer(3)]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint tcnt [[threads_per_threadgroup]]) {
    uint m = g.m, n = g.n;
    uint size = min(m, n);

    threadgroup float red[QR_THREADS];
    threadgroup float2 sh;     // broadcast v0 / scalars
    threadgroup float  sh_beta;
    threadgroup float  betas[QR_MAXN];

    // R starts as zero (we fill the upper triangle); clear it.
    for (uint e = tid; e < size * n; e += tcnt) r[e] = float2(0.0f, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 1: factorise. Column j's reflector zeroes A[j+1:, j]. ---
    for (uint j = 0u; j < size; ++j) {
        // ‖A[j:, j]‖² via a strided row reduction.
        float part = 0.0f;
        for (uint rr = j + tid; rr < m; rr += tcnt) {
            float2 z = a[rr + j * m];
            part += z.x * z.x + z.y * z.y;
        }
        red[tid] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float s2 = 0.0f;
            for (uint t = 0u; t < tcnt; ++t) s2 += red[t];
            float sigma = sqrt(s2);
            float2 x0 = a[j + j * m];
            float x0n = length(x0);
            // α = −e^{i·arg(x0)}·σ (points opposite x0 to avoid cancellation).
            float2 phase = (x0n > 0.0f) ? (x0 / x0n) : float2(1.0f, 0.0f);
            float2 alpha = -phase * sigma;
            // v0 = x0 − α; ‖v‖² = |v0|² + (σ² − |x0|²).
            float2 v0 = x0 - alpha;
            float vtv = (v0.x * v0.x + v0.y * v0.y) + (s2 - x0n * x0n);
            sh = v0;
            sh_beta = (vtv > 0.0f) ? (2.0f / vtv) : 0.0f;
            betas[j] = sh_beta;
            a[j + j * m] = v0;        // store the reflector's head in A
            r[j + j * size] = alpha;  // R[j,j] = α
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 v0 = sh;
        float beta = sh_beta;
        // Apply H_j to columns c > j, one column per striding thread.
        for (uint c = j + 1u + tid; c < n; c += tcnt) {
            // w = v_jᴴ · A[j:, c]  (v_j[j]=v0, v_j[r>j]=a[r+jm]).
            float2 w = cmulq(cconjq(v0), a[j + c * m]);
            for (uint rr = j + 1u; rr < m; ++rr) {
                w += cmulq(cconjq(a[rr + j * m]), a[rr + c * m]);
            }
            w = beta * w;
            // A[j:, c] -= w · v_j ; the head becomes R[j,c].
            float2 head = a[j + c * m] - cmulq(w, v0);
            a[j + c * m] = head;
            r[j + c * size] = head;
            for (uint rr = j + 1u; rr < m; ++rr) {
                a[rr + c * m] -= cmulq(w, a[rr + j * m]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 2: Q = H_0 … H_{size-1} · I_{m×size}, applied in reverse. ---
    for (uint e = tid; e < m * size; e += tcnt) {
        uint rr = e % m, cc = e / m;
        q[e] = (rr == cc) ? float2(1.0f, 0.0f) : float2(0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int jj = int(size) - 1; jj >= 0; --jj) {
        uint j = uint(jj);
        float beta = betas[j];
        float2 v0 = a[j + j * m]; // reflector head
        // Apply H_j to each column of Q (size columns), one per striding thread.
        for (uint c = tid; c < size; c += tcnt) {
            float2 w = cmulq(cconjq(v0), q[j + c * m]);
            for (uint rr = j + 1u; rr < m; ++rr) {
                w += cmulq(cconjq(a[rr + j * m]), q[rr + c * m]);
            }
            w = beta * w;
            q[j + c * m] -= cmulq(w, v0);
            for (uint rr = j + 1u; rr < m; ++rr) {
                q[rr + c * m] -= cmulq(w, a[rr + j * m]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
