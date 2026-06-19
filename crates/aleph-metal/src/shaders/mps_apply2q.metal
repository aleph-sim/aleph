#include <metal_stdlib>
using namespace metal;

// Uniform for the 2q gate-apply on Θ. MUST match the Rust `Apply2qMeta` struct
// (mps/kernel.rs): two used uints + two pad => 16 bytes.
//   ri        : right bond of site j (column block stride)
//   i_is_msb  : 1 when the left site (i) holds the matrix MSB qubit (qa==i),
//               0 when the right site (j) does. Selects the physical→matrix-index
//               map `out()` (CPU MPS convention, ADR-0004).
struct Apply2qMeta {
    uint ri;
    uint i_is_msb;
    uint _pad0;
    uint _pad1;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// Θ' = U·Θ, in place. `mat` is the row-major 4×4 gate in qa-MSB/qb-LSB order.
// One thread per (l, r): gathers its four entries Θ[l, (a,b), r] indexed by the
// matrix index cc = out(phys_i, phys_j), applies the 4×4, scatters back. Each
// thread owns a disjoint set of four cells, so the in-place gather→scatter is
// race-free. Grid = la·ri. cols = 2·ri; cell (l,a,b,r) lives at
//   (l*2+pi)*cols + (pj*ri + r), where (pi,pj) = decode(cc) per i_is_msb.
kernel void apply_2q_theta(device float2*       theta [[buffer(0)]],
                           device const float2* mat   [[buffer(1)]],
                           constant Apply2qMeta& g     [[buffer(2)]],
                           uint tid                    [[thread_position_in_grid]]) {
    uint ri = g.ri;
    uint cols = 2u * ri;
    uint l = tid / ri;
    uint r = tid % ri;

    uint   idx[4];
    float2 v[4];
    for (uint cc = 0u; cc < 4u; ++cc) {
        uint pi, pj;
        if (g.i_is_msb != 0u) {
            pi = cc >> 1;
            pj = cc & 1u;
        } else {
            pj = cc >> 1;
            pi = cc & 1u;
        }
        uint index = (l * 2u + pi) * cols + (pj * ri + r);
        idx[cc] = index;
        v[cc] = theta[index];
    }

    for (uint rr = 0u; rr < 4u; ++rr) {
        float2 acc = float2(0.0, 0.0);
        for (uint d = 0u; d < 4u; ++d) {
            acc += cmul(mat[rr * 4u + d], v[d]);
        }
        theta[idx[rr]] = acc;
    }
}
