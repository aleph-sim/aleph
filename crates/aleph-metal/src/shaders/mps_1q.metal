#include <metal_stdlib>
using namespace metal;

// Uniform for a 1q gate applied to one MPS site tensor. Field order/offsets MUST
// match the Rust `Mps1q` struct (see mps/kernel.rs). 4×float2 (row-major 2×2)
// then `right` and a u32 pad => 40 bytes, no internal padding.
//   m[4]  : row-major 2×2, m[0]=u00 m[1]=u01 m[2]=u10 m[3]=u11
//   right : the site's right bond dimension (stride for the physical index)
struct Mps1q {
    float2 m[4];
    uint   right;
    uint   _pad;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// One thread per (l, r) pair of the site's (left, 2, right) tensor; each updates
// the p=0 / p=1 amplitudes at that (l, r). Grid = left·right. Mirrors the CPU
// MPS `apply_1q`: data[(l*2+p)*right + r].
kernel void apply_1q_site(device float2*  data [[buffer(0)]],
                          constant Mps1q&  g    [[buffer(1)]],
                          uint tid              [[thread_position_in_grid]]) {
    uint right = g.right;
    uint l = tid / right;
    uint r = tid % right;
    uint i0 = (l * 2u + 0u) * right + r;
    uint i1 = (l * 2u + 1u) * right + r;
    float2 a0 = data[i0];
    float2 a1 = data[i1];
    data[i0] = cmul(g.m[0], a0) + cmul(g.m[1], a1);
    data[i1] = cmul(g.m[2], a0) + cmul(g.m[3], a1);
}
