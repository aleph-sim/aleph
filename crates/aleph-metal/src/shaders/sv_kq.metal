#include <metal_stdlib>
using namespace metal;

// Uniform for a generic dense k-qubit gate. Field order/offsets MUST match the
// Rust `GateKqMeta` struct (see sv/kernel.rs). All-uint => 48 bytes, no padding.
//   k         : number of target qubits (2..5; 1 also valid but 1q uses apply_1q)
//   sorted[j] : target bit positions ASCENDING, for zero-bit insertion
//   tbit[j]   : 1u<<q[j] in LOGICAL/MSB order (q[0] is the matrix-index MSB)
//   ctrl_mask : external-control mask (0 when none)
struct GateKqMeta {
    uint k;
    uint sorted[5];
    uint tbit[5];
    uint ctrl_mask;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// Grid = 2^(n-k). One thread per group of 2^k amplitudes. `mat` is row-major
// 2^k x 2^k (M[r*dim + c]). dim <= 32 (k <= 5), so the thread-local arrays fit.
kernel void apply_kq(device float2*       amps [[buffer(0)]],
                     device const float2* mat  [[buffer(1)]],
                     constant GateKqMeta&  g    [[buffer(2)]],
                     uint tid                   [[thread_position_in_grid]]) {
    uint k = g.k;
    uint dim = 1u << k;

    // Reconstruct the base index (all target bits clear) by inserting k zero
    // bits at the ascending sorted positions. Ascending order is required so
    // each insertion shifts the still-to-come higher slots up correctly.
    uint base = tid;
    for (uint j = 0u; j < k; ++j) {
        uint p = g.sorted[j];
        uint mask = (1u << p) - 1u;
        base = ((base & ~mask) << 1) | (base & mask);
    }

    // No-op unless all external controls are set (always true when ctrl_mask==0).
    if ((base & g.ctrl_mask) != g.ctrl_mask) {
        return;
    }

    // Global index of each local matrix index l: bit (k-1-j) of l -> tbit[j].
    uint   gidx[32];
    float2 v[32];
    for (uint l = 0u; l < dim; ++l) {
        uint off = 0u;
        for (uint j = 0u; j < k; ++j) {
            if (((l >> (k - 1u - j)) & 1u) != 0u) {
                off |= g.tbit[j];
            }
        }
        gidx[l] = base | off;          // base's target bits are clear => | == +
        v[l] = amps[gidx[l]];
    }

    // Dense mat-vec; read all inputs above before any write (in-place safe).
    for (uint r = 0u; r < dim; ++r) {
        float2 acc = float2(0.0, 0.0);
        for (uint c = 0u; c < dim; ++c) {
            acc += cmul(mat[r * dim + c], v[c]);
        }
        amps[gidx[r]] = acc;
    }
}
