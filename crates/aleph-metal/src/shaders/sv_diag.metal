#include <metal_stdlib>
using namespace metal;

// Per-term descriptor; field order MUST match Rust `DiagTermDesc` (16 bytes).
struct DiagTermDesc {
    uint  cond_offset;  // start in cond_masks[]
    uint  n_conds;      // number of parity conditions (0 == global phase)
    float angle;        // radians added when all conditions fire
    uint  _pad;
};

// Uniform; matches Rust `DiagMeta`.
struct DiagMeta {
    uint n_terms;
};

// One thread per amplitude index x. phi(x) = sum of term angles whose parity
// conditions all fire: parity(mask & x) is odd for every mask in the term.
// Masks are <=28-bit (MAX_METAL_QUBITS=28), so `uint` is exact.
kernel void apply_diagonal_phase(device float2*             amps       [[buffer(0)]],
                                 device const uint*         cond_masks [[buffer(1)]],
                                 device const DiagTermDesc* terms      [[buffer(2)]],
                                 constant DiagMeta&         meta       [[buffer(3)]],
                                 uint x                                [[thread_position_in_grid]]) {
    float phi = 0.0;
    for (uint t = 0u; t < meta.n_terms; ++t) {
        DiagTermDesc d = terms[t];
        bool all_fire = true;
        for (uint c = 0u; c < d.n_conds; ++c) {
            uint m = cond_masks[d.cond_offset + c];
            if ((popcount(m & x) & 1u) == 0u) {
                all_fire = false;
                break;
            }
        }
        if (all_fire) {
            phi += d.angle;
        }
    }
    float2 a = amps[x];
    float cs = cos(phi);
    float sn = sin(phi);
    amps[x] = float2(a.x * cs - a.y * sn, a.x * sn + a.y * cs);
}
