#include <metal_stdlib>
using namespace metal;

// Uniform block for one single-qubit gate. Field order/offsets MUST match the
// Rust `Gate1q` struct (see sv/kernel.rs). `m` is the 2x2 matrix row-major:
// m[0]=m00, m[1]=m01, m[2]=m10, m[3]=m11.
struct Gate1q {
    float2 m[4];
    uint   target;     // target qubit index
    uint   t_bit;      // 1u << target
    uint   ctrl_mask;  // control mask (0 for plain 1q gates; forward-compat)
    uint   _pad;       // pad to 48 bytes (matches Metal's 8-byte struct align)
};

// Complex multiply on float2 (re, im).
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// One thread per amplitude PAIR. Grid size = 2^(n-1). `tid` indexes pairs;
// insert a 0 bit at position `target` to recover the base index `i` (target
// bit clear), then `j = i | t_bit` is its partner.
kernel void apply_1q(device float2* amps   [[buffer(0)]],
                     constant Gate1q& g     [[buffer(1)]],
                     uint tid               [[thread_position_in_grid]]) {
    // `target + 1` is a valid 32-bit shift because the host caps allocation at
    // MAX_METAL_QUBITS = 28, so target ≤ 27 and the shift amount ≤ 28 < 32
    // (a `<< 32` would be MSL-undefined).
    uint lo = tid & (g.t_bit - 1u);
    uint hi = (tid >> g.target) << (g.target + 1u);
    uint i  = hi | lo;
    // No-op when controls are not all set (always applies when ctrl_mask == 0).
    if ((i & g.ctrl_mask) != g.ctrl_mask) {
        return;
    }
    uint j  = i | g.t_bit;
    float2 a = amps[i];
    float2 b = amps[j];
    amps[i] = cmul(g.m[0], a) + cmul(g.m[1], b);
    amps[j] = cmul(g.m[2], a) + cmul(g.m[3], b);
}
