#include <metal_stdlib>
using namespace metal;

// Uniforms for the GPU-resident centre-move install kernels (P5.8-04). MUST match
// the Rust `QrInstallMeta` struct (mps/kernel.rs): six used uints + two pad = 32 B.
//   rows  : grouped rows of the QR block (right move: li·2; left move: 2·ri)
//   mid   : shared bond that R contracts over (right: site i right; left: li)
//   size  : kept bond = min(rows, mid) (= QR `size`)
//   nbr   : neighbour's outer bond (right: site i+1 right `nr`; left: site i-1 left `pl`)
//   phys  : physical leg of the *neighbour* group (right: nr; left: pl·2 handled via rows)
//   _flag : unused here
struct QrInstallMeta {
    uint rows;
    uint mid;
    uint size;
    uint nbr;
    uint phys;
    uint _f0;
    uint _f1;
    uint _f2;
};

inline float2 cmuli(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cconji(float2 z) { return float2(z.x, -z.y); }

// LEFT move pack: grouped-right adjoint of site i into column-major `a` for the QR.
//   A = GRᴴ, (cols × li) col-major, cols = 2·ri:
//   a[(p*ri+r) + l*cols] = conj(GR[l, p*ri+r]) = conj(site_i[(l*2+p)*ri + r]).
//   meta: rows = cols (=2·ri), mid = li, phys = ri.  Grid = cols·li.
kernel void qr_pack_gr_adj(device const float2* site_i [[buffer(0)]],
                           device float2*      a       [[buffer(1)]],
                           constant QrInstallMeta& g   [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
    uint cols = g.rows, li = g.mid, ri = g.phys;
    if (tid >= cols * li) return;
    uint pc = tid % cols; // = p*ri + r
    uint l = tid / cols;
    uint p = pc / ri;
    uint r = pc % ri;
    a[pc + l * cols] = cconji(site_i[(l * 2u + p) * ri + r]);
}

// RIGHT move, site i ← Q (left-canonical), reshaped (li, 2, size) row-major:
//   site_i[row*size + t] = Q[row + t*rows]   (Q column-major rows×size).
// Grid = rows·size.
kernel void qr_install_q_right(device const float2* q     [[buffer(0)]],
                               device float2*      site_i [[buffer(1)]],
                               constant QrInstallMeta& g  [[buffer(2)]],
                               uint tid [[thread_position_in_grid]]) {
    uint rows = g.rows, size = g.size;
    if (tid >= rows * size) return;
    uint row = tid / size;
    uint t = tid % size;
    site_i[row * size + t] = q[row + t * rows];
}

// RIGHT move, absorb R into site i+1: new site (size, 2, nr) row-major,
//   site_j[(t*2+p)*nr + r] = Σ_c R[t + c*size] · GR_{i+1}[c, p*nr+r],
//   GR_{i+1}[c, p*nr+r] = A_{i+1}[(c*2+p)*nr + r].   (mid = old shared bond.)
// Grid = size·2·nr.
kernel void qr_absorb_right(device const float2* r      [[buffer(0)]],
                            device const float2* a_j    [[buffer(1)]],
                            device float2*      site_j  [[buffer(2)]],
                            constant QrInstallMeta& g    [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    uint mid = g.mid, size = g.size, nr = g.nbr;
    if (tid >= size * 2u * nr) return;
    uint t = tid / (2u * nr);
    uint pr = tid % (2u * nr);
    uint p = pr / nr;
    uint rr = pr % nr;
    float2 acc = float2(0.0f, 0.0f);
    for (uint c = 0u; c < mid; ++c) {
        acc += cmuli(r[t + c * size], a_j[(c * 2u + p) * nr + rr]);
    }
    site_j[(t * 2u + p) * nr + rr] = acc;
}

// LEFT move, site i ← Q_aᴴ (right-canonical), reshaped (size, 2, ri) row-major:
//   site_i[(t*2+p)*ri + r] = conj(Q_a[p*ri+r, t]) = conj(q[(p*ri+r) + t*cols]),
// where cols = 2·ri = `rows` of the QR block. Grid = size·2·ri.
kernel void qr_install_q_left(device const float2* q     [[buffer(0)]],
                              device float2*      site_i [[buffer(1)]],
                              constant QrInstallMeta& g  [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
    uint cols = g.rows, size = g.size, ri = g.phys;
    if (tid >= size * 2u * ri) return;
    uint t = tid / (2u * ri);
    uint pr = tid % (2u * ri);
    uint p = pr / ri;
    uint rr = pr % ri;
    site_i[(t * 2u + p) * ri + rr] = cconji(q[(p * ri + rr) + t * cols]);
}

// LEFT move, absorb Rᴴ into site i-1: new grouped-left (pl·2, size) → site (pl,2,size):
//   new_h[row*size + t] = Σ_c GL_{i-1}[row, c] · conj(R_a[t + c*size]),
//   GL_{i-1}[row, c] = A_{i-1}[row*li + c].   (mid = li = R contraction length.)
// `rows` here = pl·2 (the grouped-left row count). Grid = rows·size.
kernel void qr_absorb_left(device const float2* r      [[buffer(0)]],
                           device const float2* a_h    [[buffer(1)]],
                           device float2*      site_h  [[buffer(2)]],
                           constant QrInstallMeta& g    [[buffer(3)]],
                           uint tid [[thread_position_in_grid]]) {
    uint rows = g.rows, mid = g.mid, size = g.size;
    if (tid >= rows * size) return;
    uint row = tid / size;
    uint t = tid % size;
    float2 acc = float2(0.0f, 0.0f);
    for (uint c = 0u; c < mid; ++c) {
        acc += cmuli(a_h[row * mid + c], cconji(r[t + c * size]));
    }
    site_h[row * size + t] = acc;
}
