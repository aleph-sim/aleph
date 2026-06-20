#include <metal_stdlib>
using namespace metal;

// Uniform for the two-site contraction. MUST match the Rust `ContractMeta`
// struct (mps/kernel.rs): two used uints + two pad => 16 bytes.
//   c  : shared bond dimension (A.right == B.left), the summation length
//   ri : right bond of site B (column stride for the physical index p')
struct ContractMeta {
    uint c;
    uint ri;
    uint _pad0;
    uint _pad1;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// Θ = A · B as a row-major (la·2) × (2·ri) matrix, one thread per output entry.
//   A : site i, shape (la, 2, c), row-major a[(la_i*2+p)*c + cc]
//   B : site j, shape (c, 2, ri), row-major b[(cc*2+p')*ri + rj]
//   Θ[row][col] with row = la_i*2+p, col = p'*ri+rj, cols = 2·ri:
//     Θ[row*cols+col] = Σ_cc A[row*c+cc] · B[(cc*2+p')*ri+rj]
// Mirrors the CPU MPS group_left_view × group_right_view matmul. Grid = rows·cols.
kernel void contract_2site(device const float2* A     [[buffer(0)]],
                           device const float2* B     [[buffer(1)]],
                           device float2*       theta [[buffer(2)]],
                           constant ContractMeta& g   [[buffer(3)]],
                           uint tid                   [[thread_position_in_grid]]) {
    uint c = g.c;
    uint ri = g.ri;
    uint cols = 2u * ri;
    uint row = tid / cols;
    uint col = tid % cols;
    uint pp = col / ri;  // physical index of site j
    uint rj = col % ri;  // right bond of site j
    float2 acc = float2(0.0, 0.0);
    for (uint cc = 0u; cc < c; ++cc) {
        float2 a = A[row * c + cc];
        float2 b = B[(cc * 2u + pp) * ri + rj];
        acc += cmul(a, b);
    }
    theta[row * cols + col] = acc;
}
