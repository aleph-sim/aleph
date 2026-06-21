#include <metal_stdlib>
using namespace metal;

// Uniform for the GPU column-major pack (P5.8-03). MUST match the Rust `PackMeta`
// struct (mps/kernel.rs): two used uints + a flag + pad => 16 bytes.
//   rows, cols : the gated two-site block Θ′ shape (row-major, rows × cols)
//   wide       : 1 when rows < cols — Θ′ is factored as its adjoint Aᴴ (so the
//                Jacobi kernel always sees a tall m ≥ n block), else 0.
struct PackMeta {
    uint rows;
    uint cols;
    uint wide;
    uint _pad0;
};

// Pack the row-major Θ′ into the **column-major** `A` the one-sided Jacobi kernel
// consumes, entirely on the GPU — so Θ′ never round-trips to the host between the
// contraction and the SVD (the P5.7 audit's GPU→host→GPU bounce). One thread per
// output entry; grid = rows·cols (= m·n either orientation).
//   tall (wide=0): m=rows, n=cols.  A[i + t*m] = Θ[i*cols + t]
//   wide (wide=1): m=cols, n=rows.  A[i + t*m] = conj(Θ[t*cols + i])
// Mirrors the host pack in `gpu_jacobi::gpu_thin_svd` it replaces.
kernel void pack_theta(device const float2* theta [[buffer(0)]],
                       device float2*       a     [[buffer(1)]],
                       constant PackMeta&   g     [[buffer(2)]],
                       uint tid                   [[thread_position_in_grid]]) {
    uint rows = g.rows;
    uint cols = g.cols;
    uint m = (g.wide != 0u) ? cols : rows;
    uint n = (g.wide != 0u) ? rows : cols;
    if (tid >= m * n) {
        return;
    }
    uint i = tid % m; // row of A (column-major)
    uint t = tid / m; // column of A
    if (g.wide != 0u) {
        float2 z = theta[t * cols + i];
        a[i + t * m] = float2(z.x, -z.y); // conj
    } else {
        a[i + t * m] = theta[i * cols + t];
    }
}
