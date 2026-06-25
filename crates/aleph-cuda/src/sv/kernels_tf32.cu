// P5.11-05 Tensor-core (TF32) fused-block matvec.
//
// P5.10-01 established that the wall past k=3 fusion is the **O(4^k) dense
// matvec compute** itself, not register spill — the warp-tiled `apply_kq_tiled`
// removed the spill yet k=4,5 still lose to k=3. The Ada card's TF32 tensor
// cores (~8× the FP32 FMA rate) sit idle on that path. This file recasts the
// per-group `2^k × 2^k` complex matvec as a **batched GEMM** `M · V` and runs it
// on the tensor cores: one warp multiplies the gate matrix `M` (dim×dim) by a
// tile of `WMMA_N = 16` group vectors stacked as columns of `V` (dim×16). With
// 16 columns per pass the tensor core runs near-full instead of the 1/16 a bare
// matvec would use.
//
// Compiled as a **separate NVRTC module** with `--gpu-architecture=sm_89` and the
// CUDA include path (mma.h) — the base FP32 module targets the NVRTC default
// arch, which predates TF32. Only k=4 (dim=16) and k=5 (dim=32) route here; k≤3
// already beats wider fusion on the warp-tiled kernel, and dim<16 wastes the
// 16-wide WMMA tile. Convention/layout identical to `kernels_f32.cu` (ADR 0004),
// so the same oracle suite pins this path.
//
// Complex GEMM via four real GEMMs: with M = Mr + i·Mi and V = Vr + i·Vi,
//   Re(out) = Mr·Vr − Mi·Vi,   Im(out) = Mr·Vi + Mi·Vr.
// TF32 truncates the 23-bit mantissa to 10 bits (~1e-3 relative per op), so the
// oracle budget for this path is 1e-4, not the 1e-5 of the FP32 ALU kernels.

#include <mma.h>
using namespace nvcuda;

struct cplx  { double re; double im; }; // matrix coefficient (matches f64 params)
struct cplxf { float  re; float  im; }; // amplitude — half the bytes of cplx

// Matches the Rust `GateKqParams` / CUDA `GateKq` (12×u32, 48 B).
struct GateKq {
    unsigned k;
    unsigned qbit[5];
    unsigned sorted[5];
    unsigned ctrl_mask;
};

// TF32 WMMA tile: 16×16 output, 8-deep contraction step (the native tf32 k).
#define WM 16
#define WN 16
#define WK 8

// Insert zero bits at the ascending `sorted` positions → base index of group
// `gid` with all k target bits clear. Identical scheme to `apply_kq_f32`.
__device__ __forceinline__ unsigned long long base_of(unsigned long long gid, const GateKq& g) {
    unsigned long long base = gid;
    for (unsigned j = 0; j < g.k; ++j) {
        unsigned p = g.sorted[j];
        unsigned long long mask = (1ULL << p) - 1ULL;
        base = ((base & ~mask) << 1) | (base & mask);
    }
    return base;
}

// Local matrix index `l` → global state-index offset (OR of the set target bits).
__device__ __forceinline__ unsigned long long off_of(unsigned l, const GateKq& g) {
    unsigned long long off = 0ULL;
    for (unsigned j = 0; j < g.k; ++j) {
        if ((l >> j) & 1u) off |= (unsigned long long)g.qbit[j];
    }
    return off;
}

// ---- k=4 (dim=16): one warp per tile of 16 groups, one 16×16 WMMA tile ----
#define K4_WARPS 4

extern "C" __global__
void apply_kq_tf32_k4(cplxf* amps, const cplx* mat, GateKq g, unsigned long long n_groups) {
    const unsigned dim = 16u;
    __shared__ float Mr[256], Mi[256];                 // gate matrix, cast to float once
    __shared__ float Vr[K4_WARPS][256], Vi[K4_WARPS][256]; // gathered group tiles (col-major)
    __shared__ float Or[K4_WARPS][256], Oi[K4_WARPS][256]; // GEMM output (row-major)
    // Per-tile index tables: the global amp index is base(group) | off(local), so
    // off depends only on the local row (dim values) and base only on the group
    // column (WN values) — precompute each once instead of per-element (P5.11-05).
    __shared__ unsigned long long Off[K4_WARPS][16], Base[K4_WARPS][16];

    // M is identical for every block; load + tf32-cast once, cooperatively.
    for (unsigned e = threadIdx.x; e < dim * dim; e += blockDim.x) {
        Mr[e] = (float)mat[e].re;
        Mi[e] = (float)mat[e].im;
    }
    __syncthreads();

    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long tile = (unsigned long long)blockIdx.x * K4_WARPS + warp;
    unsigned long long g0 = tile * WN;
    if (g0 >= n_groups) return; // whole warp exits uniformly (safe for WMMA collectives)

    // lanes 0..15 fill the off table; lanes 16..31 the base table (dim = WN = 16).
    unsigned long long* off = Off[warp];
    unsigned long long* bse = Base[warp];
    off[lane & 15u] = off_of(lane & 15u, g);
    if (lane >= 16u) bse[lane - 16u] = base_of(g0 + (lane - 16u), g);
    __syncwarp();

    // Gather V[c][t] = amp(group g0+t, local c) into col-major shared (ld = dim).
    float* vr = Vr[warp];
    float* vi = Vi[warp];
    for (unsigned e = lane; e < dim * WN; e += 32u) {
        unsigned c = e & (dim - 1u);
        unsigned t = e >> 4;
        if (g0 + t < n_groups) {
            cplxf a = amps[bse[t] | off[c]];
            vr[c + t * dim] = a.re;
            vi[c + t * dim] = a.im;
        } else {
            vr[c + t * dim] = 0.0f;
            vi[c + t * dim] = 0.0f;
        }
    }
    __syncwarp();

    // Accumulators: Re and Im of the 16×16 output tile.
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> accR, accI;
    wmma::fill_fragment(accR, 0.0f);
    wmma::fill_fragment(accI, 0.0f);

    // Two contraction steps (K=16 = 2·WK). For each: load Mr/Mi a-fragments
    // (row-major, ld=dim) and Vr/Vi b-fragments (col-major, ld=dim), tf32-round,
    // accumulate the four real products into Re/Im.
    for (unsigned ks = 0; ks < dim; ks += WK) {
        wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::tf32, wmma::row_major> aMr, aMi;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::tf32, wmma::col_major> bVr, bVi;
        wmma::load_matrix_sync(aMr, Mr + ks, dim);
        wmma::load_matrix_sync(aMi, Mi + ks, dim);
        wmma::load_matrix_sync(bVr, vr + ks, dim);
        wmma::load_matrix_sync(bVi, vi + ks, dim);
        for (int i = 0; i < aMr.num_elements; i++) aMr.x[i] = wmma::__float_to_tf32(aMr.x[i]);
        for (int i = 0; i < aMi.num_elements; i++) aMi.x[i] = wmma::__float_to_tf32(aMi.x[i]);
        for (int i = 0; i < bVr.num_elements; i++) bVr.x[i] = wmma::__float_to_tf32(bVr.x[i]);
        for (int i = 0; i < bVi.num_elements; i++) bVi.x[i] = wmma::__float_to_tf32(bVi.x[i]);
        wmma::mma_sync(accR, aMr, bVr, accR);   // + Mr·Vr
        wmma::mma_sync(accI, aMr, bVi, accI);   // + Mr·Vi
        wmma::mma_sync(accI, aMi, bVr, accI);   // + Mi·Vr
        // − Mi·Vi : negate the a-fragment in place, then accumulate into Re.
        for (int i = 0; i < aMi.num_elements; i++) aMi.x[i] = -aMi.x[i];
        wmma::mma_sync(accR, aMi, bVi, accR);   // − Mi·Vi
    }

    wmma::store_matrix_sync(Or[warp], accR, dim, wmma::mem_row_major);
    wmma::store_matrix_sync(Oi[warp], accI, dim, wmma::mem_row_major);
    __syncwarp();

    // Scatter out[m][t] back to amp(group g0+t, local m).
    float* orr = Or[warp];
    float* oii = Oi[warp];
    for (unsigned e = lane; e < dim * WN; e += 32u) {
        unsigned m = e & (dim - 1u);
        unsigned t = e >> 4;
        if (g0 + t < n_groups) {
            cplxf o; o.re = orr[m * dim + t]; o.im = oii[m * dim + t];
            amps[bse[t] | off[m]] = o;
        }
    }
}

// ---- k=5 (dim=32): one warp per tile of 16 groups, 32×32 matrix = 2×2 WMMA ----
#define K5_WARPS 2

extern "C" __global__
void apply_kq_tf32_k5(cplxf* amps, const cplx* mat, GateKq g, unsigned long long n_groups) {
    const unsigned dim = 32u;
    __shared__ float Mr[1024], Mi[1024];                 // 32×32 gate matrix (row-major)
    __shared__ float Vr[K5_WARPS][512], Vi[K5_WARPS][512]; // 32×16 group tiles (col-major, ld=32)
    __shared__ float Or[K5_WARPS][512], Oi[K5_WARPS][512]; // 32×16 output (row-major, ld=16)
    __shared__ unsigned long long Off[K5_WARPS][32], Base[K5_WARPS][16]; // per-tile index tables

    for (unsigned e = threadIdx.x; e < dim * dim; e += blockDim.x) {
        Mr[e] = (float)mat[e].re;
        Mi[e] = (float)mat[e].im;
    }
    __syncthreads();

    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long tile = (unsigned long long)blockIdx.x * K5_WARPS + warp;
    unsigned long long g0 = tile * WN;
    if (g0 >= n_groups) return;

    // off has dim=32 entries (one per lane); base has WN=16 (lanes 0..15).
    unsigned long long* off = Off[warp];
    unsigned long long* bse = Base[warp];
    off[lane] = off_of(lane, g);
    if (lane < 16u) bse[lane] = base_of(g0 + lane, g);
    __syncwarp();

    float* vr = Vr[warp];
    float* vi = Vi[warp];
    for (unsigned e = lane; e < dim * WN; e += 32u) {
        unsigned c = e & (dim - 1u);
        unsigned t = e >> 5;
        if (g0 + t < n_groups) {
            cplxf a = amps[bse[t] | off[c]];
            vr[c + t * dim] = a.re;          // col-major, ld = dim = 32
            vi[c + t * dim] = a.im;
        } else {
            vr[c + t * dim] = 0.0f;
            vi[c + t * dim] = 0.0f;
        }
    }
    __syncwarp();

    // Two output row-tiles (rows 0..15, 16..31); each is an independent 16×16 GEMM
    // over the full 32-deep contraction (4 WK steps), complex.
    for (unsigned mt = 0; mt < 2u; ++mt) {
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> accR, accI;
        wmma::fill_fragment(accR, 0.0f);
        wmma::fill_fragment(accI, 0.0f);
        for (unsigned ks = 0; ks < dim; ks += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::tf32, wmma::row_major> aMr, aMi;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::tf32, wmma::col_major> bVr, bVi;
            const float* mr = Mr + (mt * WM) * dim + ks; // row-tile mt, k-step ks; ld = dim
            const float* mi = Mi + (mt * WM) * dim + ks;
            wmma::load_matrix_sync(aMr, mr, dim);
            wmma::load_matrix_sync(aMi, mi, dim);
            wmma::load_matrix_sync(bVr, vr + ks, dim);   // col-major rows ks..ks+7; ld = dim
            wmma::load_matrix_sync(bVi, vi + ks, dim);
            for (int i = 0; i < aMr.num_elements; i++) aMr.x[i] = wmma::__float_to_tf32(aMr.x[i]);
            for (int i = 0; i < aMi.num_elements; i++) aMi.x[i] = wmma::__float_to_tf32(aMi.x[i]);
            for (int i = 0; i < bVr.num_elements; i++) bVr.x[i] = wmma::__float_to_tf32(bVr.x[i]);
            for (int i = 0; i < bVi.num_elements; i++) bVi.x[i] = wmma::__float_to_tf32(bVi.x[i]);
            wmma::mma_sync(accR, aMr, bVr, accR);
            wmma::mma_sync(accI, aMr, bVi, accI);
            wmma::mma_sync(accI, aMi, bVr, accI);
            for (int i = 0; i < aMi.num_elements; i++) aMi.x[i] = -aMi.x[i];
            wmma::mma_sync(accR, aMi, bVi, accR);
        }
        // Output row-tile mt occupies rows mt*16.. of the 32×16 Or/Oi (ld = WN = 16).
        wmma::store_matrix_sync(Or[warp] + mt * WM * WN, accR, WN, wmma::mem_row_major);
        wmma::store_matrix_sync(Oi[warp] + mt * WM * WN, accI, WN, wmma::mem_row_major);
    }
    __syncwarp();

    float* orr = Or[warp];
    float* oii = Oi[warp];
    for (unsigned e = lane; e < dim * WN; e += 32u) {
        unsigned m = e & (dim - 1u);   // output local index (row)
        unsigned t = e >> 5;           // group within tile (column)
        if (g0 + t < n_groups) {
            cplxf o; o.re = orr[m * WN + t]; o.im = oii[m * WN + t];
            amps[bse[t] | off[m]] = o;
        }
    }
}
