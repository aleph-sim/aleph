// P5.10-03 FP32 CUDA state-vector + diagonal gate kernels.
//
// The FP64 GPU SV is memory-bandwidth bound and FP64 is compute-weak on Ada
// (~1/64 the FP32 rate). Storing the `2^n` amplitudes as **float** complex
// (`cplxf`, 8 B) halves both the footprint (so n=31 fits the 20 GiB card) and
// the bytes moved per sweep, for ~2× throughput, at FP32 accuracy (~1e-5).
//
// Layout/convention identical to the FP64 `kernels.cu` / `diag.cu` (ADR 0004),
// so the same oracle suite pins this path. The key trick: the gate **matrix**
// stays `double` inside every per-gate uniform — the structs are byte-identical
// to the FP64 Rust param structs (`Gate1qParams`, `GateKqParams`, `CnotParams`,
// `Diag1qParams`, `DiagKqParams`), so the host reuses the exact same param
// builders. Only the amplitude buffer is float; matrix coefficients are cast to
// float (`cf`) at point of use, so the arithmetic runs on the fast FP32 ALU.

struct cplx  { double re; double im; }; // matrix coefficient (matches f64 params)
struct cplxf { float  re; float  im; }; // amplitude — half the bytes of cplx

__device__ __forceinline__ cplxf cfk(float re, float im) { cplxf r; r.re = re; r.im = im; return r; }
// double-complex matrix entry → float-complex (the only narrowing in the path).
__device__ __forceinline__ cplxf cf(cplx z) { return cfk((float)z.re, (float)z.im); }
__device__ __forceinline__ cplxf cmulf(cplxf a, cplxf b) {
    return cfk(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
__device__ __forceinline__ cplxf caddf(cplxf a, cplxf b) { return cfk(a.re + b.re, a.im + b.im); }

// --- Gate1q (matches Rust Gate1qParams: cplx m[4] then 4×u32) ---
struct Gate1q {
    cplx     m[4];
    unsigned target;
    unsigned t_bit;
    unsigned ctrl_mask;
    unsigned _pad;
};

// One thread per amplitude PAIR; grid covers 2^(n-1). Mirror of FP64 apply_1q.
extern "C" __global__
void apply_1q_f32(cplxf* amps, Gate1q g, unsigned long long n_pairs) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;
    unsigned long long lo = tid & (unsigned long long)(g.t_bit - 1u);
    unsigned long long hi = (tid >> g.target) << (g.target + 1u);
    unsigned long long i  = hi | lo;
    if ((i & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;
    unsigned long long j = i | (unsigned long long)g.t_bit;
    cplxf a = amps[i];
    cplxf b = amps[j];
    amps[i] = caddf(cmulf(cf(g.m[0]), a), cmulf(cf(g.m[1]), b));
    amps[j] = caddf(cmulf(cf(g.m[2]), a), cmulf(cf(g.m[3]), b));
}

// --- Cnot (matches Rust CnotParams: 4×u32) ---
struct Cnot { unsigned ctrl; unsigned targ; unsigned lo; unsigned hi; };

// One thread per control=1 amplitude pair; grid covers 2^(n-2). Permutation.
extern "C" __global__
void apply_cnot_f32(cplxf* amps, Cnot g, unsigned long long n_groups) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_groups) return;
    unsigned long long base = tid;
    unsigned long long ml = (1ULL << g.lo) - 1ULL;
    base = ((base & ~ml) << 1) | (base & ml);
    unsigned long long mh = (1ULL << g.hi) - 1ULL;
    base = ((base & ~mh) << 1) | (base & mh);
    unsigned long long i = base | (1ULL << g.ctrl);
    unsigned long long j = i | (1ULL << g.targ);
    cplxf tmp = amps[i];
    amps[i] = amps[j];
    amps[j] = tmp;
}

// --- GateKq (matches Rust GateKqParams: k, qbit[5], sorted[5], ctrl_mask) ---
struct GateKq {
    unsigned k;
    unsigned qbit[5];
    unsigned sorted[5];
    unsigned ctrl_mask;
};

// One thread per 2^k group; grid covers 2^(n-k). `mat` is the row-major
// 2^k×2^k matrix as **double** (uploaded once); cast to float per entry.
extern "C" __global__
void apply_kq_f32(cplxf* amps, const cplx* mat, GateKq g, unsigned long long n_groups) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_groups) return;
    unsigned k = g.k;
    if (k > 5u) return;
    unsigned dim = 1u << k;
    unsigned long long base = tid;
    for (unsigned j = 0; j < k; ++j) {
        unsigned p = g.sorted[j];
        unsigned long long mask = (1ULL << p) - 1ULL;
        base = ((base & ~mask) << 1) | (base & mask);
    }
    if ((base & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;
    unsigned long long gidx[32];
    cplxf v[32];
    for (unsigned l = 0; l < dim; ++l) {
        unsigned long long off = 0ULL;
        for (unsigned j = 0; j < k; ++j) {
            if ((l >> j) & 1u) off |= (unsigned long long)g.qbit[j];
        }
        gidx[l] = base | off;
        v[l] = amps[gidx[l]];
    }
    for (unsigned r = 0; r < dim; ++r) {
        cplxf acc = cfk(0.0f, 0.0f);
        for (unsigned c = 0; c < dim; ++c) {
            acc = caddf(acc, cmulf(cf(mat[r * dim + c]), v[c]));
        }
        amps[gidx[r]] = acc;
    }
}

// --- Multi1q (matches Rust Multi1qParams: cplx mats[20], m, sorted[5], pad[2]) ---
// FP32 mirror of the FP64 `apply_1q_multi` (P5.9-03), ported to FP32 in P5.11-04.
// A batch of m DISJOINT 1q gates applied in one state sweep: gather the 2^m local
// block, apply each gate as a butterfly along its own local bit, scatter back.
struct Multi1q {
    cplx     mats[20];
    unsigned m;
    unsigned sorted[5];
    unsigned _pad[2];
};

extern "C" __global__
void apply_1q_multi_f32(cplxf* amps, Multi1q g, unsigned long long n_groups) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_groups) return;
    unsigned m = g.m;
    if (m == 0u || m > 5u) return; // guards the fixed 32-entry stacks
    unsigned dim = 1u << m;

    // Reconstruct the base index (all m target bits clear) by inserting zero bits
    // at the ascending sorted positions — same scheme as apply_kq_f32.
    unsigned long long base = tid;
    for (unsigned j = 0; j < m; ++j) {
        unsigned p = g.sorted[j];
        unsigned long long mask = (1ULL << p) - 1ULL;
        base = ((base & ~mask) << 1) | (base & mask);
    }

    // Gather the local block: local bit j ↔ qubit sorted[j].
    unsigned long long gidx[32];
    cplxf v[32];
    for (unsigned l = 0; l < dim; ++l) {
        unsigned long long off = 0ULL;
        for (unsigned j = 0; j < m; ++j) {
            if ((l >> j) & 1u) off |= (1ULL << g.sorted[j]);
        }
        gidx[l] = base | off;
        v[l] = amps[gidx[l]];
    }

    // Apply gate j as a butterfly over local bit j (step = 1<<j); distinct bits ⇒
    // sequential application is their tensor product. Matrices cast f64→f32 at use.
    for (unsigned j = 0; j < m; ++j) {
        unsigned step = 1u << j;
        cplxf m00 = cf(g.mats[j * 4 + 0]), m01 = cf(g.mats[j * 4 + 1]);
        cplxf m10 = cf(g.mats[j * 4 + 2]), m11 = cf(g.mats[j * 4 + 3]);
        for (unsigned l = 0; l < dim; ++l) {
            if ((l & step) == 0u) {
                cplxf a = v[l];
                cplxf b = v[l | step];
                v[l]        = caddf(cmulf(m00, a), cmulf(m01, b));
                v[l | step] = caddf(cmulf(m10, a), cmulf(m11, b));
            }
        }
    }

    for (unsigned l = 0; l < dim; ++l) amps[gidx[l]] = v[l];
}

// Fused multi-qubit diagonal operator (P5.9-06), FP32 amplitudes (P5.11-04).
// amps[x] *= exp(i·φ(x)); φ accumulated and sincos'd in **double** for accuracy,
// then folded into the float amplitude. CSR-encoded terms (angles/conds/offsets).
extern "C" __global__
void apply_phase_poly_f32(cplxf* amps,
                          const double* angles,
                          const unsigned long long* conds,
                          const unsigned* offsets,
                          unsigned n_terms,
                          unsigned long long n_amps) {
    unsigned long long x = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (x >= n_amps) return;

    double phi = 0.0;
    for (unsigned t = 0; t < n_terms; ++t) {
        bool all = true;
        unsigned end = offsets[t + 1];
        for (unsigned c = offsets[t]; c < end; ++c) {
            if ((__popcll(conds[c] & x) & 1) == 0) { all = false; break; }
        }
        if (all) phi += angles[t];
    }

    double s, co;
    sincos(phi, &s, &co);
    cplxf a = amps[x];
    amps[x] = cfk((float)(a.re * co - a.im * s), (float)(a.re * s + a.im * co));
}

// Warp-cooperative register-tiled fused-block kernel (P5.10-01), FP32 (P5.11-04).
// One thread owns ONE amplitude; a group of dim=2^k contiguous lanes cooperatively
// owns one block via warp shuffles. The 2^k×2^k matrix lives in shared memory as
// **float** complex (cast once on load); the matvec is an intra-warp shuffle
// reduction (no local-memory spill, the wall apply_kq_f32 hits at k=4,5).
extern "C" __global__
void apply_kq_tiled_f32(cplxf* amps, const cplx* mat, GateKq g, unsigned long long n_amps) {
    extern __shared__ cplxf smatf[];
    unsigned k = g.k;
    unsigned dim = 1u << k;
    for (unsigned idx = threadIdx.x; idx < dim * dim; idx += blockDim.x) {
        smatf[idx] = cf(mat[idx]);
    }
    __syncthreads();

    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_amps) return;

    unsigned lane  = (unsigned)(tid & 31u);
    unsigned sl    = lane & (dim - 1u);
    unsigned gbase = lane - sl;
    unsigned long long gid = tid >> k;

    unsigned long long base = gid;
    for (unsigned j = 0; j < k; ++j) {
        unsigned p = g.sorted[j];
        unsigned long long mask = (1ULL << p) - 1ULL;
        base = ((base & ~mask) << 1) | (base & mask);
    }
    if ((base & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;

    unsigned long long off = 0ULL;
    for (unsigned j = 0; j < k; ++j) {
        if ((sl >> j) & 1u) off |= (unsigned long long)g.qbit[j];
    }
    unsigned long long myidx = base | off;
    cplxf v = amps[myidx];

    unsigned mask = (dim == 32u) ? 0xffffffffu : (((1u << dim) - 1u) << gbase);
    cplxf acc = cfk(0.0f, 0.0f);
    for (unsigned c = 0; c < dim; ++c) {
        float vr = __shfl_sync(mask, v.re, gbase + c);
        float vi = __shfl_sync(mask, v.im, gbase + c);
        acc = caddf(acc, cmulf(smatf[sl * dim + c], cfk(vr, vi)));
    }
    amps[myidx] = acc;
}

// --- Diag1q (matches Rust Diag1qParams: cplx d0, cplx d1, 2×u32) ---
struct Diag1q { cplx d0; cplx d1; unsigned t_bit; unsigned ctrl_mask; };

extern "C" __global__
void apply_diag_1q_f32(cplxf* amps, Diag1q g, unsigned long long n_amps) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n_amps) return;
    if ((i & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;
    cplxf d = (i & (unsigned long long)g.t_bit) ? cf(g.d1) : cf(g.d0);
    amps[i] = cmulf(amps[i], d);
}

// --- DiagK (matches Rust DiagKqParams: k, qbit[5], ctrl_mask) ---
struct DiagK { unsigned k; unsigned qbit[5]; unsigned ctrl_mask; };

extern "C" __global__
void apply_diag_f32(cplxf* amps, const cplx* diag, DiagK g, unsigned long long n_amps) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n_amps) return;
    if ((i & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;
    unsigned l = 0u;
    for (unsigned j = 0; j < g.k; ++j) {
        if (i & (unsigned long long)g.qbit[j]) l |= (1u << j);
    }
    amps[i] = cmulf(amps[i], cf(diag[l]));
}
