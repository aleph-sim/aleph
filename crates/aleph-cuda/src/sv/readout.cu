// P5-05 GPU-resident readout kernels (FP64).
//
// These reduce the device-resident state vector on the GPU so only small
// results cross PCIe: a scalar (norm / branch probability / expectation), a
// 2^k marginal vector, or `shots` sampled indices — never the full 2^n state.
// Both CUDA state-vector backends (hand-written and cuStateVec) share these,
// since both store the state as the same interleaved [re, im] f64 buffer with
// `amps[i]` the amplitude of basis state `i` (qubit q ↦ index bit q, ADR 0004).
//
// `cplx` mirrors the host buffer layout exactly (see kernels.cu); deliberately
// not CUDA's double2 so NVRTC needs no include paths.

struct cplx { double re; double im; };

// Software double-precision atomic add via 64-bit CAS. Used instead of the
// built-in atomicAdd(double*) so the PTX compiles on any virtual architecture
// NVRTC targets (the intrinsic needs sm_60+; the CAS form is arch-independent).
// NVIDIA CUDA C Programming Guide, "Atomic Functions" (the documented fallback).
__device__ __forceinline__ double atomicAddD(double* addr, double val) {
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double next = __longlong_as_double((long long)assumed) + val;
        old = atomicCAS(a, assumed, (unsigned long long)__double_as_longlong(next));
    } while (assumed != old);
    return __longlong_as_double((long long)old);
}

// Block-tree reduction of two per-thread partial sums, writing this block's two
// subtotals to partials[2·blockIdx + {0,1}] — NO global atomic, so there is no
// cross-block contention (`final_reduce2` then collapses the per-block pairs to
// a scalar). BLOCK threads, one element each.
#define BLOCK 256u
__device__ __forceinline__ void block_reduce2(double c0, double c1, double* partials) {
    __shared__ double s0[BLOCK];
    __shared__ double s1[BLOCK];
    unsigned t = threadIdx.x;
    s0[t] = c0;
    s1[t] = c1;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0u; s >>= 1) {
        if (t < s) { s0[t] += s0[t + s]; s1[t] += s1[t + s]; }
        __syncthreads();
    }
    if (t == 0u) {
        partials[2 * blockIdx.x] = s0[0];
        partials[2 * blockIdx.x + 1] = s1[0];
    }
}

// Second pass: sum the `m` per-block pairs in `partials` to (out[0], out[1]).
// Launched as a SINGLE block that grid-strides over all pairs — one tree
// reduction, still no atomics.
extern "C" __global__
void final_reduce2(const double* partials, unsigned long long m, double* out) {
    __shared__ double s0[BLOCK];
    __shared__ double s1[BLOCK];
    unsigned t = threadIdx.x;
    double c0 = 0.0, c1 = 0.0;
    for (unsigned long long b = t; b < m; b += blockDim.x) {
        c0 += partials[2 * b];
        c1 += partials[2 * b + 1];
    }
    s0[t] = c0;
    s1[t] = c1;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0u; s >>= 1) {
        if (t < s) { s0[t] += s0[t + s]; s1[t] += s1[t + s]; }
        __syncthreads();
    }
    if (t == 0u) { out[0] = s0[0]; out[1] = s1[0]; }
}

// partials[2·blk]   += Σ|amp_i|²            (total norm²)
// partials[2·blk+1] += Σ_{i & qbit ≠ 0}|amp_i|²  (the "qubit set" branch prob).
// Pass qbit = 0 to get the total alone.
extern "C" __global__
void reduce_abs2_branch(const cplx* amps, double* partials, unsigned long long N, unsigned long long qbit) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    double p = 0.0, p1 = 0.0;
    if (i < N) {
        cplx a = amps[i];
        p = a.re * a.re + a.im * a.im;
        if (qbit && (i & qbit)) p1 = p;
    }
    block_reduce2(p, p1, partials);
}

// Expectation of a Pauli string. With P|j⟩ = c_j |j ⊕ flip⟩ and
//   c_j = i^numY · (-1)^popcount(j & sign_mask)
// (flip = bits with X or Y, sign_mask = bits with Y or Z), we have
//   ⟨ψ|P|ψ⟩ = i^numY · Σ_i conj(ψ_i) ψ_{i⊕flip} (-1)^popcount((i⊕flip)&sign_mask).
// Note the sign is evaluated at j = i⊕flip (the index P maps *from*), not i —
// they coincide only when flip and sign_mask don't overlap (i.e. no Y). This
// kernel block-reduces the complex inner sum S into `partials` (real, imag
// per block); `final_reduce2` collapses it and the host folds in the global
// i^numY phase and takes the real part.
extern "C" __global__
void expect_pauli(const cplx* amps, double* partials, unsigned long long N,
                  unsigned long long flip, unsigned long long sign_mask) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    double sre = 0.0, sim = 0.0;
    if (i < N) {
        unsigned long long j = i ^ flip;
        cplx a = amps[i];
        cplx b = amps[j];
        // conj(a) * b = (a.re - i a.im)(b.re + i b.im)
        double re = a.re * b.re + a.im * b.im;
        double im = a.re * b.im - a.im * b.re;
        double s = (__popcll(j & sign_mask) & 1) ? -1.0 : 1.0;
        sre = re * s;
        sim = im * s;
    }
    block_reduce2(sre, sim, partials);
}

// In-place measurement collapse: keep the branch matching `outcome` (scaled by
// `scale` = 1/√p), zero the other.
extern "C" __global__
void collapse(cplx* amps, unsigned long long N, unsigned long long qbit,
              unsigned outcome, double scale) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned bit = (i & qbit) ? 1u : 0u;
    if (bit == outcome) { amps[i].re *= scale; amps[i].im *= scale; }
    else { amps[i].re = 0.0; amps[i].im = 0.0; }
}

// Marginal probabilities over `k` qubits. `pos[j]` is the global index-bit of
// output bit j; each amplitude's |·|² is added into out[bin] via atomic. out has
// 2^k entries (caller zeroes it first).
extern "C" __global__
void marginal(const cplx* amps, double* out, unsigned long long N,
              const unsigned* pos, unsigned k) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    cplx a = amps[i];
    double p = a.re * a.re + a.im * a.im;
    unsigned bin = 0u;
    for (unsigned j = 0; j < k; ++j) {
        if ((i >> pos[j]) & 1ull) bin |= (1u << j);
    }
    atomicAddD(&out[bin], p);
}

// probs[i] = |amp_i|² (for the sampling CDF).
extern "C" __global__
void abs2_into(const cplx* amps, double* probs, unsigned long long N) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    cplx a = amps[i];
    probs[i] = a.re * a.re + a.im * a.im;
}

// One Hillis-Steele inclusive-scan step: out[i] = in[i] + (i>=d ? in[i-d] : 0).
// Double-buffered by the host across ⌈log2 N⌉ launches to build the CDF.
extern "C" __global__
void scan_step(const double* in, double* out, unsigned long long N, unsigned long long d) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = in[i] + (i >= d ? in[i - d] : 0.0);
}

// Inverse-CDF sample: for each shot, find the smallest i with cdf[i] > target[j]
// (upper bound), matching the CPU sampler's partition_point semantics.
extern "C" __global__
void sample_search(const double* cdf, unsigned long long N, const double* targets,
                   unsigned long long shots, unsigned long long* out) {
    unsigned long long j = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (j >= shots) return;
    double t = targets[j];
    unsigned long long lo = 0, hi = N - 1, ans = N - 1;
    while (lo <= hi) {
        unsigned long long mid = lo + ((hi - lo) >> 1);
        if (cdf[mid] > t) { ans = mid; if (mid == 0) break; hi = mid - 1; }
        else lo = mid + 1;
    }
    out[j] = ans;
}
