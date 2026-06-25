// P5.11-04 GPU-resident readout kernels (FP32 amplitudes).
//
// The FP32 mirror of `readout.cu`: identical reductions and inverse-CDF sampling,
// but the amplitude buffer is `cplxf` (interleaved [re, im] f32) instead of `cplx`
// (f64). Every accumulation/CDF/partial stays **double** for numerical stability
// (the only FP32 is the amplitude read); kernel names match the FP64 module so the
// host wrapper (`GpuReadoutF32`) loads them identically. See `readout.cu` for the
// derivations — the comments here only flag the FP32 divergence (amplitude type).

struct cplx  { double re; double im; }; // unused for amps; kept for symmetry
struct cplxf { float  re; float  im; }; // amplitude — half the bytes of cplx

// Software f64 atomic add via 64-bit CAS (arch-independent; see readout.cu).
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

// partials[2·blk] += Σ|amp|²; partials[2·blk+1] += Σ_{i&qbit≠0}|amp|². The
// per-amplitude square is computed in double from the f32 read.
extern "C" __global__
void reduce_abs2_branch(const cplxf* amps, double* partials, unsigned long long N, unsigned long long qbit) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    double p = 0.0, p1 = 0.0;
    if (i < N) {
        cplxf a = amps[i];
        double re = (double)a.re, im = (double)a.im;
        p = re * re + im * im;
        if (qbit && (i & qbit)) p1 = p;
    }
    block_reduce2(p, p1, partials);
}

// Expectation of a Pauli string (see readout.cu). conj(a)*b in double.
extern "C" __global__
void expect_pauli(const cplxf* amps, double* partials, unsigned long long N,
                  unsigned long long flip, unsigned long long sign_mask) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    double sre = 0.0, sim = 0.0;
    if (i < N) {
        unsigned long long j = i ^ flip;
        cplxf a = amps[i];
        cplxf b = amps[j];
        double are = (double)a.re, aim = (double)a.im;
        double bre = (double)b.re, bim = (double)b.im;
        double re = are * bre + aim * bim;
        double im = are * bim - aim * bre;
        double s = (__popcll(j & sign_mask) & 1) ? -1.0 : 1.0;
        sre = re * s;
        sim = im * s;
    }
    block_reduce2(sre, sim, partials);
}

// In-place measurement collapse (scale is f64, cast to f32 at the multiply).
extern "C" __global__
void collapse(cplxf* amps, unsigned long long N, unsigned long long qbit,
              unsigned outcome, double scale) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned bit = (i & qbit) ? 1u : 0u;
    if (bit == outcome) { amps[i].re *= (float)scale; amps[i].im *= (float)scale; }
    else { amps[i].re = 0.0f; amps[i].im = 0.0f; }
}

// Marginal probabilities over k qubits (|·|² in double, atomic into out).
extern "C" __global__
void marginal(const cplxf* amps, double* out, unsigned long long N,
              const unsigned* pos, unsigned k) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    cplxf a = amps[i];
    double re = (double)a.re, im = (double)a.im;
    double p = re * re + im * im;
    unsigned bin = 0u;
    for (unsigned j = 0; j < k; ++j) {
        if ((i >> pos[j]) & 1ull) bin |= (1u << j);
    }
    atomicAddD(&out[bin], p);
}

// probs[i] = |amp_i|² (double) for the sampling CDF.
extern "C" __global__
void abs2_into(const cplxf* amps, double* probs, unsigned long long N) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    cplxf a = amps[i];
    double re = (double)a.re, im = (double)a.im;
    probs[i] = re * re + im * im;
}

// One Hillis-Steele inclusive-scan step (double CDF; precision-independent).
extern "C" __global__
void scan_step(const double* in, double* out, unsigned long long N, unsigned long long d) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = in[i] + (i >= d ? in[i - d] : 0.0);
}

// Inverse-CDF upper-bound search (double; precision-independent).
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
