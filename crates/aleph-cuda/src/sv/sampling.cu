// #555: fixed-order local prefixes and hierarchical inverse transforms.
// SampleAmplitude is the readout module's cplx/cplxf. BLOCK must match Rust's 256.
// Nonnegative prefixes are sequential within each group: zero bins remain exact
// plateaus. Groups execute independently, with no full-state Hillis–Steele pass.
__device__ __forceinline__ double sample_weight(SampleAmplitude a) {
    double re = (double)a.re, im = (double)a.im;
    return re * re + im * im;
}

extern "C" __global__ void sample_local_cdf(
    const SampleAmplitude* amps, double* cdf, unsigned long long n) {
    __shared__ double s[256];
    unsigned t = threadIdx.x;
    unsigned long long i = (unsigned long long)blockIdx.x * 256 + t;
    s[t] = i < n ? sample_weight(amps[i]) : 0.0;
    __syncthreads();
    if (t == 0) {
        double sum = 0.0;
        for (unsigned j = 0; j < 256; ++j) {
            sum += s[j];
            s[j] = sum;
        }
    }
    __syncthreads();
    if (i < n) cdf[i] = s[t];
}

// Input: 0 = norm partial pairs, 1 = local-CDF block ends, 2 = contiguous totals.
// Output includes both group-local CDFs and totals for the next hierarchy level.
extern "C" __global__ void sample_mass_cdf(
    const double* in, double* cdf, double* totals, unsigned long long n,
    unsigned long long offset, unsigned source_kind, unsigned long long state_n) {
    __shared__ double s[256];
    unsigned t = threadIdx.x;
    unsigned long long i = (unsigned long long)blockIdx.x * 256 + t;
    double v = 0.0;
    if (i < n) {
        unsigned long long at = source_kind == 0 ? 2 * i : i;
        if (source_kind == 1) {
            at = (i + 1) * 256;
            if (at > state_n) at = state_n;
            --at;
        }
        v = in[at];
    }
    s[t] = v;
    __syncthreads();
    if (t == 0) {
        double sum = 0.0;
        for (unsigned j = 0; j < 256; ++j) {
            sum += s[j];
            s[j] = sum;
        }
        totals[blockIdx.x] = sum;
    }
    __syncthreads();
    if (i < n) cdf[offset + i] = s[t];
}

__device__ __forceinline__ double below_total(double r, double total) {
    double target = r * total;
    return target < total ? target : nextafter(total, 0.0);
}

// Inverse transform within a positive-mass group, then remap the residual uniform
// to [0,1) conditional on that choice. Strict upper-bound skips zero plateaus.
__device__ __forceinline__ unsigned long long local_choice(
    const double* cdf, unsigned long long start, unsigned long long len, double* r) {
    double target = below_total(*r, cdf[start + len - 1]);
    unsigned long long lo = 0, hi = len;
    while (lo < hi) {
        unsigned long long mid = lo + (hi - lo) / 2;
        if (cdf[start + mid] > target) hi = mid;
        else lo = mid + 1;
    }
    // Finiteness/norm were checked before building the hierarchy. Positive mass
    // and target < total guarantee lo < len, including r rounded up to one.
    unsigned long long selected = lo < len ? lo : len - 1;
    double prev = selected ? cdf[start + selected - 1] : 0.0;
    double mass = cdf[start + selected] - prev;
    *r = mass > 0.0 ? (target - prev) / mass : 0.0;
    *r = fmax(0.0, fmin(*r, nextafter(1.0, 0.0)));
    return selected;
}

extern "C" __global__ void sample_hierarchy(
    const SampleAmplitude* amps, const double* leaf_cdf, const double* cdf,
    unsigned long long n, const double* uniforms, unsigned long long shots,
    unsigned use_leaf_cdf, unsigned long long* out) {
    unsigned long long shot = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= shots) return;
    // At most seven levels even for 2^63 amplitudes; supported states are <=2^31.
    unsigned long long sizes[8], offsets[8];
    unsigned levels = 0;
    unsigned long long m = (n + 255) / 256, offset = 0;
    while (true) {
        sizes[levels] = m;
        offsets[levels] = offset;
        ++levels;
        if (m <= 256) break;
        offset += m;
        m = (m + 255) / 256;
    }
    double r = uniforms[shot];
    unsigned long long group = 0;
    for (int l = (int)levels - 1; l >= 0; --l) {
        unsigned long long start = group * 256, len = sizes[l] - start;
        if (len > 256) len = 256;
        group = start + local_choice(cdf, offsets[l] + start, len, &r);
    }
    unsigned long long start = group * 256, len = n - start;
    if (len > 256) len = 256;
    if (use_leaf_cdf) {
        out[shot] = start + local_choice(leaf_cdf, start, len, &r);
        return;
    }
    // Sparse-shot mode: two bounded reads of one selected block, rather than a
    // full-state CDF allocation/pass. Accumulate in FP64 for both amplitude types.
    double total = 0.0;
    for (unsigned long long j = 0; j < len; ++j) total += sample_weight(amps[start + j]);
    double target = below_total(r, total), sum = 0.0;
    unsigned long long last = start;
    for (unsigned long long j = 0; j < len; ++j) {
        double p = sample_weight(amps[start + j]);
        if (p > 0.0) {
            last = start + j;
            sum += p;
            if (sum > target) {
                out[shot] = last;
                return;
            }
        }
    }
    out[shot] = last; // Rounded tail: last positive bin, never an empty trailing bin.
}
