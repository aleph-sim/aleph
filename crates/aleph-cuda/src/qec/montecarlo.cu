// Q3-03 on-device Monte-Carlo: noisy-syndrome generation + logical-error counting kernels.
//
// These bracket the GPU Union-Find decode (uf.cu's `uf_decode`, reused unmodified) so a whole
// threshold cell — sample → decode → score — runs entirely on the device; only the final logical-
// error COUNT (8 bytes) is copied back, never the syndromes. One thread per shot throughout.

typedef unsigned int u32;
typedef unsigned long long u64;

// Counter-based uniform in [0,1): a stateless SplitMix64 hash of (seed, shot, mechanism). The DEM
// sampler needs one independent Bernoulli coin per (shot, mechanism); a stateless hash gives that
// without per-shot RNG state and is reproducible from the seed.
__device__ __forceinline__ double uniform01(u64 seed, u64 shot, u32 m) {
    u64 z = seed + shot * 0x9E3779B97F4A7C15ULL + (u64)m * 0xD1B54A32D192ED03ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);  // (z >> 11) / 2^53
}

// Draw one shot per thread directly from the DEM: for each mechanism, flip a biased coin and, on
// heads, XOR its detectors into the packed syndrome and its observables into the truth mask.
// Mirrors the CPU harness's DEM-Bernoulli sampler (`run_dem_experiment`), so the generated syndrome
// distribution — and thus the logical-error rate — matches within sampling error.
extern "C" __global__ void dem_sample(
    const double *mech_prob,  // [n_mech]
    const u32 *det_off,       // [n_mech + 1]
    const u32 *det_idx,       // [total detector incidences]
    const u64 *mech_obs,      // [n_mech] observable bitmask per mechanism
    u32 n_mech, u32 words_per_shot, u32 n_shots, u64 seed, u64 shot_base,
    u32 *syn_words,  // [n_shots * words_per_shot]  (output)
    u64 *truth) {    // [n_shots]                   (output)
    u32 li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= n_shots) return;
    u64 gshot = shot_base + (u64)li;
    u32 *w = syn_words + (size_t)li * words_per_shot;
    for (u32 i = 0; i < words_per_shot; ++i) w[i] = 0;
    u64 t = 0;
    for (u32 m = 0; m < n_mech; ++m) {
        if (uniform01(seed, gshot, m) < mech_prob[m]) {
            for (u32 k = det_off[m]; k < det_off[m + 1]; ++k) {
                u32 d = det_idx[k];
                w[d >> 5] ^= (1u << (d & 31));
            }
            t ^= mech_obs[m];
        }
    }
    truth[li] = t;
}

// Score a decoded batch: a shot is a logical error iff the decoder's predicted observable flips
// differ from the truth on any observable (within the low `num_observables` bits). Accumulate the
// count into a single device counter.
extern "C" __global__ void mispredict_reduce(
    const u64 *pred, const u64 *truth, u64 low_mask, u32 n_shots, unsigned long long *counter) {
    u32 li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= n_shots) return;
    if (((pred[li] ^ truth[li]) & low_mask) != 0ULL) {
        atomicAdd(counter, 1ULL);
    }
}
