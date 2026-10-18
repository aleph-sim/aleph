// Q3-02 GPU min-sum belief-propagation decoder.
//
// By default, ONE THREAD PER SYNDROME SHOT, mirroring the GPU Union-Find decoder (Q3-01). Each thread replays
// the CPU `BpDecoder`'s exact min-sum schedule on its own syndrome against the shared read-only
// Tanner graph. All arithmetic is `double` in the identical edge order as the CPU reference, so the
// hard-decision error vector — and thus the correction — is reproduced bit-for-bit (IEEE inf/NaN
// semantics included). Throughput comes from batch parallelism across shots.
//
// BP_COOPERATIVE assigns a full warp to one shot: checks and variables execute in parallel,
// with phase fences and unchanged serial reductions inside each node. The default remains
// one thread per shot; the CPU decoder is the oracle for both schedules.

typedef unsigned int u32;
typedef unsigned char u8;
typedef unsigned long long u64;

// IEEE +infinity as a double, so the min sentinels match the CPU's `f64::INFINITY` exactly (a
// degree-1 check legitimately emits ±inf; using a finite sentinel would diverge under inf/NaN math).
__device__ __forceinline__ double pos_inf() { return __longlong_as_double(0x7FF0000000000000LL); }

extern "C" __global__ void bp_decode(
    // --- shared read-only Tanner graph ---
    const double *lambda,   // [n_vars] prior LLRs
    const u64 *obs,         // [n_vars] observable masks
    const u32 *var_off,     // [n_vars + 1]
    const u32 *edge_check,  // [n_edges]
    const u32 *edge_var,    // [n_edges]
    const u32 *check_off,   // [n_checks + 1]
    const u32 *check_edges, // [n_edges]
    u32 n_vars, u32 n_edges, u32 n_checks, u32 max_iter, double alpha,
    // --- input syndromes (packed detector bits) ---
    const u32 *syn_words, u32 words_per_shot, u32 n_shots,
    // --- output: one observable-flip bitmask per shot ---
    u64 *out_mask,  // [n_shots]
    // --- per-shot scratch ---
    double *m_vc_all,  // [n_shots * n_edges]
    double *e_cv_all,  // [n_shots * n_edges]
    u8 *ehat_all,      // [n_shots * n_vars]
    u8 *s_all) {       // [n_shots * n_checks]
#ifdef BP_COOPERATIVE
    // Whole warps own shots; the bound is uniform, so every valid warp has 32 participants.
    u32 shot = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const u32 first = threadIdx.x % 32;
    const u32 increment = 32;
#else
    u32 shot = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 first = 0;
    const u32 increment = 1;
#endif
    if (shot >= n_shots) return;

    // Compile-time scratch placement; cooperative scheduling uses shot-major storage.
#if defined(BP_WARP_TILED)
    const size_t step = 32;
    const size_t tile = shot / 32;
    const size_t lane = shot % 32;
    double *m_vc = m_vc_all + tile * n_edges * 32 + lane;
    double *e_cv = e_cv_all + tile * n_edges * 32 + lane;
    u8 *ehat = ehat_all + tile * n_vars * 32 + lane;
    u8 *s = s_all + tile * n_checks * 32 + lane;
#elif defined(BP_EDGE_MAJOR)
    const size_t step = n_shots;
    double *m_vc = m_vc_all + shot;
    double *e_cv = e_cv_all + shot;
    u8 *ehat = ehat_all + shot;
    u8 *s = s_all + shot;
#else
    const size_t step = 1;
    double *m_vc = m_vc_all + (size_t)shot * n_edges;
    double *e_cv = e_cv_all + (size_t)shot * n_edges;
    u8 *ehat = ehat_all + (size_t)shot * n_vars;
    u8 *s = s_all + (size_t)shot * n_checks;
#endif
    const u32 *words = syn_words + (size_t)shot * words_per_shot;

    // Syndrome bits over checks (= detectors).
    for (u32 c = first; c < n_checks; c += increment) {
        s[(size_t)c * step] = (words[c >> 5] >> (c & 31)) & 1u;
    }
    // Init M_{v→c} = λ_v.
    for (u32 e = first; e < n_edges; e += increment) {
        m_vc[(size_t)e * step] = lambda[edge_var[e]];
    }

#ifdef BP_COOPERATIVE
    __syncwarp();
#endif
    const double INF = pos_inf();
    for (u32 it = 0; it < max_iter; ++it) {
        // --- check → variable (min-sum) ---
        for (u32 c = first; c < n_checks; c += increment) {
            u32 lo = check_off[c], hi = check_off[c + 1];
            bool neg = s[(size_t)c * step] != 0;  // running sign: true ⇒ negative
            double min1 = INF, min2 = INF;
            u32 argmin = 0xffffffffu;
            for (u32 k = lo; k < hi; ++k) {
                u32 edge = check_edges[k];
                double m = m_vc[(size_t)edge * step];
                if (m < 0.0) neg = !neg;
                double a = fabs(m);
                if (a < min1) {
                    min2 = min1;
                    min1 = a;
                    argmin = edge;
                } else if (a < min2) {
                    min2 = a;
                }
            }
            for (u32 k = lo; k < hi; ++k) {
                u32 edge = check_edges[k];
                double m = m_vc[(size_t)edge * step];
                bool excl_neg = (m < 0.0) ? !neg : neg;
                double ex_min = (edge == argmin) ? min2 : min1;
                double mag = alpha * ex_min;
                e_cv[(size_t)edge * step] = excl_neg ? -mag : mag;
            }
        }
#ifdef BP_COOPERATIVE
        __syncwarp(); // all check messages become visible before variable sums
#endif
        // --- variable → check + posterior hard decision ---
        for (u32 v = first; v < n_vars; v += increment) {
            u32 lo = var_off[v], hi = var_off[v + 1];
            double total = lambda[v];
            for (u32 e = lo; e < hi; ++e) total += e_cv[(size_t)e * step];
            ehat[(size_t)v * step] = (total < 0.0) ? 1 : 0;
            for (u32 e = lo; e < hi; ++e) m_vc[(size_t)e * step] = total - e_cv[(size_t)e * step];
        }
#ifdef BP_COOPERATIVE
        __syncwarp(); // all hard decisions and next-iteration messages are complete
#endif
        // --- convergence: H ê == s ---
        bool ok = true;
#ifdef BP_COOPERATIVE
        for (u32 c = first; c < n_checks; c += increment) {
#else
        for (u32 c = 0; c < n_checks && ok; ++c) {
#endif
            u32 lo = check_off[c], hi = check_off[c + 1];
            u8 parity = 0;
            for (u32 k = lo; k < hi; ++k) parity ^= ehat[(size_t)edge_var[check_edges[k]] * step];
            if (parity != s[(size_t)c * step]) ok = false;
        }
#ifdef BP_COOPERATIVE
        ok = __all_sync(0xffffffffu, ok);
#endif
        if (ok) break;
    }

    u64 mask = 0;
    for (u32 v = first; v < n_vars; v += increment) {
        if (ehat[(size_t)v * step]) mask ^= obs[v];
    }
#ifdef BP_COOPERATIVE
    // XOR is exact and associative; no floating-point reductions are reordered.
    for (u32 delta = 16; delta; delta >>= 1) mask ^= __shfl_down_sync(0xffffffffu, mask, delta);
    if (first == 0) out_mask[shot] = mask;
#else
    out_mask[shot] = mask;
#endif
}
