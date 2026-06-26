// Q3-02 GPU min-sum belief-propagation decoder.
//
// ONE THREAD PER SYNDROME SHOT, mirroring the GPU Union-Find decoder (Q3-01). Each thread replays
// the CPU `BpDecoder`'s exact min-sum schedule on its own syndrome against the shared read-only
// Tanner graph. All arithmetic is `double` in the identical edge order as the CPU reference, so the
// hard-decision error vector — and thus the correction — is reproduced bit-for-bit (IEEE inf/NaN
// semantics included). Throughput comes from batch parallelism across shots.
//
// BP is dense/regular (no pointer-chasing), so a block-cooperative variant could parallelise a
// single decode; one-thread-per-shot is the simplest design that guarantees the numeric match and
// already saturates the card on a large batch. The CPU decoder is the oracle.

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
    u32 shot = blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= n_shots) return;

    double *m_vc = m_vc_all + (size_t)shot * n_edges;
    double *e_cv = e_cv_all + (size_t)shot * n_edges;
    u8 *ehat = ehat_all + (size_t)shot * n_vars;
    u8 *s = s_all + (size_t)shot * n_checks;
    const u32 *words = syn_words + (size_t)shot * words_per_shot;

    // Syndrome bits over checks (= detectors).
    for (u32 c = 0; c < n_checks; ++c) {
        s[c] = (words[c >> 5] >> (c & 31)) & 1u;
    }
    // Init M_{v→c} = λ_v.
    for (u32 e = 0; e < n_edges; ++e) {
        m_vc[e] = lambda[edge_var[e]];
    }

    const double INF = pos_inf();
    for (u32 it = 0; it < max_iter; ++it) {
        // --- check → variable (min-sum) ---
        for (u32 c = 0; c < n_checks; ++c) {
            u32 lo = check_off[c], hi = check_off[c + 1];
            bool neg = s[c] != 0;  // running sign: true ⇒ negative
            double min1 = INF, min2 = INF;
            u32 argmin = 0xffffffffu;
            for (u32 k = lo; k < hi; ++k) {
                u32 edge = check_edges[k];
                double m = m_vc[edge];
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
                double m = m_vc[edge];
                bool excl_neg = (m < 0.0) ? !neg : neg;
                double ex_min = (edge == argmin) ? min2 : min1;
                double mag = alpha * ex_min;
                e_cv[edge] = excl_neg ? -mag : mag;
            }
        }
        // --- variable → check + posterior hard decision ---
        for (u32 v = 0; v < n_vars; ++v) {
            u32 lo = var_off[v], hi = var_off[v + 1];
            double total = lambda[v];
            for (u32 e = lo; e < hi; ++e) total += e_cv[e];
            ehat[v] = (total < 0.0) ? 1 : 0;
            for (u32 e = lo; e < hi; ++e) m_vc[e] = total - e_cv[e];
        }
        // --- convergence: H ê == s ---
        bool ok = true;
        for (u32 c = 0; c < n_checks && ok; ++c) {
            u32 lo = check_off[c], hi = check_off[c + 1];
            u8 parity = 0;
            for (u32 k = lo; k < hi; ++k) parity ^= ehat[edge_var[check_edges[k]]];
            if (parity != s[c]) ok = false;
        }
        if (ok) break;
    }

    u64 mask = 0;
    for (u32 v = 0; v < n_vars; ++v) {
        if (ehat[v]) mask ^= obs[v];
    }
    out_mask[shot] = mask;
}
