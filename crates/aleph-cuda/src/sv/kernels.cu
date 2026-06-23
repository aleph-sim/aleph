// P5-02 CUDA FP64 state-vector gate kernels.
//
// Standard dense-statevector gate application by the bit-insertion butterfly:
// one thread per amplitude group, reconstruct the group's base index by
// inserting zero bits at the target positions, gather the 2^k block, apply the
// dense matvec, scatter back. This is the canonical GPU SV kernel (Jones,
// Brown, Bautista-Salinas & Benjamin, "QuEST and High Performance Simulation
// of Quantum Computers", Sci. Rep. 9:10736, 2019, Methods), and matches aleph's
// CPU `apply_kq` kernel one-to-one.
//
// Convention (ADR 0004, enforced by the CPU-oracle test): `amps[i]` is the
// amplitude of the basis state whose qubit q has value (i >> q) & 1. The host
// supplies `qbit[j]` = the global state-index bit for matrix-index bit j
// (`gate.matrix()` is MSB-first, so the host sets `qbit[j] = 1 << qubits[k-1-j]`);
// the kernel just OR's in `qbit[j]` for each set bit of the local matrix index.
//
// `cplx` is a plain {re, im} pair (16 bytes, 8-byte aligned) — deliberately not
// CUDA's `double2`, so the source needs no CUDA vector headers and NVRTC
// compiles it with zero include paths. Its byte layout matches the host's
// interleaved `f64` amplitude buffer exactly.

struct cplx { double re; double im; };

// Per-gate uniform for a single-qubit gate. Layout MUST match the Rust
// `Gate1qParams` struct (sv/kernel.rs): cplx m[4] (row-major 2x2) then four u32.
struct Gate1q {
    cplx     m[4];      // m[0]=m00 m[1]=m01 m[2]=m10 m[3]=m11
    unsigned target;
    unsigned t_bit;     // 1u << target
    unsigned ctrl_mask; // external-control mask (0 for a plain 1q gate)
    unsigned _pad;
};

// Per-gate uniform for a generic dense k-qubit gate (k in 2..=5; the 1q fast
// path uses apply_1q). Layout MUST match the Rust `GateKqParams` struct.
struct GateKq {
    unsigned k;
    unsigned qbit[5];   // qbit[j] = global state-index bit for matrix-index bit j
    unsigned sorted[5]; // target positions ASCENDING, for zero-bit insertion
    unsigned ctrl_mask;
};

__device__ __forceinline__ cplx cmk(double re, double im) { cplx r; r.re = re; r.im = im; return r; }
__device__ __forceinline__ cplx cmul(cplx a, cplx b) {
    return cmk(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
__device__ __forceinline__ cplx cadd(cplx a, cplx b) { return cmk(a.re + b.re, a.im + b.im); }

// One thread per amplitude PAIR; grid covers n_pairs = 2^(n-1). Insert a zero
// bit at `target` to recover the base index i (target bit clear); j = i | t_bit
// is its partner. Indices are 64-bit so n up to ~30 is shift-safe.
extern "C" __global__
void apply_1q(cplx* amps, Gate1q g, unsigned long long n_pairs) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    unsigned long long lo = tid & (unsigned long long)(g.t_bit - 1u);
    unsigned long long hi = (tid >> g.target) << (g.target + 1u);
    unsigned long long i  = hi | lo;
    if ((i & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;

    unsigned long long j = i | (unsigned long long)g.t_bit;
    cplx a = amps[i];
    cplx b = amps[j];
    amps[i] = cadd(cmul(g.m[0], a), cmul(g.m[1], b));
    amps[j] = cadd(cmul(g.m[2], a), cmul(g.m[3], b));
}

// Per-batch uniform for a layer of m DISJOINT single-qubit gates (P5.9-03).
// Layout MUST match the Rust `Multi1qParams` struct (sv/kernel.rs).
// `mats[j]` is the row-major 2x2 of the gate on the j-th (ascending) qubit
// `sorted[j]`; gates act on distinct qubits, so they commute and the whole
// batch is one tensor product applied in a single state sweep.
struct Multi1q {
    cplx     mats[20];   // 5 gates × 4 complex (m00 m01 m10 m11); gate j = mats[j*4..]
    unsigned m;          // gates in this batch (1..=5)
    unsigned sorted[5];  // target positions ASCENDING, for zero-bit insertion
    unsigned _pad[2];
};

// One thread per group of 2^m amplitudes; grid covers n_groups = 2^(n-m).
// Gather the 2^m local block (the m disjoint target axes), apply each gate as a
// butterfly along its own local bit, scatter back. m butterflies = O(m·2^m)
// work per group — far cheaper than apply_kq's dense O(4^m) matvec — while
// collapsing m separate full-state passes into one. In-place safe: the whole
// block is read into registers before any write.
extern "C" __global__
void apply_1q_multi(cplx* amps, Multi1q g, unsigned long long n_groups) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_groups) return;

    unsigned m = g.m;
    if (m == 0u || m > 5u) return; // guards the fixed 32-entry stacks
    unsigned dim = 1u << m;

    // Reconstruct the base index (all m target bits clear) by inserting zero
    // bits at the ascending sorted positions — same scheme as apply_kq.
    unsigned long long base = tid;
    for (unsigned j = 0; j < m; ++j) {
        unsigned p = g.sorted[j];
        unsigned long long mask = (1ULL << p) - 1ULL;
        base = ((base & ~mask) << 1) | (base & mask);
    }

    // Gather the local block: local bit j ↔ qubit sorted[j].
    unsigned long long gidx[32];
    cplx v[32];
    for (unsigned l = 0; l < dim; ++l) {
        unsigned long long off = 0ULL;
        for (unsigned j = 0; j < m; ++j) {
            if ((l >> j) & 1u) off |= (1ULL << g.sorted[j]);
        }
        gidx[l] = base | off; // base's target bits are clear ⇒ | == +
        v[l] = amps[gidx[l]];
    }

    // Apply gate j as a butterfly over local bit j (step = 1<<j). Each gate
    // touches a distinct bit, so sequential application is their tensor product.
    for (unsigned j = 0; j < m; ++j) {
        unsigned step = 1u << j;
        cplx m00 = g.mats[j * 4 + 0], m01 = g.mats[j * 4 + 1];
        cplx m10 = g.mats[j * 4 + 2], m11 = g.mats[j * 4 + 3];
        for (unsigned l = 0; l < dim; ++l) {
            if ((l & step) == 0u) {
                cplx a = v[l];
                cplx b = v[l | step];
                v[l]        = cadd(cmul(m00, a), cmul(m01, b));
                v[l | step] = cadd(cmul(m10, a), cmul(m11, b));
            }
        }
    }

    for (unsigned l = 0; l < dim; ++l) amps[gidx[l]] = v[l];
}

// Fused multi-qubit diagonal operator (P5.9-06): amps[x] *= exp(i·φ(x)) where
// φ(x) = Σ_t angles[t] · [∀ c in conds[offsets[t]..offsets[t+1]]: parity(c & x) odd].
// A long controlled-phase ladder (QFT/QPE) collapses via FuseDiagonalRuns into one
// such phase polynomial, so this single coalesced sweep replaces ~n²/2 cphase
// passes. One thread per amplitude; each amplitude is read and written exactly
// once (fully coalesced) — the term loop is per-thread integer work over the small
// CSR arrays (L2-cached), with one sincos at the end.
extern "C" __global__
void apply_phase_poly(cplx* amps,
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
            // parity(conds[c] & x) odd ⇔ __popcll is odd. An empty cond range
            // (offsets[t]==offsets[t+1]) leaves `all` true ⇒ global phase.
            if ((__popcll(conds[c] & x) & 1) == 0) { all = false; break; }
        }
        if (all) phi += angles[t];
    }

    double s, co;
    sincos(phi, &s, &co);
    cplx a = amps[x];
    amps[x] = cmk(a.re * co - a.im * s, a.re * s + a.im * co);
}

// Per-gate uniform for a plain CNOT (P5.9-04). Layout MUST match the Rust
// `CnotParams` struct. A CNOT is a permutation, not a rotation: it swaps the
// two amplitudes that differ in the target bit whenever the control bit is 1.
struct Cnot {
    unsigned ctrl;    // control qubit index
    unsigned targ;    // target qubit index
    unsigned lo;      // min(ctrl, targ) — for ascending zero-bit insertion
    unsigned hi;      // max(ctrl, targ)
};

// One thread per amplitude pair in the control=1 subspace; grid covers
// n_groups = 2^(n-2). Reconstruct the index with control & target bits clear,
// set control=1, and swap amps[targ=0] ↔ amps[targ=1]. Touches only the
// control=1 half of the state with zero FLOPs — vs apply_kq's full 2^n sweep
// plus a 4×4 matvec. Pure permutation, so it is trivially in-place safe.
extern "C" __global__
void apply_cnot(cplx* amps, Cnot g, unsigned long long n_groups) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_groups) return;

    // Insert two zero bits at the ascending {lo, hi} positions (same scheme as
    // apply_kq with k=2), yielding the index with control=0, target=0.
    unsigned long long base = tid;
    unsigned long long ml = (1ULL << g.lo) - 1ULL;
    base = ((base & ~ml) << 1) | (base & ml);
    unsigned long long mh = (1ULL << g.hi) - 1ULL;
    base = ((base & ~mh) << 1) | (base & mh);

    unsigned long long i = base | (1ULL << g.ctrl); // control=1, target=0
    unsigned long long j = i | (1ULL << g.targ);     // control=1, target=1
    cplx tmp = amps[i];
    amps[i] = amps[j];
    amps[j] = tmp;
}

// One thread per group of 2^k amplitudes; grid covers n_groups = 2^(n-k).
// `mat` is row-major 2^k x 2^k (M[r*dim + c]). dim <= 32 (k <= 5), so the
// thread-local arrays fit. In-place safe: all inputs are read before any write.
extern "C" __global__
void apply_kq(cplx* amps, const cplx* mat, GateKq g, unsigned long long n_groups) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= n_groups) return;

    unsigned k = g.k;
    if (k > 5u) return; // guards the fixed 32-entry stacks (host contract: k<=5)
    unsigned dim = 1u << k;

    // Reconstruct the base index (all target bits clear) by inserting k zero
    // bits at the ascending sorted positions. Ascending order is required so
    // each insertion shifts the still-to-come higher slots up correctly.
    unsigned long long base = tid;
    for (unsigned j = 0; j < k; ++j) {
        unsigned p = g.sorted[j];
        unsigned long long mask = (1ULL << p) - 1ULL;
        base = ((base & ~mask) << 1) | (base & mask);
    }
    if ((base & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;

    // Global index of each local matrix index l: bit j of l -> qubits[j].
    unsigned long long gidx[32];
    cplx v[32];
    for (unsigned l = 0; l < dim; ++l) {
        unsigned long long off = 0ULL;
        for (unsigned j = 0; j < k; ++j) {
            if ((l >> j) & 1u) off |= (unsigned long long)g.qbit[j]; // host maps bit j MSB-first
        }
        gidx[l] = base | off; // base's target bits are clear => | == +
        v[l] = amps[gidx[l]];
    }
    for (unsigned r = 0; r < dim; ++r) {
        cplx acc = cmk(0.0, 0.0);
        for (unsigned c = 0; c < dim; ++c) {
            acc = cadd(acc, cmul(mat[r * dim + c], v[c]));
        }
        amps[gidx[r]] = acc;
    }
}
