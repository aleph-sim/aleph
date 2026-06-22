// P5-06 custom CUDA FP64 *diagonal*-gate kernels.
//
// The niche cuQuantum's generic `custatevecApplyMatrix` (and our own dense
// `apply_kq`) overpays for: a gate whose matrix is diagonal multiplies each
// amplitude by a single phase. There is no partner to gather and no 2^k block
// matvec — one coalesced, in-place pass over the 2^n amplitudes, ~1 complex
// multiply each. This is the workhorse of QFT (controlled-Phase), QAOA /
// Trotterised Ising (Rz + ZZ), and phase oracles / Grover diffusion
// (multi-controlled Z), so routing those gates here is a real circuit-level win.
//
// Convention matches `kernels.cu` exactly (ADR 0004): `amps[i]` is the amplitude
// of the basis state whose qubit q has value (i >> q) & 1. The host supplies
// `qbit[j]` = `1 << operand_qubit_for_matrix_bit_j` (MSB-first operand order, the
// same mapping `apply_kq` uses), and the external-control `ctrl_mask`. A thread
// whose amplitude does not satisfy the controls leaves it untouched.
//
// `cplx` is the same plain {re, im} pair as `kernels.cu` (16 B, 8-byte aligned),
// byte-identical to the host's interleaved f64 amplitude buffer, so NVRTC needs
// no CUDA vector headers.

struct cplx { double re; double im; };

__device__ __forceinline__ cplx cmul(cplx a, cplx b) {
    cplx r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

// Per-gate uniform for a single-qubit diagonal gate (Z, S, T, Rz, Phase, and any
// of them under external controls — CZ / CPhase / multi-controlled Z). Layout
// MUST match the Rust `Diag1qParams` struct (sv/diag.rs): two cplx then two u32.
struct Diag1q {
    cplx     d0;        // diagonal entry for target-bit 0  (matrix m00)
    cplx     d1;        // diagonal entry for target-bit 1  (matrix m11)
    unsigned t_bit;     // 1u << target
    unsigned ctrl_mask; // external-control mask (all must be set to fire)
};

// One thread per amplitude; grid covers n_amps = 2^n. In-place: read-modify-write
// of a single element, no cross-thread dependence.
extern "C" __global__
void apply_diag_1q(cplx* amps, Diag1q g, unsigned long long n_amps) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n_amps) return;
    if ((i & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;
    cplx d = (i & (unsigned long long)g.t_bit) ? g.d1 : g.d0;
    amps[i] = cmul(amps[i], d);
}

// Per-gate uniform for a generic k-qubit diagonal gate (k in 2..=3 in practice;
// the 1q case uses apply_diag_1q). Layout MUST match the Rust `DiagKqParams`.
struct DiagK {
    unsigned k;
    unsigned qbit[5];   // qbit[j] = global state-index bit for matrix-index bit j
    unsigned ctrl_mask;
};

// One thread per amplitude. `diag` is the 2^k-entry diagonal (cplx[dim]); the
// local index l is assembled from the operand bits of i, then amps[i] *= diag[l].
extern "C" __global__
void apply_diag(cplx* amps, const cplx* diag, DiagK g, unsigned long long n_amps) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n_amps) return;
    if ((i & (unsigned long long)g.ctrl_mask) != (unsigned long long)g.ctrl_mask) return;
    unsigned l = 0u;
    for (unsigned j = 0; j < g.k; ++j) {
        if (i & (unsigned long long)g.qbit[j]) l |= (1u << j);
    }
    amps[i] = cmul(amps[i], diag[l]);
}
