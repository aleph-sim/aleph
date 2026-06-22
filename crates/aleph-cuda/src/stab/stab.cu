// P5-07 GPU stabilizer (Aaronson-Gottesman CHP tableau) Clifford kernels.
//
// Layout mirrors the CPU backend's ColMajor (qubit-major) orientation, which is
// the GPU-friendly one: a qubit column is a contiguous `Wr = ceil((2n+1)/64)`
// word-span, and a Clifford gate updates that span word-parallel. The bit math
// is identical to `aleph-stab`'s `h_words`/`s_words`/`cnot_words` (AG 2004 §2),
// so the GPU tableau is bit-for-bit equal to the CPU one after the same gates.
//
//   x[a*Wr + w], z[a*Wr + w]  = word `w` of qubit `a`'s column (over the 2n+1
//                               generator-row axis; row r ↦ word r>>6, bit r&63)
//   sign[w]                   = sign bits, packed over the same row axis
//
// Parallelism per gate is `Wr`, which grows with n — the regime P5-07 targets.

struct StabOp { unsigned op; unsigned a; unsigned b; }; // op: 0=H 1=S 2=CNOT 3=X 4=Y 5=Z

typedef unsigned long long u64;

// Apply gate (op,a,b) to column word `w`; mutate x/z in place; RETURN the sign
// delta for word `w` (the caller XORs it into sign[w], plain or atomic). All AG
// §2 single-step rules, bit-parallel over the 64 generator rows in the word.
__device__ __forceinline__
u64 apply_word(u64* x, u64* z, unsigned op, unsigned a, unsigned b, unsigned Wr, unsigned w) {
    u64 ia = (u64)a * Wr + w;
    u64 xa = x[ia], za = z[ia];
    switch (op) {
        case 0: { x[ia] = za; z[ia] = xa; return xa & za; }          // H: r^=x&z; swap x,z
        case 1: { z[ia] = za ^ xa;        return xa & za; }          // S: r^=x&z; z^=x
        case 2: {                                                     // CNOT(a,b)
            u64 ib = (u64)b * Wr + w;
            u64 xb = x[ib], zb = z[ib];
            x[ib] = xb ^ xa;   // x_b ^= x_a
            z[ia] = za ^ zb;   // z_a ^= z_b
            return xa & zb & ~(xb ^ za);
        }
        case 3: return za;            // X: r ^= z
        case 4: return xa ^ za;       // Y: r ^= x ^ z
        case 5: return xa;            // Z: r ^= x
    }
    return 0;
}

// Initialise |0…0⟩ on `n` qubits: destabiliser c = X_c (row c), stabiliser
// c = Z_c (row n+c). x/z/sign are pre-zeroed; one thread per qubit, no races.
extern "C" __global__
void stab_init(u64* x, u64* z, unsigned n, unsigned Wr) {
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    u64 rc = c;                 // destabiliser row
    x[(u64)c * Wr + (rc >> 6)] |= (1ULL << (rc & 63));
    u64 rs = (u64)n + c;        // stabiliser row
    z[(u64)c * Wr + (rs >> 6)] |= (1ULL << (rs & 63));
}

// One Clifford gate: `Wr` threads, one per column word. No sign race (single
// gate), so a plain XOR.
extern "C" __global__
void stab_gate(u64* x, u64* z, u64* sign, unsigned op, unsigned a, unsigned b, unsigned Wr) {
    unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= Wr) return;
    u64 delta = apply_word(x, z, op, a, b, Wr, w);
    sign[w] ^= delta;
}

// A whole layer of gates on DISJOINT qubits in one launch: `n_ops * Wr` threads,
// thread (g,w) applies gate g's column word w. Disjoint qubits ⇒ no x/z race;
// many gates share the same sign word, so the sign update is an atomicXor.
extern "C" __global__
void stab_layer(u64* x, u64* z, u64* sign, const StabOp* ops, unsigned n_ops, unsigned Wr) {
    u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    unsigned g = (unsigned)(tid / Wr);
    if (g >= n_ops) return;
    unsigned w = (unsigned)(tid % Wr);
    StabOp o = ops[g];
    u64 delta = apply_word(x, z, o.op, o.a, o.b, Wr, w);
    if (delta) atomicXor(&sign[w], delta);
}
