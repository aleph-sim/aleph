# Playbook: Quantum Fourier Transform (QFT)

> **Algorithm-specific optimization guide.** Read after `OPTIMIZATION GUIDE.md` and `OPTIMIZATION CYCLE.md`. Applies the methodology to QFT.

-----

## Quick Reference

|Property               |Value                                                               |
|-----------------------|--------------------------------------------------------------------|
|Primary backends       |State vector (dense), MPS (if structured input)                     |
|Key gates              |H, controlled-phase (CP / CPhase), SWAP                             |
|Gate count             |~n²/2 gates for n qubits                                            |
|Circuit depth          |O(n)                                                                |
|Entanglement           |High (max for random input)                                         |
|Memory complexity      |O(2ⁿ) for state vector                                              |
|Primary bottleneck     |Diagonal controlled-phase gates dominate; cache pressure at larger n|
|Best-case backend match|MPS for low-entanglement input states; SV otherwise                 |

**Target to beat**: Qiskit Aer `statevector` simulator, single-thread, FP64.

**Phase 1 success metric**: ≤2× of Qiskit Aer wall-time on `qft_20` and `qft_25`.
**Phase 4 success metric**: ≤1.2× of Qiskit Aer; ≥80% of memory bandwidth ceiling.

-----

## Algorithm Overview

QFT maps `|x⟩ → (1/√N) Σ_y e^(2πixy/N) |y⟩`. Standard decomposition:

```
for j in 0..n:
    H(qubit[j])
    for k in (j+1)..n:
        CP(qubit[k], qubit[j], angle = π / 2^(k-j))
# Optional final SWAP layer to reverse bit order
```

Gate count: n Hadamards + n(n−1)/2 controlled-phases + ⌊n/2⌋ SWAPs (if needed).

**Why it’s a flagship benchmark**:

- Used inside Shor, QPE, many other algorithms.
- Stresses controlled-phase (diagonal) gates heavily.
- Has structured locality — adjacent qubits get most attention, far ones contribute small angles.
- Has known approximations (truncate small-angle phases) that allow accuracy/speed trade-offs.

-----

## Computational Profile

For `qft_n` on state vector:

|Component                |Share of runtime|Bottleneck character                                 |
|-------------------------|----------------|-----------------------------------------------------|
|Controlled-phase gates   |~80% (large n)  |Memory-bound; diagonal, single pass over state vector|
|Hadamard gates           |~10%            |Memory-bound; pair iteration                         |
|SWAP gates               |~5–10%          |Memory-bound; pair swap                              |
|Other (parsing, dispatch)|<5%             |Should be negligible                                 |

Per-gate, the bottleneck is **memory bandwidth**. A controlled-phase touches 1/4 of the state vector (only amplitudes with both control and target = 1); a Hadamard touches all amplitudes. Arithmetic per amplitude is trivial (≤4 FLOPs); the limit is reading and writing the state vector from RAM.

**Roofline**: at n=25, state vector = 16 × 2²⁵ = 512 MB. One pass at 80 GB/s = 6.4 ms. QFT has ~325 gates at n=25; if each gate did a full pass, runtime ≈ 2.1 s. In practice many gates touch only a fraction, so the practical floor is ~500 ms on this hardware.

-----

## Optimization Ladder

Specific opportunities for QFT, in ROI order.

### Rank 1: Algorithm — Approximate QFT (AQFT)

Cut off controlled-phase rotations with small angles (below precision ε). Standard cutoff: drop CP gates with angle < π/2^k for k > log₂(1/ε).

**Impact**: at n=25, dropping ~half of the gates with negligible accuracy loss for typical applications.

**Trade-off**: introduces O(ε·n²) error. For chemistry/finance applications where final results have inherent noise, this is invisible.

**When to use**: when the consumer of the QFT result is itself approximate (variational algorithms, sampling-based protocols).

**Implementation**: a `Truncate` pass in `aleph-ir` that drops small-angle rotations under a configurable threshold.

### Rank 2: Backend selection

- **Input state mostly |0⟩**: SV backend has nothing to do until the first H. MPS would notice this and stay efficient.
- **Input state is a product state**: still SV-bound; MPS doesn’t save much because QFT generates entanglement.
- **Input state already random/entangled**: SV is optimal; MPS would struggle.

The backend selector should look at the input state’s entanglement entropy estimate; route to MPS if low, SV otherwise.

### Rank 3: IR-level — Gate fusion specific to QFT

The structure `H · CP(θ_1) · CP(θ_2) · ... · CP(θ_k)` on the same target qubit is common. Fuse the CP sequence into one diagonal operation (sum the angles per (control, target) pair).

**More aggressive**: the entire chain of CPs on a target qubit, after the H, is a diagonal “phase polynomial”. The accumulated effect on each amplitude is `e^(i Σ θ_k · ctrl_k)`. This can be applied in a **single pass**, not k passes.

**Impact**: replaces O(n) diagonal gates with 1 fused diagonal application per Hadamard.

**Implementation**: `PhasePolynomialFusion` pass — group consecutive diagonal gates by their target structure, apply as one.

### Rank 4: Specialized controlled-phase kernel

A generic CP gate uses a 4×4 matrix; only 2 entries are non-trivial (diagonal). A specialized kernel:

- Skip all amplitudes where control bit = 0 (no work).
- For amplitudes where both bits = 1, multiply by `e^(iθ)`.
- Precompute `(cos θ, sin θ)` once outside the loop.

**Impact**: ~3–5× faster than generic 2q kernel.

### Rank 5: Memory layout & access patterns

- SoA layout (covered globally in P1-01).
- **Block-wise iteration** for far-apart qubits: when control and target are far in qubit index, the memory stride is large. Iterating in blocks improves L2 reuse.

### Rank 6: SIMD

- Hadamard SIMD: same as generic 1q gate (P1-03).
- CP SIMD: less obvious because the operation is conditional (only act if control bit set). Use masked SIMD (AVX-512 `_mm512_mask_*` intrinsics) for clean vectorization.

### Rank 7: Multi-threading

QFT parallelizes well: each gate’s state vector update is embarrassingly parallel. Standard Rayon chunking applies (P2-01).

**Note**: at small n (n ≤ 16), thread overhead exceeds work; don’t parallelize.

### Rank 8: GPU

State vector on GPU: standard cuStateVec or hand-written kernels. QFT-specific optimization: batch many CP gates into one kernel launch to avoid per-gate launch overhead.

-----

## Pitfalls

**1. Confusing endianness**: QFT in textbooks uses big-endian; many simulators use little-endian. The bit-reversal SWAP layer at the end is sometimes included, sometimes not. Document and test explicitly.

**2. Floating-point precision in small angles**: at large n, `cos(π/2^k)` for k > 30 is `≈ 1` and `sin` is `≈ 0`. FP64 has ~15 decimal digits; very small angles are below precision and should either be skipped or computed via Taylor expansion.

**3. Forgetting AQFT for the relevant benchmarks**: VQE/QAOA papers often use AQFT, not exact QFT. If you’re benchmarking against them, match what they did.

**4. SWAP layer interpretation**: Qiskit’s QFT has a final reversal layer; some don’t. Test against the same convention. Off-by-one (or full bit-reversal) errors are common.

**5. Composition with QPE**: in QPE, QFT runs on a register that’s classically correlated with another. The “phase precision” of the QFT determines the QPE precision. Be careful when truncating.

-----

## Baseline Comparisons

Reference times on the workstation (Ryzen 9 7950X, single-thread, FP64) for exact QFT, no AQFT:

|n |Qiskit Aer (ms)|Target Phase 1 (ms)|Target Phase 4 (ms)|
|--|---------------|-------------------|-------------------|
|10|0.4            |≤0.8               |≤0.5               |
|15|4.1            |≤8.2               |≤4.9               |
|20|71             |≤142               |≤85                |
|25|2,400          |≤4,800             |≤2,880             |
|30|97,000         |≤200,000           |≤120,000           |

These numbers are illustrative; regenerate from the actual reference machine before relying on them.

-----

## Phase-by-Phase Sub-goals

### Phase 0 (Foundation)

- [ ] Naive QFT runs end-to-end on naive backend up to n=20.
- [ ] OpenQASM input for QFT parses correctly.
- [ ] Result matches Qiskit Aer to 1e-12.
- [ ] Benchmark fixtures: qft_10, qft_15, qft_20.

### Phase 1 (Single-thread CPU)

- [ ] Specialized controlled-phase kernel (5× over generic 2q).
- [ ] Phase polynomial fusion pass.
- [ ] AVX2 Hadamard + SIMD-friendly CP.
- [ ] Within 2× of Qiskit Aer single-thread at n=20, n=25.

### Phase 2 (Multi-thread CPU)

- [ ] Parallel QFT: ≥6× speedup on 8 cores at n=25.

### Phase 3 (Alternative backends)

- [ ] MPS backend handles QFT on low-entanglement inputs.
- [ ] Backend selector routes correctly.

### Phase 4 (Algorithm benchmarks)

- [ ] AQFT pass implemented; benchmark with ε = 1e-5, 1e-10.
- [ ] Comparison report: us vs. Qiskit Aer at n = 10, 15, 20, 22, 24, 26, 28.
- [ ] Reach ≤1.2× of Qiskit Aer single-thread.

### Phase 5 (GPU)

- [ ] cuStateVec QFT benchmark.
- [ ] Custom QFT kernel: at least one regime beats cuStateVec.

-----

## Success Metrics

A QFT optimization PR is considered successful if:

1. **Correctness**: result matches Qiskit Aer to ≤1e-10 on all sizes in [10, 28].
1. **Speed**: phase-appropriate target met (table above).
1. **Generality**: no regression on other diagonal-gate-heavy benchmarks (QPE, Grover oracle).
1. **Ceiling proximity**: ≥80% of memory bandwidth roofline (Phase 1+).

-----

## References

- Nielsen & Chuang, *Quantum Computation and Quantum Information*, §5.1.
- Coppersmith, “An approximate Fourier transform useful in quantum factoring” (1994). — AQFT.
- Häner, Roetteler, Svore, “Factoring using 2n+2 qubits with Toffoli based modular multiplication” (2017). — AQFT in practice.
- Qiskit `QFT` class: `qiskit.circuit.library.QFT`.