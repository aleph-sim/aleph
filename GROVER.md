# Playbook: Grover’s Algorithm

> **Algorithm-specific optimization guide.** Read after `OPTIMIZATION GUIDE.md` and `OPTIMIZATION CYCLE.md`.

-----

## Quick Reference

|Property               |Value                                                   |
|-----------------------|--------------------------------------------------------|
|Primary backends       |State vector (dense). MPS struggles (high entanglement).|
|Key gates              |H, X, Z, multi-controlled Z (oracle), Hadamard wall     |
|Gate count             |O(√N · n) for n qubits                                  |
|Circuit depth          |O(√N · depth_oracle)                                    |
|Entanglement           |High during iterations                                  |
|Memory complexity      |O(2ⁿ) state vector                                      |
|Primary bottleneck     |Repeated oracle + diffusion; multi-controlled gates     |
|Best-case backend match|SV; stabilizer if oracle is Clifford (rare)             |

**Target to beat**: Qiskit Aer `statevector` simulator, single-thread, FP64.

**Phase 1 success metric**: within 2× of Qiskit Aer for grover_16.
**Phase 4 success metric**: within 1.3× of Qiskit Aer; oracle-call amortization shows ≥1.5× over naive.

-----

## Algorithm Overview

Find marked items in an unstructured search space of size N = 2ⁿ using O(√N) oracle queries.

Structure:

```
1. Initialize: |ψ⟩ = H^⊗n |0⟩^⊗n  (uniform superposition)
2. Repeat √N times:
     a. Apply oracle O: flips sign of marked states.
     b. Apply diffusion D: 2|ψ_init⟩⟨ψ_init| − I.
3. Measure.
```

Diffusion is implemented as `H^⊗n · (2|0⟩⟨0| − I) · H^⊗n`. The middle operator is “flip the sign of |0⟩^⊗n”, typically realized as `X^⊗n · (multi-controlled-Z) · X^⊗n`.

**Why it’s a flagship benchmark**:

- Repeated structure → great test of fusion and caching.
- Multi-controlled gates → stresses MCX/MCZ specializations.
- Real-world relevance (search, optimization).

-----

## Computational Profile

For `grover_n` with k = √N iterations:

|Component     |Share of runtime                     |Notes                                           |
|--------------|-------------------------------------|------------------------------------------------|
|Oracle        |30–60% (depends on oracle complexity)|Domain-specific; can be Clifford or non-Clifford|
|Diffusion     |30–50%                               |Fixed structure: H wall + MCZ + H wall          |
|Hadamard walls|10–20%                               |All-qubit Hadamard layers                       |
|Initialization|<1%                                  |One H wall                                      |

**Multi-controlled gates** are the hot spot inside diffusion. For n=20, MCZ on 20 qubits naively decomposes to ~40 CNOTs + T gates, which means 40+ state vector passes per iteration. Specialized MCZ kernel cuts this to 1 pass.

**Iteration count**: ~⌊π/4 · √N⌋. At n=20, ~804 iterations. The same circuit pattern repeats; cache effects are favorable if working set fits in L2/L3.

-----

## Optimization Ladder

### Rank 1: Algorithm — Amplitude amplification fusion

Multiple Grover iterations can be analyzed as a single rotation in the 2D subspace spanned by `|marked⟩` and `|unmarked⟩`. If we know we’re running k iterations, sometimes we can simulate the final rotation directly (skipping iteration-by-iteration evolution).

**Caveat**: this is “cheating” if we’re benchmarking the simulator’s per-gate cost. Use it for the application user, not for fair simulator comparisons.

### Rank 2: Specialized multi-controlled gate kernel

MCZ on n qubits: only one amplitude (the `|1...1⟩` state) gets its sign flipped. Naive: O(2ⁿ) work. Specialized: O(1) work — just negate `state[N-1]`.

More generally, MCZ on a subset S of qubits flips the sign of amplitudes where all bits in S are 1. Specialized kernel iterates these amplitudes directly (no full sweep).

**Impact**: massive. At n=20, MCZ goes from ~40 CNOT decompositions (40+ passes) to one negation, or ~5 specialized 2-bit-mask passes. Speedup: 10–40×.

This is the single biggest QGrover-specific win.

### Rank 3: Fused oracle + diffusion

The diffusion `H · X · MCZ · X · H` (on each qubit, applied as walls) has fixed structure. Hadamard wall + X wall are both pure 1q operations that can be fused with the multi-controlled gate’s surroundings.

**Concretely**: instead of n Hadamards + n X gates + MCZ + n X gates + n Hadamards (4n + 1 operations), express the whole diffusion as a single “Grover diffusion” composite operation with a custom kernel.

The composite kernel:

- Computes `mean = (Σ state[i]) / N`.
- Sets `state[i] = 2·mean − state[i]` for each amplitude.

This is O(2ⁿ) total work for the whole diffusion, vs. O(n · 2ⁿ) for the decomposed version. Speedup: n× per diffusion.

**Implementation**: register a “Grover diffusion” pattern matcher in the IR; when detected, emit a fused composite op.

### Rank 4: Oracle structure exploitation

Different oracles have different costs:

- **Bit-string match oracle** (mark single state): just flips one amplitude. O(1).
- **Polynomial oracle** (mark states satisfying a Boolean formula): the formula can sometimes be evaluated more efficiently than gate-by-gate.

For benchmarking, standardize on bit-string match oracles (simplest, well-defined). For research workloads, oracle compilation deserves its own optimization pass.

### Rank 5: Iteration caching

Each iteration applies the same sequence of operations. We can:

- Precompute fused gate matrices once.
- Stay in cache: if the state vector fits in L3, iteration overhead drops drastically.

**Cache fit threshold**: at n=20, state vector = 16 MB; fits in 32 MB L3. At n=22, 64 MB — overflows. The benchmark should test both regimes.

### Rank 6: Memory layout, SIMD

Standard SoA + SIMD applies (as in P1-01, P1-03). The diffusion’s mean-and-reflect pattern is highly vectorizable.

### Rank 7: Multi-threading

Both oracle and diffusion are embarrassingly parallel over state vector chunks. Standard rayon chunking.

For the mean computation in diffusion: parallel reduction, then broadcast. Avoid race conditions on the accumulator.

### Rank 8: GPU

GPU diffusion: standard reduction + map. Well-suited for GPU. The repeated structure means we can keep state on GPU across iterations.

-----

## Pitfalls

**1. Wrong iteration count**: too few iterations under-amplifies; too many overshoots and decreases marked probability. Optimal is `⌊π/4 · √(N/M)⌋` where M is the number of marked states. Verify by checking the marked-state probability.

**2. Oracle correctness**: easy to flip the wrong bit. Test each new oracle with a tiny case (n=3, manually verify) before benchmarking.

**3. Decomposed MCZ tests**: if your specialized MCZ disagrees with the decomposed reference, debug *both* — both implementations are easy to get wrong.

**4. Floating-point drift**: after many iterations, accumulated FP error can be visible. Renormalize after every few iterations if you observe drift > 1e-8.

**5. Comparing against “Grover with measurement”**: some implementations measure between iterations (wrong); some don’t. Use noiseless simulation without intermediate measurement for benchmarks.

**6. Numerical underflow of marked amplitude**: for very small marked sets, intermediate amplitudes can be tiny. Use FP64; FP32 may not suffice.

-----

## Baseline Comparisons

Reference times on workstation (Ryzen 9 7950X, single-thread, FP64), bit-string match oracle, full iteration count:

|n |Iterations|Qiskit Aer (ms)|Target Phase 1 (ms)|Target Phase 4 (ms)|
|--|----------|---------------|-------------------|-------------------|
|8 |12        |0.6            |≤1.2               |≤0.8               |
|12|50        |22             |≤44                |≤29                |
|16|201       |480            |≤960               |≤624               |
|20|804       |14,500         |≤29,000            |≤18,800            |

Regenerate from reference machine before relying on these.

-----

## Phase-by-Phase Sub-goals

### Phase 0 (Foundation)

- [ ] Naive Grover up to n=10 against bit-string oracle.
- [ ] Marked-state probability >0.95 after correct iteration count.
- [ ] Result matches Qiskit Aer to 1e-12.

### Phase 1 (Single-thread CPU)

- [ ] Specialized MCZ kernel.
- [ ] Composite “Grover diffusion” pattern in IR.
- [ ] AVX2 for diffusion reflection.
- [ ] Within 2× of Qiskit Aer at n=16.

### Phase 2 (Multi-thread CPU)

- [ ] Parallel diffusion (mean reduction + reflect).
- [ ] ≥6× scaling on 8 cores at n=20.

### Phase 3 (Alternative backends)

- [ ] N/A — Grover doesn’t benefit from MPS or stabilizer.

### Phase 4 (Algorithm benchmarks)

- [ ] Benchmark suite includes n = 8, 12, 16, 20.
- [ ] Iteration caching documented.
- [ ] Comparison report vs. Qiskit Aer at all sizes.

### Phase 5 (GPU)

- [ ] GPU diffusion kernel.
- [ ] State stays on GPU across iterations.
- [ ] Benchmark vs. cuStateVec.

-----

## Success Metrics

A Grover optimization PR is considered successful if:

1. **Correctness**: marked-state probability ≥0.95 (theoretical optimum) after correct iteration count; matches reference to 1e-10.
1. **Speed**: phase-appropriate target met.
1. **Multi-controlled performance**: MCZ kernel runs in O(1) for single-marked oracles, O(2^|S|) for subset-marked.
1. **Diffusion fusion**: composite diffusion runs in 1 pass over state vector.

-----

## References

- Grover, “A fast quantum mechanical algorithm for database search” (1996).
- Brassard, Høyer, Mosca, Tapp, “Quantum Amplitude Amplification and Estimation” (2000).
- Nielsen & Chuang, §6.
- Qiskit tutorial: <https://qiskit.org/textbook/ch-algorithms/grover.html>