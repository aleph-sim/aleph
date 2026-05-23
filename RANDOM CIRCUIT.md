# Playbook: Random Circuits (Sycamore-style)

> **Algorithm-specific optimization guide.** Read after `OPTIMIZATION GUIDE.md` and `OPTIMIZATION CYCLE.md`.

Random circuits are the **worst-case stress test** for state vector simulation. There’s nowhere to hide.

-----

## Quick Reference

|Property               |Value                                                              |
|-----------------------|-------------------------------------------------------------------|
|Primary backends       |State vector only. MPS fails (max entanglement). Stabilizer N/A.   |
|Key gates              |Random 1q (√X, √Y, √W); 2q entanglers (iSWAP, fSim) on a brick-wall|
|Gate count             |O(n · depth)                                                       |
|Circuit depth          |Typically 10–30                                                    |
|Entanglement           |Maximum (volume-law) — no compressibility                          |
|Memory complexity      |O(2ⁿ) state vector, strict                                         |
|Primary bottleneck     |Raw kernel throughput; memory bandwidth                            |
|Best-case backend match|SV — by elimination                                                |

**Target to beat**: Qiskit Aer + cuQuantum on supremacy-style benchmarks; published Google / IBM numbers.

**Phase 1 success metric**: within 2× of Qiskit Aer single-thread at n=25, depth=20.
**Phase 4 success metric**: within 1.3× of cuStateVec on GPU at n=30+.

-----

## Algorithm Overview

Random circuits sample uniformly (or close to it) from the unitary group. They serve as:

- Benchmarks for “quantum supremacy” / “quantum advantage” experiments.
- Linear cross-entropy benchmarking (XEB) basis.
- Worst-case stress test for simulators (since they produce maximum entanglement).

Standard structure (Sycamore-style):

```
Initial layer: random 1q gate on each qubit.
For each cycle (1 to depth):
  Random 1q gate on each qubit.
  2q entangling layer on a brick-wall pattern of pairs.
```

1q gate alphabet: {√X, √Y, √W} where √W = √((X+Y)/√2). Three distinct gates ensure circuits don’t fall into special subspaces (e.g., Clifford-only).

2q entanglers: iSWAP-like or fSim gates. NOT just CNOTs (which are Clifford and would let stabilizer methods simulate).

-----

## Computational Profile

For random_n_d (n qubits, depth d):

|Component       |Share of runtime|Notes                                       |
|----------------|----------------|--------------------------------------------|
|1q gates        |30–40%          |Generic 1q kernel (not specialized)         |
|2q entanglers   |50–60%          |Generic 2q kernel; fSim has 4×4 dense matrix|
|Init/measurement|<5%             |One H per qubit (or random init)            |

**No diagonal optimizations apply**. Random 1q gates aren’t Pauli-X / Z / H — they’re random 2×2 unitaries. Random 2q gates aren’t CNOT / CZ — they’re general 4×4. The hot path is the **generic 1q and 2q kernels** at full strength.

**This is what tests every other optimization**. If the generic kernels are slow, every benchmark suffers. Random circuits expose the truth.

-----

## Optimization Ladder

### Rank 1: Algorithm — N/A (this is the worst case)

You can’t algorithmically beat random circuits except by approximation (low-rank methods, tensor network contraction with bounded error). For honest state vector benchmarks, the algorithm is fixed.

For **tensor network contraction** approaches (Google’s “supremacy” verification techniques): different game; not state vector simulation. Use cuTensorNet for these workloads.

### Rank 2: Backend selection — State vector only

For honest benchmarks, lock to SV. Document if any other backend is used.

For **research** (low-rank approximation, etc.): see references below for techniques.

### Rank 3: IR-level — Gate fusion

Random circuits have many 1q gates interleaved with 2q gates. Fusion still helps:

- 1q · 2q · 1q on the same pair → one 4×4 matrix.
- Two consecutive 1q gates on the same qubit between 2q layers → one 1q gate.

Typical fusion gain: 1.3–1.8× on Sycamore-style circuits.

### Rank 4: Generic kernel optimization

Random circuits live or die on generic kernels. Every micro-optimization here pays off:

- SoA layout (P1-01).
- Bit-twiddling indexing (P1-02).
- AVX2 / AVX-512 (P1-03, P1-04).
- For 2q kernels: tile the inner loop for L1 cache (typical state vector chunk = 4 amplitudes / quadruplet × N quadruplets).

This is **the** rank where random circuits define success. If you’re 50% of Qiskit Aer on random circuits, fix this before anything else.

### Rank 5: Memory access pattern — block by qubit

For 2q gates between low-index qubits (e.g., qubits 2 and 3), amplitude quadruplets are close in memory — cache-friendly. For high-index qubits (e.g., qubits 23 and 24), strides are huge — cache-hostile.

**Optimization**: process amplitudes in blocks tuned to L2 cache size. For qubits where the stride exceeds cache, use software prefetching.

### Rank 6: Multi-threading

Standard rayon parallelism. Random circuits scale well because there’s no shared state during gate application.

### Rank 7: GPU

GPU shines on random circuits. The 4090’s HBM gives 1 TB/s bandwidth vs. 80 GB/s on CPU. 12× speedup is realistic.

cuStateVec is highly tuned for this; competing with it requires careful work.

### Rank 8: Multi-GPU & distributed

For n ≥ 33 (≥64 GB state vector), single GPU isn’t enough. Multi-GPU state vector partitioning. The all-to-all communication for 2q gates on global qubits is the bottleneck.

-----

## Pitfalls

**1. Lucky cancellations**: a “random” circuit chosen with a deterministic seed may have accidental structure. Use proper RNG with documented seed.

**2. Non-random 1q gates**: if all 1q gates are Clifford (e.g., {H, S, X}), the whole circuit is Clifford and stabilizer simulators trivially win. Use non-Clifford alphabet.

**3. Brick-wall depth confusion**: depth = number of 2q layers, or number of (1q + 2q) layers, depending on convention. Document yours; align with reference papers.

**4. Comparing to “supremacy” experiments unfairly**: Google’s Sycamore uses 53 qubits depth 20; that’s ~3 ExaFLOPs of work. You won’t beat it on consumer hardware. Compare on equivalent-cost benchmarks.

**5. XEB metric**: linear cross-entropy benchmarking computes a fidelity-like score from samples. Useful for verifying our simulator (XEB ≈ 1 means noiseless). Don’t confuse it with execution time.

**6. Skipping the SWAP issue**: in fixed-topology hardware, random circuits need routing. We’re simulating, so we use direct gates without SWAP overhead. Don’t accidentally add routing penalties.

-----

## Baseline Comparisons

Reference times on workstation (Ryzen 9 7950X, single-thread, FP64), random_n_d:

|n |depth|Qiskit Aer (ms)|Target Phase 1 (ms)|Target Phase 4 (ms)|
|--|-----|---------------|-------------------|-------------------|
|15|10   |12             |≤24                |≤15                |
|20|15   |250            |≤500               |≤320               |
|25|20   |12,000         |≤24,000            |≤15,600            |
|28|20   |130,000        |≤260,000           |≤170,000           |

GPU targets (RTX 4090, Phase 5):

- n=28, depth=20: cuStateVec ~3,500 ms; target ≤4,500 ms.
- n=30, depth=20: cuStateVec ~15,000 ms; target ≤19,500 ms.

-----

## Phase-by-Phase Sub-goals

### Phase 0 (Foundation)

- [ ] Random circuit generator with documented seed.
- [ ] Benchmark fixtures: random_15_10, random_20_15, random_25_20.
- [ ] XEB computation: matches Qiskit Aer to 1e-10 fidelity.

### Phase 1 (Single-thread CPU)

- [ ] Generic 1q and 2q kernels are SIMD-optimized (P1-03, P1-07).
- [ ] Gate fusion across 1q-2q boundaries (P1-10).
- [ ] Within 2× of Qiskit Aer at n=20, depth=15.

### Phase 2 (Multi-thread CPU)

- [ ] Parallel random circuit at n=25, depth=20: ≥6× on 8 cores.

### Phase 3 (Alternative backends)

- [ ] Confirm MPS *fails* gracefully (high entanglement → χ blows up → out-of-memory).
- [ ] Confirm stabilizer rejects (non-Clifford gates → error).

### Phase 4 (Algorithm benchmarks)

- [ ] XEB benchmark at n = 20, 24, 28.
- [ ] Comparison report vs. Qiskit Aer single + multi thread.

### Phase 5 (GPU)

- [ ] cuStateVec random circuit benchmark.
- [ ] Custom kernel within 1.3× of cuStateVec.
- [ ] At n=30 with 64 GB system: confirm RAM exhausts before runtime.

### Phase 6 (Multi-GPU)

- [ ] n=32, 33 across 2 H100s.
- [ ] n=35+ across nodes via MPI.

-----

## Success Metrics

A random circuit optimization PR is considered successful if:

1. **Correctness**: XEB ≈ 1 (within statistical bounds); amplitudes match reference to 1e-10.
1. **Speed**: phase-appropriate target met.
1. **Generic kernel quality**: any improvement here benefits *every other algorithm*.
1. **Memory ceiling**: simulator gracefully refuses requests above available RAM, with clear error.

-----

## Special Considerations

### Statistical sampling vs. full state vector

For very large n, even one full state vector is impractical. Real “supremacy” experiments work by:

1. Sampling bitstrings from the circuit (real or simulated).
1. Computing per-sample amplitudes via tensor network contraction.

This avoids materializing the state vector. We may add a “single-amplitude” mode (Phase 5+) for research workloads. The methodology is separate from state vector simulation and out of scope for the main benchmarks.

### Approximate methods

Various research methods (low-rank truncation, partial state vector, Pauli path simulation) trade accuracy for scale. These are advanced; document carefully if used in benchmarks.

-----

## References

- Arute et al. (Google), “Quantum supremacy using a programmable superconducting processor” (2019).
- Boixo et al., “Characterizing quantum supremacy in near-term devices” (2018).
- Pednault et al. (IBM), “Pareto-Efficient Quantum Circuit Simulation Using Tensor Network Contraction” (2020).
- Markov, Shi, “Simulating Quantum Computation by Contracting Tensor Networks” (2008).
- Liu et al. “Closing the ‘quantum supremacy’ gap” (2021).
- cuStateVec random circuit benchmarks: <https://docs.nvidia.com/cuda/cuquantum/>