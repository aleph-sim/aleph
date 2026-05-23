# Playbook: QAOA (Quantum Approximate Optimization Algorithm)

> **Algorithm-specific optimization guide.** Read after `OPTIMIZATION_GUIDE.md` and `OPTIMIZATION_CYCLE.md`.

-----

## Quick Reference

|Property               |Value                                                    |
|-----------------------|---------------------------------------------------------|
|Primary backends       |MPS (shallow), State vector (deeper or smaller n)        |
|Key gates              |Rz, Rzz (cost Hamiltonian), Rx (mixer)                   |
|Gate count per layer   |O(n) Rx + O(m) Rzz, m = # edges/clauses                  |
|Circuit depth          |p layers (typically p = 1 to 10)                         |
|Entanglement           |Low for small p, grows with p                            |
|Iterations             |Like VQE — many (50–1000) optimizer steps                |
|Primary bottleneck     |Same as VQE — repeated short circuits, expectation values|
|Best-case backend match|MPS for low p, SV for moderate n                         |

**Target to beat**: Qiskit Aer + Qiskit Optimization, single-thread.

**Phase 1 success metric**: time per QAOA expectation eval within 2× of Qiskit Aer.
**Phase 4 success metric**: MPS backend handles QAOA at n = 30+ for p ≤ 3.

-----

## Algorithm Overview

Hybrid quantum-classical algorithm for combinatorial optimization. Approximates the ground state of a problem Hamiltonian H_C encoding the optimization objective.

Structure (depth p):

```
|ψ(β, γ)⟩ = U_M(β_p) U_C(γ_p) ... U_M(β_1) U_C(γ_1) |+⟩^⊗n

Where:
  U_C(γ) = exp(-iγ H_C)   # cost unitary
  U_M(β) = exp(-iβ H_M)   # mixer, typically H_M = Σ X_i
  |+⟩^⊗n is the uniform superposition
```

For Max-Cut on graph G = (V, E): H_C = Σ_{(i,j)∈E} (I − Z_i Z_j) / 2. The Z_i Z_j term decomposes to a Rzz(2γ) gate per edge.

**Why it’s a flagship benchmark**:

- Highly structured: alternating cost / mixer layers.
- The cost layer is **all diagonal** (in Z basis) — huge optimization opportunity.
- Shallow circuits → MPS friendly.
- Real industrial relevance (logistics, finance, ML).

-----

## Computational Profile

For QAOA on n qubits, m edges, p layers:

|Component               |Share of runtime|Notes                            |
|------------------------|----------------|---------------------------------|
|Cost layer (Rzz gates)  |40–60%          |All diagonal — one pass per layer|
|Mixer layer (Rx gates)  |10–20%          |Per-qubit, fast                  |
|Initialization (H wall) |<5%             |One time                         |
|Expectation value of H_C|20–30%          |Diagonal: just sum               |
|Optimizer overhead      |<5%             |                                 |

**Key insight 1**: the cost layer is **entirely diagonal** in the computational basis. The whole U_C(γ) is a single multiplicative phase application — one pass over the state vector regardless of m. This is **the** QAOA optimization.

**Key insight 2**: ⟨H_C⟩ in the computational basis is also diagonal: ⟨ψ|H_C|ψ⟩ = Σ |ψ_x|² · C(x) where C(x) is the classical cost of bitstring x. Single pass.

-----

## Optimization Ladder

### Rank 1: Algorithm — Diagonal cost layer fusion

Instead of applying Rzz gates one by one, recognize that U_C(γ) is diagonal and compute the phase per basis state:

```
phase(x) = exp(-i γ · C(x))
state[x] *= phase(x)  for each x
```

Where C(x) is the classical cost of bitstring x for the problem instance. For Max-Cut: C(x) = number of edges cut by x.

This is **one pass over the state vector regardless of m** (number of edges). Naively, m Rzz gates = m passes; with this optimization, 1 pass. Speedup: m×.

**Implementation**: a `CostHamiltonianFusion` pass in `aleph-ir`. The pass identifies “diagonal Hamiltonian evolution” patterns and emits a custom kernel that takes the cost function and applies it directly.

For graph problems, the cost function can be precomputed as a `Vec<f64>` of length 2ⁿ (cost of every bitstring), then `state[x] *= exp(-iγ · cost[x])`. Trade-off: O(2ⁿ) precomputation, but reused across all p layers and all optimizer iterations.

### Rank 2: Backend selection — MPS for low p

For p ≤ 5, entanglement is bounded; MPS with χ = 32–256 handles QAOA at n = 30, 40, 50+.

This is **the** way to scale QAOA beyond SV limits. The simulator should auto-route shallow QAOA to MPS.

**Caveat**: Rzz on non-adjacent qubits requires SWAPs in MPS, which inflate bond dimension. For dense graphs, MPS performance degrades. Best for sparse graphs.

### Rank 3: Diagonal mixer alternatives

Standard mixer H_M = Σ X_i; U_M(β) = ⊗_i Rx_i(2β). This is per-qubit, so it’s fast (n Rx gates per layer).

Some variants (e.g., XY mixer) use 2q gates; benchmark these separately.

### Rank 4: Expectation value via classical sum

⟨H_C⟩ = Σ |ψ_x|² · C(x). With the precomputed cost array (Rank 1), this is one sum over the state vector. No basis rotation needed.

**Compare** to the general Pauli grouping in VQE: QAOA’s specific structure means we never need basis rotation for measuring H_C — it’s always diagonal.

### Rank 5: Parameter caching across iterations

Same VQE trick: symbolic parameters β_1, γ_1, …, β_p, γ_p. Compile once, update per iteration.

Especially valuable for QAOA because the cost array (from Rank 1) doesn’t depend on parameters — precompute once for the entire optimization run.

### Rank 6: Gradient computation

If using parameter-shift gradients: 2 evaluations per parameter, 4p parameters → 4p+1 expectation evals per gradient step. Same caching trick as VQE applies.

### Rank 7: Memory layout, SIMD, multi-threading

Standard global optimizations apply. The diagonal phase application is highly vectorizable:

- Load amplitude pair (re, im).
- Load precomputed (cos, sin) for the phase.
- Complex multiply.
- Store.

SIMD: 4 (AVX2) or 8 (AVX-512) amplitudes per instruction.

### Rank 8: GPU

The whole pipeline maps cleanly to GPU: precompute cost array on GPU, evolve state via diagonal phase multiplications (single-pass kernels), measure via parallel reduction.

For very large n where state vector doesn’t fit: tensor network methods on GPU (cuTensorNet) become necessary.

-----

## Pitfalls

**1. Naive Rzz iteration**: applying each Rzz separately misses the m× speedup. The single biggest QAOA-specific perf bug.

**2. Cost array memory**: at n=30, the cost array is 16 GB (FP64) — too large. Workarounds:

- Compute cost on the fly per amplitude (cheap for simple problems like Max-Cut).
- Use FP32 for the cost (8 GB).
- Drop to MPS where this isn’t an issue.

**3. Wrong problem encoding**: Max-Cut, Max-3-SAT, TSP all have different Hamiltonian decompositions. Verify with a small case.

**4. Beta-gamma symmetry**: QAOA has multiple equivalent (β, γ) solutions. Don’t compare absolute parameters across runs; compare expectation values.

**5. Optimizer landscape pathology**: QAOA loss landscape has many local minima. Different random initializations give different answers. For benchmarking, fix the seed.

**6. Wrong reference**: comparing approximation ratios depends on the optimizer; comparing expectation evaluation times is more meaningful.

**7. Non-adjacent qubit gates in MPS**: dense graphs require many SWAPs; MPS becomes expensive. Sparsity matters.

-----

## Baseline Comparisons

Reference times on workstation, Max-Cut on random 3-regular graphs:

|n |p|m (edges)|Per-eval Qiskit Aer (ms)|Target Phase 1 (ms)|Target Phase 4 (ms)|
|--|-|---------|------------------------|-------------------|-------------------|
|10|1|15       |1.2                     |≤2.4               |≤1.5               |
|10|3|15       |3.5                     |≤7.0               |≤4.6               |
|16|2|24       |35                      |≤70                |≤45                |
|20|3|30       |850                     |≤1700              |≤1100              |

MPS targets at higher n (Phase 4):

- n=30, p=2: feasible on MPS, infeasible on SV. Target: ≤30 seconds per eval.
- n=40, p=2: target ≤5 minutes per eval.

-----

## Phase-by-Phase Sub-goals

### Phase 0 (Foundation)

- [ ] Rzz, Rx parametric gates implemented.
- [ ] QAOA Max-Cut at n=4, p=1 runs end-to-end.
- [ ] Expectation value of H_C matches Qiskit to 1e-12.

### Phase 1 (Single-thread CPU)

- [ ] `CostHamiltonianFusion` pass.
- [ ] Diagonal expectation primitive.
- [ ] Symbolic parameters.
- [ ] Within 2× of Qiskit per eval at n=16, p=3.

### Phase 2 (Multi-thread CPU)

- [ ] Parallel cost evolution (diagonal multiply is trivially parallel).

### Phase 3 (Alternative backends)

- [ ] MPS backend handles QAOA p=2, p=3 on sparse graphs at n=30.
- [ ] Backend selector picks MPS when (n × p × avg-degree) below threshold.

### Phase 4 (Algorithm benchmarks)

- [ ] Comparison: Max-Cut at n = 10, 14, 18, 22.
- [ ] Approximation ratio reproduces known QAOA results.
- [ ] MPS-vs-SV crossover documented.

### Phase 5 (GPU)

- [ ] GPU diagonal evolution kernel.
- [ ] Benchmark vs. cuStateVec.

-----

## Success Metrics

A QAOA optimization PR is considered successful if:

1. **Correctness**: approximation ratio matches QAOA theory for known graphs.
1. **Diagonal-layer speedup**: m× over naive Rzz-by-Rzz.
1. **MPS scaling**: at least one configuration (n=30, p=2) runs on MPS but not SV.
1. **No VQE regression**: QAOA-specific tricks shouldn’t hurt VQE.

-----

## References

- Farhi, Goldstone, Gutmann, “A Quantum Approximate Optimization Algorithm” (2014).
- Hadfield et al., “From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz” (2019).
- Crooks, “Performance of the Quantum Approximate Optimization Algorithm on the Maximum Cut Problem” (2018). — benchmarks.
- Wang et al., “XY mixers: Analytical and numerical results for QAOA” (2020). — mixer variants.
- Qiskit Optimization: <https://github.com/qiskit-community/qiskit-optimization>