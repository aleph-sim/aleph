# Playbook: Variational Quantum Eigensolver (VQE)

> **Algorithm-specific optimization guide.** Read after `OPTIMIZATION_GUIDE.md` and `OPTIMIZATION_CYCLE.md`.

VQE is **the** most common NISQ workload. Optimizing for VQE has the highest real-world payoff of any algorithm in this playbook set.

-----

## Quick Reference

|Property               |Value                                                                   |
|-----------------------|------------------------------------------------------------------------|
|Primary backends       |State vector (small-to-medium n), MPS (low-entanglement ansatze)        |
|Key gates              |Parametric: Rx, Ry, Rz, U3; entanglers: CNOT, CZ                        |
|Gate count per ansatz  |O(n · depth · entangler_density)                                        |
|Circuit depth          |Typically shallow (depth 2–10)                                          |
|Entanglement           |Varies (often low for hardware-efficient ansatze)                       |
|Iterations             |Many (100–10000+ optimizer steps)                                       |
|Primary bottleneck     |Repeated short circuits; expectation value evaluation; parameter updates|
|Best-case backend match|MPS for shallow ansatze; SV otherwise                                   |

**Target to beat**: Qiskit Aer + Qiskit Nature, single-thread, FP64.

**Phase 1 success metric**: time per expectation evaluation within 2× of Qiskit Aer.
**Phase 4 success metric**: full H₂ ground-state convergence within 1.3× of Qiskit; gradient evaluation showcases parameter-shift batching benefit.

-----

## Algorithm Overview

Variational hybrid quantum-classical algorithm. Estimates the ground-state energy of a Hamiltonian H by minimizing ⟨ψ(θ)| H |ψ(θ)⟩ over parameters θ of a parameterized ansatz.

Loop:

```
1. Prepare |ψ(θ)⟩ = U(θ) |0...0⟩
2. Measure ⟨H⟩ = Σ_i c_i · ⟨P_i⟩  where H = Σ c_i P_i (Pauli decomposition)
3. Classical optimizer suggests new θ
4. Repeat until convergence
```

For each iteration: one ansatz circuit, multiple expectation value measurements (one per Pauli string in H).

**Why it’s the flagship NISQ benchmark**:

- It’s what people actually run on real quantum hardware.
- Test cases (H₂, LiH, BeH₂) are well-studied with known answers.
- It stresses **everything that matters for short structured circuits**: parametric gate handling, fusion, expectation values, batch evaluation.

-----

## Computational Profile

For one full VQE run with N_iter optimization steps and N_paulis Hamiltonian terms:

|Component                    |Share of runtime|Notes                                                        |
|-----------------------------|----------------|-------------------------------------------------------------|
|Ansatz state preparation     |30–50%          |Same circuit shape every iteration, just different parameters|
|Expectation value evaluation |40–60%          |One pass over state vector per Pauli string                  |
|Parameter updates (classical)|<1%             |Just optimizer overhead                                      |
|Initialization, allocation   |<5%             |Should be amortized                                          |

**Key insight**: the circuit *structure* doesn’t change between iterations; only parameters. This unlocks heavy reuse: compiled circuit, fused gates (parametric fusion), even cached intermediate states.

**Expectation values dominate when |H| is large**. H₂ in STO-3G has 15 Pauli terms. Larger molecules have hundreds to thousands. Smart grouping (commuting Paulis measured together) cuts this by 3–10×.

-----

## Optimization Ladder

### Rank 1: Algorithm — Pauli grouping

Multiple Pauli strings can be measured simultaneously if they commute (or pairwise commute under a basis rotation). Standard groupings:

- **Tensor product basis** (TPB): trivially commuting (e.g., XIIX, XIIZ both diagonal in X⊗I⊗I⊗{X,Z}).
- **General commuting groups**: more powerful, requires basis rotation.

**Impact**: 3–10× reduction in number of measurement settings for chemistry Hamiltonians.

**Implementation**: a `PauliGrouper` utility in `aleph-core` that takes a list of Pauli strings and returns groups with shared measurement bases.

### Rank 2: Backend selection — MPS for shallow ansatze

Hardware-efficient ansatze with depth ≤ ~10 keep entanglement bounded. MPS with modest bond dimension χ (32–128) handles them exactly. This means **VQE on 50+ qubits** becomes possible.

The simulator should auto-detect: if ansatz depth × entangler density < threshold, route to MPS.

**Impact**: enables VQE at scales impossible for SV. For n ≤ 25, SV may still be faster due to constant factors; benchmark to find the crossover.

### Rank 3: Parametric gate compilation & caching

Across VQE iterations, the same circuit runs with different θ. We can:

- **Compile the circuit topology once**: fix qubit indices, gate dispatch, fusion structure.
- **Update parameters cheaply**: per iteration, just substitute new θ values into precomputed positions.

This requires the IR to support **symbolic parameters**. Without it, every iteration re-parses, re-fuses, re-optimizes — wasted work.

**Impact**: removes circuit compilation overhead from the inner loop. For short circuits, this can be the dominant cost.

### Rank 4: Batch parameter-shift gradients

Most VQE classical optimizers (gradient-based) compute ∂⟨H⟩/∂θ_j via the parameter-shift rule:

- E_j+ = ⟨H⟩ at θ with θ_j → θ_j + π/2
- E_j− = ⟨H⟩ at θ with θ_j → θ_j − π/2
- Gradient = (E_j+ − E_j−) / 2

For an ansatz with p parameters, this means 2p+1 expectation evaluations per gradient step.

**Optimization**: many of these circuits share structure. Cached intermediate states (after the unchanged gates before parameter θ_j) can be reused. This is the “intermediate state caching” trick.

**Impact**: for deep ansatze, ~2× speedup on gradient evaluation.

### Rank 5: Expectation value via diagonalization

For diagonal Pauli strings (all Z or all I), ⟨P⟩ = Σ |ψ_i|² · (±1) depending on parity. Single pass.

For non-diagonal: rotate basis (apply 1q gates to map P to all-Z), then sum diagonal contributions, then undo the rotation (or just rotate, measure, discard the state — depends on whether we need to reuse).

A specialized “expectation_value(state, pauli)” primitive avoids redundant rotations when running multiple Paulis on the same state.

### Rank 6: Gate fusion for parametric chains

Hardware-efficient ansatze have patterns like `Ry(θ_1) · Rz(θ_2) · Ry(θ_3)` on the same qubit. This fuses to a single U3(θ_1, θ_2, θ_3) gate. With symbolic parameters, the fusion is **symbolic**: the U3 angles are computed from the input θ each iteration.

**Implementation**: extend `Fuse1qPass` to handle parametric inputs.

### Rank 7: Memory layout, SIMD, multi-threading

Standard global optimizations (P1-01 onward). Nothing VQE-specific.

For short circuits at small n, **per-iteration overhead dominates**. Reduce: avoid heap allocations in the inner loop; reuse state vector buffers; minimize Python ↔ Rust crossings if running from Python.

### Rank 8: GPU — batched VQE

A single VQE iteration is small. The GPU is underutilized. **Batch many VQE evaluations** (e.g., for population-based optimizers, or for multiple Hamiltonians) into one GPU launch. Each lane of the GPU runs one circuit.

This is research-level and only pays off when N_circuits is large (e.g., for differential evolution with population 100).

-----

## Pitfalls

**1. Parametric gate without symbolic support**: re-parsing the OpenQASM each iteration is the most common VQE perf bug. Symbolic params are required, not optional.

**2. Allocations in the inner loop**: every `Vec::new()` per iteration kills performance for short circuits. Reuse buffers via thread-local or per-iteration arena allocation.

**3. Measuring all Paulis individually**: if there are 200 Pauli terms and you don’t group, you’re doing 5–10× more work than necessary.

**4. Mistaken Hamiltonian**: the Jordan-Wigner / Bravyi-Kitaev encoding determines the Pauli decomposition. Use a standard library (OpenFermion) to generate; don’t hand-write.

**5. Optimizer choice**: COBYLA vs. L-BFGS-B vs. SPSA vs. Adam — they have different evaluation counts and convergence properties. For benchmarking, fix one (typically COBYLA for noiseless, gradient-based for noisy).

**6. Wrong convergence criterion**: comparing “we converged to X” against another simulator is fragile (optimizers vary). Compare “time per expectation value at fixed circuit” or “wall-time to reach energy E ± ε from same θ₀”.

**7. State vector backend for too-large n**: VQE on 30 qubits with SV is wasteful. Auto-route to MPS.

-----

## Baseline Comparisons

Reference times on workstation, hardware-efficient ansatz (Ry-CNOT alternating), depth 4:

|n |Paulis (H₂)|Per-expectation Qiskit Aer (ms)|Target Phase 1 (ms)|Target Phase 4 (ms)|
|--|-----------|-------------------------------|-------------------|-------------------|
|4 |15         |0.3                            |≤0.6               |≤0.4               |
|8 |~100       |5                              |≤10                |≤6                 |
|12|~500       |80                             |≤160               |≤105               |

Larger systems: use actual chemistry Hamiltonians from PySCF / OpenFermion.

-----

## Phase-by-Phase Sub-goals

### Phase 0 (Foundation)

- [ ] Parametric gates (Rx, Ry, Rz, U3) implemented.
- [ ] Hardware-efficient ansatz works end-to-end.
- [ ] Energy of H₂ ground state matches FCI to chemical accuracy (1.6e-3 Hartree).

### Phase 1 (Single-thread CPU)

- [ ] Symbolic parameter support in IR.
- [ ] Parametric gate fusion (Ry · Rz · Ry → U3, with symbolic angles).
- [ ] Expectation value primitive with basis rotation.
- [ ] Within 2× of Qiskit Aer per expectation eval at n=4.

### Phase 2 (Multi-thread CPU)

- [ ] Parallel ansatz execution.
- [ ] Parallel expectation values across Pauli terms (one thread per Pauli).

### Phase 3 (Alternative backends)

- [ ] MPS-based VQE for n ≥ 20 with shallow ansatz.
- [ ] Backend selector chooses MPS for shallow / SV for deep.

### Phase 4 (Algorithm benchmarks)

- [ ] Pauli grouping (TPB minimum, general optional).
- [ ] Parameter-shift gradient with intermediate state caching.
- [ ] Full H₂ optimization benchmark vs. Qiskit.
- [ ] H₂O or LiH benchmark on larger n.

### Phase 5 (GPU)

- [ ] GPU expectation values.
- [ ] Optional: batched VQE on GPU.

-----

## Success Metrics

A VQE optimization PR is considered successful if:

1. **Correctness**: ground-state energy matches FCI to chemical accuracy on test molecules.
1. **Per-eval speed**: phase-appropriate target.
1. **No regression on QFT, Grover**: VQE-specific tricks shouldn’t hurt general benchmarks.
1. **Symbolic params don’t slow concrete-only paths**: if a circuit has only concrete params, the symbolic path adds zero overhead.

-----

## References

- Peruzzo et al., “A variational eigenvalue solver on a photonic quantum processor” (2014). — original VQE paper.
- McClean, Romero, Babbush, Aspuru-Guzik, “The theory of variational hybrid quantum-classical algorithms” (2016).
- Tilly et al., “The Variational Quantum Eigensolver: A review of methods and best practices” (2021). — comprehensive survey.
- Verteletskyi, Yen, Izmaylov, “Measurement optimization in the variational quantum eigensolver using a minimum clique cover” (2020). — Pauli grouping.
- OpenFermion: <https://github.com/quantumlib/OpenFermion> — chemistry Hamiltonian generation.
- Qiskit Nature: <https://github.com/qiskit-community/qiskit-nature>
- PennyLane’s VQE tutorials: <https://pennylane.ai/qml/demos/tutorial_vqe.html>