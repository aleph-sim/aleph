# Playbook: Stabilizer Circuits (Surface Code & QEC)

> **Algorithm-specific optimization guide.** Read after `OPTIMIZATION GUIDE.md` and `OPTIMIZATION CYCLE.md`.

This playbook covers **Clifford circuits** — circuits using only `{H, S, CNOT, measurement}` and the Pauli gates. The killer app is **quantum error correction** simulation (surface code, repetition code, etc.).

-----

## Quick Reference

|Property               |Value                                                                   |
|-----------------------|------------------------------------------------------------------------|
|Primary backend        |Stabilizer (Aaronson-Gottesman tableau). SV is wrong choice.            |
|Key gates              |H, S (and Sdg), CNOT, Pauli X/Y/Z, measurements                         |
|Gate count             |Often huge (10⁴ to 10⁶+) — QEC cycles repeat many times                 |
|Qubit count            |Often huge (100s to 1000s) — physical qubits in QEC code                |
|Entanglement           |High (stabilizer states are highly entangled, just specially structured)|
|Memory complexity      |**O(n²)** — NOT exponential! This is the magic.                         |
|Primary bottleneck     |Tableau row operations during gate updates                              |
|Best-case backend match|Stabilizer, **always**. Wrong backend = orders of magnitude slowdown.   |

**Target to beat**: Stim (Craig Gidney’s simulator). Stim is the gold standard for stabilizer simulation.

**Phase 3 success metric**: within 3× of Stim on surface code cycles at distance d=5.
**Phase 4 success metric**: within 1.5× of Stim at d=11, and handle d=21+ for research workloads.

-----

## Algorithm Overview

The Aaronson-Gottesman stabilizer formalism represents a quantum state by its stabilizer group: the set of Pauli operators that leave the state invariant. For n qubits, the state is uniquely determined by n commuting independent stabilizers.

Encoded as a tableau:

```
     X_1 X_2 ... X_n  |  Z_1 Z_2 ... Z_n  |  phase
S_1  [ ... binary ... ]  [ ... binary ... ]   [bit]
S_2  ...
...
S_n  ...
D_1  [destabilizers ...]
...
D_n  
```

Each row is a Pauli operator encoded as 2n bits (X part, Z part) plus a sign bit. Gates update this tableau:

- **H on qubit q**: swap X and Z columns for qubit q across all rows.
- **S on qubit q**: replace Z bit with (X bit XOR Z bit) for qubit q.
- **CNOT(c, t)**: row updates per Aaronson-Gottesman §3.
- **Measurement**: Gaussian elimination over GF(2); either deterministic or random outcome.

Gate complexity: O(n) per Clifford gate. Total complexity: O(n²) memory, O(n · #gates) time.

**Why this is the killer app**:

- Surface code uses thousands of physical qubits with millions of gates per logical operation.
- State vector for 1000 qubits = 10^300 amplitudes. Impossible.
- Stabilizer for 1000 qubits = ~1 MB tableau. Trivial.

-----

## Computational Profile

For a surface code distance-d cycle (≈ 2d² physical qubits, ~O(d²) gates per cycle):

|Component    |Share of runtime|Notes                                              |
|-------------|----------------|---------------------------------------------------|
|CNOT updates |60–80%          |Most common gate in QEC; tableau row XOR operations|
|H, S updates |10–20%          |Cheap per-gate                                     |
|Measurements |10–20%          |Gaussian elimination — most expensive operation    |
|Pauli updates|<5%             |Just sign bit changes                              |

**Bottleneck character**: tableau operations are **memory-light** (it fits in L1/L2 cache for moderate n) but require fast bitwise operations. The hot loop is XOR’ing rows of bits.

For very large n (n > 10,000), the tableau exceeds L3; behavior shifts to memory-bound.

-----

## Optimization Ladder

### Rank 1: Algorithm — Choosing this backend at all

**This is the entire ROI**. If a Clifford circuit accidentally runs on SV, it’s 10⁹–10⁶⁰⁰× slower than necessary. Auto-routing is the most valuable single optimization in the entire project for QEC workloads.

Implementation: scan circuit gates; if all are in `{H, S, Sdg, CNOT, X, Y, Z, measurement, reset}`, route to stabilizer.

### Rank 2: Bit-packed tableau representation

The tableau is 2n+1 rows of 2n+1 bits each. Pack into `u64` words (64 bits per word). Row operations become XOR over arrays of u64.

For n=1000: 2000 bits per row ÷ 64 = 32 u64 per row. CNOT update XORs ~2-3 rows ≈ 100 u64 operations. Sub-microsecond per CNOT.

**Implementation**: `BitVec` or hand-rolled `Vec<u64>`. The latter is typically 1.5–2× faster due to inlining and avoiding the safety overhead.

### Rank 3: SIMD for bit operations

XOR’ing two `Vec<u64>` of length L: trivially SIMD’able. AVX2: 4 u64 per instruction. AVX-512: 8 u64 per instruction.

For n=1000, 32 u64 per row → 4 AVX-512 ops per row XOR. Single-digit nanoseconds per gate.

### Rank 4: Cache-friendly tableau layout

The tableau is conceptually 2D (rows × columns). Layout options:

- **Row-major**: row XOR is contiguous; column extraction (for measurement) is strided. Default; good for most operations.
- **Column-major**: opposite trade-off. Bad for CNOT, good for measurement.
- **Bit-sliced**: separate `u64` arrays per bit position. Some operations become trivial; others awkward.

Stim uses row-major bit-packed; we should start there.

### Rank 5: Measurement optimization

Measurement is the most expensive operation: find a stabilizer (or destabilizer) anticommuting with the measured Pauli, then do row reduction.

Optimization: maintain the tableau in “reduced” form proactively, so measurements are fast at the cost of slightly slower gate updates. Stim does this; we should evaluate the trade-off.

### Rank 6: Batched simulations

For Monte Carlo QEC studies, the same circuit runs thousands of times with different noise realizations. Batching:

- Multiple shots in one tableau (pack m shots into m bit positions of each u64).
- SIMD across shots.

**Impact**: 10–100× for Monte Carlo workloads. This is how Stim achieves its speed for QEC decoder benchmarking.

### Rank 7: Multi-threading

Within a single tableau: limited parallelism (most operations are inherently sequential or have small parallelism).
Across multiple shots: trivially parallel. Use rayon.

### Rank 8: GPU stabilizer

Research-level (P5-07). The tableau on GPU: bit-matrix in HBM, row XOR as a kernel. Theoretical speedup limited by the small per-op work.

The interesting GPU win: massively batched shots (10⁴–10⁶ shots in parallel). Useful for QEC decoder ML training.

-----

## Pitfalls

**1. Routing non-Clifford circuits to stabilizer**: stabilizer can’t simulate T, Rx(θ), etc. Strict gate validation needed. Clear error message on rejection.

**2. Phase tracking bugs**: the sign bit of stabilizers is easy to get wrong. Test against Stim on small circuits.

**3. Measurement determinism**: when a measurement is deterministic (the Pauli is in the stabilizer group), there’s no randomness. When it’s random (Pauli anticommutes), use a proper RNG. Mixing these breaks circuits.

**4. Tableau row ordering**: the order of stabilizers in the tableau is implementation-defined. Don’t compare tableaus directly; compare equivalent stabilizer groups.

**5. Decomposing non-Clifford gates**: tempting to approximate T as Clifford for “free” stabilizer simulation. **Don’t.** This is wrong (T isn’t Clifford). For T-gate simulation: stabilizer-rank methods (research-level).

**6. “Stabilizer rank” methods**: extend stabilizer to handle bounded T gates. Linear pay in T count, but very fast for small T counts. Out of scope for v1.0; consider for v2.

**7. Reset / re-initialize**: stabilizer can simulate `reset` (project + flip if needed). Implementation detail; verify correctness vs. Stim.

-----

## Baseline Comparisons

Reference times on workstation, surface code 1-cycle (X and Z stabilizer measurements):

|Distance d|# physical qubits|# gates|Stim (ms)|Target Phase 3 (ms)|Target Phase 4 (ms)|
|----------|-----------------|-------|---------|-------------------|-------------------|
|3         |17               |~50    |0.05     |≤0.15              |≤0.075             |
|5         |49               |~150   |0.2      |≤0.6               |≤0.3               |
|7         |97               |~300   |0.5      |≤1.5               |≤0.75              |
|11        |241              |~750   |2        |≤6                 |≤3                 |
|21        |881              |~2700  |12       |≤36                |≤18                |

For Monte Carlo (1000 shots, batched in Stim):

- d=5, 1000 shots, full memory: Stim 20 ms; target ≤60 ms (Phase 3), ≤30 ms (Phase 4).

-----

## Phase-by-Phase Sub-goals

### Phase 0 (Foundation)

- [ ] N/A — stabilizer backend not yet built.

### Phase 1, 2 (CPU optimization)

- [ ] N/A — stabilizer in Phase 3.

### Phase 3 (Alternative backends)

- [ ] Aaronson-Gottesman tableau with bit-packed `Vec<u64>` rows (P3-01).
- [ ] Measurement with Gaussian elimination (P3-02).
- [ ] Backend trait integration (P3-03).
- [ ] Within 3× of Stim on d=5 surface code cycle.
- [ ] Verified against Stim on 100+ random Clifford circuits.

### Phase 4 (Algorithm benchmarks)

- [ ] Surface code 1-cycle benchmark for d = 3, 5, 7, 9, 11 (P4-07).
- [ ] SIMD bit operations.
- [ ] Within 1.5× of Stim at d=11.

### Phase 5 (GPU — optional, P5-07)

- [ ] GPU stabilizer (research): batched Monte Carlo shots.
- [ ] At least one regime where GPU beats CPU Stim.

-----

## Success Metrics

A stabilizer optimization PR is considered successful if:

1. **Correctness**: matches Stim on shared test circuits (100+ random Clifford, surface code).
1. **Speed**: phase-appropriate target met.
1. **Memory**: tableau size = O(n²) with bit packing; no surprises.
1. **Validation**: non-Clifford gates rejected clearly.

-----

## Domain Notes

### What can stabilizer simulate?

- **Yes**: all Clifford circuits — H, S, CNOT, Pauli, measurements, resets.
- **No**: T, Rz(θ) for general θ, Toffoli (CCX), arbitrary 1q unitaries.
- **Stabilizer rank methods can**: T + Clifford with cost exponential in T-count (small T-count is fine).

### Surface code basics

Logical qubit encoded across d² ≈ d² physical data qubits, with ancilla qubits for syndrome measurement. Each “cycle” measures all stabilizers, producing a syndrome. Decoder corrects errors based on syndromes.

For simulation purposes: surface code circuits are large Clifford circuits with measurements. Standard stabilizer simulator handles them.

For noise simulation: add Pauli errors after each gate (still Clifford!). Stabilizer simulator handles noisy Clifford circuits efficiently.

### Why not write our own decoder?

Decoders (MWPM, union-find, ML-based) consume syndromes from the simulator and predict corrections. They’re separate from the simulator. PyMatching / Stim’s `Decode` cover this; we focus on the simulator side.

-----

## References

- Aaronson, Gottesman, “Improved Simulation of Stabilizer Circuits” (2004). — **The** reference for the algorithm.
- Gidney, “Stim: a fast stabilizer circuit simulator” (2021). — Stim paper, also documents tricks.
- <https://github.com/quantumlib/Stim> — Stim implementation; read the source.
- Fowler, Mariantoni, Martinis, Cleland, “Surface codes: Towards practical large-scale quantum computation” (2012).
- Bravyi, Gosset, “Improved Classical Simulation of Quantum Circuits Dominated by Clifford Gates” (2016). — stabilizer rank methods.
- PyMatching: <https://github.com/oscarhiggott/PyMatching> — decoder, useful pair to stabilizer simulator.