# Quantum Simulator — Project Roadmap

> A high-performance quantum circuit simulator written in Rust, with pluggable backends, CUDA acceleration, and a long-term path toward distributed multi-GPU execution.

-----

## 1. Project Vision

Build a quantum circuit simulator that:

1. Is **correct first** — produces results indistinguishable from reference simulators (Qiskit Aer, Stim) on shared benchmarks.
1. Is **competitive in a niche** — at least one regime (e.g., MPS for shallow circuits, stabilizer for QEC, single-GPU dense for medium scale) where it matches or beats existing tools.
1. Has a **clean, extensible architecture** — pluggable backends, backend-agnostic IR, OpenQASM 3.0 as the lingua franca.
1. Is **CPU-first, then GPU, then distributed** — each layer fully optimized and benchmarked before moving to the next.

This is a long-term, evolving project. It is not “yet another Python wrapper” — the goal is to produce a tool that researchers and educators actually choose for specific workloads.

-----

## 2. Why This Exists / Differentiation

The quantum simulation space is crowded (Qiskit Aer, PennyLane Lightning, Cirq, Stim, Intel-QS, cuQuantum). Building something useful means picking deliberate angles:

- **Rust-native core** with a clean Python frontend via `pyo3` — most fast simulators are C++; Rust gives memory safety and fearless concurrency for the threading-heavy parts.
- **Backend-agnostic IR** with automatic backend selection — analyze the circuit (Clifford-only? shallow? structured?) and pick the optimal engine (stabilizer / MPS / dense state vector).
- **Honest benchmarks from day one** — every optimization PR includes before/after numbers vs. Qiskit Aer and others.
- **Optimized for the algorithms people actually run** — VQE, QAOA, QFT, Grover, surface code cycles — not just synthetic benchmarks.

We do not try to beat NVIDIA cuQuantum at dense state vector on a single H100. We integrate it as a backend. We try to beat it (or complement it) where it does not optimize: stabilizer on GPU, MPS heuristics, hybrid approaches.

-----

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Frontends                                              │
│  • Rust API   • Python API (pyo3)   • CLI               │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Parser & Frontend Language                             │
│  • OpenQASM 3.0   • Native Rust DSL                     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Circuit IR (backend-agnostic)                          │
│  • Gate sequence  • Metadata  • Symbolic parameters     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  IR Optimization Passes                                 │
│  • Gate fusion  • Cancellation  • Dead code             │
│  • Commutation analysis  • Routing                      │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Backend Selection (heuristic + manual override)        │
└──┬───────────┬───────────┬───────────┬───────────┬─────┘
   │           │           │           │           │
   ▼           ▼           ▼           ▼           ▼
┌──────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐
│Dense │  │  MPS   │  │Stabili-│  │Decision│  │ Future:  │
│ SV   │  │ tensor │  │  zer   │  │diagram │  │ hybrid,  │
│(CPU/ │  │network │  │(Clifford│ │        │  │ multi-GPU│
│ GPU) │  │        │  │ + low-T)│ │        │  │ MPI      │
└──────┘  └────────┘  └────────┘  └────────┘  └──────────┘
```

Each backend implements a common `Backend` trait. The frontend, IR, parser, and optimization passes are all shared.

-----

## 4. Technology Stack

**Core language**: Rust (edition 2021+).

**Key crates**:

- `num-complex` — complex number primitives
- `rayon` — data parallelism
- `criterion` — benchmarking
- `proptest` — property-based testing
- `pyo3` + `maturin` — Python bindings
- `cudarc` or raw `cust` — CUDA bindings
- `mpi` — MPI bindings (for distributed phase)
- `clap` — CLI parsing

**External dependencies**:

- OpenQASM 3.0 (input format)
- cuQuantum (cuStateVec, cuTensorNet) — integrated as a backend
- Qiskit (Python, used as oracle for correctness tests)
- Stim (used as oracle for stabilizer correctness)

**Build / CI**:

- GitHub Actions
- Criterion benchmark regression detection
- Cross-platform testing (Linux primary, macOS, Windows)

-----

## 5. Phases Overview

|Phase|Goal                                  |Duration (full-time)|Key Deliverable                                                                  |
|-----|--------------------------------------|--------------------|---------------------------------------------------------------------------------|
|0    |Foundation: structure, naive simulator|2–3 weeks           |End-to-end pipeline: parser → IR → naive backend → measurement                   |
|1    |Single-threaded CPU optimization      |3–4 weeks           |SIMD-optimized state vector backend, gate fusion, ≤2× of Qiskit Aer single-thread|
|2    |Multi-threaded CPU                    |2 weeks             |Near-linear scaling on 16+ cores                                                 |
|3    |Alternative backends (Stabilizer, MPS)|4–6 weeks           |Stabilizer + MPS backends working, auto-selection heuristic                      |
|4    |Algorithm benchmarks + first release  |1–2 weeks           |Public benchmark report, v0.1 release                                            |
|4.5  |CPU parity vs Aer/Stim                |2–4 weeks           |Every parity-matrix cell ≤ 1.2× its reference; docs/perf/parity.md|
|5    |GPU backend (single-GPU)              |2–3 months          |cuQuantum integration + custom CUDA where it adds value                          |
|6    |Multi-GPU and distributed             |2–3 months          |NCCL intra-node + MPI inter-node                                                 |

Estimates are aggressive solo full-time. Realistic part-time: roughly double.

-----

## 6. Algorithm Coverage (Benchmark Suite)

The simulator is evaluated on these algorithms across all phases:

**Tier 1 — must work from Phase 0**:

- Bell / GHZ state preparation
- Quantum Fourier Transform (QFT)
- Grover’s algorithm
- Random circuits (Google supremacy-style)

**Tier 2 — added in Phase 4**:

- Quantum Phase Estimation (QPE)
- VQE with hardware-efficient ansatz (e.g., H₂ ground state)
- QAOA on Max-Cut
- Surface code 1-cycle (stabilizer showcase)

**Tier 3 — research targets, later phases**:

- Shor’s algorithm (small N)
- Hamiltonian simulation (Trotter-Suzuki)
- Quantum kernels / QML circuits
- Amplitude estimation

-----

## 7. Success Metrics

**Per phase**:

- Phase 0 ✅ **met**: 25-qubit GHZ runs end-to-end, all property tests pass, benchmark harness produces reports. See `docs/perf/phase0.md`.
- Phase 1 ✅ **met**: single-thread time within 2× of Qiskit Aer for QFT, Grover, random circuits at 25 qubits. All 16 matrix cells ≤ 2× Aer on EPYC (worst: qft_n25 = 1.73×; aleph is *faster* than Aer on Grover, random, and GHZ at n=25). See `docs/perf/phase1.md`.
- Phase 2: ≥12× speedup on 16 cores vs. single-thread.
- Phase 3: Stabilizer backend handles 1000+ qubit Clifford circuits; MPS handles 100+ qubit shallow circuits.
- Phase 4: Published benchmark report, GitHub release v0.1.
- Phase 4.5: every competitive-matrix cell ≤ 1.2× its reference (Aer MT statevector, Aer MPS, Stim), or a documented structural exception with profiling evidence; published in docs/perf/parity.md. v0.2 + PyPI (P4-09) gate on this.
- Phase 5: GPU backend within 1.5× of cuQuantum standalone.
- Phase 6: Distributed run on 4+ nodes with reasonable scaling efficiency.

**Project-level**:

- ≥1 published benchmark report per major phase.
- 100% correctness vs. Qiskit/Stim oracles in CI.
- Documented public API stable from v0.1 onward.

-----

## 8. References

- Aaronson, Gottesman. “Improved Simulation of Stabilizer Circuits” (2004).
- Vidal. “Efficient Classical Simulation of Slightly Entangled Quantum Computations” (2003).
- Pednault et al. “Pareto-Efficient Quantum Circuit Simulation Using Tensor Network Contraction” (2020).
- Google “Quantum Supremacy Using a Programmable Superconducting Processor” (2019).
- NVIDIA cuQuantum documentation: <https://docs.nvidia.com/cuda/cuquantum/>
- Qiskit Aer: <https://github.com/Qiskit/qiskit-aer>
- Stim: <https://github.com/quantumlib/Stim>
- PennyLane Lightning: <https://github.com/PennyLaneAI/pennylane-lightning>
- OpenQASM 3.0 specification: <https://openqasm.com/>

-----

## 9. How to Use This Repository

- `ROADMAP.md` (this file): Strategy and phase overview. Read first.
- `BACKLOG.md`: Detailed issue specifications. Source of truth for GitHub Issues.
- `CREATE ISSUES.md`: Instructions for Claude Code / scripts to create GitHub Issues from `BACKLOG.md`.

To create the GitHub backlog: follow `CREATE ISSUES.md`.