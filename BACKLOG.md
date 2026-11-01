# Quantum Simulator — Detailed Backlog

> **Source of truth for GitHub Issues.**
> This document contains the mainline phase issues, including Phase 7's research instruments and Phase 8's performance recovery. Each issue is structured so that an AI agent (e.g., Claude Code) or a human can create a GitHub Issue directly from its contents. The companion QEC backlog is `docs/qec/BACKLOG.md`.

-----

## How to Read This Document

Each issue follows the same template:

```
### [P{phase}-{nn}] {Title}

**Labels:** `area:*`, `type:*`, `priority:*`
**Milestone:** Phase {n}
**Estimate:** S / M / L / XL  (S ≈ <1 day, M ≈ 1–3 days, L ≈ 3–7 days, XL ≈ >1 week)
**Depends on:** P{phase}-{nn}, ...

**Description** — short summary.

**Context** — why this matters, what problem it solves.

**Technical Details** — implementation guidance, algorithms, references.

**Acceptance Criteria** — testable bullet points; all must be true to close.

**Testing Requirements** — unit, property, integration, benchmark tests.

**References** — links to papers, other implementations, docs.
```

-----

## Label System

- **Area**: `area:core`, `area:parser`, `area:ir`, `area:backend-sv`, `area:backend-mps`, `area:backend-stab`, `area:backend-gpu`, `area:backend-dist`, `area:bench`, `area:infra`, `area:docs`, `area:python`, `area:cli`
- **Type**: `type:feature`, `type:optimization`, `type:bug`, `type:refactor`, `type:test`, `type:docs`, `type:infra`
- **Priority**: `priority:critical`, `priority:high`, `priority:medium`, `priority:low`
- **Difficulty**: `good-first-issue`, `help-wanted`, `research`

## Milestones

- Phase 0 — Foundation
- Phase 1 — Single-Thread CPU Optimization
- Phase 2 — Multi-Thread CPU
- Phase 3 — Alternative Backends
- Phase 4 — Algorithm Benchmarks & v0.1 Release
- Phase 4.5 — CPU Parity
- Phase 4.6 — CPU Depth
- Phase 5 — GPU Backend
- Phase 5.5 — Apple/Metal GPU
- Phase 6 — Multi-GPU & Distributed
- Phase 7 — Research Instruments
- Phase 8 — Performance Audit and Recovery

-----

# Phase 0 — Foundation

Goal: working end-to-end pipeline (parser → IR → naive backend → measurement) with full testing and benchmarking infrastructure. Correctness over speed.

-----

### [P0-01] Setup Rust workspace and project structure

**Labels:** `area:infra`, `type:infra`, `priority:critical`
**Milestone:** Phase 0
**Estimate:** S
**Depends on:** —

**Description**
Initialize the Cargo workspace with crate layout that anticipates the full architecture.

**Context**
A well-designed workspace upfront avoids painful refactors later. We’ll have multiple crates (core, parser, IR, backends, python bindings, CLI).

**Technical Details**
Proposed crate layout:

```
aleph/
├── Cargo.toml          (workspace root)
├── crates/
│   ├── aleph-core/      (Complex, StateVector, Gate, Circuit types)
│   ├── aleph-ir/        (Circuit IR, optimization passes)
│   ├── aleph-parser/    (OpenQASM 3.0 parser)
│   ├── aleph-backend/   (Backend trait + naive impl)
│   ├── aleph-sv/        (state vector backends, CPU/GPU)
│   ├── aleph-mps/       (MPS backend)
│   ├── aleph-stab/      (stabilizer backend)
│   ├── aleph-cli/       (command-line tool)
│   └── aleph-py/        (pyo3 bindings)
└── benches/            (cross-crate benchmarks)
```

Use `edition = "2021"`, `rust-version = "1.75"` minimum. Add `rustfmt.toml` and `clippy.toml`.

**Acceptance Criteria**

- [x] `cargo build --workspace` succeeds
- [x] `cargo test --workspace` succeeds (no tests yet, but exits 0)
- [x] `cargo clippy --workspace -- -D warnings` succeeds
- [x] `cargo fmt --check` succeeds
- [x] README.md with build instructions exists

**Testing Requirements**

- Empty placeholder test in each crate to prove test harness runs.

**References**

- <https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html>

-----

### [P0-02] CI/CD pipeline with GitHub Actions

**Labels:** `area:infra`, `type:infra`, `priority:critical`
**Milestone:** Phase 0
**Estimate:** S
**Depends on:** P0-01

**Description**
Set up GitHub Actions workflows for build, test, lint, format, and benchmark regression detection.

**Context**
CI from day one means broken commits never reach main. Benchmark regression detection catches performance regressions before merge.

**Technical Details**
Workflows needed:

- `.github/workflows/ci.yml`: build + test + clippy + fmt on Linux/macOS, stable + beta Rust.
- `.github/workflows/bench.yml`: run criterion benchmarks on PR, comment results.
- Cache `~/.cargo` and `target/` for speed.

Use `actions-rs/toolchain` or `dtolnay/rust-toolchain`.

**Acceptance Criteria**

- [x] CI runs on every PR and main push
- [x] Build, test, clippy, fmt all gating
- [x] Linux + macOS matrix
- [x] Stable Rust required; beta allowed to fail
- [x] Benchmark workflow exists (may be no-op until P0-04)

**Testing Requirements**

- Open a test PR with intentional clippy failure; CI must catch it.

**References**

- <https://github.com/actions-rs>
- <https://github.com/bencherdev/bencher> for benchmark tracking (optional)

-----

### [P0-03] Choose and integrate complex number primitives

**Labels:** `area:core`, `type:feature`, `priority:critical`
**Milestone:** Phase 0
**Estimate:** S
**Depends on:** P0-01

**Description**
Decide on `Complex64` representation: `num-complex` crate vs. custom `(f64, f64)` struct.

**Context**
Every operation touches Complex64. The choice affects SIMD compatibility, memory layout, and arithmetic ergonomics. We may end up with SoA (separate real/imag arrays) later, but the abstract type still matters.

**Technical Details**
Evaluate:

- `num_complex::Complex64`: standard, well-tested, but uses AoS layout.
- Custom `Complex { re: f64, im: f64 }`: more control, possibly better for `#[repr(C)]` interop with CUDA later.
- Plan for SoA: define a `StateVector` that internally may use SoA even if individual ops use Complex64.

**Acceptance Criteria**

- [x] Decision documented in `docs/decisions/0001-complex-type.md` (ADR format)
- [x] Type aliased as `aleph_core::Complex` for forward compatibility
- [x] All current usage routed through this alias

**Testing Requirements**

- Unit test: basic arithmetic, magnitude, phase, conjugate.

**References**

- <https://docs.rs/num-complex/>
- ADR template: <https://adr.github.io/>

-----

### [P0-04] Criterion benchmark harness

**Labels:** `area:bench`, `area:infra`, `type:infra`, `priority:high`
**Milestone:** Phase 0
**Estimate:** M
**Depends on:** P0-01, P0-02

**Description**
Set up `criterion` benchmarks at the workspace level with a standard set of benchmark circuits.

**Context**
“Without benchmarks, you don’t know if optimizations work.” Every optimization PR must produce before/after numbers.

**Technical Details**

- Add `criterion` as dev-dependency.
- Create `benches/` directory at workspace root.
- Standard benchmark fixtures:
  - GHZ state preparation (n = 10, 15, 20, 25)
  - QFT (n = 10, 15, 20)
  - Random circuit (n = 20, depth = 20)
  - Bell pair (n = 2)
- Each benchmark exposed as `cargo bench --bench {name}`.
- Output to `target/criterion/`; HTML reports.

**Acceptance Criteria**

- [x] `cargo bench` produces output
- [x] At least 4 benchmark fixtures wired up
- [x] Documentation in `docs/benchmarking.md`
- [x] CI runs benchmarks on PR (may not gate, just report)

**Testing Requirements**

- Benchmarks succeed against the naive backend (P0-09) once it exists.

**References**

- <https://github.com/bheisler/criterion.rs>
- <https://bheisler.github.io/criterion.rs/book/>

-----

### [P0-05] Property-based testing infrastructure

**Labels:** `area:infra`, `type:test`, `priority:high`
**Milestone:** Phase 0
**Estimate:** M
**Depends on:** P0-01

**Description**
Set up `proptest` with quantum-specific generators and invariants.

**Context**
Quantum simulators have rich invariants that are perfect for property-based testing: unitarity (norm preservation), reversibility (gate then inverse = identity), measurement probability sums to 1, etc. Property tests catch entire classes of bugs that examples miss.

**Technical Details**
Define generators:

- `arb_state_vector(n)`: random normalized state vector of n qubits.
- `arb_gate()`: random 1q or 2q gate (Pauli, Clifford, parametric).
- `arb_circuit(n, depth)`: random circuit.

Define invariants:

- After any gate application, state vector remains normalized (‖ψ‖ = 1 ± ε).
- For any gate G with inverse G†: G† G |ψ⟩ = |ψ⟩.
- Sum of measurement probabilities = 1.
- Diagonal gates leave magnitudes unchanged.

**Acceptance Criteria**

- [x] `proptest` integrated, at least 4 generators
- [x] At least 4 invariant tests passing
- [x] Tests run as part of `cargo test`
- [x] Documentation in `docs/testing.md`

**Testing Requirements**

- This issue *is* the testing infrastructure.

**References**

- <https://github.com/proptest-rs/proptest>
- <https://propertesting.com/>

-----

### [P0-06] Define `Gate` enum and parametric gate support

**Labels:** `area:core`, `type:feature`, `priority:critical`
**Milestone:** Phase 0
**Estimate:** M
**Depends on:** P0-03

**Description**
Design the `Gate` type that represents quantum gates in the IR.

**Context**
Gate representation choices propagate everywhere. We need to support: standard gates (H, X, Y, Z, S, T, CNOT, CZ, SWAP, Toffoli), parametric gates (Rx(θ), Ry(θ), Rz(θ), U3, controlled-Rx), and arbitrary unitaries (as 2×2 or 4×4 matrices).

**Technical Details**
Proposed structure:

```rust
pub enum Gate {
    // Standard 1q gates
    H, X, Y, Z, S, Sdg, T, Tdg,
    // Parametric 1q gates
    Rx(f64), Ry(f64), Rz(f64), Phase(f64), U3(f64, f64, f64),
    // Standard 2q gates
    Cnot, Cz, Swap, Iswap,
    // Parametric 2q gates
    CRx(f64), CRy(f64), CRz(f64),
    // Multi-controlled
    Toffoli, Ccz,
    // Arbitrary unitary
    Unitary1q([[Complex; 2]; 2]),
    Unitary2q([[Complex; 4]; 4]),
}

pub struct GateInstance {
    pub gate: Gate,
    pub qubits: SmallVec<[u32; 4]>,  // target qubits
    pub controls: SmallVec<[u32; 2]>, // control qubits (for generic controlled)
}
```

Consider: should parameters be `f64` or symbolic (for parameterized circuits in VQE)? Decision: support both via a `Param` enum (`Concrete(f64)` or `Symbolic(SymbolId)`); start with `f64` only and add symbolic in Phase 4.

**Acceptance Criteria**

- [x] `Gate` enum covers all gates in Tier 1 algorithms
- [x] `GateInstance` carries qubit indices
- [x] Each gate has a method `matrix() -> SmallMatrix<Complex>` for naive use
- [x] Each gate has `is_diagonal()`, `is_clifford()`, `inverse()` methods
- [x] Unit tests for matrix correctness

**Testing Requirements**

- Unit tests: each gate’s matrix matches textbook definition.
- Property: `gate.matrix() * gate.inverse().matrix()` ≈ identity.

**References**

- Nielsen & Chuang, Chapter 4.
- Qiskit’s `QuantumCircuit` API for reference.

-----

### [P0-07] Circuit IR data structure

**Labels:** `area:ir`, `type:feature`, `priority:critical`
**Milestone:** Phase 0
**Estimate:** M
**Depends on:** P0-06

**Description**
Design the backend-agnostic circuit representation that backends consume and optimization passes transform.

**Context**
The IR sits between the parser and the backend. It must be efficient to walk, easy to transform (for optimization passes), and rich enough to carry metadata (e.g., “this block came from a fused sequence”).

**Technical Details**
Proposed structure:

```rust
pub struct Circuit {
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub instructions: Vec<Instruction>,
    pub metadata: CircuitMetadata,
}

pub enum Instruction {
    Gate(GateInstance),
    Measure { qubit: u32, clbit: u32 },
    Reset(u32),
    Barrier(SmallVec<[u32; 8]>),
}

pub struct CircuitMetadata {
    pub name: Option<String>,
    pub generated_from: Option<String>,
}
```

The IR should support efficient iteration, qubit dependency analysis (for optimization passes), and slicing into “layers” (gates that can execute in parallel).

**Acceptance Criteria**

- [x] `Circuit` builder API: `circuit.h(0); circuit.cnot(0, 1); circuit.measure(0, 0);`
- [x] Iteration API: `circuit.instructions()`
- [x] Layer extraction: `circuit.layers()` returns groups of commuting instructions
- [x] Serialization to/from OpenQASM 3.0 string (depends on P0-08)

**Testing Requirements**

- Unit: build a Bell pair circuit, iterate, verify count.
- Property: layer extraction preserves semantic ordering.

**References**

- Qiskit DAGCircuit: <https://qiskit.org/documentation/stubs/qiskit.dagcircuit.DAGCircuit.html>

-----

### [P0-08] OpenQASM 3.0 parser (minimal subset)

**Labels:** `area:parser`, `type:feature`, `priority:high`
**Milestone:** Phase 0
**Estimate:** L
**Depends on:** P0-07

**Description**
Implement an OpenQASM 3.0 parser supporting the subset needed for Tier 1 algorithms.

**Context**
OpenQASM 3.0 is the de facto standard. Supporting it from day one gives us free interop with Qiskit (export from Qiskit → run in our simulator → compare).

**Technical Details**
Minimal subset to support:

- `OPENQASM 3.0;` header
- `include "stdgates.inc";`
- `qubit[N] q;` declarations
- `bit[N] c;` declarations
- Standard gates: h, x, y, z, s, t, cx, cz, swap, ccx, rx, ry, rz, u3
- `measure q[i] -> c[i];`
- `barrier q;`
- Comments

Skip for now: classical control flow, custom gate definitions, subroutines, OpenPulse. These come in Phase 4+.

Use `nom` or `pest` parser combinators. `nom` is more common in the Rust ecosystem.

**Acceptance Criteria**

- [x] Parse the Tier 1 algorithm OpenQASM files (GHZ, QFT, Grover, random circuit)
- [x] Produce equivalent `Circuit` IR
- [x] Round-trip: Circuit → OpenQASM → Circuit produces equivalent result
- [x] Helpful error messages with line/column info

**Testing Requirements**

- Unit: parse each Tier 1 algorithm.
- Property: random Circuit → OpenQASM → parse → equivalent Circuit.
- Compare parsed circuits’ execution results against Qiskit (oracle).

**References**

- <https://openqasm.com/>
- <https://github.com/Qiskit/openqasm>
- <https://github.com/rust-bakery/nom>

-----

### [P0-09] `Backend` trait and naive CPU state vector backend

**Labels:** `area:backend`, `area:backend-sv`, `type:feature`, `priority:critical`
**Milestone:** Phase 0
**Estimate:** L
**Depends on:** P0-06, P0-07

**Description**
Define the backend abstraction and implement the simplest possible correct state vector backend.

**Context**
This is our reference implementation. Future optimizations are compared against it for correctness. Simplicity matters more than speed here.

**Technical Details**
Trait:

```rust
pub trait Backend {
    type State;
    
    fn allocate(&mut self, num_qubits: u32) -> Self::State;
    fn apply_gate(&mut self, state: &mut Self::State, gate: &GateInstance);
    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> bool;
    fn sample(&mut self, state: &Self::State, shots: u32) -> Vec<u64>;
    fn expectation_value(&mut self, state: &Self::State, pauli: &PauliString) -> f64;
    fn probabilities(&mut self, state: &Self::State, qubits: &[u32]) -> Vec<f64>;
}
```

Naive implementation:

- Allocate `Vec<Complex>` of size 2^n.
- For each gate, build full 2^n × 2^n matrix and multiply (no, this is too slow — use indexed application but no SIMD, no fusion, no specialization).
- Indexed application: for 1q gate on qubit q, iterate all pairs (i, i ⊕ 2^q) where bit q of i is 0; apply 2×2 matrix.
- For 2q gate on qubits (a, b): iterate quadruplets.

**Acceptance Criteria**

- [x] Runs all Tier 1 algorithms (GHZ, QFT, Grover, random) up to 20 qubits
- [x] Produces correct results vs. Qiskit oracle on all benchmarks
- [x] All property tests pass
- [x] Code is readable and commented — this is the reference

**Testing Requirements**

- Unit: each gate type applied to specific basis states; results match textbook.
- Property: all invariants from P0-05 hold.
- Integration: full circuits compared against Qiskit Aer to ≤1e-10 amplitude difference.

**References**

- Nielsen & Chuang, Chapter 4.
- <https://github.com/Qiskit/qiskit-aer/blob/main/src/simulators/statevector/statevector_state.hpp>

-----

### [P0-10] Oracle comparison test harness (vs. Qiskit)

**Labels:** `area:infra`, `type:test`, `priority:critical`
**Milestone:** Phase 0
**Estimate:** M
**Depends on:** P0-08, P0-09

**Description**
Build a test harness that runs the same circuit through our simulator and Qiskit Aer, asserting equivalence.

**Context**
Qiskit Aer is the gold standard. Any disagreement is a bug in our simulator (almost always). This harness catches regressions immediately.

**Technical Details**

- Use Python subprocess from Rust tests, or generate fixture data in Python ahead of time and load.
- For each test: define a circuit in OpenQASM, run in both, compare final state vector (or sampled distribution if state vector access is hidden).
- Tolerance: 1e-10 for amplitudes (FP64); 1e-5 for sampled probabilities at 100k shots.

Decision needed: subprocess at test time (slower, fresher) vs. pre-generated fixtures (faster, may drift).
Recommendation: pre-generate fixtures, regenerate via `scripts/regen-fixtures.sh`.

**Acceptance Criteria**

- [x] At least 10 circuits in the test corpus
- [x] All tests pass against the naive backend
- [x] `scripts/regen-fixtures.sh` regenerates from Python (amendment, see spec §10.1)
- [x] Documented in `docs/testing.md`

**Testing Requirements**

- Self-validating: failure of this harness means correctness regression.

**References**

- <https://qiskit.org/documentation/stubs/qiskit_aer.AerSimulator.html>

-----

### [P0-11] Measurement, sampling, and probability primitives

**Labels:** `area:core`, `area:backend-sv`, `type:feature`, `priority:high`
**Milestone:** Phase 0
**Estimate:** M
**Depends on:** P0-09

**Description**
Implement projective measurement, multi-shot sampling, marginal probabilities, and Pauli expectation values.

**Context**
These are not gates but are critical primitives. Sampling is hot in VQE/QAOA loops; expectation values are *the* primary output of variational algorithms.

**Technical Details**

- `measure(qubit)`: compute P(0), sample, collapse state, renormalize.
- `sample(shots)`: compute cumulative distribution from |ψ|², draw `shots` samples. Use alias method for O(1) sampling per shot.
- `probabilities(qubits)`: marginal distribution by summing |ψ|² over non-target qubits.
- `expectation_value(pauli_string)`: ⟨ψ|P|ψ⟩ for Pauli strings (efficient: rotate to Z-basis, then sum diagonal contributions).

**Acceptance Criteria**

- [x] All four primitives implemented for naive backend
- [x] Sampling distribution converges to |ψ|² (statistical test with 1M shots)
- [x] Expectation value tests vs. analytical results for known states

**Testing Requirements**

- Unit: measure |0⟩ → always 0, P(0) = 1.
- Unit: measure |+⟩ → 0/1 with equal probability.
- Unit: ⟨0|Z|0⟩ = 1, ⟨+|X|+⟩ = 1.
- Property: ∑ probabilities = 1.
- Statistical: 1M shots of Bell state → 50/50 split on `00`/`11` within tolerance.

**References**

- Vose’s alias method: <https://en.wikipedia.org/wiki/Alias_method>

-----

### [P0-12] CLI tool (`aleph` binary)

**Labels:** `area:cli`, `type:feature`, `priority:medium`
**Milestone:** Phase 0
**Estimate:** M
**Depends on:** P0-08, P0-09, P0-11

**Description**
Build a command-line tool to run OpenQASM files and print results.

**Context**
A CLI is the simplest user-facing interface. Useful for demos, scripting, CI.

**Technical Details**
Commands:

- `aleph run circuit.qasm --shots 1024` → print sample distribution.
- `aleph run circuit.qasm --statevector` → print full state vector (small n only).
- `aleph run circuit.qasm --expectation "ZZ"` → expectation value.
- `aleph bench circuit.qasm` → run and print timing.
- `aleph version`, `aleph help`.

Use `clap` v4 with derive.

**Acceptance Criteria**

- [x] All commands listed work
- [x] Help text auto-generated and readable
- [x] Exit codes: 0 success, non-zero on error
- [x] Documented in README

**Testing Requirements**

- Integration: invoke CLI from shell tests; check output format.

**References**

- <https://docs.rs/clap/latest/clap/>

-----

# Phase 1 — Single-Thread CPU Optimization

Goal: bring naive single-thread performance to within 2× of Qiskit Aer single-threaded on standard benchmarks.

-----

### [P1-01] Struct-of-Arrays (SoA) memory layout for amplitudes

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 1
**Estimate:** M
**Depends on:** P0-09

**Description**
Store state vector as two parallel `Vec<f64>` (real and imaginary parts) instead of `Vec<Complex>`.

**Context**
SoA layout enables better SIMD vectorization. AVX-512 can process 8 f64 lanes at once; with AoS Complex64, only 4 real/imag pairs fit, and arithmetic gets stuck in lane-crossing shuffles.

**Technical Details**

- Add new `StateVector` representation with `re: Vec<f64>`, `im: Vec<f64>`.
- Existing `Vec<Complex>` representation kept for naive backend (reference).
- Gates re-implemented to operate on the SoA layout.
- Add conversion functions between AoS and SoA for tests.

**Acceptance Criteria**

- [x] SoA backend produces identical results to naive backend (≤1e-12 difference)
- [ ] Benchmark: SoA vs. naive on QFT-20 — expect ~1.5–2× improvement just from cache effects (layout-only port lands ~0.96–1.2× on M-series; expected — strided AoS reads hide cache wins on Apple silicon. Closed by P1-02 + P1-03; EPYC bencher.dev numbers are the source of truth.)
- [x] All Phase 0 tests pass against SoA backend

**Testing Requirements**

- Equivalence test vs. naive backend on full Tier 1.
- All property tests against SoA.

**References**

- <https://en.wikipedia.org/wiki/AoS_and_SoA>
- <https://github.com/QuEST-Kit/QuEST> (uses SoA)

-----

### [P1-02] Bit-manipulation indexing for 1-qubit gate application — **DEFERRED, FOLDED INTO P1-03**

**Status:** Standalone implementation attempted in PR #76 (closed without merge, 2026-05-26). See [ADR 0007](docs/decisions/0007-soa-x86-perf-finding.md) for the full perf investigation. Short version: on x86 with AVX-512, LLVM auto-vectorizes P1-01's "branchy" predicate-loop as a masked loop (`vporq`/`vpsllvq` packed-quadword ops processing 4-8 indices per cycle); P1-02's branch-free restructure broke that transformation and regressed QFT-20 by ~30% (332 ms → 428 ms on EPYC self-hosted). The bit-manip pattern is still the right shape — but only when paired with explicit SIMD intrinsics (P1-03) where it lets `vmovupd` consume unit-stride inner blocks directly. Implementing it as a standalone optimization is a pessimization.

**P1-03 will subsume P1-02:** the nested block/pair indexing pattern lands as the SoA SIMD kernel's inner loop, not as a layout-only change. Update the P1-03 spec to incorporate the bit-manip work directly.

**Labels:** `area:backend-sv`, `type:optimization`, `priority:critical`
**Milestone:** Phase 1
**Estimate:** ~~M~~ (folded into P1-03)
**Depends on:** P1-01

**Acceptance Criteria**

- [x] ~~All 1q gates implemented with this pattern~~ — superseded; P1-03 SIMD will use the pattern as its inner loop shape.
- [x] ~~Benchmark: 2–3× improvement over P1-01 on QFT-20~~ — superseded; P1-03 AC absorbs this target.
- [x] ~~All correctness tests pass~~ — n/a (no implementation lands).

**References**

- [ADR 0007 — SoA layout-only optimization on x86 loses to LLVM masked-loop auto-vec](docs/decisions/0007-soa-x86-perf-finding.md)
- <https://github.com/QuEST-Kit/QuEST/blob/master/QuEST/src/CPU/QuEST_cpu.c> — QuEST's bit-manip pattern works because they write SIMD intrinsics from day one, not relying on LLVM auto-vec.

-----

### [P1-03] SIMD intrinsics for 1-qubit gates (AVX-512 on AoS layout)

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 1
**Estimate:** L
**Depends on:** P1-01

**Status:** Shipped 2026-05-26 (TBD squash hash). Absorbs P1-04 (#16). The implementation deliberately diverges from the original "SoA + AVX2 + AVX-512" plan after forensic investigation — see [ADR 0008](docs/decisions/0008-aos-avx512-beats-soa-simd.md). Short version: hand-written SIMD on the SoA layout (PR #78) gave essentially zero speedup vs LLVM-auto-vec'd scalar SoA because the bottleneck is load µops (4 streams), not FLOPs. Re-targeting the same intrinsics at the **AoS layout** (PR #79 / this PR, packed-complex via `_mm512_permute_pd` + `_mm512_fmaddsub_pd`) kept the 2-stream cache pattern and unlocked the win.

**Description (as shipped)**
Runtime-dispatched AVX-512F path on `kernels::aos::apply_1q`. One `_mm512_loadu_pd` reads 4 complex pairs (8 doubles, one cache-line per side); inner kernel does packed-complex × scalar-complex multiply via `vfmaddsub` and the `(re, im) → (im, re)` lane swap. Scalar fallback (LLVM auto-vec to `vmulpd xmm`) handles non-x86 hosts, sub-LANES targets, and the `min(controls) ≤ target` orientation.

**Acceptance Criteria**

- [x] ~~AVX2 kernels for at least 5 gate types~~ — revised: AVX-512F packed-complex kernel covers all generic 2×2 unitaries (H, X, Z, diagonal, generic) via a single code path. AVX2-only path dropped per ADR 0008 (SoA-with-SIMD doesn't beat AoS-without-SIMD; AVX2-on-SoA would inherit the same finding).
- [x] Runtime feature detection works — `std::is_x86_feature_detected!("avx512f")` in `kernels::aos::apply_1q`; non-x86 + non-AVX-512F hosts hit the LLVM-auto-vec'd scalar body.
- [x] Scalar fallback identical results — `aos_apply_1q_matches_scalar_reference` proptest (96 cases) + 112 generated oracle tests + `aleph-oracle::soa_vs_naive::all_fixtures_match_naive` all assert SIMD ≡ scalar within 1e-12.
- [x] ~~Benchmark: 2–4× improvement over P1-01 SoA baseline~~ — **partially met**. EPYC 8124P side-by-side: `qft/n15/naive` **2.01×** (clean), `qft/n20/naive` **1.80×** (target was 2.00×, ~10% short). The remaining gap is algorithmic — ADR 0008's perf-stat shows the SIMD tier is exhausted; further gains need IR-level optimisation (P1-08 gate fusion) per CLAUDE.md's perf hierarchy.
- [x] ~~Inner loop uses nested block/pair bit-manipulation indexing~~ — applied differently than originally specified. The AoS path uses block / pair-stride indexing for the controlled SIMD case (via renormalised `expand_with_fixed`), and the uncontrolled fast path uses a plain outer-stride loop. Both produce contiguous SIMD reads. P1-02's bit-manip framing was specific to SoA; the AoS shape needed its own approach.

**Note on the SoA path:** P1-01's `SoaSvBackend` remains in tree, but unchanged by this PR. It's competitive with AoS on non-x86 (where LLVM NEON auto-vec is close to AoS parity) and remains useful for any future workload where SoA's lower per-gate memory footprint pays off. Default backend selection is *not* changed in this ticket — that's a separate decision once we have more workload data.

**Note on the originally-spec'd SoA-SIMD direction:** PR #78 implemented SoA + AVX2 + AVX-512 per the original spec but produced no measurable speedup on EPYC (qft/n20 flat at 312 ms vs P1-01's 310 ms). Closed without merge — see [ADR 0008](docs/decisions/0008-aos-avx512-beats-soa-simd.md) for the perf-stat / objdump forensic trail.

**Testing Requirements (met)**

- Per-host forced-path proptest in `kernels/aos.rs` (96 cases, target ∈ 0..6, control count ∈ 0..=2, both control orientations).
- 112 generated oracle tests (build.rs).
- Cross-arch local sweep (M-series scalar path stays green).
- EPYC bench numbers in PR body (`qft/{n10,n15,n20}/{naive,soa}` + `ghz/n20`).

**References**

- [ADR 0008 — AoS + hand-written AVX-512 beats SoA + hand-written SIMD on QFT](docs/decisions/0008-aos-avx512-beats-soa-simd.md) — **read first**; covers the forensic perf-stat / objdump trail and decision rationale.
- [ADR 0007 — SoA layout-only optimization on x86 loses to LLVM masked-loop auto-vec](docs/decisions/0007-soa-x86-perf-finding.md) — the predecessor finding.
- <https://software.intel.com/sites/landingpage/IntrinsicsGuide/> — specifically `_mm512_permute_pd`, `_mm512_fmaddsub_pd`, `_mm512_set1_pd`.
- <https://doc.rust-lang.org/std/arch/index.html>

-----

### [P1-04] SIMD (AVX-512) for 1-qubit gates — **DEFERRED, FOLDED INTO P1-03**

**Labels:** `area:backend-sv`, `type:optimization`, `priority:medium`
**Milestone:** Phase 1
**Estimate:** ~~M~~ (folded into P1-03)
**Depends on:** ~~P1-03~~ (was a follow-up; now bundled in)

**Status:** Shipped as part of P1-03 (see PR — `Closes #15, #16`). The original P1-03/P1-04 split into "AVX2 then AVX-512" was rendered moot by the layout pivot — see [ADR 0008](docs/decisions/0008-aos-avx512-beats-soa-simd.md). P1-03 ships a single AVX-512F packed-complex kernel on the AoS layout; an AVX2-only variant was not implemented because the AVX-512 forensic finding (load-µop bottleneck on SoA) applies equally to AVX2 — the layout choice, not the lane count, was the critical decision. GitHub issue #16 closes alongside #15 via the P1-03 PR.

**Description (historical)**
Extend SIMD kernels to AVX-512 (8 f64 lanes).

**Acceptance Criteria**

- [x] ~~AVX-512 kernels for hot gate types~~ — see `crates/aleph-sv/src/kernels/aos.rs::apply_1q_avx512`; generic 2×2 unitary covers the listed types.
- [x] ~~Runtime feature detection~~ — `std::is_x86_feature_detected!("avx512f")` in `kernels::aos::apply_1q`.
- [x] ~~Benchmark: improvement over AVX2 on AVX-512 hardware~~ — N/A by the layout pivot; baseline is now LLVM-auto-vec'd `vmulpd xmm` AoS, not AVX2. EPYC `qft/n20/naive`: 305.7 ms (no AVX-512) → 172.3 ms (with AVX-512) = 1.77×.
- [x] ~~No regression on AVX2-only hardware~~ — the dispatcher falls back to scalar (LLVM auto-vec) when AVX-512F is absent; CI runs on non-AVX-512 macOS and Linux confirm no regression.

**References**
- [ADR 0008](docs/decisions/0008-aos-avx512-beats-soa-simd.md) — why AVX2-only path was not implemented.

**References**

- <https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html>

-----

### [P1-05] Specialized Pauli-X kernel

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 1
**Estimate:** S
**Depends on:** P1-02

**Description**
Pauli-X swaps amplitudes at index i and i ⊕ 2^q. Implement as pure swap, no multiplication.

**Context**
Pauli-X is common (often appears via decompositions). A swap kernel skips all arithmetic. Same idea for Y (swap + phase) and Z (negate half).

**Technical Details**
For X: `std::mem::swap(&mut re[i0], &mut re[i1]); std::mem::swap(&mut im[i0], &mut im[i1]);`
For Y: swap + multiply one half by ±i (sign flip + swap of re/im).
For Z: `re[i1] = -re[i1]; im[i1] = -im[i1];` (only the i1 half).

**Acceptance Criteria**

- [x] X, Y, Z specialized kernels
- [x] Benchmark: 3–10× speedup over generic 1q kernel for these gates
- [x] Correctness preserved

**Testing Requirements**

- Equivalence vs. generic kernel.

**References**

- N/A — straightforward optimization.

-----

**§15.1 — P1-05 amendment (2026-05-28).** The original 2025-09 spec
called for "X, Y, Z specialised kernels" on an SoA substrate. Phase-1
substrate work moved the default x86 path to AoS+AVX-512 (ADR 0008)
and added a diagonal-1q fast path covering Z (ADR 0009). The P1-05
implementation differs from the original spec as follows:

- **Scope.** Z is removed from P1-05 scope (covered by P1-06 diagonal
  fast path). X, Y, and a generic anti-diagonal kernel are added,
  dispatched by a new `classify_1q_antidiag` classifier in
  `kernels/mod.rs`.
- **Substrate.** AoS + SoA parity. Three-tier SIMD dispatch (Tier A
  packed, Tier B in-register lane permute, Tier C scalar) mirrors
  ADR 0010. Tier-B contract tightened to require
  `controls.iter().all(|&c| c >= log2(LANES))` (see ADR 0011 for the
  failure mode).
- **Acceptance.** "3–10× speedup over generic 1q kernel" is a
  micro-bench AC measured at L2-resident state (n ≤ 14); n=20
  wall-clock is informational per the bandwidth-bound regime
  documented in ADR 0008. Workload-level delta (grover_n20_iters5)
  recorded in `docs/perf/phase1-vs-qiskit.md` "P1-05 update" but not
  gating.

Updated AC checklist:

- [x] X kernel (pure swap, AoS + SoA, three tiers)
- [x] Y kernel (swap + sign-flip, AoS + SoA, three tiers — Tier B
      SoA wraps scalar; see ADR 0011 Open Question 1)
- [x] Generic anti-diagonal kernel (full multiply, AoS + SoA, three
      tiers — Tier B SoA wraps scalar)
- [x] `is_antidiagonal_2x2` + `classify_1q_antidiag` in `kernels/mod.rs`
- [x] Micro-bench: 3–10× speedup over generic-scalar 1q kernel on
      L2-resident state (EPYC Tier-A: X 3.89×, Y 3.97×, antidiag 3.80×;
      Tier-B 4.22–4.55×). Vs the pre-P1-05 dispatch's AVX-512 generic
      path the lift is 1.03–1.27× (bandwidth-bound; see ADR 0011).
- [x] ADR 0011 documents the dispatch and Open Questions

-----

### [P1-06] Specialized diagonal-gate kernel

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 1
**Estimate:** S
**Depends on:** P1-02

**Description**
For diagonal gates (Z, S, T, Rz, Phase), no amplitude mixing happens; only phases multiply.

**Context**
Diagonal gates are very common (especially in QFT, QPE, variational ansatze). Optimized kernel walks the state vector once with no pair iteration.

**Technical Details**
For Rz(θ) on qubit q: multiply amplitudes where bit q = 0 by e^(-iθ/2), where bit q = 1 by e^(+iθ/2). Precompute the two phase factors; single loop.

Diagonal 2q gates (CZ, CPhase) are similar.

**Acceptance Criteria**

- [ ] Specialized kernel for diagonal 1q and 2q gates
- [ ] Benchmark: 2–3× speedup over generic kernel
- [ ] Integration with gate-dispatch logic

**Testing Requirements**

- Equivalence vs. generic kernel.

**References**

- <https://github.com/QuEST-Kit/QuEST>

-----

### [P1-07] 2-qubit gate generic kernel + specialized CNOT/CZ/SWAP

**Labels:** `area:backend-sv`, `type:optimization`, `priority:critical`
**Milestone:** Phase 1
**Estimate:** L
**Depends on:** P1-03

**Description**
Generic 4×4 unitary application kernel plus specialized fast paths for the most common 2q gates.

**Context**
2q gates dominate circuit cost. CNOT is the single most common entangler.

**Technical Details**
Generic 2q: iterate quadruplets (i00, i01, i10, i11) where bits (a, b) take all four values; apply 4×4 matrix.

CNOT (with control = c, target = t): for amplitudes where bit c = 1, swap pairs differing at bit t. No multiplication.

CZ: amplitudes where both bits = 1 get negated; everything else unchanged. Single pass.

SWAP: swap amplitudes where bits differ.

iSWAP, sqrt-SWAP: specialized as needed.

**Acceptance Criteria**

- [x] Generic 2q kernel
- [x] Specialized CNOT, CZ, SWAP
- [x] SIMD versions of each
- [~] Benchmark: CNOT 5–10× faster than generic 2q kernel — **partially missed**, achieved **2.50×** on EPYC (39 ms / 97 ms). Root cause: bandwidth-bound at n=20 (16 MiB state spills L3); generic 2q is now also AVX-512 (Task 5) so the per-µop lead collapses to the bandwidth ratio (CNOT touches half the state, generic touches all). Workload-level qft_n20 (**1.90× vs P1-06, 1.30× Aer**) remains the binding success criterion and clears the ROADMAP § 7 exit with margin. See `docs/perf/phase1-vs-qiskit.md` § "P1-07 update" and ADR 0010 § "Performance shape" for the post-mortem.

**Testing Requirements**

- Equivalence vs. naive for each gate.

**References**

- <https://arxiv.org/abs/1601.07195> (“Quantum Supremacy” simulation paper has discussion)
- ADR 0010 ([[0010-2q-specialised-paths]]) — dispatch tree, three-tier SIMD coverage, AoS / SoA Tier C distinction.

-----

### [P1-08] Multi-controlled gate kernels (Toffoli, CCZ, MCX)

**Labels:** `area:backend-sv`, `type:optimization`, `priority:medium`
**Milestone:** Phase 1
**Estimate:** M
**Depends on:** P1-07

**Description**
Specialized kernels for Toffoli (CCX), CCZ, and arbitrary multi-controlled gates.

**Context**
Toffoli is used heavily in arithmetic circuits (Shor’s algorithm) and error correction. Doing it as a sequence of CNOTs and T gates is correct but slow.

**Technical Details**

For CCX on (c0, c1, t): for amplitudes where both control bits = 1, swap pairs at target. Single pass.

For CCZ on (c0, c1, t): for amplitudes where both control bits = 1, apply phase flip (multiply by −1).

For arbitrary MCX with k controls: amplitudes where all k controls = 1 are affected; iterate with a mask check.

**Spec amendment:** see `docs/superpowers/specs/2026-05-28-p1-08-multi-controlled-design.md` §1.

**Implementation:** Substrate is AoS + AVX-512 packed-complex (per ADR 0008). Matrix-shape dispatch at `apply_3q` prelude (per ADR 0012) detects Toffoli/CCZ 8×8 matrix patterns and routes to `dispatch_toffoli` / `dispatch_ccz`. Fall-through is `apply_3q_generic` (the original scalar 8×8 matrix multiply). Symmetric SoA implementation in `kernels::soa` with LANES_SOA=8 (vs AoS LANES=4) — Tier B has three sub-tiers for SoA (`t ∈ {0, 1, 2}`).

MCX with k controls is **implicit via P1-05**: a `Gate::X` with extra `controls` is dispatched through `apply_1q` to P1-05’s specialised anti-diagonal kernel, which already accepts arbitrary `controls.len()` via `control_mask(controls)`.

**Acceptance Criteria**

- [x] CCX, CCZ specialised (AoS + AVX-512 packed-complex). Tier A clean + outer-walk; Tier B in-zmm permute for `t ∈ {0, 1}` (Toffoli); single Tier A path for CCZ via `vxorpd` sign-flip.
- [x] Generic MCX with up to 8 controls — **implicit via P1-05** anti-diagonal kernel (`apply_1q` with k external controls); verified by `mcx_k{2,4,6}_n20` benches + `multi_ctrl_mcx_k7_8q_oracle` test.
- [x] Benchmark — `toffoli_chain_n{15,20}`, `ccz_chain_n{15,20}`, `mcx_k{2,4,6}_n20` synthetic chains on EPYC; numbers in `docs/perf/p1-08-multi-controlled.md`.
- [ ] Workload anti-regression — qft_n20 / grover_iter5_n20 / random_brickwall_n20_d20 within 2% on EPYC. **Partial.** qft_n20 −0.95% ✅; random_brickwall_n20_d20 **+3.12% ⚠️** (above gate but code-presence — random has zero Toffoli/CCZ; analysis in `docs/perf/p1-08-multi-controlled.md`); grover_n20_iters5 deferred to P1-14 (single-iter wall-clock too long for criterion sample sweep). Phase-1 ROADMAP §7 exit (≤ 2× Aer) still cleared on all three.

**Testing Requirements**

- Integer-only indexing-coverage tests (`classify_toffoli`, `ccz_pairs_unique`) exhaustively verify dispatch tier classification on n=6. Catches bit-collision bugs pre-SIMD.
- Property tests (proptest, 64 cases): CCX/CCZ involutivity, CCZ qubit-order symmetry.
- Oracle tests vs Qiskit Aer: CCX, CCCX, Grover-3q-CCZ, MCX-k7 (the P1-05 verification anchor).
- Equivalence vs. decomposed (CCX = 6 CNOTs + T gates).

**References**

- Nielsen & Chuang, Section 4.3.
- ADR 0012: multi-controlled SIMD dispatch pattern.
- Spec: `docs/superpowers/specs/2026-05-28-p1-08-multi-controlled-design.md`.
- Plan: `docs/superpowers/plans/2026-05-28-p1-08-multi-controlled.md`.

-----

### [P1-09] Gate fusion pass — adjacent 1q gates

**Labels:** `area:ir`, `type:optimization`, `priority:critical`
**Milestone:** Phase 1
**Estimate:** L
**Depends on:** P0-07, P1-02

**Description**
IR pass that merges consecutive 1-qubit gates on the same qubit into a single 2×2 unitary.

**Context**
This is one of the highest-ROI optimizations. Reduces state vector passes by 2–10× for typical circuits with rotation sequences (VQE ansatze are full of these).

**Technical Details**
Walk the IR, maintain per-qubit “pending” 2×2 matrix. When a 2q gate or barrier appears on qubit q, flush its pending matrix as a single `Unitary1q` gate. Multiply matrices on the right as new gates arrive.

Edge cases: parametric gates with symbolic params should not be fused (will be relevant in Phase 4). Diagonal+diagonal stays diagonal — preserve the “diagonal” flag for downstream specialization.

**Acceptance Criteria**

- [x] Fusion pass implemented
- [x] On VQE ansatz: ≥3× reduction in gate count
- [x] Result of fused circuit = result of unfused (to ≤1e-12)
- [x] Pass is opt-in via `Circuit.optimize()` or similar

**Testing Requirements**

- Equivalence test: original vs. fused circuit execution.
- Property: fusion preserves semantics for random circuits.

**References**

- Häner, Steiger. “0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit” — fusion technique discussion.
- <https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/transpiler/passes/optimization/>

-----

### [P1-10] Gate fusion pass — 2q + adjacent 1q

**Labels:** `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 1
**Estimate:** L
**Depends on:** P1-09

**Description**
Extend fusion to absorb 1q gates adjacent to 2q gates into a fused 4×4 unitary.

**Context**
Often a CNOT is preceded/followed by 1q rotations on its qubits. Fusing them produces a single 4×4 gate, halving the state vector passes.

**Technical Details**
Track 2q gates and any 1q gates immediately preceding/following on the same qubits with no intervening operations. Compute the 4×4 product. Replace.

Care: maintain order; a 1q gate after a 2q on qubit a, but before a 2q on qubits (a, b), can be fused with the first 2q only if it doesn’t precede unrelated work on b.

**Acceptance Criteria**

- [ ] 2q+1q fusion implemented
- [ ] On QAOA depth-10 circuit: ≥1.5× reduction in gate count beyond P1-09
- [ ] Correctness preserved

**Testing Requirements**

- Same as P1-09 with extended corpus.

**References**

- Same as P1-09.

-----

### [P1-11] Dead code elimination pass

**Labels:** `area:ir`, `type:optimization`, `priority:medium`
**Milestone:** Phase 1
**Estimate:** M
**Depends on:** P0-07

**Description**
Remove gates whose effects cannot reach any measured qubit (DCE).

**Context**
Some circuits have ancilla operations that aren’t measured. Reverse data-flow from measurements identifies “live” gates; rest can be dropped.

**Technical Details**
Build a use-def graph. Mark all measured qubits as live. Walk circuit in reverse; for each gate, if any of its qubits is live, mark all of its qubits as live (since gates create entanglement). Otherwise, gate is dead. Remove.

Conservative version: only remove gates whose qubits are never measured AND never touched by subsequent measured-qubit-touching gates.

**Acceptance Criteria**

- [ ] DCE pass implemented
- [ ] Test corpus shows reduction on at least one fixture
- [ ] No false positives (no removal of live gates)

**Testing Requirements**

- Unit: hand-crafted circuit with deliberate dead branch.
- Equivalence: DCE’d vs. original circuit produce same measurement distribution.

**References**

- Standard compiler DCE adapted to quantum.

-----

### [P1-12] Gate cancellation pass (H·H, X·X, Rz(θ)·Rz(-θ))

**Labels:** `area:ir`, `type:optimization`, `priority:medium`
**Milestone:** Phase 1
**Estimate:** M
**Depends on:** P0-07

**Description**
Identify self-inverse and inverse-pair patterns and eliminate them.

**Context**
After transpilation or naive circuit construction, redundant gates often appear: H·H, X·X, CNOT·CNOT (same qubits), Rz(θ)·Rz(-θ). Detecting and removing them is cheap and effective.

**Technical Details**
Linear pass: maintain a “last gate per qubit” stack. When a new gate arrives, check if it cancels the prior gate on those qubits. If yes, pop. If no, push.

Tricky cases: gates separated by commuting unrelated gates. Use commutation analysis (next phase) for the powerful version; start with adjacent-only.

**Acceptance Criteria**

- [ ] Adjacent cancellation pass
- [ ] At least 5 cancellation patterns
- [ ] Correctness preserved

**Testing Requirements**

- Unit: hand-crafted cancellation patterns.
- Property: random circuit with injected redundant pairs simplifies to original.

**References**

- Qiskit `Optimize1qGates`, `CommutativeCancellation`.

-----

### [P1-13] Commutation analysis (foundational)

**Labels:** `area:ir`, `type:optimization`, `priority:medium`
**Milestone:** Phase 1
**Estimate:** L
**Depends on:** P0-07

**Description**
Identify pairs of gates that commute. Enables more aggressive cancellation and fusion.

**Context**
Many gates commute: X commutes with X, Z with Z, Rz with Z, etc. Knowing this lets us reorder gates to bring cancellable pairs together.

**Technical Details**
Static table of commutation relations between gate types. For parametric gates: Rz(α) commutes with Rz(β) (same qubit) and with any Z on same qubit. CNOT commutes with X on target and with Z on control.

Provide an API: `gates_commute(g1: &GateInstance, g2: &GateInstance) -> bool`.

**Acceptance Criteria**

- [ ] Commutation table covers standard gates
- [ ] API exposed for other passes
- [ ] Unit tests for all entries in table

**Testing Requirements**

- For each commuting pair: applying in either order produces same state.
- For each non-commuting pair: applying in either order produces different states (sanity).

**References**

- <https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/circuit/commutation_checker.py>

-----

### [P1-14] Phase 1 performance report

**Labels:** `area:bench`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 1
**Estimate:** M
**Depends on:** P1-01 through P1-13

**Description**
Comprehensive benchmark report comparing single-thread performance against Qiskit Aer single-threaded.

**Context**
Phase exit criterion. Without numbers, we don’t know if Phase 1 succeeded.

**Technical Details**

- Run all Tier 1 algorithms at n = 15, 20, 22, 25 qubits.
- Compare wall-clock time vs. Qiskit Aer (single thread).
- Measure: time per gate, total time, peak memory.
- Identify any benchmarks where we’re worse than 2× of Qiskit Aer.
- Publish in `docs/perf/phase1.md`.

**Acceptance Criteria**

- [x] Report committed — `docs/perf/phase1.md` (2026-05-31).
- [x] All Tier 1 algorithms benchmarked — full {GHZ, QFT, Grover, random} × n{15,20,22,25} matrix on EPYC 8124P.
- [x] Targets met or specific follow-up issues filed for misses — **all 16 cells ≤ 2× Aer; ROADMAP §7 met (worst: qft_n25 = 1.73×); no follow-ups needed.** (Stage-0's qft_n20 2.39× miss → 1.22× post-P1-06/07.)

**Testing Requirements**

- Benchmark CI runs the report’s measurements. The full matrix is gated behind `ALEPH_BENCH_FULL_MATRIX=1` (a manual EPYC run); CI runs a cheap subset (n ≤ 20, no grover) to stay under the Bench workflow's 30-min timeout.

**References**

- Qiskit Aer config: `simulator = AerSimulator(method='statevector', max_parallel_threads=1)`.

-----

# Phase 2 — Multi-Thread CPU

Goal: near-linear scaling on 16+ cores.

-----

### [P2-01] Rayon-based parallel gate application

**Labels:** `area:backend-sv`, `type:optimization`, `priority:critical`
**Milestone:** Phase 2
**Estimate:** M
**Depends on:** P1-07

**Description**
Parallelize state vector kernels across cores using rayon.

**Context**
Each apply_gate is embarrassingly parallel over chunks of the state vector. Rayon makes this trivial; what’s not trivial is avoiding false sharing and tuning chunk sizes.

**Technical Details**

- Use `rayon::join` or `par_chunks_mut` over state vector slices.
- Tune chunk size: too small → overhead dominates; too large → poor load balance.
- Heuristic: chunk size ≥ L1 cache size / 16 ≈ 4 KB / 16 ≈ 256 amplitudes minimum.

**Acceptance Criteria**

- [ ] Parallel 1q and 2q kernels
- [ ] On 8 cores: ≥6× speedup on QFT-25
- [ ] No correctness regressions

**Testing Requirements**

- Equivalence tests on parallel backend.

**References**

- <https://docs.rs/rayon/>

-----

### [P2-02] Cache-line padding to prevent false sharing

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 2
**Estimate:** S
**Depends on:** P2-01

**Description**
Ensure parallel writes don’t trigger cache-line ping-pong between cores.

**Context**
When threads write to memory locations within the same 64-byte cache line, each write invalidates the line on other cores. This destroys parallel scaling.

**Technical Details**

- Audit kernels: identify shared writable data, accumulators.
- For per-thread state (sample counts, statistics), pad to cache-line size with `#[repr(align(64))]`.
- For state vector itself: chunk sizes ensure threads work on independent cache lines.

**Acceptance Criteria**

- [x] Audit complete, padding applied where needed (64-byte-aligned `AlignedBuf`; per-thread struct padding deferred — no parallel accumulator exists yet)
- [x] No false-sharing patterns identified by perf tools (`perf c2c`: 28 shared lines / 24 local HITM across 230k records — noise; see `docs/perf/phase2-p2-02.md`)
- [x] Scaling efficiency vs. P2-01 — measured **flat within noise** (bandwidth-bound; the audit found no contention to remove). The deliverable is the alignment guarantee + NUMA hook for P2-03, not a speedup.

**Testing Requirements**

- Benchmark: measure scaling with and without padding.

**References**

- <https://lwn.net/Articles/255364/> (Drepper, “What Every Programmer Should Know About Memory”)

-----

### [P2-03] NUMA-aware allocation

**Labels:** `area:backend-sv`, `type:optimization`, `priority:medium`
**Milestone:** Phase 2
**Estimate:** M
**Depends on:** P2-01

**Description**
On multi-socket systems, allocate state vector chunks on memory local to the cores processing them.

**Context**
On multi-socket NUMA, accessing remote memory is 2–3× slower than local. Default allocators put everything on socket 0; threads on socket 1 then suffer.

**Technical Details**

- Use `mimalloc` or `jemalloc` with NUMA policies.
- Interleave allocation across NUMA nodes (`numactl --interleave=all`) as the simple default.
- For advanced: partition state vector with first-touch policy.

**Acceptance Criteria**

- [x] NUMA-aware build option — non-default `numa` cargo feature (`aleph-core`/`aleph-sv`); `AlignedBuf::zeroed_first_touch` parallel first-touch init.
- [x] Benchmark on 2-socket machine: improvement over default allocator — Xeon Silver 4114 (2 NUMA nodes), QFT-25 first-touch **−37.7 %** vs default allocator (and beats interleave's −31.8 %); see `docs/numa.md`.
- [x] Documentation on enabling — `docs/numa.md` (enable, locality contract, interleave fallback, results) + `scripts/numa-bench.sh`.

**Testing Requirements**

- Benchmark on actual NUMA hardware (CI may skip if unavailable).

**References**

- <https://github.com/microsoft/mimalloc>
- <https://man7.org/linux/man-pages/man3/numa.3.html>

-----

### [P2-04] Chunked parallelism tuning

**Labels:** `area:backend-sv`, `type:optimization`, `priority:medium`
**Milestone:** Phase 2
**Estimate:** M
**Depends on:** P2-01

**Description**
Empirically tune chunk sizes per gate type and qubit position.

**Context**
The right chunk size depends on the gate (work per amplitude varies) and the qubit (low qubits → small strides, high qubits → large strides). Tuning helps the last 20–30% of scaling.

**Technical Details**

- Auto-tune at runtime: small probe runs to pick chunk size.
- Or: pre-tuned table per CPU model.
- Start with table; add auto-tune later.

**Acceptance Criteria**

- [ ] Tuned chunk size table for one reference CPU (e.g., Ryzen 9 7950X)
- [ ] Benchmark improvement over fixed default

**Testing Requirements**

- Benchmark suite shows improvement.

**References**

- N/A — empirical work.

-----

### [P2-05] Phase 2 scaling efficiency report

**Labels:** `area:bench`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 2
**Estimate:** S
**Depends on:** P2-01 through P2-04

**Description**
Report scaling efficiency from 1 to 64 cores.

**Context**
Phase exit criterion.

**Technical Details**

- Run all Tier 1 algorithms at n = 25 with thread counts 1, 2, 4, 8, 16, 32, 64.
- Compute efficiency = (speedup / threads).
- Target: ≥75% efficiency at 16 threads.
- Publish in `docs/perf/phase2.md`.

**Acceptance Criteria**

- [ ] Report committed
- [ ] Scaling target met or follow-ups filed

**Testing Requirements**

- Benchmark CI on machine with sufficient cores.

**References**

- Amdahl’s law for context: <https://en.wikipedia.org/wiki/Amdahl%27s_law>

-----

### [P2-06] Diagonal gate fusion pass

**Labels:** `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 2
**Depends on:** P1-09, P1-10

**Description**
Add an IR pass that fuses a run of consecutive **diagonal** gates acting on overlapping qubits into a single diagonal operation, applied to the state vector in one memory pass.

**Context**
P2-05 showed state-vector simulation is memory-bandwidth-bound: each gate streams the full 2^n state (512 MiB at n=25) with near-zero arithmetic intensity, so wall-clock is dominated by passes over memory, not FLOPs. The worst Tier-1 workload is QFT: the live `tier1_scaling` sweep measured the Aer-comparable `qft_n25.qasm` fixture (1526 ops, ≈92% controlled-phase) at only 2.16×@8 / 2.30×@16 on EPYC. Controlled-phase, `rz`, `p`, `z`, `s`, `t` are all **diagonal in the computational basis**, and a product of diagonal operators is itself diagonal — so the entire cphase ladder between two `H` gates in QFT can collapse into a *single* per-amplitude phase multiply instead of one full-state pass per gate. The existing `Fuse2q` pass only merges adjacent 2q gates into a dense 4×4; it does not exploit diagonality. This is the highest-ROI memory-pass reduction available at the IR level (no new hardware), targeting exactly the workload that scales worst.

**Technical Details**

- New `passes::FuseDiagonalRuns` pass: walk the circuit per qubit-set, accumulate maximal runs of diagonal gates (no intervening non-diagonal gate on any shared qubit) into one operation.
- Represent the fused result as a per-amplitude phase vector or an extended `Gate::Unitary1qDiag`/multi-qubit diagonal variant; kernel applies it in a single `par_units` pass (multiply each amplitude by its accumulated complex phase).
- Classify diagonal gates centrally (reuse the `DIAGONAL_EPS_SQ` predicate already used by the 1q-diagonal kernel dispatch).
- Wire into `default_pipeline()` after cancellation/DCE, before/with `Fuse2q`; run-to-fixpoint compatible.

**Acceptance Criteria**

- [ ] QFT-25 controlled-phase ladder collapses to ≤ 2 diagonal passes per qubit; total gate-pass count drops ≥ 5× vs unfused.
- [ ] Oracle equivalence vs unfused within 1e-12 across Tier-1 fixtures (raw and via the pipeline).
- [ ] Criterion improvement on `tier1_scaling`/qft (fixture) on the EPYC bench box, reported in the PR.

**Testing Requirements**

- Property test: fused diagonal run ≡ sequential application on a generic input state (not |0…0⟩).
- Standalone pass test + pipeline idempotence test.
- Benchmark before/after on the EPYC box.

**References**

- `docs/perf/phase2.md` §1, §3 (bandwidth-bound finding; QFT cphase dominance).
- Qiskit Aer `fusion` / diagonal-gate handling (read, re-implement).

-----

### [P2-07] Deep k-qubit gate fusion (FuseKq, k ≤ 5)

**Labels:** `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 2
**Depends on:** P1-09, P1-10

**Description**
Generalise the existing 1q/2q fusion to fuse runs of adjacent gates spanning up to **k = 5** qubits into a single dense 2^k × 2^k unitary applied in one state-vector pass.

**Context**
The general antidote to the memory wall (P2-05) is raising arithmetic intensity: one pass over the 512 MiB state that does a 2^k × 2^k matrix–vector product per 2^k-amplitude block does O(2^k) FLOPs per amplitude moved, instead of O(1) for a single gate. Fusing up to ~5 qubits is the standard technique in Qiskit Aer and qsim and is the biggest lever for fusible circuits (VQE/QAOA/random brick-wall), which P2-05 measured at 2.5–2.8×@16 — better than QFT but still far from linear because they currently apply many small gates. `Fuse1qRuns` + `Fuse2q` already exist; this extends the same machinery to a configurable max-k.

**Technical Details**

- New `passes::FuseKq { max_qubits: usize }` (default 4–5; tunable): greedily grow a fused block over adjacent gates sharing a small qubit support, bounded by `max_qubits`.
- Build the dense 2^k × 2^k matrix by composing member gates in circuit order; emit a `Gate::UnitaryKq` (generalise the existing 2q dense kernel to a k-qubit dense kernel over the 2^k-amplitude block).
- Cost model: only fuse when the fused dense apply is cheaper than the sum of member passes (avoid fusing across very high qubits where the dense block explodes cache footprint — interacts with P2-04 grain findings and P2-09).
- AVX-512 dense k-qubit kernel (AoS + SoA), reuse the renormalised outer-walk indexing pattern from P1-07.

**Acceptance Criteria**

- [ ] Configurable `max_qubits`; `default_pipeline()` uses a sensible default (4 or 5).
- [ ] VQE/QAOA/random pass-count reduction and criterion speedup on EPYC, reported in the PR.
- [ ] Oracle equivalence vs unfused within 1e-12 across Tier-1 fixtures.

**Testing Requirements**

- Property test: fused k-qubit block ≡ sequential application on a generic state, for k = 3,4,5.
- Indexing-coverage test for the k-qubit dense kernel (integer reproduction of `block | offsets | j` disjointness).
- Benchmark before/after on the EPYC box.

**References**

- `docs/perf/phase2.md` §3 (arithmetic-intensity argument).
- Smelyanskiy et al. / Qiskit Aer fusion; qsim gate fusion (read, re-implement).
- ADR on P1-07 2q dense kernel (renormalised outer-walk).

-----

### [P2-08] Optional FP32 (single-precision) state-vector mode

**Labels:** `area:backend-sv`, `area:core`, `type:optimization`, `priority:medium`
**Milestone:** Phase 2
**Depends on:** P1-01, P0-09

**Description**
Add an opt-in single-precision (`Complex<f32>`) state-vector backend variant that halves the bytes moved per gate.

**Context**
On a bandwidth-bound kernel (P2-05), wall-clock scales with bytes streamed. `Complex<f64>` is 16 bytes/amplitude; `Complex<f32>` is 8 — a direct ~2× on memory traffic and therefore ~1.5–2× wall-clock for large n where DRAM streaming dominates, at the cost of accuracy (≈1e-6 instead of ≈1e-10). This is the cheapest single change that attacks the actual bottleneck (bytes), and is standard in Aer (`precision: single`). FP64 remains the default and the oracle-reference path; FP32 is an explicit large-n performance mode.

**Technical Details**

- Parameterise the SV backends over the float type (generic `T: Float` or a parallel `f32` instantiation of `CpuState`/`SoaState` + kernels), or a dedicated `Fp32SvBackend`.
- AVX-512 `f32` kernels: 16 lanes/zmm vs 8 for `f64` — extend the existing kernel set or generate via the same macros.
- CLI / API flag (`--precision f32`); default stays `f64`.
- Keep the conversion utilities in `aleph-core::statevector` consistent across precisions.

**Acceptance Criteria**

- [ ] Opt-in FP32 mode; FP64 remains default and unchanged.
- [ ] ~1.5–2× wall-clock vs FP64 on a bandwidth-bound workload at n ≥ 24 (EPYC), reported in the PR.
- [ ] Oracle equivalence vs Qiskit Aer single-precision within 1e-5 amplitudes; FP64 oracle path untouched.

**Testing Requirements**

- Property tests (normalization, unitarity) at the FP32 tolerance.
- Oracle comparison at 1e-5 for FP32; existing 1e-10 FP64 oracles still pass.
- Benchmark FP32 vs FP64 on the EPYC box.

**References**

- `docs/perf/phase2.md` §1 (bytes-moved is the bottleneck).
- Qiskit Aer `precision` option (read, re-implement).

-----

### [P2-09] Cache-blocked multi-gate application

**Labels:** `area:backend-sv`, `type:optimization`, `priority:medium`
**Milestone:** Phase 2
**Depends on:** P2-01

**Description**
Apply a *batch* of gates to each cache-resident tile of the state vector before moving on, and reorder qubit indices so frequently-interacting qubits map to low (cache-local) bits — keeping data hot in L2/L3 instead of streaming the whole state from DRAM per gate.

**Context**
P2-05 established that the per-gate full-state DRAM stream is the limiter. Gates on **low** qubits touch near-contiguous addresses that fit in cache; gates on **high** qubits stride across the whole array. If consecutive gates act within a cache-sized tile, the tile can be loaded once and many gates applied while it is hot — turning N DRAM passes into 1. This is the hardest CPU-side lever (it changes the apply schedule, not just a kernel) but it is the one that genuinely *avoids* the memory wall rather than working within it; complements P2-06/P2-07 (which reduce pass count) and the P2-03 NUMA placement.

**Technical Details**

- Block the state into L2/L3-sized tiles; for each tile, apply the maximal prefix of upcoming gates whose support is confined to the tile's low-qubit window before advancing.
- Qubit-relabelling pass (IR level) that maps high-interaction qubits to low bit positions, with the inverse permutation applied to results/measurements.
- Interacts with P2-04 grain tuning and P2-07 fusion; gate scheduling becomes tile-aware.
- Validate with `perf stat -e cache-misses,LLC-load-misses` (L2/L3 miss reduction is the primary signal, not just wall-clock).

**Acceptance Criteria**

- [ ] Measurable L2/L3 cache-miss reduction (`perf stat`) on a low-qubit-heavy circuit, reported in the PR.
- [ ] Speedup in the cache-resident regime (intermediate n where tiling helps), reported on the EPYC box.
- [ ] Oracle equivalence preserved (qubit relabelling is transparent to results) within 1e-12.

**Testing Requirements**

- Oracle equivalence with and without relabelling/tiling across Tier-1 fixtures.
- Property test: qubit-permutation round-trip preserves the state.
- `perf stat` cache-miss before/after on the EPYC box.

**References**

- `docs/perf/phase2.md` §1, §3 (memory-pass argument; low- vs high-qubit stride).
- QuEST / Aer cache-blocking and qubit-reordering (read, re-implement).

-----

# Phase 3 — Alternative Backends

Goal: stabilizer and MPS backends working; automatic backend selection.

-----

### [P3-01] Stabilizer simulator — Aaronson-Gottesman tableau

**Labels:** `area:backend-stab`, `type:feature`, `priority:critical`
**Milestone:** Phase 3
**Estimate:** XL
**Depends on:** P0-09

**Description**
Implement stabilizer simulator using the Aaronson-Gottesman tableau formalism.

**Context**
Clifford circuits (H, S, CNOT, measurements) can be simulated in O(n²) time and space per gate. This means thousands of qubits on a laptop. Essential for error correction simulation.

**Technical Details**

- Represent state as a (2n+1) × (2n+1) binary tableau (stabilizers + destabilizers + sign bits).
- Each Clifford gate updates the tableau in O(n) time.
- Reference: Aaronson & Gottesman (2004), “Improved Simulation of Stabilizer Circuits”.
- Use `BitVec` for the tableau rows.

**Acceptance Criteria**

- [ ] Stabilizer simulator handles H, S, CNOT, X, Y, Z gates
- [ ] Stabilizer of 1000 qubits, depth 100, runs in <1 second
- [ ] Verified against Stim on shared circuits
- [ ] Correctly rejects non-Clifford gates

**Testing Requirements**

- Equivalence vs. Stim on 100 random Clifford circuits.
- Property: Bell state preparation gives correct stabilizers.

**References**

- Aaronson, Gottesman. <https://arxiv.org/abs/quant-ph/0406196>
- <https://github.com/quantumlib/Stim> (Craig Gidney’s implementation)

-----

### [P3-02] Stabilizer — measurements with collapse

**Labels:** `area:backend-stab`, `type:feature`, `priority:critical`
**Milestone:** Phase 3
**Estimate:** L
**Depends on:** P3-01

**Description**
Implement projective measurement on stabilizer states.

**Context**
Two cases: deterministic measurement (qubit’s Z is in the stabilizer group) and random measurement. The random case requires updating the tableau by Gaussian elimination.

**Technical Details**

- Algorithm from Aaronson-Gottesman §3.
- Deterministic: outcome determined by tableau structure.
- Random: pick a stabilizer row that anticommutes with Z_q, use it to update.

**Acceptance Criteria**

- [ ] Measurement implemented for stabilizer backend
- [ ] Deterministic + random cases correct
- [ ] Bell pair: measuring qubit 0 → outcome forces qubit 1

**Testing Requirements**

- Equivalence vs. Stim.
- Statistical: GHZ state measurements → all-0 or all-1 50/50.

**References**

- Same as P3-01.

-----

### [P3-03] Stabilizer backend integration with `Backend` trait

**Labels:** `area:backend-stab`, `area:backend`, `type:feature`, `priority:high`
**Milestone:** Phase 3
**Estimate:** M
**Depends on:** P3-01, P3-02

**Description**
Wire stabilizer simulator into the unified `Backend` trait.

**Context**
End-to-end usability requires the stabilizer backend to plug into the same pipeline as state vector.

**Technical Details**

- Implement `Backend` for `StabilizerBackend`.
- Reject non-Clifford gates with clear error.
- Expose via CLI: `aleph run circuit.qasm --backend stabilizer`.

**Acceptance Criteria**

- [ ] Stabilizer reachable through unified API
- [ ] Clear errors on non-Clifford gates
- [ ] CLI option works

**Testing Requirements**

- Integration: run surface-code-cycle.qasm through stabilizer backend.

**References**

- Same as P3-01.

-----

### [P3-04] MPS backend — basic 1D chain

**Labels:** `area:backend-mps`, `type:feature`, `priority:high`, `research`
**Milestone:** Phase 3
**Estimate:** XL
**Depends on:** P0-09

**Description**
Implement Matrix Product State backend with fixed bond dimension and SVD truncation.

**Context**
MPS is the right backend for shallow / structured circuits with bounded entanglement. Enables 100+ qubit simulation for VQE/QAOA where state vector cannot.

**Technical Details**

- Represent state as a chain of tensors, each of shape (χ, 2, χ).
- 1q gate: contract with the local tensor.
- 2q gate (nearest neighbor): contract two tensors, apply gate, SVD-truncate back to bond dimension χ.
- 2q gate (non-adjacent): SWAP intermediate qubits first.
- Parameter: max bond dimension χ.
- Use `ndarray` or `nalgebra` for tensor operations.

**Acceptance Criteria**

- [ ] MPS backend handles 1q and 2q gates
- [ ] Bond dimension truncation with controlled error
- [ ] On VQE H₂ circuit at 4 qubits: matches state vector to machine precision
- [ ] On QAOA depth-3 at 50 qubits: produces reasonable results

**Testing Requirements**

- Equivalence vs. state vector backend for small n.
- Property: small bond dimension on weakly entangled states gives near-exact results.

**References**

- Vidal, “Efficient Classical Simulation of Slightly Entangled Quantum Computations” (2003).
- White, “Density Matrix Formulation for Quantum Renormalization Groups” (1992).
- <https://github.com/ITensor/ITensors.jl>
- <https://github.com/jcmgray/quimb>

-----

### [P3-05] MPS — SVD truncation with controlled error

**Labels:** `area:backend-mps`, `type:feature`, `priority:high`, `research`
**Milestone:** Phase 3
**Estimate:** L
**Depends on:** P3-04

**Description**
Implement SVD-based truncation with both fixed χ and error-bounded modes.

**Context**
After each 2q gate, the bond between two sites grows. We must truncate. Two strategies: fixed χ (predictable memory, possibly large error), or error-bounded (truncate based on Schmidt values below threshold).

**Technical Details**

- SVD of the combined tensor.
- Discard singular values below ε or beyond top χ.
- Track accumulated truncation error.
- Use LAPACK via `ndarray-linalg`.

**Acceptance Criteria**

- [ ] SVD truncation works in both modes
- [ ] Tracked error reported
- [ ] Performance benchmark vs. fixed χ

**Testing Requirements**

- Property: trivial truncation (χ = ∞) preserves state exactly.
- Property: error-bounded mode never exceeds specified error on test circuits.

**References**

- Schollwöck. “The density-matrix renormalization group in the age of matrix product states” (2011).

-----

### [P3-06] MPS — non-adjacent 2q gates via SWAP

**Labels:** `area:backend-mps`, `type:feature`, `priority:medium`
**Milestone:** Phase 3
**Estimate:** M
**Depends on:** P3-04

**Description**
Support 2q gates between non-adjacent qubits by inserting SWAP gates.

**Context**
MPS is naturally 1D; arbitrary topology requires SWAP networks. Smart ordering minimizes the number of SWAPs.

**Technical Details**

- For 2q gate on qubits (i, j) with |i - j| > 1: SWAP qubit i forward (or j backward) until adjacent, apply gate, SWAP back.
- Could be optimized at the IR level: choose qubit ordering that minimizes total SWAPs.

**Acceptance Criteria**

- [ ] Non-adjacent gates work
- [ ] Correctness verified vs. state vector
- [ ] Benchmark vs. always-SWAP-back vs. lazy approach

**Testing Requirements**

- Equivalence on circuits with non-local gates.

**References**

- Same as P3-04.

-----

### [P3-07] Automatic backend selection heuristic

**Labels:** `area:backend`, `area:ir`, `type:feature`, `priority:high`
**Milestone:** Phase 3
**Estimate:** L
**Depends on:** P3-03, P3-04

**Description**
Analyze circuit and pick the best backend automatically.

**Context**
Users shouldn’t need to know which backend to use. Heuristic: Clifford-only → stabilizer; shallow + structured → MPS; otherwise → state vector (and warn if too large).

**Technical Details**

- Scan circuit, compute features: gate types, depth, qubit count, entanglement estimate.
- Rules:
1. If all gates are Clifford → stabilizer.
1. If depth × bond-dim-estimate < threshold → MPS.
1. If n ≤ 28 → state vector.
1. Else → warn and use state vector (or refuse).

**Acceptance Criteria**

- [x] Heuristic implemented as `select_backend(circuit) -> BackendKind`
- [x] Manual override available
- [x] Test corpus selects expected backend in each category

**Testing Requirements**

- Unit: each rule fires on the right input.

**References**

- N/A.

-----

### [P3-08] Stabilizer bit-slicing (O(n/64) tableau)

**Labels:** `area:backend`, `type:optimization`, `priority:medium`
**Milestone:** Phase 3 (deferred)
**Estimate:** M
**Depends on:** P3-02

**Status:** Deferred follow-up. **Schedule at the end of Phase 4 (or later)** — after the Phase-4 benchmarks identify where the stabilizer backend actually hurts. Per the golden rule, do not optimize blindly; measure first.

**Description**
Replace the current naive O(n)-per-gate row-major tableau update with Stim-style bit-sliced columns so gate application and `rowsum` become O(n/64) using packed-`u64` word operations (and optionally SIMD). This is the single biggest lever to close the speed gap with Stim.

**Context**
P3-01 deliberately shipped the naive O(n) row-major hot loop (hoisted word-offset/mask + branchless updates got it to ~0.48 s for 1000q×depth100 on EPYC, under the 1 s exit bar). Bit-slicing was explicitly deferred because it complicates the P3-02 `rowsum` sign-tracking. We are correct vs the Stim oracle but several× slower than Stim on large/deep Clifford circuits.

**Technical Details**

- Store the x/z tableaux as bit-packed columns (or rows) of `u64`; apply H/S/CNOT and Pauli sign rules with word-wise XOR/AND across `ceil(n/64)` words.
- Re-derive the AG `rowsum` (phase exponent `g` + sign bit) under the packed layout — this is the tricky part; keep the existing scalar implementation as an oracle to diff against.
- Keep the `BitGrid` accessor API; add SIMD (AVX-512 `vp*q`) only after the scalar bit-sliced version is proven.

**Acceptance Criteria**

- [x] Bit-sliced tableau passes the existing Stim oracle + symplectic-invariant proptests bit-for-bit vs the scalar implementation. _(word-parallel + AVX-512 `rowsum`; bit-exact vs preserved scalar `rowsum` and Stim oracles green d=3..11.)_
- [x] Measured speedup on a large/deep Clifford bench (criterion before/after), reported honestly. _(surface-d11 1.375→0.842 ms = 1.63×, 12.52×→7.66× vs Stim; the design's ≤2× target was not met — cycle is now gate-bound, see `docs/perf/surface_code.md`.)_

**References**

- Stim (quantumlib/Stim) bit-sliced tableau; Aaronson–Gottesman §3.

-----

### [P3-09] MPS multithreading + lazy SWAP permutation tracking

**Labels:** `area:backend`, `type:optimization`, `priority:medium`
**Milestone:** Phase 3 (deferred)
**Estimate:** L
**Depends on:** P3-06

**Status:** Deferred follow-up. **Schedule at the end of Phase 4 (or later)** — after benchmarks show MPS hot spots. Measure first.

**Description**
Two MPS performance follow-ups: (1) multithread the per-bond SVD / tensor contractions (currently single-threaded faer); (2) replace the always-swap-back non-adjacent-2q strategy with **lazy permutation tracking** so a long-range gate does not pay the double cost of forward + reverse SWAP ladders on every application.

**Context**
P3-04/05 shipped a single-threaded MPS (faer SVD, no parallelism). P3-06 added non-adjacent 2q gates via an always-swap-back SWAP network: each long-range gate runs a forward ladder, applies the gate, then runs the reverse ladder — `2·(distance−1)` NN SWAPs per gate, and the lazy strategy was explicitly documented as deferred.

**Technical Details**

- Track a current site↔qubit permutation in `MpsState`; route reads (measure/sample/expectation/probabilities/dense) through it instead of forcing `site == qubit`. Only swap when genuinely needed; amortize across consecutive long-range gates.
- Parallelize the bond SVD / canonical-form sweeps (rayon over independent bonds where the orthogonality-center discipline allows), being careful not to break the truncation-error accounting.

**Acceptance Criteria**

- [x] Lazy-permutation path matches the always-swap-back result vs `NaiveSvBackend` to 1e-10 (the P3-06 oracle), with fewer applied SWAPs on a long-range benchmark. *(d−1 vs 2(d−1) SWAPs by counter; EPYC wall-clock −9/−17/−22 % at dist 4/8/11; see `docs/perf/mps_parallel.md`.)*
- [x] Multithreaded SVD shows a measured speedup on a wide-bond bench, with the truncation-error oracle (ε=0 ⇒ exact) still passing. *(1.57× @16T at χ=512 on EPYC; parallelism is a measured pessimization at χ≤256 → shipped opt-in via the `parallel` cargo feature, default off. ε=0 and Par-invariance oracles pass.)*

**References**

- P3-06 design/notes (always-swap-back; lazy strategy deferred). Schollwöck MPS review for canonical-form parallelism.

-----

### [P3-10] MPS 100+ qubit shallow-circuit demo (close the Phase-3 exit metric)

**Labels:** `area:backend`, `area:bench`, `type:test`, `priority:medium`
**Milestone:** Phase 3 (deferred)
**Estimate:** S
**Depends on:** P3-04, P3-05

**Status:** Deferred validation. **Natural to fold into Phase 4 benchmarking** — run it as one of the Phase-4 benches rather than as standalone work.

**Description**
Demonstrate the MPS backend running a **100+ qubit shallow, low-entanglement circuit** within a sane time/bond budget, with the result validated against a tractable reference. This closes the ROADMAP Phase-3 exit metric — *"MPS handles 100+ qubit shallow circuits"* — with an actual measured number instead of on faith (P3-04/05 were only validated at ~50 qubits).

**Context**
ROADMAP §7 Phase-3 exit: *"Stabilizer backend handles 1000+ qubit Clifford circuits; MPS handles 100+ qubit shallow circuits."* The stabilizer half is demonstrated (1000q×depth100 ≈ 0.48 s). The MPS half is architecturally supported (cap 1024 qubits) but our shipped tests topped out around 50 qubits, so the ≥100-qubit claim is currently unverified.

**Technical Details**

- Pick a shallow nearest-neighbor circuit at `n ≥ 100` (e.g. a few layers of NN brickwork / shallow QAOA) where the entanglement stays low so a modest χ suffices.
- Run it on the MPS backend at bounded χ; assert it completes under an explicit wall-time / memory budget and that `max_bond_reached()` / truncation error stay sane.
- Full state-vector reference is intractable at `n ≥ 100`, so validate **local observables** (single-/two-qubit expectations) against an analytic or lightweight reference, or check a known invariant of the chosen circuit.
- Record the number in a short perf note (or the Phase-4 report).

**Acceptance Criteria**

- [x] MPS backend runs a `≥ 100`-qubit shallow circuit to completion within a stated time/memory budget.
- [x] Result validated against a tractable reference (local observables / known answer), not just "it ran".
- [x] Number recorded so the Phase-3 MPS exit metric is closed with evidence.

**Result (2026-06-10):** n=128 / depth-6 / χ=64 non-Clifford brickwork runs exactly (max bond 8, truncation 1.07e-13) in **10.3 ms** on EPYC; ⟨Z⟩/⟨ZZ⟩ validated to 1e-10 vs an exact light-cone SV reference. CI guard `mps_128q_shallow_demo`. See `docs/perf/mps_100q.md`.

**References**

- ROADMAP §7 Phase-3 exit metric. P3-04/P3-05 design notes.

-----

### [P3-11] Stabilizer word-parallel gate kernels (H/S/CNOT) — close the gate-bound gap

**Labels:** `area:backend`, `type:optimization`, `priority:medium`
**Milestone:** Phase 3 (deferred)
**Estimate:** L
**Depends on:** P3-08

**Status:** Deferred follow-up, **created from P3-08's profiling** (PR #134). The original P3-08 scope said "gate application *and* `rowsum` become O(n/64)"; P3-08 deliberately narrowed to `rowsum` (it was the measured hot path) and **explicitly scoped gates out**. The profile after P3-08 flipped the bottleneck — this ticket picks up the gate half. Schedule when stabilizer gate throughput matters (e.g. deeper QEC / large Clifford circuits); not urgent for v0.1.

**Description**
Word-parallelize (and where it pays, SIMD) the Clifford **gate** kernels (`H`, `S`, `CNOT`, and the Pauli sign updates) in the stabilizer tableau, so a gate stops touching the tableau one bit at a time per row.

**Context**
After P3-08 word-parallelized `rowsum`, a `perf record` of the surface-code d=11 cycle attributes time as **`Tableau::cnot` 70.9% + `Tableau::h` 15.3% ≈ 86%**, with `measure`/`rowsum` down to ~5–11% (see `docs/perf/surface_code.md` P3-08 addendum). So the measurement path is no longer the bottleneck; the gate kernels are. P3-08 got surface-d11 from **12.52× → 7.66× vs Stim** (1.63× cycle speedup); the remaining gap is almost entirely gates. This is the natural next lever toward the P3-08 design's unmet hard target (surface-d11 ≤ 2× Stim).

**Technical Details**

- Each Clifford gate touches **column `a`** (and `b` for `CNOT`) across all `2n+1` rows: it reads/modifies the single bit at word `a>>6`, mask `1<<(a&63)`, in every row. Under the current **row-major** `BitGrid` that is a **strided, single-bit, branchy `get`/`set` per row** — the opposite access pattern from `rowsum` (which wants contiguous row XOR, which is exactly why row-major is right *for `rowsum`*).
- The core tension: **`rowsum` wants row-major; gates want column-major.** Options to evaluate (likely an ADR):
  1. **Stay row-major, de-scalarize:** hoist the word/mask out of the row loop and apply each gate's per-row update with branchless word arithmetic across the 2n rows (extends the P3-01 hoisting). Cheapest; bounded upside since it's still one row per step.
  2. **Dual / transposed layout:** keep a column-major shadow (or transpose on demand) so a gate's target column is a contiguous `u64` span → word-parallel/SIMD **across rows**. Then `rowsum` either uses the row-major copy or pays a strided cost — measure the sync/transpose overhead. This is essentially Stim's bit-sliced approach.
  3. **SIMD across rows** once the column is contiguous (option 2), mirroring the P3-08 AVX-512 + `rowsum_dispatch` pattern.
- Preserve the existing scalar gate kernels as a `#[cfg(test)]` reference and **bit-exact diff** against them; the Stim oracles (d=3..11) remain the independent end-to-end gate. Do not weaken the correctness gate to chase the perf number (P3-08 precedent).

**Acceptance Criteria**

- [x] Word-parallel (and/or SIMD) `H`/`S`/`CNOT` kernels, bit-for-bit identical to the preserved scalar kernels (proptest) with all Stim oracles green at d=3..11.
- [x] Measured surface-code cycle speedup (criterion before/after), reported honestly; restate the aleph/Stim d=11 ratio. **Stretch MET:** d=11 = 4.69× cycle speedup, aleph/Stim **7.66× → 1.64×** (≤ 2× target reached; d=3/5 now faster than Stim). See `docs/perf/surface_code.md` P3-11 addendum.
- [x] If a layout change lands (option 2/3), an ADR documenting the row-major vs column-major vs dual trade-off for the stabilizer tableau. → ADR 0013 (lazy dual-orientation tableau).

**References**

- P3-08 design + perf addendum (`docs/perf/surface_code.md`); PR #134.
- Stim (quantumlib/Stim) bit-sliced tableau; Aaronson–Gottesman §2–3.

-----

### [P3-12] MPS: Gate::Swap as an O(1) permutation relabel

**Labels:** `area:backend`, `type:optimization`, `priority:medium`
**Milestone:** Phase 4.5 (adopted from Phase 3)
**Estimate:** S
**Depends on:** P3-09

**Description**
Route explicit `Gate::Swap` through the P3-09 lazy-permutation maps as a pure relabel (swap two entries in `site_of_qubit`/`qubit_of_site`) instead of the current physical path (SWAP ladder + theta gemm + truncated SVD per gate).

**Context**
P3-09 `/code-review` finding: the lazy router makes a logical SWAP expressible as a constant-time map update with exactly zero tensor work, zero bond growth, and zero truncation error — but a user-level `swap a, b` still pays `(d−1)+1` truncated SVDs and accrues avoidable `trunc_error`. Inverts the PR's own amortization story: `CNOT(0,4)` is lazily routed while `Swap(0,4)` physically drags tensors. SWAP-dense circuits (routing-aware compiler output) are the motivating workload.

**Technical Details**

- Fast path at the top of `MpsState::apply_2q` (or in `MpsBackend::apply_gate` dispatch before `matrix_4x4`): `if matches!(g.gate, Gate::Swap) { relabel; return Ok(()); }`. Composes with the router: subsequent gates route through the updated permutation.
- Decide whether `swaps_applied` counts relabels (recommendation: no — it counts *physical* SWAPs; add a separate `relabels` stat if needed).

**Acceptance Criteria**

- [x] Explicit-SWAP circuits match `NaiveSvBackend` to 1e-10 (extend the SV oracle with SWAP-dense cases, including SWAP→CNOT interleavings and reads after relabel). — *`swap_dense_matches_sv` + `random_swap_injection_matches_sv` proptest in `sv_equivalence.rs`.*
- [x] A SWAP-dense benchmark shows the relabel path applying zero physical SWAPs (`swaps_applied` unchanged) with measured wall-clock win. — *`benches/swap_dense.rs`: relabel vs CNOT-decomposed of the same permutation; the SWAP gates discharge as relabels (`relabels`++, no physical SWAP from them). Local M-series n=14/χ=32: **8.0 µs vs 62.7 µs (≈7.8×)**.*
- [x] `trunc_error` for an explicit-SWAP circuit at saturated χ is bit-identical to the SWAP-free relabeled equivalent. — *`swap_relabel_adds_no_truncation_error`: `to_bits()`-identical `trunc_error` and bit-identical final state for SWAP·∏Gτ·SWAP vs ∏G at χ=2.*

**Testing Requirements**

- Oracle equivalence incl. controlled gates after relabel (MSB convention through the permutation); proptest with random SWAP injection.

**References**

- P3-09 design spec + `docs/perf/mps_parallel.md`; /code-review finding #4 (PR #146).

-----

### [P3-13] MPS: size-thresholded per-call parallelism (replace the process-global Par control plane)

**Labels:** `area:backend`, `type:optimization`, `priority:medium`
**Milestone:** Phase 4.5 (adopted from Phase 3)
**Estimate:** M
**Depends on:** P3-09

**Description**
Choose faer parallelism per operation from the measured χ-crossover instead of faer's process-global default: small ops run `Par::Seq`, wide-bond ops use the rayon pool. Removes the feature-unification trap where any crate enabling `faer/rayon` silently flips every `MpsBackend` user in the process onto the χ≤256 pessimization.

**Context**
P3-09 measured the crossover (EPYC 16c): rayon pool is a 1.5×–19× pessimization at χ≤256 and a 1.57× win at χ=512 (@16T). The current control plane is the `parallel` cargo feature + faer's global (`GLOBAL_PARALLELISM` defaults to `Rayon(0)` once the feature is compiled in anywhere in the graph). The three matmul sites in `mps.rs` already pass `get_global_parallelism()` explicitly — substituting a size-thresholded helper there is trivial; SVD/QR go through high-level `thin_svd()`/`qr()` which read the global, so they need faer's lower-level APIs (explicit `Par` + `MemStack`) or an upstream knob.

**Technical Details**

- `fn par_for(rows, cols) -> Par`: `Par::Seq` below a threshold calibrated from `docs/perf/mps_parallel.md` (crossover between 512×1024 and 1024×2048 operands), global otherwise.
- Matmul sites: direct substitution. SVD/QR: evaluate `faer::linalg::svd::*`/`qr::*` with explicit `Par` — weigh verbosity against the win; measure before committing (the SVD is the dominant cost).
- Re-evaluate whether the `parallel` cargo feature can then default ON safely (small ops no longer regress), simplifying the user story.
- Also fixes the thread-invariance test's global-toggle isolation (compare Seq vs rayon as plain arguments).

**Acceptance Criteria**

- [ ] With parallelism compiled in and 16 threads, `nn_qaoa` (χ=64) and `wide_bond` χ=128/256 are within noise of the sequential build (no pessimization), and χ=512 retains ≥ the P3-09 1.57× speedup. — *χ≤256 cells: met; χ=512: 1.52× of the 1.57× retained (structural single-threshold ceiling; see docs/perf/mps_parallel.md § P3-13)*
- [x] ε=0 and Par-invariance oracles pass.

**Testing Requirements**

- EPYC thread sweep across χ=64/128/256/512; criterion baselines vs the P3-09 numbers in `docs/perf/mps_parallel.md`.

**References**

- `docs/perf/mps_parallel.md` (crossover data); /code-review findings #3 and #5 (PR #146); faer 0.24 `set_global_parallelism` docs.

-----

### [P3-14] MPS: hot-path scratch arena (kill the per-gate allocation churn)

**Labels:** `area:backend`, `type:optimization`, `priority:low`
**Milestone:** Phase 4.5 (adopted from Phase 3)
**Estimate:** M
**Depends on:** P3-09

**Description**
Reuse workspace buffers across gates in `apply_2q_adjacent`/`move_center_*` instead of allocating ~8–12 fresh heap buffers per 2q gate (theta, theta2, SVD workspace, `u_kept`/`vt_kept`, Q/R/absorbed/qh, two `Site`s per op).

**Context**
P3-09 honest note: the `long_range` dist1 microcircuit cell regressed +11.9 % — at χ≤32 allocator round-trips and faer workspace setup dominate the math. The copy chain also includes avoidable full-matrix copies (`svd.U()` → `u_kept` → `Site`, twice per gate) and three `Mat::zeros` memsets immediately overwritten by `Accum::Replace` gemms, plus the `q.adjoint().to_owned()` materialization in `move_center_left`.

**Technical Details**

- Persistent scratch in `MpsState` (faer `MemBuffer`/`MemStack` + two `Mat`s sized to the max bond, grown monotonically), threaded into the hot path.
- Return `MatRef` subviews from `truncated_svd` (or write directly into `Site` buffers), folding the `s·Vᴴ` scaling and V-conjugation into the single write.
- Make `from_group_right_faer` generic over conjugated views (or write the site with explicit transpose+conj indexing) to drop the `to_owned()`.
- Measure first: profile the dist1 cell to confirm the allocator attribution before restructuring (CLAUDE.md hierarchy).

**Acceptance Criteria**

- [ ] `long_range` dist1 within ±5 % of the pre-P3-09 baseline on EPYC; no regression on any other `long_range`/`nn_qaoa`/`wide_bond` cell.
- [ ] Full oracle suite unchanged (1e-10).

**Testing Requirements**

- criterion before/after on EPYC; `cargo flamegraph`/`perf` evidence for the allocation attribution.

**References**

- `docs/perf/mps_parallel.md` honest notes; /code-review finding #7 (PR #146).

-----

### [P3-15] Hoist the bit-permutation helper into aleph-core (dedupe aleph-sv ↔ aleph-mps)

**Labels:** `area:core`, `type:refactor`, `priority:low`
**Milestone:** Phase 3 (deferred)
**Estimate:** S
**Depends on:** P3-09, P2-09

**Description**
The workspace now has two independent copies of the physical→logical bit-permutation index arithmetic: `aleph-sv/src/perm.rs` (`bit_permute_buf`, P2-09 `unpermute_state`) and the P3-09 scatter pass in `MpsState::dense_statevector`. Hoist one tested helper into `aleph-core` and use it from both.

**Context**
P3-09 /code-review reuse finding: the MPS copy is scatter-based with the inverse map, so a reviewer cannot see by inspection that the two agree; any bit-order convention fix (cf. the ADR-0004 bit-order doc issue noted at v0.1 release) must be applied twice. `aleph-sv`'s helper is `pub(crate)` and carries asymmetric-permutation unit tests that MPS would inherit for free.

**Technical Details**

- New `aleph-core` module (or extend `statevector` conversions, where AoS/SoA conversion utilities already live per CLAUDE.md) with the gather-form helper + its tests; keep the in-place cycle-following variant in scope only if the MPS 2×2^n peak-memory note (test-only path) is judged worth fixing in the same move.

**Acceptance Criteria**

- [x] One shared helper; both call sites migrated; aleph-sv perm tests moved/extended to cover the MPS usage (asymmetric 3-cycle case included). — *`aleph_core::bit_permute_buf` (+ tests incl. `asymmetric_three_cycle_matches_scatter` proving gather(perm=`site_of_qubit`) ≡ MPS's `qubit_of_site` scatter); aleph-sv `perm.rs` reduced to typed wrappers; MPS `dense_statevector` phase-2 calls the helper.*
- [x] No behavior change (oracle suites of both crates green). — *aleph-sv + aleph-mps full suites unchanged.*

**References**

- `crates/aleph-sv/src/perm.rs`; `MpsState::dense_statevector` phase-2; /code-review finding #8 (PR #146).

-----

### [P3-16] Shared bench/test fixtures: brickwall builder + distribution-closeness helper

**Labels:** `area:bench`, `type:refactor`, `priority:low`
**Milestone:** Phase 3 (deferred)
**Estimate:** S
**Depends on:** P3-09

**Description**
Deduplicate the circuit fixtures multiplied across the workspace: ≥6 private brickwall builders (incl. two added by P3-09 in `wide_bond.rs` and the thread-invariance test), the 4× copy-pasted `g()` GateInstance helper inside aleph-mps alone, and the ad-hoc ±0.02 empirical-distribution check that re-implements aleph-oracle's calibrated 5σ `assert_distribution_close`.

**Context**
P3-09 /code-review reuse finding: the `wide_bond` χ-saturation constants (L1=16, L2=20) are calibrated against its local builder with no shared test — editing the builder silently invalidates the saturation claim. `aleph-benches` (benches/src/lib.rs) already hosts `random_brickwall_circuit` and the other Tier-1 builders; aleph-oracle's distribution helper is private. Note: P3-13 moved the thread-invariance brickwall into `mps.rs::state_invariant_seq_vs_rayon` as a hand-rolled `MpsState` gate loop (to use the test-only `par_override` seam), so the unification pass must restructure that test, not just swap a builder call.

**Technical Details**

- Parameterize `aleph_benches::random_brickwall_circuit` (gate choice / angle schedule) or add a `brickwall_ry_cnot_rz` variant; make aleph-benches a dev-dependency of aleph-mps; replace both P3-09 builders and assert `max_bond_reached` saturation in a cheap test so the calibration is pinned.
- Expose `assert_distribution_close` (pub testing module or aleph-test) and use it in `lazy_perm_sample_matches_probabilities`.

**Acceptance Criteria**

- [x] One brickwall definition serves wide_bond + thread-invariance (+ existing duplicates where the swap is mechanical); saturation pinned by a test, not a comment. — *`aleph_benches::brickwall_ry_cnot_rz` drives both `wide_bond.rs` and the restructured `state_invariant_seq_vs_rayon` (via a `#[cfg(test)] MpsBackend::with_par_override` seam so the test runs a `Circuit` through `run`); `brickwall_saturates_bond_cap` pins the builder's cap-saturation property (cheap n=10/χ=16 cell). Shared `aleph_benches::g` replaces the copy-pasted `GateInstance` helper across all aleph-mps benches + tests.*
- [x] MPS sampling test uses the calibrated distribution helper. — *`assert_distribution_close` made `pub` in aleph-oracle; `lazy_perm_sample_matches_probabilities` and the `sample_matches_probabilities` unit test now use the 5σ band instead of ad-hoc ±0.02.*

**References**

- `benches/src/lib.rs`; `crates/aleph-oracle/src/harness.rs` (`assert_distribution_close`); /code-review findings #10 (PR #146).

-----

# Phase 4 — Algorithm Benchmarks & v0.1 Release

Goal: comprehensive benchmarks against published baselines; first public release.

-----

### [P4-01] QFT benchmark and reference implementation

**Labels:** `area:bench`, `type:test`, `priority:high`
**Milestone:** Phase 4
**Estimate:** S
**Depends on:** P1-14

**Description**
Implement and benchmark QFT at n = 10, 15, 20, 25, 30.

**Context**
QFT is the canonical “many controlled phases” test. Exercises diagonal-gate optimization.

**Technical Details**

- OpenQASM file: standard QFT decomposition.
- Benchmarks measure time per qubit count.
- Compare: our backend(s) vs. Qiskit Aer.

**Acceptance Criteria**

- [x] QFT runs to 30 qubits on state vector
- [x] Results match Qiskit
- [x] Benchmark report row

**Testing Requirements**

- QFT(|0…0⟩) followed by inverse QFT gives back |0…0⟩.

**References**

- Nielsen & Chuang, §5.1.

-----

### [P4-02] Grover benchmark

**Labels:** `area:bench`, `type:test`, `priority:high`
**Milestone:** Phase 4
**Estimate:** S
**Depends on:** P1-14

**Description**
Grover for n = 4, 8, 12, 16 with marked-state oracles.

**Context**
Tests oracle (multi-controlled phase) + diffusion pattern. Lots of repeated structure → good for IR optimization tests.

**Technical Details**

- Standard Grover circuit with O(√N) iterations.
- Verify amplified probability for marked state.

**Acceptance Criteria**

- [ ] Grover converges for tested sizes
- [ ] Benchmark report row

**Testing Requirements**

- Marked-state probability > 0.9 after √N iterations.

**References**

- Nielsen & Chuang, §6.

-----

### [P4-03] QPE benchmark

**Labels:** `area:bench`, `type:test`, `priority:medium`
**Milestone:** Phase 4
**Estimate:** M
**Depends on:** P4-01

**Description**
Quantum phase estimation on a known unitary.

**Context**
Builds on QFT; foundational for chemistry and Shor.

**Technical Details**

- Estimate phase of a U with known eigenvalues.
- Compare estimate accuracy and runtime.

**Acceptance Criteria**

- [ ] QPE returns correct phase within bit precision
- [ ] Benchmark report row

**Testing Requirements**

- Phase estimate matches analytical eigenvalue.

**References**

- Nielsen & Chuang, §5.2.

-----

### [P4-04] VQE benchmark — H₂ molecule

**Labels:** `area:bench`, `type:test`, `priority:critical`
**Milestone:** Phase 4
**Estimate:** L
**Depends on:** P1-14

**Description**
Variational quantum eigensolver for H₂ ground state energy with hardware-efficient ansatz.

**Context**
VQE is the dominant NISQ-era workload. Many short circuits run, many expectation values. Heavy on parameterized gates and gate fusion.

**Technical Details**

- Hamiltonian from Jordan-Wigner mapping (or use a fixed precomputed Pauli sum).
- Hardware-efficient ansatz: alternating Ry + CNOT chain.
- Classical optimizer: SciPy / nlopt called from Python frontend.
- Measure: time per energy evaluation, total optimization time.

**Acceptance Criteria**

- [ ] VQE converges to known H₂ ground state energy
- [ ] Benchmark: time per energy eval at 4, 6, 8 qubits
- [ ] Comparison with Qiskit / PennyLane

**Testing Requirements**

- Energy within chemical accuracy of FCI result.

**References**

- Peruzzo et al. “A variational eigenvalue solver on a quantum processor” (2014).
- McClean et al. “The theory of variational hybrid quantum-classical algorithms” (2016).

-----

### [P4-05] QAOA benchmark — Max-Cut on small graphs

**Labels:** `area:bench`, `type:test`, `priority:critical`
**Milestone:** Phase 4
**Estimate:** L
**Depends on:** P1-14

**Description**
QAOA p=1, p=2, p=3 on Max-Cut for graphs of 6, 10, 14 nodes.

**Context**
Common NISQ algorithm with structured (cost + mixer) layers. Good MPS candidate.

**Technical Details**

- Random graphs of given sizes.
- Compare classical optimum (exact for small graphs) to QAOA result.
- Run with multiple backends; MPS should shine for low p.

**Acceptance Criteria**

- [ ] QAOA produces approximations within 0.9× optimal for small graphs
- [ ] Benchmark across backends
- [ ] Time scaling report

**Testing Requirements**

- Sanity: p = 1 gives expected approximation ratio.

**References**

- Farhi, Goldstone, Gutmann. “A Quantum Approximate Optimization Algorithm” (2014).

-----

### [P4-06] Random circuit benchmark (Sycamore-style)

**Labels:** `area:bench`, `type:test`, `priority:high`
**Milestone:** Phase 4
**Estimate:** M
**Depends on:** P1-14

**Description**
Random circuit at n = 20, 24, 28, 30, depth 10–20.

**Context**
Worst case for state vector simulation (maximum entanglement). Useful for stress-testing optimizations.

**Technical Details**

- Random single-qubit gates from {√X, √Y, √W} (Sycamore-style).
- 2-qubit entangling gates on a brick-wall pattern.
- Measure final-state amplitude distributions.

**Acceptance Criteria**

- [ ] Runs at n = 30 on state vector
- [ ] Benchmark report row
- [ ] Linear cross-entropy benchmarking (XEB) value computed

**Testing Requirements**

- XEB ≈ 1 for our simulator (we’re noiseless).

**References**

- Arute et al. (Google) “Quantum supremacy using a programmable superconducting processor” (2019).

-----

### [P4-07] Surface code 1-cycle benchmark (stabilizer)

**Labels:** `area:bench`, `area:backend-stab`, `type:test`, `priority:high`
**Milestone:** Phase 4
**Estimate:** M
**Depends on:** P3-03

**Description**
Surface code 1 syndrome extraction cycle at distance d = 3, 5, 7, 9, 11.

**Context**
Showcases the stabilizer backend. QEC is *the* killer app for stabilizer simulation.

**Technical Details**

- Generate surface code stabilizer measurement circuits.
- Run with stabilizer backend; compare to Stim.
- Measure time per cycle.

**Acceptance Criteria**

- [x] Cycles run to d = 11 (≈ 240 physical qubits)
- [x] Match Stim output
- [x] Benchmark report row

**Testing Requirements**

- [x] Logical X / Z operator detection works as expected.

**References**

- Fowler et al. “Surface codes: Towards practical large-scale quantum computation” (2012).
- <https://github.com/quantumlib/Stim>

-----

### [P4-08] v0.1 public release — benchmark report + Python bindings

**Labels:** `area:docs`, `area:python`, `area:infra`, `type:feature`, `priority:critical`
**Milestone:** Phase 4
**Estimate:** L
**Depends on:** P4-01 through P4-07

**Description**
Tag v0.1, publish benchmark report, release Python bindings on PyPI.

**Context**
First public milestone. After this, the project has a footprint.

**Technical Details**

- pyo3 bindings for `Circuit`, `Backend`, `run`.
- maturin packaging.
- PyPI release.
- Comprehensive benchmark report: `docs/perf/v0.1.md`.
- README updated with quickstart.
- LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md.
- GitHub release with binaries (Linux, macOS).

**Acceptance Criteria**

- [ ] `pip install aleph` works
- [ ] Python quickstart works
- [ ] Benchmark report published
- [ ] GitHub release tagged

**Testing Requirements**

- Install from clean environment; tutorial works.

**References**

- <https://www.maturin.rs/>
- <https://pyo3.rs/>

-----

### [P4-09] PyPI publication for aleph-sim

**Labels:** `area:python`, `area:infra`, `type:feature`, `priority:high`
**Milestone:** Phase 4
**Estimate:** S
**Depends on:** P4-08

**Description**
Publish the `aleph-sim` wheels to PyPI so `pip install aleph-sim` works without downloading from a GitHub release.

**Context**
Deferred by owner decision during P4-08: v0.1 shipped wheels attached to the GitHub release only. The release pipeline (`release.yml`) already builds portable artifacts (manylinux_2_28 x86_64 + macOS arm64, abi3-py312); publication is the only missing step.

**Technical Details**

- PyPI **trusted publishing** (OIDC) from GitHub Actions — no long-lived API token. Configure the publisher on PyPI for this repo + `release.yml`, then add a `publish` job (`pypi` environment, `permissions: id-token: write`) using `pypa/gh-action-pypi-publish`, gated on the release job.
- Dry-run against TestPyPI first; verify `pip install -i https://test.pypi.org/simple/ aleph-sim` in a clean venv.
- Decide trigger discipline: publish on tag push together with the draft release, or only after the owner publishes the GitHub release (e.g. `release: types: [published]` workflow trigger — safer, keeps the manual verification gate).
- Verify the package name `aleph-sim` is still free at execution time; register on first upload.

**Acceptance Criteria**

- [x] `pip install aleph-sim` works in a clean venv (Linux x86_64 + macOS arm64, Python ≥ 3.12)
- [x] Publication is automated in `release.yml` (no manual twine step)
- [x] README/crate-README install instructions updated to prefer PyPI

**Testing Requirements**

- TestPyPI dry-run before the real upload; clean-venv install + `scripts/python/test_aleph.py` suite against the PyPI-installed package.

**References**

- <https://docs.pypi.org/trusted-publishers/>
- <https://github.com/pypa/gh-action-pypi-publish>

-----

### [P4-10] CI job for the Python binding tests

**Labels:** `area:python`, `area:infra`, `type:infra`, `priority:high`
**Milestone:** Phase 4
**Estimate:** S
**Depends on:** P4-08

**Description**
Add a CI job that builds the wheel with maturin and runs `scripts/python/test_aleph.py`, so the Python bindings are gated per-PR instead of only at release time.

**Context**
P4-08 shipped the binding test suite (14 tests) as a manual release gate because `ci.yml` had no maturin step. Binding regressions (e.g. a gate-method signature change, an error-mapping break) currently surface only when someone builds a wheel by hand. The suite runs in <1 s; the cost is the maturin build (~2-4 min warm).

**Technical Details**

- New `test-python` job in `ci.yml`: GitHub-hosted `ubuntu-latest` (do NOT add load to the self-hosted runner), `actions/setup-python` 3.12, `maturin build --release --features python` (or `maturin develop` into a venv), then `python -m unittest discover -s scripts/python -v`.
- Make it a **gating** check like the Rust test jobs; same trusted-PR guard as the other jobs.
- Reuse `Swatinem/rust-cache` so the Rust dep tree is warm; expect ~3-5 min cold.
- Keep the suite import-guard (`skipUnless`) so local runs without the module still skip gracefully.

**Acceptance Criteria**

- [ ] PRs that break a Python binding fail CI
- [ ] Job runs on GitHub-hosted runners only
- [ ] Wall-clock ≤ ~6 min warm cache

**Testing Requirements**

- Deliberately break a binding on a scratch branch and confirm the job goes red; revert.

**References**

- `scripts/python/test_aleph.py` (P4-08), `.github/workflows/release.yml` smoke-test step (same recipe).

-----

### [P4-11] numpy-backed `statevector()` return

**Labels:** `area:python`, `type:feature`, `priority:medium`
**Milestone:** Phase 4
**Estimate:** M
**Depends on:** P4-08

**Description**
Return the state vector to Python as a numpy `complex128` array instead of a list of `PyComplex` objects.

**Context**
P4-08's `RunResult.statevector()` materializes one Python complex object per amplitude — 2^n heap objects (~56 B each). At n=25 that is ~1.9 GiB of Python objects for a 512 MiB state; at the n=28 cap it OOMs. The docstring warns about it; the fix is the standard approach in Qiskit/QuEST bindings: hand numpy a contiguous buffer.

**Technical Details**

- Add `numpy` crate (rust-numpy, matches pyo3 0.22) behind the existing `python` feature; return `Bound<'py, PyArray1<Complex64>>`.
- `aleph_core::Complex` is `num_complex::Complex<f64>` and `CpuState::amplitudes()` is a contiguous `&[Complex]` — `PyArray1::from_slice` (copy) is the simple correct first step; a zero-copy view into the stored `CpuState` is possible but must pin the buffer's lifetime to the `RunResult` object (rust-numpy `unsafe` borrow APIs) — copy first, optimize later if profiling demands.
- `numpy` becomes a runtime dependency of the wheel — add to `pyproject.toml` `dependencies`.
- Keep `.counts()` unchanged.

**Acceptance Criteria**

- [ ] `statevector()` returns `numpy.ndarray` dtype `complex128`, shape `(2**n,)`
- [ ] n=25 statevector retrieval allocates O(state) memory, not O(state × 56 B objects)
- [ ] Existing Python tests updated and green; amplitude values identical to v0.1 behavior

**Testing Requirements**

- Bell + H amplitude assertions via numpy; a memory smoke at n≥20 (RSS delta sanity, not a hard gate).

**References**

- <https://github.com/PyO3/rust-numpy>

-----

### [P4-12] Unify backend vocabulary across CLI and Python (+ auto-select in Python)

**Labels:** `area:python`, `area:cli`, `type:refactor`, `priority:medium`
**Milestone:** Phase 4
**Estimate:** M
**Depends on:** P4-08

**Description**
One backend-name vocabulary for both user surfaces, parsed in one place, and expose the CLI's `auto` backend selection to Python.

**Context**
P4-08 shipped a fork: the CLI accepts `--backend statevector|stabilizer|mps|auto` (default `auto`), the Python API accepts `backend="sv"|"mps"|"stab"` (default `"sv"`, no auto). `aleph_backend::BackendKind` + `select_explained()` (P3-07) already exist as the shared seam; the Python binding string-matches its own names instead. Users translating README CLI examples to Python hit `ValueError: unknown backend "statevector"`, and Python users never get Clifford→stabilizer auto-routing. Adding a 4th backend (FP32 exists; GPU in Phase 5) currently means editing two match sites with no compiler aid.

**Technical Details**

- Single parse function in `aleph-backend` (e.g. `BackendKind::from_user_str`) accepting canonical names AND the established aliases (`sv`/`statevector`, `stab`/`stabilizer`), used by both `aleph-cli` (clap `value_parser`) and `aleph-py`.
- Python `run(..., backend="auto")` routes through `select_explained()` like the CLI; consider making `auto` the Python default in v0.2 (breaking-change note in the changelog — v0.1 defaulted to `"sv"`).
- Error message lists the canonical names + aliases, same text on both surfaces.
- Document the alias table once (README Backends section).

**Acceptance Criteria**

- [ ] Every name the CLI accepts works in Python and vice versa
- [ ] `backend="auto"` works in Python (Clifford circuit → stabilizer, etc., with `select_explained` reasoning)
- [ ] One parse site; adding a backend variant is a compile-error-guided change
- [ ] CLI behavior unchanged (existing names keep working)

**Testing Requirements**

- Cross-surface parity test (table-driven: name → resolved BackendKind, both surfaces); Python auto-select test on a Clifford circuit; CLI integration tests stay green.

**References**

- `crates/aleph-backend/src/select.rs` (P3-07 `select_explained`), `crates/aleph-cli/src/cli.rs` `BackendChoice`.

-----

# Phase 4.5 — CPU Parity

Close (or honestly explain) wall-clock gaps vs mainstream simulators on CPU
before GPU work starts. Exit: every competitive-matrix cell ≤ 1.2× its
reference, or a documented structural exception with profiling evidence.
Design: `docs/superpowers/specs/2026-06-11-phase-4.5-cpu-parity-design.md`.

**Adopted tickets:** P3-12 (#148), P3-13 (#149), P3-14 (#150) are part of this
phase (the MPS levers). They keep their IDs and issue numbers; only their
milestone moves to "Phase 4.5 — CPU Parity". Order: P4.5-01 → (P4.5-02 ∥
P3-12..14) → P4.5-06 → P4.5-07.

### [P4.5-01] Competitive benchmark matrix vs Aer (MT statevector + MPS)

**Labels:** `area:bench`, `type:infra`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** M
**Depends on:** —

**Description** — Measure where aleph actually stands against the references
multi-threaded, before any tuning. Two new rows: (1) aleph SV 16 threads vs
Aer statevector 16 OMP threads, default settings both sides, Tier-1 fixtures
@ n=25; (2) aleph-mps (sequential default) vs Aer `matrix_product_state` on
three MPS workloads consumed byte-identically from the same QASM fixtures.
The stabilizer row is imported from `docs/perf/surface_code.md` (1.64× @ d=11)
without re-measurement.

**Context** — Phase 1 proved aleph ahead of Aer single-thread; Phase 2 only
measured self-scaling. The MT and MPS cells have never been measured, and the
phase's tuning scope (P4.5-06) is defined by this matrix, not guessed.

**Technical Details** — Extend `scripts/qiskit-baseline/run.py` with
`--threads N`, `--from-qasm`, and `--out` (existing fixtures are the source of
truth, including the legacy `grover_n25_iters5.qasm`). aleph MT side =
existing `tier1_scaling_fused` criterion group, `RAYON_NUM_THREADS=16`. New
`scripts/mps-baseline/run.py` builds brickwork-n128-d6, long-range-n12, and
wide-bond-n26 circuits, exports QASM3 fixtures, times Aer MPS with matched
bond caps; new `crates/aleph-mps/benches/parity.rs` times aleph on the same
fixtures. χ chosen so brickwork (χ=64 ≫ max bond 8) and long-range
(χ=64 = exact at n=12) truncate on neither side — equal fidelity by
construction; wide-bond reports both sides' truncation metrics with a caveat.
All measurements on the idle-verified EPYC box.

**Acceptance Criteria**
- [x] `docs/perf/parity.md` exists with the full matrix, per-cell ratio, and a ≤ 1.2× verdict per cell.
- [x] Both sides of every cell consumed byte-identical circuits (QASM fixtures), same box, same session; versions and configs pinned in the report.
- [x] Gap list section explicitly scopes P4.5-06 (or states "no MT gaps").
- [x] Iteration-capped grover reported as such; Aer default fusion disclosed.

**Testing Requirements** — harness smoke runs at small n locally;
`cargo bench -p aleph-mps --bench parity -- --test` passes in CI; fixture
QASM files parse via aleph-parser (bench panics on parse failure).

**References** — spec § 3; `docs/perf/phase1.md`, `docs/perf/phase2.md`.

### [P4.5-02] Stabilizer: word-parallel transpose + zero_row/copy_row

**Labels:** `area:backend-stab`, `type:optimization`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** M
**Depends on:** —

**Description** — Attack the two levers deferred from P3-11: the
orientation-transpose (~30% of the surface-d11 cycle) and `zero_row`/
`copy_row` (~33%), both still scalar in the dual-orientation tableau.

**Context** — The stabilizer cell is the one *known* parity gap: 1.64× Stim
@ d=11 (`docs/perf/surface_code.md`). These two hot spots are the identified
remainder after the P3-11 word-parallel gate work.

**Technical Details** — Word-parallel (u64 / AVX-512) implementations of the
transpose between X/Z bit-plane orientations and of row clear/copy in the
tableau, mirroring the P3-11 approach (ADR 0013). Bit-exact vs scalar;
Stim oracles d=3..11 unchanged.

**Acceptance Criteria**
- [x] surface-d11 cycle time improves; target ≤ 1.2× Stim, else documented structural verdict with profile evidence per spec § 5.
- [x] Bit-exact scalar↔SIMD equivalence tests; Stim oracle d=3..11 green.
- [x] Before/after criterion numbers (EPYC) in the PR.

**Testing Requirements** — existing stim_oracle suites; new unit tests for
transpose/zero_row/copy_row word-parallel paths on irregular n (not multiples
of 64).

**References** — `docs/perf/surface_code.md` P3-11 addendum; ADR 0013.

### [P4.5-06] Close the MT gaps surfaced by the parity matrix

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** M
**Depends on:** P4.5-01

**Description** — Deliberate placeholder: scope is the gap list from
`docs/perf/parity.md` (P4.5-01), not guessed in advance. Re-spec this entry
once the matrix lands; if the matrix shows no cell > 1.2×, close as no-op
with a comment linking the report.

**Context** — Spec § 4. The escalation ladder (profile → algorithm → layout →
SIMD → threads) and the one-PR-cycle-per-lever timebox from spec § 5 apply.

**Acceptance Criteria**
- [x] Every SV-MT/MPS cell > 1.2× in parity.md either brought ≤ 1.2× or closed with a documented structural verdict.

**Testing Requirements** — standard (unit + property + oracle + before/after
criterion numbers per change).

**References** — spec § 4–5; `docs/perf/parity.md` (deliverable of P4.5-01).

### [P4.5-07] Final parity report, verdicts, and v0.2 gate

**Labels:** `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** S
**Depends on:** P4.5-02, P4.5-06

**Description** — Re-measure changed cells, finalize `docs/perf/parity.md`
with a verdict per cell (≤ 1.2× or structural exception + deferred ticket),
update ROADMAP § 7 (phase met/not-met) and CLAUDE.md project status, then tag
v0.2 and execute PyPI publication (P4-09, #142).

**Acceptance Criteria**
- [x] parity.md final: every cell has a verdict; exceptions carry profiling evidence and a deferred ticket.
- [x] ROADMAP § 7 + CLAUDE.md updated; v0.2 tagged; P4-09 unblocked/executed.

**Testing Requirements** — measurement protocol only (idle-verified EPYC);
no code changes expected.

**References** — spec § 2; P4-09 (#142).

-----

# Phase 4.6 — CPU Depth

Goal: spend the pre-GPU window on (a) **QEC throughput** — the two
profile-driven stabilizer levers left visible after P4.5-02 — and (b) the
largest product gap vs Aer: **noise models**. Phase 5 (GPU) stays blocked on
permanent CUDA hardware (owner decision 2026-06-12: no GPU box yet, and the
measure-first culture does not survive rented-by-the-hour dev loops).

**Adopted tickets:** P3-12 (#148), P3-13 (#149), P3-14 (#150) — the MPS
levers — and P4-10 (#143), P4-11 (#144), P4-12 (#145) — Python/CLI polish —
keep their IDs and issue numbers; only their milestone moves to
"Phase 4.6 — CPU Depth". Order: **A** (adopted: P3-13 → P3-14 → P3-12, with
P4-10..12 interleaved as breathers) → **B** (P4.6-01 → P4.6-02) → **C**
(P4.6-03 → P4.6-04 → P4.6-05).

### [P4.6-01] Stabilizer: word-parallel `measure` column scans

**Labels:** `area:backend-stab`, `type:optimization`, `priority:high`
**Milestone:** Phase 4.6
**Estimate:** M
**Depends on:** —

**Description** — After P4.5-02 the surface-d11 cycle profile is
measure-dominated: `Tableau::measure` is 57.8% self time, and the cost is its
per-bit strided column reads under RowMajor (`x.get(row, a)`): the
random-branch `find` over stabilizer rows, the elimination loop over all `2n`
rows, and the deterministic-branch destabilizer scan.

**Context** — The same per-bit-vs-word disease P4.5-02 cured in
`zero_row`/`copy_row`, except here the access is a *column* under RowMajor, so
no contiguous read exists (ADR 0013's layout tension, now on the measurement
side).

**Technical Details** — Profile first (`perf annotate` on `measure`), then
pick the cheapest lever. Candidate options, in rising complexity: (1) hoist
the column word-offset/mask out of the row loops and walk `words[row*stride +
(a>>6)]` directly (the P3-01 gate-hoisting trick — verify what rustc already
emits before assuming a win); (2) AVX-512 strided gather of 8 rows' column
words per step + mask test; (3) an incrementally maintained per-qubit
"x-column" bit-vector (invalidation on every rowsum/copy/zero — likely too
much bookkeeping; reject unless (1)/(2) stall). Mirror the P4.5-02 testing
pattern for any new word-parallel helper.

**Acceptance Criteria**
- [x] surface-d11 cycle time improves and `measure`'s self share drops materially (out of the top profile slot; soft goal: d=11 ≤ 0.65× Stim) — else a documented structural verdict with profile evidence. — *Structural verdict: both candidate levers regress on EPYC (extraction +39.7%, index strength-reduction +1.9% at d=11); the strided loads already overlap rowsum work and the `imul` enabled load pipelining. `docs/perf/surface_code.md` P4.6-01 addendum.*
- [x] Stim oracles d=3..11 green; new unit tests for any new helper on irregular n (not multiples of 64). — *No helper ships (both prototypes reverted); both passed the unit + Stim suites before revert.*
- [x] Before/after criterion numbers (EPYC) in the PR. — *Regression table in the perf addendum + PR body.*

**Testing Requirements** — existing stim_oracle suites; equivalence +
mutation-test pattern from P4.5-02 for new helpers.

**References** — `docs/perf/surface_code.md` P4.5-02 addendum (profile);
ADR 0013.

### [P4.6-02] Stabilizer: batched-shot sampling (Pauli-frame simulator)

**Labels:** `area:backend-stab`, `type:feature`, `priority:high`
**Milestone:** Phase 4.6
**Estimate:** L
**Depends on:** —

**Description** — Multi-shot sampling currently re-runs the full CHP
simulation once per shot. Implement frame-based sampling à la Stim: one CHP
reference run records the measurement structure; M shots are then M Pauli
frames propagated word-parallel (bit-packed across shots — 64 shots per u64
word, 512 per zmm), each flipping recorded reference outcomes where its frame
anticommutes with the measurement.

**Context** — For QEC users multi-shot throughput is the headline number, and
Stim's frame simulator beats per-shot tableau simulation by orders of
magnitude. No amount of single-shot kernel work can reach this; it is the
highest-leverage CPU feature left in the stabilizer backend.

**Technical Details** — New sampler in `aleph-stab` (e.g. `FrameSampler`):
reference run via the existing `Tableau`, capturing per-measurement reference
outcomes + which measurements were random; frame state = X/Z bit-planes over
qubits × shot-words; Clifford gates act on frames by conjugation (H swaps
x/z frame bits, S/CNOT per the same tables as the tableau kernels — all
word-parallel over shots); a measurement flips the reference outcome wherever
the frame has an X component on the measured qubit, and randomizes the frame's
Z there (Z-basis collapse). Pauli-noise hooks come nearly for free in frame
simulators — leave the seam visible for P4.6-04 but do NOT build noise here.
Wire into the backend sampling path only for Clifford+measure circuits above a
shot threshold (selection via the existing dispatch). Read, don't copy:
Gidney, "Stim: a fast stabilizer circuit simulator", Quantum 5, 497 (2021),
§ 4; quantumlib/Stim `FrameSimulator`.

**Acceptance Criteria**
- [x] Sampling M=1024 shots of the surface-d11 cycle is ≥ 10× faster than 1024 sequential single-shot runs (EPYC, criterion numbers in the PR). — *`benches/benches/frame_sampler.rs`: **EPYC (idle, AVX-512) 98.74 ms → 1.52 ms = 65×**; local aarch64 scalar 65.70 ms → 1.07 ms = 61× (≫10×; win is the 64×-per-batch x/z amortization, structural/platform-independent).*
- [x] Distribution oracle: frame-sampled counts match per-shot CHP sampling within the 1e-5 / 100k-shot tolerance (`docs/testing.md`), plus a Stim cross-check on the surface-code fixtures. — *`frame_sampler.rs` checks batched vs the EXACT uniform-on-support distribution (stronger than per-shot) with the 5σ band + a random-Clifford proptest; frame ≡ per-shot/tableau, which the existing `surface_code_stim_oracle`/`stim_measure_oracle` validate vs Stim → frame ≡ Stim transitively.*
- [x] Deterministic seeding: same seed → same shot table; non-Clifford circuits are cleanly rejected (fall back to per-shot path). — *`batched_deterministic_same_seed`; non-Clifford is rejected at `apply_gate`, so `sample` only ever sees a Clifford state.*

**Testing Requirements** — distribution-closeness helper (soft dependency on
P3-16 — inline a local copy if P3-16 hasn't landed); proptest invariants
(frame propagation matches gate conjugation on random Clifford circuits).

**References** — Gidney 2021 § 4 (frame simulation); `docs/testing.md`
distribution tolerances; P3-16 (#152).

### [P4.6-03] Noise models: design spec + ADR

**Labels:** `area:core`, `type:docs`, `priority:high`
**Milestone:** Phase 4.6
**Estimate:** M
**Depends on:** —

**Description** — Decide and write down how aleph represents noise *before*
any implementation. The spec must answer: (1) **simulation strategy** —
stochastic Kraus trajectories on the SV backend (sample one Kraus branch per
channel application; O(2^n) memory, cost scales with shots) vs a
density-matrix backend (exact, O(4^n), ~14-qubit ceiling) vs both eventually
(Aer ships both; expected recommendation: trajectories first, DM later only
if demanded); (2) **noise IR** — channel set v1 (depolarizing 1q/2q,
amplitude damping, phase damping, bit/phase-flip, measurement/readout error)
and the attachment model: an Aer-style `NoiseModel` object mapping gate
kinds/qubit sets → channels, applied at execution time — NOT per-gate IR
pollution (golden rule 4: the IR stays backend-agnostic and noise-free);
(3) **API surface** — Python `aleph.run(c, noise=...)` + CLI; (4) **oracle
strategy** — vs Aer under a byte-identical NoiseModel at 100k shots / 1e-5;
(5) **frame-sampler integration** — what Pauli channels look like in
P4.6-02's sampler (frame simulators absorb Pauli noise nearly for free).

**Acceptance Criteria**
- [ ] Spec in `docs/superpowers/specs/` answers (1)–(5) with explicit trade-offs; ADR accepted under `docs/decisions/`.
- [ ] P4.6-04 and P4.6-05 re-specced in BACKLOG from the spec (amend + re-sync issues).

**Testing Requirements** — none (design doc); the oracle protocol it defines
becomes P4.6-04's testing section.

**References** — Aer noise model docs/implementation (read, don't copy);
Nielsen & Chuang § 8 (quantum operations); `docs/testing.md`.

### [P4.6-04] Noise models: implementation per spec

**Labels:** `area:backend-sv`, `type:feature`, `priority:high`
**Milestone:** Phase 4.6
**Estimate:** L
**Depends on:** P4.6-03

**Description** — Build `aleph_sv::noise` per the P4.6-03 spec: `NoiseModel` /
`QuantumError` / `ReadoutError` config types + v1 channel constructors;
`apply_channel` (general quantum-jump — pᵢ=‖Kᵢ|ψ〉‖², sample, apply, renormalize
— with a Pauli fast-path that skips the norm for depolarizing/flip channels);
`run_noisy(circuit, &NoiseModel, shots, seed) -> Counts` (rayon over shots,
per-shot RNG = hash(seed, shot)); per-qubit readout error at terminal sampling.
The driver works directly on `CpuState`; the IR, `Backend` trait, and noiseless
`run()` are untouched (noise is a separate entry point). v1 is SV-only; terminal
measurement only (mid-circuit measure/reset under noise = v1.1).

**Acceptance Criteria**
- [ ] Channel set v1 (depolarizing 1q/2q, amplitude/phase damping, bit/phase-flip, readout error) works end-to-end via `run_noisy` on the SV backend.
- [ ] Oracle vs Aer under a byte-identical NoiseModel: 1e-5 at 100k shots on the spec's fixture set (depol on H/CX, amp+phase damping after H, asymmetric readout, depol+readout GHZ-3), compared via `aleph_oracle::assert_distribution_close`.
- [ ] CPTP property tests: each channel's Σpᵢ=1 (1e-12) and ‖state‖=1 post-`apply_channel`; deterministic seeding (same seed → identical counts); empty NoiseModel reproduces the noiseless distribution; noiseless `run()` criterion benchmark unchanged.

**Testing Requirements** — per the P4.6-03 spec § "Oracle protocol"; distribution
oracles vs Aer + the CPTP/trace/determinism property tests above.

**References** — `docs/superpowers/specs/2026-06-13-p46-03-noise-models-design.md`;
ADR `docs/decisions/0014-noise-trajectories.md`.

### [P4.6-05] Noise models: Python/CLI surface + docs

**Labels:** `area:python`, `type:feature`, `priority:medium`
**Milestone:** Phase 4.6
**Estimate:** M
**Depends on:** P4.6-04

**Description** — Aer-compatible noise API per the P4.6-03 spec § 4. Python
(pyo3): `aleph.NoiseModel()` with `add_quantum_error(err, gates, qubits)`,
`add_all_qubit_quantum_error(err, gates)`, `add_readout_error(probs, qubits)`;
error factories mirroring Aer names — `depolarizing_error(p, num_qubits)`,
`amplitude_damping_error(gamma)`, `phase_damping_error(lam)`,
`pauli_error([...])`; `aleph.run(circuit, shots, noise=nm, seed=...)` dispatching
to `run_noisy` (no `noise=` → existing noiseless path). CLI: `--noise
<preset>:<p>` for the single-parameter presets (depolarizing, readout); full
`NoiseModel` construction stays in Python. README/crate-README examples +
release-notes entry.

**Acceptance Criteria**
- [x] Python `NoiseModel` + error factories (Aer names) + `aleph.run(..., noise=)` per spec, with tests in `scripts/python/test_aleph.py`; CLI exposure for at least a depolarizing preset.
- [x] README + crate-README examples; docs updated; release-notes entry.

**Testing Requirements** — python behaviour tests against a locally built
wheel; CLI assert_cmd tests.

**References** — P4.6-03 spec; P4-12 (#145) backend-vocabulary work (keep the
two API surfaces consistent).

-----

# Phase 5 — GPU Backend

Goal: GPU state vector backend within 1.5× of cuQuantum standalone.

-----

### [P5-01] CUDA toolchain setup and `cudarc` integration

**Labels:** `area:backend-gpu`, `area:infra`, `type:infra`, `priority:critical`
**Milestone:** Phase 5
**Estimate:** M
**Depends on:** P4-08

**Description**
Add CUDA bindings via `cudarc` and ensure CI can build GPU code.

**Context**
First step toward GPU. Get the toolchain working before any kernel work.

**Technical Details**

- Choose `cudarc` (safe Rust) or `cust` (lower-level).
- Test: allocate GPU memory, copy data, free.
- CI: GPU runner (self-hosted or AWS).

**Acceptance Criteria**

- [ ] GPU memory allocation works
- [ ] CI builds GPU code
- [ ] Simple “copy 1M floats to GPU and back” test

**Testing Requirements**

- Round-trip data test.

**References**

- <https://github.com/coreylowman/cudarc>

-----

### [P5-02] GPU state vector backend — basic

**Labels:** `area:backend-gpu`, `area:backend-sv`, `type:feature`, `priority:critical`
**Milestone:** Phase 5
**Estimate:** XL
**Depends on:** P5-01

**Description**
Implement state vector backend on GPU with hand-written CUDA kernels for 1q and 2q gates.

**Context**
Even before cuQuantum integration, we want our own GPU baseline.

**Technical Details**

- SoA state vector on GPU (two `f64` arrays).
- CUDA kernels for 1q and 2q gates, mirroring CPU patterns.
- Thread per amplitude pair.
- Tune block size (typically 256 or 512).

**Acceptance Criteria**

- [ ] Runs Tier 1 algorithms on GPU
- [ ] Correct vs. CPU backend
- [ ] At least 10× faster than CPU multi-thread on a consumer GPU (RTX 4090) at n = 28

**Testing Requirements**

- Equivalence vs. CPU on all Tier 1.

**References**

- <https://developer.nvidia.com/cuda-toolkit>
- QuEST GPU implementation as reference.

-----

### [P5-03] cuQuantum integration as a backend

**Labels:** `area:backend-gpu`, `type:feature`, `priority:critical`
**Milestone:** Phase 5
**Estimate:** L
**Depends on:** P5-01

**Description**
Integrate NVIDIA cuQuantum (cuStateVec) as an optional backend.

**Context**
cuQuantum is the gold standard for GPU state vector. We benchmark our own kernels against it; for users wanting maximum performance on NVIDIA hardware, we route through cuQuantum.

**Technical Details**

- Bind cuStateVec via FFI.
- Wrap as a `Backend` impl.
- Build flag: `--features cuquantum`.

**Acceptance Criteria**

- [ ] cuQuantum backend works
- [ ] Within expected performance of standalone cuQuantum
- [ ] Optional dependency

**Testing Requirements**

- Equivalence vs. our GPU backend.

**References**

- <https://docs.nvidia.com/cuda/cuquantum/>

-----

### [P5-04] GPU memory management strategy

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5
**Estimate:** M
**Depends on:** P5-02

**Description**
Design memory allocation strategy: pool, pinned host memory, streams.

**Context**
Naive allocation per gate hurts performance. Need pooled allocator and async transfers.

**Technical Details**

- GPU memory pool (cudaMallocAsync or custom).
- Pinned host memory for staging.
- CUDA streams for overlap of transfer and compute.

**Acceptance Criteria**

- [ ] Allocator pool implemented
- [ ] Benchmark: allocation overhead negligible

**Testing Requirements**

- Stress test: many small circuits, no memory leaks.

**References**

- <https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/>

-----

### [P5-05] CPU↔GPU transfer optimization

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:medium`
**Milestone:** Phase 5
**Estimate:** M
**Depends on:** P5-04

**Description**
Minimize transfers; keep state vector on GPU across gates.

**Context**
PCIe is slow vs. HBM. Transfers should happen only at start (input state) and end (results).

**Technical Details**

- Lazy transfer: data stays on GPU until explicitly read.
- Result computation (expectation, sampling) done on GPU when possible.

**Acceptance Criteria**

- [ ] Only initial state and final results crossed PCIe
- [ ] Verified by profiling

**Testing Requirements**

- nvprof / Nsight Systems traces confirm no spurious transfers.

**References**

- NVIDIA Nsight Systems documentation.

-----

### [P5-06] Custom CUDA kernels for niches cuQuantum misses

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:medium`, `research`
**Milestone:** Phase 5
**Estimate:** XL
**Depends on:** P5-03

**Description**
Identify and implement custom kernels where cuQuantum is suboptimal or absent.

**Context**
cuQuantum is great but not perfect; e.g., very small qubit counts, batched circuits, specific gate sequences may benefit from custom kernels.

**Technical Details**

- Profile cuQuantum on a range of workloads.
- Identify regimes where simpler kernels beat it.
- Implement and benchmark.

**Acceptance Criteria**

- [ ] At least one regime documented where custom > cuQuantum
- [ ] Custom kernels integrated with backend selection

**Testing Requirements**

- Benchmark suite covers identified regimes.

**References**

- N/A — research-level.

-----

### [P5-07] GPU stabilizer backend (research)

**Labels:** `area:backend-gpu`, `area:backend-stab`, `type:feature`, `priority:low`, `research`
**Milestone:** Phase 5
**Estimate:** XL
**Depends on:** P3-03, P5-01

**Description**
Implement stabilizer simulator on GPU.

**Context**
Stim is CPU; cuQuantum doesn’t do stabilizers. A GPU stabilizer simulator could enable massive QEC simulations.

**Technical Details**

- Tableau on GPU as bit-matrix.
- Gate updates as bit operations across rows.
- Likely bottlenecked by warp-level synchronization patterns.

**Acceptance Criteria**

- [ ] GPU stabilizer for n ≥ 1000, depth ≥ 1000
- [ ] Faster than Stim on large inputs

**Testing Requirements**

- Equivalence vs. CPU stabilizer / Stim.

**References**

- Aaronson, Gottesman (2004).
- May require novel research.

-----

### [P5-08] GPU benchmark report

**Labels:** `area:bench`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 5
**Estimate:** M
**Depends on:** P5-02, P5-03

**Description**
Comprehensive GPU benchmark report.

**Context**
Phase exit criterion.

**Technical Details**

- Tier 1 + 2 algorithms on GPU.
- Compare: our GPU, cuQuantum, Qiskit Aer GPU.
- Target hardware: RTX 4090 (consumer), A100, H100 (data center).

**Acceptance Criteria**

- [ ] Report published
- [ ] Targets met

**Testing Requirements**

- CI benchmarks on GPU runner.

**References**

- Phase 1, 2 reports for template.

-----

# Phase 5.9 — CUDA state-vector throughput → Aer-GPU parity

Phase 5 met its exit gate (aleph FP64 GPU SV within 1.22× of cuStateVec on every
Tier-1 + Tier-2 workload at n=28). But the P5-08 report found **Qiskit Aer-GPU is
2–5× faster than both aleph and cuStateVec** — and *not* via a better kernel or
lower precision (Aer is FP64, its own engine, `cuStateVec_enable=False`). The gap
is entirely **above the kernel**, two levers:

1. **Gate fusion** — Aer fuses runs of gates into larger unitaries before
   simulating, cutting full-state passes (the dominant cost, since SV is
   bandwidth-bound). aleph *already has* the IR passes (`Fuse2q`, the P2-08
   diagonal-run fusion) but the GPU backend never sees them. Disabling Aer's
   fusion moves it back toward the per-gate engines (Grover 3.89→5.33 s, QAOA
   6.54→15.03 s), quantifying this lever.
2. **Batched dispatch** — even fusion-off Aer beats our cuStateVec column (QFT
   6.4 vs 10.6 s): it streams many gates per launch, where our path is one
   kernel launch (and, for cuStateVec, a workspace query) per gate.

Goal: close the gap to Aer-GPU on the Tier-1 + Tier-2 SV suite — bandwidth-pass
reduction the project already knows how to do on the CPU side, applied to the GPU
backend. Box: `openwebgui.splynx.com` (RTX 4000 SFF Ada; aer-gpu venv at
`/root/aervenv`). Exit metric in ROADMAP § 7.

-----

### [P5.9-01] Feed the IR gate-fusion passes into the GPU run path

**Labels:** `area:backend-gpu`, `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 5
**Estimate:** M
**Depends on:** P5-02, P5-06, P1-10 (Fuse2q), P2-08 (diagonal-run fusion)

**Description**
Run the existing IR fusion passes (`Fuse2q` + P2-08 diagonal-run fusion) on the
circuit before the GPU backends apply it, so adjacent 2q gates collapse to one
4×4 (`apply_kq`, k=2 — already supported) and diagonal runs collapse to one
phase multiply (`apply_diag`, P5-06). The cheapest, highest-ROI lever: no new
kernels, reuses passes that already gate-test against the CPU oracle.

**Technical Details**
- Apply the passes in `CudaSvBackend`/`CuStateVecBackend` ingestion (or a shared
  GPU run helper), guarded by a flag so the unfused path stays benchable.
- QFT/QPE benefit from diagonal-run fusion; brickwall/VQE/QAOA from `Fuse2q`.

**Acceptance Criteria**
- [ ] Fused GPU run is oracle-equal to unfused (1e-10) across Tier-1 + Tier-2.
- [ ] Benchmark shows per-workload speedup vs unfused; no Tier-1 regression.

**Testing Requirements**
- Oracle vs CPU `NaiveSvBackend` on the fused circuit; A/B benchmark (fused vs not).

**References**
- `docs/perf/p5-08-gpu-report.md` (the gap analysis).

-----

### [P5.9-02] 4–5q fused dense kernel + IR blocks

**Labels:** `area:backend-gpu`, `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 5
**Estimate:** L
**Depends on:** P5.9-01

**Description**
Let fusion emit up to 5-qubit dense blocks (Aer's default `fusion_max_qubit=5`)
and apply them on GPU. The `apply_kq` kernel already guards `k ≤ 5`; what's
missing is an IR/`GateMatrix` representation for 4q/5q dense unitaries, the host
flatten/launch for k=4,5, and the fusion pass producing them.

**Acceptance Criteria**
- [ ] 4q/5q fused blocks apply on GPU, oracle-equal to unfused (1e-10).
- [ ] Benchmark improvement on dense-2q workloads (random/VQE/QAOA) vs P5.9-01.

**Testing Requirements**
- Oracle vs CPU SV; benchmark vs P5.9-01 baseline.

-----

### [P5.9-03] Batched disjoint-layer dispatch (one memory sweep per layer)

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:medium`
**Milestone:** Phase 5
**Estimate:** L
**Depends on:** P5-02

**Description**
Apply all disjoint-qubit gates of a circuit layer in a **single pass over the
state**, cutting both kernel launches and memory sweeps (N disjoint 1q gates →
1 sweep, not N). Mirrors the P5-07 `stab_layer` idea for the SV backend; schedule
the circuit into maximal disjoint layers in the IR.

**Acceptance Criteria**
- [ ] Layer dispatch oracle-equal to per-gate (1e-10).
- [ ] Benchmark: fewer sweeps/launches → measured speedup on layered circuits.

**Testing Requirements**
- Oracle vs CPU SV; benchmark vs per-gate dispatch.

-----

### [P5.9-04] Specialized 2q kernels (CNOT / diagonal-2q)

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:low`
**Milestone:** Phase 5
**Estimate:** M
**Depends on:** P5-02

**Description**
Tuned kernels for the common 2q gates (CNOT as a permutation, `CZ`/`CRz` as
diagonal-2q) to close the residual 1.13–1.22× vs cuStateVec on dense-2q-heavy
workloads, instead of the generic 4×4 `apply_kq` matvec.

**Acceptance Criteria**
- [ ] Oracle-equal (1e-10); benchmark improvement on CNOT-heavy workloads.

**Testing Requirements**
- Oracle vs CPU SV; benchmark.

-----

### [P5.9-05] Re-bench vs Aer-GPU + verdict

**Labels:** `area:bench`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 5
**Estimate:** S
**Depends on:** P5.9-01, P5.9-02, P5.9-03

**Description**
Re-run the P5-08 harness with all Phase-5.9 work enabled, update
`docs/perf/p5-08-gpu-report.md` (or a new `p5.9` report), and state whether the
Aer-GPU parity exit metric (ROADMAP § 7) is reached.

**Acceptance Criteria**
- [ ] Updated report published with the post-fusion numbers.
- [ ] Exit metric explicitly evaluated (met / not-met + verdict).

**Testing Requirements**
- Full Tier-1 + Tier-2 re-bench, aleph vs cuStateVec vs Aer-GPU.

-----

### [P5.9-06] GPU diagonal phase-polynomial kernel (close QFT/QPE vs Aer-GPU)

**Labels:** `area:backend-gpu`, `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 5
**Depends on:** P5.9-01, P5.9-05
**Estimate:** M

**Description**
The P5.9-05 verdict left **QFT (1.73×) and QPE (1.75×)** as the only Tier-1+Tier-2
cells still over the 1.5×-vs-Aer-GPU exit gate. Both are long **controlled-phase
ladders**: diagonal, but they do not collapse under `Fuse1q`/`Fuse2q`/`FuseKq`
(no dense ≤3q block forms), so the GPU still streams the full state once per
cphase gate (~350 passes for QFT-28). aleph already has `FuseDiagonalRuns`, which
collapses such a ladder into one `Instruction::DiagonalPhase` (a phase polynomial
`φ(x) = Σ_t angle_t · [∀ m ∈ conds_t: parity(m & x) odd]`), and the CPU SV backend
already applies it in one streaming pass. The GPU backend does not — it inherits
the trait default that rejects `DiagonalPhase`.

This ticket adds the GPU kernel and wires the pass in, turning QFT/QPE's hundreds
of cphase passes into **one** coalesced sweep `amps[x] *= exp(i·φ(x))`.

**Technical Details**
- New NVRTC kernel `apply_phase_poly(amps, angles, conds, offsets, n_terms, n_amps)`:
  one thread per amplitude `x`, accumulate `φ` over the CSR-encoded terms
  (`conds[offsets[t]..offsets[t+1]]`), `sincos(φ)`, complex-multiply `amps[x]`.
  Fully coalesced (each amplitude read/written once).
- Override `CudaSvBackend::apply_diagonal_phase`: flatten the `DiagonalPhase`
  terms to host `angles:[f64]` / `conds:[u64]` / `offsets:[u32]`, upload, launch
  over `2^n` threads.
- Add `FuseDiagonalRuns` to `fuse_for_gpu` (canonical order: after DCE, before
  `Fuse1q`). `run_layered` already routes `DiagonalPhase` through the trait method.

**Acceptance Criteria**
- [ ] Fused GPU run with the diagonal kernel is oracle-equal to unfused CPU SV
  (1e-10) across Tier-1 + Tier-2 (esp. QFT, QPE).
- [ ] QFT and QPE move from ~1.73–1.75× to **≤ 1.5×** of Aer-GPU at n=28 (the
  P5.9-05 exit gate), with no regression on the already-passing cells.

**Testing Requirements**
- Oracle vs CPU `NaiveSvBackend`; A/B benchmark (diagonal-fusion on vs off) and
  re-run of the P5.9-05 verdict table.

-----

# Phase 5.10 — single-GPU SV: reach & throughput

Phase 5.9 reached Aer-GPU parity on the single RTX 4000 Ada box. Phase 5.10 is the
set of **single-GPU** levers that still have a measurable exit *on the hardware we
have* — pushing throughput and qubit reach without a multi-GPU cluster. (True
distributed / multi-node SV is Phase 6, deferred until multi-GPU hardware is
available: its exit metric — scaling efficiency across nodes — is unmeasurable on
one card, so building it now would ship an unverifiable perf claim.) All three
tickets below oracle-test and benchmark entirely on `openwebgui.splynx.com`.

### [P5.10-01] Register-tiled fused-block kernel (unlock k=4,5 fusion)

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5
**Depends on:** P5.9-02
**Estimate:** L

**Description**
P5.9-02b measured that the generic `apply_kq` only profits from fusion up to
**k=3**; k=4,5 regress because the dense `2^k × 2^k` matvec per amplitude group is
O(4^k) and the `v[32]`/`gidx[32]` thread-local arrays (768 B at k=5) spill to
local memory. Aer reaches its `fusion_max_qubit=5` default with register-tiled
fused-block kernels. This ticket writes a kernel that keeps the `2^k` block in
registers / shared memory without spilling (e.g. a warp-cooperative tile where
each lane owns a slice of the block and the matvec is a shared-memory reduction),
so k=4,5 fused blocks beat the k=3 result instead of regressing.

**Acceptance Criteria**
- [ ] k=4 and k=5 fused blocks apply faster than the k=3 `apply_kq` baseline on
  dense workloads (random/VQE/QAOA), oracle-equal to unfused CPU SV (1e-10).
- [ ] `MAX_FUSE_QUBITS` raised to the new measured sweet spot; P5.9-05 verdict
  re-run shows no regression and a measurable gain on the dense-2q cells.

**Testing Requirements**
- Oracle vs CPU `NaiveSvBackend`; A/B benchmark vs the generic `apply_kq` at k=3,4,5.

-----

### [P5.10-02] Host-memory paging for n > 30 (out-of-core single-GPU SV)

**Labels:** `area:backend-gpu`, `type:feature`, `priority:medium`
**Milestone:** Phase 5
**Depends on:** P5-02, P5-05
**Estimate:** L

**Description**
`MAX_CUDA_QUBITS = 30` is the largest state that fits the 20 GiB card (16 GiB of
FP64 at n=30). Beyond that the state must live in host memory and stream through
the device. Implement an out-of-core path: hold the `2^n` state in pinned host
memory, process each gate over device-sized tiles with overlapped H2D/D2H copies
(double-buffered on the stream), so n=31–33 runs on one GPU at a documented
bandwidth-bound throughput cost. The P5-05 retaining pool and stream plumbing are
the foundation; the new work is the tiling schedule and the host-resident state.

**Acceptance Criteria**
- [ ] Out-of-core run is oracle-equal to the in-core path / CPU SV (1e-10) at
  small n with paging forced on.
- [ ] n=31 (and ideally 32) runs on the 20 GiB card; report the throughput hit
  vs in-core (transfer-overlap efficiency).

**Testing Requirements**
- Oracle vs CPU `NaiveSvBackend` with forced tiling at small n; throughput bench
  at n=30 (in-core) vs n=31 (paged).

-----

### [P5.10-03] FP32 / mixed-precision GPU state vector

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:medium`
**Milestone:** Phase 5
**Depends on:** P5-02
**Estimate:** M

**Description**
FP64 on the Ada card is compute-weak (~1/64 the FP32 rate); the GPU SV is
bandwidth-bound, and FP32 halves both the memory footprint (2 GiB at n=28) and the
bytes moved per sweep, for ~2× throughput and +1 qubit of reach. Add an FP32
(optionally mixed-precision) variant of `CudaSvBackend` — mirroring the Metal
track's FP32 ceiling — with a 1e-5 oracle tolerance instead of FP64's 1e-10.

**Acceptance Criteria**
- [ ] FP32 backend oracle-equal to the FP64 backend / Qiskit within 1e-5 across
  Tier-1 + Tier-2.
- [ ] Benchmark shows ~2× memory + throughput vs FP64; +1 qubit reach on the card.

**Testing Requirements**
- Oracle vs FP64 `CudaSvBackend` (1e-5); throughput + reach benchmark vs FP64.

-----

# Phase 5.11 — single-GPU SV: deeper reach & throughput

Phase 5.10 shipped three single-GPU levers — the register-tiled fused-block
kernel (P5.10-01), out-of-core host-memory paging to n=31 (P5.10-02), and an FP32
backend (n=31 in-core, ~2× throughput; P5.10-03). Phase 5.11 **stacks and combines
them** and attacks the two walls they exposed: the **PCIe-serialised paging cost**
(~18.5× the in-core rate, no copy/compute overlap) and the **O(4^k) dense-matvec
compute ceiling** (k=4,5 fusion still loses to k=3 even with the spill removed).
Every ticket is measurable on the one RTX 4000 SFF Ada box (`openwebgui.splynx.com`,
20 GiB card, 62 GiB host RAM); true multi-GPU scaling remains Phase 6 (deferred).

### [P5.11-01] FP32 out-of-core paging → n=32

**Labels:** `area:backend-gpu`, `type:feature`, `priority:high`
**Milestone:** Phase 5
**Depends on:** P5.10-02, P5.10-03
**Estimate:** M

**Description**
`run_paged` (P5.10-02) streams an **FP64** host state, so n=32 needs 64 GiB of
pinned host memory — past this box's 62 GiB. The FP32 buffer (P5.10-03) halves
that: FP32 paged n=32 is `2^32 · 8 B = 32 GiB` of pinned host, comfortably inside
62 GiB. Generalise the paging executor over precision (or add an FP32 paged path):
hold the host state and stream the tiles as `f32`, and route each tile group's
compute through `CudaSvBackendF32::apply_gate` instead of the FP64 backend. The
tiling scheme (high/low split, co-resident gather, qubit remap) is unchanged —
only the scalar type and the per-tile kernels differ. Sets a new single-GPU reach
record (n=32) on the same card.

**Acceptance Criteria**
- [ ] FP32 paged run oracle-equal to the FP32 in-core backend (and CPU FP64 within
  1e-5) at small `n` with paging forced on.
- [ ] **n=32 runs on the 20 GiB card** (32 GiB FP32 pinned host); report wall time
  and effective host↔device throughput.

**Testing Requirements**
- Oracle vs `CudaSvBackendF32` / CPU at small `n` forced-tiled; n=32 reach bench.

-----

### [P5.11-02] Overlapped (double-buffered) paging copies

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5
**Depends on:** P5.10-02
**Estimate:** M

**Description**
`run_paged`'s tile copies are **synchronous**: cudarc inserts a host sync whenever
a borrowed slice backs an async copy, so H2D, compute, and D2H never overlap
(P5.10-02 measured ~18.5× the in-core cost; the state buffer is pinned so each copy
is fast pinned DMA — the loss is purely the lack of overlap). Issue the tile copies
via raw-FFI `memcpy_htod_async` / `memcpy_dtoh_async` on a dedicated copy stream,
with two (ping/pong) device tile-group buffers and CUDA events, so
gather(i+1) / compute(i) / scatter(i−1) run concurrently and keep the PCIe link
saturated in both directions. This cannot beat the PCIe-vs-resident bandwidth ratio
that sets the floor, but should recover a meaningful fraction by hiding compute and
overlapping H2D with D2H.

**Acceptance Criteria**
- [ ] Oracle-equal to the synchronous paged path (1e-10) at small `n` forced-tiled.
- [ ] Measured speedup vs the synchronous paging baseline at n=30/31; report the
  transfer-overlap efficiency (achieved vs peak PCIe bandwidth).

**Testing Requirements**
- Oracle vs the synchronous path; A/B throughput bench vs P5.10-02.

-----

### [P5.11-03] Amortise PCIe over multi-gate tile residency

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5
**Depends on:** P5.10-02
**Estimate:** L

**Description**
Today every paged gate streams the whole `2^n` state once (one read + one write).
But a **run of consecutive gates acting only on low (within-tile) qubits** can be
applied to a tile *while it is resident*, before scattering — collapsing N per-gate
PCIe passes into one. Schedule the circuit into maximal low-qubit-only runs
(optionally relabelling qubits so the hot/most-acted qubits are low), load each
tile once, apply the whole run on-device, scatter once. For locality-rich circuits
(1q layers, nearest-neighbour 2q on low qubits) this cuts PCIe traffic by the run
length — the single largest paging lever — and composes with IR fusion (apply
`UnitaryKq` blocks within a residency) and with P5.11-02's overlap.

**Acceptance Criteria**
- [x] Oracle-equal to per-gate paging (1e-10) at small `n` forced-tiled.
- [x] On a locality-rich circuit at n=31, **≥2× fewer full-state PCIe passes** than
  per-gate paging; report the wall-time speedup. (**26× fewer passes, 9.93×
  wall-time** at n=31; `docs/perf/p5.11-03-multi-gate-residency.md`.)

**Testing Requirements**
- Oracle vs per-gate paging; A/B bench (per-gate vs batched-residency) at n=30/31.

-----

### [P5.11-04] First-class FP32 backend: GPU-resident readout + throughput levers

**Labels:** `area:backend-gpu`, `type:feature`, `priority:medium`
**Milestone:** Phase 5
**Depends on:** P5.10-03
**Estimate:** L

**Description**
`CudaSvBackendF32` is a per-gate apply path with host-side amplitude readout only:
it does not implement the full `Backend` trait (no GPU-resident measure / sample /
expectation / probabilities) and skips the Phase 5.9 / 5.10-01 throughput levers
(IR fusion, `apply_1q_multi` layer batching, `apply_phase_poly`, `apply_kq_tiled`).
Port the `GpuReadout` reductions and the apply-side levers to FP32 so it is a
drop-in first-class backend whose throughput wins **stack on the 2× precision win**
(e.g. FP32 + `apply_phase_poly` for QFT/QPE, FP32 + fusion for dense workloads).

**Acceptance Criteria**
- [x] FP32 backend implements the full `Backend` trait, oracle-equal to FP64 within
  1e-5 (worst readout |Δ| 7.2e-7; sampling matches the exact distribution at 400k
  shots). `GpuReadoutF32` + `readout_f32.cu`.
- [x] FP32 + fusion / phase-poly beats per-gate FP32 on QFT/QPE/dense workloads
  (QFT 2.07×, dense 1.82×); stacked **3.92× / 3.58× vs FP64 per-gate**
  (`docs/perf/p5.11-04-fp32-first-class.md`).

**Testing Requirements**
- Readout oracle (measure/sample/expectation/probabilities) vs FP64; throughput
  bench vs per-gate FP32 and vs FP64.

-----

### [P5.11-05] Tensor-core (TF32) fused-block matvec — break the k=4,5 compute wall

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:medium`
**Milestone:** Phase 5
**Depends on:** P5.10-01, P5.10-03
**Estimate:** L

**Description**
P5.10-01 established that the wall past k=3 fusion is the **O(4^k) dense matvec
compute itself**, not register spill — the warp-tiled `apply_kq_tiled` removed the
spill yet k=4,5 still lose to k=3. The Ada card's **TF32 / FP16 tensor cores**
(~8× the FP32 FMA rate) sit idle on this path. Recast the `2^k × 2^k` fused-block
matvec as a tensor-core MMA (TF32 for ~1e-5, or FP16 inputs with FP32 accumulate)
so the compute that gated wide fusion gets cheap enough to finally pay. Pairs
naturally with the FP32 amplitude buffer (P5.10-03 / P5.11-04). If it works, this
is the lever that lets `MAX_FUSE_QUBITS` rise past 3.

**Acceptance Criteria**
- [x] TF32 fused-block apply oracle-equal to the FP32 dense apply within 1e-4
  (TF32 mantissa) across dense workloads. (≤7.7e-5 across random/VQE at k=4,5.)
- [x] k=4 and/or k=5 fused blocks **beat the k=3 baseline** on dense workloads (the
  P5.10-01 metric, now via cheaper compute); raise `MAX_FUSE_QUBITS` if met. (k=4
  TF32 beats tiled-k3 1.02×/1.18× on random/VQE at n=28; k=5 stays 0.94–0.95×, so
  the win caps at 4. Exposed as FP32-only `MAX_FUSE_QUBITS_TF32=4` /
  `fuse_for_gpu_tf32` — **not** a bump of the shared `MAX_FUSE_QUBITS`, which also
  feeds the FP64 backend that has no TF32 kernel and would regress at k=4.)

**Testing Requirements**
- Oracle vs FP32 dense apply; A/B bench tiled vs tensor-core at k=3,4,5.

-----

### [P5.11-06] Backend auto-select: precision + reach policy

**Labels:** `area:backend`, `area:backend-gpu`, `type:feature`, `priority:low`
**Milestone:** Phase 5
**Depends on:** P5.10-02, P5.10-03
**Estimate:** S

**Description**
The backend selector (`aleph-backend/src/select.rs`) does not know about FP32 or
paging, so users must hand-wire those backends. Add a precision + reach policy:
route to `CudaSvBackendF32` when the caller's accuracy tolerance admits FP32 (and
it buys reach / throughput), and to the paged executor when
`n > MAX_CUDA_QUBITS` (FP64) or `n > MAX_CUDA_QUBITS_F32` (FP32). Expose the
precision / reach choice through the CLI and Python bindings so the Phase 5.10/5.11
reach is reachable without manually constructing backends.

**Acceptance Criteria**
- [x] Selector routes `n > 30` FP64 (or `n > 31` FP32) to the paged executor and
  honours an explicit FP32/precision flag; covered by unit tests. (`select_from_full`
  + `SelectEnv`/`Precision`/`Reach` in `aleph-backend`; `BackendKind::{Cuda,CudaF32}`;
  9 new unit tests; in-core caps pinned to `aleph-cuda` by `cuda_caps_match`.)
- [x] CLI / Python expose the precision + reach choice; documented. (CLI: `--backend
  cuda`/`cuda-f32`, `--precision auto|f64|f32` (default `auto` ⇒ FP32 on a GPU pick),
  out-of-core paging past the cap; Python: `backend="cuda"/"cuda-f32"`, `precision=`
  kwarg. Both behind a Linux `cuda` feature, mirroring the `metal` gating. e2e n=31
  in-core (auto + explicit) and n=32 paged verified on the RTX 4000 Ada.)

**Testing Requirements**
- Selector unit tests; an end-to-end CLI run at n=31.

-----

# Phase 5.7 — GPU-resident MPS (Apple/Metal)

Goal: take the MPS-on-Metal scaffold from "correct but CPU-SVD-bottlenecked" to a
genuinely GPU-resident, usable MPS backend.

> **Backlog note.** The Apple/Metal GPU track (sub-phases 5.5 SV foundation, 5.6
> hardening, 5.7 this) was built ahead of its backlog entries — the formal Phase 5
> above is the CUDA/cuQuantum plan; Metal is a separate single-box track on the M4
> Mac Mini (documented in `docs/perf/metal.md`). This section starts
> closing that gap: it records the shipped P5.7 work and queues the open Metal
> follow-ups. 5.5/5.6 remain documented only in `docs/perf/metal.md` /
> `phase5.5.md`; retro-filing them is optional cleanup, not tracked here.
>
> All entries below use the **Phase 5.5 — Apple/Metal GPU** milestone (the GitHub
> milestone for the whole Metal track). FP32 is the Metal accuracy ceiling (~1e-5).

-----

### [P5.7-01] Complex one-sided Jacobi SVD — CPU reference + faer fallback — **SHIPPED (PR #216)**

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:feature`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** M

One-sided Jacobi thin-SVD in Rust (`mps/jacobi.rs`) — the CPU reference for the
Metal kernel, wired as the faer convergence fallback. σ = ‖column‖ avoids squaring
the condition number (the FP32/GPU-friendly route; cf. cuSOLVER `gesvdj`). Validated
against faer to 1e-10. See `docs/perf/phase5.7.md`.

-----

### [P5.7-02] GPU-resident Jacobi SVD — Metal kernel + dispatch — **SHIPPED (PR #220)**

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:feature`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** L

The same algorithm as a Metal kernel (`mps_jacobi.metal`): one threadgroup factors
one two-site block, threads stride the rows, threadgroup memory holds only the 2×2
reduction + rotation scalars. `gpu_jacobi.rs` dispatch handles wide blocks via `Aᴴ`.
Validated standalone against the CPU reference / faer (reconstruction 1e-4, σ 1e-4).

-----

### [P5.7-03] Wire GPU-resident Jacobi SVD into MetalMpsBackend — **SHIPPED (PR #218)**

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:optimization`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** M

Per-gate split now runs on the GPU (faer = CPU fallback); whole per-gate path is
GPU-resident. Holds the 1e-5 oracle end-to-end. Split phase ~1390 ms → ~300 ms
(~4.6×) on the n=12 d=24 brickwall. Report: `docs/perf/phase5.7.md`.

-----

### [P5.7-04] Batched layer-parallel SVD dispatch — **SHIPPED (PR #231)**

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** M
**Depends on:** P5.7-03

> **Shipped.** `jacobi_svd_batched` (one threadgroup per block, keyed off
> `threadgroup_position_in_grid`) + a backend-side scheduler `run_batched` that
> groups disjoint NN 2q gates into layers and factors each layer in a single
> dispatch, one `commit`/`wait` per layer. On n=12 d=24 the split phase drops
> ~1.55× and contract+apply ~6× vs P5.7-03; criterion total ~1.30× (small bond),
> ~1.05× (bond-saturating). Oracle + proptest cover both `run` and `run_batched`.
> See `docs/perf/phase5.7.md` "Batched layer-parallel SVD (P5.7-04)".

**Description**
Factor all independent two-site SVDs of a brickwall layer in a single kernel launch
(one threadgroup per block) and drop the per-gate `wait_until_completed`.

**Context**
P5.7-03 made the SVD GPU-resident (~4.6× the faer split) but it still runs one
threadgroup per gate with a full GPU sync per gate, so per-dispatch latency is now
~85% of per-gate time (`docs/perf/phase5.7.md`). Gates within a brickwall layer act
on disjoint site pairs, so their splits are independent and batchable — the named
next lever.

**Technical Details**

- Group disjoint NN 2q gates into layers (the generic gate-by-gate runner doesn't
  expose this; either a layering pass or a backend-side scheduler).
- Pack each block's `(Θ, dims)` with per-block buffer offsets; dispatch grid =
  `num_blocks` threadgroups; the kernel keys off `threadgroup_position_in_grid`.
- Pool the `A`/`V`/`sig` buffers to kill per-gate allocation churn.
- One `commit`/`wait` per layer instead of per gate.

**Acceptance Criteria**

- [x] A layer's blocks factor in one dispatch; oracle still 1e-5.
- [x] Criterion shows the per-gate split share drop materially vs P5.7-03 on
  n=12 d=24 and on a larger-bond case.

**Testing Requirements**
Oracle/proptest unchanged; unit test that a disjoint-gate layer matches sequential
application; bench in the MPS report.

**References**
`docs/perf/phase5.7.md` "what's next"; cuSOLVER `gesvdjBatched`.

-----

### [P5.7-05] MPS-on-Metal readout: measure / sample / probabilities / expectation — **SHIPPED (PR #232)**

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:feature`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** L
**Depends on:** P5.7-03

> **Shipped (PR #232).** `mps/readout.rs`: `measure`/`sample`/`probabilities`/
> `expectation_value` via doubled (bra⊗ket) transfer-matrix sweeps — exact for the
> non-canonical scaffold (no dependence on P5.7-07 canonical form), bond×bond
> environments only, **no `2^n`**. Perfect (Ferris–Vidal) sampling uses
> precomputed right environments; `measure` projects + rescales the site leg;
> `with_seed` now drives a real RNG. Oracle (`mps_readout.rs`) matches CPU MPS +
> exact FP64 SV within 1e-5, sampling distribution-close at 20k shots, and a
> GHZ n=26 case checks readout with no dense vector. Contractions run host-side
> (zero-copy from unified memory); GPU offload of the transfer step is a noted
> follow-up.

**Description**
Implement the `Backend` readout methods that currently return
`UnsupportedInstruction` on `MetalMpsState` — `measure`, `sample`, `probabilities`,
`expectation_value` — without forming the dense `2^n` vector.

**Context**
The scaffold can evolve NN circuits but can only read out via the dense (test-only,
`2^n`) contraction. Real use needs sampling and expectations at MPS scale.

**Technical Details**

- MPS sampling via conditional probabilities from site contractions (left/right
  environment sweep); expectation via `⟨ψ|P|ψ⟩` along the chain; run the hot
  contractions on the GPU.
- Seeded RNG for reproducibility (`with_seed` already exists).

**Acceptance Criteria**

- [x] measure/sample/probabilities/expectation match CPU MPS / Aer within tolerance
  (1e-5 amplitudes; 1e-5 probabilities @100k shots).
- [x] No `2^n` allocation in the readout path.

**Testing Requirements**
Oracle vs CPU MPS + Aer; sampling distribution-closeness (reuse P3-16 helper);
property: probabilities sum to 1.

**References**
Schollwöck MPS review (sampling); the CPU MPS readout in `aleph-mps`.

-----

### [P5.7-06] SWAP router for non-nearest-neighbour 2q gates (Metal MPS) — **SHIPPED (PR #233)**

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:feature`, `priority:medium`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** M
**Depends on:** P5.7-03

> **Shipped (PR #233).** `apply_2q_routed`: a non-adjacent 2q gate walks the higher
> qubit down to its neighbour with a **physical** SWAP network (each SWAP a NN 2q
> gate through the existing contract+apply+SVD path), applies the gate with its
> original MSB/LSB orientation, then unwinds the SWAPs — restoring the scaffold's
> site≡qubit-order invariant (so readout/dense stay correct). Lazy permutation
> tracking (P3-12) was deliberately avoided to keep that invariant. Wired into both
> `run` and `run_batched` (non-NN gates flush the pending layer, then route).
> Oracle: long-range/QAOA-style + reversibility; the random property test now emits
> arbitrary-distance pairs. All match CPU MPS / exact SV within 1e-5.

**Description**
Support non-adjacent 2q gates by routing with SWAPs (currently
`UnsupportedInstruction`).

**Context**
NN-only blocks QAOA / long-range circuits. The CPU MPS already does this (P3-06,
lazy-SWAP P3-12); port the approach to Metal.

**Technical Details**

- SWAP network to bring qubits adjacent → apply → (optionally) swap back; or lazy
  permutation tracking (P3-12) to avoid physical swaps. SWAP is itself an NN 2q gate
  through the existing contract+apply+SVD path.

**Acceptance Criteria**

- [x] Arbitrary 2q gate on any pair matches CPU MPS / exact SV within 1e-5.
- [x] Oracle extended with non-NN circuits (long-range CX, QAOA-style).

**Testing Requirements**
Oracle with non-NN circuits; property reversibility.

**References**
P3-06, P3-12 in this backlog.

-----

### [P5.7-07] Canonical-form MPS on Metal → real truncation — **SHIPPED (PR #234)**

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:feature`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** L
**Depends on:** P5.7-03

> **Shipped (PR #234).** `mps/canonical.rs` tracks an orthogonality centre and
> moves it one site at a time via a thin SVD (SVD-based QR/LQ). The gate-by-gate
> path (`run`/`apply_2q_nn`) moves the centre onto the active bond before each
> two-site split, so the gated block has Frobenius² = ⟨ψ|ψ⟩ and the split rescales
> the kept σ by `1/√Σ_kept σ²` — **applying** a bond-cap truncation with controlled
> error instead of refusing it. The `MPS_TRUNC_TOL` refusal is retired on this
> path; `run_batched` stays exact-only (no single centre) and still refuses.
> Oracle: a compressible brickwall at capped χ matches the CPU MPS at the same cap
> (~fp32), the truncated state stays normalised, and a canonical-sweep round-trip
> preserves the dense state; all exact (uncapped) oracles still hold at 1e-5.

**Description**
Maintain an orthogonality centre (canonical form) so a two-site split can
renormalize the kept σ and **apply** truncation instead of refusing it (P5.6-02).

**Context**
The scaffold is non-canonical, so any real bond-cap truncation would corrupt
amplitudes and is therefore refused. Truncation is the whole point of MPS for
large/entangled circuits; without it the GPU MPS only runs exact (small-bond) cases.

**Technical Details**

- Track the orthogonality centre; move it onto the active bond before each 2q split
  (QR/normalization sweeps, GPU where hot); after SVD, absorb σ and rescale by
  `1/√(kept weight)` as the CPU MPS does. Retire the `MPS_TRUNC_TOL` refusal once
  canonical.

**Acceptance Criteria**

- [x] Truncated runs (bond cap below natural bond) match CPU MPS within the
  controlled truncation error.
- [x] The bond-cap refusal is replaced by correct compression; oracle on a
  compressible circuit at a capped χ.

**Testing Requirements**
Oracle vs CPU MPS at matched `max_bond`; truncation-error accounting; property
normalization.

**References**
P3-05 (MPS SVD truncation with controlled error); Schollwöck.

-----

### [P5.7-08] MPS-on-Metal benchmark + Phase 5.7 exit report — **SHIPPED (PR #235)**

**Labels:** `area:backend-mps`, `area:bench`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** M
**Depends on:** P5.7-04

> **Shipped (PR #235).** `benches/mps_vs_cpu.rs`: GPU MPS (`run`/`run_batched`) vs
> CPU MPS across an n-sweep on NN and bond-saturating brickwalls, with a pre-timing
> correctness guard. Exit report appended to `docs/perf/phase5.7.md`. **Verdict:
> exit metric NOT met** — the f64 CPU MPS is 12–190× faster than the GPU MPS at all
> measured scales (n ≤ 14); per-op dispatch latency + serial host overhead dominate
> until χ/n are far larger than the scaffold's dense-readout test cap allows. The
> phase succeeds on correctness/feature-completeness (GPU-resident SVD, batching,
> readout, SWAP routing, canonical truncation) and fails its performance metric;
> the NN gap halves per +2 qubits (crossover extrapolated ~n≈18–20). Honest caveats
> + future levers in the report.

**Description**
A committed criterion benchmark comparing GPU MPS vs CPU MPS across workloads, plus
a consolidated Phase 5.7 exit report with a verdict and a stated exit metric.

**Context**
Current numbers are an ad-hoc ignored timing test on one n=12 d=24 brickwall. A
proper sweep and an exit metric (e.g. GPU MPS ≥ CPU MPS on ≥1 regime) are needed to
call the sub-phase done.

**Technical Details**

- Criterion bench under the `metal` feature (device-or-skip); sweep n / depth / χ;
  report GPU MPS vs CPU MPS wall-clock; fold in P5.7-04 batching.
- Update `docs/perf/phase5.7.md` / `metal.md` with the verdict.

**Acceptance Criteria**

- [x] Bench committed + runs locally.
- [x] Report published with honest caveats; Phase 5.7 exit metric stated and
  evaluated (NOT met — see report verdict).

**Testing Requirements**
The bench itself; numbers reproduced in the report.

**References**
`docs/perf/metal.md`, `phase5.5.md` as templates.

-----

# Phase 5.8 — GPU-resident MPS rewrite (Apple/Metal)

> **Status: COMPLETE (2026-06-21).** All six issues shipped (#238–#243, PRs
> #244–#249). The backend is now fully GPU-resident — zero steady-state device
> allocation, the whole per-gate split + the QR centre moves on one command buffer
> (U/V/σ never read back), and lazy-permutation SWAP routing — and oracle-correct.
> **But the exit metric (GPU MPS ≥ CPU MPS on ≥1 regime) is NOT met:** ~16× slower
> than the f64 CPU MPS at n=16 χ=256, gap *growing* with n (~22× at n=20). The wall
> is the single-threadgroup per-block factorisation; a multi-threadgroup
> factorisation is the precondition for any crossover. Full verdict:
> `docs/perf/phase5.8.md`.

Goal: make the Metal MPS backend **faster than the f64 CPU MPS** in its target
regime (large bond χ, large n), by eliminating the per-gate host round-trips,
allocations, and CPU-side linear algebra that Phase 5.7's exit bench exposed.

> **Why.** Phase 5.7 shipped a correct, feature-complete GPU MPS but it is 12–190×
> *slower* than the Phase-3 CPU MPS at n ≤ 14 — its exit metric is unmet
> (`docs/perf/phase5.7.md`). The audit (`docs/perf/phase5.7-audit.md`) traced the
> loss to: the canonical `run` path doing **host f64 SVDs (`move_center`) on top of
> the GPU SVD** — up to O(n) per gate; **Θ bouncing GPU→host→GPU every gate**;
> **no device-buffer pooling** (a fresh `new_buffer_with_data` per gate); **3
> `wait_until_completed` per gate**; an **O(χ³) host reconstruction guard per gate**;
> and **physical SWAP-with-unwind** for non-NN gates. The CPU MPS avoids every one of
> these (zero-alloc `Scratch` arena, **QR not SVD** for canonicalisation, lazy
> permutation routing, gemm-everywhere). This phase ports those wins to the GPU.
>
> **Honesty up front.** At small χ (≤128) the problem is too small to beat a tuned
> CPU; the realistic win is **χ ≳ 256 and n > 14**, which the scaffold's 28-qubit
> dense-readout test cap cannot currently measure — so P5.8-01 lifts that cap first.
> All entries use the **Phase 5.5 — Apple/Metal GPU** milestone. FP32 is the Metal
> accuracy ceiling (~1e-5).

-----

### [P5.8-01] Large-χ / large-n MPS test + bench harness

**Labels:** `area:backend-mps`, `area:bench`, `type:infra`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** S
**Depends on:** P5.7-08

**Description**
Make the large-bond regime — where a GPU MPS can actually win — measurable.

**Context**
The current oracle/bench cap (28 qubits, dense `2^n` readout) hides the only regime
where the GPU should beat the CPU. Readout is already `2^n`-free since P5.7-05, so
correctness at large n can be checked without the dense vector.

**Technical Details**

- Add bench cells at n ∈ {16, 20, 24} on bond-saturating circuits (χ capped at
  256/512/1024) to `benches/mps_vs_cpu.rs`; report GPU vs CPU and the χ-sweep.
- Correctness at large n via `2^n`-free checks: norm = 1, an analytic GHZ/`Z`-string
  expectation, and `run` vs `run_batched` agreement — no dense compare.
- Keep the small-n dense oracle for exactness; gate the large-n cells `#[ignore]`
  if > 30 s.

**Acceptance Criteria**

- [ ] Bench reports a χ-sweep at n ≥ 16 without a `2^n` allocation.
- [ ] Large-n correctness asserted by `2^n`-free invariants.

**Testing Requirements**
The bench itself; the invariant checks reproduced in the report.

**References**
`docs/perf/phase5.7-audit.md`; P5.7-05 (`2^n`-free readout).

-----

### [P5.8-02] Device-buffer pool (kill per-gate allocation)

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** M
**Depends on:** P5.7-08

**Description**
A reusable, monotonically-growing device-buffer pool (the GPU analogue of the CPU
MPS `Scratch` arena), so the hot path stops calling `new_buffer_with_data` per gate.

**Context**
`DeviceBuffer::from_slice` allocates a fresh `MTLBuffer` every call (`buffer.rs`);
the per-gate path allocates ~6 + 2·(centre steps) of them (Θ, Jacobi A/V/σ, two new
site tensors, canonical moves). The CPU MPS allocates **zero** in steady state via
`Scratch` (`aleph_mps::mps`).

**Technical Details**

- Pool keyed by role/size, `grow`-on-demand, reset (not freed) per gate; sub-views
  at exact shape. Site tensors mutated in place where possible instead of rebuilt.
- Pre-size to χ_max so steady state never reallocates.

**Acceptance Criteria**

- [ ] Steady-state per-gate device allocations ≈ 0 (instrument/assert in a test).
- [ ] Oracle unchanged; bench shows a per-gate-overhead drop.

**Testing Requirements**
Oracle/proptest unchanged; an allocation-count probe; bench numbers.

**References**
`aleph_mps::mps::Scratch` (P3-14); `docs/perf/phase5.7-audit.md`.

-----

### [P5.8-03] GPU-resident per-gate pipeline (no host round-trip)

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** L
**Depends on:** P5.8-02

**Description**
Keep the two-site block and site tensors resident on the GPU across
contract → gate-apply → SVD → write-back, with the σ-handling/packing on-device,
so Θ no longer bounces GPU→host→GPU every gate.

**Context**
Today Θ is contracted on the GPU, **read to host** for a column-major repack,
re-uploaded into a fresh Jacobi buffer, factored, **read back**, and the factors
re-uploaded as new site buffers — negating unified memory. The host pack/sort/
rebuild loops and `.to_vec()` copies are ~96% of per-gate time (P5.7-03 report).

**Technical Details**

- GPU kernels (or one fused kernel) for the column-major pack, σ-sort/select, and
  the `U` / `diag(σ)·Vᴴ` site assembly; collapse the contract+apply+SVD into one
  command buffer with a single `commit`/`wait` per gate (or per layer).
- Move the P5.7-07 reconstruction guard on-device or make it O(k) (e.g. check column
  orthogonality of `U` rather than a full O(χ³) host reconstruction).

**Acceptance Criteria**

- [ ] No host read of Θ between contract and SVD; ≤ 1 `wait` per gate (≤ 1 per layer
  batched).
- [ ] Oracle unchanged; the split phase's host share drops materially.

**Testing Requirements**
Oracle/proptest unchanged; per-phase timing in the report.

**References**
`docs/perf/phase5.7-audit.md` (bounce + host-pack findings).

-----

### [P5.8-04] GPU QR/LQ canonicalisation (replace host SVD moves)

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** L
**Depends on:** P5.8-03

**Description**
Move the orthogonality centre with a **GPU QR/LQ** instead of a host f64 SVD, so the
canonical `run` path stops paying CPU SVDs on top of the GPU SVD.

**Context**
`canonical::move_center_*` runs a host faer SVD per stepped site — up to O(n) f64
SVDs per gate, GPU idle (`docs/perf/phase5.7-audit.md`). The CPU MPS uses **QR**
(2–4× cheaper than SVD) for moves and SVD only for the truncating split
(`aleph_mps::mps::move_center_right/left`). The GPU MPS should do the same on-device.

**Technical Details**

- A Metal Householder/Givens QR kernel (or block-Householder) for the grouped-left
  view; LQ via QR of the adjoint for left moves. Keep f64-fallback for accuracy.
- Reuse the P5.8-02 pool; absorb R via an on-GPU gemm, not a host scalar loop.

**Acceptance Criteria**

- [ ] Centre moves run on the GPU; no host SVD on the move path.
- [ ] Canonical `run` ≈ `run_batched` cost on non-truncating circuits; oracle holds.

**Testing Requirements**
`move_center_preserves_state` + oracle; truncation oracle unchanged; bench.

**References**
`aleph_mps::mps` QR moves; Golub & Van Loan §5 (Householder QR).

-----

### [P5.8-05] Lazy-permutation SWAP routing (no physical swap-back)

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:optimization`, `priority:medium`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** M
**Depends on:** P5.8-03

**Description**
Route non-NN 2q gates by tracking a `qubit_of_site`/`site_of_qubit` permutation
(lazy, P3-12 style) instead of a physical SWAP network that is unwound afterwards.

**Context**
`apply_2q_routed` does adjacent SWAPs (each a gemm + truncated SVD) **and unwinds
them in reverse** (`backend.rs`), ~2× the SWAP cost the CPU MPS pays — it leaves the
permutation in place (`aleph_mps::mps::apply_2q`). The unwind was added to keep
site≡qubit order for readout/dense; lazy tracking must update those paths instead.

**Technical Details**

- Add the permutation maps to `MetalMpsState`; thread them through readout, dense,
  and canonical centre indexing. Discharge user `Gate::Swap` as an O(1) relabel.
- Keep correctness: the dense/readout convention must follow the permutation.

**Acceptance Criteria**

- [ ] Non-NN gate cost ≈ half the current routed cost; oracle (incl. non-NN +
  reversibility) unchanged.
- [ ] User `Swap` touches no tensors.

**Testing Requirements**
Oracle with non-NN/QAOA circuits; readout oracle under a non-trivial permutation.

**References**
`aleph_mps::mps` lazy SWAP (P3-09/P3-12).

-----

### [P5.8-06] Phase 5.8 exit re-bench + verdict

**Labels:** `area:backend-mps`, `area:bench`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 5.5 — Apple/Metal GPU
**Estimate:** S
**Depends on:** P5.8-04

**Description**
Re-run the GPU-vs-CPU MPS sweep after the rewrite and state whether the exit metric
(GPU MPS ≥ CPU MPS on ≥ 1 regime) is now met.

**Context**
P5.7-08 left the metric unmet. This records whether P5.8-02…05 closed the gap, and
in which regime (expected: large χ / large n).

**Technical Details**

- Re-run `benches/mps_vs_cpu.rs` (with the P5.8-01 large-χ cells); update
  `docs/perf/phase5.7-audit.md` / a new `phase5.8.md` with before/after and the
  crossover χ/n.

**Acceptance Criteria**

- [ ] Exit metric re-evaluated with numbers; crossover point stated (or, if still
  unmet, an honest analysis of why and what remains).

**Testing Requirements**
The bench; numbers reproduced in the report.

**References**
`docs/perf/phase5.7.md` exit report; `phase5.7-audit.md`.

-----

# Phase 6 — Multi-GPU and Distributed

Goal: Distributed state vector simulation across multiple nodes.

-----

### [P6-01] NCCL integration for intra-node multi-GPU

**Labels:** `area:backend-dist`, `type:feature`, `priority:critical`
**Milestone:** Phase 6
**Estimate:** L
**Depends on:** P5-08

**Description**
Use NCCL for state vector partition exchange across GPUs within a node.

**Context**
8 GPUs on one node communicate via NVLink. NCCL gives near-bandwidth-optimal collectives.

**Technical Details**

- Partition state vector across GPUs by high qubits.
- All-to-all for global qubit gates.
- Local gates need no communication.

**Acceptance Criteria**

- [ ] 2 GPUs work
- [ ] 4 GPUs work
- [ ] 8 GPUs work
- [ ] Scaling efficiency >70% at 8 GPUs for QFT-32

**Testing Requirements**

- Equivalence vs. single-GPU for small n.

**References**

- <https://github.com/NVIDIA/nccl>

-----

### [P6-02] State vector partitioning strategy

**Labels:** `area:backend-dist`, `type:feature`, `priority:critical`
**Milestone:** Phase 6
**Estimate:** L
**Depends on:** P6-01

**Description**
Design which qubits live where; minimize communication.

**Context**
Partition strategy is the central design choice for distributed state vector.

**Technical Details**

- “Global” qubits → indexed by GPU rank.
- “Local” qubits → indexed within each GPU’s slice.
- Gates on local qubits: free.
- Gates on global qubits: amplitude exchange.
- Strategy: keep gates as local as possible; insert SWAPs if helpful.

**Acceptance Criteria**

- [ ] Partition strategy documented
- [ ] Communication count reported per circuit

**Testing Requirements**

- Equivalence vs. single-GPU.

**References**

- Häner, Steiger. “0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit” (2017).
- Intel-QS paper.

-----

### [P6-03] Gate routing — local vs global decisions

**Labels:** `area:backend-dist`, `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 6
**Estimate:** L
**Depends on:** P6-02

**Description**
IR pass that reorders gates and inserts qubit relabelings to minimize global-qubit gates.

**Context**
Communication is the bottleneck. Minimizing it by clever scheduling is the highest-ROI optimization for distributed.

**Technical Details**

- Cost model: communication per global-qubit gate.
- ILP or heuristic to minimize total comm.

**Acceptance Criteria**

- [ ] Pass implemented
- [ ] Measured reduction in communication

**Testing Requirements**

- Equivalence after routing.

**References**

- N/A — research-level optimization.

-----

### [P6-04] MPI integration for inter-node

**Labels:** `area:backend-dist`, `type:feature`, `priority:critical`
**Milestone:** Phase 6
**Estimate:** XL
**Depends on:** P6-01

**Description**
Extend distributed backend across nodes via MPI.

**Context**
Multiple nodes via InfiniBand. Same patterns as NCCL but cross-node.

**Technical Details**

- Use `mpi` crate (binding to MPI).
- Hybrid NCCL (intra-node) + MPI (inter-node).
- Pinned memory for RDMA staging.

**Acceptance Criteria**

- [ ] 2 nodes work
- [ ] 4 nodes work
- [ ] Scaling efficiency reported

**Testing Requirements**

- Equivalence with intra-node and single-GPU.

**References**

- <https://github.com/rsmpi/rsmpi>
- <https://www.open-mpi.org/>

-----

### [P6-05] Communication-aware circuit compiler

**Labels:** `area:backend-dist`, `area:ir`, `type:optimization`, `priority:medium`, `research`
**Milestone:** Phase 6
**Estimate:** XL
**Depends on:** P6-03

**Description**
Full circuit-level optimizer that considers communication cost.

**Context**
Phase 6’s most ambitious goal. Combine commutation, fusion, partitioning, and routing into one global optimizer.

**Technical Details**

- Multi-objective: gate count + communication + memory.
- Heuristic search (simulated annealing, ILP, RL).

**Acceptance Criteria**

- [ ] Optimizer implemented
- [ ] Benchmark improvement vs. naive distributed

**Testing Requirements**

- Equivalence preserved.

**References**

- Research literature on quantum circuit compilation.

-----

### [P6-06] Distributed benchmark report and v1.0 release

**Labels:** `area:bench`, `area:docs`, `type:docs`, `priority:critical`
**Milestone:** Phase 6
**Estimate:** L
**Depends on:** P6-01 through P6-05

**Description**
Comprehensive distributed scaling report; v1.0 release.

**Context**
Project’s grand culmination.

**Technical Details**

- Strong + weak scaling at 1, 2, 4, 8, 16, 32 GPUs.
- Up to as many qubits as memory allows.
- Final v1.0 release with stability guarantees.

**Acceptance Criteria**

- [ ] Report published
- [ ] v1.0 tagged
- [ ] API stability commitment documented

**Testing Requirements**

- Full benchmark suite passes.

**References**

- Previous phase reports for template.

-----

# Phase 7 — Research Instruments

Goal: use the existing simulator, QEC decoders, and hardware golden models to answer new
research questions: which faults cause failure, how frequent rare failures are, which
observables require simulation, how robust decoder priors are, and whether corrections
arrive before operational deadlines.

This phase spans the simulator and QEC tracks. The specifications below are the source of
truth; QEC backlog documents link here rather than duplicate these issues. It does not
require completion of the Phase 6 distributed release. All proposed API names and new
module paths below are implementation suggestions, not existing APIs.

**Exit metric:** P7-01 through P7-06 meet their scoped acceptance criteria and P7-00 publishes
`docs/perf/p7-research-instruments.md`, including exact/reference validation, reproducible
commands, resource/statistical-efficiency measurements, negative results, and supported
regimes. No speedup or scientific novelty is assumed before measurement.

**GitHub milestone:** [Phase 7 — Research Instruments](https://github.com/aleph-sim/aleph/milestone/20).

| ID | GitHub issue |
|----|--------------|
| P7-00 | [Research instruments integration, examples, and phase exit report](https://github.com/aleph-sim/aleph/issues/542) |
| P7-01 | [Decoder failure witness capture, minimization, and cross-decoder replay](https://github.com/aleph-sim/aleph/issues/543) |
| P7-02 | [Rare-event logical error estimation with fault-count stratification](https://github.com/aleph-sim/aleph/issues/544) |
| P7-03 | [Exact observable-driven execution via causal-cone extraction](https://github.com/aleph-sim/aleph/issues/545) |
| P7-04 | [Decoder robustness experiments under prior mismatch, drift, and bursts](https://github.com/aleph-sim/aleph/issues/546) |
| P7-05 | [Deadline-aware syndrome stream and fallback trace replay](https://github.com/aleph-sim/aleph/issues/547) |
| P7-06 | [Exact near-Clifford expectation engine with sparse Pauli propagation](https://github.com/aleph-sim/aleph/issues/548) |

**Recommended order:** P7-01 → P7-02; P7-03 independently; P7-04 starts with static prior
mismatch; P7-05 reuses Q4/Q7 timing and policy work; P7-06 starts with exact Pauli propagation.

-----

### [P7-00] Research instruments integration, examples, and phase exit report

**Labels:** `area:docs`, `area:bench`, `type:docs`, `priority:high`
**Milestone:** Phase 7
**Estimate:** L
**Depends on:** P7-01 through P7-06 (for closure; coordination starts immediately)

**Description**
Coordinate the six research instruments, establish a shared reproducibility contract, and
publish the phase exit report with runnable examples and measured limits.

**Context**
Aleph already has CPU/GPU state-vector, MPS and stabilizer engines, a communication-aware
compiler, DEM experiment infrastructure, multiple software decoders, and FPGA golden models.
Phase 7 turns those components into research workflows. Another isolated benchmark or
unqualified rare-event/real-time claim would not establish useful new capability.

**Technical Details**

- Keep P7-01/P7-02/P7-04 in the QEC experiment/tooling layer, P7-03 in query planning,
  P7-05 in trace/replay tooling, and P7-06 in an expectation engine. Preserve the backend-agnostic
  IR and current noiseless execution semantics. Do not force expectation-only capabilities
  into APIs that promise amplitudes or samples.
- Each issue supplies a runnable Rust example or CLI/tool command. Document input schema,
  configuration, seed, model/circuit hash, decoder parameters, software revision, precision,
  resource limits, and hardware where relevant. Keep stochastic streams reproducible across
  worker counts, following the existing `(seed, shot_index)` experiment convention.
- Use shared artifacts where appropriate: P7-01 failure witnesses, P7-02 sampling diagnostics,
  P7-04 model provenance, P7-05 timing traces, and P7-03/P7-06 query-plan statistics.
  Identify any schema conversion explicitly; do not hide unmatched model/trace assumptions.
- Report time, peak memory or allocation estimates, decoder calls, uncertainty and effective
  sampling diagnostics as appropriate. Include failed/unsupported regimes and full benchmark
  commands. Use existing oracle tools (FP64 SV/Aer, Stim, PyMatching) within their supported scope.
- Deliver one end-to-end QEC study (sampling → failure witness → cross-decoder replay) and one
  observable-query study (full SV versus causal cone and, where supported, Pauli propagation).
- Python convenience bindings, automatic backend selection, GPU kernels, and RTL changes are
  follow-ups unless a capability issue explicitly needs them. No new release/version commitment.

**Phase Checklist**

- [x] [[P7-01] Decoder failure witness capture, minimization, and cross-decoder replay](https://github.com/aleph-sim/aleph/issues/543)
- [x] [[P7-02] Rare-event logical error estimation with fault-count stratification](https://github.com/aleph-sim/aleph/issues/544)
- [x] [[P7-03] Exact observable-driven execution via causal-cone extraction](https://github.com/aleph-sim/aleph/issues/545)
- [x] [[P7-04] Decoder robustness experiments under prior mismatch, drift, and bursts](https://github.com/aleph-sim/aleph/issues/546)
- [x] [[P7-05] Deadline-aware syndrome stream and fallback trace replay](https://github.com/aleph-sim/aleph/issues/547)
- [x] [[P7-06] Exact near-Clifford expectation engine with sparse Pauli propagation](https://github.com/aleph-sim/aleph/issues/548)

**Acceptance Criteria**

- [ ] All six child issues are linked in a phase checklist and have their scoped deliverables verified.
- [ ] Reproducibility/provenance conventions and runnable examples are documented.
- [ ] `docs/perf/p7-research-instruments.md` includes both end-to-end studies, correctness evidence,
      measurement methodology, resource limits, and negative results.
- [ ] Research estimates, queue-model results, and actual hardware measurements are clearly distinguished.
- [ ] Existing public execution semantics are preserved; new interfaces state their supported inputs.

**Testing Requirements**

- Smoke-run every documented example on its stated environment; small deterministic examples belong in CI.
- Validate artifact round trips and links; reuse each child's meaningful oracle/statistical checks.
- Run required workspace checks for implementation changes; documentation-only changes need link/schema checks.

**References**

- `ROADMAP.md`, Phase 7; `docs/qec/ROADMAP.md`.
- `crates/aleph-qec/src/experiment.rs`; `docs/benchmarking.md`.
- P7-01 through P7-06.

-----

### [P7-01] Decoder failure witness capture, minimization, and cross-decoder replay

**Labels:** `area:qec`, `area:decoder`, `type:feature`, `priority:high`
**Milestone:** Phase 7
**Estimate:** L
**Depends on:** Q0-04 (existing experiment harness)

**Description**
Capture the fired DEM mechanisms behind a failed shot, reduce that fault set while preserving
logical failure, and replay the resulting witness across compatible software decoders and
the hardware golden model.

**Context**
`sample_shots` currently returns syndromes and true observable flips; aggregate logical error
rates do not explain why a decoder failed. Aleph's MWPM, Union-Find, BP/relay-BP and fixed-point
models make cross-decoder failure analysis a practical extension of the existing infrastructure.
A short, reproducible witness can expose trapping sets, bad priors, and regression cases.

**Technical Details**

1. Add an opt-in trace sampler or companion sampler retaining fired mechanism indices. Keep
   ordinary sampling's hot path unchanged when capture is disabled. Store the exact DEM,
   its hash, fault indices, seed/shot identity, decoder configuration, syndrome, true logical
   flips, predicted correction, and available convergence/validity diagnostics.
2. Reconstruct syndrome and truth by XORing the chosen mechanisms' detector and observable
   supports. The failure predicate is a logical mismatch for a fixed decoder configuration.
   Non-convergence is a separate diagnostic, not automatically a logical failure.
3. Use deterministic chunk deletion/delta debugging, followed by a single-fault deletion
   sweep until no deletion preserves failure. The result is **1-minimal**, not a proof of
   globally minimum weight. Respect a decoder-call/time budget and label interrupted results
   as incomplete. Keep stochastic decoder choices fixed so the predicate is stable.
4. Export/import a versioned witness artifact and replay it through compatible decoders.
   Keep original priors fixed during minimization; do not rebuild them from the reduced fault set.
   For fixed-point/RTL reference vectors, use the existing reference-vector path and state which
   code, dimensions, window layout and configuration are supported.
5. Distinguish DEM mechanism indices from physical circuit fault locations. Merged mechanisms
   may have several physical origins; report circuit locations only when provenance exists.
   Group witnesses by a documented signature to avoid counting identical examples repeatedly.
6. Provide a Rust example/tool that captures a failure, minimizes it, and prints a decoder
   comparison. Save useful witnesses as small regression fixtures.

**Acceptance Criteria**

- [ ] Witness round-trip reconstructs the original syndrome, truth, correction and failure.
- [ ] A completed minimization preserves logical failure and passes the explicit 1-minimality check.
- [ ] Budgets, incomplete status, decoder configuration and DEM-versus-circuit provenance are explicit.
- [ ] At least two compatible software decoders replay the same witness; one supported fixed-point
      golden-model witness is exported and checked through the existing reference-vector path.
- [ ] At least one surface-code and one bivariate-bicycle failure case are documented with
      starting/remaining fault counts, decoder calls and runtime.

**Testing Requirements**

- Exhaustively enumerate tiny DEMs to validate reconstruction and 1-minimality, including empty
  syndrome with nonzero logical flips, duplicate supports, and multi-observable cases.
- Check deterministic capture/replay across worker counts and stable decoder seeds.
- Benchmark capture-disabled overhead and minimizer work; compare reconstructed shots with
  the existing sampler and oracle decoders on supported DEMs.

**References**

- `crates/aleph-qec/src/experiment.rs`, `dem.rs`, `decoder.rs`, `refvec.rs`.
- Q0-04: https://github.com/aleph-sim/aleph/issues/298.
- `docs/qec/q7-07-nonconvergence-policy.md` (validity versus logical failure).

-----

### [P7-02] Rare-event logical error estimation with fault-count stratification

**Labels:** `area:qec`, `area:decoder`, `area:bench`, `type:feature`, `priority:high`, `research`
**Milestone:** Phase 7
**Estimate:** XL
**Depends on:** Q0-04; P7-01 (failure artifact/replay integration)

**Description**
Add a rare-event estimator for independent DEM mechanisms, starting with fault-count
stratification, valid uncertainty estimates, and an explicit accounting of unsampled tails.
Compare its statistical efficiency with direct Monte Carlo at matched decoder-call budgets.

**Context**
The current experiment loop samples independent Bernoulli mechanisms and counts failures.
At low logical error rates most calls produce no failure, and zero observed failures do not
establish zero risk. Recent work studies rare-event sampling specifically for circuit-level
QEC and bivariate-bicycle codes with Relay decoding; Aleph has the needed DEM/decoder substrate.

**Technical Details**

1. For a fault set F and K=|F|, estimate `P(fail) = sum_k P(K=k) P(fail | K=k)`.
   Compute binomial weights only for equal mechanism probabilities. For unequal probabilities,
   use a Poisson-binomial dynamic program and an exact conditional sampler (suffix/prefix DP
   is a starting point). Uniform k-subset draws are generally wrong for unequal probabilities.
2. Handle deterministic p=0/p=1 mechanisms explicitly and use stable arithmetic/log-domain
   calculations where needed. Preserve XOR semantics of merged DEM mechanisms. Operate on
   DEM faults; do not label their count as physical circuit fault weight without provenance.
3. Start with fixed, predeclared per-stratum sample budgets. Estimate binomial conditional
   rates with intervals valid for zero/all failures, then combine simultaneous stratum bounds
   and exact weights conservatively. Keep unsampled probability mass as an explicit interval
   contribution (worst-case failure probability 1), rather than silently assuming zero.
   Do not reuse `LogicalErrorResult`'s Wald half-width for this estimator.
4. Report per-stratum weights, sample counts, failures, interval method, omitted tail mass,
   decoder calls, runtime, and reproducibility data. A target-precision run must stop with
   either a justified bound or explicit budget exhaustion. Adaptive allocation/stopping needs
   its own statistical justification; it is not required by the fixed-budget MVP.
5. Integrate witness capture with P7-01 for sampled failures. Keep the decoder's chosen priors
   fixed at the target physical model, even if a future importance/splitting sampler uses a
   different proposal distribution.
6. Benchmark surface-code and bivariate-bicycle fixtures. Compare direct Monte Carlo where
   failures are measurable and exact enumeration on small DEMs. Document where stratification
   gains efficiency and where it does not. Splitting/Metropolis and correlated circuit-noise
   sampling are separate follow-ups, with mixing/bias checks before scientific claims.

**Acceptance Criteria**

- [ ] Equal- and unequal-probability independent DEMs have correct stratum weights and conditional sampling.
- [ ] Results include a non-degenerate bound for zero failures and an explicit unsampled-tail bound.
- [ ] Tiny-model estimates agree with exact enumeration; direct Monte Carlo agrees in accessible regimes.
- [ ] Fixed-budget repeated trials validate the stated interval coverage within reported Monte Carlo uncertainty.
- [ ] A reproducible surface-code/bivariate-bicycle report compares interval width, runtime and decoder
      calls at matched budgets, including an operating point where direct sampling is inefficient.
- [ ] Failures export to P7-01; no rare-rate claim relies solely on extrapolating an empirical fit.

**Testing Requirements**

- Enumerate small heterogeneous DEMs; test p=0/p=1, underflow, overlapping supports and multiple observables.
- Check conditional fault-set frequencies and seed determinism independently of decoder output.
- Use offline repeated-seed coverage/efficiency experiments; keep a small deterministic/statistical
  sanity suite in CI instead of expensive rare-event campaigns.

**References**

- Bravyi and Vargo, Simulation of rare events in quantum error correction:
  https://arxiv.org/abs/1308.6270.
- Rare Event Simulation of Quantum Error-Correcting Circuits:
  https://arxiv.org/abs/2509.13678.
- Fail fast: techniques to probe rare events in quantum error correction:
  https://arxiv.org/abs/2511.15177.
- `crates/aleph-qec/src/experiment.rs`, `decoder.rs`, `dem.rs`.

-----

### [P7-03] Exact observable-driven execution via causal-cone extraction

**Labels:** `area:ir`, `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 7
**Estimate:** L
**Depends on:** P1-11 (existing DCE); P3-10 (existing exact light-cone demo/reference)

**Description**
Plan and execute a requested Pauli expectation on its backward causal cone, with compact
qubit remapping, instead of allocating a full state vector for every circuit qubit.

**Context**
The existing `DeadCodeElim` seeds liveness from measurements and deliberately does nothing
for circuits without them. P3-10 already validates local MPS observables with an exact
light-cone state-vector reference. Making that idea a reusable query path can make large,
shallow, local circuits tractable when the requested observable has a small cone.

**Technical Details**

1. Define an expectation-query planner separate from full-state/sample execution. MVP input:
   unitary circuits starting in |0...0> and Hermitian Pauli strings or real-coefficient sums.
   Reject measure/reset/noise/classical control with actionable errors. General entangled
   initial states require a separate boundary-state treatment and are outside this issue.
2. Walk backwards from each observable's non-identity support. Keep gates touching live
   support and add all their targets/external controls. For fused diagonal/tiled instructions,
   inspect the constituent representation or conservatively retain the full touched support;
   never discard an instruction whose semantics are not understood. Preserve relevant barrier
   ordering and do not interpret a barrier itself as a physical entangling operation.
3. Build a compact circuit and bidirectional original-to-compact map; remap the observable
   with the project's existing qubit/string ordering. Dropped gates have no effect on this
   expectation, but the compact state is not the full circuit's final state.
4. Handle identity observables analytically. For sums, start with per-term cones, deduplicate
   identical plans, and only combine cones when the cost/memory policy justifies it. Do not
   accidentally replace many small cones with one intractable union.
5. Execute compact circuits on exact FP64 SV with an explicit memory cap. Report original and
   active qubits, retained gates, estimated memory, planning/execution time, and query value.
   If the cone is too large, return a clear resource-limit error or an explicitly requested
   compatible backend; do not silently approximate or change the full-state API.
6. Promote the existing P3-10 reference logic where appropriate. Supply a Rust example/tool
   and document complexity in cone width/depth rather than total qubit count.

**Acceptance Criteria**

- [ ] Supported Pauli expectations agree with unpruned FP64 SV/Aer to 1e-10 on small oracle fixtures.
- [ ] Compact maps, controls, Pauli ordering and multi-term coefficients are correct and inspectable.
- [ ] Unsupported operations and resource-limit cases produce explicit errors.
- [ ] A >=100-qubit shallow local circuit computes a tractable local expectation and matches
      an analytic or independent exact-cone reference without full 2^n allocation.
- [ ] Benchmarks include a useful small-cone case and a deep/global case with little or no reduction,
      reporting planning overhead and memory/time changes.
- [ ] Full-state and sampling paths retain their current semantics.

**Testing Requirements**

- Property-test random small unitary circuits/observables against full SV, including non-NN gates,
  external controls, fused instructions, disjoint cones, identity, Y signs and qubit permutations.
- Test rejection of nonunitary input and memory caps before allocation.
- Reuse P3-10 fixtures and publish reproducible scaling versus cone width/depth.

**References**

- `crates/aleph-ir/src/passes/dce.rs`, `instruction.rs`.
- `docs/perf/mps_100q.md`; P3-10: https://github.com/aleph-sim/aleph/issues/123.
- `docs/superpowers/specs/2026-06-10-p3-10-mps-100q-demo-design.md`.

-----

### [P7-04] Decoder robustness experiments under prior mismatch, drift, and bursts

**Labels:** `area:qec`, `area:decoder`, `area:bench`, `type:feature`, `priority:medium`, `research`
**Milestone:** Phase 7
**Estimate:** XL
**Depends on:** Q0-04 (existing harness); P7-01 (diagnostic witness integration)

**Description**
Separate the true generating noise process from the decoder's assumed model, then measure
logical failure and convergence under static prior mismatch, time-varying drift, and correlated bursts.

**Context**
The current harness intentionally samples the same DEM used to build the decoder, which is
correct for matched-model benchmarks. Deployment raises a different question: how badly does
a fixed decoder degrade when probabilities, bias or correlations change? A deliberate
mismatch mode should coexist with the matched baseline and record both models explicitly.

**Technical Details**

1. MVP: sample a generating DEM and construct the decoder from an assumed DEM with identical
   detector/observable identities. Validate structural compatibility; matching counts alone
   are insufficient. Store hashes and both configurations. Start with unchanged mechanism
   supports and perturbed probabilities, so any effect is attributable to prior mismatch.
2. Sweep true versus assumed noise rate and data/measurement or X/Z bias where the selected
   code/model supports it. Compare matched and mismatched decoders on identical generated
   shots (paired experiment) and use a justified interval for the difference.
3. Extend generation to a deterministic schedule of probabilities, then a named latent-state
   burst process with configurable onset, duration, affected mechanisms and strength. Explicitly
   record correlations and time ordering; an independent static DEM cannot encode temporal
   correlation by simply changing individual marginal probabilities.
4. For correlated runs, report uncertainty from independent streams/blocks with a documented
   method rather than treating rounds as independent Bernoulli samples. Store burst state and
   time index in witness provenance; independently regenerating each round would lose the model.
5. Report LER, non-convergence/validity where available, decoder cost and robustness curves.
   A bounded periodic prior-refresh comparator is optional; any fitted updates must use past
   data only and be evaluated on held-out streams. No RL/controller dependency in this issue.
6. Provide static mismatch and drift/burst examples for surface and bivariate-bicycle models,
   with at least two compatible decoder families. Save representative failures through P7-01.

**Acceptance Criteria**

- [ ] The static matched case reproduces existing harness results; incompatible model identities are rejected.
- [ ] Paired static-mismatch curves and full model provenance are reproducible.
- [ ] Time-varying and correlated-burst generators preserve their configured temporal statistics,
      and uncertainty reflects stream/block independence assumptions.
- [ ] At least two decoder families are compared on supported surface/bivariate-bicycle fixtures;
      logical failure is distinguished from non-convergence.
- [ ] Reports include parameter ranges, negative results, runtime and saved failure witnesses.

**Testing Requirements**

- Tiny-model enumeration for static generators; deterministic seeds across worker counts.
- Check measured burst occupancy/duration/correlation against configured laws on fixed campaigns.
- Zero drift/zero burst strength reduces to the matched/static baseline; test detector/observable
  schema mismatches and prohibit access to future samples in any refresh experiment.

**References**

- `crates/aleph-qec/src/experiment.rs`, `dem.rs`, `fixed_bp.rs`.
- Q0-04: https://github.com/aleph-sim/aleph/issues/298.
- `docs/qec/q7-07-nonconvergence-policy.md`.

-----

### [P7-05] Deadline-aware syndrome stream and fallback trace replay

**Labels:** `area:qec`, `area:decoder`, `area:fpga`, `area:bench`, `type:feature`, `priority:medium`
**Milestone:** Phase 7
**Estimate:** L
**Depends on:** Q4-02, Q4-03, Q7-07 (existing stream, latency and fallback work)

**Description**
Build a discrete-event experiment that replays decode traces against syndrome arrivals,
finite queue capacity, correction deadlines and fallback policies. Measure tail latency,
missed deadlines, dropped work and logical outcomes together.

**Context**
Q4 already provides streaming/backlog handling and latency instrumentation. Q7-07 shows a
specific accuracy/latency conflict: host OSD can repair flagged shots but is too slow for the
real-time path. Phase 7 asks whether a complete operational policy keeps up with a workload,
including rare long service times and host/interface delays; it does not rebuild those decoders.

**Technical Details**

1. Define a versioned trace schema with arrival/round identity, decode service time or cycles
   plus clock, code/window/configuration, validity, logical outcome and conditional fallback
   outcome/time where measured. Label synthetic, golden-model and measured-silicon sources.
   Preserve timing/outcome correlations; do not independently shuffle service times and failures.
2. Start with deterministic single-server FIFO event replay; add configurable worker count,
   arrivals, finite queue length and per-job correction deadline relative to arrival. Specify
   what a job represents (round/window/block) and when a committed correction becomes usable.
   Model transfer/host overhead explicitly; raw kernel cycles alone are not end-to-end latency.
3. Compare flag-only, fixed iteration budget, early-exit and optional host fallback using actual
   policy-specific traces/outcomes. Put fallback work on a declared queue/resource. Handle
   simultaneous events deterministically and report overflow/drop/cancel policies.
4. Report waiting/service/end-to-end times (p50/p95/p99/max), queue occupancy, utilization,
   throughput, deadline misses, overflow, flags and decoded logical error separately. Define
   operational failure only under an explicit policy for late/missing corrections: a deadline
   miss is not automatically a physical logical error. Count already-failed/late jobs without
   double counting when presenting a policy-dependent combined metric.
5. Preserve sequential dependencies for streaming windows/commits. MVP uses ordered traces;
   independent trace resampling requires an explicit assumption and cannot establish correlated
   backlog behavior. Do not claim a closed-loop quantum-control simulation from trace replay.
6. Benchmark arrival-rate/deadline sweeps around saturation, including stable average load
   with rare service-time spikes and correlated flagged episodes. Reuse existing hardware
   reference/timing data; do not modify RTL or invent a measured fallback latency.

**Acceptance Criteria**

- [ ] Deterministic finite-queue replay supports documented arrivals, deadlines, worker and fallback policies.
- [ ] Existing Q4/Q7 traces or a documented adapter feed at least one measured/golden-model case;
      every timing source and unmeasured component is identified.
- [ ] A report compares operational policies on identical ordered input, including queue saturation
      and a tail-latency case that mean decode time alone misses.
- [ ] Deadline misses, overflows, validity and logical failures are distinct, with explicit combined-policy semantics.
- [ ] Trace correlations, round/window units and commit dependencies are preserved and documented.

**Testing Requirements**

- Hand-calculated event timelines for empty/stable/overloaded queues, simultaneous arrival/completion,
  finite capacity, multiple workers and fallback contention; verify conservation of submitted work.
- Validate constant-service single-server behavior against analytical utilization/backlog expectations.
- Round-trip traces and compare no-queue/no-deadline replay's logical outcomes with the original campaign.

**References**

- Q4-02: https://github.com/aleph-sim/aleph/issues/313.
- Q4-03: https://github.com/aleph-sim/aleph/issues/314.
- Q7-07: https://github.com/aleph-sim/aleph/issues/458.
- `docs/perf/qec-q4-realtime.md`; `docs/qec/q7-07-nonconvergence-policy.md`.
- `hw/product/README.md`; `crates/aleph-qec/src/relay_window.rs`.

-----

### [P7-06] Exact near-Clifford expectation engine with sparse Pauli propagation

**Labels:** `area:backend-stab`, `area:ir`, `area:bench`, `type:feature`, `priority:medium`, `research`
**Milestone:** Phase 7
**Estimate:** XL
**Depends on:** P3-01 through P3-03 (existing stabilizer conventions); P7-03 (shared expectation-query contract)

**Description**
Compute Pauli expectations by exact Heisenberg-picture sparse Pauli propagation through
Clifford gates and supported Pauli rotations, with explicit limits on term growth and memory.
Establish the useful near-Clifford regime before adding approximate pruning.

**Context**
Stabilizer simulation is efficient for Clifford circuits, while generic SV allocates 2^n
amplitudes. For expectation queries, Clifford conjugation maps each Pauli term to a signed
Pauli term; non-Clifford rotations can branch terms. This creates a possible regime where
cost follows observable term growth rather than total qubit count. It is an expectation
engine, not a universal replacement for state-vector amplitudes or sampling.

**Technical Details**

1. Represent Pauli strings with bit-packed X/Z supports and correctly tracked phases; use
   canonical keys and coefficients. Start with |0...0> inputs and Hermitian Pauli sums.
   Evaluate the final backward-propagated sum analytically: strings containing X/Y have zero
   expectation on |0...0>, while I/Z strings contribute their signed coefficients.
2. Reverse-propagate `U† O U`. Support a documented Clifford gate set and Pauli rotations
   such as Rx/Ry/Rz and, if represented correctly, ZZ rotations. Fix `R_P(theta)=exp(-i theta P/2)`
   conventions before implementation; derive commuting/anticommuting updates and verify signs
   with dense matrices. T/T-dagger may be treated as Z rotations up to irrelevant global phase.
3. Merge identical Pauli terms, handling exact algebraic cancellations without heuristic
   magnitude pruning. "Exact" means no intentional truncation, with stated floating-point
   roundoff. Unsupported generic gates, controls, measure/reset/noise and initial states
   receive explicit errors or an explicitly requested exact fallback.
4. Enforce maximum terms and memory before uncontrolled expansion. Return a structured
   resource-limit result rather than an incomplete expectation. Report peak terms, branching
   rotations, coefficient norms, merge counts, time and memory; cancellations can make the
   number of non-Clifford gates alone a poor predictor of cost.
5. Reuse the P7-03 query contract and cone preprocessing where supported; compare propagation
   both with and without cone extraction. Keep opt-in execution until supported semantics
   and measured crossover justify any automatic routing.
6. Benchmark Clifford-only, a few non-Clifford rotations, growing magic/branch counts, and
   adversarial term-growth cases against FP64 SV and MPS on tractable overlaps. A >=100-qubit
   tractable near-Clifford example needs an independent analytic/cone reference. Approximate
   coefficient pruning, sampling and their error guarantees belong to a separate follow-up.

**Acceptance Criteria**

- [ ] Supported expectations agree with FP64 SV/Aer to 1e-10 on small fixtures, including Y phases,
      negative angles, T/T-dagger and multiple observable terms.
- [ ] Clifford-only propagation has no non-Clifford branching and matches existing reference conventions.
- [ ] No coefficient truncation is hidden; resource exhaustion and unsupported semantics are explicit.
- [ ] A >=100-qubit supported near-Clifford example is independently validated without full-state allocation.
- [ ] A crossover/resource-growth report states useful regimes and adverse cases; GPU/approximate
      capabilities are not implied by the CPU exact prototype.
- [ ] Runnable query example and term/memory diagnostics follow the P7-03 contract.

**Testing Requirements**

- Property-test conjugation against dense matrices on small random supported circuits, including
  controls within supported Clifford gates, qubit permutations and signed Pauli products.
- Check inverse-gate cancellation, identity observables, multiple-term merging and rotation limits
  at Clifford angles. Validate resource checks before expansion/allocation.
- Benchmark peak terms and runtime versus rotation count, topology and observable support.

**References**

- Begusic, Hejazi and Chan, Simulating quantum circuit expectation values by Clifford
  perturbation theory: https://arxiv.org/abs/2306.04797 (research context; MVP is untruncated).
- `crates/aleph-stab`, `crates/aleph-core`, `crates/aleph-ir/src/instruction.rs`.
- P7-03 observable query planning.

-----

# Phase 8 — Performance Audit and Recovery

Tracker: [#550](https://github.com/aleph-sim/aleph/issues/550).
The [integration report](docs/perf/p8-performance-recovery.md) records the
2026-10-18 snapshot: 13/16 capabilities closed, with P8-01/P8-11/P8-12 open.
This phase remains in progress. Closed task state does not imply default adoption
or a universal speedup; research opt-ins and negative regimes stay explicit.

The specifications below restore the existing GitHub issues to the local backlog.
Issue links identify the existing items: do not create duplicates. The checkboxes
in each specification are acceptance requirements, not a synchronized status feed.

-----

### [P8-00] Performance audit integration and end-to-end recovery report

**GitHub:** [#550](https://github.com/aleph-sim/aleph/issues/550) — open as of 2026-10-18.

**Labels:** `area:bench`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** P8-01 through P8-16 (for closure; coordination starts immediately)

**Description**
Track the audit findings, coordinate incremental repairs, and publish an end-to-end report showing which public workflows improved and which hypotheses were rejected.

**Context**
The October 2026 source audit found gaps between benchmark execution and public frontends, eager full-state downloads, sampling scratch omitted from reach policy, and algorithms that retain avoidable work. Historical negative results identify where another broad rewrite is unlikely to pay. Phase 7 adds research capabilities; Phase 8 makes existing execution reliable and efficient.

**Technical Details**
- Keep confirmed source findings separate from unmeasured performance hypotheses. The local 12-qubit probe validated GPU-safe fusion semantics (brickwall 295→32 instructions, QFT on a prepared state 90→34, amplitude error <2e-16); these are instruction counts, not measured GPU speedups.
- Coordinate the issue order: P8-08 and P8-01/P8-02 first; P8-03/P8-04 next; P8-05/P8-06/P8-07 for sampling and trajectories. P8-13 supplies measurement discipline throughout. Hardware experiments proceed independently after their baselines are captured.
- Require one focused PR per capability, meaningful oracle/contract checks, and before/after measurements on an idle relevant machine. Avoid new performance claims from the overloaded audit host or unreviewed synthetic-only fixtures.
- Publish `docs/perf/p8-performance-recovery.md` with public API commands, startup/compile/execute/sample/materialize timing, launches, transferred bytes, peak memory, precision, seeds, machine/software identity, and negative results.
- Preserve strong existing practices: sparse algorithms, lazy permutations, pass/traffic reduction, reference implementations, deterministic shots, and numerical fallback guards. Do not remove safety checks to manufacture speed.
- Research tasks may close with a reproducible no-go result under their stated criteria. The phase report must label that result; it must not imply the speed target was reached.

**Phase Checklist**

- [ ] [P8-01: Use optimized CPU and Metal execution in the CLI](https://github.com/aleph-sim/aleph/issues/551)
- [x] [P8-02: Integrate CUDA fusion and layered dispatch into public frontends](https://github.com/aleph-sim/aleph/issues/552)
- [x] [P8-03: Materialize Python GPU amplitudes lazily](https://github.com/aleph-sim/aleph/issues/553)
- [x] [P8-04: Plan backend reach from requested operations and peak memory](https://github.com/aleph-sim/aleph/issues/554)
- [x] [P8-05: Replace full-state multi-pass CUDA CDF sampling](https://github.com/aleph-sim/aleph/issues/555)
- [x] [P8-06: Avoid building CPU alias tables for one or few samples](https://github.com/aleph-sim/aleph/issues/556)
- [x] [P8-07: Bound noisy trajectory memory and reuse per-worker state](https://github.com/aleph-sim/aleph/issues/557)
- [x] [P8-08: Probe GPU availability without compiling discarded backends](https://github.com/aleph-sim/aleph/issues/558)
- [x] [P8-09: Coalesce GPU QEC scratch while retaining serial-shot arithmetic](https://github.com/aleph-sim/aleph/issues/559)
- [x] [P8-10: Prototype cooperative GPU BP with explicit accuracy and latency trade-offs](https://github.com/aleph-sim/aleph/issues/560)
- [ ] [P8-11: Prove a competitive Metal MPS factorization before another pipeline rewrite](https://github.com/aleph-sim/aleph/issues/561)
- [ ] [P8-12: Validate communication-aware plans on real multi-GPU workloads](https://github.com/aleph-sim/aleph/issues/562)
- [x] [P8-13: Make public-workflow baselines and experiment gates reproducible](https://github.com/aleph-sim/aleph/issues/563)
- [x] [P8-14: Evaluate combined tile residency and overlapped paging](https://github.com/aleph-sim/aleph/issues/564)
- [x] [P8-15: Bound DEM experiment batches and validate decoder result contracts](https://github.com/aleph-sim/aleph/issues/565)
- [x] [P8-16: Use valid logical-error intervals for zero and rare observed failures](https://github.com/aleph-sim/aleph/issues/566)

**Acceptance Criteria**
- [ ] Every child has a verified repair or its explicitly allowed, measured research verdict.
- [ ] The report covers at least CPU CLI, CUDA Python counts-only, large-state sampling, and noisy trajectories.
- [ ] Modelled reach, measured allocation limits, benchmark kernel speed, and public API speed are distinguished.
- [ ] Current baselines and historical limitations are linked without treating completed work as missing.

**Testing Requirements**
- Smoke-run documented commands and reuse child oracle/memory/transfer checks.
- Verify reproducible raw data and current baselines; keep expensive hardware performance campaigns outside correctness CI.

**References**
- `ROADMAP.md`, Phase 8; `BACKLOG.md`, P8-01 through P8-16.
- `docs/perf/p5.9-gpu-fusion.md`, `phase5.8.md`, `q1-03b-sparse-blossom.md`.
- Phase 7 tracker: https://github.com/aleph-sim/aleph/issues/542.

-----

### [P8-01] Use optimized CPU and Metal execution in the CLI

**GitHub:** [#551](https://github.com/aleph-sim/aleph/issues/551) — open as of 2026-10-18.

**Labels:** `area:cli`, `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 8
**Estimate:** M
**Depends on:** P1-09, P1-10, P2-07

**Description**
Route supported CPU/Metal CLI execution through the backend-appropriate optimized pipeline instead of the generic raw-gate driver.

**Context**
`crates/aleph-cli/src/exec.rs::run_with_backend` currently calls `aleph_backend::run` directly. CPU Python uses `run_optimized`; Metal Python uses `MetalSvBackend::run_optimized`. Thus public frontends can execute different pipelines for the same backend while performance reports time optimized drivers.

**Technical Details**
1. Separate execution from output formatting so the helper does not force every backend through raw `run`. Use the existing CPU optimized driver and the Metal supported wrapper; keep an explicit unfused baseline for A/B work.
2. Preserve CPU/Metal qubit-permutation restoration, amplitude ordering, expectation strings, measurement behavior, barriers, precision flags, and error handling. Do not apply SV fusion to stabilizer/MPS/noisy trajectories.
3. Check the FP32 CPU backend's optimized-instruction support before enabling the default there. Any unsupported subset needs explicit handling and tests rather than a silent raw fallback.
4. Cover measurements carefully: CLI returns final-state views, while DCE is seeded by measurements. Optimizing must preserve the CLI's final-state/sample contract, not merely the retained classical measurement distribution; conservatively skip unsafe passes for those queries.
5. Measure actual CLI wall time including planning and output selection on tiny and large workloads. Record retained instructions and planning cost.

**Acceptance Criteria**
- [ ] Supported CLI CPU/Metal cases use the intended optimized driver; raw A/B remains available.
- [ ] Final-state, sampling, expectations, permutation and measurement semantics match the documented contract.
- [ ] Unsupported FP32/IR combinations fail clearly or use a documented safe policy.
- [ ] Public CLI before/after measurements show useful workload gains and expose tiny-input planning overhead.

**Testing Requirements**
- Compare CLI outputs with unfused exact SV and Python equivalents on small fixtures, including nontrivial input preparation, Y terms, controls, barriers and measurements.
- Use deterministic structural checks to prove the optimized path ran, plus idle-machine workload benchmarks.

**References**
- `crates/aleph-cli/src/exec.rs`, `crates/aleph-backend/src/lib.rs`.
- `crates/aleph-py/src/run.rs`, `crates/aleph-metal/src/sv/backend.rs`.
- `docs/perf/parity.md`.

-----

### [P8-02] Integrate CUDA fusion and layered dispatch into public frontends

**GitHub:** [#552](https://github.com/aleph-sim/aleph/issues/552) — closed as of 2026-10-18.

**Labels:** `area:backend-gpu`, `area:cli`, `area:python`, `area:ir`, `type:optimization`, `priority:high`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** P5.9-01, P5.9-03, P5.9-06

**Description**
Make CLI/Python CUDA execution use the existing GPU-safe fusion and layered dispatch mechanisms that the CUDA benchmark suite already measures.

**Context**
The earlier fusion issue #272 is closed and the kernels/helpers exist. The remaining gap is public integration: Python `run_cuda_py` calls raw `run`, and CLI's generic helper does the same. The audit probe demonstrated substantial instruction-count reduction with the existing GPU-safe passes, but did not measure GPU time.

**Technical Details**
1. Add a shared CUDA execution-planning/helper boundary used by CLI, Python, and end-to-end benchmarks. Use GPU-safe fusion, not CPU relabel/tile passes, then supported layered execution. Avoid duplicate optimization and retain raw A/B execution.
2. Preserve measure/reset/barrier behavior, final-state output semantics, controls, seed handling and DiagonalPhase dispatch. If a pass is only safe for measurement-distribution queries, exclude it when frontends request final-state views or re-sample the collapsed final state.
3. Keep FP64 and ordinary FP32 accuracy contracts. Width-4 TF32 fusion has a different tolerance and optional hardware/module availability: require an explicit precision policy, and use width 3 when TF32 is unavailable/disabled. Do not silently expand FP32 error to the TF32 budget.
4. Report planning time, resulting gate kinds/launches, execution time, sampling and transfer separately. Include QFT/QPE, brickwall, VQE and QAOA through the actual public entry points.
5. Reuse existing helpers/oracles and make compiled-plan reuse possible without requiring a new compiler architecture in this issue.

**Acceptance Criteria**
- [ ] CLI/Python use a shared supported CUDA plan and demonstrably emit/execute fusion and layered batches.
- [ ] Small public-API results match raw FP64 SV at each declared precision tolerance.
- [ ] TF32 capability/fallback and precision policy are explicit and tested.
- [ ] Current frontends are benchmarked before/after against the same workloads and hardware, including planning overhead.

**Testing Requirements**
- Extend fusion/layered oracles with frontend-path checks, measurements, barriers and nontrivial prepared states.
- Test no-TF32 capability and unsupported IR; run transfer/launch counters so correctness-only CI cannot pass via a raw path.
- GPU benchmarks must execute on real supported hardware, not silently skip.

**References**
- Previous integration issue: https://github.com/aleph-sim/aleph/issues/272 (closed; helper substrate).
- `crates/aleph-cuda/src/fusion.rs`, `sv/backend.rs`, `sv/fp32.rs`.
- `crates/aleph-py/src/run.rs`, `crates/aleph-cli/src/exec.rs`.
- `docs/perf/p5.9-gpu-fusion.md`, `p5.11-05-tf32-fused-block.md`.

-----

### [P8-03] Materialize Python GPU amplitudes lazily

**GitHub:** [#553](https://github.com/aleph-sim/aleph/issues/553) — closed as of 2026-10-18.

**Labels:** `area:python`, `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** P5-05, P5.11-04

**Description**
Keep GPU results device-resident for counts-only Python calls and download/widen amplitudes only when statevector() is requested.

**Context**
CUDA Python always calls `state.amplitudes_vec()` after sampling; Metal Python always calls `to_aos_f64()`. At n=28 FP32 CUDA this downloads 2 GiB and creates a 4 GiB complex128 host vector even when only counts are needed. The core readout's small-result transfer guarantee does not currently cover the frontend.

**Technical Details**
1. Extend AmpStore/RunResult with feature-gated owned GPU states or a safe lazy materialization holder. Preserve device/context lifetime and satisfy PyO3 sendability/thread ownership without unsafe assertions.
2. Keep sampling before final result construction and hold only the resources required by the result. Make statevector() produce the established complex128 NumPy representation and correct qubit order; define whether repeated calls reuse a host cache.
3. Release the GIL appropriately for device synchronization/copy, and surface device-copy errors as Python errors. Do not change counts-only semantics or force GPU availability in CPU-only builds.
4. Add public-API transfer accounting for CUDA; on Metal use appropriate materialization/allocation counters rather than pretending unified memory transfers are PCIe copies.
5. Measure counts-only versus requested-statevector time and peak host/device memory. Explain the lifetime/memory trade-off of retaining a GPU result.

**Acceptance Criteria**
- [ ] Counts-only Python GPU runs perform no full-state host materialization.
- [ ] statevector() returns oracle-correct complex128 amplitudes on demand, including repeated calls and object cleanup.
- [ ] Thread/context lifetime and failed-copy paths are tested without new unchecked unsafe assumptions.
- [ ] Public transfer and peak-memory measurements demonstrate the removed work.

**Testing Requirements**
- CPU-only, CUDA and Metal feature builds; small amplitude/order oracles and lifetime tests.
- Assert counts-only DTOH bytes scale with shots/scalars, not 2^n, and materialization happens only on explicit access.

**References**
- `crates/aleph-py/src/run.rs` (AmpStore, RunResult, run_cuda_py, Metal arm).
- `crates/aleph-cuda/src/sv/state.rs`, `sv/fp32.rs`.
- `crates/aleph-cuda/tests/transfer.rs`; `docs/perf/p5-05-lazy-transfer.md`.

-----

### [P8-04] Plan backend reach from requested operations and peak memory

**GitHub:** [#554](https://github.com/aleph-sim/aleph/issues/554) — closed as of 2026-10-18.

**Labels:** `area:backend`, `area:backend-gpu`, `area:cli`, `area:python`, `type:bug`, `priority:high`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** P5.11-06

**Description**
Replace state-only CUDA reach assumptions with a capability-aware peak-memory plan that includes sampling, outputs, scratch and runtime device capacity.

**Context**
Reach::for_kind uses fixed 30/31-qubit caps for the reference card. CUDA sampling allocates two 2^n FP64 CDF buffers. At n=30 FP32, 8 GiB state plus 16 GiB CDF already needs at least 24 GiB; at n=31 it needs 48 GiB. These lower bounds cannot fit the reference 20 GiB card, even though its state-only policy selects in-core execution.

**Technical Details**
1. Model requested operations explicitly: execute-only, counts, expectation, marginals, amplitude export and precision. Estimate peak concurrent allocations, optional retained pool resources, matrix/readout scratch and transfer/materialization buffers with checked arithmetic.
2. Probe runtime total/free device capacity cheaply and reserve documented headroom. A snapshot is advisory because other processes may allocate; retain actionable allocation failures.
3. Keep abstract selection independent of CUDA implementation types. Pass capability/memory data from the frontend and make decisions explainable. Support smaller/larger devices instead of treating one card's cap as universal.
4. For an operation that does not fit, reject before large allocation or select an explicitly supported route. The current paged path is norm-only: never silently replace requested counts/expectations with a norm.
5. Coordinate with P8-05 so estimates use the actual sampler's memory model. Cover host export limits separately from device capacity.

**Acceptance Criteria**
- [ ] Estimates include current sampling scratch and output requirements with overflow guards.
- [ ] The n=30/n=31 FP32 counts cases cannot be labelled feasible on a 20 GiB budget with the current sampler.
- [ ] Synthetic capability tests cover multiple card sizes and requested views; explanations identify the limiting allocation.
- [ ] Allocation failures remain clear under changing shared-device memory; no requested output is silently discarded.

**Testing Requirements**
- GPU-free budget/overflow boundary tests and real-device constrained-memory checks without intentionally OOMing the host.
- Compare measured peak allocation with estimates for execution, sampling and export on tractable sizes.

**References**
- `crates/aleph-backend/src/select.rs` (Reach::for_kind, SelectEnv).
- `crates/aleph-cuda/src/sv/readout.rs`, `sv/readout_f32.rs`.
- `crates/aleph-cli/src/exec.rs::run_cuda_paged`, `crates/aleph-py/src/run.rs`.

-----

### [P8-05] Replace full-state multi-pass CUDA CDF sampling

**GitHub:** [#555](https://github.com/aleph-sim/aleph/issues/555) — closed as of 2026-10-18.

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:high`
**Milestone:** Phase 8
**Estimate:** XL
**Depends on:** P5-05, P5.11-04, P8-04

**Description**
Replace Hillis-Steele full-state scan with an efficient scan or hierarchical sampling method, reducing work and scratch while preserving the distribution contract.

**Context**
Both precision readouts allocate two FP64 arrays of 2^n entries and launch scan_step for offsets 1,2,4,... . Building a CDF therefore makes n full-state scan passes even for a small shot count. Device-resident execution avoids PCIe traffic but still does avoidable device-memory work.

**Technical Details**
1. Baseline the existing sampler separately from gate execution, including check_norm, allocation, scan launches and result download. Sweep state size and shot count.
2. Prototype block-local scan with block totals/offsets, or a hierarchical sampler using block probability totals and selected-block sampling. Compare correctness, memory, read passes and shot-count crossover before selecting a default.
3. CUB DeviceScan is a reference for efficient scans; the current NVRTC/Rust integration may require a compiled shim or a custom supported kernel path. Do not add an unmeasured build dependency merely to copy an API.
4. Preserve finite/norm checks, zero-probability bins, deterministic seeded execution within the chosen algorithm, and FP64 accumulation where needed. Parallel summation may alter exact seed-to-outcome mapping; specify that change and verify distributions instead of promising unchanged bits without evidence.
5. Bound allocation before launch and update P8-04's memory model. Retain the old implementation as an explicit benchmark/reference path until validation completes.

**Acceptance Criteria**
- [ ] The default avoids n full-array Hillis-Steele passes and has a documented peak-memory bound.
- [ ] Uniform, sparse, concentrated, mixed-phase and boundary distributions pass numerical/statistical oracles.
- [ ] Repeated fixed-seed runs are deterministic under the documented policy.
- [ ] Before/after data cover FP32/FP64 and small/large shot counts, reporting time, passes and peak scratch.

**Testing Requirements**
- Small exact CDF/reference checks; zero bins, NaNs, norm drift, zero shots and extreme concentration.
- Real-GPU scaling campaigns and transfer checks; meaningful distribution tests with declared sample sizes/tolerances.

**References**
- `crates/aleph-cuda/src/sv/readout.rs`, `sv/readout_f32.rs`, `sv/readout.cu`, `sv/readout_f32.cu`.
- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html.
- `docs/perf/p5-05-lazy-transfer.md`.

-----

### [P8-06] Avoid building CPU alias tables for one or few samples

**GitHub:** [#556](https://github.com/aleph-sim/aleph/issues/556) — closed as of 2026-10-18.

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 8
**Estimate:** M
**Depends on:** P0-11

**Description**
Add a low-shot CPU sampling path that avoids alias-table construction when its preprocessing cost cannot be amortized.

**Context**
sample_impl always computes probabilities and builds an AliasTable. Every noisy trajectory calls sample(state, 1), then discards the table. The table builds thresholds, aliases, scaled weights and index stacks over 2^n amplitudes for a single draw.

**Technical Details**
1. Implement a validated cumulative/streaming weighted draw for shots=1, then measure whether a small-shot threshold is justified. Separate norm/finiteness validation from materializing every probability when possible.
2. Keep the alias path for repeated draws. Preserve normalization drift handling, zero-probability bins, degenerate states and seeded reproducibility within each algorithm.
3. Document any seed-to-outcome change from choosing a different sampling algorithm; do not accidentally change the noisy shot-seed derivation or worker-count independence.
4. Measure sampler-only and end-to-end noisy trajectory workloads, reporting allocation count/bytes, time and crossover by n/shots. Limit the initial PR to the low-shot path rather than rewriting all sampling.

**Acceptance Criteria**
- [ ] Single-shot execution does not build an alias table and retains required state validation.
- [ ] Empirical distributions and exact degenerate cases agree with the established sampler contract.
- [ ] Noisy results remain deterministic across worker counts for fixed configuration/seed.
- [ ] Allocation/time measurements establish a useful regime and preserve the many-shot baseline.

**Testing Requirements**
- NaN/invalid norm, tiny valid branches, uniform/Bell/sparse/random states, zero and many shots.
- Reuse sampling distribution oracles; run paired low-shot versus alias benchmarks on an idle CPU.

**References**
- `crates/aleph-sv/src/measure.rs::sample_impl`, `sampling.rs::AliasTable`.
- `crates/aleph-sv/src/noise/mod.rs::run_one_shot`.
- `crates/aleph-sv/benches/sample.rs`.

-----

### [P8-07] Bound noisy trajectory memory and reuse per-worker state

**GitHub:** [#557](https://github.com/aleph-sim/aleph/issues/557) — closed as of 2026-10-18.

**Labels:** `area:backend-sv`, `area:python`, `area:cli`, `type:optimization`, `priority:medium`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** P4.6-04, P8-06

**Description**
Reuse trajectory buffers and choose concurrency from a memory budget rather than allocating a fresh dense state for every active shot.

**Context**
run_noisy parallelizes over shots; run_one_shot constructs a backend and allocates a state each time. Peak memory scales with active trajectories and sampler scratch, plus a dense 2^n count histogram. Existing channel lookup is already hoisted outside the shot loop, so repeating that optimization would not address this cost.

**Technical Details**
1. Add a worker/chunk execution context with reusable state and scratch. Reset to |0...0> correctly before each shot and seed stochastic work by the existing absolute shot index, independent of chunking/scheduling.
2. Estimate per-trajectory state/channel/sampler peak memory and fixed result memory. Select bounded concurrency from a declared memory policy; clarify that Rayon uses a shared pool and nested kernel work is not automatically threads-squared oversubscription.
3. Start with the current public histogram format, then expose a sparse counts path to frontends if dense 2^n output itself dominates. Preserve the legacy Rust API via an explicit adapter/compatibility policy.
4. Keep noise attached to original gate boundaries. Generic noiseless fusion cannot cross channel insertions without changing the model. Do not remove unitarity/channel validation; any prevalidated plan must prove its immutable-input contract.
5. Benchmark state reuse, sampler changes and concurrency separately to attribute gains; include small states and large memory-bound trajectories.

**Acceptance Criteria**
- [ ] Reused buffers do not leak state between shots; fixed-seed results are worker/chunk invariant.
- [ ] Concurrency and result allocations respect a documented budget or fail before excessive allocation.
- [ ] Gate/channel ordering and the legacy output contract remain correct.
- [ ] Before/after measurements include peak memory, allocations, throughput and noise-model identity.

**Testing Requirements**
- Independent trajectory oracle versus the existing path, including Pauli and Kraus channels, readout noise and zero noise.
- Test consecutive shots with different seeds and constrained concurrency; report dense-result limits.

**References**
- `crates/aleph-sv/src/noise/mod.rs`, `noise/apply.rs`, `noise/model.rs`.
- `docs/decisions/0014-noise-trajectories.md`.
- `crates/aleph-py/src/run.rs::hist_to_counts`.

-----

### [P8-08] Probe GPU availability without compiling discarded backends

**GitHub:** [#558](https://github.com/aleph-sim/aleph/issues/558) — closed as of 2026-10-18.

**Labels:** `area:backend-gpu`, `area:cli`, `area:python`, `type:optimization`, `priority:high`
**Milestone:** Phase 8
**Estimate:** M
**Depends on:** P5.11-06

**Description**
Separate device capability probing from backend/kernel construction and avoid unnecessary GPU probes for explicit backend choices.

**Context**
Both frontends construct SelectEnv before deciding between Auto and Fixed. CUDA availability calls CudaSvBackendF32::new(), compiling/loading modules; CLI Metal availability constructs MetalSvBackend too. A CUDA-enabled explicit CPU run may therefore pay GPU setup, and a selected GPU is then constructed again.

**Technical Details**
1. Resolve explicit requests without irrelevant availability probes. For auto, use a cheap device/capability query and circuit structure to determine which environment facts are actually needed.
2. Separate capabilities from loaded kernels; construct the selected backend once. If module caching is needed, key it by device/context/precision/build configuration and define invalidation/lifetime/error behavior.
3. Keep auto fallback and explicit-backend errors clear. A discovered device does not guarantee kernel compilation succeeds, so handle construction failures deliberately instead of confusing availability with readiness.
4. Measure cold process startup and warm repeated Python calls separately, using construction/compilation counters in addition to wall time.

**Acceptance Criteria**
- [ ] Explicit CPU/stabilizer/MPS requests do not compile CUDA/Metal modules.
- [ ] Auto probing does not construct and discard a complete backend; selected backend is initialized once.
- [ ] Device discovery versus compilation failure behavior is documented and tested.
- [ ] Cold/warm public API measurements and counters show removed setup work.

**Testing Requirements**
- CPU-only builds, absent devices, explicit GPU failure and mocked capability/constructor paths.
- Actual CUDA/Metal smoke tests; verify counters without timing-sensitive CI assertions.

**References**
- `crates/aleph-cli/src/exec.rs::cuda_available`, `metal_available`, run_circuit.
- `crates/aleph-py/src/run.rs::cuda_available`, run_circuit.
- `crates/aleph-cuda/src/sv/fp32.rs` constructors; `crates/aleph-metal/src/context.rs`.

-----

### [P8-09] Coalesce GPU QEC scratch while retaining serial-shot arithmetic

**GitHub:** [#559](https://github.com/aleph-sim/aleph/issues/559) — closed as of 2026-10-18.

**Labels:** `area:decoder`, `area:backend-gpu`, `type:optimization`, `priority:medium`, `research`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** Q3-01, Q3-02

**Description**
Test transposed/tiled per-shot scratch layouts for GPU BP, retaining the existing per-shot arithmetic schedule; evaluate Union-Find only after the regular BP case.

**Context**
GPU BP and UF use one thread per shot with scratch indexed as shot*stride+element. At a common edge loop iteration, adjacent warp lanes access addresses separated by a full graph stride. The source identifies a plausible coalescing problem; its magnitude has not been measured with hardware counters.

**Technical Details**
1. Profile the unchanged kernels with representative graph sizes, shot batches and error rates. Record transactions/bandwidth, occupancy, iteration divergence and scratch footprint.
2. Add an edge-major or tiled AoSoA BP scratch variant so adjacent lanes access adjacent shot slots for a common edge, preserving each shot's edge order and FP64 operations. Keep the current layout as A/B reference and update scratch/batch budgeting.
3. Compare layout-only changes before introducing cooperative scheduling or reduced precision. Graph reads shared by lanes and divergent early stops may limit gains; state those regimes explicitly.
4. UF's irregular parent/BFS accesses may behave differently: profile it independently and only retain a layout variant with measured benefit. Preserve tie-breaking and correction identity.

**Acceptance Criteria**
- [ ] BP layout-only variant matches the reference corrections and convergence behavior on the established oracle campaign.
- [ ] Hardware profiling confirms or rejects the proposed coalescing bottleneck.
- [ ] Time, scratch and batch-size crossover are reported against the same-decoder reference.
- [ ] A measured no-go with reverted candidate code is acceptable; no unsupported speedup claim.

**Testing Requirements**
- Existing 100k-shot BP oracle, degree-1 infinities, graph extremes and deterministic repeats.
- UF correction/tie-order checks if attempted; explicit real-GPU profiling, not construction-skip success.

**References**
- `crates/aleph-cuda/src/qec/bp.cu`, `bp.rs`, `uf.cu`, `uf.rs`.
- `docs/perf/qec-q3-gpu-bp.md`, `qec-q3-gpu-uf.md`.
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (coalesced access).

-----

### [P8-10] Prototype cooperative GPU BP with explicit accuracy and latency trade-offs

**GitHub:** [#560](https://github.com/aleph-sim/aleph/issues/560) — closed as of 2026-10-18.

**Labels:** `area:decoder`, `area:backend-gpu`, `type:optimization`, `priority:low`, `research`
**Milestone:** Phase 8
**Estimate:** XL
**Depends on:** P8-09

**Description**
Prototype warp/block-cooperative BP after layout profiling, and establish whether it improves throughput or single-shot latency at an acceptable accuracy/resource cost.

**Context**
The current exact-match BP kernel serializes all checks/variables/iterations inside one thread per shot. Layout fixes do not remove that serial critical path. Cooperative execution is a hypothesis, not a demonstrated improvement, and reductions can alter numerical/tie behavior.

**Technical Details**
1. Pick a bounded graph/degree regime from P8-09 data. Parallelize independent check updates, then variable updates, with a clear message schedule and synchronization contract.
2. Keep FP64 initially. Preserve within-node reduction order where practical; otherwise document numeric tolerance and quantify hard-decision/convergence disagreement, paired logical error and latency. A rate agreement alone does not establish bitwise equivalence.
3. Compare with both original and layout-improved kernels across batch sizes; expose throughput versus single-shot latency, occupancy, scratch and tails. Do not transfer claims from surface codes to qLDPC without a supported fixture.
4. Keep the exact-match reference and do not change defaults until the target operating regime is measured. FP32, relay-BP and OSD integration are separate follow-ups.
5. Use a predeclared go/no-go experiment; if synchronization outweighs parallel work, publish the negative result and stop the rewrite.

**Acceptance Criteria**
- [ ] Supported cooperative prototype has a documented schedule and oracle/accuracy comparison.
- [ ] Throughput, single-shot/tail latency and memory are measured at matched precision/decoder parameters.
- [ ] At least one supported qLDPC fixture or an explicit surface-only limitation is included.
- [ ] Adoption or rejection follows measured data; negative results can close this research issue.

**Testing Requirements**
- Tiny Tanner graph reference updates, degree-1 and infinity cases, varied syndrome weight and convergence tails.
- Paired decoder outcomes and real-hardware batch/latency campaigns; report rounding/tie disagreements.

**References**
- `crates/aleph-cuda/src/qec/bp.cu`, `crates/aleph-qec/src/bp.rs`.
- `docs/perf/qec-q3-gpu-bp.md`; P8-09.

-----

### [P8-11] Prove a competitive Metal MPS factorization before another pipeline rewrite

**GitHub:** [#561](https://github.com/aleph-sim/aleph/issues/561) — open as of 2026-10-18.

**Labels:** `area:backend-mps`, `area:backend-gpu`, `type:optimization`, `priority:medium`, `research`
**Milestone:** Phase 8
**Estimate:** XL
**Depends on:** P5.8-06

**Description**
Benchmark and prototype multi-threadgroup/tiled MPS SVD/QR factorization, with a go/no-go gate before integrating another large Metal MPS rewrite.

**Context**
Phase 5.8 removed major host/pipeline costs but the n=16 chi=256 benchmark changed only a few percent; GPU remained 16–22x slower than CPU. gpu_jacobi and move_gpu still dispatch one threadgroup per factorization, including blocks up to 512x512. The numerical guard/f64 fallback is essential and must stay.

**Technical Details**
1. Isolate representative two-site split and center-move matrices at chi=32/64/128/256/512, including ill-conditioned and truncating cases. Measure pack/factor/sort/install/guard/fallback separately against the current GPU path and faer CPU.
2. Prototype blocked Jacobi or QR stages across multiple threadgroups/dispatches with explicit inter-dispatch dependencies. Do not assume increasing threadgroup size unlocks the whole GPU.
3. Validate reconstruction, orthogonality/canonical form, discarded weight, norm and final observables at the existing tolerance. Quantify fallback frequency and its cost rather than disabling guards.
4. Establish a useful factorization crossover before wiring a complete MPS pipeline. If it exists, integrate only that regime and rerun end-to-end MPS against CPU with actual bond-size distributions and dispatch/sync costs.
5. Small-bond CPU fallback and batching are independent controls; avoid claiming fully resident execution when the chosen path falls back to CPU.

**Acceptance Criteria**
- [ ] A stage-level matrix/shape benchmark explains the existing wall and fallback frequency.
- [ ] Prototype preserves reconstruction/canonical/truncation correctness on hard fixtures.
- [ ] An integration go/no-go verdict is backed by factorization and projected/measured end-to-end costs.
- [ ] No-go can close the research issue with documented limits; only measured beneficial regimes change production routing.

**Testing Requirements**
- Existing CPU/FP64 SV MPS oracles and chi-saturating examples.
- Shape/conditioning sweeps, fallback accounting, and idle-Metal before/after timings; preserve current reference implementation.

**References**
- `crates/aleph-metal/src/mps/gpu_jacobi.rs`, `move_gpu.rs`.
- `crates/aleph-metal/src/shaders/mps_jacobi.metal`, `mps_qr.metal`.
- `docs/perf/phase5.7-audit.md`, `phase5.8.md`.

-----

### [P8-12] Validate communication-aware plans on real multi-GPU workloads

**GitHub:** [#562](https://github.com/aleph-sim/aleph/issues/562) — open as of 2026-10-18.

**Labels:** `area:backend-dist`, `area:ir`, `area:bench`, `type:optimization`, `priority:medium`, `research`
**Milestone:** Phase 8
**Estimate:** XL
**Depends on:** P6-05

**Description**
Benchmark the current compiler on actual multi-GPU hardware and diagnose compute-pass inflation, representation loss and communication before choosing further scheduling changes.

**Context**
The earlier AWS n=28 random FP32 run measured 0.87x speedup on two L4s and 1.06x on four. Distributed plans inflated local compute/fusion work before link cost. P6-05 and state-aware cost calibration have since improved plans, but a predicted compiler gain on emulated ranks is not an actual multi-device measurement.

**Technical Details**
1. Establish current fused single-device, naive distributed, lookahead and compiled baselines with identical circuit/precision/query and reported hardware topology. Record NCCL/P2P/host staging capability rather than assuming it.
2. Decompose wall time into planning, local kernel kinds/passes, exchange bytes/events, synchronization and readout. Detect lost diagonal specialization, extra local swaps, fusion fragmentation and host serialization using the current code, not stale pre-fix reports.
3. Compare useful work and achieved time for both fixed-size strong scaling and fixed-memory reach. QFT's free SWAP relabeling can make superlinear speedup a change in work, so annotate that baseline.
4. Select one measured schedule/representation lever and validate it separately. Do not add hardware, widen fusion or tune a link model when compute inflation is the dominant term.
5. Freeze calibration choices before held-out workload evaluation. Keep predictions and measured results in separate columns and include uncompetitive cases.

**Acceptance Criteria**
- [ ] Current multi-device measurements compare single-GPU, naive/lookahead and compiled plans on at least 2/4 GPUs when hardware is available.
- [ ] Kernel/communication/work inflation explains scaling limits with attributable measurements.
- [ ] A selected optimization has before/after oracle and end-to-end evidence, or a documented no-go conclusion.
- [ ] Hardware-unavailable results remain labelled prediction-only and do not close measured scaling claims.

**Testing Requirements**
- Existing distributed equivalence/host-overlap tests, small actual multi-device oracles, held-out plan selection cases.
- Strong/weak scaling with topology, precision and total-work counters; do not disrupt shared services without the repository's authorized isolation procedure.

**References**
- `docs/perf/p6-multi-gpu.md`, `p6-05-compiler.md`.
- `crates/aleph-ir/src/dist/compile.rs`, `specialize.rs`.
- `crates/aleph-cuda/src/dist/device_sv.rs`, `cost.rs`, `nccl.rs`.

-----

### [P8-13] Make public-workflow baselines and experiment gates reproducible

**GitHub:** [#563](https://github.com/aleph-sim/aleph/issues/563) — closed as of 2026-10-18.

**Labels:** `area:bench`, `area:infra`, `area:docs`, `type:feature`, `priority:high`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** —

**Description**
Create a common end-to-end benchmark/provenance workflow and require bottleneck/crossover evidence before expensive optimization campaigns.

**Context**
Historical evidence includes an inert min_amps axis in a 360-cell chunk sweep, Metal pipeline work ahead of competitive factorization, tiny TF32 margins measured best-of, GPU-MWPM/UF algorithm swaps inside speedup headlines, and cost-model contamination from services restarted by a watchdog. Existing reports are valuable but their comparison contracts differ.

**Technical Details**
1. Record revision, exact public entry point, precision/error budget, optimizer flags, workload seed/model, shots, threads/workers, warmed/cold state, compiler/device identity, topology and raw repeat data in a machine-readable run manifest.
2. Time startup, parse/compile, execution, sample and materialization separately, plus total caller wall time. Run the same public route used by customers; kernel-only results are an additional tier.
3. Use paired/interleaved A/B order and report medians/spread and uncertainty for small wins. Treat best-of as an explicitly labelled throughput ceiling, not sufficient evidence of a stable 2% gain. Verify idleness throughout; obey shared-service isolation authorization and restoration rules.
4. Require a preflight experiment that the swept parameter changes the executed path/counters. Sweep thresholds near the actual transition instead of spending trials on an inert axis. Use a bottleneck upper bound/crossover estimate and a declared stop condition before broad rewrites.
5. Compare same algorithm/precision/core budget first. Keep workflow algorithm changes in separate tables with accuracy data. Re-measure MWPM after Sparse Blossom; archive older dense-MWPM numbers as historical.
6. Add structural/transfer/memory guards through public API paths. Feature compilation or tests that silently skip GPU construction do not establish hardware execution; hardware campaigns must explicitly report executed/skipped status.

**Acceptance Criteria**
- [ ] Reusable manifests and commands capture total/stage timing, provenance, repeats and resource counters.
- [ ] At least CPU CLI and CUDA Python reference workflows use the harness.
- [ ] Current same-decoder and same-resource baselines are separate from historical/cross-algorithm headlines.
- [ ] Historical no-gain cases are summarized with their causes and experiment stop rules.
- [ ] CI guards exercise intended frontend paths structurally; real-hardware results cannot be recorded as passes when skipped.

**Testing Requirements**
- Validate manifest schema, subprocess failures, skipped hardware, warm/cold labels and counter capture.
- Reproduce a known fusion/transfer improvement and a small/no-gain case; avoid wall-time thresholds in correctness CI.

**References**
- `docs/perf/phase2-p2-04.md`, `phase5.8.md`, `p5.11-05-tf32-fused-block.md`.
- `docs/perf/qec-q3-montecarlo.md`, `q1-03b-sparse-blossom.md`, `p6-05-compiler.md`.
- `.github/workflows/ci.yml`; `docs/benchmarking.md`.

-----

### [P8-14] Evaluate combined tile residency and overlapped paging

**GitHub:** [#564](https://github.com/aleph-sim/aleph/issues/564) — closed as of 2026-10-18.

**Labels:** `area:backend-gpu`, `type:optimization`, `priority:low`, `research`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** P5.11-02, P5.11-03, P8-13

**Description**
Evaluate a paging executor that combines existing low-qubit multi-gate residency with copy overlap, using the measured duplex ceiling and real locality distribution to bound expectations.

**Context**
run_paged_batched amortizes transfers over low-only gate runs and achieved up to 9.93x on a locality-rich fixture. run_paged_overlapped overlaps per-gate copies and measured about 1.10x with a 1.24x duplex ceiling. They are separate entry points; the report's conceptual composability does not establish a combined production path or multiplicative end-to-end gains.

**Technical Details**
1. Baseline both existing executors and inspect actual low-run lengths/high-qubit interruptions on held-out workloads. Retain synchronous residency as the main comparison.
2. Prototype an event-ordered ring of resident tile groups: apply a complete safe low-only run per load, overlap independent groups, and enforce prior scatter completion before host-memory reuse and high-gate transitions.
3. Budget all device buffers and pinned host memory; avoid losing tile size/headroom to an oversized overlap ring. Handle shallow pipelines with the proven synchronous fallback.
4. Measure copies, bytes, residency, fill/drain and compute separately. Large locality gains are fixture-dependent; overlap benefits cannot exceed the measured link/pipeline opportunity.
5. Start FP64, mirror FP32 only if worthwhile. Qubit relabelling/locality-aware scheduling is a later measured lever, not necessary to prove safe composition.

**Acceptance Criteria**
- [ ] Prototype preserves paging order and matches the synchronous reference at declared precision.
- [ ] Memory budgeting and high-qubit flush/fallback behavior are explicit.
- [ ] Held-out workload A/B measures traffic, wall time and regimes with no gain.
- [ ] Adopt only a measured useful configuration; a documented no-go result can close the research issue.

**Testing Requirements**
- Forced small tiles with mixed controls/targets and alternating low/high gates; compare complete amplitudes against CPU/reference paging.
- Real-device event-order tests and interleaved idle-machine timing; zero hidden synchronous copies.

**References**
- `crates/aleph-cuda/src/sv/paged.rs`, `paged_f32.rs`.
- `docs/perf/p5.11-02-overlapped-paging.md`, `p5.11-03-multi-gate-residency.md`.

-----

### [P8-15] Bound DEM experiment batches and validate decoder result contracts

**GitHub:** [#565](https://github.com/aleph-sim/aleph/issues/565) — closed as of 2026-10-18.

**Labels:** `area:qec`, `area:decoder`, `type:bug`, `priority:medium`
**Milestone:** Phase 8
**Estimate:** L
**Depends on:** Q0-04

**Description**
Process logical-error experiments in bounded batches and reject incomplete decoder outputs before scoring instead of materializing all shots and silently truncating zip iteration.

**Context**
run_dem_experiment samples all syndromes/truths into vectors, decodes the full batch, then zips predictions with truths. Memory grows with total shots. If a custom decoder returns fewer predictions, zip skips unreturned shots but the reported denominator is still the requested shot count; missing observable bits are also treated as false by mispredicted.

**Technical Details**
1. Introduce bounded batch size/memory policy and sample by absolute shot index so chunking preserves the existing deterministic stream. Keep sample_shots as an explicit artifact API for callers needing every shot.
2. Validate decode_batch output length and each correction's observable schema before scoring. Return a clear decoder-contract error for incomplete results instead of counting only the returned prefix.
3. Accumulate exact counts across batches with checked arithmetic. Expose processed/requested counts for any intentional partial/budgeted experiment; never silently use an unprocessed denominator.
4. Consider packed syndromes/truth bits and reusable per-worker sampling scratch after the batch boundary is correct. Benchmark small fast decoders where allocation overhead matters and large campaigns where memory dominates.
5. Preserve external PyMatching batch amortization and existing decoder thread-local caches.

**Acceptance Criteria**
- [ ] Total experiment memory is bounded by configured batch size, not total shots.
- [ ] Short/long prediction vectors and incompatible observable lengths fail explicitly.
- [ ] Fixed-seed exact tallies match the previous valid full-batch path across batch sizes/workers.
- [ ] Sampling, decode, scoring time and peak memory are reported for representative campaigns.

**Testing Requirements**
- Custom intentionally malformed decoders, zero/partial final batches, multiple observables and deterministic mechanisms.
- Existing CPU/oracle experiments and fixed-seed chunk invariance; packed conversion round trips if introduced.

**References**
- `crates/aleph-qec/src/experiment.rs::run_dem_experiment`, sample_shots, mispredicted.
- `crates/aleph-qec/src/decoder.rs::Decoder`, `syndrome.rs`.
- Phase 7 P7-01/P7-02 rely on trustworthy experiment scoring.

-----

### [P8-16] Use valid logical-error intervals for zero and rare observed failures

**GitHub:** [#566](https://github.com/aleph-sim/aleph/issues/566) — closed as of 2026-10-18.

**Labels:** `area:qec`, `area:bench`, `type:bug`, `priority:high`
**Milestone:** Phase 8
**Estimate:** M
**Depends on:** Q0-04

**Description**
Replace or supplement the current Wald half-width with a documented interval that remains meaningful for zero/all failures and small counts.

**Context**
LogicalErrorResult::new computes 1.96*sqrt(p*(1-p)/n). Zero observed failures yields rate=0 and ci95=0, which cannot establish zero logical risk. The same boundary problem occurs at all failures. Phase 7 rare-event estimators need their own interval method; the existing direct Monte Carlo baseline also needs valid uncertainty.

**Technical Details**
1. Choose and document an interval (Wilson or exact binomial are starting points), its confidence level, two/one-sided interpretation and zero-shot behavior. Represent lower/upper bounds rather than assuming symmetry around the sample proportion.
2. Inventory Rust/Python/CLI/report consumers of ci95 and define a compatibility/migration policy. Do not silently reinterpret a public half-width as an asymmetric interval.
3. Update comparison/report helpers to use the declared bounds correctly; overlapping intervals are not a proof of decoder equivalence. Preserve paired comparisons where matching shot streams are available.
4. Keep ordinary Bernoulli experiment intervals separate from stratified/weighted/correlated estimators in P7-02/P7-04.
5. Validate coverage offline for small/rare cases and document zero-event upper bounds; zero-shot experiments have no evidence, not a measured zero failure probability.

**Acceptance Criteria**
- [ ] Zero/all-failure and small-count experiments have valid documented bounds; zero-shot semantics are explicit.
- [ ] Public result/report consumers follow a compatible, tested migration policy.
- [ ] Reference binomial cases and offline coverage checks support the stated method.
- [ ] Weighted/correlated estimators cannot accidentally reuse the ordinary direct-sampling interval.

**Testing Requirements**
- Exact boundary/reference cases, large n, monotonic bounds and invalid count inputs.
- Offline repeated-sampling coverage within reported simulation uncertainty; small deterministic cases in CI.

**References**
- `crates/aleph-qec/src/decoder.rs::LogicalErrorResult::new`.
- `crates/aleph-qec/src/experiment.rs`.
- `BACKLOG.md`, P7-02 and P7-04 (distinct estimator contracts).

-----

# Cross-phase CI Infrastructure

### [CI-01] Reduce CI load: docs fast path, then dependency-aware code checks

**GitHub:** [#597](https://github.com/aleph-sim/aleph/issues/597).

**Labels:** `area:infra`, `type:infra`, `priority:high`
**Estimate:** L (two sequential, independently reviewable stages)
**Scope:** Implement both stages in one task and one PR, as requested by the user: **A — documentation fast path**, then **B — dependency-aware code checks**. Keep separate validation gates inside the same PR; do not close the issue until both stages are verified.

## Execution order

- [ ] **Stage A:** Fast hosted checks for approved docs-only changes, retaining the existing full route for code/config/unknown changes.
- [ ] **Stage B:** Select code checks from affected dependencies, consumers, features and platforms; retain conservative full-suite fallbacks and integration audits.

The gate/classifier infrastructure from A is reused by B. Stage B must not delay the docs improvement or silently reduce A's safety guarantees.

## Problem and observed evidence

Documentation-only PRs currently run the same eight CI jobs as production changes. `.github/workflows/ci.yml` has no change classifier or documentation fast path: rustfmt, Clippy (including CUDA/NCCL), Linux stable/beta workspace suites, two Python wheel/test jobs and macOS stable/beta workspace/Metal suites all run.

For example, [report PR #596](https://github.com/aleph-sim/aleph/pull/596), head `d02d8b51dfb3cb6c1d21bff94e81f3ac692e1922`, changes reports, backlog/roadmap and saved validation receipts. In [CI run 39158713634](https://github.com/aleph-sim/aleph/actions/runs/39158713634), macOS stable took **22m45s** and Linux beta **13m16s**; Linux stable did not start until 19:41:25 UTC after the run's first jobs started around 19:25:39. These are individual observed job durations/start times, not a completed-run total, billing estimate or causal proof of every queue delay. No Rust implementation changed.

The desired outcome is concrete: ordinary prose/report PRs should get meaningful checks in minutes without occupying the self-hosted development/measurement runner or rebuilding Python wheels and the Rust workspace.

## Stage A: a safe docs-only route

1. Add a small **GitHub-hosted**, read-only change-planning job. Keep its workflow triggered for every relevant PR, so the merge result always exists. Retain all current heavy checks for every change that is not confidently classified as documentation-only.
2. Use an explicit allowlist of prose/report locations (for example approved root Markdown files, `docs/**` Markdown/images and identified report receipts). A file extension or `docs/` prefix alone is insufficient: documentation trees can contain fixtures, generated artifacts, scripts or inputs used by tests/builds. Audit repository consumers before approving non-prose paths; unknown formats/paths take the full route. `.rs` documentation edits, rustdoc, examples, Python notebooks/scripts and build inputs remain code-bearing changes.
3. Compute the **complete** PR diff against the appropriate base/merge-base, including deleted/renamed paths and both sides of a rename. Do not rely on a truncated changed-files response or a shallow-fetch failure. A mixed docs+code PR takes the full route. An incomplete/ambiguous diff falls back to full CI; inability to produce a valid plan fails the merge gate.
4. The docs route performs useful bounded checks: changed local link/asset targets, relevant Markdown consistency and JSON/JSONL/schema/checksum validation where a known report format declares it. Avoid live external-link checks as a blocking network-flaky gate. Do not execute arbitrary fenced shell snippets. Run only explicitly mapped executable-example checks where required; otherwise classify those changes for the existing full route.
5. Ordinary docs-only PRs must launch **no self-hosted Rust jobs, no workspace build/test, no CUDA/Metal feature build and no Python wheel build**. Publish a concise job summary explaining the classification and omitted suites. Do not label omitted tests as executed/passed.
6. CI workflows/actions, classifier scripts/tests, `.github/**` policy, Cargo manifests/lockfile, toolchain/config/build scripts, packaging configuration, source/test scripts and unrecognized paths must never use the docs shortcut. Initially select the unchanged full CI route for all of these.

## Reliable merge semantics

- Add one stable final gate (for example `ci-required`) that always evaluates the successful plan and **all selected jobs**, even when dependencies failed. A job-level `always()`/`needs` design must explicitly reject failed planning, malformed/missing outputs, failed/cancelled/missing selected jobs and a selected job unexpectedly skipped.
- An intentionally unselected heavy suite is allowed to be skipped only when explained by the valid docs plan and successful docs checks. This is different from treating skipped/cancelled checks as test success.
- Do not solve this solely with workflow-level `paths-ignore`. A filtered-out workflow can leave expected checks pending; a conditionally skipped job can otherwise appear acceptable without proving the intended checks ran. See [GitHub's required-check semantics](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks#handling-skipped-but-required-checks).
- Bind plan/result reporting to the current PR head and relevant base/tested revision; stale results from a previous commit must not authorize a merge. Recompute on every new commit/base update. Preserve existing concurrency cancellation, but a cancelled run cannot satisfy the gate.
- Audit branch rules and merge-monitor expectations before rollout. On 2026-10-18 the branch-protection API reported `Branch not protected` and the repository ruleset list was empty; therefore the current all-checks policy also lives in our manual/automation merge instructions. Recheck this at implementation time. Do not silently enable/disable protection or relax any current PR's authorized merge conditions.
- Update the documented merge policy and future monitors to use the new aggregate **only after it is validated and accepted**. Existing monitors requiring eight/nine completed successful jobs retain their contract until explicitly migrated. This task must not retroactively waive PR #596's checks.
- Preserve fork/self-hosted trust boundaries, read-only token defaults and checkout without persisted credentials. Docs planning/checks should work on hosted runners for forks; untrusted changes must not gain self-hosted execution or secrets. Do not use `pull_request_target` to execute PR code. Changes to the classifier/workflow themselves require the full route and review.

## Main-branch and full-check policy

Use the same safe docs classification for main pushes so merging a report does not immediately re-run the expensive suite there. Define the push comparison range explicitly; an unavailable range selects full CI. Code-bearing main pushes keep existing full validation in this first task.

Provide a documented full-suite override and retain a broad default-branch audit (scheduled or manually requested). Treat separately dispatched validation as extra evidence, not an automatic substitute for the PR's required check; GitHub's event semantics matter. Do not enable timed performance benchmarks on a shared runner as part of this change. `bench.yml` is currently manual-only; preserve that policy. Release/tag/wheel publication behavior remains unchanged.

`hw.yml` already path-filters PRs but runs on every main push. Apply/audit the same docs exception there so a docs merge does not run RTL co-simulation unnecessarily. Preserve checks for RTL, generated hardware inputs and the Rust golden model. Do not simply assume its current `aleph-qec/**` list captures every transitive golden-generator input; record any broader dependency gap as follow-up work rather than weakening it.

## Rollout and tests

1. Add table-driven classifier tests and temporarily report the proposed route in shadow mode while the old merge gates stay authoritative. Exercise both successful routes before skipping expensive jobs.
2. Cases must include root prose; docs prose/images; known report JSON/JSONL and checksum receipts; malformed receipts; mixed docs+Rust; deleted/renamed files; docs renamed into source and vice versa; paths with whitespace; more than 300 changed files with code at the end; missing/truncated diff; unknown path; workflow/classifier edit; Cargo.lock/toolchain edit; new PR commit/base movement; and fork PRs.
3. Gate tests must show failure for classifier failure, absent outputs, cancelled/failed selected jobs and unexpected selected-job skips; a valid docs plan plus passing docs checks must succeed with explicitly unselected heavy jobs omitted.
4. Run a real docs-only PR and a mixed/code-bearing control. Record selected jobs, check conclusions, self-hosted job count, queue wait, elapsed wall and summed runner time separately. Demonstrate that the docs shortcut also holds for a docs-only main push.
5. Initial docs target: **no expensive jobs and roughly 1–3 minutes active execution with warm hosted tooling**. Queue time is reported separately, not promised. Compare several real runs, retain failures and choose acceptance thresholds from measured setup costs; do not claim a saving based only on predicted job counts.

## Stage A acceptance criteria

- [ ] Pure approved documentation changes receive useful hosted checks and a stable successful merge gate without Rust/Python/GPU/RTL suites or self-hosted allocation.
- [ ] Mixed, ambiguous and code/config changes retain existing full CI coverage; no regression is hidden by a skipped job or planner failure.
- [ ] The same docs exception works on main without changing code-bearing main/release behavior.
- [ ] Classifier and merge-gate edge/security cases have meaningful automated coverage; a real docs and real code control validate routing.
- [ ] Before/after route, elapsed/queue/runner-time evidence and known limitations are documented.
- [ ] Merge instructions and future monitors describe selected-success versus intentionally-unselected checks; migration does not bypass existing PR authorization.

## Stage B: dependency-aware checks for code changes

Start only after Stage A routing/gate behavior is verified. Replace the unconditional full route for confidently understood code changes with a conservative affected-check plan. File location is the seed, not the entire affected surface.

### Dependency and configuration model

1. Derive the workspace package graph from Cargo metadata. Include normal, build, dev/test, optional-feature and target-specific dependencies; compute the transitive reverse-consumer closure needed by the selected tests. Shared test/oracle helpers can connect otherwise separate packages. Cycles must terminate deterministically. Record the feature/platform assumptions instead of treating a default-feature graph as complete.
2. Maintain explicit non-Cargo edges for Python bindings/tests, CLI integration, shader source/includes, generated artifacts, golden-vector generators, scripts, fixtures and benchmark harnesses. Verify actual consumers before excluding checks. Unknown/unmapped inputs take the full route.
3. Select **package + test scope + platform + feature set**. A crate name alone is not a coverage plan: Metal/CUDA crates are empty without their respective features. Do not use a blanket `--all-features` build that mixes unsupported platform configurations; use audited configurations.
4. Preserve current contracts: Linux debug, macOS release/cross-platform coverage, stable/beta behavior, MPS no-default-features, CUDA/NCCL feature checks, Metal feature checks, Python 3.10/3.12 wheel/binding contracts, and RTL golden checks where affected. Any reduction of these dimensions is a separate explicit policy decision, not a side effect of path filtering. The aggregate must inspect actual selected job outcomes even when a legacy beta job is marked continue-on-error.
5. Planner/workflow changes, workspace manifests/lockfile/toolchain/build configuration, ambiguous diffs and unknown paths retain the full route initially. Optimize these broad triggers only after independent evidence that a narrower plan is safe.

### Expected routing examples — validate against the actual graph

| Change seed | Required affected coverage | What it must not assume |
|---|---|---|
| Approved prose/report only | Stage A hosted docs checks and aggregate | No Rust/Python/GPU/RTL execution merely because another workflow also handles code |
| `aleph-metal` / Metal shaders | Real `metal` feature build/lint/tests on macOS; affected feature-enabled CLI/Python consumer builds/contracts; any impacted reference/test packages | Default workspace build proves Metal coverage, or every macOS test is a Metal test |
| `aleph-cuda` / CUDA kernels | `cuda` and relevant `nccl` checks, pure no-device contracts, affected CUDA-feature CLI/Python consumers; dependency-selected reference tests | CUDA compilation proves execution on a physical GPU |
| `aleph-sv` / CPU kernels or noise | CPU tests/oracles plus CLI/Python and any transitive GPU/reference consumers; relevant Linux/macOS and toolchain configurations | CPU edits cannot break another backend's oracle or a public frontend |
| `aleph-core`, `aleph-ir`, `aleph-backend` | Broad consumer closure, normally approaching the full matrix | Shared interfaces are isolated to their own unit tests |
| `aleph-qec` | QEC plus affected Python/CUDA consumers and RTL golden-model coverage | RTL depends only on files directly inside `hw/` |
| Parser or CLI | Parser/CLI plus affected frontend/IR consumers and public-route contracts, according to the graph | Source parsing or output semantics are backend-independent in their tests |
| Python wrappers/scripts | Relevant wheel/binding/integration matrix and explicitly mapped script contracts | A Python-only diff cannot affect CUDA-Q/public workflow behavior |
| RTL/golden fixtures/generators | Hardware co-simulation plus affected Rust/generator/consumer checks | Generated data are inert documentation |
| Unknown/mixed/common configuration | Union of all affected scopes; full fallback when uncertain | Pick just the largest edited folder or the first matching rule |

These examples define intended reasoning, not a frozen hand-maintained substitute for graph tests. The closure may legitimately be broad because current dev/oracle dependencies connect packages. Prefer a correct broad result to an impressive but unsafe reduction in job count.

### Execution and merge behavior

- Emit a versioned, deterministic machine-readable plan with changed files, source/base identities, selected configurations and a reason for every selected/omitted suite. The job summary must make the plan reviewable.
- Build reusable job/configuration groups so selecting one backend does not run the entire workspace inside a supposedly targeted job. Conversely, `cargo test -p ...` alone does not test reverse consumers: add their relevant builds/contracts explicitly.
- Stage A's always-evaluated aggregate remains the single selection authority. It must validate **every selected matrix entry**, not just one successful sibling. Missing entries, planner errors, malformed/false-string outputs, and unexpected skips/cancellations fail. Intentionally omitted configurations remain visibly unexecuted.
- Keep broad integration on code-bearing main pushes during initial rollout and provide a full default-branch periodic/manual audit. A main/nightly job is a backstop, not permission to omit a known affected consumer before merge. Docs-only main pushes still take A's fast route. Keep release CI and timed benchmark isolation unchanged.
- Provide a safe full-check override. An override may broaden coverage; PR text/labels/inputs must not let an untrusted author reduce the computed plan. Preserve the hosted/fork/self-hosted boundaries established by A.
- Actual Metal/CUDA device validation and performance campaigns remain distinct from compile-only/no-device checks. Route supported hardware checks where available; otherwise report the missing hardware evidence honestly. This task does not provision GPUs or certify performance from a build.

### Validation and rollout

1. Run B in **shadow mode**: compare its selected sets against the existing full CI and replay representative historical diffs before removing jobs. Include shared core/IR, pure Metal, CUDA, QEC, CPU noise, frontend, Python, scripts/fixtures and configuration changes.
2. Add graph/classifier fixtures for normal/dev/build/optional/target edges, dependency cycles, consumers through shared oracle helpers, new/deleted/renamed packages and shader includes, lock/config changes and unknown inputs. Include deliberate broken-consumer examples proving the consumer is selected, not merely that the graph contains a direct edge.
3. Test scheduler/aggregate behavior on empty optional matrices, partial matrix failure, unexpected skip, classifier failure, cancelled superseded runs and a new PR head/base. Preserve A's docs tests throughout.
4. Validate at least a docs-only case and representative Metal, CUDA, CPU, QEC and shared-code selections. Use real applicable build/test runs where infrastructure supports them; explicitly separate routing-fixture proof from physical GPU execution. A deliberate syntax/contract regression in a selected consumer must fail its planned gate.
5. Compare several before/after code PRs with the same declared coverage boundary. Record queue time, elapsed critical path, summed runner time and self-hosted utilization separately. Report legitimate broad/no-savings cases; do not set a universal speedup target before measuring dependency overlap.
6. Document the package/configuration map, full override, adding a new crate/feature, failure diagnostics, rollback to full CI, and the updated exact-head merge-monitor contract. Require an accepted shadow-mode report before switching authoritative selection.

### Stage B acceptance criteria

- [ ] Dependency-derived package/consumer/configuration selection is implemented with audited non-Cargo edges and a fail-safe full fallback.
- [ ] Backend-local edits avoid unrelated expensive suites **where the verified dependency graph permits**, while shared-interface edits keep broad coverage.
- [ ] Feature-enabled frontend/reference contracts, cross-platform/toolchain and hardware-golden requirements cannot disappear through empty default builds or skipped matrix entries.
- [ ] Graph, security and aggregate failure cases pass; shadow-mode and representative actual selected builds/tests are documented.
- [ ] Measured queue/elapsed/runner-time evidence demonstrates the effect, including cases where selection remains broad.
- [ ] Merge instructions, future monitors, maintenance/override/rollback documentation are migrated deliberately; current PRs are never waived retroactively.

**Issue closure:** both A and B acceptance lists are satisfied in the same reviewed PR. Implement the docs path first, then validate dependency-aware code selection before merging the combined change.

## References

- Repository workflows: [.github/workflows/ci.yml](https://github.com/aleph-sim/aleph/blob/d02d8b51dfb3cb6c1d21bff94e81f3ac692e1922/.github/workflows/ci.yml), [hw.yml](https://github.com/aleph-sim/aleph/blob/d02d8b51dfb3cb6c1d21bff94e81f3ac692e1922/.github/workflows/hw.yml), [bench.yml](https://github.com/aleph-sim/aleph/blob/d02d8b51dfb3cb6c1d21bff94e81f3ac692e1922/.github/workflows/bench.yml).
- [GitHub workflow filtering/diff limits](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#git-diff-comparisons): do not assume native path filters inspect an unlimited diff.
- [GitHub required checks / skipped jobs / event semantics](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).
- Phase 8 tracker [#550](https://github.com/aleph-sim/aleph/issues/550), report PR [#596](https://github.com/aleph-sim/aleph/pull/596). This is cross-project CI infrastructure, not an extra capability counted in Phase 8's existing 16-task exit ledger.


-----

# Appendix: Counts and Order

- **Phase 0**: 12 issues
- **Phase 1**: 14 issues
- **Phase 2**: 5 issues
- **Phase 3**: 7 issues
- **Phase 4**: 8 issues
- **Phase 5**: 8 issues
- **Phase 6**: 6 issues

**Original Phases 0–6 baseline: 60 issues.** Later sub-phase additions are specified above.

- **Phase 7**: 7 issues (one integration/report tracker + six capability issues).
- **Phase 8**: 17 issues (one integration/report tracker + sixteen capability issues).

Recommended ordering for first weeks:

1. P0-01 → P0-02 → P0-03 (foundation in place)
1. P0-04 → P0-05 (testing & benchmark harnesses)
1. P0-06 → P0-07 → P0-09 (core types and naive backend)
1. P0-08 → P0-10 → P0-11 → P0-12 (parser, oracle, primitives, CLI)
1. Phase 1 in roughly the order listed
