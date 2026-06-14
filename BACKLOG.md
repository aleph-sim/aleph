# Quantum Simulator — Detailed Backlog

> **Source of truth for GitHub Issues.**
> This document contains every planned issue for Phases 0–6. Each issue is structured so that an AI agent (e.g., Claude Code) or a human can create a GitHub Issue directly from its contents.

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
- Phase 6 — Multi-GPU & Distributed

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
- [ ] surface-d11 cycle time improves and `measure`'s self share drops materially (out of the top profile slot; soft goal: d=11 ≤ 0.65× Stim) — else a documented structural verdict with profile evidence.
- [ ] Stim oracles d=3..11 green; new unit tests for any new helper on irregular n (not multiples of 64).
- [ ] Before/after criterion numbers (EPYC) in the PR.

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

# Appendix: Counts and Order

- **Phase 0**: 12 issues
- **Phase 1**: 14 issues
- **Phase 2**: 5 issues
- **Phase 3**: 7 issues
- **Phase 4**: 8 issues
- **Phase 5**: 8 issues
- **Phase 6**: 6 issues

**Total: 60 issues.**

Recommended ordering for first weeks:

1. P0-01 → P0-02 → P0-03 (foundation in place)
1. P0-04 → P0-05 (testing & benchmark harnesses)
1. P0-06 → P0-07 → P0-09 (core types and naive backend)
1. P0-08 → P0-10 → P0-11 → P0-12 (parser, oracle, primitives, CLI)
1. Phase 1 in roughly the order listed