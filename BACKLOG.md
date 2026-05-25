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

- [ ] `cargo build --workspace` succeeds
- [ ] `cargo test --workspace` succeeds (no tests yet, but exits 0)
- [ ] `cargo clippy --workspace -- -D warnings` succeeds
- [ ] `cargo fmt --check` succeeds
- [ ] README.md with build instructions exists

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

- [ ] CI runs on every PR and main push
- [ ] Build, test, clippy, fmt all gating
- [ ] Linux + macOS matrix
- [ ] Stable Rust required; beta allowed to fail
- [ ] Benchmark workflow exists (may be no-op until P0-04)

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

- [ ] Decision documented in `docs/decisions/0001-complex-type.md` (ADR format)
- [ ] Type aliased as `aleph_core::Complex` for forward compatibility
- [ ] All current usage routed through this alias

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

- [ ] `cargo bench` produces output
- [ ] At least 4 benchmark fixtures wired up
- [ ] Documentation in `docs/benchmarking.md`
- [ ] CI runs benchmarks on PR (may not gate, just report)

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

- [ ] `proptest` integrated, at least 4 generators
- [ ] At least 4 invariant tests passing
- [ ] Tests run as part of `cargo test`
- [ ] Documentation in `docs/testing.md`

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

- [ ] `Gate` enum covers all gates in Tier 1 algorithms
- [ ] `GateInstance` carries qubit indices
- [ ] Each gate has a method `matrix() -> SmallMatrix<Complex>` for naive use
- [ ] Each gate has `is_diagonal()`, `is_clifford()`, `inverse()` methods
- [ ] Unit tests for matrix correctness

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

- [ ] `Circuit` builder API: `circuit.h(0); circuit.cnot(0, 1); circuit.measure(0, 0);`
- [ ] Iteration API: `circuit.instructions()`
- [ ] Layer extraction: `circuit.layers()` returns groups of commuting instructions
- [ ] Serialization to/from OpenQASM 3.0 string (depends on P0-08)

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

- [ ] Parse the Tier 1 algorithm OpenQASM files (GHZ, QFT, Grover, random circuit)
- [ ] Produce equivalent `Circuit` IR
- [ ] Round-trip: Circuit → OpenQASM → Circuit produces equivalent result
- [ ] Helpful error messages with line/column info

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

- [ ] Runs all Tier 1 algorithms (GHZ, QFT, Grover, random) up to 20 qubits
- [ ] Produces correct results vs. Qiskit oracle on all benchmarks
- [ ] All property tests pass
- [ ] Code is readable and commented — this is the reference

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
Recommendation: pre-generate fixtures, regenerate via `make regen-fixtures`.

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

- [ ] All four primitives implemented for naive backend
- [ ] Sampling distribution converges to |ψ|² (statistical test with 1M shots)
- [ ] Expectation value tests vs. analytical results for known states

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

- [ ] All commands listed work
- [ ] Help text auto-generated and readable
- [ ] Exit codes: 0 success, non-zero on error
- [ ] Documented in README

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

- [ ] SoA backend produces identical results to naive backend (≤1e-12 difference)
- [ ] Benchmark: SoA vs. naive on QFT-20 — expect ~1.5–2× improvement just from cache effects
- [ ] All Phase 0 tests pass against SoA backend

**Testing Requirements**

- Equivalence test vs. naive backend on full Tier 1.
- All property tests against SoA.

**References**

- <https://en.wikipedia.org/wiki/AoS_and_SoA>
- <https://github.com/QuEST-Kit/QuEST> (uses SoA)

-----

### [P1-02] Bit-manipulation indexing for 1-qubit gate application

**Labels:** `area:backend-sv`, `type:optimization`, `priority:critical`
**Milestone:** Phase 1
**Estimate:** M
**Depends on:** P1-01

**Description**
Replace naive index loops with bit-twiddling that iterates state vector pairs in cache-friendly order.

**Context**
For a 1q gate on qubit q, every pair of amplitudes (i, i ⊕ 2^q) where bit q is 0 must be processed. Naive iteration is `for i in 0..N { if bit q of i is 0 { process(i, i^mask) } }` — branch-heavy and cache-unfriendly. The right pattern iterates blocks.

**Technical Details**
Pattern:

```rust
let mask = 1usize << q;
let lo_mask = mask - 1;
let hi_mask = !((mask << 1) - 1);
for k in 0..(n_amps >> 1) {
    let i0 = ((k & hi_mask) << 1) | (k & lo_mask);
    let i1 = i0 | mask;
    // apply 2x2 gate to (state[i0], state[i1])
}
```

This visits pairs in a deterministic, branch-free order. Combined with SoA, it’s vectorizable.

**Acceptance Criteria**

- [ ] All 1q gates implemented with this pattern
- [ ] Benchmark: 2–3× improvement over P1-01 on QFT-20
- [ ] All correctness tests pass

**Testing Requirements**

- Equivalence vs. naive backend.

**References**

- <https://github.com/QuEST-Kit/QuEST/blob/master/QuEST/src/CPU/QuEST_cpu.c>

-----

### [P1-03] SIMD (AVX2) for 1-qubit gates

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 1
**Estimate:** L
**Depends on:** P1-02

**Description**
Hand-write AVX2 intrinsics for hot 1q-gate kernels.

**Context**
AVX2 gives 4 f64 lanes; for SoA state vector this means processing 4 amplitude pairs per instruction. Compiler auto-vectorization sometimes works but is unreliable for our patterns.

**Technical Details**

- Use `std::arch::x86_64` intrinsics with runtime CPU feature detection.
- Provide scalar fallback when AVX2 unavailable.
- Implement: generic 2×2 unitary, Pauli-X (swap), Pauli-Z (negate half), Hadamard, diagonal.
- Process 4 pairs at a time (= 8 f64 in `re`, 8 f64 in `im`).

**Acceptance Criteria**

- [ ] AVX2 kernels for at least 5 gate types
- [ ] Runtime feature detection works
- [ ] Scalar fallback identical results
- [ ] Benchmark: 2–4× improvement over P1-02 on AVX2-capable hardware

**Testing Requirements**

- Equivalence test: AVX2 vs. scalar on randomized inputs.
- Property tests against AVX2 backend.

**References**

- <https://software.intel.com/sites/landingpage/IntrinsicsGuide/>
- <https://doc.rust-lang.org/std/arch/index.html>

-----

### [P1-04] SIMD (AVX-512) for 1-qubit gates

**Labels:** `area:backend-sv`, `type:optimization`, `priority:medium`
**Milestone:** Phase 1
**Estimate:** M
**Depends on:** P1-03

**Description**
Extend SIMD kernels to AVX-512 (8 f64 lanes).

**Context**
On AVX-512 hardware (Ice Lake+, Zen 4+), this doubles throughput over AVX2. Detected at runtime; falls back gracefully.

**Technical Details**
Same patterns as AVX2 but with 512-bit registers (`__m512d`). Note: some older Intel chips downclock under AVX-512; benchmark on real hardware before promoting it as default.

**Acceptance Criteria**

- [ ] AVX-512 kernels for hot gate types
- [ ] Runtime feature detection
- [ ] Benchmark: improvement over AVX2 on AVX-512 hardware
- [ ] No regression on AVX2-only hardware

**Testing Requirements**

- Equivalence test on multiple CPU types in CI matrix.

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

- [ ] X, Y, Z specialized kernels
- [ ] Benchmark: 3–10× speedup over generic 1q kernel for these gates
- [ ] Correctness preserved

**Testing Requirements**

- Equivalence vs. generic kernel.

**References**

- N/A — straightforward optimization.

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

- [ ] Generic 2q kernel
- [ ] Specialized CNOT, CZ, SWAP
- [ ] SIMD versions of each
- [ ] Benchmark: CNOT 5–10× faster than generic 2q kernel

**Testing Requirements**

- Equivalence vs. naive for each gate.

**References**

- <https://arxiv.org/abs/1601.07195> (“Quantum Supremacy” simulation paper has discussion)

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
For CCX on (c1, c2, t): for amplitudes where both control bits = 1, swap pairs at target. Single pass.

For arbitrary MCX with k controls: amplitudes where all k controls = 1 are affected; iterate with a mask check.

**Acceptance Criteria**

- [ ] CCX, CCZ specialized
- [ ] Generic MCX with up to 8 controls
- [ ] Benchmark: 3–5× faster than decomposed equivalent

**Testing Requirements**

- Equivalence vs. decomposed (CCX = 6 CNOTs + T gates).

**References**

- Nielsen & Chuang, Section 4.3.

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

- [ ] Fusion pass implemented
- [ ] On VQE ansatz: ≥3× reduction in gate count
- [ ] Result of fused circuit = result of unfused (to ≤1e-12)
- [ ] Pass is opt-in via `Circuit.optimize()` or similar

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

- [ ] Report committed
- [ ] All Tier 1 algorithms benchmarked
- [ ] Targets met or specific follow-up issues filed for misses

**Testing Requirements**

- Benchmark CI runs the report’s measurements.

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

- [ ] Audit complete, padding applied where needed
- [ ] No false-sharing patterns identified by perf tools
- [ ] Scaling efficiency improves vs. P2-01

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

- [ ] NUMA-aware build option
- [ ] Benchmark on 2-socket machine: improvement over default allocator
- [ ] Documentation on enabling

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

- [ ] Heuristic implemented as `select_backend(circuit) -> BackendKind`
- [ ] Manual override available
- [ ] Test corpus selects expected backend in each category

**Testing Requirements**

- Unit: each rule fires on the right input.

**References**

- N/A.

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

- [ ] QFT runs to 30 qubits on state vector
- [ ] Results match Qiskit
- [ ] Benchmark report row

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

- [ ] Cycles run to d = 11 (≈ 240 physical qubits)
- [ ] Match Stim output
- [ ] Benchmark report row

**Testing Requirements**

- Logical X / Z operator detection works as expected.

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