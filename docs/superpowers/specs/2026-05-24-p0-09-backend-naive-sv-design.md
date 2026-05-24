# P0-09 — `Backend` trait and naive CPU state vector backend

**Issue:** P0-09 (see `BACKLOG.md`)
**Depends on:** P0-06 (`Gate`), P0-07 (`Circuit`, `GateInstance`), P0-08 (parser, for integration tests)
**Date:** 2026-05-24

---

## 1. Goal

Define the backend abstraction (`Backend` trait + `BackendError`) and ship the simplest possible correct CPU state vector implementation (`NaiveSvBackend`). This is the **reference** against which every future backend and every future optimization is checked. Simplicity, readability, and correctness dominate; performance does not.

---

## 2. Scope

### In scope

- 6-method `Backend` trait: `allocate`, `apply_gate`, `measure`, `sample`, `expectation_value`, `probabilities`.
- Shared concrete `BackendError` enum (no associated `type Error`).
- `pub fn run<B: Backend>(b, c) -> Result<B::State, BackendError>` helper in `aleph-backend`.
- `Pauli` enum + `PauliString` in `aleph-core` (so backends can share without depending on each other).
- `NaiveSvBackend` + opaque `CpuState` in `aleph-sv`, with single-threaded indexed gate application for 1q / 2q / 3q gates, including external `controls`.
- Per-backend RNG with explicit seed (`NaiveSvBackend::with_seed(u64)`) plus `::new()` entropy-seeded constructor.
- Tier-1 algorithms (GHZ, QFT, Grover, random Clifford+T) running through `parse → run` at **≥ 20 qubits**.

### Out of scope

- Multi-threading, rayon, SIMD (P0-11+).
- MPS / stabilizer / GPU backends (later phases).
- Qiskit oracle harness (P0-10).
- Shot-noise / observable optimization beyond the naive copy-state path (P0-11).
- Persistence / serialization of `CpuState` (not needed yet).
- Mid-circuit `Param::Symbolic` evaluation — naive backend rejects symbolic params.

---

## 3. Architecture

Three crates participate:

```
aleph-core    +  Pauli, PauliString
aleph-backend    Backend trait, BackendError, run<B>
aleph-sv         NaiveSvBackend (impl Backend), CpuState
```

`aleph-backend` depends on `aleph-core` and `aleph-ir`. `aleph-sv` depends on `aleph-backend`, `aleph-core`, `aleph-ir`, and `rand`. Backends never depend on each other.

Dataflow:

```
Circuit ──┐
          ▼
       run<B>(backend, circuit)
          │
          ├─ allocate(num_qubits)             → State
          ├─ for each GateInstance:
          │     apply_gate(&mut state, &gate)
          └─ return state
```

Measurement / sampling / expectation / probabilities are called by the *user* on the returned state, not by `run` itself.

---

## 4. Crate layout

```
crates/
  aleph-core/
    src/
      lib.rs       (re-exports)
      complex.rs   (existing)
      gate.rs      (existing, from P0-06)
      pauli.rs     (NEW — Pauli, PauliString)
  aleph-backend/
    src/
      lib.rs       (Backend trait, BackendError, run<B>)
  aleph-sv/
    src/
      lib.rs       (re-exports NaiveSvBackend, CpuState)
      backend.rs   (impl Backend for NaiveSvBackend)
      state.rs     (CpuState struct + getters)
      kernels.rs   (apply_1q, apply_2q, apply_3q)
      measure.rs   (measure, sample, expectation_value, probabilities)
    tests/
      tier1.rs     (GHZ, QFT, Grover, random — via parser)
    benches/
      naive_sv.rs  (criterion: H wall at n ∈ {10, 15, 20})
```

---

## 5. Dependencies

Workspace `[workspace.dependencies]` additions:

```toml
rand = "0.8"
```

`aleph-sv/Cargo.toml`: `rand.workspace = true`. Tests use the same `rand` for deterministic seeding.

No other new deps. `proptest`, `thiserror`, `num-complex`, `criterion` already in the workspace.

---

## 6. Public API

### 6.1 `aleph-core::pauli`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli { I, X, Y, Z }

impl Pauli {
    /// 2×2 matrix in basis |0⟩, |1⟩.
    pub fn matrix(self) -> [[Complex<f64>; 2]; 2];
}

#[derive(Debug, Clone, PartialEq)]
pub struct PauliString {
    pub coefficient: f64,
    /// (qubit, pauli) pairs. Qubits not listed are implicit identity.
    /// Sorted by qubit; no duplicates.
    pub terms: Vec<(u32, Pauli)>,
}

impl PauliString {
    pub fn new(coefficient: f64, terms: Vec<(u32, Pauli)>) -> Result<Self, PauliError>;
    pub fn identity(coefficient: f64) -> Self;
}

#[derive(Debug, thiserror::Error)]
pub enum PauliError {
    #[error("duplicate qubit {qubit} in Pauli string")]
    DuplicateQubit { qubit: u32 },
    #[error("non-finite coefficient")]
    NonFiniteCoefficient,
}
```

### 6.2 `aleph-backend`

```rust
pub trait Backend {
    type State;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError>;
    fn apply_gate(&mut self, state: &mut Self::State, gate: &GateInstance)
        -> Result<(), BackendError>;
    fn measure(&mut self, state: &mut Self::State, qubit: u32)
        -> Result<bool, BackendError>;
    fn sample(&mut self, state: &Self::State, shots: u32)
        -> Result<Vec<u64>, BackendError>;
    fn expectation_value(&mut self, state: &Self::State, pauli: &PauliString)
        -> Result<f64, BackendError>;
    fn probabilities(&mut self, state: &Self::State, qubits: &[u32])
        -> Result<Vec<f64>, BackendError>;
}

pub fn run<B: Backend>(backend: &mut B, circuit: &Circuit)
    -> Result<B::State, BackendError>;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BackendError {
    QubitOutOfRange { qubit: u32, num_qubits: u32 },
    DuplicateQubit { qubit: u32 },
    ArityMismatch { kind: &'static str, expected: usize, got: usize },
    UnsupportedGate { kind: &'static str },
    UnsupportedInstruction { kind: &'static str },
    SymbolicParam,
    NonFiniteParam { kind: &'static str },
    NonUnitaryMatrix { deviation: f64 },
    EmptyCircuit,
    DegenerateMeasurement { qubit: u32, probability: f64 },
    TooManyQubits { requested: u32, limit: u32 },
    InvalidPauliString { reason: &'static str },
    InvalidState { reason: &'static str },
}
```

(Variant list as of round-2 review; see § 11.1 and § 11.2 for the rationale behind each. `QubitCountMismatch` was removed as unreachable.)

`run` returns `EmptyCircuit` if the circuit has zero instructions AND zero qubits; a 0-instruction non-zero-qubit circuit returns an all-`|0…0⟩` state. (Rationale: `|0…0⟩` is a valid output; only the truly-empty case is an error.)

### 6.3 `aleph-sv`

```rust
pub struct NaiveSvBackend {
    rng: rand::rngs::StdRng,
}

impl NaiveSvBackend {
    pub fn new() -> Self;                      // entropy seed
    pub fn with_seed(seed: u64) -> Self;       // explicit seed
}

impl Default for NaiveSvBackend { fn default() -> Self { Self::new() } }

impl Backend for NaiveSvBackend {
    type State = CpuState;
    // ... (the six methods)
}

pub struct CpuState {
    num_qubits: u32,
    amps: Vec<Complex<f64>>,
}

impl CpuState {
    pub fn num_qubits(&self) -> u32;
    pub fn amplitudes(&self) -> &[Complex<f64>];
}
```

Soft cap: `allocate(num_qubits)` returns `BackendError::TooManyQubits { requested, limit: 28 }` for `num_qubits > 28`. Rationale: 2^28 × 16 bytes = 4 GiB; comfortable on a 16 GiB laptop. Acceptance target is 20 qubits.

---

## 7. Algorithms

### 7.1 Qubit ordering convention (from P0-06)

`qubits[0]` is the **MSB** of the matrix index. For a 2-qubit gate on `[a, b]`, basis order is `|a b⟩`:

```
matrix row/col 0 → a=0, b=0
matrix row/col 1 → a=0, b=1
matrix row/col 2 → a=1, b=0
matrix row/col 3 → a=1, b=1
```

This matches `Gate::Cnot` (`qubits = [control, target]`) whose matrix swaps rows 2 ↔ 3 — the control sits at the MSB position. The P0-06 docstring on `Gate::Unitary2q` is the source of truth for this convention.

Qubits in the *global state-vector* index follow the natural mapping: qubit `q` is at bit position `q` of the global amplitude index `i`. The MSB convention only affects how a gate's matrix rows/cols map onto a chosen set of target qubits, not how those qubits live in the global vector.

### 7.2 1q kernel (indexed-pair)

```text
apply_1q(amps, target, controls, m):
    t_bit    = 1 << target
    ctrl_msk = OR over (1 << c) for c in controls
    for i in 0 .. amps.len():
        if i & t_bit != 0:                  continue   // visit each pair once
        if (i & ctrl_msk) != ctrl_msk:      continue   // skip uncontrolled
        j = i | t_bit
        a = amps[i]; b = amps[j]
        amps[i] = m[0][0]*a + m[0][1]*b
        amps[j] = m[1][0]*a + m[1][1]*b
```

### 7.3 2q kernel (quadruplet)

Two target bits `t0, t1` define a 4-element subspace per outer iteration. **MSB convention:** matrix index `k` is `(bit_at_t0 << 1) | bit_at_t1` — i.e. `qubits[0]` lives at the high bit of `k`, `qubits[1]` at the low bit. For each base `i` with both target bits clear and all controls set, load `[amps[i], amps[i|t1_bit], amps[i|t0_bit], amps[i|t0_bit|t1_bit]]` (in that order so they correspond to `k = 0, 1, 2, 3`), multiply by 4×4, store back. Targets must be distinct (`DuplicateQubit`) and disjoint from controls.

### 7.4 3q kernel (octuplet)

Same shape, 8-element subspace, 8×8 matrix. **MSB convention:** matrix index `k`'s bits map to `(qubits[0], qubits[1], qubits[2])` from MSB to LSB — `qubits[0]` is bit 2 of `k`, `qubits[1]` is bit 1, `qubits[2]` is bit 0. Targets distinct; targets ∩ controls = ∅.

### 7.5 `apply_gate` dispatch

1. Validate every qubit in `gate.qubits` and `gate.controls` against `state.num_qubits` → `QubitOutOfRange`.
2. Reject duplicates within `qubits`, within `controls`, and across (`qubits ∩ controls`) → `DuplicateQubit`.
3. Reject `Param::Symbolic` → `SymbolicParam`.
4. Materialise matrix via `Gate::matrix()` (from P0-06's `GateMatrix::{M2x2, M4x4, M8x8}`).
5. Dispatch on arity:
   - `M2x2` → `apply_1q`
   - `M4x4` → `apply_2q`
   - `M8x8` → `apply_3q`
6. `Gate::Unitary1q { matrix }` / `Unitary2q { matrix }` carry their matrix directly; same dispatch.

External `controls` are handled uniformly inside the kernels — no separate "controlled" code path. `Cnot[a,b]` (intrinsic) and `X[b]` with `controls=[a]` (external) must produce identical states.

### 7.6 `measure(state, qubit) -> bool`

```text
p1 = Σ |amps[i]|² for i where ((i >> qubit) & 1) == 1
outcome = rng.gen::<f64>() < p1
p = if outcome { p1 } else { 1.0 - p1 }
if p < 1e-300: return Err(DegenerateMeasurement { qubit, probability: p })
norm = sqrt(p)
for i in 0 .. amps.len():
    if (((i >> qubit) & 1) == 1) == outcome:
        amps[i] /= norm
    else:
        amps[i]  = 0
return outcome
```

### 7.7 `sample(state, shots) -> Vec<u64>`

Build CDF once: `cdf[i] = Σ_{k ≤ i} |amps[k]|²`. For each shot, draw `u = rng.gen::<f64>()` and binary-search (`slice::partition_point`) the CDF for `u`. Returns basis indices `0 .. 2^n`. Cost: O(N + shots·log N). No state mutation.

### 7.8 `expectation_value(state, pauli) -> f64`

```text
tmp = state.amps.clone()
for (qubit, p) in pauli.terms where p != I:
    apply_1q(&mut tmp, qubit, &[], p.matrix())
ev = Re( Σᵢ conj(state.amps[i]) * tmp[i] )
return pauli.coefficient * ev
```

Naive (O(N · |terms|), allocates one copy). P0-11 will specialise the Pauli-Z-only case.

### 7.9 `probabilities(state, qubits) -> Vec<f64>`

Marginal over the named subset. Returns a vector of length `2^qubits.len()`. For each basis index `i ∈ 0 .. N`, gather the bits at positions `qubits[0], qubits[1], …` into key `k` (with `qubits[0]` as LSB to match the same convention), accumulate `|amps[i]|²` into `out[k]`. Empty `qubits` slice → `vec![1.0]`. Duplicate qubits → `DuplicateQubit`. Out-of-range → `QubitOutOfRange`.

### 7.10 `run<B>`

```text
if circuit.num_qubits == 0 and circuit.instructions.is_empty():
    return Err(EmptyCircuit)
state = backend.allocate(circuit.num_qubits)?
for gate in &circuit.instructions:
    backend.apply_gate(&mut state, gate)?
return Ok(state)
```

---

## 8. Testing

### 8.1 Unit (co-located `mod tests`)

- Each `Gate` variant applied to each computational basis state on the smallest sufficient qubit count; assert specific output amplitudes within `1e-12`.
- `apply_1q`, `apply_2q`, `apply_3q` correctness on contrived 2/3/4-qubit states.
- External controls: `Gate::Cnot[a, b]` ≡ `Gate::X[b]` with `controls = [a]` on random states.
- `measure` collapse: prepare `(|0⟩ + |1⟩)/√2`, measure, assert post-state is a basis state and matches outcome.
- `sample`: 10k shots on `(|00⟩ + |11⟩)/√2` with fixed seed, assert observed bit-correlation = 1.0 exactly and outcomes ⊂ {0, 3}.
- `expectation_value`: `⟨0|Z|0⟩ = 1`, `⟨+|X|+⟩ = 1`, `⟨0|X|0⟩ = 0`, `⟨−|Z|−⟩ = −1`.
- `probabilities`: marginals sum to 1; single-qubit marginal of `|+⟩` is `[0.5, 0.5]`.
- Every `BackendError` variant has a triggering test.

### 8.2 Property (`proptest`, `mod tests`)

Generators: H, X, Y, Z, S, Sdg, T, Tdg, CX, CZ, SWAP, Rx/Ry/Rz with arbitrary `f64` angles in `[-2π, 2π]`. State sizes `n ∈ 1..=8`.

- **Normalisation:** after any sequence of ≤ 50 generated gates, `(Σ|amp|²) − 1 ∈ (-1e-10, 1e-10)`.
- **Reversibility:** apply gate then its inverse on a random state → original within `1e-10`.
- **Involution:** `H·H`, `X·X`, `CX·CX`, `SWAP·SWAP` all reduce to identity within `1e-12` on random states.
- **Control equivalence:** for each intrinsic controlled gate that has a base form in `Gate` (`Cnot` ↔ `X`, `Cz` ↔ `Z`), the intrinsic form on `[a, b]` ≡ the base form on `[b]` with `controls=[a]` on random states.
- **Sampling consistency:** with fixed seed and 10k shots on a small circuit, empirical frequencies within 3σ of `probabilities(all qubits)`.

### 8.3 Integration (`crates/aleph-sv/tests/tier1.rs`)

Inputs are OpenQASM 3.0 strings parsed via `aleph_parser::parse` and run through `aleph_backend::run` with `NaiveSvBackend`.

- **GHZ-n** for `n ∈ {2, 3, 5, 10, 20}`: assert `|amps[0]|² ≈ |amps[2^n - 1]|² ≈ 0.5` within `1e-10`; all other `|amps|² < 1e-20`.
- **QFT-n** on `|1⟩` for `n ∈ {3, 5}`: compare against analytic Fourier amplitudes within `1e-10`.
- **Grover-3** (one marked state): assert `P(marked) > 0.78` after one iteration. (One-iteration Grover on 3 qubits with 1 marked state gives `P ≈ 0.781`.)
- **Random Clifford+T** on 8 qubits with fixed RNG seed: check normalisation only; determinism (same seed → same final state). Oracle deferred to P0-10.

### 8.4 Benchmarks (`crates/aleph-sv/benches/naive_sv.rs`)

One criterion benchmark: H wall (apply H to every qubit) on `n ∈ {10, 15, 20}`. Establishes the baseline P0-11 will measure against. No regression gate yet.

---

## 9. Acceptance criteria

- [ ] `aleph-core` exports `Pauli`, `PauliString`, `PauliError`.
- [ ] `aleph-backend` exports `Backend`, `BackendError`, `run`.
- [ ] `aleph-sv` exports `NaiveSvBackend`, `CpuState`.
- [ ] All six trait methods implemented and tested.
- [ ] Tier-1 algorithms (GHZ-20, QFT-5, Grover-3, random-8) run end-to-end via `parse → run` and pass their integration assertions.
- [ ] Property suite passes (normalisation, reversibility, involution, control equivalence, sampling).
- [ ] `cargo test --workspace` green.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] `cargo bench --bench naive_sv` produces a baseline (no perf gate this PR).
- [ ] `unsafe` count remains zero in `aleph-sv` / `aleph-backend`.

---

## 10. Open questions

None at the time of writing. Likely amendments (record in § 11 if they happen):

- Whether to add an `expectation_value` fast path for Pauli-Z-only strings before P0-11 lands. Default: no, this is the naive backend.
- Whether `run` should accept `&mut Circuit` to allow in-place IR rewrites. Default: no, `&Circuit` is enough.

---

## 11. Amendments

### 11.1 Code-review pass — error-model and validation hardening

Records all 10 findings from the `/code-review` pass on the initial implementation. Every one was CONFIRMED or PLAUSIBLE; all are now fixed on the branch.

**Error enum changes:**
- **Added** `BackendError::ArityMismatch { kind, expected, got }` — apply_gate now validates `gate.qubits.len() == gate.gate.arity()` before kernel dispatch (was an index-out-of-bounds panic).
- **Added** `BackendError::NonFiniteParam { kind: &'static str }` — `Gate::matrix()` returns both `GateError::SymbolicParam` and `GateError::NonFiniteParam`; the dispatch now routes them to distinct `BackendError` variants instead of collapsing both into `SymbolicParam`.
- **Added** `BackendError::UnsupportedInstruction { kind }` — distinguishes a non-Gate IR instruction (`Reset`) from a non-supported gate. `run<B>` now uses this for `Reset` instead of misnaming it `UnsupportedGate`.
- **Added** `BackendError::NonUnitaryMatrix { deviation }` — `apply_gate` now rejects user-supplied `Gate::Unitary1q` / `Gate::Unitary2q` matrices whose `‖U·U† − I‖_max > AMPLITUDE_TOL`. Intrinsic gates skip the check (unitary by construction).
- **Added** `BackendError::InvalidPauliString { reason }` — surfaces invariant violations in callers that bypass `PauliString::new` via the public fields.
- **Removed** `BackendError::QubitCountMismatch` — unreachable through the current API (run always passes `circuit.num_qubits()` to `allocate`). Re-add when a future API exposes pre-allocated state to user-driven `apply_gate`.

**Validation hardening:**
- `expectation_value_impl` now revalidates `PauliString` invariants (finite coefficient, sorted/dedup terms, in-range qubits) at kernel entry. The `pub` fields made `::new`'s checks bypassable; trusting the invariants produced silently-wrong values for duplicate-qubit terms.
- `apply_gate` validates arity, qubit range, qubit duplication, parameter finiteness, and (for user matrices) unitarity — in that order.

**Measurement correctness and reproducibility:**
- `measure_impl` clamps `p1` to `[0, 1]` after the FP sum to absorb drift; without this, `1.0 − p1` could be slightly negative and `p.sqrt()` would return NaN, silently poisoning the state. The threshold check (`p < 1e-300`) is now applied to both branches before consuming RNG state: degenerate cases either error out (both branches tiny) or are decided deterministically (one branch tiny → outcome forced). RNG is only consumed when the choice is genuinely random, preserving `with_seed` reproducibility on highly polarized states and error paths.

**New driver:**
- `run_with_outcomes<B>(backend, circuit) -> (B::State, Vec<MeasurementRecord>)` returns the per-Measure-instruction `(instruction_index, qubit, clbit, outcome)`. `run<B>` is now a thin wrapper that discards outcomes. P0-10 oracle comparison against shot-based Qiskit Aer references needs the recorded outcomes.

**Test coverage:**
- Reversibility property tests added for `Ry`, `Rz`, `S·Sdg`, `T·Tdg`, `Iswap·IswapDg`, plus involution tests for `X²`, `Y²`, `Z²`, `Swap²`. Previously only `Rx` had a generic adjoint-reversibility proptest.
- Regression tests added for every new `BackendError` variant.

**Refuted candidates:** None — all 10 candidates survived verification.

**Deferred:**
- The QFT-3 integration test still checks only magnitudes (`|a|² = 1/8`), not phase relations. Phase-sensitive QFT correctness will be covered by P0-10's Qiskit oracle harness.
- `Gate::Unitary1q/2q` unitarity check happens at *apply* time, not *construction* time. P0-06 documented "no unitarity check at construction"; adding one to the constructor would be a P0-06 amendment, out of scope here.

**API additions** (also recorded in `aleph_core`):
- `Gate::name(&self) -> &'static str` — stable variant-name strings used by `BackendError::{ArityMismatch, NonFiniteParam, UnsupportedGate}` messages.

**Dependency hygiene:**
- Dropped unused `num-complex` and `thiserror` declarations from `aleph-sv/Cargo.toml`.

### 11.2 Second-pass review — NaN containment and defense-in-depth

The first amendment's fixes shipped two regressions and missed several adjacent gaps. Round-2 review surfaced 10 candidates; all are now fixed.

**HIGH — NaN containment regressions in round 1:**
- `measure_impl`'s `p1.clamp(0.0, 1.0)` propagates NaN unchanged. A state with any NaN amplitude still poisoned the entire vector. Fix: explicit `if !p1.is_finite()` guard returning `BackendError::InvalidState { reason: "non-finite amplitude norm²" }` before the clamp.
- `unitarity_deviation` tracked the worst element with `if dev > worst` (and then briefly `worst.max(dev)`). Both swallow NaN — `>` because NaN comparisons return false, `f64::max` per IEEE-754-2008 minNum/maxNum. A `Gate::Unitary1q` with NaN entries passed the check at zero deviation. Fix: explicit `if dev.is_nan() { return f64::NAN }` early-return inside `max_dev`.

**MEDIUM — hardening propagation:**
- `sample_impl` now mirrors `measure_impl`'s defenses: rejects empty state, rejects non-finite per-amp `|a|²`, rejects total norm² outside a √n·`AMPLITUDE_TOL` drift budget.
- The unitarity check now runs **unconditionally** on every dispatched matrix (`Gate::Unitary1q/2q` *and* intrinsic gates). Cost is constant per gate (≤ 8×8 multiply) and negligible against the kernel itself. Catches pathological cases like `Gate::Rx(1e18)` where argument-reduction precision loss leaves `cos²+sin²` measurably below 1.

**MEDIUM — visibility / API hygiene:**
- `aleph_sv::kernels::apply_{1,2,3}q` are now `pub(crate)`. They were `pub fn`, which meant external crates could bypass `apply_gate`'s arity/bounds/duplicate guards.

**LOW — doc and test drift:**
- `run<B>` doc comment now references `BackendError::UnsupportedInstruction` (not the old `UnsupportedGate`) for `Reset` and cross-references `run_with_outcomes`.
- `aleph_core::Gate::name()` has a pinning test asserting each `(variant, string)` pair so a future rename is a deliberate edit, not a silent drift.
- `aleph-ir`'s local `gate_variant_name` now delegates to `Gate::name()` instead of duplicating the table.
- A Tier-1 integration test exercises `run_with_outcomes` end-to-end: parses a Bell-pair-with-measurements circuit, asserts the two `MeasurementRecord`s are in instruction order and that outcomes are perfectly correlated.

**Spec self-consistency:**
- § 6.2's `BackendError` code block was still listing the removed `QubitCountMismatch` — now updated to match the post-§-11.1 enum surface.

**New error variant introduced this round:**
- `BackendError::InvalidState { reason: &'static str }` — surfaces upstream state corruption (NaN amplitudes, empty state, norm² off by more than the drift budget) at the measurement / sampling boundary.

**Refuted candidates:** None — all 10 candidates survived verification.

**Deferred (out of scope for P0-09):**
- `Gate::inverse()` does not validate parameter finiteness; calling `Rx(NaN).inverse()` returns `Rx(NaN)` without error. A future P0-06 amendment can add the check at the IR layer.
- `BackendError` does not carry `#[non_exhaustive]`. Adding it now would force downstream `match` arms to add `_` catchalls before P0-10's CLI lands. Revisit when the error surface settles.

