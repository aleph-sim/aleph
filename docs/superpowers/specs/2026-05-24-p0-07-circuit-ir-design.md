# P0-07 — Circuit IR data structure

**Status:** Approved (design phase)
**Date:** 2026-05-24
**Issue:** P0-07 (see `BACKLOG.md:340`)
**Depends on:** P0-06 (`Gate`, `GateInstance`).
**Blocks:** P0-08 (OpenQASM parser), P0-09 (naive state-vector backend).

---

## 1. Purpose

Introduce the backend-agnostic circuit intermediate representation that every backend consumes and every optimization pass transforms. After this issue:

- Tier 1 algorithms (GHZ, QFT, Grover, random circuit) can be expressed as `Circuit` values.
- P0-08 (OpenQASM parser) can lower into `Circuit`.
- P0-09 (naive state-vector backend) can iterate `Circuit::instructions()` and dispatch on `Instruction`.

This spec covers the type design, builder API, layer extraction, error model, and testing requirements. Implementation lands as a single PR titled `[P0-07] Circuit IR data structure`.

## 2. Scope

**In scope**

- New `aleph-ir` crate module structure (currently a stub).
- `Circuit`, `Instruction`, `CircuitMetadata`, `CircuitError` types.
- Builder methods covering the Tier 1 gate set (mini set + generic `add_gate`/`add_instruction`).
- `instructions()` accessor, `len()`/`is_empty()`, `metadata()`.
- `layers()` with the **disjoint OR shared-diagonal** commutation rule.
- Unit + `proptest` tests for builder validation, layer extraction invariants, Bell pair / GHZ construction.

**Out of scope**

- OpenQASM 3.0 serialization → P0-08.
- Optimization passes (gate fusion, cancellation, commutation) → P1+.
- Conditional gates / classical control flow → Phase 1+.
- Full pairwise commutation table → Phase 1+ (see BACKLOG §1082).
- DAG/graph representation → leave the door open by keeping `instructions: Vec<_>` private.

## 3. Decisions

Three design questions were resolved during brainstorming:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Layer extraction commutation scope | **Disjoint qubits OR both gates diagonal on shared qubit.** | Uses the existing `Gate::is_diagonal()` for nearly-free smarts; matches the spec phrase "groups of commuting instructions" without diving into a Phase-1 commutation table. |
| 2 | Builder error handling | **`Result<&mut Self, CircuitError>` per method.** | Stricter contract for a library API. Fluent chaining with `?` still works; tests use `.unwrap()`. |
| 3 | Builder method surface | **Mini set of Tier-1 convenience methods + generic `add_gate`/`add_instruction`.** | Spec example `circuit.h(0)` works; new `Gate` variants don't break the IR API. |

Sub-decisions resolved during design presentation:

- `with_name` is **consuming** (`mut self -> Self`) — set once at construction.
- `layers()` returns `Vec<Vec<usize>>` (indices into `instructions()`), not `Vec<Vec<&Instruction>>` — simpler lifetimes, lets backends index lazily.
- Both `add_gate(GateInstance)` and `add_instruction(Instruction)` exist — the former is convenience over the latter.

## 4. Types

### 4.1 `Circuit`

```rust
#[derive(Debug, Clone)]
pub struct Circuit {
    pub(crate) num_qubits: u32,
    pub(crate) num_clbits: u32,
    pub(crate) instructions: Vec<Instruction>,
    metadata: CircuitMetadata,
}
```

All non-metadata fields are **private**. Access is via `num_qubits()`/`num_clbits()` (getters), `instructions()` (slice), and `layers()` (groups). The field-private design future-proofs a DAG-style refactor (Phase 1+ optimization passes) and prevents the API-misuse panic that would arise from mutating `num_qubits` downward after construction (see § 12.2.3).

### 4.2 `Instruction`

```rust
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Apply a gate on concrete qubits.
    Gate(GateInstance),
    /// Mid-circuit or terminal measurement of `qubit` into `clbit`.
    Measure { qubit: u32, clbit: u32 },
    /// Reset `qubit` to `|0⟩` (mid-circuit reset).
    Reset(u32),
    /// Barrier forbidding optimization passes from crossing this point
    /// for the listed qubits. SmallVec inline 8 — barriers larger than
    /// that spill to heap.
    Barrier(SmallVec<[u32; 8]>),
}
```

Qubit ordering inside `Instruction::Gate(_)` follows the `GateInstance` convention from P0-06 (e.g. `[control, target]` for `Cnot`).

### 4.3 `CircuitMetadata`

```rust
#[derive(Debug, Clone, Default)]
pub struct CircuitMetadata {
    pub name: Option<String>,
    pub generated_from: Option<String>,
}
```

Kept tiny on purpose. New metadata fields are added as use cases arrive (e.g. P0-08 might populate `generated_from = Some("openqasm:3.0")`).

### 4.4 `CircuitError`

```rust
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CircuitError {
    #[error("qubit {qubit} out of range (circuit has {num_qubits} qubits)")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },
    #[error("clbit {clbit} out of range (circuit has {num_clbits} clbits)")]
    ClbitOutOfRange { clbit: u32, num_clbits: u32 },
    #[error("duplicate qubit index {qubit} in instruction")]
    DuplicateQubit { qubit: u32 },
    #[error("gate {gate} has arity {expected} but {got} qubits supplied")]
    ArityMismatch { gate: &'static str, expected: usize, got: usize },
    #[error("barrier must cover at least one qubit")]
    EmptyBarrier,
    #[error("gate {gate} has {controls} external controls but max is {max}")]
    TooManyControls { gate: &'static str, controls: usize, max: usize },
}
```

`gate: &'static str` (not `Gate`) so the error stays `Clone + Copy`-friendly and the variant name appears in `Display` output without `Debug`-formatting the entire payload. `EmptyBarrier` and `TooManyControls` were added in § 12.2/§ 12.3 — see those amendments for rationale.

## 5. Public API

### 5.1 Construction

```rust
impl Circuit {
    pub fn new(num_qubits: u32, num_clbits: u32) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn with_generated_from(self, source: impl Into<String>) -> Self;
}
```

Both `with_*` are **consuming** so they only appear at construction time:

```rust
let c = Circuit::new(2, 2).with_name("bell").with_generated_from("hand-coded");
```

### 5.2 Builder methods (mini set)

All return `Result<&mut Self, CircuitError>`. Listed by gate group; argument order follows the per-gate qubit convention from P0-06.

```rust
// 1q standard
h(q), x(q), y(q), z(q), s(q), sdg(q), t(q), tdg(q)

// 1q parametric
rx(theta, q), ry(theta, q), rz(theta, q), phase(theta, q),
u3(theta, phi, lambda, q)

// 2q (control, target / q0, q1)
cnot(c, t), cz(q0, q1), swap(q0, q1)

// 3q
ccx(c0, c1, target)         // alias of Toffoli

// Non-gate
measure(qubit, clbit), reset(qubit),
barrier(qubits: impl IntoIterator<Item = u32>)

// Generic escape hatches
add_gate(GateInstance), add_instruction(Instruction)
```

Gates outside this mini set (`Iswap`, `IswapDg`, `CRx`/`CRy`/`CRz`, `Ccz`, `Unitary1q`, `Unitary2q`) go through `add_gate(GateInstance::new(...))`.

### 5.3 Inspection

```rust
pub fn instructions(&self) -> &[Instruction];
pub fn len(&self) -> usize;
pub fn is_empty(&self) -> bool;
pub fn metadata(&self) -> &CircuitMetadata;
```

### 5.4 Layer extraction

```rust
pub fn layers(&self) -> Vec<Vec<usize>>;
```

Returns groups of indices into `instructions()`. Within each group, instructions can execute in parallel. See §6.

## 6. Layer extraction algorithm

**Rule:** two instructions `A` and `B` (with `A` appearing before `B` in `instructions()`) belong to the **same** layer if and only if:

- Their used-qubit sets are disjoint, **or**
- The intersection consists only of qubits where both `A` and `B` are `Instruction::Gate(g)` with `g.gate.is_diagonal() == true`.

**Used-qubit set** is the union of `gate.qubits` and `gate.controls` for `Gate` instructions; `{qubit}` for `Measure`/`Reset`; the contained `SmallVec` for `Barrier`.

**Classical bits:** A `Measure { qubit, clbit }` is also blocked from sharing a layer with any other `Measure` or `Reset` that touches the **same clbit** (classical write conflict). Phase 0 backends serialize classical ops anyway, so this restriction is conservative but correct.

**Algorithm:**

```
layers: Vec<Vec<usize>> = vec![];
// For each qubit, the (layer index, instruction index) of the most
// recent instruction that touched it.
last_for_qubit: HashMap<u32, (usize, usize)> = empty;
// For each clbit, just the layer index — clbit writes never commute.
last_for_clbit: HashMap<u32, usize> = empty;

for (i, inst) in instructions.iter().enumerate() {
    let qubits  = used_qubits(inst);    // SmallVec
    let clbits  = used_clbits(inst);    // SmallVec, empty for non-Measure

    // Earliest layer that does not violate any per-qubit/per-clbit
    // dependency. Start at 0 (parallel with everything) and bump
    // forward for each blocking predecessor.
    let mut earliest = 0usize;

    for q in &qubits {
        if let Some(&(prev_layer, prev_idx)) = last_for_qubit.get(q) {
            // Can the new instruction share `prev_layer` with the
            // prior instruction on this qubit? Only if both are
            // diagonal Gate instructions.
            if commute_on_qubit(&instructions[prev_idx], inst) {
                earliest = earliest.max(prev_layer);
            } else {
                earliest = earliest.max(prev_layer + 1);
            }
        }
    }
    for c in &clbits {
        if let Some(&prev_layer) = last_for_clbit.get(c) {
            earliest = earliest.max(prev_layer + 1);
        }
    }

    if earliest == layers.len() {
        layers.push(vec![i]);
    } else {
        layers[earliest].push(i);
    }

    for q in qubits { last_for_qubit.insert(q, (earliest, i)); }
    for c in clbits { last_for_clbit.insert(c, earliest); }
}

fn commute_on_qubit(a: &Instruction, b: &Instruction) -> bool {
    match (a, b) {
        (Instruction::Gate(ga), Instruction::Gate(gb)) =>
            ga.gate.is_diagonal() && gb.gate.is_diagonal(),
        // Measure/Reset/Barrier never commute with anything on the
        // shared qubit.
        _ => false,
    }
}
```

**Used-qubit set helpers:**

- `Gate(g)`: `g.qubits.iter().chain(g.controls.iter())`
- `Measure { qubit, .. }`: `[qubit]`
- `Reset(q)`: `[q]`
- `Barrier(qs)`: `qs.iter()`

**Used-clbit set:**

- `Measure { clbit, .. }`: `[clbit]`
- Others: empty.

**Complexity:** O(Σ arity(inst)) ≈ O(n · avg_arity), since each instruction does constant work per touched qubit/clbit (HashMap lookup + insert) and `commute_on_qubit` is O(1) per call. For Phase-0 gates, avg_arity ≤ 3 plus controls ≤ 2, so effectively O(n).

**Barriers and resets:** the `commute_on_qubit` returns `false` for any non-`Gate` instruction, so barriers and resets force a layer break on every touched qubit. This is the conservative-correct behaviour the spec wants.

## 7. Error handling

- Builder methods return `Result<&mut Self, CircuitError>` and never panic on user input.
- `add_gate` validates `qubits.len() == gate.arity()` and that every index is in range.
- `barrier` validates uniqueness of the supplied qubit list and that every index is in range.
- `measure` validates both qubit and clbit ranges.
- No `unwrap()`/`expect()` outside `#[cfg(test)]` per CLAUDE.md § Code Conventions.

## 8. Testing

### 8.1 Unit tests

- Construction: `Circuit::new(0, 0).is_empty()`, `Circuit::new(3, 0).num_qubits == 3`.
- Builder happy path: Bell pair (`h(0); cnot(0,1); measure; measure`), GHZ-3, QFT-2.
- Builder error path:
  - `h(5)` on 3-qubit circuit → `QubitOutOfRange`.
  - `measure(0, 5)` on 3-clbit circuit → `ClbitOutOfRange`.
  - `barrier([0, 0])` → `DuplicateQubit`.
  - `add_gate(GateInstance::new(Cnot, [0]))` → `ArityMismatch`.
- Metadata: `with_name("bell").metadata().name == Some("bell")`.
- Layer extraction:
  - Empty circuit → `[]`.
  - Single H → `[[0]]`.
  - Two H on disjoint qubits → `[[0, 1]]` (parallel).
  - H(0) then X(0) → `[[0], [1]]` (sequential, non-diagonal).
  - Z(0) then Phase(0.5, 0) → `[[0, 1]]` (parallel, both diagonal).
  - Cnot(0,1) then H(2) → `[[0, 1]]` (disjoint qubits).
  - Measure(0,0) then Measure(0,1) → `[[0], [1]]` (same qubit blocks).
  - Measure(0,0) then Measure(1,0) → `[[0], [1]]` (same clbit blocks).
  - Barrier([0,1]) blocks subsequent gates on those qubits.

### 8.2 Property tests (`proptest`)

1. **Layer flattening preserves order:** for any randomly-built circuit, flattening `layers()` in order and joining all indices reproduces `0..circuit.len()`.
2. **No conflicting pair in a layer:** for every pair of indices in the same layer, the conflict predicate is false.
3. **Bell pair invariant:** `Circuit::new(2,2).h(0)?.cnot(0,1)?.measure(0,0)?.measure(1,1)?` has exactly 4 instructions and 3 layers.

### 8.3 Out of scope

- Oracle tests against Qiskit's `DAGCircuit.layers()` — arrives with P0-09's Qiskit-Aer comparison.
- Performance benchmarks of layer extraction — arrives when a real workload (P0-09 random-circuit bench) exercises it.

## 9. File layout

```
crates/aleph-ir/
├── Cargo.toml                  # MODIFY: add aleph-core, smallvec, thiserror, proptest
└── src/
    ├── lib.rs                  # re-exports
    ├── circuit.rs              # Circuit + CircuitMetadata + builder methods
    ├── instruction.rs          # Instruction enum
    ├── error.rs                # CircuitError
    └── layers.rs               # layer extraction algorithm
```

`circuit.rs` will be the largest file; if it crosses ~600 lines during implementation, consider splitting the builder methods into a `circuit/builder.rs` submodule.

## 10. Acceptance criteria

Mirrors `BACKLOG.md:379`:

- [ ] `Circuit::new(num_qubits, num_clbits)` constructs an empty circuit.
- [ ] Builder API covers Tier-1 gates (`h`, `cnot`, `measure`, `rx`, etc.) returning `Result<&mut Self, CircuitError>`.
- [ ] `circuit.instructions()` iteration API exists.
- [ ] `circuit.layers()` returns groups of indices with the disjoint-OR-diagonal commutation rule.
- [ ] Unit tests cover happy path + every `CircuitError` variant.
- [ ] Property tests pass on default `proptest` budget for layer-extraction invariants.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] `cargo test --workspace` passes in both debug and release modes.

OpenQASM round-trip from the BACKLOG is **explicitly deferred** to P0-08.

## 11. Open Questions

None. Sub-decisions raised during design review (consuming `with_name`, indices in `layers()`, dual `add_gate`/`add_instruction`) are recorded in §3. If implementation surfaces a new decision, the spec gets a §12 amendment before code lands (same pattern as P0-06).

## 12. Amendments

### 12.1 Layer-extraction monotonicity (2026-05-24)

**Issue:** §6 pseudocode and §8.2 property #1 were internally inconsistent. §6 implemented ASAP scheduling — each instruction gets its earliest possible layer based only on per-qubit/per-clbit dependencies — but §8.2 property #1 ("flattening `layers()` in order reproduces `0..circuit.len()`") requires that the layer assignment be **non-decreasing** in instruction index. ASAP scheduling violates this on inputs like `cnot(0,1); h(0); h(2)` where `h(2)` backfills to layer 0 while `h(0)` sits in layer 1 → flattened order `[0, 2, 1]`.

**Resolution:** the algorithm enforces monotonicity by also bumping `earliest` to at least the previous instruction's assigned layer. Backfilling into the **current** open layer still happens (preserving the §8.1 unit-test case `cnot(0,1); h(2) → [[0, 1]]`), but no instruction can land in a layer strictly earlier than the one immediately before it.

```
let mut prev_assigned_layer = 0;
for (i, inst) in ... {
    let mut earliest = prev_assigned_layer;
    // ... (same per-qubit / per-clbit logic) ...
    // place at `earliest`
    prev_assigned_layer = earliest;
}
```

The property test in `tests/layers_properties.rs::layers_flatten_to_0_to_len` is the authoritative check. The §6 pseudocode is interpreted with this monotonicity constraint applied.

### 12.2 Post-merge code-review hardening (2026-05-24)

A multi-angle code review on the freshly-implemented branch surfaced four real correctness gaps and two coverage gaps. All are fixed in the same branch before merge.

**Correctness fixes:**

1. **Duplicate-qubit guard now release-safe.** `Circuit::validate_gate` now checks uniqueness across `gate.qubits ∪ gate.controls` in addition to arity and range. Previously, gates like `Cnot(0, 0)`, `Swap(0, 0)`, `Toffoli(0, 0, 0)`, or `controlled(X, [0], [0])` slipped through the IR in release builds — `GateInstance::new`'s own `check_qubit_uniqueness` is `#[cfg(debug_assertions)]`-gated and inert outside debug. `CircuitError::DuplicateQubit` is reused (its message is already generic). Tests: `add_gate_rejects_duplicate_qubit_cnot`, `add_gate_rejects_duplicate_qubit_toffoli`, `add_gate_rejects_qubit_control_overlap`, `add_gate_rejects_oob_control`.

2. **Empty `Barrier` rejected.** `Instruction::Barrier(empty)` was previously accepted and acted as a no-op for layer extraction, allowing disjoint gates on either side to share its layer — violating the documented "synchronization point" semantics. New error variant `CircuitError::EmptyBarrier`; `validate_instruction` rejects empty barriers up front. Test: `barrier_rejects_empty`.

3. **`num_qubits` / `num_clbits` fields encapsulated.** Previously `pub`, allowing user code to mutate them downward after instructions were added, which then panicked inside `extract_layers` via OOB indexing of `last_for_qubit`. Now `pub(crate)`, exposed through `num_qubits()` / `num_clbits()` getters. The counts are immutable for the `Circuit`'s lifetime.

4. **`Circuit::new` bounds `num_qubits` / `num_clbits`.** New `MAX_QUBITS = 65_535` / `MAX_CLBITS = 65_535` constants. `Circuit::new` `assert!`s on overflow, replacing the previous silent acceptance of `u32::MAX` (which triggered ~100 GB allocation in `extract_layers`). Untrusted callers (P0-08 parser, RPC handlers) should add a `try_new` fallible variant returning `CircuitError` — explicitly deferred to P0-08, where the parser is the actual untrusted-input boundary. Tests: `new_panics_on_too_many_qubits`, `new_panics_on_too_many_clbits`, `new_accepts_max_qubits`, `new_accepts_zero_qubits_zero_clbits`.

**Coverage fixes:**

5. **`add_gate`'s controls-OOB branch is now tested.** The `.chain(gate.controls.iter())` range-check path had no coverage; only the targets path was exercised. Added `add_gate_rejects_oob_control`.

6. **Proptest variant coverage broadened.** `arb_circuit` previously sampled only H/Z/S/X/CNOT; the spec §6 algorithm's distinguishing branches (clbit collision via `Measure`, `Reset`/`Barrier` non-commutation, `GateInstance::controlled` qubit union, parametric-diagonal commutation via `Rz`/`Phase`) were unreachable. Rebuilt around an `OpKind` enum that samples every Phase-0 `Instruction` variant including `Measure`/`Reset`/`Barrier`, parametric gates (`Rx`/`Ry`/`Rz`/`Phase`), 3-qubit (`Toffoli`/`Ccz`), and one `GateInstance::controlled` shape. Added a fourth property: `same_clbit_writes_serialize`, asserting that any two `Measure` instructions writing to the same clbit end up in different layers (the §6 clbit non-commutation rule, previously unverified by proptest).

**Design clarification:** the IR's `validate_gate` intentionally admits arbitrary external `controls` beyond what the base `Gate` semantically expects — `GateInstance::controlled` is a generic mechanism, and per-gate "is this a sensible control set?" decisions belong to backends. Added a doc comment on `validate_gate` to pin this policy. `ArityMismatch`'s message still references only `qubits.len()` vs `gate.arity()`; backends that mis-handle extra controls should fail loudly in their own layer.

**Refuted candidates** (kept for the record, not fixed):
- "`commute_on_qubit`'s `_ => false` catch-all silently widens on future `Instruction` variants" — by design; new variants get added with intent, and the default-conservative behavior is the right starting point for an unknown variant.
- "`pub(crate)` on `instructions` allows internal bypass of validation" — load-bearing for the in-crate inspection tests; any in-crate optimization pass that needs to splice instructions is expected to call `add_gate`/`add_instruction` and would be reviewed.
- "Spec/code drift in §12.1 monotonicity not flagged in algorithm doc-comment" — actually documented inline in `layers.rs`; see the `prev_assigned_layer` block comment.

### 12.3 Second-round review polish (2026-05-24)

A second `/code-review` pass on the § 12.2 fix-up commit surfaced one new correctness gap and six polish items. All fixed in the same branch.

**Correctness:**

1. **`validate_gate` bounds `gate.controls.len()` against [`MAX_GATE_CONTROLS`].** New `MAX_GATE_CONTROLS = 8` constant + new error variant `CircuitError::TooManyControls`. `GateInstance` has `pub` fields (`qubits`, `controls`), so a caller can construct an instance with an arbitrarily-large `controls` `SmallVec` directly via struct-literal syntax. The previous round added uniqueness-checking across `qubits ∪ controls` but used an O(N²) linear-scan `seen.contains(&q)`; an adversarial `controls.len() == 1_000_000` would hang the IR for minutes. Phase-0 gates use 0–2 controls (`aleph-core::GateInstance::controlled` inline buffer is `[u32; 2]`); a bound of 8 is generous. Tests: `add_gate_rejects_too_many_controls`, `add_gate_accepts_max_gate_controls`.

**Polish:**

2. **`MAX_QUBITS`/`MAX_CLBITS`/`MAX_GATE_CONTROLS` re-exported from `lib.rs`.** The panic message in `Circuit::new` references `MAX_QUBITS`, but the `circuit` module was private. Downstream code now imports via `aleph_ir::{MAX_QUBITS, MAX_CLBITS, MAX_GATE_CONTROLS}`.
3. **`Circuit::barrier` rustdoc now documents all three rejection paths** (`EmptyBarrier`, `DuplicateQubit`, `QubitOutOfRange`) and guides users with filter-built qubit lists on how to handle the empty case.
4. **`Instruction::Barrier` variant doc-comment** now states the "at least one qubit" requirement and points at `CircuitError::EmptyBarrier`.
5. **Within-controls duplicate now has a dedicated unit test** (`add_gate_rejects_duplicate_within_controls`) — the unified uniqueness loop already handled this case, but no test pinned the branch.
6. **Proptest `arb_op` is `nc=0`-safe.** Previously, `(0u32..nq, 0u32..nc).prop_map(...)` would panic on strategy construction if `nc == 0` (empty range). Measurement is now conditionally included only when `nc >= 1`, and its weight bumped from 1 (~6.7%) to 4 (~21%) so the `same_clbit_writes_serialize` property actually sees collisions within the default proptest budget. That property also switched from `nc=2` to `nc=1`, guaranteeing collisions when any two measurements appear.
7. **`every_op_kind_is_constructible` smoke test now uses `assert_eq!(c.len(), ops.len())`** instead of the previous `c.len() > before`, which would have passed even if only one of 20 ops survived. The unused `Param` import and its dead `let _ = Param::Concrete(0.0)` import-use guard line were removed.
8. **Stale `layers_properties.proptest-regressions` seed comment removed** — the seed previously pinned the original §6/§12.1 monotonicity bug (fixed in `dc1f34c`/`02380ce`) under the pre-`OpKind` strategy; its comment is now misleading. The file is kept (proptest will write new seeds on demand) but the stale entry is cleared.

**Deferred (documented as `TODO(P0-08)` in code):** the fallible `Circuit::try_new` returning `Result<Self, CircuitError>` for untrusted-input boundaries. The current `Circuit::new` panic on bounds violation is a documented programmer-error contract; the parser in P0-08 is the actual untrusted entry point and will add `try_new` then.
