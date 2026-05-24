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
    pub num_qubits: u32,
    pub num_clbits: u32,
    instructions: Vec<Instruction>,
    metadata: CircuitMetadata,
}
```

`instructions` is **private** so a future refactor to a DAG-style representation (Phase 1+ optimization passes) doesn't break public callers. Access is via `instructions()` (slice) and `layers()` (groups).

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
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CircuitError {
    #[error("qubit {qubit} out of range (circuit has {num_qubits} qubits)")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },
    #[error("clbit {clbit} out of range (circuit has {num_clbits} clbits)")]
    ClbitOutOfRange { clbit: u32, num_clbits: u32 },
    #[error("duplicate qubit index {qubit} in instruction")]
    DuplicateQubit { qubit: u32 },
    #[error("gate {gate} has arity {expected} but {got} qubits supplied")]
    ArityMismatch { gate: &'static str, expected: usize, got: usize },
}
```

`gate: &'static str` (not `Gate`) so the error stays `Clone + Copy`-friendly and the variant name appears in `Display` output without `Debug`-formatting the entire payload.

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
