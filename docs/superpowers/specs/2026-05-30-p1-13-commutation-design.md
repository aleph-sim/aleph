# [P1-13] Commutation analysis (foundational) — design

> Status: approved (brainstorm) — pending implementation plan.
> Issue: BACKLOG `[P1-13] Commutation analysis (foundational)` (#25).
> Depends on: P0-07 (Circuit IR), P1-09 (`passes` module).

## 1. Goal

Provide a foundational primitive that answers "do these two gates
commute?" so later passes (more aggressive cancellation, commutation-
aware fusion) can reorder gates safely. This ticket ships the primitive
and its tests only — no consuming pass, no `default_pipeline` change.

```rust
pub fn gates_commute(a: &GateInstance, b: &GateInstance) -> bool
```

Acceptance criteria (BACKLOG #25):

- [ ] Commutation table covers standard gates.
- [ ] API exposed for other passes.
- [ ] Unit tests for all entries in the table.

Testing requirements (BACKLOG):

- For each commuting pair: applying in either order produces the same state.
- For each non-commuting pair: applying in either order produces different states (sanity).

## 2. Architecture

- New module `crates/aleph-ir/src/passes/commute.rs`; `pub fn
  gates_commute(a: &GateInstance, b: &GateInstance) -> bool`.
- Re-export `pub use commute::gates_commute;` from
  `crates/aleph-ir/src/passes/mod.rs`. It is **not** a `Pass` and is
  **not** added to `default_pipeline()`.
- **No `aleph-core` changes.** Leans on existing `Gate::is_diagonal()`,
  `Gate::arity()`, and the `GateInstance { gate, qubits, controls }`
  fields, plus `Instruction`-free gate-vs-gate reasoning.
- API scope is gate-vs-gate only (per BACKLOG). Measure/Reset/Barrier
  are the consuming pass's concern (they are fences); `gates_commute`
  takes two `&GateInstance`.

## 3. Soundness invariant

`gates_commute(a, b) == true` **must be sound**: the two operations
genuinely commute (`A·B == B·A` as operators on the full Hilbert
space), so a consumer may swap their order without changing the
circuit's action. When unsure, return **`false`** — a false negative
only forgoes an optimisation; a false positive would let a consumer
reorder gates and silently corrupt the state. The function is
**symmetric**: `gates_commute(a, b) == gates_commute(b, a)`.

## 4. Rules (first match wins; otherwise `false`)

Let `supp(g) = used_qubits` = targets ∪ external controls.

1. **Disjoint support → `true`.** `supp(a) ∩ supp(b) == ∅`. Operators
   on disjoint qubits always commute.
2. **Both diagonal → `true`.** `a.gate.is_diagonal() &&
   b.gate.is_diagonal()`. Diagonal matrices in the computational basis
   always commute, and a controlled-diagonal gate is still diagonal, so
   this holds on any qubits and with any external controls. Covers
   `Z, S, Sdg, T, Tdg, Rz, Phase, Cz, Ccz, CRz, Unitary1qDiag`
   mutually (e.g. `Rz·Z`, `S·T`, `Cz·Rz`).
3. **Structurally identical → `true`.** Same `gate` (by `PartialEq`),
   same `qubits` (positional), same `controls` (as a set). An operator
   commutes with itself (`X·X`, `H·H`, `Cnot(0,1)·Cnot(0,1)`,
   identically-controlled gates).
4. **CNOT control/target relations** — only when **neither** instance
   has external `controls` (both bare). For `Cnot(c, t)` (i.e.
   `gate == Cnot`, `qubits == [c, t]`) against a single-qubit gate
   `G(q)` (`G.arity() == 1`, no controls):
   - `q == t` and `G` commutes with `X` — i.e. `G ∈ {X, Rx(_)}` →
     `true` (X and Rx are functions of `I` and `X`, which pass through
     CNOT on the target).
   - `q == c` and `G.is_diagonal()` (`Z, Rz, S, Sdg, T, Tdg, Phase`) →
     `true` (CNOT is block-diagonal in the control's Z-eigenbasis).
   Evaluated for both argument orders.
5. **Otherwise → `false`.**

**Deliberately conservative `false` (deferred to future work):**

- `Cnot·Cnot` with partial overlap (shared control-only or shared
  target-only commute; control-of-one == target-of-other does not).
- Different non-diagonal 1q gates on the same qubit (`X·H`, `H·Y`).
- `Y`/`Ry` on a CNOT target (they anticommute with the relevant Pauli
  in general).
- Externally-controlled gates beyond rules 1–3.

These only forgo optimisations; none is a correctness risk.

## 5. Testing

- **Unit** (co-located in `commute.rs`) — one test per rule/entry
  (AC: "unit tests for all entries"):
  - Disjoint: `H(0) ∥ X(1)` → true; `Cnot(0,1) ∥ Z(2)` → true.
  - Both-diagonal: `Z·Rz`, `S·T`, `Phase·Cz`, `Rz·CRz`, `Cz·Ccz`,
    controlled-`Z` (ctrl-Z) ∥ `Rz` → true.
  - Identical: `H(0)·H(0)`, `Cnot(0,1)·Cnot(0,1)`, identically-
    controlled `X` → true.
  - CNOT/target: `Cnot(0,1) ∥ X(1)` → true; `∥ Rx(θ,1)` → true;
    `∥ Z(1)` → false; `∥ Y(1)` → false.
  - CNOT/control: `Cnot(0,1) ∥ Z(0)` → true; `∥ Rz(θ,0)` → true;
    `∥ X(0)` → false.
  - Non-commuting sanity: `X(0)·Z(0)` → false; `H(0)·X(0)` → false;
    `Cnot(0,1)·Cnot(1,2)` (control==target overlap) → false.
  - Symmetry: a loop asserting `gates_commute(a,b) ==
    gates_commute(b,a)` over all the above cases.
- **SV oracle** (`crates/aleph-sv/tests/commute_oracle.rs`, mirrors the
  P1-12 oracle) — the soundness guard:
  - For every pair the table reports `true`: build circuits `[a, b]`
    and `[b, a]`, assert identical full state vectors within `1e-12`.
  - For a sample of `false` pairs (`X·Z`, `H·X`, `Cnot ∥ Z(target)`):
    assert the two orderings produce **different** states (BACKLOG
    sanity).
  - **Proptest:** random `GateInstance` pairs on a small register; when
    `gates_commute(a, b)` is `true`, assert reorder preserves the state
    within `1e-12`. Strong false-positive guard. (Only the
    `commute ⟹ equal` direction is proptested; "different states" for
    `false` pairs is left to the deterministic unit cases, since some
    non-commuting pairs still yield identical states on a specific input
    and would flake a property test.)

## 6. Out of scope (deferred)

- Any consuming pass (commutation-aware cancellation/fusion) — future
  ticket; if added after DCE it may re-expose the single-pass
  idempotence concern noted in P1-12, at which point a run-to-fixpoint
  wrapper in `PassPipeline::run` is the robust fix.
- The conservative-`false` cases listed in §4.
- Matrix-commutator numeric fallback (rejected: introduces FP tolerance
  and non-deterministic "entries"; the rule table is deterministic and
  testable).
