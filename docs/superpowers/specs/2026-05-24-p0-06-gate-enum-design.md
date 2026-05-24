# P0-06 — `Gate` enum + parametric gate support

**Status:** Approved (design phase)
**Date:** 2026-05-24
**Issue:** P0-06 (see `BACKLOG.md:278`)
**Depends on:** P0-03 (Complex type)
**Blocks:** P0-07 (Circuit IR), P0-09 (naive backend), all P0-10/11/12 work that consumes the IR.

---

## 1. Purpose

Introduce the canonical representation of quantum gates used by every backend, parser, and IR pass in `aleph`. After this issue:

- Tier 1 algorithms (GHZ, QFT, Grover, random circuit) can be expressed in terms of `Gate` + `GateInstance`.
- P0-07 (`Circuit` IR) can store a `Vec<GateInstance>` directly.
- P0-09 (naive state-vector backend) can dispatch on `Gate` to apply unitaries.

This spec covers the type design, public API, error model, testing requirements, and supporting ADRs. Implementation lands as a single PR titled `[P0-06] Gate enum + parametric gate support`.

## 2. Scope

**In scope**

- New module `aleph_core::gate` (file `crates/aleph-core/src/gate.rs`) with `Gate`, `Param`, `SymbolId`, `GateInstance`, `GateMatrix`, `GateError`.
- Methods on `Gate`: `matrix()`, `is_diagonal()`, `is_clifford()`, `inverse()`, `arity()`.
- Constructors on `GateInstance`: `new()`, `controlled()`.
- Unit and `proptest` property tests covering matrix correctness, identity round-trip, diagonal/Clifford classification.
- Two ADRs: `0002-gate-clifford-detection.md`, `0003-gate-matrix-representation.md`.
- New `aleph-core` dependencies: `smallvec`, `thiserror`, `proptest` (dev).

**Out of scope**

- Symbolic parameters with concrete behaviour. `Param::Symbolic` exists in the enum but has no public constructor in Phase 0 and `matrix()` returns `Err(GateError::SymbolicParam)` if it is ever encountered.
- Stabilizer-style Clifford detection for parametric gates with `θ = k·π/2` angles. Deferred to Phase 2 stabilizer backend work; see ADR-0002.
- Generic n-qubit unitaries (n ≥ 4). Only `Unitary1q` and `Unitary2q` variants exist; arbitrary-arity gates land alongside MPS work in Phase 1+.
- `Circuit` / IR data structures (P0-07) and any backend code (P0-09).
- Oracle tests against Qiskit / Stim. They arrive in P0-09 when there is an actual backend to compare.

## 3. Decisions

Three design questions were resolved during brainstorming:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Concrete `f64` params or `Param` enum from day one? | **`Param` enum with `Concrete(f64)` only.** `Symbolic(SymbolId)` variant exists but has no public constructor. | Avoids a breaking change in Phase 4 (VQE) when backends start matching on `Param`. Boilerplate cost is contained to `matrix()` and `inverse()`. |
| 2 | Matrix representation | **`GateMatrix` enum (`M2x2` / `M4x4` / `M8x8`).** Stack-allocated nested arrays, single uniform return type. | Zero deps, no heap allocation per `matrix()` call, lets backends dispatch with `match m { M2x2(a) => ..., M4x4(a) => ... }`. `ndarray` would force a heap alloc on every kernel invocation. See ADR-0003. |
| 3 | `is_clifford()` for parametric gates | **Always `false` for parametric variants.** | Phase 0 has no stabilizer backend that would benefit. A tolerance-based check (`θ mod π/2 ≈ 0`) is its own design problem with silent-bug failure modes if the tolerance is wrong. See ADR-0002. |

## 4. Types

### 4.1 `Param`

```rust
/// A gate parameter — either a concrete real number or a placeholder for
/// symbolic substitution. Phase 0 only constructs `Concrete`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Param {
    Concrete(f64),
    /// Reserved for Phase 4 (VQE / parametrized circuits). No public
    /// constructor in Phase 0; encountering this variant in `matrix()`
    /// returns `GateError::SymbolicParam`.
    Symbolic(SymbolId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(u32);

impl From<f64> for Param {
    fn from(v: f64) -> Self { Param::Concrete(v) }
}
```

`SymbolId`'s tuple field stays `pub(crate)` (or fully private) so Phase 0 code outside `aleph-core` cannot construct a `Symbolic` variant.

### 4.2 `Gate`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    // --- 1q standard ---
    H, X, Y, Z, S, Sdg, T, Tdg,
    // --- 1q parametric ---
    Rx(Param), Ry(Param), Rz(Param), Phase(Param),
    U3(Param, Param, Param),
    // --- 2q standard ---
    Cnot, Cz, Swap, Iswap,
    // --- 2q parametric ---
    CRx(Param), CRy(Param), CRz(Param),
    // --- 3q standard ---
    Toffoli, Ccz,
    // --- arbitrary unitary, owned ---
    Unitary1q(Box<[[Complex; 2]; 2]>),
    Unitary2q(Box<[[Complex; 4]; 4]>),
}
```

**Why `Box<...>` for `Unitary*` variants:** without indirection the enum's discriminant size would be dominated by `Unitary2q` (256 B), forcing every cache line of a `Vec<GateInstance>` to carry empty bytes for the common case of standard gates. `Box` keeps the variant payload to 8 B (pointer), so the enum stays ~24 B (driven by `U3(Param, Param, Param)`). The hot path — iterating standard / parametric gates — stays cache-friendly. Arbitrary unitaries are rare and tolerate the heap indirection.

### 4.3 `GateInstance`

```rust
#[derive(Debug, Clone)]
pub struct GateInstance {
    pub gate: Gate,
    /// Target qubit indices in spec-defined order
    /// (e.g. for `Cnot` this is `[control, target]`).
    pub qubits: SmallVec<[u32; 4]>,
    /// External controls applied generically on top of the underlying gate.
    /// Empty for non-controlled instances. Phase 0 backends may reject
    /// non-empty `controls`; this is recorded so OpenQASM `ctrl @` lowers
    /// cleanly later.
    pub controls: SmallVec<[u32; 2]>,
}
```

**Qubit ordering convention:** target qubits are stored in the order required by the gate's canonical matrix (e.g. `Cnot.qubits = [control, target]`, `Toffoli.qubits = [ctrl0, ctrl1, target]`, `Swap.qubits = [q0, q1]`). This is documented in the rustdoc for each variant. Backends rely on this contract — violating it silently mis-applies the gate.

### 4.4 `GateMatrix`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum GateMatrix {
    M2x2([[Complex; 2]; 2]),
    M4x4([[Complex; 4]; 4]),
    M8x8([[Complex; 8]; 8]),
}
```

Sizes: M2x2 = 64 B, M4x4 = 256 B, M8x8 = 1024 B. All stack-friendly. Backends pattern-match on the variant to pick the right kernel.

### 4.5 `GateError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    #[error("symbolic parameter cannot produce a concrete matrix")]
    SymbolicParam,
}
```

## 5. Public API

```rust
impl Gate {
    /// Unitary matrix in the computational basis.
    ///
    /// Returns `Err(GateError::SymbolicParam)` if any parameter is
    /// `Param::Symbolic`. In Phase 0 this branch is unreachable through
    /// the public API; the `Result` return type is forward-compatible
    /// with Phase 4 symbolic params.
    pub fn matrix(&self) -> Result<GateMatrix, GateError>;

    /// True iff the matrix is diagonal in the computational basis.
    /// Diagonal-true variants: Z, S, Sdg, T, Tdg, Rz, Phase, CRz, Cz, Ccz.
    /// All others (including diagonal U3 specializations) → false.
    pub fn is_diagonal(&self) -> bool;

    /// True iff the gate belongs to the Clifford group.
    /// True for: H, X, Y, Z, S, Sdg, Cnot, Cz, Swap, Iswap.
    /// (Iswap = SWAP · (S ⊗ S), composition of Clifford generators.)
    /// False for all parametric variants (see ADR-0002), all U3,
    /// T, Tdg, Toffoli, Ccz, and both `Unitary*` variants
    /// (cannot inspect generic matrices without dedicated detection).
    pub fn is_clifford(&self) -> bool;

    /// Inverse gate. Same arity, conjugate-transpose of matrix.
    /// Self-inverse: H, X, Y, Z, Cnot, Cz, Swap, Toffoli, Ccz.
    /// Adjoint pairs: S ↔ Sdg, T ↔ Tdg.
    /// Parametric: Rx(θ) → Rx(-θ), Ry(θ) → Ry(-θ), Rz(θ) → Rz(-θ),
    ///             Phase(θ) → Phase(-θ), CRx/CRy/CRz analogous,
    ///             U3(θ, φ, λ) → U3(-θ, -λ, -φ).
    /// Iswap: returns `Unitary2q(iSWAP†)` — Iswap is not self-inverse
    ///        and Phase 0 does not add an `IswapDg` variant to keep
    ///        enum surface minimal.
    /// Unitary*: returns the conjugate-transpose matrix wrapped in the
    ///           same Unitary* variant.
    pub fn inverse(&self) -> Gate;

    /// Number of target qubits (1, 2, or 3).
    /// Does **not** count generic external controls on a `GateInstance`.
    pub fn arity(&self) -> usize;
}

impl GateInstance {
    pub fn new(gate: Gate, qubits: impl Into<SmallVec<[u32; 4]>>) -> Self;
    pub fn controlled(
        gate: Gate,
        qubits: impl Into<SmallVec<[u32; 4]>>,
        controls: impl Into<SmallVec<[u32; 2]>>,
    ) -> Self;
}
```

`arity()` is provided so backend dispatch (`match gate.arity() { 1 => ..., 2 => ..., 3 => ... }`) can route to a kernel without computing the matrix first. It must agree with `matrix()`:

- `arity() == 1` ⇔ `matrix() == Ok(M2x2(_))`
- `arity() == 2` ⇔ `matrix() == Ok(M4x4(_))`
- `arity() == 3` ⇔ `matrix() == Ok(M8x8(_))`

This invariant is checked by a property test.

## 6. Error Handling

- All methods that could fail return `Result<_, GateError>` (only `matrix()` in this PR).
- No `panic!` on any input that can be constructed through the public API.
- No `unwrap()` / `expect()` outside `#[cfg(test)]` per CLAUDE.md § Code Conventions.

## 7. Testing

Co-located in `crates/aleph-core/src/gate.rs` under `#[cfg(test)] mod tests`.

### 7.1 Unit tests (textbook matrices)

For every concrete gate variant, assert `gate.matrix()?` equals the matrix from Nielsen & Chuang Ch. 4. For parametric gates, test three concrete angles each: `0`, `π/2`, `π`. Tolerance: `AMPLITUDE_TOL` (`1e-10`).

### 7.2 Unit tests (predicates and metadata)

- `is_diagonal()` truth table: assert each variant returns the documented value.
- `is_clifford()` truth table: same.
- `inverse()` returns the documented variant (e.g. `Gate::S.inverse() == Gate::Sdg`).
- `arity()` returns the expected 1 / 2 / 3 for every variant.

### 7.3 Property tests (`proptest`)

1. **Inverse identity:** for every concrete gate variant (and `Rx/Ry/Rz/Phase/U3/CRx/CRy/CRz` with random `θ ∈ [-2π, 2π]`), compute `M = matrix()?`, `M⁻¹ = inverse().matrix()?`, assert `M · M⁻¹ ≈ I` (element-wise within `AMPLITUDE_TOL`). Matrix-multiply helper local to the test module.
2. **Negation symmetry:** for `Rx/Ry/Rz/Phase/CRx/CRy/CRz`, `matrix(Param::Concrete(-θ)) == inverse(matrix(Param::Concrete(θ)))`.
3. **Arity / matrix size invariant:** for every gate, `arity()` and the variant of `matrix()?` agree.
4. **Unitarity:** `M · M† ≈ I` for every concrete and parametric gate (random angle).

### 7.4 Out of scope here

- Oracle tests against Qiskit / Stim — arrive with P0-09 (naive backend) since there is no backend yet to compare.
- Symbolic param tests — Phase 4.

## 8. ADRs

### 8.1 `docs/decisions/0002-gate-clifford-detection.md`

- Title: "Always-false `is_clifford` for parametric gates in Phase 0"
- Context: Clifford rotations exist for `θ = k·π/2`; detecting them needs a tolerance, which is a separate design exercise.
- Decision: parametric variants return `false`.
- Consequence: Phase 2 stabilizer backend will revisit. Documents safe-default reasoning (false negatives degrade performance; false positives silently corrupt results).

### 8.2 `docs/decisions/0003-gate-matrix-representation.md`

- Title: "Stack-allocated `GateMatrix` enum"
- Context: backends need to read gate matrices on the hot path. Options surveyed: `ndarray::Array2`, const-generic per-arity methods, enum of fixed-size arrays.
- Decision: `enum GateMatrix { M2x2, M4x4, M8x8 }` over fixed `[[Complex; N]; N]` payloads.
- Consequence: no heap allocation per `matrix()`; backends pattern-match. An n-qubit `Dense(Array2)` variant can be added non-breakingly when Phase 1 needs it.

## 9. Dependencies to Add

In `crates/aleph-core/Cargo.toml`:

```toml
[dependencies]
num-complex = { workspace = true }
smallvec    = "1"
thiserror   = "1"

[dev-dependencies]
proptest    = "1"
```

`smallvec` and `thiserror` are workspace-wide foundations and will recur in P0-07+; pinning them at the workspace root is left for the implementing PR if no other crate has done it yet.

## 10. Acceptance Criteria

Mirrors BACKLOG.md:320 with concrete checks:

- [ ] `Gate` enum covers all gates in Tier 1 algorithms (H, X, Y, Z, S, T, CNOT, CZ, SWAP, Toffoli, Rx/Ry/Rz, Phase, U3).
- [ ] `GateInstance` carries target qubits and (separately) generic controls.
- [ ] `Gate::matrix() -> Result<GateMatrix, GateError>` exists and is correct for every concrete variant.
- [ ] `Gate::is_diagonal()`, `Gate::is_clifford()`, `Gate::inverse()`, `Gate::arity()` exist with the documented semantics.
- [ ] Unit tests pass; property tests pass on default `proptest` budget.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is green.
- [ ] `cargo fmt --check` is green.
- [ ] ADRs 0002 and 0003 are committed alongside the code change.

## 11. Open Questions

None. All three brainstorming decisions are recorded in §3. If implementation surfaces a fourth, it returns here for a spec amendment before code lands.

## 12. Amendments (post-merge)

The sections above record the design that was approved before implementation. The amendments below record decisions that were forced by code review on the implementation PR. The original sections are retained verbatim for historical record; **the amendments here override them where they conflict**.

### 12.1 `Gate::IswapDg` variant added (overrides §4.2 and §5)

Code review of the initial implementation flagged that the originally-planned `Iswap.inverse() → Gate::Unitary2q(iSWAP†)` fallback broke an important algebraic invariant: the inverse of a Clifford gate must itself be Clifford, but `Unitary2q` returns `false` from `is_clifford()`. A future stabilizer-backend dispatcher routing via `if g.is_clifford() { stab_apply } else { sv_apply }` would silently fall off the fast path mid-circuit on any `iSWAP†`.

**Decision:** Add `Gate::IswapDg` as a 2q standard variant. `Iswap.inverse() == IswapDg`, `IswapDg.inverse() == Iswap`, and both return `true` from `is_clifford()`.

**Updated enum literal:**

```rust
// --- 2q standard ---
Cnot, Cz, Swap, Iswap, IswapDg,
```

**Updated `is_clifford()` set:** `H, X, Y, Z, S, Sdg, Cnot, Cz, Swap, Iswap, IswapDg`.

**Updated `inverse()` doc:** `Iswap` ↔ `IswapDg` form an adjoint pair (no `Unitary2q` fallback).

### 12.2 `GateError::NonFiniteParam` added (overrides §4.5)

Code review found that `From<f64> for Param` silently accepted `NaN`/`Inf`, propagating all-NaN matrices through the backend with no diagnostic.

**Decision:** `Gate::matrix()` rejects non-finite concrete params with `GateError::NonFiniteParam`. The full enum is now:

```rust
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GateError {
    #[error("symbolic parameter cannot produce a concrete matrix")]
    SymbolicParam,
    #[error("parameter must be finite (was NaN or infinite)")]
    NonFiniteParam,
}
```

Note: `GateError` also derives `PartialEq` (added during implementation for ergonomic test assertions — see `expect_err` helper in `kinds.rs`).

`Gate::inverse()` deliberately does **not** reject non-finite params; the matrix-level guard is the single chokepoint. See `inverse()` rustdoc for NaN-equality semantics.

### 12.3 `GateMatrix` does **not** derive `PartialEq` (overrides §4.4)

The §4.4 sample shows `#[derive(Debug, Clone, PartialEq)]`. Code review identified this as a float-equality footgun (CLAUDE.md § Common Mistakes).

**Decision:** `GateMatrix` derives only `Debug, Clone`. Tests use `approx_eq_*` helpers at `AMPLITUDE_TOL` tolerance. Error-case tests use an `expect_err` pattern-match helper instead of `assert_eq!` on `Result<GateMatrix, GateError>`.

### 12.4 `GateInstance` ctors `debug_assert` qubit invariants (overrides §4.3)

`GateInstance::new` and `GateInstance::controlled` `debug_assert!`:

1. `qubits.len() == gate.arity()`
2. Every index in `qubits ∪ controls` appears at most once.

Both checks are debug-only (no-op in release). The `should_panic` test cases are gated with `#[cfg(debug_assertions)]` so `cargo test --release` does not report them as false failures.

### 12.5 Spec §10 acceptance criteria addendum

Add to the §10 checklist:

- [ ] `Gate::matrix()` returns `Err(GateError::NonFiniteParam)` for `Param::Concrete(NaN)`/`Inf` — unit + property tested.
- [ ] `Gate::IswapDg` is in the Clifford set; `Iswap.inverse() == IswapDg`.
- [ ] `is_diagonal()` and `is_clifford()` are written as exhaustive `match` (not `matches!`) so new variants force a compile-time decision.
- [ ] `GateInstance::new`/`controlled` debug-assert arity and qubit-index uniqueness; tests for these are gated with `#[cfg(debug_assertions)]`.
- [ ] `GateMatrix` does not derive `PartialEq`.
