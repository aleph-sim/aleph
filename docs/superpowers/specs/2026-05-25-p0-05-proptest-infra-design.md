# P0-05 — Property-based testing infrastructure

**Issue:** P0-05 (see `BACKLOG.md`, GitHub #5)
**Depends on:** P0-04 (workspace deps + MSRV 1.85; proptest 1.x already pinned)
**Date:** 2026-05-25

---

## 1. Goal

Stand up a dedicated `aleph-test` crate that hosts the proptest
strategies the workspace has been duplicating across crates. The
crate replaces the inline `random_*`/`arb_*` helpers in
`aleph-parser/tests/`, `aleph-ir/tests/`, and `aleph-sv/src/*.rs`
with a single source of truth — and closes the last remaining
BACKLOG P0-05 invariant ("diagonal gates leave magnitudes
unchanged") with one new proptest.

This ticket is **completion** rather than fresh implementation:
proptest is already integrated, ~25 property tests already pass
under `cargo test --workspace`. P0-05 turns ad-hoc inline
strategies into a small, focused, reusable library.

---

## 2. Scope

### In scope

- New `crates/aleph-test/` dev-only library crate with four
  modules: `state`, `gate`, `circuit`, `pauli`.
- Eight public proptest strategies + two helpers (§4).
- Migration of duplicated strategies in
  `aleph-parser/tests/round_trip_property.rs` and
  `aleph-ir/tests/layers_properties.rs` to use the shared crate.
- Migration of test-private strategies in
  `aleph-sv/src/backend.rs` (`random_1q_gate_strategy`) and
  `aleph-sv/src/measure.rs` (`random_normalised_state`,
  `RandomOp`, `any_random_op`).
- One new invariant test:
  `aleph-sv/src/backend.rs::tests::diagonal_gate_preserves_magnitudes`.
- New "Property-based testing (P0-05)" section in
  `docs/testing.md` cataloguing generators, invariants, and
  failure-persistence guidance.
- `aleph-test` listed as `[dev-dependencies]` from every consumer
  crate.  `proptest` itself remains workspace-pinned; no version
  bump.

### Out of scope (deferred)

- Generic backend-runner fixture exercising the BACKLOG invariants
  against arbitrary `Backend` implementations. Only one backend
  (`NaiveSvBackend`) exists today; the abstraction has no
  consumer. Revisit in P3+ when MPS / Stab / GPU backends land.
- Custom shrinking strategies for `Complex` / state-vector inputs.
  proptest's built-in shrinking on the underlying `f64` and
  `Vec<_>` is sufficient at the current invariant set.
- Stateful proptest (`TestRunner` with a state machine).
  Mid-circuit measurement & feed-forward (P0-13+) is where this
  pays off.
- `arb_circuit` with classical-control instructions.  The IR
  doesn't support them yet.
- Replacing the proptest-regressions persistence convention or
  bumping default `cases:` budgets.  Existing defaults (256 cases)
  are appropriate.

---

## 3. Architecture

```
crates/aleph-test/
├── Cargo.toml              proptest = regular dep; aleph-core + aleph-ir = regular deps
└── src/
    ├── lib.rs              `pub mod state; pub mod gate; pub mod circuit; pub mod pauli;`
    ├── state.rs            arb_state_vector(n) → Vec<Complex>
    ├── gate.rs             arb_1q_gate / arb_2q_gate / arb_gate / arb_diagonal_1q_gate
    ├── circuit.rs          arb_circuit / arb_op / distinct_pair / distinct_triple
    └── pauli.rs            arb_pauli_string(n, mix_xy) → PauliString
```

Consumers depend on `aleph-test` only via `[dev-dependencies]`.
The crate is `publish = false`. Production builds are unaffected.

**Dep graph** (no cycles; dev-deps do not participate in the
acyclic graph):

```
aleph-test → aleph-core, aleph-ir, proptest

aleph-parser   (dev) → aleph-test
aleph-ir       (dev) → aleph-test          ← dev-dep cycle through aleph-test
                                              is allowed by cargo
aleph-sv       (dev) → aleph-test
aleph-cli      (dev) → aleph-test          (for future prop tests)
aleph-oracle   (dev) → aleph-test          (for future prop tests)
```

---

## 4. Generators — public API

### 4.1 `state.rs`

```rust
/// Random normalized state vector of `n` qubits.  Output length is
/// `2^n`; total norm² lies within `validate_state`'s drift budget
/// (`√n · AMPLITUDE_TOL`).
///
/// Samples (re, im) ∈ [-1, 1] uniformly per amplitude then
/// renormalises.  Not uniformly distributed on the Bloch sphere —
/// intentional: pathological near-degenerate states are part of
/// the input space we want to surface.
pub fn arb_state_vector(n: u32) -> impl Strategy<Value = Vec<Complex>>;
```

### 4.2 `gate.rs`

```rust
/// Random 1-qubit gate.  Vocabulary:
/// H, X, Y, Z, S, Sdg, T, Tdg, Rx(θ), Ry(θ), Rz(θ).
/// Rotation angles ∈ [-2π, 2π].
pub fn arb_1q_gate() -> impl Strategy<Value = Gate>;

/// Random 2-qubit gate.  Vocabulary: Cnot, Cz, Swap, Iswap, IswapDg.
pub fn arb_2q_gate() -> impl Strategy<Value = Gate>;

/// Union of `arb_1q_gate` and `arb_2q_gate`, weighted ~70/30
/// toward 1-qubit (matches typical circuit density).
pub fn arb_gate() -> impl Strategy<Value = Gate>;

/// Diagonal-only 1q subset for the "leaves magnitudes unchanged"
/// invariant.  Vocabulary: Z, S, Sdg, T, Tdg, Rz(θ).
pub fn arb_diagonal_1q_gate() -> impl Strategy<Value = Gate>;
```

### 4.3 `circuit.rs`

```rust
/// Random `Circuit` with `nq` qubits, `nc` classical bits, and
/// `n_ops` instructions.  Operations span 1q / 2q gates and
/// (when `nc > 0`) measurements.
///
/// BACKLOG names the generator `arb_circuit(n, depth)`; this
/// project uses the three-parameter form because clbit count and
/// op count vary independently in practice.  "depth" in the
/// BACKLOG was a loose synonym for op count.
pub fn arb_circuit(nq: u32, nc: u32, n_ops: usize)
    -> impl Strategy<Value = Circuit>;

/// Single random op for the supplied circuit shape.  Composable
/// with `prop_map` if the test needs raw op-level control.
pub fn arb_op(nq: u32, nc: u32) -> BoxedStrategy<OpKind>;

pub fn distinct_pair(nq: u32) -> impl Strategy<Value = (u32, u32)>;
pub fn distinct_triple(nq: u32) -> impl Strategy<Value = (u32, u32, u32)>;
```

`OpKind` is the internal enum currently inlined in parser/ir
tests; it gets moved into this module verbatim (with derived
`Clone, Debug`) so callers can pattern-match before applying.

### 4.4 `pauli.rs`

```rust
/// Random `PauliString` with terms on qubits in `[0, n)`.
///   `mix_xy = false` → Z-only strings (exercises the Z fast path).
///   `mix_xy = true`  → full {I, X, Y, Z} (mixed fallthrough).
/// Coefficient defaults to 1.0; callers wanting random coefficients
/// compose with `(-2.0..=2.0).prop_flat_map(|c| ...)`.
pub fn arb_pauli_string(n: u32, mix_xy: bool)
    -> impl Strategy<Value = PauliString>;
```

---

## 5. Migration map

The existing duplicated / private generators are **replaced**, not
duplicated.

| Location | Symbol | Disposition |
|---|---|---|
| `aleph-parser/tests/round_trip_property.rs` | `OpKind`, `arb_op`, `arb_circuit`, `distinct_pair`, `distinct_triple` | DELETE; import from `aleph_test::circuit` |
| `aleph-ir/tests/layers_properties.rs` | same five helpers (duplicated) | DELETE; import from `aleph_test::circuit` |
| `aleph-sv/src/backend.rs` | `random_1q_gate_strategy` | DELETE; switch callers to `aleph_test::gate::arb_1q_gate` |
| `aleph-sv/src/measure.rs` | `random_normalised_state(n, seed)` | DELETE; rebuild state via `arb_state_vector(n).prop_map(|amps| CpuState { num_qubits: n, amps })` inside the `#[cfg(test)] mod tests` of aleph-sv (CpuState fields are `pub(crate)`) |
| `aleph-sv/src/measure.rs` | `RandomOp` enum + `any_random_op` | DELETE; switch to `aleph_test::circuit::arb_op` and adapt the test to consume `OpKind` |

**LOC delta estimate:**
- aleph-parser: ~80 LOC deleted, ~3 LOC added (use + call).
- aleph-ir: ~80 LOC deleted, ~3 LOC added.
- aleph-sv: ~50 LOC deleted, ~10 LOC added.
- aleph-test: ~250 LOC of new strategy code.
- Net: ~50 LOC delta, with massive DRY improvement.

**Naming convention chosen:** `arb_*` (proptest community
standard). The pre-existing `random_*` names in aleph-sv are
renamed during migration.

---

## 6. New invariant — diagonal-gates-leave-magnitudes

Added inline in `aleph-sv/src/backend.rs::tests`:

```rust
proptest! {
    /// Diagonal 1q gates (Z, S, Sdg, T, Tdg, Rz(θ)) only rotate
    /// phases; they MUST leave |aᵢ| invariant for every basis
    /// state.  The existing reversibility proptests verify a
    /// stronger property — but this targets magnitudes directly
    /// and would surface a single-direction bug (e.g. a Z kernel
    /// that accidentally scales an amplitude).  Cz is also
    /// diagonal but excluded here to keep the strategy 1q-only;
    /// a separate Cz proptest can land later if a real bug
    /// motivates it.
    #[test]
    fn diagonal_gate_preserves_magnitudes(
        op in aleph_test::gate::arb_diagonal_1q_gate(),
        q in 0u32..4u32,
    ) {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(4).unwrap();
        // Non-trivial preamble so the state isn't |0…0⟩.
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0])).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0, 1])).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![2])).unwrap();
        let before: Vec<f64> = s.amplitudes().iter().map(|a| a.norm()).collect();
        b.apply_gate(&mut s, &GateInstance::new(op, smallvec![q])).unwrap();
        let after: Vec<f64> = s.amplitudes().iter().map(|a| a.norm()).collect();
        for (b, a) in before.iter().zip(after.iter()) {
            prop_assert!((b - a).abs() < 1e-12, "|a| changed: {b} → {a}");
        }
    }
}
```

That's the only new proptest. The other three BACKLOG-listed
invariants (norm preservation, reversibility, ∑P=1) already pass
under `cargo test --workspace` from the P0-06…P0-12 work.

---

## 7. Documentation — `docs/testing.md`

Appended section catalogs every shared strategy and every active
invariant test, plus the failure-persistence convention.

```markdown
## Property-based testing (P0-05)

The workspace uses [proptest] for invariant testing. Shared
strategies live in the `aleph-test` crate (`crates/aleph-test/`),
consumed as a `[dev-dependencies]` entry by every crate that
needs them. No production code depends on `proptest`.

### Generators

| Strategy | Module | What it produces |
|---|---|---|
| `arb_state_vector(n)` | `aleph_test::state` | Normalised `Vec<Complex>` of length `2^n` |
| `arb_1q_gate()` / `arb_2q_gate()` / `arb_gate()` | `aleph_test::gate` | Random `Gate` |
| `arb_diagonal_1q_gate()` | `aleph_test::gate` | Z / S / Sdg / T / Tdg / Rz only |
| `arb_circuit(nq, nc, n_ops)` | `aleph_test::circuit` | Valid `Circuit` |
| `arb_op(nq, nc)` | `aleph_test::circuit` | Single random `OpKind` |
| `arb_pauli_string(n, mix_xy)` | `aleph_test::pauli` | `PauliString` |
| `distinct_pair(nq)` / `distinct_triple(nq)` | `aleph_test::circuit` | Raw qubit-tuple helpers |

### Invariants exercised

| Invariant | Where |
|---|---|
| Norm preservation after any gate | `aleph-sv/src/backend.rs::tests::normalisation_invariant` |
| Reversibility (`G†·G·ψ = ψ`) | 10+ proptests in `aleph-sv/src/backend.rs` (`*_then_*_negative_returns_identity`, `*_squared_is_identity`) |
| Diagonal gates leave \|aᵢ\| invariant | `aleph-sv/src/backend.rs::tests::diagonal_gate_preserves_magnitudes` |
| Σ P(outcome) = 1 over full basis | `aleph-sv/src/measure.rs::tests::probabilities_full_basis_sums_to_one` |
| Z fast path ≡ slow path (Z-only Pauli) | `aleph-sv/src/measure.rs::tests::z_fast_path_matches_slow_path` |
| Parser ↔ emitter round-trip | `aleph-parser/tests/round_trip_property.rs::parse_emit_roundtrip` |
| IR layer partitioning correctness | `aleph-ir/tests/layers_properties.rs` |
| f64 round-trip through serde_json | `aleph-oracle/src/fixture.rs::tests::f64_pair_round_trips_through_serde_json` |
| Pauli arg parser ↔ Display | `aleph-cli/src/pauli.rs::tests::z_only_round_trip` |

### Failure persistence

proptest writes shrunk failure seeds to
`<crate>/proptest-regressions/*.txt`. **Commit these files** —
they replay historical failure cases on every future run,
preventing regression of bugs the suite previously caught.

### Adding a property test

1. Pick or compose a strategy from `aleph_test::*`.
2. Inside a `proptest! { #[test] fn ... { ... } }` block, assert
   the invariant with `prop_assert!` (not plain `assert!` — the
   former shrinks).
3. Default `ProptestConfig::default()` (256 cases) is fine for
   most tests; bump `cases: N` for expensive setups.

[proptest]: https://github.com/proptest-rs/proptest
```

---

## 8. Acceptance-criteria mapping

| BACKLOG P0-05 AC | Where satisfied |
|---|---|
| `proptest` integrated, at least 4 generators | §4 ships 8 public strategies + 2 helpers (>>= 4) |
| At least 4 invariant tests passing | §7 catalogues 9 invariant tests (>>= 4) |
| Tests run as part of `cargo test` | All proptests run under `cargo test --workspace`; CI invokes it already |
| Documentation in `docs/testing.md` | §7's new "Property-based testing (P0-05)" section |

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `aleph-test` becomes a god-crate | One module per responsibility (state / gate / circuit / pauli). A fifth unrelated module is a signal to split. |
| Migration changes proptest sample distributions and breaks a previously-passing test | Each migrated test is re-run after migration; any new `proptest-regressions/*.txt` seeds get committed alongside the migration. |
| Shared strategy hides per-test intent | `arb_circuit(nq, nc, n_ops)` is the same shape everywhere it's used today. Tests with idiosyncratic needs keep their bespoke strategies inline. |
| `aleph-test` adds compile-time cost | `proptest` is already pulled in by every test-running crate; `aleph-test` is small (~250 LOC). Net compile-time impact is near-zero. |
| Naming churn (`random_*` → `arb_*`) makes git blame noisy in aleph-sv | The migration is one focused commit per crate; subsequent edits read the renamed symbols. Blame for the underlying logic is preserved on the original implementing commit; only the call-site changes are new history. |

---

## 10. Workflow notes

Standard P0-06…P0-12 workflow:

- Branch: `p0-05-proptest-infra`.
- Implementation order (drives the plan):
  1. Scaffold `crates/aleph-test/` (Cargo.toml + lib.rs + 4 empty modules).
  2. Workspace-level `Cargo.toml`: add `aleph-test` to `members`.
  3. `state.rs`: `arb_state_vector` + unit-test it (red-green TDD).
  4. `gate.rs`: `arb_1q_gate` / `arb_2q_gate` / `arb_gate` /
     `arb_diagonal_1q_gate` + unit tests.
  5. `circuit.rs`: move `OpKind`, `arb_op`, `arb_circuit`,
     `distinct_pair`, `distinct_triple` from parser/ir tests +
     unit tests.
  6. `pauli.rs`: `arb_pauli_string` + unit test (Z-only and mixed).
  7. Migrate `aleph-parser/tests/round_trip_property.rs`:
     delete inline helpers, switch to `aleph_test::circuit`.
  8. Migrate `aleph-ir/tests/layers_properties.rs`: same.
  9. Migrate `aleph-sv/src/backend.rs`:
     `random_1q_gate_strategy` → `arb_1q_gate`.
  10. Migrate `aleph-sv/src/measure.rs`: `random_normalised_state`
      → `arb_state_vector`; `RandomOp` → `arb_op`.
  11. Add `diagonal_gate_preserves_magnitudes` proptest to
      `aleph-sv/src/backend.rs::tests`.
  12. `docs/testing.md` "Property-based testing (P0-05)" section.
  13. `BACKLOG.md` tick P0-05 ACs.
  14. `cargo fmt` + `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace`.
  15. PR.

- Branch is squash-merged with `Closes #5` in the PR body
  (now that we know GitHub auto-close requires the **issue**
  number — see P0-12 retro).
