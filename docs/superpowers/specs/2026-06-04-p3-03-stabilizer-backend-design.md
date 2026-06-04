# P3-03 — Stabilizer backend ↔ `Backend` trait + CLI — Design

**Issue:** #34 (P3-03)
**Milestone:** Phase 3 — Alternative Backends
**Depends on:** P3-01 (`2dbbb8e`), P3-02 (`3a51a91`)
**Status:** Design approved, pending spec review
**Date:** 2026-06-04

---

## 1. Goal & scope

Wire the stabilizer simulator into the unified `Backend` trait and expose
it through the CLI (`aleph run … --backend stabilizer`), so Clifford
circuits run end-to-end through the same pipeline as the state vector.

**In scope:**

- `StabilizerBackend` (`crates/aleph-stab/src/backend.rs`) implementing
  `Backend` with `State = Tableau`.
- `allocate`, `apply_gate` (rejecting non-Clifford), `measure`, `sample`,
  `expectation_value`. `probabilities` returns a clear Unsupported error.
- CLI `--backend {statevector, stabilizer}` flag + a stabilizer run path.
- A `surface-code-cycle.qasm` Clifford fixture for the integration test.

**Out of scope:** MPS (P3-04..06), auto-selection (P3-07), `probabilities`
for stabilizer, X/Y-basis measurement, mid-circuit reset.

---

## 2. Crate wiring

- `aleph-stab` gains a dependency on `aleph-backend` (for the `Backend`
  trait + `BackendError`). No cycle: `aleph-backend` depends only on
  `aleph-core`/`aleph-ir`. This mirrors `aleph-sv`.
- `aleph-stab` exports `StabilizerBackend` from `lib.rs`.
- `aleph-cli` gains a dependency on `aleph-stab`.

---

## 3. `StabilizerBackend` (`crates/aleph-stab/src/backend.rs`)

```rust
pub struct StabilizerBackend { rng: StdRng }
impl StabilizerBackend {
    pub fn new() -> Self;            // entropy-seeded (StdRng::from_entropy)
    pub fn with_seed(seed: u64) -> Self;  // reproducible
}
impl Default for StabilizerBackend { fn default() -> Self { Self::new() } }
impl Backend for StabilizerBackend { type State = Tableau; … }
```

Mirrors `NaiveSvBackend`'s constructor shape (`new`/`with_seed`).

### 3.1 Method-by-method

| Method | Behavior |
|--------|----------|
| `allocate(n)` | `Tableau::new(n as usize)`; reject `n > 65_536` → `TooManyQubits { requested: n, limit: 65_536 }` (generous guard; stabilizer is O(n²)). |
| `apply_gate(state, g)` | `dispatch::apply_gate(state, g)` with error mapping (§3.2). |
| `measure(state, q)` | `state.measure(q as usize, &mut self.rng)` with error mapping. |
| `sample(state, shots)` | For each shot: `state.clone()`, `measure` qubits `0..n`, pack outcomes into a `u64` (bit `q` = qubit `q`'s outcome). Returns `Vec<u64>`. Requires `n ≤ 64` (the trait's `u64` row contract); `n > 64` → `TooManyQubits { requested: n, limit: 64 }`. |
| `expectation_value(state, pauli)` | §4. |
| `probabilities(_, _)` | `Err(UnsupportedInstruction { kind: "probabilities" })`. |
| `apply_diagonal_phase` | inherited default (Unsupported) — stabilizer runs the raw circuit, never the SV-optimized `DiagonalPhase`. |
| `apply_tiled_block` | inherited default (replays each gate via `apply_gate`) — Clifford-safe. |
| `unpermute_state` | inherited default (Unsupported) — not on the raw path. |

### 3.2 Error mapping `StabError` → `BackendError`

- `StabError::NonClifford { gate }` → `BackendError::UnsupportedGate { kind: gate }`.
- `StabError::QubitOutOfRange { qubit, num_qubits }` → `BackendError::QubitOutOfRange { qubit, num_qubits }`.

A small `fn map_err(StabError) -> BackendError` in `backend.rs`.

---

## 4. `expectation_value` algorithm

For a Pauli string `P` with real coefficient `c` and sparse `terms` on a
stabilizer state `|ψ⟩`: `⟨ψ|P|ψ⟩ = c · s`, with `s ∈ {+1, −1, 0}`.

Implemented as a `pub(crate)` method on `Tableau` (in `tableau.rs`, where
it has access to internals and the verified `rowsum`):

```rust
/// ⟨ψ|P|ψ⟩ for the unsigned Pauli P given by (x_p, z_p) per qubit:
/// +1/-1 if P (up to sign) is in the stabilizer group, 0 if P
/// anticommutes with some stabilizer generator.
pub(crate) fn pauli_eigenvalue(&self, x_p: &[bool], z_p: &[bool]) -> i8;
```

Algorithm (read-only; clones `self` internally to use the scratch row +
`rowsum`, or accumulates into a local buffer — implementer's choice, both
are O(n²) and side-effect-free on `self`):

1. **Commute check:** for each stabilizer row `k ∈ [n, 2n)`, the symplectic
   product `⊕_j (x_p[j]·z(k,j) ⊕ z_p[j]·x(k,j))`. If any is `1` (odd) → `P`
   anticommutes with a generator → return `0`.
2. **In-group sign:** `P` commutes with all stabilizers ⇒ (pure stabilizer
   state, maximal abelian group) `P ∈ ⟨generators⟩` up to sign. The
   coefficients are `c_k = ⟨P, destabilizer_k⟩` (symplectic product with
   row `k ∈ [0, n)`). Accumulate `∏_{k: c_k=1} stab_(n+k)` via `rowsum`
   into a fresh scratch row (clone), tracking the sign. The accumulated
   Pauli equals `(x_p, z_p)` (by construction); its sign `r` gives the
   eigenvalue `s = (-1)^r` (since `∏stab` fixes `|ψ⟩` with +1, so
   `P|ψ⟩ = (-1)^r |ψ⟩`).
3. Return `s` (`+1` or `-1`).

`StabilizerBackend::expectation_value` then:
- Builds `(x_p, z_p)` from `pauli.terms` (`X`→x, `Z`→z, `Y`→both); rejects
  any qubit index `≥ n` → `BackendError::QubitOutOfRange`.
- Returns `Ok(pauli.coefficient * s as f64)`.

Reuses the verified `g`/`rowsum` machinery — no new phase math.

---

## 5. CLI integration

- Add `BackendKind { StateVector, Stabilizer }` (clap `ValueEnum`,
  `Default = StateVector`) and `#[arg(long)] backend: BackendKind` to the
  `Run` subcommand.
- **`StateVector`** (default): unchanged — the existing precision-based
  `run_with_backend` path.
- **`Stabilizer`:** a new `run_stabilizer` helper in `exec.rs`. The
  existing generic `run_with_backend` cannot be reused — its
  `B::State: AmpsF64` bound is unsatisfiable for `Tableau` (no dense
  amplitudes). `run_stabilizer`:
  - builds `StabilizerBackend::{with_seed|new}`, runs the circuit via
    `aleph_backend::run`,
  - if `--statevector` (or `--force-statevector`) is set → error:
    "the stabilizer backend has no dense state vector; drop --statevector",
  - `--shots` (default 1024 when no other view) → `sample` → `format_counts`,
  - `--expectation P…` → `expectation_value` per Pauli → `format_expectation`,
  - `--precision` is irrelevant to stabilizer; documented as ignored.

The `BackendKind` branch sits alongside the existing `match precision`
block in the `Run` handler.

---

## 6. `surface-code-cycle.qasm` fixture

Add `oracle/circuits/surface-code-cycle.qasm` — one Clifford
stabilizer-measurement round of a small surface-code patch (data qubits +
X/Z ancillas, H/CNOT entangling, ancilla measurements). Purely Clifford,
so it runs on the stabilizer backend and (being small) also on the SV
backend for cross-checks. `random_clifford_n4_d20.qasm` (existing) is a
second integration input.

---

## 7. Testing

1. **Unit (`backend.rs`):**
   - allocate + apply H/S/CNOT + measure round-trips.
   - non-Clifford gate (`T`, `Rz(θ)`) → `UnsupportedGate`.
   - `sample` on GHZ-5 → every shot is all-0 or all-1.
   - `expectation_value`: Bell state ⟨ZZ⟩=+1, ⟨XX⟩=+1, ⟨ZI⟩=0; |0⟩ ⟨Z⟩=+1, ⟨X⟩=0.
   - `probabilities` → `UnsupportedInstruction`.
   - Pauli term qubit out of range → `QubitOutOfRange`.
2. **Cross-backend oracle (`tests/`):** for random Clifford circuits,
   `StabilizerBackend` sample counts match `NaiveSvBackend` sample counts
   (seeded, statistical band) — stabilizer ≡ state vector on Clifford
   circuits.
3. **Expectation oracle:** `StabilizerBackend::expectation_value` vs
   `NaiveSvBackend::expectation_value` on random Clifford circuits ×
   random Pauli strings — agree to 1e-9.
4. **CLI integration (`assert_cmd`, in `aleph-cli`):**
   - `aleph run surface-code-cycle.qasm --backend stabilizer --shots 1024 --seed 0`
     exits 0 and prints counts.
   - `--backend stabilizer` on a non-Clifford QASM → non-zero exit, clear error.
   - `--backend stabilizer --statevector` → non-zero exit, clear error.

---

## 8. Acceptance-criteria mapping

| AC | Covered by |
|----|-----------|
| Stabilizer reachable through unified API | §3 (`Backend` impl), §7.1–7.3 |
| Clear errors on non-Clifford gates | §3.2, §7.1, §7.4 |
| CLI option works | §5, §7.4 |
| Integration: run surface-code-cycle.qasm | §6, §7.4 |

---

## 9. Risks & notes

- **`sample` ≤ 64 qubits:** the `Backend::sample → Vec<u64>` contract packs
  one bitstring per `u64`, so stabilizer sampling is capped at 64 qubits
  even though the state itself scales further. Documented; `n > 64` errors
  clearly. (The AC fixture is small.)
- **`expectation_value` purity:** must not mutate the passed `&Self::State`
  (the trait takes `&state`). The clone-and-scratch (or local-accumulator)
  approach keeps it side-effect-free.
- **CLI `--statevector` on stabilizer:** deliberately an error rather than
  a silent no-op — a dense dump is meaningless/impossible for the
  stabilizer representation.
- **`pauli_eigenvalue` correctness** rests on the pure-stabilizer-state
  property (a Pauli commuting with all `n` generators is in the group up
  to sign). The expectation oracle vs `NaiveSvBackend` is the guard.
