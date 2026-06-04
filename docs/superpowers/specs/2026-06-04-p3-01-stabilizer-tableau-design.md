# P3-01 — Stabilizer simulator (Aaronson-Gottesman tableau) — Design

**Issue:** P3-01 (GitHub issue for `area:backend-stab` / Phase 3 opener)
**Milestone:** Phase 3 — Alternative Backends
**Status:** Design approved, pending spec review
**Date:** 2026-06-04
**Branch:** `p3-01-stabilizer-tableau`

---

## 1. Goal & scope

Implement the core of a stabilizer simulator using the Aaronson-Gottesman
(CHP) tableau formalism in the `aleph-stab` crate (currently an empty
stub). Clifford circuits simulate in O(n) time per gate, O(n²) space —
thousands of qubits on a laptop.

**In scope (P3-01):**

- `Tableau` core type: identity init, native Clifford gate updates, O(n)/gate.
- Full IR Clifford gate set: `H, X, Y, Z, S, Sdg, Cnot, Cz, Swap, Iswap, IswapDg`.
- `apply_gate(&mut Tableau, &GateInstance)` dispatch that rejects non-Clifford gates.
- Stabilizer readout API (Pauli strings + signs) for oracle comparison.
- Tests: unit (Bell/GHZ), proptest (symplectic invariants), Stim oracle.
- Criterion bench: 1000 qubits, depth 100, asserted < 1s on EPYC.

**Explicitly out of scope (later tickets):**

- Measurement with collapse → **P3-02** (the scratch row and `rowsum`
  primitive are reserved here but not exercised).
- `Backend` trait impl + CLI `--backend stabilizer` → **P3-03**.
- Anything MPS → P3-04..06; auto-selection → P3-07.

---

## 2. Crate & module layout

```
crates/aleph-stab/
├── Cargo.toml          # + aleph-core, aleph-ir deps (NO Backend yet)
└── src/
    ├── lib.rs          # re-exports Tableau, StabError, apply_gate
    ├── bits.rs         # dependency-free packed-bit row helper
    ├── tableau.rs      # Tableau core + native Clifford ops + readout
    ├── dispatch.rs     # Gate → tableau ops; non-Clifford rejection
    └── error.rs        # thiserror StabError
```

`Gate`, `GateInstance`, `Pauli`, and `PauliString` all live in
`aleph-core`, so the only new crate dependency is `aleph-core`
(`aleph-ir` is **not** needed in P3-01 — no `Instruction`/`Backend`).

**Dependency policy:** no `bitvec` crate. The bit operations we need
(get/set/toggle/swap a single column bit within a packed row) are a few
lines over `Vec<u64>`; a custom helper keeps the hot loop under our
control and adds no dependency (CLAUDE.md golden rule). The only new
crate dependency is `aleph-core` (for `Gate`, `GateInstance`, `Pauli`,
`PauliString`), already in-workspace.

---

## 3. Tableau representation

Row-major CHP layout (Aaronson-Gottesman 2004, §2), matching the BACKLOG
"BitVec for the tableau rows" hint but with a hand-rolled packed store:

- **Rows:** `2n + 1`.
  - `0 .. n` — **destabilizer** generators.
  - `n .. 2n` — **stabilizer** generators.
  - `2n` — **scratch** row, reserved for P3-02 `rowsum` (allocated here,
    unused in P3-01).
- **Each row** holds `n` x-bits, `n` z-bits, and one phase bit `r`
  (`r = 1` ⇒ leading `-` sign; the imaginary `i` factors cancel in pure
  Clifford evolution and are tracked only transiently inside `rowsum`,
  which is P3-02).
- **Packed storage** (`bits.rs`): each row is `ceil(n/64)` `u64` words for
  x, the same for z, plus one bit for `r`. Column-bit access = word index
  `c >> 6`, mask `1 << (c & 63)`.

**Identity init — `Tableau::new(n)`:** destabilizer `i` = X_i
(`x[i][i]=1`), stabilizer `i` = Z_i (`z[n+i][i]=1`), all `r = 0`. This is
the stabilizer state |0…0⟩.

**Cost:** every gate loops over `2n` rows doing O(1) bit work → O(n) per
gate, O(n²) memory. Meets the AC. Word-level bit-slicing (Stim-style,
O(n/64) per gate) is a deliberately deferred optimization — only pursued
if EPYC numbers miss the < 1s target.

---

## 4. Gate operations

### 4.1 Primitive tableau updates (applied to every row `i ∈ 0..2n`)

From Aaronson-Gottesman §2 (the CHP update rules):

| Gate | Phase update | Bit update |
|------|--------------|------------|
| `H(a)`     | `r ^= x_a & z_a` | swap `x_a, z_a` |
| `S(a)`     | `r ^= x_a & z_a` | `z_a ^= x_a` |
| `CNOT(a,b)`| `r ^= x_a & z_b & (x_b ^ z_a ^ 1)` | `x_b ^= x_a`; `z_a ^= z_b` |

### 4.2 Pauli gates (direct sign rules)

Paulis conjugate each stabilizer to `± itself`, so they only flip `r`:

| Gate | Phase update |
|------|--------------|
| `X(a)` | `r ^= z_a` |
| `Z(a)` | `r ^= x_a` |
| `Y(a)` | `r ^= x_a ^ z_a` |

### 4.3 Composed Clifford gates

Built from the primitives, each unit-tested for exact equivalence against
the existing state-vector backend (`NaiveSvBackend`):

- `Sdg(a) = S(a); S(a); S(a)`  (S† = S³, since S⁴ = I)
- `Cz(a,b) = H(b); CNOT(a,b); H(b)`
- `Swap(a,b) = CNOT(a,b); CNOT(b,a); CNOT(a,b)`
- `Iswap(a,b)`, `IswapDg(a,b)` — standard H/S/CNOT decompositions, the
  exact factorization pinned by an SV-equivalence unit test (the test is
  the source of truth; the decomposition is whatever passes it at 1e-12).

### 4.4 Dispatch & rejection — `dispatch::apply_gate`

```rust
pub fn apply_gate(t: &mut Tableau, inst: &GateInstance) -> Result<(), StabError>;
```

- Reuses the existing `Gate::is_clifford()` (aleph-core) as the gate.
- Non-Clifford gate ⇒ `StabError::NonClifford { gate }`.
- Qubit index ≥ `n` ⇒ `StabError::QubitOutOfRange`.
- Maps each Clifford `Gate` variant + its `GateInstance::qubits` to the
  native/composed op. Control/target ordering follows the IR convention
  (`Cnot` = `[control, target]`).

---

## 5. Stabilizer readout API

Reuse the existing `aleph_core::{Pauli, PauliString}` types — no new
Pauli type in aleph-stab. `PauliString { coefficient: f64, terms:
Vec<(u32, Pauli)> }` is sparse (identity terms omitted); the generator's
sign maps to `coefficient = +1.0 | -1.0`.

```rust
impl Tableau {
    /// The n stabilizer generators (rows n..2n) as signed Pauli strings.
    pub fn stabilizers(&self) -> Vec<PauliString>;
    /// The n destabilizer generators (rows 0..n) — for invariant tests.
    pub fn destabilizers(&self) -> Vec<PauliString>;
}
```

`(x,z)` → Pauli mapping: `(0,0)=I` (omitted from `terms`), `(1,0)=X`,
`(0,1)=Z`, `(1,1)=Y`; `r=1` ⇒ `coefficient = -1.0`. This is the
comparison surface for the Stim oracle and Bell/GHZ unit assertions.

Plus invariant helpers for property tests:

```rust
impl Tableau {
    /// Symplectic inner product of two rows (0 = commute, 1 = anticommute).
    fn rows_anticommute(&self, i: usize, j: usize) -> bool;
}
```

---

## 6. Testing strategy

Per CLAUDE.md "Testing Requirements" — unit + property + oracle + bench.

### 6.1 Unit tests (co-located in `tableau.rs`)

- Bell `H(0); CNOT(0,1)` ⇒ stabilizers `{+XX, +ZZ}` (canonicalized).
- GHZ-3 `H(0); CNOT(0,1); CNOT(1,2)` ⇒ `{+XXX, +ZZI, +IZZ}` (canonical).
- Each composed gate (`Sdg, Cz, Swap, Iswap, IswapDg`) verified against
  `NaiveSvBackend`: prepare a random small (n≤5) Clifford circuit, apply
  the gate under test in both backends, then assert every tableau
  stabilizer generator fixes the SV state. Computed via
  `Backend::expectation_value(ψ, P)`: a generator with sign `s` and
  unsigned Pauli `P` fixes `|ψ⟩` iff `⟨ψ|P|ψ⟩ = s` (±1.0 to 1e-12). No
  manual amplitude manipulation needed.
- `apply_gate` rejects `T`, `Rz(θ)`, `Toffoli`, … with `NonClifford`.

### 6.2 Property tests (`proptest`, alongside unit)

For random Clifford circuits (random sequences over the 11-gate set on
n ≤ 12 qubits):

- **Symplectic invariant:** destabilizer `i` anticommutes with
  stabilizer `i`; commutes with every other stabilizer and destabilizer;
  stabilizers mutually commute. (This is the canonical tableau
  well-formedness invariant — must hold after every gate.)
- **Generic-state oracle (not |0…0⟩):** prepare a generic state, evolve
  in both stabilizer and SV backends, assert each stabilizer generator
  fixes the SV state. (Lesson from P1-13: a |0…0⟩-only oracle misses
  bugs — prep a generic state.)

### 6.3 Stim oracle (EPYC-gated)

- 100 random Clifford circuits: run through `Tableau`, read out
  `stabilizers()`, compare the **canonical** stabilizer group against
  Stim's (`stim.Tableau` / `TableauSimulator.canonical_stabilizers()`).
- Implemented mirroring the existing `aleph-oracle` Qiskit pattern
  (Python subprocess; fixtures gated by the same build/feature
  mechanism). `stim` added to the EPYC oracle venv alongside
  `qiskit-aer`.
- Marked `#[ignore]` locally/CI (like the slow Qiskit oracles); run on
  the EPYC box. Group comparison is sign-and-generator canonical, not
  row-order-sensitive (Gaussian-eliminate both to RREF before comparing).

### 6.4 Benchmark (criterion)

- `stab_clifford_1000q_depth100`: 1000-qubit, depth-100 random Clifford
  circuit. Assert wall-clock < 1s on a **verified-idle** EPYC box
  (idle-check rule, CLAUDE.md / `feedback-check-server-clean`).
- Report numbers in the PR per the golden rule.

---

## 7. Acceptance criteria mapping

| AC | Covered by |
|----|-----------|
| Handles H, S, CNOT, X, Y, Z (+ full Clifford set) | §4 |
| 1000q depth 100 < 1s | §6.4 bench |
| Verified against Stim | §6.3 oracle |
| Correctly rejects non-Clifford gates | §4.4 + §6.1 |

---

## 8. Risks & notes

- **Iswap/IswapDg decomposition** is the one place a wrong factorization
  silently produces a valid-but-wrong Clifford; the SV-equivalence unit
  test is the guard. Pin it before relying on it.
- **Perf headroom:** O(n) row-loop should clear < 1s comfortably
  (~10⁸–10⁹ bit ops). If EPYC disagrees, bit-slicing is the escape hatch
  — but it complicates P3-02's `rowsum`, so we stay row-major unless
  forced.
- **Stim canonicalization:** stabilizer *groups* are equal even when
  generator *rows* differ; compare canonical (RREF) forms, never raw rows.
- **No `Backend` impl in this PR** — keep aleph-stab free of the backend
  trait so P3-03 owns that boundary cleanly.
