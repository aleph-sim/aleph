# P3-07 — Automatic Backend Selection Heuristic — Design

**Issue:** #38 (P3-07). **Depends on:** P3-03 (stabilizer backend), P3-04 (MPS backend) — both merged.
**Date:** 2026-06-05. **Status:** approved, pre-implementation.

## Goal

Analyze a parsed circuit and pick the best backend automatically, so users need
not know which backend fits their circuit. Implements the AC:

- `select_backend(circuit) -> BackendKind` heuristic.
- Manual override available.
- A test corpus selects the expected backend in each category.

## Decisions (locked during brainstorming)

1. **MPS heuristic = conservative structural.** MPS is chosen only for clearly
   low-entanglement circuits (all two-qubit gates nearest-neighbor AND shallow)
   AND only when the state vector cannot fit (`n > 28`). This avoids silently
   routing a high-entanglement circuit into a lossy bounded-χ MPS run.
2. **`auto` becomes the CLI default.** The existing explicit choices
   (`statevector` / `stabilizer` / `mps`) remain and act as manual overrides.
3. **Too-large = warn + proceed.** When the resolved backend is state vector but
   `n > 28`, print a stderr warning and run anyway (matches the existing SV
   soft-cap behavior, which warns rather than refuses). The user stays in control.

## Placement

A new `select` module in **`aleph-backend`**.

- `aleph-backend` already owns the `Backend` trait and the `run(circuit, …)`
  driver, and already depends on `aleph-ir` (it consumes `Circuit`).
- `BackendKind { Statevector, Stabilizer, Mps }` here is an abstract *label*; it
  does **not** import the concrete `aleph-sv` / `aleph-stab` / `aleph-mps`
  crates (those depend on `aleph-backend`, not vice-versa), so there is no cycle.
- Rejected: `aleph-ir` (naming concrete backends would leak backend knowledge
  into the IR — golden-rule violation) and CLI-only (not reusable by the Python
  bindings / future API, and not independently unit-testable as a library).

## Architecture — `aleph-backend/src/select.rs`

```rust
/// Resolved (abstract) backend label produced by the heuristic.
pub enum BackendKind { Statevector, Stabilizer, Mps }

/// Read-only structural features of a circuit, computed in one scan.
pub struct CircuitFeatures {
    pub num_qubits: u32,
    pub depth: usize,                    // circuit.layers().len()
    pub twoq_depth: usize,               // layers containing >= 1 two-qubit gate
    pub all_clifford: bool,              // every Gate is_clifford(); Measure/Barrier ok
    pub all_twoq_nearest_neighbor: bool, // every 2q gate has |q0 - q1| == 1
    pub all_gates_at_most_2q: bool,      // no gate acts on 3+ qubits (MPS can't)
}

pub struct Selection { pub kind: BackendKind, pub reason: &'static str }
pub fn analyze(c: &Circuit) -> CircuitFeatures;
pub fn select_from(f: &CircuitFeatures) -> Selection;    // pure; trivially unit-testable
pub fn select_explained(c: &Circuit) -> Selection;       // kind + human-readable reason
pub fn select_backend(c: &Circuit) -> BackendKind;       // AC-exact signature; = select_explained(c).kind
```

`select_backend`/`analyze`/`select_from` are **pure and total** — read-only
scans, no `Result`, no panic. Worst case they fall through to `Statevector`.

### Feature extraction (`analyze`)

Single pass over `circuit.instructions()`, plus `circuit.layers()` for depth:

- `num_qubits` = `circuit.num_qubits()`.
- `all_clifford` = every `Instruction::Gate(g)` has `g.gate.is_clifford()`.
  `Measure` and `Barrier` are allowed (stabilizer supports measurement).
  Any `DiagonalPhase` / `TiledBlock` instruction (SV-only optimization
  artifacts; not expected pre-optimization) ⇒ `all_clifford = false`.
- `all_twoq_nearest_neighbor` = for every gate acting on exactly two qubits,
  `|q0 - q1| == 1`. Gates of arity ≠ 2 do not affect this flag.
- `all_gates_at_most_2q` = no `Gate` acts on 3+ qubits. The MPS backend supports
  only 1q and (nearest-neighbor) 2q gates, so a 3q+ gate (Toffoli/CCZ/…)
  disqualifies MPS — without this guard a large non-Clifford circuit whose only
  multi-qubit gate is a Toffoli would route to MPS and fail at runtime.
- `twoq_depth` = number of layers (from `circuit.layers()`) that contain at
  least one two-qubit gate. (`depth` = total layer count, kept for diagnostics.)

> **`Reset` is not a selection input.** Verified during planning: the shared
> `run` driver (`aleph-backend/src/lib.rs`) rejects `Instruction::Reset` with
> `UnsupportedInstruction { kind: "reset" }` for **every** backend. A circuit
> containing a reset therefore fails on whichever backend it lands on, so reset
> presence cannot change the *viable* choice — it is deliberately not a feature.

### Decision rule (`select_from`, ordered)

1. `all_clifford` → **Stabilizer** (O(n²) memory; thousands of qubits).
2. `num_qubits <= SV_EXACT_CAP` (= 28) → **Statevector** (exact and fits — never
   risk MPS approximation when an exact backend works and is fast).
3. `all_twoq_nearest_neighbor && all_gates_at_most_2q && twoq_depth <= MPS_DEPTH_THRESHOLD`
   → **Mps** (the only place bounded-χ approximation is used, and only because SV
   can't fit; the 3q+ guard keeps gates the MPS backend can't apply out of it).
4. else → **Statevector** (too large; the CLI warns and proceeds).

Named constants (documented in code):

- `SV_EXACT_CAP: u32 = 28` — the project-wide SV soft cap (matches
  `aleph-sv` / `aleph-cli`).
- `MPS_DEPTH_THRESHOLD: usize = 64` — soft guard against pathological
  entanglement growth in a nearest-neighbor circuit. The MPS backend itself
  bounds memory via χ, so this is conservative, not a hard correctness bound.
  Starting value; validated/adjusted by the test corpus.

## CLI wiring — `aleph-cli`

- The clap `--backend` enum gains `Auto` and becomes the **default**. To avoid a
  name collision with `aleph_backend::BackendKind`, the CLI clap enum is renamed
  `BackendChoice { Auto, Statevector, Stabilizer, Mps }` (mechanical rename
  across `cli.rs` / `exec.rs` / tests).
- `BackendChoice::resolve(&Circuit, wants_amplitudes: bool) -> aleph_backend::BackendKind`:
  - `Auto` → `select_backend(circuit)`, then a **view-compatibility override**:
    if the pick is `Stabilizer` but `wants_amplitudes` is true
    (`--statevector` / `--force-statevector`), downgrade to `Statevector` and
    `eprintln!` a one-line note (stabilizer has no dense state vector). MPS and
    SV can both produce amplitudes, so only the stabilizer pick needs this.
  - explicit `Statevector` / `Stabilizer` / `Mps` → returned verbatim (**manual
    override**; no analysis run).
- On an `Auto` resolution, print to **stderr**:
  `auto-selected backend: <kind> (<reason>)`. stdout (counts / statevector view)
  stays clean and pipeable.
- **Too-large warning:** after resolution, if `kind == Statevector && num_qubits > 28`,
  `eprintln!` a soft warning and proceed.
- The resolved `aleph_backend::BackendKind` maps to the existing dispatch
  (`run_stabilizer` / `run_mps` / `run_with_backend`); no change to those paths.

## Error handling

- `select_backend` / `analyze` / `select_from`: pure, total, no new error type.
- Backend-specific errors (MPS rejecting a 3q/long-range gate, stabilizer
  rejecting non-Clifford) stay where they are. The conservative heuristic will
  not hand a backend something it rejects; if a future gate slips through, the
  backend's own error still fires.

## Testing

- **Unit (`select_from`)** — one test per rule arm, building `CircuitFeatures`
  directly: Clifford→Stabilizer; small dense→Statevector; large NN-shallow→Mps;
  large dense→Statevector; large NN-but-deep→Statevector.
- **Integration (`analyze` + `select_backend`)** — build real circuits with the
  `Circuit` builder and assert the chosen kind (AC "test corpus per category"):
  - all-Clifford GHZ → Stabilizer.
  - small `Rz`/`T` circuit (n ≤ 28) → Statevector.
  - 30-qubit nearest-neighbor shallow brickwork (2q gates on adjacent qubits) → Mps.
  - 30-qubit long-range (a 2q gate spanning distant qubits) → Statevector.
- **CLI (`assert_cmd`)** — `--backend auto` on a Clifford fixture prints
  `auto-selected backend: stabilizer` to stderr and counts on stdout;
  `--backend auto --statevector` on a Clifford circuit shows the downgrade note;
  explicit `--backend mps` overrides (no auto-select line).
- No oracle / proptest — this is a routing decision, not a numerical kernel; the
  correctness of each backend is already gated by P3-01..06.

## Out of scope (YAGNI)

Real entanglement-entropy / treewidth estimation, lazy MPS permutation tracking,
GPU / distributed backend kinds, per-backend cost modeling beyond the structural
rule above.
