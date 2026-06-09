# P4-07 — Surface-code 1-cycle benchmark (stabilizer) — Design

**Issue:** #45 (`[P4-07]`)
**Date:** 2026-06-09
**Depends on:** P3-03 (stabilizer backend), the existing Stim oracle pattern.

## Goal

Showcase the stabilizer backend on QEC — the killer app for stabilizer
simulation. Implement one surface-code syndrome-extraction cycle at distance
d = 3, 5, 7, 9, 11, prove it matches Stim, and produce a time-per-cycle
benchmark report row.

### Acceptance criteria (from BACKLOG)

- [ ] Cycles run to d = 11 (≈ 240 physical qubits).
- [ ] Match Stim output.
- [ ] Benchmark report row.
- [ ] Testing: logical X / Z operator detection works as expected.

## Why this ticket is structurally different from P4-01..06

The Phase-4 framework (`run.py` + `report.py` + `phase4_*.rs`) is hardwired
around **Qiskit Aer as baseline** and the **state-vector backend** on the
aleph side, timing committed QASM corpus. P4-07 instead uses the
**stabilizer backend**, compares to **Stim**, and runs a circuit with
mid-circuit measurements. It therefore reuses the *Stim-subprocess oracle*
idiom (`crates/aleph-stab/tests/stim_*_oracle.rs`), **not** the Aer report
tooling. `run.py` / `report.py` are left untouched (zero blast radius on the
existing six family rows).

## Locked decisions (from brainstorming)

1. **Circuit source = hand-rolled rotated builder** (Rust). One builder feeds
   both our backend (`Vec<GateInstance>`) and a generated Stim program string,
   so both sides run the identical circuit by construction. No new parser
   surface (no DETECTOR/OBSERVABLE/reset translation).
2. **Primary correctness oracle = postselected canonical stabilizer-group
   equivalence vs Stim**, lifted from `stim_measure_oracle.rs`. Handles the
   random X-ancilla outcomes via postselection.
3. **Report = dedicated `docs/perf/surface_code.md`** rendered by a small
   script (QAOA precedent), comparing aleph vs Stim time-per-cycle. The
   Aer-bound `report.py` is not touched.

## 1. Workload: rotated surface code

Rotated surface-code layout (Fowler et al. 2012; Tomita & Svore 2014 for the
rotated variant), distance d (odd):

- **Data qubits:** d × d grid → d² qubits.
- **Ancilla (measure) qubits:** d² − 1, alternating X-type and Z-type on the
  plaquettes/faces, including the weight-2 boundary stabilizers.
- **Total:** 2d² − 1 → d=3:17, d=5:49, d=7:97, d=9:161, **d=11:241** (≈240 ✓).

Concrete qubit indexing (to be fixed in the plan): data qubits indexed
`0..d²` by row-major grid position; ancillas indexed `d²..2d²−1`. Each ancilla
records its type (X or Z) and its ≤4 data-qubit neighbours (2 on the
boundary). Canonical **N-W-E-S** CX interaction order so X- and Z-checks
commute (the standard schedule that avoids hook-error ordering hazards). In a
noiseless single cycle the schedule does not change the measured group
(everything commutes), but we use the canonical order so the logical-detection
test exercises a genuinely-correct code.

### One syndrome-extraction cycle

- **X-ancilla** `a` over data neighbours `{d_i}`: `H a; CX a d_i (×4, N-W-E-S);
  H a; measure a` → measures X⊗X⊗X⊗X.
- **Z-ancilla** `a` over `{d_i}`: `CX d_i a (×4, N-W-E-S); measure a` →
  measures Z⊗Z⊗Z⊗Z.

For a single cycle from a fresh |0…0⟩ data state, ancillas start in |0⟩, so no
explicit reset is needed.

### Builder API (`benches/src/lib.rs`)

```rust
pub struct SurfaceCode {
    pub distance: usize,
    pub num_qubits: usize,          // 2d² − 1
    pub data: Vec<u32>,             // d² data qubit indices
    pub x_ancillas: Vec<Ancilla>,   // type X
    pub z_ancillas: Vec<Ancilla>,   // type Z
    pub logical_x: Vec<u32>,        // data support of X̄ (a lattice row)
    pub logical_z: Vec<u32>,        // data support of Z̄ (a lattice column)
}
pub struct Ancilla { pub index: u32, pub data_neighbours: Vec<u32> }

impl SurfaceCode {
    pub fn new(distance: usize) -> Self;
    /// One syndrome-extraction cycle as gates (no measurements appended;
    /// caller measures ancillas in `x_ancillas`/`z_ancillas` order).
    pub fn cycle_gates(&self) -> Vec<GateInstance>;
    /// Ancilla measurement order matching the Stim program.
    pub fn ancilla_order(&self) -> Vec<u32>;
}

/// The identical circuit as a Stim program string (H/CX/M only), measurements
/// emitted in `ancilla_order()`. Mirrors stim_oracle::stim_program.
pub fn surface_code_stim_program(d: usize) -> String;
```

Gate set used: `Gate::H`, `Gate::Cnot`, `Gate::X`/`Gate::Z` (for error/logical
injection in tests) — all native to both `StabilizerBackend` and Stim.

## 2. aleph execution path

Drive `StabilizerBackend` directly — **no IR optimization passes** (those are
SV-specific). Per cycle:

```
let mut be = StabilizerBackend::with_seed(seed);
let mut t = be.allocate(2d²−1)?;
for g in sc.cycle_gates() { be.apply_gate(&mut t, &g)?; }
let syndrome: Vec<bool> =
    sc.ancilla_order().iter().map(|a| be.measure(&mut t, *a)).collect()?;
```

The `sample` 64-qubit u64 cap does **not** apply — we collect syndrome bits
ourselves via per-qubit `measure`.

## 3. Correctness vs Stim — `benches/tests/surface_code_stim_oracle.rs`

`#[ignore]` (requires python3 + stim; run on the EPYC oracle venv), one test
parametrised over d ∈ {3,5,7,9,11}. Per d, **postselected stabilizer-group
equivalence**, structurally identical to `stim_measure_oracle.rs`:

1. Run our cycle, obtain ancilla outcomes `b[]` (in `ancilla_order()`).
2. Python subprocess: `sim.do(stim.Circuit(prog_without_M))`, then for each
   ancilla `postselect_z(a, desired_value=b[k])` in the same order; also read
   `peek_z` before each postselect for the determinism cross-check.
3. Compare `canonical_stabilizers()` (re-canonicalised through
   `stim.Tableau.from_stabilizers(...).to_stabilizers(canonicalize=True)` on
   both sides) as a sorted set.
4. Determinism cross-check: `peek_z == +1 ⇒ b false`, `−1 ⇒ b true`, `0 ⇒` no
   constraint.

Note on the Stim program: the oracle needs the gates **without** the trailing
`M` lines (it postselects instead of measuring), so the Stim program is built
from `cycle_gates()` translated to text; ancilla order comes from
`ancilla_order()`. (The committed `.stim` files in §5, which *do* include `M`,
are for timing only.)

## 4. Logical detection — `benches/tests/surface_code_logical.rs`

Fast CI gate, **no Stim**, fully deterministic via the Z-stabilizers
(deterministic from |0…0⟩ = logical |0⟩_L). Tested at d=3 and d=5 with a fixed
seed:

- **No error:** every Z-ancilla outcome = 0.
- **Single physical X error** on a data qubit q (apply `X q` before the cycle):
  exactly the Z-ancilla(s) adjacent to q flip to 1 — a detection event; all
  others stay 0.
- **Logical X̄** (apply `X` on the full `logical_x` row): commutes with every
  Z-stabilizer ⇒ **no** Z-ancilla fires — correctly **undetectable**, the
  defining property of a logical operator.
- **X-basis mirror:** prepare logical |+⟩_L (apply `H` to all data, giving
  |+…+⟩), use the X-ancillas (now deterministic); logical Z̄ on `logical_z`
  is undetectable; a single physical `Z` error fires the adjacent X-ancilla(s).

This satisfies the AC "logical X/Z operator detection works as expected" as a
gated test. Determinism is read straight from `measure` outcomes (no RNG
matching needed — the relevant ancillas are deterministic in the chosen basis).

## 5. Benchmark & report — `docs/perf/surface_code.md`

- **aleph bench:** `benches/benches/phase4_surface_code.rs`, criterion,
  `iter_batched(|| be.allocate(n), |t| {apply cycle_gates; measure ancillas},
  BatchSize::SmallInput)` so allocation is excluded from the timed cycle.
  d ∈ {3,5,7,9,11} — all fast (tableau is O(n²); one cycle is a few hundred
  gates + d²−1 measurements). Registered `harness = false`.
- **Stim corpus:** tiny bin `benches/src/bin/surface_dump.rs` prints
  `surface_code_stim_program(d)` (including `M` lines) → committed
  `scripts/surface_code/circuits/surface_d{3,5,7,9,11}.stim`. Single source of
  truth, human-inspectable.
- **Stim timing:** `scripts/surface_code/stim_timing.py` reads each `.stim`,
  times one cycle (`stim.TableauSimulator().do(circuit)` over N runs, median),
  writes `surface-stim.json`.
- **aleph extraction:** reuse `scripts/bench-report/extract_criterion.py` if it
  generalises to the `surface_code` group; otherwise a thin reader in the
  render script. Produces per-d median ms.
- **Render:** `scripts/surface_code/render_report.py` (+ `test_render.py`
  golden, stdlib `unittest`) merges aleph + Stim JSON + meta →
  `docs/perf/surface_code.md`:

  ```
  | d | qubits | aleph (ms) | Stim (ms) | aleph / Stim |
  ```

## 6. Honesty caveat (expected result)

Stim is a purpose-built, heavily-optimised stabilizer simulator; our CHP
tableau will almost certainly be **slower per cycle** (likely 1–2 orders of
magnitude). The report states this plainly. The deliverable is **correctness
parity with Stim** plus a documented, reproducible time-per-cycle row — not
beating Stim. (Contrast with the SV families where aleph competes with Aer.)

## Crate wiring

- `benches/Cargo.toml`: add deps `aleph-stab = { path = ... }`,
  `rand = { workspace = true }`; register
  `[[bench]] phase4_surface_code` (`harness = false`) and
  `[[bin]] surface_dump`.
- New test files in `benches/tests/` use `StabilizerBackend` + the shared
  builder from `benches/src/lib.rs`. This is safe: `aleph-benches` is a
  separate crate depending on `aleph-stab`, so `StabilizerBackend`'s `Backend`
  impl is a single monomorphisation (the "crate-compiled-twice trait mismatch"
  only bit `aleph-backend`'s *own* tests — P4-04 lesson).

## Out of scope (YAGNI)

- Multi-round detector circuits, decoders (MWPM/UF), noise models, observable
  flips — this ticket is one *noiseless* cycle. (Multi-round detection logic in
  §4 uses only two static error injections, not a noise model.)
- Unrotated surface code.
- Any change to `run.py` / `report.py` / `phase4.md`.
- Committing a QASM corpus (the Rust builder is the source of truth; only the
  `.stim` timing files are committed).

## File manifest

| File | Action |
|------|--------|
| `benches/src/lib.rs` | add `SurfaceCode`, `cycle_gates`, `surface_code_stim_program` |
| `benches/src/bin/surface_dump.rs` | new — dump `.stim` files |
| `benches/benches/phase4_surface_code.rs` | new — criterion cycle timing |
| `benches/tests/surface_code_stim_oracle.rs` | new — postselected group oracle (`#[ignore]`) |
| `benches/tests/surface_code_logical.rs` | new — logical/physical detection gate |
| `benches/Cargo.toml` | add deps + bench/bin registration |
| `scripts/surface_code/circuits/surface_d{3,5,7,9,11}.stim` | new — committed timing corpus |
| `scripts/surface_code/stim_timing.py` | new — Stim timing |
| `scripts/surface_code/render_report.py` | new — render `surface_code.md` |
| `scripts/surface_code/test_render.py` | new — golden test |
| `docs/perf/surface_code.md` | new — the report (EPYC numbers) |
| `BACKLOG.md` | check the four P4-07 ACs |
