# [P2-06] Diagonal gate fusion pass — design

> Status: approved (brainstorm 2026-06-02). Implements GitHub issue #106.
> Depends on: P1-09 (`Fuse1qRuns`, `passes` module), P1-10 (`Fuse2q`).

## 1. Problem and motivation

P2-05 (`docs/perf/phase2.md`) established that state-vector simulation is
**memory-bandwidth-bound**: each gate streams the full `2^n` state (512 MiB at
n=25) with near-zero arithmetic intensity, so wall-clock is dominated by the
*number of passes over memory*, not FLOPs. The worst Tier-1 workload is QFT —
the live `tier1_scaling` sweep measured the Aer-comparable `qft_n25.qasm`
fixture at only 2.16×@8 / 2.30×@16 on EPYC.

The lever: controlled-phase, `rz`, `p`, `z`, `s`, `t` are all **diagonal in the
computational basis**, and a product of diagonal operators is itself diagonal.
The entire cphase ladder between two `H` gates in QFT can therefore collapse
into a *single* per-amplitude phase multiply — one memory pass instead of one
pass per gate.

### 1.1 Critical finding — two QFT representations fuse differently

There are two QFT encodings in the repo, and they are **not** equivalent for a
naive "fuse consecutive diagonal gates" pass:

- **Builder QFT** (`benches/src/lib.rs::qft_circuit`, the 3.37× number): emits
  `GateInstance::controlled(Gate::Phase, [j], [k])` — controlled-Phase gates,
  all diagonal. The whole ladder between two `H`s is a contiguous diagonal run.
- **Fixture QFT** (`scripts/qiskit-baseline/circuits/qft_n25.qasm`, the
  Aer-comparable file the acceptance criteria name): already **lowered to `p` +
  `cx`** (900 `p`, 600 `cx`, 25 `h`). The `cx`/`Cnot` gates are **not
  diagonal**, so a pure diagonal-run fuser sees them as run-breakers and gets
  almost nothing. The backlog's "≈92% controlled-phase" description of the
  fixture is misleading — it is decomposed `p`+`cx`.

Per the brainstorm decision, this pass must collapse **both** forms. That
requires *absorbing the interleaved `cx`s*, which the design below does via
monomial-matrix tracking.

### 1.2 Why this is sound — monomial algebra

A `cx` is a **permutation** matrix; a diagonal gate is **diagonal**. Any product
of permutations and diagonals is a **monomial** matrix `M = P · D` (exactly one
non-zero per row/column = one permutation `P` × one diagonal `D`). Walking a
maximal run of `{diagonal gates ∪ cx}` and accumulating `(P, D)`:

- `cx` updates `P` (a GF(2)-linear bit map): `cx(c,t)` does `x_t ^= x_c`.
- A diagonal gate's phase condition is **remapped through the current `P`** as
  it is absorbed.

`cx`s in QFT come in conjugating pairs (`cx · D · cx`), so the running `P`
returns to the **identity** at every `H` boundary. When `P == I`, the run is a
pure diagonal and we emit one fused operation. Worked example verified by hand:
the fixture's `p(π/4) q24 · cx(24,23) · p(-π/4) q23 · cx(24,23) · p(π/4) q23`
reconstructs exactly `cp(π/2)` on (24,23), i.e.
`diag(1,1,1,e^{iπ/2})` — using the identity
`x_a ∧ x_b = (x_a + x_b − (x_a ⊕ x_b)) / 2`.

Collapsing one QFT level into one diagonal pass turns the `~n²/2` cphase gates
into `~n` diagonal passes + `n` H passes ⇒ ≈ `n/4` ≈ 6× pass reduction at n=25,
matching the AC's "≥ 5×".

## 2. IR representation

New `Instruction` variant (not a `Gate` variant — a full-register diagonal with
masks over **absolute** qubit indices is neither fixed-arity nor relocatable, so
it does not belong in the `Gate` enum the Golden Rules protect; keeping it out
also keeps `Gate::arity()`/`matrix()` honest):

```rust
// crates/aleph-ir/src/instruction.rs
pub enum Instruction {
    Gate(GateInstance),
    Measure { qubit: u32, clbit: u32 },
    Reset(u32),
    Barrier(SmallVec<[u32; 8]>),
    DiagonalPhase(Box<DiagonalPhase>),   // NEW
}

pub struct DiagonalPhase {
    pub n_qubits: u32,            // mask bit-width; asserted ≤ 64
    pub terms: Vec<PhaseTerm>,
}

pub struct PhaseTerm {
    pub conds: SmallVec<[u64; 2]>, // AND of these parity-conditions
    pub angle: f64,                // radians, added when all conds' parities == 1
}
```

**Semantics.** Amplitude at basis index `x` is multiplied by

```
exp( i · Σ_t  angle_t · [ ∀ m ∈ conds_t : parity(m & x) == 1 ] )
```

where `parity(v) = popcount(v) mod 2`. Examples:

- `p(θ) q[j]` → `PhaseTerm { conds: [1<<j], angle: θ }`.
- controlled-Phase(θ), ctrl `k`, tgt `j` → `PhaseTerm { conds: [1<<k, 1<<j], angle: θ }`.
- After `cx(c,t)` conjugation a target-bit condition mask gains the control bit:
  `1<<t` becomes `(1<<t) ^ (1<<c)` — the form is closed under `cx`-conjugation
  because conjugation only ever XORs bits *into* existing masks.

`u64` masks cap fusion at ≤ 64 qubits — far above the n ≤ 28 software cap; the
pass asserts `n_qubits ≤ 64`.

### 2.1 Blast radius of the new variant

Every exhaustive `match` on `Instruction` gains an arm:
- `Instruction::used_qubits` → union of all bits set across all `conds` masks.
- `Instruction::used_clbits` → empty.
- Backends (`aleph-sv` `CpuState`/`SoaState`, `aleph-backend` naive) → dispatch
  to the new kernel (§4).
- `layers.rs` → treat as diagonal-with-diagonal commuting where applicable;
  conservatively, may be treated as occupying all its support qubits.
- Passes (`cancel`, `dce`, `commute`, `fuse_1q`, `fuse_2q`) → opaque: they do
  not produce or consume it. `FuseDiagonalRuns` treats an existing
  `DiagonalPhase` as a run-breaker (ensures idempotence).
- `aleph-parser` never produces it; `emit` (QASM round-trip) → must either lower
  it back to its terms or refuse with a clear error. v1: refuse with
  `UnsupportedInstruction` (the op only ever exists post-optimization, never
  round-tripped). This asymmetry is documented; a future ticket may lower it.

## 3. The pass — `passes::FuseDiagonalRuns`

New file `crates/aleph-ir/src/passes/fuse_diagonal.rs`, exported from
`passes/mod.rs`.

**Run definition.** A *run* is a maximal contiguous sequence of instructions
each of which is either (a) a `Gate` whose `gate.is_diagonal()` is true
(including controlled-diagonal `GateInstance`s with external `controls`), or (b)
a `Gate` that is `Cnot`. Any other instruction — non-diagonal non-cx gate,
`Measure`, `Reset`, `Barrier`, or an existing `DiagonalPhase` — ends the run.
`Barrier` is always a hard fence (never crossed).

**State carried across a run.**
- `P`: GF(2) permutation as `n` row-masks (`Vec<u64>`), initialized to identity
  (`P.row[i] = 1<<i`). Invariant: the run-so-far equals `P · D`.
- `terms`: `Vec<PhaseTerm>` accumulating `D`.

**Per-instruction update.**
- **`Cnot(c, t)`**: `P.row[t] ^= P.row[c]`. (No term emitted; `cx` is pure
  permutation.)
- **diagonal gate** on target bit(s) `b…` with optional external controls `k…`,
  contributing angle `θ` when its fire-condition holds: push
  `PhaseTerm { conds: [P.row[k] for k in controls] ++ [P.row[b] for b in target-condition], angle: θ }`.
  - 1q diagonals (`Z, S, Sdg, T, Tdg, Rz, Phase, Unitary1qDiag`): the
    diagonal `diag(d0, d1)` becomes a global-phase term plus a `q`-conditioned
    term. Concretely, factor `diag(d0,d1) = d0 · diag(1, d1/d0)`: emit one
    `PhaseTerm { conds: [P.row[q]], angle: arg(d1) − arg(d0) }` and fold
    `arg(d0)` into a run-global phase term (`conds: []`, always fires). Global
    phase is physically irrelevant for measurement but **must** be preserved for
    the 1e-12 statevector oracle.
  - `Cz` (diag on (a,b), fires `e^{iπ}` at `11`): `conds: [P.row[a], P.row[b]]`,
    `angle: π`.
  - `Ccz`: three conds.
  - `CRz(θ)`: controlled `Rz` → controlled `diag(e^{-iθ/2}, e^{iθ/2})`; expand
    like the 1q case but gated on the control parity.
- **run end** (next instr is a run-breaker or end-of-circuit):
  - if `P == I` (every `P.row[i] == 1<<i`): canonicalize `terms` — merge terms
    with identical `conds` (sum angles), drop terms whose `angle.rem_euclid(2π)`
    is within `DIAGONAL_EPS` of 0, sort `conds` within each term and terms by
    key for determinism — and, **if the cost model (below) approves**, replace
    the whole run with a single `Instruction::DiagonalPhase`.
  - if `P != I`: **emit the run unchanged** (conservative, always correct). QFT
    always lands `P == I` at `H` boundaries, so this fallback does not fire on
    the target workload. (A future ticket may split the run at the last `P == I`
    prefix; v1 is whole-run-or-nothing.)

**Cost model.** Emit the fused `DiagonalPhase` only when the run (a) spans **> 1
qubit** *and* (b) absorbed **≥ 2** diagonal gates. A single-qubit diagonal run
is left untouched for `Fuse1qRuns` + the P1-06 1q-diagonal kernel, which is
cheaper than the generic term-evaluating kernel. This keeps the pass from
pessimizing cheap cases.

**Determinism.** No `HashMap` iteration in output order; canonicalization sorts.
This matters for the idempotence and oracle tests.

## 4. Backend kernel

New `apply_diagonal_phase(&mut self, dp: &DiagonalPhase)` on the SV state types,
applied as a single `par_units` streaming pass over the amplitude array. Per the
brainstorm decision, full kernel rigor: **scalar baseline + AVX-512, in both AoS
(`CpuState`) and SoA (`SoaState`)**, EPYC-validated.

- **Scalar**: for each amplitude index `x`, accumulate
  `φ = Σ_t (all conds parity-1 ? angle_t : 0)`, then multiply the amplitude by
  `(cos φ, sin φ)`. `par_units` over the existing chunk policy (P2-04).
- **AVX-512**: `VPOPCNTQ` (Zen 4 / EPYC 8124P has AVX512-VPOPCNTDQ) for per-lane
  parities; mask-blend + FMA to accumulate `angle`s into a per-lane `φ`; then
  `e^{iφ}`.
  - **Open risk → resolved in the plan, settled by measurement:** AVX-512 has no
    `sincos` intrinsic. Options: (i) a vectorized polynomial sincos
    (sleef-style, with range reduction), or (ii) scalar-extract sincos into a
    lane buffer then SIMD complex-multiply. Because the workload is
    bandwidth-bound, (ii) may already hide the sincos cost behind the memory
    stream — measure on EPYC before committing to (i). The PR reports which won
    and the numbers, honestly (cf. P2-04 negative findings).

`aleph-backend`'s naive backend gets a simple scalar implementation for oracle
parity.

## 5. Pipeline integration

`PassPipeline::default_pipeline()` becomes:

```
[CancelInversePairs, DeadCodeElim, FuseDiagonalRuns, Fuse1qRuns, Fuse2q]
```

`FuseDiagonalRuns` runs **before `Fuse2q`** so raw `cx`s are still present to
absorb — `Fuse2q` would otherwise bury them inside non-diagonal dense 4×4
`Unitary2q` blocks, defeating the permutation cancellation. It runs **after
`Cancel`/`DCE`** (per the existing rationale: exact inverse pairs are deleted,
not fused). Idempotence: an emitted `DiagonalPhase` is a run-breaker for the
pass, so a second `optimize()` is a no-op — the pipeline still reaches a fixpoint
in one `optimize()`.

## 6. Testing requirements

1. **Property test** (`proptest`): a fused run ≡ sequential application of the
   original gates on a **generic** input state (not `|0…0⟩` — per the P1-13
   lesson that `|0…0⟩` oracles miss conjugation bugs), to 1e-12.
2. **Unit tests**:
   - `cx·p·cx` reconstructs the exact `cp(θ)` diagonal (the §1.2 example).
   - multi-bit mask conjugation across two interleaved `cx` pairs.
   - term canonicalization: identical-`conds` merge, near-zero drop, global-phase
     preservation.
   - the `P != I` conservative fallback leaves the run byte-for-byte unchanged.
   - cost-model: a lone 1q-diagonal run is **not** converted.
3. **Standalone pass test** (pass run directly, not only via pipeline — the P1-10
   lesson) **+ pipeline idempotence test** (`optimize()` twice == once).
4. **Oracle tests**: equivalence vs the unfused circuit across all Tier-1
   fixtures, **both raw (pass alone) and via the full pipeline**, to 1e-12 for
   amplitudes (global phase included).
5. **Benchmark**: `tier1_scaling/qft` on **both** the decomposed fixture
   (`qft_n25.qasm`) and the builder QFT, before/after, on the EPYC bench box
   (verified idle per CLAUDE.md), reported in the PR.

## 7. Acceptance criteria (from BACKLOG #106) and how this meets them

- [ ] QFT-25 cphase ladder collapses to ≤ 2 diagonal passes per qubit; total
  gate-pass count drops ≥ 5× vs unfused → met via per-level collapse (§1.2),
  measured as instruction count before/after.
- [ ] Oracle equivalence vs unfused within 1e-12 across Tier-1 fixtures (raw and
  via pipeline) → §6.4.
- [ ] Criterion improvement on `tier1_scaling`/qft on EPYC, reported in the PR →
  §6.5, both fixture and builder.

## 8. Out of scope / follow-ups

- Splitting a run at the last `P == I` prefix when `P != I` overall (v1 is
  whole-run-or-nothing).
- Absorbing `Swap`, `X`, `Y` into runs (Swap is a clean permutation; X/Y are
  affine/monomial and need a constant-offset extension to `P` — deferred).
- QASM `emit` lowering of `DiagonalPhase` back to gates (v1 refuses).
- Vectorized-polynomial sincos if scalar-extract proves to be the SIMD
  bottleneck (decided by measurement; may itself become a follow-up).
