# [P1-12] Gate cancellation pass — design

> Status: approved (brainstorm) — pending implementation plan.
> Issue: BACKLOG `[P1-12] Gate cancellation pass (H·H, X·X, Rz(θ)·Rz(-θ))`.
> Depends on: P0-07 (Circuit IR), P1-09 (`passes` module + `Pass` trait).

## 1. Goal

Eliminate adjacent self-inverse and inverse-pair gates from a `Circuit`
as an IR-level optimisation pass. After transpilation or naive circuit
construction, redundant gates appear (H·H, X·X, CNOT·CNOT on the same
qubits, Rz(θ)·Rz(−θ)); detecting and removing them is cheap and
effective. Adjacent-only in this ticket; commutation-aware cancellation
is deferred to P1-13.

Acceptance criteria (BACKLOG):

- [ ] Adjacent cancellation pass.
- [ ] At least 5 cancellation patterns.
- [ ] Correctness preserved.

## 2. Placement & architecture

- New module `crates/aleph-ir/src/passes/cancel.rs`, type
  `CancelInversePairs`, implementing the existing `Pass` trait
  (`passes/mod.rs`). Re-export from `passes/mod.rs`.
- `default_pipeline()` becomes `[DeadCodeElim, CancelInversePairs,
  Fuse1qRuns, Fuse2q]`.
  - **Cancel before fusion is load-bearing.** If `Fuse1qRuns` ran
    first it would collapse `Rz(θ)·Rz(−θ)` into a single
    `Unitary1qDiag` (numerically ≈ I but still executed), making the
    pair invisible to inverse-equality detection. Running Cancel first
    removes the pair outright; fusion then works over a smaller input.
  - DCE stays first: cheap, shrinks the input, independent of Cancel.
- **No `aleph-core` changes.** The pass leans entirely on the existing
  `Gate::inverse()` and `#[derive(PartialEq)]` on `Gate`.

## 3. Cancellation rule (single, generic)

Two gate instructions `P` (earlier) and `G` (later) cancel iff **all**
hold:

1. **Adjacency on the full support:** `G` immediately follows `P` on
   every qubit in their shared support — no instruction between them
   touches any of those qubits.
2. **Identical support:** same target qubits **positionally** and the
   same **set** of external controls.
3. **Inverse:** `P.gate == G.gate.inverse()`.

This one rule covers H·H, X·X, Y·Y, Z·Z, S·Sdg, T·Tdg, CNOT·CNOT,
CZ·CZ, SWAP·SWAP, Iswap·IswapDg, Rz(θ)·Rz(−θ), Phase(θ)·Phase(−θ),
CRz(θ)·CRz(−θ), Toffoli·Toffoli, Ccz·Ccz, and their externally-
controlled forms — roughly twelve patterns, well past the ≥5 AC.

**Conservative choices (correctness first):**

- **Targets compared positionally.** `Cnot = [control, target]` is not
  symmetric, so positional comparison is mandatory there. For the
  genuinely symmetric gates (`Cz`, `Swap`, `Ccz`) this means
  `Cz(0,1)·Cz(1,0)` is **not** cancelled in v1. Positional comparison
  never produces a wrong cancellation; it only occasionally misses one.
  Symmetric-gate qubit-order normalisation is an explicit deferral.
- **Controls compared as a set** (control order is semantically
  irrelevant).
- **Parametric pairs cancel only on exact f64 equality** of `−θ`.
  `Rz(0.3)·Rz(−0.3)` cancels (`Rz(0.3).inverse() == Rz(-0.3)`); angles
  that are close-but-unequal do not (that is fusion/tolerance
  territory, not cancellation). NaN params never equal themselves, so
  they never cancel — safe by default.

## 4. Algorithm — single forward pass, per-qubit live-index stacks

State:

- `result: Vec<Instruction>` — output, append-only.
- `removed: Vec<bool>` — tombstones, parallel to `result`.
- `live: HashMap<u32, Vec<usize>>` — per qubit, a stack of indices of
  still-live `result` instructions that touched it.

For each incoming instruction:

- **Gate `G`** with support set `S`:
  - Candidate `i` exists iff every `q ∈ S` has a non-empty `live[q]`
    whose **top is the same index `i`**, AND `used_qubits(result[i])`
    as a set `== S`, AND `result[i]` is a `Gate` equal to
    `G.gate.inverse()` (targets positional, controls as set).
  - If candidate: set `removed[i] = true`, pop `live[q]` for all
    `q ∈ S`, and **do not append `G`** (the pair is gone).
  - Else: append `G` at index `k`, push `k` onto `live[q]` for all
    `q ∈ S`.
- **Measure / Reset / Barrier:** append to `result`; push its index
  onto `live[q]` for every touched qubit. These are **never popped**,
  so they act as barriers — no gate cancels across a measurement,
  reset, or explicit barrier. `Barrier` is preserved verbatim.

The "same top index for all of `S`" + "support equality" conditions
make nested cancellation fall out for free:

- `X H H X` (qubit 0) → after the H·H pop, the two X's become adjacent
  and cancel → empty.
- `CNOT(0,1) X(0) X(0) CNOT(0,1)` → inner X·X cancels, then the two
  CNOTs share top index on both qubits and cancel → empty.

Finalise: collect `result[i]` where `!removed[i]`, preserving order;
this replaces `circuit.instructions`. Read `&circuit.instructions`
directly — **not** `std::mem::take` (P1-09 lesson: an early return
must not leave the circuit empty). `n_qubits` / `n_clbits` / metadata
are untouched.

`PassStats`: `transformations` = number of cancellation events;
`gates_before − gates_after == 2 × transformations`.

## 5. Correctness notes

- **No measurement dependence.** Unlike DCE, Cancel removes only pairs
  that are algebraically the identity, so the output state vector is
  identical regardless of which qubits are later observed. The SV
  oracle is preserved unconditionally.
- **Barriers/Measure/Reset** strictly block cancellation across them
  (§4).
- **Metadata invariants** (qubit/clbit counts) preserved — only
  `instructions` is rebuilt.

## 6. Testing

- **Unit** (co-located in `cancel.rs`):
  - Each pattern cancels: H·H, X·X, Y·Y, Z·Z, S·Sdg, T·Tdg, CNOT·CNOT,
    CZ·CZ, SWAP·SWAP, Iswap·IswapDg, Rz(θ)·Rz(−θ), Toffoli·Toffoli.
  - Non-cancellation: different qubits, different controls, unequal
    params, `Cz(0,1)` vs `Cz(1,0)` (documents the deferral).
  - Barrier / Measure / Reset between a pair blocks cancellation.
  - Nested: `X H H X → ∅`, `CNOT X X CNOT → ∅`.
  - **Standalone pass run** (not only via the pipeline) — P1-10
    lesson: a `pub` pass must be tested in isolation.
- **Property** (`proptest`):
  - Inject `g·g†` pairs at random positions into a random base circuit;
    the pass must return the base circuit (BACKLOG requirement).
  - SV-state equivalence before/after the pass within `1e-12` across a
    fixture set (mirrors the 32-case P1-09 oracle check).
- **Pipeline** test in `passes/mod.rs`: `default_pipeline` cancels a
  redundant pair that fusion alone would not remove.

## 7. Benchmark

- Fixture `cancel_redundant(base_n, base_depth, pairs)` in
  `bench_fixtures.rs`: a useful base contour plus `pairs` injected
  `g·g†` redundancies interleaved through it.
- Bench `crates/aleph-ir/benches/cancel.rs`, gated behind the
  `bench-fixtures` feature.
- The crate-level `bench-fixtures` steps already in
  `.github/workflows/bench.yml` (`cargo bench -p aleph-ir --features
  bench-fixtures [--no-run]`) compile and run every feature-gated bench
  in the crate generically, so the new `cancel` bench needs no workflow
  change — only the `[[bench]]` registration in `Cargo.toml`.
- Metric: pass wall-clock + reduction ratio `gates_before/gates_after`.
  Target ≈ `≥ 3×` on a pair-dominated fixture (parity with the
  self-imposed targets in P1-09 VQE 3.09× and P1-10 QAOA 3.85×).

## 8. Out of scope (deferred)

- Commutation-aware cancellation (reorder to bring cancellable pairs
  together) — P1-13.
- Symmetric-gate qubit-order normalisation (`Cz(0,1) ≡ Cz(1,0)`).
- Angle-merging of non-cancelling rotation pairs (`Rz(α)·Rz(β) →
  Rz(α+β)`) — that is fusion's job; this pass only deletes exact
  inverse pairs.
