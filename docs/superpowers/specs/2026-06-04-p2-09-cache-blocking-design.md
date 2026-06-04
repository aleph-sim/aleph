# [P2-09] Cache-Blocked Multi-Gate Application — Design

**Issue:** #109 · **Milestone:** Phase 2 (final ticket) · **Branch:** `p2-09-cache-blocking`
**Date:** 2026-06-04

## Goal

Reduce the per-gate full-state DRAM stream — the limiter established by P2-05 — by
(1) applying a *batch* of gates to each cache-resident tile before advancing, and
(2) relabelling qubit indices so frequently-interacting qubits map to low
(cache-local) bit positions. This is the one CPU-side lever that *avoids* the
memory wall (turning N DRAM passes into 1 for a run of cache-confinable gates)
rather than working within it. It complements the fusion passes (P2-06/P2-07,
which reduce pass *count*) and the NUMA placement (P2-03).

## Acceptance Criteria (from BACKLOG §[P2-09])

1. Measurable L2/L3 cache-miss reduction (`perf stat`) on a low-qubit-heavy
   circuit, reported in the PR.
2. Speedup in the cache-resident regime (intermediate `n` where tiling helps),
   reported on the EPYC box.
3. Oracle equivalence preserved (qubit relabelling is transparent to results)
   within 1e-12.

## Decisions (brainstorm forks resolved)

| Fork | Decision | Why |
|------|----------|-----|
| Scope | **Both** the tile-major driver **and** the qubit-relabelling pass | Full BACKLOG ticket; relabelling extends the driver's coverage from already-low-qubit circuits to arbitrary ones. |
| Tile-major driver representation | **New `Instruction::TiledBlock { gates, tile_bits }`** emitted by a pass; backend executes it tile-major | Consistent with the existing fused IR constructs (`DiagonalPhase`, `UnitaryKq`); isolated and unit-testable. |
| Relabelling transparency | **Output-layer permutation tracking, realized as a single final index-remap (gather)** inside `run_optimized` returning a logical-order state | One O(2^n) gather pass instead of ~n SWAP passes; the returned state is logical-order so `Backend`/`measure`/`sample`/`HasAmplitudes`/the oracle harness are **unchanged**. Mid-circuit measure qubits are mapped through π during the run. |
| Tile size `t` | Default **`tile_bits = 15`** for EPYC (2^15 amps × 16 B = 512 KiB, fits the 1 MiB/core L2 with working-set headroom), CPU-model dispatch via the P2-04 tuning layer; empirically swept | L2 is per-core (1 MiB on EPYC 8124P); a per-thread tile must stay hot in L2. |
| Pipeline order | `[RelabelQubits, Cancel, DCE, FuseDiagonalRuns, Fuse1qRuns, Fuse2q, FuseKq, TileBlock]` | Relabel first (max locality for fusion to exploit); TileBlock last (group what remains after fusion). |

## Architecture

### Mechanism — why tile-major is correct and a win

The state is `2^n` amplitudes. A gate on qubit `q` pairs amplitudes at stride
`2^q`. For **low** `q` (`q < t`) those pairs lie within a `2^t`-amplitude tile;
for **high** `q` they stride across the whole array. Today the run loop is
**gate-major**:

```
for gate in run:           // each gate = one full-state DRAM pass
    for tile in tiles: apply gate to tile
```

For a run of gates all confined to `[0, t)`, reordering to **tile-major** keeps
each tile hot across the whole run:

```
for tile in tiles:         // one DRAM pass over the whole state
    for gate in run: apply gate to tile
```

Correctness: a gate on `q < t` applied to the contiguous sub-slice
`state[tile_start .. tile_start + 2^t]` (with `tile_start` a multiple of `2^t`)
pairs exactly the same amplitudes as applying it to the whole state — because
bit `q` of an absolute index equals bit `q` of the in-tile offset when
`tile_start` is `2^t`-aligned and `q < t`. The existing kernels already index
relative to the slice they're handed, so they apply unchanged to the sub-slice.
No floating-point reduction is reordered, so the result is **bit-identical** to
gate-major (same guarantee `par_blocks` already provides across thread counts).

**Controls ≥ t** are constant across a tile (they are determined by
`tile_start`'s high bits): for each tile the gate is either fully active or
fully skipped. **Controls < t** are handled by the kernel within the tile. The
`TileBlock` pass only groups gates whose **targets are all `< t`**; a gate with a
target `≥ t` ends the current block. (Controls `≥ t` are permitted within a
block — the executor masks per tile.)

### Component 1 — `TileBlock` pass (`aleph-ir`)

`crates/aleph-ir/src/passes/tile_block.rs`. Runs **last** in `default_pipeline`.
Walks the post-fusion instruction stream and greedily accumulates a maximal run
of consecutive `Instruction::Gate` whose gate targets are all `< tile_bits` into
one `Instruction::TiledBlock { gates: Vec<GateInstance>, tile_bits: u8 }`. Any
instruction that is not such a gate (a high-target gate, `Measure`, `Barrier`,
`Reset`, `DiagonalPhase`, or a `TiledBlock` boundary) flushes the current run
and is emitted verbatim. A run of length 1 is emitted as a plain `Gate` (no
benefit from a one-gate block). `tile_bits` comes from the tile policy
(below); the pass does not relabel — it consumes whatever qubit layout it is
given.

### Component 2 — `Instruction::TiledBlock` + backend executor

- New variant `Instruction::TiledBlock(Box<TiledBlock>)` where
  `TiledBlock { gates: Vec<GateInstance>, tile_bits: u8 }`. Exhaustive `match`
  sites updated (instruction.rs, circuit.rs validation/layers, parser
  `emit.rs`/`lower.rs` — emit/parse may reject it as a non-surface construct
  like `DiagonalPhase` is handled, error.rs, backend run loop). The parser does
  **not** produce `TiledBlock` (it is pass-emitted only), mirroring how
  `DiagonalPhase`/`UnitaryKq` are internal.
- `Backend` trait gains `fn apply_tiled_block(&mut self, state, &TiledBlock) ->
  Result<(), BackendError>` with a **default impl** that simply replays
  `apply_gate` for each gate in order (so backends without a tiled fast path —
  and the SoA/f32 backends initially — stay correct). The run loop dispatches
  `TiledBlock` to it.
- `NaiveSvBackend` overrides it with the tile-major executor: split
  `state.amps` into `2^tile_bits`-amplitude tiles; rayon-parallel over tiles;
  per tile, for each gate, evaluate `≥ tile_bits` controls against the tile
  index (skip the gate for this tile if any such control is 0) and apply the
  gate to the tile sub-slice via the existing AoS kernels. The kernel call uses
  the gate's targets/controls **as-is** for the `< tile_bits` bits; the
  `≥ tile_bits` controls are consumed by the per-tile active/skip test and
  dropped from the slice-local control set.

### Component 3 — `RelabelQubits` pass + permutation tracking

- `crates/aleph-ir/src/passes/relabel.rs`. Runs **first**. Builds a weighted
  qubit-interaction signal from the gate stream — primarily, for each qubit, how
  often it appears in gates and how often pairs of qubits co-occur in the same
  gate / adjacent gates. Assigns the highest-traffic qubits to the lowest bit
  positions (a permutation `π: logical → physical`). Rewrites every gate's
  qubit/control indices through `π`. Records `π` on the `Circuit`
  (`Circuit.qubit_permutation: Option<Box<[u32]>>`, `None` = identity).
- **Heuristic guard:** the pass only commits a non-identity `π` when it
  estimates a net win — i.e. the relabelling makes enough additional gates
  tile-confinable (targets `< tile_bits`) to outweigh the one final gather pass.
  Otherwise it leaves `π = None` (identity) and is a no-op. Conservative by
  design — correctness never depends on the heuristic, only the speedup does.
- **Transparency (single final remap):** `run_optimized_with_outcomes` reads
  `optimized.qubit_permutation`. If present:
  - Mid-circuit `Measure { qubit, .. }`: the qubit is mapped `π(qubit)` before
    calling `backend.measure`, and the recorded `MeasurementRecord.qubit` is the
    original logical qubit (un-mapped) so the outcome ordering contract holds.
  - After the run, apply one index-remap so the returned state is in **logical**
    order: `out[i] = phys[ bitperm(i, π) ]` (a single gather; reuses an aligned
    scratch buffer). The state handed back to the caller is logical-order, so
    `HasAmplitudes`, the oracle harness, `sample`, and the CLI are **untouched**.
  - Raw `run` / `run_with_outcomes` (the oracle reference path) never relabel
    and are byte-for-byte unchanged.

### Component 4 — tile policy

Extend the P2-04 `kernels::tuning` layer (or a sibling `TilePolicy`) with a
`tile_bits` selected by CPU model via `cpuid`, default **15** on EPYC-class
AVX-512 parts, with a documented fallback. The `TileBlock` pass queries it. A
sweep on EPYC confirms/adjusts the default (same data-driven discipline as
P2-04's grain).

## Data flow

```
circuit ──optimize()──▶ [RelabelQubits → fusion passes → TileBlock] ──▶ optimized IR
                              │ (records π on Circuit)                      │
run_optimized(&mut backend) ──┘                                            │
   for inst in optimized:                                                  │
     Gate/DiagonalPhase ─▶ apply_gate / apply_diagonal_phase (map measure q through π)
     TiledBlock         ─▶ backend.apply_tiled_block ─▶ tile-major executor
   ── final: if π set, single gather phys→logical ──▶ logical-order State
```

## Testing

1. **Oracle equivalence (AC #3), 1e-12:** Tier-1 fixtures through the full
   relabel+tile pipeline vs the raw `run` reference; identical within 1e-12.
   Both `NaiveSvBackend`'s tiled executor and the default-impl replay path are
   covered.
2. **Tile-major ≡ gate-major (bit-exact):** a generated low-qubit run applied
   tile-major must equal the gate-major result exactly (no tolerance) — the
   correctness core of the executor; cover controls `< t`, controls `≥ t`
   (per-tile active/skip), and `tile_bits` at/above `n` (degenerate single
   tile).
3. **Permutation round-trip (property):** for a random `π`, relabel + final
   gather restores the logical state exactly; `MeasurementRecord.qubit` values
   are logical.
4. **Thread-invariance:** tiled executor result independent of
   `RAYON_NUM_THREADS` (tiles are disjoint, no cross-tile reduction).
5. **`perf stat` (AC #1):** L2/L3 cache-miss + LLC-load-miss before/after on a
   constructed low-qubit-heavy circuit, EPYC, reported in the PR.
6. **Wall-clock (AC #2):** cache-resident-regime speedup on EPYC, reported.
   Honest counter-case: a high-qubit-bound circuit (random brick-wall) shows no
   win — report both.

## Validation ops

- SIMD + cache-miss measurement on EPYC ([[aleph-bench-server]], `perf` 7.0.0
  present; L2 = 1 MiB/core, L3 = 16 MiB/CCX). `perf stat -e
  cache-misses,LLC-load-misses,L1-dcache-load-misses`. Deliver via git bundle;
  idle-check ([[feedback-check-server-clean]]); `cargo check --target
  x86_64-unknown-linux-gnu` for SIMD codegen locally (aarch64 dev box).

## Out of scope (YAGNI)

- Tiled executor for the SoA and FP32 backends — they inherit the correct
  default-impl replay (`apply_gate` per gate); a tiled f32/SoA executor is a
  follow-up if the AoS win justifies it.
- Hierarchical (L2-within-L3) two-level tiling — single-level L2 tiling first.
- Relabelling that physically permutes the stored state mid-run (we only ever
  remap once at the output boundary).
- Cross-`TiledBlock`/`DiagonalPhase` interleaving optimizations.
