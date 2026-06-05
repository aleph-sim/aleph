# P3-06 — MPS: non-adjacent 2q gates via SWAP

**Issue:** #37 (`area:backend-mps`, `type:feature`, `priority:medium`, M)
**Milestone:** Phase 3
**Date:** 2026-06-05
**Status:** Approved (brainstorming)
**Depends on:** P3-04 (#35, merged), P3-05 (#36, merged)

## Goal

Let the MPS backend apply a 2-qubit gate between **non-adjacent** qubits by
inserting a nearest-neighbor SWAP network, applying the gate on the now-adjacent
pair, then undoing the SWAPs. Today `MpsState::apply_2q` rejects `|a−b| > 1` with
`MpsError::NonNearestNeighbor`; this replaces that rejection with a swap-and-back
path.

## Background (P3-04/05)

- `MpsState::apply_2q(g, u)` handles only nearest-neighbor pairs (`|qa−qb| == 1`),
  else returns `MpsError::NonNearestNeighbor { a, b }` (mapped by the backend to
  `BackendError::InvalidState`). It moves the orthogonality center, contracts the
  two sites, applies the 4×4 gate (qubits[0]=MSB, ADR-0004), SVD-truncates per
  the `TruncationPolicy`.
- A SWAP is an ordinary 2q gate; its 4×4 matrix is supported by `matrix_4x4`, so
  a nearest-neighbor SWAP runs through the existing `apply_2q` machinery.
- Site `q` corresponds to qubit `q` throughout (ADR-0004); every readout path
  (`dense_statevector`, `measure`, `sample`, `expectation`, `probabilities`)
  relies on this.

## Key decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Strategy | **always-swap-back** only. After the gate, undo every SWAP so site = qubit again. The lazy (permutation-tracking) strategy is **deferred** (a future optimization; it would touch every readout path). |
| Which qubit moves | Move the qubit at the **higher** site `hi` down to `lo+1` (leaving `lo` in place), via a ladder of adjacent SWAPs. |
| Benchmark | Measure the SWAP-network cost as a **function of gate distance** only (lazy not implemented). Document the deferral explicitly in the bench/docs — do not silently omit it. |

## Architecture

All changes are inside `crates/aleph-mps/src/mps.rs`. No new modules; readout
paths and the backend/CLI are untouched (site = qubit invariant preserved).

### `swap_adjacent(i)` — private helper

Applies a SWAP on sites `(i, i+1)`. Constructs the SWAP `GateInstance`
(`Gate::Swap` on `[i, i+1]`; if the variant name differs, build the 4×4 SWAP
matrix directly) and routes it through the existing nearest-neighbor 2q path
(the same contraction + SVD machinery — SWAP is unitary, so no truncation error
beyond round-off). This physically exchanges the two sites' qubit states.

### `apply_2q` — swap-and-back path

Replace the early `if qa.abs_diff(qb) != 1 { return Err(NonNearestNeighbor…) }`
with:
- If `|qa − qb| == 1`: the existing nearest-neighbor body (unchanged).
- Else (`> 1`): let `lo = min(qa, qb)`, `hi = max(qa, qb)`.
  1. **Forward ladder:** for `k` from `hi-1` down to `lo+1`, `swap_adjacent(k)`.
     This moves the qubit originally at site `hi` to site `lo+1`; the qubit at
     `lo` is untouched. The two target qubits now occupy sites `lo` and `lo+1`.
  2. **Apply the gate** on the adjacent pair via the nearest-neighbor body, with
     a `GateInstance` whose `qubits` are the two adjacent sites **in the same
     relative order as the original `g.qubits`**, so control/target (the
     qubits[0]=MSB convention) is preserved. Concretely: whichever of `(qa, qb)`
     equals `lo` keeps site `lo`; the other now sits at `lo+1`; build the
     adjacent `GateInstance` mapping `qa → its site`, `qb → its site`.
  3. **Reverse ladder:** for `k` from `lo+1` up to `hi-1`, `swap_adjacent(k)`,
     restoring the original site = qubit layout.

The gate's SVD truncation happens once (step 2) under the active
`TruncationPolicy`; the SWAPs are unitary and add only rounding-level
`trunc_error`. `max_bond_seen` is updated by each `apply_2q` (including SWAPs)
as in P3-05.

### Error handling
- 3q+ gates and external controls: rejected as before (unchanged).
- Non-adjacent 2q: now supported — no longer returns `NonNearestNeighbor`.
- The `MpsError::NonNearestNeighbor` variant and its `BackendError::InvalidState`
  mapping are **retained** (still used as a defensive internal invariant guard,
  e.g. if a ladder leaves the pair non-adjacent — which must not happen). A
  comment notes it is no longer reached on the normal 2q path.

## Testing

1. **Unit:** GHZ-4 built with non-adjacent CNOTs (e.g. `CNOT(0,2)`, `CNOT(0,3)`)
   reconstructs the expected dense state; a `SWAP(0,2)` exchanges two qubits.
2. **Oracle vs `NaiveSvBackend`** (χ large = exact, 1e-10): circuits with
   non-adjacent 2q gates including **asymmetric control/target** — `CNOT(0,3)`,
   `CNOT(3,0)`, `CZ(0,4)`, `CNOT(2,0)` — on 5–6 qubits. This is the guard for
   control/target correctness through the SWAP network (as it was for the 2q
   convention in P3-04).
3. **Property (`proptest`):** random circuits mixing 1q + arbitrary-distance 2q
   gates on 4–5 qubits; MPS dense == SV dense to 1e-9 (χ large).
4. **Invariants:** norm ≈ 1 and the canonical form survive non-adjacent gates
   (covered transitively by the dense oracle).

## Performance

A new bench (extend `benches/nn_qaoa.rs` or a small `benches/long_range.rs`)
applies `CNOT(0, k)` for increasing `k` (distance 1…n−1) on a fixed-n MPS and
records wall-time, documenting the O(distance) SWAP overhead. **No ratio gate**
(P3-06 has no perf AC). The bench comment states the lazy strategy is deferred,
so the curve reflects always-swap-back (2·(distance−1) SWAPs per gate).

## Decomposition (M → ~4 tasks)

1. `swap_adjacent` helper + `apply_2q` swap-and-back path + unit tests
   (GHZ via non-adjacent CNOTs, SWAP exchange).
2. Oracle equivalence vs `NaiveSvBackend` (asymmetric, non-adjacent) + proptest.
3. Long-range distance-scaling benchmark.
4. Docs (lib.rs note; the deferred-lazy comment) + final gate.

## References
- Schollwöck (2011) §4.2 — SWAP networks for non-local gates on MPS.
- P3-04 design: `docs/superpowers/specs/2026-06-05-p3-04-mps-basic-chain-design.md`.
