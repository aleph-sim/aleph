# P3-09: MPS lazy SWAP permutation tracking + multithreading — design

**Issue:** #125 `[P3-09]` · **Crate:** `aleph-mps` · **Date:** 2026-06-11
**Decision:** approved approaches — A1 (full lazy permutation with routed reads) + B2 (faer
`rayon` parallelism + gemm-recast of the hot contractions). One PR, two stages.

## Problem

1. Non-adjacent 2q gates use an always-swap-back SWAP network (`apply_2q`, mps.rs): a
   forward ladder, the gate, then a reverse ladder — `2·(d−1)` nearest-neighbor SWAPs per
   gate, each paying a truncated SVD. The reverse ladder exists only to restore the
   `site == qubit` invariant that every read path assumes.
2. Everything is single-threaded: faer SVD built without its `rayon` feature, QR on
   nalgebra, and the two-site contractions are hand-rolled nested loops.

## Stage 1 — lazy permutation (A1)

### State

`MpsState` gains:

- `qubit_of_site: Vec<u32>` and `site_of_qubit: Vec<usize>` — mutually inverse maps,
  initialised to identity, O(1) lookup both ways.
- `swaps_applied: u64` — counts physical `swap_adjacent` applications; public getter
  `swaps_applied()` (evidence for AC-1's "fewer applied SWAPs").

`swap_adjacent(k)` updates both maps and the counter. No `Backend` trait changes.

### Gate application

- `apply_2q(g, u)`: resolve `sa = site_of_qubit[qa]`, `sb = site_of_qubit[qb]`. If
  adjacent, apply directly. Otherwise ladder the qubit at the **higher** site down until
  adjacent to the lower one — `d−1` SWAPs — apply the gate, and **do not swap back**.
  Consecutive long-range gates on nearby qubits amortise: the qubits stay where the last
  gate left them.
- `apply_2q_adjacent` is refactored to take site indices plus which site carries
  `g.qubits[0]` (the matrix MSB, ADR-0004). Today the MSB orientation is derived from
  qubit names assuming `site == qubit`; under a permutation that derivation is wrong.
- `apply_1q(q)` routes through `site_of_qubit[q]`.

### Read routing

| Read | Change |
|---|---|
| `measure(q)` | move center to `site_of_qubit[q]`, collapse there |
| `sample` | site `i` outcome packs into bit `qubit_of_site[i]` |
| `probabilities(qs)` | `out_bit_for_site[site_of_qubit[q]] = pos` |
| `dense_statevector` | site `q`'s physical bit shifts by `qubit_of_site[q]` |
| `expectation` | already routed via `apply_1q`; `overlap` is site-wise on a clone that shares the permutation — unchanged |

### Correctness note (AC-1)

At finite χ the lazy ordering produces a *different truncation pattern* than
always-swap-back, so 1e-10 equivalence vs `NaiveSvBackend` is asserted in the exact
regime (χ large enough / ε=0) — the same contract the P3-06 oracle uses.

## Stage 2 — multithreading (B2)

- **Cargo:** faer features become `["linalg", "std", "rayon"]` (still
  `default-features = false`). faer shares the global rayon pool, so `RAYON_NUM_THREADS`
  controls it — consistent with `aleph-sv`.
- **QR → faer:** `thin_qr` (tensor.rs) ported from nalgebra to faer. It runs on every
  orthogonality-center move and nalgebra's QR is single-threaded and slow.
- **Theta-build → one gemm:** the layouts already agree —
  `group_left(site_i) (li·2 × mi) · group_right(site_j) (mi × 2·ri)` *is* the matrix the
  hand-rolled loops currently build and feed to the SVD (row `l·2+a`, col `b·ri+r`). The
  intermediate flat vec and the final reshape disappear.
- **Gate-apply stays loops:** applying U to theta costs O(16·li·ri), a factor χ/4 cheaper
  than theta-build O(4·li·mi·ri) — not on the hot path.
- **`absorb_into_left/right` → gemm:** every center move, O(χ³).
- **Out of scope:** transfer contractions in `overlap`/`probabilities` (read path). Only
  revisit in a follow-up commit if the wide-bond bench shows them hot.
- Use zero-copy faer `MatRef` views over `Site` data where strides allow; copies only
  where layout forces them.

## Testing

- Existing `tests/sv_equivalence.rs` proptest oracle exercises the lazy path
  automatically; verify the generator emits long-range 2q gates, extend it if not.
- New unit tests:
  - proptest invariant: `qubit_of_site` ∘ `site_of_qubit` = identity after arbitrary
    gate sequences;
  - interleaved gate → measure → gate with a non-identity permutation;
  - `dense_statevector` / `sample` / `probabilities` under a non-identity permutation;
  - reversed-order CNOT (`qubits = [hi, lo]`) on permuted sites — the MSB-convention
    risk case;
  - SWAP counter: on a long-range circuit, lazy count strictly below the
    always-swap-back count (deterministic expected numbers).
- ε=0 ⇒ exact truncation oracle (AC-2) — existing tests plus a wide-bond case.
- Thread invariance: results agree to 1e-10 between `RAYON_NUM_THREADS=1` and `=8`
  (not bit-exact — parallel SVD may round differently).

## Benchmarks and measurement

- `benches/long_range.rs` (existing): before/after wall-clock for AC-1; report
  `swaps_applied` alongside.
- New `benches/wide_bond.rs`: random brickwall, n≈24, depth saturating χ=256 — SVD
  matrices 512×512 where parallelism is visible. Thread sweep 1/2/4/8/16.
- Perf boxes: EPYC (`195.154.249.85`, verify idle first — shared CI runner) and Ryzen
  (`49.12.173.85`, no AVX-512; ship a fresh `git bundle` via scp and verify HEAD before
  trusting numbers).
- Local aarch64 numbers are development-only, not AC evidence.

## Delivery

One PR `[P3-09]`, commits staged lazy-perm first, multithreading second, each with its
own oracle and bench evidence. PR body: `Closes #125`, before/after criterion numbers,
SWAP-count table, thread-sweep table. BACKLOG.md checkboxes flipped in the same PR.

## Acceptance criteria mapping

| AC | Evidence |
|---|---|
| Lazy matches always-swap-back vs `NaiveSvBackend` to 1e-10, fewer SWAPs | sv_equivalence oracle (exact regime) + `swaps_applied` counter + long_range bench |
| Multithreaded SVD speedup on wide-bond bench, ε=0 oracle still passing | wide_bond thread sweep on EPYC + Ryzen; existing ε=0 tests green |

## Risks

- **MSB-orientation bug** in the site-based `apply_2q_adjacent` — the highest-risk
  change; covered by reversed-order + permuted-site oracle tests.
- **faer 0.24 API drift** for parallelism control (`set_global_parallelism` naming) —
  resolve at implementation time.
- **Truncation-pattern divergence** at small χ is expected, not a bug; oracles pin the
  exact regime only.
