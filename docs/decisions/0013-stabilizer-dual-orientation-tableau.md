# 0013 — Stabilizer dual-orientation tableau (P3-11)

## Status

Accepted, 2026-06-10.

## Context

P3-08 word-parallelized the CHP measurement hot path (`rowsum`). A
`perf record` of the surface-code d=11 cycle afterwards showed the
bottleneck had flipped to the **gate** kernels: `Tableau::cnot` 70.9% +
`Tableau::h` 15.3% ≈ 86%, with `measure`/`rowsum` down to ~5–11%
(`docs/perf/surface_code.md`, P3-08 addendum). The cycle was gate-bound,
and surface-d11 sat at **7.66× Stim** — short of the P3-08 design's hard
target of ≤ 2× Stim.

The structural reason is a **layout tension**:

- A Clifford **gate** touches one qubit **column** (`a`, and `b` for
  CNOT) across all `2n+1` generator rows. Under the row-major `BitGrid`
  (P3-01/P3-08) that column is a strided, single-bit-per-row access —
  the gate runs `2n` scalar iterations, each twiddling one bit at word
  `a>>6`. This is exactly the opposite of what `rowsum` wants.
- `rowsum` (the measurement path) XORs two full generator **rows**, which
  is contiguous and word-parallel precisely *because* the layout is
  row-major. P3-08 depends on this.

So `rowsum` wants row-major; gates want column-major. The two cannot both
be contiguous in one fixed layout.

## Decision

Make the tableau **lazily dual-orientation** (Stim's bit-sliced idea):

- `Tableau` carries an `Orientation { RowMajor, ColMajor }` flag. The
  physical `x`/`z` storage is the same `BitGrid` with transposed
  dimensions: RowMajor `(2n+1) × n`, ColMajor `n × (2n+1)` (grid "row"
  `j` = qubit column `j`'s bits over all `2n+1` generators, contiguous).
- **Gates** run in ColMajor: a column is a contiguous
  `W = ceil((2n+1)/64)`-word span, so H/S/CNOT/Paulis update the whole
  column **word-parallel** (scalar `u64` kernel + an AVX-512 `avx512f`
  kernel, runtime-dispatched, mirroring P3-08's `rowsum_dispatch`).
- **`rowsum`/`measure`/readout** run in RowMajor (P3-08 unchanged).
- Each public method ensures the orientation it needs (`ensure_col_major`
  for gates, `ensure_row_major` for measurement/readout); a **bit-
  transpose** runs only on an orientation change. A surface-code cycle is
  a gate-batch then a measure-batch → **~2 transposes/cycle**, amortized
  over thousands of gate ops.
- `sign` is a packed `BitVec` over the generator-row axis, which the
  transpose preserves, so `sign` is **orientation-invariant** (never
  transposed) and aligns word-for-word with a ColMajor column span —
  enabling word-parallel sign updates inside the gate kernels.
- The transpose itself is a **blocked 64×64** bit-transpose (Warren,
  *Hacker's Delight* 2nd ed. §7-3), kept bit-exact against a scalar
  reference by a diff test.

### Alternatives rejected

- **Stay row-major, de-scalarize the gate loop (option 1).** The P3-01
  hoisting already does this; it is still one row per step. Bounded
  upside, cannot reach ≤2× Stim. Rejected.
- **Column-major only.** Gates become fast but `rowsum` degrades to a
  per-`(h,i)` strided single-bit gather/scatter across columns →
  O(n²)-ish measurement, regressing the measure-heavy QEC workload that
  is the whole point. Rejected.
- **Always-synced dual copies (write-through both layouts).** No
  transpose latency, but every gate pays the row-major write too,
  defeating the gate win. Rejected.

## Consequences

### Measured outcome (EPYC 8124P, single-thread, idle, target-cpu=native)

Surface-code cycle, before (pre-P3-11 `main`) vs after (P3-11), same box:

| d | qubits | before (ms) | after (ms) | cycle speedup | aleph/Stim before → after |
|--:|-------:|------------:|-----------:|--------------:|--------------------------:|
| 3 | 17 | 0.004 | 0.003 | 1.33× | 0.53× → 0.40× |
| 5 | 49 | 0.033 | 0.011 | 3.17× | 2.55× → 0.81× |
| 7 | 97 | 0.133 | 0.037 | 3.57× | 5.84× → 1.63× |
| 9 | 161 | 0.370 | 0.092 | 4.01× | 6.60× → 1.65× |
| 11 | 241 | 0.848 | 0.181 | **4.69×** | 7.71× → **1.64×** |

**The ≤ 2× Stim hard target is met** at d=7, 9, 11 (1.63–1.65×); at
d=3, 5 aleph is now *faster* than Stim. P3-11 takes surface-d11 from
7.66× to **1.64× Stim** — the gap the P3-08 design left open.

A fresh `perf record` of the d=11 cycle confirms the gate kernels are no
longer the bottleneck:

```
29.7%  BitGrid::transpose   (the new orientation-bridge cost)
26.0%  Tableau::measure
19.1%  Tableau::zero_row     (still scalar per-bit column clear)
13.5%  Tableau::copy_row     (still scalar per-bit row copy)
 5.5%  Tableau::rowsum
 1.6%  Tableau::cnot         (was 70.9%)
 0.4%  Tableau::h            (was 15.3%)
```

Gates dropped from ~86% to ~2%. The remaining time is the transpose
bridge (~30%) plus the measurement-collapse helpers (`zero_row`/
`copy_row`, still row-major per-bit) — the natural next levers, but ≤2×
is already reached so they are deferred.

### Trade-offs / caveats

- **Interleave pathology.** A pathological `gate, measure, gate, measure,
  …` sequence transposes on every op. This stays *correct* (each method
  ensures its orientation) but loses the amortization. The surface-code
  target is batched (all gates, then all measures), so this does not
  arise there; it is a known cost for adversarial interleavings.
- **Scratch row.** The ColMajor gate kernels span all `2n+1` rows, so
  they dirty the scratch row `2n` (the row-major path looped `0..2n`).
  This is safe because every scratch-row consumer (`measure`'s
  deterministic branch, `pauli_eigenvalue`) calls `zero_row(2n)` before
  reading it. Documented at the kernels and enforced by a
  `debug_assert!(RowMajor)` on the row-major-only helpers.

### Correctness gate

- ColMajor scalar + AVX-512 gate kernels are **bit-exact** vs the
  preserved `#[cfg(test)]` row-major `*_scalar` references (proptest over
  random Clifford circuits, n up to 130; AVX-512 path exercised on EPYC).
- The blocked transpose is bit-exact vs a scalar reference (diff test,
  including non-64-multiple and the 483×241 surface-d11 shape).
- All Stim oracles green at **d = 3..11** (`surface_code_stim_oracle`,
  `stim_oracle`, `stim_measure_oracle`), plus the local `sv_equivalence`
  state-vector cross-check that exercises the gate→measure transpose
  bridge end-to-end.

## References

- Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits"
  (2004), §2–3.
- Warren, *Hacker's Delight* 2nd ed., §7-3 (bit-matrix transpose).
- P3-08 design + perf addendum (`docs/perf/surface_code.md`); PR #134.
- Stim (quantumlib/Stim) bit-sliced tableau + on-demand transpose.
