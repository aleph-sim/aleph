# P3-11 — Stabilizer word-parallel gate kernels (lazy dual-orientation tableau)

**Date:** 2026-06-09
**Issue:** #135 (BACKLOG P3-11), depends on P3-08 (#124, PR #134)
**Status:** Design approved → implementation

## Problem

P3-08 word-parallelized `rowsum`, which flipped the surface-code d=11 bottleneck.
A `perf record` of the cycle now attributes **`Tableau::cnot` 70.9% + `Tableau::h`
15.3% ≈ 86%**; `measure`/`rowsum` are down to ~5–11%. P3-08 took surface-d11 from
**12.52× → 7.66× vs Stim** (1.63× cycle speedup); the remaining gap is almost
entirely the gate kernels.

Each Clifford gate touches **column `a`** (and `b` for CNOT) across all `2n+1`
rows: it reads/modifies the single bit at word `a>>6`, mask `1<<(a&63)`, in every
row. Under the current **row-major** `BitGrid` that is a strided, single-bit,
per-row update (≈`2n` scalar iterations per gate; ~482 at d=11). The current
kernels already hoist the word/mask and use branchless word arithmetic (P3-01),
but they are still **one row per step** — bounded.

The core tension: **`rowsum` wants row-major** (contiguous row XOR — exactly why
P3-08 is fast), **gates want column-major** (a gate's target column should be a
contiguous `u64` span so the per-row update is word-parallel/SIMD *across rows*).

## Goal / success bar

**Push hard for the stretch target ≤ 2× Stim @ d=11**, via a correct layout
change. Correctness is non-negotiable and gated exactly as P3-08: bit-for-bit
identical to the preserved scalar kernels (proptest) with all Stim oracles green
at d=3..11. Perf is reported honestly even if ≤2× is not reached.

## Chosen approach — lazy dual-orientation (Stim-style)

`Tableau` carries an **orientation flag** `{RowMajor, ColMajor}`. The physical
storage is the same `BitGrid`, with transposed dimensions per orientation:

- **RowMajor**: `BitGrid(2n+1, n)` — each generator row is contiguous. Required by
  `rowsum`/`measure`/`pauli_eigenvalue`/`stabilizers`/readout. Preserves P3-08.
- **ColMajor**: `BitGrid(n, 2n+1)` — "row `j`" of the grid holds **column `j`**:
  the bits of column `j` across all `2n+1` generators, contiguous. Required by
  gates → word/SIMD across rows.

`sign` becomes a **packed bit-vector** of length `2n+1` (replacing `Vec<bool>`).
The generator-row axis is preserved under transpose, so `sign` is **never
transposed** and aligns word-for-word with a ColMajor column span — enabling
word-parallel sign updates inside the gate kernels. `measure`/`rowsum` access
individual sign bits via get/set, unchanged in behaviour.

Each public method first calls `ensure_*`:

- gates (`h`, `s`, `cnot`, `x/y/z`, `sdg`, `cz`, `swap`, `iswap`, …) → `ensure_col_major()`
- `measure`, `sample`, `expectation_value`, `stabilizers`/`destabilizers`,
  `rows_anticommute`, test read accessors → `ensure_row_major()`

A transpose runs **only on orientation change**. A surface-code cycle is a
gate-batch followed by a measure-batch → **~2 transposes/cycle**, each
O((2n+1)·n / 64) word-ops, amortized over thousands of gate ops. **Risk:** a
pathological `gate, measure, gate, measure, …` interleave transposes on every op
(still *correct*, just slow). The surface-code target is batched; this risk is
documented in the ADR.

### Gate kernels (ColMajor, word-parallel over `W = ceil((2n+1)/64)` words)

For column `a` (and `b`), let `xcol(a)`/`zcol(a)` be the `W`-word spans:

- **H(a):** `sign ^= xcol(a) & zcol(a)`; swap `xcol(a) ↔ zcol(a)`.
- **S(a):** `sign ^= xcol(a) & zcol(a)`; `zcol(a) ^= xcol(a)`.
- **CNOT(a,b):** `sign ^= xcol(a) & zcol(b) & ~(xcol(b) ^ zcol(a))`;
  `xcol(b) ^= xcol(a)`; `zcol(a) ^= zcol(b)`.
- **X(a):** `sign ^= zcol(a)`. **Z(a):** `sign ^= xcol(a)`.
  **Y(a):** `sign ^= xcol(a) ^ zcol(a)`.

All updates are word-parallel over `W` words (chunk-loop for any `n`; ~8 words at
d=11). Composed gates (`sdg = s³`, `cz = h·cnot·h`, `swap`, `iswap`, `iswap_dg`)
route through these primitives, so a gate batch never transposes mid-sequence.
The CNOT sign formula reads `xcol(a)`/`zcol(b)` **before** writing `xcol(b)`/
`zcol(a)`; since `a ≠ b` the spans are disjoint and a single pass is safe (the
sign word must be computed from original `xcol(b)`/`zcol(a)` before the XORs).

### Transpose kernel

Bit-transpose of the logical `(2n+1) × n` matrix ↔ `n × (2n+1)`, for `x` and `z`
independently (`sign` untouched). Start with a straightforward correct
bit-by-bit transpose as the reference; add a blocked 64×64 transpose
(Warren, *Hacker's Delight* §7-3) behind the same API, validated by a diff test
against the reference and a round-trip property (`transpose∘transpose = id`).

### SIMD

AVX-512 variants of the gate kernels (`_mm512_and/xor/andnot/or` over `__m512i`,
8×`u64` = 512 generator-rows per step) plus a `*_dispatch` mirroring P3-08's
`rowsum_dispatch` (`is_x86_feature_detected!("avx512f")`). The scalar word kernel
is the fallback and the bit-exact reference. The transpose may also get an
AVX-512 path if profiling shows it on the hot path post-change.

## Correctness gate

1. **Preserve** the current row-major scalar gate kernels as `#[cfg(test)]`
   reference (`h_scalar`, `s_scalar`, `cnot_scalar`, …) operating on a RowMajor
   tableau.
2. **Equivalence proptest:** drive a random Clifford circuit (H/S/CNOT/X/Y/Z mix)
   through both the new ColMajor kernels and the preserved row-major reference;
   assert identical full tableaux (x, z, sign) after normalizing orientation.
   Use a generic (non-|0…0⟩) start where applicable (P1-13 lesson).
3. **Transpose tests:** round-trip identity; blocked-vs-scalar bit-exact diff;
   small and multi-word `n` (e.g. n ∈ {1,2,7,63,64,65,127,241}).
4. **AVX-512 bit-exact** vs scalar word kernel (skipped when feature absent, as in
   `rowsum`'s test), run on EPYC.
5. **All Stim oracles green at d=3..11**, not weakened.

## Performance validation

- Criterion before/after on the surface-code 1-cycle benchmark (P4-07) and any
  gate microbench; report on a **verified-idle** EPYC box (check `uptime` ~0,
  `pgrep cargo bench`). Re-measure Stim apples-to-apples.
- Honest restate of the aleph/Stim d=11 ratio; note whether ≤2× is reached and,
  if not, the new bottleneck from a fresh `perf record`.

## ADR

`docs/decisions/0013-stabilizer-dual-orientation-tableau.md`: row-major vs
column-major vs dual-orientation trade-off; the transpose amortization argument;
the interleave-pathology caveat; why `sign` is orientation-invariant.

## Out of scope

- Transpose-free measurement (a fully bit-sliced measurement à la Stim) — only if
  the transpose proves to dominate after measurement; otherwise the row-major
  P3-08 path stays.
- Multi-threading the gate batch (Phase-2 territory; stabilizer is single-thread).

## Task breakdown (for writing-plans)

1. **`PackedBits` + transpose primitive** in `bits.rs` (or a new module): packed
   sign bit-vector type; bit-transpose `(rows×cols)↔(cols×rows)` scalar reference
   + blocked 64×64; tests (round-trip, diff, multi-word n).
2. **Orientation infrastructure** in `Tableau`: `orientation` flag, `sign` →
   packed, `ensure_row_major`/`ensure_col_major` (transpose on change). Keep gates
   row-major for now (no kernel change) so all existing tests stay green — proves
   the transpose round-trips end-to-end.
3. **ColMajor scalar word-parallel gate kernels** (H/S/CNOT/X/Y/Z); preserve old
   row-major kernels as `#[cfg(test)]` reference; equivalence proptest.
4. **AVX-512 gate kernels + dispatch**; bit-exact vs scalar on EPYC.
5. **EPYC validation, criterion before/after, ADR 0013, perf report update**,
   honest aleph/Stim d=11 restate.
