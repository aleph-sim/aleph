# P3-08 — Word-parallel + SIMD `rowsum` (close the Stim gap) — Design

**Issue:** #124 (`[P3-08]` Stabilizer bit-slicing / O(n/64) tableau)
**Date:** 2026-06-09
**Depends on:** P3-02 (measurement/collapse). Motivated by P4-07, which measured the gap.

## Goal

Make the stabilizer backend's measurement hot path competitive with Stim by
word-parallelizing (and then SIMD-accelerating) `rowsum`, the CHP row-product +
phase-tracking routine. P4-07 measured aleph at **12.5× slower than Stim** on a
surface-code d=11 cycle; this ticket closes that gap with a hard target of
**≤ 2× Stim single-thread at d=11**.

## Why `rowsum`, and why now

P3-08 was deliberately deferred until "the Phase-4 benchmarks identify where the
stabilizer backend actually hurts" (per the golden rule: measure first). P4-07
did exactly that. Findings, grounded in the code:

- The tableau is **already bit-packed** row-major `Vec<u64>` (`bits.rs`
  `BitGrid`, `stride = ⌈cols/64⌉`). Single/two-qubit gate kernels already touch
  one/two words per row — O(rows), already minimal for row-major. **Gates are
  not the bottleneck and are out of scope.**
- `rowsum` (`tableau.rs:297`) is the lone scalar **per-bit** hot path. Both its
  loops use `BitGrid::get`/`set` one bit at a time, throwing away the packing:
  ```rust
  for j in 0..n { acc += g(x.get(i,j), z.get(i,j), x.get(h,j), z.get(h,j)); } // phase
  for j in 0..n { x.set(h,j, x.get(h,j)^x.get(i,j)); z.set(h,j, …); }          // XOR
  ```
- A random measurement calls `rowsum` into up to ~n rows, each `rowsum` O(n) bit
  ops → **O(n²) per random measurement**. A surface-d11 cycle has 120 ancilla
  measurements at n=241 → this dominates the 1.375 ms.
- Row words are **contiguous in memory** (row `h` = words `[h·stride,
  (h+1)·stride)`), so word-parallel and SIMD over the stride dimension are clean
  contiguous loads — no gather. (Gate kernels touch a single column across rows,
  which *would* be strided/gather — another reason gates are out of scope.)

## Locked decisions (from brainstorming)

1. **Scope = both phases in this ticket:** scalar word-parallel `rowsum`, then
   AVX-512 SIMD on top. Both gated by the Stim oracle.
2. **Hard AC:** surface-code d=11 cycle **≤ 2× Stim** single-thread on EPYC
   (in addition to the bit-exact correctness AC). Reaching it may require the
   SIMD phase — that is why SIMD is in-scope, not deferred.
3. **Layout stays row-major** (no column-major transpose) — row-major is already
   ideal for `rowsum` (XOR of contiguous words). The P3-08 backlog title says
   "columns (or rows)"; rows is the right choice here.
4. **Focus is `rowsum` only.** Gate kernels, `sample`, `pauli_eigenvalue`,
   `rows_anticommute` are out of scope (YAGNI; not the QEC bottleneck).

## The `rowsum` algorithm

`rowsum(h, i)` replaces Pauli row `h` with the product (row h)·(row i),
tracking the ± sign. Aaronson–Gottesman §3. Two parts:

### Part 1 — phase exponent (read-only, original bits)

```
acc = 2·sign[h] + 2·sign[i] + Σ_j g(x_i[j], z_i[j], x_h[j], z_h[j])
new sign[h] = (acc mod 4) == 2      // invariant: acc mod 4 ∈ {0, 2}
```

The scalar `g` (kept verbatim as oracle, `tableau.rs`):
```rust
fn g(x1,z1,x2,z2) -> i32 {           // (x1,z1)=row i bit, (x2,z2)=row h bit
    match (x1,z1) {
        (false,false) => 0,
        (true ,false) => z2*(2*x2-1),        // X_i:  +1 if zh&xh, -1 if zh&!xh
        (false,true ) => x2*(1-2*z2),        // Z_i:  +1 if xh&!zh, -1 if xh&zh
        (true ,true ) => z2 - x2,            // Y_i:  +1 if zh&!xh, -1 if xh&!zh
    }
}
```

**Word-parallel form.** For each word, build two bitmasks from the four
words `xi, zi, xh, zh` (per-bit booleans) and popcount them:

```
plus  = (xi & !zi & zh & xh)        // X_i, zh&xh
      | (!xi & zi & xh & !zh)       // Z_i, xh&!zh
      | (xi & zi & zh & !xh)        // Y_i, zh&!xh
minus = (xi & !zi & zh & !xh)       // X_i, zh&!xh
      | (!xi & zi & xh & zh)        // Z_i, xh&zh
      | (xi & zi & xh & !zh)        // Y_i, xh&!zh
acc += popcount(plus) - popcount(minus)   // summed over all stride words
```

This is exactly Σ_j g over the word's 64 bits (each bit contributes +1, −1, or 0,
matching the three g cases; all other (xi,zi) combinations contribute 0). Derived
once here; the bit-exact oracle test (below) is the actual guarantee.

### Part 2 — bit XOR (write)

```
for w in 0..stride: x_h[w] ^= x_i[w];  z_h[w] ^= z_i[w];
```

**Ordering:** Part 1 reads the *original* bits of rows h and i, so it must run
before Part 2 overwrites row h. Implementation: phase pass first (read-only),
then XOR pass.

## Architecture / components

- **`crates/aleph-stab/src/rowsum.rs`** (new): the word-level kernels, with one
  clear responsibility each.
  - `rowsum_words(xh: &mut [u64], zh: &mut [u64], xi: &[u64], zi: &[u64]) -> i64`
    — scalar word-parallel; does the XOR in-place on `xh`/`zh` and returns the
    phase `acc` contribution (Σ g over words; caller adds the `2·sign` terms and
    reduces mod 4). Pure, portable, no `unsafe`.
  - `rowsum_avx512(...)` — same signature/contract, AVX-512 + VPOPCNTQ, behind
    `#[target_feature(enable="avx512f,avx512vpopcntdq")]` `unsafe fn` with a
    SAFETY block; only called after `is_x86_feature_detected!`.
- **`crates/aleph-stab/src/bits.rs`**: add contiguous-slice accessors to
  `BitGrid`:
  - `row_words(&self, r: usize) -> &[u64]` (the `stride` words of row r)
  - `row_words_mut(&mut self, r: usize) -> &mut [u64]`
  - A helper to borrow two rows' word-slices for the in-place XOR (row h mut, row
    i shared) — e.g. via `split_at_mut` on the backing `words`, or two separate
    immutable reads of row i copied/aliased carefully. (The plan resolves the
    borrow mechanics; both rows live in the same `Vec<u64>`.)
- **`crates/aleph-stab/src/tableau.rs`**: `Tableau::rowsum` becomes a thin
  dispatcher:
  - default → `rowsum_words`,
  - on `is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")` → `rowsum_avx512`.
  - Compute `acc = 2·sign[h] + 2·sign[i] + <kernel phase>`, set
    `sign[h] = acc.rem_euclid(4) == 2` (keep the `debug_assert!(m==0||m==2)`).
  - The old per-bit body is preserved verbatim as **`rowsum_scalar`** (private,
    `#[cfg(test)]`-reachable) for the oracle diff. The scalar `g` stays.

## Testing / correctness gate

1. **Bit-exact diff oracle** (the safety net for the phase formula): a proptest
   over random tableaux (random valid CHP states or random row pairs) asserting
   `rowsum_words` produces identical `x`/`z` bits **and** sign as the preserved
   `rowsum_scalar`, for many (h, i) pairs. On EPYC, the same diff for
   `rowsum_avx512`.
2. **Stim oracles unchanged & green:** `crates/aleph-stab/tests/stim_oracle.rs`,
   `stim_measure_oracle.rs`, and `benches/tests/surface_code_stim_oracle.rs`
   (d=3..11) — independent confirmation of the phase formula end-to-end.
3. **Existing invariants:** symplectic-invariant proptests and
   `rowsum_bit_involution` continue to pass.

## Benchmarks / acceptance criteria

- [ ] **AC-1 (correctness):** `rowsum_words` (and `rowsum_avx512` on EPYC) are
  bit-for-bit + sign identical to the preserved scalar `rowsum`, via proptest;
  all Stim oracles pass at d=3..11.
- [ ] **AC-2 (hard perf):** surface-code d=11 cycle **≤ 2× Stim** single-thread
  on EPYC (re-using the P4-07 `phase4_surface_code` bench vs `stim_timing.py`).
- [ ] **AC-3 (reporting):** criterion before/after for the measurement path,
  reported honestly; update the `docs/perf/surface_code.md` row(s) with the new
  numbers + a note on the speedup. Add a measurement-heavy Clifford microbench
  if the surface bench alone doesn't isolate `rowsum`.

## Risks & mitigations

- **The phase word-formula is the bug-prone part** → mitigated by the bit-exact
  scalar-oracle proptest (catches any per-bit disagreement) and the independent
  Stim oracle. Both must be green before the scalar `rowsum` is removed from the
  hot path.
- **SIMD only builds/validates on x86_64** (local dev is aarch64) → use
  `cargo check --target x86_64-unknown-linux-gnu` to validate compilation, full
  validation on EPYC; GitHub-hosted `test linux` is x86_64 and gates it on PR.
  Follow the aleph-sv `is_x86_feature_detected!` + scalar-fallback pattern and
  the P2-06 `avx512vpopcntdq` precedent (note the underscore-free feature name).
- **AC-2 may not be reachable even with SIMD** (Stim is hand-tuned AVX; EPYC
  frequency-throttles) → if so, report the honest achieved ratio and the
  profiling evidence; the target is ≤2×, the deliverable is the honest number.
  Do **not** weaken the correctness gate to chase the perf number.

## Out of scope (YAGNI)

- Gate-kernel SIMD (strided/gather, not the QEC bottleneck).
- `sample`, `pauli_eigenvalue`, `rows_anticommute` word-parallelization (may
  reuse the new slice accessors opportunistically, but no dedicated work).
- Column-major / transposed layout.
- Multi-threading (measurement is inherently sequential per qubit; out of scope).

## File manifest

| File | Action |
|------|--------|
| `crates/aleph-stab/src/rowsum.rs` | new — `rowsum_words` + `rowsum_avx512` |
| `crates/aleph-stab/src/bits.rs` | add `row_words`/`row_words_mut` slice accessors |
| `crates/aleph-stab/src/tableau.rs` | `rowsum` → dispatcher; preserve scalar as `rowsum_scalar` (test oracle) |
| `crates/aleph-stab/src/lib.rs` | `mod rowsum;` |
| `crates/aleph-stab/tests/` or in-module | bit-exact `rowsum_words` ≡ scalar proptest |
| `benches/benches/phase4_surface_code.rs` | reused for AC-2 (no change expected); optional measurement microbench |
| `docs/perf/surface_code.md` + `docs/perf/data/surface-*.json` | re-measure & update row(s) |
| `BACKLOG.md` | tick P3-08 ACs |
