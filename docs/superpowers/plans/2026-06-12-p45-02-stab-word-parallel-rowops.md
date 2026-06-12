# P4.5-02 Stabilizer: word-parallel transpose + zero_row/copy_row — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last CPU-parity gap (stabilizer surface-d11 at 1.64× Stim) by word-parallelizing the two remaining scalar hot spots: `Tableau::zero_row`/`copy_row` (~33% of the d=11 cycle, per-bit loops over contiguous rows) and the 64×64 transpose block kernel (~30%, scalar delta-swap → AVX-512 in-register network).

**Architecture:** Three independent micro-optimizations inside `aleph-stab`, no API or layout changes. (1) In RowMajor orientation a generator row is `stride = ceil(n/64)` contiguous `u64` words, so `zero_row` becomes a word `fill(0)` and `copy_row` a `copy_from_slice` via the existing `row_pair_mut`. (2) The blocked transpose keeps its load/store loop but the 64×64 kernel gains an AVX-512 variant: the whole block is 64 u64 = 8 zmm registers, and Warren's delta-swap network runs vectorized (levels j=32/16/8 pair whole vectors; j=4/2/1 pair lanes via permute+mask-blend). Runtime dispatch mirrors `rowsum_dispatch` (P3-08) / `gates.rs` (P3-11): `is_x86_feature_detected!("avx512f")`, scalar fallback everywhere else.

**Tech Stack:** Rust 1.89, `core::arch::x86_64` AVX-512F intrinsics, criterion, Stim (Python) baseline. Validation on EPYC 8124P (`ssh root@195.154.249.85`, AVX-512; local machine is aarch64 → scalar paths only).

**Issue:** #155. Branch: `p4.5-02-stab-word-parallel-rowops` off `origin/main`, in the main checkout (NO worktree). PR title: `[P4.5-02] Stabilizer: word-parallel transpose + zero_row/copy_row`, body `Closes #155`.

**Acceptance criteria (BACKLOG):**
- surface-d11 cycle time improves; target ≤ 1.2× Stim, else documented structural verdict with profile evidence.
- Bit-exact scalar↔SIMD equivalence tests; Stim oracles d=3..11 green.
- Before/after criterion numbers (EPYC) in the PR.

**Current profile (d=11 cycle, post-P3-11, from ADR 0013):**
```
29.7%  BitGrid::transpose    ← Task 3
26.0%  Tableau::measure      (out of scope)
19.1%  Tableau::zero_row     ← Task 1
13.5%  Tableau::copy_row     ← Task 2
 5.5%  Tableau::rowsum
 2.0%  gates
```
Expected: Tasks 1+2 alone remove ~30% of the cycle (1.64× → ~1.15× Stim); Task 3 adds headroom.

## File Structure

- Modify: `crates/aleph-stab/src/tableau.rs` — `zero_row` (~line 526), `copy_row` (~line 510); keep old bodies as `#[cfg(test)] *_scalar` references; new equivalence tests in the in-file `mod tests`.
- Modify: `crates/aleph-stab/src/bits.rs` — `transpose()` (~line 106) takes a kernel fn pointer; new `transpose64_kernel()` dispatcher + `transpose64_avx512()`; new gated equivalence test in the in-file `mod tests`.
- Modify: `docs/perf/surface_code.md` — P4.5-02 addendum (Task 6, after EPYC numbers exist).

No new files. No changes outside `aleph-stab` + docs.

---

### Task 0: Branch + claim issue

- [ ] **Step 0.1: Create branch from origin/main in the main checkout**

```bash
cd /Users/ex/GitHub/aleph
git fetch origin
git checkout -b p4.5-02-stab-word-parallel-rowops origin/main
```

- [ ] **Step 0.2: Claim the issue**

```bash
gh issue comment 155 --body "Working on this"
```

---

### Task 1: Word-parallel `zero_row`

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs:523-536` (`zero_row`)
- Test: same file, `mod tests`

**Why this is safe:** in RowMajor, grid row `r` is exactly the generator row — `stride` contiguous words in `x` and in `z`. `BitGrid` guarantees padding bits past col `n` are already zero, so `fill(0)` preserves the padding invariant.

- [ ] **Step 1.1: Keep the old per-bit body as a test reference.** In `tableau.rs`, directly below the current `zero_row`, add:

```rust
/// Pre-P4.5-02 per-bit reference, kept for the equivalence test in this file.
#[cfg(test)]
fn zero_row_scalar(&mut self, r: usize) {
    debug_assert!(
        self.orientation == Orientation::RowMajor,
        "zero_row_scalar needs RowMajor"
    );
    for j in 0..self.n {
        self.x.set(r, j, false);
        self.z.set(r, j, false);
    }
    self.sign.set(r, false);
}
```

- [ ] **Step 1.2: Add the test helpers + equivalence test** in `tableau.rs` `mod tests`. The helper scrambles a tableau with a deterministic random Clifford circuit so rows and signs are non-trivial, then forces RowMajor (gates leave it ColMajor):

```rust
/// Deterministically scrambled tableau in RowMajor orientation.
fn random_row_major_tableau(n: usize, seed: u64) -> Tableau {
    use rand::Rng;
    let mut t = Tableau::new(n);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..4 * n {
        let a = rng.gen_range(0..n);
        match rng.gen_range(0..4) {
            0 => t.h(a).unwrap(),
            1 => t.s(a).unwrap(),
            2 if n > 1 => {
                let mut b = rng.gen_range(0..n);
                if b == a {
                    b = (b + 1) % n;
                }
                t.cnot(a, b).unwrap();
            }
            _ => t.x_gate(a).unwrap(),
        }
    }
    t.ensure_row_major();
    t
}

/// Full-state compare incl. padding words (word-level, not per-bit).
fn assert_tableaus_bit_identical(a: &Tableau, b: &Tableau, ctx: &str) {
    for row in 0..2 * a.n + 1 {
        assert_eq!(a.sign.get(row), b.sign.get(row), "sign {ctx} row={row}");
        assert_eq!(a.x.row_words(row), b.x.row_words(row), "x {ctx} row={row}");
        assert_eq!(a.z.row_words(row), b.z.row_words(row), "z {ctx} row={row}");
    }
}

#[test]
fn zero_row_matches_scalar_reference() {
    // Irregular n (not multiples of 64) per the P4.5-02 testing requirement;
    // 241 = surface d=11. Rows cover destab/stab/scratch.
    for &n in &[3usize, 70, 130, 241] {
        let t0 = random_row_major_tableau(n, 0xA5A5 + n as u64);
        for &r in &[0usize, n / 2, n, 2 * n] {
            let mut a = t0.clone();
            let mut b = t0.clone();
            a.zero_row(r);
            b.zero_row_scalar(r);
            assert_tableaus_bit_identical(&a, &b, &format!("zero_row n={n} r={r}"));
        }
    }
}
```

- [ ] **Step 1.3: Run the test — must pass against the unchanged impl** (sanity: at this point `zero_row` ≡ old body, so this validates the harness itself):

Run: `cargo test -p aleph-stab zero_row_matches_scalar_reference`
Expected: PASS

- [ ] **Step 1.4: Swap in the word-parallel implementation.** Replace the body of `zero_row` (keep `zero_row_scalar` as-is):

```rust
/// Reset a row to the identity Pauli with `+` sign. Word-parallel: in
/// RowMajor a generator row is `stride` contiguous words per grid, so the
/// clear is a word fill (the per-bit loop was ~19% of the surface-d11
/// cycle). Padding bits past col `n` are already zero, so `fill(0)`
/// preserves the BitGrid padding invariant.
///
/// Precondition: RowMajor (direct `(row, col)` grid access).
fn zero_row(&mut self, r: usize) {
    debug_assert!(
        self.orientation == Orientation::RowMajor,
        "zero_row needs RowMajor"
    );
    self.x.row_words_mut(r).fill(0);
    self.z.row_words_mut(r).fill(0);
    self.sign.set(r, false);
}
```

- [ ] **Step 1.5: Run the equivalence test — must still pass:**

Run: `cargo test -p aleph-stab zero_row_matches_scalar_reference`
Expected: PASS

- [ ] **Step 1.6: Mutation-test the oracle** (P1-08 lesson: an equivalence test must be shown to fail on a broken kernel). Temporarily comment out `self.z.row_words_mut(r).fill(0);`, run the test, confirm FAIL on a `z` row compare. Restore the line, re-run, confirm PASS.

- [ ] **Step 1.7: Run the crate suite:**

Run: `cargo test -p aleph-stab`
Expected: all green (measure/oracle tests exercise `zero_row` on both branches).

- [ ] **Step 1.8: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P4.5-02] zero_row: per-bit loop -> word fill

In RowMajor a generator row is stride contiguous words, so the clear is
a fill(0) per grid plus the sign bit. The per-bit loop was 19.1% of the
surface-d11 cycle (ADR 0013 profile). Old body kept as #[cfg(test)]
zero_row_scalar; equivalence test compares full words incl. padding on
irregular n (70, 130, 241), mutation-tested."
```

---

### Task 2: Word-parallel `copy_row`

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs:507-521` (`copy_row`)
- Test: same file, `mod tests`

**Why this is safe:** same contiguity argument; `row_pair_mut(dst, src)` (already used by `rowsum`) gives simultaneous `&mut dst`/`&src` row slices and `debug_assert`s `dst != src`. The only caller is `measure` (`copy_row(p - n, p)`, always distinct rows — verify with `grep -n "copy_row(" crates/aleph-stab/src/*.rs` and check no other caller appeared). `src`'s padding bits are zero, so the word copy preserves the invariant.

- [ ] **Step 2.1: Keep the old body as a test reference.** Below `copy_row`, add:

```rust
/// Pre-P4.5-02 per-bit reference, kept for the equivalence test in this file.
#[cfg(test)]
fn copy_row_scalar(&mut self, dst: usize, src: usize) {
    debug_assert!(
        self.orientation == Orientation::RowMajor,
        "copy_row_scalar needs RowMajor"
    );
    for j in 0..self.n {
        self.x.set(dst, j, self.x.get(src, j));
        self.z.set(dst, j, self.z.get(src, j));
    }
    let s = self.sign.get(src);
    self.sign.set(dst, s);
}
```

- [ ] **Step 2.2: Add the equivalence test** (reuses Task 1's helpers; covers `dst < src`, `dst > src`, adjacent rows, scratch row):

```rust
#[test]
fn copy_row_matches_scalar_reference() {
    for &n in &[3usize, 70, 130, 241] {
        let t0 = random_row_major_tableau(n, 0xC0FE + n as u64);
        for &(dst, src) in &[(0usize, 2 * n), (n / 2, n / 2 + 1), (2 * n, 0), (n, n - 1)] {
            let mut a = t0.clone();
            let mut b = t0.clone();
            a.copy_row(dst, src);
            b.copy_row_scalar(dst, src);
            assert_tableaus_bit_identical(&a, &b, &format!("copy_row n={n} {dst}<-{src}"));
        }
    }
}
```

- [ ] **Step 2.3: Run — must pass against the unchanged impl:**

Run: `cargo test -p aleph-stab copy_row_matches_scalar_reference`
Expected: PASS

- [ ] **Step 2.4: Swap in the word-parallel implementation:**

```rust
/// Copy a full generator row (x bits, z bits, sign) from `src` to `dst`.
/// Word-parallel via `row_pair_mut` (a RowMajor generator row is `stride`
/// contiguous words; the per-bit loop was ~13.5% of the surface-d11
/// cycle). `src`'s padding bits are zero, so the word copy preserves the
/// BitGrid padding invariant.
///
/// Precondition: RowMajor; `dst != src` (enforced by `row_pair_mut`).
fn copy_row(&mut self, dst: usize, src: usize) {
    debug_assert!(
        self.orientation == Orientation::RowMajor,
        "copy_row needs RowMajor"
    );
    let (xd, xs) = self.x.row_pair_mut(dst, src);
    xd.copy_from_slice(xs);
    let (zd, zs) = self.z.row_pair_mut(dst, src);
    zd.copy_from_slice(zs);
    let s = self.sign.get(src);
    self.sign.set(dst, s);
}
```

- [ ] **Step 2.5: Run the test — must still pass:**

Run: `cargo test -p aleph-stab copy_row_matches_scalar_reference`
Expected: PASS

- [ ] **Step 2.6: Mutation-test the oracle.** Temporarily drop the sign propagation (comment out the last two lines), confirm the test FAILS on a sign compare (the scrambled tableau has non-trivial signs); restore, confirm PASS.

- [ ] **Step 2.7: Run the crate suite:**

Run: `cargo test -p aleph-stab`
Expected: all green.

- [ ] **Step 2.8: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P4.5-02] copy_row: per-bit loop -> word copy via row_pair_mut

copy_from_slice over the contiguous RowMajor row words in x and z, plus
the sign bit. Was 13.5% of the surface-d11 cycle (ADR 0013 profile).
Old body kept as #[cfg(test)] copy_row_scalar; equivalence test covers
dst<src, dst>src, adjacent and scratch rows on irregular n,
mutation-tested on the sign path."
```

---

### Task 3: AVX-512 64×64 transpose kernel + dispatch

**Files:**
- Modify: `crates/aleph-stab/src/bits.rs` — `transpose()` (~line 106), new `transpose64_kernel()` + `transpose64_avx512()` next to `transpose64` (~line 178)
- Test: same file, `mod tests`

**Design:** the 64×64 block (64 u64) is exactly 8 zmm registers. Warren's delta-swap network (`transpose64`) runs fully in registers:
- Levels j=32/16/8: rows `k` and `k+j` sit at the same lane of vectors `k/8` and `k/8 + j/8` → plain vector-pair delta-swap, no permutes.
- Levels j=4/2/1: the partner row `lane ^ j` lives in the same vector → one `vpermq` to materialize partners, mask-blends to pick each pair's low/high row, one delta-swap expression updates all 8 lanes.
Shift-by-`j` uses `_mm512_s{r,l}lv_epi64` with a `set1` count so the levels stay table-driven (the `_epi64` immediate-shift intrinsics need const generics). Only `avx512f` is required. Dispatch hoists kernel selection out of the block loop into a fn pointer — one indirect call per 64×64 block (~hundreds of word ops), negligible, and the scalar path keeps working on aarch64/Ryzen.

- [ ] **Step 3.1: Write the gated equivalence test first** (in `bits.rs` `mod tests`; mirrors `rowsum`'s `avx512_matches_reference_when_available` pattern — empty body off x86_64, skip-with-message when the CPU lacks AVX-512):

```rust
#[test]
fn transpose64_avx512_matches_scalar_when_available() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping transpose64_avx512 test: avx512f unavailable");
            return;
        }
        let mut rng = 0xABCD_EF01_2345_6789u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for case in 0..64 {
            let mut a = [0u64; 64];
            match case {
                0 => {}                       // all-zero
                1 => a = [u64::MAX; 64],      // all-one
                2 => a[63] = 1u64 << 63,      // lone corner bit
                3 => a[0] = 1,                // other corner
                _ => a.iter_mut().for_each(|w| *w = next()),
            }
            let mut want = a;
            super::transpose64(&mut want);
            let mut got = a;
            unsafe { super::transpose64_avx512(&mut got) };
            assert_eq!(got, want, "case {case}");
        }
    }
}
```

- [ ] **Step 3.2: Verify it fails to compile** (function doesn't exist yet) — this is the TDD "red":

Run: `cargo check --target x86_64-unknown-linux-gnu -p aleph-stab --tests` (cross-check from aarch64; P2-04 lesson — this target compiles the SIMD path locally)
Expected: error: cannot find function `transpose64_avx512`

- [ ] **Step 3.3: Implement the kernel + dispatcher.** Add below `transpose64` in `bits.rs`:

```rust
/// Pick the 64×64 block kernel once per transpose: AVX-512 when available
/// (`is_x86_feature_detected!` caches the cpuid result), scalar delta-swap
/// otherwise and on non-x86.
fn transpose64_kernel() -> fn(&mut [u64; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f verified present immediately above. The
            // non-capturing closure coerces to `fn`.
            return |a| unsafe { transpose64_avx512(a) };
        }
    }
    transpose64
}

/// AVX-512 64×64 bit-transpose. The whole block (64 u64 = 8 zmm) is held in
/// registers and the delta-swap network of [`transpose64`] (Warren, Hacker's
/// Delight 2nd ed. §7-3) runs vectorized:
/// - j = 32/16/8: rows `k` and `k+j` sit at the same lane of vectors `k/8`
///   and `k/8 + j/8` — vector-pair delta-swap, no shuffles.
/// - j = 4/2/1: the partner row is `lane ^ j` within one vector — a lane
///   permute materializes partners; mask-blends select each pair's low row
///   `u` and high row `d` so one delta-swap expression updates all lanes
///   (low lanes get `t << j`, high lanes get `t`, exactly the scalar loop).
/// Bit-exact with [`transpose64`] (`transpose64_avx512_matches_scalar_when_available`).
///
/// # Safety
/// Caller must ensure `avx512f` is available (checked by
/// [`transpose64_kernel`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn transpose64_avx512(a: &mut [u64; 64]) {
    use core::arch::x86_64::*;
    let p = a.as_mut_ptr();
    let mut v = [_mm512_setzero_si512(); 8];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = _mm512_loadu_si512(p.add(8 * i) as *const __m512i);
    }
    // Cross-vector levels: pair distance d = j/8 vectors.
    for &(j, m, d) in &[
        (32u64, 0x0000_0000_FFFF_FFFFu64, 4usize),
        (16, 0x0000_FFFF_0000_FFFF, 2),
        (8, 0x00FF_00FF_00FF_00FF, 1),
    ] {
        let cnt = _mm512_set1_epi64(j as i64);
        let mask = _mm512_set1_epi64(m as i64);
        let mut k = 0usize;
        while k < 8 {
            if k & d == 0 {
                let lo = v[k];
                let hi = v[k + d];
                let t =
                    _mm512_and_si512(_mm512_xor_si512(_mm512_srlv_epi64(lo, cnt), hi), mask);
                v[k + d] = _mm512_xor_si512(hi, t);
                v[k] = _mm512_xor_si512(lo, _mm512_sllv_epi64(t, cnt));
            }
            k += 1;
        }
    }
    // Within-vector levels: partner = lane ^ j; `hm` marks each pair's high
    // lane.
    for &(j, m, hm, idx) in &[
        (
            4u64,
            0x0F0F_0F0F_0F0F_0F0Fu64,
            0b1111_0000u8,
            [4i64, 5, 6, 7, 0, 1, 2, 3],
        ),
        (2, 0x3333_3333_3333_3333, 0b1100_1100, [2, 3, 0, 1, 6, 7, 4, 5]),
        (1, 0x5555_5555_5555_5555, 0b1010_1010, [1, 0, 3, 2, 5, 4, 7, 6]),
    ] {
        let cnt = _mm512_set1_epi64(j as i64);
        let mask = _mm512_set1_epi64(m as i64);
        let perm = _mm512_setr_epi64(
            idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7],
        );
        for vec in v.iter_mut() {
            let w = _mm512_permutexvar_epi64(perm, *vec);
            let u = _mm512_mask_blend_epi64(hm, *vec, w); // pair's low row
            let d = _mm512_mask_blend_epi64(hm, w, *vec); // pair's high row
            let t = _mm512_and_si512(_mm512_xor_si512(_mm512_srlv_epi64(u, cnt), d), mask);
            *vec = _mm512_xor_si512(
                *vec,
                _mm512_mask_blend_epi64(hm, _mm512_sllv_epi64(t, cnt), t),
            );
        }
    }
    for (i, vec) in v.iter().enumerate() {
        _mm512_storeu_si512(p.add(8 * i) as *mut __m512i, *vec);
    }
}
```

(If `_mm512_loadu_si512`/`_mm512_storeu_si512` pointer types differ on our toolchain, mirror the exact casts used in `rowsum.rs:73,102` — that file already compiles on MSRV 1.89.)

- [ ] **Step 3.4: Wire the dispatcher into `transpose()`.** Two edits in `BitGrid::transpose`: hoist the kernel before the block loop and call it instead of `transpose64`:

```rust
pub(crate) fn transpose(&self) -> BitGrid {
    let kernel = transpose64_kernel();
    ...
            transpose64(&mut tmp);   // ← replace with: kernel(&mut tmp);
    ...
}
```

Also update the `transpose` doc comment's kernel sentence to: "Processes 64×64 bit blocks via a per-call-dispatched kernel ([`transpose64`] scalar, or [`transpose64_avx512`] when the CPU has AVX-512)".

- [ ] **Step 3.5: Cross-compile check (SIMD path compiles), then local tests (scalar path + dispatch on aarch64):**

Run: `cargo check --target x86_64-unknown-linux-gnu -p aleph-stab --tests`
Expected: clean.
Run: `cargo test -p aleph-stab`
Expected: all green (`transpose64_avx512_matches_scalar_when_available` is an empty pass on aarch64; `blocked_transpose_matches_scalar` + `transpose_roundtrip_and_values` exercise the dispatch end-to-end incl. the 483×241 d=11 shape).

- [ ] **Step 3.6: Commit**

```bash
git add crates/aleph-stab/src/bits.rs
git commit -m "[P4.5-02] AVX-512 64x64 transpose kernel behind runtime dispatch

The 64x64 block is 8 zmm registers; Warren's delta-swap network runs
in-register (j>=8 as vector pairs, j<8 via vpermq + mask blends).
Dispatch hoisted to a fn pointer chosen once per transpose, mirroring
rowsum_dispatch; scalar path unchanged for aarch64/non-AVX-512. The
transpose bridge was 29.7% of the surface-d11 cycle (ADR 0013)."
```

---

### Task 4: Full local validation

- [ ] **Step 4.1: Workspace tests** — `cargo test --workspace`. Expected: green (Stim oracle tests `stim_oracle`/`stim_measure_oracle`/`surface_code_stim_oracle` + `sv_equivalence` cover the measure path end-to-end).
- [ ] **Step 4.2: Lints** — `cargo clippy --workspace --all-targets -- -D warnings` and `cargo clippy --target x86_64-unknown-linux-gnu -p aleph-stab --all-targets -- -D warnings` (the second catches SIMD-path lints invisible on aarch64). Expected: clean.
- [ ] **Step 4.3: Format** — `cargo fmt` then `cargo fmt --check`. Expected: clean.
- [ ] **Step 4.4: Commit any fixups; push the branch** — `git push -u origin p4.5-02-stab-word-parallel-rowops`. GitHub-hosted `test linux` CI is x86_64 → compiles the AVX-512 path and runs tests (hosted runners typically lack AVX-512, so the gated test skips there; EPYC covers it in Task 5).

---

### Task 5: EPYC validation + before/after benchmarks

Box: `ssh root@195.154.249.85` (EPYC 8124P, AVX-512). Existing clone `/root/aleph-p45`; cargo lives under `~/.rustup/toolchains/*/bin` (NOT on PATH — resolve it first). Python venv `/root/parity-venv` (uv py3.12).

- [ ] **Step 5.1: Verify the box is idle** (MANDATORY — CLAUDE.md perf guideline): `uptime` (load ≈ 0) and `pgrep -af "cargo bench|bencher run|Runner.Worker"` (no competing jobs), plus `cat /proc/mdstat` (no resync; P2-03 lesson). Do NOT push to `main`/`benches/**` during the measurement window.
- [ ] **Step 5.2: Transfer the branch via git bundle** (avoids CI-trigger race on the shared runner):

```bash
cd /Users/ex/GitHub/aleph
git bundle create /tmp/p4502.bundle origin/main..p4.5-02-stab-word-parallel-rowops origin/main
scp /tmp/p4502.bundle root@195.154.249.85:/root/
ssh root@195.154.249.85 'cd /root/aleph-p45 && git fetch /root/p4502.bundle p4.5-02-stab-word-parallel-rowops:p4502 && git fetch /root/p4502.bundle origin/main:p4502-main && git log -1 --oneline p4502'
```

Verify the printed HEAD matches the local branch tip (P2-02 lesson: confirm before trusting numbers).
- [ ] **Step 5.3: Correctness on real AVX-512 hardware:** on the box, `git checkout p4502` then `cargo test -p aleph-stab` — confirm `transpose64_avx512_matches_scalar_when_available`, both `*_matches_scalar_reference` tests, and `blocked_transpose_matches_scalar` PASS (not skipped — no "skipping" line in output). Then the Stim oracles: `cargo test -p aleph-benches --test surface_code_stim_oracle` (plus `cargo test -p aleph-stab --tests` for stim_oracle/stim_measure_oracle if the venv provides stim; oracle fixtures are committed, check the test docs).
- [ ] **Step 5.4: Criterion before/after** (single-thread bench, `target-cpu=native`):

```bash
git checkout p4502-main
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches --bench phase4_surface_code -- --save-baseline p4502-before
git checkout p4502
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches --bench phase4_surface_code -- --baseline p4502-before
```

Record the per-d medians and the change % for d=3,5,7,9,11.
- [ ] **Step 5.5: Stim same-session numbers:** `/root/parity-venv/bin/python scripts/surface_code/stim_timing.py --out /root/p4502-stim.json --runs 50` (install `stim` into the venv first if missing: `/root/parity-venv/bin/pip install stim`). Compute aleph/Stim ratio per d from the criterion medians.
- [ ] **Step 5.6: Fresh profile for the report** (AC requires profile evidence for the verdict either way):

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches --bench phase4_surface_code --no-run
BENCH=$(ls -t target/release/deps/phase4_surface_code-* | grep -v '\.d$' | head -1)
perf record -g -o /root/p4502.perf.data -- $BENCH --bench 11 --profile-time 10
perf report -i /root/p4502.perf.data --stdio --percent-limit 1 | head -50
```

Capture the new top-of-profile breakdown (expect `measure`/`transpose` to dominate; `zero_row`/`copy_row` should be gone).
- [ ] **Step 5.7: Verdict check.** If d=11 aleph/Stim ≤ 1.2×: AC met. If not: the profile from 5.6 is the structural-verdict evidence (likely `Tableau::measure`'s strided per-bit column scans — out of this ticket's scope); document honestly in Task 6 (per spec § 5, one-PR-cycle-per-lever timebox).

---

### Task 6: Docs — `docs/perf/surface_code.md` P4.5-02 addendum

- [ ] **Step 6.1: Append a `### P4.5-02` addendum** after the P3-11 section, same shape as the P3-11 one: what changed (word-fill `zero_row`, word-copy `copy_row`, AVX-512 transpose kernel), before/after table (d, aleph before ms, aleph after ms, cycle speedup, aleph/Stim before → after), the fresh d=11 `perf` breakdown, and the verdict line (≤1.2× met, or structural exception + what the remaining bottleneck is). Update the HTML comment at the top of the file (lines 21-23) to note the file now reflects P4.5-02 numbers.
- [ ] **Step 6.2: Commit:**

```bash
git add docs/perf/surface_code.md
git commit -m "[P4.5-02] surface_code.md addendum: word-parallel rowops + AVX-512 transpose numbers"
```

(BACKLOG checkbox flips are deferred to the P4.5-07 meta pass, same as P4.5-01. `docs/perf/parity.md` stab row is also re-measured/finalized in P4.5-07.)

---

### Task 7: PR

- [ ] **Step 7.1: Self-review the full diff** (`git diff origin/main...HEAD`) with fresh eyes.
- [ ] **Step 7.2: Push and open the PR:**

```bash
git push -u origin p4.5-02-stab-word-parallel-rowops
gh pr create --title "[P4.5-02] Stabilizer: word-parallel transpose + zero_row/copy_row" --body "..."
```

Body must include: `Closes #155`; approach summary (3 levers, dispatch pattern, padding-invariant argument); test results (equivalence + mutation-test notes, Stim oracles d=3..11 on EPYC, AVX-512 test exercised on EPYC); the criterion before/after table + aleph/Stim ratios; the fresh perf profile; follow-ups (e.g. `Tableau::measure` column scans if still hot).
- [ ] **Step 7.3: Run `/code-review` (high effort) on the branch; fix CONFIRMED findings; re-run until clean.**
- [ ] **Step 7.4: CI green → merge (squash), confirm #155 auto-closes, verify main builds.**

---

## Self-Review Notes

- **Spec coverage:** AC-1 (d11 improvement + ≤1.2× target or structural verdict w/ profile) → Tasks 5.4–5.7 + 6; AC-2 (bit-exact scalar↔SIMD + Stim oracles d3..11) → Tasks 1.2/2.2/3.1 + 5.3; AC-3 (before/after criterion on EPYC in PR) → Tasks 5.4 + 7.2. Testing req (irregular n) → n ∈ {3, 70, 130, 241} in Tasks 1–2; the blocked-transpose tests already cover 483×241/65×9/64×65.
- **Padding invariant:** `fill(0)` and `copy_from_slice` both preserve "high bits past `cols` are zero" because the source state already satisfies it; the word-level `row_words` compare in the tests asserts it explicitly.
- **`row_pair_mut` precondition:** `dst != src` — sole caller `measure` passes `(p−n, p)` with `p ≥ n`, always distinct; Step 2 verifies no new callers.
- **Scratch-row dirtying (ADR 0013 caveat):** unaffected — `zero_row`/`copy_row` semantics are unchanged, only their inner loops.
- **Fn-pointer indirection on the scalar path:** one indirect call per 64×64 block (≈768 word ops) — noise; aarch64/Ryzen unaffected in any measurable way.
- **Type consistency:** `transpose64_kernel() -> fn(&mut [u64; 64])` matches both `transpose64` and the closure-wrapped `transpose64_avx512`; test names referenced in commits match the test code.
