# P3-08 Word-parallel + SIMD `rowsum` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scalar per-bit `rowsum` (the stabilizer measurement hot path) with a word-parallel `u64` implementation, then an AVX-512 one, closing the surface-code d=11 gap with Stim to ≤ 2× single-thread.

**Architecture:** Add contiguous-row word-slice accessors to `BitGrid`; put the word-level + AVX-512 `rowsum` kernels in a new `aleph-stab` module; turn `Tableau::rowsum` into a thin dispatcher (AVX-512 when detected, else scalar word-parallel). The old per-bit `rowsum` is preserved as a private reference and diffed bit-for-bit by a deterministic-random oracle; the existing Stim oracles (d=3..11) are the independent end-to-end gate. Layout stays row-major.

**Tech Stack:** Rust (`u64` bit ops + `count_ones`; `core::arch::x86_64` AVX-512 `_mm512_*` + VPOPCNTQ behind `is_x86_feature_detected!`), criterion, EPYC for SIMD/perf validation, python3+stim oracle.

**Spec:** `docs/superpowers/specs/2026-06-09-p3-08-rowsum-bitparallel-design.md`

---

## Key facts (verified against the codebase)

- `crates/aleph-stab/src/bits.rs`: `BitGrid { words: Vec<u64>, stride: usize, cols: usize }`, row-major, `stride = ⌈cols/64⌉`; row `r` occupies `words[r*stride .. (r+1)*stride]`. Existing accessors: `get`/`set`/`row_stride`/`word(idx)`/`word_mut(idx)`. All `pub(crate)`.
- `crates/aleph-stab/src/tableau.rs`: `Tableau { n, x: BitGrid, z: BitGrid, sign: Vec<bool> }`, rows `0..2n+1`. `fn g(x1,z1,x2,z2) -> i32` (free fn) and `fn rowsum(&mut self, h, i)` (method, line ~297). `rowsum` callers (`measure`) always pass `h != i`.
- `crates/aleph-stab/src/lib.rs`: `mod bits; mod tableau; …` (bits is private to the crate; new `mod rowsum;` goes here).
- High bits beyond column `n` in the last word of every row are **always zero**: `BitGrid::zeros` zeroes all words, gate kernels only `set`/mask valid columns, and `rowsum`'s XOR of two zero-high-bit words preserves zero. So word kernels need **no tail masking** — out-of-range bits contribute 0 to both the phase masks and the XOR.
- Local dev is aarch64; x86 SIMD compiles/validates only on EPYC. `cargo check --target x86_64-unknown-linux-gnu` validates compilation. GitHub-hosted `test linux` is x86_64 and gates the scalar path on PR; AVX-512 runtime path is exercised on EPYC.
- No stim locally (system python 3.14 has no wheel); Stim oracle runs on EPYC venv (`stim` 1.16.0).

## The `g` → word formula (the crux; derived once, oracle-verified)

For one bit position, with `(xi, zi)` = row i's (x,z) bit and `(xh, zh)` = row h's:
`g(xi,zi,xh,zh)` ∈ {−1,0,1}. Over a `u64` word, the count of +1 and −1 contributions is the popcount of these masks (validated below against the scalar `g`):

```
plus  = ( xi & !zi & zh &  xh)   // X_i: g=+1 when zh&xh
      | (!xi &  zi & xh & !zh)   // Z_i: g=+1 when xh&!zh
      | ( xi &  zi & zh & !xh)   // Y_i: g=+1 when zh&!xh
minus = ( xi & !zi & zh & !xh)   // X_i: g=-1 when zh&!xh
      | (!xi &  zi & xh &  zh)   // Z_i: g=-1 when xh&zh
      | ( xi &  zi & xh & !zh)   // Y_i: g=-1 when xh&!zh
phase_word = popcount(plus) - popcount(minus)
```
`(xi,zi)=(0,0)` (identity on i) contributes nothing — correct.

---

## Task 1: `BitGrid` row word-slice accessors

**Files:**
- Modify: `crates/aleph-stab/src/bits.rs` (add accessors + unit test)

- [ ] **Step 1: Write the failing test**

Add to `bits.rs` `#[cfg(test)] mod tests`:
```rust
    #[test]
    fn row_words_roundtrip_and_pair() {
        let mut g = BitGrid::zeros(4, 130); // stride 3 words/row
        g.set(1, 0, true);
        g.set(1, 129, true);
        g.set(2, 64, true);
        // row_words: row 1 has bit 0 (word0) and bit 129 (word2)
        let r1 = g.row_words(1);
        assert_eq!(r1.len(), 3);
        assert_eq!(r1[0], 1u64);
        assert_eq!(r1[2], 1u64 << (129 - 128));
        // row_pair_mut: borrow row 1 (mut) and row 2 (shared) at once, dst<src
        {
            let (dst, src) = g.row_pair_mut(1, 2);
            assert_eq!(dst.len(), 3);
            assert_eq!(src.len(), 3);
            assert_eq!(src[1], 1u64); // row 2 bit 64 → word1 bit0
            dst[1] ^= src[1]; // row1 word1 gets bit 64
        }
        assert!(g.get(1, 64));
        // dst>src ordering also works
        {
            let (dst, src) = g.row_pair_mut(2, 1);
            dst[0] ^= src[0]; // row2 word0 ^= row1 word0 (bit0)
        }
        assert!(g.get(2, 0));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p aleph-stab --lib bits::tests::row_words_roundtrip_and_pair 2>&1 | head`
Expected: compile error — `row_words`/`row_pair_mut` not found.

- [ ] **Step 3: Implement the accessors**

Add to `impl BitGrid` in `bits.rs` (after the existing `word_mut`):
```rust
    /// The `stride` words of row `r` (contiguous).
    #[inline]
    pub(crate) fn row_words(&self, r: usize) -> &[u64] {
        let s = self.stride;
        &self.words[r * s..r * s + s]
    }

    /// Mutable contiguous words of row `r`.
    #[inline]
    pub(crate) fn row_words_mut(&mut self, r: usize) -> &mut [u64] {
        let s = self.stride;
        &mut self.words[r * s..r * s + s]
    }

    /// Mutable words of row `dst` and shared words of row `src`, borrowed
    /// simultaneously. Requires `dst != src` (rows live in one backing `Vec`,
    /// split via `split_at_mut`).
    #[inline]
    pub(crate) fn row_pair_mut(&mut self, dst: usize, src: usize) -> (&mut [u64], &[u64]) {
        debug_assert_ne!(dst, src, "row_pair_mut needs distinct rows");
        let s = self.stride;
        if dst < src {
            let (lo, hi) = self.words.split_at_mut(src * s);
            (&mut lo[dst * s..dst * s + s], &hi[..s])
        } else {
            let (lo, hi) = self.words.split_at_mut(dst * s);
            (&mut hi[..s], &lo[src * s..src * s + s])
        }
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p aleph-stab --lib bits::tests::row_words_roundtrip_and_pair`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab && cargo fmt --check`
```bash
git add crates/aleph-stab/src/bits.rs
git commit -m "[P3-08] BitGrid: contiguous row word-slice accessors (row_words/row_pair_mut)"
```

---

## Task 2: scalar word-parallel `rowsum_words` + bit-exact oracle

**Files:**
- Create: `crates/aleph-stab/src/rowsum.rs`
- Modify: `crates/aleph-stab/src/lib.rs` (add `mod rowsum;`)

- [ ] **Step 1: Create the module with the scalar word kernel, a reference, and a deterministic-random bit-exact test**

Create `crates/aleph-stab/src/rowsum.rs`:
```rust
//! Word-parallel `rowsum` kernels for the CHP tableau. `rowsum(h, i)`
//! left-multiplies Pauli row `i` onto row `h`: it XORs the x/z bit-vectors
//! and returns the Aaronson–Gottesman phase exponent Σ_j g(row_i[j], row_h[j])
//! (the caller adds the `2·sign` terms and reduces mod 4). See AG (2004) §2.
//!
//! Rows are contiguous `u64` words (see `BitGrid::row_pair_mut`). Out-of-range
//! high bits in the last word are always zero, so no tail masking is needed.

/// Scalar word-parallel kernel. Reads the original bits of both rows to
/// compute the phase, then XORs row `i` into row `h` in place. Returns the
/// phase exponent contribution (Σ g, may be negative).
pub(crate) fn rowsum_words(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
    debug_assert_eq!(xh.len(), xi.len());
    debug_assert_eq!(zh.len(), zi.len());
    debug_assert_eq!(xh.len(), zh.len());
    let mut acc: i64 = 0;
    // Phase pass (original bits).
    for w in 0..xh.len() {
        let (xiw, ziw, xhw, zhw) = (xi[w], zi[w], xh[w], zh[w]);
        let plus = (xiw & !ziw & zhw & xhw)
            | (!xiw & ziw & xhw & !zhw)
            | (xiw & ziw & zhw & !xhw);
        let minus = (xiw & !ziw & zhw & !xhw)
            | (!xiw & ziw & xhw & zhw)
            | (xiw & ziw & xhw & !zhw);
        acc += plus.count_ones() as i64 - minus.count_ones() as i64;
    }
    // XOR pass (write row h).
    for w in 0..xh.len() {
        xh[w] ^= xi[w];
        zh[w] ^= zi[w];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-bit reference (the pre-P3-08 algorithm), used only to validate the
    /// word kernels. Mirrors the original `Tableau::g` + per-bit loops.
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        let x2 = x2 as i32;
        let z2 = z2 as i32;
        match (x1, z1) {
            (false, false) => 0,
            (true, false) => z2 * (2 * x2 - 1),
            (false, true) => x2 * (1 - 2 * z2),
            (true, true) => z2 - x2,
        }
    }

    fn rowsum_ref(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
        let bits = xh.len() * 64;
        let get = |w: &[u64], j: usize| (w[j >> 6] >> (j & 63)) & 1 == 1;
        let mut acc: i64 = 0;
        for j in 0..bits {
            acc += g(get(xi, j), get(zi, j), get(xh, j), get(zh, j)) as i64;
        }
        for w in 0..xh.len() {
            xh[w] ^= xi[w];
            zh[w] ^= zi[w];
        }
        acc
    }

    // Deterministic xorshift RNG (no proptest dep; reproducible).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn words_match_per_bit_reference() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for stride in [1usize, 2, 4, 8, 13] {
            for _ in 0..2000 {
                let xi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let zi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let xh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let zh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();

                let (mut xa, mut za) = (xh0.clone(), zh0.clone());
                let pa = rowsum_words(&mut xa, &xi, &mut za, &zi);

                let (mut xb, mut zb) = (xh0.clone(), zh0.clone());
                let pb = rowsum_ref(&mut xb, &xi, &mut zb, &zi);

                assert_eq!(pa, pb, "phase mismatch at stride {stride}");
                assert_eq!(xa, xb, "x XOR mismatch at stride {stride}");
                assert_eq!(za, zb, "z XOR mismatch at stride {stride}");
            }
        }
    }
}
```

- [ ] **Step 2: Wire the module**

In `crates/aleph-stab/src/lib.rs`, add `mod rowsum;` alongside the other `mod` lines (no `pub use` — it's crate-internal).

- [ ] **Step 3: Run the bit-exact oracle**

Run: `cargo test -p aleph-stab --lib rowsum::tests::words_match_per_bit_reference`
Expected: PASS (the word formula equals the per-bit `g` reference over 2000 random cases per stride). If it FAILS, the `plus`/`minus` masks are wrong — fix the masks against `g`, do **not** touch the reference.

- [ ] **Step 4: Lint + commit**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab && cargo fmt --check`
```bash
git add crates/aleph-stab/src/rowsum.rs crates/aleph-stab/src/lib.rs
git commit -m "[P3-08] Scalar word-parallel rowsum_words + bit-exact per-bit oracle"
```

---

## Task 3: wire `Tableau::rowsum` to the word kernel; preserve scalar as reference

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Preserve the old body as a `#[cfg(test)]` reference and rewrite `rowsum`**

Replace the existing `fn rowsum(&mut self, h, i)` (around line 297) with:
```rust
    /// Left-multiply stabilizer/destabilizer row `i` onto row `h`, tracking the
    /// sign. Dispatches to the word-parallel kernel (AVX-512 when available).
    fn rowsum(&mut self, h: usize, i: usize) {
        let base = 2 * self.sign[h] as i64 + 2 * self.sign[i] as i64;
        let (xh, xi) = self.x.row_pair_mut(h, i);
        let (zh, zi) = self.z.row_pair_mut(h, i);
        let phase = crate::rowsum::rowsum_words(xh, xi, zh, zi);
        let m = (base + phase).rem_euclid(4);
        debug_assert!(m == 0 || m == 2, "rowsum phase {m} not in {{0, 2}}");
        self.sign[h] = m == 2;
    }

    /// Pre-P3-08 per-bit reference, kept for the equivalence test in this file.
    #[cfg(test)]
    fn rowsum_scalar(&mut self, h: usize, i: usize) {
        let mut acc: i32 = 2 * self.sign[h] as i32 + 2 * self.sign[i] as i32;
        for j in 0..self.n {
            acc += g(self.x.get(i, j), self.z.get(i, j), self.x.get(h, j), self.z.get(h, j));
        }
        let m = acc.rem_euclid(4);
        debug_assert!(m == 0 || m == 2, "rowsum phase {m} not in {{0, 2}}");
        self.sign[h] = m == 2;
        for j in 0..self.n {
            let xh = self.x.get(h, j) ^ self.x.get(i, j);
            let zh = self.z.get(h, j) ^ self.z.get(i, j);
            self.x.set(h, j, xh);
            self.z.set(h, j, zh);
        }
    }
```
Note: `g` is now only referenced from the `#[cfg(test)]` `rowsum_scalar`. Add `#[cfg(test)]` to the free `fn g(...)` (or `#[allow(dead_code)]` if simpler) so a non-test build doesn't warn about an unused function under `-D warnings`. (Prefer `#[cfg(test)]` on `g` — it is genuinely test-only now.)

- [ ] **Step 2: Add a Tableau-level equivalence test**

Add to `tableau.rs` `#[cfg(test)] mod tests` (the module that holds `rowsum_bit_involution`):
```rust
    #[test]
    fn rowsum_matches_scalar_reference() {
        // Drive both implementations from identical entangled states and assert
        // the full tableau (x, z, sign) agrees after the same rowsum.
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, n: usize) -> usize {
                let mut x = self.0;
                x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x;
                (x as usize) % n
            }
        }
        for n in [1usize, 2, 3, 7, 8, 9, 65] {
            let mut rng = Rng(0xD1B54A32D192ED03 ^ n as u64);
            // Build a shared random Clifford state.
            let mut base = Tableau::new(n);
            for _ in 0..(6 * n + 10) {
                match rng.below(3) {
                    0 => { let _ = base.h(rng.below(n)); }
                    1 => { let _ = base.s(rng.below(n)); }
                    _ => {
                        let a = rng.below(n);
                        let b = (a + 1 + rng.below(n.max(2) - 1)) % n.max(1);
                        if n > 1 && a != b { let _ = base.cnot(a, b); }
                    }
                }
            }
            let rows = 2 * n;
            for _ in 0..200 {
                let h = rng.below(rows);
                let mut i = rng.below(rows);
                if i == h { i = (i + 1) % rows; }
                let mut a = base.clone();
                let mut b = base.clone();
                a.rowsum(h, i);
                b.rowsum_scalar(h, i);
                for r in 0..rows {
                    for c in 0..n {
                        assert_eq!(a.x(r, c), b.x(r, c), "x[{r},{c}] n={n}");
                        assert_eq!(a.z(r, c), b.z(r, c), "z[{r},{c}] n={n}");
                    }
                    assert_eq!(a.sign_row(r), b.sign_row(r), "sign[{r}] n={n}");
                }
            }
        }
    }
```
This uses the existing `pub(crate)` read accessors. Verify their exact names first: `grep -n "fn x(\|fn z(\|fn sign\|pub(crate) fn" crates/aleph-stab/src/tableau.rs`. If a sign-row reader does not exist, add a `#[cfg(test)] fn sign_row(&self, r: usize) -> bool { self.sign[r] }` helper. Adjust `a.x(r,c)`/`a.z(r,c)` to whatever the existing accessor names are (the `rowsum_bit_involution` test already calls `t.x(r,j)`/`t.z(r,j)`, so those exist — reuse them).

- [ ] **Step 3: Run the equivalence test + full crate suite**

Run: `cargo test -p aleph-stab`
Expected: all pass, including `rowsum_matches_scalar_reference`, `rowsum_bit_involution`, the measurement tests, and the backend tests. (The Stim oracles are `#[ignore]`d and run on EPYC in Task 5.) If any measurement/Bell/GHZ test regresses, the dispatcher or formula is wrong — debug before proceeding.

- [ ] **Step 4: Lint + commit**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab && cargo fmt --check`
```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-08] Tableau::rowsum dispatches to word kernel; scalar kept as test oracle"
```

---

## Task 4: AVX-512 `rowsum_avx512` + runtime dispatch

**Files:**
- Modify: `crates/aleph-stab/src/rowsum.rs` (add SIMD kernel + dispatcher)
- Modify: `crates/aleph-stab/src/tableau.rs` (call the dispatcher instead of `rowsum_words` directly)

- [ ] **Step 1: Add the AVX-512 kernel and a dispatcher**

Append to `crates/aleph-stab/src/rowsum.rs` (outside the `tests` module):
```rust
/// Dispatch to the AVX-512 kernel when the CPU supports it, else the scalar
/// word kernel. Same contract as [`rowsum_words`].
#[inline]
pub(crate) fn rowsum_dispatch(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            // SAFETY: both features verified present immediately above; slices
            // are equal-length and the kernel only does aligned-agnostic
            // (`loadu`/`storeu`) accesses within bounds.
            return unsafe { rowsum_avx512(xh, xi, zh, zi) };
        }
    }
    rowsum_words(xh, xi, zh, zi)
}

/// AVX-512 + VPOPCNTQ implementation of [`rowsum_words`]. Processes 8 `u64`
/// (512 bits) per step; a scalar tail handles the remaining `len % 8` words.
///
/// # Safety
/// Caller must ensure `avx512f` and `avx512vpopcntdq` are available (checked by
/// [`rowsum_dispatch`]). Slices must be equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn rowsum_avx512(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
    use core::arch::x86_64::*;
    let len = xh.len();
    let chunks = len / 8;
    let mut acc_v = _mm512_setzero_si512(); // 8 lanes of signed i64: Σ(plus-minus)
    for c in 0..chunks {
        let off = c * 8;
        let xiw = _mm512_loadu_si512(xi.as_ptr().add(off) as *const i32);
        let ziw = _mm512_loadu_si512(zi.as_ptr().add(off) as *const i32);
        let xhw = _mm512_loadu_si512(xh.as_ptr().add(off) as *const i32);
        let zhw = _mm512_loadu_si512(zh.as_ptr().add(off) as *const i32);

        // helpers
        let nzi = _mm512_andnot_si512(ziw, _mm512_set1_epi64(-1)); // !zi
        let nzh = _mm512_andnot_si512(zhw, _mm512_set1_epi64(-1)); // !zh
        let nxi = _mm512_andnot_si512(xiw, _mm512_set1_epi64(-1)); // !xi
        let nxh = _mm512_andnot_si512(xhw, _mm512_set1_epi64(-1)); // !xh

        let and4 = |a, b, cc, d| _mm512_and_si512(_mm512_and_si512(a, b), _mm512_and_si512(cc, d));

        let plus = _mm512_or_si512(
            _mm512_or_si512(and4(xiw, nzi, zhw, xhw), and4(nxi, ziw, xhw, nzh)),
            and4(xiw, ziw, zhw, nxh),
        );
        let minus = _mm512_or_si512(
            _mm512_or_si512(and4(xiw, nzi, zhw, nxh), and4(nxi, ziw, xhw, zhw)),
            and4(xiw, ziw, xhw, nzh),
        );
        // per-lane popcount (VPOPCNTQ), accumulate plus - minus
        let pc_plus = _mm512_popcnt_epi64(plus);
        let pc_minus = _mm512_popcnt_epi64(minus);
        acc_v = _mm512_add_epi64(acc_v, _mm512_sub_epi64(pc_plus, pc_minus));

        // XOR pass for this chunk
        let nx = _mm512_xor_si512(xhw, xiw);
        let nz = _mm512_xor_si512(zhw, ziw);
        _mm512_storeu_si512(xh.as_mut_ptr().add(off) as *mut i32, nx);
        _mm512_storeu_si512(zh.as_mut_ptr().add(off) as *mut i32, nz);
    }
    let mut acc = _mm512_reduce_add_epi64(acc_v);
    // scalar tail
    for w in (chunks * 8)..len {
        let (xiw, ziw, xhw, zhw) = (xi[w], zi[w], xh[w], zh[w]);
        let plus = (xiw & !ziw & zhw & xhw) | (!xiw & ziw & xhw & !zhw) | (xiw & ziw & zhw & !xhw);
        let minus = (xiw & !ziw & zhw & !xhw) | (!xiw & ziw & xhw & zhw) | (xiw & ziw & xhw & !zhw);
        acc += plus.count_ones() as i64 - minus.count_ones() as i64;
        xh[w] ^= xi[w];
        zh[w] ^= zi[w];
    }
    acc
}
```
Notes for the implementer:
- `_mm512_andnot_si512(a, b)` computes `(!a) & b`; with `b = all-ones` (`_mm512_set1_epi64(-1)`) it yields `!a`. Confirm against `core::arch` docs; if clearer, compute `!a` as `_mm512_xor_si512(a, all_ones)`.
- `_mm512_popcnt_epi64` requires `avx512vpopcntdq` (enabled in `target_feature`). `_mm512_reduce_add_epi64` is a convenience intrinsic (lowers to a shuffle+add tree).
- The XOR result is stored back into `xh`/`zh` in the same chunk, after the phase masks for that chunk are computed from the loaded originals — ordering is safe within a chunk.

- [ ] **Step 2: Point the Tableau dispatcher at `rowsum_dispatch`**

In `crates/aleph-stab/src/tableau.rs` `fn rowsum`, change the kernel call:
```rust
        let phase = crate::rowsum::rowsum_dispatch(xh, xi, zh, zi);
```
(from `rowsum_words` in Task 3).

- [ ] **Step 3: Extend the bit-exact test to cover the SIMD path (runs only where AVX-512 exists)**

Add to `rowsum.rs` `mod tests`:
```rust
    #[test]
    fn avx512_matches_reference_when_available() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")) {
                return; // no AVX-512 here (e.g. GitHub macos / non-AVX512 linux) — skip
            }
            let mut rng = Rng(0xC2B2AE3D27D4EB4F);
            for stride in [1usize, 2, 7, 8, 9, 16, 17] {
                for _ in 0..1000 {
                    let xi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let zi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let xh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let zh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let (mut xa, mut za) = (xh0.clone(), zh0.clone());
                    let pa = unsafe { rowsum_avx512(&mut xa, &xi, &mut za, &zi) };
                    let (mut xb, mut zb) = (xh0.clone(), zh0.clone());
                    let pb = rowsum_ref(&mut xb, &xi, &mut zb, &zi);
                    assert_eq!(pa, pb, "avx512 phase mismatch stride {stride}");
                    assert_eq!(xa, xb, "avx512 x mismatch stride {stride}");
                    assert_eq!(za, zb, "avx512 z mismatch stride {stride}");
                }
            }
        }
    }
```

- [ ] **Step 4: Validate compilation for x86_64 (locally, aarch64 host)**

Run: `rustup target add x86_64-unknown-linux-gnu 2>/dev/null; cargo check -p aleph-stab --target x86_64-unknown-linux-gnu --all-targets 2>&1 | tail -20`
Expected: compiles clean (this is the only local check of the SIMD code; it does not run it). Also run the scalar suite on the host: `cargo test -p aleph-stab` (the AVX-512 test self-skips on aarch64).

- [ ] **Step 5: Lint + commit**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab && cargo fmt --check`
(If clippy flags the `as *const i32` casts or the closure, address minimally; the AVX-512 body is `#[cfg(target_arch="x86_64")]` so clippy on aarch64 won't lint it — re-run clippy on EPYC in Task 5.)
```bash
git add crates/aleph-stab/src/rowsum.rs crates/aleph-stab/src/tableau.rs
git commit -m "[P3-08] AVX-512 rowsum kernel + runtime dispatch (VPOPCNTQ)"
```

---

## Task 5: EPYC validation, perf measurement, report, ACs, PR

**Files:**
- Modify: `docs/perf/surface_code.md` + `docs/perf/data/surface-aleph.json` (+ meta) — new aleph numbers
- Modify: `BACKLOG.md` (tick P3-08 ACs)

This mirrors the P4-07 EPYC workflow (memory `phase4-status`). Surface-code runs are fast (minutes), so run synchronously.

- [ ] **Step 1: Local gate before shipping to EPYC**

Run:
```bash
cargo fmt --check
cargo clippy -p aleph-stab --all-targets -- -D warnings
cargo test -p aleph-stab
cargo check -p aleph-stab --target x86_64-unknown-linux-gnu --all-targets
```
Expected: all green.

- [ ] **Step 2: Transfer branch to EPYC and run the full correctness + SIMD gate**

Per memory `aleph_bench_server` / `phase4-status`: verify idle (`uptime`, no `cargo bench`), transfer via git bundle into `/tmp/aleph-p114/aleph`, ensure stim 1.16.0 in the venv. Then:
```bash
export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
source scripts/qiskit-baseline/.venv/bin/activate   # python3 -> stim-enabled
# unit + SIMD bit-exact (AVX-512 test now actually runs)
RUSTFLAGS="-C target-cpu=native" cargo test -p aleph-stab
RUSTFLAGS="-C target-cpu=native" cargo clippy -p aleph-stab --all-targets -- -D warnings
# Stim oracles (independent end-to-end gate), all distances
RUSTFLAGS="-C target-cpu=native" cargo test -p aleph-stab --test stim_oracle -- --ignored
RUSTFLAGS="-C target-cpu=native" cargo test -p aleph-stab --test stim_measure_oracle -- --ignored
RUSTFLAGS="-C target-cpu=native" cargo test -p aleph-benches --test surface_code_stim_oracle -- --ignored
```
Expected: `avx512_matches_reference_when_available` runs and passes; all Stim oracles pass (this is **AC-1**). If any Stim oracle fails, the phase formula is wrong — stop and debug (the scalar oracle in Task 2/3 should already have caught it, so this would indicate a SIMD-specific bug).

- [ ] **Step 3: Measure before/after + the surface-d11 vs Stim ratio (AC-2)**

```bash
# aleph timing (single-thread) — the NEW numbers
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 cargo bench -p aleph-benches --bench phase4_surface_code
# extract medians (group "surface_code") -> /tmp/surface-aleph.json (same reader as P4-07)
# Stim timing (unchanged corpus)
python3 scripts/surface_code/stim_timing.py --out /tmp/surface-stim.json --runs 50
```
Record the d=11 ratio `aleph_ms / stim_ms`. **AC-2 target: ≤ 2.0×.** For the honest before/after, also note the prior d=11 ratio from git (`docs/perf/surface_code.md` on `main`: 12.52×). If AC-2 is met → record. If not met even with AVX-512, capture a flamegraph/`perf` to confirm `rowsum` is no longer dominant and report the honest achieved ratio (do not weaken correctness).

- [ ] **Step 4: Re-render the report and tick ACs**

```bash
# regenerate meta (rustc/stim/date), render surface_code.md from the new JSONs
python3 scripts/surface_code/render_report.py --aleph /tmp/surface-aleph.json --stim /tmp/surface-stim.json --meta /tmp/surface-meta.json --out docs/perf/surface_code.md
```
Add a one-line note under the table: the P3-08 speedup (e.g. "d=11: 12.5× → N× vs Stim after word-parallel+AVX-512 rowsum"). scp the updated `surface_code.md` + `docs/perf/data/surface-*.json` back to the local checkout. Tick the two P3-08 acceptance criteria in `BACKLOG.md` (`grep -n "P3-08" BACKLOG.md` → the AC checkboxes around the "bit-sliced tableau passes the existing Stim oracle…" and "Measured speedup…" lines).

- [ ] **Step 5: Commit, push, PR**

```bash
git add docs/perf/surface_code.md docs/perf/data/surface-*.json BACKLOG.md
git commit -m "[P3-08] EPYC: word-parallel+SIMD rowsum results vs Stim + tick ACs"
git push -u origin p3-08-rowsum-bitparallel
gh pr create --title "[P3-08] Word-parallel + SIMD rowsum (close the Stim gap)" --body "<summary>"
```
PR body: `Closes #124`, approach summary, AC-1 (Stim oracles green d=3..11 + bit-exact scalar/SIMD), AC-2 (surface-d11 ratio: before 12.5× → after N×, vs the ≤2× target — state honestly whether met), AC-3 (before/after criterion + updated report), and the honesty note. Clean up EPYC scratch.

---

## Self-review notes

- **Spec coverage:** word-parallel scalar `rowsum` (Task 2), AVX-512 (Task 4), dispatcher + preserved scalar oracle (Task 3), bit-exact diff (Tasks 2 & 4), Stim oracle gate + AC-2 perf + report (Task 5), row-major kept (no transpose), gates/sample/etc out of scope (untouched). ✓
- **Type consistency:** kernel signature `rowsum_words(xh:&mut[u64], xi:&[u64], zh:&mut[u64], zi:&[u64]) -> i64` and `rowsum_dispatch`/`rowsum_avx512` share it identically across Tasks 2–4; `BitGrid::row_pair_mut(dst, src) -> (&mut [u64], &[u64])` defined in Task 1 and consumed in Task 3/4; `Tableau::rowsum` calls `rowsum_words` in Task 3 then `rowsum_dispatch` in Task 4 (explicit swap). ✓
- **Crux risk** (`g`→word formula) is gated three ways: per-bit reference proptest (Task 2), Tableau-level scalar-equivalence (Task 3), Stim oracle d=3..11 (Task 5). ✓
- **SIMD-can't-run-locally** handled: `cargo check --target x86_64` for compile (Task 4 Step 4), self-skipping AVX-512 test, full run on EPYC (Task 5). ✓
- One verify-before-use flagged inline: the exact names of the `pub(crate)` tableau read accessors (`x`/`z`/sign) — Task 3 Step 2 says to grep and reuse the ones `rowsum_bit_involution` already uses.
```
