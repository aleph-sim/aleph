# P3-11 Word-Parallel Gate Kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Word-parallelize (and SIMD) the Clifford `H`/`S`/`CNOT` (and Pauli) gate kernels in the stabilizer tableau by running gates in a column-major orientation, bridged to the row-major `rowsum`/`measure` path by an on-demand bit-transpose.

**Architecture:** `Tableau` carries an orientation flag `{RowMajor, ColMajor}` over the same `BitGrid` storage with transposed dimensions. Gates run in ColMajor (a column is a contiguous `W = ceil((2n+1)/64)`-word span → word/SIMD update across all `2n+1` rows). `rowsum`/`measure` run in RowMajor (P3-08, unchanged). `sign` becomes a packed bit-vector over the orientation-invariant generator-row axis, enabling word-parallel sign updates. Read-only logical accessors are orientation-agnostic; a transpose fires only when a method needs the other orientation (~2/cycle for surface code).

**Tech Stack:** Rust 2021 (MSRV 1.89), `std::arch::x86_64` AVX-512 intrinsics (gated by `is_x86_feature_detected!`), criterion, the existing Stim oracle harness.

**Spec:** `docs/superpowers/specs/2026-06-09-p3-11-word-parallel-gates-design.md`

## File structure

- `crates/aleph-stab/src/bits.rs` — add `BitVec` (packed sign bits), `BitGrid::rows()`, `BitGrid::transpose()` (scalar ref in Task 1, blocked in Task 5).
- `crates/aleph-stab/src/gates.rs` — **new**: word-parallel + AVX-512 gate kernels operating on column/sign word-spans (`h_*`, `s_*`, `cnot_*`, `sign_xor_words`, `y_sign_words`, `*_dispatch`).
- `crates/aleph-stab/src/tableau.rs` — orientation flag, `sign: BitVec`, `get_x`/`get_z`, `ensure_row_major`/`ensure_col_major`; rewire public gates to ColMajor kernels; preserve old kernels as `#[cfg(test)]` `*_scalar` references; equivalence proptest.
- `crates/aleph-stab/src/error.rs` — add `StabError::DuplicateQubit`.
- `crates/aleph-stab/src/backend.rs` — map `DuplicateQubit`.
- `crates/aleph-stab/src/lib.rs` — `mod gates;`.
- `crates/aleph-stab/benches/` — surface-code cycle bench already exists (P4-07); add a gate microbench if missing (Task 6).
- `docs/decisions/0013-stabilizer-dual-orientation-tableau.md` — **new** ADR (Task 6).
- `docs/perf/surface_code.md` — P3-11 addendum (Task 6).

---

### Task 1: `BitVec` + scalar bit-transpose primitive

**Files:**
- Modify: `crates/aleph-stab/src/bits.rs`
- Test: same file, `#[cfg(test)] mod tests`

- [ ] **Step 1: Write failing tests for `BitVec` and `transpose`**

Add to `bits.rs` tests module:

```rust
#[test]
fn bitvec_set_get_and_words() {
    let mut v = super::BitVec::zeros(130); // 3 words
    assert_eq!(v.words().len(), 3);
    assert!(!v.get(129));
    v.set(129, true);
    assert!(v.get(129));
    v.set(129, false);
    assert!(!v.get(129));
    v.set(0, true);
    v.set(64, true);
    assert_eq!(v.words()[0], 1u64);
    assert_eq!(v.words()[1], 1u64);
    // word-level mutation visible through get
    v.words_mut()[2] ^= 1u64 << 1;
    assert!(v.get(129));
}

#[test]
fn grid_rows_accessor() {
    let g = super::BitGrid::zeros(5, 70); // 5 rows, stride 2
    assert_eq!(g.rows(), 5);
    assert_eq!(super::BitGrid::zeros(1, 1).rows(), 1);
}

#[test]
fn transpose_roundtrip_and_values() {
    // Deterministic fill, transpose, check (c,r)==(r,c), and T∘T == id.
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for &(rows, cols) in &[(1usize, 1usize), (3, 5), (7, 64), (65, 9), (128, 130), (483, 241)] {
        let mut g = super::BitGrid::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                if next() & 1 == 1 {
                    g.set(r, c, true);
                }
            }
        }
        let t = g.transpose();
        assert_eq!(t.rows(), cols, "transpose row count ({rows}x{cols})");
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(t.get(c, r), g.get(r, c), "({r},{c}) {rows}x{cols}");
            }
        }
        // round trip
        let tt = t.transpose();
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(tt.get(r, c), g.get(r, c), "roundtrip ({r},{c})");
            }
        }
    }
}
```

- [ ] **Step 2: Run tests; verify they fail to compile (`BitVec`, `rows`, `transpose` missing)**

Run: `cargo test -p aleph-stab bits:: 2>&1 | tail -20`
Expected: compile error — `BitVec` / `rows` / `transpose` not found.

- [ ] **Step 3: Implement `BitVec`, `BitGrid::rows`, `BitGrid::transpose` (scalar)**

Add to `bits.rs` (after the `BitGrid` impl block):

```rust
impl BitGrid {
    /// Number of rows (`words.len() / stride`). `stride ≥ 1` always.
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        self.words.len() / self.stride
    }

    /// Bit-transpose: returns a `cols × rows` grid with out bit `(c, r)` =
    /// `self` bit `(r, c)`. Scalar reference (a blocked kernel replaces the
    /// body in P3-11 Task 5, validated against this via a diff test).
    pub(crate) fn transpose(&self) -> BitGrid {
        let rows = self.rows();
        let mut out = BitGrid::zeros(self.cols, rows);
        for r in 0..rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    out.set(c, r, true);
                }
            }
        }
        out
    }
}

/// Packed bit-vector of `len` bits in `ceil(len/64)` u64 words. Unused high
/// bits in the final word are always zero (`set` only touches valid indices),
/// so word-parallel `&`/`^` consumers need no tail masking.
#[derive(Clone)]
pub(crate) struct BitVec {
    words: Vec<u64>,
    len: usize,
}

impl BitVec {
    pub(crate) fn zeros(len: usize) -> Self {
        BitVec {
            words: vec![0u64; len.div_ceil(64).max(1)],
            len,
        }
    }

    #[inline]
    pub(crate) fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len, "bit {i} out of range {}", self.len);
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }

    #[inline]
    pub(crate) fn set(&mut self, i: usize, val: bool) {
        debug_assert!(i < self.len, "bit {i} out of range {}", self.len);
        let (w, m) = (i >> 6, 1u64 << (i & 63));
        if val {
            self.words[w] |= m;
        } else {
            self.words[w] &= !m;
        }
    }

    #[inline]
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub(crate) fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}
```

- [ ] **Step 4: Run tests; verify pass**

Run: `cargo test -p aleph-stab bits:: 2>&1 | tail -20`
Expected: PASS (all bits tests, including the new three).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/bits.rs
git commit -m "[P3-11] Add BitVec + BitGrid::rows/transpose (scalar reference)"
```

---

### Task 2: Orientation infrastructure in `Tableau` (gates still row-major)

Adds the orientation flag, `sign: BitVec`, orientation-agnostic reads, and the `ensure_*` transposers — **without changing gate kernels yet**. All existing tests must stay green, proving the machinery and transpose round-trip end-to-end.

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`
- Test: same file.

- [ ] **Step 1: Write a failing test forcing an orientation flip**

Add to `tableau.rs` tests:

```rust
#[test]
fn orientation_flip_preserves_logical_bits() {
    // Build a generic state (row-major), force a col-major flip and back,
    // and confirm every logical (x,z,sign) bit is unchanged.
    let t = generic_state();
    let snap: Vec<(bool, bool)> = (0..2 * t.num_qubits())
        .flat_map(|r| (0..t.num_qubits()).map(move |c| (r, c)))
        .map(|(r, c)| (t.x(r, c), t.z(r, c)))
        .collect();
    let signs: Vec<bool> = (0..2 * t.num_qubits() + 1).map(|r| t.sign(r)).collect();

    let mut t2 = t.clone();
    t2.ensure_col_major();
    // reads are orientation-agnostic: identical logical bits in col-major
    let snap2: Vec<(bool, bool)> = (0..2 * t2.num_qubits())
        .flat_map(|r| (0..t2.num_qubits()).map(move |c| (r, c)))
        .map(|(r, c)| (t2.x(r, c), t2.z(r, c)))
        .collect();
    assert_eq!(snap, snap2, "col-major reads diverged");
    t2.ensure_row_major();
    for r in 0..2 * t2.num_qubits() + 1 {
        assert_eq!(t2.sign(r), signs[r], "sign row {r} after flip-back");
    }
    for (i, &(r, c)) in (0..2 * t2.num_qubits())
        .flat_map(|r| (0..t2.num_qubits()).map(move |c| (r, c)))
        .collect::<Vec<_>>()
        .iter()
        .enumerate()
    {
        assert_eq!((t2.x(r, c), t2.z(r, c)), snap[i], "bit ({r},{c}) after flip-back");
    }
}
```

- [ ] **Step 2: Run; verify it fails (no `ensure_col_major`/`ensure_row_major`)**

Run: `cargo test -p aleph-stab orientation_flip 2>&1 | tail -15`
Expected: compile error — methods not found.

- [ ] **Step 3: Add orientation flag, `BitVec` sign, agnostic reads, `ensure_*`**

In `tableau.rs`:

3a. Imports + enum + struct field. Change `use crate::bits::BitGrid;` to:

```rust
use crate::bits::{BitGrid, BitVec};

/// Physical layout of the `x`/`z` grids. Gates need ColMajor (a column is a
/// contiguous word-span); `rowsum`/`measure` need RowMajor (a generator row is
/// contiguous, per P3-08). `sign` is orientation-invariant (the generator-row
/// axis is preserved by the transpose).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Orientation {
    RowMajor,
    ColMajor,
}
```

Change the struct's `sign: Vec<bool>` to `sign: BitVec` and add `orientation: Orientation`:

```rust
pub struct Tableau {
    n: usize,
    /// x-bits. Dimensions depend on `orientation`:
    /// RowMajor = `(2n+1) × n`, ColMajor = `n × (2n+1)`.
    x: BitGrid,
    z: BitGrid,
    /// sign bit per generator row (`true` = `-`); length `2n+1`.
    sign: BitVec,
    orientation: Orientation,
}
```

3b. In `Tableau::new`, replace `sign: vec![false; rows]` with `sign: BitVec::zeros(rows)` and add `orientation: Orientation::RowMajor`. (The `x.set`/`z.set` setup stays — `new` builds RowMajor.)

3c. Add the orientation-agnostic reads + transposers (inside `impl Tableau`):

```rust
/// Logical x-bit of generator `row`, qubit `col`, regardless of orientation.
#[inline]
fn get_x(&self, row: usize, col: usize) -> bool {
    match self.orientation {
        Orientation::RowMajor => self.x.get(row, col),
        Orientation::ColMajor => self.x.get(col, row),
    }
}
/// Logical z-bit of generator `row`, qubit `col`, regardless of orientation.
#[inline]
fn get_z(&self, row: usize, col: usize) -> bool {
    match self.orientation {
        Orientation::RowMajor => self.z.get(row, col),
        Orientation::ColMajor => self.z.get(col, row),
    }
}

/// Ensure RowMajor (generator rows contiguous) for `rowsum`/`measure`/readout.
fn ensure_row_major(&mut self) {
    if self.orientation == Orientation::ColMajor {
        self.x = self.x.transpose();
        self.z = self.z.transpose();
        self.orientation = Orientation::RowMajor;
    }
}
/// Ensure ColMajor (qubit columns contiguous) for word-parallel gates.
fn ensure_col_major(&mut self) {
    if self.orientation == Orientation::RowMajor {
        self.x = self.x.transpose();
        self.z = self.z.transpose();
        self.orientation = Orientation::ColMajor;
    }
}
```

3d. Rewrite the read accessors `x`/`z`/`sign` to be orientation-agnostic:

```rust
#[allow(dead_code)]
#[inline]
pub(crate) fn x(&self, row: usize, col: usize) -> bool {
    self.get_x(row, col)
}
#[allow(dead_code)]
#[inline]
pub(crate) fn z(&self, row: usize, col: usize) -> bool {
    self.get_z(row, col)
}
#[allow(dead_code)]
#[inline]
pub(crate) fn sign(&self, row: usize) -> bool {
    self.sign.get(row)
}
```

3e. Replace every `self.sign[...]` usage with `self.sign.get(...)` / `self.sign.set(.., ..)`. The sites are in `h`, `s`, `cnot`, `x_gate`, `y_gate`, `z_gate` (read+write), `rowsum`, `rowsum_scalar`, `copy_row`, `zero_row`, `measure`, `pauli_eigenvalue`. Examples:
- `self.sign[i] ^= xa & za;` → `let v = self.sign.get(i) ^ (xa & za); self.sign.set(i, v);`
- `self.sign[h] = m == 2;` → `self.sign.set(h, m == 2);`
- `2 * self.sign[h] as i64` → `2 * self.sign.get(h) as i64`
- `self.sign[scratch]` (read) → `self.sign.get(scratch)`
- `self.sign[p] = outcome;` → `self.sign.set(p, outcome);`

3f. In `row_to_pauli`, `rows_anticommute`, and `pauli_eigenvalue`'s `anti_with`/`debug_assert` closures, replace `self.x.get(row,col)`/`self.z.get(...)` (and `t.x.get`/`t.z.get`) with `self.get_x`/`self.get_z` (resp. `t.get_x`/`t.get_z`) so they read correctly in either orientation.

3g. **Gate methods stay row-major for now**: at the top of each gate (`h`, `s`, `cnot`, `x_gate`, `y_gate`, `z_gate`) after `check_qubit`, insert `self.ensure_row_major();`. They keep using `self.x.word(...)` etc. (row-major), unchanged. At the top of `measure` (after `check_qubit`), insert `self.ensure_row_major();`. `pauli_eigenvalue` is `&self`: keep the first anticommute loop on `self` via `get_x`/`get_z`; for the rowsum part change `let mut t = self.clone();` to `let mut t = self.clone(); t.ensure_row_major();`.

- [ ] **Step 4: Build + run the whole stab suite (incl. the new test)**

Run: `cargo test -p aleph-stab 2>&1 | tail -25`
Expected: PASS — all pre-existing tests plus `orientation_flip_preserves_logical_bits`. (Gates still row-major, so the only new behaviour exercised is the explicit flip in the test.)

- [ ] **Step 5: Clippy + fmt**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab --check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-11] Add orientation flag + packed sign + transpose bridge (gates still row-major)"
```

---

### Task 3: Column-major scalar word-parallel gate kernels

Introduce `gates.rs`, flip the public gates to ColMajor word kernels, preserve the old row-major bodies as `#[cfg(test)]` `*_scalar` references, and add the equivalence proptest. Also add the `DuplicateQubit` guard for 2-qubit gates (the ColMajor CNOT needs two distinct spans; `row_pair_mut` requires `a != b`).

**Files:**
- Create: `crates/aleph-stab/src/gates.rs`
- Modify: `crates/aleph-stab/src/lib.rs`, `tableau.rs`, `error.rs`, `backend.rs`
- Test: `gates.rs` (kernel unit tests), `tableau.rs` (equivalence proptest)

- [ ] **Step 1: Write the gate kernels + their unit tests in `gates.rs`**

Create `crates/aleph-stab/src/gates.rs`:

```rust
//! Word-parallel Clifford gate kernels over column-major spans.
//!
//! In ColMajor orientation a qubit column is a contiguous `W = ceil((2n+1)/64)`
//! word-span (one `BitGrid` row); `sign` is a matching `W`-word bit-vector over
//! the same `2n+1` generator-row axis. Each kernel updates the whole column
//! word-parallel. Unused high bits in the final word are zero (BitGrid/BitVec
//! guarantee), so no tail masking is needed. See Aaronson–Gottesman (2004) §2.

/// H(a): `sign ^= x_a & z_a`; swap `x_a, z_a`.
pub(crate) fn h_words(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    debug_assert_eq!(xa.len(), za.len());
    debug_assert_eq!(xa.len(), sign.len());
    for w in 0..xa.len() {
        sign[w] ^= xa[w] & za[w];
        core::mem::swap(&mut xa[w], &mut za[w]);
    }
}

/// S(a): `sign ^= x_a & z_a`; `z_a ^= x_a`.
pub(crate) fn s_words(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
    for w in 0..xa.len() {
        sign[w] ^= xa[w] & za[w];
        za[w] ^= xa[w];
    }
}

/// CNOT(a,b): `sign ^= x_a & z_b & ~(x_b ^ z_a)`; `x_b ^= x_a`; `z_a ^= z_b`.
/// `xb`/`za` are read (for the sign) before being written, in one pass.
pub(crate) fn cnot_words(
    xa: &[u64],
    xb: &mut [u64],
    za: &mut [u64],
    zb: &[u64],
    sign: &mut [u64],
) {
    for w in 0..xa.len() {
        sign[w] ^= xa[w] & zb[w] & !(xb[w] ^ za[w]);
        xb[w] ^= xa[w];
        za[w] ^= zb[w];
    }
}

/// `sign ^= col`. X uses the z-column; Z uses the x-column.
pub(crate) fn sign_xor_words(col: &[u64], sign: &mut [u64]) {
    for w in 0..col.len() {
        sign[w] ^= col[w];
    }
}

/// Y(a): `sign ^= x_a ^ z_a`.
pub(crate) fn y_sign_words(xa: &[u64], za: &[u64], sign: &mut [u64]) {
    for w in 0..xa.len() {
        sign[w] ^= xa[w] ^ za[w];
    }
}

/// Dispatch wrappers (AVX-512 lands in Task 4; for now they just call scalar).
#[inline]
pub(crate) fn h_dispatch(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    h_words(xa, za, sign);
}
#[inline]
pub(crate) fn s_dispatch(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
    s_words(xa, za, sign);
}
#[inline]
pub(crate) fn cnot_dispatch(
    xa: &[u64],
    xb: &mut [u64],
    za: &mut [u64],
    zb: &[u64],
    sign: &mut [u64],
) {
    cnot_words(xa, xb, za, zb, sign);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Per-bit reference over a single column: applies the AG single-qubit
    // update bit-by-bit across `bits` generator rows. Independent of the
    // word kernels above.
    fn get(w: &[u64], j: usize) -> bool {
        w[j >> 6] & (1u64 << (j & 63)) != 0
    }
    fn put(w: &mut [u64], j: usize, v: bool) {
        let (i, m) = (j >> 6, 1u64 << (j & 63));
        if v {
            w[i] |= m;
        } else {
            w[i] &= !m;
        }
    }

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
    fn h_matches_per_bit() {
        let mut rng = Rng(0xDEADBEEF1234);
        for w in [1usize, 2, 5, 8, 9] {
            let bits = w * 64;
            let mut xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let (x0, z0, s0) = (xa.clone(), za.clone(), sg.clone());
            h_words(&mut xa, &mut za, &mut sg);
            // reference
            let (mut xr, mut zr, mut sr) = (x0.clone(), z0.clone(), s0.clone());
            for j in 0..bits {
                let (xj, zj) = (get(&x0, j), get(&z0, j));
                if get(&s0, j) ^ (xj & zj) != get(&sr, j) {
                    put(&mut sr, j, get(&s0, j) ^ (xj & zj));
                }
                put(&mut xr, j, zj); // swap
                put(&mut zr, j, xj);
            }
            assert_eq!(xa, xr, "h x w={w}");
            assert_eq!(za, zr, "h z w={w}");
            assert_eq!(sg, sr, "h sign w={w}");
        }
    }

    #[test]
    fn s_matches_per_bit() {
        let mut rng = Rng(0xABCDEF01);
        for w in [1usize, 3, 8] {
            let bits = w * 64;
            let xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let (z0, s0) = (za.clone(), sg.clone());
            s_words(&xa, &mut za, &mut sg);
            let (mut zr, mut sr) = (z0.clone(), s0.clone());
            for j in 0..bits {
                let (xj, zj) = (get(&xa, j), get(&z0, j));
                put(&mut sr, j, get(&s0, j) ^ (xj & zj));
                put(&mut zr, j, zj ^ xj);
            }
            assert_eq!(za, zr, "s z w={w}");
            assert_eq!(sg, sr, "s sign w={w}");
        }
    }

    #[test]
    fn cnot_matches_per_bit() {
        let mut rng = Rng(0x55AA55AA);
        for w in [1usize, 2, 8, 9] {
            let bits = w * 64;
            let xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut xb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let zb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let (xb0, za0, s0) = (xb.clone(), za.clone(), sg.clone());
            cnot_words(&xa, &mut xb, &mut za, &zb, &mut sg);
            let (mut xbr, mut zar, mut sr) = (xb0.clone(), za0.clone(), s0.clone());
            for j in 0..bits {
                let (xaj, xbj, zaj, zbj) =
                    (get(&xa, j), get(&xb0, j), get(&za0, j), get(&zb, j));
                put(&mut sr, j, get(&s0, j) ^ (xaj & zbj & !(xbj ^ zaj)));
                put(&mut xbr, j, xbj ^ xaj);
                put(&mut zar, j, zaj ^ zbj);
            }
            assert_eq!(xb, xbr, "cnot xb w={w}");
            assert_eq!(za, zar, "cnot za w={w}");
            assert_eq!(sg, sr, "cnot sign w={w}");
        }
    }
}
```

- [ ] **Step 2: Register the module; run the kernel tests (fail until `mod gates;` added)**

Add `mod gates;` to `lib.rs` (after `mod error;`).
Run: `cargo test -p aleph-stab gates:: 2>&1 | tail -20`
Expected: PASS (the three kernel tests).

- [ ] **Step 3: Add `DuplicateQubit` error + mapping**

In `error.rs`, add a variant:

```rust
    /// A 2-qubit gate referenced the same qubit for both operands (e.g.
    /// `CNOT(a, a)`); the column-major kernels require two distinct columns.
    #[error("2-qubit gate referenced qubit {qubit} twice")]
    DuplicateQubit { qubit: u32 },
```

In `backend.rs` `map_stab_err`, add an arm:

```rust
        StabError::DuplicateQubit { qubit } => BackendError::DuplicateQubit { qubit },
```

- [ ] **Step 4: Rewire public gates to ColMajor kernels; preserve old bodies as `*_scalar`**

In `tableau.rs`:

4a. For each of `h`, `s`, `cnot`, `x_gate`, `y_gate`, `z_gate`: copy the **current row-major body** (the one added/kept in Task 2, with the `ensure_row_major()` call) into a new `#[cfg(test)]` method named `<name>_scalar` (e.g. `fn h_scalar(&mut self, a: usize) -> Result<(), crate::StabError>`). These are the bit-exact references.

4b. Replace the public gate bodies with the ColMajor word-kernel versions:

```rust
pub fn h(&mut self, a: usize) -> Result<(), crate::StabError> {
    self.check_qubit(a)?;
    self.ensure_col_major();
    let xa = self.x.row_words_mut(a);
    let za = self.z.row_words_mut(a);
    let sign = self.sign.words_mut();
    crate::gates::h_dispatch(xa, za, sign);
    Ok(())
}

pub fn s(&mut self, a: usize) -> Result<(), crate::StabError> {
    self.check_qubit(a)?;
    self.ensure_col_major();
    let xa = self.x.row_words(a);
    let za = self.z.row_words_mut(a);
    let sign = self.sign.words_mut();
    crate::gates::s_dispatch(xa, za, sign);
    Ok(())
}

pub fn cnot(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
    self.check_qubit(a)?;
    self.check_qubit(b)?;
    if a == b {
        return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
    }
    self.ensure_col_major();
    // x grid: row b mutable (x_b), row a shared (x_a).
    let (xb, xa) = self.x.row_pair_mut(b, a);
    // z grid: row a mutable (z_a), row b shared (z_b).
    let (za, zb) = self.z.row_pair_mut(a, b);
    crate::gates::cnot_dispatch(xa, xb, za, zb, self.sign.words_mut());
    Ok(())
}

pub fn x_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
    self.check_qubit(a)?;
    self.ensure_col_major();
    crate::gates::sign_xor_words(self.z.row_words(a), self.sign.words_mut());
    Ok(())
}

pub fn z_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
    self.check_qubit(a)?;
    self.ensure_col_major();
    crate::gates::sign_xor_words(self.x.row_words(a), self.sign.words_mut());
    Ok(())
}

pub fn y_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
    self.check_qubit(a)?;
    self.ensure_col_major();
    let xa = self.x.row_words(a);
    let za = self.z.row_words(a);
    crate::gates::y_sign_words(xa, za, self.sign.words_mut());
    Ok(())
}
```

> **Borrow note:** `h`/`y` take two simultaneous borrows from *different* grids (`self.x` and `self.z`) plus `self.sign` — three distinct fields, allowed. `cnot` uses `row_pair_mut` on each grid for the same-grid `(a,b)` pair.

4c. Guard `a == b` at the top of the composed 2-qubit gates so they never mutate then fail mid-sequence. In `cz`, `swap`, `iswap`, `iswap_dg`, immediately after the `check_qubit` calls (add `check_qubit` if absent — they currently delegate; add explicit checks), insert:

```rust
    if a == b {
        return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
    }
```

For `cz`:
```rust
pub fn cz(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
    self.check_qubit(a)?;
    self.check_qubit(b)?;
    if a == b {
        return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
    }
    self.h(b)?;
    self.cnot(a, b)?;
    self.h(b)
}
```
Apply the same prelude to `swap`, `iswap`, `iswap_dg` (keep their existing decomposition bodies).

- [ ] **Step 5: Add the equivalence proptest in `tableau.rs`**

```rust
#[test]
fn colmajor_gates_match_scalar_reference() {
    // Drive identical random Clifford circuits through the public ColMajor
    // kernels and the preserved row-major *_scalar references; assert the
    // full logical tableau (x, z, sign) agrees. a != b enforced for 2q gates.
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x as usize) % n
        }
    }
    for n in [1usize, 2, 3, 8, 9, 64, 65, 130] {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ (n as u64).wrapping_mul(0x9E37));
        let mut a = Tableau::new(n);
        let mut b = Tableau::new(n);
        for _ in 0..(20 * n + 50) {
            let pick = rng.below(7);
            let q = rng.below(n);
            match pick {
                0 => {
                    a.h(q).unwrap();
                    b.h_scalar(q).unwrap();
                }
                1 => {
                    a.s(q).unwrap();
                    b.s_scalar(q).unwrap();
                }
                2 => {
                    a.x_gate(q).unwrap();
                    b.x_gate_scalar(q).unwrap();
                }
                3 => {
                    a.y_gate(q).unwrap();
                    b.y_gate_scalar(q).unwrap();
                }
                4 => {
                    a.z_gate(q).unwrap();
                    b.z_gate_scalar(q).unwrap();
                }
                _ => {
                    if n >= 2 {
                        let mut q2 = rng.below(n);
                        if q2 == q {
                            q2 = (q2 + 1) % n;
                        }
                        a.cnot(q, q2).unwrap();
                        b.cnot_scalar(q, q2).unwrap();
                    }
                }
            }
        }
        for r in 0..2 * n {
            assert_eq!(a.sign(r), b.sign(r), "sign[{r}] n={n}");
            for c in 0..n {
                assert_eq!(a.x(r, c), b.x(r, c), "x[{r},{c}] n={n}");
                assert_eq!(a.z(r, c), b.z(r, c), "z[{r},{c}] n={n}");
            }
        }
    }
}

#[test]
fn cnot_duplicate_qubit_rejected() {
    let mut t = Tableau::new(2);
    assert!(matches!(
        t.cnot(1, 1),
        Err(crate::StabError::DuplicateQubit { qubit: 1 })
    ));
    assert!(matches!(
        t.swap(0, 0),
        Err(crate::StabError::DuplicateQubit { qubit: 0 })
    ));
}
```

- [ ] **Step 6: Run the full stab suite + Stim oracles**

Run: `cargo test -p aleph-stab 2>&1 | tail -30`
Expected: PASS — kernel tests, equivalence proptest, duplicate-qubit test, all pre-existing decomposition/measurement tests, **and the Stim oracle tests (d=3..11)**. (If the Stim oracles are `#[ignore]`d, run `cargo test -p aleph-stab -- --ignored 2>&1 | tail -30` and confirm green.)

- [ ] **Step 7: Clippy + fmt**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab --check`
Expected: clean. (Resolve any `dead_code` on `*_scalar` — they are `#[cfg(test)]` so used only in tests; that is fine.)

- [ ] **Step 8: Commit**

```bash
git add crates/aleph-stab/src/gates.rs crates/aleph-stab/src/lib.rs \
        crates/aleph-stab/src/tableau.rs crates/aleph-stab/src/error.rs \
        crates/aleph-stab/src/backend.rs
git commit -m "[P3-11] Column-major word-parallel H/S/CNOT/Pauli gate kernels"
```

---

### Task 4: AVX-512 gate kernels + dispatch

SIMD the hot `H`/`S`/`CNOT` kernels (8×`u64`/step), mirroring `rowsum`'s dispatch. `sign_xor_words`/`y_sign_words` stay scalar (single-pass, cold). Bit-exact vs the scalar word kernels, validated on EPYC.

**Files:**
- Modify: `crates/aleph-stab/src/gates.rs`
- Test: same file.

- [ ] **Step 1: Add AVX-512 kernels + wire dispatch**

In `gates.rs`, replace the three `*_dispatch` bodies and append the AVX-512 functions:

```rust
#[inline]
pub(crate) fn h_dispatch(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f verified present; all slices equal length; the
            // kernel uses unaligned loads/stores within bounds and a scalar
            // tail for the `len % 8` remainder.
            return unsafe { h_avx512(xa, za, sign) };
        }
    }
    h_words(xa, za, sign);
}

#[inline]
pub(crate) fn s_dispatch(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return unsafe { s_avx512(xa, za, sign) };
        }
    }
    s_words(xa, za, sign);
}

#[inline]
pub(crate) fn cnot_dispatch(
    xa: &[u64],
    xb: &mut [u64],
    za: &mut [u64],
    zb: &[u64],
    sign: &mut [u64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return unsafe { cnot_avx512(xa, xb, za, zb, sign) };
        }
    }
    cnot_words(xa, xb, za, zb, sign);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn h_avx512(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    use core::arch::x86_64::*;
    let len = xa.len();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let xw = _mm512_loadu_si512(xa.as_ptr().add(off) as *const __m512i);
        let zw = _mm512_loadu_si512(za.as_ptr().add(off) as *const __m512i);
        let sw = _mm512_loadu_si512(sign.as_ptr().add(off) as *const __m512i);
        let ns = _mm512_xor_si512(sw, _mm512_and_si512(xw, zw));
        _mm512_storeu_si512(sign.as_mut_ptr().add(off) as *mut __m512i, ns);
        // swap x and z
        _mm512_storeu_si512(xa.as_mut_ptr().add(off) as *mut __m512i, zw);
        _mm512_storeu_si512(za.as_mut_ptr().add(off) as *mut __m512i, xw);
    }
    for w in (chunks * 8)..len {
        sign[w] ^= xa[w] & za[w];
        core::mem::swap(&mut xa[w], &mut za[w]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn s_avx512(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
    use core::arch::x86_64::*;
    let len = xa.len();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let xw = _mm512_loadu_si512(xa.as_ptr().add(off) as *const __m512i);
        let zw = _mm512_loadu_si512(za.as_ptr().add(off) as *const __m512i);
        let sw = _mm512_loadu_si512(sign.as_ptr().add(off) as *const __m512i);
        let ns = _mm512_xor_si512(sw, _mm512_and_si512(xw, zw));
        _mm512_storeu_si512(sign.as_mut_ptr().add(off) as *mut __m512i, ns);
        _mm512_storeu_si512(za.as_mut_ptr().add(off) as *mut __m512i, _mm512_xor_si512(zw, xw));
    }
    for w in (chunks * 8)..len {
        sign[w] ^= xa[w] & za[w];
        za[w] ^= xa[w];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cnot_avx512(
    xa: &[u64],
    xb: &mut [u64],
    za: &mut [u64],
    zb: &[u64],
    sign: &mut [u64],
) {
    use core::arch::x86_64::*;
    let len = xa.len();
    let chunks = len / 8;
    let ones = _mm512_set1_epi64(-1);
    for c in 0..chunks {
        let off = c * 8;
        let xaw = _mm512_loadu_si512(xa.as_ptr().add(off) as *const __m512i);
        let xbw = _mm512_loadu_si512(xb.as_ptr().add(off) as *const __m512i);
        let zaw = _mm512_loadu_si512(za.as_ptr().add(off) as *const __m512i);
        let zbw = _mm512_loadu_si512(zb.as_ptr().add(off) as *const __m512i);
        let sw = _mm512_loadu_si512(sign.as_ptr().add(off) as *const __m512i);
        // ~(x_b ^ z_a)
        let nxnz = _mm512_andnot_si512(_mm512_xor_si512(xbw, zaw), ones);
        // x_a & z_b & ~(x_b ^ z_a)
        let term = _mm512_and_si512(_mm512_and_si512(xaw, zbw), nxnz);
        let ns = _mm512_xor_si512(sw, term);
        _mm512_storeu_si512(sign.as_mut_ptr().add(off) as *mut __m512i, ns);
        _mm512_storeu_si512(
            xb.as_mut_ptr().add(off) as *mut __m512i,
            _mm512_xor_si512(xbw, xaw),
        );
        _mm512_storeu_si512(
            za.as_mut_ptr().add(off) as *mut __m512i,
            _mm512_xor_si512(zaw, zbw),
        );
    }
    for w in (chunks * 8)..len {
        sign[w] ^= xa[w] & zb[w] & !(xb[w] ^ za[w]);
        xb[w] ^= xa[w];
        za[w] ^= zb[w];
    }
}
```

- [ ] **Step 2: Add bit-exact SIMD-vs-scalar tests (skipped without AVX-512)**

Append to `gates.rs` tests:

```rust
#[test]
fn avx512_gates_match_scalar_when_available() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            return; // skip on non-AVX512 hosts (local aarch64, GH macos, Ryzen)
        }
        let mut rng = Rng(0x0F1E2D3C4B5A6978);
        for w in [1usize, 2, 7, 8, 9, 16, 17] {
            for _ in 0..500 {
                let xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                let za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                let zb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                let xb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                let sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();

                // H
                let (mut xa1, mut za1, mut s1) = (xa.clone(), za.clone(), sg.clone());
                let (mut xa2, mut za2, mut s2) = (xa.clone(), za.clone(), sg.clone());
                unsafe { h_avx512(&mut xa1, &mut za1, &mut s1) };
                h_words(&mut xa2, &mut za2, &mut s2);
                assert_eq!((&xa1, &za1, &s1), (&xa2, &za2, &s2), "H w={w}");

                // S
                let (mut za1, mut s1) = (za.clone(), sg.clone());
                let (mut za2, mut s2) = (za.clone(), sg.clone());
                unsafe { s_avx512(&xa, &mut za1, &mut s1) };
                s_words(&xa, &mut za2, &mut s2);
                assert_eq!((&za1, &s1), (&za2, &s2), "S w={w}");

                // CNOT
                let (mut xb1, mut za1, mut s1) = (xb.clone(), za.clone(), sg.clone());
                let (mut xb2, mut za2, mut s2) = (xb.clone(), za.clone(), sg.clone());
                unsafe { cnot_avx512(&xa, &mut xb1, &mut za1, &zb, &mut s1) };
                cnot_words(&xa, &mut xb2, &mut za2, &zb, &mut s2);
                assert_eq!((&xb1, &za1, &s1), (&xb2, &za2, &s2), "CNOT w={w}");
            }
        }
    }
}
```

- [ ] **Step 3: Local build/test (scalar path exercised on aarch64)**

Run: `cargo test -p aleph-stab gates:: 2>&1 | tail -20`
Expected: PASS. Note: on local aarch64 the AVX-512 test early-returns; the SIMD path is only exercised on EPYC (Step 5).

- [ ] **Step 4: Cross-check SIMD compiles for x86_64 without an EPYC box**

Run: `cargo check -p aleph-stab --target x86_64-unknown-linux-gnu 2>&1 | tail -15`
Expected: clean (catches `__m512i`/intrinsic typos; per P2-04 lesson). If the target is missing: `rustup target add x86_64-unknown-linux-gnu`.

- [ ] **Step 5: Validate the SIMD path on EPYC**

On the EPYC box ([[aleph_bench_server]], `ssh root@195.154.249.85`; transfer via git-bundle per the P3-08 ops notes), with `RUSTFLAGS="-C target-cpu=native"`:

Run: `cargo test -p aleph-stab gates:: 2>&1 | tail -20` and `cargo test -p aleph-stab 2>&1 | tail -20`
Expected: PASS including `avx512_gates_match_scalar_when_available` (now actually exercised) and the Stim oracles.

- [ ] **Step 6: Clippy + fmt + commit**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab --check`

```bash
git add crates/aleph-stab/src/gates.rs
git commit -m "[P3-11] AVX-512 H/S/CNOT gate kernels + feature dispatch"
```

---

### Task 5: Blocked 64×64 bit-transpose (close the bridge cost)

The scalar transpose is `O(rows·cols)` get/set and runs ~2×/cycle; at d=11 it would dominate the post-gate-speedup cycle. Replace its body with a 64×64-block transpose (Warren, *Hacker's Delight* §7-3), validated bit-exact against the scalar version (now renamed as the reference).

**Files:**
- Modify: `crates/aleph-stab/src/bits.rs`
- Test: same file.

This is a refactor-under-test: add a scalar reference + diff test (green while both are scalar), then swap `transpose`'s body to blocked — the diff test now meaningfully gates the blocked impl.

- [ ] **Step 1: Add a `transpose_scalar` reference (copy of the current body) + diff test**

In `bits.rs`, add `transpose_scalar` as a verbatim copy of the current scalar `transpose` body (keep both for now; `transpose_scalar` is the oracle):

```rust
impl BitGrid {
    /// Scalar bit-transpose reference; the blocked `transpose` is diffed
    /// against this. `#[cfg(test)]` — used only by the diff test.
    #[cfg(test)]
    pub(crate) fn transpose_scalar(&self) -> BitGrid {
        let rows = self.rows();
        let mut out = BitGrid::zeros(self.cols, rows);
        for r in 0..rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    out.set(c, r, true);
                }
            }
        }
        out
    }
}
```

Add the diff test:

```rust
#[test]
fn blocked_transpose_matches_scalar() {
    let mut rng = 0xD1B54A32D192ED03u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for &(rows, cols) in &[
        (1usize, 1usize),
        (3, 5),
        (64, 64),
        (65, 64),
        (64, 65),
        (130, 70),
        (483, 241), // surface d=11
        (200, 200),
    ] {
        let mut g = super::BitGrid::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                if next() & 1 == 1 {
                    g.set(r, c, true);
                }
            }
        }
        let a = g.transpose();
        let b = g.transpose_scalar();
        assert_eq!(a.rows(), b.rows(), "rows {rows}x{cols}");
        // exhaustive bit compare over original coordinates
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(a.get(c, r), b.get(c, r), "blocked!=scalar ({r},{c}) {rows}x{cols}");
            }
        }
    }
}
```

- [ ] **Step 2: Run; verify green baseline (both bodies still scalar → identical)**

Run: `cargo test -p aleph-stab blocked_transpose 2>&1 | tail -15`
Expected: PASS (trivially — `transpose` is still scalar; this pins the oracle before the swap).

- [ ] **Step 3: Replace `transpose`'s body with the blocked implementation**

In `bits.rs`, add the 64×64 in-register helper and **replace the body of the existing `transpose`** (leave `transpose_scalar` as the reference):

```rust
/// Transpose a 64×64 bit-matrix held as 64 rows of `u64` (bit `c` of `a[r]`
/// is element `(r,c)`), in place. Warren, *Hacker's Delight* 2nd ed. §7-3.
#[inline]
fn transpose64(a: &mut [u64; 64]) {
    let mut j = 32usize;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            let t = (a[k] ^ (a[k + j] >> j)) & m;
            a[k] ^= t;
            a[k + j] ^= t << j;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

impl BitGrid {
    /// Blocked bit-transpose: `rows × cols` → `cols × rows`. Processes 64×64
    /// bit blocks via [`transpose64`]; edge blocks are zero-padded (BitGrid
    /// guarantees out-of-range high bits are zero). Bit-exact with
    /// [`BitGrid::transpose_scalar`] (proved by `blocked_transpose_matches_scalar`).
    pub(crate) fn transpose(&self) -> BitGrid {
        let rows = self.rows();
        let cols = self.cols;
        let mut out = BitGrid::zeros(cols, rows);
        let src_stride = self.stride; // words per source row
        let dst_stride = out.stride; // words per dest row (= ceil(rows/64))
        let row_blocks = rows.div_ceil(64);
        let col_blocks = cols.div_ceil(64);
        for rb in 0..row_blocks {
            let r0 = rb * 64;
            let rmax = (r0 + 64).min(rows);
            for cb in 0..col_blocks {
                let c0 = cb * 64;
                let cmax = (c0 + 64).min(cols);
                // Load block: tmp[k] = source row (r0+k), bits [c0, c0+64).
                let mut tmp = [0u64; 64];
                for (k, slot) in tmp.iter_mut().enumerate().take(rmax - r0) {
                    *slot = self.words[(r0 + k) * src_stride + cb];
                }
                transpose64(&mut tmp);
                // Store: out row (c0+k) word at block rb = tmp[k].
                for (k, &val) in tmp.iter().enumerate().take(cmax - c0) {
                    out.words[(c0 + k) * dst_stride + rb] = val;
                }
            }
        }
        out
    }
}
```

> **Edge correctness:** source rows `≥ rows` are left as `tmp[k]=0`; source cols `≥ cols` in the loaded word are zero (BitGrid invariant), so transposed bits landing in out-rows `≥ cols` or out-cols `≥ rows` are zero and never stored (the `take(cmax-c0)` / `take(rmax-r0)` bound the stores). `cb` indexes the source col-word and equals the dest col-block; `rb` indexes the dest col-word. The diff test covers non-multiple-of-64 `rows`/`cols`.

- [ ] **Step 4: Run the diff test + the full suite**

Run: `cargo test -p aleph-stab 2>&1 | tail -25`
Expected: PASS — `blocked_transpose_matches_scalar`, `transpose_roundtrip_and_values`, `orientation_flip_preserves_logical_bits`, the equivalence proptest, and the Stim oracles all green (the orientation bridge now uses the blocked transpose end-to-end).

- [ ] **Step 5: Clippy + fmt + commit**

Run: `cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab --check`

```bash
git add crates/aleph-stab/src/bits.rs
git commit -m "[P3-11] Blocked 64x64 bit-transpose for the orientation bridge"
```

---

### Task 6: EPYC validation, criterion before/after, ADR + perf report

**Files:**
- Bench: `crates/aleph-stab/benches/` (reuse the P4-07 surface-code cycle bench; add a gate microbench if none exists)
- Create: `docs/decisions/0013-stabilizer-dual-orientation-tableau.md`
- Modify: `docs/perf/surface_code.md`
- Modify: `BACKLOG.md` (tick P3-11 acceptance boxes honestly)

- [ ] **Step 1: Locate/confirm the surface-code cycle benchmark**

Run: `ls crates/aleph-stab/benches/ benches/ 2>/dev/null; grep -rln "surface" crates/aleph-stab/benches benches 2>/dev/null`
Expected: find the P4-07 surface-code bench. If a focused gate microbench (apply N H/S/CNOT to an n-qubit tableau) is absent, add one in `crates/aleph-stab/benches/gates.rs` measuring `h`/`cnot` throughput at n ∈ {50, 100, 241} (criterion). Keep it tiny and feature-free.

- [ ] **Step 2: Capture the "before" baseline on a verified-idle EPYC box**

On EPYC ([[feedback-check-server-clean]]: confirm `uptime` load ≈ 0 and `pgrep -af "cargo bench|bencher run|Runner.Worker"` is empty first). Check out **`origin/main`** (pre-P3-11), build the surface bench, and record d=3..11 cycle times + the Stim apples-to-apples numbers.

Run (EPYC): `RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-stab --bench <surface_bench> 2>&1 | tee /tmp/p3-11-before.txt`

- [ ] **Step 3: Capture the "after" numbers on the same idle box**

Check out the `p3-11-word-parallel-gates` branch (transfer via git-bundle), rebuild, rerun the identical bench.

Run (EPYC): `RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-stab --bench <surface_bench> 2>&1 | tee /tmp/p3-11-after.txt`

Also re-profile the cycle to find the new bottleneck:
Run (EPYC): `perf record -g --call-graph dwarf -- <bench-binary> --bench --profile-time 5 <surface/d11 filter>` then `perf report --stdio | head -40`.
Record: gate % vs transpose % vs measure/rowsum %.

- [ ] **Step 4: Compute the honest aleph/Stim d=11 ratio**

From `/tmp/p3-11-after.txt` and the re-measured Stim cycle, compute aleph-d11 / Stim-d11. Compare to P3-08's 7.66×. State whether ≤2× (stretch) is reached; if not, give the number and the dominant cost from Step 3's profile.

- [ ] **Step 5: Write ADR 0013**

Create `docs/decisions/0013-stabilizer-dual-orientation-tableau.md` covering: the row-major-vs-column-major tension; why dual-orientation with lazy transpose beats option 1 (row-major de-scalarize, bounded) and option-column-major-only (regresses `rowsum`/measure); the ~2-transpose/cycle amortization and the `gate,measure` interleave pathology caveat; why `sign` is orientation-invariant; the blocked-transpose choice; and the measured outcome (gate %, cycle speedup, aleph/Stim ratio). Follow the format of `docs/decisions/0012-*.md`.

- [ ] **Step 6: Update `docs/perf/surface_code.md`**

Add a "P3-11 — word-parallel gates" section: before/after cycle table (d=3..11), the new `perf` breakdown, the aleph/Stim d=11 restate, and an honest verdict on the ≤2× target. Reference [[p3-08-planned]]'s addendum it supersedes.

- [ ] **Step 7: Tick P3-11 acceptance criteria in `BACKLOG.md`**

Check the boxes that are met (word-parallel + SIMD kernels bit-exact w/ Stim oracles green; measured cycle speedup reported; ADR landed). Leave the ≤2× stretch box honestly unchecked if not reached, with a one-line note (mirroring the P3-08 honesty precedent).

- [ ] **Step 8: Commit**

```bash
git add docs/decisions/0013-stabilizer-dual-orientation-tableau.md \
        docs/perf/surface_code.md BACKLOG.md crates/aleph-stab/benches/
git commit -m "[P3-11] EPYC validation, ADR 0013, perf report, honest Stim restate"
```

---

## Final verification (before PR)

- [ ] `cargo test --workspace 2>&1 | tail -20` — green (locally; SIMD path validated separately on EPYC per Task 4/6).
- [ ] `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check` — clean.
- [ ] Stim oracles d=3..11 green (EPYC if `#[ignore]`d/slow).
- [ ] `/code-review` high-effort on the diff (P3-08 precedent: caught real issues post-CI-green).
- [ ] Open PR `[P3-11] Stabilizer word-parallel gate kernels (H/S/CNOT)` with `Closes #135`, before/after numbers, the aleph/Stim d=11 restate, and the honest ≤2× verdict.

## Self-review notes (spec coverage)

- Spec §"Gate kernels" → Task 3 (scalar) + Task 4 (SIMD). ✓
- Spec §"Transpose kernel" (scalar ref + blocked + diff + round-trip) → Task 1 (scalar+round-trip) + Task 5 (blocked+diff). ✓
- Spec §"orientation flag / sign packed / ensure_* / get_x/get_z" → Task 2. ✓
- Spec §"Correctness gate" (preserved scalar refs, equivalence proptest, transpose tests, AVX-512 bit-exact, Stim oracles) → Tasks 1/3/4/5. ✓
- Spec §"Performance validation" (idle EPYC, before/after, Stim restate, re-profile) → Task 6. ✓
- Spec §"ADR 0013" → Task 6 Step 5. ✓
- `a != b` precondition for `row_pair_mut`-based CNOT → Task 3 `DuplicateQubit` guard (cnot + composed 2q gates). ✓
- Method-name consistency: `ensure_row_major`/`ensure_col_major`, `get_x`/`get_z`, `*_dispatch`/`*_words`/`*_avx512`, `*_scalar` refs, `transpose`/`transpose_scalar`/`transpose64`, `BitVec`, `BitGrid::rows`. ✓
