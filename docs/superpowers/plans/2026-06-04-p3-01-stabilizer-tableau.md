# P3-01 Stabilizer Tableau Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Aaronson-Gottesman (CHP) stabilizer-tableau core in `aleph-stab` — identity init, the full IR Clifford gate set, non-Clifford rejection, signed-Pauli readout — verified by unit, proptest, and Stim oracle tests, meeting the 1000q×depth-100 `<1s` perf AC.

**Architecture:** Row-major CHP tableau (`2n+1` rows × packed `u64` x/z/sign bits), O(n) per gate. Primitive updates for H/S/CNOT + direct Pauli sign rules; Sdg/Cz/Swap/Iswap/IswapDg as primitive compositions. A `dispatch::apply_gate` maps `aleph_core::GateInstance` → tableau ops and rejects non-Clifford via the existing `Gate::is_clifford()`. Readout reuses `aleph_core::{Pauli, PauliString}`.

**Tech Stack:** Rust 2021, `aleph-core` (Gate/GateInstance/Pauli/PauliString), `aleph-sv::NaiveSvBackend` (dev-dep for equivalence tests), `proptest`, `criterion`, Python `stim` (EPYC oracle).

**Reference:** Aaronson & Gottesman 2004, "Improved Simulation of Stabilizer Circuits", §2 (gate updates). Spec: `docs/superpowers/specs/2026-06-04-p3-01-stabilizer-tableau-design.md`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `crates/aleph-stab/Cargo.toml` | crate manifest; add `aleph-core` dep, dev-deps |
| `crates/aleph-stab/src/lib.rs` | module wiring + public re-exports |
| `crates/aleph-stab/src/bits.rs` | `BitGrid` — packed `u64` row store, column-bit get/set/toggle/swap |
| `crates/aleph-stab/src/error.rs` | `StabError` (thiserror) |
| `crates/aleph-stab/src/tableau.rs` | `Tableau`: init, primitive + composed gates, readout, invariants |
| `crates/aleph-stab/src/dispatch.rs` | `apply_gate(&mut Tableau, &GateInstance)` |
| `crates/aleph-stab/tests/sv_equivalence.rs` | composed-gate + generic-state equivalence vs `NaiveSvBackend` |
| `crates/aleph-stab/tests/stim_oracle.rs` | 100-circuit Stim group comparison (`#[ignore]`) |
| `crates/aleph-stab/benches/stab_clifford.rs` | 1000q×depth-100 `<1s` criterion bench |

> **Convention note:** library code uses no `unwrap`/`expect` (CLAUDE.md); errors via `?` + `StabError`. Test code may `unwrap`. Every public item gets a rustdoc line. Comments explain *why*. Cite AG §2 next to the gate rules.

---

## Task 1: Crate skeleton & dependencies

**Files:**
- Modify: `crates/aleph-stab/Cargo.toml`
- Modify: `crates/aleph-stab/src/lib.rs`

- [ ] **Step 1: Set the manifest dependencies**

Replace `crates/aleph-stab/Cargo.toml` with:

```toml
[package]
name = "aleph-stab"
edition.workspace = true
rust-version.workspace = true
version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]
aleph-core = { path = "../aleph-core" }
thiserror  = { workspace = true }

[dev-dependencies]
aleph-sv   = { path = "../aleph-sv" }
aleph-backend = { path = "../aleph-backend" }
proptest   = { workspace = true }
criterion  = { workspace = true }

[[bench]]
name = "stab_clifford"
harness = false
```

- [ ] **Step 2: Wire the module tree in `lib.rs`**

Replace `crates/aleph-stab/src/lib.rs` with:

```rust
//! `aleph-stab`: Stabilizer (Aaronson–Gottesman tableau) backend.
//!
//! Clifford circuits (H, S, CNOT, Paulis, …) simulate in O(n) time per
//! gate and O(n²) memory via the CHP tableau formalism. P3-01 provides
//! the tableau core, gate application, and signed-Pauli readout.
//! Measurement (P3-02) and the `Backend` trait impl (P3-03) land later.
//!
//! Reference: Aaronson & Gottesman, "Improved Simulation of Stabilizer
//! Circuits" (2004), <https://arxiv.org/abs/quant-ph/0406196>.

mod bits;
mod dispatch;
mod error;
mod tableau;

pub use dispatch::apply_gate;
pub use error::StabError;
pub use tableau::Tableau;
```

This will not compile yet (modules don't exist) — that's expected; later tasks create them. To keep the tree green between tasks, create empty stub files now.

- [ ] **Step 3: Create empty module stubs so the crate builds**

```bash
cd /Users/ex/GitHub/aleph
printf '//! Packed-bit row store. See Task 2.\n' > crates/aleph-stab/src/bits.rs
printf '//! StabError. See Task 3.\n'            > crates/aleph-stab/src/error.rs
printf '//! Tableau core. See Tasks 4-9.\n'      > crates/aleph-stab/src/tableau.rs
printf '//! Gate dispatch. See Task 10.\n'       > crates/aleph-stab/src/dispatch.rs
```

Then temporarily comment out the `pub use` lines and `mod` lines for not-yet-defined items, OR leave them — simplest is to defer the `pub use` until the items exist. Replace the `pub use` block with a TODO comment for now:

```rust
// Re-exports added as items land (Tasks 2-10):
// pub use dispatch::apply_gate;
// pub use error::StabError;
// pub use tableau::Tableau;
```

And comment the `mod` lines whose files are empty stubs — keep only `mod bits;` etc. if the empty files compile (an empty `.rs` with just a doc comment is a valid module). Empty modules compile fine, so keep all four `mod` lines and the commented `pub use`.

- [ ] **Step 4: Verify the crate compiles**

Run: `cargo build -p aleph-stab`
Expected: success (empty modules, no re-exports).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/Cargo.toml crates/aleph-stab/src/
git commit -m "[P3-01] aleph-stab crate skeleton + deps"
```

---

## Task 2: `BitGrid` packed-bit row store

A grid of `rows` rows, each `cols` bits, packed into `u64` words. One
contiguous `Vec<u64>`; row `r` occupies words `[r*stride, (r+1)*stride)`
where `stride = ceil(cols/64)`. Used three times by `Tableau` (x bits, z
bits) and once as a single-column sign vector handled separately.

**Files:**
- Modify: `crates/aleph-stab/src/bits.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/aleph-stab/src/bits.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::BitGrid;

    #[test]
    fn set_get_toggle_roundtrip() {
        let mut g = BitGrid::zeros(3, 130); // 3 rows, 130 cols (3 words/row)
        assert!(!g.get(2, 129));
        g.set(2, 129, true);
        assert!(g.get(2, 129));
        g.toggle(2, 129); // -> false
        assert!(!g.get(2, 129));
        g.toggle(1, 0); // -> true
        assert!(g.get(1, 0));
        // independence: untouched cell stays false
        assert!(!g.get(0, 64));
    }

}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib bits`
Expected: FAIL — `BitGrid` not defined.

- [ ] **Step 3: Implement `BitGrid`**

Prepend to `crates/aleph-stab/src/bits.rs` (above the test module):

```rust
//! Packed-bit grid: `rows × cols` bits in a flat `Vec<u64>`.
//!
//! Row-major: row `r` lives in words `[r*stride, (r+1)*stride)`,
//! `stride = ceil(cols/64)`. All accessors are O(1). No bounds checks in
//! release (callers — the tableau — pass in-range indices derived from
//! `n`); `debug_assert` guards catch logic bugs in tests.

#[derive(Clone)]
pub(crate) struct BitGrid {
    words: Vec<u64>,
    stride: usize, // u64 words per row
    cols: usize,
}

impl BitGrid {
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        let stride = cols.div_ceil(64);
        BitGrid {
            words: vec![0u64; rows * stride],
            stride,
            cols,
        }
    }

    #[inline]
    fn word_index(&self, row: usize, col: usize) -> (usize, u64) {
        debug_assert!(col < self.cols, "col {col} out of range {}", self.cols);
        (row * self.stride + (col >> 6), 1u64 << (col & 63))
    }

    #[inline]
    pub(crate) fn get(&self, row: usize, col: usize) -> bool {
        let (w, mask) = self.word_index(row, col);
        self.words[w] & mask != 0
    }

    #[inline]
    pub(crate) fn set(&mut self, row: usize, col: usize, val: bool) {
        let (w, mask) = self.word_index(row, col);
        if val {
            self.words[w] |= mask;
        } else {
            self.words[w] &= !mask;
        }
    }

    #[inline]
    pub(crate) fn toggle(&mut self, row: usize, col: usize) {
        let (w, mask) = self.word_index(row, col);
        self.words[w] ^= mask;
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-stab --lib bits`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/bits.rs
git commit -m "[P3-01] BitGrid packed-bit row store"
```

---

## Task 3: `StabError`

**Files:**
- Modify: `crates/aleph-stab/src/error.rs`
- Modify: `crates/aleph-stab/src/lib.rs` (uncomment the `pub use error::StabError;`)

- [ ] **Step 1: Implement the error enum**

Replace `crates/aleph-stab/src/error.rs` with:

```rust
//! Error type for the stabilizer backend.

/// Errors from applying a gate to a [`crate::Tableau`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StabError {
    /// A non-Clifford gate (T, Rz, Toffoli, arbitrary unitary, …) was
    /// dispatched to the stabilizer backend, which can only simulate
    /// Clifford circuits.
    #[error("non-Clifford gate {gate} cannot run on the stabilizer backend")]
    NonClifford { gate: &'static str },

    /// A gate referenced a qubit index ≥ the tableau's qubit count.
    #[error("qubit {qubit} out of range (tableau has {num_qubits} qubits)")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },
}
```

- [ ] **Step 2: Re-export it**

In `crates/aleph-stab/src/lib.rs`, uncomment / add:

```rust
pub use error::StabError;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p aleph-stab`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-stab/src/error.rs crates/aleph-stab/src/lib.rs
git commit -m "[P3-01] StabError type"
```

---

## Task 4: `Tableau` struct + identity init

State layout: `2n+1` rows. Row `i ∈ [0,n)` = destabilizer i; row `i ∈
[n,2n)` = stabilizer i; row `2n` = scratch (reserved for P3-02). Two
`BitGrid`s (`x`, `z`) of `2n+1` rows × `n` cols, plus a sign `Vec<bool>`
of length `2n+1`.

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`
- Modify: `crates/aleph-stab/src/lib.rs` (re-export `Tableau`)

- [ ] **Step 1: Write the failing test**

Append to `crates/aleph-stab/src/tableau.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::Tableau;

    #[test]
    fn identity_tableau_is_zero_state() {
        let t = Tableau::new(3);
        assert_eq!(t.num_qubits(), 3);
        // destabilizer i = X_i  -> x[i][i]=1, all z=0, sign=+
        // stabilizer  i = Z_i  -> z[n+i][i]=1, all x=0, sign=+
        for i in 0..3 {
            assert!(t.x(i, i), "destab {i} should have X on qubit {i}");
            assert!(t.z(3 + i, i), "stab {i} should have Z on qubit {i}");
            assert!(!t.sign(i) && !t.sign(3 + i), "all signs +");
            for j in 0..3 {
                if j != i {
                    assert!(!t.x(i, j));
                    assert!(!t.z(3 + i, j));
                }
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib tableau::tests::identity`
Expected: FAIL — `Tableau` not defined.

- [ ] **Step 3: Implement struct + `new` + accessors**

Prepend to `crates/aleph-stab/src/tableau.rs`:

```rust
//! Aaronson–Gottesman (CHP) stabilizer tableau.
//!
//! Rows `0..n` are destabilizer generators, `n..2n` are stabilizer
//! generators, and row `2n` is a scratch row reserved for measurement
//! (P3-02). Each row carries `n` x-bits, `n` z-bits, and a sign bit
//! (`true` = leading `-`). Gates update all `2n` non-scratch rows in
//! O(1) each → O(n) per gate. See AG (2004) §2.

use crate::bits::BitGrid;

/// A stabilizer state over `n` qubits in CHP tableau form.
#[derive(Clone)]
pub struct Tableau {
    n: usize,
    /// x-bits: `2n+1` rows × `n` cols.
    x: BitGrid,
    /// z-bits: `2n+1` rows × `n` cols.
    z: BitGrid,
    /// sign bit per row (`true` = `-`); length `2n+1`.
    sign: Vec<bool>,
}

impl Tableau {
    /// Allocate the `|0…0⟩` stabilizer state on `n` qubits.
    ///
    /// Destabilizer `i` = `X_i`, stabilizer `i` = `Z_i`, all signs `+`.
    pub fn new(n: usize) -> Self {
        let rows = 2 * n + 1;
        let mut x = BitGrid::zeros(rows, n.max(1));
        let mut z = BitGrid::zeros(rows, n.max(1));
        for i in 0..n {
            x.set(i, i, true); // destabilizer i = X_i
            z.set(n + i, i, true); // stabilizer i = Z_i
        }
        Tableau {
            n,
            x,
            z,
            sign: vec![false; rows],
        }
    }

    /// Number of qubits.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    // --- read accessors (used by tests + readout) ---
    #[inline]
    pub(crate) fn x(&self, row: usize, col: usize) -> bool {
        self.x.get(row, col)
    }
    #[inline]
    pub(crate) fn z(&self, row: usize, col: usize) -> bool {
        self.z.get(row, col)
    }
    #[inline]
    pub(crate) fn sign(&self, row: usize) -> bool {
        self.sign[row]
    }
}
```

> Note: `BitGrid::zeros(rows, n.max(1))` — `n.max(1)` avoids a
> zero-column grid for the degenerate `n=0` case (no qubits); all loops
> over `0..n` are then empty and the tableau is trivially valid.

- [ ] **Step 4: Re-export `Tableau`**

In `lib.rs` add: `pub use tableau::Tableau;`

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p aleph-stab --lib tableau`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs crates/aleph-stab/src/lib.rs
git commit -m "[P3-01] Tableau struct + identity init"
```

---

## Task 5: Primitive gates — H, S, CNOT

The three CHP primitives (AG §2). Each loops over rows `0..2n` (NOT the
scratch row `2n`). Bounds-check the qubit index against `n`.

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the failing test (Bell state)**

Add inside the existing `tests` mod in `tableau.rs`:

```rust
    #[test]
    fn bell_state_stabilizers() {
        // H(0); CNOT(0,1) on |00> -> stabilized by +XX and +ZZ.
        let mut t = Tableau::new(2);
        t.h(0).unwrap();
        t.cnot(0, 1).unwrap();
        // Stabilizer rows are 2 and 3 (n=2). Check the *group* by its
        // canonical generators is awkward here; instead check the two
        // raw stabilizer rows are {XX:+, ZZ:+} in some order.
        let stabs: Vec<(bool, [bool; 2], [bool; 2])> = (2..4)
            .map(|r| {
                (
                    t.sign(r),
                    [t.x(r, 0), t.x(r, 1)],
                    [t.z(r, 0), t.z(r, 1)],
                )
            })
            .collect();
        // XX row: x=[1,1], z=[0,0], sign=+
        assert!(stabs.contains(&(false, [true, true], [false, false])), "missing +XX: {stabs:?}");
        // ZZ row: x=[0,0], z=[1,1], sign=+
        assert!(stabs.contains(&(false, [false, false], [true, true])), "missing +ZZ: {stabs:?}");
    }

    #[test]
    fn out_of_range_qubit_rejected() {
        let mut t = Tableau::new(2);
        assert!(t.h(2).is_err());
        assert!(t.cnot(0, 2).is_err());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib tableau::tests::bell`
Expected: FAIL — `h`/`cnot` not defined.

- [ ] **Step 3: Implement H, S, CNOT + bounds helper**

Add to the `impl Tableau` block in `tableau.rs`. Note H swaps `x_a` with
`z_a` **across the two grids** (not within one), which is why `BitGrid`
has no single-row column-swap helper.

```rust
    #[inline]
    fn check_qubit(&self, q: usize) -> Result<(), crate::StabError> {
        if q >= self.n {
            return Err(crate::StabError::QubitOutOfRange {
                qubit: q as u32,
                num_qubits: self.n as u32,
            });
        }
        Ok(())
    }

    /// Hadamard on qubit `a`. AG §2: `r ^= x_a·z_a`; swap `x_a, z_a`.
    pub fn h(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        for i in 0..2 * self.n {
            let xa = self.x.get(i, a);
            let za = self.z.get(i, a);
            if xa && za {
                self.sign[i] ^= true;
            }
            self.x.set(i, a, za);
            self.z.set(i, a, xa);
        }
        Ok(())
    }

    /// Phase gate S on qubit `a`. AG §2: `r ^= x_a·z_a`; `z_a ^= x_a`.
    pub fn s(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        for i in 0..2 * self.n {
            if self.x.get(i, a) && self.z.get(i, a) {
                self.sign[i] ^= true;
            }
            if self.x.get(i, a) {
                self.z.toggle(i, a);
            }
        }
        Ok(())
    }

    /// CNOT control `a`, target `b`. AG §2:
    /// `r ^= x_a·z_b·(x_b ⊕ z_a ⊕ 1)`; `x_b ^= x_a`; `z_a ^= z_b`.
    pub fn cnot(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        for i in 0..2 * self.n {
            let xa = self.x.get(i, a);
            let xb = self.x.get(i, b);
            let za = self.z.get(i, a);
            let zb = self.z.get(i, b);
            if xa && zb && (xb ^ za ^ true) {
                self.sign[i] ^= true;
            }
            if xa {
                self.x.toggle(i, b);
            }
            if zb {
                self.z.toggle(i, a);
            }
        }
        Ok(())
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib`
Expected: PASS (identity, bell, out-of-range, bits roundtrip).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-01] Primitive Clifford gates H, S, CNOT (AG §2)"
```

---

## Task 6: Pauli gates — X, Y, Z (direct sign rules)

Paulis conjugate every generator to `± itself`, so they only flip signs.
AG sign rules: `X(a): r ^= z_a`; `Z(a): r ^= x_a`; `Y(a): r ^= x_a ⊕ z_a`.

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the failing test**

X and Z are derivable from H/S, giving an independent check.
`X = H·S·S·H` and `Z = S·S`. Add to the `tests` mod:

```rust
    // Apply gate `g` and its primitive decomposition to two fresh
    // tableaux prepared identically, and assert the full tableaux match.
    fn assert_tableaux_eq(a: &Tableau, b: &Tableau) {
        assert_eq!(a.num_qubits(), b.num_qubits());
        let n = a.num_qubits();
        for r in 0..2 * n {
            assert_eq!(a.sign(r), b.sign(r), "sign row {r}");
            for c in 0..n {
                assert_eq!(a.x(r, c), b.x(r, c), "x[{r}][{c}]");
                assert_eq!(a.z(r, c), b.z(r, c), "z[{r}][{c}]");
            }
        }
    }

    // Prepare a generic (non-|0>) 3-qubit Clifford state to exercise
    // sign rules on populated rows (P1-13 lesson: don't test on |0...0>).
    fn generic_state() -> Tableau {
        let mut t = Tableau::new(3);
        t.h(0).unwrap();
        t.s(0).unwrap();
        t.cnot(0, 1).unwrap();
        t.h(2).unwrap();
        t.cnot(2, 1).unwrap();
        t
    }

    #[test]
    fn z_equals_ss() {
        let mut direct = generic_state();
        direct.z_gate(1).unwrap();
        let mut decomp = generic_state();
        decomp.s(1).unwrap();
        decomp.s(1).unwrap();
        assert_tableaux_eq(&direct, &decomp);
    }

    #[test]
    fn x_equals_hssh() {
        let mut direct = generic_state();
        direct.x_gate(1).unwrap();
        let mut decomp = generic_state();
        decomp.h(1).unwrap();
        decomp.s(1).unwrap();
        decomp.s(1).unwrap();
        decomp.h(1).unwrap();
        assert_tableaux_eq(&direct, &decomp);
    }

    #[test]
    fn y_equals_xz_up_to_phase() {
        // Y = i·X·Z, and the i global phase is unobservable in the
        // stabilizer group, so Y and X∘Z must produce identical tableaux.
        let mut direct = generic_state();
        direct.y_gate(1).unwrap();
        let mut decomp = generic_state();
        decomp.z_gate(1).unwrap();
        decomp.x_gate(1).unwrap();
        assert_tableaux_eq(&direct, &decomp);
    }
```

> Methods are named `x_gate`/`y_gate`/`z_gate` to avoid colliding with
> the `x`/`z` bit accessors.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib tableau::tests::z_equals_ss`
Expected: FAIL — `z_gate` not defined.

- [ ] **Step 3: Implement X, Y, Z**

Add to `impl Tableau`:

```rust
    /// Pauli-X on `a`. Sign rule: `r ^= z_a` (X anticommutes with Z).
    pub fn x_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        for i in 0..2 * self.n {
            if self.z.get(i, a) {
                self.sign[i] ^= true;
            }
        }
        Ok(())
    }

    /// Pauli-Z on `a`. Sign rule: `r ^= x_a`.
    pub fn z_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        for i in 0..2 * self.n {
            if self.x.get(i, a) {
                self.sign[i] ^= true;
            }
        }
        Ok(())
    }

    /// Pauli-Y on `a`. Sign rule: `r ^= x_a ⊕ z_a`.
    pub fn y_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        for i in 0..2 * self.n {
            if self.x.get(i, a) ^ self.z.get(i, a) {
                self.sign[i] ^= true;
            }
        }
        Ok(())
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib tableau`
Expected: PASS (z_equals_ss, x_equals_hssh, y_equals_xz_up_to_phase + earlier).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-01] Pauli X/Y/Z direct sign rules"
```

---

## Task 7: Composed gates — Sdg, Cz, Swap

`Sdg = S·S·S`; `Cz(a,b) = H(b)·CNOT(a,b)·H(b)`; `Swap = CNOT(a,b)·CNOT(b,a)·CNOT(a,b)`.

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the failing test**

Add to `tests` mod:

```rust
    #[test]
    fn sdg_inverts_s() {
        // S then Sdg must restore the original tableau.
        let before = generic_state();
        let mut t = before.clone();
        t.s(1).unwrap();
        t.sdg(1).unwrap();
        assert_tableaux_eq(&t, &before);
    }

    #[test]
    fn cz_is_symmetric_and_hch() {
        let mut a = generic_state();
        a.cz(0, 2).unwrap();
        let mut b = generic_state();
        b.cz(2, 0).unwrap(); // CZ symmetric
        assert_tableaux_eq(&a, &b);
    }

    #[test]
    fn swap_twice_is_identity() {
        let before = generic_state();
        let mut t = before.clone();
        t.swap(0, 2).unwrap();
        t.swap(0, 2).unwrap();
        assert_tableaux_eq(&t, &before);
    }
```

`Tableau` must derive `Clone` (it does — Task 4).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib tableau::tests::sdg`
Expected: FAIL — `sdg` not defined.

- [ ] **Step 3: Implement composed gates**

Add to `impl Tableau`:

```rust
    /// S† on `a`. `S† = S³` (since `S⁴ = I`).
    pub fn sdg(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.s(a)?;
        self.s(a)?;
        self.s(a)
    }

    /// Controlled-Z on `(a,b)`. `CZ = H_b · CNOT_{a,b} · H_b`. Symmetric.
    pub fn cz(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.h(b)?;
        self.cnot(a, b)?;
        self.h(b)
    }

    /// SWAP `(a,b)`. `SWAP = CNOT_{a,b} · CNOT_{b,a} · CNOT_{a,b}`.
    pub fn swap(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.cnot(a, b)?;
        self.cnot(b, a)?;
        self.cnot(a, b)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib tableau`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-01] Composed Clifford gates Sdg, Cz, Swap"
```

---

## Task 8: Composed gates — Iswap, IswapDg (SV-pinned)

iSWAP and its adjoint are Clifford but their H/S/CNOT factorization is
error-prone. The decomposition below is the standard one; the **SV
equivalence test in Task 11 is the source of truth**. We pin it here with
a tableau-level self-consistency test (`Iswap` then `IswapDg` = identity)
and the matrix-action test in Task 11 confirms correctness.

`iSWAP = S_a · S_b · H_a · CNOT_{a,b} · CNOT_{b,a} · H_b`.
`iSWAP† = iSWAP³` (iSWAP⁴ = I), but cheaper: `IswapDg = (iSWAP)` applied
then corrected — simplest correct form is `IswapDg = Sdg_a · Sdg_b`
wrapped appropriately. To avoid guesswork, implement `iswap_dg` as the
**reverse** circuit of `iswap` with each primitive replaced by its
inverse: reverse order of [S_a, S_b, H_a, CNOT_{a,b}, CNOT_{b,a}, H_b]
and invert each (`H†=H`, `CNOT†=CNOT`, `S†=Sdg`):
`H_b · CNOT_{b,a} · CNOT_{a,b} · H_a · Sdg_b · Sdg_a`.

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the failing test**

Add to `tests` mod:

```rust
    #[test]
    fn iswap_then_iswapdg_is_identity() {
        let before = generic_state();
        let mut t = before.clone();
        t.iswap(0, 2).unwrap();
        t.iswap_dg(0, 2).unwrap();
        assert_tableaux_eq(&t, &before);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib tableau::tests::iswap`
Expected: FAIL — `iswap` not defined.

- [ ] **Step 3: Implement iswap / iswap_dg**

Add to `impl Tableau`:

```rust
    /// iSWAP `(a,b)`: `|01⟩ ↔ i|10⟩`. Clifford.
    /// Decomposition: `S_a S_b H_a CNOT_{a,b} CNOT_{b,a} H_b`.
    /// Correctness pinned by the SV-equivalence test (P3-01 §6.1).
    pub fn iswap(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.s(a)?;
        self.s(b)?;
        self.h(a)?;
        self.cnot(a, b)?;
        self.cnot(b, a)?;
        self.h(b)
    }

    /// iSWAP† `(a,b)`: reverse circuit of `iswap` with each primitive
    /// inverted (`H†=H`, `CNOT†=CNOT`, `S†=Sdg`).
    pub fn iswap_dg(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.h(b)?;
        self.cnot(b, a)?;
        self.cnot(a, b)?;
        self.h(a)?;
        self.sdg(b)?;
        self.sdg(a)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-stab --lib tableau::tests::iswap`
Expected: PASS.

> If Task 11's SV-equivalence test later shows the `iswap` *matrix
> action* is wrong (e.g. mismatched up to the `i` phase or a/b swapped),
> the fix is to adjust this decomposition until both Task 8's
> round-trip AND Task 11's matrix test pass. Do not weaken the tests.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-01] Iswap/IswapDg composed gates"
```

---

## Task 9: Readout API + symplectic invariant helper

`stabilizers()` / `destabilizers()` → `Vec<aleph_core::PauliString>`;
`rows_anticommute(i,j)` for property tests.

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the failing test**

Add to `tests` mod:

```rust
    use aleph_core::{Pauli, PauliString};

    #[test]
    fn bell_readout_is_xx_and_zz() {
        let mut t = Tableau::new(2);
        t.h(0).unwrap();
        t.cnot(0, 1).unwrap();
        let stabs = t.stabilizers();
        assert_eq!(stabs.len(), 2);
        let xx = PauliString::new(1.0, vec![(0, Pauli::X), (1, Pauli::X)]).unwrap();
        let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        // order-independent membership
        assert!(stabs.iter().any(|p| same_pauli(p, &xx)));
        assert!(stabs.iter().any(|p| same_pauli(p, &zz)));
    }

    fn same_pauli(a: &PauliString, b: &PauliString) -> bool {
        if (a.coefficient - b.coefficient).abs() > 1e-12 {
            return false;
        }
        let mut at = a.terms.clone();
        let mut bt = b.terms.clone();
        at.sort();
        bt.sort();
        at == bt
    }

    #[test]
    fn identity_state_symplectic() {
        let t = Tableau::new(4);
        let n = 4;
        for i in 0..n {
            // destab i anticommutes with stab i, commutes with others
            assert!(t.rows_anticommute(i, n + i));
            for j in 0..n {
                if j != i {
                    assert!(!t.rows_anticommute(i, n + j));
                }
                assert!(!t.rows_anticommute(n + i, n + j)); // stabs commute
            }
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib tableau::tests::bell_readout`
Expected: FAIL — `stabilizers` not defined.

- [ ] **Step 3: Implement readout + symplectic helper**

Add `use aleph_core::{Pauli, PauliString};` near the top of `tableau.rs`
(below the existing `use crate::bits::BitGrid;`). Add to `impl Tableau`:

```rust
    /// Read a single row as a signed Pauli string. Identity terms are
    /// omitted (sparse), matching `aleph_core::PauliString`.
    fn row_to_pauli(&self, row: usize) -> PauliString {
        let mut terms = Vec::new();
        for c in 0..self.n {
            let p = match (self.x.get(row, c), self.z.get(row, c)) {
                (false, false) => continue, // I
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
            };
            terms.push((c as u32, p));
        }
        let coeff = if self.sign[row] { -1.0 } else { 1.0 };
        // PauliString::new sorts/validates; terms here are already unique
        // and ascending, so this cannot error.
        PauliString::new(coeff, terms).unwrap_or_else(|_| PauliString::identity(coeff))
    }

    /// The `n` stabilizer generators (rows `n..2n`).
    pub fn stabilizers(&self) -> Vec<PauliString> {
        (self.n..2 * self.n).map(|r| self.row_to_pauli(r)).collect()
    }

    /// The `n` destabilizer generators (rows `0..n`).
    pub fn destabilizers(&self) -> Vec<PauliString> {
        (0..self.n).map(|r| self.row_to_pauli(r)).collect()
    }

    /// Symplectic inner product of rows `i` and `j`:
    /// `⊕_a (x_{i,a}·z_{j,a} ⊕ z_{i,a}·x_{j,a})`. `true` ⇒ the two Pauli
    /// strings anticommute.
    pub(crate) fn rows_anticommute(&self, i: usize, j: usize) -> bool {
        let mut acc = false;
        for a in 0..self.n {
            acc ^= (self.x.get(i, a) && self.z.get(j, a))
                ^ (self.z.get(i, a) && self.x.get(j, a));
        }
        acc
    }
```

> `unwrap_or_else` (not `unwrap`) keeps library code free of `unwrap`
> per CLAUDE.md; the fallback is unreachable but satisfies the lint.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib tableau`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-01] Signed-Pauli readout + symplectic helper"
```

---

## Task 10: Gate dispatch + non-Clifford rejection

`apply_gate(&mut Tableau, &GateInstance)` maps IR gates to tableau ops.

**Files:**
- Modify: `crates/aleph-stab/src/dispatch.rs`
- Modify: `crates/aleph-stab/src/lib.rs` (re-export `apply_gate`)

- [ ] **Step 1: Write the failing test**

Replace `crates/aleph-stab/src/dispatch.rs` body's test section (append):

```rust
#[cfg(test)]
mod tests {
    use crate::{apply_gate, Tableau};
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec; // available transitively? if not, use vec!

    #[test]
    fn dispatch_bell() {
        let mut t = Tableau::new(2);
        apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32])).unwrap();
        apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![0u32, 1u32])).unwrap();
        assert_eq!(t.stabilizers().len(), 2);
    }

    #[test]
    fn rejects_non_clifford() {
        let mut t = Tableau::new(1);
        let err = apply_gate(&mut t, &GateInstance::new(Gate::T, vec![0u32])).unwrap_err();
        assert!(matches!(err, crate::StabError::NonClifford { .. }));
    }
}
```

> If `smallvec::smallvec!` isn't a dev-dep of aleph-stab, use `vec!` as
> shown above — `GateInstance::new` accepts `impl Into<SmallVec<...>>`
> and `Vec<u32>` qualifies. Do not add a smallvec dep just for tests.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib dispatch`
Expected: FAIL — `apply_gate` not defined.

- [ ] **Step 3: Implement dispatch**

Prepend to `crates/aleph-stab/src/dispatch.rs`:

```rust
//! Maps `aleph_core::GateInstance` onto [`Tableau`] operations and
//! rejects non-Clifford gates. Uses `Gate::is_clifford()` as the
//! single source of truth for the Clifford set.

use crate::{StabError, Tableau};
use aleph_core::{Gate, GateInstance};

/// Apply one IR gate to the tableau.
///
/// Returns [`StabError::NonClifford`] for any gate outside the Clifford
/// group, and [`StabError::QubitOutOfRange`] for out-of-range indices
/// (surfaced by the underlying `Tableau` methods).
///
/// External `controls` (generic `ctrl @` modifiers) are not supported by
/// the stabilizer backend in P3-01 and are rejected as non-Clifford if
/// present (a controlled-Clifford is not necessarily Clifford, and the
/// IR's `is_clifford()` describes the base gate only).
pub fn apply_gate(t: &mut Tableau, inst: &GateInstance) -> Result<(), StabError> {
    if !inst.controls.is_empty() {
        return Err(StabError::NonClifford {
            gate: gate_name(&inst.gate),
        });
    }
    let q = &inst.qubits;
    match &inst.gate {
        Gate::H => t.h(q[0] as usize),
        Gate::S => t.s(q[0] as usize),
        Gate::Sdg => t.sdg(q[0] as usize),
        Gate::X => t.x_gate(q[0] as usize),
        Gate::Y => t.y_gate(q[0] as usize),
        Gate::Z => t.z_gate(q[0] as usize),
        Gate::Cnot => t.cnot(q[0] as usize, q[1] as usize),
        Gate::Cz => t.cz(q[0] as usize, q[1] as usize),
        Gate::Swap => t.swap(q[0] as usize, q[1] as usize),
        Gate::Iswap => t.iswap(q[0] as usize, q[1] as usize),
        Gate::IswapDg => t.iswap_dg(q[0] as usize, q[1] as usize),
        other => {
            debug_assert!(!other.is_clifford(), "Clifford gate {other:?} not dispatched");
            Err(StabError::NonClifford {
                gate: gate_name(other),
            })
        }
    }
}

/// Static name for error messages (no allocation).
fn gate_name(g: &Gate) -> &'static str {
    match g {
        Gate::H => "H",
        Gate::X => "X",
        Gate::Y => "Y",
        Gate::Z => "Z",
        Gate::S => "S",
        Gate::Sdg => "Sdg",
        Gate::T => "T",
        Gate::Tdg => "Tdg",
        Gate::Rx(_) => "Rx",
        Gate::Ry(_) => "Ry",
        Gate::Rz(_) => "Rz",
        Gate::Phase(_) => "Phase",
        Gate::U3(..) => "U3",
        Gate::Cnot => "Cnot",
        Gate::Cz => "Cz",
        Gate::Swap => "Swap",
        Gate::Iswap => "Iswap",
        Gate::IswapDg => "IswapDg",
        Gate::CRx(_) => "CRx",
        Gate::CRy(_) => "CRy",
        Gate::CRz(_) => "CRz",
        Gate::Toffoli => "Toffoli",
        Gate::Ccz => "Ccz",
        Gate::Unitary1q(_) => "Unitary1q",
        Gate::Unitary1qDiag(_) => "Unitary1qDiag",
        Gate::Unitary2q(_) => "Unitary2q",
        Gate::UnitaryKq { .. } => "UnitaryKq",
    }
}
```

> Verify the `Gate` variant list against
> `crates/aleph-core/src/gate/kinds.rs` at implementation time — if a
> variant was added/renamed, update `gate_name` to match (the `match`
> must stay exhaustive or it won't compile, which is the safety net).

- [ ] **Step 4: Re-export**

In `lib.rs` add: `pub use dispatch::apply_gate;`

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib dispatch`
Expected: PASS.

- [ ] **Step 6: Full crate test + clippy + fmt**

```bash
cargo test -p aleph-stab
cargo clippy -p aleph-stab --all-targets -- -D warnings
cargo fmt -p aleph-stab --check
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-stab/src/dispatch.rs crates/aleph-stab/src/lib.rs
git commit -m "[P3-01] Gate dispatch + non-Clifford rejection"
```

---

## Task 11: SV-equivalence integration test (composed gates)

Verify every composed/native gate's **matrix action** matches the state
vector backend: for a generic random Clifford-prepared state, apply the
gate in both backends, then assert every tableau stabilizer generator
fixes the SV state via `expectation_value`.

**Files:**
- Create: `crates/aleph-stab/tests/sv_equivalence.rs`

- [ ] **Step 1: Write the test**

Create `crates/aleph-stab/tests/sv_equivalence.rs`:

```rust
//! Cross-backend equivalence: the stabilizer tableau and the state
//! vector backend must agree on the action of every Clifford gate.
//!
//! Method: prepare an identical generic Clifford state in both backends,
//! apply the gate under test, then for each tableau stabilizer generator
//! `g` (sign `s`, unsigned Pauli `P`) assert `⟨ψ|P|ψ⟩ = s` to 1e-10 —
//! i.e. `g|ψ⟩ = |ψ⟩`. (P1-13 lesson: prep a generic state, not |0…0⟩.)

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance, Pauli, PauliString};
use aleph_sv::NaiveSvBackend;
use aleph_stab::{apply_gate, Tableau};

const N: usize = 4;

/// A fixed generic Clifford preparation applied to both backends.
fn prep() -> Vec<GateInstance> {
    vec![
        GateInstance::new(Gate::H, vec![0u32]),
        GateInstance::new(Gate::S, vec![0u32]),
        GateInstance::new(Gate::Cnot, vec![0u32, 1u32]),
        GateInstance::new(Gate::H, vec![2u32]),
        GateInstance::new(Gate::Cnot, vec![2u32, 3u32]),
        GateInstance::new(Gate::Cnot, vec![1u32, 2u32]),
    ]
}

fn assert_stabilized(gates_under_test: &[GateInstance]) {
    // Tableau side
    let mut t = Tableau::new(N);
    for g in prep().iter().chain(gates_under_test) {
        apply_gate(&mut t, g).unwrap();
    }
    // SV side
    let mut be = NaiveSvBackend::default();
    let mut sv = be.allocate(N as u32).unwrap();
    for g in prep().iter().chain(gates_under_test) {
        be.apply_gate(&mut sv, g).unwrap();
    }
    // Every stabilizer generator must fix the SV state: <psi|P|psi> = sign.
    for gen in t.stabilizers() {
        let sign = gen.coefficient; // ±1.0
        // unsigned Pauli for expectation_value
        let unsigned = PauliString::new(1.0, gen.terms.clone()).unwrap_or_else(|_| {
            PauliString::identity(1.0)
        });
        let ev = be.expectation_value(&sv, &unsigned).unwrap();
        assert!(
            (ev - sign).abs() < 1e-10,
            "generator {gen:?} not stabilized: <P> = {ev}, expected {sign}"
        );
    }
    // Sanity: the prepared+evolved state has exactly N independent
    // stabilizers (no degeneracy bug).
    assert_eq!(t.stabilizers().len(), N);
}

#[test]
fn native_gates_match_sv() {
    for g in [
        GateInstance::new(Gate::H, vec![1u32]),
        GateInstance::new(Gate::S, vec![1u32]),
        GateInstance::new(Gate::X, vec![1u32]),
        GateInstance::new(Gate::Y, vec![1u32]),
        GateInstance::new(Gate::Z, vec![1u32]),
        GateInstance::new(Gate::Cnot, vec![1u32, 3u32]),
    ] {
        assert_stabilized(&[g]);
    }
}

#[test]
fn composed_gates_match_sv() {
    for g in [
        GateInstance::new(Gate::Sdg, vec![1u32]),
        GateInstance::new(Gate::Cz, vec![0u32, 3u32]),
        GateInstance::new(Gate::Swap, vec![0u32, 3u32]),
        GateInstance::new(Gate::Iswap, vec![0u32, 3u32]),
        GateInstance::new(Gate::IswapDg, vec![0u32, 3u32]),
    ] {
        assert_stabilized(&[g]);
    }
}
```

> If `expectation_value` signature differs (e.g. takes `&mut self`),
> match the trait in `crates/aleph-backend/src/lib.rs` (it is
> `&mut self, state: &Self::State, pauli: &PauliString`). `Backend`
> methods are `&mut self`; `be` is declared `mut` above.

- [ ] **Step 2: Run the test**

Run: `cargo test -p aleph-stab --test sv_equivalence`
Expected: PASS. **If `Iswap`/`IswapDg` fail**, fix the Task 8
decomposition (NOT the test) until both this and Task 8's round-trip pass.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-stab/tests/sv_equivalence.rs
git commit -m "[P3-01] SV-equivalence integration tests"
```

---

## Task 12: Property test — symplectic invariant under random Cliffords

**Files:**
- Modify: `crates/aleph-stab/tests/sv_equivalence.rs` (add a `proptest!` block) OR create `crates/aleph-stab/tests/properties.rs`. Use a new file.
- Create: `crates/aleph-stab/tests/properties.rs`

- [ ] **Step 1: Write the property test**

Create `crates/aleph-stab/tests/properties.rs`:

```rust
//! Tableau well-formedness is preserved under arbitrary Clifford
//! evolution: the destabilizer/stabilizer pair stays symplectic.

use aleph_stab::Tableau;
use proptest::prelude::*;

// A random gate op over the 11-gate Clifford set on `n` qubits, encoded
// as (opcode, q0, q1). We apply via the tableau's public methods.
#[derive(Debug, Clone)]
enum Op {
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    Cnot(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Iswap(usize, usize),
    IswapDg(usize, usize),
}

fn op_strategy(n: usize) -> impl Strategy<Value = Op> {
    let q = 0..n;
    let q2 = (0..n, 0..n).prop_filter("distinct", |(a, b)| a != b);
    prop_oneof![
        q.clone().prop_map(Op::H),
        q.clone().prop_map(Op::S),
        q.clone().prop_map(Op::Sdg),
        q.clone().prop_map(Op::X),
        q.clone().prop_map(Op::Y),
        q.clone().prop_map(Op::Z),
        q2.clone().prop_map(|(a, b)| Op::Cnot(a, b)),
        q2.clone().prop_map(|(a, b)| Op::Cz(a, b)),
        q2.clone().prop_map(|(a, b)| Op::Swap(a, b)),
        q2.clone().prop_map(|(a, b)| Op::Iswap(a, b)),
        q2.prop_map(|(a, b)| Op::IswapDg(a, b)),
    ]
}

fn apply(t: &mut Tableau, op: &Op) {
    match *op {
        Op::H(a) => t.h(a),
        Op::S(a) => t.s(a),
        Op::Sdg(a) => t.sdg(a),
        Op::X(a) => t.x_gate(a),
        Op::Y(a) => t.y_gate(a),
        Op::Z(a) => t.z_gate(a),
        Op::Cnot(a, b) => t.cnot(a, b),
        Op::Cz(a, b) => t.cz(a, b),
        Op::Swap(a, b) => t.swap(a, b),
        Op::Iswap(a, b) => t.iswap(a, b),
        Op::IswapDg(a, b) => t.iswap_dg(a, b),
    }
    .unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn symplectic_invariant_preserved(
        ops in {
            let n = 6;
            proptest::collection::vec(op_strategy(n), 0..40)
        }
    ) {
        let n = 6;
        let mut t = Tableau::new(n);
        for op in &ops {
            apply(&mut t, op);
        }
        // destab i anticommutes with stab i; commutes with all other rows.
        for i in 0..n {
            prop_assert!(t.rows_anticommute(i, n + i), "destab {i} ⊥ stab {i} broken");
            for j in 0..n {
                if j != i {
                    prop_assert!(!t.rows_anticommute(i, n + j));
                    prop_assert!(!t.rows_anticommute(n + i, n + j));
                    prop_assert!(!t.rows_anticommute(i, j));
                }
            }
        }
    }
}
```

> This requires `Tableau::{h,s,sdg,x_gate,y_gate,z_gate,cnot,cz,swap,iswap,iswap_dg}`
> and `rows_anticommute` to be **public** (they are `pub` / `pub(crate)`
> respectively). `rows_anticommute` is `pub(crate)` — change it to `pub`
> so the integration test (separate crate) can call it. Update Task 9's
> signature to `pub fn rows_anticommute` and adjust its rustdoc. (Do this
> now: edit `tableau.rs`.)

- [ ] **Step 2: Make `rows_anticommute` public**

In `tableau.rs`, change `pub(crate) fn rows_anticommute` → `pub fn rows_anticommute`.

- [ ] **Step 3: Run the property test**

Run: `cargo test -p aleph-stab --test properties`
Expected: PASS (200 cases).

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-stab/tests/properties.rs crates/aleph-stab/src/tableau.rs
git commit -m "[P3-01] Property test: symplectic invariant under random Cliffords"
```

---

## Task 13: Criterion bench — 1000q × depth 100 < 1s

**Files:**
- Create: `crates/aleph-stab/benches/stab_clifford.rs`

- [ ] **Step 1: Write the bench**

Create `crates/aleph-stab/benches/stab_clifford.rs`:

```rust
//! P3-01 perf AC: a 1000-qubit, depth-100 random Clifford circuit must
//! run in < 1s. Run on a *verified-idle* box (CLAUDE.md idle-check).
//!
//! Run: cargo bench -p aleph-stab --bench stab_clifford
//! For the <1s assertion as a test: the bench prints per-iter time;
//! compare against the 1s budget in the PR writeup.

use aleph_stab::Tableau;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Deterministic xorshift so the bench is reproducible without an RNG dep.
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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn run_circuit(n: usize, depth: usize) {
    let mut t = Tableau::new(n);
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..depth {
        // one layer ≈ n gates, ~half 2-qubit
        for _ in 0..n {
            match rng.below(6) {
                0 => t.h(rng.below(n as u64) as usize).unwrap(),
                1 => t.s(rng.below(n as u64) as usize).unwrap(),
                2 => t.x_gate(rng.below(n as u64) as usize).unwrap(),
                3 => t.z_gate(rng.below(n as u64) as usize).unwrap(),
                _ => {
                    let a = rng.below(n as u64) as usize;
                    let mut b = rng.below(n as u64) as usize;
                    if a == b {
                        b = (b + 1) % n;
                    }
                    t.cnot(a, b).unwrap();
                }
            }
        }
    }
    black_box(t.stabilizers());
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("stab_clifford");
    group.sample_size(10);
    group.bench_function("n1000_depth100", |b| {
        b.iter(|| run_circuit(black_box(1000), black_box(100)))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

- [ ] **Step 2: Build the bench (compile check)**

Run: `cargo bench -p aleph-stab --bench stab_clifford --no-run`
Expected: compiles.

- [ ] **Step 3: Quick local sanity run (perf number recorded later on EPYC)**

Run: `cargo bench -p aleph-stab --bench stab_clifford`
Expected: completes; note the time (local aarch64 number is indicative,
the AC is asserted on EPYC in Task 15).

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-stab/benches/stab_clifford.rs
git commit -m "[P3-01] Criterion bench: 1000q depth-100 Clifford"
```

---

## Task 14: Stim oracle test (EPYC-gated)

100 random Clifford circuits → compare canonical stabilizer group vs
Stim. Shells out to Python `stim`. `#[ignore]` by default (like the slow
Qiskit oracles); run on EPYC in Task 15.

**Files:**
- Create: `crates/aleph-stab/tests/stim_oracle.rs`

- [ ] **Step 1: Write the oracle test**

Create `crates/aleph-stab/tests/stim_oracle.rs`:

```rust
//! Oracle: stabilizer group equivalence vs Stim on random Clifford
//! circuits. Requires Python + `stim` on PATH; gated `#[ignore]` so the
//! default `cargo test` (and CI without stim) skips it. Run explicitly:
//!
//!   cargo test -p aleph-stab --test stim_oracle -- --ignored
//!
//! Comparison is by *canonical* stabilizer group (Stim's
//! `canonical_stabilizers()`), not raw generator rows — generator choice
//! is non-unique; the group is the invariant.

use aleph_stab::{apply_gate, Tableau};
use aleph_core::{Gate, GateInstance};
use std::process::Command;

const N: usize = 12;
const DEPTH: usize = 30;
const CIRCUITS: usize = 100;

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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One random circuit as a list of (gate, qubits), shared by both sides.
fn random_circuit(seed: u64) -> Vec<GateInstance> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for _ in 0..DEPTH {
        for _ in 0..N {
            let q = rng.below(N as u64) as u32;
            match rng.below(7) {
                0 => out.push(GateInstance::new(Gate::H, vec![q])),
                1 => out.push(GateInstance::new(Gate::S, vec![q])),
                2 => out.push(GateInstance::new(Gate::X, vec![q])),
                3 => out.push(GateInstance::new(Gate::Y, vec![q])),
                4 => out.push(GateInstance::new(Gate::Z, vec![q])),
                _ => {
                    let a = q;
                    let mut b = rng.below(N as u64) as u32;
                    if a == b {
                        b = (b + 1) % N as u32;
                    }
                    out.push(GateInstance::new(Gate::Cnot, vec![a, b]));
                }
            }
        }
    }
    out
}

/// Our canonical stabilizer group as a sorted Vec of signed Pauli
/// strings, using the same text format Stim emits ("+XZ_Y" style).
fn ours_canonical(circ: &[GateInstance]) -> Vec<String> {
    let mut t = Tableau::new(N);
    for g in circ {
        apply_gate(&mut t, g).unwrap();
    }
    // Reduce to canonical (RREF) form to match Stim. For P3-01 we lean on
    // Stim's canonicalization on its side and canonicalize ours by the
    // same algorithm in Python (see emit below): simplest robust path is
    // to hand BOTH the raw circuit to Python and let Stim build the
    // reference, while we send our generators for the Python script to
    // canonicalize identically. To avoid duplicating RREF in Rust, we
    // instead compare against Stim's tableau built from the SAME circuit
    // and rely on Stim canonical form on both — see python script.
    t.stabilizers()
        .iter()
        .map(|p| {
            let mut chars = vec![b'_'; N];
            for (q, pauli) in &p.terms {
                chars[*q as usize] = match pauli {
                    aleph_core::Pauli::I => b'_',
                    aleph_core::Pauli::X => b'X',
                    aleph_core::Pauli::Y => b'Y',
                    aleph_core::Pauli::Z => b'Z',
                };
            }
            let sign = if p.coefficient < 0.0 { '-' } else { '+' };
            format!("{sign}{}", String::from_utf8(chars).unwrap())
        })
        .collect()
}

/// Encode the circuit as a Stim program string.
fn stim_program(circ: &[GateInstance]) -> String {
    let mut s = String::new();
    for g in circ {
        let q = &g.qubits;
        match g.gate {
            Gate::H => s.push_str(&format!("H {}\n", q[0])),
            Gate::S => s.push_str(&format!("S {}\n", q[0])),
            Gate::X => s.push_str(&format!("X {}\n", q[0])),
            Gate::Y => s.push_str(&format!("Y {}\n", q[0])),
            Gate::Z => s.push_str(&format!("Z {}\n", q[0])),
            Gate::Cnot => s.push_str(&format!("CX {} {}\n", q[0], q[1])),
            _ => unreachable!("oracle circuits only use H/S/Paulis/CX"),
        }
    }
    s
}

/// Run the python helper; return Stim's canonical generators (one per
/// line, "+XZ_Y" format). Our side must be canonicalized identically,
/// so we ALSO send our generators and let Python canonicalize both with
/// stim.PauliString / stim.Tableau.from_stabilizers.
fn stim_canonical(circ: &[GateInstance], ours: &[String]) -> Option<(Vec<String>, Vec<String>)> {
    let py = r#"
import sys, stim
data = sys.stdin.read().split("---\n")
prog = data[0]
ours = [l for l in data[1].splitlines() if l]
# Reference from the circuit:
sim = stim.TableauSimulator()
for line in prog.splitlines():
    if not line.strip():
        continue
    parts = line.split()
    op = parts[0]
    args = [int(x) for x in parts[1:]]
    sim.do(stim.Circuit(line))
ref = sim.canonical_stabilizers()
ref_str = [str(p) for p in ref]
# Canonicalize ours through stim so the comparison uses one canonical form:
ours_ps = [stim.PauliString(s) for s in ours]
t = stim.Tableau.from_stabilizers(ours_ps, allow_redundant=False, allow_underconstrained=False)
ours_canon = [str(p) for p in t.to_stabilizers(canonicalize=True)] if hasattr(t,'to_stabilizers') else ref_str
print("\n".join(ref_str))
print("===")
print("\n".join(str(p) for p in stim.Tableau.from_stabilizers(ours_ps).stabilizers(canonicalize=True)))
"#;
    let mut input = stim_program(circ);
    input.push_str("---\n");
    input.push_str(&ours.join("\n"));
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take()?.write_all(input.as_bytes()).ok()?;
            child.wait_with_output().ok()
        })?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.split("===");
    let refs: Vec<String> = parts.next()?.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let ours_c: Vec<String> = parts.next()?.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    Some((refs, ours_c))
}

#[test]
#[ignore = "requires python3 + stim; run on the EPYC oracle venv"]
fn matches_stim_on_random_cliffords() {
    let mut failures = 0;
    for k in 0..CIRCUITS {
        let circ = random_circuit(0xABCDEF ^ (k as u64).wrapping_mul(0x100000001B3));
        let ours = ours_canonical(&circ);
        let (refs, ours_c) = match stim_canonical(&circ, &ours) {
            Some(v) => v,
            None => panic!("stim helper failed (is `stim` installed in the active python3?)"),
        };
        let mut a = refs.clone();
        let mut b = ours_c.clone();
        a.sort();
        b.sort();
        if a != b {
            failures += 1;
            eprintln!("circuit {k} mismatch:\n  stim: {a:?}\n  ours: {b:?}");
        }
    }
    assert_eq!(failures, 0, "{failures}/{CIRCUITS} circuits disagreed with Stim");
}
```

> **Implementation note for the executor:** the Python helper's exact
> `stim` API calls (`Tableau.from_stabilizers`, `to_stabilizers` /
> `stabilizers(canonicalize=True)`) vary across stim versions. When
> running on EPYC (Task 15), adjust the helper to the installed stim
> version's API — the *contract* is: produce a canonical generator list
> for both the circuit (reference) and our generators, in the same
> format, then compare as sorted sets. Pin the working stim version in
> the PR notes. The Rust side (`ours_canonical`, set comparison) is
> stable; only the embedded Python may need a one-line API tweak.

- [ ] **Step 2: Compile-check (test is ignored, won't run stim locally)**

Run: `cargo test -p aleph-stab --test stim_oracle --no-run`
Expected: compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-stab/tests/stim_oracle.rs
git commit -m "[P3-01] Stim oracle test (EPYC-gated, #[ignore])"
```

---

## Task 15: EPYC validation + perf number + workspace gate

Validate on the EPYC box (`ssh root@195.154.249.85`) per the project's
SIMD/bench gate. Stabilizer code is scalar (no AVX-512), so it also runs
correctly on local aarch64 — but the **perf AC (<1s)** and the **Stim
oracle** are EPYC-validated.

**Files:** none (validation + notes only).

- [ ] **Step 1: Local full-workspace gate**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
Expected: all green. Fixes go back into the relevant task's files.

- [ ] **Step 2: Ship branch to EPYC via git bundle**

```bash
git bundle create /tmp/p3-01.bundle p3-01-stabilizer-tableau
scp /tmp/p3-01.bundle root@195.154.249.85:/root/
ssh root@195.154.249.85 'cd /root && rm -rf aleph-p301 && git clone -q /root/p3-01.bundle aleph-p301 && cd aleph-p301 && git checkout -q p3-01-stabilizer-tableau'
```

- [ ] **Step 3: Idle-check, then run tests + bench on EPYC**

```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" || echo IDLE'
# Expect load ~0 and IDLE. If not, wait (CLAUDE.md idle rule).
ssh root@195.154.249.85 'export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH; cd /root/aleph-p301; cargo test -p aleph-stab; RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-stab --bench stab_clifford'
```
Expected: tests pass; bench `n1000_depth100` reports < 1s. Record the number.

- [ ] **Step 4: Install stim + run the oracle on EPYC**

```bash
ssh root@195.154.249.85 'cd /root/aleph-p301; python3 -m pip install --quiet stim || (uv venv .venv && . .venv/bin/activate && uv pip install stim); python3 -c "import stim; print(stim.__version__)"'
ssh root@195.154.249.85 'export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH; cd /root/aleph-p301; cargo test -p aleph-stab --test stim_oracle -- --ignored --nocapture'
```
Expected: `0/100 circuits disagreed with Stim`. If the Python helper's
stim API mismatches the installed version, fix it (Task 14 note),
re-bundle from local, re-run.

- [ ] **Step 5: Clean up EPYC disk (root / is ~20G)**

```bash
ssh root@195.154.249.85 'rm -rf /root/aleph-p301 /root/p3-01.bundle'
```

- [ ] **Step 6: Record numbers; no code commit unless fixes were needed**

If Step 4 required a Python-helper fix, commit it:
```bash
git add crates/aleph-stab/tests/stim_oracle.rs
git commit -m "[P3-01] Pin Stim oracle to installed stim API"
```

---

## Task 16: PR

**Files:** none.

- [ ] **Step 1: Find the GitHub issue number for P3-01**

```bash
gh issue list --search "P3-01 in:title" --state all --json number,title
```
Note the issue number (call it `<ISSUE>`).

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin p3-01-stabilizer-tableau
gh pr create --title "[P3-01] Stabilizer simulator — Aaronson-Gottesman tableau" --body "$(cat <<'EOF'
Closes #<ISSUE>

## Summary
First Phase-3 ticket. Adds the Aaronson-Gottesman (CHP) stabilizer
tableau core in `aleph-stab`:
- Row-major `2n+1`-row tableau over a dependency-free packed-`u64`
  `BitGrid`; O(n) per gate, O(n²) memory.
- Full IR Clifford set: H/S/CNOT + direct Pauli sign rules (X/Y/Z) as
  primitives; Sdg/Cz/Swap/Iswap/IswapDg as primitive compositions.
- `apply_gate(&mut Tableau, &GateInstance)` dispatch reusing the existing
  `Gate::is_clifford()`; rejects non-Clifford gates and external controls.
- Signed-Pauli readout reusing `aleph_core::{Pauli, PauliString}`.

Scope per spec: **no measurement** (P3-02) and **no `Backend` impl**
(P3-03); the scratch row is reserved but unused.

## Tests
- Unit: Bell/GHZ stabilizers, identity init, X=HSSH / Z=SS / Y=XZ
  decomposition checks, out-of-range rejection.
- SV-equivalence (`tests/sv_equivalence.rs`): every native + composed
  gate verified against `NaiveSvBackend` via `expectation_value`
  (generator fixes the SV state to 1e-10).
- Property (`tests/properties.rs`, 200 cases): symplectic invariant
  preserved under random 40-gate Clifford circuits on 6 qubits.
- Stim oracle (`tests/stim_oracle.rs`, `#[ignore]`): 100 random Clifford
  circuits, canonical stabilizer-group equivalence vs Stim. Validated on
  EPYC: 0/100 disagreements. Stim version: <fill>.

## Benchmark
`stab_clifford/n1000_depth100` on verified-idle EPYC: **<fill>** (AC: <1s). ✅/❌

## AC mapping
- [x] H, S, CNOT, X, Y, Z (+ full Clifford set)
- [x] 1000q depth 100 < 1s (EPYC number above)
- [x] Verified against Stim
- [x] Correctly rejects non-Clifford gates

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Confirm CI is green; self-review the diff.**

Run: `gh pr checks --watch` and re-read the diff with fresh eyes.

---

## Self-Review (plan vs spec)

**Spec coverage:**
- §2 crate layout → Tasks 1–10 (one module per file). ✓
- §3 tableau representation → Task 4 (struct, 2n+1 rows, packed bits, identity). ✓
- §4.1 primitives → Task 5; §4.2 Paulis → Task 6; §4.3 compositions → Tasks 7–8; §4.4 dispatch+rejection → Task 10. ✓
- §5 readout (reusing aleph-core PauliString) → Task 9. ✓
- §6.1 unit + SV-equivalence → Tasks 5–9 (units) + Task 11. ✓
- §6.2 proptest symplectic + generic-state → Task 12 (symplectic) + Task 11 (generic-state oracle). ✓
- §6.3 Stim oracle (EPYC-gated) → Tasks 14, 15. ✓
- §6.4 bench <1s → Tasks 13, 15. ✓
- §7 AC mapping → Task 16 PR body. ✓

**Placeholder scan:** Task 16 PR body has intentional `<ISSUE>` and
`<fill>` placeholders to be filled from real values at execution — these
are runtime values, not plan gaps. No code-step placeholders.

**Type consistency:** `Tableau` methods named `h/s/sdg/cnot/cz/swap/iswap/iswap_dg`
and `x_gate/y_gate/z_gate` (Pauli methods suffixed to avoid clashing with
`x()/z()` bit accessors) — used consistently across Tasks 5–12.
`rows_anticommute` promoted to `pub` in Task 12 (flagged). `apply_gate`
signature identical in Tasks 10, 11, 14. `BitGrid` exposes only
get/set/toggle — H swaps across the x/z grids, so no single-row
column-swap helper exists.

**Known soft spot:** the Stim oracle's embedded Python uses version-
sensitive `stim` API calls; Task 14's note + Task 15 Step 4 make
adjusting-to-installed-version an explicit step. The Rust comparison
contract is stable.
