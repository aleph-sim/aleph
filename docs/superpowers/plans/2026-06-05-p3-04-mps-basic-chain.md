# P3-04 MPS backend — basic 1D chain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Matrix Product State (MPS) backend (`aleph-mps`) supporting 1q gates, nearest-neighbor 2q gates with fixed-χ SVD truncation, measurement, perfect sampling, expectation values, and joint-subset probabilities, wired into the `Backend` trait and the CLI.

**Architecture:** Mixed-canonical MPS (vector of rank-3 site tensors + an orthogonality center). 1q gates contract locally (no SVD); 2q nearest-neighbor gates contract two sites, apply the 4×4 gate, SVD-truncate to χ. SVD/QR via `nalgebra` (pure Rust, no LAPACK). `sample` uses perfect sampling (Ferris–Vidal); `probabilities` uses a doubled transfer-matrix sweep.

**Tech Stack:** Rust 2021, `nalgebra` (complex SVD/QR), `num_complex::Complex<f64>` (= `aleph_core::Complex`), `rand`, `proptest`, `assert_cmd`.

**Spec:** `docs/superpowers/specs/2026-06-05-p3-04-mps-basic-chain-design.md`

**Conventions locked (ADR 0004):**
- `amps[i]` ↔ qubit `q` has value `(i >> q) & 1`. Qubit 0 = LSB of the amplitude index. MPS dense reconstruction must use `index = Σ_q p_q · 2^q` where `p_q` is the physical value at site `q` (site `q` = qubit `q`).
- 2q gate matrix index: combined = `(qubits[1]_bit << 1) | qubits[0]_bit`. `qubits[0]` is the LSB of the 4×4 matrix index, `qubits[1]` the MSB.

**Crate fact:** `aleph_core::Complex` is `num_complex::Complex<f64>`; `nalgebra` re-exports the same `num_complex::Complex` (0.4) — `DMatrix<Complex>` interoperates with no conversion.

---

## File Structure

```
crates/aleph-mps/
  Cargo.toml          — deps: aleph-core, aleph-backend, aleph-ir, nalgebra, rand; dev: aleph-sv, proptest
  src/
    lib.rs            — crate doc, module decls, `MpsError`, re-exports (MpsState, MpsBackend)
    tensor.rs         — `Site` rank-3 tensor + reshape ↔ nalgebra DMatrix + SVD/QR primitives
    mps.rs            — `MpsState`: init, dense reconstruction, canonicalization, apply_1q, apply_2q,
                        expectation, measure, sample, probabilities
    gate.rs           — `GateInstance` → 2×2 / 4×4 unitary extraction
    backend.rs        — `MpsBackend` impl `Backend`, `MpsError → BackendError`
  tests/
    sv_equivalence.rs — oracle vs NaiveSvBackend + proptests + VQE-H2 + NN-QAOA acceptance
crates/aleph-cli/src/
  cli.rs              — add `BackendKind::Mps`, `--max-bond`
  exec.rs             — add `run_mps` path
Cargo.toml (root)     — add `nalgebra` to [workspace.dependencies]
```

---

## Task 0: Crate scaffolding + `MpsError`

**Files:**
- Modify: `Cargo.toml` (root, `[workspace.dependencies]`)
- Modify: `crates/aleph-mps/Cargo.toml`
- Modify: `crates/aleph-mps/src/lib.rs`

- [ ] **Step 1: Add `nalgebra` to workspace deps**

In root `Cargo.toml` under `[workspace.dependencies]`, add after the `serde_json` line:

```toml
nalgebra = "0.33"
```

- [ ] **Step 2: Fill `crates/aleph-mps/Cargo.toml`**

Replace the empty `[dependencies]` section with:

```toml
[dependencies]
aleph-core = { path = "../aleph-core" }
aleph-backend = { path = "../aleph-backend" }
aleph-ir = { path = "../aleph-ir" }
nalgebra = { workspace = true }
rand = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
aleph-sv = { path = "../aleph-sv" }
aleph-parser = { path = "../aleph-parser" }
proptest = { workspace = true }
```

- [ ] **Step 3: Write `lib.rs` with module decls + `MpsError`**

Replace `crates/aleph-mps/src/lib.rs` with:

```rust
//! `aleph-mps`: Matrix Product State (MPS) backend.
//!
//! Mixed-canonical MPS with fixed bond-dimension χ truncation. Handles 1q
//! gates and nearest-neighbor 2q gates; non-adjacent 2q gates (SWAP networks)
//! are P3-06, error-bounded truncation is P3-05.
//!
//! See `docs/superpowers/specs/2026-06-05-p3-04-mps-basic-chain-design.md`.

mod backend;
mod gate;
mod mps;
mod tensor;

pub use backend::MpsBackend;
pub use mps::MpsState;

/// Errors raised by the MPS state operations, before mapping to the shared
/// `aleph_backend::BackendError`.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MpsError {
    #[error("qubit {qubit} out of range for {num_qubits}-qubit state")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },

    #[error("gate `{kind}` is not supported by the MPS backend")]
    UnsupportedGate { kind: &'static str },

    #[error("2q gate on non-adjacent qubits {a} and {b}; the basic MPS chain only supports nearest-neighbor 2q gates (SWAP networks are P3-06)")]
    NonNearestNeighbor { a: u32, b: u32 },

    #[error("gate `{kind}` carries external controls, which the MPS backend does not support")]
    ExternalControls { kind: &'static str },

    #[error("gate `{kind}` has a non-finite (NaN or infinite) parameter")]
    NonFiniteParam { kind: &'static str },

    #[error("measurement of qubit {qubit} on a degenerate branch (p = {probability:e})")]
    DegenerateMeasurement { qubit: u32, probability: f64 },
}
```

> Note: `backend`, `gate`, `mps`, `tensor` modules don't exist yet — create empty stubs so the crate compiles: `echo "" > crates/aleph-mps/src/{backend,gate,mps,tensor}.rs`. Each task below fills one in. For Step 4 to pass, the stubs must at least define the `pub use`d items, so temporarily comment out the `pub use` lines and the `mod` lines for not-yet-written modules, OR create minimal stubs. Use minimal stubs: `tensor.rs`/`gate.rs`/`mps.rs` empty; `backend.rs` empty. Comment out the two `pub use` lines and `mod backend/gate/mps/tensor` until their tasks land, re-enabling per task. (Track this in each task's steps.)

For this task, write `lib.rs` with ONLY the `MpsError` enum and its doc — no `mod`/`pub use` lines yet:

```rust
//! `aleph-mps`: Matrix Product State (MPS) backend.
//! ... (doc as above) ...

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MpsError { /* ... as above ... */ }

#[cfg(test)]
mod tests {
    use super::MpsError;
    #[test]
    fn error_messages_render() {
        let e = MpsError::NonNearestNeighbor { a: 0, b: 3 };
        assert!(e.to_string().contains("non-adjacent"));
    }
}
```

- [ ] **Step 4: Build + test**

Run: `cargo test -p aleph-mps`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/aleph-mps/Cargo.toml crates/aleph-mps/src/lib.rs
git commit -m "[P3-04] aleph-mps crate scaffolding + MpsError"
```

---

## Task 1: `Site` rank-3 tensor + reshape primitives

**Files:**
- Create: `crates/aleph-mps/src/tensor.rs`
- Modify: `crates/aleph-mps/src/lib.rs` (add `mod tensor;`)

A site tensor has shape `(left, 2, right)` stored row-major: `data[(l*2 + p)*right + r]`. nalgebra `DMatrix` is column-major; we own the flat layout and convert at SVD/QR boundaries.

- [ ] **Step 1: Write failing tests**

Add to `tensor.rs`:

```rust
//! Rank-3 MPS site tensor `(left, 2, right)` and its reshape ↔ nalgebra views.

use aleph_core::Complex;
use nalgebra::DMatrix;

/// A single MPS site tensor of shape `(left, 2, right)`.
/// Row-major flat storage: `data[(l*2 + p)*right + r]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub left: usize,
    pub right: usize,
    pub data: Vec<Complex>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Complex;

    fn c(re: f64) -> Complex { Complex::new(re, 0.0) }

    #[test]
    fn ket0_site_shape() {
        let s = Site::ket0();
        assert_eq!((s.left, s.right), (1, 1));
        assert_eq!(s.get(0, 0, 0), c(1.0));
        assert_eq!(s.get(0, 1, 0), c(0.0));
    }

    #[test]
    fn group_left_roundtrip() {
        // 2×2×3 tensor, fill with distinct values; group-left → (2*2, 3) matrix
        // → back to Site must be identity.
        let mut s = Site::zeros(2, 3);
        for l in 0..2 { for p in 0..2 { for r in 0..3 {
            *s.get_mut(l, p, r) = c((l*100 + p*10 + r) as f64);
        }}}
        let m = s.to_group_left(); // (left*2) rows × right cols
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 3);
        let back = Site::from_group_left(&m, 2, 3);
        assert_eq!(back, s);
    }

    #[test]
    fn group_right_roundtrip() {
        let mut s = Site::zeros(2, 3);
        for l in 0..2 { for p in 0..2 { for r in 0..3 {
            *s.get_mut(l, p, r) = c((l*100 + p*10 + r) as f64);
        }}}
        let m = s.to_group_right(); // left rows × (2*right) cols
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 6);
        let back = Site::from_group_right(&m, 2, 3);
        assert_eq!(back, s);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-mps tensor`
Expected: FAIL (methods undefined).

- [ ] **Step 3: Implement `Site`**

Add the impl to `tensor.rs`:

```rust
impl Site {
    /// `(left, 2, right)` of zeros.
    pub fn zeros(left: usize, right: usize) -> Self {
        Site { left, right, data: vec![Complex::new(0.0, 0.0); left * 2 * right] }
    }

    /// The (1,2,1) tensor `[1, 0]` — a qubit in state |0⟩.
    pub fn ket0() -> Self {
        let mut s = Site::zeros(1, 1);
        s.data[0] = Complex::new(1.0, 0.0);
        s
    }

    #[inline]
    pub fn idx(&self, l: usize, p: usize, r: usize) -> usize {
        (l * 2 + p) * self.right + r
    }
    #[inline]
    pub fn get(&self, l: usize, p: usize, r: usize) -> Complex { self.data[self.idx(l, p, r)] }
    #[inline]
    pub fn get_mut(&mut self, l: usize, p: usize, r: usize) -> &mut Complex {
        let i = self.idx(l, p, r); &mut self.data[i]
    }

    /// Reshape to a `(left*2) × right` column-major matrix (groups the left
    /// bond and physical index into rows — the form for moving the
    /// orthogonality center rightward via QR).
    pub fn to_group_left(&self) -> DMatrix<Complex> {
        DMatrix::from_fn(self.left * 2, self.right, |row, r| {
            let l = row / 2; let p = row % 2; self.get(l, p, r)
        })
    }
    pub fn from_group_left(m: &DMatrix<Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        for row in 0..left * 2 { for r in 0..right {
            let l = row / 2; let p = row % 2; *s.get_mut(l, p, r) = m[(row, r)];
        }}
        s
    }

    /// Reshape to a `left × (2*right)` matrix (groups physical + right bond
    /// into columns — the form for moving the center leftward).
    pub fn to_group_right(&self) -> DMatrix<Complex> {
        DMatrix::from_fn(self.left, 2 * self.right, |l, col| {
            let p = col / self.right; let r = col % self.right; self.get(l, p, r)
        })
    }
    pub fn from_group_right(m: &DMatrix<Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        for l in 0..left { for col in 0..2 * right {
            let p = col / right; let r = col % right; *s.get_mut(l, p, r) = m[(l, col)];
        }}
        s
    }
}
```

- [ ] **Step 4: Add `mod tensor;` to lib.rs**

In `lib.rs`, add `mod tensor;` above the `MpsError` enum.

- [ ] **Step 5: Run tests**

Run: `cargo test -p aleph-mps tensor`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-mps/src/tensor.rs crates/aleph-mps/src/lib.rs
git commit -m "[P3-04] Site rank-3 tensor + reshape primitives"
```

---

## Task 2: `MpsState` init + dense reconstruction

**Files:**
- Create: `crates/aleph-mps/src/mps.rs`
- Modify: `crates/aleph-mps/src/lib.rs` (`mod mps;`, `pub use mps::MpsState;`)

`dense_statevector()` contracts the whole chain into a `Vec<Complex>` of length `2^n` — TEST-ONLY scale (small n), used to compare against `NaiveSvBackend::amplitudes()`. Amplitude index = `Σ_q p_q · 2^q` (ADR 0004).

- [ ] **Step 1: Write failing tests**

Add to `mps.rs`:

```rust
//! Mixed-canonical MPS state: init, dense reconstruction, canonicalization,
//! gate application, expectation, measurement, sampling, probabilities.

use aleph_core::Complex;
use crate::tensor::Site;

/// Mixed-canonical MPS. Sites left of `center` are left-canonical, sites right
/// are right-canonical; the center site carries the norm.
#[derive(Debug, Clone)]
pub struct MpsState {
    pub(crate) sites: Vec<Site>,
    pub(crate) center: usize,
    pub(crate) max_bond: usize,
    pub(crate) trunc_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm_sq(v: &[Complex]) -> f64 { v.iter().map(|c| c.norm_sqr()).sum() }

    #[test]
    fn ket0_dense_is_e0() {
        let s = MpsState::new(3, 64);
        let v = s.dense_statevector();
        assert_eq!(v.len(), 8);
        assert!((v[0].re - 1.0).abs() < 1e-12);
        assert!((norm_sq(&v) - 1.0).abs() < 1e-12);
        for k in 1..8 { assert!(v[k].norm() < 1e-12); }
    }

    #[test]
    fn single_qubit_dense() {
        let s = MpsState::new(1, 64);
        let v = s.dense_statevector();
        assert_eq!(v.len(), 2);
        assert!((v[0].re - 1.0).abs() < 1e-12);
        assert!(v[1].norm() < 1e-12);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-mps mps::tests::ket0`
Expected: FAIL (`MpsState::new` undefined).

- [ ] **Step 3: Implement `new` + `dense_statevector`**

```rust
impl MpsState {
    /// Allocate |0…0⟩ on `n` qubits with bond cap `max_bond`.
    pub fn new(n: usize, max_bond: usize) -> Self {
        let sites = (0..n).map(|_| Site::ket0()).collect();
        MpsState { sites, center: 0, max_bond: max_bond.max(1), trunc_error: 0.0 }
    }

    pub fn num_qubits(&self) -> usize { self.sites.len() }
    pub fn truncation_error(&self) -> f64 { self.trunc_error }

    /// Contract the whole chain into a dense `2^n` amplitude vector.
    /// TEST/SMALL-n ONLY (allocates 2^n). Amplitude index uses the ADR-0004
    /// convention: qubit `q` (== site `q`) occupies bit `q`.
    pub fn dense_statevector(&self) -> Vec<Complex> {
        let n = self.sites.len();
        // Running set of partial amplitude vectors keyed by right bond index.
        // Start: a 1×1 "row" = [1] over the left boundary (left bond dim 1).
        // acc[r] holds the amplitude contribution for each value of the current
        // right bond r, for the basis prefix encoded in `out` index bits.
        // We build the full 2^n vector by iterating sites and expanding.
        let mut amps: Vec<Complex> = vec![Complex::new(1.0, 0.0)]; // dim = left bond of site 0 = 1
        let mut left_dim = 1usize;
        // `amps` is indexed [basis_prefix * left_dim + l]. After all sites
        // left_dim == 1 so it collapses to the 2^n vector.
        for (q, site) in self.sites.iter().enumerate() {
            debug_assert_eq!(site.left, left_dim);
            let prefix_count = amps.len() / left_dim;
            let mut next = vec![Complex::new(0.0, 0.0); prefix_count * 2 * site.right];
            for prefix in 0..prefix_count {
                for p in 0..2usize {
                    let new_prefix = prefix | (p << q); // set bit q (sites are in order)
                    for r in 0..site.right {
                        let mut acc = Complex::new(0.0, 0.0);
                        for l in 0..left_dim {
                            acc += amps[prefix * left_dim + l] * site.get(l, p, r);
                        }
                        next[new_prefix * site.right + r] += acc;
                    }
                }
            }
            amps = next;
            left_dim = site.right;
        }
        debug_assert_eq!(left_dim, 1);
        amps
    }
}
```

> The `new_prefix = prefix | (p << q)` step relies on prefixes accumulating one bit per site in site order; since `prefix` only has bits `< q` set before site `q`, the OR sets bit `q` cleanly. Final `left_dim == 1` ⇒ `amps` has length `2^n` indexed by qubit-bit `q`.

- [ ] **Step 4: Wire lib.rs**

Add `mod mps;` and `pub use mps::MpsState;` to `lib.rs`.

- [ ] **Step 5: Run tests**

Run: `cargo test -p aleph-mps mps`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-mps/src/mps.rs crates/aleph-mps/src/lib.rs
git commit -m "[P3-04] MpsState init + dense reconstruction helper"
```

---

## Task 3: 1q gate extraction + `apply_1q`

**Files:**
- Create: `crates/aleph-mps/src/gate.rs`
- Modify: `crates/aleph-mps/src/mps.rs`, `crates/aleph-mps/src/lib.rs`

1q gate on site `i`: `A'[l,p',r] = Σ_p U[p'][p] · A[l,p,r]`. Canonicality preserved ⇒ no SVD, no center move.

- [ ] **Step 1: Write `gate.rs` 2×2 extraction + failing test**

```rust
//! Extract dense unitary matrices from `GateInstance` for the MPS backend.

use aleph_core::{Complex, GateInstance, GateMatrix};
use crate::MpsError;

/// Extract a 1q gate's 2×2 matrix. Rejects external controls and non-1q gates.
pub(crate) fn matrix_2x2(g: &GateInstance) -> Result<[[Complex; 2]; 2], MpsError> {
    if !g.controls.is_empty() {
        return Err(MpsError::ExternalControls { kind: g.gate.name() });
    }
    match g.gate.matrix() {
        Ok(GateMatrix::M2x2(m)) => {
            if m.iter().flatten().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
                return Err(MpsError::NonFiniteParam { kind: g.gate.name() });
            }
            Ok(m)
        }
        _ => Err(MpsError::UnsupportedGate { kind: g.gate.name() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn x_matrix() {
        let g = GateInstance::new(Gate::X, smallvec![0u32]);
        let m = matrix_2x2(&g).unwrap();
        assert!((m[0][1].re - 1.0).abs() < 1e-12);
        assert!((m[1][0].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_controls() {
        let g = GateInstance::controlled(Gate::X, smallvec![1u32], smallvec![0u32]);
        assert!(matches!(matrix_2x2(&g), Err(MpsError::ExternalControls { .. })));
    }
}
```

Add `smallvec = { workspace = true }` to `[dev-dependencies]` in `crates/aleph-mps/Cargo.toml` (tests construct `GateInstance`).

- [ ] **Step 2: Write `apply_1q` failing test in `mps.rs`**

Add to `mps.rs` tests module:

```rust
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    #[test]
    fn x_on_zero_is_one() {
        let mut s = MpsState::new(1, 64);
        s.apply_1q(0, &crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap());
        let v = s.dense_statevector();
        assert!(v[0].norm() < 1e-12);
        assert!((v[1].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn h_on_zero_is_plus() {
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        assert!((v[0].re - inv).abs() < 1e-12); // |00>
        assert!((v[1].re - inv).abs() < 1e-12); // |01> (q0=1)
        assert!(v[2].norm() < 1e-12);
        assert!(v[3].norm() < 1e-12);
    }
```

Add `smallvec` to dev-deps already done in Step 1.

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p aleph-mps`
Expected: FAIL (`apply_1q` undefined).

- [ ] **Step 4: Implement `apply_1q`**

In `mps.rs`:

```rust
impl MpsState {
    /// Apply a 1q unitary to site `i` (qubit `i`). Preserves canonical form,
    /// so neither the center nor any SVD is touched.
    pub(crate) fn apply_1q(&mut self, i: usize, u: &[[Complex; 2]; 2]) {
        let site = &mut self.sites[i];
        for l in 0..site.left {
            for r in 0..site.right {
                let a0 = site.get(l, 0, r);
                let a1 = site.get(l, 1, r);
                *site.get_mut(l, 0, r) = u[0][0] * a0 + u[0][1] * a1;
                *site.get_mut(l, 1, r) = u[1][0] * a0 + u[1][1] * a1;
            }
        }
    }
}
```

- [ ] **Step 5: Wire lib.rs**

Add `mod gate;` to `lib.rs`.

- [ ] **Step 6: Run tests**

Run: `cargo test -p aleph-mps`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-mps/src/gate.rs crates/aleph-mps/src/mps.rs crates/aleph-mps/src/lib.rs crates/aleph-mps/Cargo.toml
git commit -m "[P3-04] 1q gate extraction + apply_1q"
```

---

## Task 4: Canonicalization — QR center moves

**Files:**
- Modify: `crates/aleph-mps/src/tensor.rs` (QR split helpers), `crates/aleph-mps/src/mps.rs`

Move the orthogonality center one site at a time. Moving right (i → i+1): group site `i` left as `(left*2, right)` matrix `M`, QR-decompose `M = Q·R` (thin), set site `i = reshape(Q)` (left-canonical), absorb `R` into site `i+1`'s left bond: `A[i+1]'[l',p,r] = Σ_l R[l',l] A[i+1][l,p,r]`. Moving left is symmetric using `to_group_right` and an LQ (QR on the conjugate transpose).

- [ ] **Step 1: Write failing tests**

Add to `mps.rs` tests:

```rust
    /// Left-canonical check: Σ_{l,p} conj(A[l,p,r1]) A[l,p,r2] == δ.
    fn is_left_canonical(site: &Site) -> bool {
        for r1 in 0..site.right { for r2 in 0..site.right {
            let mut acc = Complex::new(0.0, 0.0);
            for l in 0..site.left { for p in 0..2 {
                acc += site.get(l, p, r1).conj() * site.get(l, p, r2);
            }}
            let expect = if r1 == r2 { 1.0 } else { 0.0 };
            if (acc.re - expect).abs() > 1e-9 || acc.im.abs() > 1e-9 { return false; }
        }}
        true
    }

    #[test]
    fn move_center_right_makes_left_canonical_and_preserves_state() {
        let mut s = MpsState::new(3, 64);
        // Put some entanglement-free content via 1q gates so tensors are non-trivial.
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h); s.apply_1q(1, &h);
        let before = s.dense_statevector();
        s.move_center_to(2);
        assert_eq!(s.center, 2);
        assert!(is_left_canonical(&s.sites[0]));
        assert!(is_left_canonical(&s.sites[1]));
        let after = s.dense_statevector();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).norm() < 1e-9, "state changed under canonicalization");
        }
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-mps move_center`
Expected: FAIL (`move_center_to` undefined).

- [ ] **Step 3: Implement QR helpers + center moves**

In `tensor.rs`, add a thin-QR helper returning `(Q, R)` with `Q` of shape `(rows, k)`, `R` of shape `(k, cols)`, `k = min(rows, cols)`:

```rust
/// Thin QR: returns (Q, R) with Q `rows×k`, R `k×cols`, k = min(rows, cols).
pub fn thin_qr(m: &DMatrix<Complex>) -> (DMatrix<Complex>, DMatrix<Complex>) {
    let qr = m.clone().qr();
    let q_full = qr.q();        // rows×rows
    let r_full = qr.r();        // rows×cols (upper-triangular)
    let k = m.nrows().min(m.ncols());
    let q = q_full.columns(0, k).into_owned();
    let r = r_full.rows(0, k).into_owned();
    (q, r)
}
```

In `mps.rs`:

```rust
use nalgebra::DMatrix;
use crate::tensor::thin_qr;

impl MpsState {
    /// Multiply matrix `r` into site `i`'s LEFT bond:
    /// A'[l',p,r2] = Σ_l r[l',l] · A[l,p,r2].
    fn absorb_into_left(&mut self, i: usize, r: &DMatrix<Complex>) {
        let site = &self.sites[i];
        let new_left = r.nrows();
        let mut out = Site::zeros(new_left, site.right);
        for lp in 0..new_left { for p in 0..2 { for r2 in 0..site.right {
            let mut acc = Complex::new(0.0, 0.0);
            for l in 0..site.left { acc += r[(lp, l)] * site.get(l, p, r2); }
            *out.get_mut(lp, p, r2) = acc;
        }}}
        self.sites[i] = out;
    }

    /// Multiply matrix `l` into site `i`'s RIGHT bond:
    /// A'[l2,p,r'] = Σ_r A[l2,p,r] · l[r,r'].
    fn absorb_into_right(&mut self, i: usize, l: &DMatrix<Complex>) {
        let site = &self.sites[i];
        let new_right = l.ncols();
        let mut out = Site::zeros(site.left, new_right);
        for l2 in 0..site.left { for p in 0..2 { for rp in 0..new_right {
            let mut acc = Complex::new(0.0, 0.0);
            for r in 0..site.right { acc += site.get(l2, p, r) * l[(r, rp)]; }
            *out.get_mut(l2, p, rp) = acc;
        }}}
        self.sites[i] = out;
    }

    /// Move center one step right: site i → left-canonical via QR, R into i+1.
    fn move_center_right(&mut self) {
        let i = self.center;
        let m = self.sites[i].to_group_left();      // (left*2) × right
        let (q, r) = thin_qr(&m);                    // q:(left*2)×k, r:k×right
        let k = q.ncols();
        self.sites[i] = Site::from_group_left(&q, self.sites[i].left, k);
        self.absorb_into_left(i + 1, &r);
        self.center += 1;
    }

    /// Move center one step left: site i → right-canonical, R into i-1.
    fn move_center_left(&mut self) {
        let i = self.center;
        // Right-canonical: group as left × (2*right), LQ via QR on the
        // conjugate transpose: Mᴴ = Q R  ⇒  M = Rᴴ Qᴴ; set site = Qᴴ (right
        // isometry), absorb Rᴴ into site i-1's right bond.
        let m = self.sites[i].to_group_right();      // left × (2*right)
        let mh = m.adjoint();                         // (2*right) × left
        let (q, r) = thin_qr(&mh);                     // q:(2*right)×k, r:k×left
        let k = q.ncols();
        let site_mat = q.adjoint();                    // k × (2*right) — right-canonical
        self.sites[i] = Site::from_group_right(&site_mat, k, self.sites[i].right);
        let r_into = r.adjoint();                      // left × k
        self.absorb_into_right(i - 1, &r_into);
        self.center -= 1;
    }

    /// Move the orthogonality center to site `target`.
    pub(crate) fn move_center_to(&mut self, target: usize) {
        while self.center < target { self.move_center_right(); }
        while self.center > target { self.move_center_left(); }
    }
}
```

> `from_group_right(&site_mat, k, right)`: `site_mat` is `k × (2*right)`, so left bond = `k`, right bond = `right`. Confirm dimensions in the test (the isometry check covers correctness).

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-mps`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/tensor.rs crates/aleph-mps/src/mps.rs
git commit -m "[P3-04] QR-based orthogonality-center moves"
```

---

## Task 5: 2q nearest-neighbor gate + SVD truncation

**Files:**
- Modify: `crates/aleph-mps/src/gate.rs` (4×4 extraction), `crates/aleph-mps/src/mps.rs`, `crates/aleph-mps/src/tensor.rs` (SVD helper)

Algorithm for a 2q gate on qubits `(qa, qb)` with `|qa-qb| == 1`:
1. Let `i = min(qa, qb)` (left site), `j = i+1` (right site). Move center to `i`.
2. Build two-site tensor `Θ[l, a, b, r]` where `a` = physical of site `i`, `b` = physical of site `j`: `Θ[l,a,b,r] = Σ_m sites[i][l,a,m] · sites[j][m,b,r]`.
3. Apply the 4×4 gate. The gate matrix `U` is indexed by `(qubits[1]_bit<<1)|qubits[0]_bit` (ADR 0004). Map each (a,b) for sites (i,j)=(qubit i, qubit j) to gate indices using the gate's actual `qubits` order: build `gate_index(phys_i, phys_j)` = `(bit_of(qubits[1])<<1)|bit_of(qubits[0])` where `bit_of(qubit i)=phys_i`, `bit_of(qubit j)=phys_j`. `Θ'[l,a',b',r] = Σ_{a,b} U[out(a',b')][out(a,b)] · Θ[l,a,b,r]`.
4. Reshape `Θ'` to matrix `M` of shape `(l*2) × (2*r)` (rows = (l,a'), cols = (b',r)). SVD: `M = U_s · diag(s) · V_sᴴ`.
5. Keep `χ = min(rank, max_bond)` largest singular values. `discarded = Σ_{k≥χ} s_k²`; `trunc_error += discarded`. Renormalize kept values: `kept_norm = sqrt(Σ_{k<χ} s_k²)`; if `kept_norm > 0`, scale kept singular values by `1/kept_norm * sqrt(Σ_all s_k²)`? — No: simpler, divide the kept singular-value vector by `sqrt(Σ_{k<χ} s_k²) / total_norm`. Since the pre-gate state is normalized and the gate is unitary, `Σ_all s_k² == 1`; after discarding, renormalize kept by `1/sqrt(Σ_{k<χ} s_k²)` to restore unit norm.
6. New site `i` = reshape(`U_s[:, :χ]`) into `(l, 2, χ)` (left-canonical). New site `j` = reshape(`diag(s_kept) · V_sᴴ[:χ, :]`) into `(χ, 2, r)` (the new center). Set `center = j`.

- [ ] **Step 1: Write `matrix_4x4` in `gate.rs` + test**

```rust
/// Extract a 2q gate's 4×4 matrix. Rejects controls / non-2q gates.
pub(crate) fn matrix_4x4(g: &GateInstance) -> Result<[[Complex; 4]; 4], MpsError> {
    if !g.controls.is_empty() {
        return Err(MpsError::ExternalControls { kind: g.gate.name() });
    }
    match g.gate.matrix() {
        Ok(GateMatrix::M4x4(m)) => {
            if m.iter().flatten().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
                return Err(MpsError::NonFiniteParam { kind: g.gate.name() });
            }
            Ok(m)
        }
        _ => Err(MpsError::UnsupportedGate { kind: g.gate.name() }),
    }
}
```

Test (CNOT is `[[1000],[0100],[0001],[0010]]` in the ADR-0004 ordering — assert a couple of entries):

```rust
#[test]
fn cnot_matrix_shape() {
    let g = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);
    let m = matrix_4x4(&g).unwrap();
    // ADR-0004: index = (q1<<1)|q0, q0 is control. Spot-check it's a permutation
    // with a single 1 per row.
    for row in &m {
        let ones: usize = row.iter().filter(|c| (c.re - 1.0).abs() < 1e-12).count();
        assert_eq!(ones, 1);
    }
}
```

- [ ] **Step 2: Write `truncated_svd` in `tensor.rs` + the 2q failing test in `mps.rs`**

`tensor.rs`:

```rust
/// SVD of `m` truncated to at most `max_bond` singular values, renormalized to
/// preserve total weight (assumes input came from a normalized state). Returns
/// `(u_kept, s_kept, vt_kept, discarded_weight)`:
///   u_kept: rows×χ, s_kept: length χ (real), vt_kept: χ×cols, discarded: f64.
pub fn truncated_svd(
    m: &DMatrix<Complex>,
    max_bond: usize,
) -> (DMatrix<Complex>, Vec<f64>, DMatrix<Complex>, f64) {
    let svd = m.clone().svd(true, true);
    let u = svd.u.expect("computed u");
    let vt = svd.v_t.expect("computed v_t");
    let s: Vec<f64> = svd.singular_values.iter().copied().collect();
    let rank = s.len();
    let chi = rank.min(max_bond.max(1));
    let discarded: f64 = s[chi..].iter().map(|x| x * x).sum();
    let kept_weight: f64 = s[..chi].iter().map(|x| x * x).sum();
    let scale = if kept_weight > 0.0 { (1.0 / kept_weight).sqrt() } else { 1.0 };
    let s_kept: Vec<f64> = s[..chi].iter().map(|x| x * scale).collect();
    let u_kept = u.columns(0, chi).into_owned();
    let vt_kept = vt.rows(0, chi).into_owned();
    (u_kept, s_kept, vt_kept, discarded)
}
```

> nalgebra's `svd(true, true)` returns singular values in DESCENDING order; rely on that for "keep top χ".

`mps.rs` failing test:

```rust
    fn c(re: f64) -> Complex { Complex::new(re, 0.0) } // if not already defined in this mod

    #[test]
    fn bell_via_h_cnot() {
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let cnot = crate::gate::matrix_4x4(&GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32])).unwrap();
        s.apply_2q(&GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]), &cnot).unwrap();
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        assert!((v[0].re - inv).abs() < 1e-10); // |00>
        assert!(v[1].norm() < 1e-10);
        assert!(v[2].norm() < 1e-10);
        assert!((v[3].re - inv).abs() < 1e-10); // |11>
        assert!(s.truncation_error() < 1e-12);  // χ=64, nothing discarded
    }

    #[test]
    fn rejects_non_adjacent() {
        let mut s = MpsState::new(3, 64);
        let cnot = crate::gate::matrix_4x4(&GateInstance::new(Gate::Cnot, smallvec![0u32, 2u32])).unwrap();
        let err = s.apply_2q(&GateInstance::new(Gate::Cnot, smallvec![0u32, 2u32]), &cnot).unwrap_err();
        assert!(matches!(err, MpsError::NonNearestNeighbor { a: 0, b: 2 }));
    }
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p aleph-mps bell_via_h_cnot`
Expected: FAIL (`apply_2q` undefined).

- [ ] **Step 4: Implement `apply_2q`**

In `mps.rs` (uses `truncated_svd`):

```rust
use crate::tensor::truncated_svd;
use aleph_core::GateInstance;
use crate::MpsError;

impl MpsState {
    /// Apply a 2q gate (its 4×4 matrix `u`) on nearest-neighbor qubits.
    pub(crate) fn apply_2q(&mut self, g: &GateInstance, u: &[[Complex; 4]; 4]) -> Result<(), MpsError> {
        let qa = g.qubits[0];
        let qb = g.qubits[1];
        if qa.abs_diff(qb) != 1 {
            return Err(MpsError::NonNearestNeighbor { a: qa, b: qb });
        }
        let i = qa.min(qb) as usize; // left site (= qubit i)
        let j = i + 1;               // right site (= qubit j)
        self.move_center_to(i);

        let (li, mi, ri) = (self.sites[i].left, self.sites[i].right, self.sites[j].right);
        // Θ[l,a,b,r] = Σ_m A_i[l,a,m] A_j[m,b,r]
        let mut theta = vec![Complex::new(0.0, 0.0); li * 2 * 2 * ri];
        let theta_idx = |l: usize, a: usize, b: usize, r: usize| ((l * 2 + a) * 2 + b) * ri + r;
        for l in 0..li { for a in 0..2 { for b in 0..2 { for r in 0..ri {
            let mut acc = Complex::new(0.0, 0.0);
            for m in 0..mi { acc += self.sites[i].get(l, a, m) * self.sites[j].get(m, b, r); }
            theta[theta_idx(l, a, b, r)] = acc;
        }}}}

        // gate output index per ADR-0004: (bit_of(qubits[1])<<1)|bit_of(qubits[0]),
        // bit_of(qubit i)=a (site i = qubit i), bit_of(qubit j)=b.
        let out = |phys_i: usize, phys_j: usize| -> usize {
            let bit_q0 = if g.qubits[0] as usize == i { phys_i } else { phys_j };
            let bit_q1 = if g.qubits[1] as usize == i { phys_i } else { phys_j };
            (bit_q1 << 1) | bit_q0
        };
        // Θ'[l,a',b',r] = Σ_{a,b} U[out(a',b')][out(a,b)] Θ[l,a,b,r]
        let mut theta2 = vec![Complex::new(0.0, 0.0); li * 2 * 2 * ri];
        for l in 0..li { for r in 0..ri {
            for ap in 0..2 { for bp in 0..2 {
                let mut acc = Complex::new(0.0, 0.0);
                for a in 0..2 { for b in 0..2 {
                    acc += u[out(ap, bp)][out(a, b)] * theta[theta_idx(l, a, b, r)];
                }}
                theta2[theta_idx(l, ap, bp, r)] = acc;
            }}
        }}

        // Reshape to M[(l,a'), (b',r)] = (li*2) × (2*ri), SVD-truncate.
        let m = DMatrix::from_fn(li * 2, 2 * ri, |row, col| {
            let l = row / 2; let a = row % 2;
            let b = col / ri; let r = col % ri;
            theta2[theta_idx(l, a, b, r)]
        });
        let (u_s, s_kept, vt_s, discarded) = truncated_svd(&m, self.max_bond);
        self.trunc_error += discarded;
        let chi = s_kept.len();

        // New site i = reshape(u_s) into (li, 2, chi) [from_group_left].
        self.sites[i] = Site::from_group_left(&u_s, li, chi);
        // New site j = reshape(diag(s) · vt_s) into (chi, 2, ri) [from_group_right].
        let mut sv = vt_s.clone();
        for row in 0..chi { for col in 0..sv.ncols() { sv[(row, col)] *= Complex::new(s_kept[row], 0.0); } }
        self.sites[j] = Site::from_group_right(&sv, chi, ri);
        self.center = j;
        Ok(())
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p aleph-mps`
Expected: PASS.

- [ ] **Step 6: Add a CZ + a reversed-qubit-order test**

Append to `mps.rs` tests:

```rust
    #[test]
    fn cnot_reversed_qubit_order() {
        // CNOT with qubits [1,0]: control=q1, target=q0. Prep q1=|1> then CNOT.
        let mut s = MpsState::new(2, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![1u32])).unwrap();
        s.apply_1q(1, &x); // q1 = 1
        let g = GateInstance::new(Gate::Cnot, smallvec![1u32, 0u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        s.apply_2q(&g, &cnot).unwrap();
        let v = s.dense_statevector();
        // q1=1 (control) flips q0 → both 1 → |11> index 3.
        assert!((v[3].re - 1.0).abs() < 1e-10);
    }
```

Run: `cargo test -p aleph-mps`; Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-mps/src/gate.rs crates/aleph-mps/src/mps.rs crates/aleph-mps/src/tensor.rs
git commit -m "[P3-04] 2q nearest-neighbor gate + fixed-χ SVD truncation"
```

---

## Task 6: `expectation_value`

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs`

`⟨ψ|P|ψ⟩` for a `PauliString`: clone the chain into ψ′, apply each single-qubit Pauli (via `apply_1q` with the Pauli's 2×2 matrix), then compute the overlap `⟨ψ|ψ′⟩` by a left-to-right transfer sweep: `E (1×1 start) ; E' = Σ_p A[i]_pᴴ(bra) · E · A[i]_p(ket)`. Multiply by `coefficient`, return `.re`.

- [ ] **Step 1: Write failing tests**

```rust
    use aleph_core::{Pauli, PauliString};

    fn bell(max_bond: usize) -> MpsState {
        let mut s = MpsState::new(2, max_bond);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let g = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        s.apply_2q(&g, &cnot).unwrap();
        s
    }

    #[test]
    fn expectation_bell() {
        let s = bell(64);
        let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        let xx = PauliString::new(1.0, vec![(0, Pauli::X), (1, Pauli::X)]).unwrap();
        let zi = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        assert!((s.expectation(&zz).unwrap() - 1.0).abs() < 1e-10);
        assert!((s.expectation(&xx).unwrap() - 1.0).abs() < 1e-10);
        assert!(s.expectation(&zi).unwrap().abs() < 1e-10);
        let half = PauliString::new(0.5, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        assert!((s.expectation(&half).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn expectation_oor() {
        let s = bell(64);
        let p = PauliString::new(1.0, vec![(5, Pauli::Z)]).unwrap();
        assert!(matches!(s.expectation(&p), Err(MpsError::QubitOutOfRange { qubit: 5, .. })));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-mps expectation`; Expected: FAIL.

- [ ] **Step 3: Implement `expectation` + `overlap`**

```rust
impl MpsState {
    /// ⟨self|other⟩ via a left-to-right transfer sweep. Both must have the
    /// same qubit count and matching physical dimension (2).
    fn overlap(&self, other: &MpsState) -> Complex {
        // E: bra_bond × ket_bond, start 1×1 identity [1].
        let mut e = DMatrix::<Complex>::from_element(1, 1, Complex::new(1.0, 0.0));
        for i in 0..self.sites.len() {
            let bra = &self.sites[i];
            let ket = &other.sites[i];
            // E'[lb', lk'] would build right; we contract incrementally:
            // E_new[rb, rk] = Σ_p Σ_{lb,lk} conj(bra[lb,p,rb]) E[lb,lk] ket[lk,p,rk]
            let mut e_new = DMatrix::<Complex>::zeros(bra.right, ket.right);
            for p in 0..2 {
                // tmp[lb, rk] = Σ_lk E[lb,lk] ket[lk,p,rk]
                let mut tmp = DMatrix::<Complex>::zeros(bra.left, ket.right);
                for lb in 0..bra.left { for rk in 0..ket.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for lk in 0..ket.left { acc += e[(lb, lk)] * ket.get(lk, p, rk); }
                    tmp[(lb, rk)] = acc;
                }}
                // E_new[rb, rk] += Σ_lb conj(bra[lb,p,rb]) tmp[lb,rk]
                for rb in 0..bra.right { for rk in 0..ket.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for lb in 0..bra.left { acc += bra.get(lb, p, rb).conj() * tmp[(lb, rk)]; }
                    e_new[(rb, rk)] += acc;
                }}
            }
            e = e_new;
        }
        e[(0, 0)]
    }

    /// ⟨ψ|P|ψ⟩ for a Pauli string. Returns coefficient · Re⟨ψ|Pψ⟩.
    pub(crate) fn expectation(&self, p: &PauliString) -> Result<f64, MpsError> {
        let n = self.sites.len();
        let mut pp = self.clone();
        for (q, pauli) in &p.terms {
            let qi = *q as usize;
            if qi >= n { return Err(MpsError::QubitOutOfRange { qubit: *q, num_qubits: n as u32 }); }
            if let aleph_core::Pauli::I = pauli { continue; }
            let m = pauli.matrix();
            pp.apply_1q(qi, &m);
        }
        let ov = self.overlap(&pp);
        Ok(p.coefficient * ov.re)
    }
}
```

> `aleph_core::Pauli::matrix(self) -> [[Complex;2];2]` exists (pauli.rs:21).

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-mps expectation`; Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/mps.rs
git commit -m "[P3-04] expectation_value via Pauli-applied overlap sweep"
```

---

## Task 7: `measure` with collapse

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs`

Move center to qubit `q`. With identity environment at the center, the single-site reduced density-matrix diagonal is `p(b) = Σ_{l,r} |A[l,b,r]|²`. Sample with rng, project site `q` onto |outcome⟩ (zero the other physical component), renormalize site by `1/√p`. Center stays at `q`.

- [ ] **Step 1: Write failing tests**

```rust
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn measure_zero_is_zero() {
        let mut s = MpsState::new(1, 64);
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(s.measure(0, &mut rng).unwrap(), false);
    }

    #[test]
    fn measure_ghz_correlated() {
        // GHZ-3: H(0), CNOT(0,1), CNOT(1,2). Measuring all qubits → all equal.
        let mut s = MpsState::new(3, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..2u32 {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        let mut rng = StdRng::seed_from_u64(7);
        let b0 = s.measure(0, &mut rng).unwrap();
        let b1 = s.measure(1, &mut rng).unwrap();
        let b2 = s.measure(2, &mut rng).unwrap();
        assert_eq!(b0, b1);
        assert_eq!(b1, b2);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-mps measure`; Expected: FAIL.

- [ ] **Step 3: Implement `measure`**

```rust
use rand::Rng;

impl MpsState {
    /// Measure qubit `q` in the Z basis, collapsing the state. Returns the bit.
    pub(crate) fn measure<R: Rng>(&mut self, q: usize, rng: &mut R) -> Result<bool, MpsError> {
        let n = self.sites.len();
        if q >= n { return Err(MpsError::QubitOutOfRange { qubit: q as u32, num_qubits: n as u32 }); }
        self.move_center_to(q);
        let site = &self.sites[q];
        let mut p0 = 0.0;
        for l in 0..site.left { for r in 0..site.right { p0 += site.get(l, 0, r).norm_sqr(); } }
        let mut p1 = 0.0;
        for l in 0..site.left { for r in 0..site.right { p1 += site.get(l, 1, r).norm_sqr(); } }
        let total = p0 + p1;
        if total <= 0.0 {
            return Err(MpsError::DegenerateMeasurement { qubit: q as u32, probability: total });
        }
        let p0n = p0 / total;
        let outcome = rng.gen::<f64>() >= p0n; // true (=1) with prob p1
        let keep = if outcome { 1usize } else { 0usize };
        let pk = if outcome { p1 } else { p0 };
        let scale = (total / pk).sqrt(); // renormalize: divide by sqrt(pk/total)
        let site = &mut self.sites[q];
        for l in 0..site.left { for r in 0..site.right {
            let drop = 1 - keep;
            *site.get_mut(l, drop, r) = Complex::new(0.0, 0.0);
            let v = site.get(l, keep, r);
            *site.get_mut(l, keep, r) = v * Complex::new(scale, 0.0);
        }}
        Ok(outcome)
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-mps measure`; Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/mps.rs
git commit -m "[P3-04] single-qubit measurement with collapse"
```

---

## Task 8: `sample` (perfect sampling) + `probabilities`

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs`

`MAX_PROB_QUBITS = 20`.

**`sample`:** canonicalize a working clone to right-canonical (center = 0). For each shot, sweep left→right keeping a left boundary row-vector `bnd` (length = left bond, starts `[1]`). At site `i`: for each `b∈{0,1}` compute the partial vector `w_b[r] = Σ_l bnd[l]·A[i][l,b,r]`; `prob(b) = ‖w_b‖²` (since the rest of the chain is right-canonical/isometric ⇒ identity environment); sample `b`; set `bnd ← w_b / √prob(b)`. Pack `1u64<<q` per the convention.

**`probabilities`:** doubled transfer-matrix sweep (spec). Maintain `Vec<(pattern, E)>` where `E` is a `bra_bond × ket_bond` matrix (here bra==ket==self). For sites not in the subset, contract over physical (sum); for sites in the subset, split into two patterns appending the bit. Map patterns → output index by each requested qubit's position in the `qubits` slice.

- [ ] **Step 1: Write failing tests**

```rust
    fn ghz(n: usize) -> MpsState {
        let mut s = MpsState::new(n, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..(n as u32 - 1) {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        s
    }

    #[test]
    fn sample_ghz_all_equal() {
        let s = ghz(4);
        let mut rng = StdRng::seed_from_u64(3);
        let shots = s.sample(500, &mut rng);
        for sh in shots { assert!(sh == 0b0000 || sh == 0b1111, "bad GHZ shot {sh:04b}"); }
    }

    #[test]
    fn probabilities_plus_state() {
        // H on q0 of a 2q state → marginal of q0 is uniform.
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let p = s.probabilities(&[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-10);
        assert!((p[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn probabilities_bell_joint() {
        let s = ghz(2); // Bell
        let p = s.probabilities(&[0, 1]).unwrap();
        // (|00>+|11>)/√2 → p[00]=0.5, p[11]=0.5; index bit pos0=q0, pos1=q1.
        assert!((p[0b00] - 0.5).abs() < 1e-10);
        assert!(p[0b01].abs() < 1e-10);
        assert!(p[0b10].abs() < 1e-10);
        assert!((p[0b11] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn probabilities_empty_subset_is_one() {
        let s = ghz(2);
        assert_eq!(s.probabilities(&[]).unwrap(), vec![1.0]);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-mps sample probabilities`; Expected: FAIL.

- [ ] **Step 3: Implement `sample` + `probabilities`**

```rust
/// Largest subset size `probabilities` will materialize (output is 2^k).
pub(crate) const MAX_PROB_QUBITS: usize = 20;

impl MpsState {
    /// Perfect sampling (Ferris–Vidal 2012). Does not mutate `self`.
    /// Each shot packs qubit `q` into bit `q`.
    pub(crate) fn sample<R: Rng>(&self, shots: u32, rng: &mut R) -> Vec<u64> {
        let n = self.sites.len();
        // Work on a right-canonical clone (center = 0) so the right environment
        // is the identity at every site during the left→right sweep.
        let mut work = self.clone();
        work.move_center_to(0);
        let mut out = Vec::with_capacity(shots as usize);
        for _ in 0..shots {
            // bnd: row vector over the current left bond, starts [1] (left bond
            // of site 0 is 1).
            let mut bnd = vec![Complex::new(1.0, 0.0)];
            let mut bits = 0u64;
            for i in 0..n {
                let site = &work.sites[i];
                // w_b[r] = Σ_l bnd[l] A[l,b,r]
                let mut w = [
                    vec![Complex::new(0.0, 0.0); site.right],
                    vec![Complex::new(0.0, 0.0); site.right],
                ];
                for b in 0..2 { for r in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for l in 0..site.left { acc += bnd[l] * site.get(l, b, r); }
                    w[b][r] = acc;
                }}
                let p0: f64 = w[0].iter().map(|c| c.norm_sqr()).sum();
                let p1: f64 = w[1].iter().map(|c| c.norm_sqr()).sum();
                let total = p0 + p1;
                let outcome = rng.gen::<f64>() * total >= p0; // 1 with prob p1/total
                let b = if outcome { 1usize } else { 0usize };
                if outcome { bits |= 1u64 << i; }
                let pk = if outcome { p1 } else { p0 };
                let scale = if pk > 0.0 { (1.0 / pk).sqrt() } else { 0.0 };
                bnd = w[b].iter().map(|c| *c * Complex::new(scale, 0.0)).collect();
            }
            out.push(bits);
        }
        out
    }

    /// Exact joint marginal over `qubits` (length 2^k). Matches the SV backend
    /// contract: empty → [1.0]; output bit `pos` corresponds to `qubits[pos]`.
    /// Validation (duplicate / out-of-range) is left to the backend wrapper.
    pub(crate) fn probabilities(&self, qubits: &[u32]) -> Result<Vec<f64>, MpsError> {
        let n = self.sites.len();
        if qubits.is_empty() { return Ok(vec![1.0]); }
        if qubits.len() > MAX_PROB_QUBITS {
            // Backend maps to TooManyQubits; here surface UnsupportedGate-style.
            return Err(MpsError::UnsupportedGate { kind: "probabilities(subset too large)" });
        }
        // out_bit_for_site[site] = Some(pos) if site is requested at slice pos.
        let mut out_bit_for_site: Vec<Option<usize>> = vec![None; n];
        for (pos, &q) in qubits.iter().enumerate() {
            if (q as usize) >= n {
                return Err(MpsError::QubitOutOfRange { qubit: q, num_qubits: n as u32 });
            }
            out_bit_for_site[q as usize] = Some(pos);
        }
        // envs: (output_index_so_far, E matrix bra_bond×ket_bond). Start [1].
        let mut envs: Vec<(usize, DMatrix<Complex>)> =
            vec![(0usize, DMatrix::from_element(1, 1, Complex::new(1.0, 0.0)))];
        for i in 0..n {
            let site = &self.sites[i];
            let contract_p = |e: &DMatrix<Complex>, p: usize| -> DMatrix<Complex> {
                // E'[rb,rk] = Σ_{lb,lk} conj(A[lb,p,rb]) E[lb,lk] A[lk,p,rk]
                let mut tmp = DMatrix::<Complex>::zeros(site.left, site.right);
                for lb in 0..site.left { for rk in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for lk in 0..site.left { acc += e[(lb, lk)] * site.get(lk, p, rk); }
                    tmp[(lb, rk)] = acc;
                }}
                let mut e_new = DMatrix::<Complex>::zeros(site.right, site.right);
                for rb in 0..site.right { for rk in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for lb in 0..site.left { acc += site.get(lb, p, rb).conj() * tmp[(lb, rk)]; }
                    e_new[(rb, rk)] = acc;
                }}
                e_new
            };
            match out_bit_for_site[i] {
                None => {
                    // Trace out: sum over physical.
                    for (_, e) in envs.iter_mut() {
                        *e = &contract_p(e, 0) + &contract_p(e, 1);
                    }
                }
                Some(pos) => {
                    let mut next = Vec::with_capacity(envs.len() * 2);
                    for (idx, e) in &envs {
                        next.push((*idx, contract_p(e, 0)));            // bit 0
                        next.push((*idx | (1 << pos), contract_p(e, 1))); // bit 1
                    }
                    envs = next;
                }
            }
        }
        let dim = 1usize << qubits.len();
        let mut out = vec![0.0; dim];
        for (idx, e) in envs {
            debug_assert_eq!((e.nrows(), e.ncols()), (1, 1));
            out[idx] = e[(0, 0)].re;
        }
        Ok(out)
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-mps`; Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/mps.rs
git commit -m "[P3-04] perfect sampling + joint-subset probabilities"
```

---

## Task 9: `MpsBackend` impl `Backend`

**Files:**
- Create: `crates/aleph-mps/src/backend.rs`
- Modify: `crates/aleph-mps/src/lib.rs` (`mod backend;`, `pub use backend::MpsBackend;`)

- [ ] **Step 1: Write `backend.rs` + failing trait-level tests**

```rust
//! `MpsBackend`: the `aleph_backend::Backend` impl over `MpsState`.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Gate, GateInstance, PauliString};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::mps::MAX_PROB_QUBITS;
use crate::{MpsError, MpsState};

/// MPS backend with a configurable max bond dimension χ.
pub struct MpsBackend {
    rng: StdRng,
    max_bond: usize,
}

const MAX_QUBITS: u32 = 1024;
const DEFAULT_MAX_BOND: usize = 128;

impl MpsBackend {
    pub fn new() -> Self { Self { rng: StdRng::from_entropy(), max_bond: DEFAULT_MAX_BOND } }
    pub fn with_seed(seed: u64) -> Self { Self { rng: StdRng::seed_from_u64(seed), max_bond: DEFAULT_MAX_BOND } }
    pub fn with_max_bond(mut self, chi: usize) -> Self { self.max_bond = chi.max(1); self }
}
impl Default for MpsBackend { fn default() -> Self { Self::new() } }

fn map_mps_err(e: MpsError) -> BackendError {
    match e {
        MpsError::QubitOutOfRange { qubit, num_qubits } => BackendError::QubitOutOfRange { qubit, num_qubits },
        MpsError::UnsupportedGate { kind } => BackendError::UnsupportedGate { kind },
        MpsError::ExternalControls { kind } => BackendError::UnsupportedGate { kind },
        MpsError::NonFiniteParam { kind } => BackendError::NonFiniteParam { kind },
        MpsError::NonNearestNeighbor { .. } => BackendError::InvalidState {
            reason: "non-adjacent 2q gate requires a SWAP network (see P3-06)",
        },
        MpsError::DegenerateMeasurement { qubit, probability } => {
            BackendError::DegenerateMeasurement { qubit, probability }
        }
    }
}

impl Backend for MpsBackend {
    type State = MpsState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_QUBITS {
            return Err(BackendError::TooManyQubits { requested: num_qubits, limit: MAX_QUBITS });
        }
        Ok(MpsState::new(num_qubits as usize, self.max_bond))
    }

    fn apply_gate(&mut self, state: &mut Self::State, gate: &GateInstance) -> Result<(), BackendError> {
        match gate.gate.arity() {
            1 => {
                let m = crate::gate::matrix_2x2(gate).map_err(map_mps_err)?;
                let q = gate.qubits[0] as usize;
                if q >= state.num_qubits() {
                    return Err(BackendError::QubitOutOfRange { qubit: gate.qubits[0], num_qubits: state.num_qubits() as u32 });
                }
                state.apply_1q(q, &m);
                Ok(())
            }
            2 => {
                let m = crate::gate::matrix_4x4(gate).map_err(map_mps_err)?;
                for &q in &gate.qubits {
                    if q as usize >= state.num_qubits() {
                        return Err(BackendError::QubitOutOfRange { qubit: q, num_qubits: state.num_qubits() as u32 });
                    }
                }
                state.apply_2q(gate, &m).map_err(map_mps_err)
            }
            _ => Err(BackendError::UnsupportedGate { kind: gate.gate.name() }),
        }
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        state.measure(qubit as usize, &mut self.rng).map_err(map_mps_err)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        let n = state.num_qubits();
        if n > 64 {
            return Err(BackendError::TooManyQubits { requested: n as u32, limit: 64 });
        }
        Ok(state.sample(shots, &mut self.rng))
    }

    fn expectation_value(&mut self, state: &Self::State, pauli: &PauliString) -> Result<f64, BackendError> {
        state.expectation(pauli).map_err(map_mps_err)
    }

    fn probabilities(&mut self, state: &Self::State, qubits: &[u32]) -> Result<Vec<f64>, BackendError> {
        // Validate duplicates here (matches the SV backend contract).
        let mut seen = Vec::new();
        for &q in qubits {
            if seen.contains(&q) { return Err(BackendError::DuplicateQubit { qubit: q }); }
            seen.push(q);
        }
        if qubits.len() > MAX_PROB_QUBITS {
            return Err(BackendError::TooManyQubits { requested: qubits.len() as u32, limit: MAX_PROB_QUBITS as u32 });
        }
        state.probabilities(qubits).map_err(map_mps_err)
    }
}

#[cfg(test)]
mod tests {
    use super::MpsBackend;
    use aleph_backend::{Backend, BackendError};
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    #[test]
    fn bell_sample_correlated() {
        let mut be = MpsBackend::with_seed(0);
        let mut s = be.allocate(2).unwrap();
        be.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        be.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32])).unwrap();
        for sh in be.sample(&s, 200).unwrap() { assert!(sh == 0b00 || sh == 0b11); }
    }

    #[test]
    fn rejects_three_qubit_gate() {
        let mut be = MpsBackend::with_seed(0);
        let mut s = be.allocate(3).unwrap();
        let err = be.apply_gate(&mut s, &GateInstance::new(Gate::Toffoli, smallvec![0u32, 1u32, 2u32])).unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedGate { .. }));
    }

    #[test]
    fn rejects_non_adjacent() {
        let mut be = MpsBackend::with_seed(0);
        let mut s = be.allocate(3).unwrap();
        let err = be.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0u32, 2u32])).unwrap_err();
        assert!(matches!(err, BackendError::InvalidState { .. }));
    }

    #[test]
    fn probabilities_duplicate_rejected() {
        let mut be = MpsBackend::with_seed(0);
        let s = be.allocate(2).unwrap();
        assert!(matches!(be.probabilities(&s, &[0, 0]), Err(BackendError::DuplicateQubit { qubit: 0 })));
    }
}
```

> Verify `Gate::Toffoli` is the actual variant name (check `crates/aleph-core/src/gate/kinds.rs`); if it differs (e.g. `Gate::Ccx`), use the real name.

- [ ] **Step 2: Wire lib.rs**

Add `mod backend;` and `pub use backend::MpsBackend;`.

- [ ] **Step 3: Run + clippy + fmt**

Run: `cargo test -p aleph-mps && cargo clippy -p aleph-mps --all-targets -- -D warnings && cargo fmt -p aleph-mps --check`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-mps/src/backend.rs crates/aleph-mps/src/lib.rs
git commit -m "[P3-04] MpsBackend impl Backend + error mapping"
```

---

## Task 10: CLI integration — `--backend mps` + `--max-bond`

**Files:**
- Modify: `crates/aleph-cli/src/cli.rs`, `crates/aleph-cli/src/exec.rs`, `crates/aleph-cli/Cargo.toml`

- [ ] **Step 1: Add `aleph-mps` dep to CLI**

In `crates/aleph-cli/Cargo.toml` `[dependencies]`, add: `aleph-mps = { path = "../aleph-mps" }`.

- [ ] **Step 2: Extend `BackendKind` + add `--max-bond`**

In `cli.rs`, add the `Mps` variant:

```rust
    /// MPS tensor network (bounded entanglement). χ via --max-bond.
    Mps,
```

In the `Run` subcommand struct, add after the `backend` arg:

```rust
        /// MPS max bond dimension χ (only used by --backend mps).
        #[arg(long, default_value_t = 128)]
        max_bond: usize,
```

Update the `backend` doc line to mention `mps`. Then propagate `max_bond` through `main.rs`/`lib.rs` to `run_circuit` (add a `max_bond: usize` parameter to `run_circuit`). Find the call site in `crates/aleph-cli/src/lib.rs` or `main.rs` and thread it through.

- [ ] **Step 3: Add `run_mps` + dispatch in `exec.rs`**

Add `use aleph_mps::MpsBackend;`. Add a `max_bond: usize` parameter to `run_circuit` and the dispatch:

```rust
    if backend == BackendKind::Mps {
        return run_mps(
            &circuit, effective_shots, print_statevector || force_statevector,
            &paulis, n, seed, max_bond, &seed_label, out,
        );
    }
```

Add the function (mirrors `run_stabilizer`):

```rust
#[allow(clippy::too_many_arguments)]
fn run_mps<W: Write>(
    circuit: &aleph_ir::Circuit,
    effective_shots: Option<u32>,
    statevector_requested: bool,
    paulis: &[(String, aleph_core::PauliString)],
    n: u32,
    seed: Option<u64>,
    max_bond: usize,
    seed_label: &str,
    out: &mut W,
) -> Result<()> {
    if statevector_requested {
        return Err(anyhow!(
            "the MPS backend does not expose a dense state vector; drop --statevector \
             (use --shots and/or --expectation instead)"
        ));
    }
    let mut backend = match seed {
        Some(s) => MpsBackend::with_seed(s),
        None => MpsBackend::new(),
    }.with_max_bond(max_bond);
    let state = run(&mut backend, circuit).context("running circuit (mps)")?;
    if let Some(shots) = effective_shots {
        let samples = backend.sample(&state, shots).context("sampling final state")?;
        output::format_counts(out, &samples, shots, n, seed_label)?;
    }
    if !paulis.is_empty() {
        writeln!(out, "expectation values:")?;
        for (raw, ps) in paulis {
            let v = backend.expectation_value(&state, ps)
                .with_context(|| format!("computing expectation value for {raw:?}"))?;
            output::format_expectation(out, raw, v)?;
        }
    }
    Ok(())
}
```

- [ ] **Step 4: Add integration tests**

In the CLI's integration test file (find existing `crates/aleph-cli/tests/*.rs`, follow its `assert_cmd` style), add tests that:
- Run a Bell QASM with `--backend mps --shots 200` and assert output contains counts for `00` and `11` only.
- Run with `--backend mps --statevector` and assert it errors with "does not expose a dense state vector".

Use the existing test's fixture-writing helper. Example skeleton (adapt to the file's existing helpers):

```rust
#[test]
fn mps_backend_bell_counts() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bell.qasm");
    std::fs::write(&path, "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n").unwrap();
    let mut cmd = Command::cargo_bin("aleph").unwrap();
    cmd.args(["run", path.to_str().unwrap(), "--backend", "mps", "--shots", "200", "--seed", "1"]);
    cmd.assert().success().stdout(predicate::str::contains("00").or(predicate::str::contains("11")));
}
```

- [ ] **Step 5: Run + clippy + fmt**

Run: `cargo test -p aleph-cli && cargo clippy -p aleph-cli --all-targets -- -D warnings && cargo fmt --check`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-cli
git commit -m "[P3-04] CLI --backend mps + --max-bond"
```

---

## Task 11: Oracle equivalence + property tests

**Files:**
- Create: `crates/aleph-mps/tests/sv_equivalence.rs`

Compare MPS (χ large = exact) against `NaiveSvBackend` via dense reconstruction and expectation, plus proptests for the canonical invariant and norm.

- [ ] **Step 1: Write the oracle equivalence tests**

```rust
//! Oracle equivalence vs NaiveSvBackend + MPS invariants.
//! MPS dense reconstruction must match the SV amplitude vector (ADR-0004).

use aleph_backend::{run, Backend};
use aleph_core::{Complex, Gate, GateInstance, Pauli, PauliString};
use aleph_mps::{MpsBackend, MpsState};
use aleph_sv::NaiveSvBackend;
use smallvec::smallvec;

/// Reconstruct the MPS dense vector by running the same circuit on a fresh
/// MpsBackend and pulling the internal dense_statevector via a test hook.
/// (dense_statevector is pub(crate); expose it for tests through the State.)
fn mps_dense(circuit: &aleph_ir::Circuit, chi: usize) -> Vec<Complex> {
    let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
    let st: MpsState = run(&mut be, circuit).unwrap();
    st.dense_statevector()
}

fn sv_dense(circuit: &aleph_ir::Circuit) -> Vec<Complex> {
    let mut be = NaiveSvBackend::with_seed(0);
    let st = run(&mut be, circuit).unwrap();
    st.amplitudes().to_vec()
}

#[test]
fn bell_matches_sv() {
    let mut c = aleph_ir::Circuit::new(2);
    c.push_gate(GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
    c.push_gate(GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32])).unwrap();
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) { assert!((x - y).norm() < 1e-10); }
}
```

> Prerequisite: `dense_statevector` must be `pub` (it was defined `pub` in Task 2 — confirm, and add a doc note "test/debug only, allocates 2^n"). `truncation_error` is already `pub`. The integration test is an external crate, so it can only touch `pub` items + the `Backend` trait. Also confirm the `aleph_ir::Circuit` builder API: check `Circuit::new`, `push_gate`, `is_empty` names in `crates/aleph-ir/src`. If they differ (e.g. `Circuit::try_new`, `add_gate`), use the real names. Inspect before writing.

- [ ] **Step 2: Add a multi-qubit nearest-neighbor circuit oracle + expectation oracle**

```rust
#[test]
fn nn_chain_matches_sv() {
    // 5-qubit: layer of H, then nearest-neighbor CNOT ladder, then RZ rotations.
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n);
    for q in 0..n { c.push_gate(GateInstance::new(Gate::H, smallvec![q])).unwrap(); }
    for q in 0..n - 1 { c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap(); }
    // Use an available parameterized 1q gate; check kinds.rs for Rz(theta) constructor.
    for q in 0..n { c.push_gate(GateInstance::new(Gate::rz(0.3 + q as f64 * 0.1), smallvec![q])).unwrap(); }
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) { assert!((x - y).norm() < 1e-10); }
}

#[test]
fn expectation_matches_sv() {
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n);
    for q in 0..n { c.push_gate(GateInstance::new(Gate::H, smallvec![q])).unwrap(); }
    for q in 0..n - 1 { c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap(); }
    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();
    for terms in [vec![(0u32, Pauli::Z), (1, Pauli::Z)], vec![(0, Pauli::X), (2, Pauli::X)], vec![(1, Pauli::Z)]] {
        let p = PauliString::new(1.0, terms).unwrap();
        let em = mps.expectation_value(&ms, &p).unwrap();
        let es = sv.expectation_value(&svs, &p).unwrap();
        assert!((em - es).abs() < 1e-10, "expectation mismatch: {em} vs {es}");
    }
}

#[test]
fn probabilities_matches_sv() {
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n);
    for q in 0..n { c.push_gate(GateInstance::new(Gate::H, smallvec![q])).unwrap(); }
    for q in 0..n - 1 { c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap(); }
    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();
    for subset in [vec![0u32], vec![0, 2], vec![1, 3, 0]] {
        let pm = mps.probabilities(&ms, &subset).unwrap();
        let ps = sv.probabilities(&svs, &subset).unwrap();
        assert_eq!(pm.len(), ps.len());
        for (x, y) in pm.iter().zip(ps.iter()) { assert!((x - y).abs() < 1e-10); }
    }
}
```

> Confirm a parameterized rotation constructor (e.g. `Gate::rz(f64)`); inspect `kinds.rs`. If rotations need `aleph_core::Param`, build accordingly. If simpler, drop RZ and use only Clifford 1q+2q for the oracle (still exercises the MPS paths).

- [ ] **Step 3: Add proptests (canonical invariant, norm, χ=∞ exact)**

```rust
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random nearest-neighbor 1q+2q circuit on 4 qubits, χ=64 (no truncation):
    /// MPS dense must equal SV dense to 1e-9, and norm ≈ 1.
    #[test]
    fn random_nn_circuit_matches_sv(seq in prop::collection::vec(0u8..6, 0..30)) {
        let n = 4u32;
        let mut c = aleph_ir::Circuit::new(n);
        let mut rngq = 0u32;
        for op in seq {
            rngq = (rngq + 1) % n;
            match op {
                0 => c.push_gate(GateInstance::new(Gate::H, smallvec![rngq])).unwrap(),
                1 => c.push_gate(GateInstance::new(Gate::X, smallvec![rngq])).unwrap(),
                2 => c.push_gate(GateInstance::new(Gate::S, smallvec![rngq])).unwrap(),
                3 => c.push_gate(GateInstance::new(Gate::Y, smallvec![rngq])).unwrap(),
                _ => {
                    let q = rngq.min(n - 2);
                    c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap();
                }
            }
        }
        if c.is_empty() { return Ok(()); }
        let a = mps_dense(&c, 64);
        let b = sv_dense(&c);
        let mut norm = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            prop_assert!((x - y).norm() < 1e-9);
            norm += x.norm_sqr();
        }
        prop_assert!((norm - 1.0).abs() < 1e-9);
    }
}
```

> `push_gate` returns `Result`; `.unwrap()` inside the closure is fine in tests. Confirm `Circuit::is_empty` exists; otherwise guard on `seq` emptiness. Confirm `Gate::S`, `Gate::Y` variant names.

- [ ] **Step 4: Add weak-entanglement small-χ near-exact test**

```rust
#[test]
fn small_chi_weak_entanglement_near_exact() {
    // Shallow nearest-neighbor circuit keeps Schmidt rank low; χ=4 should be
    // near-exact and trunc_error tiny.
    let n = 6u32;
    let mut c = aleph_ir::Circuit::new(n);
    for q in 0..n { c.push_gate(GateInstance::new(Gate::H, smallvec![q])).unwrap(); }
    for q in 0..n - 1 { c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap(); }
    let a = mps_dense(&c, 4);
    let b = sv_dense(&c);
    let mut err = 0.0;
    for (x, y) in a.iter().zip(b.iter()) { err += (x - y).norm_sqr(); }
    assert!(err.sqrt() < 1e-6, "L2 error {} too large", err.sqrt());
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p aleph-mps --test sv_equivalence`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-mps/tests/sv_equivalence.rs crates/aleph-mps/src/mps.rs
git commit -m "[P3-04] oracle equivalence vs NaiveSv + invariant proptests"
```

---

## Task 12: VQE-H₂ + NN-QAOA acceptance, bench, docs

**Files:**
- Modify: `crates/aleph-mps/tests/sv_equivalence.rs` (acceptance tests)
- Create: `crates/aleph-mps/benches/nn_qaoa.rs` (+ bench wiring in `Cargo.toml`)
- Modify: `crates/aleph-mps/src/lib.rs` (crate docs), `CLAUDE.md` (backend table if present)

- [ ] **Step 1: VQE-H₂ @ 4 qubits machine-precision test**

Build a 4-qubit hardware-efficient ansatz with nearest-neighbor entanglers (a representative VQE-H₂ circuit: layers of `Ry` rotations + NN CNOT ladder). Compare MPS (χ=64) dense to SV dense at 1e-10.

```rust
#[test]
fn vqe_h2_matches_sv_machine_precision() {
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n);
    let thetas = [0.31, 0.59, 0.27, 0.18, 0.44, 0.62, 0.11, 0.53];
    let mut t = thetas.iter().cycle();
    for _layer in 0..2 {
        for q in 0..n { c.push_gate(GateInstance::new(Gate::ry(*t.next().unwrap()), smallvec![q])).unwrap(); }
        for q in 0..n - 1 { c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap(); }
    }
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) { assert!((x - y).norm() < 1e-10); }
}
```

> Confirm `Gate::ry(f64)` constructor name in `kinds.rs`; adapt if it's `Gate::Ry(Param)`.

- [ ] **Step 2: NN-QAOA depth-3 @ 50 qubits "reasonable" test (`#[ignore]`)**

```rust
#[test]
#[ignore = "50-qubit MPS run; minutes-scale, runs on CI nightly"]
fn qaoa50_nn_ring_runs_reasonably() {
    let n = 50u32;
    let mut c = aleph_ir::Circuit::new(n);
    for q in 0..n { c.push_gate(GateInstance::new(Gate::H, smallvec![q])).unwrap(); }
    for _p in 0..3 {
        // Cost layer: nearest-neighbor ring ZZ via CNOT–RZ–CNOT on (q, q+1).
        for q in 0..n - 1 {
            c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap();
            c.push_gate(GateInstance::new(Gate::rz(0.7), smallvec![q + 1])).unwrap();
            c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap();
        }
        // Mixer: RX on every qubit.
        for q in 0..n { c.push_gate(GateInstance::new(Gate::rx(0.5), smallvec![q])).unwrap(); }
    }
    let mut be = MpsBackend::with_seed(1).with_max_bond(64);
    let st = run(&mut be, &c).unwrap();
    // "Reasonable": bounded truncation, normalized, non-degenerate sampling.
    assert!(st.truncation_error() < 1e-1, "trunc_error {}", st.truncation_error());
    let shots = be.sample(&st, 1000).unwrap();
    let distinct: std::collections::HashSet<u64> = shots.iter().copied().collect();
    assert!(distinct.len() > 1, "sampling produced a single bitstring");
}
```

> `truncation_error()` is `pub` (Task 2). Confirm `Gate::rx` / `Gate::rz` names. Time-box: if it exceeds 30s locally it stays `#[ignore]` per CLAUDE.md.

- [ ] **Step 3: Bench (criterion) for NN-QAOA scaling**

Add to `crates/aleph-mps/Cargo.toml`:

```toml
[dev-dependencies]
criterion = { workspace = true }

[[bench]]
name = "nn_qaoa"
harness = false
```

Create `crates/aleph-mps/benches/nn_qaoa.rs` benching the NN-QAOA depth-3 circuit at n ∈ {10, 20, 30} with χ=64 (wall-time only; no ratio gate — P3-04 has no perf AC). Use the `Gate` constructors confirmed above. Keep the bench small enough to finish quickly.

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use aleph_backend::run;
use aleph_core::{Gate, GateInstance};
use aleph_mps::MpsBackend;
use smallvec::smallvec;

fn qaoa_circuit(n: u32) -> aleph_ir::Circuit {
    let mut c = aleph_ir::Circuit::new(n);
    for q in 0..n { c.push_gate(GateInstance::new(Gate::H, smallvec![q])).unwrap(); }
    for _ in 0..3 {
        for q in 0..n - 1 {
            c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap();
            c.push_gate(GateInstance::new(Gate::rz(0.7), smallvec![q + 1])).unwrap();
            c.push_gate(GateInstance::new(Gate::Cnot, smallvec![q, q + 1])).unwrap();
        }
        for q in 0..n { c.push_gate(GateInstance::new(Gate::rx(0.5), smallvec![q])).unwrap(); }
    }
    c
}

fn bench(cr: &mut Criterion) {
    let mut g = cr.benchmark_group("nn_qaoa_chi64");
    for n in [10u32, 20, 30] {
        let c = qaoa_circuit(n);
        g.bench_function(format!("n{n}"), |b| b.iter(|| {
            let mut be = MpsBackend::with_seed(0).with_max_bond(64);
            run(&mut be, &c).unwrap()
        }));
    }
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
```

- [ ] **Step 4: Crate docs + CLAUDE.md**

Expand `lib.rs` crate doc with a short "Usage" example (allocate, apply gates, sample). If `CLAUDE.md` has a backend list/table, add the MPS backend row. If not, skip.

- [ ] **Step 5: Full workspace gate**

Run:
```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo build --release -p aleph-mps
```
Expected: all PASS. (The `#[ignore]`d 50-qubit test is skipped.)

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-mps CLAUDE.md
git commit -m "[P3-04] VQE-H2 + NN-QAOA acceptance, bench, docs"
```

---

## Final verification (before PR)

- [ ] `cargo test --workspace` green.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] Run the `#[ignore]`d 50-qubit QAOA test once locally (`cargo test -p aleph-mps --release -- --ignored qaoa50`) and record wall-time in the PR body.
- [ ] No `unwrap()`/`expect()` in library code (only in tests / the `svd().u.expect` which is a documented infallible-after-compute call — add a SAFETY-style comment).
- [ ] Self-review the diff.

## PR

- Title: `[P3-04] MPS backend — basic 1D chain`
- Body: `Closes #35` (verify the issue number for P3-04 via `gh issue list` — memory says #35). Summary of approach, test results (oracle 1e-10, proptests, VQE-H₂ machine precision, QAOA-50 wall-time), note that perf has no ratio AC, note nalgebra (pure Rust, no LAPACK) dependency justification, and that error-bounded truncation (P3-05) + non-adjacent gates (P3-06) are deferred.

---

## Notes for the implementer (read before starting)

1. **The 2q gate-index convention (ADR-0004) is the canonical footgun** (see CLAUDE.md + memory P1-07/P1-10). The `out(phys_i, phys_j)` mapping in Task 5 and the dense-reconstruction bit order in Task 2 must agree with `NaiveSvBackend`. The oracle test (Task 11) is the guard — if `bell_matches_sv` fails, suspect the index mapping first.
2. **nalgebra SVD returns descending singular values** and complex `u`/`v_t`; singular values are real (`f64`). QR `q()`/`r()` give full-size factors — slice to thin.
3. **Verify API names before writing each task**: `aleph_ir::Circuit` builder (`new`/`push_gate`/`is_empty`), `Gate` rotation constructors (`rx`/`ry`/`rz` vs `Rx(Param)`), `Gate::Toffoli` vs `Gate::Ccx`, `Pauli::matrix`. Grep `crates/aleph-core/src/gate/kinds.rs` and `crates/aleph-ir/src/`.
4. **Pure Rust ⇒ local == EPYC.** No `is_x86_feature_detected!` path; local aarch64 runs the same code, so EPYC validation is not required for correctness (unlike SV kernels).
5. **No `unwrap` in lib code.** The only `expect` is on `svd.u`/`svd.v_t` (always `Some` when `svd(true, true)`); annotate it.
