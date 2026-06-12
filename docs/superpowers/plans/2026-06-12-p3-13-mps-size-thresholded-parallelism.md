# [P3-13] MPS Size-Thresholded Per-Call Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Choose faer parallelism per operation from the measured χ-crossover (P3-09, `docs/perf/mps_parallel.md`) instead of faer's process-global default: small ops run `Par::Seq`, wide-bond ops use the rayon pool — removing the feature-unification trap where compiling `faer/rayon` anywhere in the graph silently flips every `MpsBackend` user onto the χ≤256 pessimization (1.5×–19× slower).

**Architecture:** A new `aleph-mps/src/linalg.rs` module hosts (a) the size-threshold policy `par_for(rows, cols) -> Par` calibrated from P3-09 data, and (b) explicit-`Par` replicas of faer's high-level `thin_svd()`/`qr()` (which hard-read `get_global_parallelism()`): `thin_svd_par` over `faer::linalg::svd::svd` and `thin_qr_par` over `faer::linalg::qr::no_pivoting::factor::qr_in_place` + the householder apply. All five hot-path call sites in `mps.rs` (theta gemm, truncated SVD, two thin-QR center moves, two absorption gemms) thread a per-op `Par` through. `MpsState` gains a `pub(crate) par_override: Option<Par>` test knob so the Par-invariance oracle compares Seq vs rayon **as plain arguments** (no global mutation — fixes the P3-09 review finding). After EPYC measurement validates the AC, the `parallel` cargo feature flips to default-ON (small ops no longer regress), with the `wide_bond` bench gaining a runtime env guard so push-to-main Bench on the shared EPYC runner stays cheap.

**Tech Stack:** Rust 1.89, faer 0.24 (`linalg`, `std`, optional `rayon`), criterion, proptest. EPYC box `195.154.249.85` for measurement.

**Issue:** #149 · **Branch:** `p3-13-mps-size-thresholded-par` (in main checkout — NO worktree) · **PR title:** `[P3-13] MPS: size-thresholded per-call parallelism`

---

## Key facts (verified against faer 0.24.0 source, 2026-06-12)

- `MatRef::thin_svd()` → `Svd::new_imp` and `MatRef::qr()` → `Qr::new_imp` both call `get_global_parallelism()` internally (`faer-0.24.0/src/linalg/solvers.rs:1115,1344`). There is no per-call knob at the high level — replication over the low-level API is required.
- Low-level signatures:
  - `faer::linalg::svd::svd(A, s: DiagMut, u: Option<MatMut>, v: Option<MatMut>, par: Par, stack: &mut MemStack, params: Spec<SvdParams, T>) -> Result<(), SvdError>` + `svd_scratch::<T>(m, n, compute_u, compute_v, par, params)`.
  - `faer::linalg::qr::no_pivoting::factor::{qr_in_place(A: MatMut, Q_coeff: MatMut, par, stack, params) -> QrInfo, qr_in_place_scratch::<T>(m, n, block_size, par, params), recommended_block_size::<T>(m, n)}`.
  - `faer::linalg::householder::{apply_block_householder_sequence_on_the_left_in_place_with_conj(basis: MatRef, coeff: MatRef, Conj::No, matrix: MatMut, par, stack), apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(basis_nrows, block_size, rhs_ncols)}`.
- `faer::dyn_stack` is `pub extern crate` — `faer::dyn_stack::{MemBuffer, MemStack}` works, **no new Cargo dependency**.
- Public types: `faer::Par` (`Copy + PartialEq + Eq + Debug`; the `Rayon` variant and `Par::rayon(n)` are `#[cfg(feature = "rayon")]`), `faer::Conj`, `faer::diag::Diag`.
- After `qr_in_place`, the factored matrix holds R in the upper trapezoid and householder vectors strictly below the diagonal; faer's `Qr` builds `Q_basis` as the unit-diagonal lower trapezoid of the first `size = min(m,n)` columns (`split_LU`), and `compute_thin_Q()` applies the householder sequence to `Mat::identity(m, size)`.
- `aleph_core::Complex == faer::c64` (P3-09) — canonical element type, no `Conj::Yes` handling needed in the SVD replica.
- Threshold calibration (`docs/perf/mps_parallel.md`, EPYC 16c): χ=256 ops (largest operand 512×1024 = 524 288 elements) are a rayon pessimization; χ=512 ops (1024×2048 = 2 097 152) win 1.57× @16T. `PAR_MIN_ELEMS = 1 << 20` (1 048 576) is the geometric midpoint.
- Without the `rayon` feature, `get_global_parallelism()` is always `Par::Seq`, so `par_for` degrades to a no-op — single code path for both builds.
- Only aleph-mps uses faer in this workspace (`grep faer crates/*/Cargo.toml`); `aleph-py` and `aleph-cli` depend on aleph-mps **with default features** — they pick up the default flip in Task 9.

## File structure

- **Create** `crates/aleph-mps/src/linalg.rs` — threshold policy + explicit-`Par` thin-SVD/thin-QR wrappers + their unit tests. One responsibility: "faer with `Par` as an argument".
- **Modify** `crates/aleph-mps/src/lib.rs` — register `mod linalg;`, update the parallelism doc section.
- **Modify** `crates/aleph-mps/src/tensor.rs` — `truncated_svd` gains a `par: faer::Par` parameter; body swaps `m.thin_svd()` for `linalg::thin_svd_par(m, par)`.
- **Modify** `crates/aleph-mps/src/mps.rs` — `par_override` field, `choose_par` helper, five call sites rewired, new state-level invariance unit test.
- **Modify** `crates/aleph-mps/tests/sv_equivalence.rs` — delete the global-toggle `results_invariant_across_parallelism` test (replaced by the unit test).
- **Modify** `crates/aleph-mps/Cargo.toml` + `crates/aleph-mps/benches/wide_bond.rs` — Task 9 default-flip + runtime bench guard (measurement-gated).
- **Modify** `docs/perf/mps_parallel.md`, `BACKLOG.md` — results section, checkboxes.

---

### Task 0: Branch + claim the issue

- [ ] **Step 0.1: Create the branch (no worktree) and claim**

```bash
cd /Users/ex/GitHub/aleph
git fetch origin
git checkout -b p3-13-mps-size-thresholded-par origin/main
gh issue comment 149 --body "Working on this."
```

Expected: branch created tracking origin/main; comment posted.

---

### Task 1: `linalg.rs` — threshold policy `par_for`

**Files:**
- Create: `crates/aleph-mps/src/linalg.rs`
- Modify: `crates/aleph-mps/src/lib.rs` (add `mod linalg;`)

- [ ] **Step 1.1: Write the new module with the policy and failing-first tests**

Create `crates/aleph-mps/src/linalg.rs`:

```rust
//! Explicit-`Par` faer wrappers and the size-threshold parallelism policy
//! (P3-13). faer's high-level `thin_svd()`/`qr()` hard-read the process-global
//! parallelism, which the `parallel` cargo feature flips to rayon for every
//! caller in the build graph (feature unification) — a 1.5×–19× pessimization
//! at χ ≤ 256 (docs/perf/mps_parallel.md). These wrappers take `Par` per call
//! so each operation chooses from its own operand size instead.

use crate::MpsError;
use aleph_core::Complex;
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder::{
    apply_block_householder_sequence_on_the_left_in_place_scratch,
    apply_block_householder_sequence_on_the_left_in_place_with_conj,
};
use faer::linalg::qr::no_pivoting::factor::{
    qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::{Conj, Mat, MatRef, Par};

/// Minimum operand element count (`rows · cols`) for the rayon pool to pay
/// off. Calibrated from the P3-09 EPYC sweep (docs/perf/mps_parallel.md):
/// χ=256 ops (up to 512×1024 = 524 288 elements) are a rayon pessimization,
/// while χ=512 ops (1024×2048 = 2 097 152) win 1.57× @16T. 2^20 is the
/// geometric midpoint of that interval; the P3-13 EPYC sweep validates both
/// sides.
const PAR_MIN_ELEMS: usize = 1 << 20;

/// Whether a `rows × cols` operand is large enough to amortize fork-join
/// overhead (feature-independent threshold arithmetic, unit-tested directly).
#[inline]
pub(crate) fn wants_parallel(rows: usize, cols: usize) -> bool {
    rows.saturating_mul(cols) >= PAR_MIN_ELEMS
}

/// `Par` for one operation on a `rows × cols` operand: the global setting
/// (rayon when the `parallel` feature is compiled in) above the threshold,
/// `Par::Seq` below it. Without the feature the global is always `Par::Seq`,
/// so this degrades to a no-op.
pub(crate) fn par_for(rows: usize, cols: usize) -> Par {
    if wants_parallel(rows, cols) {
        faer::get_global_parallelism()
    } else {
        Par::Seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_boundaries() {
        // Largest χ=256 operand (the measured pessimization) stays sequential.
        assert!(!wants_parallel(512, 1024));
        assert!(!wants_parallel(0, usize::MAX)); // saturating, not panicking
        // 1024×1024 (= PAR_MIN_ELEMS exactly) and up may parallelize.
        assert!(wants_parallel(1024, 1024));
        assert!(wants_parallel(1024, 2048));
        assert!(wants_parallel(usize::MAX, usize::MAX));
    }

    #[test]
    fn par_for_below_threshold_is_seq() {
        assert_eq!(par_for(512, 512), Par::Seq);
    }

    #[test]
    fn par_for_above_threshold_follows_global() {
        assert_eq!(par_for(2048, 2048), faer::get_global_parallelism());
    }
}
```

Register the module in `crates/aleph-mps/src/lib.rs` — next to the existing `mod` items (e.g., after `mod gate;`):

```rust
mod linalg;
```

- [ ] **Step 1.2: Run the module tests — verify they pass**

```bash
cargo test -p aleph-mps linalg:: 2>&1 | tail -5
```

Expected: `test result: ok. 3 passed` (the SVD/QR wrappers come in Tasks 2–3; this task is policy-only).

- [ ] **Step 1.3: Commit**

```bash
git add crates/aleph-mps/src/linalg.rs crates/aleph-mps/src/lib.rs
git commit -m "[P3-13] Add size-threshold parallelism policy par_for

PAR_MIN_ELEMS = 2^20 elements, the geometric midpoint of the P3-09
crossover (512x1024 pessimizes, 1024x2048 wins 1.57x @16T on EPYC)."
```

---

### Task 2: `thin_svd_par` — explicit-`Par` thin SVD

**Files:**
- Modify: `crates/aleph-mps/src/linalg.rs`

- [ ] **Step 2.1: Write the failing bit-exactness test first**

Append inside `mod tests` in `crates/aleph-mps/src/linalg.rs`:

```rust
    /// Deterministic full-rank-ish complex test matrix.
    fn test_matrix(m: usize, n: usize) -> Mat<Complex> {
        Mat::from_fn(m, n, |i, j| {
            Complex::new(
                ((i * 7 + j * 3) % 11) as f64 * 0.37 - 1.1,
                ((i * 5 + j) % 7) as f64 * 0.23 - 0.6,
            )
        })
    }

    /// The replica must be BIT-EXACT vs faer's high-level thin_svd when given
    /// the same Par the high-level call reads from the global — any divergence
    /// means the low-level invocation differs (wrong compute mode, params, or
    /// conj handling), which a tolerance test could mask.
    #[test]
    fn thin_svd_par_matches_high_level_bit_exact() {
        for (m, n) in [(8usize, 5usize), (5, 8), (6, 6)] {
            let a = test_matrix(m, n);
            let hl = a.thin_svd().unwrap();
            let (u, s, v) = thin_svd_par(a.as_ref(), faer::get_global_parallelism()).unwrap();
            let size = Ord::min(m, n);
            assert_eq!(u.shape(), (m, size));
            assert_eq!(v.shape(), (n, size));
            for t in 0..size {
                assert_eq!(s.as_ref()[t], hl.S()[t], "S[{t}] ({m}x{n})");
            }
            for r in 0..m {
                for c in 0..size {
                    assert_eq!(u[(r, c)], hl.U()[(r, c)], "U[({r},{c})] ({m}x{n})");
                }
            }
            for r in 0..n {
                for c in 0..size {
                    assert_eq!(v[(r, c)], hl.V()[(r, c)], "V[({r},{c})] ({m}x{n})");
                }
            }
        }
    }
```

- [ ] **Step 2.2: Run it — verify it fails to compile (function not defined)**

```bash
cargo test -p aleph-mps linalg:: 2>&1 | grep -m1 "cannot find\|error"
```

Expected: `error[E0425]: cannot find function thin_svd_par`.

- [ ] **Step 2.3: Implement `thin_svd_par`**

Add above `#[cfg(test)]` in `crates/aleph-mps/src/linalg.rs`:

```rust
/// Thin SVD with an explicit `Par`: `A = U · diag(S) · Vᴴ` with
/// `U: m × size`, `V: n × size`, `size = min(m, n)`. Mirrors
/// `faer::linalg::solvers::Svd::new_thin` — which hard-reads the global
/// parallelism (faer-0.24.0 solvers.rs:1344) — for the canonical c64 element
/// type (no conjugation pass needed: `aleph_core::Complex == faer::c64`).
pub(crate) fn thin_svd_par(
    a: MatRef<'_, Complex>,
    par: Par,
) -> Result<(Mat<Complex>, Diag<Complex>, Mat<Complex>), MpsError> {
    let (m, n) = a.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::<Complex>::zeros(m, size);
    let mut v = Mat::<Complex>::zeros(n, size);
    let mut s = Diag::<Complex>::zeros(size);
    svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut MemBuffer::new(svd_scratch::<Complex>(
            m,
            n,
            ComputeSvdVectors::Thin,
            ComputeSvdVectors::Thin,
            par,
            Default::default(),
        ))),
        Default::default(),
    )
    .map_err(|_| MpsError::SvdFailed)?;
    Ok((u, s, v))
}
```

- [ ] **Step 2.4: Run the tests — verify they pass**

```bash
cargo test -p aleph-mps linalg:: 2>&1 | tail -3
```

Expected: `test result: ok. 4 passed`.

- [ ] **Step 2.5: Commit**

```bash
git add crates/aleph-mps/src/linalg.rs
git commit -m "[P3-13] Add thin_svd_par: thin SVD with explicit Par

Replicates faer's Svd::new_thin over linalg::svd::svd; bit-exact
equivalence test pins the replication."
```

---

### Task 3: `thin_qr_par` — explicit-`Par` thin QR

**Files:**
- Modify: `crates/aleph-mps/src/linalg.rs`

- [ ] **Step 3.1: Write the failing bit-exactness test first**

Append inside `mod tests`:

```rust
    /// Same bit-exactness rationale as the SVD test. Covers m>n, m<n, m=n —
    /// the m<n case exercises the trapezoidal (not triangular) R and the
    /// size×size householder basis.
    #[test]
    fn thin_qr_par_matches_high_level_bit_exact() {
        for (m, n) in [(8usize, 5usize), (5, 8), (6, 6)] {
            let a = test_matrix(m, n);
            let hl = a.qr();
            let hq = hl.compute_thin_Q();
            let hr = hl.thin_R();
            let (q, r) = thin_qr_par(a.to_owned(), faer::get_global_parallelism());
            let size = Ord::min(m, n);
            assert_eq!(q.shape(), (m, size));
            assert_eq!(r.shape(), (size, n));
            assert_eq!(q.shape(), hq.shape());
            assert_eq!((r.nrows(), r.ncols()), (hr.nrows(), hr.ncols()));
            for i in 0..m {
                for j in 0..size {
                    assert_eq!(q[(i, j)], hq[(i, j)], "Q[({i},{j})] ({m}x{n})");
                }
            }
            for i in 0..size {
                for j in 0..n {
                    assert_eq!(r[(i, j)], hr[(i, j)], "R[({i},{j})] ({m}x{n})");
                }
            }
        }
    }
```

- [ ] **Step 3.2: Run it — verify it fails to compile**

```bash
cargo test -p aleph-mps linalg:: 2>&1 | grep -m1 "cannot find\|error"
```

Expected: `error[E0425]: cannot find function thin_qr_par`.

- [ ] **Step 3.3: Implement `thin_qr_par`**

Add above `#[cfg(test)]`:

```rust
/// Thin QR with an explicit `Par`: returns `(thin_Q, thin_R)` with
/// `thin_Q: m × size` (orthonormal columns), `thin_R: size × n` upper
/// trapezoidal, `size = min(m, n)`. Mirrors `faer::linalg::solvers::Qr::new`
/// + `compute_thin_Q()`/`thin_R()` — which hard-read the global parallelism
/// (faer-0.24.0 solvers.rs:1115,1196). Takes the input by value: it doubles
/// as the in-place factorization workspace (the high-level path makes the
/// same `to_owned()` copy internally).
pub(crate) fn thin_qr_par(mut qr: Mat<Complex>, par: Par) -> (Mat<Complex>, Mat<Complex>) {
    let (m, n) = qr.shape();
    let size = Ord::min(m, n);
    let block_size = recommended_block_size::<Complex>(m, n);
    let mut q_coeff = Mat::<Complex>::zeros(block_size, size);
    let _ = qr_in_place(
        qr.as_mut(),
        q_coeff.as_mut(),
        par,
        MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<Complex>(
            m,
            n,
            block_size,
            par,
            Default::default(),
        ))),
        Default::default(),
    );
    // After qr_in_place: R sits in the upper trapezoid, householder vectors
    // strictly below the diagonal (faer qr/no_pivoting/factor.rs docs).
    let mut thin_r = Mat::<Complex>::zeros(size, n);
    for i in 0..size {
        for j in i..n {
            thin_r[(i, j)] = qr[(i, j)];
        }
    }
    // Householder basis = unit-diagonal lower trapezoid of the first `size`
    // columns — exactly faer's split_LU L factor (solvers.rs:955).
    let mut basis = Mat::<Complex>::zeros(m, size);
    for j in 0..size {
        basis[(j, j)] = Complex::new(1.0, 0.0);
        for i in (j + 1)..m {
            basis[(i, j)] = qr[(i, j)];
        }
    }
    let mut thin_q = Mat::<Complex>::identity(m, size);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        basis.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        thin_q.as_mut(),
        par,
        MemStack::new(&mut MemBuffer::new(
            apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex>(
                m, block_size, size,
            ),
        )),
    );
    (thin_q, thin_r)
}
```

- [ ] **Step 3.4: Run the tests — verify they pass**

```bash
cargo test -p aleph-mps linalg:: 2>&1 | tail -3
```

Expected: `test result: ok. 5 passed`.

- [ ] **Step 3.5: Add the Seq-vs-rayon reconstruction invariance test (parallel feature only)**

Append inside `mod tests`:

```rust
    /// Both helpers must produce a valid factorization under either Par —
    /// reconstructions (which are unique, unlike the factors' phases/signs)
    /// must match the input to 1e-12. Run via:
    /// cargo test -p aleph-mps --features parallel
    #[cfg(feature = "parallel")]
    #[test]
    fn helpers_reconstruct_under_seq_and_rayon() {
        let (m, n) = (48usize, 32usize);
        let a = test_matrix(m, n);
        for par in [Par::Seq, Par::rayon(0)] {
            let (u, s, v) = thin_svd_par(a.as_ref(), par).unwrap();
            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex::new(0.0, 0.0);
                    for k in 0..n {
                        acc += u[(i, k)] * s.as_ref()[k] * v[(j, k)].conj();
                    }
                    assert!(
                        (acc - a[(i, j)]).norm() < 1e-12,
                        "SVD reconstruction ({i},{j}) under {par:?}"
                    );
                }
            }
            let (q, r) = thin_qr_par(a.to_owned(), par);
            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex::new(0.0, 0.0);
                    for k in 0..n {
                        acc += q[(i, k)] * r[(k, j)];
                    }
                    assert!(
                        (acc - a[(i, j)]).norm() < 1e-12,
                        "QR reconstruction ({i},{j}) under {par:?}"
                    );
                }
            }
        }
    }
```

- [ ] **Step 3.6: Run with the parallel feature — verify all pass**

```bash
cargo test -p aleph-mps --features parallel linalg:: 2>&1 | tail -3
```

Expected: `test result: ok. 6 passed`.

- [ ] **Step 3.7: Commit**

```bash
git add crates/aleph-mps/src/linalg.rs
git commit -m "[P3-13] Add thin_qr_par: thin QR with explicit Par

Replicates faer's Qr::new + compute_thin_Q/thin_R over qr_in_place +
the block householder apply. Bit-exact test vs the high-level path;
Seq-vs-rayon reconstruction invariance under --features parallel."
```

---

### Task 4: Thread `Par` through `truncated_svd` and all `mps.rs` call sites

**Files:**
- Modify: `crates/aleph-mps/src/tensor.rs` (signature + body of `truncated_svd`, lines ~121–134; its 6 test call sites)
- Modify: `crates/aleph-mps/src/mps.rs` (struct field, `choose_par`, `apply_2q_adjacent`, `move_center_right`, `move_center_left`)

- [ ] **Step 4.1: Change `truncated_svd` to take `par`**

In `crates/aleph-mps/src/tensor.rs`, change the signature and the SVD call (keep the rest of the body — sigma extraction, suffix sums, policy logic — untouched):

```rust
pub fn truncated_svd(
    m: faer::MatRef<'_, Complex>,
    policy: &TruncationPolicy,
    par: faer::Par,
) -> Result<TruncatedSvd, MpsError> {
    let rows = m.nrows();
    let cols = m.ncols();

    // Reliable complex SVD via faer (singular values nonnegative, nonincreasing).
    let (fu, fs, fv) = crate::linalg::thin_svd_par(m, par)?;
    let fs = fs.as_ref();
    let k = fs.column_vector().nrows(); // = min(rows, cols)
```

The doc comment on the function keeps its "Why faer" section; extend the first paragraph with one sentence: `The SVD runs under the caller-chosen `par` (P3-13 size-thresholded parallelism).`

Everything downstream (`fu[(r, t)]`, `fv[(c, t)].conj()`) compiles unchanged against the owned `Mat`s.

- [ ] **Step 4.2: Fix the 6 `truncated_svd` test call sites in tensor.rs**

In `crates/aleph-mps/src/tensor.rs` tests, every `truncated_svd(<args>)` gains a trailing `faer::Par::Seq` argument. Example (the same mechanical edit in `truncated_svd_reconstructs_complex_full_rank`, `truncated_svd_rank1_complex_collapses_to_chi1`, `error_bounded_keeps_minimal_chi`, `error_bounded_tiny_eps_keeps_all`, `error_bounded_cap_overrides_eps`, `fixed_bond_matches_legacy`):

```rust
        let (u, s, vt, _disc) =
            truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(64), faer::Par::Seq).unwrap();
```

- [ ] **Step 4.3: Add `par_override` + `choose_par` to `MpsState`**

In `crates/aleph-mps/src/mps.rs`, add to the struct (after `swaps_applied`):

```rust
    /// Test-only override forcing every faer op down one `Par` regardless of
    /// the size threshold — lets the Par-invariance oracle compare `Seq` vs
    /// `rayon` as plain arguments instead of toggling faer's process global
    /// (P3-13).
    pub(crate) par_override: Option<faer::Par>,
```

Initialize in `with_policy` (after `swaps_applied: 0,`):

```rust
            par_override: None,
```

Add the helper method to `impl MpsState` (e.g., right before `apply_1q`):

```rust
    /// Per-operation parallelism for a `rows × cols` faer operand: the
    /// size-threshold policy, unless a test override pins it.
    fn choose_par(&self, rows: usize, cols: usize) -> faer::Par {
        self.par_override
            .unwrap_or_else(|| crate::linalg::par_for(rows, cols))
    }
```

- [ ] **Step 4.4: Rewire `apply_2q_adjacent`**

Replace the theta gemm's `faer::get_global_parallelism()` and the SVD call. After `let ri = self.sites[j].right;` insert:

```rust
        let par = self.choose_par(li * 2, 2 * ri);
```

The gemm's last argument becomes `par`:

```rust
        matmul(
            theta.as_mut(),
            Accum::Replace,
            self.sites[i].group_left_view(),
            self.sites[j].group_right_view(),
            Complex::new(1.0, 0.0),
            par,
        );
```

The SVD call becomes:

```rust
        let (u_s, s_kept, vt_s, discarded) = truncated_svd(theta2.as_ref(), &self.policy, par)?;
```

(One `par` for both ops is correct: gemm output and SVD input are the same `(li·2) × (2·ri)` matrix.)

- [ ] **Step 4.5: Rewire `move_center_right`**

Replace the body's QR + gemm (the function currently reads `let qr = self.sites[i].group_left_view().qr();` etc.):

```rust
    fn move_center_right(&mut self) {
        let i = self.center;
        let left = self.sites[i].left;
        let right = self.sites[i].right;
        let (q, r) = crate::linalg::thin_qr_par(
            self.sites[i].group_left_view().to_owned(),
            self.choose_par(left * 2, right),
        );
        let k = q.ncols();
        let next_right = self.sites[i + 1].right;
        // A'[l',p,r2] = Σ_l R[l',l] · A[l,p,r2]  ==  R · group_right(A).
        let mut absorbed = faer::Mat::<Complex>::zeros(k, 2 * next_right);
        matmul(
            absorbed.as_mut(),
            Accum::Replace,
            r.as_ref(),
            self.sites[i + 1].group_right_view(),
            Complex::new(1.0, 0.0),
            self.choose_par(k, 2 * next_right),
        );
        self.sites[i + 1] = Site::from_group_right_faer(absorbed.as_ref(), k, next_right);
        self.sites[i] = Site::from_group_left_faer(q.as_ref(), left, k);
        self.center += 1;
    }
```

- [ ] **Step 4.6: Rewire `move_center_left`**

```rust
    fn move_center_left(&mut self) {
        let i = self.center;
        let right = self.sites[i].right;
        let left = self.sites[i].left;
        // LQ via thin QR of the adjoint; `.to_owned()` materializes the
        // lazily-conjugated view (same copy the high-level Qr::new made).
        let (q, r) = crate::linalg::thin_qr_par(
            self.sites[i].group_right_view().adjoint().to_owned(),
            self.choose_par(2 * right, left),
        );
        let k = q.ncols();
        let prev_left = self.sites[i - 1].left;
        // A'[l2,p,r'] = Σ_r A[l2,p,r] · Rᴴ[r,r']  ==  group_left(A) · Rᴴ.
        let mut absorbed = faer::Mat::<Complex>::zeros(prev_left * 2, k);
        matmul(
            absorbed.as_mut(),
            Accum::Replace,
            self.sites[i - 1].group_left_view(),
            r.adjoint(),
            Complex::new(1.0, 0.0),
            self.choose_par(prev_left * 2, k),
        );
        self.sites[i - 1] = Site::from_group_left_faer(absorbed.as_ref(), prev_left, k);
        // M = group_right(A) = Rᴴ·Qᴴ; Qᴴ has orthonormal rows — the
        // right-canonical site.
        // `.adjoint()` yields a lazily-conjugated view (`ComplexConj` element
        // type); materialize it to the canonical complex type for the
        // grouped-right reshape. k × (2·right) — a small bond-sized copy.
        let qh = q.adjoint().to_owned();
        self.sites[i] = Site::from_group_right_faer(qh.as_ref(), k, right);
        self.center -= 1;
    }
```

- [ ] **Step 4.7: Run the full crate test suite (both feature configs) — verify everything passes**

```bash
cargo test -p aleph-mps 2>&1 | tail -5
cargo test -p aleph-mps --features parallel 2>&1 | tail -5
```

Expected: all tests pass in both runs (the oracle suite at 1e-9/1e-10 is unaffected: at `Par::Seq` the wrappers are bit-exact replicas of what ran before).

- [ ] **Step 4.8: Check `faer::get_global_parallelism` no longer appears in src/**

```bash
grep -rn "get_global_parallelism" crates/aleph-mps/src/
```

Expected: exactly one hit — inside `linalg::par_for` (the policy's single read point).

- [ ] **Step 4.9: Commit**

```bash
git add crates/aleph-mps/src/tensor.rs crates/aleph-mps/src/mps.rs
git commit -m "[P3-13] Thread per-op Par through the MPS hot path

theta gemm + truncated SVD share one size-chosen Par; both center-move
QRs and absorption gemms choose from their own operand dims. The global
is now read in exactly one place (par_for)."
```

---

### Task 5: Replace the global-toggle invariance test

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (new unit test)
- Modify: `crates/aleph-mps/tests/sv_equivalence.rs` (delete `results_invariant_across_parallelism` and its `ParGuard`)

- [ ] **Step 5.1: Add the state-level Seq-vs-rayon unit test**

Append inside `mod tests` in `crates/aleph-mps/src/mps.rs`:

```rust
    /// Same circuit under sequential and rayon-parallel faer must agree to
    /// 1e-10 (not bit-exact: parallel kernels may round differently).
    ///
    /// Replaces the former tests/sv_equivalence.rs global-toggle test
    /// (P3-09): `par_override` forces EVERY op down the chosen path
    /// regardless of the size threshold — Seq vs rayon as plain arguments,
    /// no process-global mutation, no cross-test isolation hazard (P3-13).
    ///
    /// n=10 with 10 brickwall layers grows the central bond to χ = 16
    /// (measured; only every second layer crosses the middle cut), so the
    /// rayon branch sees real multi-column SVD/QR/gemm work.
    #[cfg(feature = "parallel")]
    #[test]
    fn state_invariant_seq_vs_rayon() {
        use aleph_core::Param;
        let run = |par: faer::Par| -> Vec<Complex> {
            let n = 10usize;
            let mut s = MpsState::new(n, 128);
            s.par_override = Some(par);
            let h_of = |q: u32| {
                crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![q])).unwrap()
            };
            for q in 0..n as u32 {
                s.apply_1q(q as usize, &h_of(q));
            }
            for layer in 0..10u32 {
                let mut q = layer % 2;
                while (q as usize) + 1 < n {
                    let ry = crate::gate::matrix_2x2(&GateInstance::new(
                        Gate::Ry(Param::Concrete(0.3 + (q + layer * n as u32) as f64 * 0.11)),
                        smallvec![q],
                    ))
                    .unwrap();
                    s.apply_1q(q as usize, &ry);
                    let gi = GateInstance::new(Gate::Cnot, smallvec![q, q + 1]);
                    let u = crate::gate::matrix_4x4(&gi).unwrap();
                    s.apply_2q(&gi, &u).unwrap();
                    q += 2;
                }
            }
            // Exercise the lazy router too.
            let gi = GateInstance::new(Gate::Cnot, smallvec![0u32, 9u32]);
            let u = crate::gate::matrix_4x4(&gi).unwrap();
            s.apply_2q(&gi, &u).unwrap();
            s.dense_statevector()
        };
        let a = run(faer::Par::Seq);
        let b = run(faer::Par::rayon(0));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).norm() < 1e-10, "parallelism changed the state");
        }
    }
```

- [ ] **Step 5.2: Delete the old test from `tests/sv_equivalence.rs`**

Remove the entire block from the comment `// Needs faer's rayon backend: run via cargo test -p aleph-mps --features parallel` through the closing brace of `fn results_invariant_across_parallelism()` (the `#[cfg(feature = "parallel")]` test with the `ParGuard` RAII struct and `set_global_parallelism` calls — currently the last item in the file).

- [ ] **Step 5.3: Run both feature configs — verify pass, and the old test is gone**

```bash
cargo test -p aleph-mps --features parallel 2>&1 | tail -5
grep -rn "set_global_parallelism" crates/aleph-mps/
```

Expected: tests pass including `state_invariant_seq_vs_rayon`; grep returns nothing.

- [ ] **Step 5.4: Commit**

```bash
git add crates/aleph-mps/src/mps.rs crates/aleph-mps/tests/sv_equivalence.rs
git commit -m "[P3-13] Par-invariance oracle via par_override, not global toggle

Seq vs rayon compared as plain arguments; closes the P3-09 review
finding about test isolation through faer's process global."
```

---

### Task 6: Workspace validation

- [ ] **Step 6.1: Full workspace gates (what CI runs)**

```bash
cargo test --workspace 2>&1 | tail -3
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo clippy -p aleph-mps --features parallel --all-targets -- -D warnings 2>&1 | tail -3
```

Expected: all green. If clippy flags `needless_range_loop` in `linalg.rs`/test loops, add the established `#[allow(clippy::needless_range_loop)]` with the same one-line justification used throughout the crate.

- [ ] **Step 6.2: Push and open a draft PR (CI starts churning while we measure)**

```bash
git push -u origin p3-13-mps-size-thresholded-par
gh pr create --draft --title "[P3-13] MPS: size-thresholded per-call parallelism" --body "Closes #149. Draft until EPYC AC numbers land. Summary/benches to follow."
```

---

### Task 7: EPYC measurement (AC validation)

**Box:** `ssh root@195.154.249.85` (EPYC 8124P 16c, AVX-512). Repo checkout: `~/p3-10` (GitHub remote — if `git fetch` 404s after the org move, run `git remote set-url origin https://github.com/aleph-sim/aleph.git`). Cargo needs `export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH`.

- [ ] **Step 7.1: Verify the box is idle (CLAUDE.md rule — non-negotiable)**

```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" || echo IDLE; cat /proc/mdstat | grep -c resync || true'
```

Expected: load ≈ 0.00, `IDLE`, no md resync. If not idle, wait — do not measure.

- [ ] **Step 7.2: Sync the branch and build**

```bash
ssh root@195.154.249.85 'export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH && cd ~/p3-10 && git fetch origin && git checkout p3-13-mps-size-thresholded-par && git rev-parse HEAD'
```

Expected: HEAD matches the pushed commit (verify against `git rev-parse HEAD` locally — the P2-02 lesson: never trust numbers from an unverified HEAD).

- [ ] **Step 7.3: Run the AC sweep (nohup, no `set -e`, check logs not DONE markers)**

Run each cell sequentially in one script to avoid self-contention:

```bash
ssh root@195.154.249.85 'export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH && cd ~/p3-10 && nohup bash -c "
  # 1) true sequential reference (no feature)
  cargo bench -p aleph-mps --bench nn_qaoa -- --save-baseline p313-seq
  cargo bench -p aleph-mps --bench long_range -- --save-baseline p313-seq
  # 2) AC: parallel compiled in, 16 threads — small ops must not regress
  RAYON_NUM_THREADS=16 cargo bench -p aleph-mps --features parallel --bench nn_qaoa -- --baseline p313-seq
  RAYON_NUM_THREADS=16 cargo bench -p aleph-mps --features parallel --bench long_range -- --baseline p313-seq
  # 3) AC: wide_bond chi sweep at 16T (chi512 env-gated)
  RAYON_NUM_THREADS=16 WIDE_BOND_CHI512=1 cargo bench -p aleph-mps --features parallel --bench wide_bond
  # 4) wide_bond t=1 reference on the same build
  RAYON_NUM_THREADS=1 WIDE_BOND_CHI512=1 cargo bench -p aleph-mps --features parallel --bench wide_bond
" > ~/p313-bench.log 2>&1 &'
```

Poll with `ssh root@195.154.249.85 'tail -5 ~/p313-bench.log'`; budget ~45–90 min (the χ512 cells dominate). Re-verify idleness after completion.

- [ ] **Step 7.4: Evaluate against the AC**

| Cell | AC target | P3-09 reference (16T / seq) |
|---|---|---|
| `nn_qaoa` all cells, 16T parallel build | within noise (±~3%) of seq baseline | was +155–225% with global rayon |
| `wide_bond` n20 χ128, 16T | within noise of 322.9 ms seq | was 748.6 ms |
| `wide_bond` n24 χ256, 16T | within noise of 3.843 s seq | was 4.533 s |
| `wide_bond` n26 χ512, 16T | ≤ ~29.6 s (retain ≥1.57× vs 46.50 s seq) | was 29.61 s |

If χ=512 loses its win (threshold starves the symmetric 1024×1024 ops): lower `PAR_MIN_ELEMS` toward `512 * 1024 + 1` and re-run the χ512 + χ256 cells. If χ≤256 regresses: raise it. Record whatever the final calibration is; one re-run loop is acceptable, more means the single-threshold model is wrong — stop and reassess against the ticket.

---

### Task 8: Default-ON flip (conditional on Task 7 AC passing)

**Decision rule:** flip `parallel` into the default feature set **iff** every Task-7 AC row passed. If any failed, skip this task, keep the feature opt-in, and record why in the perf doc + PR body.

**Files:**
- Modify: `crates/aleph-mps/Cargo.toml`
- Modify: `crates/aleph-mps/benches/wide_bond.rs`
- Modify: `crates/aleph-mps/src/lib.rs` (parallelism doc section, ~line 54)

- [ ] **Step 8.1: Flip the feature default and rewrite the comment**

In `crates/aleph-mps/Cargo.toml` replace the `[features]` section:

```toml
[features]
default = ["parallel"]
# faer's rayon-parallel kernels. Safe ON by default since P3-13: every faer
# call chooses Par per-operation from the measured size threshold
# (src/linalg.rs PAR_MIN_ELEMS), so small-bond ops always run sequentially
# (chi <= 256 was a 1.5x-19x pessimization under the old process-global
# control plane) and only wide-bond ops (chi >= ~512, 1.57x win @16T on
# EPYC) enter the rayon pool. Thread count: RAYON_NUM_THREADS.
# Disable with default-features = false for a rayon-free build.
parallel = ["faer/rayon"]
```

- [ ] **Step 8.2: Add the runtime env guard to wide_bond (the bench now compiles in `cargo bench --workspace`)**

With `parallel` in the defaults, `required-features = ["parallel"]` is always satisfied, so push-to-main Bench on the shared EPYC runner would inherit minutes of saturating sweep cells. Gate at runtime instead. In `crates/aleph-mps/benches/wide_bond.rs`, at the top of `fn bench(cr: &mut Criterion)` insert:

```rust
    // Runtime gate: `parallel` is a default feature since P3-13, so this
    // bench now compiles under `cargo bench --workspace` — but its sweep
    // cells would tax every push-to-main Bench run on the shared EPYC
    // runner. Opt in explicitly:
    // WIDE_BOND=1 [WIDE_BOND_CHI512=1] RAYON_NUM_THREADS=N \
    //   cargo bench -p aleph-mps --bench wide_bond
    if std::env::var_os("WIDE_BOND").is_none() {
        eprintln!("wide_bond: skipped (set WIDE_BOND=1 to run the sweep)");
        return;
    }
```

Update the module doc comment (lines ~10–15) to describe the env-var gate instead of the `required-features` skip, and update the matching NOTE comment in `crates/aleph-mps/Cargo.toml` above the `[[bench]] name = "wide_bond"` entry:

```toml
# NOTE: `parallel` is a default feature since P3-13, so required-features no
# longer skips this bench in `cargo bench --workspace`. A runtime env guard
# (WIDE_BOND=1) inside the bench keeps the sweep cells out of push-to-main
# Bench runs on the shared EPYC runner; WIDE_BOND_CHI512=1 additionally
# enables the ~47 s/iter chi=512 cell.
```

Note: Task 7's bench invocations set `WIDE_BOND_CHI512` but not `WIDE_BOND` — Task 7 runs BEFORE this guard exists, so they work as written. Any re-measurement after this step must add `WIDE_BOND=1`.

- [ ] **Step 8.3: Update the lib.rs parallelism doc**

In `crates/aleph-mps/src/lib.rs` (~line 54), replace the `parallel` feature paragraph with:

```rust
//! The `parallel` cargo feature (default ON since P3-13) enables faer's
//! rayon-parallel kernels. Parallelism is chosen per operation from a size
//! threshold (`linalg::par_for`): small-bond ops always run sequentially —
//! compiling the feature in cannot pessimize χ ≤ 256 workloads — and only
//! wide-bond operands (≥ 2^20 elements, χ ≈ 512+) enter the rayon pool.
//! Control thread count via `RAYON_NUM_THREADS`; opt out entirely with
//! `default-features = false`.
```

- [ ] **Step 8.4: Re-run workspace gates (defaults changed for aleph-py/aleph-cli too)**

```bash
cargo test --workspace 2>&1 | tail -3
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo bench -p aleph-mps --bench wide_bond 2>&1 | tail -3
```

Expected: all green; the last command prints the `wide_bond: skipped` line and exits fast (guard works without the env var).

- [ ] **Step 8.5: Commit**

```bash
git add crates/aleph-mps/Cargo.toml crates/aleph-mps/benches/wide_bond.rs crates/aleph-mps/src/lib.rs
git commit -m "[P3-13] Default-ON parallel feature; runtime-gate wide_bond

Size-thresholded Par makes compiled-in rayon safe for small bonds
(EPYC-validated), so users get the chi>=512 win without a feature flag.
wide_bond now env-gated (WIDE_BOND=1) to keep push-to-main Bench cheap."
```

---

### Task 9: Perf report + BACKLOG bookkeeping

**Files:**
- Modify: `docs/perf/mps_parallel.md`
- Modify: `BACKLOG.md` (P3-13 acceptance checkboxes, ~line 1944)

- [ ] **Step 9.1: Append a P3-13 section to `docs/perf/mps_parallel.md`**

Structure (fill the measured numbers from Task 7):

```markdown
## P3-13: size-thresholded per-call parallelism (2026-06-XX)

**Branch:** `p3-13-mps-size-thresholded-par` @ <sha> · **Issue:** #149 · EPYC verified idle.

Every faer call now passes `Par` explicitly: ops below `PAR_MIN_ELEMS = 2^20`
elements (calibrated from the P3-09 crossover above) run `Par::Seq`; larger
ops use the rayon pool. The `parallel` feature default flipped ON.

| cell | seq build | P3-13 parallel @16T | P3-09 global-rayon @16T |
|---|---|---|---|
| nn_qaoa n10 | <x> | <x> (±n%) | +155–225 % |
| nn_qaoa n20 | <x> | <x> | … |
| nn_qaoa n30 | <x> | <x> | … |
| long_range dist1/4/8/11 | <x> | <x> | … |
| wide_bond n20 χ128 | <x> | <x> | 748.6 ms (vs 322.9 seq) |
| wide_bond n24 χ256 | <x> | <x> | 4.533 s (vs 3.843 seq) |
| wide_bond n26 χ512 | <x> | <x> (≥1.57× retained: yes/no) | 29.61 s (vs 46.50 seq) |

**AC verdict:** …

Honest notes: <anything that regressed, threshold re-calibration if any,
cells within/outside noise>.
```

- [ ] **Step 9.2: Flip the P3-13 acceptance checkboxes in BACKLOG.md**

In the `### [P3-13]` section (~line 1965), turn the two `- [ ]` acceptance criteria into `- [x]` (only if genuinely met; otherwise leave honest and note in the PR).

- [ ] **Step 9.3: Commit**

```bash
git add docs/perf/mps_parallel.md BACKLOG.md
git commit -m "[P3-13] Perf report: EPYC AC numbers + BACKLOG checkboxes"
```

---

### Task 10: Code review, PR finalization, merge

- [ ] **Step 10.1: Self-review the full diff with fresh eyes**

```bash
git diff origin/main...HEAD --stat
git diff origin/main...HEAD
```

Checklist: no leftover `get_global_parallelism` outside `par_for`; no `unwrap()` in non-test library code; comments explain why; no stray debug code.

- [ ] **Step 10.2: Run /code-review (high effort) and address findings**

Use the code-review skill at high effort on the branch. Fix CONFIRMED findings; commit each fix separately.

- [ ] **Step 10.3: Finalize the PR**

Un-draft and fill the body:

```bash
gh pr ready
gh pr edit --body "$(cat <<'EOF'
Closes #149.

## Summary
<approach: par_for threshold + explicit-Par thin SVD/QR replicas + per-op
threading; par_override invariance oracle; default-ON flip (or: kept opt-in
because <reason>)>

## Tests
<workspace + --features parallel results; bit-exact replica tests; ε=0 and
Par-invariance oracles>

## Benchmarks (EPYC, verified idle)
<the Task-9 table>

## Notes
<threshold calibration; anything deferred>
EOF
)"
```

- [ ] **Step 10.4: Wait for green CI, merge (squash), close out**

CI must be green (build, test, clippy, fmt; remember the shared-runner serialization — a queued Bench run can delay PR CI ~30 min). Let the PR sit briefly, re-review, then squash-merge. Verify issue #149 auto-closed.

---

## Self-review notes (spec coverage)

- Ticket TD-1 (`par_for(rows, cols)` + calibrated threshold) → Tasks 1, 7 (calibration loop).
- Ticket TD-2 (matmul direct substitution; SVD/QR via lower-level explicit-`Par` API; "measure before committing" — the SVD is the dominant cost and the bit-exact replicas keep `Par::Seq` behavior identical, so the only measured risk is the threshold itself, covered by Task 7) → Tasks 2–4, 7.
- Ticket TD-3 (re-evaluate default-ON) → Task 8 (explicit decision rule).
- Ticket TD-4 (thread-invariance test via plain arguments) → Task 5.
- AC-1 (nn_qaoa χ=64 + wide_bond χ=128/256 within noise @16T; χ=512 retains ≥1.57×) → Task 7 table.
- AC-2 (ε=0 and Par-invariance oracles pass) → existing ε=0 test untouched (Task 4.7 runs it); invariance in Task 5.
- Testing requirement (EPYC sweep vs P3-09 baselines in mps_parallel.md) → Tasks 7, 9.
