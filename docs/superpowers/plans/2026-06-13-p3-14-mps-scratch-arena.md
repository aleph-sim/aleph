# P3-14 MPS Hot-Path Scratch Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-gate heap allocation churn in the MPS hot path (`apply_2q_adjacent`, `move_center_right/_left`) by reusing pooled faer workspace buffers grown monotonically, collapsing the five-pass SVD-factor copy chain into one direct write per `Site`.

**Architecture:** A `Scratch` struct of reusable `Mat`s + one faer `MemBuffer` lives on `MpsState`. Each faer op (gemm/SVD/QR) takes a `submatrix_mut` view sized to its actual operand; `MemBuffer` regrows only when a larger `StackReq` appears. New `svd_into`/`qr_into` primitives in `linalg.rs` write factors into caller buffers; a pure `svd_truncation_plan` holds the χ-selection math; new in-place `Site` fillers write factors (with folded conjugation/scaling) directly into existing `data` Vecs.

**Tech Stack:** Rust 2021, faer 0.24 low-level API (`faer::linalg::svd::svd`, `faer::linalg::qr::no_pivoting`, `faer::dyn_stack::{MemBuffer, MemStack}`, `StackReq`), `aleph_core::Complex` (= `faer::c64`).

---

## Background facts (verified, bake into your reasoning)

- `faer::linalg::svd::svd(A: MatRef, s: DiagMut, u: Option<MatMut>, v: Option<MatMut>, par, stack: &mut MemStack, params)` — asserts only on **shapes** (`u.nrows()==A.nrows()`, `u.ncols()==size`), accepts **strided** MatMut (no contiguity assert; only a `perf-warn`-feature log for row-major). `size = min(m,n)`.
- `svd_scratch::<Complex>(m, n, ComputeSvdVectors::Thin, ComputeSvdVectors::Thin, par, Default::default()) -> StackReq`.
- `MemStack::new(&mut mem)` works via DerefMut from `&mut MemBuffer`. `MemStack::new(&mut mem).can_hold(req: StackReq) -> bool`. `MemBuffer::new(req: StackReq)`.
- `Mat::<Complex>::zeros(r, c)`, `m.as_ref()`, `m.as_mut()`, `m.as_ref().submatrix(row, col, nrows, ncols) -> MatRef`, `m.as_mut().submatrix_mut(row, col, nrows, ncols) -> MatMut`. `.adjoint()` is a lazy conjugate-transpose view (no alloc).
- `Diag::<Complex>::zeros(size)`, `s.as_mut() -> DiagMut`, `s.as_ref()[t] -> Complex` (singular value is in `.re`).
- QR low-level (already used in `linalg.rs::thin_qr_par`): `recommended_block_size::<Complex>(m,n)`, `qr_in_place(qr: MatMut, q_coeff: MatMut, par, stack, params)`, `qr_in_place_scratch::<Complex>(m,n,block_size,par,default)`, `apply_block_householder_sequence_on_the_left_in_place_with_conj(basis: MatRef, coeff: MatRef, Conj::No, dst: MatMut, par, stack)`, `apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex>(m, block_size, size)`.
- `aleph_core::Complex == faer::c64`; all views zero-copy.
- `MpsState` is `#[derive(Clone)]`; cloned in `expectation()` (1q-only) and per-shot sampling. `#[cfg(test)] par_override` field exists.
- `choose_par(rows, cols) -> faer::Par` is the size-threshold policy (unchanged). Keep all `choose_par` call sites and their `(rows, cols)` arguments byte-identical to today.
- Crate has a `clippy.toml` `disallowed-methods` fence on `faer::get_global_parallelism` — do not call it outside `par_for`/tests.

## File structure

- `crates/aleph-mps/src/linalg.rs` — add `ensure_mem`, `svd_into`, `qr_into`; re-express `thin_svd_par`/`thin_qr_par` as thin allocating delegates; keep bit-exact tests (repointed where needed).
- `crates/aleph-mps/src/tensor.rs` — extract pure `svd_truncation_plan`; add `Site::fill_left_from` and `Site::fill_right_from_scaled_conj`; keep `truncated_svd` as the test-facing reconstruction reference (now delegating to the plan).
- `crates/aleph-mps/src/mps.rs` — add `Scratch` struct + field on `MpsState`; rewrite `apply_2q_adjacent`, `move_center_right`, `move_center_left` to use the arena.
- `docs/perf/mps_parallel.md` — append a P3-14 allocation-attribution + before/after note (Task 9).

## Peak-memory note (carry into review)

The arena holds ~9 pooled `Mat`s + `MemBuffer`, each grown to actual operand sizes (≈16 MB each at χ=512). The previous code freed buffers between gates, so steady-state peak rises (bounded, ≈100–150 MB at χ=512, small vs the state). Unifying time-disjoint buffers (e.g. `absorbed`↔`theta`) is a documented follow-up, intentionally not done in v1 for clarity. Call this out in the PR body.

---

## Task 1: Spike — confirm strided `MatMut` submatrix works with faer svd + qr

**Files:**
- Test (temporary, will be deleted in this task): `crates/aleph-mps/src/linalg.rs` (add a `#[test]` in the existing `mod tests`)

- [ ] **Step 1: Add a spike test that runs svd + qr through a submatrix of a larger backing Mat**

Add to `mod tests` in `crates/aleph-mps/src/linalg.rs`:

```rust
#[test]
fn spike_strided_submatrix_svd_and_qr() {
    use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
    // Backing buffers larger than the operand; we use a 5×4 sub-block.
    let (m, n) = (5usize, 4usize);
    let size = Ord::min(m, n);
    let a = test_matrix(m, n);

    // Pooled-style outputs: bigger backing, strided submatrix views.
    let mut u_buf = Mat::<Complex>::zeros(8, 8);
    let mut v_buf = Mat::<Complex>::zeros(8, 8);
    let mut s = Diag::<Complex>::zeros(size);
    let par = Par::Seq;
    let req = svd_scratch::<Complex>(m, n, ComputeSvdVectors::Thin, ComputeSvdVectors::Thin, par, Default::default());
    let mut mem = MemBuffer::new(req);
    svd(
        a.as_ref(),
        s.as_mut(),
        Some(u_buf.as_mut().submatrix_mut(0, 0, m, size)),
        Some(v_buf.as_mut().submatrix_mut(0, 0, n, size)),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .unwrap();

    // Reconstruct U·diag(s)·Vᴴ from the strided sub-views; must equal A.
    let u = u_buf.as_ref().submatrix(0, 0, m, size);
    let v = v_buf.as_ref().submatrix(0, 0, n, size);
    for i in 0..m {
        for j in 0..n {
            let mut acc = Complex::new(0.0, 0.0);
            for k in 0..size {
                acc += u[(i, k)] * s.as_ref()[k] * v[(j, k)].conj();
            }
            assert!((acc - a[(i, j)]).norm() < 1e-12, "strided SVD recon ({i},{j})");
        }
    }
}
```

- [ ] **Step 2: Run the spike**

Run: `cargo test -p aleph-mps spike_strided_submatrix_svd_and_qr -- --nocapture`
Expected: PASS. If it PANICS on a stride assert, STOP — switch the whole plan to the fallback (per-buffer exact-size monotonic Mats, no submatrix; each buffer reallocated only on shape change). Record which op rejected strides in the PR.

- [ ] **Step 3: Delete the spike test**

Remove the `spike_strided_submatrix_svd_and_qr` function (it was a one-shot risk check; the real coverage is the bit-exact + oracle tests).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "[P3-14] Spike: confirm faer svd accepts strided submatrix MatMut

Verified faer 0.24 svd reconstructs through submatrix views of a larger
backing Mat (row_stride==1, generic col_stride). Arena can pool oversized
Mats and take submatrix views. Spike test removed after passing.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: `linalg.rs` — `ensure_mem` + `svd_into` pooled SVD primitive

**Files:**
- Modify: `crates/aleph-mps/src/linalg.rs`

- [ ] **Step 1: Add `ensure_mem` and `svd_into` above `thin_svd_par`**

In `crates/aleph-mps/src/linalg.rs`, add (the imports `MemBuffer, MemStack`, `svd, svd_scratch, ComputeSvdVectors`, `Diag` are already present; add `use faer::dyn_stack::StackReq;` and `use faer::diag::DiagMut;` and `use faer::MatMut;` to the existing `use` block):

```rust
/// Grow `mem` so a subsequent `MemStack::new(mem)` can satisfy `req`.
/// Effectively monotonic: only rebuilds when the current buffer cannot hold
/// the request, so it never shrinks below a size it already serves.
pub(crate) fn ensure_mem(mem: &mut MemBuffer, req: StackReq) {
    if !MemStack::new(mem).can_hold(req) {
        *mem = MemBuffer::new(req);
    }
}

/// Thin SVD writing factors into caller-provided buffers, using `mem` as scratch
/// (grown as needed). `u_out` must be `m × size`, `v_out` `n × size`, `s_out`
/// `size`, with `size = min(m, n)` — typically `submatrix_mut` views of larger
/// pooled `Mat`s (the arena, P3-14). No allocation in steady state.
pub(crate) fn svd_into(
    a: MatRef<'_, Complex>,
    par: Par,
    u_out: MatMut<'_, Complex>,
    v_out: MatMut<'_, Complex>,
    s_out: DiagMut<'_, Complex>,
    mem: &mut MemBuffer,
) -> Result<(), MpsError> {
    let (m, n) = a.shape();
    let req = svd_scratch::<Complex>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        Default::default(),
    );
    ensure_mem(mem, req);
    svd(
        a,
        s_out,
        Some(u_out),
        Some(v_out),
        par,
        MemStack::new(mem),
        Default::default(),
    )
    .map_err(|_| MpsError::SvdFailed)
}
```

- [ ] **Step 2: Re-express `thin_svd_par` as an allocating delegate to `svd_into`**

Replace the body of `thin_svd_par` with:

```rust
pub(crate) fn thin_svd_par(a: MatRef<'_, Complex>, par: Par) -> Result<ThinSvd, MpsError> {
    let (m, n) = a.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::<Complex>::zeros(m, size);
    let mut v = Mat::<Complex>::zeros(n, size);
    let mut s = Diag::<Complex>::zeros(size);
    let mut mem = MemBuffer::new(StackReq::new::<Complex>(0));
    svd_into(a, par, u.as_mut(), v.as_mut(), s.as_mut(), &mut mem)?;
    Ok((u, s, v))
}
```

- [ ] **Step 3: Run the existing bit-exact SVD test (now exercising `svd_into` indirectly)**

Run: `cargo test -p aleph-mps thin_svd_par_matches_high_level_bit_exact`
Expected: PASS (delegation must stay bit-exact).

- [ ] **Step 4: Add a direct bit-exact test for `svd_into` into pooled (oversized) buffers**

Add to `mod tests`:

```rust
#[test]
fn svd_into_pooled_matches_high_level_bit_exact() {
    for (m, n) in [(8usize, 5usize), (5, 8), (6, 6), (160, 128)] {
        let size = Ord::min(m, n);
        let a = test_matrix(m, n);
        let hl = a.thin_svd().unwrap();
        // Oversized pooled backing + strided sub-views.
        let mut u = Mat::<Complex>::zeros(m + 3, size + 3);
        let mut v = Mat::<Complex>::zeros(n + 3, size + 3);
        let mut s = Diag::<Complex>::zeros(size);
        let mut mem = MemBuffer::new(StackReq::new::<Complex>(0));
        svd_into(
            a.as_ref(),
            faer::get_global_parallelism(),
            u.as_mut().submatrix_mut(0, 0, m, size),
            v.as_mut().submatrix_mut(0, 0, n, size),
            s.as_mut(),
            &mut mem,
        )
        .unwrap();
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

(The `#[allow(clippy::disallowed_methods)]` on `mod tests` already permits `get_global_parallelism` here.)

- [ ] **Step 5: Run the new test**

Run: `cargo test -p aleph-mps svd_into_pooled_matches_high_level_bit_exact`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "[P3-14] linalg: svd_into pooled-buffer SVD primitive + ensure_mem

svd_into writes thin-SVD factors into caller MatMut/DiagMut views (typ.
submatrices of pooled Mats) using a caller MemBuffer grown via ensure_mem.
thin_svd_par now delegates to it (bit-exact, existing test green). New test
pins svd_into bit-exact vs faer high-level through oversized strided views.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: `linalg.rs` — `qr_into` pooled QR primitive

**Files:**
- Modify: `crates/aleph-mps/src/linalg.rs`

- [ ] **Step 1: Add `qr_into` (mirrors `thin_qr_par` but writes into caller buffers)**

Add below `svd_into`. `qr_in` is the input/factor workspace (caller has already copied the source matrix in); `q_coeff`, `thin_q`, `thin_r` are caller outputs sized exactly; `mem` grown as needed:

```rust
/// Thin QR writing `Q` (m × size) into `thin_q` and `R` (size × n) into
/// `thin_r`, factoring in place over `qr_in` (caller copies the source matrix
/// into it first). `q_coeff` is the householder block-T scratch sized
/// `block_size × size` where `block_size = recommended_block_size(m, n)`. `mem`
/// is grown as needed. Mirrors `thin_qr_par` with zero allocation in steady
/// state (P3-14). `size = min(m, n)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qr_into(
    mut qr_in: MatMut<'_, Complex>,
    par: Par,
    mut q_coeff: MatMut<'_, Complex>,
    mut thin_q: MatMut<'_, Complex>,
    mut thin_r: MatMut<'_, Complex>,
    mem: &mut MemBuffer,
) {
    let (m, n) = qr_in.shape();
    let size = Ord::min(m, n);
    let block_size = recommended_block_size::<Complex>(m, n);
    ensure_mem(mem, qr_in_place_scratch::<Complex>(m, n, block_size, par, Default::default()));
    let _ = qr_in_place(
        qr_in.rb_mut(),
        q_coeff.rb_mut(),
        par,
        MemStack::new(mem),
        Default::default(),
    );
    // R: upper trapezoid of the factored qr_in. Column-major source → j-outer.
    for j in 0..n {
        for i in 0..Ord::min(j + 1, size) {
            thin_r[(i, j)] = qr_in[(i, j)];
        }
    }
    // Reuse qr_in's first `size` columns as the householder basis (faer
    // split_LU convention: strict-upper zeroed, unit diagonal). No extra alloc.
    for j in 0..size {
        for i in 0..j {
            qr_in[(i, j)] = Complex::new(0.0, 0.0);
        }
        qr_in[(j, j)] = Complex::new(1.0, 0.0);
    }
    // thin_q := identity, then apply the householder sequence.
    thin_q.fill(Complex::new(0.0, 0.0));
    for d in 0..size {
        thin_q[(d, d)] = Complex::new(1.0, 0.0);
    }
    ensure_mem(
        mem,
        apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex>(m, block_size, size),
    );
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        qr_in.rb().subcols(0, size),
        q_coeff.rb(),
        Conj::No,
        thin_q.rb_mut(),
        par,
        MemStack::new(mem),
    );
}
```

Note: `rb()`/`rb_mut()` are faer's reborrow helpers (already in faer's prelude via `MatMut`; if not in scope add `use faer::prelude::*;` — but `linalg.rs` uses `qr.as_mut()` style today, so prefer `qr_in.as_mut()`/`qr_in.as_ref()` if `rb` is unavailable). Verify which compiles; both are non-consuming reborrows.

- [ ] **Step 2: Re-express `thin_qr_par` as an allocating delegate to `qr_into`**

Replace `thin_qr_par`'s body with:

```rust
pub(crate) fn thin_qr_par(qr: Mat<Complex>, par: Par) -> (Mat<Complex>, Mat<Complex>) {
    let (m, n) = qr.shape();
    let size = Ord::min(m, n);
    let block_size = recommended_block_size::<Complex>(m, n);
    let mut qr_in = qr; // consumed as the in-place workspace
    let mut q_coeff = Mat::<Complex>::zeros(block_size, size);
    let mut thin_q = Mat::<Complex>::zeros(m, size);
    let mut thin_r = Mat::<Complex>::zeros(size, n);
    let mut mem = MemBuffer::new(StackReq::new::<Complex>(0));
    qr_into(
        qr_in.as_mut(),
        par,
        q_coeff.as_mut(),
        thin_q.as_mut(),
        thin_r.as_mut(),
        &mut mem,
    );
    (thin_q, thin_r)
}
```

- [ ] **Step 3: Run the existing bit-exact QR test (now exercising `qr_into`)**

Run: `cargo test -p aleph-mps thin_qr_par_matches_high_level_bit_exact`
Expected: PASS.

- [ ] **Step 4: Run the full linalg test module + reconstruction test**

Run: `cargo test -p aleph-mps --features parallel helpers_reconstruct_under_seq_and_rayon` then `cargo test -p aleph-mps linalg`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "[P3-14] linalg: qr_into pooled-buffer QR primitive

qr_into factors in place over a caller qr_in workspace and writes Q/R into
caller buffers, reusing qr_in's columns as the householder basis (faer
split_LU convention). thin_qr_par delegates to it; bit-exact tests green.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: `tensor.rs` — extract pure `svd_truncation_plan`

**Files:**
- Modify: `crates/aleph-mps/src/tensor.rs`

- [ ] **Step 1: Add the pure plan function above `truncated_svd`**

```rust
/// Pure χ-selection + renormalization for a truncated SVD given the (descending,
/// nonnegative) singular values `sigmas`. Returns `(chi, discarded, scale)`:
/// - `chi` ∈ [1, len] singular values to keep,
/// - `discarded` = Σ_{j≥chi} σ_j² (the dropped Schmidt weight),
/// - `scale` = renormalization factor for the kept σ so the state stays unit
///   weight (input must come from a normalized state).
///
/// Null directions numerically zero relative to σ_max (1e-7·σ_max) are pruned
/// before applying the policy, so the bond is never inflated with Gram noise.
pub(crate) fn svd_truncation_plan(sigmas: &[f64], policy: &TruncationPolicy) -> (usize, f64, f64) {
    let k = sigmas.len();
    let s_max = sigmas.first().copied().unwrap_or(0.0);
    let eps = 1e-7 * s_max.max(f64::MIN_POSITIVE);
    let significant = sigmas.iter().filter(|&&s| s > eps).count().max(1);

    // Suffix sums of σ²: suffix_sq[t] = Σ_{j≥t} σ_j².
    let mut suffix_sq = vec![0.0_f64; k + 1];
    for t in (0..k).rev() {
        suffix_sq[t] = suffix_sq[t + 1] + sigmas[t] * sigmas[t];
    }
    let chi = match *policy {
        TruncationPolicy::FixedBond(max_bond) => significant.min(max_bond.max(1)),
        TruncationPolicy::ErrorBounded { epsilon, max_bond } => {
            let cap = significant.min(max_bond.max(1));
            let mut chosen = cap;
            #[allow(clippy::needless_range_loop)]
            for keep in 1..=cap {
                if suffix_sq[keep] <= epsilon {
                    chosen = keep;
                    break;
                }
            }
            chosen
        }
    };
    let discarded = suffix_sq[chi];
    let kept_weight = suffix_sq[0] - suffix_sq[chi];
    let scale = if kept_weight > 0.0 {
        (1.0 / kept_weight).sqrt()
    } else {
        1.0
    };
    (chi, discarded, scale)
}
```

- [ ] **Step 2: Rewrite `truncated_svd` to use the plan (keeps it as the test-facing reconstruction reference)**

Replace the body of `truncated_svd` from the `let s_max ...` line through the `Ok((u_kept, s_kept, vt_kept, discarded))` with:

```rust
    let (chi, discarded, scale) = svd_truncation_plan(&sigmas, policy);

    let u_kept = faer::Mat::from_fn(rows, chi, |r, t| fu[(r, t)]);
    let vt_kept = faer::Mat::from_fn(chi, cols, |t, c| fv[(c, t)].conj());
    let s_kept: Vec<f64> = (0..chi).map(|t| sigmas[t] * scale).collect();
    Ok((u_kept, s_kept, vt_kept, discarded))
```

(The lines computing `rows`, `cols`, `fu/fs/fv`, `k`, `sigmas` stay as-is at the top of `truncated_svd`.)

- [ ] **Step 3: Port the 6 χ-selection unit tests onto `svd_truncation_plan`**

Add these to `mod tests` (they assert the same χ/discarded the old `truncated_svd` tests did, but on the pure function — keep the existing `truncated_svd` reconstruction tests too):

```rust
#[test]
fn plan_fixed_bond_caps_chi() {
    let s = vec![1.0, 0.1, 0.01, 0.001];
    let (chi, _, _) = svd_truncation_plan(&s, &TruncationPolicy::FixedBond(2));
    assert_eq!(chi, 2);
}

#[test]
fn plan_error_bounded_keeps_minimal_chi() {
    let s = vec![1.0, 0.1, 0.01, 0.001];
    let (chi, disc, _) =
        svd_truncation_plan(&s, &TruncationPolicy::ErrorBounded { epsilon: 1e-3, max_bond: 64 });
    assert_eq!(chi, 2);
    assert!(disc <= 1e-3 + 1e-15);
}

#[test]
fn plan_tiny_eps_keeps_all() {
    let s = vec![1.0, 0.1, 0.01, 0.001];
    let (chi, disc, _) =
        svd_truncation_plan(&s, &TruncationPolicy::ErrorBounded { epsilon: 0.0, max_bond: 64 });
    assert_eq!(chi, 4);
    assert!(disc < 1e-12);
}

#[test]
fn plan_cap_overrides_eps() {
    let s = vec![1.0, 0.1, 0.01, 0.001];
    let (chi, _, _) =
        svd_truncation_plan(&s, &TruncationPolicy::ErrorBounded { epsilon: 10.0, max_bond: 1 });
    assert_eq!(chi, 1);
}

#[test]
fn plan_prunes_null_directions() {
    // A rank-1 spectrum padded with numerical zeros must collapse to χ=1.
    let s = vec![1.0, 1e-15, 1e-16, 0.0];
    let (chi, _, scale) = svd_truncation_plan(&s, &TruncationPolicy::FixedBond(64));
    assert_eq!(chi, 1);
    assert!((scale - 1.0).abs() < 1e-9, "unit-weight input → scale≈1");
}
```

- [ ] **Step 4: Run all tensor tests**

Run: `cargo test -p aleph-mps tensor`
Expected: PASS (the 2 reconstruction tests + the 6 original + the 5 new plan tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "[P3-14] tensor: extract pure svd_truncation_plan

Pulls the χ-selection / null-pruning / renormalization out of truncated_svd
into a pure (chi, discarded, scale) function so the hot path can drive the
SVD into pooled buffers and write Sites directly without the owned-factor
copies. truncated_svd kept as the test-facing reconstruction reference.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: `tensor.rs` — in-place `Site` fillers

**Files:**
- Modify: `crates/aleph-mps/src/tensor.rs`

- [ ] **Step 1: Add `fill_left_from` and `fill_right_from_scaled_conj` to `impl Site`**

```rust
    /// Overwrite this site in place as a left-canonical tensor of shape
    /// `(left, 2, right)` whose grouped-left matrix `(left·2) × right` is `m`.
    /// Reuses the existing `data` allocation (resized), avoiding a fresh `Site`.
    /// `m` must be at least `(left·2) × right`; only that top-left block is read.
    pub fn fill_left_from(&mut self, m: faer::MatRef<'_, Complex>, left: usize, right: usize) {
        self.left = left;
        self.right = right;
        self.data.clear();
        self.data.resize(left * 2 * right, Complex::new(0.0, 0.0));
        #[allow(clippy::needless_range_loop)]
        for row in 0..left * 2 {
            for r in 0..right {
                self.data[row * right + r] = m[(row, r)];
            }
        }
    }

    /// Overwrite this site in place as a right-canonical tensor of shape
    /// `(left, 2, right)` whose grouped-right matrix `left × (2·right)` is the
    /// scaled conjugate `conj(v[(col, l)]) · sv[l]` — i.e. the singular-value
    /// folding `s·Vᴴ` for the V factor (or the bare conjugate when `sv` is all
    /// ones, e.g. a right-canonical Qᴴ). `v` is read as `cols × left` (its row
    /// = grouped-right column index `col`, its col = bond index `l`). `sv` has
    /// length `left`. Reuses the existing `data` allocation.
    pub fn fill_right_from_scaled_conj(
        &mut self,
        v: faer::MatRef<'_, Complex>,
        sv: &[f64],
        left: usize,
        right: usize,
    ) {
        self.left = left;
        self.right = right;
        self.data.clear();
        self.data.resize(left * 2 * right, Complex::new(0.0, 0.0));
        #[allow(clippy::needless_range_loop)]
        for l in 0..left {
            let s = Complex::new(sv[l], 0.0);
            for col in 0..2 * right {
                // grouped-right entry (l, col) = conj(V[col, l]) · s
                self.data[l * 2 * right + col] = v[(col, l)].conj() * s;
            }
        }
    }
```

- [ ] **Step 2: Add a focused unit test that the two fillers match `from_group_*`**

```rust
#[test]
fn fill_left_matches_from_group_left() {
    let m = faer::Mat::from_fn(4, 3, |i, j| Complex::new(i as f64 + 1.0, j as f64 - 0.5));
    let reference = Site::from_group_left_faer(m.as_ref(), 2, 3);
    let mut s = Site::ket0(); // wrong shape on purpose; filler must resize
    s.fill_left_from(m.as_ref(), 2, 3);
    assert_eq!(s, reference);
}

#[test]
fn fill_right_scaled_conj_matches_manual() {
    // V is (cols=2·right) × (left); here left=2, right=3 → V is 6×2.
    let left = 2usize;
    let right = 3usize;
    let v = faer::Mat::from_fn(2 * right, left, |i, j| {
        Complex::new(i as f64 * 0.1 + 1.0, j as f64 * 0.2 - 0.3)
    });
    let sv = [2.0_f64, 0.5];
    let mut s = Site::ket0();
    s.fill_right_from_scaled_conj(v.as_ref(), &sv, left, right);
    assert_eq!((s.left, s.right), (left, right));
    for l in 0..left {
        for col in 0..2 * right {
            let expected = v[(col, l)].conj() * Complex::new(sv[l], 0.0);
            assert_eq!(s.data[l * 2 * right + col], expected);
        }
    }
}
```

- [ ] **Step 3: Run the filler tests**

Run: `cargo test -p aleph-mps fill_`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "[P3-14] tensor: in-place Site fillers (fill_left_from, fill_right_from_scaled_conj)

Resize-in-place writers that reuse the Site data Vec instead of allocating a
fresh Site, folding V-conjugation and singular-value scaling into one indexed
write for the right-canonical factor. Unit-tested vs from_group_* / manual.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: `mps.rs` — `Scratch` struct + field on `MpsState`

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs`

- [ ] **Step 1: Add imports and the `Scratch` struct**

At the top of `mps.rs`, extend the faer imports and add:

```rust
use faer::dyn_stack::{MemBuffer, StackReq};
use faer::diag::Diag;
use faer::Mat;
```

Then add the struct (near the `MpsState` definition):

```rust
/// Reusable per-state workspace for the 2q hot path (P3-14). Each `Mat` grows
/// monotonically to the largest operand seen; ops take `submatrix_mut` views at
/// their exact shape. `mem` is the faer scratch shared sequentially across the
/// SVD/QR ops within one gate.
///
/// NOTE: peak scratch memory rises vs the alloc-per-gate code (≈100–150 MB at
/// χ=512); buffers used at disjoint times (absorbed↔theta) could be unified —
/// documented follow-up, not done in v1 for clarity.
struct Scratch {
    theta: Mat<Complex>,
    theta2: Mat<Complex>,
    svd_u: Mat<Complex>,
    svd_v: Mat<Complex>,
    qr_in: Mat<Complex>,
    q_coeff: Mat<Complex>,
    thin_q: Mat<Complex>,
    thin_r: Mat<Complex>,
    absorbed: Mat<Complex>,
    mem: MemBuffer,
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch {
            theta: Mat::new(),
            theta2: Mat::new(),
            svd_u: Mat::new(),
            svd_v: Mat::new(),
            qr_in: Mat::new(),
            q_coeff: Mat::new(),
            thin_q: Mat::new(),
            thin_r: Mat::new(),
            absorbed: Mat::new(),
            mem: MemBuffer::new(StackReq::new::<Complex>(0)),
        }
    }
}

// Cloning a state must NOT copy transient workspace: scratch holds no semantic
// state (always written before read, regrown on demand), so a clone starts
// empty. This keeps `#[derive(Clone)]` on MpsState cheap and correct for the
// expectation()/sampling clone paths.
impl Clone for Scratch {
    fn clone(&self) -> Self {
        Scratch::default()
    }
}
```

Add a helper on `Scratch` to grow + view a buffer:

```rust
impl Scratch {
    /// Ensure `buf` is at least `rows × cols`, regrowing monotonically, and
    /// return a `rows × cols` mutable submatrix view.
    fn view_mut<'a>(
        buf: &'a mut Mat<Complex>,
        rows: usize,
        cols: usize,
    ) -> faer::MatMut<'a, Complex> {
        if buf.nrows() < rows || buf.ncols() < cols {
            // Grow to cover the request (keep the larger of each dim).
            let nr = buf.nrows().max(rows);
            let nc = buf.ncols().max(cols);
            *buf = Mat::zeros(nr, nc);
        }
        buf.as_mut().submatrix_mut(0, 0, rows, cols)
    }
}
```

- [ ] **Step 2: Add the `scratch` field to `MpsState` and initialize it in `with_policy`**

In the `MpsState` struct add (after `swaps_applied`, before the `#[cfg(test)] par_override`):

```rust
    /// Reusable hot-path workspace (P3-14). Not part of the logical state; see
    /// `Scratch`'s clone-as-empty.
    scratch: Scratch,
```

In `with_policy`'s struct literal add `scratch: Scratch::default(),` (before `#[cfg(test)] par_override: None,`).

- [ ] **Step 3: Build to confirm it compiles (field unused for now)**

Run: `cargo build -p aleph-mps`
Expected: builds (a `dead_code`/unused warning on `Scratch` fields is acceptable at this step; the next tasks consume them). If clippy `-D warnings` would block, add a temporary `#[allow(dead_code)]` on `Scratch` and remove it in Task 8.

- [ ] **Step 4: Run the MPS test suite (behavior unchanged)**

Run: `cargo test -p aleph-mps`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "[P3-14] mps: add Scratch arena field to MpsState

Pooled faer workspace (theta/theta2/svd_u/svd_v/qr buffers/absorbed/mem) with
monotonic view_mut growth and clone-as-empty semantics. Not yet wired into the
hot path. State behavior unchanged.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: `mps.rs` — rewrite `apply_2q_adjacent` to use the arena

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs`

- [ ] **Step 1: Replace the body of `apply_2q_adjacent` (after the `move_center_to(i)` call) with the pooled version**

Keep the function signature, the NN guard, the `i/j` computation, the `move_center_to(i)` call, and the `out` closure exactly as today. Replace the buffer allocations and SVD/Site-write section. The new body after `self.move_center_to(i);`:

```rust
        let li = self.sites[i].left;
        let ri = self.sites[j].right;
        let par = self.choose_par(li * 2, 2 * ri);

        // Destructure for independent field borrows: &sites and &mut scratch
        // simultaneously (distinct fields).
        let MpsState {
            sites, scratch, policy, ..
        } = self;

        // Θ as (li·2) × (2·ri): grouped-left × grouped-right. Pooled, no memset
        // (Accum::Replace overwrites every entry).
        {
            let theta = Scratch::view_mut(&mut scratch.theta, li * 2, 2 * ri);
            matmul(
                theta,
                Accum::Replace,
                sites[i].group_left_view(),
                sites[j].group_right_view(),
                Complex::new(1.0, 0.0),
                par,
            );
        }

        // out closure (MSB/LSB index map) — same as before.
        let out = |phys_i: usize, phys_j: usize| -> usize {
            let bit_msb = if s_msb == i { phys_i } else { phys_j };
            let bit_lsb = if s_lsb == i { phys_i } else { phys_j };
            (bit_msb << 1) | bit_lsb
        };

        // Θ' = U·Θ. theta2 is += accumulated → must be zeroed first.
        {
            let mut theta2 = Scratch::view_mut(&mut scratch.theta2, li * 2, 2 * ri);
            theta2.fill(Complex::new(0.0, 0.0));
            let theta = scratch.theta.as_ref().submatrix(0, 0, li * 2, 2 * ri);
            for ap in 0..2usize {
                for bp in 0..2usize {
                    let row_u = out(ap, bp);
                    for a in 0..2usize {
                        for b in 0..2usize {
                            let u_entry = u[row_u][out(a, b)];
                            if u_entry == Complex::new(0.0, 0.0) {
                                continue;
                            }
                            for r in 0..ri {
                                for l in 0..li {
                                    theta2[(l * 2 + ap, bp * ri + r)] +=
                                        u_entry * theta[(l * 2 + a, b * ri + r)];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Truncated SVD of Θ' into pooled u/v/s, then write Sites directly.
        let rows = li * 2;
        let cols = 2 * ri;
        let size = rows.min(cols);
        let mut s_diag = Diag::<Complex>::zeros(size);
        {
            let theta2 = scratch.theta2.as_ref().submatrix(0, 0, rows, cols);
            let u_out = Scratch::view_mut(&mut scratch.svd_u, rows, size);
            let v_out = Scratch::view_mut(&mut scratch.svd_v, cols, size);
            crate::linalg::svd_into(theta2, par, u_out, v_out, s_diag.as_mut(), &mut scratch.mem)?;
        }
        let sigmas: Vec<f64> = (0..size).map(|t| s_diag.as_ref()[t].re).collect();
        let (chi, discarded, scale) = crate::tensor::svd_truncation_plan(&sigmas, policy);
        self.trunc_error += discarded;
        self.max_bond_seen = self.max_bond_seen.max(chi);

        let s_kept: Vec<f64> = (0..chi).map(|t| sigmas[t] * scale).collect();

        // Site i ← left-canonical from U[:, 0..chi]  (li·2 × chi).
        {
            let u_view = self.scratch.svd_u.as_ref().submatrix(0, 0, rows, chi);
            self.sites[i].fill_left_from(u_view, li, chi);
        }
        // Site j ← right-canonical from s·Vᴴ. V is (cols × size); we read its
        // first `chi` columns. fill_right_from_scaled_conj reads V[col, t] and
        // folds conj + s_kept[t]. (chi × ri grouped-right; left=chi, right=ri.)
        {
            let v_view = self.scratch.svd_v.as_ref().submatrix(0, 0, cols, chi);
            self.sites[j].fill_right_from_scaled_conj(v_view, &s_kept, chi, ri);
        }
        self.center = j;

        Ok(())
```

Note: `policy` from the destructure is `&TruncationPolicy`; `svd_truncation_plan` takes `&TruncationPolicy` — pass `policy` directly. After the destructure block ends, `self` is usable again for `self.trunc_error`/`self.sites`/`self.scratch` (the destructure borrow ends at the end of the `{ }` blocks that used `sites`/`scratch`; ensure the `let MpsState { .. } = self;` is inside a scope or that subsequent `self.` uses don't overlap a live `sites`/`scratch` borrow). If the borrow checker complains, wrap the gemm+theta2+svd section in an inner `{ }` block so the destructured borrows drop before the `self.trunc_error += ...` lines, and recompute `li/ri/etc` as needed, OR re-borrow `self.scratch`/`self.sites` directly in the later blocks (as written above the later blocks use `self.scratch`/`self.sites`, so the destructure must end first — keep the destructure scoped to only the Θ/Θ' gemm region and use `self.` for the SVD + Site writes). Prefer the `self.`-direct form for the SVD/Site-write section to minimize borrow friction.

- [ ] **Step 2: Build**

Run: `cargo build -p aleph-mps`
Expected: builds. Resolve any borrow-checker error by scoping the destructure tightly (see the note) — do not change behavior.

- [ ] **Step 3: Run the focused 2-site oracle + the random oracle proptest**

Run: `cargo test -p aleph-mps`
Expected: PASS — in particular `random_nn_circuit_matches_sv`, `regression_svd_norm_loss_seq`, and the long-range SWAP oracle. These are the real correctness net for the direct Site writes.

- [ ] **Step 4: Run the parallel-feature suite too**

Run: `cargo test -p aleph-mps --features parallel`
Expected: PASS (incl. `state_invariant_seq_vs_rayon`).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "[P3-14] mps: pooled arena in apply_2q_adjacent

Θ/Θ' gemms write into pooled theta/theta2 (no memset; Replace overwrites),
SVD runs into pooled svd_u/svd_v via svd_into, and the two new Sites are
written directly via fill_left_from / fill_right_from_scaled_conj — collapsing
the five-pass factor copy chain to one write each. Math unchanged; full oracle
suite green (seq + rayon).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 8: `mps.rs` — rewrite `move_center_right` / `move_center_left` to use the arena

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs`

- [ ] **Step 1: Rewrite `move_center_right`**

```rust
    fn move_center_right(&mut self) {
        let i = self.center;
        let left = self.sites[i].left;
        let right = self.sites[i].right;
        let m = left * 2;
        let n = right;
        let size = m.min(n);
        let par = self.choose_par(m, n);
        let next_right = self.sites[i + 1].right;

        // Copy the grouped-left view into the pooled QR workspace.
        {
            let mut qr_in = Scratch::view_mut(&mut self.scratch.qr_in, m, n);
            let src = self.sites[i].group_left_view();
            for c in 0..n {
                for r in 0..m {
                    qr_in[(r, c)] = src[(r, c)];
                }
            }
        }
        let block_size = crate::linalg::recommended_block_size_complex(m, n);
        {
            let qr_in = Scratch::view_mut(&mut self.scratch.qr_in, m, n);
            let q_coeff = Scratch::view_mut(&mut self.scratch.q_coeff, block_size, size);
            let thin_q = Scratch::view_mut(&mut self.scratch.thin_q, m, size);
            let thin_r = Scratch::view_mut(&mut self.scratch.thin_r, size, n);
            // qr_in/q_coeff/thin_q/thin_r all distinct fields → can't borrow via
            // four view_mut at once. See Step 1b.
            let _ = (qr_in, q_coeff, thin_q, thin_r);
        }
        // ... see Step 1b for the actual borrow-safe call
        unimplemented!()
    }
```

This naive form does NOT compile (four `&mut self.scratch.*` borrows). Use Step 1b instead.

- [ ] **Step 1b: Borrow-safe `move_center_right` (destructure scratch once)**

Replace `move_center_right` with this final version:

```rust
    fn move_center_right(&mut self) {
        let i = self.center;
        let left = self.sites[i].left;
        let right = self.sites[i].right;
        let m = left * 2;
        let n = right;
        let size = m.min(n);
        let par = self.choose_par(m, n);
        let next_right = self.sites[i + 1].right;
        let block_size = crate::linalg::recommended_block_size_complex(m, n);

        // Grow all four QR buffers up front (separate calls; each ends its
        // borrow), then destructure scratch for simultaneous &mut access.
        Scratch::grow(&mut self.scratch.qr_in, m, n);
        Scratch::grow(&mut self.scratch.q_coeff, block_size, size);
        Scratch::grow(&mut self.scratch.thin_q, m, size);
        Scratch::grow(&mut self.scratch.thin_r, size, n);
        Scratch::grow(&mut self.scratch.absorbed, size, 2 * next_right);

        // Copy grouped-left into qr_in.
        {
            let mut qr_in = self.scratch.qr_in.as_mut().submatrix_mut(0, 0, m, n);
            let src = self.sites[i].group_left_view();
            for c in 0..n {
                for r in 0..m {
                    qr_in[(r, c)] = src[(r, c)];
                }
            }
        }

        // QR into pooled buffers.
        {
            let Scratch {
                qr_in,
                q_coeff,
                thin_q,
                thin_r,
                mem,
                ..
            } = &mut self.scratch;
            crate::linalg::qr_into(
                qr_in.as_mut().submatrix_mut(0, 0, m, n),
                par,
                q_coeff.as_mut().submatrix_mut(0, 0, block_size, size),
                thin_q.as_mut().submatrix_mut(0, 0, m, size),
                thin_r.as_mut().submatrix_mut(0, 0, size, n),
                mem,
            );
        }
        let k = size; // thin Q has `size` columns

        // absorbed = R · group_right(site[i+1])  (k × 2·next_right).
        {
            let r_view = self.scratch.thin_r.as_ref().submatrix(0, 0, k, n);
            let mut absorbed =
                self.scratch.absorbed.as_mut().submatrix_mut(0, 0, k, 2 * next_right);
            matmul(
                absorbed.as_mut(),
                Accum::Replace,
                r_view,
                self.sites[i + 1].group_right_view(),
                Complex::new(1.0, 0.0),
                self.choose_par(k, 2 * next_right),
            );
        }
        // Write site i+1 (right-canonical from absorbed) and site i (left-canonical from Q).
        {
            let absorbed = self.scratch.absorbed.as_ref().submatrix(0, 0, k, 2 * next_right);
            self.sites[i + 1].fill_left_from_grouped_right(absorbed, k, next_right);
        }
        {
            let q_view = self.scratch.thin_q.as_ref().submatrix(0, 0, m, k);
            self.sites[i].fill_left_from(q_view, left, k);
        }
        self.center += 1;
    }
```

Two helpers referenced above must be added:
- `Scratch::grow(buf, rows, cols)` — the growth half of `view_mut` (Step 1c).
- `Site::fill_left_from_grouped_right(m, left, right)` — `absorbed` is a *grouped-right* `k × (2·next_right)` matrix being written into site `i+1` whose new `left=k, right=next_right`. This is exactly the old `from_group_right_faer(absorbed, k, next_right)` reshape, done in place (Step 1d).

- [ ] **Step 1c: Add `Scratch::grow` and refactor `view_mut` to use it**

```rust
impl Scratch {
    fn grow(buf: &mut Mat<Complex>, rows: usize, cols: usize) {
        if buf.nrows() < rows || buf.ncols() < cols {
            let nr = buf.nrows().max(rows);
            let nc = buf.ncols().max(cols);
            *buf = Mat::zeros(nr, nc);
        }
    }

    fn view_mut<'a>(
        buf: &'a mut Mat<Complex>,
        rows: usize,
        cols: usize,
    ) -> faer::MatMut<'a, Complex> {
        Self::grow(buf, rows, cols);
        buf.as_mut().submatrix_mut(0, 0, rows, cols)
    }
}
```

- [ ] **Step 1d: Add `Site::fill_left_from_grouped_right` to `tensor.rs`**

This mirrors the old `from_group_right_faer` (a `left × (2·right)` grouped-right matrix → tensor `(left,2,right)`), in place:

```rust
    /// Overwrite this site in place from a `left × (2·right)` grouped-right
    /// matrix `m` (row `l`, col `p·right + r`) — the in-place equivalent of
    /// `from_group_right_faer`. Reuses the existing `data` allocation.
    pub fn fill_from_grouped_right(&mut self, m: faer::MatRef<'_, Complex>, left: usize, right: usize) {
        self.left = left;
        self.right = right;
        self.data.clear();
        self.data.resize(left * 2 * right, Complex::new(0.0, 0.0));
        #[allow(clippy::needless_range_loop)]
        for l in 0..left {
            for col in 0..2 * right {
                self.data[l * 2 * right + col] = m[(l, col)];
            }
        }
    }
```

Correction: in Step 1b call `self.sites[i + 1].fill_from_grouped_right(absorbed, k, next_right);` (rename from the placeholder `fill_left_from_grouped_right`). Use `fill_from_grouped_right` consistently.

- [ ] **Step 1e: Add `recommended_block_size_complex` wrapper in `linalg.rs`**

`recommended_block_size` is imported in `linalg.rs` but private to module use; expose a tiny wrapper so `mps.rs` can size `q_coeff` identically to `qr_into`:

```rust
/// Block size faer uses for the m×n householder QR (P3-14 sizes q_coeff to match).
pub(crate) fn recommended_block_size_complex(m: usize, n: usize) -> usize {
    recommended_block_size::<Complex>(m, n)
}
```

- [ ] **Step 2: Rewrite `move_center_left` symmetrically**

```rust
    fn move_center_left(&mut self) {
        let i = self.center;
        let right = self.sites[i].right;
        let left = self.sites[i].left;
        // LQ via QR of the adjoint of the grouped-right view: (2·right) × left.
        let m = 2 * right;
        let n = left;
        let size = m.min(n);
        let par = self.choose_par(m, n);
        let prev_left = self.sites[i - 1].left;
        let block_size = crate::linalg::recommended_block_size_complex(m, n);

        Scratch::grow(&mut self.scratch.qr_in, m, n);
        Scratch::grow(&mut self.scratch.q_coeff, block_size, size);
        Scratch::grow(&mut self.scratch.thin_q, m, size);
        Scratch::grow(&mut self.scratch.thin_r, size, n);
        Scratch::grow(&mut self.scratch.absorbed, prev_left * 2, size);

        // qr_in := adjoint(group_right(site[i]))  → element (col, l) = conj(gr[l, col]).
        {
            let mut qr_in = self.scratch.qr_in.as_mut().submatrix_mut(0, 0, m, n);
            let gr = self.sites[i].group_right_view(); // left × (2·right)
            for c in 0..n {
                // c indexes `left`
                for r in 0..m {
                    // r indexes `2·right`
                    qr_in[(r, c)] = gr[(c, r)].conj();
                }
            }
        }
        {
            let Scratch { qr_in, q_coeff, thin_q, thin_r, mem, .. } = &mut self.scratch;
            crate::linalg::qr_into(
                qr_in.as_mut().submatrix_mut(0, 0, m, n),
                par,
                q_coeff.as_mut().submatrix_mut(0, 0, block_size, size),
                thin_q.as_mut().submatrix_mut(0, 0, m, size),
                thin_r.as_mut().submatrix_mut(0, 0, size, n),
                mem,
            );
        }
        let k = size;

        // absorbed = group_left(site[i-1]) · Rᴴ   (prev_left·2 × k).
        {
            let r_view = self.scratch.thin_r.as_ref().submatrix(0, 0, k, n);
            let mut absorbed =
                self.scratch.absorbed.as_mut().submatrix_mut(0, 0, prev_left * 2, k);
            matmul(
                absorbed.as_mut(),
                Accum::Replace,
                self.sites[i - 1].group_left_view(),
                r_view.adjoint(),
                Complex::new(1.0, 0.0),
                self.choose_par(prev_left * 2, k),
            );
        }
        {
            let absorbed = self.scratch.absorbed.as_ref().submatrix(0, 0, prev_left * 2, k);
            self.sites[i - 1].fill_left_from(absorbed, prev_left, k);
        }
        // Site i ← right-canonical = Qᴴ. Q is (2·right) × k; the grouped-right
        // matrix of the new site is Qᴴ = (k × 2·right) with entry (t, col) =
        // conj(Q[col, t]). fill_right_from_scaled_conj with sv = all-ones reads
        // exactly conj(Q[col, t]).
        {
            let q_view = self.scratch.thin_q.as_ref().submatrix(0, 0, m, k);
            let ones = vec![1.0_f64; k];
            self.sites[i].fill_right_from_scaled_conj(q_view, &ones, k, right);
        }
        self.center -= 1;
    }
```

This drops the `q.adjoint().to_owned()` (`qh`) materialization entirely — `fill_right_from_scaled_conj` reads `conj(Q[col, t])` directly.

- [ ] **Step 3: Build**

Run: `cargo build -p aleph-mps`
Expected: builds. Fix borrow errors only by scoping (the destructure-of-scratch pattern is already used above).

- [ ] **Step 4: Run the full suite (seq + parallel)**

Run: `cargo test -p aleph-mps && cargo test -p aleph-mps --features parallel`
Expected: PASS — especially `move_center_right_makes_left_canonical_and_preserves_state`, `move_center_left_preserves_state`, the random oracle, and `state_invariant_seq_vs_rayon`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "[P3-14] mps: pooled arena in move_center_right/_left

QR factors into pooled qr_in/q_coeff/thin_q/thin_r; R-absorption gemm targets
the pooled absorbed buffer (Replace, no memset); new sites written via in-place
fillers. move_center_left's q.adjoint().to_owned() dropped — the right-canonical
Qᴴ site is filled by reading conj(Q) directly. Canonicalization + oracle tests
green (seq + rayon).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 9: Cleanup, workspace lint/test, perf-note scaffold

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (remove any temporary `#[allow(dead_code)]`)
- Modify: `crates/aleph-mps/src/tensor.rs` (remove now-unused `from_group_right_faer`/`from_group_left_faer` ONLY if no remaining callers — check first; tests in `tensor.rs` still use them, so likely keep)
- Modify: `docs/perf/mps_parallel.md`

- [ ] **Step 1: Check for now-dead code**

Run: `cargo build -p aleph-mps 2>&1 | grep -i "never used\|dead_code"`
Expected: no warnings. If `from_group_*_faer` are flagged dead (their only callers were the hot path), keep them ONLY if `tensor.rs` tests still reference them; otherwise delete them and the `faer_from_group_roundtrip` test that exercises them. Resolve so the next step is clean.

- [ ] **Step 2: Remove any temporary `#[allow(dead_code)]` added in Task 6 Step 3**

- [ ] **Step 3: Workspace clippy + fmt (what CI gates)**

Run:
```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```
Expected: clean. If fmt fails, run `cargo fmt --all` and re-check (CI uses `--all`).

- [ ] **Step 4: Full workspace test**

Run: `cargo test --workspace`
Expected: PASS.

- [ ] **Step 5: Add the P3-14 perf-note scaffold to `docs/perf/mps_parallel.md`**

Append a `## P3-14 — scratch arena` section with: the allocation-attribution summary (what was allocated per gate before; what is pooled now), and a placeholder table for the EPYC before/after numbers to be filled in Task 10. Example skeleton:

```markdown
## P3-14 — hot-path scratch arena

Per-2q-gate allocations before P3-14: theta + theta2 (memset+overwrite),
fresh SVD u/v/s + MemBuffer, u_kept/vt_kept copies, two Site Vecs; center moves
added to_owned() copies + per-call QR MemBuffers + qh adjoint copy. P3-14 pools
all of these on `MpsState` (monotonic growth, submatrix views) and writes the
new Sites directly (one pass each), dropping the qh materialization.

### EPYC before/after (criterion, current main baseline)

| cell | main | P3-14 | Δ |
|------|------|-------|---|
| long_range dist1 | _TBD_ | _TBD_ | _TBD_ |
| long_range dist4/8/11 | _TBD_ | _TBD_ | _TBD_ |
| nn_qaoa | _TBD_ | _TBD_ | _TBD_ |
| wide_bond χ64/128/256/512 | _TBD_ | _TBD_ | _TBD_ |

(Numbers filled after the EPYC measurement round; see PR body.)
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "[P3-14] cleanup + perf-note scaffold

Workspace clippy/fmt/test green; mps_parallel.md gains the P3-14
allocation-attribution section + EPYC before/after table (numbers pending
measurement).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 10: EPYC measurement (improve-vs-main bar)

**Files:**
- Modify: `docs/perf/mps_parallel.md` (fill the table)

This task runs on the EPYC bench server (memory: `aleph-bench-server`, `ssh root@195.154.249.85`). Per P3-13: the single self-hosted runner serializes everything; **drain the bench queue first** (`gh run list`), and pushing the branch will itself trigger a ~30 min `cargo bench --workspace` — measure on a verified-idle box.

- [ ] **Step 1: Verify the box is idle**

Run (on EPYC): `uptime` (load ≈ 0) and `pgrep -af "cargo bench|bencher run|Runner.Worker"` (empty). Wait if busy.

- [ ] **Step 2: Baseline current main**

```bash
git checkout main && git pull
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --features parallel --bench long_range -- --save-baseline p314-main
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --features parallel --bench nn_qaoa -- --save-baseline p314-main
WIDE_BOND=1 RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --features parallel --bench wide_bond -- --save-baseline p314-main
```

- [ ] **Step 3: Measure the branch against the baseline**

```bash
git checkout p3-14-mps-scratch-arena
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --features parallel --bench long_range -- --baseline p314-main
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --features parallel --bench nn_qaoa -- --baseline p314-main
WIDE_BOND=1 RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --features parallel --bench wide_bond -- --baseline p314-main
```

- [ ] **Step 4: Record the numbers + verdict**

Fill the `mps_parallel.md` table. Verdict per the AC: **dist1 must improve vs main; no regression on any other cell.** Report flat-but-not-worse honestly (no PR failure for a flat non-dist1 cell within criterion noise). Commit:

```bash
git add docs/perf/mps_parallel.md
git commit -m "[P3-14] perf: EPYC before/after numbers + verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 11: PR

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin p3-14-mps-scratch-arena
```

PR title: `[P3-14] MPS: hot-path scratch arena (kill per-gate allocation churn)`
PR body must include:
- `Closes #150` (the GitHub **issue** number, not the PR — CLAUDE.md).
- Approach summary (pooled `Scratch` arena, `svd_into`/`qr_into`, `svd_truncation_plan`, in-place Site fillers, dropped `qh` copy).
- Test results: full oracle suite green (seq + `--features parallel`); bit-exact SVD/QR; plan + filler unit tests.
- EPYC criterion before/after table; the dist1-improve verdict; honest note on flat cells.
- Peak-memory tradeoff note (≈100–150 MB scratch at χ=512; buffer unification a documented follow-up).
- Spike result (faer accepts strided submatrix MatMut).

- [ ] **Step 2: Wait for CI green (gating: clippy/fmt/test linux+macos), then merge**

Per memory: Linux gating jobs may queue 10–15 min on the shared runner; macOS finishes first. If fmt red → `cargo fmt --all`, re-push.

---

## Self-review notes (author)

- **Spec coverage:** arena struct + monotonic growth (Task 6) ✓; submatrix views + spike (Task 1) ✓; svd_into/MemBuffer reuse (Task 2) ✓; qr_into + drop to_owned-as-alloc (Task 3, 8) ✓; svd_truncation_plan + direct Site writes folding s·Vᴴ conj (Tasks 4, 5, 7) ✓; drop qh via conjugated write (Task 8) ✓; Site Vec reuse in place (Task 5, 7, 8) ✓; clone-as-empty (Task 6) ✓; no unsafe ✓; testing (bit-exact + plan + filler + oracle) ✓; EPYC measurement improve-vs-main (Task 10) ✓.
- **Type/name consistency:** Site filler used as `fill_from_grouped_right` (Step 1d correction supersedes the `fill_left_from_grouped_right` placeholder in Step 1b — use `fill_from_grouped_right` in the final code). `recommended_block_size_complex` defined in Task 8 Step 1e, used in both move_center fns. `svd_into`/`qr_into`/`ensure_mem`/`svd_truncation_plan`/`Scratch::{grow,view_mut}` defined before use.
- **Borrow-checker:** the destructure-of-`self.scratch` pattern (Task 8) and the scoped-blocks pattern (Task 7) are the sanctioned way to hold `&sites` + `&mut scratch.*` together; if friction remains, scope tighter — never change the math.
- **Known faer-name risk:** `Mat::new()` (empty 0×0) and `MatMut::fill` / `rb()`/`rb_mut()` vs `as_mut()`/`as_ref()` — if a name doesn't resolve, the alternative non-consuming reborrow is `as_mut()`/`as_ref()` (already used in `linalg.rs`). Resolve at build time; behavior is identical.
```
