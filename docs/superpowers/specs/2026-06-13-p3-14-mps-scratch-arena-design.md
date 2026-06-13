# P3-14 — MPS hot-path scratch arena (kill per-gate allocation churn)

**Issue:** #150 (Phase 4.6, adopted from Phase 3; depends on P3-09)
**Type:** optimization · priority:low · estimate M
**Status:** design approved 2026-06-13

## Problem

Every adjacent 2q gate (`MpsState::apply_2q_adjacent`) and every center move
(`move_center_right`/`move_center_left`) allocates a fresh batch of heap buffers,
most of which are overwritten before they are read or copied multiple times:

Per 2q gate:
- `theta` = `Mat::zeros(li·2, 2·ri)` — memset, then immediately overwritten by an
  `Accum::Replace` gemm (the memset is wasted).
- `theta2` = `Mat::zeros(li·2, 2·ri)` — genuinely needs zero-init (`+=` accumulated).
- Inside `truncated_svd` → `thin_svd_par`: fresh `u`, `v`, `s`, **and a fresh
  `MemBuffer` SVD scratch every call**.
- `u_kept` / `vt_kept` — two full `Mat::from_fn` copies of the SVD factors.
- Two `from_group_*_faer` calls — two fresh `Site` `Vec`s (sites i and j).

That is the *five-pass* factor chain `svd → u_kept → Site_i` and
`svd → vt_kept → sv-scale → Site_j`.

Per center move:
- `group_left_view().to_owned()` (and `group_right_view().adjoint().to_owned()` in
  `_left`) materializes a QR workspace copy.
- `thin_qr_par` internals: `q_coeff`, `thin_r`, `thin_q = Mat::identity`, and **two
  fresh `MemBuffer`s**.
- `absorbed = Mat::zeros(...)` — memset, then `Accum::Replace` gemm.
- `move_center_left` additionally materializes `qh = q.adjoint().to_owned()`.

P3-09's honest note: the `long_range` dist1 microcircuit cell regressed +11.9 % —
at χ ≤ 32 the allocator round-trips and faer workspace setup dominate the math.

## Goals / Non-goals

**Goals**
- Reuse workspace buffers across gates instead of reallocating per op.
- Collapse the factor copy chain to a single write into each new `Site`.
- Drop the avoidable `to_owned()` materializations (the `qh` conjugate copy).
- Recover the dist1 regression; no regression on any other cell.

**Non-goals**
- No change to the truncation math, the `Par` size-threshold policy (P3-13), the
  lazy SWAP router (P3-09), or any public API behavior.
- No `unsafe`.
- No new dependency (faer 0.24 low-level API is already used in `linalg.rs`).

## Decisions (from brainstorming)

1. **Full arena**, built regardless of a pre-profile (we still measure after; the
   AC is perf-based).
2. **dist1 success bar = improve vs current `main`** (post-P3-13). The literal
   "pre-P3-09 baseline" in the ticket is no longer cleanly reproducible — P3-13's
   parallel-build flavor already stepped the bencher.dev long_range baseline
   +1.4–5.5 % on these µs-scale cells. We measure current main on EPYC and require
   dist1 to improve, with no regression elsewhere. Report flat-but-not-worse
   honestly rather than failing the PR.
3. **Reuse `Site` Vecs in place** — write new factors into the existing
   `sites[i]/[j].data` buffers (resize, reuse capacity) rather than allocating
   fresh `Site`s.

## Design

### Arena struct & lifecycle

A `Scratch` struct held as a field on `MpsState`:

```rust
struct Scratch {
    theta:   Mat<Complex>,   // grouped Θ            (Accum::Replace target)
    theta2:  Mat<Complex>,   // U·Θ                  (zeroed each gate, += target)
    svd_u:   Mat<Complex>,   // SVD left factor
    svd_v:   Mat<Complex>,   // SVD right factor
    svd_s:   Diag<Complex>,  // singular values
    qr_in:   Mat<Complex>,   // QR/LQ in-place input + factor workspace
    q_coeff: Mat<Complex>,   // householder block-T coeffs
    thin_q:  Mat<Complex>,   // QR Q output
    mem:     MemBuffer,      // faer scratch for svd / qr / householder
}
```

- Each buffer **grows monotonically to the actual operand sizes seen**, not the
  policy cap — a circuit that only reaches χ=64 never allocates χ=512 buffers.
- Ops take **submatrix views** (`buf.as_mut().submatrix(0, 0, m, n)`) at the exact
  shape each call needs.
- `mem` is rebuilt only when a call's `StackReq` exceeds the stored capacity. The
  faer ops within a gate run sequentially, so one shared `mem` is safe.

**Spike first (Task 1):** confirm faer 0.24's `svd` / `qr_in_place` accept a
strided `MatMut` (a submatrix of a larger backing Mat). If they reject non-unit
column strides, fall back to per-buffer *exact-size* Mats grown monotonically
(still kills steady-state churn; only the shape-change reallocs remain).

### Hot-path rewrites

`apply_2q_adjacent` destructures `self` into independent field borrows
(`sites`, `scratch`, `policy`, `center`) so `&sites[i]` and `&mut scratch` can be
held at once. Then:
- **Θ gemm** → `scratch.theta` submatrix, `Accum::Replace`, no memset.
- **Θ' = U·Θ** → zero the `scratch.theta2` submatrix (it is `+=`-accumulated),
  existing 6-deep loop writes into it.
- **SVD + Site writes** → see next section.

`move_center_right`/`_left`:
- `to_owned()` / `adjoint().to_owned()` of the source view become a **copy into
  `scratch.qr_in`** (the copy is unavoidable — QR factors in place over a live
  Site we cannot destroy — but it is no longer a fresh allocation).
- `absorbed` → a pooled submatrix with `Accum::Replace`, no memset.
- `thin_qr_par` is changed to write Q/R into caller-provided pooled buffers
  (`qr_in`, `q_coeff`, `thin_q`, `mem`) instead of allocating per call.
- `move_center_left`'s `qh = q.adjoint().to_owned()` is dropped: the
  right-canonical Site is written by reading `conj(q[(col, t)])` directly (see the
  conjugated-write idiom below).

Router/permutation logic, `choose_par`, and `Par` selection are unchanged — this
is pure storage reuse.

### `truncated_svd` refactor + direct Site writes

Split into:

1. **`svd_truncation_plan(sigmas, policy) -> (chi, discarded, scale)`** — pure, no
   allocation; holds the χ-selection + renormalization logic. The 6 existing
   `truncated_svd` unit tests (which only assert χ / discarded) port onto it.
2. **Caller runs the SVD into pooled `svd_u/v/s`, calls the plan, then writes both
   Sites directly** (one pass each):
   - **Site i** (left-canonical, `li × 2 × chi`): resize `sites[i].data` to
     `li·2·chi`, set `left=li, right=chi`, then
     `data[row·chi + r] = svd_u[(row, r)]` for `row∈0..li·2, r∈0..chi`.
   - **Site j** (right-canonical, `chi × 2 × ri`): resize `sites[j].data` to
     `chi·2·ri`, set `left=chi, right=ri`, then fold conjugation + scaling into the
     single write:
     `data[t·2·ri + col] = conj(svd_v[(col, t)]) · (s[t] · scale)`
     for `t∈0..chi, col∈0..2·ri`.

This replaces `u_kept` + `vt_kept` + the `sv` scale + both `from_group_*_faer`
copies with one indexed write each. The same conjugated-write idiom serves
`move_center_left`.

**Site Vec reuse:** add `Site` methods that resize `data` in place and fill from a
`MatRef` (`fill_left_from`, `fill_right_from_scaled_conj`). `from_group_*_faer`
remain for tests / non-hot callers; the hot path uses the in-place fillers.

### Clone semantics

`MpsState` is cloned in `expectation()` (1q-only, never touches scratch) and by the
backend for per-shot sampling. Scratch is transient workspace (grown on demand,
always written before read), so cloning its contents is meaningless. `Scratch` gets
a hand-written `Clone` returning `Scratch::default()` (empty, regrows on first
use), with a loud doc comment. `MpsState`'s `#[derive(Clone)]` and the
`#[cfg(test)] par_override` field are unaffected.

### Safety

No `unsafe`. Monotonic growth reallocates via safe `Mat::zeros`. The
"uninitialized backing for Replace" is just a zeroed buffer we do not re-zero — it
is safe-initialized memory; we rely on `Accum::Replace` overwriting the live
submatrix.

## Testing

- **Primary net (unchanged, must stay green):** `random_nn_circuit_matches_sv`
  (1e-9), `regression_svd_norm_loss_seq`, the long-range SWAP-network oracle
  (1e-10) — these caught the original nalgebra SVD bug.
- **Ported** `svd_truncation_plan` unit tests (χ / discarded / scale).
- **New** focused 2-site fill unit test: build a known 2-site block, apply a gate,
  assert the two filled Sites reconstruct the expected dense vector — a fill /
  transpose / conj bug fails fast without a full random circuit.
- `linalg.rs` bit-exact `thin_svd_par` / `thin_qr_par` tests must pass after they
  are rewired to write into caller-provided buffers.
- Par-invariance oracle (`state_invariant_seq_vs_rayon`) unchanged.

## Measurement (EPYC)

Per the "improve vs current main" bar:
1. Baseline current `main` (post-P3-13) on EPYC: `long_range` (dist 1/4/8/11),
   `nn_qaoa`, `wide_bond` (χ sweep via `WIDE_BOND=1`).
2. Re-run on branch with criterion `--baseline`.
3. **Success:** dist1 improves (recovers the cited +11.9 % P3-09 regression); no
   regression on any other cell; oracle suite 1e-10.
4. Numbers in the PR body + a short note in `docs/perf/mps_parallel.md`.
5. Drain the self-hosted bench queue before each measurement round (P3-13 lesson:
   push-to-PR triggers ~30 min `cargo bench --workspace` on the shared runner).

## Acceptance Criteria (from BACKLOG, retargeted)

- [ ] `long_range` dist1 improves vs current main on EPYC; no regression on any
      other `long_range` / `nn_qaoa` / `wide_bond` cell.
- [ ] Full oracle suite unchanged (1e-10).
- [ ] criterion before/after on EPYC in the PR; allocation-attribution note.

## Risks

- **Strided-`MatMut` rejection by faer low-level ops** — mitigated by the Task-1
  spike + exact-size-buffer fallback.
- **Direct conjugated Site write is the highest-risk code** — guarded by the
  focused fill unit test + the random oracle proptests.
- **Flat perf** — plausible given P3-13 / P2-0x bandwidth-bound history; we report
  honestly. No-regression + oracle-green is the floor; dist1 improvement is the
  target.
