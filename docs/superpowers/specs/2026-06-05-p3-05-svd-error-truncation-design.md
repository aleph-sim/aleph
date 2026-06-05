# P3-05 — MPS: SVD truncation with controlled error

**Issue:** #36 (`area:backend-mps`, `type:feature`, `priority:high`, `research`, L)
**Milestone:** Phase 3
**Date:** 2026-06-05
**Status:** Approved (brainstorming)
**Depends on:** P3-04 (#35, merged `42081af`)

## Goal

Add an **error-bounded** SVD-truncation mode to the MPS backend alongside the
existing fixed-χ mode, expose both through a `TruncationPolicy`, and report the
tracked truncation error. Builds directly on P3-04, which already performs
fixed-χ truncation, accumulates discarded Schmidt weight (`trunc_error`), and
exposes it via `MpsState::truncation_error()`.

## Background (what P3-04 already provides)

- `tensor::truncated_svd(m, max_bond) -> (u, s_kept, vt, discarded)` — keeps the
  top `min(significant, max_bond)` singular values via a Hermitian Gram-matrix
  eigendecomposition (nalgebra's complex SVD is unreliable for rank-deficient
  matrices), where `significant` drops values below the noise floor
  `1e-7·σ_max`. Returns the discarded squared weight.
- `MpsState { …, max_bond: usize, trunc_error: f64 }`; `apply_2q` calls
  `truncated_svd(&m, self.max_bond)` and accumulates `discarded` into
  `trunc_error`. `truncation_error()` is public.
- `MpsBackend { …, max_bond }` with `with_max_bond(χ)`; CLI `--max-bond`.

## Scope

### In scope (P3-05)
- `TruncationPolicy` enum (fixed-χ and error-bounded modes).
- Error-bounded χ selection in `truncated_svd`.
- Plumb the policy through `MpsState` and `MpsBackend`.
- Track and report the max bond dimension reached.
- CLI `--max-error <ε>` + reporting of accumulated truncation error and max χ.
- Property + oracle tests for both modes; a fixed-χ vs error-bounded benchmark.

### Out of scope
- A higher-precision SVD that resolves discarded weight below the Gram floor
  (~1e-14). Documented as a limitation.
- Non-adjacent gates (P3-06), auto-backend-selection (P3-07).

## Key decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Error-bounded criterion | **Discarded weight ≤ ε per bond** (standard DMRG 2-norm truncation error), capped by `max_bond`, floored by the P3-04 noise threshold. |
| `ε` interpretation | **Absolute** discarded squared weight `Σ_{discarded} σ²`. The two-site block is normalized in canonical form (‖M‖²=1), so absolute and relative coincide. |
| Config surface | A `TruncationPolicy` enum + `MpsBackend::with_truncation`; `with_max_bond` kept as `FixedBond` sugar. CLI gains `--max-error`. |
| `--max-error` without `--max-bond` | `max_bond` cap defaults to 128. |
| Reporting | CLI prints accumulated truncation error AND max bond reached, in **both** modes. |

## Architecture

### `TruncationPolicy` (`crates/aleph-mps/src/tensor.rs`, re-exported from `lib.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationPolicy {
    /// Keep at most χ singular values (the largest).
    FixedBond(usize),
    /// Keep the fewest singular values whose discarded squared weight is ≤ ε,
    /// never exceeding `max_bond`.
    ErrorBounded { epsilon: f64, max_bond: usize },
}
```

### `truncated_svd` selection logic

After computing the descending significant singular values (noise floor
`1e-7·σ_max` dropped as in P3-04):

- `FixedBond(χ)`: `keep = min(significant, χ.max(1))` (current behavior).
- `ErrorBounded { epsilon, max_bond }`: choose the **smallest** `keep` such that
  `Σ_{j ≥ keep} σ_j² ≤ epsilon`; clamp `keep` to `[1, min(significant, max_bond)]`.
  (Equivalently: drop the smallest singular values while the running discarded
  tail stays ≤ ε; stop when adding the next would exceed ε or the cap is hit.)

`discarded` = `Σ_{j ≥ keep} σ_j²` (computed from the full significant tail plus
the noise-floor remainder); renormalization (`scale = 1/√kept_weight`) is
unchanged. Signature changes from `max_bond: usize` to `policy: &TruncationPolicy`.

### `MpsState`

- Field `max_bond: usize` → `policy: TruncationPolicy`.
- New field `max_bond_seen: usize` (running max of the post-truncation bond χ),
  updated in `apply_2q`; accessor `max_bond_reached() -> usize`.
- `new(n, max_bond)` retained as sugar: `policy = FixedBond(max_bond)`,
  `max_bond_seen = 1`. New `with_policy(n, policy)`.
- `apply_2q` calls `truncated_svd(&m, &self.policy)` and updates
  `max_bond_seen = max_bond_seen.max(chi)`.

### `MpsBackend`

- Field `max_bond: usize` → `policy: TruncationPolicy`. Default `FixedBond(128)`.
- `with_max_bond(χ)` retained (sets `FixedBond(χ)`); new `with_truncation(policy)`.
- `allocate` builds `MpsState::with_policy(n, self.policy)`.

### CLI (`crates/aleph-cli`)

- New `--max-error <f64>` (optional) on `run`. Resolution: `Some(ε)` →
  `ErrorBounded { epsilon: ε, max_bond }`; else `FixedBond(max_bond)`.
- `run_mps` takes the resolved policy; after the run, prints
  `truncation error: <Σ discarded>` and `max bond χ: <max_bond_reached>`
  (both modes).

## Error handling
- `--max-error` ≤ 0 or non-finite → CLI error before running.
- Everything else inherits P3-04's `MpsError`/`BackendError` mapping.

## Testing

1. **Unit (`truncated_svd`, ErrorBounded):** construct a matrix with a known
   singular spectrum (e.g. diag-like with σ = [1, 0.1, 0.01, 0.001]); assert the
   chosen χ and that `discarded ≤ ε` for several ε; assert the `max_bond` cap
   overrides ε; assert ε huge → χ=1.
2. **Property — exactness:** `ErrorBounded { epsilon: 0.0, .. }` (or `FixedBond`
   ≥ rank) reproduces the state-vector exactly (MPS dense == NaiveSv to 1e-10)
   on random nearest-neighbor circuits.
3. **Property — bound honored:** on test circuits in error-bounded mode, every
   single truncation's `discarded ≤ ε` whenever the `max_bond` cap was not the
   binding constraint. (Asserted at the `truncated_svd` level to isolate it.)
4. **Oracle — bounded deviation:** a weakly-entangled circuit run with a moderate
   ε stays within the accumulated discarded-weight budget of the exact SV state
   (L2 deviation bounded by the reported `truncation_error`, up to first order).
5. **CLI integration:** `aleph run … --backend mps --max-error 1e-6` succeeds and
   prints the `truncation error:` and `max bond χ:` lines; `--max-error 0`/negative
   rejected.

## Performance

`benches/nn_qaoa.rs` extended to compare `FixedBond(64)` vs
`ErrorBounded { epsilon: 1e-8, max_bond: 64 }` on the NN-QAOA depth-3 circuit
(wall-time + reached χ). No ratio gate — P3-05 has no perf AC; the bench
documents the time/accuracy trade-off.

## Numerical caveat (documented)

The P3-04 Gram-matrix truncation resolves singular values down to ~`1e-7·σ_max`,
so the smallest reliably-controllable discarded weight is ~`1e-14`. An `ε` below
that behaves like `ε ≈ 1e-14`. Resolving finer would require a higher-precision
SVD (e.g. one-sided Jacobi or LAPACK) — out of scope; noted for a future ticket.

## Decomposition (L → ~6 tasks)

1. `TruncationPolicy` enum + `truncated_svd` branching + unit tests.
2. `MpsState`: `policy` field, `max_bond_seen` + `max_bond_reached`, `with_policy`,
   `new` sugar; `apply_2q` wiring.
3. `MpsBackend`: `policy` field, `with_truncation`, `with_max_bond` sugar.
4. CLI `--max-error` + policy resolution + reporting in `run_mps` + integration tests.
5. Property + oracle tests (exactness, bound-honored, bounded-deviation).
6. Bench (fixed-χ vs error-bounded) + docs (lib.rs, numerical caveat).

## References
- Schollwöck, "The density-matrix renormalization group in the age of matrix
  product states" (2011) — §4.1 truncation, discarded weight.
- P3-04 design: `docs/superpowers/specs/2026-06-05-p3-04-mps-basic-chain-design.md`.
