# ADR 0006 — Explicit NaN guards before any FP comparison

**Status**: Accepted (2026-05-25, retroactive)
**Issues**: [P0-09](../../BACKLOG.md), [P0-10](../../BACKLOG.md),
[P0-11](../../BACKLOG.md)

## Context

Three Phase-0 tickets regressed in code review on the same
class of bug: a NaN floating-point value silently passing a
`<` / `>` / `.clamp(...)` / `.max(...)` check that the author
intended as a reject.

* **P0-09** review round 2: `f64::clamp(0.0, 1.0)` on a NaN
  measurement probability returned NaN, which then produced NaN
  amplitudes during renormalisation — the test missed it
  because `NaN > 1e-10` is `false`, so the tolerance check
  passed.
* **P0-09** review round 3: same pattern in
  `unitarity_deviation` — `if dev > worst` swallowed NaN
  matrices, letting a non-unitary user matrix through.
* **P0-10** review: `assert_state_close` compared amplitudes via
  `delta > STATE_TOLERANCE` without first rejecting NaN — the
  exact regression class P0-09 had just fixed in two other
  sites.
* **P0-11** review round 2: distribution-oracle band formula
  `5.0 * (p * (1.0 - p).max(0.0) / n_f).sqrt()` could produce
  NaN for `p < 0` (sqrt of negative), and `delta > NaN` would
  silently pass.

In every case the symptom was the same: **a NaN propagated
past a tolerance check that looked correct in isolation but
relied on IEEE-754 comparisons that return `false` on either
side of NaN**.

## Decision

**Every FP comparison that gates correctness must be preceded
by an explicit `is_finite()` (or `is_nan()`) guard, or use a
NaN-rejecting wrapper.**

The pattern, codified:

```rust
// REJECTED — NaN silently passes:
if value > threshold { return Err(...); }

// REQUIRED — NaN rejected loudly:
if !value.is_finite() {
    return Err(BackendError::InvalidState { reason: "non-finite value" });
}
if value > threshold { return Err(...); }
```

For aggregations across many values (e.g. norm² sum), do the
`is_finite()` check **inside the accumulation loop**, not after
— a single NaN amp poisons the sum.

For `Backend` and `Oracle` code, the conventional error variant
is `InvalidState { reason: &'static str }` or a parallel
panic-with-structured-message in test harness contexts.

`f64::clamp`, `f64::max`, `f64::min`, and `Ord::cmp` on `f64`
all have NaN-swallowing semantics (or, in `Ord::cmp`'s case,
don't compile because `f64: !Ord`).  **Treat all of them as
NaN-unsafe** and either guard before or use `is_nan() ||
is_infinite()` short-circuit fall-throughs.

## Consequences

* New invariant-style proptests in `aleph-test::state` and
  `aleph-sv::measure` exercise `validate_state` on hand-crafted
  NaN-bearing states (`f64::NAN`-filled amp slots) to ensure
  every primitive rejects loudly.
* Code review checklist: any new `>`/`<`/`.clamp`/`.max` on a
  `f64` must have an `is_finite()` guard within 5 lines
  upstream OR a comment explaining why the value is provably
  finite at that point.
* Future backends (MPS, stabiliser, GPU) inherit this rule.
  The CUDA reduction in P5 is the most likely future violation
  — flag this ADR in `aleph-sv-gpu`'s design spec.

## Alternatives considered

* **Wrap `f64` in a newtype that rejects NaN at construction.**
  Considered but rejected: every kernel computation produces
  intermediate values that *could* be NaN under pathological
  inputs (e.g. `0.0 / 0.0` in a non-finite-input edge case).
  Forcing a wrapping check on every intermediate would balloon
  perf cost.  Better: rely on a small number of well-known
  *boundary* sites — `validate_state`, kernel `apply_*` entry,
  oracle compare — to be the guard layer.

* **`#[deny(clippy::float_cmp)]` workspace-wide.**  Clippy's
  `float_cmp` lints `==`/`!=`, not the more dangerous
  `>`/`<`/`.clamp`.  Not the right tool.

* **Property test pinning `f64::clamp` semantics.**  Useful but
  not sufficient — the lint is on **specific call sites**, not
  on the function itself.

## References

* `crates/aleph-sv/src/measure.rs::validate_state` — the
  workspace-canonical NaN-rejection pattern.
* `crates/aleph-sv/src/backend.rs::unitarity_deviation` —
  fixed-then-fixed-again in P0-09 review rounds 2 + 3.
* `crates/aleph-oracle/src/harness.rs::assert_state_close`
  (line 66-78) — explicit NaN guard mirroring this rule.
* `crates/aleph-oracle/src/harness.rs::assert_distribution_close`
  (line ~195) — same guard on the distribution-oracle path,
  added in P0-11 review round 2.
* `crates/aleph-sv/src/sampling.rs::AliasTable::build` — the
  `while let (Some, Some) = (small.pop(), large.pop())` lockstep
  bug (P0-11) was a different class but same root cause:
  trusting IEEE/operator semantics that silently fail on edge.
