# Contributing to Aleph

Thank you for your interest in contributing. This guide covers everything you need to get a change merged.

---

## Getting started

**Prerequisites:** Rust ≥ 1.89 (edition 2021). Install via [rustup](https://rustup.rs/).

```bash
git clone https://github.com/aleph-sim/aleph.git
cd aleph
cargo build --workspace
cargo test --workspace
```

All three commands must succeed before you start making changes.

---

## Before you open a PR

Run these checks locally — CI enforces all of them:

```bash
# Warnings are errors
cargo clippy --workspace --all-targets -- -D warnings

# Format (must be clean)
cargo fmt --check

# All tests green
cargo test --workspace
```

Fix every warning and format issue before pushing.

---

## PR conventions

- **Branch names:** use `pNN-NN-short-description` for work tracked in the backlog (e.g. `p5-01-cuda-toolchain`); use a descriptive name otherwise.
- **PR title:** tag with the backlog issue ID when applicable — e.g. `[P5-01] CUDA toolchain setup`.
- **PR body must include:**
  - `Closes #<issue-number>` — use the GitHub **issue** number, not the PR number.
  - A summary of the approach taken.
  - Test results (which test suites pass, any oracle comparison output).
  - Criterion benchmark numbers (before **and** after) for any performance-related change. No optimization PR merges without them.
  - Notes on anything deliberately left out or deferred.
- **One issue, one PR.** Match the issue's acceptance criteria exactly.

---

## Testing requirements

Every change to a backend or kernel must include all four of the following:

1. **Unit tests** — specific gates on specific basis states; compare to textbook results.
2. **Property tests** (`proptest`) — invariants from `docs/testing.md`: unitarity, normalization, reversibility, measurement probabilities sum to 1.
3. **Oracle tests** — full-circuit equivalence against a trusted reference:
   - State-vector backends: compare against Qiskit Aer; amplitude tolerance `1e-10` (FP64).
   - Stabilizer backend: compare against Stim.
4. **Benchmark** — before/after Criterion numbers in the PR description.

**Floating-point comparisons always use a tolerance — never equality.**

```rust
// correct
assert!((actual - expected).abs() < 1e-10);

// wrong
assert_eq!(actual, expected);
```

Slow oracle tests may be marked `#[ignore]` only if they exceed 30 seconds. CI runs ignored tests on a nightly schedule.

---

## Performance work

Correctness first, speed second. A faster simulator that gives wrong answers is worthless.

**Before optimizing:**

1. Profile first (`cargo flamegraph`, `perf`). Never optimize blindly.
2. Benchmark the smallest possible unit (a gate kernel bench is more meaningful than a full-circuit bench when targeting that kernel).
3. Measure with `cargo bench -- --baseline <name>` and report relative speedup, not absolute time.

**Optimization ROI hierarchy** (work top-down; don't jump to the bottom):

1. Choose the right backend (stabilizer vs. MPS vs. state vector).
2. IR-level optimization (gate fusion, cancellation, commutation).
3. Memory layout (cache-friendly data structures).
4. SIMD intrinsics.
5. Multi-threading.
6. GPU.

---

## Code style

- `rustfmt` is enforced. Run `cargo fmt` before every commit.
- `clippy -D warnings` is enforced. Fix all warnings.
- **No `unwrap()` or `expect()` in library code** (outside tests and one-time init). Use `?` with concrete error types via `thiserror`.
- **Comments explain *why*, not *what*.** The code shows what.
- **Cite your sources.** Algorithm-heavy code must include a reference comment:
  ```rust
  // Aaronson-Gottesman §3, eq. 15
  ```
- **No `unsafe` without a SAFETY comment** explaining why the invariant holds. SIMD intrinsics are the primary legitimate use.
- Public items require rustdoc comments; include examples for non-trivial APIs.
- Prefer `&[T]` over `&Vec<T>` in function signatures.

---

## Architecture rules

- **Backend-agnostic IR.** Never let backend-specific concerns leak into `aleph-core` or `aleph-ir`. Backends consume IR; IR knows nothing about backends.
- New backends must go through the IR — do not parse OpenQASM directly in a backend.
- Every new dependency needs justification in the PR. Prefer the standard library and existing workspace crates.

---

## Where to start

- [`BACKLOG.md`](BACKLOG.md) — detailed specifications for every open issue.
- [`ROADMAP.md`](ROADMAP.md) — strategic direction and phase definitions.
- [Open GitHub Issues](https://github.com/aleph-sim/aleph/issues) — pick something tagged `good first issue` or comment on an issue you want to tackle so we can avoid duplicate work.
