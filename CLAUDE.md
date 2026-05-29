# CLAUDE.md

> Instructions for **Claude Code** working on this repository.
> This file is read automatically at the start of every session. Keep it terse, current, and actionable.

-----

## Project Overview

This is a **high-performance quantum circuit simulator** written in Rust, with pluggable backends (state vector, MPS, stabilizer), CUDA acceleration, and a long-term path to distributed multi-GPU execution.

Read these three files before doing substantial work:

- `ROADMAP.md` — strategic vision and phases.
- `BACKLOG.md` — detailed issue specifications (source of truth for all issues).
- `CREATE ISSUES.md` — how the GitHub backlog is synced from `BACKLOG.md`.

The project is in **Phase 0** (foundation) at the time of writing. See `ROADMAP.md` § 5 for phase definitions.

-----

## Golden Rules

1. **Correctness first, speed second.** A faster simulator that gives the wrong answer is worthless. Property tests and oracle comparisons (vs. Qiskit Aer / Stim) gate every change.
1. **Measure before and after every optimization.** No optimization PR merges without criterion benchmark numbers in the description.
1. **One issue, one PR.** Match the issue’s acceptance criteria exactly. Tag the PR title with the issue ID: `[P0-01] Set up Rust workspace`.
1. **Backend-agnostic IR.** Never let backend-specific concerns leak into `aleph-core` or `aleph-ir`. Backends consume IR; IR knows nothing about backends.
1. **No `unsafe` without justification.** Document why in a comment with a SAFETY block. SIMD intrinsics are the main legitimate use.

-----

## Repository Layout

```
aleph/
├── Cargo.toml              # workspace root
├── ROADMAP.md              # strategy
├── BACKLOG.md              # detailed issues
├── CREATE ISSUES.md        # backlog → GitHub sync
├── CLAUDE.md               # this file
├── README.md               # public-facing intro
├── crates/
│   ├── aleph-core/          # Complex, StateVector, Gate, Circuit types
│   ├── aleph-ir/            # Circuit IR + optimization passes
│   ├── aleph-parser/        # OpenQASM 3.0 parser
│   ├── aleph-backend/       # Backend trait + naive impl
│   ├── aleph-sv/            # state vector backends (CPU, later GPU)
│   ├── aleph-mps/           # MPS backend
│   ├── aleph-stab/          # stabilizer backend
│   ├── aleph-cli/           # `aleph` binary
│   └── aleph-py/            # pyo3 Python bindings
├── benches/                # cross-crate criterion benchmarks
├── docs/
│   ├── decisions/          # ADRs (Architecture Decision Records)
│   ├── perf/               # per-phase performance reports
│   ├── benchmarking.md
│   └── testing.md
└── scripts/
    ├── create-labels.sh
    ├── create-milestones.sh
    └── sync-issues.sh
```

If a directory above doesn’t exist yet, that’s expected — early phases haven’t created it. Don’t create files in non-existent layouts; follow the issue you’re working on.

-----

## Code Conventions

### Rust style

- Edition: **2021**. Minimum Rust: **1.89** (raised from 1.85 in P1-03 to enable AVX-512 intrinsics, stabilised in Rust 1.89.0; previous bumps: 1.75 → 1.85 in P0-04 for criterion 0.5's transitive deps).
- `rustfmt` enforced via `cargo fmt`. Settings in `rustfmt.toml` (default if absent).
- `clippy` warnings treated as errors in CI: `cargo clippy --workspace --all-targets -- -D warnings`.
- No `unwrap()` or `expect()` in library code outside of tests and `lazy_static!`-style one-time init. Use `?` with concrete error types via `thiserror`.
- Public items get rustdoc comments. Include examples for non-trivial APIs.
- Prefer `&[T]` over `&Vec<T>` in function signatures.
- Prefer iterators and `rayon::par_iter` over manual loops where readable.

### Naming

- Crates: `aleph-{area}` (kebab-case).
- Modules: `snake_case`.
- Types: `PascalCase`.
- Functions, variables: `snake_case`.
- Constants: `SCREAMING_SNAKE_CASE`.

### Error handling

- Library crates: define a crate-local `Error` enum with `thiserror`. Never `panic!` on user input.
- CLI / examples: `anyhow::Result` is fine.
- Test code: `.unwrap()` is OK.

### Comments

- Comments explain **why**, not **what**. The code shows what.
- Top of every file: brief module purpose. Top of every public function: one-line doc + examples if non-obvious.
- Algorithm-heavy code: cite the paper / source in a comment. Example: `// Aaronson-Gottesman §3, eq. 15`.

-----

## Build, Test, Bench

Standard commands (run from workspace root):

```bash
# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Lint and format check (what CI runs)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check

# Run benchmarks
cargo bench --workspace

# Run a specific benchmark
cargo bench --bench qft -- --baseline main

# Build with all optimizations (for honest benchmarks)
cargo build --release --workspace
RUSTFLAGS="-C target-cpu=native" cargo bench --workspace
```

GPU-specific (Phase 5+):

```bash
cargo build --workspace --features cuda
cargo test --workspace --features cuda
```

Python bindings (Phase 4+):

```bash
cd crates/aleph-py
maturin develop --release
python -c "import aleph; print(aleph.version())"
```

-----

## Testing Requirements

Every change to backend / kernel code requires:

1. **Unit tests** — specific gates on specific basis states; compare to textbook results.
1. **Property tests** (`proptest`) — invariants from `docs/testing.md`: unitarity, normalization, reversibility, measurement probability sum = 1.
1. **Oracle tests** — full-circuit equivalence vs. Qiskit Aer (state vector) or Stim (stabilizer). Tolerance: 1e-10 for amplitudes (FP64), 1e-5 for probabilities at 100k shots.
1. **Benchmark** — before/after criterion numbers in the PR description.

Test naming:

- Unit tests: `mod tests { #[test] fn test_pauli_x_on_zero() { ... } }` co-located with the code.
- Integration tests: under `tests/` at the crate root.
- Property tests: alongside unit tests, marked with `proptest!` macro.

Do not skip oracle tests because they’re slow. Mark them with `#[ignore]` only if they exceed 30 seconds; CI runs ignored tests on a nightly schedule.

-----

## Performance Guidelines

### When optimizing

1. **Profile first.** Use `cargo flamegraph` or `perf` (Linux). Never optimize blindly.
1. **Benchmark the smallest possible unit.** A gate kernel benchmark is more meaningful than a full circuit benchmark when targeting that kernel.
1. **Compare against the previous best** via criterion’s `--baseline`. Report relative speedup, not absolute time (absolute times vary by machine).
1. **Memory bandwidth is usually the bottleneck**, not FLOPS. Look at cache misses (`perf stat -e cache-misses`).
1. **Don’t reach for SIMD before you’ve tried better algorithms.** Algorithm changes give 10–1000×; SIMD gives 2–4×.

### When in doubt

The performance hierarchy by ROI:

1. Choose the right backend (stabilizer vs. MPS vs. state vector).
1. IR-level optimization (gate fusion, cancellation, commutation).
1. Memory layout (SoA, cache blocking).
1. SIMD.
1. Multi-threading.
1. GPU.

Work top-down. Don’t jump to GPU when CPU has unrealized wins.

-----

## PR Workflow

1. **Pick an issue.** Comment “Working on this” so others don’t duplicate.
1. **Branch name**: `pNN-NN-short-description`. Example: `p0-01-rust-workspace`.
1. **Commits**: small, logical, with clear messages. Body explains why.
1. **PR title**: `[P0-01] Set up Rust workspace`.
1. **PR body**: must include
- Reference to issue (`Closes #<issue-number>`).  **Use the
  issue number, not the PR number** — `Closes #67` to close
  GitHub Issue 67, not to self-reference the PR.  Prose like
  "Closes P0-12" does NOT trigger GitHub's auto-close.  P0-06,
  P0-07, P0-08, and P0-11 all merged with the wrong reference
  and had to be manually closed later.
- Summary of approach.
- Test results (passing tests, oracle comparison).
- Benchmark numbers if applicable.
- Notes on anything left out or follow-ups.
1. **CI must be green.** Build, tests, clippy, fmt all gating.
1. **Self-review first.** Re-read the diff with fresh eyes before requesting review.

For Phase 0, while there’s no team yet: open PRs anyway, let them sit for an hour, re-review, then merge. The discipline pays off when others join.

-----

## Common Mistakes to Avoid

- **Mixing AoS and SoA.** Pick one per backend; document the choice.
- **Letting `Vec<Complex>` and `(Vec<f64>, Vec<f64>)` representations diverge.** Conversion utilities live in `aleph-core::statevector`.
- **Floating-point equality in tests.** Always use a tolerance: `assert!((actual - expected).abs() < 1e-10)`.
- **NaN-silent comparisons.** `NaN > x` and `NaN < x` both return `false` in IEEE-754; `f64::clamp`, `f64::max`, `f64::min` all *swallow* NaN (the latter by IEEE-2008 minNum semantics). Any `>`/`<`/`.clamp`/`.max` gating correctness must be preceded by an explicit `is_finite()` reject — see ADR 0006. Three Phase-0 review rounds regressed on this (P0-09 ×2, P0-10, P0-11); the pattern is sneaky and easy to miss.
- **Eager-pop in `while let` over multiple `pop()`s.** `while let (Some(a), Some(b)) = (small.pop(), large.pop())` evaluates BOTH pops before the pattern match — when one stack is empty you still consume an element from the other, leaking it. Use `while !small.is_empty() && !large.is_empty()` and pop inside the loop. (P0-11 alias-table bug; caught in test by skewed uniform distribution.)
- **`Closes #<PR-number>` instead of `<issue-number>` in PR bodies.** See PR Workflow above; GitHub auto-close needs the issue number.
- **Mutating state vectors in place during tests without restoring.** Tests should be independent; use fresh state per test.
- **Adding dependencies without thinking.** Every new crate is a security and maintenance liability. Justify in the PR.
- **Optimizing code that isn’t on the hot path.** Profile.
- **Skipping the IR.** New backends must go through the IR, not parse OpenQASM directly.
- **Hardcoding qubit counts or gate types in kernels.** Kernels take generic `&GateInstance`; dispatch is centralized.

-----

## When You Get Stuck

If an issue is unclear:

1. Re-read the issue body in `BACKLOG.md` (richer than GitHub Issues sometimes).
1. Check `ROADMAP.md` for higher-level context.
1. Look at the referenced papers / implementations in the issue’s “References” section.
1. Look at how Qiskit Aer, QuEST, or Stim solved the same problem.

If after that it’s still unclear, comment on the GitHub Issue with specific questions before starting work. Don’t guess.

If a planned change conflicts with this CLAUDE.md or the architecture, **stop and ask**. Update this file (with rationale in an ADR under `docs/decisions/`) before proceeding.

-----

## Working with External References

This project relies on prior art. Always cite:

- Algorithm sources: paper + section + equation if applicable.
- Implementation references: link to specific file + line in upstream repo.
- Place citations in comments adjacent to the code they justify.

Key external repos worth reading:

- <https://github.com/Qiskit/qiskit-aer> — production C++ simulator
- <https://github.com/QuEST-Kit/QuEST> — clean C implementation, great for reading
- <https://github.com/quantumlib/Stim> — fast stabilizer simulator
- <https://github.com/PennyLaneAI/pennylane-lightning> — competing Rust/C++ effort
- <https://github.com/NVIDIA/cuQuantum> — GPU reference

Never copy code from these projects (license incompatibility risk). Read, understand, re-implement.

-----

## Updating This File

This file evolves with the project. Update it when:

- A new convention is established (add to “Code Conventions”).
- A common mistake is found in review (add to “Common Mistakes”).
- The build/test command set changes.
- The repository layout changes.
- A phase completes (update “Project Overview”).

Open a PR titled `[meta] Update CLAUDE.md: <reason>`. Don’t bundle CLAUDE.md changes with feature work.

-----

## Quick Reference

|Need to…             |Run                                                    |
|---------------------|-------------------------------------------------------|
|Build                |`cargo build --workspace`                              |
|Test                 |`cargo test --workspace`                               |
|Lint                 |`cargo clippy --workspace --all-targets -- -D warnings`|
|Format               |`cargo fmt`                                            |
|Benchmark            |`cargo bench --workspace` [^bench-features]            |
|Benchmark (IR fuse)  |`cargo bench -p aleph-ir --features bench-fixtures`    |
|Profile (Linux)      |`cargo flamegraph --bench qft`                         |

[^bench-features]: Per-crate benches with `required-features` are silently skipped by `cargo bench --workspace`. Run them explicitly with their feature flag — e.g., `cargo bench -p aleph-ir --features bench-fixtures` for the P1-09 `fuse_1q` bench.
|Add a new issue      |Edit `BACKLOG.md`, then follow `CREATE ISSUES.md`      |
|Update GitHub issues |Re-run `scripts/sync-issues.sh`                        |
|Run the CLI          |`cargo run --bin aleph -- run circuit.qasm`             |
|Python bindings (dev)|`cd crates/aleph-py && maturin develop`                 |

-----

## Glossary

- **State vector**: full 2^n complex array; exact but exponential memory.
- **MPS** (Matrix Product State): tensor network; efficient for low-entanglement states.
- **Stabilizer**: tableau representation; efficient for Clifford circuits.
- **Clifford gate**: H, S, CNOT, and compositions; classically simulable.
- **Gate fusion**: merging adjacent gates into one for fewer state vector passes.
- **SoA / AoS**: Struct of Arrays / Array of Structs memory layout.
- **Oracle test**: equivalence test against a trusted reference (Qiskit, Stim).
- **Tier 1 algorithms**: GHZ, QFT, Grover, random circuit (must work from Phase 0).
- **IR**: backend-agnostic circuit intermediate representation in `aleph-ir`.
- **Backend**: `Backend` trait implementor; one of SV, MPS, Stab, GPU, etc.