# QFT parity — enable the optimization pipeline in the run path

> Status: design (brainstorm). Follows P1-14 (Phase-1 perf report) and the
> `[meta]` Phase-1-complete fixup. Not a numbered backlog item yet — a
> Phase-1 follow-up to close the lone QFT gap.

## 1. Problem & root cause

The P1-14 report found QFT is the only Tier-1 family where aleph is slower than
Qiskit Aer single-thread: `qft_n25 = 1.73× Aer`, `qft_n20 = 1.22×`. Every other
family aleph wins.

**Root cause (proven by profiling, not assumed):** the `qiskit_baseline` bench —
and the `run()` driver generally — applied the **raw parsed circuit** gate by
gate. The IR optimization passes shipped in P1-09…P1-13 (`CancelInversePairs`,
`DeadCodeElim`, `Fuse1qRuns`, `Fuse2q`) were **never invoked in the run path**.
The driver doc even states "callers run passes themselves" — and no caller
(bench, oneshot, CLI) did. So P1-14 measured *raw* aleph against *fusion-
optimized* Aer (Aer runs its gate-fusion by default).

QFT is hit hardest because it is almost entirely controlled-phase gates, which
the basis decomposes to `cx + p`. Fusion collapses these adjacency runs into one
2q block each, turning 970 gates into 190 — a 5× reduction in state-vector
passes. Grover/random/GHZ benefit less (they have fewer fusible adjacencies) and
already won without fusion.

### Measured fusion effect (this is the whole win — no new kernels needed)

| circuit | raw gates → fused | EPYC wall-clock (raw→opt) | speedup |
|---------|-------------------|---------------------------|---------|
| qft_n20 | 970 → 190         | 596 ms → 243 ms           | **2.45×** |
| qft_n25 | 1525 → 325        | (to measure; ~2.4× expected) | — |

Local (relative) speedups confirm fusion helps **every** family and harms none:
GHZ 1.1×, QFT 3.9×, Grover 1.4×, random 2.0×.

**Consequence:** turning on the existing pipeline flips qft_n20 from 1.22×
*behind* to ~0.53× *ahead* of Aer. No new Gate variant, kernel, or pass is
required. The infrastructure (incl. a 2q-diagonal no-shuffle kernel and
`Unitary1qDiag`) already exists; the gap was purely that optimization wasn't
wired into execution.

## 2. Goal & non-goals

**Goal:** wire the existing optimization pipeline into the run path, prove it is
semantics-preserving end-to-end, re-measure honestly (optimized aleph vs
fusion-optimized Aer), and update `docs/perf/phase1.md`. Expected outcome: QFT
reaches ≤ 1× Aer at all n; all other families improve too.

**Non-goals:**
- No new passes, Gate variants, or kernels. (If re-measurement somehow shows QFT
  still > 1×, a *follow-up* spec would consider commutation-aware or
  `Unitary2qDiag` fusion — but profiling says that's unnecessary.)
- No change to pass semantics or to `run()`'s raw behavior.
- Not touching multi-thread / Phase 2 work.

## 3. Architecture

A new driver pair in `aleph-backend` (which already depends on `aleph-ir`, so no
new dependency):

```rust
/// Run `circuit` through the default optimization pipeline, then simulate.
pub fn run_optimized<B: Backend>(backend: &mut B, circuit: &Circuit)
    -> Result<B::State, BackendError>;

/// Same, preserving measurement outcomes (see `run_with_outcomes`).
pub fn run_optimized_with_outcomes<B: Backend>(backend: &mut B, circuit: &Circuit)
    -> Result<(B::State, Vec<MeasurementRecord>), BackendError>;
```

- `run()` / `run_with_outcomes()` stay **raw** — they remain the debug/oracle
  reference and don't break existing callers.
- The new functions clone the circuit, run `PassPipeline::default_pipeline()`,
  then delegate to the raw driver. (Clone because the pass pipeline mutates in
  place and the caller's `&Circuit` is shared.)
- A `PassError` from the pipeline maps to a new `BackendError` variant
  (e.g. `Optimization(PassError)`).

**Why in `aleph-backend`, not `aleph-ir`:** the helper composes pipeline+driver,
both of which live at/below the backend layer. Keeping `run()` pass-agnostic
preserves the backend-agnostic contract (§ "Backend-agnostic" in the driver
doc); `run_optimized` is an opt-in convenience that doesn't change that.

## 4. Correctness plan (critical — passes now affect real output)

Per-pass oracle tests already exist at 1e-12 (`fuse_2q_oracle.rs`,
`fuse_1q_oracle.rs`, `cancel_oracle.rs`, `dce_oracle.rs`, `commute_oracle.rs`),
proving each pass semantics-preserving in isolation. We add an **end-to-end**
guard:

- New test file `crates/aleph-backend/tests/run_optimized_oracle.rs`
  (`aleph-sv` is already a dev-dependency there).
- For every Tier-1 fixture (the committed `circuits/*.qasm` at small n) plus
  property-generated circuits (reuse `aleph-test` strategies): assert
  `run_optimized(c)` state amplitudes ≡ `run(c)` within 1e-12.
- Measurement path: `run_optimized_with_outcomes` with a fixed seed must produce
  the same outcomes as `run_with_outcomes` on circuits with measurements
  (fusion must not reorder across measurement/barrier — verify the pipeline
  already respects barriers, which P1-09 established).
- This pins the **whole pipeline + sim** path, not just individual passes — the
  exact gap that let raw-vs-optimized slip through in P1-14.

## 5. Benchmark + re-measurement + report

- `benches/benches/qiskit_baseline.rs`: switch the aleph side from `run` to
  `run_optimized`. This is the honest comparison — Aer fuses by default, so we
  now compare optimized-vs-optimized.
- `benches/src/bin/oneshot.rs` (RSS): also use `run_optimized` for consistency
  (fusion changes peak buffers negligibly but keeps the path identical to the
  bench).
- **Re-measure the full matrix on EPYC** with `run_optimized` (the heavy phase;
  grover_n25 is again the multi-hour long pole). Update `docs/perf/phase1.md`
  **in place**: replace the headline tables with optimized numbers, and add a
  short note that the prior figures were raw single-gate kernel throughput
  (preserve that framing so the improvement is legible, not hidden). Re-state the
  ROADMAP §7 verdict with the optimized numbers (QFT now ≤ 1×).
- Keep the CI-safe gating (`ALEPH_BENCH_FULL_MATRIX`) and sample budgets as-is.

## 6. Verification

- `cargo test --workspace` green, incl. the new end-to-end oracle.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check`
  clean (separate gates).
- EPYC: AVX-512 still emitted (`objdump`), full matrix re-measured, numbers in
  the report reproducible via the documented commands.
- The bench's CI subset still completes well under the Bench workflow's 30-min
  timeout.

## 7. Risks & mitigations

- **A pass reorders across a measurement/barrier → wrong outcomes.** Mitigation:
  the measurement-outcome oracle in §4; barriers are already respected by the
  passes (P1-09). Low risk, explicitly tested.
- **Clone cost per `run_optimized`.** Negligible vs simulation; the bench's
  `iter_with_setup` already rebuilds the backend each iteration. If it ever
  matters, an in-place `optimize(&mut Circuit)` variant can be added later.
- **EPYC re-measurement is long** (grover_n25). Accepted; same protocol as
  P1-14, driven in one sitting on the idle runner. Not pushing to `benches/**`
  mid-run.

## 8. Out of scope / follow-ups

- Commutation-aware fusion or a `Unitary2qDiag` IR variant — only if
  re-measurement unexpectedly leaves QFT > 1× (profiling says it won't).
- Making `run_optimized` the default for the CLI (`aleph run`) — reasonable, but
  a separate change with its own UX consideration (a `--no-opt` flag).
