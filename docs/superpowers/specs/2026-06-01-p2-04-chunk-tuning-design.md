# P2-04 — Chunked parallelism tuning (design)

**Issue:** [P2-04] Chunked parallelism tuning (GitHub #30)
**Milestone:** Phase 2 (multi-threaded CPU)
**Depends on:** P2-01 (rayon-parallel SV kernels), P2-02 (aligned buffers), P2-03 (NUMA first-touch)
**Date:** 2026-06-01

## Goal

Empirically tune the parallel chunking knobs **per gate type and per target-qubit
position**, deliver a pre-tuned table selected by CPU model at runtime, and show a
benchmark improvement over the current fixed default — or report honestly that the
path is bandwidth-bound and the win is flat.

## Background

The SV kernels parallelize via two helpers in `crates/aleph-sv/src/kernels/mod.rs`:

- `par_blocks(count, len, block_of, body)` — runs `body(block_of(k))` for `k in
  0..count`, sequential when `len < par_min_amps()` (env `ALEPH_PAR_MIN_AMPS`,
  default `1<<18`), else rayon `into_par_iter().with_min_len(64)`.
- `par_units(...)` — flattens outer-block × inner-SIMD-unit into one parallel
  dimension so top-qubit gates (where `outer_count == 1`) still parallelize;
  delegates to `par_blocks`.

Today both knobs are global/fixed: the sequential cutoff is one process-wide value
and the rayon grain is a hardcoded `with_min_len(64)`. The right values depend on
the gate (work per amplitude varies) and the target qubit (low qubit → small
stride; high qubit → `par_units` regime). P2-04 makes these per-(gate, position).

**Honesty up front.** P2-02 (aligned buffers) and P2-03 (NUMA) both landed
*bandwidth-bound and flat within noise*; the SV gate path has a documented
~1.5–2.7× single-socket scaling ceiling. The deliverable here is the **tunable
table + plumbing + an honest cross-box measurement**, framed exactly as P2-02/03
were. A measurable win is a bonus, not the premise.

## Mechanism — explicit `ChunkPolicy` parameter (Approach A)

The policy is threaded as an explicit value through the kernel layers. No hidden
state. Chosen over a scoped thread-local (Approach C) because rayon runs `body` on
worker threads where a dispatch-thread thread-local is invisible; C only works
because the policy is consumed at iterator-construction time, and that invariant is
a latent footgun (a future policy read inside `body` would silently revert to the
default on every worker — bit-identical results, quiet perf regression). An explicit
parameter makes that failure mode structurally impossible and matches the repo's
"explicit, bit-identical, no hidden state" ethos (the `par_blocks` doc comment, the
P2-01 precise-capture lesson).

**Leaf-local realization.** `ChunkPolicy` is an explicit parameter of
`par_blocks`/`par_units`, but it is computed **in the leaf kernel** one line before
the call — not threaded through the dispatchers. Each leaf kernel already *is* a
specific gate class (e.g. `apply_1q_diagonal_avx512` ⟹ `OneQDiag`) and has the
`target`(s) and `len`, so it derives its own `PosClass` locally. This keeps the
entry fns (`apply_1q`/`apply_2q`/`apply_3q` × {aos, soa}) and the ~10 sub-dispatchers
**untouched**, and is the most misuse-proof realization of A (no long threading
chain to get wrong, no hidden state).

Shape:

```rust
fn par_blocks(policy: ChunkPolicy, count, len, block_of, body) {
    if len < policy.min_amps { /* sequential */ }
    else { (0..count).into_par_iter().with_min_len(policy.grain).for_each(...) }
}

unsafe fn apply_1q_diagonal_avx512(amps, target, controls, d0, d1) {
    let n = amps.len().trailing_zeros();
    let policy = resolve_policy(GateClass::OneQDiag, pos_class(target, n));
    par_blocks(policy, count, len, |k| ..., |i0| ...);   // explicit param
}
```

Edit surface: the `par_blocks`/`par_units` signatures + all **23 call sites** (each
gains one preceding `let policy = resolve_policy(...)` line). No entry-fn or
dispatcher signature changes.

## Components

### CPU detection — `cpu_model() -> RefCpu`

- x86_64: read the CPU brand string via the `__cpuid` intrinsic (leaves
  `0x8000_0002..=0x8000_0004`). One small `unsafe` with a SAFETY note; the brand
  leaves are present on any AVX-capable x86. No new dependency.
- Match known substrings → `enum RefCpu { Epyc8124P, Ryzen3900, Generic }`.
- Non-x86 or unrecognized → `Generic`.
- Cache in a `OnceLock`.
- `ALEPH_CPU_MODEL=epyc|ryzen|generic` forces selection (tests + cross-box sweeps).

### Types & taxonomy

```rust
#[derive(Clone, Copy)] struct ChunkPolicy { min_amps: usize, grain: usize }

enum GateClass {                       // 9 classes, mirroring the real kernels
    OneQGeneric, OneQDiag, OneQAntidiag,
    TwoQDense, TwoQCnot, TwoQCz, TwoQSwap, TwoQDiag,
    ThreeQ,
}

enum PosClass { Low, Mid, High }
```

`PosClass` is derived from the **maximum** target index (dominant stride) relative
to `n`:

- `High` if `max_target >= n - HIGH_BAND` (the `par_units` regime — `outer_count`
  small),
- `Low`  if `max_target < LOW_BAND` (small stride / low-bit kernels),
- `Mid`  otherwise.

`HIGH_BAND` and `LOW_BAND` are fixed design constants (start at 2 each; revisit only
if the sweep shows a sharp boundary elsewhere).

### The table — `chunk_policy(cpu, class, pos) -> ChunkPolicy`

A pure `match` (CPU → class → pos → `ChunkPolicy`). 9 × 3 = 27 cells per CPU.

**No-regression guarantee.** `RefCpu::Generic` returns the *current defaults* for
every cell (`min_amps = 1<<18`, `grain = 64`). On unknown hardware, and for every
cell we choose not to tune, behavior is byte-for-byte unchanged from today.

**YAGNI.** Only cells with real Tier-1 traffic are tuned away from the default:
`OneQDiag` (cphase ladder), `TwoQCnot`, `OneQGeneric` (H), `TwoQDiag`. The remaining
cells stay at the Generic default *explicitly* in each CPU table. Sweeping all 27
cells is not justified when ~5 carry the workload.

### Policy resolution & precedence

In the dispatch fns:

```rust
fn resolve_policy(class, pos) -> ChunkPolicy {
    let mut p = chunk_policy(cpu_model(), class, pos);
    if let Some(v) = env_usize("ALEPH_PAR_MIN_AMPS") { p.min_amps = v; }
    if let Some(v) = env_usize("ALEPH_PAR_GRAIN")    { p.grain    = v; }
    p
}
```

Each env var, **if present, overrides its field** of the table-resolved policy
(per-field, not all-or-nothing) — so the sweep sets both while a debugging session
can pin just one. The two env reads are cached in `OnceLock`s (read once at first
use, as the current `par_min_amps()` already does) — no per-gate env syscall in the
hot path. This single env path is both the **sweep instrument** and a
debugging knob. (The previous global `par_min_amps()` OnceLock is subsumed by this;
its `ALEPH_PAR_MIN_AMPS` semantics are preserved — when set it still forces the
sequential cutoff, now per-field rather than process-global.)

### Sweep harness

A gated bench (`chunk_tune`, behind a `bench-fixtures`-style feature) applies one
gate class repeatedly on a 25-qubit state at a chosen target, reading
`ALEPH_TUNE_GATE` / `ALEPH_TUNE_TARGET`, and constructs the `ChunkPolicy` directly
from `ALEPH_PAR_MIN_AMPS` + `ALEPH_PAR_GRAIN` (bypassing the table). A driver script
walks the grid:

- `min_amps ∈ {2^16, 2^17, 2^18, 2^19, 2^20}`
- `grain ∈ {16, 32, 64, 128, 256, 512}`

on **EPYC 8124P** (AVX-512 path) and **Ryzen 9 3900** (scalar path), each
idle-verified per CLAUDE.md (`uptime` ~0, `pgrep -af "cargo bench|bencher run"`).
Record best-median per cell; assemble the table literal from the winners. The
**primary reference CPU** is designated in the P2-05 report based on which box shows
a cleaner/larger signal.

## Correctness & testing

- **Policy-invariance test (key).** Changing `min_amps`/`grain` only re-partitions
  tasks — never changes which amplitude a `body` writes, and there is no
  cross-thread FP reduction — so results are bit-identical. Run all oracle fixtures
  under `{default, force-sequential, force-all-parallel, grain=1, grain=huge}` and
  assert equality within 1e-12 (expect exact). Extends the existing thread-count
  invariant.
- `chunk_policy` unit tests: `Generic` cell == current default; known-CPU cell ==
  table value; `PosClass` boundary cases (`max_target` at `LOW_BAND`, at
  `n - HIGH_BAND`).
- CPU detection: `ALEPH_CPU_MODEL` forcing returns the expected `RefCpu`; an
  unrecognized brand → `Generic`.
- **Benchmark (the AC).** Tuned table vs fixed default on the reference box; honest
  before/after criterion numbers in the PR, even if flat.

## Acceptance criteria (from BACKLOG)

- [ ] Tuned chunk-size table for one reference CPU (EPYC or Ryzen — designated from
      the sweep).
- [ ] Benchmark improvement over the fixed default (or honest flat report).

## Out of scope

- Runtime auto-tuning (probe runs to pick chunk size). BACKLOG says "start with
  table; add auto-tune later." Deferred.
- Tuning the `HIGH_BAND`/`LOW_BAND` position thresholds as runtime parameters.
- Per-cell tuning of the ~22 low-traffic cells (left at Generic default).
- NUMA-interaction tuning (P2-03 already shipped first-touch; chunk × NUMA
  interaction is a P2-05 reporting concern, not a new knob here).
