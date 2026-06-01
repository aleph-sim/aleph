# P2-01 — Rayon-Based Parallel Gate Application — Design

**Date:** 2026-06-01
**Issue:** P2-01 (BACKLOG.md, Phase 2 — Multi-Thread CPU)
**Depends on:** P1-07 (2q kernels), and the full P1-03..P1-08 AoS/SoA SIMD kernel set.
**Status:** Approved (brainstorming) — ready for implementation plan.

## Goal

Parallelize the state-vector gate kernels across CPU cores with rayon, as the
foundation of Phase 2. This is the *fundament* ticket only — false-sharing
padding (P2-02), NUMA (P2-03), chunk tuning (P2-04), and the scaling report
(P2-05) are explicitly **out of scope** and handled as separate later efforts.

Phase 2 phase-exit target is ≥12× on 16 cores (ROADMAP §7); the formal
validation of that target lives in P2-05. P2-01's own acceptance gate is the
BACKLOG figure: **≥6× speedup on QFT-25 at 8 cores vs. 1 core.**

## Background

Every SV kernel (AoS in `kernels/aos.rs`, SoA in `kernels/soa.rs`, for 1q / 2q /
3q, scalar + AVX-512 + low-bit variants) is built around an **outer block walk**.
Two driver shapes exist:

- **Uncontrolled:** `while block < len { outer_iter(block); block += outer_step }`
- **Controlled:** `for k in 0..outer_count { outer_iter(expand_with_fixed(k, fixed) << shift) }`

In both, the loop body is a closure `outer_iter(block)` that touches a
**pairwise-disjoint** set of amplitudes (this disjointness is already a mandated
invariant of the P1 SIMD kernels — `block | offsets[k] | j` bits are
disjoint). That disjointness is exactly what makes the outer block dimension
embarrassingly parallel: distinct blocks never write the same amplitude.

rayon is **not yet** a workspace dependency.

## Architecture

### One shared parallel driver, not rayon-per-kernel

Rather than duplicate rayon plumbing across ~30 kernels, introduce a single
driver in `kernels/mod.rs`:

```rust
// Sync wrapper over a raw pointer. Blocks are pairwise-disjoint, so
// concurrent writes into the shared buffer are sound (the SIMD-kernel
// disjointness invariant). SAFETY justification lives at the def site.
#[derive(Clone, Copy)]
struct BlockPtr(*mut f64);
unsafe impl Send for BlockPtr {}
unsafe impl Sync for BlockPtr {}

/// Calls `body(block_of(k))` for k in 0..count. Sequential below the
/// threshold, rayon-parallel above it. The result is bit-identical
/// regardless of thread count: each block writes disjoint memory with no
/// floating-point reduction, so there is no reordering of FP ops →
/// oracle equality (1e-12) holds across any RAYON_NUM_THREADS.
fn par_blocks(
    count: usize,
    len: usize,
    block_of: impl Fn(usize) -> usize + Sync,
    body: impl Fn(usize) + Sync,
)
```

Each kernel collapses its two driver loops into one `par_blocks(...)` call,
with the existing `outer_iter` becoming `body`. The uncontrolled path is
normalized to `count = len / outer_step`, `block_of(k) = k * outer_step`. The
controlled path keeps `block_of(k) = expand_with_fixed(k, fixed) << shift`.

The kernel *bodies* (the SIMD inner walk) are untouched — only the driver that
sequences blocks changes. This reuses all P1-03..P1-08 SIMD work verbatim.

## Threshold & Determinism

- **Parallelism gate:** parallelize only when `len >= PAR_MIN_AMPS`
  (initial `1 << 18` = 256K amplitudes ≈ 4 MiB). Below it, rayon overhead
  outweighs the win and small circuits / tests stay sequential and fast. The
  exact constant is tuned empirically on EPYC; the default is safe.
- **No Cargo feature flag.** rayon becomes a normal workspace dependency —
  mandated by the Phase 2 BACKLOG and justified per CLAUDE.md's
  "justify new dependencies" rule. Parallelism is always-on above the
  threshold, mirroring how the `is_x86_feature_detected` SIMD path is
  always-on.
- **Thread count:** the rayon global pool, controlled by `RAYON_NUM_THREADS`.
  That env var is the lever P2-05's scaling report will sweep; we do not build
  any thread-count API in P2-01.
- **Determinism:** each block writes disjoint memory; there is no
  cross-thread floating-point reduction. The result is therefore bit-identical
  for any thread count, and oracle equivalence (1e-12 vs. sequential) is
  preserved.

## SoA — mirror of the same driver

SoA kernels in `kernels/soa.rs` share the identical block walk but over two
streams (real, imag). The same `par_blocks` driver applies; the `body` closure
captures two `BlockPtr`s (re, im). The block-disjointness invariant is
identical to AoS, so the SoA conversion is mechanical.

## Testing & Acceptance Criteria

### Tests

- **Correctness (primary gate):** run the existing `all_fixtures_match_naive`
  workhorse (SoA ≡ AoS ≡ Naive within 1e-12) under
  `RAYON_NUM_THREADS` ∈ {1, 2, 4, 8} to prove thread-count invariance of the
  result.
- **Property:** existing unitarity / normalization proptests stand unchanged —
  the parallel path does not alter the math.
- **No regression:** full `cargo test --workspace`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo fmt --check`.

### Benchmark (BACKLOG AC)

- QFT-25, measured on EPYC: **≥6× speedup at 8 cores vs. 1 core**
  (`RAYON_NUM_THREADS=1` baseline). The ROADMAP ≥12×/16-core figure is
  measured and noted but formally validated later in P2-05.

### Acceptance Criteria for the PR

1. Parallel 1q and 2q kernels in **both** AoS and SoA (3q/Toffoli included if
   the conversion is cheap — same driver).
2. QFT-25 ≥6× speedup at 8 cores vs. 1 core on EPYC.
3. Thread-count-invariant oracle equivalence (the `{1,2,4,8}` sweep passes at
   1e-12).
4. Zero correctness regressions; CI green (build, test, clippy, fmt).

## Out of Scope (explicit)

- Cache-line / false-sharing padding (P2-02).
- NUMA-aware allocation (P2-03).
- Per-gate / per-qubit chunk-size tuning beyond the single `PAR_MIN_AMPS`
  threshold (P2-04).
- The 1→64 core scaling-efficiency report and the ≥12×/16-core phase-exit
  validation (P2-05).
- Parallelizing measurement / sampling (`measure*.rs`, `sampling.rs`) — P2-01
  is gate application only.

## References

- rayon: <https://docs.rs/rayon/>
- Memory [[p1-07-merged]] — canonical disjoint-block invariant
  `block | offsets[k] | j` pairwise-disjoint.
- ADR 0008 — bandwidth-bound nature of large-n kernels (informs why the win is
  expected to come from cores, and why small n stays sequential).
