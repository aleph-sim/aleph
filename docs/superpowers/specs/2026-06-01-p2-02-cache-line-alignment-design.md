# P2-02 — Cache-line-aligned state buffers (false-sharing audit) — Design

**Issue:** P2-02 — Cache-line padding to prevent false sharing (BACKLOG.md)
**Phase:** 2 (multi-threaded CPU)
**Depends on:** P2-01 (rayon parallel gate application, merged `c494334`)
**Date:** 2026-06-01
**Estimate:** S

---

## 1. Problem & honest framing

P2-02's BACKLOG brief is "prevent cache-line ping-pong between cores." The
pre-implementation audit (below) shows this codebase has **no material
false-sharing surface today**, for two independent reasons:

1. **State-vector parallel writes are large, contiguous, disjoint chunks.**
   The only rayon usage in the workspace is `par_blocks`/`par_units`
   (`crates/aleph-sv/src/kernels/mod.rs`). It splits the block index with
   `into_par_iter().with_min_len(64)`, so every task owns ≥ 64 consecutive
   SIMD units. One unit = `LANES = 4` complex = exactly **64 bytes = one
   cache line**. Tasks write pairwise-disjoint ranges, so the *only* lines two
   threads could share are the single boundary line between adjacent tasks
   (~1 per task split, out of millions). Negligible by construction.

2. **No per-thread accumulators exist.** The spec's primary target —
   "sample counts, statistics" — is not yet parallel. `measure.rs`,
   `measure_soa.rs`, and `sampling.rs` are sequential; there is no
   `fold`/`reduce`/atomic/`Mutex` in any hot parallel path. There is nothing
   to pad.

3. **Large allocations are already incidentally aligned.** The parallel path
   only engages at `ALEPH_PAR_MIN_AMPS = 1 << 18` (4 MiB). glibc malloc serves
   allocations above ~128 KiB via `mmap`, which returns page-aligned (4 KiB)
   memory — already 64-byte aligned. So on the Linux bench boxes the residual
   boundary line from (1) is, in practice, already eliminated.

Additionally, P2-01's own report (`docs/perf/phase2-p2-01.md`) established that
the scaling ceiling on both bench boxes is **memory bandwidth + the EPYC
frequency throttle**, not cache-line contention (8→16 cores buys only
3.37×→3.69×, a bandwidth plateau).

**Conclusion that shapes scope:** a false-sharing "fix" cannot meaningfully move
the scaling numbers, because there is no contention to remove. P2-02's durable
value is therefore reframed as a **guarantee**, not a speedup:

- **(a) Portability/robustness** — *guarantee* 64-byte alignment instead of
  relying on the allocator's mmap heuristic (which does not hold for small
  states, non-glibc allocators, or future custom allocators).
- **(b) Aligned AVX-512 load/store enablement** — the alignment precondition,
  should we ever want `_mm512_load_pd` over `loadu` (see §4, out of default
  scope).
- **(c) The allocation hook P2-03 (NUMA-aware allocation) needs** — a
  hand-rolled owned buffer is the natural place to add `numa_alloc` /
  first-touch later. The `aligned-vec` crate would not compose with that.

The report (§6) states all of this plainly. AC #3 ("scaling efficiency improves
vs P2-01") is interpreted honestly: we measure it, and we expect — and report —
flat-within-noise, with the perf-tool audit (AC #2) as the substantive evidence.

---

## 2. Component: `AlignedBuf<T>` — `crates/aleph-core/src/aligned.rs`

A fixed-size, 64-byte-aligned, owned heap buffer. The minimal allocation
primitive; deliberately *not* a growable `Vec` (state vectors never resize).

```rust
pub const CACHE_LINE: usize = 64;

pub struct AlignedBuf<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}
```

**API surface:**

- `AlignedBuf::<T>::zeroed(len) -> Self` — `std::alloc::alloc_zeroed` with
  `Layout::from_size_align(len * size_of::<T>(), CACHE_LINE)`. Both `f64` and
  `Complex` (two `f64`) are zero-valid, so the all-zero bit pattern is a valid
  `|0…0⟩` buffer; the backend then sets element 0. Returns a buffer whose data
  pointer is `% 64 == 0`.
- `AlignedBuf::<T>::from_slice(src: &[T]) -> Self where T: Copy` — `zeroed`-shaped
  allocation followed by `copy_from_slice`; for test/example ergonomics
  replacing `vec![…]` literals.
- `Deref<Target = [T]>` + `DerefMut` — provides indexing, iteration,
  `as_ptr`/`as_mut_ptr`, `len`, `iter`, `to_vec`, and **deref coercion** so
  existing `&mut state.amps` arguments still bind to kernel params typed
  `&mut [T]` with no call-site change.
- `Drop` — deallocates with the *same* `Layout` used to allocate.
- `unsafe impl<T: Send> Send` / `unsafe impl<T: Sync> Sync` — mirrors `Vec<T>`
  (an owned heap region; sound to send/share under the same bounds).

**Edge cases & SAFETY:**

- `len == 0` → store `NonNull::dangling()` (correctly aligned for `T`), perform
  **no** allocation and **no** deallocation in `Drop`. (n = 0 qubits ⇒ dim = 1,
  so a zero-length buffer is only a defensive path, but `alloc(0)` is UB and
  must be avoided.)
- One contained `// SAFETY:` block documents: the layout invariant (size =
  `len * size_of::<T>()`, align = 64, same layout used for alloc and dealloc);
  the zero-init validity bound (only ever instantiated for `f64`/`Complex`,
  both zeroable); and the `alloc_zeroed` null-return handling
  (`handle_alloc_error`).
- `Layout` construction is guarded: `len * size_of::<T>()` overflow is
  impossible for our domain (`dim ≤ 2^28`, `size_of::<Complex>() = 16` ⇒
  ≤ 4 GiB < `isize::MAX`), but the constructor still returns/handles the
  `Layout` error path rather than `unwrap` in library code (CLAUDE.md: no
  `unwrap` in library code).

This is the only new `unsafe` in the change; it is a contained, well-documented
allocator — an accepted use per CLAUDE.md ("No `unsafe` without justification").

---

## 3. Wiring `AlignedBuf` into the state structs

- `CpuState.amps: AlignedBuf<Complex>` (`crates/aleph-sv/src/state.rs:15`)
- `SoaState.{re, im}: AlignedBuf<f64>` (`crates/aleph-sv/src/soa_state.rs:9–10`)

**Allocation sites:**

- `NaiveSvBackend::allocate` (`backend.rs:52`):
  `let mut amps = AlignedBuf::zeroed(dim); amps[0] = Complex::new(1.0, 0.0);`
- `SoaSvBackend::allocate` (`soa_backend.rs:57,62`):
  `re = AlignedBuf::zeroed(dim); re[0] = 1.0; im = AlignedBuf::zeroed(dim);`

**Field-access blast radius:** ~94 references touch these fields, but the vast
majority are slice operations (`state.amps[i]`, `&mut state.amps`, iteration,
`.amplitudes()` returning `&[Complex]`) that work unchanged through
`Deref`/`DerefMut`. Only **construction** sites change:

- The two `allocate` methods (above).
- `vec![…]` literals in `state.rs`, `soa_state.rs`, and any unit/proptest
  fixtures → `AlignedBuf::from_slice(&[…])`.

No change to kernels, `ComplexPtr`/`BlockPtr`, conversion utilities,
`HasAmplitudes`, sampling, or measurement — they all consume slices/pointers.

---

## 4. AVX-512 aligned loads — OUT of default scope

Kernels keep `_mm512_loadu_pd` / `_mm512_storeu_pd`. Switching to aligned
`_mm512_load_pd` / `store` requires *proving* every unit offset is 64-aligned
(misalignment is UB, not a slowdown), for a benefit that is ≈ 0 on modern x86
when an access does not cross a cache line. We will measure a candidate switch
on a **standalone micro-bench** and adopt it **only if** it shows a real win
*and* the per-unit alignment proof holds. Default outcome: no change. This keeps
the ticket's correctness surface limited to the allocator.

---

## 5. Testing

**Unit (aleph-core, runs on all targets):**

- `zeroed(n)` data pointer satisfies `(ptr as usize) % 64 == 0` for
  `n ∈ {0, 1, 4, 1000}`.
- `zeroed(n)` yields all-zero contents and `len() == n`.
- `from_slice(&[…])` round-trips contents and length.
- Run the `aligned` module tests under **miri**
  (`cargo +nightly miri test -p aleph-core aligned`) to validate the
  alloc/dealloc/Drop/`NonNull::dangling` unsafe paths (no leaks, no UB).

**Backend alignment assertions:**

- After `NaiveSvBackend::allocate` and `SoaSvBackend::allocate`, assert
  `(buf.as_ptr() as usize) % 64 == 0` for a representative `num_qubits`.

**Correctness preserved by existing suites (must stay green, untouched):**

- AoS ≡ SoA ≡ Naive oracle equivalence at 1e-12 (`all_fixtures_match_naive`
  and the generated oracle tests).
- The P2-01 thread-sweep invariant (`scripts/p2-01-thread-sweep.sh`): bit-
  identical across `RAYON_NUM_THREADS ∈ {1,2,4,8}`.

Because no amplitude arithmetic changes, these passing unchanged is the primary
correctness evidence.

---

## 6. Benchmark, audit, report (AC #1, #2, #3)

All measurement on a **verified-idle** box (CLAUDE.md idle-check:
`uptime` load ≈ 0, `pgrep -af "cargo bench|bencher run|Runner.Worker"` clear)
to avoid the CI-race contamination that bit P2-01.

- **Scaling re-measure (AC #3):** QFT-25 scaling on the idle EPYC vs the P2-01
  baseline (`run_optimized`, not raw `run`). Honest expectation: flat within
  criterion CI.
- **False-sharing audit (AC #2):** `perf c2c record`/`report` on a QFT-25 run at
  8–16 threads on the EPYC; capture the HITM (cache-line contention) summary
  showing no cross-core hot lines. This is the substantive deliverable for AC #2.
- **Report:** `docs/perf/phase2-p2-02.md` — documents the audit (the three
  reasons from §1), the `perf c2c` evidence, the flat scaling result, and the
  reframing of the change as a guarantee + NUMA hook (P2-03), not a speedup.
- **AC #1 ("audit complete, padding applied where needed"):** satisfied by the
  audit + the alignment guarantee; explicitly note that per-thread struct
  padding (`#[repr(align(64))]`) is deferred until a parallel accumulator
  actually exists to contend on (will land with a future parallel sampler).

---

## 7. Out of scope / follow-ups

- `#[repr(align(64))]` per-thread accumulator padding — deferred; no parallel
  accumulator exists yet (note in report, link to future parallel-sampler work).
- AVX-512 aligned `load`/`store` switch — measured, adopted only on a proven win.
- NUMA-aware / first-touch allocation — **P2-03**, builds on `AlignedBuf`.
- Chunk-size tuning — **P2-04**.

---

## 8. PR / process

- Branch `p2-02-cache-line-alignment` off `main` (no worktrees — per user
  preference).
- One PR, title `[P2-02] Cache-line-aligned state buffers`, `Closes #<issue>`
  (look up the GitHub issue number for P2-02 — use the **issue** number, not
  the PR number).
- PR body: approach summary, test results (unit + miri + oracle + thread-sweep),
  `perf c2c` audit summary, scaling numbers, and the honest framing.
- CI green (build, test, clippy `-D warnings`, fmt) before merge.
