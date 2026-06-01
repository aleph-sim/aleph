# P2-03 — NUMA-aware allocation (first-touch on `AlignedBuf`)

**Date:** 2026-06-01
**Issue:** P2-03 (GitHub #29) — *NUMA-aware allocation*
**Depends on:** P2-02 (`AlignedBuf`, merged `c4222e6`)
**Status:** design approved, pending spec review

-----

## 1. Goal

On a multi-node NUMA machine the default allocator faults the entire state
vector onto the node of the *allocating* thread (node 0). Worker threads
running on other nodes then pay the remote-access penalty (~2.1× on the
target Xeon, `node distances` 10/21) for half of every gate sweep.

P2-03 delivers a **first-touch allocation path** that faults the state
vector's pages in parallel from rayon's global worker pool, so pages are
distributed across nodes instead of piling onto node 0. It builds directly
on `AlignedBuf` — the P2-02 module docs already name "the allocation hook
P2-03 (NUMA first-touch) will extend."

The path is gated behind a non-default `numa` cargo feature: when off,
behaviour and codegen are byte-for-byte identical to today (`zeroed`),
with zero overhead.

## 2. Hardware context (why the benchmark needs a specific box)

Both current bench boxes are **single NUMA node** and cannot exhibit any
NUMA effect:

- EPYC 8124P (primary): `numactl --hardware` reports `available: 1 nodes
  (0)` — NPS1, 32 CPUs, one node, distances `10`.
- Ryzen 9 3900 (secondary): single-socket desktop part, one node.

The benchmark AC ("improvement over default allocator") is validated on a
**2-socket Intel Xeon Silver 4114** box being provisioned for this ticket:
2× 10C/20T, 128 GB (64 GB/node), **2 NUMA nodes** over UPI. It is a
Skylake-SP part, so it **has AVX-512** (1 FMA unit on the Silver SKU) —
the production AoS AVX-512 kernel path runs there, so NUMA is validated on
the real compute path, not a scalar fallback. Absolute throughput is lower
than the EPYC (2.2 GHz, single AVX-512 FMA, Skylake AVX-512 downclock), but
that is irrelevant: we measure the *local-vs-remote ratio* on one machine,
not absolute speed. Optionally Skylake SNC can expose 4 nodes for a harsher
test; not required.

## 3. Scope

**In scope:** first-touch parallel-init constructor on `AlignedBuf`; `numa`
feature wiring in `aleph-core` and `aleph-sv`; correctness + partition-
coverage + oracle-equivalence tests (run on single-node hardware); an
init-cost criterion bench; a 3-policy NUMA benchmark on the Xeon; `docs/
numa.md`.

**Out of scope (YAGNI):** mimalloc / jemalloc global allocators; libnuma
FFI (`mbind`, `numa_alloc_onnode`); `move_pages`-based placement
verification; runtime auto-tuning of chunk size (that is P2-04). A
per-worker affinity hook is **conditionally** in scope — see §5.

## 4. Architecture

### 4.1 Core component

A new constructor in `crates/aleph-core/src/aligned.rs`, compiled only
under `aleph-core`'s `numa` feature:

```rust
#[cfg(feature = "numa")]
pub fn zeroed_first_touch(len: usize) -> Self
```

Semantics: observationally identical to `zeroed` (all-zero buffer, correct
`len`, 64-byte aligned, `len == 0` → the `NonNull::dangling()` sentinel).
The difference is *who faults the pages*:

- `zeroed` calls `alloc_zeroed`. Its pages are lazily faulted by whichever
  thread first touches them during the *first gate* — under rayon
  work-stealing, placement is unpredictable and tends to land on node 0.
- `zeroed_first_touch` calls `alloc` (uninitialised — so glibc does **not**
  pre-fault), then runs a **rayon parallel pass over contiguous equal
  chunks** of the buffer, each worker `write_bytes(0)` across its range.
  That first write faults each page locally *and* zero-initialises it.

The const-asserts (`!needs_drop::<T>()`, non-ZST) and the `len == 0`
dangling-sentinel branch carry over unchanged. The parallel pass MUST use
rayon's **global** pool (the same pool the gate kernels use via
`par_blocks`/`par_units`) so that init-time and compute-time thread→node
mappings can agree.

### 4.2 Partition: contiguous chunks

The init pass splits `[0, len)` into contiguous equal ranges, mirroring how
`par_units → par_blocks` (and rayon's default recursive halving with
`with_min_len(64)`) split the compute. Contiguous distribution is what
QuEST/Aer do and is the best *static* approximation of the per-gate
partition (which varies by target qubit). A pure-integer partition function
makes the chunking unit-testable without NUMA hardware.

### 4.3 Call sites

Two one-line allocation sites switch under the feature:

- `aleph-sv/src/backend.rs:52` — `AlignedBuf::<Complex>::zeroed(dim)`
- `aleph-sv/src/soa_backend.rs:58,63` — the `re` / `im` `f64` buffers

```rust
#[cfg(feature = "numa")]
let amps = AlignedBuf::<Complex>::zeroed_first_touch(dim);
#[cfg(not(feature = "numa"))]
let amps = AlignedBuf::<Complex>::zeroed(dim);
```

(Helper indirection acceptable to avoid repeating the `cfg` pair at three
sites — decided in planning.)

### 4.4 Feature gating

- `aleph-core`: `[features] numa = ["dep:rayon"]`, with `rayon` added as an
  **optional** dependency. Off by default → `AlignedBuf` stays dep-free and
  codegen is unchanged.
- `aleph-sv`: `[features] numa = ["aleph-core/numa"]`.

## 5. First-touch benefit: two tiers

- **Tier 1 (no pinning) — the ticket core.** A parallel init spreads
  first-touches across both nodes, yielding a roughly balanced placement
  (≈50/50), which already beats the all-on-node-0 default allocator. This
  is enough to satisfy the AC and needs **no new crates and no pinning**.
- **Tier 2 (pinned + matched partition) — stretch.** With workers pinned so
  each worker's contiguous range is on its local node, placement is truly
  local and beats even interleave. Pinning is done **first via external
  `numactl --cpunodebind` / `taskset` in the bench script**. A per-worker
  affinity hook (rayon `start_handler` + `sched_setaffinity`, behind the
  `numa` feature) is added **only if** we decide to machine-prove
  first-touch > interleave — that decision is deferred to the planning
  stage to keep the core dependency-light (CLAUDE.md "justify every dep").

## 6. Benchmark methodology (on the 2-socket Xeon)

Measure three memory-placement policies on the **same** machine and circuit
(Tier-1 workloads at high qubit count — QFT-28 / random-brickwall — where
high-qubit gates sweep the whole state and remote access bites hardest):

1. **Baseline — default allocator, no pinning.** `zeroed` → `alloc_zeroed`;
   all pages on node 0. The "everything on socket 0" pathology; the
   denominator for the AC.
2. **Interleave — `numactl --interleave=all`.** Round-robin pages; balanced,
   no locality. The "simple default"; works without the feature.
3. **First-touch — `--features numa`.** Parallel contiguous init (Tier 1),
   optionally + external pinning (Tier 2).

Expected ordering: **first-touch ≥ interleave > baseline**. AC met when #3
(and #2) beat #1. Report local/remote ratio and scaling efficiency in
`docs/numa.md`.

Per CLAUDE.md / [[feedback-check-server-clean]]: verify the box is idle
(`uptime`, `pgrep cargo bench`) before each measurement.

## 7. Testing requirements

All tests run on single-node hardware (EPYC / local); **none block on the
Xeon**:

1. **Correctness** — `zeroed_first_touch` returns an all-zero, correctly
   sized, 64-aligned buffer for `n ∈ {1, 4, 1000, ≥ a few pages}`, and is
   observationally identical to `zeroed`. Include the `Complex` element
   type end-to-end (mirrors the existing `complex_zeroed_and_round_trip`).
2. **Partition coverage** — pure-integer test that the contiguous-chunk
   partition function covers `[0, len)` exactly once with pairwise-disjoint
   chunks, in the style of the existing `par_blocks_visits_each_block_once`
   tests. This is the machine-checkable "structurally correct first-touch"
   guarantee that needs no NUMA hardware.
3. **Oracle equivalence** — a full QFT-12 run under `--features numa`
   produces a bit-identical state vector vs the non-`numa` build (1e-12).
   Guards that swapping the constructor changes nothing observable.
4. **Init-cost bench** — criterion `numa_first_touch` measuring `alloc +
   init` and `alloc + first_gate` *together* (first-touch moves page-fault
   cost from the first gate into allocation; it does not add new work), to
   show no wall-clock regression on a single node.

## 8. Documentation & bookkeeping

- **`docs/numa.md`** — how to enable (`--features numa`); the locality
  contract (needs first-touch OS policy + pinned workers); the interleave
  fallback (`numactl --interleave=all`, no feature needed); and the 3-policy
  Xeon benchmark results (local/remote numbers, efficiency).
- **`BACKLOG.md`** — tick all three AC after measuring; if Tier 2
  (first-touch > interleave) is not reached, file an honest follow-up (as
  P2-01 did for its unmet AC).
- **ADR** — only if first-touch becomes the *default* policy choice; a short
  ADR in `docs/decisions/`. Otherwise skip.

## 9. Risks

- **Pinning is required for true locality.** Without it, Tier 1 gives a
  balanced (interleave-like) placement — still beats baseline, but
  first-touch will not exceed interleave until Tier 2 pinning lands. Framed
  honestly in §5; the bench reports what each tier achieves.
- **Bandwidth wall (ADR 0008).** Our scaling is memory-bandwidth-bound; the
  NUMA win is bounded by how much of the sweep actually hits remote memory.
  Largest effect expected on high-qubit gates over the full state.
- **`alloc` vs `alloc_zeroed` semantics.** `zeroed_first_touch` must fully
  write every byte before any read; the parallel pass covers `[0, len)`
  exactly once (test #2 guards this) so no slot is left uninitialised.
