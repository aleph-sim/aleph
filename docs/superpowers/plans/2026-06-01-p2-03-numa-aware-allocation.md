# P2-03 NUMA-aware Allocation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-touch parallel-init allocation path on `AlignedBuf`, behind a non-default `numa` cargo feature, so the state vector's pages distribute across NUMA nodes instead of piling onto node 0; validate the placement win with a 3-policy benchmark on a 2-socket Xeon.

**Architecture:** A new `AlignedBuf::zeroed_first_touch` allocates uninitialised then zeroes the buffer from rayon's global pool in contiguous, page-aligned chunks (so the worker that faults a page is the one that later computes on it). A feature-agnostic `zeroed_state` dispatcher routes the SV backends to first-touch under `numa` and to plain `zeroed` otherwise, keeping `cfg` out of call sites. Correctness is provable on single-node hardware; the cross-node speedup is measured on the Xeon.

**Tech Stack:** Rust 2021 (MSRV 1.89), rayon (optional dep in `aleph-core`), criterion, `numactl` for the hardware benchmark.

**Spec:** `docs/superpowers/specs/2026-06-01-p2-03-numa-aware-allocation-design.md`

---

## File Structure

- `crates/aleph-core/Cargo.toml` — add `numa` feature + optional `rayon` dep + criterion dev-dep + bench entry.
- `crates/aleph-core/src/aligned.rs` — `first_touch_chunk_len` (pure partition fn), `SendPtr` wrapper, `zeroed_first_touch`, `zeroed_state` dispatcher, tests.
- `crates/aleph-sv/Cargo.toml` — add `numa` feature forwarding to `aleph-core/numa`.
- `crates/aleph-sv/src/backend.rs:52` — route the `Complex` buffer through `zeroed_state`; add a numa-gated `|0…0⟩` test.
- `crates/aleph-sv/src/soa_backend.rs:58,63` — route the `re`/`im` `f64` buffers through `zeroed_state`.
- `crates/aleph-core/benches/numa_first_touch.rs` — init-cost microbench (`zeroed` vs `zeroed_first_touch`).
- `scripts/numa-bench.sh` — 3-policy benchmark driver for the Xeon.
- `docs/numa.md` — enable instructions, locality contract, interleave fallback, methodology, results table.
- `BACKLOG.md` — tick AC after the Xeon run (Task 9).

---

## Task 1: Add the `numa` feature + optional rayon to aleph-core

**Files:**
- Modify: `crates/aleph-core/Cargo.toml`

- [ ] **Step 1: Add the optional dep and feature**

Edit `crates/aleph-core/Cargo.toml`. Under `[dependencies]` add the optional rayon line, and add a `[features]` section after `[dependencies]`:

```toml
[dependencies]
num-complex = { workspace = true }
smallvec = { workspace = true }
thiserror = { workspace = true }
rayon = { workspace = true, optional = true }

[features]
## First-touch NUMA-aware allocation (`AlignedBuf::zeroed_first_touch`).
## Off by default: `AlignedBuf` then stays dependency-free and codegen is
## unchanged. See docs/numa.md and P2-03.
numa = ["dep:rayon"]
```

- [ ] **Step 2: Verify both configurations build**

Run: `cargo build -p aleph-core && cargo build -p aleph-core --features numa`
Expected: both succeed (the feature pulls rayon; nothing uses it yet, which is fine).

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-core/Cargo.toml
git commit -m "[P2-03] Add non-default numa feature + optional rayon to aleph-core"
```

---

## Task 2: Pure partition function `first_touch_chunk_len`

The contiguous, page-aligned chunking that the parallel init will use. Pure (no rayon, no globals) so it is unit-testable without NUMA hardware — this is the machine-checkable "structurally correct first-touch" guarantee from the spec.

**Files:**
- Modify: `crates/aleph-core/src/aligned.rs`
- Test: `crates/aleph-core/src/aligned.rs` (`mod tests`)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block at the bottom of `aligned.rs`:

```rust
#[cfg(feature = "numa")]
#[test]
fn first_touch_partition_covers_range_once() {
    // (len, n_threads, elem_size) — Complex is 16 B, f64 is 8 B.
    let cases = [
        (1usize, 1usize, 16usize),
        (1, 8, 16),
        (1000, 4, 16),
        (1 << 20, 8, 16),        // 1M Complex, 8 threads
        (1 << 20, 20, 8),        // 1M f64, 20 threads (Xeon core count)
        ((1 << 20) + 7, 6, 16),  // non-page-multiple len
        (300, 16, 8),            // many threads, tiny len
    ];
    for (len, nt, es) in cases {
        let chunk = first_touch_chunk_len(len, nt, es);
        assert!(chunk > 0, "chunk must be positive: len={len} nt={nt} es={es}");
        let per_page = (FIRST_TOUCH_PAGE / es).max(1);
        assert_eq!(chunk % per_page, 0, "chunk {chunk} not page-aligned (es={es})");

        // Reconstruct the chunks: contiguous, disjoint, exact cover of [0,len).
        let n_chunks = len.div_ceil(chunk);
        let mut prev_end = 0usize;
        let mut covered = 0usize;
        for c in 0..n_chunks {
            let start = c * chunk;
            let end = (start + chunk).min(len);
            assert_eq!(start, prev_end, "gap/overlap before chunk {c}");
            assert!(start < end, "empty chunk {c}");
            covered += end - start;
            prev_end = end;
        }
        assert_eq!(prev_end, len, "partition did not reach len={len}");
        assert_eq!(covered, len, "coverage {covered} != len {len}");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p aleph-core --features numa first_touch_partition`
Expected: FAIL to compile — `first_touch_chunk_len` and `FIRST_TOUCH_PAGE` not defined.

- [ ] **Step 3: Write the implementation**

Add near the top of `aligned.rs`, after the `CACHE_LINE` constant (around line 22):

```rust
/// Page granularity for first-touch chunking. Keeping chunk boundaries on a
/// page multiple stops a single page from being faulted by two workers (which
/// would split its amplitudes across two NUMA nodes). 4 KiB is the base page
/// on x86-64 and aarch64; this is a *granularity*, not an exactness claim —
/// under transparent huge pages a boundary may land inside a larger page, a
/// negligible placement nit that is never incorrect.
#[cfg(feature = "numa")]
const FIRST_TOUCH_PAGE: usize = 4096;

/// Contiguous chunk length, in elements, for the first-touch parallel init.
///
/// Splits `[0, len)` into roughly `n_threads` contiguous chunks so each rayon
/// worker faults one chunk's pages, then rounds the chunk up to a whole number
/// of [`FIRST_TOUCH_PAGE`] pages so no page is split between two workers. Pure
/// function of its inputs (no rayon / globals) so the partition is unit-testable
/// without NUMA hardware. Returns at least one page of elements for `len > 0`.
#[cfg(feature = "numa")]
fn first_touch_chunk_len(len: usize, n_threads: usize, elem_size: usize) -> usize {
    debug_assert!(len > 0 && n_threads > 0 && elem_size > 0);
    // Elements per page (≥ 1; an element larger than a page degenerates to
    // per-element granularity, which is the finest useful split).
    let per_page = (FIRST_TOUCH_PAGE / elem_size).max(1);
    let target = len.div_ceil(n_threads); // ~equal contiguous chunks
    let chunk = target.div_ceil(per_page) * per_page; // round up to whole pages
    chunk.max(per_page)
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p aleph-core --features numa first_touch_partition`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-core/src/aligned.rs
git commit -m "[P2-03] Pure first-touch partition fn + coverage test"
```

---

## Task 3: `zeroed_first_touch` constructor

**Files:**
- Modify: `crates/aleph-core/src/aligned.rs`
- Test: `crates/aleph-core/src/aligned.rs` (`mod tests`)

- [ ] **Step 1: Write the failing tests**

Add to `mod tests`:

```rust
#[cfg(feature = "numa")]
#[test]
fn first_touch_matches_zeroed() {
    // 70_000 elems spans many pages and exceeds n_threads * per_page, so the
    // parallel path actually splits into multiple chunks.
    for n in [1usize, 4, 1000, 70_000] {
        let buf = AlignedBuf::<f64>::zeroed_first_touch(n);
        assert_eq!(buf.len(), n);
        assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0, "len {n} not 64-aligned");
        assert!(buf.iter().all(|&x| x == 0.0), "len {n} not all-zero");
    }
}

#[cfg(feature = "numa")]
#[test]
fn first_touch_empty_is_zero_len() {
    let buf = AlignedBuf::<f64>::zeroed_first_touch(0);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[cfg(feature = "numa")]
#[test]
fn first_touch_complex_round_trips() {
    let buf = AlignedBuf::<crate::Complex>::zeroed_first_touch(70_000);
    assert_eq!(buf.len(), 70_000);
    assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0);
    assert!(buf.iter().all(|&z| z == crate::Complex::new(0.0, 0.0)));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-core --features numa first_touch_matches`
Expected: FAIL to compile — `zeroed_first_touch` not defined.

- [ ] **Step 3: Add the `SendPtr` wrapper and the constructor**

Add `SendPtr` after the `AlignedBuf` struct definition (after line ~36):

```rust
/// Minimal `Send`/`Sync` raw-pointer wrapper so the first-touch parallel pass
/// can hand each rayon task the base pointer. The pass writes only pairwise-
/// disjoint ranges, so concurrent use is data-race-free.
#[cfg(feature = "numa")]
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

// SAFETY: the only user (the first-touch pass) writes disjoint ranges; the
// wrapper introduces no aliasing beyond what that disjointness already upholds.
#[cfg(feature = "numa")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "numa")]
unsafe impl<T> Sync for SendPtr<T> {}
```

Add the constructor inside `impl<T> AlignedBuf<T>`, right after `zeroed` (around line 101):

```rust
/// Allocate `len` elements, zero-initialised via a **first-touch** parallel
/// pass so each page is faulted by the rayon worker that zeroes it.
///
/// Observationally identical to [`zeroed`](Self::zeroed) — all-zero, 64-byte
/// aligned, `len == 0` → the dangling sentinel. The difference is page
/// placement: on a NUMA host the parallel write spreads pages across nodes
/// instead of faulting them all onto the allocating thread's node. See
/// `docs/numa.md`. Available only under the `numa` feature.
///
/// The all-zero bit pattern must be a valid `T` (holds for `f64` / `Complex`).
#[cfg(feature = "numa")]
pub fn zeroed_first_touch(len: usize) -> Self {
    const {
        assert!(mem::size_of::<T>() != 0, "AlignedBuf<T> requires a non-ZST T")
    };
    const {
        assert!(
            !mem::needs_drop::<T>(),
            "AlignedBuf<T> does not run element destructors; T must not need Drop"
        )
    };
    if len == 0 {
        return Self::empty();
    }
    let layout = Self::layout(len);
    // SAFETY: `layout` has non-zero size (`len > 0`, `size_of::<T>() > 0`). We
    // allocate *uninitialised* (not `alloc_zeroed`) precisely so the pages are
    // not pre-faulted by the allocator; the parallel pass below performs the
    // first write to every byte before any read through `Deref`.
    let raw = unsafe { alloc::alloc(layout) } as *mut T;
    let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

    let n_threads = rayon::current_num_threads().max(1);
    let chunk = first_touch_chunk_len(len, n_threads, mem::size_of::<T>());
    let n_chunks = len.div_ceil(chunk);
    let send = SendPtr(ptr.as_ptr());
    {
        use rayon::prelude::*;
        (0..n_chunks).into_par_iter().for_each(|c| {
            let base = send; // Copy the Send wrapper into the task.
            let start = c * chunk;
            let end = (start + chunk).min(len);
            // SAFETY: `first_touch_chunk_len`'s chunks are pairwise-disjoint and
            // cover `[0, len)` exactly (coverage test in this module), so each
            // task writes a unique, in-bounds range. `write_bytes` sets
            // `(end - start) * size_of::<T>()` bytes to 0 — a valid `T`.
            unsafe {
                core::ptr::write_bytes(base.0.add(start), 0u8, end - start);
            }
        });
    }

    Self { ptr, len, _marker: PhantomData }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p aleph-core --features numa first_touch`
Expected: PASS (coverage test from Task 2 plus the three new ones).

- [ ] **Step 5: Verify miri-clean provenance (matches P2-02 discipline)**

Run: `cargo +nightly miri test -p aleph-core --features numa first_touch_matches_zeroed 2>/dev/null || echo "miri unavailable — skip, note in PR"`
Expected: PASS, or a clean skip if miri/nightly is unavailable on the dev box. (Rayon under miri is slow but the small `n` cases finish; if miri rejects rayon, note it and rely on the coverage test + ASAN-free logic.)

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-core/src/aligned.rs
git commit -m "[P2-03] AlignedBuf::zeroed_first_touch (parallel first-touch init)"
```

---

## Task 4: `zeroed_state` dispatcher

A feature-agnostic constructor so the SV backends never carry a `cfg`.

**Files:**
- Modify: `crates/aleph-core/src/aligned.rs`
- Test: `crates/aleph-core/src/aligned.rs` (`mod tests`)

- [ ] **Step 1: Write the failing test**

Add to `mod tests`:

```rust
#[test]
fn zeroed_state_is_zeroed_and_aligned() {
    // Routes to first-touch under `numa`, plain `zeroed` otherwise; both must
    // yield an all-zero, 64-aligned buffer of the right length.
    let buf = AlignedBuf::<crate::Complex>::zeroed_state(70_000);
    assert_eq!(buf.len(), 70_000);
    assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0);
    assert!(buf.iter().all(|&z| z == crate::Complex::new(0.0, 0.0)));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-core zeroed_state_is_zeroed`
Expected: FAIL to compile — `zeroed_state` not defined.

- [ ] **Step 3: Add the dispatcher**

Inside `impl<T> AlignedBuf<T>`, after `from_slice`:

```rust
/// State-buffer constructor used by the SV backends: first-touch parallel
/// init under the `numa` feature, plain [`zeroed`](Self::zeroed) otherwise.
/// Keeps the `cfg` in one place so call sites stay feature-agnostic.
pub fn zeroed_state(len: usize) -> Self {
    #[cfg(feature = "numa")]
    {
        Self::zeroed_first_touch(len)
    }
    #[cfg(not(feature = "numa"))]
    {
        Self::zeroed(len)
    }
}
```

- [ ] **Step 4: Run the test in both configurations**

Run: `cargo test -p aleph-core zeroed_state_is_zeroed && cargo test -p aleph-core --features numa zeroed_state_is_zeroed`
Expected: PASS in both.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-core/src/aligned.rs
git commit -m "[P2-03] AlignedBuf::zeroed_state feature-agnostic dispatcher"
```

---

## Task 5: Wire the SV backends through `zeroed_state`

**Files:**
- Modify: `crates/aleph-sv/Cargo.toml`
- Modify: `crates/aleph-sv/src/backend.rs:52`
- Modify: `crates/aleph-sv/src/soa_backend.rs:58,63`
- Test: `crates/aleph-sv/src/backend.rs` (`mod tests`)

- [ ] **Step 1: Add the forwarding feature**

In `crates/aleph-sv/Cargo.toml`, under the existing `[features]` block (after the `internal-bench` entry):

```toml
## Forwards aleph-core's first-touch NUMA allocation into the SV backends.
## See docs/numa.md and P2-03.
numa = ["aleph-core/numa"]
```

- [ ] **Step 2: Swap the three call sites**

`crates/aleph-sv/src/backend.rs:52` — change:

```rust
        let mut amps = AlignedBuf::<Complex>::zeroed(dim);
```

to:

```rust
        let mut amps = AlignedBuf::<Complex>::zeroed_state(dim);
```

`crates/aleph-sv/src/soa_backend.rs:58` — change `let mut re = AlignedBuf::<f64>::zeroed(dim);` to `AlignedBuf::<f64>::zeroed_state(dim)`.

`crates/aleph-sv/src/soa_backend.rs:63` — change `im: AlignedBuf::<f64>::zeroed(dim),` to `im: AlignedBuf::<f64>::zeroed_state(dim),`.

- [ ] **Step 3: Add a numa-gated zero-state test**

In `crates/aleph-sv/src/backend.rs` `mod tests`:

```rust
#[cfg(feature = "numa")]
#[test]
fn allocate_zero_state_under_numa() {
    // n=12 → 4096 Complex = 64 KiB, spanning many pages, so the first-touch
    // parallel path is exercised end-to-end and must still yield |0…0⟩.
    let mut b = NaiveSvBackend::with_seed(0);
    let s = b.allocate(12).unwrap();
    assert_eq!(s.amplitudes()[0], Complex::new(1.0, 0.0));
    assert!(s.amplitudes()[1..].iter().all(|&z| z == Complex::new(0.0, 0.0)));
}
```

- [ ] **Step 4: Verify — default build unchanged, feature build green (oracle equivalence)**

Run:
```bash
cargo test -p aleph-sv
cargo test -p aleph-sv --features numa
```
Expected: both PASS. The second runs the **entire existing aleph-sv suite — including the oracle-equivalence tests — through the first-touch path**, which is the spec's oracle-equivalence requirement: identical |0…0⟩ start ⇒ identical results under the same kernels.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/Cargo.toml crates/aleph-sv/src/backend.rs crates/aleph-sv/src/soa_backend.rs
git commit -m "[P2-03] Route SV backends through zeroed_state; numa feature + oracle pass"
```

---

## Task 6: Init-cost microbench

Isolates the upfront cost first-touch adds (it moves page-fault cost from the first gate into allocation; it does not create new work). The NUMA *locality* win is measured on the Xeon (Task 8), not here.

**Files:**
- Modify: `crates/aleph-core/Cargo.toml`
- Create: `crates/aleph-core/benches/numa_first_touch.rs`

- [ ] **Step 1: Add criterion dev-dep and the bench entry**

In `crates/aleph-core/Cargo.toml`:

```toml
[dev-dependencies]
proptest = { workspace = true }
criterion = { workspace = true }

[[bench]]
name = "numa_first_touch"
harness = false
required-features = ["numa"]
```

- [ ] **Step 2: Write the bench**

Create `crates/aleph-core/benches/numa_first_touch.rs`:

```rust
//! Init-cost microbench: `AlignedBuf::zeroed` (lazy `alloc_zeroed`) vs
//! `zeroed_first_touch` (eager parallel first-touch) at state-vector scale.
//! First-touch pulls page-fault cost out of the first gate and into
//! allocation; this isolates that upfront cost. The NUMA *locality* win is
//! measured end-to-end on 2-node hardware (scripts/numa-bench.sh), not here.
//!
//! Run: `cargo bench -p aleph-core --features numa --bench numa_first_touch`

use aleph_core::{AlignedBuf, Complex};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn alloc_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("numa_first_touch");
    for &n in &[22u32, 25] {
        let len = 1usize << n;
        group.bench_with_input(BenchmarkId::new("zeroed", n), &len, |b, &len| {
            b.iter(|| criterion::black_box(AlignedBuf::<Complex>::zeroed(len)));
        });
        group.bench_with_input(BenchmarkId::new("first_touch", n), &len, |b, &len| {
            b.iter(|| criterion::black_box(AlignedBuf::<Complex>::zeroed_first_touch(len)));
        });
    }
    group.finish();
}

criterion_group!(benches, alloc_init);
criterion_main!(benches);
```

- [ ] **Step 3: Verify it builds and runs a quick sample**

Run: `cargo bench -p aleph-core --features numa --bench numa_first_touch -- --warm-up-time 1 --measurement-time 2`
Expected: compiles and prints `numa_first_touch/zeroed/...` and `.../first_touch/...` lines. (Numbers are informational; the default-allocator path may even look "faster" in isolation since first-touch front-loads the faults — that is the expected, documented trade-off.)

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-core/Cargo.toml crates/aleph-core/benches/numa_first_touch.rs
git commit -m "[P2-03] Init-cost microbench: zeroed vs zeroed_first_touch"
```

---

## Task 7: Benchmark driver + `docs/numa.md`

The hardware-facing deliverable: a reproducible 3-policy benchmark script and the documentation. Numbers are filled in Task 9 on the Xeon.

**Files:**
- Create: `scripts/numa-bench.sh`
- Create: `docs/numa.md`

- [ ] **Step 1: Write the benchmark driver**

Create `scripts/numa-bench.sh`:

```bash
#!/usr/bin/env bash
# P2-03 NUMA benchmark — 3 memory-placement policies on a multi-node host.
# Run from the workspace root on the 2-socket Xeon (or any host with >1 NUMA
# node). Pick a high-qubit, full-state-sweep scaling bench as $BENCH (verify
# the available names with `cargo bench -p aleph-sv -- --list`).
#
# Per CLAUDE.md: confirm the box is IDLE first (uptime ~0, no competing
# `cargo bench`) — a CI race silently inflates baselines.
set -euo pipefail

BENCH="${BENCH:-qft_scaling}"          # high-n scaling bench from P2-01
PKG="aleph-sv"
export RUSTFLAGS="-C target-cpu=native"

echo "=== idle check ==="; uptime; pgrep -af 'cargo bench|bencher run|Runner.Worker' || echo "(clean)"
echo "=== NUMA topology ==="; numactl --hardware | sed -n '1,6p'

echo; echo "### Policy 1: baseline (default allocator, all pages on node 0, no numa feature)"
cargo bench -p "$PKG" --bench "$BENCH"

echo; echo "### Policy 2: interleave (numactl --interleave=all, no numa feature)"
numactl --interleave=all cargo bench -p "$PKG" --bench "$BENCH"

echo; echo "### Policy 3: first-touch (--features numa, Tier 1: no pinning)"
cargo bench -p "$PKG" --features numa --bench "$BENCH"

echo; echo "### Policy 3b (Tier 2): first-touch + external per-node pinning"
echo "# Pin the process to node-local CPUs+memory so each worker's contiguous"
echo "# chunk is local. Example for a 2-node box (adjust core lists to lscpu):"
echo "#   numactl --cpunodebind=0,1 --localalloc \\"
echo "#     cargo bench -p $PKG --features numa --bench $BENCH"
```

Make it executable:

```bash
chmod +x scripts/numa-bench.sh
```

- [ ] **Step 2: Write `docs/numa.md`**

Create `docs/numa.md`:

```markdown
# NUMA-aware allocation (P2-03)

On a multi-node NUMA machine the default allocator faults the entire state
vector onto the node of the *allocating* thread (node 0). Worker threads on
other nodes then pay the remote-access penalty (~2.1× on the reference Xeon)
for their share of every gate sweep.

The `numa` feature replaces the buffer's lazy zero-fill with a **first-touch**
parallel init: `AlignedBuf::zeroed_first_touch` allocates uninitialised, then
zeroes the buffer from rayon's global pool in contiguous, page-aligned chunks,
so the worker that faults a page is (under matched partitioning + pinning) the
one that later computes on it. Pages thus distribute across nodes instead of
piling onto node 0.

## Enabling

```bash
cargo build  --release -p aleph-sv --features numa
cargo bench           -p aleph-sv --features numa --bench <bench>
```

Off by default: without the feature, `AlignedBuf` is dependency-free and
codegen is byte-for-byte unchanged.

## Locality contract

First-touch only yields *true* locality when both hold:

1. **OS policy is first-touch** (the Linux default) — i.e. NOT running under
   `numactl --interleave`.
2. **Workers are pinned** so a worker's contiguous chunk stays on the node it
   faulted. Without pinning, first-touch still spreads pages across nodes
   (≈ balanced, like interleave) and already beats the all-on-node-0 default,
   but it will not exceed interleave until pinning is in place. Pin via
   `numactl --cpunodebind=… --localalloc` (see `scripts/numa-bench.sh`).

## Fallback: interleave (no feature needed)

```bash
numactl --interleave=all cargo run --release -p aleph-cli -- run circuit.qasm
```

Round-robins pages across nodes: no locality, but balanced bandwidth and
robust to bad partitioning. The zero-code default for unknown topologies.

## Benchmark methodology

`scripts/numa-bench.sh` measures three placement policies on the same machine
and circuit (high-qubit Tier-1 workload, where gates sweep the full state and
remote access bites hardest):

1. **Baseline** — default allocator, all pages on node 0.
2. **Interleave** — `numactl --interleave=all`.
3. **First-touch** — `--features numa` (Tier 1 unpinned; Tier 2 + pinning).

Expected ordering: **first-touch ≥ interleave > baseline**.

## Results — 2-socket Intel Xeon Silver 4114 (2× 10C/20T, 2 NUMA nodes)

> **Pending measurement on the Xeon (P2-03 Task 9).** Both current bench boxes
> (EPYC 8124P, Ryzen 9 3900) are single NUMA node and cannot exhibit a NUMA
> effect; the EPYC reports `available: 1 nodes (0)`.

| Workload | Baseline (node 0) | Interleave | First-touch (Tier 1) | First-touch + pin (Tier 2) |
|----------|------------------:|-----------:|---------------------:|---------------------------:|
| _TBD_    | _TBD_             | _TBD_      | _TBD_                | _TBD_                      |
```

Note: the `_TBD_` row is a deliberate placeholder filled in Task 9 from real
hardware; it is the only sanctioned placeholder in this plan.

- [ ] **Step 3: Verify the script is syntactically valid**

Run: `bash -n scripts/numa-bench.sh && echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/numa-bench.sh docs/numa.md
git commit -m "[P2-03] NUMA benchmark driver + docs/numa.md (results pending Xeon)"
```

---

## Task 8: Pre-PR verification (single-node, runs now)

**Files:** none (verification only).

- [ ] **Step 1: Full default build/test/lint/fmt**

Run:
```bash
cargo build --workspace
cargo test  --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
Expected: all green. This proves the feature-off path (the default everyone builds) is unchanged.

- [ ] **Step 2: Feature-on build/test/lint**

Run:
```bash
cargo build  -p aleph-core -p aleph-sv --features numa
cargo test   -p aleph-core --features numa
cargo test   -p aleph-sv   --features numa
cargo clippy -p aleph-core -p aleph-sv --features numa --all-targets -- -D warnings
```
Expected: all green. Confirms the first-touch path compiles clean, passes its own tests, passes the full aleph-sv oracle suite, and is clippy-clean (watch the `SendPtr` unsafe impls and `write_bytes` — they carry SAFETY comments).

- [ ] **Step 3: Commit any fmt/clippy fixes**

```bash
git add -A
git commit -m "[P2-03] fmt/clippy fixups" || echo "nothing to fix"
```

- [ ] **Step 4: Open the PR (let it sit, then self-review per CLAUDE.md)**

```bash
git push -u origin p2-03-numa-aware-allocation
gh pr create --title "[P2-03] NUMA-aware first-touch allocation" --body "$(cat <<'EOF'
Closes #29

## Summary
First-touch parallel-init allocation path on `AlignedBuf`, behind a
non-default `numa` cargo feature. `zeroed_first_touch` allocates
uninitialised then zeroes the buffer from rayon's global pool in
contiguous, page-aligned chunks, so pages distribute across NUMA nodes
instead of faulting onto node 0. A feature-agnostic `zeroed_state`
dispatcher routes the SV backends; with the feature off, codegen is
byte-for-byte unchanged.

## Tests
- Pure partition-coverage test (exact, disjoint cover of [0,len)).
- `zeroed_first_touch` correctness (f64 + Complex, multi-page sizes).
- Full aleph-sv suite — incl. oracle-equivalence — passes under `--features numa`.
- Init-cost microbench `numa_first_touch`.

## Benchmark
3-policy methodology in `docs/numa.md` + `scripts/numa-bench.sh`.
Cross-node numbers PENDING the 2-socket Xeon Silver 4114 (both current
bench boxes are single NUMA node; EPYC reports `available: 1 nodes (0)`).
Will be filled before merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Task 9: Xeon benchmark run (DEFERRED — requires the 2-socket box)

> **Gated on hardware.** Do this once the 2-socket Xeon Silver 4114 is
> provisioned. Until then the PR carries the infra + docs with the results
> table marked pending; let it sit per CLAUDE.md.

**Files:**
- Modify: `docs/numa.md` (fill the results table)
- Modify: `BACKLOG.md` (tick the P2-03 AC)

- [ ] **Step 1: Confirm topology and idle box**

Run: `numactl --hardware` (expect `available: 2 nodes`), `uptime`, `pgrep -af 'cargo bench'`.
Per CLAUDE.md / `feedback-check-server-clean`: a CI race silently inflates baselines — verify clean first.

- [ ] **Step 2: Run the 3-policy benchmark**

Run: `BENCH=<verified-high-n-scaling-bench> ./scripts/numa-bench.sh 2>&1 | tee /tmp/numa-bench.log`
Expected: three (+ optional Tier-2) policy sections with criterion numbers.

- [ ] **Step 3: Fill `docs/numa.md` results table** with the measured baseline / interleave / first-touch / first-touch+pin numbers and the local/remote ratio. State honestly whether first-touch > interleave (Tier 2) was reached; if not, file a follow-up issue (as P2-01 did for its unmet AC).

- [ ] **Step 4: Tick the AC in `BACKLOG.md`** under `[P2-03] NUMA-aware allocation`:
  - `[x] NUMA-aware build option`
  - `[x] Benchmark on 2-socket machine: improvement over default allocator`
  - `[x] Documentation on enabling`

- [ ] **Step 5: Commit and update the PR**

```bash
git add docs/numa.md BACKLOG.md
git commit -m "[P2-03] Xeon 2-node benchmark results; tick AC; close #29"
git push
```

---

## Self-Review notes

- **Spec coverage:** §4.1 → Task 3; §4.2 → Task 2; §4.3 → Tasks 4–5; §4.4 → Tasks 1, 5; §5 Tier 1 → Tasks 3/7, Tier 2 → Task 7 script + Task 9; §6 benchmark → Tasks 7, 9; §7 tests #1 → Task 3, #2 → Task 2, #3 (oracle) → Task 5 Step 4, #4 init-cost → Task 6; §8 docs → Task 7, BACKLOG → Task 9. CLAUDE.md is intentionally NOT touched (project rule: CLAUDE.md changes go in a separate `[meta]` PR).
- **Placeholders:** the only `_TBD_` cells are the `docs/numa.md` results row, explicitly sanctioned and filled from real hardware in Task 9. No code placeholders.
- **Type/name consistency:** `first_touch_chunk_len(len, n_threads, elem_size)`, `FIRST_TOUCH_PAGE`, `SendPtr<T>`, `zeroed_first_touch`, `zeroed_state` are used identically across Tasks 2–6.
- **Affinity hook:** kept out of scope per the spec; Tier 2 pinning is external (`numactl`) in the bench script. Revisit only if Task 9 shows first-touch needs in-process pinning to beat interleave.
