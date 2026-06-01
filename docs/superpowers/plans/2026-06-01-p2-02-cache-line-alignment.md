# P2-02 Cache-line-aligned state buffers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee 64-byte (cache-line) alignment for the state-vector buffers via a hand-rolled `AlignedBuf<T>`, audit the codebase for false sharing, and document the (expected-flat) scaling result honestly.

**Architecture:** A new `AlignedBuf<T>` owned-allocation primitive in `aleph-core` (raw `alloc_zeroed` at `align = 64`, `Deref`/`DerefMut` to `[T]`, `Drop`, `Send`/`Sync`) replaces the plain `Vec` in `CpuState.amps` and `SoaState.{re,im}`. No amplitude arithmetic changes, so existing oracle/thread-sweep suites are the correctness gate. The buffer is the allocation hook P2-03 (NUMA first-touch) will extend.

**Tech Stack:** Rust 2021 (MSRV 1.89), `std::alloc`, `core::ptr`/`NonNull`, rayon (existing), criterion + `perf c2c` for measurement.

**Spec:** `docs/superpowers/specs/2026-06-01-p2-02-cache-line-alignment-design.md`
**GitHub issue:** #28

---

## File Structure

- **Create** `crates/aleph-core/src/aligned.rs` — `AlignedBuf<T>`, `CACHE_LINE`, unit tests. Sole new file; one contained `unsafe` allocator.
- **Modify** `crates/aleph-core/src/lib.rs` — add `pub mod aligned;` + re-export.
- **Modify** `crates/aleph-sv/src/state.rs` — `CpuState.amps: AlignedBuf<Complex>`; fix the test fixture.
- **Modify** `crates/aleph-sv/src/backend.rs` — `NaiveSvBackend::allocate` uses `AlignedBuf::zeroed`; add alignment assertion test.
- **Modify** `crates/aleph-sv/src/soa_state.rs` — `SoaState.{re,im}: AlignedBuf<f64>`; fix test fixtures.
- **Modify** `crates/aleph-sv/src/soa_backend.rs` — `SoaSvBackend::allocate` uses `AlignedBuf::zeroed`; add alignment assertion test.
- **Create** `docs/perf/phase2-p2-02.md` — audit + scaling report.
- **Modify** `BACKLOG.md` — tick P2-02 acceptance checkboxes.

---

## Task 1: `AlignedBuf<T>` in aleph-core

**Files:**
- Create: `crates/aleph-core/src/aligned.rs`
- Modify: `crates/aleph-core/src/lib.rs` (add module + re-export, after the `gate` block near line 37)
- Test: inline `#[cfg(test)] mod tests` in `aligned.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/aleph-core/src/aligned.rs` with ONLY the tests first (no `AlignedBuf` yet) so the module fails to compile / tests fail:

```rust
//! `AlignedBuf<T>` — a fixed-size, cache-line-aligned, owned heap buffer.
//!
//! State vectors never resize, so a growable `Vec` is unnecessary; what we
//! want is a *guaranteed* 64-byte (cache-line) base alignment so that the
//! AoS SIMD units (`LANES = 4` complex = 64 bytes, in `aleph-sv`) sit on the
//! cache-line grid and parallel tasks never share a boundary line (P2-02).
//! It is also the allocation hook P2-03 (NUMA first-touch) will extend.
//!
//! Intended for `Copy`/POD element types (`f64`, `aleph_core::Complex`): the
//! buffer does NOT run element destructors on drop (it only frees the block).
//! `zeroed` relies on the all-zero bit pattern being a valid `T`, which holds
//! for `f64` and `Complex` (`#[repr(C)] { re: f64, im: f64 }`).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_is_cache_line_aligned() {
        for n in [0usize, 1, 4, 1000] {
            let buf = AlignedBuf::<f64>::zeroed(n);
            assert_eq!(
                buf.as_ptr() as usize % CACHE_LINE,
                0,
                "len {n} not 64-aligned"
            );
            assert_eq!(buf.len(), n);
        }
    }

    #[test]
    fn zeroed_contents_are_zero() {
        let buf = AlignedBuf::<f64>::zeroed(64);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn from_slice_round_trips() {
        let src = [1.0_f64, -2.0, 3.5, 4.0, 5.0];
        let buf = AlignedBuf::from_slice(&src);
        assert_eq!(&*buf, &src);
        assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[test]
    fn from_empty_slice_is_zero_len() {
        let buf = AlignedBuf::<f64>::from_slice(&[]);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn deref_mut_writes_through() {
        let mut buf = AlignedBuf::<f64>::zeroed(4);
        buf[2] = 9.0;
        assert_eq!(buf[2], 9.0);
        assert_eq!(buf[0], 0.0);
    }

    #[test]
    fn clone_is_independent_copy() {
        let mut a = AlignedBuf::<f64>::from_slice(&[1.0, 2.0, 3.0]);
        let b = a.clone();
        a[0] = 99.0;
        assert_eq!(&*b, &[1.0, 2.0, 3.0]);
        assert_eq!(b.as_ptr() as usize % CACHE_LINE, 0);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p aleph-core aligned 2>&1 | tail -20`
Expected: compile error — `cannot find type AlignedBuf` / `CACHE_LINE` not found.

- [ ] **Step 3: Implement `AlignedBuf<T>`**

Prepend the implementation above the `#[cfg(test)]` block in `crates/aleph-core/src/aligned.rs`:

```rust
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::{fmt, mem, slice};
use std::alloc::{self, Layout};

/// Cache-line size on all targets we support (x86-64, aarch64). 64 bytes =
/// exactly `LANES = 4` complex amplitudes, the AoS SIMD unit width.
pub const CACHE_LINE: usize = 64;

/// A fixed-size, `CACHE_LINE`-aligned, owned heap buffer of `T`.
///
/// See the module docs for the element-type contract (POD/`Copy`, no
/// destructors run). Construct with [`AlignedBuf::zeroed`] or
/// [`AlignedBuf::from_slice`]; access through `Deref`/`DerefMut` to `[T]`.
pub struct AlignedBuf<T> {
    /// Non-null, 64-aligned. For `len == 0` this is a 64-aligned sentinel
    /// (`CACHE_LINE as *mut T`) that is never dereferenced.
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    /// Layout for `len` elements at 64-byte alignment.
    ///
    /// `len` is bounded by the caller (`dim = 2^n`, `n ≤ 28`), so
    /// `len * size_of::<T>() ≤ 2^32` never approaches `isize::MAX` on a
    /// 64-bit target and the constructor cannot fail in-domain. An
    /// out-of-domain `len` is treated as an unsatisfiable allocation.
    fn layout(len: usize) -> Layout {
        let size = len.saturating_mul(mem::size_of::<T>());
        match Layout::from_size_align(size, CACHE_LINE) {
            Ok(l) => l,
            // Unreachable for in-domain `len`; route to the OOM handler.
            Err(_) => alloc::handle_alloc_error(Layout::new::<u8>()),
        }
    }

    /// 64-aligned, non-null, never-dereferenced sentinel for `len == 0`.
    fn empty() -> Self {
        // SAFETY: `CACHE_LINE` (64) is non-zero, so `new_unchecked` is sound;
        // the pointer is aligned for any `T` we use (size ∈ {8,16} | 64) and
        // is only ever handed to `slice::from_raw_parts(_, 0)`, which is
        // valid for a non-null, aligned pointer at length 0.
        let ptr = unsafe { NonNull::new_unchecked(CACHE_LINE as *mut T) };
        Self { ptr, len: 0, _marker: PhantomData }
    }

    /// Allocate `len` elements, zero-initialised.
    ///
    /// The all-zero bit pattern must be a valid `T` (holds for `f64` /
    /// `Complex`).
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        let layout = Self::layout(len);
        // SAFETY: `layout` has non-zero size (`len > 0`, `size_of::<T>() > 0`).
        let raw = unsafe { alloc::alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self { ptr, len, _marker: PhantomData }
    }

    /// Allocate and copy the contents of `src`.
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Copy,
    {
        let len = src.len();
        if len == 0 {
            return Self::empty();
        }
        let layout = Self::layout(len);
        // SAFETY: non-zero size; we initialise all `len` slots immediately
        // below via `copy_nonoverlapping` before any read.
        let raw = unsafe { alloc::alloc(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        // SAFETY: `src` holds `len` `T`; `ptr` owns `len` aligned, allocated
        // (uninitialised) slots; the regions do not overlap (fresh alloc).
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), ptr.as_ptr(), len);
        }
        Self { ptr, len, _marker: PhantomData }
    }
}

impl<T> Deref for AlignedBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` is non-null, 64-aligned, and points to `len`
        // initialised `T` (zeroed or copied); for `len == 0` the sentinel is
        // a valid zero-length base.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: same invariants as `deref`; `&mut self` gives unique access.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return; // `empty()` never allocated.
        }
        let layout = Self::layout(self.len);
        // SAFETY: `ptr` came from `alloc`/`alloc_zeroed` with exactly this
        // layout; we free the block once. Element destructors are
        // intentionally not run (POD element contract, see module docs).
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T: Copy> Clone for AlignedBuf<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedBuf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

// SAFETY: `AlignedBuf<T>` owns a unique heap region of `T` with no interior
// mutability or shared ownership — identical sharing semantics to `Vec<T>`,
// so it is `Send`/`Sync` under the same bounds.
unsafe impl<T: Send> Send for AlignedBuf<T> {}
unsafe impl<T: Sync> Sync for AlignedBuf<T> {}
```

- [ ] **Step 4: Register the module**

In `crates/aleph-core/src/lib.rs`, after the `gate` re-export block (around line 37), add:

```rust
pub mod aligned;
pub use aligned::{AlignedBuf, CACHE_LINE};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p aleph-core aligned 2>&1 | tail -20`
Expected: all six tests PASS.

- [ ] **Step 6: Lint + format**

Run: `cargo clippy -p aleph-core --all-targets -- -D warnings && cargo fmt -p aleph-core`
Expected: no warnings; no diff after fmt (or fmt fixes applied).

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-core/src/aligned.rs crates/aleph-core/src/lib.rs
git commit -m "[P2-02] Add AlignedBuf<T>: 64-byte-aligned owned buffer"
```

---

## Task 2: Validate the unsafe paths under miri

**Files:** none (verification only)

- [ ] **Step 1: Run miri on the aligned module**

Run: `cargo +nightly miri test -p aleph-core aligned 2>&1 | tail -30`
Expected: all six tests PASS with no UB / leak diagnostics.

If the nightly toolchain or miri component is not installed:
```bash
rustup toolchain install nightly --component miri
```
If miri is genuinely unavailable in the environment, record that in the eventual PR body and rely on the alignment + content assertions from Task 1 plus the leak-free `Drop` (covered by the `clone_is_independent_copy` allocate/free cycle). Do NOT skip silently.

- [ ] **Step 2: No commit** (verification step; nothing changed).

---

## Task 3: Wire `AlignedBuf` into the AoS path (`CpuState` + `NaiveSvBackend`)

**Files:**
- Modify: `crates/aleph-sv/src/state.rs:15` (field), `:38` (test fixture)
- Modify: `crates/aleph-sv/src/backend.rs:52` (allocate); add a test
- Test: inline tests in both files

- [ ] **Step 1: Add the failing alignment-assertion test**

In `crates/aleph-sv/src/backend.rs`, inside the existing `#[cfg(test)] mod tests`, add:

```rust
#[test]
fn allocated_state_is_cache_line_aligned() {
    let mut b = NaiveSvBackend::with_seed(0);
    let s = b.allocate(20).unwrap();
    assert_eq!(
        s.amplitudes().as_ptr() as usize % aleph_core::CACHE_LINE,
        0
    );
}
```

- [ ] **Step 2: Run it to verify it fails to compile / fails**

Run: `cargo test -p aleph-sv allocated_state_is_cache_line_aligned 2>&1 | tail -15`
Expected: PASS *by accident* is possible (mmap), so this is a guard not a driver — but it must at least compile. If `with_seed` is not the constructor, use `NaiveSvBackend::new()`. (Confirm the constructor name from `backend.rs:19/26`.)

- [ ] **Step 3: Change the field type**

In `crates/aleph-sv/src/state.rs`:
- Line 5: `use aleph_core::Complex;` → `use aleph_core::{AlignedBuf, Complex};`
- Line 15: `pub(crate) amps: Vec<Complex>,` → `pub(crate) amps: AlignedBuf<Complex>,`
- Line 38 (test fixture): `amps: vec![Complex::new(0.0, 0.0); 8],`
  → `amps: AlignedBuf::from_slice(&[Complex::new(0.0, 0.0); 8]),`

(`amplitudes()` at line 25 returns `&self.amps` → `&[Complex]` unchanged via `Deref`.)

- [ ] **Step 4: Update the allocation site**

In `crates/aleph-sv/src/backend.rs`:
- Ensure `AlignedBuf` is imported (add `use aleph_core::AlignedBuf;` near the `Complex` import).
- Lines 52–53:
  ```rust
  let mut amps = vec![Complex::new(0.0, 0.0); dim];
  amps[0] = Complex::new(1.0, 0.0);
  ```
  →
  ```rust
  let mut amps = AlignedBuf::<Complex>::zeroed(dim);
  amps[0] = Complex::new(1.0, 0.0);
  ```

- [ ] **Step 5: Fix any remaining `Vec`-specific usages**

Run: `cargo build -p aleph-sv 2>&1 | tail -30`
For each error about a method missing on `AlignedBuf` (e.g. `.push`, `.into_vec`, `.capacity`), inspect the call site:
- Slice-able reads/writes → already work via `Deref`/`DerefMut`.
- Need an owned `Vec` → `state.amps.to_vec()`.
- Construction `vec![…]` → `AlignedBuf::from_slice(&[…])` or `AlignedBuf::zeroed(n)`.
Expected after fixes: `aleph-sv` builds (SoA still on `Vec` until Task 4 — that's fine, they are independent fields).

- [ ] **Step 6: Run AoS tests**

Run: `cargo test -p aleph-sv --lib backend 2>&1 | tail -20` then
`cargo test -p aleph-sv state 2>&1 | tail -20`
Expected: PASS, including `allocated_state_is_cache_line_aligned` and `getters_match_construction`.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-sv/src/state.rs crates/aleph-sv/src/backend.rs
git commit -m "[P2-02] Back CpuState.amps with AlignedBuf<Complex>"
```

---

## Task 4: Wire `AlignedBuf` into the SoA path (`SoaState` + `SoaSvBackend`)

**Files:**
- Modify: `crates/aleph-sv/src/soa_state.rs:9-10` (fields), `:64-65,76-77,93-94` (fixtures)
- Modify: `crates/aleph-sv/src/soa_backend.rs:57,62` (allocate); add a test
- Test: inline tests in both files

- [ ] **Step 1: Add the failing alignment-assertion test**

In `crates/aleph-sv/src/soa_backend.rs`, inside the existing `#[cfg(test)] mod tests`, add:

```rust
#[test]
fn allocated_soa_state_is_cache_line_aligned() {
    let mut b = SoaSvBackend::new();
    let s = b.allocate(20).unwrap();
    assert_eq!(s.re().as_ptr() as usize % aleph_core::CACHE_LINE, 0);
    assert_eq!(s.im().as_ptr() as usize % aleph_core::CACHE_LINE, 0);
}
```

(Confirm `re()`/`im()` getter names from `soa_state.rs` — the file already exposes `re(&self) -> &[f64]`; add `im()` use likewise.)

- [ ] **Step 2: Run it to verify it compiles**

Run: `cargo test -p aleph-sv allocated_soa_state_is_cache_line_aligned 2>&1 | tail -15`
Expected: compiles; passes (alignment may be incidentally true pre-change — guard, not driver).

- [ ] **Step 3: Change the field types**

In `crates/aleph-sv/src/soa_state.rs`:
- Line 4: add `AlignedBuf` to the import: `use aleph_core::{AlignedBuf, Complex};`
- Lines 9–10:
  ```rust
  pub(crate) re: Vec<f64>,
  pub(crate) im: Vec<f64>,
  ```
  →
  ```rust
  pub(crate) re: AlignedBuf<f64>,
  pub(crate) im: AlignedBuf<f64>,
  ```
- Test fixtures at lines ~64–65, ~76–77, ~93–94: replace each `vec![…]` with `AlignedBuf::from_slice(&[…])`. Example:
  ```rust
  re: vec![0.5, -0.25],
  im: vec![0.0, 0.75],
  ```
  →
  ```rust
  re: AlignedBuf::from_slice(&[0.5, -0.25]),
  im: AlignedBuf::from_slice(&[0.0, 0.75]),
  ```

- [ ] **Step 4: Update the allocation site**

In `crates/aleph-sv/src/soa_backend.rs`:
- Add `use aleph_core::AlignedBuf;` near existing imports.
- Lines 57–62:
  ```rust
  let mut re = vec![0.0; dim];
  re[0] = 1.0;
  Ok(SoaState {
      num_qubits,
      re,
      im: vec![0.0; dim],
  })
  ```
  →
  ```rust
  let mut re = AlignedBuf::<f64>::zeroed(dim);
  re[0] = 1.0;
  Ok(SoaState {
      num_qubits,
      re,
      im: AlignedBuf::<f64>::zeroed(dim),
  })
  ```

- [ ] **Step 5: Fix any remaining `Vec`-specific usages**

Run: `cargo build -p aleph-sv 2>&1 | tail -30`
Resolve method-not-found errors as in Task 3 Step 5 (slice ops work via deref; owned needs `.to_vec()`; construction uses `from_slice`/`zeroed`). Pay attention to `measure_soa.rs` and `sampling.rs` — they read `re`/`im` as slices (fine), but check for any `.clone()` of the whole `Vec` (works — `AlignedBuf: Clone`) or capacity calls.
Expected: `aleph-sv` builds clean.

- [ ] **Step 6: Run SoA tests**

Run: `cargo test -p aleph-sv soa 2>&1 | tail -25`
Expected: PASS, including the new alignment assertion and the `soa_state` fixture tests.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-sv/src/soa_state.rs crates/aleph-sv/src/soa_backend.rs
git commit -m "[P2-02] Back SoaState.{re,im} with AlignedBuf<f64>"
```

---

## Task 5: Full-workspace correctness gate

**Files:** none (verification; fix-ups only if something breaks)

- [ ] **Step 1: Full test suite (the oracle + thread-sweep gate)**

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: all green, including `all_fixtures_match_naive` and the generated AoS≡SoA≡Naive oracle equivalence (1e-12). Because no amplitude arithmetic changed, any failure here is a wiring bug — fix it before proceeding.

- [ ] **Step 2: Thread-invariance sweep (bit-identical across thread counts)**

Run: `ALEPH_PAR_MIN_AMPS=0 RAYON_NUM_THREADS=4 cargo test --workspace 2>&1 | tail -15`
(Forcing the parallel path at small `n` exercises `par_blocks` over the aligned buffers.)
Expected: identical pass set; no nondeterminism.

- [ ] **Step 3: Lint + format across the workspace**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
Expected: no warnings, no diff.

- [ ] **Step 4: Commit (only if Step 1–3 required fix-ups)**

```bash
git add -A
git commit -m "[P2-02] Fix-ups from full-workspace correctness gate"
```

---

## Task 6: Bench, false-sharing audit, and report (EPYC)

> Runs on the bench server `ssh root@195.154.249.85` (EPYC 8124P, AVX-512).
> **MUST verify the box is idle first** (CLAUDE.md): `uptime` load ≈ 0 and
> `pgrep -af "cargo bench|bencher run|Runner.Worker"` returns nothing. Do NOT
> push to `benches/**` while measuring (it races the self-hosted CI runner).

**Files:**
- Create: `docs/perf/phase2-p2-02.md`

- [ ] **Step 1: Confirm idle box**

Run (on EPYC): `uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" || echo CLEAR`
Expected: load ≈ 0 and `CLEAR`. If not clean, wait / `systemctl restart` the runner per the Stage-0 ops notes, then re-check.

- [ ] **Step 2: Re-measure QFT-25 scaling vs P2-01 baseline**

On EPYC, build release and run the existing QFT scaling bench through the
optimized path at 1/8/16 threads (mirror the P2-01 procedure in
`docs/perf/phase2-p2-01.md` §2):
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --workspace
for t in 1 8 16; do RAYON_NUM_THREADS=$t taskset -c 0-$((t-1)) \
  cargo bench -p aleph-sv --bench qft_scaling -- --sample-size 10 2>&1 | tee /tmp/p2-02-t$t.txt; done
```
Record times + speedups. Expected: within criterion CI of the P2-01 numbers
(3.37×@8, 3.69×@16) — i.e. **flat**, confirming the bandwidth ceiling.

- [ ] **Step 3: `perf c2c` false-sharing audit (AC #2)**

```bash
RAYON_NUM_THREADS=16 perf c2c record -- \
  cargo run --release -p aleph-cli -- run <qft25.qasm> >/dev/null
perf c2c report --stdio 2>&1 | tee /tmp/p2-02-c2c.txt | head -60
```
Inspect the "Shared Data Cache Line Table" / HITM summary. Expected: no hot
shared cache lines attributable to the state buffer (confirming no false
sharing). Capture the headline counts for the report.

- [ ] **Step 4: Write the report**

Create `docs/perf/phase2-p2-02.md` covering, per spec §6:
- The audit: the three reasons there is no material false-sharing surface
  (large disjoint chunks; no parallel accumulators; large allocs already
  mmap-page-aligned), now backed by the `perf c2c` HITM evidence (Step 3).
- The alignment guarantee: `AlignedBuf` gives 64-byte alignment unconditionally
  (independent of allocator heuristics), aligning AoS SIMD units to the
  cache-line grid.
- Scaling result (Step 2): flat within noise vs P2-01, as expected for a
  bandwidth-bound kernel.
- Framing: the change is a robustness/guarantee + NUMA hook for P2-03, not a
  speedup. `#[repr(align(64))]` per-thread padding is deferred until a parallel
  accumulator exists (future parallel sampler).
- AVX-512 aligned-load switch: measured / left out (per spec §4); state the
  outcome.

- [ ] **Step 5: Tick BACKLOG acceptance criteria**

In `BACKLOG.md`, under `[P2-02]` (line ~1230), tick the three AC checkboxes
(`- [ ]` → `- [x]`), noting in the report that AC #3 is "measured flat, audit
shows nothing to improve."

- [ ] **Step 6: Commit**

```bash
git add docs/perf/phase2-p2-02.md BACKLOG.md
git commit -m "[P2-02] Scaling + false-sharing audit report (phase2-p2-02.md)"
```

---

## Task 7: Open the PR

**Files:** none

- [ ] **Step 1: Push the branch**

```bash
git push -u origin p2-02-cache-line-alignment
```

- [ ] **Step 2: Open the PR**

Title: `[P2-02] Cache-line-aligned state buffers`
Body MUST include:
- `Closes #28` (the **issue** number, not the PR number — CLAUDE.md).
- Approach summary (AlignedBuf, swapped into both state structs).
- Test results: `cargo test --workspace` green; miri on `aligned` (or note if
  unavailable); thread-sweep invariant; oracle equivalence preserved.
- `perf c2c` audit summary (no false sharing) + QFT-25 scaling numbers (flat).
- Honest framing: guarantee + NUMA hook, not a speedup; deferred per-thread
  padding.

```bash
gh pr create --title "[P2-02] Cache-line-aligned state buffers" --body "$(cat <<'EOF'
Closes #28

## Summary
Hand-rolled `AlignedBuf<T>` (aleph-core) guarantees 64-byte alignment for the
state-vector buffers; swapped into `CpuState.amps` and `SoaState.{re,im}`.

## Audit (false sharing)
No material false-sharing surface: large disjoint parallel chunks, no parallel
accumulators yet, and large allocs were already mmap-page-aligned. `perf c2c`
HITM report attached — no hot shared cache lines. See `docs/perf/phase2-p2-02.md`.

## Tests
- `cargo test --workspace` green; AoS≡SoA≡Naive oracle @1e-12 unchanged.
- miri on `aleph-core aligned` clean (alloc/dealloc/Drop/sentinel).
- Thread-sweep invariant (`ALEPH_PAR_MIN_AMPS=0`, 1/2/4/8 threads) bit-identical.

## Benchmark
QFT-25 scaling flat within CI vs P2-01 (3.37×@8, 3.69×@16) — bandwidth-bound,
nothing to gain from alignment, as predicted.

## Notes / follow-ups
Guarantee + P2-03 NUMA hook, not a speedup. Per-thread `#[repr(align(64))]`
padding deferred until a parallel accumulator exists. AVX-512 aligned-load
switch measured / left out (no win).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Verify CI is green**, then let it sit per the project's solo-merge discipline before squash-merging.

---

## Self-review notes

- **Spec coverage:** §2 AlignedBuf → Task 1; §2 miri → Task 2; §3 wiring → Tasks 3–4; §5 testing (unit/miri/oracle/thread-sweep/alignment asserts) → Tasks 1,2,3,4,5; §6 bench/audit/report → Task 6; §7 deferrals noted in report (Task 6 Step 4) + PR (Task 7); §8 PR/process → Task 7. All covered.
- **Type consistency:** `AlignedBuf<T>`, `CACHE_LINE`, `zeroed(len)`, `from_slice(&[T]) where T: Copy`, `as_ptr`/`len`/`is_empty` (via `Deref`), `to_vec()` (via `Deref`) used consistently across Tasks 1–4. `Clone` (bound `T: Copy`) backs the `#[derive(Clone)]` on both state structs; `Debug` (bound `T: Debug`) backs `#[derive(Debug)]`.
- **No placeholders:** every code/step is concrete. The one runtime-confirmed detail is the `NaiveSvBackend`/`SoaSvBackend` constructor name (`new()` vs `with_seed()`), flagged inline in Tasks 3/4 Step 1.
```
