# P2-01 — Rayon Parallel Gate Application — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the state-vector gate kernels (AoS + SoA, 1q/2q, 3q if cheap) across CPU cores with rayon, gated by a runtime amplitude threshold, with bit-identical results.

**Architecture:** Introduce one shared `par_blocks` driver in `kernels/mod.rs` that sequences the already-disjoint outer block walk either sequentially (below threshold) or via rayon (above it). Each kernel keeps its SIMD inner walk verbatim; only its outer driver loop is replaced by a `par_blocks(...)` call. A `BlockPtr(*mut f64)` Send+Sync wrapper carries the raw write pointer across threads; disjoint blocks make concurrent writes sound. The threshold is read once from `ALEPH_PAR_MIN_AMPS` (default `1<<18`), letting tests force the parallel path at small n.

**Tech Stack:** Rust 1.89, rayon, AVX-512 intrinsics (`core::arch::x86_64`), criterion, existing `aleph-oracle` equivalence harness.

**Design doc:** `docs/superpowers/specs/2026-06-01-p2-01-rayon-parallel-design.md`

---

## File Structure

- `Cargo.toml` (workspace root) — add `rayon` to `[workspace.dependencies]`.
- `crates/aleph-sv/Cargo.toml` — depend on workspace `rayon`.
- `crates/aleph-sv/src/kernels/mod.rs` — **new** `par_blocks` driver, `BlockPtr` wrapper, `par_min_amps()` threshold reader, and unit tests. This is the only genuinely new code; everything else is mechanical site conversion.
- `crates/aleph-sv/src/kernels/aos.rs` — convert ~25 driver loops to `par_blocks`.
- `crates/aleph-sv/src/kernels/soa.rs` — convert ~15 driver loops to `par_blocks`.
- `crates/aleph-oracle/tests/soa_vs_naive.rs` — already the equivalence workhorse; reused unchanged, exercised under a thread/threshold sweep (Task 8).
- `benches/` — QFT-25 parallel-scaling bench (Task 9).

### The canonical transform (referenced by every conversion task)

Every kernel today ends in **one of two** driver shapes wrapping a `let outer_iter = |block: usize| { ... };` closure.

**Shape U (uncontrolled, strided):**
```rust
let amps_ptr = amps.as_mut_ptr() as *mut f64;
let outer_iter = |block: usize| { /* SIMD inner walk, uses amps_ptr */ };
if controls.is_empty() {
    let outer_step = target_bit << 1;
    let mut block = 0usize;
    while block < len { outer_iter(block); block += outer_step; }
    return;
}
```

**Shape C (controlled, expand_with_fixed):**
```rust
let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
for k in 0..outer_count {
    let block = crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1);
    outer_iter(block);
}
```

**After conversion**, both become a `par_blocks` call. Two edits per kernel:

1. Replace `let amps_ptr = amps.as_mut_ptr() as *mut f64;` with
   `let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);`
   and make `outer_iter`'s first line `let amps_ptr = bp.ptr();`.
   (SoA: wrap both — `let re_bp = BlockPtr(re.as_mut_ptr()); let im_bp = BlockPtr(im.as_mut_ptr());` and rederive both inside via `re_bp.ptr()` / `im_bp.ptr()`.)

   **CRITICAL (proven on EPYC in Task 1):** extract the pointer through
   the `bp.ptr()` `&self` accessor, NEVER a direct `bp.0` field read.
   Rust 2021 disjoint closure capture would grab the bare `*mut f64`
   field (`!Sync`) and the closure fails `par_blocks`' `Fn + Sync` bound;
   `bp.ptr()` forces whole-`BlockPtr` capture (`&BlockPtr: Sync`).

2. Replace the driver loop:

   Shape U →
   ```rust
   if controls.is_empty() {
       let outer_step = target_bit << 1;
       let count = len / outer_step;
       crate::kernels::par_blocks(count, len, |k| k * outer_step, outer_iter);
       return;
   }
   ```
   Shape C →
   ```rust
   crate::kernels::par_blocks(
       outer_count,
       len,
       |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
       outer_iter,
   );
   ```

`outer_iter` is now passed by value (it is `Copy`-free but `Fn(usize) + Sync` — its only captures are `BlockPtr` (Send+Sync), `__m512d` broadcasts (plain SIMD value types, Send+Sync), and `Copy` scalars). `block_of` captures `&fixed_above` (a `SmallVec` of POD — Sync) by reference and `Copy` scalars; it is `Fn(usize) + Sync`.

**Why this is sound (state in every conversion commit message):** distinct `block` bases address pairwise-disjoint amplitude sets (the P1 SIMD-kernel invariant `block | offsets | j` has disjoint bit-fields), so parallel writes never race. AVX-512 intrinsics inside `outer_iter` run in `unsafe` blocks whose feature precondition (`avx512f`) was checked by the dispatch `is_x86_feature_detected!` before the kernel was called; that holds machine-wide, so executing the closure on a rayon worker thread is as sound as the sequential call it replaces.

---

## Task 1: Parallel driver infrastructure + first kernel (proof)

**Files:**
- Modify: `Cargo.toml` (workspace root, `[workspace.dependencies]`)
- Modify: `crates/aleph-sv/Cargo.toml`
- Modify: `crates/aleph-sv/src/kernels/mod.rs`
- Modify: `crates/aleph-sv/src/kernels/aos.rs:247-369` (`apply_1q_avx512`)

- [ ] **Step 1: Add rayon dependency**

In root `Cargo.toml` under `[workspace.dependencies]` add:
```toml
rayon = "1.10"
```
In `crates/aleph-sv/Cargo.toml` under `[dependencies]` add:
```toml
rayon = { workspace = true }
```

- [ ] **Step 2: Run to confirm it resolves**

Run: `cargo build -p aleph-sv`
Expected: builds clean (rayon downloaded, no usage yet).

- [ ] **Step 3: Write the failing unit test for `par_blocks`**

Append to `crates/aleph-sv/src/kernels/mod.rs` (inside or add a `#[cfg(test)] mod par_tests`):
```rust
#[cfg(test)]
mod par_tests {
    use super::{par_blocks, BlockPtr};

    // par_blocks must touch every block base exactly once, whether it
    // runs sequentially or in parallel. We write k into slot block_of(k)
    // and assert the full permutation came back.
    fn run_once(count: usize, force_parallel: bool) -> Vec<usize> {
        // SAFETY env override drives the threshold; len=0 path uses the
        // explicit `len` arg only for the threshold compare.
        let mut out = vec![usize::MAX; count];
        let bp = BlockPtr(out.as_mut_ptr() as *mut f64); // reuse wrapper for Send
        let len = if force_parallel { usize::MAX } else { 0 };
        par_blocks(count, len, |k| k, move |slot| {
            // SAFETY: slots are the identity permutation of 0..count,
            // disjoint, in bounds; we round-trip through the f64 ptr only
            // to reuse BlockPtr — write the usize via a typed ptr.
            let p = bp.0 as *mut usize;
            unsafe { *p.add(slot) = slot };
        });
        out
    }

    #[test]
    fn par_blocks_covers_all_sequential() {
        let v = run_once(1000, false);
        assert!(v.iter().enumerate().all(|(i, &x)| x == i));
    }

    #[test]
    fn par_blocks_covers_all_parallel() {
        let v = run_once(1000, true);
        assert!(v.iter().enumerate().all(|(i, &x)| x == i));
    }
}
```

- [ ] **Step 4: Run it, expect a compile failure (symbols undefined)**

Run: `cargo test -p aleph-sv par_blocks 2>&1 | head -20`
Expected: FAIL — `cannot find function par_blocks` / `cannot find struct BlockPtr`.

- [ ] **Step 5: Implement the driver in `kernels/mod.rs`**

Add near the top of `kernels/mod.rs` (after the existing `use`/module decls):
```rust
use std::sync::OnceLock;

/// Raw write pointer shareable across rayon worker threads.
///
/// SAFETY: the only constructor sites hand this to `par_blocks`, whose
/// `block_of` produces pairwise-disjoint block bases; each parallel task
/// writes a disjoint amplitude range, so no two threads ever touch the
/// same byte. The pointer therefore behaves as a set of disjoint
/// `&mut` slices, which is `Send + Sync`.
#[derive(Clone, Copy)]
pub(crate) struct BlockPtr(pub(crate) *mut f64);
unsafe impl Send for BlockPtr {}
unsafe impl Sync for BlockPtr {}

/// Minimum state-vector length (in amplitudes) before gate kernels go
/// parallel. Below this, rayon's task overhead outweighs the win and the
/// kernel runs sequentially (keeping small circuits and unit tests fast
/// and trivially deterministic). Overridable via `ALEPH_PAR_MIN_AMPS`
/// (read once) — tests set it to 0 to force the parallel path at small n;
/// it is also the knob P2-04 will tune.
fn par_min_amps() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ALEPH_PAR_MIN_AMPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1usize << 18)
    })
}

/// Run `body(block_of(k))` for every `k` in `0..count`.
///
/// Sequential when the state vector (`len` amplitudes) is below
/// `par_min_amps()`, otherwise rayon-parallel. `block_of` maps a block
/// index to its base amplitude; the bases MUST be pairwise-disjoint
/// blocks (the SIMD-kernel invariant) so parallel `body` calls never
/// race. The result is bit-identical across thread counts: each block
/// writes disjoint memory with no cross-thread floating-point reduction.
pub(crate) fn par_blocks(
    count: usize,
    len: usize,
    block_of: impl Fn(usize) -> usize + Sync,
    body: impl Fn(usize) + Sync,
) {
    if len < par_min_amps() {
        for k in 0..count {
            body(block_of(k));
        }
    } else {
        use rayon::prelude::*;
        // with_min_len keeps fine-grained (low-target) kernels from
        // drowning in per-element task overhead: each task runs a
        // contiguous batch of blocks sequentially.
        (0..count)
            .into_par_iter()
            .with_min_len(64)
            .for_each(|k| body(block_of(k)));
    }
}
```

- [ ] **Step 6: Run the unit tests, expect PASS**

Run: `cargo test -p aleph-sv par_blocks`
Expected: both `par_blocks_covers_all_sequential` and `_parallel` PASS.

- [ ] **Step 7: Convert `apply_1q_avx512` (the proof kernel)**

In `crates/aleph-sv/src/kernels/aos.rs`, apply the canonical transform to `apply_1q_avx512` (lines ~284, ~333-341, ~365-368):
  - Change `let amps_ptr = amps.as_mut_ptr() as *mut f64;` → `let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);`
  - Make `outer_iter`'s body start with `let amps_ptr = bp.0;`
  - Replace the uncontrolled `while block < len { … }` block with:
    ```rust
    if controls.is_empty() {
        let outer_step = target_bit << 1;
        let count = len / outer_step;
        crate::kernels::par_blocks(count, len, |k| k * outer_step, outer_iter);
        return;
    }
    ```
  - Replace the controlled `for k in 0..outer_count { … }` with:
    ```rust
    crate::kernels::par_blocks(
        outer_count,
        len,
        |k| crate::kernels::expand_with_fixed(k, &fixed_above) << (target + 1),
        outer_iter,
    );
    ```

- [ ] **Step 8: Compile — confirm the target_feature closure crosses the rayon boundary**

Run: `cargo build -p aleph-sv`
Expected: builds clean. If it fails with a `Send`/`Sync`/target-feature error on the closure, apply the documented fallback: extract `outer_iter`'s body into a `#[cfg(target_arch="x86_64")] #[target_feature(enable="avx512f")] unsafe fn apply_1q_avx512_block(bp: BlockPtr, block: usize, /* broadcasts as args */)` and have `outer_iter` be `|block| unsafe { apply_1q_avx512_block(bp, block, …) }`. Record which path was needed in the commit message (it determines Tasks 2–7).

- [ ] **Step 9: Equivalence still holds (sequential, default threshold)**

Run: `cargo test -p aleph-oracle --test soa_vs_naive`
Expected: PASS (n≤10 fixtures run sequentially; conversion must not change results).

- [ ] **Step 10: Force the parallel path at small n and re-run**

Run: `ALEPH_PAR_MIN_AMPS=0 cargo test -p aleph-oracle --test soa_vs_naive`
Expected: PASS — parallel `apply_1q_avx512` produces identical state vectors.

- [ ] **Step 11: EPYC spike — confirm it runs on real AVX-512 hardware**

On the bench server (`ssh root@195.154.249.85`), build+run the same two test invocations (`RAYON_NUM_THREADS=8 ALEPH_PAR_MIN_AMPS=0 cargo test -p aleph-oracle --test soa_vs_naive`).
Expected: PASS. This proves the inherited-target_feature closure executes correctly on rayon worker threads on the production CPU before we replicate the pattern across 40 sites.

- [ ] **Step 12: Commit**

```bash
git add Cargo.toml crates/aleph-sv/Cargo.toml crates/aleph-sv/src/kernels/mod.rs crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P2-01] par_blocks driver + first parallel kernel (apply_1q_avx512)

BlockPtr Send+Sync wrapper + env-gated threshold (ALEPH_PAR_MIN_AMPS,
default 1<<18). Disjoint outer blocks -> bit-identical across threads.
Proven on EPYC under RAYON_NUM_THREADS=8 ALEPH_PAR_MIN_AMPS=0."
```

---

## Task 2: Convert remaining AoS 1q kernels

**Files:** Modify `crates/aleph-sv/src/kernels/aos.rs`

Apply the **canonical transform** (verbatim recipe from File Structure section, same as Task 1 Step 7) to each site below. For the `_lowbit` kernels the driver is Shape U with `outer_step = LANES` (i.e. `block += LANES`) → `count = len / LANES`, `block_of = |k| k * LANES`.

| Fn | ~line | Shapes present |
|----|-------|----------------|
| `apply_1q_diagonal_avx512` | 518 | U (590) + C (611) |
| `apply_1q_x_avx512` | 632 | U (669) + C (685) |
| `apply_1q_y_avx512` | 708 | U (782) + C (800) |
| `apply_1q_antidiag_avx512` | 821 | U (874) + C (889) |
| `apply_1q_x_avx512_lowbit` | 917 | U-LANES (967) |
| `apply_1q_y_avx512_lowbit` | 1006 | U-LANES (1079) |
| `apply_1q_antidiag_avx512_lowbit` | 1108 | U-LANES (1176) |

- [ ] **Step 1: Convert all 7 fns** using the canonical recipe. Lowbit kernels: `outer_step = LANES`.

- [ ] **Step 2: Build**

Run: `cargo build -p aleph-sv`
Expected: clean.

- [ ] **Step 3: Equivalence, default + forced-parallel**

Run: `cargo test -p aleph-oracle --test soa_vs_naive`
Run: `ALEPH_PAR_MIN_AMPS=0 cargo test -p aleph-oracle --test soa_vs_naive`
Expected: both PASS.

- [ ] **Step 4: Clippy**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P2-01] Parallelize remaining AoS 1q kernels (diag/x/y/antidiag + lowbit)"
```

---

## Task 3: Convert AoS 2q kernels

**Files:** Modify `crates/aleph-sv/src/kernels/aos.rs`

All AoS 2q AVX-512 kernels use **Shape C** (`for k in 0..outer_count { … expand_with_fixed … }`). The `block_of` expression is whatever currently sits inside that loop (copy it verbatim into the `par_blocks` closure); the shift/fixed naming differs per kernel — preserve each kernel's own local variables.

| Fn | ~line | driver line |
|----|-------|-------------|
| `apply_2q_avx512` (dense) | 1447 | 1579 |
| `apply_2q_cnot_avx512` | 1632 | 1698 |
| `apply_2q_cnot_avx512_tier_b` | 1748 | 1795 |
| `apply_2q_cnot_avx512_tier_c` | 1852 | 1897 |
| `apply_2q_swap_avx512` | 2014 | 2088 |
| `apply_2q_swap_avx512_tier_b` | 2143 | 2203 |
| `apply_2q_swap_avx512_tier_c` | 2263 | 2301 |
| `apply_2q_cz_avx512` | 2413 | 2462 |
| `apply_2q_diagonal_avx512` | 2526 | 2619 |

Recipe per fn: wrap the write pointer in `BlockPtr`, rederive inside `outer_iter`, and replace
```rust
for k in 0..outer_count { let block = <expr>; outer_iter(block); }
```
with
```rust
crate::kernels::par_blocks(outer_count, len, |k| <expr>, outer_iter);
```
(`len` = `amps.len()`; bind it if not already in scope.)

- [ ] **Step 1: Convert all 9 fns.**

- [ ] **Step 2: Build** — `cargo build -p aleph-sv` — clean.

- [ ] **Step 3: Equivalence** — both default and `ALEPH_PAR_MIN_AMPS=0` runs of `cargo test -p aleph-oracle --test soa_vs_naive` PASS.

- [ ] **Step 4: Clippy** — `cargo clippy -p aleph-sv --all-targets -- -D warnings` clean.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P2-01] Parallelize AoS 2q kernels (dense/cnot/swap/cz/diagonal, all tiers)"
```

---

## Task 4: Convert AoS 3q kernels (if cheap — same recipe)

**Files:** Modify `crates/aleph-sv/src/kernels/aos.rs`

The 3q Toffoli/CCZ outer-walk kernels and the `apply_3q` dispatch arms (driver loops at lines 2859, 3255, and the `for k in 0..outer_count` arms around 3969/4094/4229/4300/4462/4622/4697/4935/5125) follow Shape C. Convert with the identical recipe. **Tier-B0/B1 in-register kernels** (e.g. `apply_toffoli_avx512_tier_b0`, line 2943) have no outer block loop (they process the whole register set in one pass) — **leave those sequential**; they only fire at tiny qubit counts below the threshold anyway.

- [ ] **Step 1: Convert each outer-walk 3q kernel / dispatch arm** that has a `for k in 0..outer_count` driver. Skip the in-register tier-b kernels.

- [ ] **Step 2: Build** — `cargo build -p aleph-sv` — clean.

- [ ] **Step 3: Equivalence** — default + `ALEPH_PAR_MIN_AMPS=0` runs PASS (the `kernel_ccx`, `kernel_ccz`, `mcx_*` fixtures cover these).

- [ ] **Step 4: Clippy** — clean.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P2-01] Parallelize AoS 3q outer-walk kernels (toffoli/ccz/mcx)"
```

---

## Task 5: Convert SoA 1q kernels

**Files:** Modify `crates/aleph-sv/src/kernels/soa.rs`

SoA kernels carry **two** streams. Use two `BlockPtr`s: `let re_bp = crate::kernels::BlockPtr(re.as_mut_ptr()); let im_bp = crate::kernels::BlockPtr(im.as_mut_ptr());` and rederive both at the top of `outer_iter` (`let re_ptr = re_bp.0; let im_ptr = im_bp.0;`). `len = re.len()`. Then convert the driver exactly as the AoS recipe.

| Fn | ~line | Shapes |
|----|-------|--------|
| `apply_1q_x_soa_avx512` | 2275 | U (2309) + C (2324) |
| `apply_1q_y_soa_avx512` | 2342 | U (2397) + C (2412) |
| `apply_1q_antidiag_soa_avx512` | 2432 | U (2490) + C (2505) |
| `apply_1q_x_soa_avx512_lowbit` | 2540 | U-LANES (2591) |

- [ ] **Step 1: Convert all 4 fns** (two-pointer variant of the recipe).
- [ ] **Step 2: Build** — `cargo build -p aleph-sv` clean.
- [ ] **Step 3: Equivalence** — default + `ALEPH_PAR_MIN_AMPS=0` PASS.
- [ ] **Step 4: Clippy** — clean.
- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/soa.rs
git commit -m "[P2-01] Parallelize SoA 1q kernels (x/y/antidiag + lowbit)"
```

---

## Task 6: Convert SoA 2q kernels

**Files:** Modify `crates/aleph-sv/src/kernels/soa.rs`

Two-pointer recipe, Shape C, for each:

| Fn | ~line | driver |
|----|-------|--------|
| `apply_2q_cnot_avx512` | 941 | 1002 |
| `apply_2q_cnot_avx512_tier_b` | 1055 | 1103 |
| `apply_2q_cnot_avx512_tier_c` | 1172 | 1222 |
| `apply_2q_swap_avx512` | 1262 | 1325 |
| `apply_2q_swap_avx512_tier_b` | 1383 | 1443 |
| `apply_2q_swap_avx512_tier_c` | 1513 | 1560 |
| `apply_2q_cz_avx512` | 1591 | 1634 |
| `apply_2q_diagonal_avx512` | 1685 | 1761 |

- [ ] **Step 1: Convert all 8 fns.**
- [ ] **Step 2: Build** — clean.
- [ ] **Step 3: Equivalence** — default + forced-parallel PASS.
- [ ] **Step 4: Clippy** — clean.
- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/soa.rs
git commit -m "[P2-01] Parallelize SoA 2q kernels (cnot/swap/cz/diagonal, all tiers)"
```

---

## Task 7: Convert SoA 3q kernels (if cheap)

**Files:** Modify `crates/aleph-sv/src/kernels/soa.rs`

SoA 3q outer-walk kernels/dispatch arms (drivers around 255/541/3117/3334/3386 and the `apply_toffoli_avx512_tier_a_outer_walk_soa` / `apply_ccz_avx512_tier_a_outer_walk_soa`). Same as Task 4: convert outer-walk Shape-C drivers, leave in-register tier-b kernels sequential.

- [ ] **Step 1: Convert each SoA 3q outer-walk driver.**
- [ ] **Step 2: Build** — clean.
- [ ] **Step 3: Equivalence** — default + forced-parallel PASS.
- [ ] **Step 4: Clippy** — clean.
- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/soa.rs
git commit -m "[P2-01] Parallelize SoA 3q outer-walk kernels"
```

---

## Task 8: Thread-count-invariance verification

**Files:** Create `scripts/p2-01-thread-sweep.sh`

- [ ] **Step 1: Write the sweep script**

Create `scripts/p2-01-thread-sweep.sh`:
```bash
#!/usr/bin/env bash
# P2-01: prove gate kernels are thread-count invariant. Forces the
# parallel path at small n (ALEPH_PAR_MIN_AMPS=0) and runs the SoA≡AoS≡
# Naive equivalence workhorse across thread counts. Bit-identical output
# means every count passes the 1e-12 oracle.
set -euo pipefail
for t in 1 2 4 8; do
  echo "== RAYON_NUM_THREADS=$t =="
  RAYON_NUM_THREADS=$t ALEPH_PAR_MIN_AMPS=0 \
    cargo test -p aleph-oracle --test soa_vs_naive -- --nocapture
done
echo "All thread counts agree within 1e-12."
```
Make it executable: `chmod +x scripts/p2-01-thread-sweep.sh`.

- [ ] **Step 2: Run locally**

Run: `./scripts/p2-01-thread-sweep.sh`
Expected: all four thread counts PASS. (On non-AVX-512 hosts this exercises the scalar dispatch path; still valid for the rayon driver itself.)

- [ ] **Step 3: Run on EPYC**

On `ssh root@195.154.249.85`: run the same script.
Expected: all PASS on real AVX-512 — the authoritative correctness gate.

- [ ] **Step 4: Commit**

```bash
git add scripts/p2-01-thread-sweep.sh
git commit -m "[P2-01] Thread-count-invariance sweep script (1/2/4/8 threads, forced parallel)"
```

---

## Task 9: QFT-25 parallel scaling benchmark

**Files:** Modify the QFT bench under `benches/` (find with `ls benches/`); add a parallel-scaling variant or document the env-driven measurement.

- [ ] **Step 1: Locate the QFT bench**

Run: `ls benches/ && grep -rln "qft" benches/`
Expected: identifies the existing QFT criterion bench.

- [ ] **Step 2: Establish the 1-core baseline on EPYC**

On EPYC, build release with native target and run QFT at n=25, pinned to 1 thread:
```bash
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench --bench qft -- qft/25 --save-baseline p2_01_t1
```
(Use the bench's actual id; `cargo bench --bench qft -- --list` to confirm.)

- [ ] **Step 3: Measure at 8 cores**

```bash
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=8 \
  cargo bench --bench qft -- qft/25 --baseline p2_01_t1
```
Expected: criterion reports the speedup. **AC gate: ≥6× faster than the `p2_01_t1` baseline.**

- [ ] **Step 4: Also record 16 cores (for the ROADMAP figure, not gating here)**

```bash
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=16 \
  cargo bench --bench qft -- qft/25 --baseline p2_01_t1
```
Record the number for the PR body (formal ≥12×/16-core validation is P2-05).

- [ ] **Step 5: If <6× at 8 cores — tune before proceeding**

Sweep `ALEPH_PAR_MIN_AMPS` and the `with_min_len` grain (try 32/64/128/256), re-measure. If still short, profile with `perf stat` on EPYC for memory-bandwidth saturation (expected ceiling per ADR 0008) and document the finding. Do not lower the AC; record the gap and file a follow-up if bandwidth-bound.

- [ ] **Step 6: Commit any bench additions**

```bash
git add benches/
git commit -m "[P2-01] QFT-25 parallel-scaling bench; EPYC 8-core >=6x recorded"
```

---

## Task 10: Tune threshold, full gate, PR

**Files:** `crates/aleph-sv/src/kernels/mod.rs` (final `PAR_MIN_AMPS` default), PR.

- [ ] **Step 1: Lock the production default**

Set the `unwrap_or(1usize << 18)` default in `par_min_amps()` to the value that gave the best QFT-25 scaling without regressing small-n latency (confirm small-n: the existing micro-benches must not regress vs. main — run `cargo bench -p aleph-sv` style micro if present, or a quick n=14 QFT). Document the chosen value in the doc comment.

- [ ] **Step 2: Full workspace gate**

Run: `cargo test --workspace`
Run: `cargo clippy --workspace --all-targets -- -D warnings`
Run: `cargo fmt --check`
Expected: all clean.

- [ ] **Step 3: Final EPYC confirmation**

On EPYC: `./scripts/p2-01-thread-sweep.sh` PASS, and re-confirm QFT-25 8-core ≥6×.

- [ ] **Step 4: Open the PR**

```bash
git push -u origin p2-01-rayon-parallel
gh pr create --title "[P2-01] Rayon-based parallel gate application" --body "$(cat <<'EOF'
Closes #<P2-01 issue number — look up in GitHub, do NOT use the PR number>

## Approach
One shared `par_blocks` driver in `kernels/mod.rs` parallelizes the
already-disjoint outer block walk of every AoS+SoA SIMD kernel. Kernel
inner walks unchanged; only drivers converted. `BlockPtr` Send+Sync
carries the write pointer; disjoint blocks make concurrent writes sound.
Threshold `ALEPH_PAR_MIN_AMPS` (default 1<<18) keeps small circuits
sequential.

## Correctness
- SoA≡AoS≡Naive oracle harness PASS at default threshold.
- Thread-count-invariance: `scripts/p2-01-thread-sweep.sh` PASS for
  RAYON_NUM_THREADS ∈ {1,2,4,8} with ALEPH_PAR_MIN_AMPS=0 (bit-identical,
  no FP reduction) — on EPYC.
- `cargo test --workspace`, clippy, fmt all green.

## Benchmarks (EPYC)
- QFT-25, 8 cores vs 1 core: <X>× (AC ≥6×) ✓
- QFT-25, 16 cores vs 1 core: <Y>× (ROADMAP ≥12× context; formal exit in P2-05)

## Out of scope
P2-02 (false-sharing padding), P2-03 (NUMA), P2-04 (chunk tuning),
P2-05 (scaling report), measurement/sampling parallelism.
EOF
)"
```

- [ ] **Step 5: Self-review the diff**

Run: `git diff main...HEAD --stat` then re-read the kernel conversions with fresh eyes — confirm every converted site rederives the pointer inside `outer_iter` and passes the correct `len`/`count`/`block_of`. Let the PR sit, re-review, then merge once CI is green.

---

## Self-Review Notes (plan vs. spec)

- **Spec "parallel 1q+2q AoS+SoA"** → Tasks 1,2,3 (AoS 1q/2q), 5,6 (SoA 1q/2q). ✓
- **Spec "3q if cheap"** → Tasks 4,7, explicitly gated as same-recipe / skip in-register tiers. ✓
- **Spec "threshold-gated, no feature flag, RAYON_NUM_THREADS lever"** → Task 1 `par_min_amps()` + `par_blocks`. ✓
- **Spec "thread-count-invariant oracle equivalence {1,2,4,8}"** → Task 8 sweep. ✓
- **Spec "QFT-25 ≥6× at 8 cores on EPYC"** → Task 9. ✓
- **Spec "zero regressions, CI green"** → Task 10. ✓
- **Risk flagged:** target_feature-closure-across-rayon soundness/compile — proven in Task 1 (Step 8 fallback documented) before the 40-site sweep, matching the P1 "EPYC-validate per AVX-512 group" discipline.
