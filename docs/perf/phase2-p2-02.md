# P2-02 — Cache-line-aligned state buffers & false-sharing audit

**Phase:** 2 (multi-threaded CPU)
**Issue:** #28 (P2-02)
**Date:** 2026-06-01
**Depends on:** P2-01 (rayon parallel gate application, `c494334`)
**Hardware:**
- **EPYC** — AMD EPYC 8124P (Siena), 16 physical / 32 SMT, single socket, 1 NUMA
  node, AVX-512, all-core frequency-throttled (~55%, see `phase2-p2-01.md` §3).
- **Ryzen** — AMD Ryzen 9 3900, 12 physical / 24 SMT, **no AVX-512** (scalar
  kernel path). Second box for a cross-path corroboration of the bandwidth wall.

Toolchain: Rust 1.95, `RUSTFLAGS="-C target-cpu=native"`, criterion release
builds, measured on **verified-idle** boxes (CLAUDE.md idle-check; load ≈ 0,
no competing `cargo bench`/runner jobs).

---

## 1. Summary

P2-02 swaps the state-vector backing store (`CpuState.amps`,
`SoaState.{re,im}`) from a plain `Vec` to a new 64-byte-aligned owned buffer,
`aleph_core::AlignedBuf<T>`, and audits the parallel kernels for false sharing.

**The audit's headline finding is that there is no material false-sharing
surface to remove**, for three independent reasons established below (§2). A
`perf c2c` run over a 16-thread QFT-25 confirms it directly: **28 shared cache
lines and 24 local HITM events across 230 k sampled records** — noise, not a
ping-pong pattern (§4). Accordingly the scaling numbers are **flat within noise
vs P2-01** on both boxes (§3), exactly as predicted.

The value P2-02 delivers is therefore a **guarantee, not a speedup**:

1. **Alignment is now guaranteed**, not incidental. The previous `Vec` was only
   8-byte aligned; large state vectors *happened* to be 64-byte (indeed page-)
   aligned because glibc serves >128 KiB allocations via `mmap`. `AlignedBuf`
   makes 64-byte alignment hold unconditionally — for small states, non-glibc
   allocators, and any future custom allocator.
2. **The AoS SIMD unit (`LANES = 4` complex = exactly 64 bytes) now sits on the
   cache-line grid**, so the only theoretically-shareable line — the boundary
   between two rayon tasks' contiguous chunks — is eliminated by construction.
3. **It is the allocation hook P2-03 (NUMA-aware allocation) needs.** A
   hand-rolled owned buffer is where `numa_alloc` / first-touch placement will
   live; the `aligned-vec` crate would not have composed with that.

`AlignedBuf` is ~160 lines with one contained `unsafe` allocator; it is validated
by unit tests, **miri (including `-Zmiri-strict-provenance`)**, and the unchanged
AoS≡SoA≡Naive oracle suite (1e-12) plus the P2-01 thread-invariance sweep.

---

## 2. The audit — why there is no false sharing to remove

False sharing requires two threads to write the *same* 64-byte cache line. In
this codebase that essentially cannot happen on the hot path:

1. **State-vector writes are large, contiguous, disjoint chunks.** The only
   rayon usage in the workspace is `par_blocks`/`par_units` (`kernels/mod.rs`),
   which splits the block index with `into_par_iter().with_min_len(64)`. Each
   task owns ≥ 64 consecutive SIMD units; one unit is `LANES = 4` complex = 64
   bytes = one cache line. Tasks write pairwise-disjoint ranges, so the only
   line two threads could share is the single boundary line between adjacent
   tasks (~1 per split, out of tens of millions). With `AlignedBuf` even that
   line is eliminated, because the buffer base is on the cache-line grid.

2. **There are no per-thread accumulators.** The BACKLOG brief targets "sample
   counts, statistics" — but `measure.rs`, `measure_soa.rs`, and `sampling.rs`
   are still sequential. There is no `fold`/`reduce`/atomic/`Mutex` in any
   parallel path, hence nothing to pad with `#[repr(align(64))]` (deferred, §6).

3. **The scaling ceiling is bandwidth, not contention.** P2-01 already
   established (`phase2-p2-01.md` §5) that the 8→16-core plateau is
   memory-bandwidth saturation. A contention fix cannot move a bandwidth-bound
   number.

---

## 3. Scaling — flat vs P2-01 (as predicted)

### EPYC (AVX-512), QFT-25, `qft_scaling` + `qft_scaling_fused`, idle box

| Threads | Time (raw) | Speedup vs T1 | P2-01 (reference) |
|--------:|-----------:|--------------:|------------------:|
| 1  | 7.90 s | 1.00× | (8.41 s) |
| 8  | 2.39 s | **3.31×** | 3.37× |
| 16 | 2.18 s | **3.62×** | 3.69× |

The 8× and 16× speedup ratios are statistically identical to P2-01's clean
numbers (3.37× / 3.69×) — alignment did not change scaling. The single-thread
time differs (~7.90 s vs 8.41 s) within run-to-run / session variance; we do not
claim that as an alignment win. Fused (`run_optimized`) measures identically to
raw (7.90/2.39/2.18 s), consistent with P2-01's finding that the QFT
controlled-phase ladder does not fuse.

### Ryzen (scalar, no AVX-512), QFT-25 + QFT-22

| Threads | QFT-25 time | QFT-25 speedup | QFT-22 time | QFT-22 speedup |
|--------:|------------:|---------------:|------------:|---------------:|
| 1  | 12.64 s | 1.00× | 1.292 s | 1.00× |
| 8  | 6.00 s | **2.11×** | 0.405 s | 3.18× |
| 12 | 6.02 s | **2.10×** | 0.323 s | 3.99× |

QFT-25 hits **2.11×@8** — identical to P2-01's documented Ryzen scalar number,
confirming alignment changed nothing on the scalar path either — then **plateaus
at 12 threads** (6.00 → 6.02 s, no gain) — the same memory-bandwidth wall seen on
EPYC, now reproduced on a second box and a second code path. The smaller QFT-22
(state fits cache far better, so less bandwidth-bound) keeps scaling to 3.99×@12,
which sharpens the diagnosis: the QFT-25 plateau is bandwidth, not a
parallelization defect. Fused (`run_optimized`) == raw here too.

---

## 4. False-sharing audit — `perf c2c` (AC #2)

`perf c2c record` over a single 16-thread QFT-25 run through `run_optimized`
(the `oneshot` bin on `scripts/qiskit-baseline/circuits/qft_n25.qasm`), EPYC:

```
Total records                     : 230469
Load Operations                   :  45612
Load Local HITM                   :     24      <- false-sharing signal
Load Remote HITM                  :      4
Total Shared Cache Lines          :     28
Store HITs on shared lines        :      4
```

**Interpretation:** 28 shared cache lines and 24+4 HITM events across an entire
33 M-amplitude (512 MiB) simulation is noise. A workload with real false sharing
shows thousands–millions of HITMs concentrated on a *few* hot lines; here the
Shared Data Cache Line Table shows every shared line with exactly **1** HITM
(3.57% each, scattered addresses — not a contiguous amplitude region). These are
incidental cross-core touches (rayon work-stealing deque metadata, allocator
bookkeeping, the odd boundary line), not a ping-pong pattern on the amplitude
array. **No false-sharing pattern is identified — AC #2 satisfied.**

---

## 5. Implementation notes — `AlignedBuf<T>`

- `crates/aleph-core/src/aligned.rs`. `zeroed(len)` (zero-init via `alloc_zeroed`
  — `f64`/`Complex` are zero-valid) and `from_slice(&[T])` for `T: Copy`.
- `Deref`/`DerefMut` to `[T]` — all existing slice/index/iter call sites and
  kernel pointers are unchanged; only the two `allocate()` sites and test
  fixtures touch the new constructors.
- Fixed-size (state vectors never resize). `Drop` frees with the identical
  `Layout`; element destructors are intentionally not run, and the POD contract
  is **machine-enforced** by `const { assert!(!needs_drop::<T>()) }` +
  `const { assert!(size_of::<T>() != 0) }` guards in both constructors.
- `len == 0` uses `NonNull::dangling()` (provenance-clean; verified under
  `-Zmiri-strict-provenance`), never dereferenced.
- AVX-512 aligned `load`/`store` (`_mm512_load_pd` vs `loadu`) was left **out of
  scope**: it requires proving every unit offset is 64-aligned (UB otherwise)
  for ≈0 benefit on modern x86 when accesses don't cross a line. Kernels keep
  `loadu`.

---

## 6. Acceptance criteria

- **AC #1 — Audit complete, padding applied where needed.** ✅ Audit in §2;
  alignment guaranteed via `AlignedBuf`. Per-thread `#[repr(align(64))]` padding
  is **deferred** — there is no parallel accumulator to contend on yet; it will
  land with the future parallel sampler.
- **AC #2 — No false-sharing patterns identified by perf tools.** ✅ `perf c2c`
  (§4): 28 shared lines / 24 local HITM, no hot line.
- **AC #3 — Scaling efficiency improves vs P2-01.** Measured **flat within
  noise** (§3) — honestly, there was no contention to remove (bandwidth-bound).
  The substantive deliverable is the alignment guarantee + the audit evidence,
  not a speedup.

---

## 7. Follow-ups

- **P2-03** — NUMA-aware allocation, building on `AlignedBuf` (first-touch /
  `numa_alloc`).
- Per-thread accumulator padding once a parallel sampler/statistics path exists.
- AVX-512 aligned load/store — revisit only if a micro-bench shows a real win.
