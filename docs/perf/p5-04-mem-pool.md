# P5-04 — GPU memory pool

A retaining device memory pool so repeated state allocation (the "many small
circuits" workload) reuses freed GPU blocks instead of round-tripping the OS
allocator.

## What it is

`cudarc` already routes every `CudaSlice` allocation through CUDA's
**stream-ordered** allocator (`cuMemAllocAsync`) and frees through
`cuMemFreeAsync` on drop, on pool-capable devices (CUDA 11.2+; our sm_89 box
qualifies). The gap: the device's default pool ships with a **release threshold
of 0**, so every async-freed block is returned to the OS at the next
synchronization, and the next same-size allocation pays a fresh `cuMemAlloc`.

P5-04 raises the default pool's `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` to "retain
everything" (`u64::MAX`) at `CudaContext` construction. The default pool becomes
a caching pool: freed blocks stay reserved and the next allocation is a pool
hit. Because **both** GPU backends (`CudaSvBackend`, `CuStateVecBackend`)
allocate through `CudaContext`, both get the pool with zero per-backend change.

- `src/pool.rs` — `MemPool` wraps the device default pool: configure (set the
  release threshold), query `RESERVED_MEM_CURRENT` / `USED_MEM_CURRENT`, and
  `cuMemPoolTrimTo`. All via FFI cudarc already exposes in `cudarc::driver::sys`.
- `CudaContext` holds the pool and exposes `pool_reserved_bytes()`,
  `pool_used_bytes()`, `trim_pool()`, `synchronize()`, and a hidden
  `set_pool_release_threshold()` (the A/B benchmark hook).

### Why not a custom Rust free-list, pinned memory, or extra streams?

- **Custom free-list**: redundant. The driver's stream-ordered pool already does
  size-bucketed caching with correct stream-ordering semantics; we only had to
  stop it from releasing eagerly. This is exactly NVIDIA's recommended approach
  ("Using the CUDA Stream-Ordered Memory Allocator", part 1).
- **Pinned host memory**: cudarc's `alloc_pinned` always sets
  `CU_MEMHOSTALLOC_WRITECOMBINED`, which is *pessimal for host reads* — and our
  dominant transfer is the device→host amplitude **readback**. Write-combined
  staging would slow the common path, so it is deliberately deferred until a
  non-WC pinned alloc is available. (htod-only staging is a minor win — only the
  measurement re-upload — and not worth the asymmetry.)
- **Streams / transfer-compute overlap**: that is P5-05 (transfer optimization),
  which keeps the state on-GPU across gates and overlaps the few remaining
  copies. Out of scope here.

## Correctness

`tests/mem_pool.rs` (skips cleanly without a GPU):

- `pool_reuses_freed_blocks_without_growth` — 24 vs 240 allocate/free cycles of a
  64 MiB state: reserved bytes stay bounded (≤ 2× across a 10× churn increase,
  and ≤ a few states absolute), proving reuse rather than per-cycle OS
  allocation.
- `many_small_circuits_no_leak` — 2000 small circuits (6–11 qubits) run end to
  end; after dropping all states and synchronizing, the pool reports ~0 bytes in
  use (no leak).

Both GPU backends' full oracle suites still pass unchanged (the context change
is transparent to gate application and readout).

## Performance

RTX 4000 SFF Ada (20 GiB, sm_89), CUDA 13.0. One iteration = allocate `|0…0⟩`,
free, synchronize — a small circuit's allocation lifecycle. Retaining pool
(`u64::MAX` threshold) vs the un-tuned default (threshold 0, release-to-OS):

| state size      | release-to-OS | retain (pool) | speedup |
|-----------------|--------------:|--------------:|--------:|
| n=22 (64 MiB)   |    1795.7 µs  |     255.7 µs  |   7.0×  |
| n=24 (256 MiB)  |    5992.0 µs  |    1014.6 µs  |   5.9×  |
| n=26 (1024 MiB) |   24221.6 µs  |    4047.0 µs  |   6.0×  |

Allocation overhead is **negligible relative to the un-pooled baseline — a
6–7× reduction** in per-cycle allocate/free cost across state sizes (the
remaining retain-path time is dominated by zeroing the freshly-allocated `|0…0⟩`
buffer, not allocation). Reproduce:

```bash
ALEPH_POOL_N=24 cargo test -p aleph-cuda --features cuda --release \
  -- --ignored --nocapture pool_alloc_overhead
```
