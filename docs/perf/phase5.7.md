# Phase 5.7 — GPU-resident MPS SVD (one-sided Jacobi)

> Report for the **Phase 5.7** work that moves the MPS two-site SVD off the CPU and
> onto the Metal GPU. Phase 5.5/5.6 left the MPS scaffold correct but bottlenecked on
> a host SVD that blocked the GPU (`docs/perf/metal.md`: the host split was **94.5%**
> of per-gate tensor time, **17.3×** the GPU contract+apply). P5.6-07 (#214) parallelised
> that SVD on the CPU with rayon; this phase replaces it with a GPU-resident kernel.
> Perf is never gated in CI; these are reproducible local measurements.

## What shipped

| Step | What |
|------|------|
| **P5.7-01** | Complex one-sided **Jacobi thin-SVD** in Rust (`mps/jacobi.rs`) — the CPU reference, wired as the faer convergence fallback. |
| **P5.7-02** | The same algorithm as a **Metal compute kernel** (`mps_jacobi.metal`) + dispatch (`gpu_jacobi.rs`), validated standalone against the CPU reference / faer. |
| **P5.7-03** | Kernel wired into `MetalMpsBackend`: the per-gate split is now GPU-resident; only σ-readout + χ-selection stay on the host. faer remains the CPU fallback. |
| **P5.7-04** | **Batched layer-parallel SVD** (`jacobi_svd_batched` + `run_batched`): a brickwall layer's disjoint two-site splits factor in one dispatch, one `commit`/`wait` per layer instead of per gate. |
| **P5.7-05** | **Readout** (`mps/readout.rs`): `measure`/`sample`/`probabilities`/`expectation_value` via doubled transfer-matrix sweeps — bond×bond environments, no `2^n`, exact on the non-canonical scaffold. |
| **P5.7-06** | **SWAP router** (`apply_2q_routed`): non-adjacent 2q gates routed by a physical SWAP network (apply → unwind), restoring site≡qubit order; wired into `run` and `run_batched`. |

### Why one-sided Jacobi
It orthogonalizes the *columns* of Θ′ by right-multiplying 2×2 unitary rotations, so
σ = ‖column‖ is read directly instead of as √eigenvalue of the Gram matrix `AᴴA`. It
never squares the condition number — the property that makes it the FP32/GPU-friendly
SVD (cuSOLVER's `gesvdj` uses the same method). The complex 2×2 column-Gram is
real-symmetrized by a `diag(1, e^{-iφ})` phase pre-rotation before the standard real
Jacobi angle.

### Kernel shape
One threadgroup factors one two-site block: threads stride the row dimension, the
columns stay in device memory, and only the per-pair 2×2 reduction and the broadcast
rotation scalars use threadgroup memory. Threadgroup size is the largest power of two
the device allows (≤ 256; the reduction halves it). Wide blocks (`rows < cols`) are
factored as `Aᴴ` with the U/V roles swapped.

## Correctness

The single-precision GPU SVD holds the project's **1e-5** MPS oracle **end-to-end**:

- `mps_oracle` — dense statevector vs the CPU MPS backend **and** the exact FP64
  `NaiveSvBackend` (1q-only, GHZ n∈{3,5,8,10}, NN brickwall {4×6, 6×8, 8×6}).
- `mps_proptest` — random NN circuits vs CPU MPS.
- The P5.6-02 truncation guard is intact: bond-1 GHZ is **refused** (drops ~½ the
  Schmidt weight), and an in-cap run records ≈0 truncation.
- Kernel-level: `gpu_jacobi_matches_reference` reconstructs `A = UΣVᴴ` to 1e-4 and
  matches faer's FP64 σ across tall/square/wide/1×1/2×2/32×12 blocks.

FP32 is the GPU accuracy ceiling (~1e-5, not the FP64 backends' 1e-10), as for the
Metal SV backend. faer's f64 SVD stays the fallback for a non-finite GPU result.

## Performance — the host-SVD bottleneck collapses

NN brickwall **n=12, depth=24**, M4 Mac Mini, **live desktop** (`WindowServer` sharing
the GPU), 3 runs each. The timer splits per-gate time into the GPU contract+apply
dispatches vs the **two-site split** (the phase this work swaps). Absolute times and
the contract+apply column carry live-desktop variance (integrated-GPU clock state
differs between an idle-CPU-SVD run and a GPU-busy run); the **split** column is the
apples-to-apples comparison of the swapped phase.

| Split factorizer | split time (per circuit) | share of per-gate time | split ÷ contract+apply |
|------------------|--------------------------|------------------------|------------------------|
| faer CPU SVD (P5.6-07, rayon) | **~1390 ms** | 93.3% | ~14× |
| **GPU Jacobi (P5.7-03)** | **~300 ms** | ~85% | ~5.6× |

**~4.6× faster** on the split phase, the part this work changes. Per-circuit total
(contract+apply + split) falls from ~1.48 s to ~0.35 s on these runs.

The host SVD was the dominant per-gate cost since the scaffold shipped (94.5% in the
P5.5 report); it is now ~4.6× cheaper and runs on-device, so the SVD no longer
serializes the GPU behind a CPU library.

## Batched layer-parallel SVD (P5.7-04)

P5.7-03 left the split GPU-resident but still **one threadgroup per gate with a
per-gate `wait_until_completed`** — so per-gate dispatch latency was the named next
lever. P5.7-04 adds a **batched** path:

- `jacobi_svd_batched` (same `mps_jacobi.metal`, refactored to a shared
  `jacobi_block` device function): the grid is `num_blocks` threadgroups, each keys
  off `threadgroup_position_in_grid`, reads its `JacobiBlockMeta` (dims + per-block
  buffer offsets), and factors its slice of the packed `A`/`V`/`sig` buffers.
- `MetalMpsBackend::run_batched` is a backend-side scheduler: it greedily groups
  adjacent NN 2q gates that act on **disjoint** site pairs into a layer (disjoint ⇒
  commuting ⇒ batching is exact), flushing on a 1q gate, barrier, or site conflict.
  A flushed layer's contract + gate-apply run on **one** command buffer (one
  `commit`/`wait`; per-gate 4×4 buffers replace the shared scratch so the applies
  don't clobber each other), then **one** batched-Jacobi dispatch factors every
  block. faer stays the per-block CPU fallback; the truncation guard (P5.6-02) is
  unchanged — a layer is refused before any site tensor is mutated.

### Measurements — NN brickwall n=12 d=24, M4 Mac Mini, live desktop

Internal timer (`report_svd_roundtrip_cost`), summed over the run; load ≈ 3.3 (not
idle), so order-of-magnitude:

| Path | contract+apply | SVD split | total |
|------|----------------|-----------|-------|
| gate-by-gate (P5.7-03) | ~58 ms | ~378 ms | ~436 ms |
| **layer-batched (P5.7-04)** | **~9 ms** | **~244 ms** | **~254 ms** |

The split phase drops **~1.55×** (the batched dispatch removes the per-gate launch
latency; the residual is host-side packing/χ-selection/upload, which is still
per-block and serial). Contract+apply drops **~6×** from collapsing 2·N per-gate
syncs into one wait per layer.

Criterion wall-clock (`mps_batched` bench, 10 samples):

| Workload | gate-by-gate | batched | speedup |
|----------|--------------|---------|---------|
| NN brickwall n=12 d=24 (small bond) | ~448 ms | ~346 ms | **~1.30×** |
| bond-saturating n=12 (χ→64) | ~265 ms | ~251 ms | ~1.05× |

Batching wins most where per-gate dispatch latency dominates — many small blocks per
layer (small-bond brickwall). In the bond-saturating case each layer has fewer,
larger blocks whose per-block compute dwarfs the dispatch overhead, so the gain is
small; the lever there is the per-block factorization, not the launch count.

## Honesty caveats & what's next

- **Live desktop, single box.** Same caveats as P5.7-03: a contended Mini, GPU DVFS
  between runs; treat ratios as order-of-magnitude. The split column is the
  apples-to-apples comparison of the swapped phase.
- **Host packing is the new split-phase floor.** The batched dispatch is fast, but
  the split column still includes per-block host work (column-major packing, σ sort,
  site-tensor build + upload). Pooling the `A`/`V`/`sig` buffers across layers and
  moving the pack/unpack onto the GPU would chip at the residual.
- **Still exact-only.** Non-NN gates are now SWAP-routed (P5.7-06) and readout is
  supported (P5.7-05), but there is still no canonical-form renormalization, so a
  real truncation is refused, not applied — unchanged from P5.6-02 (canonical form
  is P5.7-07).
