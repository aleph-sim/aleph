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

## Honesty caveats & what's next

- **Live desktop, single workload.** One n=12 d=24 brickwall on a contended Mini;
  treat the ratio as order-of-magnitude. The contract+apply column varied ~53 ms
  (GPU-busy) vs ~95 ms (after a long idle CPU SVD) between branches — likely GPU DVFS,
  not the kernel — which is why the split column, not the percentages, is the headline.
- **Per-gate dispatch is the new bottleneck.** The split is still ~85% of per-gate
  time because the kernel runs as **one threadgroup per block with a per-gate
  `wait_until_completed`**. The next levers: batch a brickwall layer's independent
  blocks into one dispatch, and drop the per-gate sync. Larger bonds (deeper
  entanglement) will also amortize the dispatch better than these small Tier-1 blocks.
- **Still NN-only, exact-only.** No SWAP router and no canonical-form renormalization,
  so a real truncation is refused, not applied — unchanged from P5.6-02.
