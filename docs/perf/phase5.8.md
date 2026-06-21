# Phase 5.8 — GPU-resident MPS rewrite (Apple/Metal): exit report

> Report for the **Phase 5.8** work that made the Metal MPS backend fully
> GPU-resident — eliminating the per-gate host round-trips, allocations, and CPU
> linear algebra that the Phase 5.7 audit (`docs/perf/phase5.7-audit.md`) blamed for
> the GPU MPS being 12–190× slower than the f64 CPU MPS. Perf is never gated in CI;
> these are reproducible local measurements on the M4 Mac Mini.

## What shipped

| Step | What |
|------|------|
| **P5.8-01** | Large-χ / large-n test + bench harness (`benches/mps_vs_cpu.rs` `large_n` cells; `tests/mps_large_n.rs`) — the χ ≳ 256 / n > 14 regime made measurable with `2^n`-free correctness checks. |
| **P5.8-02** | **Device-buffer pool** (`DeviceBuffer` capacity reuse; in-place `SiteTensor`; `JacobiScratch`). Steady-state device allocations per gate: **0** (was ~6 + 2·moves). |
| **P5.8-03** | **GPU-resident per-gate pipeline** (`mps_pack.metal` + `mps_finalize.metal`): contract → apply → pack → Jacobi → σ-sort → truncate → assemble, **one command buffer, one `commit`/`wait` per gate**; U/V/σ never read back; recon guard on-device. |
| **P5.8-04** | **GPU Householder QR canonical moves** (`mps_qr.metal` + `mps_qr_install.metal`), with the whole move sweep **fused onto the gate command buffer** (no per-move sync). Tiny blocks (bond < 96) fall back to host f64 SVD. |
| **P5.8-05** | **Lazy-permutation SWAP routing**: non-NN gates route with a forward SWAP network that is **not unwound** (≈ half cost); user `Swap` is an O(1) relabel; readout/dense follow the permutation. |
| **P5.8-06** | This exit re-bench + verdict. |

## Correctness

The fully GPU-resident path holds the project's **1e-5** MPS oracle end-to-end (`run`
and `run_batched` vs the CPU MPS and the exact FP64 SV), including the n=8 χ=4
truncation case; the on-device finalize's reconstruction guard degrades a
mis-converged f32 Jacobi block to the f64 CPU SVD (without it, `run` drifted 1.2e-2 vs
CPU f64 at n=16; with it, 8e-6). `2^n`-free invariants (norm, GHZ `Z`-string,
`run` vs `run_batched`) hold to n=24, and the large-χ truncating regime (n=16/20,
χ=512) keeps unit norm. Lazy routing is checked under a non-trivial permutation
(readout + dense vs references) and the user `Swap` is verified zero-allocation.

## Exit metric: **GPU MPS ≥ CPU MPS on ≥ 1 regime**

**Verdict: NOT met.** The GPU MPS is faster than it was (every per-gate host
round-trip is gone, allocations are zero, the SVD/QR/sort/assembly are all on-device),
but it is still **slower than the f64 CPU MPS in every measured regime**, and at fixed
bond the gap **grows** with n.

Bond-saturating brickwall, central bond capped at χ=256, median wall-clock:

| n | cpu (f64 MPS) | gpu (FP32, GPU-resident) | ratio |
|---|---------------|--------------------------|-------|
| 16 | 0.377 s | 6.07 s | **16.1×** |
| 20 | 4.34 s | ≈ 95 s | **≈ 22×** |

(The Phase 5.7 baseline at n=16 χ=256 was 6.2 s — so the full rewrite shaved only a
few percent off the *large-bond* cell, while the small-bond NN-brickwall cells are at
parity with the pre-rewrite host path, P5.8-03/04 reports.)

## Why the GPU still loses — and why the gap grows

The host costs the audit named are genuinely gone. What remains is **the per-block
factorisation kernels themselves**: the two-site Jacobi SVD (P5.7-03) and the
Householder QR moves (P5.8-04) each factor **one block with one threadgroup** (≤ 256
threads). The CPU MPS hands each block to faer, which uses **all cores + SIMD** and a
blocked, cache-tuned LAPACK-style SVD/QR.

- At χ=256 a two-site block is up to 512×512. A single 256-thread threadgroup is far
  from enough parallelism to cover an O(χ³) factorisation; faer's multi-core blocked
  kernels win comfortably.
- As **n grows at fixed χ**, the number of (still single-threadgroup) blocks grows and
  each gate's GPU dispatch/sync overhead recurs, while faer keeps amortising across
  cores — so the ratio **rises** (16× → 22× over n=16 → 20) rather than crossing over.

This is exactly the limiter the audit flagged as the residual risk ("at large χ:
plausibly — *once the FLOPs dominate launch overhead*"). The FLOPs never get to
dominate because the single-threadgroup kernel caps the usable parallelism well below
what the block needs.

## What would actually move the needle (future work, not Phase 5.8)

1. **Multi-threadgroup factorisation.** A block-Jacobi / block-Householder that
   tiles one factorisation across *many* threadgroups (cooperative or via a small
   sequence of dispatches), so a 512×512 block uses the whole GPU rather than 256
   threads. This is the single highest-leverage change and the precondition for any
   crossover.
2. **Cross-gate batching on the GPU.** Even with a fast per-block kernel, per-gate
   dispatch/sync overhead is real; batching independent blocks across gates (beyond
   the within-layer batching of P5.7-04) would amortise it.
3. **Mixed precision only where it pays.** The f32 kernels need a recon guard + f64
   fallback; a faster, better-conditioned factorisation would reduce fallbacks.

## Bottom line

Phase 5.8 delivered what it set out to architecturally: a **correct, fully
GPU-resident, zero-host-round-trip, zero-steady-state-allocation** Metal MPS backend,
with QR (not SVD) canonicalisation and lazy SWAP routing. It did **not** make that
backend beat the f64 CPU MPS — the single-threadgroup per-block factorisation is the
wall, and closing it is a separate, larger piece of work (multi-threadgroup
factorisation). The honest state of the Apple/Metal MPS track: **architecturally
complete, performance-uncompetitive with the CPU MPS at the measured sizes.**
