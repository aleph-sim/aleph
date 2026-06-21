# Phase 5.7 audit — why the Metal MPS lost to the CPU MPS

> Post-mortem on the Phase 5.7 exit result (`phase5.7.md`): the GPU MPS is correct
> and feature-complete but **12–190× slower** than the Phase-3 f64 CPU MPS at
> n ≤ 14. This document traces the loss to specific code, contrasts with how the CPU
> MPS avoids each cost, and feeds the Phase 5.8 rewrite (`BACKLOG.md` § Phase 5.8).

## The numbers

`benches/mps_vs_cpu.rs`, M4 Mac Mini, bond cap 256 (exact), median wall-clock:

| workload | n | cpu | gpu_batched | ratio |
|----------|---|-----|-------------|-------|
| NN brickwall d=24 | 12 | 17.6 ms | 381 ms | 22× |
| NN brickwall d=24 | 14 | 95 ms | 1.10 s | 12× |
| bond-saturating | 14 | 4.6 ms | 280 ms | 61× |

Two derived facts from the P5.7 reports/benches:

- **Split phase ≈ 96% of GPU time** (n=12 d=24: ~9 ms contract+apply vs ~244 ms
  split). The "split" is mostly *host* work + allocations + small dispatches, not
  GPU FLOPs.
- **Canonicalisation tax ≈ 100 ms** (n=12 NN: `run` 483 ms − `run_batched` 381 ms),
  i.e. ~21% of the canonical path is the host SVD centre-moves alone.

## Root causes (ranked), with code evidence

1. **Canonical `run` does host f64 SVDs on top of the GPU SVD.**
   `apply_2q_nn` calls `move_center_to` before any GPU work (`backend.rs`); each
   centre step is a host faer SVD (`canonical.rs` → `svd.rs` `factor`). Worst case
   **up to O(n) f64 SVDs per single 2q gate**, GPU idle — *plus* the per-gate GPU
   Jacobi SVD. The CPU MPS uses **QR** (2–4× cheaper) for moves and SVD only for the
   truncating split.

2. **Θ bounces GPU→host→GPU every gate.** Θ is contracted on the GPU, read to host
   for a column-major repack, re-uploaded into a fresh Jacobi buffer, factored, read
   back (`.to_vec()`), and the factors re-uploaded as new site buffers. Unified
   memory's zero-copy advantage is thrown away.

3. **No device-buffer pooling.** `DeviceBuffer::from_slice` calls
   `new_buffer_with_data` every time (`buffer.rs`); the per-gate path allocates
   ~6 + 2·(centre steps) buffers. The CPU MPS allocates **zero** in steady state via
   the `Scratch` arena (`aleph_mps::mps`).

4. **Per-gate dispatch + sync.** The `run` path does **3 `wait_until_completed` per
   gate** (contract, apply, Jacobi). `run_batched` collapses these to 2 per *layer*
   (the P5.7-04 win), but it's still a full pipeline flush on microsecond-scale
   kernels.

5. **O(χ³) host reconstruction guard per gate.** The P5.7-07 accuracy guard
   (`recon_residual`) does a full f64 `‖Θ − UΣVᴴ‖` on the host every gate — necessary
   for f32-Jacobi robustness, but a real tax; it should be on-device or O(k).

6. **Physical SWAP-with-unwind for non-NN gates.** `apply_2q_routed` does adjacent
   SWAPs (each a gemm + truncated SVD) **and unwinds them in reverse** to restore
   site≡qubit order — ~2× the cost the CPU MPS pays (it routes lazily and leaves the
   permutation in place).

## What the CPU MPS does that we don't

| technique | CPU MPS (`aleph_mps`) | Metal MPS today |
|-----------|------------------------|------------------|
| buffer pooling | `Scratch` arena, zero per-gate alloc | none (fresh buffers/gate) |
| canonicalisation | **QR/LQ**, SVD only for truncating split | **SVD on every move**, host-side |
| non-NN routing | **lazy permutation**, no swap-back; user `Swap` = O(1) relabel | physical SWAP **with unwind** |
| contraction | gemm + `Accum::Replace`, no memset | Θ on GPU (good); centre-move absorb = serial host loop |
| precision | f64 throughout (lossless QR/SVD) | f32 GPU SVD + O(χ³) host recon guard + f64 fallback |

## Is it fixable? Honest assessment

- **At small χ (≤128): probably not.** The per-gate work is tiny; GPU launch/sync
  overhead (~tens of µs/dispatch) is irreducible below the CPU's whole-gate time, and
  faer on a warm cache is near-optimal. The CPU will likely keep winning here.
- **At large χ (≥256) and larger n: yes, plausibly.** SVD/contract are O(χ³); once
  the FLOPs dominate launch overhead the GPU should win. The measured gap **halves
  per +2 qubits** on the NN brickwall (104×→12× over n=8→14), with a naive crossover
  ~n≈18–20 — but the scaffold's 28-qubit dense-readout test cap can't reach it.

The fix is **architectural, not point patches**: everything resident on the GPU,
zero host work on the hot path, QR (not SVD) for moves, lazy (not physical) SWAPs,
and benchmarking in the large-χ regime. Fixing only pooling without the algorithmic
gaps (SVD→QR moves, physical→lazy SWAP, host→GPU residency) still loses.

## Plan

Phase 5.8 (`BACKLOG.md`): P5.8-01 large-χ harness → P5.8-02 buffer pool →
P5.8-03 GPU-resident pipeline → P5.8-04 GPU QR canonicalisation →
P5.8-05 lazy SWAP → P5.8-06 exit re-bench + verdict.
