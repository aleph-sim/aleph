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
| **P5.7-07** | **Canonical form → real truncation** (`mps/canonical.rs`): track an orthogonality centre (SVD-based QR/LQ moves), renormalise the kept σ, and *apply* a bond-cap truncation on the `run` path instead of refusing it. |
| **P5.7-08** | **Exit benchmark + report** (`benches/mps_vs_cpu.rs` + this report's verdict): GPU MPS vs CPU MPS sweep, exit metric stated and evaluated. |

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
- **Truncation is canonical on the `run` path (P5.7-07).** `run`/`apply_2q_nn`
  maintains an orthogonality centre and applies bond-cap truncation with the
  `1/√(kept weight)` renormalisation, matching the CPU MPS at a matched cap.
  `run_batched` has no single centre, so it stays **exact-only** and still refuses
  a real truncation — use `run` when compressing.
- **Canonicalisation is host-side, SVD-based.** The centre moves via a thin SVD per
  stepped site (an SVD-standing-in-for-QR); a QR move and GPU offload would cut the
  per-move cost — a follow-up, not needed for correctness.
- **GPU-Jacobi accuracy guard (added here).** The f32 one-sided Jacobi can
  mis-converge on an ill-conditioned two-site block (its U then not quite isometric),
  a small per-gate error that compounds the state norm over a deep/SWAP-routed
  circuit. `gpu_svd_split` now checks the reconstruction residual `‖Θ′−UΣVᴴ‖/‖Θ′‖`
  and degrades that block to the f64 faer SVD when it exceeds ~1e-3 (well-conditioned
  blocks pass at ~1e-4 and stay on the GPU). Without it, random depth-≳13 routed
  circuits drifted to a ~1e-1 norm error; with it they hold the ~1e-3 routed-f32
  budget. A higher-precision GPU SVD (or a GPU-side orthogonality polish) would let
  more blocks stay on-device — a follow-up.

-----

## Phase 5.7 exit report (P5.7-08)

### What the sub-phase delivered

The MPS-on-Metal backend went from "correct but CPU-SVD-bottlenecked" (end of 5.6)
to **GPU-resident and feature-complete**:

- the two-site split runs on the GPU (one-sided Jacobi kernel, P5.7-01/02/03);
- a brickwall layer's disjoint splits batch into one dispatch (P5.7-04);
- full readout — `measure`/`sample`/`probabilities`/`expectation_value` — with no
  `2^n` allocation (P5.7-05);
- non-nearest-neighbour gates via a SWAP router (P5.7-06);
- canonical form, so a bond-cap truncation is **applied** with controlled error,
  not refused (P5.7-07).

Correctness is gated end-to-end vs the CPU MPS **and** the exact FP64 statevector
(`mps_oracle`, `mps_readout`, `mps_proptest`), including a capped-χ truncation
oracle and a GHZ n=26 no-`2^n` readout check.

### Benchmark — GPU MPS vs CPU MPS

`benches/mps_vs_cpu.rs`, M4 Mac Mini, live desktop, load ≈ 2.4 (not idle — treat as
order-of-magnitude), bond cap 256 (no truncation; exact compare). Median wall-clock
per circuit; `gpu_batched` is `run_batched` (P5.7-04), `cpu` is `aleph_mps::MpsBackend`.

**NN random brickwall, depth 24:**

| n | cpu | gpu_batched | gpu_batched ÷ cpu |
|---|-----|-------------|-------------------|
| 8  | 0.81 ms | 84 ms | 104× |
| 10 | 3.5 ms | 236 ms | 66× |
| 12 | 17.6 ms | 381 ms | 22× |
| 14 | 95 ms | 1.10 s | 12× |

**Bond-saturating brickwall (`brickwall_ry_cnot_rz`, central bond → 2^(n/2)):**

| n | cpu | gpu_batched | gpu_batched ÷ cpu |
|---|-----|-------------|-------------------|
| 8  | 0.30 ms | 56 ms | 189× |
| 10 | 1.0 ms | 109 ms | 106× |
| 12 | 2.9 ms | 182 ms | 62× |
| 14 | 4.6 ms | 280 ms | 61× |

Batching helps ~1.1–1.3× over gate-by-gate `gpu`, consistent with P5.7-04.

### Exit metric — **not met**

The stated exit metric was *GPU MPS ≥ CPU MPS wall-clock on ≥ 1 regime*. It is **not
met**: the Phase-3 CPU MPS (f64 faer SVD, rayon, years-tuned) is **12–190× faster**
than the GPU MPS across every cell measured. Honest verdict: **Phase 5.7 succeeds on
correctness and feature-completeness, and fails its performance exit metric.**

Why the GPU loses at these scales:

- **Per-op dispatch latency dominates.** Each two-site gate is a handful of small
  Metal dispatches with a host sync; at bond χ ≤ 128 the kernels finish in
  microseconds but the launch/sync + unified-memory readback cost far more.
- **Host overhead is large and serial.** Column-major packing, σ-sort, χ-selection,
  canonical-form SVD sweeps, and site-tensor rebuild all run on the host per gate.
- **The CPU baseline is exceptional at small/medium bond.** faer's blocked SVD on a
  warm cache beats a cold GPU dispatch until the SVD FLOPs are large enough to hide
  the launch cost — i.e. χ in the high hundreds / large n.

The one encouraging trend: on the NN brickwall the gap **halves per +2 qubits**
(104× → 12× over n = 8 → 14) as growing bond gives the GPU more work to amortise; a
naive extrapolation puts crossover around n ≈ 18–20 — beyond the scaffold's 28-qubit
dense-readout test cap, so not reachable by this bench. The bond-saturating regime
plateaus near 60×, where faer stays dominant.

### What would move the needle (future work)

- **Cut per-gate host/sync overhead:** GPU-side packing/unpacking and canonical-form
  moves (QR on-device), fewer `wait_until_completed`s, a persistent buffer pool.
- **Higher-precision / faster GPU SVD:** an on-device orthogonality polish so fewer
  blocks fall back to f64 (P5.7-07 guard), and a blocked GPU SVD for large χ.
- **Push to the large-χ / large-n regime** where the GPU SVD FLOPs dominate — needs
  lifting the dense-readout test cap (readout is already `2^n`-free since P5.7-05).

A full root-cause post-mortem (per-gate host round-trips, allocations, host SVD
canonicalisation, physical SWAPs) is in **`docs/perf/phase5.7-audit.md`**; the rewrite
is scheduled as **Phase 5.8** in `BACKLOG.md`.
