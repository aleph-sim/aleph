# P2-04 — Chunked parallelism tuning (perf report)

**Issue:** #30 · **Milestone:** Phase 2 · **Date:** 2026-06-01
**Spec:** `docs/superpowers/specs/2026-06-01-p2-04-chunk-tuning-design.md`
**Raw data:** `docs/perf/data/p2-04-tune-epyc.log`, `docs/perf/data/p2-04-tune-ryzen.log`
**Sweep tool:** `scripts/tune-chunks.sh` · **Bench:** `crates/aleph-sv/benches/chunk_tune.rs`

## TL;DR

Chunk tuning has **no headroom to capture**. Across a 360-cell grid
(gate × target × `min_amps` × `grain`) on two reference CPUs, every cell is
within **~0.4 % (noise) of the pre-P2-04 default `grain = 64`**. No cell beats
the default, so the tuned table is left at `DEFAULT_POLICY` everywhere. The
genuine signals are *negative* and **confirm** the default rather than improve
on it. This is the same bandwidth-bound conclusion P2-02 (alignment) and P2-03
(NUMA) reached.

The deliverable is therefore the **tuning + CPU-dispatch infrastructure** (a
`ChunkPolicy` table selected by CPU model, threaded leaf-locally into
`par_blocks`/`par_units`, with a bit-exact invariance guarantee and an env
sweep instrument) plus the empirical validation below — not a speedup.

**Primary reference CPU: EPYC 8124P** (AVX-512). It is the only box that shows
any chunk sensitivity at all (the large-grain regression below); Ryzen 9 3900
(scalar) is the flat cross-check.

## Method

For each gate class we drove the matching kernel on a fresh `n = 25` state
(2²⁵ = 33.5 M amplitudes ≈ 512 MiB AoS) via `chunk_tune`, forcing the policy
through `ALEPH_PAR_MIN_AMPS` / `ALEPH_PAR_GRAIN` (the per-field env override in
`tuning::resolve_policy`). One `cargo bench` **process per grid point** (the env
knobs are cached once per process). Grid:

- gates: `h` (OneQGeneric), `zdiag` (OneQDiag), `cnot` (TwoQCnot), `cphase` (TwoQDiag)
- targets: `1` / `12` / `24` → PosClass Low / Mid / High
- `min_amps`: 2¹⁶ … 2²⁰ · `grain`: 16 / 32 / 64 / 128 / 256 / 512

Both boxes were verified idle before measuring (`uptime` ≈ 0, no competing
`cargo bench` / `bencher` / CI-runner jobs) per CLAUDE.md. Code was delivered by
`git bundle` (not a GitHub push) specifically to avoid triggering a CI bench on
the self-hosted EPYC runner mid-measurement.

## Finding 1 — `grain` best-vs-default is noise (≤ 0.4 %)

EPYC, per (gate, target) cell, averaged over the (inert) `min_amps` axis. Times
in ms; "best vs 64" = how much the best grain beats the default grain=64.

| cell | g16 | g32 | g64 | g128 | g256 | g512 | best vs 64 |
|---|---|---|---|---|---|---|---|
| h / 1   | 8.880 | 8.867 | 8.846 | 8.836 | 8.824 | 8.820 | +0.3 % |
| h / 12  | 8.892 | 8.884 | 8.878 | 8.868 | 8.864 | 8.856 | +0.2 % |
| h / 24  | 8.878 | 8.869 | 8.857 | 8.847 | 8.844 | 8.840 | +0.2 % |
| zdiag / 1  | 8.870 | 8.844 | 8.834 | 8.826 | 8.812 | 8.807 | +0.3 % |
| zdiag / 12 | 8.817 | 8.826 | 8.835 | 8.859 | 9.404 | 9.492 | +0.2 % |
| zdiag / 24 | 21.363 | 21.330 | 21.353 | 21.356 | 21.351 | 21.352 | +0.1 % |
| cnot / 1   | 10.561 | 10.550 | 10.532 | 10.517 | 10.510 | 10.495 | +0.4 % |
| cnot / 12  | 4.179 | 4.179 | 4.188 | 4.209 | 4.562 | 4.818 | +0.2 % |
| cnot / 24  | 10.435 | 10.470 | 10.476 | 10.450 | 10.504 | 10.459 | +0.4 % |
| cphase / 1  | 8.859 | 8.849 | 8.835 | 8.823 | 8.809 | 8.804 | +0.3 % |
| cphase / 12 | 9.106 | 9.094 | 9.093 | 9.076 | 9.067 | 9.064 | +0.3 % |
| cphase / 24 | 8.846 | 8.832 | 8.827 | 8.819 | 8.823 | 8.801 | +0.3 % |

Every "best vs 64" is ≤ 0.4 % — below run-to-run noise. There is no defensible
non-default value to encode.

## Finding 2 — large `grain` *regresses* stride-heavy AVX-512 kernels

The only effect above noise is a penalty at `grain ≥ 256` on the mid-target
stride-heavy kernels:

- **cnot / 12:** g64 = 4.188 ms → g512 = 4.818 ms = **+15.0 %**
- **zdiag / 12:** g64 = 8.835 ms → g512 = 9.492 ms = **+7.4 %**

Coarse rayon batches starve the worker pool for these kernels' block walk. The
default `grain = 64` sits safely in the optimal band, so it both maximises
throughput and avoids this cliff. Conclusion: **do not raise the grain.**

## Finding 3 — `min_amps` is inert at perf-relevant sizes

At n = 25 the state length (33.5 M) exceeds every tested `min_amps` (≤ 1.05 M),
so `len < min_amps` is always false and the kernel is always parallel — the
`min_amps` axis is flat by construction. The sequential cutoff only gates tiny
states (microseconds regardless), so the default `1 << 18` is fine and was not
meaningfully probed. (Probing it would require sweeping n near the threshold,
which is not where Tier-1 performance lives.)

## Finding 4 — Ryzen (scalar) is totally flat

On the Ryzen 9 3900 (no AVX-512, scalar kernels) every cell is flat within
~0.1–0.2 % across the *entire* grid — no grain or `min_amps` effect at all. The
scalar path is purely bandwidth-bound, exactly as P2-02/P2-03 documented. For
reference, the AVX-512 advantage at n = 25: `h` 8.85 ms (EPYC) vs 34.2 ms
(Ryzen) ≈ 3.9×; `cnot/12` 4.19 ms vs 17.4 ms ≈ 4.2×.

## Acceptance criteria

- [x] **Tuned chunk-size table for one reference CPU** — table built and
  CPU-dispatched (EPYC primary, Ryzen cross-check); empirically every probed
  cell resolves to the default, recorded as the table's measured content.
- [x] **Benchmark improvement over fixed default** — measured: **none
  available** (default already near-optimal; ≤ 0.4 % across all cells). Reported
  honestly. The actionable result is the *guard* against the +8–15 % large-grain
  regression and the validated infrastructure.

## Follow-ups

- Runtime auto-tuning remains out of scope (BACKLOG: "start with table") and is
  unmotivated by these results — there is nothing to auto-tune toward.
- If a future kernel or CPU shows real chunk sensitivity, adding a tuned cell is
  a one-line `chunk_policy` match arm, re-measured with the same sweep.
- P2-05 (Phase 2 scaling report) should fold in this bandwidth-bound theme
  alongside P2-01/02/03.
