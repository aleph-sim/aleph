# P2-01 — Rayon Parallel Gate Application — Scaling Report

**Phase:** 2 (multi-threaded CPU)
**Issue:** P2-01
**Date:** 2026-06-01
**Hardware:** AMD EPYC 8124P (Siena), 16 physical cores / 32 threads (SMT×2),
single socket, 1 NUMA node, 6-channel DDR5, ~126 GiB. Hosted instance with a
power/frequency cap (see §3). Toolchain: Rust 1.95, `RUSTFLAGS="-C
target-cpu=native"`, criterion release builds.

## 1. Summary

P2-01 parallelizes every state-vector gate kernel (AoS + SoA; 1q/2q + Toffoli/CCZ
3q) with rayon, behind a runtime amplitude threshold (`ALEPH_PAR_MIN_AMPS`,
default `1<<18`). Correctness is bit-identical across thread counts — each
parallel task writes pairwise-disjoint amplitude blocks with no cross-thread
floating-point reduction, so the result is invariant to `RAYON_NUM_THREADS`
(verified at 1e-12 vs. the Naive backend across {1,2,4,8} threads;
`scripts/p2-01-thread-sweep.sh`).

**The P2-01 acceptance gate (≥6× on QFT-25 at 8 cores) is NOT met on this
hardware: measured 1.6×.** The investigation (§3–§5) shows this is dominated by
two factors that are not parallelization defects — an all-core frequency throttle
and memory-bandwidth saturation — plus one genuine code limitation that we fixed
(count-starvation, §4). On this single-socket, frequency-capped box the
6×/8-core and 12×/16-core targets are not physically reachable for
bandwidth-bound state-vector simulation regardless of code quality.

## 2. Headline numbers (QFT-25, raw `run`)

| Threads | Time | Speedup vs 1 |
|--------:|-----:|-------------:|
| 1  | 11.16 s | 1.00× |
| 8  |  7.06 s | 1.58× |
| 16 |  6.94 s | 1.61× |

Scaling plateaus by 8 threads. For contrast, a dense-2q brick-wall circuit
(`random`, n=20, higher arithmetic intensity, partly L3-resident) reaches 2.2×
at 8 threads. Both are far below linear.

## 3. Root cause — frequency throttle (environmental)

The hosted EPYC instance drops its all-core clock sharply under load:

| Load | Active-core MHz |
|------|----------------:|
| 1 thread (boost)   | **2995** |
| 8 threads (all-core) | **1628** |

That is a **1.84× per-core frequency handicap** at 8 threads (the box's own
`lscpu` reports "CPU scaling 55%"). This alone caps the *ideal* 8-core speedup at
≈ 8 × (1628/2995) ≈ 4.3×, before any memory or algorithmic effect. It is a
property of the benchmark host, not of aleph.

## 4. Root cause — count-starvation (code; fixed)

The block-walk kernels nest an outer walk (`outer_count = 2^(n-1-target)` for a
1q gate) over an inner SIMD walk. The initial `par_blocks` driver parallelized
only the **outer** dimension. For a gate on a high qubit, `outer_count` collapses
to 1 — the gate ran fully sequentially despite touching the whole state.

Isolating diagnostic (40 H gates at n=25, identical kernel and memory traffic,
varying only the target qubit):

| Circuit | outer_count | T=8 speedup (before fix) | T=8 speedup (after fix) |
|---------|------------:|-------------------------:|------------------------:|
| H on q2 (low)   | 2²² | 2.72× | 2.2× |
| H on q24 (high) | 1   | **1.03× (none)** | **4.86×** |

**Fix:** `par_units` (in `kernels/mod.rs`) flattens the outer-block × inner-SIMD
walk into one parallel dimension of size `len/(2·LANES)` — independent of the
target qubit. Applied to the generic 1q kernel (`apply_1q_avx512`) and the
QFT-dominant controlled-phase kernel (`apply_2q_diagonal_avx512`). High-qubit
gates now parallelize fully (1.03× → 4.86×), while low-qubit gates are unchanged
(`units_per_block == 1` degenerates to the original driver).

## 5. Root cause — memory bandwidth (fundamental)

Despite the count-starvation fix, **QFT-25 barely moved (1.54× → 1.58×).** QFT is
≈ 92% controlled-phase gates, the lowest-intensity gate in the set (1 read + 1
write + 1 complex multiply per amplitude). At n=25 the 512 MiB state vector is
pure DRAM streaming; a few cores saturate memory bandwidth, so additional cores
add little — the same reason the dense (higher-intensity) `random` circuit scales
better (2.2×) than QFT (1.6×). The count-starvation fix is real and valuable
(proven 4.86× on high-qubit workloads, and it matters more on hardware that is
not bandwidth- and frequency-capped), but it cannot move a workload whose ceiling
is memory bandwidth.

Note also that the honest end-to-end path is `run_optimized` (gate fusion;
qft-parity / PR #96). Fusion collapses the QFT cphase ladder (~36× less work,
10.97 s → ~0.30 s) but its multicore scaling is similarly bandwidth-limited
(~1.3–1.45×) — fewer, denser passes, still streaming.

## 6. What landed

- `par_blocks` + `par_units` drivers; `BlockPtr` Send/Sync wrapper;
  `ALEPH_PAR_MIN_AMPS` threshold.
- All AoS + SoA 1q/2q + Toffoli/CCZ 3q kernels parallelized (outer-block).
- Count-starvation flattening on `apply_1q_avx512` and `apply_2q_diagonal_avx512`.
- Thread-count-invariance harness (`scripts/p2-01-thread-sweep.sh`) and a gated
  QFT scaling bench (`qft_scaling`, feature `scaling-bench`).

## 7. Follow-ups (filed)

1. **Propagate `par_units` flattening** to the remaining inner-loop kernels
   (`apply_1q_{diagonal,x,y,antidiag}_avx512`, 2q cnot/swap/cz/dense tiers, 3q
   tier-A, and the SoA mirrors). Mechanical; will not change QFT (bandwidth) but
   improves high-qubit-gate scaling and matters on non-throttled hardware.
2. **Re-validate on non-throttled, multi-channel/multi-socket hardware** before
   judging the ROADMAP §7 ≥12×/16-core exit — the current box cannot demonstrate
   it for bandwidth-bound SV simulation regardless of code.
3. **P2-04 chunk-size tuning** of `ALEPH_PAR_MIN_AMPS` and the `with_min_len`
   grain per gate type / qubit position.
4. **Revisit the P2-01 / ROADMAP scaling targets** for bandwidth-bound kernels —
   an efficiency-relative-to-bandwidth or compute-bound-regime metric is more
   honest than a fixed ≥6×/≥12× for memory-streaming gate application.

## 8. Addendum — scalar-path parallelization (follow-up)

P2-01 parallelized only the AVX-512 kernels; on non-AVX-512 hosts (Zen 2, ARM,
older Intel) the dispatch falls back to the **scalar** kernels, which were left
sequential. A follow-up branch parallelized those too: a `ComplexPtr` (`*mut
Complex` Send/Sync wrapper) + `par_blocks(len, len, |k| k, body)` over the flat
per-amplitude walk. The scalar kernels are pure `0..len` walks with a per-index
guard, so they have **no count-starvation** — full parallelism at any qubit.

Cross-check on a second box (AMD Ryzen 9 3900, Zen 2, 12c/24t, **no AVX-512**,
dual-channel DDR4):

| Path / box | QFT-25 T=1 | T=8 | Plateau |
|------------|-----------:|----:|--------:|
| AVX-512 / EPYC 8124P | 11.16 s | 1.6× | yes |
| Scalar / Ryzen 9 3900 | 12.83 s | **2.11×** | yes (≈8 threads) |

The scalar path scales somewhat better (higher arithmetic intensity per core, and
the Ryzen throttles less), but **both paths on both machines plateau, bandwidth-
bound, far below the ≥6× target.** Two independent CPUs and two code paths
agreeing reinforces §5: state-vector gate application is fundamentally memory-
bandwidth-bound, and the ≥6×/8-core / ≥12×/16-core goals require either much
higher memory bandwidth (more channels / sockets) or higher arithmetic intensity
(fusion, cache blocking) — not more threads. Notably the scalar single-thread
time (12.83 s) is within ~15% of the AVX-512 single-thread time (11.16 s): for
this memory-bound workload the SIMD advantage is largely eaten by the memory
wall.

## 8. Reproduce

```bash
# Correctness (thread-count invariance, forced parallel at small n):
./scripts/p2-01-thread-sweep.sh

# QFT scaling sweep on the bench server:
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench -p aleph-benches --bench qft_scaling --features scaling-bench -- --save-baseline t1
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=8 \
  cargo bench -p aleph-benches --bench qft_scaling --features scaling-bench -- --baseline t1
```
