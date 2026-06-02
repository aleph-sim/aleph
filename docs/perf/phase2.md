# Phase 2 — Multi-Threaded CPU — Scaling-Efficiency Report

**Phase:** 2 (multi-threaded CPU) · **Issue:** #31 (P2-05) · **Date:** 2026-06-02
**Consolidates:** P2-01 (#27), P2-02 (#28), P2-03 (#29), P2-04 (#30).
**Bench:** `benches/benches/tier1_scaling.rs` (feature `scaling-bench`).
**Hardware (all reference boxes, verified-idle per CLAUDE.md idle-check):**

- **EPYC** — AMD EPYC 8124P (Siena), 16 physical / 32 SMT, single socket, 1 NUMA
  node, AVX-512, **all-core frequency-throttled to ~55 %** (`phase2-p2-01.md` §3).
- **NUMA** — 2× Intel Xeon Silver 4114, 20 physical / 40 SMT, **2 sockets / 2 NUMA
  nodes** (distance 10/21), AVX-512.
- **Ryzen** — AMD Ryzen 9 3900, 12 physical / 24 SMT, **no AVX-512** (scalar path).

Toolchain: Rust 1.95, `RUSTFLAGS="-C target-cpu=native"`, criterion release builds.

## 1. Verdict

The ROADMAP §7 Phase-2 exit — **≥ 12× speedup on 16 cores vs single-thread**,
equivalently **≥ 75 % parallel efficiency at 16 threads** (the P2-05 spec's
target; 0.75 × 16 = 12) — **is not met on any hardware available to this
project, and cannot be on this hardware regardless of code.** The measured QFT-25
efficiency at 16 threads is **23 %** on the EPYC box (3.69×), and the same
saturating shape reproduces on a second CPU and a second (scalar) code path.

This is **not a parallelization defect.** State-vector gate application is
**memory-bandwidth-bound** at high core counts: at n=25 the 512 MiB state vector
is pure DRAM streaming, and the lowest-intensity gates (QFT is ~92 % controlled-
phase: 1 read + 1 write + 1 complex multiply per amplitude) saturate the memory
controllers with a handful of cores. Two independent ceilings sit below the
target — the EPYC's ~55 % all-core frequency throttle (an environmental cap that
alone limits its 16-core *ideal* to ≈ 8.6×) and the fundamental bandwidth wall —
and neither is movable by the kinds of work Phase 2 covered (parallelization,
alignment, NUMA placement, chunk tuning). The four Phase-2 tickets each
confirmed this from a different angle (§4).

Per the AC ("scaling target met **or** follow-ups filed"), we take the
follow-ups path (§6). The honest engineering conclusion: the parallelization is
**good for the regime it is in** — on EPYC, QFT-25 reaches **78 % of the box's
frequency-adjusted 8-core ceiling** — and the fixed ≥12×/≥75 % gate is the wrong
metric for a bandwidth-bound kernel (follow-up §6.2).

## 2. Headline scaling — QFT-25 (real measured data)

QFT is the canonical bandwidth-bound Tier-1 workload and the one with measured
multi-thread data across boxes (P2-01 §2/§8, P2-02 §3). `efficiency =
speedup / threads` is the spec's literal metric.

### EPYC 8124P (AVX-512), raw `run`

| Threads | Time | Speedup | Efficiency | Eff. vs freq-adjusted ideal |
|--------:|-----:|--------:|-----------:|----------------------------:|
| 1  | 8.41 s | 1.00× | — | — |
| 8  | 2.50 s | **3.37×** | 42 % | **78 %** |
| 16 | 2.28 s | **3.69×** | 23 % | 43 % |

Frequency context: the all-core clock drops 2995 → ~1620 MHz under load (~1.85×
per-core handicap), capping the *ideal* 8-core speedup at ≈ 4.3× and 16-core at
≈ 8.6× before any memory effect (`phase2-p2-01.md` §3). Against that adjusted
ceiling, 3.37×@8 is **78 %** — the parallelization is sound; the absolute number
is hardware-capped. P2-02 re-measured 3.31×@8 / 3.62×@16 (within noise;
alignment changed nothing). Fused (`run_optimized`) ≈ raw — QFT's controlled-
phase ladder acts on distinct qubit pairs and does not fuse.

### Ryzen 9 3900 (scalar, no AVX-512), raw `run`

| Threads | Time | Speedup | Efficiency |
|--------:|-----:|--------:|-----------:|
| 1  | 12.64 s | 1.00× | — |
| 8  | 6.00 s | **2.11×** | 26 % |
| 12 | 6.02 s | **2.10×** | 18 % |

(Single P2-02 re-measurement session, `phase2-p2-02.md` §3; an earlier P2-01
session measured T1=13.05 s / 2.15×@8 within run-to-run variance.) A second CPU,
a second code path: QFT-25 plateaus at **~8 threads** (8→12 buys nothing) — the
same bandwidth wall. The smaller QFT-22 (fits cache far better)
keeps scaling to 3.99×@12 (P2-02 §3), which sharpens the diagnosis: the n=25
plateau is bandwidth, not a parallelization defect.

### NUMA 2× Xeon 4114 — allocation-placement result (P2-03)

The NUMA box's measured Phase-2 contribution is the **allocation-policy** result,
not a full thread sweep: NUMA-aware **first-touch** allocation cut QFT-25 by
**−37.7 % (1.60×)** vs the default allocator — beating `numactl --interleave`
(−31.8 %) **with no thread pinning** (`docs/numa.md` results table). This is orthogonal to
the thread-scaling ceiling: correct page placement raises the achievable
bandwidth on a 2-socket box; it does not change the bandwidth-bound *shape* of
the per-thread curve.

## 3. Full Tier-1 matrix — measured + pending

The `tier1_scaling` bench measures GHZ / QFT / Grover / random at n=25, swept via
`RAYON_NUM_THREADS`. The cells below are the **measured** state of the project's
data. Cells marked **`pending HW run`** have **no fabricated numbers**: the bench
is delivered ready and produces them with one command (§7). No box reaches the
spec's 32/64-thread points except via SMT (EPYC 32t, NUMA 40t); 64 physical
threads is **unreachable on available hardware** (§5).

| Workload (n=25) | Box | T1 | T2 | T4 | T8 | T16 | T32 |
|---|---|---|---|---|---|---|---|
| QFT     | EPYC  | 8.41 s | pending | pending | **3.37×** | **3.69×** | pending |
| QFT     | Ryzen | 12.64 s | pending | pending | **2.11×** | — (12c) | — |
| GHZ     | EPYC  | pending HW run — *trivial workload, see §4.5* |
| Grover  | EPYC  | pending HW run |
| Random  | EPYC  | pending HW run |

Honest scope note: the multi-thread numbers measured during P2-01..04 targeted
QFT-25 (the workload over the Stage-0 Aer target and the clearest bandwidth
probe). GHZ/Grover/random full sweeps, and the intermediate 2/4/32-thread QFT
points, were **not** measured and are not invented here. The expectation, given
§4, is that Grover and random show the same bandwidth-bound plateau (Grover
carries Toffoli/CCZ, random is brick-wall — both higher arithmetic intensity than
QFT's cphase, so if anything they scale *slightly* better at low thread counts,
but hit the same wall); GHZ is degenerate (§4.5). Confirming this is follow-up §6.3.

## 4. Root-cause synthesis — what the four Phase-2 tickets established

### 4.1 P2-01 — parallelization + the two ceilings
Every SV gate kernel (AoS + SoA, 1q/2q/3q) is rayon-parallel behind
`ALEPH_PAR_MIN_AMPS`, bit-identical across thread counts. A **count-starvation**
bug (outer-block-only parallelism left high-qubit gates serial) was fixed with
`par_units` flattening (high-qubit gate 1.03× → 4.86× @8). The remaining limits
are environmental (frequency throttle) and fundamental (bandwidth) — not code.

### 4.2 P2-02 — contention is not the limiter
64-byte-aligned `AlignedBuf` + a false-sharing audit. `perf c2c` over a 16-thread
QFT-25: **28 shared lines / 24 local HITM across 230 k records** — noise, no
ping-pong. Scaling **flat vs P2-01**. There was no contention to remove; the
deliverable was an alignment *guarantee* (and the NUMA hook P2-03 needed).

### 4.3 P2-03 — NUMA placement helps bandwidth, not the curve shape
First-touch allocation (`zeroed_first_touch`, `numa` feature) gives **−37.7 %**
on the 2-socket box with no pinning (§2). It raises *achievable* bandwidth; it
does not change the bandwidth-bound per-thread scaling shape.

### 4.4 P2-04 — no chunk-tuning headroom
A 360-cell (gate × target × `min_amps` × `grain`) sweep on EPYC + Ryzen: every
cell within **~0.4 % of the default `grain = 64`**. Negative findings *confirm*
the default — large grain (≥256) *regresses* stride-heavy AVX-512 kernels by
+7–15 %; `min_amps` is inert at n≥21 (always parallel). Nothing to tune toward.

### 4.5 GHZ-25 is a degenerate scaling workload
GHZ-25 is 1 H + 24 CNOT = **25 gates total**, running in milliseconds and
dominated by state allocation/initialization, not gate-kernel throughput. Its
"efficiency" is allocation+setup noise, not a bandwidth-scaling signal. It is
included for spec completeness and annotated as such — never reported as a
meaningful parallel-efficiency data point.

## 5. The 64-core / ≥12× target is hardware-gated

The spec asks for thread counts up to 64; the ROADMAP exit asks for ≥12×@16. No
available box can demonstrate either:

- **No 64-physical-core box exists in the fleet.** EPYC 16c/32t, NUMA 20c/40t,
  Ryzen 12c/24t. Counts above physical cores are SMT (throughput-limited for this
  bandwidth-bound, FPU-heavy workload), and 64 is unreachable entirely.
- **The EPYC's frequency throttle** caps its 16-core *ideal* at ≈ 8.6× before any
  memory effect — ≥12×@16 is arithmetically impossible there.
- **Bandwidth** then pulls the realized EPYC 16-core figure to ~3.7×, and the
  Ryzen and NUMA boxes corroborate the saturating shape.

Demonstrating ≥12×@16 for bandwidth-bound SV simulation needs a **non-throttled,
high-memory-bandwidth, ≥16-physical-core** machine (and the 64-thread point a
≥64-core box) — hardware this project does not currently have. This is a
measurement-environment gap, recorded as a follow-up (§6.1), not an open code
defect.

## 6. Follow-ups (filed)

1. **Re-validate ≥12×@16 (and the 32/64-thread points) on non-throttled,
   higher-bandwidth, ≥32-physical-core hardware** when available. The target is
   gated on hardware we lack, not on a code defect.
2. **`[meta]` proposal: revise the ROADMAP §7 Phase-2 exit metric** toward an
   *efficiency-vs-achievable-bandwidth-ceiling* (or compute-bound-regime) form for
   memory-streaming SV kernels. A fixed ≥12×/≥75 % is not an honest gate for a
   bandwidth-bound workload (first flagged in P2-01 follow-up #4). This report
   **recommends** the `[meta]`; it does not edit ROADMAP.md here.
3. **Run the full `tier1_scaling` sweep** (GHZ/Grover/random, and the
   intermediate 2/4/32-thread QFT points) on EPYC + NUMA + Ryzen to fill the
   *pending* cells of §3. The bench is ready (§7).
4. **Propagate `par_units` flattening** to the remaining inner-loop kernels
   (carried from P2-01 follow-up #1) — improves high-qubit-gate scaling on
   non-throttled hardware; will not move the QFT bandwidth number.

## 7. Reproduce

```bash
# Correctness (thread-count invariance on the Tier-1 fixtures):
./scripts/p2-05-thread-sweep.sh

# Tier-1 scaling sweep on an idle bench box (repeat the second line per N):
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench -- --save-baseline t1
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=8 \
  cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench -- --baseline t1

# NUMA first-touch (2-socket box, P2-03):
cargo build -p aleph-benches --features "scaling-bench numa" --release
```

Measure only on a **verified-idle** box (`uptime` ≈ 0, no competing
`cargo bench`/runner jobs; idle-check per CLAUDE.md and `phase2-p2-01.md` §1);
deliver code to the self-hosted EPYC runner via `git bundle`, not a GitHub push,
to avoid racing the CI Bench job (technique recorded in `phase2-p2-04.md`).
