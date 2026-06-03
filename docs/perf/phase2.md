# Phase 2 — Multi-Threaded CPU — Scaling-Efficiency Report

**Phase:** 2 (multi-threaded CPU) · **Issue:** #31 (P2-05) · **Date:** 2026-06-02
**Consolidates:** P2-01 (#27), P2-02 (#28), P2-03 (#29), P2-04 (#30).
**Bench:** `benches/benches/tier1_scaling.rs` (feature `scaling-bench`).
**Raw data:** `docs/perf/data/p2-05-tier1-scaling.log` (live EPYC + NUMA sweep, 2026-06-02).
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
project, and cannot be on this hardware regardless of code.** A live `tier1_scaling`
sweep (§3) puts 16-thread efficiency at **14–22 % across GHZ/QFT/random on two
independent AVX-512 boxes** (EPYC single-socket and a 2-socket Xeon); the
P2-01 `qft_scaling` builder circuit measured 23 % @16t (§2). The same saturating
shape reproduces on a third CPU (Ryzen, scalar path), and the 2-socket box even
*regresses* past one socket (§3) — every angle agrees.

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

## 3. Full Tier-1 matrix — `tier1_scaling`, live sweep

The `tier1_scaling` bench (this PR) measures GHZ / QFT / random at n=25 through the
AVX-512 `NaiveSvBackend`, swept across `RAYON_NUM_THREADS` on two verified-idle
boxes (2026-06-02). Each cell is `speedup×` vs that box's T1, with the **16-thread
efficiency** (the ROADMAP target column) called out. **Grover is excluded** and
left pending — see the box below.

> **These are a *different* QFT circuit than §2.** §2 reports P2-01's `qft_scaling`,
> built by the Rust `qft_circuit(n)` (~325 gates). This bench parses the **Aer-
> comparable `qft_n25.qasm` fixture** (1526 ops — Qiskit decomposes every
> controlled-phase + adds the final swaps), so its absolute times are ~4× larger
> *and its speedup is lower* (EPYC 2.16×@8 vs the builder's 3.37×@8). That is not a
> contradiction: the fixture QFT is ≈92% controlled-phase, the **lowest-arithmetic-
> intensity** gate in the set, so it saturates memory bandwidth even harder. The
> two QFT families bracket the bandwidth-bound regime; neither is "wrong".

### EPYC 8124P (AVX-512, 16c/32t), raw `run`

| Workload | T1 | T2 | T4 | T8 | T16 (eff) | T32 |
|---|---:|---:|---:|---:|---:|---:|
| GHZ    | 1.47 s | 1.63× | 2.38× | 3.15× | 3.59× (**22%**) | 3.59× |
| QFT    | 34.5 s | 1.52× | 1.93× | 2.16× | 2.30× (**14%**) | 2.34× |
| random | 39.5 s | 1.67× | 2.23× | 2.54× | 2.73× (**17%**) | 2.77× |

### NUMA 2× Xeon 4114 (AVX-512, 20c/40t, **2 sockets**), raw `run`

| Workload | T1 | T2 | T4 | T8 | T16 (eff) | T32 | T40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GHZ    | 2.14 s | 1.52× | 2.19× | 2.57× | 2.64× (**17%**) | 2.68× | 2.65× |
| QFT    | 78.7 s | 1.49× | 2.06× | 2.14× | **2.27×** (**14%**) | 2.18× | 2.16× |
| random | 76.3 s | 1.64× | 2.35× | 2.38× | 2.33× (**15%**) | 2.30× | 2.29× |

Three findings, all reinforcing §1:

1. **16-thread efficiency is 14–22% on both AVX-512 boxes** — nowhere near the
   ≥75% / ≥12× exit. The lowest-intensity workload (QFT, ≈92% cphase) scales
   worst (2.16–2.27×@16); higher-intensity random and the gate-light GHZ scale a
   little better, exactly as a bandwidth-bound model predicts.
2. **NUMA shows a textbook cross-socket regression.** QFT peaks at **2.27×@16**
   then *goes backwards* — 2.18×@32, 2.16×@40 — as threads spill onto the second
   socket. The default allocator faults the whole 512 MiB state onto node 0, so
   socket-1 threads pay remote-memory latency and *subtract* throughput. This is
   precisely the failure mode P2-03's first-touch allocation removes (§4.3): with
   the `numa` feature the same box gains −37.7% (§2). Default-allocator scaling
   saturates at one socket's bandwidth.
3. **GHZ plateaus at an allocation floor, not a bandwidth one.** Its time bottoms
   out at ~0.41 s (EPYC) / ~0.80 s (NUMA) and stops improving (T16==T32); that
   floor is one-time state allocation + the 25-gate body, not kernel throughput
   (§4.5). Its "3.59×" is an allocation artifact, not a parallel-scaling signal.

> **Grover is pending — and intractably so at low thread counts.**
> `grover_n25_iters5.qasm` decomposes into thousands of multi-controlled gates;
> its **single-thread baseline is ≈13 CPU-hours** (criterion's measured estimate:
> 48 286 s for the 10-sample floor). A T1-anchored efficiency sweep is therefore
> not feasible in a normal measurement window, and no fabricated number is
> substituted. Follow-up §6.3: measure Grover with a reduced harness (1 sample,
> high-thread-only, or a smaller iteration count) on a dedicated long run.
>
> **Ryzen** was unavailable for a clean sweep this round — its RAID10 array was
> mid-resync (degraded, `[2/1]` mirrors), so the box was not idle. The report
> keeps its earlier-measured P2-02 QFT scalar number (§2).

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
GHZ-25 is 1 H + 24 CNOT = **25 gates total**. At n=25 each gate still streams the
full 512 MiB state, so a single run is ~1.5 s (EPYC) / ~2.1 s (NUMA) — not
milliseconds — but as threads increase the gate body parallelizes until the time
hits a **one-time state-allocation floor** (~0.41 s EPYC / ~0.80 s NUMA, measured
§3) and stops improving (T16==T32). Its apparent "3.59×" is dominated by that
fixed allocation cost, not gate-kernel throughput, so it is an allocation-bound
artifact rather than a bandwidth-scaling signal. Included for spec completeness
and annotated as such — never reported as a meaningful parallel-efficiency point.

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
3. **Measure Grover** — the one remaining `pending` cell (§3). `grover_n25_iters5`
   is ≈13 CPU-h single-threaded, so it needs a reduced harness (fewer samples /
   high-thread-only / smaller iteration count) on a dedicated long run, and a
   clean Ryzen scalar sweep once its RAID resync completes. GHZ/QFT/random are
   now measured on EPYC + NUMA (§3).
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

-----

## 8. P2-06 — diagonal-run fusion (issue #106)

`FuseDiagonalRuns` (a new IR pass) collapses a maximal run of `{diagonal gates
∪ cx}` into a single `Instruction::DiagonalPhase` applied in **one** streaming
state-vector pass. It tracks a GF(2) bit-permutation across the run, so the
interleaved `cx`s of a *decomposed* QFT are **absorbed** (monomial algebra:
`cx·D·cx` is diagonal) rather than breaking the run. Both QFT encodings
therefore collapse: the builder controlled-`Phase` ladder and the
Aer-comparable `qft_n25.qasm` fixture (lowered to `p`+`cx`). The fused operator
is a symbolic list of `(AND-of-parity-masks → angle)` terms — no `2^n` storage —
applied by a scalar + AVX-512 (`VPOPCNTQ` parity + scalar-extract sincos) kernel
in AoS and SoA. Correctness is gated at 1e-12 (global phase included) by the
`diagonal_fusion_oracle` equivalence tests on a generic input state, validated
on the EPYC AVX-512 box.

### Instruction-count (memory-pass) reduction

The acceptance target was ≥ 5× fewer gate-passes on QFT-25. Measured (gates
remaining after the pipeline, without vs with `FuseDiagonalRuns`):

| circuit              | without | with | reduction |
|----------------------|--------:|-----:|----------:|
| builder QFT n=20     |     210 |   39 |   5.38×   |
| builder QFT n=22     |     253 |   43 |   5.88×   |
| builder QFT n=25     |     325 |   49 |   6.63×   |
| fixture QFT n=25     |     300 |   47 |   6.38×   |

The remaining ~`n`+`n` instructions are the `n` Hadamards (non-diagonal, one
pass each) plus the `n` per-level fused diagonals — exactly the `≈ 2n`-pass
lower bound the design predicted.

### Wall-clock (EPYC 8124P, AoS + AVX-512 `NaiveSvBackend`, idle-verified)

End-to-end `run` time of the pipeline-optimized circuit, without vs with the
diagonal pass (criterion, `sample_size = 10`, `target-cpu=native`):

| circuit                       | without   | with      | speedup |
|-------------------------------|----------:|----------:|--------:|
| builder QFT n=22              | 205.2 ms  | 111.9 ms  | 1.83×   |
| builder QFT n=25              | 2.164 s   | 1.068 s   | 2.03×   |
| **fixture QFT n=25 (`p`+`cx`)** | 3.703 s   | 1.121 s   | **3.30×** |

The decomposed fixture — the workload the acceptance criteria name — gets the
largest speedup (**3.30×**): its `cx`+`p` decomposition carried the most
redundant full-state passes, all absorbed into the per-level diagonals.

Wall-clock speedup (2–3.3×) is smaller than the pass-count reduction (≈ 6×)
because the surviving Hadamard passes are pure memory streams while each fused
`DiagonalPhase` pass now does more arithmetic per amplitude (term evaluation).
Both are bandwidth-bound; the win is fewer DRAM streams over the `2^n` state,
exactly the Phase-2 thesis (§1). The AVX-512 kernel matched the scalar kernel
bit-for-bit on the equivalence tests; with `target-cpu=native` the `with`
column runs the SIMD path.

### Reproduce

```bash
# idle-verified EPYC box; deliver via git bundle (not a GitHub push):
RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p aleph-benches --bench diagonal_fusion --features scaling-bench
# the instruction-count reduction table prints to stderr at startup.
```

-----

## 9. P2-07 — deep k-qubit gate fusion (issue #107)

`FuseKq` (a new IR pass, runs **last** in `default_pipeline()`) greedily merges
chains of the dense 1q/2q blocks the earlier passes produced (`Unitary1q`,
`Unitary2q`) into a single dense `Gate::UnitaryKq` over ≤ `max_qubits` qubits,
applied in **one** state-vector pass by a scalar + AVX-512 generic-`k` matvec
kernel (renormalised outer-walk generalized from P1-07). Cost model: only a
block spanning **≥ 3 qubits** that absorbed **≥ 2 gates** becomes dense — so 1q/2q
and lone `Cnot`/`Cz`/`Swap` keep their specialized kernels (no regression on
cheap gates). Unlike P2-06's diagonal kernel (pure memory streaming), the dense
matvec does `2^k` complex-muls per amplitude — it **raises arithmetic
intensity**, the regime where AVX-512 actually pays.

### Instruction-count (memory-pass) reduction (EPYC, `max_qubits = 4`)

| workload (n=25)       | without | with | reduction |
|-----------------------|--------:|-----:|----------:|
| random brick-wall     |     241 |  129 |   1.87×   |
| QAOA-like             |     167 |   72 |   2.32×   |
| VQE-like              |     156 |  156 |   1.00×   |

VQE shows **no** `FuseKq` reduction — and that is correct: its entangler is `Cz`,
which is **diagonal**, so the whole CZ-ladder + `Rz` structure is already
collapsed into a `DiagonalPhase` by `FuseDiagonalRuns` (P2-06) before `FuseKq`
runs. `FuseKq`'s wins are specifically on **non-diagonal** (`Cnot`-entangled)
fusible circuits, which `FuseDiagonalRuns` cannot touch.

### Wall-clock (EPYC 8124P, AoS + AVX-512 `NaiveSvBackend`, idle-verified)

`run` of the pipeline-optimized circuit, without vs with `FuseKq` (criterion,
`sample_size = 10`, `target-cpu=native`):

| workload              |   without |     with | speedup |
|-----------------------|----------:|---------:|--------:|
| random n=22           | 608.4 ms  | 368.5 ms | 1.65×   |
| random n=25           | 5.075 s   | 2.688 s  | **1.89×** |
| QAOA n=22             | 375.0 ms  | 137.3 ms | 2.73×   |
| QAOA n=25             | 3.136 s   | 1.337 s  | **2.34×** |
| VQE n=25              | 1.887 s   | 1.887 s  | 1.00×   |

Wall-clock tracks the pass-count reduction closely (random 1.89× vs 1.87× count;
QAOA 2.34× vs 2.32×) — because dense fusion raises arithmetic intensity, the
extra per-amplitude FLOPs are not the bottleneck; fewer DRAM streams over the
`2^n` state is. The scalar↔AVX-512 kernel matched bit-for-bit on the equivalence
tests (EPYC); with `target-cpu=native` the `with` column runs the SIMD path.

### `max_qubits` sweep → default = 4

QAOA-25, `FuseKq { max_qubits }` then `run`:

| max_qubits | time     |
|-----------:|---------:|
| 2          | 3.126 s  | (≈ no fusion — span < 3 never fuses) |
| 3          | 1.903 s  |
| **4**      | **1.337 s** |
| 5          | 1.446 s  |

`max_qubits = 4` wins; **`k = 5` is slower** — the 32×32 dense block's
`2^k`-per-amplitude FLOP cost outweighs the marginal extra pass reduction. The
`FuseKq::default()` cap is therefore **4**, empirically confirmed (the same
data-driven approach P2-04 used for grain).

### Reproduce

```bash
# idle-verified EPYC box; deliver via git bundle (not a GitHub push):
RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p aleph-benches --bench fuse_kq --features scaling-bench
# the instruction-count reduction table prints to stderr at startup.
```

-----

## 10. P2-08 — optional FP32 (single-precision) state-vector mode (issue #108)

A dedicated `Fp32SvBackend` (AoS, `AlignedBuf<Complex<f32>>` = 8 B/amp vs 16)
alongside `NaiveSvBackend`. Scalar f32 kernels cover every gate; f32 AVX-512
kernels (16 f32 lanes/zmm vs 8 f64) accelerate the fused hot-types the
optimized pipeline emits — generic dense 1q, diagonal 1q, generic dense 2q,
`UnitaryKq` (k ≤ 4), and `DiagonalPhase`. Opt-in via CLI `--precision f32`;
**FP64 stays the default and the 1e-10 oracle reference, byte-for-byte
unchanged**. The f32 backend is oracle-validated against the exact Aer
fixtures at **1e-5** (30 fixtures) and SIMD≡scalar-validated on EPYC.

### Wall-clock — fused `run`, EPYC 8124P (AVX-512, 16c/32t, all-core), idle-verified

Both columns time the **fused** path (`circuit.optimize()` once, outside the
timed loop; `sample_size = 10`, `target-cpu=native`). f32/f64 ratio is the
single-precision speedup:

| workload                    |       f64 |      f32 | speedup |
|-----------------------------|----------:|---------:|--------:|
| QFT n=22                    | 111.5 ms  |  86.0 ms | 1.30×   |
| QFT n=25                    | 1.066 s   | 800.2 ms | 1.33×   |
| random brick-wall n=22      | 516.1 ms  | 268.8 ms | **1.92×** |
| random brick-wall n=25      | 3.724 s   | 2.447 s  | **1.52×** |

### Reading the result — workload-dependent, and why

The **~1.5–2× AC is met on dense workloads** (random brick-wall: 1.52× at n=25,
1.92× at n=22) but **not on QFT** (1.33×). The split is structural, not a code
defect:

- **Fused QFT is `DiagonalPhase`-dominated** (P2-06 collapses the cphase ladder
  into one diagonal per run). The f32 `DiagonalPhase` kernel halves the
  amplitude byte traffic but computes the per-index rotation angle and its
  `sin_cos` in **f64** — angle precision is deliberately preserved (only the
  final `(cos, sin)` is narrowed to f32 before the amplitude multiply). That
  transcendental work is **precision-independent**, so it is a fixed floor the
  f32 mode cannot shave: only the bytes halve, not the compute. Result: 1.33×,
  not 2×.
- **Fused random brick-wall is dense-`Unitary1q`/`Unitary2q`-dominated.** Those
  f32 AVX-512 kernels get **both** levers: half the byte traffic **and** 16
  f32 lanes/zmm (vs 8 f64) of arithmetic. Hence the win lands in the AC band.
  The n=22 figure (1.92×) exceeds n=25 (1.52×) because at n=22 the kernel is
  less purely DRAM-bound, so the doubled SIMD compute width contributes more;
  at n=25 the state (256 MiB f32 / 512 MiB f64) is deeply DRAM-bound and the
  ratio trends toward the pure byte-traffic limit minus precision-independent
  index overhead.

This is consistent with the Phase-2 through-line (§4): these kernels are
memory-bandwidth-bound, so halving bytes/amp is the dominant lever **except**
where a precision-independent compute floor (here, `DiagonalPhase`'s `sin_cos`)
intrudes. Accuracy held at the FP32 oracle tolerance (1e-5) on every fixture;
the largest observed f32-vs-f64 amplitude deviation on the Tier-1 equivalence
fixtures was **1.37e-7**, three orders under the 1e-5 bound.

**Possible follow-up** (not in this ticket): a faster vectorized f32 `sin_cos`
(or a small precomputed phase LUT) in the `DiagonalPhase` kernel would attack
the QFT floor — at some accuracy cost that must stay within the 1e-5 oracle.

### Reproduce

```bash
# idle-verified EPYC box; deliver via git bundle (not a GitHub push):
RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p aleph-benches --bench qft_precision --features scaling-bench
# emits qft_precision/{f64,f32}/{22,25} and random_precision/{f64,f32}/{22,25}.
```
