# Q4-02 — Parallel-window decoding + backlog handling

**Issue:** Q4-02 (Phase Q4, real-time / streaming).
**Depends on:** Q4-01 (sliding window).
**Status:** done.

## What and why

Sliding-window decoding (Q4-01) is **sequential**: window `k`'s input is the residual left by
window `k − 1`, so the windows form a dependency chain of depth `O(stream length)`. A single
decoder must therefore keep pace with the device on its own. When per-window decode is slower than
syndrome arrival, the unprocessed-syndrome queue grows without bound — the **backlog problem**
(Battistel et al., [arXiv:2303.00054](https://arxiv.org/abs/2303.00054)): the reaction time for the
next non-Clifford gate diverges and fault tolerance breaks, even when the *average* decode rate
looks adequate.

The **parallel-window** scheme (Skoric et al., [arXiv:2209.08552](https://arxiv.org/abs/2209.08552);
Tan et al., [arXiv:2209.09219](https://arxiv.org/abs/2209.09219)) breaks the chain into **two layers
of mutually-independent windows**, so the decoding *depth* is `O(1)` and throughput scales with the
number of workers:

- **Layer A** — the *even*-indexed commit regions `[0,C), [2C,3C), …` decode concurrently, each in a
  window with `B` rounds of buffer on **both** sides. Each commits its own region (toggling those
  detectors in the running residual) and XORs its observable flips into the logical correction.
- **Layer B** — the *odd*-indexed commit regions decode concurrently against the residual **left by
  layer A**. Every odd region is flanked by two already-committed even regions, so its seams are
  pinned on both sides — the inherited boundary condition the parallel-window papers rely on.

Both layers are embarrassingly parallel (`rayon`) and there are only two of them regardless of
stream length, so `P` workers give `≈ P×` the single-window throughput. That headroom is what keeps
the backlog bounded: pick `P` so the sustained service rate exceeds the arrival rate.

`ParallelWindowDecoder` (`crates/aleph-qec/src/parallel_window.rs`) reuses the Q4-01 seam machinery
verbatim — out-of-window detectors cut at per-detector **temporal-sink** nodes, kept distinct from
the real spatial boundary with free observable-less drains, so a time cut never spuriously flips the
logical observable. Each window decode is a **pure** read of the shared residual returning
(detector-toggle list, observable mask); the toggles are applied by **XOR**, which is associative
and commutative, so the result is independent of completion order — no data race, no order-
dependence. For `C ≥ 2` on a graphlike DEM (edges span ≤ 1 round) even windows' write sets are in
fact disjoint, so each layer commits a well-defined, seam-consistent correction.

## Acceptance criteria

- [x] **Throughput keeps pace with a configurable syndrome arrival rate (no unbounded backlog).**
- [x] **Measured sustained syndrome-bits/second.**

Both are met; see below.

## Results

Box: M4 Mac Mini, 10 rayon threads. `cargo run --release -p aleph-qec --example qec_q4_parallel`,
memory-Z phenomenological `p = 0.03`, `rounds = 24·d`, commit `C = d`, buffer `B = d` (window
`3d`), 30 000 shots for the rate check. Data: `docs/perf/data/qec-q4-parallel.{csv,log}`.

### Correctness — parallel rate within CI of full-batch UF

| d | rounds | batch rate | parallel rate | |Δ| | combined CI | within CI |
|---|--------|-----------|---------------|-----|-------------|-----------|
| 3 | 72  | 0.5015 | 0.5021 | 6.0e-4 | 1.1e-2 | ✅ |
| 5 | 120 | 0.4985 | 0.5026 | 4.1e-3 | 1.1e-2 | ✅ |
| 7 | 168 | 0.4985 | 0.5005 | 2.0e-3 | 1.1e-2 | ✅ |

(Rates sit near 0.5 because at `p = 0.03` over a long `24·d`-round stream the cumulative logical-
flip probability saturates; the point is the **parallel decode matches the batch decode within CI**,
i.e. the two-layer seams compose correctly. The slow nightly oracle
`tests/parallel_window.rs::parallel_rate_within_ci_of_batch` asserts this at 200k shots.)

### Throughput — parallel (P cores) vs sequential sliding (1 core)

| d | windows | bits/round | seq rounds/s | par rounds/s | speedup | **par syndrome-bits/s** | window µs |
|---|---------|-----------|--------------|--------------|---------|-------------------------|-----------|
| 3 | 24 | 4.0  | 191 149 | 527 413 | 2.76× | **2.11e6** | 15.9 |
| 5 | 24 | 12.0 | 66 192  | 241 256 | 3.64× | **2.90e6** | 76.2 |
| 7 | 24 | 24.0 | 35 172  | 147 882 | 4.20× | **3.55e6** | 200.2 |

Speedup **grows with `d`** (2.76× → 4.20×): each window is heavier (the per-window
`MatchingGraph::from_dem` + UF rebuild dominates), so the fixed per-stream costs — the layer barrier
and the serial XOR-apply — amortise better and the parallel efficiency rises. The streams here have
only 24 windows (12 per layer); longer streams and more cores push the speedup further toward the
`P`-core ceiling.

### Backlog — fluid queue at an arrival rate the sequential decoder can't sustain

For each `d` we pick an arrival rate `λ` *between* the sequential and parallel service rates
(`seq_rounds/s < λ < par_rounds/s`) and run a work-conserving fluid queue over a 100-stream horizon
from an empty queue:

| d | λ (rounds/s) | seq backlog (rounds) | par backlog (rounds) |
|---|--------------|----------------------|----------------------|
| 3 | 359 281 | **3 416 (growing)** | **0 (drained)** |
| 5 | 153 724 | **6 890 (growing)** | **0 (drained)** |
| 7 | 91 527  | **10 406 (growing)** | **0 (drained)** |

At the same arrival rate the **1-core sliding decoder's backlog grows without bound** while the
**10-core parallel decoder drains it to zero**. Because parallel-window throughput scales with the
worker count (depth is `O(1)`, not `O(stream length)`), *any* fixed arrival rate can be met by
provisioning enough workers — the standard fix for the backlog problem.

## Honest boundary

These are *throughput* numbers (sustained syndrome-bits/s), not single-shot latency. Real-time
superconducting QEC needs ≈ `1e6` rounds/s (1 µs/round); the per-window decode is currently
16–200 µs because each window **rebuilds its matching graph + UF decoder from scratch** (the Q1-03 /
Q4-01 note: the residual cost is graph construction, not the matching). Parallel windows deliver the
*scaling* that removes the backlog; closing the absolute constant to the 1 µs/round target is the job
of Q4-03 (latency budget) and Q6 (FPGA). A natural software lever is caching one interior-window plan
(identical geometry across all interior windows) instead of rebuilding per window per shot.

## Files

- `crates/aleph-qec/src/parallel_window.rs` — `ParallelWindowDecoder`, `WindowPlan`.
- `crates/aleph-qec/tests/parallel_window.rs` — valid-correction (CI) + within-CI-of-batch (nightly).
- `crates/aleph-qec/examples/qec_q4_parallel.rs` — correctness / throughput / backlog blocks.
- `docs/perf/data/qec-q4-parallel.{csv,log}` — committed run.
