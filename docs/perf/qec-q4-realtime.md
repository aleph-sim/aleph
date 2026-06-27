# Q4-03 — Real-time latency budget

**Issue:** Q4-03 (Phase Q4, real-time / streaming).
**Depends on:** Q4-01 (sliding window).
**Status:** done.

## What and why

The North Star is a real-time decoder that keeps pace with the device: superconducting QEC measures
a syndrome round roughly every **1 µs**, so a decoder that wants to run forever without a growing
backlog (Q4-02) must commit each round in **< 1 µs**. This note instruments where the per-round
decode time actually goes, stage by stage, and measures the gap to that 1 µs target — the gap that
motivates the Q6 FPGA decoder.

The unit of work is **one streaming window**. The Q4-01/Q4-02 window decoders take a window of
`W = 3d` rounds (commit `C = d`, buffer `d` each side), and each decode does four things:

1. **graph build** — `MatchingGraph::from_dem` + `UnionFindDecoder::from_graph` for the window DEM,
   rebuilt per window per shot;
2. **growth** — Union-Find cluster growth (the matching);
3. **peel** — spanning-forest peel of the erasure into a correction;
4. **commit** — XOR the chosen edges' observable flips into the running residual.

A window commits `C = d` rounds, so **per-round latency = window decode time / d**.
`examples/qec_q4_latency.rs` times each stage over 4000 DEM-Bernoulli syndromes (`p = 0.03`) and
reports the distribution; `UnionFindDecoder::decode_edges_timed` splits growth from peel without
touching the hot `decode_edges` path.

## Acceptance criteria

- [x] **Per-stage latency histogram (graph build, growth, peel, commit) for d ∈ {5,7,9,11}.**
- [x] **`docs/perf/qec-q4-realtime.md` with the budget and the gap to 1 µs (motivates Q6 FPGA).**

## Results

Box: M4 Mac Mini (single thread; latency, not throughput). Memory-Z phenomenological `p = 0.03`,
window `W = 3d`, 4000 samples. Data: `docs/perf/data/qec-q4-latency.{csv,log}`.

### Per-stage window latency (median µs, with p99)

| d | window W | build p50 | growth p50 | peel p50 | commit p50 | **total p50** | total p99 |
|---|----------|-----------|------------|----------|------------|---------------|-----------|
| 5  | 15 | 54.0  | 4.08  | 1.33  | 0.041 | **59.6**  | 66.2  |
| 7  | 21 | 138.7 | 11.67 | 3.79  | 0.083 | **154.1** | 167.7 |
| 9  | 27 | 299.0 | 26.79 | 8.67  | 0.166 | **334.7** | 357.4 |
| 11 | 33 | 554.0 | 54.50 | 16.58 | 0.250 | **625.8** | 661.3 |

Full p50/p90/p99/max per stage is in `qec-q4-latency.csv`.

### Budget vs the 1 µs/round target

| d | total p50 (µs) | **per-round (µs)** | gap to 1 µs |
|---|----------------|--------------------|-------------|
| 5  | 59.6  | 11.9 | **12×** |
| 7  | 154.1 | 22.0 | **22×** |
| 9  | 334.7 | 37.2 | **37×** |
| 11 | 625.8 | 56.9 | **57×** |

## Reading the budget

- **Graph build dominates — ~90% of every window** (54.0 / 59.6 at d=5; 554.0 / 625.8 at d=11). It
  is pure overhead from the streaming decoders' decision to **rebuild the matching graph + UF per
  window per shot** (the Q1-03 / Q4-01 / Q4-02 wall). The window geometry is *identical* across all
  interior windows, so this is amortisable: building one interior-window plan once and reusing it
  would erase this stage. That is the single biggest software lever and the obvious Q4 follow-up.
- **Growth is the real algorithmic cost** and the stage that actually scales with the code: 4 µs
  (d=5) → 54 µs (d=11), roughly `∝ d³` (defect count `∝ d²`, growth rounds `∝ d`). Peel is ~¼ of
  growth. Commit is negligible (< 0.3 µs).
- **Amortised lower bound.** Strip the rebuild and per-round latency becomes `(growth + peel +
  commit)/d`: **1.09 µs (d=5)**, 2.2 (d=7), 4.0 (d=9), 6.5 (d=11). So at d=5 the matching work alone
  is *already at* the 1 µs line on one CPU core; the budget is blown by redundant graph construction,
  not by the matching. At larger d the growth stage itself exceeds the budget and must be
  parallelised.

## The gap to 1 µs — why Q6 (FPGA)

Even fully amortised, a CPU core hits ~1 µs/round only at d=5 and falls behind as `d` grows (growth
is serial cluster expansion). Closing the gap at useful distances needs what an FPGA/ASIC gives:

- **build-free, fixed-array pipeline** — the decode graph is baked into the fabric (no per-window
  `from_dem`), removing the dominant 90% stage outright;
- **spatially-parallel cluster growth** — grow all clusters' frontiers in one cycle instead of a
  serial vertex scan, collapsing the growth stage that dominates the amortised budget;
- **deterministic, jitter-free latency** — the p99/p50 spread here is ~1.1× (GC-free Rust already),
  but hardware removes OS scheduling tails entirely, which matters for a hard real-time deadline.

This is exactly the Q6 (FPGA) → Q7 (ASIC) thesis from the roadmap: the integer-only, fixed-array,
near-linear weighted-UF decoder (Q2-03's recommendation) is the hardware-shaped algorithm; this
budget quantifies the 12–57× CPU gap it has to close.

## Files

- `crates/aleph-qec/src/union_find.rs` — `UnionFindDecoder::decode_edges_timed` (instrumented, opt-in).
- `crates/aleph-qec/examples/qec_q4_latency.rs` — per-stage histogram + budget.
- `docs/perf/data/qec-q4-latency.{csv,log}` — committed run.
