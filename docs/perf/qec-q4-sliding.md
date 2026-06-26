# Q4-01 — Sliding-window streaming decoder

Phase Q4 is the shift from **offline batch** decoding to **real-time streaming**: a quantum computer
emits syndromes forever, so you cannot wait for the end of the experiment to decode. The standard
real-time approach (Dennis et al.; Skoric et al. / Tan et al.) is the **sliding window** — decode a
window of `W` consecutive rounds, commit the correction for the first `C < W` rounds, slide forward
by `C`, and let the trailing `W − C` rounds (the *buffer*) supply the future context the commit region
needs. Correctness at the window seams is the hard part.

[`SlidingWindowDecoder`] implements this over the Union-Find decoder, on a memory-Z syndrome stream of
arbitrary length.

**Verdict — both acceptance criteria met:**

- **Logical error rate within CI of full-batch decoding for an adequate window.** On long streams the
  sliding-window rate converges to the batch UF rate from above and is **within the combined 95 % CI
  once the buffer `W − C ≳ d`** — at the threshold (`p = 3 %`), within CI from buffer ≥ 3 for every
  `d ∈ {3,5,7}` (and from buffer 1 at `d=3`).
- **Unbounded stream in bounded memory.** Each window's working set is `O(W)` — independent of the
  total number of rounds — so an endless stream decodes with bounded memory.

## The seam: why a naive cut flips the logical observable

The subtlety the backlog flags ("correctness near window seams is the hard part") is real and bit us:
in a memory-Z DEM every **observable-flipping mechanism is a *spatial* detector↔boundary edge**. If a
window cuts a *time-like* measurement edge and collapses its out-of-window endpoint onto the same
shared boundary, the DEM builder **merges** it with the real observable edge at that detector — and a
harmless "carry this chain forward in time" match then **spuriously flips the logical observable**.
With that bug the streaming rate was 5–10× the batch rate and never converged.

The fix: route every out-of-window detector to its **own temporal-sink node**, distinct from the
spatial boundary, with a free observable-less drain. Time cuts then never touch an observable edge.
A running **residual** carries committed corrections across seams (a committed edge is applied to the
syndrome, toggling its real detectors), so the per-window decodes compose into one valid global
correction — decoding windows independently does not.

## Convergence to batch (threshold, p = 3 %, commit C = d, 40 000 shots)

Long memory-Z stream of `rounds = 6d`. `Δ` is `|sliding − batch|`; `within CI` is `Δ ≤ ci_batch +
ci_sliding`. `window_dets` is the per-window working set (bounded).

| `d` | rounds | `W` | buffer `W−C` | batch rate | sliding rate | Δ | within CI | window dets |
|--:|--:|--:|--:|--:|--:|--:|:--:|--:|
| 3 | 18 | 4 | 1 | 0.3663 | 0.3700 | 0.0036 | ✓ | 16 |
| 3 | 18 | 6 | 3 | 0.3663 | 0.3650 | 0.0014 | ✓ | 24 |
| 3 | 18 | 12 | 9 | 0.3663 | 0.3667 | 0.0003 | ✓ | 48 |
| 5 | 30 | 6 | 1 | 0.4101 | 0.4220 | 0.0119 | ✗ | 72 |
| 5 | 30 | 8 | 3 | 0.4101 | 0.4139 | 0.0038 | ✓ | 96 |
| 5 | 30 | 10 | 5 | 0.4101 | 0.4117 | 0.0016 | ✓ | 120 |
| 5 | 30 | 20 | 15 | 0.4101 | 0.4114 | 0.0013 | ✓ | 240 |
| 7 | 42 | 8 | 1 | 0.4481 | 0.4621 | 0.0141 | ✗ | 192 |
| 7 | 42 | 11 | 4 | 0.4481 | 0.4519 | 0.0039 | ✓ | 264 |
| 7 | 42 | 14 | 7 | 0.4481 | 0.4490 | 0.0010 | ✓ | 336 |
| 7 | 42 | 28 | 21 | 0.4481 | 0.4499 | 0.0019 | ✓ | 672 |

(Full sweep incl. all `W` in [`data/qec-q4-sliding.csv`](data/qec-q4-sliding.csv).)

**Reading it.** A one-round buffer (`W = d+1`) is biased high — the commit region has too little
future context, so seam errors inflate the rate by ~0.012–0.014. By **buffer ≥ 3** the bias falls
inside the CI at every distance, and it keeps shrinking with `W` (to ~3 × 10⁻⁴ at `d=3`, buffer 9).
At `W` equal to the whole stream the decoder is exactly a batch decode (a lib unit test asserts
bit-equality). Below threshold the convergence is the same shape but needs a slightly larger buffer
at `d=7` (≈ 2d), as expected for the lower error density. So the **`d`-dependent window bound is
`W ≳ 2d` (buffer ≳ d)** — consistent with the literature.

## Bounded memory

The `window_dets` column is the largest number of detectors any single window spans — the per-window
working set. It depends only on `W` (and the code), **not** on the stream length: a `tests/`
assertion confirms a `d=5, W=3d` decoder spans the same per-window detector count whether the stream
is 20 or 80 rounds. Because each window is built, decoded and discarded as the decoder slides, an
unbounded stream is decoded in `O(W)` memory. (The residual is kept as a full array here for
simplicity; only the live window's `O(W·nz)` entries are ever in play, so it compresses to a ring
buffer in a true endless-stream deployment.)

## Reproduce

```bash
# Convergence sweep (CSV: batch vs sliding rate, within-CI flag, working set), p configurable:
cargo run --release -p aleph-qec --example qec_q4_sliding -- 40000 2024 0.03

# Fast CI guards (full-window == batch exactly; bounded working set):
cargo test -p aleph-qec --test sliding_window
# Thorough statistical match (nightly; minutes):
cargo test -p aleph-qec --test sliding_window -- --ignored
```

**Machine.** Apple M4, rustc `--release`. Seed 2024, deterministic.

## What this validates + limits

- A streaming decoder that reproduces the offline (batch) logical accuracy for an adequate window and
  runs on an unbounded stream in bounded memory — the Phase-Q4 entry point.
- **Latency, not yet addressed here.** This establishes *correctness* of windowed streaming; it does
  not yet bound per-round decode *latency* or handle the **backlog problem** (if decode is slower than
  syndrome arrival the queue grows unboundedly and fault tolerance breaks). That is [Q4-02]
  (parallel-window decoding + sustained throughput vs arrival rate), building on this.
- The per-window decode currently rebuilds its sub-graph each slide; a steady-state decoder reuses one
  interior-window plan (the bulk is time-translation-invariant), the obvious optimisation toward a
  real-time latency budget.

## References

- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, **Topological quantum memory**, J. Math. Phys. 43
  (2002) — windowed surface-code decoding.
- L. Skoric, D. E. Browne, K. M. Barnes, N. I. Gillespie, E. T. Campbell, **Parallel window decoding**,
  Nature Communications 14, 7040 (2023).
- X. Tan, F. Zhang, R. Chao, Y. Shi, J. Chen, **Scalable surface-code decoders with parallelization in
  time**, PRX Quantum 4, 040344 (2023).

[Q4-02]: ../qec/BACKLOG.md
[`SlidingWindowDecoder`]: ../../crates/aleph-qec/src/sliding.rs
