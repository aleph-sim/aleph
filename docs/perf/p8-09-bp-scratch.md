# P8-09: GPU BP scratch coalescing, with measured crossovers

## Result

Retain explicit `CudaBp::with_scratch_layout` choices for the original shot-major,
edge-major and 32-shot warp-tiled layouts. **`CudaBp::new` remains shot-major.**
The candidate improves some real-GPU batch regimes, but a universal replacement or
a batch-size-only automatic rule is contradicted by the negative controls.

On the shared RTX 4000 SFF Ada, warp-tiled BP improved the measured decode boundary
by **1.44×** at surface d11 /4096 shots /p=.03 and **1.93×** at d7 /16384 /.03.
At d7 /4096 /.001 it instead took **1.55× longer**; at d11 /4096 /.001 it took
**1.58× longer**. Smaller graphs and single-shot inputs also have losing regimes.
These are descriptive same-decoder measurements, not a logical-accuracy improvement
or a guarantee for other GPUs/graphs/traffic.

## Layout and safety contract

The Tanner graph, per-shot edge order, FP64 operations, alpha=.875, maximum 64
iterations, syndrome packing, early stop and output masks are unchanged. Each GPU
thread still decodes one complete shot. NVRTC selects the address formula at compile
time; there is no runtime branch per message and no cooperative scheduling or
precision change. The source body implements the same arithmetic for every layout.

For element `e`, shot `s`, logical current tile size `N`, and element count `E`:

| Layout | Scratch index | Allocated shot capacity |
|---|---|---|
| `ShotMajor` | `s*E + e` | N |
| `EdgeMajor` | `e*N + s` | N |
| `WarpTiled` | `(s/32)*E*32 + e*32 + s%32` | ceil(N/32)*32 |

The formula applies independently to both FP64 message arrays, hard decisions and
syndrome bits. Input packed words/output masks keep their public shot-major format.
Warp-tiled storage reduces the distance between a warp's successive graph rows;
partial warp padding is included in the budget. Each lane owns disjoint addresses.
`EdgeMajor` uses **current N**, not retained capacity: reusing a larger allocation
for a smaller decode does not reuse the wrong pitch.

Logical scratch bytes per shot are `16*edges + vars + checks + 8 + 4*words`.
The retained payload is `capacity * per_shot_bytes()` including input/output buffers.
For d7 this is **16,096 bytes/shot**, d11 **64,252**. A d7 batch of 4096 has
**65,929,216 bytes** in all layouts; a single warp-tiled shot reserves 32 logical
slots instead of one. These are device buffer payloads, not process/driver VRAM.

The 2 GiB scratch policy rounds the maximum warp-tile capacity **down** to a
multiple of 32 before splitting a batch. It rejects a graph whose single padded
unit cannot fit instead of forcing an oversized one-shot allocation. Old retained
buffers are released before growth so old+new payloads do not double that budget;
partial failed allocation state cannot masquerade as valid retained capacity.
Checked packed-word length rejects short/overflowed input before upload. Existing
prefix semantics are preserved: surplus words and nonempty input for zero shots
remain accepted. No intentional OOM or large new stress allocation was used.

Selection is explicit on Linux with the CUDA feature and a real device:

```rust
use aleph_cuda::{BpScratchLayout, CudaBp};
// `cpu` is the caller's configured aleph_qec::BpDecoder.
let gpu = CudaBp::with_scratch_layout(&cpu.tanner(), BpScratchLayout::WarpTiled)?;
let masks = gpu.decode(&syndromes)?;
// gpu.per_shot_bytes(), gpu.retained_scratch_bytes(), gpu.scratch_layout()
```

The original layout remains available as the numerical and performance reference.
No frontend routing, UF algorithm, cooperative BP kernel or FP32 path changes.

## Measurement protocol

User-authorized shared `openwebgui.splynx.com`, RTX 4000 SFF Ada (sm_89, 20,475 MiB),
driver 580.178.04, CUDA/NVRTC 13.0, Rust 1.96.0, Nsight Compute 2025.3.1. Initial GPU
utilization was zero; services stayed running. Clocks were not changed or locked.
The 315-second campaign retained 1,570 telemetry observations: SM clocks 1560–2340 MHz,
temperature 42–72 °C, used VRAM 1495–1916 MiB. GPU utilization includes this campaign.
This was not an exclusive/isolation-controlled card run; telemetry is context, not
proof that another tenant never ran. Balanced block ratios bound the observed
variation; they are not statistical confidence intervals.

`bp_layout_probe` builds the same phenomenological surface DEM and syndromes from
`StdRng` seed **559**, with rate p used for data/measurement mechanisms. It uses a
new process/decoder per invocation, allocates and warms up once, then times three
reused-buffer calls. The timed boundary is **packed-input upload + kernel + mask
download/synchronization**, excluding model construction, packing, compilation,
allocation warmup and process startup. It is not an entire Monte Carlo workflow.

The fixed campaign covers d={3,7,11}, shots={1,128,4096}, p={.001,.03}, plus six
large/odd/high-rate controls. Four mirrored three-arm blocks rotate the leading
arm; odd blocks reverse cell order. Each arm has 24 retained timed observations per
cell, grouped into four paired blocks. There are **624 invocations /1872 timings**,
including an original-kernel ABBA control. No best-of selection, result-based retries
or outlier removal. Every correction checksum agrees within its input cell.

| d | Shots | p | Shot-major ms | Edge speedup | Warp speedup | Warp four-block range |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 128 | .001 | .1154 | 1.080× | 1.088× | 1.068–1.092× |
| 3 | 128 | .03 | 2.2524 | .772× | .772× | .768–.796× |
| 3 | 4096 | .03 | 2.3689 | .782× | .788× | .768–.783× |
| 7 | 1 | .03 | 20.2429 | 1.020× | .631× | .590–.670× |
| 7 | 128 | .03 | 51.6645 | 1.233× | 1.245× | 1.241–1.246× |
| 7 | 4096 | .001 | 24.1112 | .633× | .643× | .643–.643× |
| 7 | 4096 | .03 | 74.7575 | 1.043× | 1.048× | 1.042–1.093× |
| 7 | 4096 | .15 | 87.4505 | 1.313× | 1.319× | 1.316–1.330× |
| 7 | 33 | .03 | 36.5596 | .967× | .974× | .974–.975× |
| 7 | 4097 | .03 | 74.5612 | .968× | 1.039× | 1.035–1.040× |
| 7 | 16384 | .03 | 367.6353 | 1.760× | 1.926× | 1.919–1.926× |
| 11 | 1 | .03 | 88.2292 | 1.019× | .694× | .694–.694× |
| 11 | 128 | .03 | 217.8672 | 1.394× | 1.405× | 1.404–1.408× |
| 11 | 4096 | .001 | 121.0652 | .624× | .631× | .630–.631× |
| 11 | 4096 | .03 | 412.3007 | 1.415× | 1.437× | 1.425–1.448× |
| 11 | 4096 | .15 | 412.6431 | 1.613× | 1.637× | 1.624–1.649× |

Speedup is median shot-major time / median candidate time; below one is slower.
Four-block ranges use each block's paired medians. Small differences between the
pooled ratio and range reflect grouping/drift, not removed observations. All **24**
cells, not just this compact table, are retained in the analysis JSON.

The truly unchanged kernel was built separately from base
`1de1494a3180e784817ac42039fb501e746465f6` with only a probe added. It matches the
compile-time `ShotMajor` reference: four control block ratios range .997–1.018
across three scoped cells. No material reference regression was observed in these controls;
they do not prove zero overhead everywhere.

## Hardware counters: why coalescing alone is insufficient

Matched Nsight profiles run after timing, with one launch skipped, one collected,
cache control `all` and clock control `none`. They are instrumentation, **not timing
acceptance**. Original/reference/candidates are measured with identical options.
An earlier exploratory d7 profile used cache control `none`; it is retained with its
own metadata and is not mixed into the confirmed profile table.

| d /p, 4096 shots | Layout | Load sectors/request | DRAM read GB | L1 hit % | Active lanes/instruction |
|---|---|---:|---:|---:|---:|
| 7 /.001 | Shot | 3.72 | .065 | 84.22 | 6.51 |
| 7 /.001 | Warp | 1.56 | .098 | 49.18 | 6.49 |
| 7 /.03 | Shot | 11.41 | 3.161 | 79.58 | 24.15 |
| 7 /.03 | Warp | 3.51 | 5.448 | 39.40 | 24.21 |
| 11 /.03 | Shot | 14.75 | 23.697 | 72.44 | 30.02 |
| 11 /.03 | Warp | 3.89 | 16.586 | 38.19 | 30.07 |
| 11 /.15 | Shot | 16.62 | 24.179 | 71.00 | 31.74 |
| 11 /.15 | Warp | 4.36 | 16.328 | 37.69 | 31.74 |

Load-sector metrics aggregate graph and scratch accesses, not per-array attribution.
Counters confirm the **access-coalescing problem** but reject an unconditional
throughput conclusion. On low-rate d7, active-lane utilization is about 6.5/32 and the coalesced
representation has a lower L1 hit fraction while reading more DRAM. At d11/higher rates, the same transformation reduces DRAM reads and materially
improves time. This is evidence for an interaction of locality and iteration
divergence; it is not a proven TLB diagnosis or a universally compute-bound claim.

A separate instrumented diagnostic recorded exact iteration counts on 4096 shots
per cell. Counts are identical **shot by shot across all three layouts**, and the
low-bit correction checksums equal the uninstrumented campaign. Mean iterations:
d7 /.001 **1.2686**, d7 /.03 **52.9712**, d11 /.03 and /.15 **64**. The loop-level
lane utilization `sum(iterations) / sum_warps(32*max_iterations)` is respectively
**18.18%, 82.77%, 100%, 100%**; this differs from Nsight's instruction-wide lane
metric. Reaching 64 does not distinguish nonconvergence from convergence at the
last iteration. The temporary diagnostic packs counts into unused high mask bits
on these single-observable fixtures only; it is not a production interface and its
instrumented durations are excluded from all speed claims.

Registers fall 42→40/thread in these profiles. Achieved occupancy at d7 /.03 stays
about 8%, rather than becoming an occupancy breakthrough; it depends on the small
launch grid and running phases. Source inspection alone could not establish these
counter/throughput trade-offs. Full metric values, including stores/L2 hits and
occupancy, are retained.

After BP, the **unchanged UF reference** was independently profiled at d7 /512
shots: about **5.00 load sectors/request**, **7.10 active lanes/instruction**, 7.70%
achieved occupancy. Its pointer/BFS accesses and divergent work are a separate
experiment, not evidence that BP's layout will help UF. The existing UF example
still matched CPU corrections across its five distances. No UF candidate is retained
and no UF speedup is claimed; applying this rewrite to UF without its own measured
win remains deferred.

## Correctness and validation

- All three layouts passed repetition n={8,16,32}, surface d={3,5,7} at four detector
  densities, and the **100,000-shot** d7 oracle per layout: **588,000 CPU/GPU observable
  correction comparisons, zero differences**. These are correction comparisons,
  not a claim that full per-variable vectors were compared on all 588k shots.
- A separate actual-device suite compares complete hard decisions and final parity /
  CPU convergence on degree-one infinity, isolated-variable/check, overlapping-check
  and empty graphs, iteration caps {1,2,64}, batches {1,31,32,33,129,7,0,257,3}.
  It exercises grow/shrink reuse, partial warp padding and real 17-shot subtiles
  without allocating a 2 GiB test batch. Input prefix compatibility and short-input
  rejection are checked on device.
- `ALEPH_REQUIRE_CUDA=1` makes a missing device or constructor error fatal in the
  actual GPU runs; construction-skip success is not counted as hardware evidence.
- Pure tests check zero/oversized units, padding and the 2 GiB tile bound. This test
  is added to the existing CUDA CI job with no GPU requirement.
- Stable/beta workspace all-target Clippy and fmt passed locally; final Linux CUDA
  all-target Clippy passed with `--no-deps`. An initial server invocation encountered
  a pre-existing Rust 1.96 `nonminimal_bool` lint in QEC deadline replay. Its log is
  retained; unrelated dependency code was not changed to manufacture a lint pass.
- The independent Python checker verifies all 624 CSVs against retained manifests,
  counts/repeats/input coordinates/checksums, and recomputes every median and block
  ratio. Deliberately changed raw timing data are rejected.

The measured CUDA body is byte-identical to final source. After measurement, the
host-only delta preserved the historical acceptance of surplus packed words and
expanded tests/commentary; the final device suite and scoped CUDA lint were rerun.
It does not alter the measured kernel schedule or address formulas. Source/patch,
binary and result hashes distinguish the historical measured snapshot from final
validation rather than relabeling old results as a new build.

## Reproduce and retained evidence

On a Linux CUDA host:

```sh
ALEPH_REQUIRE_CUDA=1 cargo test --release -p aleph-cuda --features cuda --test qec_bp_oracle -- --nocapture
ALEPH_REQUIRE_CUDA=1 cargo test --release -p aleph-cuda --features cuda --lib qec::bp::tests -- --nocapture
cargo run --release -p aleph-cuda --features cuda --example bp_layout_probe -- 7 16384 .03 3 warp
python3 scripts/python/check_bp_scratch.py /path/to/retained/campaign
```

The exact fixed campaign runner is `scripts/python/bench_bp_scratch.py`: pass a
three-layout probe binary, an unchanged original probe binary, and a **new** output
directory. It records GPU telemetry and keeps every observation. The baseline and
measured candidate source patches under [`data/p8-09-bp-scratch/`](data/p8-09-bp-scratch/)
apply to the base revision above in fresh `git archive` snapshots; use separate fresh
Cargo target directories to avoid stale tar-mtime artifacts. The data README gives
archive extraction/check commands. Nsight argv/metrics are retained in profile
manifests so profiling need not be inferred from a prose summary.

References: [original BP](qec-q3-gpu-bp.md), [original UF](qec-q3-gpu-uf.md),
[issue #559](https://github.com/aleph-sim/aleph/issues/559).
