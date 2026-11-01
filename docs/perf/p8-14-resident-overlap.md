# P8-14: resident tile runs with copy overlap

Issue [#564](https://github.com/aleph-sim/aleph/issues/564).

## Scope and decision rule

This experiment combines the existing [multi-gate residency](p5.11-03-multi-gate-residency.md)
and [copy-overlap](p5.11-02-overlapped-paging.md) mechanisms in FP64. Historical gains
are not multiplied. The comparator is synchronous residency at the same tile split,
plus larger synchronous tiles at equal ring-amplitude payload on deep locality cells.

The predeclared gate requires at least 1.05× median improvement in a held-out deep
operating cell, all four paired block ratios above 1, and no domination by its
same-payload larger synchronous tile. Unsupported regimes can remain synchronous;
a measured no-go can close this research issue.

## Execution contract

`run_paged_batched_overlapped(circuit, tile_qubits, depth)` is an explicit FP64 mode.
Defaults and FP32 are unchanged. Maximal runs whose operands **and controls** are
below `tile_qubits` share one load/scatter per tile. Barriers do not break a run.
Any high-qubit gate flushes it and uses remapped co-resident tile groups.

The ring reuses the existing H2D/compute/D2H streams and event ordering:

1. A segment's first gather waits for the preceding segment's final scatter.
2. Reusing a ring slot waits for that slot's preceding scatter.
3. Compute waits for the gather; scatter waits for all gates in the resident run.
4. A segment with fewer than `2*depth` groups uses synchronous residency on the
   default stream, after the preceding scatter. Future slot reuse waits for it.
5. Every stream is drained before returning the host state. The shared pipeline
   now drains even after an enqueue/apply error, before pinned/ring storage drops,
   and restores the backend context. No extra unsafe blocks were introduced.

Gate arithmetic and order within each tile are unchanged. The schedule helper only
changes how many gates execute between gather and scatter.

## Resource bounds

Pinned host payload is `16 * 2^n` bytes. The amplitude ring reserves
`depth * 16 * 2^(m + max_high_fanout)` bytes, even for low-only segments. Per-slot
matrix scratch is bounded separately (generic matrices up to 8×8; `UnitaryKq` uses
its explicit data size). Host/cgroup availability and free device memory are
preflighted with headroom; retained pool memory is trimmed before the ring.
Gate metadata, events and context overhead are outside the payload formula and covered
only by advisory headroom. Allocation errors still propagate because concurrent consumers
can change free memory.
The existing maximum-amplitude ring bound remains, and oversized shifts are rejected.

This conservative largest-group ring can waste memory on mostly low-only circuits.
A deeper ring can force a smaller tile. Therefore the experiment retains equal-payload
larger-tile synchronous controls instead of assuming equal tile width is a fair
resource comparison. It makes no full-device peak-allocation or n=31+ reach claim.

## Timing protocol

Six fixed workloads use actual CUDA paging at n=24/26, with host state at most 1 GiB
and ring amplitude payload at most 1 GiB. They deliberately fit in-core: forced
paging isolates transfer/scheduling effects without stressing the shared server.
The locality model has four deterministic rotation/CNOT layers over a fixed hot
region and a high target at `n-1`; the hot region stays fixed when the comparator
uses a larger tile. Other held-out controls have alternating low/high gates, short
runs with high/high CNOTs, a shallow pipeline, and unfused natural-order QFT.

Four mirrored blocks rotate the leading mode and reverse cell order. All 160
allocation-inclusive API samples are retained, without retries/outlier removal.
Backend construction/JIT, norm/export are outside timers; pinned/ring allocation,
initialization, execution and drain are inside. Separate synchronous per-gate
controls and instrumented Nsight timelines run **after** acceptance timing. Their
profiler overhead is never counted as an end-to-end gain.

Full-state pass/byte, low-run and launch counts printed by the probe are analytical
models, not observed bandwidth or kernel timings. Matched CUDA timelines are the
independent check of actual copies, kernels and overlap. Copy-only duplex ceilings
from historical measurements do not imply a universal total speedup ceiling when
compute and stream scheduling also change.

## Validation

Real-device full-amplitude tests compare CPU, synchronous per-gate and synchronous
residency results at finite absolute error ≤1e-10. Cases include deterministic random
circuits, high controls/targets, high/high gates, generic SWAP/CCX matrix scratch,
barriers, tile splits and ring depths 2/3/4/7. Norm and reversibility are checked.
Invalid tile/depth/host-size/instruction requests are rejected without forced OOM.
Existing overlap/residency reference suites and a GPU-free segment-order check also
run. No GPU-less skip is accepted in the hardware receipts (`ALEPH_REQUIRE_CUDA=1`).

## Results: GO for explicit FP64 opt-in

All four blocks clear the frozen gate on held-out locality n=26: **1.36×** over synchronous residency at the same split and **1.42×** over the larger synchronous tile at equal ring-amplitude payload. This supports keeping the explicit combined executor. It does not justify changing defaults or adding FP32 yet.

| Workload | n / m | Residency, ms | Per-gate overlap, ms | Combined, ms | Gain vs residency | Gain vs faster existing path | Four-block residency ratios |
|---|---:|---:|---:|---:|---:|---:|---:|
| locality | 24 / 20 | 475.86 | 5376.97 | 330.24 | 1.441× | 1.441× | 1.425–1.515× |
| locality | 26 / 22 | 2611.11 | 23635.41 | 1915.32 | 1.363× | 1.363× | 1.362–1.379× |
| alternating | 24 / 20 | 1394.22 | 1128.79 | 1129.64 | 1.234× | 0.999× | 1.231–1.239× |
| short | 24 / 20 | 394.80 | 490.80 | 358.92 | 1.100× | 1.100× | 1.086–1.105× |
| locality | 24 / 22 | 653.41 | 7474.25 | 655.31 | 0.997× | 0.997× | 0.994–0.999× |
| qft | 24 / 20 | 4667.74 | 10152.81 | 3836.60 | 1.217× | 1.217× | 1.211–1.226× |

The equal-payload larger synchronous controls use **the same circuit/hot region**: n=24/m=22/hot=20 takes 633.56 ms versus combined m=20 at 330.24 ms (1.92×), both with 128 MiB amplitude payload; n=26/m=24/hot=22 takes 2714.21 ms versus combined m=22 at 1915.32 ms (1.42×), both with 512 MiB. These observations are workload-specific; no cache-causality claim is made without cache counters.

Preserve the negative controls: the alternating circuit has no multi-gate residency to add, so combined matches the already available overlap path (0.999×). The shallow locality case falls back throughout and is slightly slower than synchronous residency (0.997×), while allocating a four-slot ring instead of one group. Users should retain synchronous residency for that regime. Short runs improve only 1.10×, well below the deep locality gain. QFT improves 1.22× against the faster existing path.

Post-primary per-gate synchronous controls at n=24/m=20 have medians 6722.41 ms (locality) and 1392.39 ms (alternating), eight observations each. These separate controls do not establish paired-block ratios with the primary campaign. Existing residency remains the main comparator.

The main campaign lasted 839.87 s. GPU telemetry spans 210–2340 MHz, 38–66 °C and 1423–2136 MiB total device memory use; utilization includes the experiment. Before setup the server showed load 0.00 and GPU 0%; the pre-timing check immediately after our own builds/tests showed load 1.01 and GPU 0%. Services and clocks stayed untouched. Four interleaved blocks reduce time drift, but this remains shared-host evidence, not a dedicated-machine confidence interval.

## Measured timeline attribution

Nine post-primary Nsight Systems 2025.3.2 timelines independently verify transfer bytes, kernel counts, stream IDs, fill/drain and overlap. Their instrumented times are **not** acceptance timings. For locality n=24/m=20, residency and combined each transfer exactly **2 GiB H2D + 2 GiB D2H**, 128 tile copies in each direction, and execute 2528 kernels. The per-gate overlap reference transfers 40 GiB each direction (160 passes), with the same 2528 kernels. Thus residency removes passes; combining it with overlap does not remove more bytes.

| Profile, locality n=24/m=20 | Setup before first tile, ms | GPU pipeline span, ms | Sum H2D / D2H / kernels, ms | H2D∩D2H, ms | Copies∩compute, ms |
|---|---:|---:|---:|---:|---:|
| Synchronous residency | 60.60 | 438.58 | 162.50 / 164.61 / 107.88 | 0 | 0 |
| Per-gate overlap | 54.95 | 5320.16 | 5097.10 / 4307.89 / 101.61 | 4097.31 | 95.23 |
| Combined residency + overlap | 55.25 | 277.09 | 248.80 / 212.29 / 114.40 | 189.81 | 108.93 |

Setup is the interval from the pinned-allocation API start to the first tile H2D: it includes host initialization, ring allocation and stream creation. It is not pure allocator latency. Combined copies take longer individually under duplex contention, and kernels do not get faster here; the gain comes from covering their time with concurrent work. The measured combined timeline has 1.26 ms to its first kernel and 2.52 ms after its last kernel to the final scatter. Those global fill/drain endpoints are not additive independent costs.

The shallow n=24/m=22 timelines show **zero H2D/D2H or copy/compute overlap**, and essentially identical GPU pipeline spans: 598.58 ms synchronous versus 598.31 ms combined. Initial amplitude memset payload is 128 MiB versus 512 MiB, plus the backend’s separate 16-byte initialization. This explains why the shallow fallback preserves correctness but does not create performance value. Short-run combined timing shows both deep low-run overlap and synchronous high/high fallback, with identical expected transfer totals.

All profiled memcpy APIs are async driver calls; combined locality has four stream synchronizations at final drain, not per tile. The shallow path deliberately remains single-stream ordered. Complete raw timelines and API counts are retained, so modeled pass/launch counts can be checked against observations. Do not sum busy-time durations to estimate wall time: they overlap.

## Final validation and review

- 2 final real-device combined-oracle tests passed, including generic SWAP/CCX matrix scratch and a host-side unsupported-k=6 error **after enqueue**, followed by reuse of the backend.
- Existing paging references passed: 2 residency tests and 3 overlap tests; 4 GPU-free schedule tests passed. The public API example compiled (2 CUDA doc tests total).
- Linux CUDA all-target Clippy (`--no-deps`), local stable/beta workspace Clippy and rustfmt passed. The server toolchain’s unrelated dependency lint is not changed by this issue. CI now explicitly runs the pure paging schedule tests without a GPU.
- Independent primary/profile checkers passed. A deliberately inconsistent modeled-transfer receipt was rejected.
- Manual self-review found no blocking findings. New bounds checks reject impossible ring/host sizes; the error path drains all streams and restores the backend context.

Hardware: RTX 4000 SFF Ada, driver 580.178.04, CUDA 13 bindings/NVRTC, Rust 1.96.0 on x86_64 Linux. See [raw data and reproduction](data/p8-14-resident-overlap/README.md) for exact measured-source identity, every sample, failed test-fixture compilation logs, final passing logs and the independent analysis commands.
