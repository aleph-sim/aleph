# P8-10: opt-in warp-cooperative FP64 BP

Issue [#560](https://github.com/aleph-sim/aleph/issues/560), following [P8-09](p8-09-bp-scratch.md).

## Verdict and scope

**GO for an explicit Rust opt-in; keep the default serial shot-major decoder.** The frozen gate required at least 1.2× improvement against the faster serial reference in a held-out cell, with all four paired blocks above 1×. All 22 measured cells improved: median API-call speedups span **1.78–35.97×**. This is a bounded operating result on one RTX 4000 SFF Ada, not an automatic dispatch policy or a guarantee for other graphs, GPUs, rates, or decoder settings. No losing cell occurred in this matrix; none has been hidden.

`CudaBp::warp_cooperative(&cpu.tanner())?` selects the new schedule. `CudaBp::new` and `with_scratch_layout` retain their existing behavior. This issue adds no CLI/Python selector, relay/OSD changes, precision downgrade or approximation.

## Schedule and synchronization

A 128-thread block contains four independent full warps, one shot per warp. Lanes initialize disjoint scratch elements, process independent check nodes, then independent variable nodes. Each node retains the original ascending-edge FP64 arithmetic, including infinity/NaN comparisons. Full-warp fences separate initialization, check writes and variable writes. A full-warp convergence ballot produces a uniform break. Observable masks use exact associative integer XOR; lane 0 stores the result.

Check parity scans all assigned checks rather than stopping at the first failed check; this adds work but leaves the convergence predicate unchanged. There are no cross-warp dependencies, new shared-memory allocations or new unsafe blocks. The shot guard is uniform across every participating warp, and global thread indexing uses `size_t` because cooperative thread counts can exceed `u32`.

The new constructor validates CSR lengths, monotone bounds, variable ownership and unique check-edge ownership before device access. Public `TannerGraph` fields cannot introduce concurrent writes through duplicate edges. Invalid graphs return `Error::Memory`; existing constructors are unchanged.

## Correctness and accuracy

- Real-GPU oracle suite: **848,000 CPU/GPU correction comparisons**, zero differences across serial shot-major, edge-major, warp-tiled and cooperative schedules, including Gross qLDPC.
- Complete hard-decision vectors and convergence predicates match CPU fixtures, including degree-one/infinite messages, empty/isolated nodes, Gross, iteration caps 1/2/64, odd and reused batch sizes and explicit 17-shot subtiles. These fixture checks are separate from the large correction-mask campaign.
- Primary measurements validate every timed returned mask vector against a CPU oracle bound to exact DEM/input SHA256, seed, shots and BP parameters. Supplementary traces validate each timed single-shot mask.
- Logical failures and nonconvergence counts in timing receipts are **CPU reference metadata**, not extra diagnostic outputs from the GPU public API. Equal corrections on identical draws preserve those observed logical outcomes; they do not improve BP decoding quality.

For example, surface11 at rate .03 and 4096 shots has 4093 CPU nonconvergences and 1809 logical failures. Gross at the same batch/rate has 33 and 32. Faster BP does not fix these accuracy limits. Gross here is a genuine hyperedge **code-capacity** model, not a circuit-level qLDPC noise model.

## Primary timing protocol

Linux x86_64, RTX 4000 SFF Ada (20 GiB), driver 580.178.04, Rust 1.96.0, CUDA 13/NVRTC, FP64, alpha .875, 64 iterations. Services and GPU clocks stayed unchanged. The go gate, held-out choices and matrix were fixed before hardware measurements; the CPU-cache/warmup implementation was finalized before the primary campaign. One surface7 pilot is excluded from acceptance.

Surface7/surface11 use phenomenological memory-Z DEMs with rounds equal to distance. Gross uses code-capacity DEMs. Physical fault draws use seed 560. The 22-cell matrix covers rates .001/.03, batches 1/128/4096 and odd 33/4097 controls. Four mirrored three-arm blocks rotate the first arm and reverse cell order on alternate blocks. Comparators are serial shot-major and serial warp-tiled from P8-09. The faster reference is selected separately for each cell/block.

The timed endpoint is reused `decode_packed` plus synchronization: upload, kernel and returned-mask download. Packing, model building, compilation, CPU oracle work and warm allocation are excluded. Single-shot cells contain 512 repeated-input observations per arm; batch cells contain 24. All observations are retained, with no outlier removal or retries. Three original-binary ABBA controls bring the total to **576 invocations / 10,512 observations**. Original/current-serial block ratios range 0.9993–1.0025, with no material regression observed in those controls.

Telemetry during the primary campaign recorded 1320–2340 MHz SM clocks, 38–72 °C and 1453–1874 MiB total GPU memory use. Utilization includes the experiment itself; this shared host is not a controlled dedicated benchmark machine. Ratios are descriptive paired-block evidence, not confidence intervals.

| Model | Rate | Shots | Cooperative median, ms | Cooperative shots/s | Speedup vs faster reference | Four-block speedup range |
|---|---:|---:|---:|---:|---:|---:|
| surface7 | 0.001 | 1 | 0.0396 | 25,235 | 11.57× | 11.63–14.03× |
| surface7 | 0.03 | 1 | 0.9572 | 1,045 | 21.22× | 18.88–21.22× |
| surface7 | 0.001 | 128 | 0.0567 | 2,258,591 | 25.28× | 24.91–25.40× |
| surface7 | 0.03 | 128 | 1.1304 | 113,234 | 35.97× | 35.94–35.99× |
| surface7 | 0.001 | 4096 | 1.3646 | 3,001,716 | 17.83× | 17.38–17.93× |
| surface7 | 0.03 | 4096 | 19.0627 | 214,870 | 3.68× | 3.61–3.70× |
| surface11 | 0.001 | 1 | 0.1297 | 7,711 | 15.02× | 14.83–15.06× |
| surface11 | 0.03 | 1 | 5.0856 | 197 | 17.23× | 17.22–17.23× |
| surface11 | 0.001 | 128 | 4.8666 | 26,302 | 19.64× | 19.62–19.64× |
| surface11 | 0.03 | 128 | 6.5631 | 19,503 | 24.06× | 24.05–24.07× |
| surface11 | 0.001 | 4096 | 9.2784 | 441,453 | 12.72× | 12.52–12.81× |
| surface11 | 0.03 | 4096 | 162.1697 | 25,257 | 1.78× | 1.77–1.82× |
| gross | 0.001 | 1 | 0.0236 | 42,430 | 9.88× | 8.52–11.53× |
| gross | 0.03 | 1 | 0.0314 | 31,808 | 11.80× | 10.51–12.45× |
| gross | 0.001 | 128 | 0.0242 | 5,282,598 | 14.36× | 12.20–14.62× |
| gross | 0.03 | 128 | 0.4684 | 273,245 | 22.92× | 22.72–23.36× |
| gross | 0.001 | 4096 | 0.1597 | 25,648,813 | 3.49× | 3.35–3.58× |
| gross | 0.03 | 4096 | 0.8113 | 5,048,809 | 13.32× | 13.37–13.80× |
| surface7 | 0.03 | 33 | 1.1055 | 29,850 | 32.81× | 32.79–34.03× |
| surface7 | 0.03 | 4097 | 19.4711 | 210,415 | 3.62× | 3.62–3.71× |
| gross | 0.03 | 33 | 0.0568 | 581,252 | 21.71× | 20.15–21.91× |
| gross | 0.03 | 4097 | 0.8196 | 4,998,957 | 13.46× | 12.33–13.47× |

## Varied-input single-shot tails (supplementary)

The primary single-shot p95/p99 measure jitter for one repeated input. To address workload variation, a separate post-primary experiment replays **64 distinct fixed physical-noise inputs per model/rate**, each as a single-shot API call in a persistent decoder. Four mirrored blocks give eight replays/input/arm: **144 invocations / 9216 timed calls / 384 unique inputs**. This supplementary experiment does not change or select the primary go verdict.

Below are empirical percentiles of the fixed 64-input mixtures with repeats. They are not population-tail guarantees or 512 independent noise samples. The surface11 low-rate p99 includes a single difficult input; rare-event tails need larger independent campaigns.

| Model | Rate | Serial shot p95 / p99, ms | Serial tiled p95 / p99, ms | Cooperative p50 / p95 / p99, ms | CPU invalid / logical failures, of 64 |
|---|---:|---:|---:|---:|---:|
| surface7 | 0.001 | 0.8016 / 1.1867 | 1.2845 / 1.8336 | 0.0403 / 0.0560 / 0.0726 | 0 / 0 |
| surface7 | 0.03 | 21.9050 / 28.3028 | 36.1916 / 37.3632 | 0.9570 / 1.2686 / 1.2692 | 50 / 16 |
| surface11 | 0.001 | 3.5701 / 92.2495 | 5.2422 / 133.9556 | 0.1297 / 0.2040 / 5.0774 | 1 / 0 |
| surface11 | 0.03 | 92.7910 / 98.4342 | 135.2343 / 146.3507 | 5.0774 / 5.0786 / 5.0839 | 64 / 33 |
| gross | 0.001 | 0.2616 / 0.2620 | 0.3510 / 0.3516 | 0.0229 / 0.0243 / 0.0311 | 0 / 0 |
| gross | 0.03 | 1.8659 / 8.1003 | 2.6366 / 11.3119 | 0.0310 / 0.1046 / 0.3997 | 1 / 1 |

## Memory, occupancy and the actual trade-off

Scratch payload remains `16*edges + variables + checks + 8 + 4*max(1,ceil(checks/32))` bytes per shot: surface7 16,096 B, surface11 64,252 B, Gross 7148 B. Cooperative shot-major storage has no padding; the serial tiled reference rounds capacity to 32 shots. The existing 2 GiB scratch cap and multi-tile behavior remain. These are buffer payloads, not allocator/context/compiler overhead or total process VRAM.

Fifteen matched Nsight profiles (cold cache, clocks unchanged) are retained separately. Profile replay durations are not acceptance timings. At 4096 shots, surface7 low-rate occupancy rose from 3.98% / 3.58% (shot/tiled) to 51.58%; surface11 high-rate from 8.26% / 8.07% to 91.44%; Gross high-rate from 5.18% / 4.96% to 39.30%. Cooperative registers/thread were 40, versus 42 shot-major / 40 tiled.

This is **not a memory-coalescing win**: Gross high-rate global load sectors/request rise from 2.43 / 1.55 to 13.00, and surface11 from 14.74 / 3.89 to 15.32. These ratios aggregate different schedules and instruction streams, not identical logical accesses. Independent node execution shortens each shot’s serial critical path despite scattered graph accesses. One-shot profiles still have only about 2% achieved occupancy, though cooperative active lanes/instruction increase from about 1 to 25–28. High-degree imbalance, scattered loads, additional convergence work and untested large graphs remain reasons to keep explicit opt-in.

## Validation and reproducibility

Actual CUDA tests required a device (`ALEPH_REQUIRE_CUDA=1`); no GPU-less skip counted as success. The full correction suite passed, as did three final BP unit tests, stable/beta workspace Clippy, final Linux CUDA all-target Clippy (`--no-deps`) and rustfmt. The server toolchain has an unrelated pre-existing dependency lint; local workspace lint covers dependencies. CI additionally runs the new pure ownership rejection test without requiring a GPU.

The first expanded Gross fixture exposed an existing test assumption (one packed word per shot); it was corrected to the graph’s actual word count. An initial new hexadecimal seed lint was corrected without changing the seed value. Failed and passing logs are retained. The independent receipt checkers passed, and an altered trace-oracle mask was rejected in a negative control.

See [data README](data/p8-10-cooperative-bp/README.md) for exact source boundaries, raw artifacts, fingerprints and commands. No performance claim is made for cold compilation, CLI/Python workflows, arbitrary rate distributions, other hardware or decoder variants.
