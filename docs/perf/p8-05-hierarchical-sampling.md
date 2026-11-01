# CUDA sampling without full-state scan passes (#555)

The default FP32/FP64 sampler replaces the two-buffer, n-pass Hillis–Steele CDF
with a hierarchy of 256-entry probability groups. It uses the norm reduction's
existing block masses for small shot counts, or builds one block-local FP64 CDF
for larger shot counts. No CUB/build dependency is added. The former algorithm
remains explicitly selectable as `SamplingAlgorithm::Reference` for A/B checks.

This PR is stacked on #581 (#554). The shared FP64 readout also serves cuStateVec;
its default therefore inherits the new sampler. Public CLI/Python output formats
are unchanged. Exact seed-to-outcome mappings can change; see the policy below.

## Algorithm and selection

`Hierarchical` reuses the per-256-amplitude norm masses, builds group-local CDFs
of those masses recursively, and traverses them with conditional inverse transforms.
A shot then reads at most 256 selected amplitudes twice, accumulating probabilities
in FP64. Beyond the mandatory norm check, it needs no full-state preprocessing.

`BlockCdf` first builds a local CDF inside every 256-amplitude block. Group masses
come from those CDF endpoints. Each shot traverses the small hierarchy and binary
searches its selected block. This trades one N-double buffer and a full-state
build pass for less per-shot work. Prefixes inside each group use fixed-order
sequential addition, with coalesced loads/stores and independent groups running
in parallel. In particular, zero probabilities remain exact CDF plateaus.

The measured conservative `Auto` policy is:

- N <= 65,536: BlockCdf.
- Otherwise, shots <= N/4096: Hierarchical.
- Otherwise: BlockCdf.

This is a calibration for the RTX 4000 SFF Ada, not a universal optimum. The
72-case sweep below compares both prototypes before choosing this policy. At
n=18, Hierarchical wins for one shot but loses at 1024; at n=22 it wins at 1024
but loses at 65,536. The conservative threshold avoids the clear measured
regressions; it does not claim to locate every crossover exactly. Explicit Rust
selectors support calibration on other devices. Auto selection is deterministic
and shared with preflight; it does not change silently according to transient
free memory. Large-shot BlockCdf requests may still exceed device capacity.

[CUB DeviceScan](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html)
was considered as the device-wide scan reference. Here, block-local prefixes
plus conditional transforms avoid needing a device-wide full-state CDF or a
compiled CUB shim. The kernels are an independent implementation.

## Before/after measurements

Host: RTX 4000 SFF Ada (20,475 MiB), driver 580.178.04, CUDA 13, Ubuntu 24.04,
Rust 1.96.0. Initial free GPU memory was 18,680 MiB and utilization 0%; host load
started at 0.91. Services stayed running, without isolation. Before each sweep
case, the script waited for utilization <=5%, with bounded retries. No competing
GPU workload was observed, but these are shared-host measurements, not an
isolated-device guarantee. No large-state reach/OOM experiment was performed.
Maximum tested state: n=22 (64 MiB FP64); maximum shots: one million.

Uniform state preparation, backend construction and NVRTC/JIT are outside all
sampling timers. Norm validation, budget checks, allocation, RNG/target upload,
search and result download are included. Reference retains the original scan
algorithm and adds the same current-budget guard and profiling hooks.

Criterion: 10 samples, 500 ms warmup, 1 s measurement per cell, profiling off.
Each state/backend is constructed once outside the measured closure. Intervals
are Criterion's reported 95% intervals, in milliseconds.

| Precision | n | Shots | Reference estimate [interval] | Auto estimate [interval] | Reference / Auto |
|---|---:|---:|---:|---:|---:|
| FP64 | 18 | 1024 | 0.14409 [0.13783, 0.14762] | 0.13723 [0.13715, 0.13731] | 1.05x |
| FP32 | 18 | 1024 | 0.14375 [0.14368, 0.14380] | 0.14500 [0.14495, 0.14508] | 0.99x |
| FP64 | 22 | 1024 | 7.1560 [7.1532, 7.1581] | 0.47761 [0.47745, 0.47769] | 14.98x |
| FP32 | 22 | 1024 | 6.8450 [6.8437, 6.8464] | 0.44612 [0.44527, 0.44843] | 15.34x |
| FP64 | 22 | 65536 | 7.3285 [7.3276, 7.3295] | 1.4046 [1.4043, 1.4051] | 5.22x |
| FP32 | 22 | 65536 | 7.0238 [7.0225, 7.0262] | 1.3136 [1.3134, 1.3139] | 5.35x |

The separate sweep starts a fresh process for each precision/mode/n/shots cell,
warms once, then records nine repetitions. It covers n=14/18/22 and shots
1/1024/65,536/1,000,000 for Reference/Hierarchical/BlockCdf. Raw min/median/max,
scratch, launch counts and GPU readings are in `data/p8-05/sweep.jsonl`.
Do not mix its fresh-process/cache/clock conditions with Criterion estimates.
At n=22, one-shot medians fall from 7.18/6.94 ms to 0.432/0.418 ms (FP64/FP32).
At one million shots they fall from 11.27/11.57 ms to 8.14/7.74 ms with BlockCdf.
Small-state high-shot cases are an explicit tradeoff: n=14, one million shots was
7.12 -> 7.23 ms FP64 and 7.47 -> 8.13 ms FP32 in this sweep. There is no blanket
speedup claim for that regime. Host RNG/upload/download and search dominate once
CDF construction is cheap; reducing build passes cannot remove those costs.

Separate synchronized stage profiles at n=22, 1024 shots (one representative
post-warmup call, ms; allocation and capacity query are included in build):

| Precision | Mode | Norm | Allocate + build | RNG/upload + search | Download |
|---|---|---:|---:|---:|---:|
| FP64 | Reference | 0.383 | 6.768 | 0.028 | 0.006 |
| FP64 | Hierarchical | 0.280 | 0.051 | 0.226 | 0.006 |
| FP64 | BlockCdf | 0.342 | 1.245 | 0.024 | 0.006 |
| FP32 | Reference | 0.297 | 6.619 | 0.028 | 0.006 |
| FP32 | Hierarchical | 0.279 | 0.050 | 0.308 | 0.006 |
| FP32 | BlockCdf | 0.297 | 1.309 | 0.024 | 0.006 |

Profiles deliberately synchronize between stages, affecting overlap and clocks;
they explain work attribution, not end-to-end throughput. Without profiling,
stage fields measure host submission time and download absorbs queued GPU work.
At n=22 construction launches drop from 23 (abs2 + 22 full-state scan passes) to
2 hierarchy kernels or 3 including local CDF construction. All modes still do
the same two-launch norm validation and one search launch.

## Exact device-memory bound

Let m0=ceil(N/256), m(j+1)=ceil(mj/256), stopping when mj<=256. Define
H = 8 * sum_j (mj + ceil(mj/256)). This includes the flattened local mass CDFs
and every retained next-level totals buffer, including the final singleton.
Sampling workspace, excluding persistent norm buffers and shot I/O, is:

- Hierarchical: H + 8 bytes (a valid dummy leaf pointer).
- BlockCdf: H + 8*N bytes.
- Reference: 16*N bytes.
- Zero shots: 0 (norm is still checked).

Targets and output add 16*shots device bytes; norm partials add
16*ceil(N/256)+16 bytes. State, execution scratch, host outputs, pool reservations
and headroom remain covered by #554's model. The generic planner accepts an
executor-supplied checked workspace override; CUDA uses the same selection and
bound functions in preflight and actual sampling. A second budget check occurs
before allocating sampling workspace, and driver errors still propagate.

| n | Reference scratch | Hierarchical scratch | BlockCdf scratch |
|---:|---:|---:|---:|
| 18 | 4,194,304 | 8,272 | 2,105,416 |
| 22 | 67,108,864 | 132,112 | 33,686,536 |

At n=22/1024 shots the measured live peak is 67,519,520 bytes FP64 and
33,965,088 bytes FP32 with Auto; old Reference peaks are 134,496,272 and
100,941,840 bytes. These are logical live allocations from the CUDA pool counter,
not driver/module memory or reserved pool pages. Every mode's profile audit
asserts that the model covers its measured peak. The ten operation audits from
#554 also pass; current rows are in `data/p8-05/allocation.jsonl`.

A synthetic 20 GiB budget now admits low-shot n=31 FP32 requests: state 16 GiB,
hierarchy 67,635,216 bytes, reductions about 128 MiB, plus outputs/execution and
headroom. This is a model result, **not a measured n=31 run**. The previous
31-qubit rejection audit is therefore replaced with an overflowing dimension;
32/33-qubit CLI rejection cases avoid allocations on the reference card.

## Correctness and reproducibility

Finite/norm validation is unchanged and runs even for zero shots. Both amplitude
precisions accumulate probabilities in FP64. Inverse transforms use strict
upper-bound selection, clamp rounded targets below the selected positive mass,
and preserve zero plateaus. The leaf fallback chooses the last positive bin.

A fixed state, resolved algorithm, precision, seed and device/build gives repeated
identical samples. Auto may choose a different variant when shots change. Grouped
floating-point sums and conditional remapping can change exact outcomes relative
to the old scan; cross-algorithm or cross-device seed identity is not promised.

Validation on the actual GPU:

- 35 CUDA library tests, including the new small exact-CDF and boundary tests.
- Raw FP32/FP64 states at n=0/1/7/8/9/16/17: sparse support, leading/trailing empty
  groups, concentrated final bin, repeatability, zero shots, and exact transfer
  counters (16 norm bytes + 8*shots). Eight independently seeded random complex
  sparse distributions per precision/mode use 100,000 shots, direct CPU-CDF
  outcome checks, zero-bin exclusion and per-bin tolerance 6*sqrt(S*p*(1-p))+2.
  Dense uniform complex states at n=9/17 additionally check 512 bins/groups with
  100,000 shots and a six-sigma bound.
- Exact boundary uniforms 0, CDF boundaries, the predecessor of a boundary and
  the predecessor of 1 for both leaf implementations; NaN/Inf/zero/bad norm errors.
- Existing FP64 readout CPU oracle, three FP32 readout oracles, and transfer test
  pass (one pre-existing throughput test remains ignored).
- 42 backend and 36 CLI library tests, workspace Clippy, Linux CUDA all-targets
  Clippy, rustfmt, ten operation allocation cases and public CUDA CLI checks.

## Reproduction

```sh
cargo test --release -p aleph-cuda --features cuda --lib \
  --test readout_oracle --test fp32_readout_oracle --test transfer -- --test-threads=1
cargo bench -p aleph-cuda --features cuda --bench sampling -- --noplot
cargo build --release -p aleph-cuda --features cuda --example sampling_audit
python3 scripts/p8-05-sampling.py target/release/examples/sampling_audit sweep.jsonl
python3 scripts/p8-05-sampling.py target/release/examples/sampling_audit profile.jsonl --profile
# Individual fresh-process audit: precision mode qubits shots repetitions profile
# Modes: reference / hierarchical / block / auto. Bounds: n<=22, shots<=1M.
target/release/examples/sampling_audit f32 auto 22 1024 9 0
```

Raw data: `data/p8-05/{criterion.json,sweep.jsonl,profile.jsonl,allocation.jsonl}`.
Source hashes and validation metadata are recorded alongside them. Criterion is
a dev-only dependency already present in the workspace; production dependencies
are unchanged. Further work could tune selected-block cooperation, cache stable
state CDFs safely, or reduce high-shot host RNG/transfer overhead. None is needed
to remove the n full-array scans, and none is claimed by these measurements.
