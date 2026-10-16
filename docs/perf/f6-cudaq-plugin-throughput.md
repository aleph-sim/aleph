# CUDA-Q plugin output and input marshalling (#516)

The Python plugin now fills a NumPy-owned float64 error matrix directly from the
native batch loop. It avoids materializing an entire uint8 error batch and then
casting that batch to float64. The existing public `decode_batch_errors` API still
returns uint8; the CUDA-Q adapter uses a private native output method. Decoder
parameters, matching, path retrace, ties, convergence, and error-column order are
unchanged.

The same loop decodes either byte or float output. Rayon now zips output chunks
with convergence flags directly, avoiding a temporary vector of mutable row
references. Float output is allocated with NumPy, stays private during the GIL-free
fill, and is returned through NumPy's normal ownership. No unsafe code is added.
The SDK accepts its C-contiguous float64 storage without another copy.

Bool/uint8 NumPy syndromes avoid a detector-sized float64 intermediate. Threshold
comparison still means `value > 0.5`; lists, soft values, NaN, and infinities retain
the previous threshold policy. Bool comparisons are viewed as uint8 instead of
copied a second time. Empty 2-D batches now validate their detector width, and
`(shots, 0)` preserves the shot count for zero-detector models. Bare `[]` remains
an empty batch.

## What profiling changed

The original issue suggested reusing retrace scratch. The current matcher already
caches its `State` per decoder/thread, including `rt_dist`, `rt_pred`, `rt_heap`,
`rt_touched`, and `rt_matched`. Reimplementing that cache would duplicate existing
work. #511's pending allocation work is not included or required here.

On the baseline d=5, 20,000-shot workload, the output is **20,000 x 3,718**: 74.36 MB
of bytes or 594.88 MB of float64. The initial profile spent about 72 ms in NumPy
casts, 50 ms in native decoding, and 34 ms reconstructing the input float array
from lists. Returning a full Rust-owned `Vec<f64>` actually regressed the plugin
path. That experiment was discarded. Keeping NumPy's allocation policy while
filling its storage from Rust improved the measured path; no claim about the
allocator's underlying mechanism is needed for that observation.

A real SDK probe confirmed that NumPy input was already accepted by this version.
Callers can already avoid `astype(float64).tolist()`. This change does not claim to
introduce NumPy input support or remove list conversion from callers that still
supply lists. SDK `BatchDecoderResult` also coerces uint8 to float64, so merely
passing it a byte matrix would move the cast rather than remove it.

## Measurement

Results and reproduction commands follow below. These measurements concern the
Python plugin and output marshalling; they do not claim a new matching algorithm,
NVIDIA decoder parity, or elimination of the inherent cost of returning every
error column rather than just observable bits.

Host: Intel Core i5-13500, Linux, rustc 1.96.0, Python 3.12.3, NumPy 2.5.3,
Stim 1.16.0; real CUDA-Q QEC 0.8.0 source build
`6ac00e5c24ab17bbb6515f09009ce69643839eed`. Baseline is main
`bb128f7e3dcc3138b7bb89d20fba070f40f0c677`. Candidate is this PR's source.
Snapshots and the native extension were isolated under `/tmp/aleph-516-Ew3ttU`;
the shared SDK environment and GPU services were unchanged. No GPU is used.

The Python harness uses a Stim rotated-memory-X circuit, rounds=distance,
after-Clifford depolarization 0.003, seed 20261016, and the same split model for
native and plugin calls. Each size ran baseline/candidate/candidate/baseline in
separate processes, seven timing samples per arm per process. Tables pool those
14 samples per version and report their median. Timing uses one Rayon/BLAS thread,
pinned to CPU 2; before each process, host utilization had to be below 10% and
CPU 2 plus its SMT sibling below 5%. This is a shared host, not an isolated-machine
claim. For example, the independently unchanged cast stage varied by roughly 2–6%.

| Workload / path | Baseline ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| d=5, 20,000 shots, native byte errors | 51.394 | 48.259 | 1.06x |
| d=5, 20,000 shots, prebuilt list plugin input | 159.497 | 139.458 | 1.14x |
| d=5, 20,000 shots, prebuilt float64 NumPy input | 124.705 | 100.021 | 1.25x |
| d=5, 20,000 shots, Stim bool NumPy input | 125.366 | 99.474 | 1.26x |
| d=5, 20,000 shots, create float64/list + plugin | 211.233 | 194.219 | 1.09x |
| d=5, 20,000 shots, create float64 NumPy + plugin | 125.183 | 102.066 | 1.23x |
| d=9, 2,000 shots, native byte errors | 35.875 | 34.505 | 1.04x |
| d=9, 2,000 shots, prebuilt list plugin input | 106.003 | 93.212 | 1.14x |
| d=9, 2,000 shots, prebuilt float64 NumPy input | 86.698 | 73.402 | 1.18x |
| d=9, 2,000 shots, Stim bool NumPy input | 87.416 | 72.895 | 1.20x |
| d=9, 2,000 shots, create float64/list + plugin | 137.679 | 125.475 | 1.10x |
| d=9, 2,000 shots, create float64 NumPy + plugin | 88.353 | 73.891 | 1.20x |

The output hashes, input hashes, and convergence flags agree across all four
processes per size. The d=5 dataset has 16 logical mismatches / 20,000 shots;
d=9 has 0 / 2,000. Both have zero nonconverged outputs. These are compatibility
checks on fixed samples, not a newly powered logical-error-rate study. Existing
PyMatching oracle tests remain the independent matching check.

Buffer accounting: removing the byte batch saves 74.36 MB at d=5 and 53.006 MB
at d=9 before the float64 result is returned. The required float64 results still
occupy 594.88 MB and 424.048 MB respectively. Per-shot decoder vectors and input
copies still exist. These are buffer sizes, not measured process-RSS reductions.

Reproduce the complete Python profile and all stage timings on each checkout:

```sh
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 2 /root/cqvenv/bin/python scripts/python/bench_cudaq_path.py \
  --distance 5 --shots 20000 --repeats 7 --profile /tmp/plugin.prof
# Repeat with --distance 9 --shots 2000, alternating baseline/candidate order.
```

The checked-in JSON records retain individual samples, output hashes, model sizes,
SDK versions, and module paths; temporary module paths identify the actual private
extensions used rather than installed package metadata.

## Core microbenchmark and validation

`RAYON_NUM_THREADS=1 taskset -c 2 cargo bench -p aleph-py --bench error_output`
compares the old byte-batch-then-cast work with direct float64 writes. Both arms
reuse the same float64 destination to isolate the batch loop, rather than
confounding NumPy's allocator with Rust's. This uses seeded phenomenological
surface-code syndromes (256 shots), not the circuit-level Python workload above.

| Distance | Byte then cast, Criterion estimate | Direct fill, Criterion estimate | Ratio |
|---|---:|---:|---:|
| 5 | 83.034 us [79.925, 86.853] | 77.840 us [77.753, 77.972] | 1.07x |
| 9 | 517.52 us [516.98, 518.54] | 511.71 us [511.11, 512.58] | 1.01x |

The modest core change is consistent with leaving matching/retrace unchanged.
The public plugin's larger gain comes from avoiding batch materialization and
conversion. The microbenchmark is not an independent end-to-end speedup claim.

Validation on the final source:

- 13 `aleph-py` Rust library tests, including all seven decoder families across
  zero shots, zero detectors, zero error columns, and ordinary error rows.
- 22 local Python QEC tests with Stim/PyMatching, no skips.
- 37 Python/native/real CUDA-Q SDK tests on the Linux server, no skips. Coverage
  includes single and batch results, noncontiguous layouts, soft thresholds,
  noncanonical uint8 values, empty-axis shapes, and split-component rejection.
- Workspace/all-target Clippy, Python-feature/all-target Clippy, and rustfmt pass.
- All nine source/config/test/benchmark file hashes agree between the local tree
  and the tested server snapshot.

The new output array remains private while native code fills it with the GIL
released. Results retain the SDK's float64 shape/dtype, bool convergence vector,
and optional-result fields. Per-shot decoder allocations and shortest-path
retrace remain future optimization opportunities; this PR does not incorporate
pending #511 or the native C++ plugin proposed in #513.
