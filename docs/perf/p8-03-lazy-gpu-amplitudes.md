# P8-03 — lazy Python GPU amplitudes

Issue: [#553](https://github.com/aleph-sim/aleph/issues/553).

## Contract and implementation

Previously GPU `aleph.run` sampled the final state and eagerly downloaded/widened
all amplitudes into a complex128 host vector, even for counts-only callers.
The result now owns a minimal final-amplitude readout holder. CUDA retains the
cudarc buffer and owning stream/context, excluding the wrapper's raw pool handle
and execution scratch. Metal finishes pending work and retains only the shared
amplitude buffer. No new `unsafe` or unchecked Send/Sync implementation is added.

The first `statevector()` releases the GIL for synchronization, CUDA download or
Metal widening, and cache lookup. A mutex serializes concurrent materialization.
On success it caches a shared immutable complex128 host vector and drops the
device holder. Each call copies that cache into an independent NumPy array using
the same NumPy copy path as before. Mutating an array cannot alter later results.
CPU statevector behavior, amplitude order, sampling, precision and execution
pipelines remain unchanged. CUDA copy failures become contextual Python
ValueErrors; the device state remains available for retry. Metal's synchronized
shared-buffer read remains infallible under its existing API contract.

Counts-only results **retain GPU amplitude memory until destroyed**. Retaining many
results can exhaust device memory sooner than the old eager host-result behavior.
Use `del result` when only its counts are needed; counts and NumPy arrays are
independent of result lifetime. After materialization, the host cache lives until
result destruction. Returned arrays have their own memory. First materialization
still needs staging/widening/cache/array memory; this change does not promise a
smaller statevector-request peak or lower GPU sampling scratch requirements.

## Actual public CUDA measurements

2026-10-13, RTX 4000 SFF Ada (20 GiB), driver 580.178.04, Linux 6.8,
Rust 1.96.0, Python 3.12.3, NumPy 2.5.0. Baseline `dce0073`, candidate
`7c8050c`; separate locked release builds and Cargo target directories. Baseline
adds only the read-only transfer-counter binding in
[data/p8-03-baseline-dtoh.patch](data/p8-03-baseline-dtoh.patch).

Each cell uses three fresh processes per revision, in paired order
baseline/candidate/candidate/baseline/baseline/candidate. Circuit: H(0), S(0),
CX(0,n−1); 32 shots, seed 7. `run_counts_ms` times public run plus counts, including
backend initialization, execution, sampling and previous eager materialization.
Read cells access statevector twice; the first includes NumPy import on demand.
RSS is the measured lifetime process peak, not a calculated allocation budget.

| Backend / n | Counts run median before → after (ms) | Counts-only peak RSS before → after (MiB) | Run DTOH before → after (bytes) |
|-------------|-------------------------------------:|----------------------------------------:|---------------------------------:|
| CUDA FP32 / 24 | 339.664 → 175.337 | 550.934 → 167.027 | 134218000 → 272 |
| CUDA FP32 / 26 | 768.768 → 280.575 | 1702.766 → 167.047 | 536871184 → 272 |
| CUDA FP64 / 24 | 352.361 → 143.234 | 672.410 → 160.594 | 268435728 → 272 |
| CUDA FP64 / 26 | 927.076 → 272.642 | 2208.562 → 160.457 | 1073742096 → 272 |

Candidate counts-only materializations are zero. The 272 bytes are a norm scalar,
sampling scalar and 32 uint64 outcomes; they do not grow with 2^n. First explicit
access transfers exactly the FP32/FP64 state bytes, increments materializations
once, releases logical device amplitude ownership and caches `16 * 2^n` host bytes.
Second access transfers zero bytes and does no widening. Deleting the result
returns both tracked resident/cache bytes to their starting values.

| Backend / n | First read median before → after (ms) | Repeat read before → after (ms) | Read-workflow peak RSS before → after (MiB) |
|-------------|--------------------------------------:|--------------------------------:|--------------------------------------------:|
| CUDA FP32 / 24 | 79.707 → 239.163 | 38.312 → 37.560 | 617.000 → 617.148 |
| CUDA FP32 / 26 | 300.818 → 656.654 | 164.603 → 134.390 | 2152.898 → 2153.000 |
| CUDA FP64 / 24 | 78.666 → 282.305 | 38.323 → 38.245 | 672.543 → 672.555 |
| CUDA FP64 / 26 | 255.782 → 897.405 | 133.990 → 134.444 | 2208.547 → 2208.598 |

The larger first-read cost moves work out of `run`; compare run + first read when
the caller actually needs amplitudes. Read-workflow peak RSS is essentially
unchanged. Repeat reads preserve a host copy rather than returning shared mutable
storage. An initial candidate cloned a Rust Vec for each call; measurements
revealed extra repeat cost, so it was replaced with an Arc cache and the existing
NumPy slice-copy path. Initial-candidate measurements are excluded from this report.

CUDA memory was sampled per process about every 100 ms using nvidia-smi. The
n=26 median sampled peaks were unchanged: 1730 MiB FP32 and 2242 MiB FP64, for
both counts-only and read workflows. They include context/pool/sampling scratch
and are lower bounds on true peaks; short n=24 candidates sometimes finished
before their maximum was sampled. No GPU peak-memory reduction is claimed.
Logical retained amplitudes after counts are 512 MiB FP32 / 1024 MiB FP64 at n=26.

The host remained shared; no services or watchdog were paused. Initial campaign
GPU utilization was 0%, with 1339 MiB used by existing services; sustained idleness
was not established. Three repeats support these observations, not a calibrated
speedup confidence interval or gate-throughput claim. Full samples, spread, stage
RSS, counters and artifact hashes are in
[data/p8-03-cuda-results.json](data/p8-03-cuda-results.json) and
[data/p8-03-manifest.json](data/p8-03-manifest.json).

## Metal evidence

On macOS arm64, Python 3.12 / NumPy 2.3.5, the retained P8-08 baseline wheel uses
the same eager Metal amplitude path. It was not rebuilt from the CUDA baseline
revision; its artifact hash is recorded. Same circuit/order/repeats as CUDA.

| n | Counts run median before → after (ms) | Counts-only peak RSS before → after (MiB) |
|---|-------------------------------------:|----------------------------------------:|
| 22 | 314.015 → 264.938 | 221.844 → 157.688 |
| 24 | 1030.263 → 949.984 | 797.828 → 541.828 |

Counts-only host widening is absent, confirmed by the materialization counter.
Metal uses unified memory: these RSS values include shared GPU storage and CPU
sampling temporaries; they are not PCIe-transfer measurements. The n=24 peak
difference is 256 MiB, the removed complex128 vector. Read-workflow n=24 medians
were 808.109 → 935.812 MiB with allocator/OS variation; no read-peak improvement
is claimed. Raw observations: [data/p8-03-metal-results.json](data/p8-03-metal-results.json).

## Verification and reproduction

- CPU-only, Metal and CUDA abi3 wheels build; 28 existing binding tests pass on each.
- Hardware-opted-in public tests execute on Metal, CUDA FP64 and CUDA FP32:
  counts transfer bounds, complex phases/order, repeated reads, independent arrays,
  cross-thread creation and concurrent first access, zero-shot cleanup, collapsed
  final state, and result/array lifetime. Explicit hardware opt-in fails on init
  errors rather than recording a device skip as success.
- Three pure Rust lazy-holder tests cover a deterministic failed-copy retry,
  successful device release, counts-only destruction and concurrent copy-once.
  Failure injection is at the copy callback; no destructive real-driver fault
  injection was performed on the shared host.
- Real CUDA FP32/FP64 readout and transfer oracles pass after the common buffer
  copy refactor. Workspace and CPU/Metal/CUDA Python all-target clippy, formatting
  and diff checks pass.

```bash
cargo test -p aleph-py --lib lazy_amplitudes
maturin build --release --locked -m crates/aleph-py/Cargo.toml --features python,cuda
# Extract wheel with zipfile; point PYTHONPATH at its directory.
ALEPH_TEST_GPU_BACKENDS=cuda,cuda-f32 python -m unittest discover -s scripts/python -p test_gpu_results.py -v
ALEPH_TEST_GPU_BACKENDS=metal python -m unittest discover -s scripts/python -p test_gpu_results.py -v
python scripts/python/bench_gpu_results.py --backend cuda-f32 --qubits 26
python scripts/python/bench_gpu_results.py --backend cuda-f32 --qubits 26 --read
```

For baseline/candidate comparisons use separate Cargo target/output directories,
extract wheels separately and select each with PYTHONPATH. Run the same candidate
benchmark script in fresh interleaved processes. Capture its JSON and ru_maxrss;
sample nvidia-smi per-process memory while each subprocess runs (about 100 ms
between polls). GPU diagnostics are logical buffer accounting, not a replacement
for measured peak RSS or allocator reservations.
