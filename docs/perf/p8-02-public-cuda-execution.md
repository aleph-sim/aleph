# Public CUDA fusion and layered execution (#552)

CLI and Python CUDA runs now share `CudaExecutionPlan` and
`prepare_cuda_execution`. They apply the existing GPU-safe fusion passes and
layered dispatch rather than leaving those optimizations in kernel benchmarks.
Plans are inspectable and reusable from Rust. The public default uses ordinary
FP64/FP32 arithmetic and width-3 fusion. Experimental TF32 is deliberately excluded.

The main n=22 public campaign improves caller wall time by 1.07–4.11x across its
20 cells, but this is not a universal speedup: a separate native-controlled-phase
FP32 QFT case regresses, and small-state setup costs dominate. Raw dispatch remains
available. The measurements below preserve these distinctions.

## Integration and semantics

The state-preserving pipeline is CancelInversePairs, FuseDiagonalRuns,
Fuse1qRuns, Fuse2q and FuseKq(width 3), followed by supported layered execution.
It excludes measurement-distribution DCE: frontends sample every final qubit,
and Python can request amplitudes later. CPU relabeling/tiling is also excluded;
pre-relabeled/tiled inputs fail explicitly instead of returning a permuted state.
Measure/barrier ordering and external controls retain their semantics. Reset
remains unsupported and now fails before backend/state construction.

Preflight sees the actual planned circuit, including fused matrices and phase
scratch, before constructing an SV backend. The context guard remains alive during
construction. State allocation still follows operation-aware device/host capacity
checks. A separate structural guard rejects >32 qubits because gate masks are
u32; free memory on a larger device cannot authorize unsupported addressing.
This is not a claim that n=32 fits or was measured on the reference card.

`CudaExecutionPlan::execute` sets strict arithmetic policy, executes raw or layered
IR, synchronizes, and propagates asynchronous errors. It does not optimize again.
The plan can be executed repeatedly without repeating fusion. Existing standalone
experimental Rust fusion/TF32 helpers remain available with their prior contracts.

Python retains lazy device amplitudes after counts; first `statevector()` still
performs explicit download/widen/cache and releases the device state. Seeded
measurement RNG consumption/order is retained. Floating-point regrouping can
change exact outcomes near a boundary; reproducibility is within a fixed version,
precision, plan and seed, rather than a raw/optimized bit-identity promise.

## Precision gate: rejected TF32 candidate

Ordinary FP32 is checked at 1e-5, FP64 at 1e-10. The legacy backend's default-on
TF32 switch must not silently widen that contract, even when input already contains
manual width-4/5 IR. Public execution explicitly disables it for both optimized
and raw modes, and generates width 3 whether the optional TF32 module is usable or
missing. Diagnostics expose availability separately from actual use.

An initial width-4 TF32 candidate failed the prior empirical 1e-4 budget on an
8-qubit coherent-state fixture: maximum amplitude error was
**1.408317795e-4 vs same-IR FP32 ALU**, and **1.408423669e-4 vs CPU FP64**.
The tolerance was not relaxed to admit it. `public_precision_audit` reproduces this
with the existing experimental helpers and records `within_previous_budget=false`;
it is a diagnostic of a rejected candidate, not a passing accuracy test.

Public tests also hide the optional TF32 module's CUDA header path in a child
process and verify successful ordinary-FP32 execution, width 3 and zero TF32 launches.
No library/service or SDK installation is modified by that test.

## Measurement contract

Host: RTX 4000 SFF Ada, driver 580.178.04, CUDA 13, Rust 1.96.0, Linux. Initial
free GPU memory was 18,680 MiB; utilization 0%. Services stayed running. The paired
harness checks utilization before every case with bounded retries and records
before/after readings and host load. There is no isolation guarantee or claim
that these readings prove absence of every short competing request. No service
was stopped and no large reach/OOM experiment was performed.

Base: main after #581, `8e6ddecec9ca947b73765ddc49a8dfd9fef0685f`.
Both arms use the same candidate binary/wheel, precision and raw workload.
Raw retains prior per-gate dispatch through the shared guarded/instrumented
boundary; it adds a circuit clone and explicit end-of-execution synchronization.
Thus this isolates the optimization rather than comparing unrelated builds.
The sampling implementation is unchanged between arms; #555 was not yet in this
base, so these are not combined sampling-plus-fusion speedup claims.

The main campaign has five families, two precisions, two public entry points,
three interleaved pairs, with alternating raw/optimized order: **120 executed
calls** at n=22, shots=32, seed=552. All five families first pass small n=8 CPU
FP64 amplitude oracles in both modes/precisions. Maximum state is 64 MiB FP64.
Host BLAS/OpenMP threads are fixed to one for validation/materialization.

QFT/QPE use an exact p/cx controlled-phase decomposition because the current
QASM parser has no cp/ctrl-modifier syntax. CLI and Python-from-QASM execute the
same source. Brickwall uses alternating disjoint CNOT edges; VQE uses chain
entanglers; QAOA uses CNOT/Rz/CNOT ZZ interactions. Native `Circuit.cp` is evaluated
separately below, so decomposition savings are not mislabeled as its speedup.

Python caller time measures `aleph.run` on an already parsed circuit, including
planning, context/JIT setup, execution, sampling, counts and native result return.
First statevector materialization is timed separately; it includes device-state
release and cache/NumPy work, not pure PCIe bandwidth. CLI caller time includes
fresh subprocess startup, parsing, setup, execution, counts output and teardown.
Python creates a new backend per call in the same process; CLI uses a fresh process.
Neither public route reuses a plan/backend. Each Python arm warms once before pairs.

Native diagnostics separate parse (CLI), fusion/clone, preflight/context acquisition,
backend/JIT initialization, synchronized execution, sampling, counts, and explicit
materialization where applicable. `total_native_ms` ends before final result/backend
release; external `caller_ms` is authoritative for total wall time. Dense fused-gate
counts and successful gate-kernel launch counters establish the executed path.
Launch counters exclude readout and are process-wide deltas, so concurrent callers
must not interpret them as per-call attribution.

## Public caller results

Median ms, raw -> optimized. Each cell has three measurements per arm; raw repeats,
min/max summaries and stages are committed. Three pairs do not establish a tight
confidence interval for small gains.

| Workload | FP64 Python | FP64 CLI | FP32 Python | FP32 CLI |
|---|---:|---:|---:|---:|
| QFT (QASM decomposition) | 581.09 -> 141.50 | 659.55 -> 231.98 | 229.71 -> 159.79 | 326.92 -> 250.57 |
| QPE (QASM decomposition) | 590.65 -> 145.89 | 674.95 -> 237.54 | 230.93 -> 161.66 | 324.27 -> 255.88 |
| Brickwall | 259.74 -> 150.02 | 344.98 -> 242.44 | 159.89 -> 131.96 | 253.16 -> 225.50 |
| VQE | 275.61 -> 169.11 | 364.45 -> 288.56 | 162.36 -> 147.86 | 258.16 -> 231.09 |
| QAOA | 244.80 -> 153.94 | 323.04 -> 229.75 | 159.22 -> 143.99 | 246.83 -> 231.57 |

Representative FP64 Python execution stages (median ms):

| Workload | Fusion raw -> optimized | Execution raw -> optimized | Gate launches raw -> optimized |
|---|---:|---:|---:|
| QFT decomposition | 0.034 -> 0.545 | 480.90 -> 39.94 | 1177 -> 43 |
| QPE decomposition | 0.038 -> 0.630 | 493.55 -> 47.55 | 1198 -> 50 |
| Brickwall | 0.010 -> 0.298 | 160.39 -> 49.03 | 327 -> 37 |
| VQE | 0.011 -> 0.445 | 174.41 -> 77.05 | 390 -> 66 |
| QAOA | 0.008 -> 0.190 | 143.89 -> 45.54 | 362 -> 44 |

Initialization is about 20 ms for FP64; preflight including primary-context
acquisition is typically 64–79 ms here. Sampling stays about 7.1 ms in both arms
with the older sampler at this revision. First FP64 statevector materialization
is roughly 88–101 ms in these cases, including result/device lifetime costs.
Fusion saves execution passes, not those setup/output costs.

### Counterexamples and limits

A separate 12-call paired campaign uses native Python `c.cp` QFT, with the same
unitary and precision checks but a different input IR:

| Precision | Caller raw -> optimized (ms) | Execution raw -> optimized (ms) |
|---|---:|---:|
| FP64 | 154.89 -> 129.81 | 57.12 -> 32.24 |
| FP32 | **144.32 -> 153.60** | **19.04 -> 27.06** |

This FP32 route is a measured regression, not hidden by the decomposition table.
`cuda_optimize=False` is the available workaround. The phase-polynomial path's
FP64 phase/angle work is a plausible limiter at this size, but this campaign does
not establish its hardware bottleneck or a portable crossover policy. A future
precision/state-size cost calibration should address it; no unmeasured heuristic
is introduced here to make every headline positive.

At n=8 brickwall, setup dominates: FP64 Python 95.00 -> 89.25 ms with overlapping
ranges [91.42,95.09]/[88.07,95.56]; FP32 CLI 194.61 -> 200.92 ms with ranges
[192.08,197.37]/[197.22,223.96]. These are not evidence of a stable small-state
speedup. Auto selection normally keeps small circuits on CPU; explicit CUDA users
can compare the raw route.

## Reusable-plan Criterion results

10 samples, 500 ms warmup, at least 1 s requested measurement; Criterion extends
collection for slow cells. Plans/backends are created outside the closure. State
allocation, execution, synchronization and result destruction are measured;
planning/JIT/sampling/materialization are excluded. This is a separate tier from
public caller time. The synthetic dense-chain bench retains its `brickwall` ID;
its chain edges differ from the staggered public brickwall fixture.

| Case, n=22 | Raw ms [95% interval] | Optimized ms [95% interval] |
|---|---:|---:|
| Dense chain FP64 | 174.10 [174.06,174.13] | 74.389 [74.201,74.558] |
| Dense chain FP32 | 38.187 [38.079,38.288] | 16.599 [16.572,16.632] |
| Native CP QFT FP64 | 57.101 [57.092,57.112] | 27.408 [27.392,27.437] |
| Native CP QFT FP32 | 13.982 [13.981,13.982] | **19.838 [19.834,19.844]** |

## Validation and memory

Three Rust planner tests cover random/nontrivial prepared states, external controls,
barriers, repeated plan execution, exact CPU amplitudes, measured final-state support,
manual width-4 strict arithmetic, unsupported IR and the addressing boundary.
The pure planning test runs in CI without a GPU. GPU-backed Rust tests retain their
normal no-device skip guard; the reported opt-in public campaigns instead fail on
missing initialization and assert actual fusion/layer launches and zero TF32 use.

Six real public CLI/Python tests cover precision, phase dispatch, final collapse,
seed repeatability, lazy transfer, module-unavailable fallback and pre-construction
rejection. Existing lazy/concurrent-result and CUDA memory frontend regressions
also pass. CLI default all-targets tests (36 library + 41 integration + 1 probe),
workspace Clippy, Linux CUDA/CLI/Python feature Clippy, Mac Python+Metal feature
Clippy and rustfmt pass.

Fresh-process n=18 planned-memory audits cover both modes and precisions:

| Precision / mode | Estimated live bytes | Measured live peak |
|---|---:|---:|
| FP32 raw | 6,310,416 | 6,308,368 |
| FP32 optimized | 6,310,896 | 6,308,368 |
| FP64 raw | 8,407,568 | 8,405,520 |
| FP64 optimized | 8,408,048 | 8,405,520 |

The planned phase scratch increases the estimate; the measured overall peak here
is still sampling, rather than simultaneous gate execution. All observed peaks
are covered. These are logical live allocations, not reserved pool/JIT memory.

## Use and reproduction

```sh
# Default optimized strict execution; raw comparison and opt-in stderr diagnostics:
aleph run circuit.qasm --backend cuda-f32 --shots 32 --cuda-diagnostics
aleph run circuit.qasm --backend cuda-f32 --shots 32 --cuda-raw --cuda-diagnostics

cargo test -p aleph-cuda --features cuda --test public_execution pure_planning_contracts
cargo test --release -p aleph-cuda --features cuda --test public_execution -- --test-threads=1
cargo bench -p aleph-cuda --features cuda --bench public_execution -- --noplot
ALEPH_TEST_PUBLIC_CUDA=1 ALEPH_CUDA_CLI=target/release/aleph \
  python scripts/python/test_public_cuda_execution.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/python/bench_public_cuda_execution.py \
  --binary target/release/aleph --output public-ab.jsonl --qubits 22 --repeats 3 --shots 32
python scripts/python/bench_public_cuda_native_qft.py --output native-qft.jsonl
# Diagnostic of the rejected candidate (prints within_previous_budget=false):
cargo run --release -p aleph-cuda --features cuda --example public_precision_audit
```

```python
result = aleph.run(circuit, backend="cuda-f32", seed=552)  # optimized, strict FP32
raw = aleph.run(circuit, backend="cuda-f32", seed=552, cuda_optimize=False)
print(result.execution_stats())
```

Raw evidence is in `data/p8-02/`: public/main and small/native-CP pairs, Criterion
intervals, memory peaks, TF32 counterexample, and exact source checkpoint hashes.
Production dependencies are unchanged; Criterion is dev-only and already in the
workspace. Remaining work includes native-CP FP32 admission/crossover tuning and
reducing repeated context/JIT/result-release costs, rather than claiming these
execution-pass savings eliminate all frontend overhead.

Integration after #555 (2026-10-16): merged main `928bdb7` and retained both
sampling and public-execution benchmarks/exports. On the real CUDA device, all
three planner tests, three sampler tests and seven public CLI/Python tests pass.
The added n=18, shots=16 GHZ fixture combines raw/optimized execution with the
hierarchical sampler in both precisions, checking full amplitudes, outcome
support and seeded repeats. Linux CUDA/CLI all-target Clippy, local workspace
Clippy, 78 default CLI tests and rustfmt pass. The benchmark evidence above
predates this integration and remains an execution-planning A/B with the former
sampler; it does not measure combined planner-plus-new-sampler speedups.
