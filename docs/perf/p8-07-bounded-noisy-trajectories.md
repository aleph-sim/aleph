# P8-07: bounded noisy trajectories and reusable worker buffers

Issue #557 follows the single-shot sampler change in #556. The old driver still
allocated a fresh dense FP64 state for every shot, a fresh probability vector for
every Kraus application, and an outcomes array followed by a dense `2^n`
histogram. Channel lookup was already outside the shot loop; repeating that
optimization alone would not remove these costs.

## Execution and compatibility

The driver partitions the absolute shot range into at most the planned number of
chunks. Each chunk owns one backend/state and the maximum required Kraus
probability buffer. The first shot uses the freshly allocated zero state; every
subsequent shot resets **all amplitudes**, sets amplitude zero to one, and resets
`StdRng` from the existing `shot_seed(seed, absolute_index)`. Channel probability
scratch is resized and cleared for each use. State/channel buffers do not persist
between separate public calls.

Gate/channel insertion order, per-shot RNG consumption and readout order are
unchanged from the #556 version. Gate validation still runs through
`NaiveSvBackend::apply_gate`; noiseless fusion is not introduced across channel
boundaries. Borrowed attachment lookup uses nested maps to support constant-time
string/slice lookup without allocating temporary keys. Resolved references are
shared immutably. Public-field channel inputs are checked for finite/nonnegative
Pauli weights, matching arity, Kraus completeness and valid readout probabilities
before trajectory allocation. Invalid inputs return `NoiseError`.

`run_noisy` retains its Rust `Vec<u64>` dense-histogram signature. It now delegates
to `run_noisy_with_options` with the default incremental buffer budget of **1 GiB**.
`run_noisy_sparse` returns ascending nonzero `(basis_index, count)` pairs. CLI and
Python use this sparse path without changing printed bitstrings, dictionary
counts, seeds, or the shots-only result contract. The noiseless path is unchanged.

## Memory policy

`NoiseOptions` declares `memory_budget_bytes` and an optional `max_workers`.
`plan_noisy` reports fixed bytes, bytes per context, maximum contexts and the
conservative peak buffer bound, using checked arithmetic. The context count is
bounded by the current shared Rayon pool, shots, explicit maximum and remaining
budget. Even when kernel tasks suspend a trajectory, the total number of owned
contexts cannot exceed the number of chunks. No private pool or threads-squared
oversubscription assumption is introduced.

The estimate includes:

- The `8 * shots` outcomes array and resolved attachment vector storage.
- Dense result `8 * 2^n`, or sparse result capacity `16 * min(shots, 2^n)`.
- Per-context FP64 state `16 * 2^n`, maximum Kraus probabilities and the existing
  one-element sample vector.
- Potential simultaneous gate/control SmallVec spills. For `UnitaryKq`, offsets,
  target/fixed arrays, both AVX512 broadcast matrices (`128 * 4^k` bytes), and
  conservative per-pool-thread gather buffers are included even on scalar hosts.

This is an **incremental owned-buffer estimate**, not an OS reservation or RSS
cap. It excludes caller-owned circuit/model/matrices, stacks, allocator overhead,
Rayon runtime/task bookkeeping and unrelated concurrent calls. Result conversion
into Python dictionaries/CLI formatting is outside the Rust driver's budget.
Actual phases do not all overlap: the estimate reserves result bytes alongside
trajectory bytes even though states are dropped before result construction.
Shared-host availability may still change; the underlying allocator failure policy
is unchanged. The backend's existing 28-qubit ceiling remains.

If fixed output plus one context cannot fit, the call returns an actionable
`MemoryBudget` error before allocating execution/result buffers. Zero-shot sparse
calls need no buffers; dense zero-shot calls still budget the full histogram.
Rust users can raise the budget explicitly. CLI/Python currently use the fixed
default policy; larger frontend budgets require a future exposed option.

Outcomes remain O(shots), and sparse output reserves its worst-case capacity even
for a concentrated distribution. This issue does not implement an unbounded-shot
streaming accumulator or use distribution-dependent estimates. Static contiguous
chunks also trade fine-grained scheduling for a strict context bound.

## Validation

The fresh-per-shot reference driver is retained only in tests, separately executing
gates/channels, terminal sampling and readout. Reused execution matches its exact
seeded counts for empty and mixed Pauli/Kraus/readout models, seeds 0/7/u64::MAX,
1/2/4 pool threads, worker limits 1/2/3/4, non-divisible shot ranges and budgets
that force one context. Alternating 4/2/4 Kraus branch counts exercises scratch
clearing and resizing. Dense and sparse results reconstruct the same histogram.

Boundary tests cover zero shots/qubits, insufficient/exact budgets, zero worker
limits, dimension/arithmetic overflow, control spills, vectorized matrix scratch
and malformed channel/readout inputs. Local validation passed 351 SV unit/property
tests, three driver tests, six stored Aer density-matrix/readout fixtures and two
public sampling/worker-invariance tests. CLI tests cover the unchanged noise output;
the sparse formatter is compared directly with the raw-sample formatter, including
zero shots. Eight Python `TestNoise` tests passed on the newly built Python 3.12
extension. Workspace/all-target clippy, Python-feature clippy and fmt passed.

## Measurement protocol

Baseline is main `e51ba789abc21295c79e1326548099cbb9d3a7d6` after #556; measured
candidate is `d3f5a3a`. Identical public `run_noisy` benchmark/allocation evaluator,
separate Cargo target directories, distinct binary hashes and exact comparison
of every nonzero count bin in all twelve evaluator cases gate timing. A later
change only clarifies the conservative reservation error text; trajectory,
sampling and planning arithmetic are unchanged.

The user selected CPU-only host `51.159.0.49`: dual Xeon Silver 4114 (20 physical
cores / 40 logical), AVX512, 123 GiB RAM, Linux 7.0, Rust 1.96. Optimized Cargo
profiles, no custom RUSTFLAGS. Four initially idle physical cores on socket 1 are
pinned: logical CPUs **3/7/9/11**, with SMT siblings 23/27/29/31 monitored too.
The monitor runs on CPU39. Shared CI/services remain running; CPU-only execution
is sufficient for this issue. One/four-worker pools, model/circuit setup are
outside timing. No dedicated-host, GPU/Metal or frontend end-to-end gain is claimed.

Each case uses seed 557, H on each qubit, depolarizing(0.05,1) after H, and qubit-0
readout `[[0.97,0.03],[0.07,0.93]]`. Mixed cases add Ry(0.73) on each qubit,
amplitude damping(0.17) then phase damping(0.23) after Ry. Matrix: n=4/1024 shots,
n=16/64, n=20/8, both models and 1/4 workers. Timings include full public dense
execution: validation, allocation/reset, channels, sampling, readout and output.

Per-case A/B/B/A order; 10 Criterion samples per repeat, 0.3 s warmup and 0.7 s
requested measurement (Criterion extends expensive cases). Each repeat first
requires <=2% busy time on all selected cores/SMT siblings. A 0.2 s per-task CPU
monitor follows active/runnable benchmark-thread endpoints and conservatively
charges sampled external CPU time to their physical cores. A repeat is rejected
when observed overlap exceeds **3% of nominal worker CPU capacity**. Endpoint
sampling can miss short tasks/migrations and cannot isolate shared-cache/turbo
interference. This is bounded shared-host evidence, not exclusive CPU isolation;
sub-5% throughput changes are not promoted as gains.

The first twenty accepted repeats passed a stricter <1% of one-core wall-time
limit on all allowed cores. Constant 1-2% background work made that limit reject
large single-worker runs repeatedly, including work on unused allowed cores.
Before inspecting final speedups, attribution was narrowed to active benchmark
cores and the later 3%-capacity limit was declared. Earlier rejected repeats are
**not promoted**. All earlier accepted stricter repeats remain. Telemetry records
every accepted/rejected run and policy version; 48 primary repeats and ten targeted
follow-up repeats are accepted, with 32 rejected attempts retained.

Reported time is the mean of both repeat median estimates, except the replicated
tiny four-worker Kraus case where all **seven** medians per version are included.
Raw samples/confidence intervals are retained. Heap counters track requested
allocations/bytes and additional live Rust heap, not RSS or retained pages.

## Results and limits

| Same dense API, one worker | Baseline allocations | Candidate allocations | Allocated bytes, baseline → candidate |
|---|---:|---:|---:|
| Pauli n=4 / 1024 shots | 2,077 | 1,032 | 295,284 → 16,896 |
| Mixed Kraus n=4 / 1024 shots | 10,277 | 1,037 | 426,588 → 17,072 |
| Pauli n=20 / 8 shots | 63 | 32 | 142,607,748 → 25,166,592 |
| Mixed Kraus n=20 / 8 shots | 423 | 53 | 142,614,028 → 25,167,408 |

The large-state live-heap peak is essentially **unchanged at the same concurrency**:
one 16 MiB state per context still dominates. Reducing total allocated bytes is not
a further 16 MiB peak reduction after #556. The bounded policy instead controls the
number of states that can be owned at once; constrained n=20 mixed tests measure
16.00/32.00/64.00 MiB for 1/2/4 contexts, all below their planned bounds. A sparse
one-context call fits a 16,778,872-byte budget while the dense adapter rejects it.
For n=4, one-worker peak drops from 16,912 to 8,760 bytes for the mixed case.

| Model / n / shots | Workers | Baseline | Candidate | Baseline / candidate |
|---|---:|---:|---:|---:|
| kraus / 16 / 64 | 1 | 1797.377 ms | 1780.021 ms | 1.010× |
| kraus / 20 / 8 | 1 | 4236.672 ms | 4059.606 ms | 1.044× |
| kraus / 4 / 1024 | 1 | 5.048 ms | 4.970 ms | 1.016× |
| pauli / 16 / 64 | 1 | 118.954 ms | 120.990 ms | 0.983× |
| pauli / 20 / 8 | 1 | 380.080 ms | 373.935 ms | 1.016× |
| pauli / 4 / 1024 | 1 | 0.971 ms | 0.937 ms | 1.036× |
| kraus / 16 / 64 | 4 | 523.233 ms | 505.017 ms | 1.036× |
| kraus / 20 / 8 | 4 | 1223.004 ms | 1173.640 ms | 1.042× |
| kraus / 4 / 1024 | 4 | 1.578 ms | 1.687 ms | 0.935× |
| pauli / 16 / 64 | 4 | 33.600 ms | 33.942 ms | 0.990× |
| pauli / 20 / 8 | 4 | 127.776 ms | 125.229 ms | 1.020× |
| pauli / 4 / 1024 | 4 | 0.311 ms | 0.288 ms | 1.081× |

The wider-state ratios span 0.983× to 1.044×, which this shared-host protocol
cannot turn into a reliable speedup claim. The measured practical gain is lower
allocation traffic and an explicit memory/context bound. State reset still writes
the entire vector, and Kraus probability/application passes and gate/unitarity
work remain; removing allocation does not remove their arithmetic or memory walks.
Four-worker tiny Pauli shows a modest 1.081× ratio on this host, not a universal
acceleration claim.

The initial tiny four-worker Kraus candidate medians were **2.802 and 1.514 ms**
versus baseline 1.570 and 1.560 ms. Five additional paired repeats per version
check this variability rather than discard the slow result. Extra candidate
medians are 1.484-1.512 ms; extra baselines 1.564-1.638 ms. Including **all seven**
medians gives baseline 1.578 ms versus candidate 1.687 ms: **6.9% slower**. The rare
initial latency outlier remains unresolved; do not advertise a gain from selecting
the typical/faster repeats. Scheduling/environment sensitivity is a hypothesis,
not an attribution proved by the current telemetry. Per-thread placement and
cycles/ref-cycles profiling are appropriate follow-up work.

The previous i5-13500/openwebgui full matrix overlapped a CI `Runner.Worker` and
is retained as explicitly contaminated diagnostic evidence, excluded from the
accepted Xeon summary. The constrained memory/correctness results on both hosts
remain valid. No services or CI jobs were stopped for these measurements.

## Data and reproduction

`data/p8-07-summary.csv` contains all twelve cases; `p8-07-criterion.json` holds
48 primary repeats; `p8-07-variance-criterion.json` holds all ten targeted repeats.
`p8-07-core-guard.jsonl` records accepted/rejected telemetry and policy changes.
Allocation, constrained-budget and manifest files record both binary/source
provenance and exact count preflight. `p8-07-contaminated-*` is the rejected
openwebgui campaign. Linux reproduction helpers are in `scripts/python`.

```sh
cargo test --locked -p aleph-sv --lib --test noise_driver --test noise_oracle --test single_shot
cargo test --locked -p aleph-cli
cargo clippy --locked --workspace --all-targets -- -D warnings
# Compile the identical harness against each version in independent targets.
CARGO_TARGET_DIR=/tmp/target-for-this-version \
  cargo bench --locked -p aleph-sv --bench noisy_trajectories --no-run
# Alternate the resulting executables after builds/tests complete.
ALEPH_BENCH_OUTPUT=/tmp/results-this-repeat \
  taskset -c 0,2,4,6 /absolute/path/to/noisy-trajectories-bench --bench
cargo run --locked --release -p aleph-sv --example trajectory_allocations
# Candidate-only constrained-memory evaluator, four pool threads:
cargo run --locked --release -p aleph-sv --example trajectory_budget
```

For automatic per-case A/B/B/A runs, copy the independently compiled executables
into an artifact directory as `bench-base` and `bench-candidate`, then run:

```sh
python3 scripts/python/select_noisy_bench_cores.py /tmp/noisy-artifacts
python3 scripts/python/bench_noisy_trajectories_guard.py /tmp/noisy-artifacts
```

The guard changes affinity only for its monitor and benchmark children, stores
all attempt logs/telemetry, and writes accepted Criterion JSON/summary CSV. Its
shared-host limits still apply even when every repeat passes the CPU guard.
