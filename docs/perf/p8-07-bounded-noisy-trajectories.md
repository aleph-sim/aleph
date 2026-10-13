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

Baseline is main `e51ba789abc21295c79e1326548099cbb9d3a7d6` after #556. The identical
public `run_noisy` benchmark and allocation evaluator are copied to that snapshot
and the candidate, with separate Cargo target directories and verified distinct
binary hashes. All twelve warmed evaluator cases compare every nonzero count bin
exactly between baseline and candidate before timing.

Intel i5-13500, Linux 6.8, Rust 1.96, optimized Cargo profiles, no custom RUSTFLAGS.
Both versions run with affinity to logical CPUs 0/2/4/6 (four physical P cores).
One- and four-worker Rayon pools are created outside timed iterations. Shared
services stay running; resource logs check for competing builds/bench runners.
These are shared-host measurements, not dedicated-machine isolation.

Each case uses seed 557, H on each qubit, depolarizing(0.05,1) after H, and qubit-0
readout `[[0.97,0.03],[0.07,0.93]]`. Mixed cases additionally apply Ry(0.73) on every
qubit, amplitude damping(0.17) then phase damping(0.23) after Ry. The matrix covers
n=4/1024 shots, n=16/64, n=20/8, both models and 1/4 workers. Timing includes full
trajectory execution, channel validation, reset/allocation, terminal sampling,
readout and dense result construction. Models/circuits/pools are prepared outside.

A/B/B/A order, 10 Criterion samples per repeat, 0.3 s warmup and 0.7 s requested
measurement (extended by Criterion for expensive cases). Reported time is the mean
of both repeat median estimates; all raw samples and confidence intervals are
retained. Heap profiles count requested allocations, allocated bytes and additional
live Rust heap peak; they are **not RSS or allocator-retained pages**.

## Allocation results and timing status

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

**Timing is pending a clean host window.** The complete initial A/B/B/A campaign
recorded a competing `Runner.Worker` in candidate-1, candidate-2 and baseline-2.
The four-worker cases showed substantial repeat variability. These raw timings
are retained as explicitly contaminated diagnostic evidence; none are accepted
as a throughput improvement/regression claim. An automatic per-case retry harness
waits for the runner/build processes to disappear and monitors them throughout
each timed run; any affected repeat is rejected. The PR remains draft until the
accepted paired timing campaign completes.

## Data and reproduction

The committed `data/p8-07-*` records contain allocation baselines/candidate,
constrained-budget measurements, exact-count preflight hashes and machine/source/
binary provenance. `p8-07-contaminated-*` stores the rejected timing campaign;
`p8-07-manifest.json` explicitly marks its status. Clean timing data will be added
before the draft is made ready.

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
