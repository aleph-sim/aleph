# Public workflow experiments (#563)

Run from the checkout with Python 3.10+; the controller needs only the standard
library. CUDA workers need the built CUDA Python package and NumPy. Build the
CLI from this revision for the CPU diagnostics protocol. No privileged operations,
service pausing, affinity changes or dependency installation happen in this harness.

```sh
cargo build -p aleph-cli --release
# Edit arm executable paths and full source revisions to match your actual build.
python scripts/benchmarks/public_workflow.py scripts/benchmarks/cpu-control.json --output /tmp/cpu-control-001
# Both arms can use the same wheel; only cuda_optimize changes.
python scripts/benchmarks/public_workflow.py scripts/benchmarks/cuda-fusion.json --output /tmp/cuda-fusion-001
```

Example revisions are placeholders for the operator to update, not proof of a
build's source. Each executable is hashed; CUDA also records the imported native
module path/hash. Record build flags/toolchain in the experiment report alongside
the manifest. A source checkout's HEAD is not silently attributed to a binary.

Each new output directory contains the exact QASM, `manifest.json` and
`runs.jsonl`. Refusing an existing directory prevents accidentally appending two
experiments. Nonzero exit (2) means failed, skipped or contaminated, never a pass.
`--skip-hardware` records an explicit skipped CUDA campaign without importing the
SDK. Requesting CUDA execution fails on missing CUDA, missing API diagnostics,
wrong output/precision, oracle mismatch or memory/transfer guard failures. CI can
compile CUDA but that is not a hardware execution receipt.

## Manifest contract

`validate_spec` is the versioned, executable schema. Unknown extension keys are
preserved; required keys, ranges, types and comparison constraints are validated.
The examples list the required fields:

- Full arm source revisions (operator-declared), executable hashes, exact argv,
  hardware/compiler/platform/topology, selected thread environment and CPU affinity.
- Model, exact QASM hash, seed, precision, fixed amplitude/norm tolerances, shots,
  optimizer and algorithm, one worker, pairs and warmup calls. Error tolerances
  cannot exceed the established FP64 1e-10 / FP32 1e-5 amplitude bounds.
- Comparison class: `same-algorithm`, `algorithm-change` or `historical`. A
  same-algorithm campaign refuses different algorithm declarations. Each manifest
  is summarized separately; never pool decoder/precision/resource classes.
- Experiment rationale: bottleneck fraction, desired speedup and stop rule. The
  Amdahl bound `1/(1-f)` must permit the target before launching a campaign.
  This is an operator-supplied hypothesis, not a hardware profile.
- Preflight counter gate (`counter-change`) or explicitly justified no-gain
  control (`control`). The CUDA example requires `output_instructions` to change
  in two real frontend calls before broad timing. Sweep thresholds around actual
  observed transitions; a missing/inert counter aborts instead of spending trials.

Raw preflight records remain distinct from measurements. Order alternates AB/BA
by pair; the preflight pair runs BA. Median paired ratios, min/max and a seeded
paired bootstrap interval are reported. Five pairs are exploratory; overlapping
intervals or a tiny best cell do not establish a stable small improvement.
No wall-time thresholds gate correctness CI.

## Timing scopes and resource limits

CPU measures ordinary `aleph run --backend sv`, with
`ALEPH_BENCH_CPU_DIAGNOSTICS=1` emitting a numeric stderr receipt. stdout remains
ordinary output; the adapter validates count widths and shot totals. Read/parse,
execution, sampling, amplitude conversion and output formatting are separately
timed. The reference CPU campaign is deliberately an identical-arm control,
without claiming to integrate the deferred #551 CPU optimizer.

CUDA measures `Circuit.from_qasm`, `aleph.run` and one lazy `statevector()` read,
including result release. Native diagnostics add execution and sampling stages.
The controller also records the entire subprocess wall time separately: imports,
warmups and preflight CPU oracle are included there, outside the reported API
caller clock. `warmup_calls=1` means one same-process full public-call warmup,
**with a new backend per call**. Zero means fresh-process/runtime-cold, not a cold
OS cache, driver, GPU or JIT cache. CLI always runs a fresh process and rejects
warm labels. CUDA `startup_ms` measures worker start through imports, not OS
process launch. CPU startup cannot be isolated and is `null`; subtracting stage
sums would confound startup with I/O, allocations and teardown. Unavailable stages
are `null`, never fabricated zeroes. Do not add overlapping API/native stages.

The CUDA preflight checks the complete timed workload against CPU amplitudes.
Every sample verifies counts, strict FP32/FP64 mode, finite normalized amplitudes,
no eager state transfer, exact state/download sizes, one materialization and
released result memory. Process peak RSS includes imports and warmups; device
byte counters describe public result ownership, not total allocator peak.

Load averages, CPU count, competing runner/cargo/bencher process names and GPU
identity/utilization/free memory are sampled before, every 0.5 seconds during,
and after each subprocess. Each snapshot has its own timestamp. Pre/post GPU
utilization and CPU load/competitors throughout reject contaminated campaigns.
During-call GPU utilization includes our own work, so it is recorded for review
but cannot identify other tenants. Load averages lag, polling can miss short
bursts, and a shared service can still contaminate accepted results. No monitor
proves exclusive use. A rejected campaign stops; it does not quietly select only
fast repeats or disable services. Isolation requires separate user authorization.

## Validation

```sh
ALEPH_BENCH_TEST_CLI="$PWD/target/release/aleph" \
  python -m unittest discover -s scripts/benchmarks -v
```

Tests cover schema/budgets, warm labels, alternating pairs, inert counter gates,
subprocess errors, explicit hardware skips, missing receipts/counters and resource
rejection. CI executes the real CPU public entrypoint without treating noisy CI
numbers as performance evidence. Real CUDA campaigns remain explicit hardware
runs and must have executed receipts; no CI compile success is called a GPU pass.

Revalidate a saved directory without executing hardware:

```sh
python scripts/benchmarks/public_workflow.py --validate-result /tmp/cuda-fusion-001
```

This checks receipts, raw pair order/count, workload hash, preflight gate and
recomputes the summary. A valid skipped/failed record still exits 2. Native
artifacts are pinned across the campaign; changing the executable or imported
wheel mid-run aborts. Build commands and RUSTFLAGS are recorded per arm as
operator declarations, alongside the controller's detected compiler identity.
CUDA reference campaigns require the public execution API from PR #583.
