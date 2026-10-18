# Phase 8: performance recovery and remaining gates

Tracker: [#550](https://github.com/aleph-sim/aleph/issues/550). Status snapshot:
**2026-10-18**, main `6114dab7e18c4deac0fda3039060e00f7661b653`.
**13 of 16 capability issues are closed; Phase 8 is not complete.** P8-01,
P8-11 and P8-12 remain open. This report integrates existing measurements; it is
not a new all-backend benchmark campaign or a v1.0 release qualification.

## What changed for callers

The audit exposed useful optimized implementations that public calls did not
reach, unnecessary GPU initialization and amplitude downloads, and memory policies
that counted the state but missed the requested output. The largest confirmed
repairs remove work: discarded backend construction, full-state exports, scan
passes and per-shot buffers. Kernel improvements alone do not remove setup,
sampling, result conversion or decoder accuracy limits.

| Workflow | Measured result | Scope and qualification |
|---|---|---|
| CPU CLI cold process | 194.561 → 1.271 ms on the 2-qubit explicit statevector request | P8-08 removed discarded CUDA construction on the CUDA-enabled Linux CLI; full-process timing, not isolated startup or CPU gate throughput. Three fresh processes per revision on a shared host. |
| CUDA Python counts-only | FP64 n=26 public run/counts 927.076 → 272.642 ms; process peak RSS 2208.562 → 160.457 MiB; run DTOH 1,073,742,096 → 272 B | P8-03 removes eager amplitudes. First explicit state read pays deferred work; read-workflow peak RSS and sampled GPU peak did not improve. |
| CUDA public fusion | n=22 FP64 brickwall Python caller 348.510 → 258.380 ms; median paired ratio 1.342, bootstrap 95% interval 1.319–1.404 | P8-13 rechecks P8-02 with the newer sampler. Includes parse/run/first state read/result release; five pairs on one shared GPU, not a universal crossover. |
| CUDA large-state sampling | n=22 /1024 shots FP64 7.1560 → 0.47761 ms (14.98×), FP32 6.8450 → 0.44612 ms (15.34×) | P8-05 sampling boundary only; preparation/JIT excluded. n=22 is the largest state measured in that campaign. |
| CPU one-shot sampling | FP64 AoS n=20 10.366 → 0.820 ms; additional sampler heap peak 36 MiB → 8 B | P8-06 prepared-state sampler, one worker. No Python/CLI whole-call ratio is inferred. |
| Noisy trajectories | n=20 /8 mixed shots, one worker: allocated bytes 142,614,028 → 25,167,408 | P8-07 reduces allocation traffic and bounds owned contexts. Same-concurrency large-state peak is essentially unchanged; wider-state time ratios 0.983–1.044× do not establish a gain. |
| DEM experiments | 1M shots: additional live requested heap 102,672,376 → 6,741,056 B (15.23× lower) | P8-15 default batches, fixture with 128 detectors/3 observables; exact tallies match. No material throughput gain established. |

The ratios above have different baselines, timing scopes, hardware and requested
outputs. **Do not multiply them into a Phase 8 speedup.** All quoted timings are
historical child-campaign observations, not timings rerun on this report's main
revision. Later source integration checks establish compatibility, not identical
performance on every subsequent revision.

## Capability and adoption ledger

“Closed” describes GitHub task state; adoption is separate. Research prototypes
can meet their criteria while remaining explicit opt-ins. The linked child reports
retain precise oracles, provenance, raw observations and adverse cases.

| ID / issue | State / PR | Mainline behavior and evidence |
|---|---|---|
| P8-01 / [#551](https://github.com/aleph-sim/aleph/issues/551) | Open; draft [#569](https://github.com/aleph-sim/aleph/pull/569) | Optimized CPU/Metal CLI candidate awaits idle-Metal timing. Main CPU CLI still uses the raw driver. Draft CPU measurements are not mainline results. |
| P8-02 / [#552](https://github.com/aleph-sim/aleph/issues/552) | Closed; [#583](https://github.com/aleph-sim/aleph/pull/583) | Default strict FP64/FP32 width-3 CUDA plans in CLI/Python; raw control retained. [Report](p8-02-public-cuda-execution.md), [data](data/p8-02/summary.json). |
| P8-03 / [#553](https://github.com/aleph-sim/aleph/issues/553) | Closed; [#568](https://github.com/aleph-sim/aleph/pull/568) | Lazy Python GPU amplitudes, copy-once host cache; results retain device amplitudes until materialized/dropped. [Report](p8-03-lazy-gpu-amplitudes.md), [manifest](data/p8-03-manifest.json). |
| P8-04 / [#554](https://github.com/aleph-sim/aleph/issues/554) | Closed; [#581](https://github.com/aleph-sim/aleph/pull/581) | Operation-aware device/host preflight; no counts-to-norm substitution. [Report](p8-04-memory-planning.md), [allocation audit](data/p8-04-allocation-audit.jsonl). |
| P8-05 / [#555](https://github.com/aleph-sim/aleph/issues/555) | Closed; [#582](https://github.com/aleph-sim/aleph/pull/582) | Default deterministic hierarchy/local-CDF policy; old scan selectable from Rust. [Report](p8-05-hierarchical-sampling.md), [raw estimates](data/p8-05/criterion.json). |
| P8-06 / [#556](https://github.com/aleph-sim/aleph/issues/556) | Closed; [#571](https://github.com/aleph-sim/aleph/pull/571) | Zero/one-shot CPU path; alias remains at ≥2 shots. [Report](p8-06-single-shot-sampling.md), [manifest](data/p8-06-manifest.json). |
| P8-07 / [#557](https://github.com/aleph-sim/aleph/issues/557) | Closed; [#572](https://github.com/aleph-sim/aleph/pull/572) | Bounded reusable trajectory contexts; sparse CLI/Python counts, dense Rust compatibility. [Report](p8-07-bounded-noisy-trajectories.md), [manifest](data/p8-07-manifest.json). |
| P8-08 / [#558](https://github.com/aleph-sim/aleph/issues/558) | Closed; [#567](https://github.com/aleph-sim/aleph/pull/567) | Lazy discovery; selected backend still initializes per call. [Report](p8-08-cheap-gpu-probes.md), [cold Python data](data/p8-08-cuda-cold-python.json). |
| P8-09 / [#559](https://github.com/aleph-sim/aleph/issues/559) | Closed; [#592](https://github.com/aleph-sim/aleph/pull/592) | Explicit edge-major/warp-tiled BP scratch; shot-major default retained. [Report](p8-09-bp-scratch.md), [raw evidence](data/p8-09-bp-scratch/README.md). |
| P8-10 / [#560](https://github.com/aleph-sim/aleph/issues/560) | Closed; [#593](https://github.com/aleph-sim/aleph/pull/593) | Explicit Rust cooperative FP64 BP; serial default retained. [Report](p8-10-cooperative-bp.md), [raw evidence](data/p8-10-cooperative-bp/README.md). |
| P8-11 / [#561](https://github.com/aleph-sim/aleph/issues/561) | Open | Competitive multi-threadgroup Metal MPS factorization unproven. [Phase 5.8](phase5.8.md) is historical negative evidence, not a new prototype verdict. |
| P8-12 / [#562](https://github.com/aleph-sim/aleph/issues/562) | Open | Current compiler needs actual multi-GPU measurements. [Compiler model](p6-05-compiler.md) gains do not establish measured scaling. |
| P8-13 / [#563](https://github.com/aleph-sim/aleph/issues/563) | Closed; [#584](https://github.com/aleph-sim/aleph/pull/584) | Public-route receipts, counter admission and contamination gates. [Report](p8-13-public-workflow-gates.md), [harness](../../scripts/benchmarks/README.md). |
| P8-14 / [#564](https://github.com/aleph-sim/aleph/issues/564) | Closed; [#594](https://github.com/aleph-sim/aleph/pull/594) | Explicit FP64 residency + overlap; defaults unchanged. [Report](p8-14-resident-overlap.md), [raw evidence](data/p8-14-resident-overlap/README.md). |
| P8-15 / [#565](https://github.com/aleph-sim/aleph/issues/565) | Closed; [#575](https://github.com/aleph-sim/aleph/pull/575) | Default 65,536-shot batches and exact decoder-result schema checks. [Report](p8-15-bounded-dem-experiments.md), [manifest](data/p8-15-manifest.json). |
| P8-16 / [#566](https://github.com/aleph-sim/aleph/issues/566) | Closed; [#570](https://github.com/aleph-sim/aleph/pull/570) | Explicit direct-binomial bounds; legacy Wald fields remain labeled. [Contract](../qec/binomial-intervals.md), [coverage](data/p8-16-binomial-coverage.json). |

Follow-up [#573 / PR #574](https://github.com/aleph-sim/aleph/pull/574)
validated manual CPU fused IR and reduced repeated validation overhead. Its
[report](p8-573-manual-fused-ir.md) preserves cold first-use regressions and
AVX512 measurements. Sparse Blossom [#578](https://github.com/aleph-sim/aleph/pull/578)
and [#595](https://github.com/aleph-sim/aleph/pull/595) are now merged too; their
[allocation](q1-03c-reuse-allocations.md) and [event-scan](q1-03c-event-scans.md)
results are related QEC work, not extra P8 capability completions. #511 was closed
by user acceptance at 1.468× the original d=11 baseline, below its 1.5× target.

## Timing, launches and output costs

### CPU CLI and discovery

The P8-08 Linux startup comparison used separate locked release binaries/wheels,
three interleaved fresh processes per revision, Bell n=2 /32 shots /seed 7.
CLI cold time includes startup, execution/output and exit. Python cold time also
includes interpreter/import; warm time is the median of calls 2–20, not a warm
retained backend. Explicit Python SV cold wall fell 212.764 → 24.286 ms; selected
CUDA-FP32 cold wall fell 373.970 → 217.536 ms. Candidate construction counters are
zero for CPU/Clifford-auto and one per explicit GPU call. No module cache was added.

P8-13's current CPU route is a **no-optimization control**: GHZ n=8 /16 shots,
identical executable, 1.542 vs 1.522 ms; paired bootstrap interval 0.396–1.714.
It establishes a working receipt boundary, not a CPU fusion win. CPU startup
cannot be isolated by subtracting overlapping native stages and stays `null`.
The draft #569 has CPU wins and QFT/tiny-input regressions, but its Metal gate and
integration review are pending. Its `--raw`/`--execution-stats` switches must not
be documented as available on this main revision.

### CUDA Python: execution is only part of caller time

P8-02's n=22 /32 shots /seed 552 same-artifact raw/optimized campaign spans five
families and both precisions in CLI/Python. Public caller improvements span
1.07–4.11× in its 20 cells, with three pairs per cell. It predates P8-05; the
sampling algorithm is held fixed within that experiment. FP64 decomposed-QFT
Python execution falls 480.90 → 39.94 ms and gate launches 1177 → 43, while fusion
planning grows 0.034 → 0.545 ms. Caller time falls 581.09 → 141.50 ms, a smaller
ratio. FP64 backend/JIT initialization is about 20 ms and context/preflight
typically 64–79 ms in that campaign. These native stage observations exclude
parts of final teardown; external caller time remains authoritative. Native-CP
FP32 QFT instead regresses (see below).

P8-13 subsequently measures the combined mainline planner/sampler through public
Python, using five pairs and one preflight pair per campaign, one thread/worker,
and a full same-process warmup with a new backend each call. At n=22 brickwall,
launches fall 327 → 37 and execution 159.754 → 49.039 ms. Sampling is about
0.59–0.62 ms; first state materialization is about 94–107 ms. The 1.342 paired
caller ratio includes parse/run/read/result release, while total subprocess wall
also includes imports/warmups/preflight. These clocks are kept separate.

For counts-only, P8-03's n=26 FP64 run transfers only 272 B (norm/sampling scalars
plus 32 outcomes). An explicit first state read downloads 1,073,741,824 B and
creates a complex128 host cache; repeated reads copy independent NumPy arrays
without another download. The first-read median grows 255.782 → 897.405 ms as
work moves out of run. Read-workflow peak RSS stays about 2208.6 MiB. A counts-only
result still owns 1 GiB of amplitudes until read or destroyed: retaining many such
results can exhaust GPU memory. Logical ownership counters are not allocator peaks.

### Sampling, allocation and reach are different claims

P8-05's n=22 /1024-shot sampler changes construction from 23 launches to two
hierarchy kernels (three with the local CDF). Two norm launches and one search
launch remain. Hierarchical scratch is 132,112 B instead of 67,108,864 B for
the reference's two full CDF buffers. Measured live pool peaks become 67,519,520 B
FP64 and 33,965,088 B FP32 versus 134,496,272 /100,941,840 B. These exclude driver,
module and reserved pool pages. The timed boundary includes norm, guards,
allocation, RNG/upload, search and download, but excludes preparation and JIT.

The planner's admission of low-shot n=31 FP32 under a synthetic 20 GiB budget is
**modelled reach**, not an actual n=31 sampling run. P8-04 real allocation audits
use n=18; P8-05 maximum measured state is n=22. Device/host headroom and concurrent
allocation races remain; passing preflight reserves no memory. Counts never
silently become a norm. State, execution scratch, sampling and export must be
budgeted together using the actual plan and selected sampler.

P8-06 one-shot CPU draws remove the alias table; ≥2 shots keep it. P8-07 then
reuses one dense state/Kraus scratch per bounded worker context. These sequential
repairs have different baselines. In P8-07, n=20 with 1/2/4 contexts measures
16/32/64 MiB additional heap, below conservative bounds. Lower allocated bytes
does not mean lower same-concurrency peak or fewer full-state reset/channel passes.
The default 1 GiB incremental buffer policy is not an RSS cap; outcomes remain
O(shots), and Python dictionaries/CLI formatting are outside that Rust budget.

## Why some approaches stay opt-in or did not pay

| Finding | Retained evidence | Consequence |
|---|---|---|
| Fewer launches need not improve small public calls | P8-13 n=8 FP64 and n=18 FP32 reduce execution/launches but paired caller intervals include 1 | Stop kernel-only tuning when setup/output dominates; retain public clocks. |
| Native CP FP32 fusion can lose | P8-02 n=22 caller 144.32 → 153.60 ms; reusable-plan execution 13.982 → 19.838 ms | Raw control remains; a precision/size crossover needs calibration before new routing. |
| TF32 width 4 violates the strict public contract | P8-02 coherent-state amplitude error 1.4084e-4 vs CPU FP64 | Public plans keep ordinary arithmetic and width 3; tolerance was not relaxed. |
| Small/high-shot sampling has little scan work to remove | P8-05 n=14 /1M FP32 samples 7.47 → 8.13 ms | Auto is an empirical deterministic policy, not universally fastest. |
| Reuse removes allocations, not trajectory arithmetic | P8-07 wide-state ratios near 1; tiny four-worker Kraus aggregate 6.9% slower including the retained outlier | Accept memory/context control, not a universal speed claim. Final narrow replication improves 1.053× but does not erase prior variability. |
| Better coalescing can worsen locality/divergence | P8-09 warp-tiled d11 /4096 /.03 improves 1.44×; at .001 takes 1.58× longer | Keep shot-major default and explicit alternatives; no batch-size-only auto rule. |
| Cooperative BP shortens the per-shot serial path | P8-10 all 22 tested cells improve 1.78–35.97× against the faster serial reference | Explicit Rust opt-in on one measured GPU; no automatic policy or accuracy gain. |
| Residency and overlap interact | P8-14 held-out locality n=26 improves 1.36× vs residency, 1.42× vs equal-payload larger synchronous tile; shallow case 0.997×, alternating 0.999× vs faster existing path | Explicit FP64 combined executor; do not multiply historical gains or infer FP32 benefit. |
| Bounds improve statistical correctness, not execution | P8-16 0/1000 failures gives [0, 0.003682], not zero uncertainty | Direct fixed-iid binomial contract only; stratification/drift need their own estimators. |

BP timing is upload + reused decode + mask download/synchronization, excluding
packing, model/JIT/oracle and warm allocation. Identical CPU/GPU corrections do
not make BP a better decoder: surface11 /.03 /4096 still has 1809 logical failures
and 4093 CPU nonconvergences in P8-10. Its varied-input tails are empirical fixed
mixtures, not population latency guarantees.

Paging timing includes pinned/ring allocation, initialization, execution and drain,
but excludes backend/JIT and norm/export. Forced n=24/26 states fit in-core; this
isolates paging schedule effects and proves no new reach. Matched timelines show
residency and combined locality each move 2 GiB in each direction, while per-gate
overlap moves 40 GiB. Combining overlap saves wall time without further reducing
residency's bytes; copy/kernel busy durations overlap and cannot be summed as wall.

## Provenance and experiment quality

| Campaigns | Machine / software | Comparison discipline |
|---|---|---|
| P8-02/03/04/05/08/09/10/13/14 | Shared Linux RTX 4000 SFF Ada, ~20 GiB, driver 580.178.04, CUDA/NVRTC 13, Rust 1.96.0. Early public Python: 3.12.3, NumPy 2.5.0. Exact builds/seeds in child manifests. | Services/clocks unchanged. Early campaigns use three pairs and boundary utilization; later campaigns keep fixed mirrored blocks, continuous telemetry and separate profiles. None proves exclusive GPU use. |
| P8-06 | i5-13500, Linux 6.8, Rust 1.96, CPU0, one Rayon worker, seed 556 | Independent Cargo targets, A/B/B/A; 10 Criterion samples/repeat. Invalid identical-binary pilot discarded. |
| P8-07 | Dual Xeon Silver 4114, Linux 7.0, Rust 1.96, CPUs 3/7/9/11 with SMT siblings monitored, 1/4 workers, seed 557 | Per-case A/B/B/A, external-overlap admission; policy changes, rejected attempts, outlier and final follow-up all retained. |
| P8-15 | Same Xeon host, one worker on CPU3, monitor CPU39, native release flags, seed 565 | A/B/B/A with CPU3/23 guard; common-stage and total boundaries are separate. |
| P8-03 Metal | macOS arm64, Python 3.12, NumPy 2.3.5 | Retained earlier eager-path wheel, not rebuilt CUDA baseline revision; unified-memory RSS is not PCIe traffic. No new Metal timing campaign here. |

Child manifests pin source/binary hashes, flags, exact inputs and raw samples.
Use those identities to reproduce a historical result; use current main with
new hashes to establish a new baseline. Several campaigns found stale shared-target
artifacts or CI interference. These failed experiments remain documented and are
excluded from accepted summaries. A feature compile, missing-device skip, or an
unchanged counter cannot stand in for actual hardware execution.

P8-13's saved receipts can be revalidated without CUDA. Retained `failed` and
`contaminated` campaigns stay non-successful; they must not be dropped merely
because executed campaigns are easier to summarize. Hardware oracles, allocation
counters, sampled RSS/VRAM and analytical budgets answer different questions.
Unavailable stage/peak observations remain unavailable, rather than being filled
with zero or guessed from allocation formulas.

## Reproduce and check the current integration

Run from the repository root. These bounded CPU commands check public contracts
and saved evidence on a busy machine; **their timings are not performance data**.
The noisy contract tests cover the mixed Pauli/Kraus/readout path and worker/seed
invariance without a new expensive trajectory campaign.

For this report update, the release CPU CLI built successfully and the commands
below passed **32 Rust + 16 Python tests**, including the actual public CLI guard.
One existing external PyMatching test remains ignored; no new PyMatching or GPU
run is claimed. All six saved P8-13 server campaigns revalidated (four executed,
one failed, one contaminated), preserving their original classifications. The
Bell smoke returned 18 `00` and 14 `11` outcomes. [Validation/provenance](data/p8-00/README.md)
records the exact base, CLI hash, commands and logs. This is scoped verification,
not a new full-workspace suite or hardware performance campaign.

```sh
cargo build --locked --release -p aleph-cli
ALEPH_BENCH_TEST_CLI="$PWD/target/release/aleph" \
  python3 -m unittest discover -s scripts/benchmarks -v
cargo test --locked -p aleph-backend --lib memory::tests
cargo test --locked -p aleph-sv --test single_shot --test noise_driver --test noise_oracle
cargo test --locked -p aleph-qec --test bounded_dem_experiment --test binomial_interval
python3 -m unittest scripts/python/test_qec_threshold_plot.py -v
python3 scripts/benchmarks/public_workflow.py --validate-result \
  docs/perf/data/p8-13/server/cuda-fusion-settled
```

For an ordinary current CPU CLI smoke, create a small input and check counts:

```sh
printf 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0],q[1];\n' > /tmp/p8-bell.qasm
target/release/aleph run /tmp/p8-bell.qasm --backend sv --shots 32 --seed 7
```

On a prepared Linux CUDA installation, use a release CLI built with `--features
cuda` and a separately extracted `python,cuda` wheel on PYTHONPATH. GPU tests
below must actually execute; missing hardware is not a pass. These are optional
hardware reproduction commands, not tests run for this documentation update:

```sh
target/release/aleph run /tmp/p8-bell.qasm --backend cuda-f32 --shots 32 --cuda-diagnostics
target/release/aleph run /tmp/p8-bell.qasm --backend cuda-f32 --shots 32 --cuda-raw --cuda-diagnostics
ALEPH_TEST_PUBLIC_CUDA=1 ALEPH_CUDA_CLI="$PWD/target/release/aleph" \
  python3 scripts/python/test_public_cuda_execution.py -v
ALEPH_TEST_GPU_BACKENDS=cuda,cuda-f32 \
  python3 -m unittest discover -s scripts/python -p test_gpu_results.py -v
ALEPH_REQUIRE_CUDA=1 cargo test --release -p aleph-cuda --features cuda \
  --test qec_bp_oracle -- --nocapture
```

```python
import aleph
c = aleph.Circuit.from_qasm('OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0],q[1];')
r = aleph.run(c, backend="cuda-f32", shots=32, seed=7)
print(r.counts())                 # counts-only; device amplitudes remain owned
print(r.execution_stats())
state = r.statevector()           # explicit download/widen/cache + independent array
del r                            # release cache/device ownership when no longer needed
```

Use the child report commands for sampling Criterion, memory audits, noisy budget
examples and paging/BP raw-data checkers. Public A/B campaigns use the
[P8-13 harness](../../scripts/benchmarks/README.md) with edited absolute artifact
paths/full revisions and a new output directory. Do not reuse placeholder
revisions, shared Cargo targets or a busy Mac for performance acceptance.

## Remaining exit gates and next decisions

1. **P8-01 / #551:** integrate current main into draft #569, review semantics and
   rerun CPU/Metal oracles. Finish paired public Metal timings on a free Mac,
   including small planning overhead and losing workloads. Until then CPU/Metal
   CLI optimizer integration remains pending; CUDA integration is already done.
2. **P8-11 / #561:** benchmark representative SVD/QR shapes and guards against
   faer/current Metal before another full pipeline rewrite. Accept a competitive
   measured regime or a documented no-go. Historical 16–22× MPS slowdowns do not
   count as having run this experiment.
3. **P8-12 / #562:** obtain actual 2/4-GPU measurements with topology, work/pass
   inflation and communication attribution. Compiler predictions and emulated
   ranks do not close measured scaling. MPI/v1.0 work remains separate Phase 6.
4. **P8-00 / #550:** update this report and tracker as those three land; smoke-run
   final documented routes and preserve adverse results. Only then evaluate the
   phase exit criteria and close the tracker. This report PR uses **Refs #550**.

Measured follow-up opportunities are native-CP FP32 routing, per-call context/JIT
costs, high-shot host sampling work, and noisy tail variability. Each needs a
separate profile, fixed output/precision, admission rule and focused issue before
implementation. The phase does not authorize another broad rewrite merely because
an individual kernel became faster.
