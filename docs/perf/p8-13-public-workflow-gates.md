# P8-13: reproducible public workflows and experiment gates

The reference controller is `scripts/benchmarks/public_workflow.py`. Its versioned
spec/receipt validation, commands and timing/resource contract are documented in
[`scripts/benchmarks/README.md`](../../scripts/benchmarks/README.md). It exercises
ordinary CPU `aleph run` and CUDA Python `aleph.run`, not a substitute kernel.

## Admission before optimization

1. Declare the exact output, model, precision/error bound, algorithm, source and
   artifact, thread/worker budget, device/topology and cold/warm scope. Separate
   same-algorithm comparisons from workflow algorithm changes and old snapshots.
2. Profile the unchanged path. Estimate its bottleneck fraction and the maximum
   possible end-to-end benefit; a target outside that bound is a no-go. The
   manifest records the hypothesis and an explicit stop rule. It does not pretend
   an operator's hypothesis is a hardware measurement.
3. Run two public-route preflight samples. A swept optimizer/threshold must change
   the requested executed counter, not just appear in a command. Otherwise stop.
   For a crossover, choose sizes around the actual observed path transition.
4. Check full-output correctness before a broad campaign, then use interleaved
   pairs and keep raw data, uncertainty, resource snapshots and all failures.
   A discarded/contaminated campaign remains evidence of a failed experiment.
5. Retain only measured improvements or document/revert a no-go. Do not turn a
   best-of ceiling, lower precision or a different decoder into a same-contract
   production speedup. Small gains need substantially more independent pairs and
   an uncertainty analysis than the five-pair reference campaign.

## Comparison ledger

| Evidence | Classification and treatment | What a new comparison must hold fixed |
|---|---|---|
| New CPU CLI GHZ identical-arm control | Current public-route/resource control; no intended optimization | Executable hash, FP64, seeded QASM, shots, one worker/thread; fresh process |
| New CUDA Python raw/fused brickwall | Current same-algorithm public-route comparison | Same native wheel, FP64, state output, shots, thread budget; only optimizer changes |
| [Sparse Blossom](q1-03b-sparse-blossom.md) | Current mainline decoder **implementation**, historical timing snapshot; sparse replaces the former dense matching default | Recalibrate the **sparse** path and PyMatching on identical DEM/syndromes, shots, precision and one core before a new speed claim |
| [GPU QEC Monte Carlo](qec-q3-montecarlo.md), GPU UF vs CPU UF | Historical **same-decoder** snapshot, 4.2x aggregate in that report; not a current CPU baseline | UF on both sides; equal noise model and correction/rate acceptance; report resource difference explicitly |
| Same report, GPU UF vs CPU dense MWPM | Historical **cross-algorithm** workflow comparison (11.8x to 107x headline); dense MWPM is archived as a baseline for current Sparse Blossom | New MWPM claims must use sparse CPU matching; retain UF/MWPM accuracy separately; never pool with same-decoder table |
| [Q1-03c PR #578](https://github.com/aleph-sim/aleph/pull/578) | Unmerged allocation candidate, not a mainline baseline | Keep candidate and main revisions separate; do not silently attribute its results to main |

The new run manifests keep comparison classes separate. This issue does not
reclassify historical measurements as newly executed calibrations or substitute
a UF/BP decoder for MWPM. A new decoder optimization must rerun the sparse/PyMatching
baseline; the simulator reference campaigns below are not decoder benchmarks.

## Historical no-gain cases and stop rules

| Case | Evidence and cause | Gate before repeating work |
|---|---|---|
| CPU chunk sweep | [P2-04](phase2-p2-04.md): 360 cells, `min_amps` below every perf-relevant state; branch never changed | Require path/counter change; sweep state size near cutoff before a grid |
| Metal MPS pipeline | [Phase 5.8](phase5.8.md): removed transfers, still ~16x/22x slower at n=16/20; single-threadgroup factorization is limiting | Benchmark a competitive factorization first; stop another pipeline rewrite without a credible crossover |
| TF32 width-4 margin | [P5.11-05](p5.11-05-tf32-fused-block.md): best-of-5 1.02x random win; [P8-02](p8-02-public-cuda-execution.md) rejects a coherent-state error counterexample | Full-state error gate before timing; paired repeated same-precision baseline, report tiny gains as uncertain |
| CUDA native CP FP32 | [P8-02](p8-02-public-cuda-execution.md): fewer gate launches yet ~6.4% slower caller wall; setup and phase work can dominate | Retain raw control and public wall clock; launch count alone cannot accept a speed claim |
| Distributed model contamination | [P6-05 §4.7–4.8](p6-05-compiler.md): shared service/watchdog restarted during timing, one cell skewed 49% | Monitor throughout; stop on contamination. Ask separately before isolation; restore explicitly authorized services |
| Sparse matching allocations | [Q1-03b](q1-03b-sparse-blossom.md): guessed micro-allocation changes did not remove the ~2.5x gap to PyMatching | Profile current sparse implementation before edits; use the same decoder and syndrome set |

## Validation and reference campaigns

The hardware-free suite tests malformed specs/receipts, finite budgets/stages,
subprocess failures, explicit hardware skips, alternating order, inert counters
and resource rejection. CI additionally runs the real CPU CLI and checks its
counts/stage receipt without imposing timing thresholds. GPU work requires an
executed hardware receipt, full CPU amplitude oracle and public transfer/memory
guards. A missing SDK/device is a failed requested run, never a passing test.

Reference run evidence and conclusions are recorded below after execution. CPU
startup is not separately observable and stays null; CUDA import startup and
entire process wall are distinct from its warmed API-call clock. Polling and load
averages are useful contamination checks but cannot prove an isolated machine.
