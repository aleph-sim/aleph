# P8-15: bounded DEM experiments and exact decoder contracts (#565)

`run_dem_experiment` previously materialized every syndrome, truth and prediction
before scoring. A short prediction vector silently truncated `zip`, while the
result still used the requested shot count as its denominator. Missing observable
bits were treated as false, and extra bits were ignored.

The experiment now processes at most 65,536 shots per batch by default. Call
`run_dem_experiment_with_options` with `DemExperimentOptions { batch_size }` to
trade memory against decoder batch startup cost. Zero is rejected, even for an
empty experiment. `sample_shots` remains an explicit, unbounded artifact API.

Each shot still uses the unchanged `splitmix64(seed, absolute_shot_index)` RNG
stream and unchanged mechanism order. Batch boundaries and Rayon scheduling do
not change samples. Batch-independent decoders therefore produce identical exact
tallies. The same decoder instance is reused; decoder caches are not reset and
external PyMatching still receives one vectorized call per batch, rather than
one subprocess per shot. Models with zero observables now get one empty
PyMatching correction per shot instead of an empty prediction batch.

Before scoring a batch, the harness checks the exact prediction count and every
prediction's exact observable width. Violations return `Error::DecoderContract`
with a zero-based absolute shot (batch start for a count mismatch). Decoder errors
propagate. Checked additions accumulate the processed/error counts; a result is
constructed only after every requested shot has been processed. There is no
partial/budget-stopped estimator or silently unprocessed denominator.

## Memory boundary

Harness memory scales with batch size and DEM dimensions, not total shots:
syndrome/truth/prediction buffers plus dense sampling scratch per active worker.
This is a shot-count policy, not an absolute byte/RSS budget. Decoder-owned caches,
external-process memory and maliciously oversized decoder outputs are outside the
harness bound. A sufficiently large model or explicitly large batch can still
exhaust memory. Packed representations and scratch reuse are deferred; this change
establishes the correct batch/contract boundary without changing the sampler.

The identical `dem_experiment_memory` example was built on baseline and candidate.
It measures additional peak live requested heap bytes through a counting System
allocator, after a one-shot Rayon warm-up; it excludes allocator metadata and RSS.
Its timings are not used. The fixture has 128 detectors, three observables, 128
independent probability-0.02 mechanisms, seed 565, and two Rayon workers.

| Shots | Baseline peak bytes | Candidate peak bytes | Baseline/candidate | Exact errors (both) |
|---:|---:|---:|---:|---:|
| 10,000 | 1,036,872 | 1,036,872 | 1.00× | 7958 |
| 100,000 | 10,278,792 | 6,739,816 | 1.53× | 79767 |
| 1,000,000 | 102,672,376 | 6,741,056 | 15.23× | 796673 |

## CPU measurements

Baseline: `c283272747877863fa4a2723421c8d710a8550b3`. Both versions use the
identical Criterion harness and independently built Cargo targets. Only benchmark
manifest/lock and measurement harness files are added to the baseline; its
experiment implementation remains unchanged. Candidate source hashes, executable
hashes, raw accepted estimates/samples and guard events accompany this report.
Different executable hashes alone are not proof of correct source provenance.

Host: user-authorized `root@51.159.0.49`, dual Xeon Silver 4114, 40 logical CPUs.
Release/bench builds use `RUSTFLAGS='-C target-cpu=native'`, thin LTO and one codegen
unit. No services or CI jobs are stopped. Timed runs use one Rayon worker pinned
to CPU 3, monitor CPU 39, and watch physical-core siblings 3/23. The script checks
topology, rejects identical binaries, requires <=2% preflight busy on both siblings
and accepts <3% sampled external CPU overlap. Runs are ordered A/B/B/A per cell;
10 Criterion samples, 200 ms warm-up and 500 ms requested measurement (extended
by Criterion when necessary). The mean of the two run medians is reported.

This is bounded shared-host evidence. Brief tasks/migrations, shared cache pressure,
turbo and other unsampled interference can escape the guard. Small apparent gains
are not presented as established speedups. Sampling/decode/scoring rows isolate
common stage work on 1,024 shots; the score row uses an identical direct comparison
in both binaries, not each version's private validation/scoring loop. Total rows
include the actual public harness, validation and allocation/free work.

| Workload | Baseline ms | Candidate ms | Candidate time change |
|---|---:|---:|---:|
| `mwpm_d3_total_1024` | 0.809699 | 0.796324 | -1.65% |
| `null_decode_1024` | 0.064297 | 0.063873 | -0.66% |
| `null_total/1024` | 1.043542 | 1.037781 | -0.55% |
| `null_total/131072` | 135.483343 | 133.541984 | -1.43% |
| `sample_1024` | 0.939768 | 0.962249 | +2.39% |
| `score_1024` | 0.004056 | 0.004155 | +2.45% |

All total cells are within 1.7% of baseline; no material throughput change is
established. Common-stage differences reach 2.5%, consistent with treating these
small timing deltas as shared-host variation. These microbenchmarks are not an
additive decomposition of the public-run totals. The guard accepted 24 runs and
rejected two overlap-contaminated attempts; maximum accepted sampled overlap
was 1.08% of one CPU.


For external decoders, campaigns above the batch limit now pay multiple process
startups. The default amortizes each startup over up to 65,536 shots; callers may
choose a larger batch when memory permits. No universal PyMatching throughput gain
is claimed from the in-process Null/MWPM timing cells.

## Validation and reproduction

Final Linux validation: **25 passing tests/examples**, comprising five sampler/harness
unit tests, two PyMatching unit tests, eight new bounded-contract integration tests,
two existing memory-experiment tests, three rustdoc examples, one real PyMatching
integration test and four Stim DEM oracle tests. Existing ignored threshold/profile
tests remain ignored; the new PyMatching test was explicitly run with `--ignored`.
Local new integration tests (8), workspace all-target clippy, rustfmt and diff checks
passed. The identical-binary guard negative test also passed.

Artifacts: [memory](data/p8-15-memory.csv), [timings](data/p8-15-summary.csv),
[raw estimates/samples](data/p8-15-criterion.json), [guard events](data/p8-15-guard.jsonl),
[source/binary manifest](data/p8-15-manifest.json), [validation logs](data/p8-15-validation.txt).

Coverage includes wrong short/long/empty prediction batches, missing/extra
observable bits, failures after a valid prefix, zero shots, exact/tail batches,
the default limit, multiple observables, deterministic mechanisms, absolute stream
identity across batch sizes/workers, and a `u64::MAX` request that fails on its
first two-shot batch without materializing the campaign. Real PyMatching checks
full-batch versus 16-shot batching and zero/three-observable schemas. Existing
surface-code DEM/Stim comparisons exercise the unchanged generative model.

An exploratory broad QEC run was interrupted in pre-existing slow decoder tests;
it is not reported as a full-suite pass. The final targeted experiment, memory,
PyMatching and oracle runs below are the validation evidence. Repository-wide
runtime coverage remains subject to PR CI. Review performed manually; no independent
review agent or code-review skill was used.

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
RAYON_NUM_THREADS=2 cargo test --release -p aleph-qec --lib experiment::tests
cargo test --release -p aleph-qec --lib pymatching::tests
cargo test --release -p aleph-qec --test bounded_dem_experiment
cargo test --release -p aleph-qec --test memory_experiment
cargo test --release -p aleph-qec --doc
PYMATCHING_PYTHON=/path/to/python cargo test --release -p aleph-qec \
  --test bounded_dem_experiment -- --ignored
STIM_PYTHON=/path/to/python cargo test --release -p aleph-qec \
  --test surface_dem_stim_oracle -- --ignored
RAYON_NUM_THREADS=2 cargo run --release -p aleph-qec \
  --example dem_experiment_memory -- 1000000
RAYON_NUM_THREADS=1 cargo bench -p aleph-qec --bench dem_experiment
# Copy independently built executables to a fresh artifacts directory first:
python3 scripts/python/bench_dem_experiments_guard.py /path/to/artifacts
```
