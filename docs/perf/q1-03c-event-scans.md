# Q1-03c follow-up: Sparse Blossom event scans (#511)

This continues [allocation reuse](q1-03c-reuse-allocations.md). Reuse removed
warmed internal allocations but did not meet #511's original 1.5× d=11 target.
The issue's initial attribution of the entire PyMatching gap to allocations was
incorrect: a fresh steady-state profile of current main puts 59.70% of sampled
cycles in `next_at` and 21.73% in `run_to_matching` (zero lost perf samples).
These percentages identify work; they are not speedup measurements.

## Change and invariants

The final change is a single scheduling guard: `schedule_node` skips scanning
when a look-at is already queued at the current time. Every `next_at` result is
clamped to that time, so none can improve the existing entry. Its sequence and tie
position remain intact. The scanner's production body is unchanged from main;
integer arithmetic, adjacency order, strict `<` tie selection, pop-time scanning,
weights and public APIs are unchanged.

For `queued == now`, the old code could push only when `t < queued`, while the
scanner returns `t >= now` or `None`. It therefore cannot push in either case.
The queued payload carries the node id; pop-time scanning reads the current
state after intervening matcher updates. Reset and valid pop still clear the
queued-time marker as before. Manual self-review found no blocking findings;
this is self-review, not an independent review service.

Test builds execute that unchanged scanner on the skipped path to assert its
lower bound. Test-only accounting excludes the assertion's scan from actual work
counts. A focused heap test checks same-time sequence preservation and earlier
rescheduling of a future entry. Production state contains no added counters or
`Cell`. The larger neighbour-filtering experiment and its original per-transition
scanner oracle are retained as rejected evidence, not shipped as production code.

The original two-guard 256-shot diagnostic recorded 54,796 queued-now skips
against 173,924 actual scanner calls at d=11 p=.03: about 24% of otherwise
requested scans are avoidable. This is work accounting, not a speedup estimate.
The final queued-only diagnostic checks the same call/skip work separately.

## Measurement protocol

The [protocol](data/q1-511-events/protocol.json) was frozen before candidate
measurements. The CPU host is shared; no CI or services are stopped. All timed
processes use CPU 3 on the Xeon Silver 4114, with SMT sibling 23 idle at admission.
The driver requires <10% total activity and <1.1% on the core and sibling over
two seconds before starting each process, and retains one-second host activity
throughout. Shared-host limitations remain even with these controls.

Three independently built snapshots use Rust 1.96.0 and
`RUSTFLAGS="-C target-cpu=native"`:

- Original pre-reuse baseline: `5a671ae26b2eb4235d024403952c2f27988bddfb`.
- Current main with allocation reuse: `ab3aed8027132f42f9df4e9ddc614403da5a3147`.
- Candidate in the held-out campaign: the queued-now guard only.

In the initial training manifests, `candidate` means both guards; in the final
held-out manifest it means queued-only. Binary SHA256 values and the retained
stage-specific source fingerprints identify each arm. Only main→candidate
isolates this patch; the historical arm quantifies cumulative change between
the snapshots.

The native Criterion fixture is byte-identical in the two baseline commits.
Cross-version probes serialize exact DEM probability bits, detector/observable
indices, 256 syndromes and correction bytes at six distances and two rates.
All three versions must produce identical files before timing. This prevents
an apparent cumulative improvement from changed workloads or corrections.

The d=11 ablation compares main, neighbour filtering alone, and both guards in
two mirrored blocks. The final campaign compares original/main/candidate at all
six distances in four mirrored blocks with rotated leading arms. Each arm has
eight process observations. Tables use the median of process Criterion mean
point estimates divided by 256 shots. Each native bootstrap interval and raw
sample is retained; no pooled confidence interval is inferred.

Public Python measurements use the unchanged `bench_mwpm_allocations.py` helper
and `bench_qec.py` fixtures: Stim circuit-level d=5/20,000 shots and d=9/5,000
shots, p=.003, seed 1, one Rayon thread, two ABBA blocks and seven samples per
process. Each version has 28 rate samples per case. Wheels are built separately
and extracted to isolated import paths; native-module paths and prediction
SHA256 values are checked.

## Training and variant selection

The initial three-arm ablation retained all 12 processes. Median d=11 times were
111.227 µs/shot for main, 109.528 for neighbour-only and 94.678 for both guards.
The both-guards/main block ratios were 1.169× and 1.186×. However, the first four
processes saw intermittent SMT-sibling activity (maximum 47.3%). This series is
training evidence, not the final acceptance measurement; no outliers were removed.

Before any held-out measurement, the protocol was amended to compare queued-only
against both guards in two mirrored two-arm blocks. Neighbour filtering must add
at least 2% in aggregate and each block to justify retaining it. Otherwise the
smaller queued-only change is preferred. The original protocol and amended one
are both retained. The second training attempt stopped during its fifth process when one recorded
interval reached 13.7% overall CPU (sibling 0%). The older tracer did not record
phases, so whether that interval touched decode is unknown. The initial attribution
to construction was too strong; see the retained protocol erratum. The whole
attempt remains invalid, with no retrospective phase reclassification. No
complete qualified two-block benefit was established, so the final variant uses
the predeclared smaller queued-only default. This does not prove neighbour
filtering harmful; the experiment remains inconclusive.

Before any held-out result, native host validity was refined to intervals touching
Criterion warmup or sample collection. The driver parses phase markers on both
ends of each one-second interval and conservatively includes crossing intervals.
It requires at least two active samples per case. Any active interval above 10%
overall CPU or 5% sibling activity stops and invalidates the whole campaign.
Construction/analysis-only activity is retained in the raw trace, outside the
decode measurement. Python retains the whole-process check. This phase rule,
its reason, and both earlier protocols are preserved; numerical adoption gates
are unchanged. No attempted results or statistical outliers are deleted.

## Validation boundaries

The two-guard production source passed 272 release QEC tests and 10 doc tests
(22 existing ignores plus the new explicit work diagnostic). This includes the existing
100,500-shot dense differential and error-estimate invariants, circuit-level DEMs,
proptests, serial/parallel determinism, warmed allocation tests and doc examples.
The explicit PyMatching suite passed all 750,000 shots at its default counts.
A separately built/extracted wheel passed all 24 public QEC Python tests with no
skips. The full run preceded the added test-only heap regression, which was checked
separately. After selecting queued-only, the final source passed 14 MWPM tests
(including dense and error-estimate controls), 25 Sparse Blossom tests, the explicit
work diagnostic, both allocation tests, a fresh 750,000-shot PyMatching sweep,
exact-corpus equality, and three targeted public Python checks. Stage-specific
source fingerprints and full logs are retained.

Local stable Rust 1.98.1 workspace all-target Clippy and beta QEC all-target
Clippy pass without exceptions, as does rustfmt. The server's Rust 1.96.0
all-target Clippy reports `nonminimal_bool` at the unchanged
`deadline_replay.rs:573` on both main and candidate. Its failure logs are retained;
only the supplementary 1.96 scoped diagnostic allows that lint. No source
allowance or CI configuration change is introduced.

## Held-out results

Native times are µs/shot; speedup is baseline time divided by candidate time.

| d | Original | Main | Queued-only | vs main | vs original |
|---:|---:|---:|---:|---:|---:|
| 7 | 25.004 | 21.141 | 17.355 | 1.218× | 1.441× |
| 9 | 65.218 | 55.294 | 45.640 | 1.212× | 1.429× |
| 11 | 132.449 | 108.496 | 90.214 | 1.203× | 1.468× |
| 13 | 224.793 | 189.900 | 157.305 | 1.207× | 1.429× |
| 15 | 392.179 | 343.669 | 281.893 | 1.219× | 1.391× |
| 17 | 580.808 | 539.293 | 434.592 | 1.241× | 1.336× |

At d=11, the four main/candidate block ratios are 1.219×, 1.212×, 1.185×, 1.228×. The cumulative original/candidate ratios are 1.463×, 1.452×, 1.479×, 1.488×.

The patch passes the predeclared adoption gate: ≥1.05× aggregate at d=11, all four blocks improve, and every measured native/Python case avoids a >5% regression. **The original #511 1.5× target remains unmet**: 1.468× aggregate and 1.452–1.488× by block. This PR references #511 and leaves it open.

| Python case | Main shots/s | Queued-only shots/s | Gain | PyMatching shots/s during candidate runs | Aleph/PyMatching |
|---|---:|---:|---:|---:|---:|
| d=5 | 322,058 | 339,727 | 1.055× | 579,571 | 0.586× |
| d=9 | 42,083 | 44,191 | 1.050× | 71,911 | 0.615× |

Each Python cell retains all 28 rate samples per version. Prediction SHA256 values agree across all eight processes. PyMatching remains faster on these public API fixtures; these data do not establish parity.

The native audit covers all 144 cells with at least 5 active host intervals per cell. Maximum activity during warmup/collection was 8.58% overall and 5.00% on the SMT sibling. Whole-process maxima, including untimed construction/analysis, were 8.58% and 5.00%. Python whole-process maxima were 8.38% and 0.00%. All traces are retained. Whole-process maxima also satisfy the earlier 10%/5% rule, so the phase refinement
is not needed for this final series to pass. These paired shared-host measurements describe the fixed fixtures; no pooled confidence interval or general speed guarantee is claimed.

## Remaining cost

The final steady-state profile has zero lost samples: `next_at` accounts for 51.04% of sampled cycles, `run_to_matching` 26.64%, and `schedule_node` 6.27%. The main control is 59.70%, 21.73%, and 5.95%, respectively. These are separate attribution profiles, not elapsed-time ratios. The guard removes redundant scanning, while neighbour/event processing remains the dominant work. Allocation reuse alone did not close this gap; further work on #511 needs a separate measured change to the remaining hot path.

## Reproduction

Build the original/main/candidate snapshots in separate plain directories and
separate Cargo target directories. Copy the std-only
`crates/aleph-qec/examples/mwpm_event_corpus.rs` probe into the older snapshots.
The config binds `original`, `main`, `candidate` to absolute native benchmark
paths and sets `cpu`, `sibling`, and the installed `python` interpreter. The
campaign creates a new output directory and refuses to reuse an existing one.

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches \
  --bench mwpm_decode --no-run
cargo test --release -p aleph-qec --lib mwpm::tests
cargo test --release -p aleph-qec --lib sparse_blossom
cargo test --release -p aleph-qec --test mwpm_allocations
PYMATCHING_PYTHON=/path/to/python cargo test --release -p aleph-qec \
  --test mwpm_pymatching_oracle -- --ignored --nocapture
python3 scripts/python/bench_sparse_events.py CONFIG.json NEW_NATIVE_DIRECTORY primary
python3 scripts/python/summarize_sparse_events.py NEW_NATIVE_DIRECTORY
python3 scripts/python/bench_sparse_events_python.py CONFIG.json \
  ISOLATED_WHEELS_DIRECTORY NEW_PYTHON_DIRECTORY
python3 scripts/python/summarize_sparse_events_python.py NEW_PYTHON_DIRECTORY
```

The isolated wheel directory contains `main-python/aleph` and
`candidate-python/aleph`, extracted from separately built wheels. Keep the native
module paths, prediction hashes, tool versions, source hashes, every Criterion
sample/interval and host traces when comparing a new run. A host-invalid attempt
is retained as such; it is not a passing benchmark with selected samples removed.

## Evidence

- [Predeclared protocol](data/q1-511-events/protocol.json), [original protocol](data/q1-511-events/protocol-original.json), [training amendment](data/q1-511-events/protocol-training-amendment.json), and [factual attribution erratum](data/q1-511-events/protocol-errata.json).
- [Native summary and per-case coverage](data/q1-511-events/primary-summary.json), [all Criterion samples/intervals](data/q1-511-events/primary-criterion-raw.json), [host traces](data/q1-511-events/primary-manifest.json), and [admission traces](data/q1-511-events/primary-host-conditions.jsonl).
- [Python summary](data/q1-511-events/python-summary.json), [all raw rates/prediction hashes](data/q1-511-events/python-raw.json), and [host/admission traces](data/q1-511-events/python-manifest.json).
- [Validation output](data/q1-511-events/validation-output.txt), [local validation](data/q1-511-events/local-validation.json), [final source hashes](data/q1-511-events/queued-final-source-sha256.json), [scanner equality proof](data/q1-511-events/scanner-equivalence.json), and [hardware/toolchain metadata](data/q1-511-events/metadata.json).
- [Cross-version corpus hashes](data/q1-511-events/corpus-sha256.json), [final queued-only corpus hashes](data/q1-511-events/queued-final-corpus-sha256.json), and [adoption decision](data/q1-511-events/adoption.json).
- [Main profile](data/q1-511-events/base-profile-report.txt), [final profile](data/q1-511-events/candidate-profile-report.txt), and [profiling invocation/binary hash](data/q1-511-events/candidate-profile-manifest.json).
- [Initial training summary](data/q1-511-events/ablation-summary.json), [all initial samples](data/q1-511-events/ablation-criterion-raw.json), [host-invalid guard attempt](data/q1-511-events/guard-ablation-manifest.json), [its raw samples](data/q1-511-events/guard-ablation-criterion-raw.json), and [rejected neighbour experiment](data/q1-511-events/rejected-neighbor-experiment.patch).
