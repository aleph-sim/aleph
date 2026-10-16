# P7-01 — Reproducible decoder failure witnesses

Issue: [#543](https://github.com/aleph-sim/aleph/issues/543).

`WitnessDecoder` builds one supported decoder against an immutable copy of the original DEM.
`capture_shot` is an opt-in companion to `sample_shots`: it draws the same StdRng stream,
seeded with the same SplitMix64(seed, absolute shot), with one coin per original mechanism.
The ordinary sampler has no trace flag, branch or trace allocation; its only source change
is exposing the existing seed mixer inside the crate. Capturing one shot includes decoding,
canonical DEM serialization and SHA-256 hashing; it is a debugging operation, not the default
Monte Carlo path.

## Capture, reduce, replay

```sh
cargo run --release -p aleph-qec --example decoder_witness -- capture /tmp/witnesses
cargo run --release -p aleph-qec --example decoder_witness -- replay /tmp/witnesses/surface-d3-r3.json
cargo test -p aleph-qec --test witness
cargo bench -p aleph-qec --bench failure_witness
```

The example scans seed 543 in ascending absolute shot order. To demonstrate nontrivial
reduction rather than selecting the first already-minimal failure, it accepts the first
surface failure with >=4 fired mechanisms and the first BB failure with >=9. It stops after
100,000 shots if none qualifies. These are selected debugging examples, not unbiased LER
samples. The ordinary DEM sampler still uses all original correlated mechanisms, including
`^` components; only matching decoder construction splits components.

Replay configurations are MWPM (default locality policy), unweighted Union-Find, floating
relay BP, and fixed relay BP. Relay alpha is 0.875 and gamma is (-0.3, 0.9); legs, sweeps,
disorder seed, early-exit policy and fixed-point widths are explicit in the JSON. No decoder
priors are inferred from reduced faults. Hypergraphs are supported by BP; matching replay
rejects incompatible non-graphlike DEMs. Additional decoders require an explicit versioned
configuration rather than an opaque name or closure.

Version-1 artifacts preserve canonical `.dem` text and its SHA-256, source seed and shot,
original and current sorted mechanism indices, decoder settings, both outcomes and reduction
receipt. Outcomes include syndrome, true and predicted logical bits, logical mismatch,
available syndrome validity and fixed-relay first-valid iteration. Non-convergence alone is
never counted as logical failure. Seed/shot identify the original draw; the reduced set is a
counterfactual, not a new random draw at that index. RNG/decoder bit stability across future
library releases is not promised: the saved indices retain the historical fault set, while import checks both the original
seed/shot draw and source-decoder outcomes and rejects mismatches. Reproducing a future
version mismatch can require the original checkout.

Provenance is explicitly `dem-mechanisms`, never physical circuit locations: a DEM mechanism
can merge several physical origins. No circuit location is fabricated. The exact-example
signature is SHA-256 over the DEM hash, serialized decoder configuration and current fault
indices; sample seed and runtime are excluded. It groups the same reduced mechanism example,
not all sets that happen to yield the same syndrome or all physical faults with one origin.

## Reduction semantics and budgets

The reducer tries deterministic complement chunk deletions, increasing granularity when no
chunk can be removed, then performs restart-on-success single deletions until one complete
sweep removes nothing. See [Zeller and Hildebrandt, TSE 2002](https://www.st.cs.uni-saarland.de/publications/files/zeller-tse-2002.pdf).
Each accepted deletion must retain logical mismatch under the *same* source decoder. No
monotonicity assumption is made. A completed result is **1-minimal**, not globally
minimum-weight, and need not be unique. An empty failing fault set is vacuously 1-minimal.

`MinimizeBudget` bounds predicate evaluations and wall time. Checks occur before each call;
an in-flight decoder call cannot be preempted and can exceed the wall deadline. Counts include
initial source replay, rejected trials and the final explicit certification sweep. One fixed
relay evaluation performs two internal passes (prediction/validity and first-valid diagnostic);
`decoder_calls` counts predicate evaluations, not individual internal passes or BP sweeps.
Decoder construction is outside the budget; structural validation is inside elapsed time.
Zero budget returns `Incomplete` without claiming a new source replay. Import of untrusted
artifacts should go through `FailureWitness::from_json` before reduction.

The receipt survives JSON round-trip, including `Incomplete`, call/time budgets, consumed
calls and elapsed microseconds. Import reconstructs both fault sets, verifies dimensions and
subset/order, reruns the source decoder on original/current sets, and independently checks
all single deletions when a receipt claims `OneMinimal`. Import certification is intentionally
outside the minimization budget and may be expensive on large artifacts. SHA-256 identifies
the model, not the author of an artifact; artifacts are not signed.

## Golden reference vector contract

The checked BB fixture is **gross [[144,12,12]] code-capacity Z noise, p=0.03**, with 72 X checks,
144 original mechanisms, 12 logical bits and a single complete block (no sliding/window
mapping). Fixed relay uses 6 legs x 10 sweeps, Q5.3 (8 signed bits, 3 fractional bits), disorder
seed 0x5E1A4B9C, full-schedule lightest-valid policy, alpha=7/8 and gamma=(-0.3,0.9).
This is the code-capacity graph/configuration built by the existing
`qec_q7_bp_graph` example's default `build(None)` path.

`fixed_reference` exports one `.syn` shot as three little-endian u32 words (detector d ->
word d/32, bit d%32; upper padding zero), and one v2 `.ref` via existing `write_ref`.
`read_ref` checks the header and recovers true/predicted u16 masks, validity and first-valid
iteration. The example unpacks `.syn`, invokes the existing independent `FixedRelayBp`
entry points, and checks their output against `.ref`; regression tests pin the exact bytes.
Export rejects >16 observables, non-source fixed configurations and schedules exceeding
the 15-bit iteration field. No remapping or iteration truncation is silently applied.

The JSON sidecar is mandatory: binary vectors do not encode DEM priors, dimensions or layout.
A hardware consumer must build the matching graph/ROM configuration. This checks software
golden/vector compatibility; no new board run or arbitrary-graph RTL support is claimed.

## Regression examples and validation

Fixtures live in `crates/aleph-qec/tests/data/witness/`:

| Model | Shot (seed 543) | Source decoder | Faults before -> after | Predicate calls |
|---|---:|---|---:|---:|
| Surface d=3, r=3, phenomenological p_data=p_meas=0.03 | 39 | MWPM | 5 -> 2 | 6 |
| Gross BB code-capacity p=0.03 | 515 | Fixed relay 6x10 Q5.3 | 9 -> 7 | 28 |

Both completed the independent 1-minimality check. The surface witness also fails under
Union-Find; the BB witness also fails under floating relay BP with the same schedule/disorder.
These shared failures are examples of decoder limitations, not assertions that either
implementation is buggy. Original and reduced outcomes remain available for analysis.

The tiny-model tests enumerate every mechanism subset in three DEMs, including duplicate
supports, correlated components, empty syndromes, logical-only mechanisms and multiple
observables. They compare XOR reconstruction, import/replay and independent single-deletion
checks. Further tests cover 1-vs-4-worker capture, ordinary-sampler equality, zero/one-call and
zero-time budgets, forged certification, incompatible models, invalid configuration/probability,
non-convergence without logical mismatch.

Performance receipts and final test totals are recorded with the PR validation results.
New `serde`/`serde_json` dependencies reuse workspace versions; `sha2` supplies the standard
SHA-256 implementation rather than introducing a custom hash implementation.

## Initial local measurement (diagnostic, not a clean-host acceptance)

Local arm64 Mac, Rust 1.98.1, `RAYON_NUM_THREADS=1`, identical existing
`dem_experiment/sample_1024` harness against main `ca5d6b3` and the patch. Two ABBA blocks:

| Source | Four per-run median times for 1024 ordinary shots (us) | Median of run medians (us) |
|---|---|---:|
| Main | 842.7, 845.7, 864.9, 1519.4 | 855.3 |
| Patch | 843.9, 864.8, 1013.9, 1514.0 | 939.4 |

The aggregate patch/main ratio is 1.098, but both binaries drift from roughly 0.84 ms to
1.51 ms within this short campaign. Load averages were 12.4–13.9 on a 10-core shared Mac.
This is **inconclusive** for capture-disabled overhead: neither a speedup nor a regression
acceptance is claimed. The sampler loop itself remains source-identical. A separate pinned-server campaign is recorded below; this local run remains inconclusive
and is retained in full.

Warmed opt-in Criterion medians on the same shared machine were capture **7.780 us**
(95% CI 7.592–7.941) and surface 5->2 minimization **18.460 us** (18.213–18.924). These are
cost observations, not portable throughput promises or isolated-hardware results.

Raw per-run medians, samples, logs, binary hashes, source fingerprints, oracle output and test
logs are under `docs/perf/data/p7-01-witnesses/`. The initial full release suite passed 240 tests
with 22 pre-existing ignores; the final focused suite passed all 8 witness tests after the last
validation changes. Workspace/all-target Clippy, rustfmt and the public Rust documentation
example passed. Stim 1.16.0 checked both original/reduced fixture reconstructions plus all 16
tiny correlated subsets; PyMatching 2.4.0 matched both surface predictions. The BB hypergraph
is explicitly excluded from PyMatching replay.

## Pinned-server follow-up

The authorized follow-up tested exactly source head `816ef96e2a5aadd6955d5967a07eaedd2101ee83`
against main `ca5d6b3989204fd056d4f6bc8242e3b1489e22f3` on the shared dual-Xeon Silver 4114
server (40 logical CPUs), Rust 1.96.0. Services and CI jobs remained running. Both snapshots
were compiled separately with the same compiler and locked dependencies; source and binary
SHA-256 values accompany the receipts. All locally recorded source fingerprints matched.

Four ABBA blocks produced 16 accepted runs of the **same existing three control cells**,
with `RAYON_NUM_THREADS=1` and affinity to CPU 3. SMT sibling 23 was also monitored. A one-second
preflight required <=2% busy time on both CPUs; each measured process required <=3% external
CPU-3 busy time (total busy minus child process CPU usage) and <=3% busy time on sibling 23.
Thirteen preflight attempts and one measured-overlap attempt were rejected and retained. Among accepted runs, external CPU-3
busy time was <=1.625%; sibling busy time was <=2.934%. These are per-process aggregate guards,
not whole-host isolation or proof against short interference bursts, cache or frequency effects.

| Ordinary capture-disabled cell, 1024 shots | Main median (us) | Patch median (us) | Paired block ratio | Block-bootstrap 95% range |
|---|---:|---:|---:|---:|
| Sampling only | 1054.8 | 1043.5 | 0.983 | 0.958–1.008 |
| Null-decoder complete experiment | 1167.7 | 1112.1 | 0.983 | 0.956–1.003 |
| MWPM d3 complete experiment | 820.3 | 811.6 | 0.979 | 0.899–1.036 |

Time columns are medians of per-run Criterion medians. Ratios are geometric means of the
within-ABBA log ratios, so dividing the displayed medians is a different statistic. The ranges
are percentile bootstrap ranges from resampling four complete blocks (all 4^4 combinations),
not a universal non-regression guarantee. No material capture-disabled regression was detected
in these cells. No speedup is claimed: all three ranges include parity, and the small block
count/shared-host conditions limit generalization. The noisy Mac run remains inconclusive.

Warmed opt-in surface d3/r3 costs on the same pinned CPU were **14.230 us** per captured shot
(95% Criterion interval 14.218–14.260), and **59.671 us** for the 5->2 reduction
(59.438–83.661; wider uncertainty retained). Capture includes a decoder call, an owned DEM
snapshot and hashing; reduction includes initial validation/replay and six predicate calls.
These opt-in costs do not run in the ordinary sampler.

The server passed all 8 witness tests, all 5 existing experiment tests, the independent
Stim/PyMatching oracle script and the complete example. The example again produced surface
5->2 / 6 calls and BB 9->7 / 28 calls, with identical signatures. Its one-shot, unpinned
correctness-run minimization times were 315 us and 9099 us respectively; these cold observations
are not the warmed Criterion surface estimate. Regenerated BB `.syn` and `.ref` bytes exactly
matched the checked-in fixtures across architectures. The fixed-vector check still makes no
new board/RTL execution claim.

The performance-validation item that kept this PR in draft is complete for these scoped
controls. Remote receipts, every executed run, raw samples, guarded feature costs, source/binary
hashes and the campaign/analysis scripts are checked in alongside the earlier local evidence.

The control campaign logs every preflight decision. The separate feature runner skipped one
busy preflight before its accepted run; that skipped check did not execute a benchmark and its
CPU readings were not serialized. All executed feature measurements are retained.
