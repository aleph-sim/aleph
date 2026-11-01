# Phase 7: reproducible research instruments

P7-00 integrates the six scoped capabilities into two reproducible studies. The six
child issues are closed; this report supplies the phase integration/exit evidence.
This is an integration and evidence PR, with no new simulator/decoder algorithm,
public execution change, automatic backend selection, or release commitment.

## Capability and evidence checklist

| Verified deliverable | Issue | Primary report and retained measurements |
|---|---|---|
| Failure capture, bounded 1-minimal reduction, cross-decoder replay | [#543](https://github.com/aleph-sim/aleph/issues/543) | [P7-01](../qec/p7-01-decoder-witnesses.md) |
| Fault-count conditional estimation, omitted-mass bounds, opt-in pre-observation routing | [#544](https://github.com/aleph-sim/aleph/issues/544) | [P7-02](../qec/p7-02-rare-events.md) |
| Exact causal cones, compact remapping, pre-allocation dense limits | [#545](https://github.com/aleph-sim/aleph/issues/545) | [P7-03](p7-03-causal-cone-expectations.md) |
| Generating/assumed/reference separation, independent-stream uncertainty, v3 replay | [#546](https://github.com/aleph-sim/aleph/issues/546) | [P7-04](../qec/p7-04-decoder-robustness.md) |
| Finite queues, fallback, deadlines, dependency/commit semantics | [#547](https://github.com/aleph-sim/aleph/issues/547) | [P7-05](../qec/p7-05-deadline-trace-replay.md) |
| Exact sparse Pauli propagation, counters and explicit resource refusal | [#548](https://github.com/aleph-sim/aleph/issues/548) | [P7-06](p7-06-sparse-pauli.md) |

Each primary report retains its commands, inputs, raw receipts, numerical/statistical
oracles and limitations. Integration does not replace those campaigns with a new
small smoke test. The new helpers live in example support modules, shared with
ordinary Cargo integration tests; production libraries are unchanged.

## Reproduction

From the repository root, with Rust stable and Python 3 (standard library only):

```sh
python3 scripts/python/run_research_instruments.py /tmp/p7-run --target-dir /tmp/p7-build
python3 scripts/python/check_research_instruments.py /tmp/p7-run
```

Both paths should be new and separate; output reuse is rejected. Use a fresh build
directory for extracted snapshots to avoid stale artifacts with restored mtimes.
The runner selects each executable from Cargo's exact package/target artifact,
records its SHA256 and freshness, and checks source hashes again after execution.
For a non-Git snapshot, `--snapshot-provenance FILE` accepts the original checkout's
`base_revision` and `source_sha256` map and verifies every declared source before
building. The base revision alone does not identify the uncommitted integration
patch: the source hash map is authoritative for the executed source.

The runner builds and executes all nine entry points below. Durations in its
manifest are correctness-smoke receipts on a shared host, not clean-host benchmarks.
`RAYON_NUM_THREADS=1`, build jobs default to four, and no services are stopped.

| Package/example | Smoke arguments after `--` | Scope |
|---|---|---|
| `aleph-qec / research_workflow` | `NEW_QEC_DIR` | Study 1 below |
| `aleph-stab / research_observables` | `NEW_QUERY_DIR` | Study 2 below; `--smoke` uses product6 instead of product16 |
| `aleph-qec / decoder_witness` | `capture OUTDIR`; `replay FILE.json` | Existing surface/BB capture and native exported-witness replay |
| `aleph-qec / rare_events` | `OUTDIR 8 1` | Eight coverage trials, one campaign seed; not a new coverage assertion |
| `aleph-qec / decoder_robustness` | `OUTDIR 4 8` | Four streams, eight steps; not a new robustness conclusion |
| `aleph-qec / robustness_coverage` | `OUTFILE.json 8` | Small oracle-path smoke |
| `aleph-qec / deadline_replay` | `campaign OUTDIR 32` | Model replay; checked by the independent Python availability oracle |
| `aleph-sv / expectation_query` | none | Existing compact-SV example |
| `aleph-stab / sparse_expectation` | none | Existing sparse example |

Each may be run as `cargo run --release -p PACKAGE --example NAME -- ARGS`.
The wrapper's manifest retains the complete executed argv, including paths and
seeds/budgets encoded in the two new helpers. Tests use the same helpers:

```sh
cargo test --release -p aleph-qec -p aleph-stab
cargo clippy --workspace --all-targets -- -D warnings
cargo +beta clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Shared reproducibility and provenance contract

1. **Identity before interpretation.** QEC artifacts retain the canonical DEM text
   and its SHA256, source decoder family/configuration, mechanism indices, seed,
   shot/rank and original/reduced faults. Priors are part of model identity. Native
   replay rejects changed models; comparing another decoder uses the same DEM.
2. **Keep sampling laws explicit.** V1 witnesses describe ordinary independent DEM
   draws; v2 preserves fault-count conditioning/enumeration; v3 preserves the actual
   robustness process and generating DEM alongside assumed decoder priors. No
   conversion silently erases conditioning or temporal provenance. Postselected or
   reduced witnesses are diagnostic examples, never fresh Monte Carlo samples.
3. **Record the endpoint and accounting.** Retain declared and actual decoder calls,
   confidence, row mode/counts, omitted probability mass, stop reason and all
   intervals. Fixed independent shots use exact binomial intervals. Correlated
   trajectories use independent streams as statistical units, not iid steps.
4. **Hash the same query inputs.** Save UTF-8 QASM and Pauli text before parsing;
   each engine reads those files with zero initial state and FP64 arithmetic.
   Store both file hashes and query identity
   `SHA256(b"p7-query-v1\0" || u64_le(qasm_byte_length) || qasm || pauli)`.
   Record source indices, compact-to-original maps, precision, numerical tolerance,
   memory/term budgets and explicit unsupported/refused results.
5. **Separate kinds of evidence.** Manifest/source/binary hashes identify software.
   Algorithm workspace/payload bounds are not RSS. Host decode timings and declared
   interface costs are not FPGA timing. A queue replay's synthetic traffic/deadline
   assumptions are not a physical closed-loop correction experiment. Hardware
   measurements retain their own machine, revision, workload and audit receipts.

The stdlib checker independently verifies hashes/signatures, Clopper–Pearson endpoints
by numerical binomial-CDF inversion, analytic controls, call accounting and explicit
resource statuses. Native Rust import, decoding, minimization and replay validate
witness semantics. The Python checker does not independently implement MWPM/UF or
prove correlated-process confidence coverage. `--compare OTHER_RUN_ROOT` compares
both validated runs' stable identities, counts, witnesses, stream outcomes and
numerical query answers (tolerance 1e-10), excluding elapsed time.

## Study 1: sampling → failure witness → cross-decoder replay

`research_workflow` fixes seed 542 before observing results. It builds a distance-3,
three-round surface memory-Z phenomenological DEM with data/measurement rates .03.
All **512** ordinary shots are scored; the earliest failure is saved for diagnosis
without changing the estimator's endpoint. The run gives **56/512** failures with
95% exact interval **[0.08368493, 0.13967527]**.

Canonical DEM SHA256:
`93e6d45a66b0af6c32fe15f3f630d772cfa4ff6d7bc57f3691e9822b253c62fe`.
The first failing shot is **4**. Its three faults reduce to **two**, certified
1-minimal in **seven** reduction calls under a 512-call/30-second cap. An additional
single-deletion sweep verifies the certificate's use; JSON import must preserve the
witness exactly. The reduced signature is
`e11c5f4f3fec78db2b922903a03adcc4502da865251cbf78355f9e8b471c81f5`.
**Both MWPM and unweighted Union-Find fail on this witness.** Cross-decoder replay
explains a common failure; it does not establish a relative logical-error rate.
No mapping from DEM mechanisms back to a unique physical circuit fault is claimed.

The same surface identity feeds fault-count strata K≤3, with 64 calls per row and
256 declared calls total. Exact small rows reduce actual scored calls to **169**.
The omitted mass is about **0.05298050**, retained in the conservative failure
interval **[0.03240277, 0.16733391]**. This tiny integration budget is not a rare-rate
precision demonstration. Conditional witnesses retain v2 provenance through export,
minimization, import and native cross-decoder replay.

An independent analytic control uses three mechanisms:

```text
error(0.1) D0 L0
error(0.2) D0
error(0.03) L0
```

Matching prefers the .2 no-logical boundary edge. Failure is parity of the independent
.1 and .03 logical mechanisms: `.1 + .03 - 2*.1*.03 = .124`.
Enumeration scores all **eight** subsets and its weighted interval encloses .124
with numerical width below 1e-12. Enumeration witnesses also preserve their distinct
v2 provenance, rather than pretending to be ordinary shots.

A separate experiment swaps the first two assumed priors, keeps the original
reference/generating model, and declares a Markov burst on mechanism 2 (onset .1,
recovery .25, multiplier 3). Four independent streams ×32 steps consume **256** paired
decoder calls. The observed paired difference is **0.0703125**, but its interval is
**[-1, 1]**: deliberately uninformative at this smoke budget. Its v3 witnesses preserve
both identities/process, survive reduction and replay under the assumed model.
Independent fault-count weights are never reused for this correlated process.

## Study 2: the same observable through alternative exact representations

`research_observables` persists each circuit and Pauli sum, then parses those files
for full FP64 SV, P7-03 compact SV, P7-06 sparse propagation with/without cones,
and an MPS overlap check. Values must agree within **1e-10**. Full SV is restricted
to ≤20 qubits **before allocation**; compact SV has a **64 MiB** workspace budget.
Sparse propagation uses the current explicit defaults of 65,536 candidate terms
and 64 MiB workspace, apart from the eight-term refusal control. These bounds
exclude owned input/plans and process overhead. MPS has bond cap 128, and discarded
weight must be finite and ≤1e-12.

| Saved query | Reference value | Compact SV | Sparse diagnostic / scope |
|---|---:|---|---|
| `ghz8`: global X + .25 I + .125 Z0Z1, phase .37 | 1.3073273456060344, analytic | Full and compact both agree | Four peak terms without cones, three with grouping |
| `local128`: three Ry/CNOT layers, Z64 + .25 I | 0.8145652527710875, compact FP64 SV | Six qubits, indices 62–67; 6,656 workspace bytes | All 128 qubits without cones or six with cones; sparse/MPS independently agree numerically |
| `ghz128`: same global weighted observable | 1.3073273456060344, analytic | Structured resource refusal at 128 active qubits; full SV not attempted | Truly global support remains 128; four/three peak terms; MPS bond two |
| `product16`: Ry(.37) on all qubits, global Z | 0.32590780369910105, analytic `cos(.37)^16` | Full and compact both agree | 65,536 terms, 4,718,592 workspace bytes; MPS bond one |

These are weighted sums, so the GHZ answer exceeding one is valid. The 128-qubit
local case uses compact SV as its reference; full SV is intentionally unavailable,
so that row does not claim an independent full-state oracle. Small GHZ/product rows
provide full-SV and analytic overlap. A separate eight-term cap must fail with
`required=16, limit=8`, returning **no partial expectation**.

CI runs product6 to bound runtime while retaining branching/refusal coverage; the
full study runs product16. This integration changes observables relative to the
single-term leaf benchmarks; do not attach those historical timing ratios to these
new multi-term fixtures. Sparse propagation produces expectations, not amplitudes,
samples or a general noise/measurement simulator. These query APIs accept unitary
circuits from zero initial state and Hermitian Pauli sums; measurement/reset and
unsupported gate forms are rejected, including outside the live support. The leaf
reports specify the complete supported gate/validation contracts. No approximate term pruning or
automatic routing is introduced.

## Measured benefit and negative regimes across the phase

The following are existing scoped measurements, not timings from the new integration
smoke. Follow each primary report for commands, exact receipts, host, revision and
confidence/measurement protocol; different rows are not a common-host leaderboard.

| Instrument | Measured benefit | Measured limit / negative control |
|---|---|---|
| P7-01 | Replayable surface/BB witnesses reduce failures to inspectable mechanism sets | Four ABBA blocks over three capture-disabled controls found no material regression; ranges include parity, so no general speedup claim |
| P7-02 | At matched actual calls, median interval width was about 2,733× smaller for rare surface and 17.1× for rare BB | At higher rates widths were 1.51× and 9.55× worse. Rare surface wall time 4.678 ms vs direct .420 ms: width gain is not throughput gain. Only 1/64 campaign runs met requested 1e-6 width |
| P7-03 | Local18→6 compact query: 24.162 µs compile+query vs 48.461 ms full SV; workspace bound 6,656 vs 10,489,856 bytes | Global16: 784.36 µs query vs 779.44 µs full SV; no cone reduction. Reused-plan execution is a separate timing tier |
| P7-04 | Static mismatch example measured +.8789 percentage points, marginal 95% interval [+.202, +1.555] pp | Exploratory, not multiplicity-adjusted. Drift/burst deltas had broad stream intervals including zero; steps are not iid samples |
| P7-05 | Replay exposes finite-queue and fallback policy costs hidden by mean latency | Synthetic mean 138.281 ns below period 160 ns still caused 45/512 drops and 106 late jobs. Correlated fallback cut decoded errors 8→3 while union SLA failures rose 8→13; no silicon throughput claim |
| P7-06 | Global single-X GHZ128: 12.59 µs, two terms, 192-byte payload, about 3.63 MiB RSS | Cones cost 87.51 µs on the same leaf case. Product16: 11.53 ms sparse vs 1.94 ms SV and 7.75 µs MPS; 65,536 terms. Local128 cone planning also lost (204.4 vs 62.7 µs without cones) |

P7-03's CPU measurements used an i5-13500; P7-06's used a Xeon Silver 4114.
P7-06's initial stale-Cargo-artifact attempt was explicitly excluded, with its audit
retained and a fresh-build rerun replacing it. The deadline report likewise retains
an excluded stale-schema attempt. Neither is silently counted as accepted evidence.
The rare-event `auto` mode is opt-in, deterministic pre-observation routing based on
a declared heuristic, not a guarantee that its interval or runtime will improve.

## Integration validation receipts

The integration archive and validation receipt under
[`data/p7-integration/`](data/p7-integration/) record actual commands, source/binary
hashes, deterministic output artifacts, Linux/Mac comparison and test results.
Validation passed 357 affected-crate release tests (25 existing ignores), 11 doc
tests, both new integration tests on Linux, stable/beta workspace all-target
Clippy and rustfmt. All nine entry points / 16 commands completed on Linux;
independent cross-platform and tampering checks passed.
The Linux run uses the user-authorized shared CPU server; no GPU or physical-board
validation is required by this integration issue. Leaf Stim/PyMatching, analytic
coverage, trace-oracle and full-SV evidence remain linked above; this smoke does not
claim to repeat their large statistical/performance campaigns.
