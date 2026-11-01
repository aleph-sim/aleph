# P8-11 — Metal MPS factorization experiment

Issue: [#561](https://github.com/aleph-sim/aleph/issues/561).

## Decision and scope

The round-robin multi-threadgroup Jacobi prototype is a **NO-GO for production
integration on the measured M5**. Accelerating the old GPU factorization is not
sufficient: current guarded split costs remain above a measured f64 CPU splitter,
and clustered spectra expose truncation-sensitive observable disagreements.
This rejects this specific prototype/regime, not all blocked SVD/QR algorithms.
No production routes, numerical guards or f64 fallback were changed.

## Frozen experiment

The [protocol](data/p8-11-metal-factorization/protocol.md) and implementation were
committed as `d928f93` before the full hardware campaign. An initial small pilot
was used to validate the shader/scratch wiring; it is not mixed into reported
campaign timings. Acquisition requires a real Metal device and fails on absence.

Apple M5 MacBook Air, 10 CPU and 10 GPU cores, 32 GiB unified memory. Four Rayon
workers were available to the existing faer-based CPU reference. Compilation
completed before the measured campaign. This was a shared desktop with an active
display, not a dedicated/thermally isolated GPU. No services or user programs
were stopped; [continuous host telemetry](data/p8-11-metal-factorization/host-load.log)
and the [manifest](data/p8-11-metal-factorization/manifest.json) preserve limits
and executable identity. CPU idle is not a measurement of GPU utilization.

The shape sweep covers chi 32/64/128/256/512, square (2chi,2chi), tall (2chi,chi)
and wide (chi,2chi) blocks. All four deterministic complex fixtures have unit
Frobenius norm: separated singular values, tightly clustered values, a seven-order
condition range, and rank deficiency. Unitary butterfly rotations densify the
matrices before FP32 rounding. A bond cap of half the thin rank exercises actual
truncation; both methods use the existing normalization policy.

One untimed pack/factor warmup per mode precedes two paired repetitions in ABBA
order. Each receipt runs the actual guarded finalize, a complete CPU factor and
CPU split oracle, reconstruction and canonical checks, plus the current GPU QR.
The guard-free finalize is an attribution-only counterfactual: its results are
never used for acceptance or production. It separates sort/installation from
the O(m*n*k) reconstruction guard without removing that guard from the real path.

## Measured results

The complete campaign contains **240 receipts** over 60 shape/conditioning
fixtures, with two observations per method. No eligible shape passed the frozen
integration gate. Square well-conditioned medians (milliseconds):

| chi / matrix | Old GPU factor | Prototype factor | Actual guard + finalize | Sort/install only | CPU full split |
|---|---:|---:|---:|---:|---:|
| 32 / 64×64 | 13.075 | 3.433 | 0.300 | 0.196 | 0.675 |
| 64 / 128×128 | 68.412 | 4.398 | 0.856 | 0.206 | 2.553 |
| 128 / 256×256 | 405.415 | 19.024 | 7.954 | 0.450 | 13.777 |
| 256 / 512×512 | 2307.915 | 133.713 | 67.030 | 0.470 | 78.381 |
| 512 / 1024×1024 | 16144.694 | 1784.881 | 780.530 | 1.585 | 416.014 |

At 1024×1024, even a hypothetical zero-cost factorization would still pay a
~781 ms current guard/finalize, versus ~416 ms for the actual CPU full split.
The measured candidate factor itself costs another ~1785 ms and 12,277 dispatches
(12 sweeps) on that fixture. Sort/install without the guard is only ~1.6 ms.
Fewer serialized column pairs did not remove the other serial O(chi³) stage.

Criterion pack/factor median estimates were 13.311 → 2.885 ms at chi=32, and
68.748 → 4.424 ms at chi=64 (about 4.61× and 15.54× relative to old GPU).
This is useful prototype evidence but **not** a CPU crossover or an end-to-end
MPS improvement. [Raw Criterion estimates/samples](data/p8-11-metal-factorization/criterion.json)
retain confidence intervals and outliers; [the strict summary](data/p8-11-metal-factorization/summary.json)
includes every rectangular and adverse cell.

The actual production guard rejected **16/240** probes: 8/120 reference and
8/120 candidate, all large rank-deficient fixtures. It correctly kept factors
with reconstruction errors as high as 1.112 (reference) / 0.679 (candidate)
from being accepted. The standalone measured CPU fallback splitter is included
in those instrumented budgets. Do not remove this guard to make timing attractive.

The stronger frozen numerical gate rejected **26/120 reference** and
**32/120 candidate** observations. Reconstruction alone is insufficient: large
ill-conditioned cases can exceed retained-isometry/norm limits while passing
that guard, and clustered-cutoff observable differences can exceed 1e-5.
Maximum observed retained Gram error was 2.84e-4; normalized weight difference
1.68e-5; Z-observable difference 7.84e-4. Discarded-weight errors stayed below
1e-5. These failures are retained, not averaged into passing means. Current
GPU QR reconstructed/isometrized all frozen probes within 1.90e-6 / 2.52e-6;
this correctness result does not establish competitive QR latency.

Host CPU idle ranged 53.93–95.70%, median 87.80%. Compiler work did not overlap
the stage campaign. Shared display/thermal activity remains a limitation, as do
two repetitions per mode and synthetic spectra rather than measured circuit bond
distributions. No production route is chosen from these timings.

## Validation

The complete release Metal suite passed **83 tests**, including CPU/FP64 SV MPS
oracles, canonical center moves, capped-bond truncation/norm examples, non-nearest
neighbor routing, large-n expectation/readout and run/batched agreement. Its four
ignores consist of two explicit hardware research tests plus two existing long
norm/timing cases. The research filter was separately run with `--include-ignored`:
**3 passed**, including real-device hard fixtures and an independent large
rank-deficient guard/fallback regression. Thus neither research hardware test
was counted as an unexecuted success. The long chi=512 n=16/20 circuit case was
not run; the full matrix sweep covers factorization through chi=512 and supports
only the scoped no-go, not an end-to-end speed claim.

Research-feature all-target Clippy on stable and beta, workspace all-target
Clippy, formatting and the 15 CI policy contracts passed. The Metal CI group now also compiles/lints the explicit research feature
and runs its pure schedule test; actual-device tests stay explicitly ignored in
headless CI. Local hardware receipts are separate evidence. Runtime code remains
excluded from normal `--features metal` backend routing.

## How the prototype differs

The production Jacobi SVD serializes every column pair inside one threadgroup.
The prototype uses one threadgroup per independent pair. A circle-method schedule
covers every pair exactly once per sweep (a dummy-column bye handles odd sizes).
Independent pairs own disjoint A/V columns; distinct compute encoders provide
tracked device-buffer dependencies between rounds. A per-sweep atomic rotation
count and completion wait implement bounded convergence (same 1e-6 pair threshold,
60-sweep cap). Column normalization is parallel over threadgroups.

The complex rotation formula is the existing one-sided Jacobi formula, with the
same row reduction. This is a multi-threadgroup scheduling experiment, not a new
precision, a larger single threadgroup, a block-Householder algorithm, or a claim
that global barriers can be replaced by threadgroup barriers. Schedule coverage
and small actual-device reconstruction tests include odd and rectangular shapes.

## Timing interpretation

All stage clocks are host wall clocks including encoder construction, submission
and completion wait; shader compilation and fixture generation are excluded.
CPU split includes widening, current faer/fallback factorization, truncation and
FP32 output assembly. GPU pack includes reset/identity preparation of pooled
scratch. Reference sweeps are internal and not instrumented (`sweeps=0` means
unknown, not no work); candidate sweep/dispatch counts are explicit.

These probes intentionally use separate commands for attribution. Their sum is
an **instrumented stage budget**, not the fused production command-buffer time
and not an end-to-end MPS speedup. A beneficial stage budget was a necessary
precondition for integration; it was not permission to change routing directly.
The CPU QR column is an **implicit faer QR diagnostic** and excludes explicit Q
formation; no matched full-QR speed ratio is claimed from it. GPU QR includes
explicit thin Q/R, whose reconstruction and orthogonality are checked separately.

A separate Criterion control measures actual pack/factor completion with pooled
buffers, 10 samples, 1 s warmup and 2 s requested measurement at chi 32/64. Both
methods include reset/pack and host dispatch/completion. Criterion is a noisy
shared-desktop measurement, not a performance CI gate.

## Accuracy and fallback interpretation

Each receipt compares full reconstruction, retained-column Gram error, discarded
weight, truncated state norm and a flattened bipartite Z observable with the
existing f64 CPU split. Finite data are required. Frozen thresholds are 1e-3 for
relative reconstruction (current guard), 1e-4 for retained isometry, and 1e-5
for the remaining three quantities. The raw [matrix receipts](data/p8-11-metal-factorization/matrix.jsonl)
retain failures instead of selecting favorable seeds/shapes.

A reconstruction guard can pass even when truncation near a clustered singular
cutoff chooses an observably different subspace. Such a failure is evidence that
the declared observable agreement was not established, not proof that every FP32
factor is algebraically invalid. Both reference and candidate can show this
sensitivity. Tolerances were not widened after seeing it.

`guard_accept` is the actual on-device finalize decision. `fallback_ms` records
the measured standalone CPU splitter cost when that decision declines; it is not
a whole backend state-rebuild timing. Frequencies apply only to these frozen
matrix probes. They do not estimate CPU fallback or bond distributions in real
MPS circuits; existing small-bond CPU center moves remain in production.

## Reproduction

Build and finish compilation before timing; require a free enough Mac and record
contention throughout. Research code is behind an explicit feature and is never
called by normal Metal execution.

```bash
cargo build --release -p aleph-metal --features factorization-research --example factorization_research
RAYON_NUM_THREADS=4 target/release/examples/factorization_research > matrix.jsonl
python3 scripts/python/summarize_metal_factorization.py matrix.jsonl summary.json
RAYON_NUM_THREADS=4 cargo bench -p aleph-metal --features factorization-research --bench factorization_research
cargo test --release -p aleph-metal --features factorization-research --lib research::tests -- --include-ignored
```

Bounded smoke knobs: `ALEPH_RESEARCH_CHIS`, `ALEPH_RESEARCH_SHAPES`,
`ALEPH_RESEARCH_FIXTURES`, `ALEPH_RESEARCH_REPEATS`. Defaults reproduce the full
frozen campaign. The summary rejects incomplete sweeps and non-finite metrics.

## Consequence

Keep the existing backend and its f64 fallback. A future experiment would need
both a lower-dispatch/blocked factorization and a parallel reconstruction guard;
QR center moves and clustered-cutoff stability remain independent constraints.
A future GO would still require real MPS bond distributions, fallback accounting
and end-to-end norm/observable/time comparisons before production adoption.
