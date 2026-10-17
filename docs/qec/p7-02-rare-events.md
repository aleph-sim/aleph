# P7-02 — Fixed-budget fault-count stratification

Issue: [#544](https://github.com/aleph-sim/aleph/issues/544).

This adds a separate estimator for **independent original DEM mechanisms**. It estimates
`P(fail) = sum_k P(K=k) P(fail | K=k)` without changing decoder priors. K is a DEM-mechanism
count, not a physical circuit-fault weight. Correlated `^` components fire together; they
are split only inside matching-decoder construction, never in sampling.

```sh
cargo run --release -p aleph-qec --example rare_events -- /tmp/rare-events 1000 16
python scripts/python/check_rare_event_bounds.py /tmp/rare-events/summary.json
cargo test -p aleph-qec --test stratified --test witness --test binomial_interval
cargo bench -p aleph-qec --bench rare_events
```

The Python check requires NumPy, Stim, PyMatching and SciPy. It independently checks stratum
weights/tails with 100-digit Decimal arithmetic over the exact binary-f64 input probabilities,
conditional intervals against SciPy beta quantiles, and tiny-model truth against PyMatching.

## Probability masses and conditional draws

`FaultCountPlan` removes p=0 mechanisms and records forced p=1 mechanisms. It builds a truncated
Poisson-binomial suffix DP over the remaining independent Bernoulli variables. For suffix i,

```
S[i,r] = (1-p[i]) S[i+1,r] + p[i] S[i+1,r-1]
P(take i | r remaining) = p[i] S[i+1,r-1] / S[i,r].
```

The table uses log-sum-exp and `ln_1p(-p)`; the conditional branch uses a stable logistic of
the take/skip log masses. It does not draw uniform k-subsets for heterogeneous priors, nor
perform rejection sampling until an ordinary shot happens to have the requested K. For equal
probabilities the same recurrence reduces to binomial weights and uniform conditional subsets.
The background conditional Bernoulli distribution is described by
[Chen and Liu (1997), Statistica Sinica 7, 875–892](https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n44/j7n44.htm).
The recurrence and update above follow directly by conditioning independent Bernoulli variables.

A separate positive, outward-rounded interval DP encloses the unconditional masses. The overflow
mass accumulates `old_tail + old_mass[max_k] * p`; it is **not** computed as `1 - almost_one`.
This retains tiny tails without cancellation. `log_weight=Some(finite_negative_value)` distinguishes
an underflowed positive mass from an impossible stratum (`None`). Underflowed positive products
retain a positive enclosing upper bound, even if the nominal linear probability is zero.

Construction enforces a caller-supplied numeric-workspace limit and a 4,000,000-f64-cell ceiling
(32 MB): suffix table, rolling mass row and stored mass bounds are included; the owned DEM and
index metadata are excluded. Large plans return a resource error rather than allocate an
unbounded quadratic table. `max_faults` must include all forced mechanisms and cannot exceed
the original mechanism count. Counts between the maximum possible K and the requested cutoff
are explicitly impossible, not sampled.

“Exact conditional sampling” refers to this target distribution and recurrence, rather than a
uniform-subset or importance-proposal approximation. Floating-point transcendental evaluation,
finite random-bit resolution and ordinary PRNG assumptions remain; no bit-level exact-rational
sampler or certified special-function implementation is claimed. Tests cover underflow,
heterogeneous conditional frequencies and independent high-precision mass/quantile checks.

## Fixed allocation, complete enumeration and uncertainty

The entire per-stratum budget vector is declared before decoding. Impossible strata cannot
have positive budgets. The sum is checked against `max_decoder_calls` before any call.
`enumerate_small=true` selects complete enumeration whenever the count of possible fault sets
fits that stratum's declared budget; this decision uses only the model and budget, never the
observed failures. Otherwise the row uses independent conditional draws. Unspent exact-row
budget is not adaptively redistributed.

Complete enumeration uses lexicographic ranks and original heterogeneous fault-set probabilities.
It sums the unconditional probabilities of failing sets directly, enclosing rounding outward.
It does **not** average heterogeneous subsets uniformly or multiply their weighted contribution
by P(K=k) a second time. In particular, the unique zero-stochastic-fault configuration costs one
call and can be classified exactly. This prevents a large, zero-failure k=0 stratum from setting
an unnecessarily large Monte Carlo upper bound on an already-known outcome.

For M Monte Carlo strata, a requested confidence of `1-alpha` assigns `alpha/M` to each row.
The row intervals are equal-tailed Clopper–Pearson intervals, including non-degenerate zero/all
failure bounds. Bonferroni then guarantees simultaneous conditional coverage under the sampling
model. Positive mass bounds multiply conditional endpoints outward, and contributions sum
outward. Exact-enumeration rows require no confidence allocation. The existing direct
`BinomialInterval95::from_counts` still uses alpha=0.05; a private generalized-alpha form supplies
these adjusted row intervals. No Wald half-width is reused.

The interval construction uses bounded beta-CDF inversion, retains conservative bracket sides
and rejects unsupported/unrepresentable parameters. Its ordinary floating-point special-function
limitations are checked numerically against SciPy; the confidence claim does not include arbitrary
implementation bugs or quantify every possible transcendental/RNG rounding bias.

Every uncalled possible row and every K above the cutoff contributes unknown failure mass
`[0, mass_upper]`. `unsampled_mass` reports their probability bounds separately. The output field
`sampled_failure_mass_estimate` estimates **evaluated contributions only**, not a total rare-event
rate obtained by silently setting the unknown tail to zero. Consumers must use
`failure_probability` for a bound on the full failure probability.

`target_width`, if supplied, is assessed only at the fixed endpoint. The status is `TargetMet`
or `BudgetExhausted`; no output-dependent optional stopping or adaptive allocation is hidden.
Without a target, the status is `FixedBudgetComplete`. Results include model SHA-256, fixed
decoder configuration, sampler identity, seed, declared/actual calls, row weights/log weights,
failures, available nonconvergence counts, method, omitted mass and elapsed time.
Nonconvergence is a diagnostic and never substitutes for logical mismatch.

## Witness integration and reproducibility

The estimator uses `WitnessDecoder`, built once against the unchanged target DEM. It retains
at most `witness_limit` failures, reusing each already-scored outcome: capture spends **no extra
decoder calls** inside the matched budget. Later JSON import/replay/minimization is separate work.

Ordinary P7-01 v1 witnesses remain unchanged and importable. New v2 artifacts carry one of:

- `FaultCountV1`: total K plus the conditional generator's campaign seed and absolute draw index;
- `FaultCountEnumerationV1`: total K plus a lexicographic rank (seed is campaign metadata).

Import regenerates the original fault set under that explicit provenance, reconstructs truth
and syndrome, and verifies source-decoder outcomes. Conditional seed/index pairs are never
presented as ordinary `sample_shots` identities. Minimization preserves original conditional
provenance and faults while the reduced set becomes an explicitly counterfactual witness.
The existing exact-example signature still groups the same model/configuration/fault set
regardless of how it was found. Legacy v1 readers cannot import v2 artifacts.

Conditional draw identity is `(seed, K, absolute sample index)` with a versioned SplitMix64
composition and rand-0.8 StdRng; it is independent of scheduling. The sampler's 1-vs-4-worker
outputs are tested. The runner is initially serial, keeping the fixed endpoint and call
accounting transparent. No automatic replacement of direct Monte Carlo is introduced.

## Predeclared validation campaign

The example fixes these cases before examining outputs, 16 seeds starting at 544:

| Case | Target noise | Cutoff | Declared calls per row |
|---|---|---:|---:|
| Surface d3/r3 low | phenomenological p_data=p_meas=0.0001 | K<=5 | 256 |
| Surface d3/r3 accessible | p_data=p_meas=0.03 | K<=5 | 256 |
| Gross BB capacity low | independent Z p=0.001 | K<=5 | 256 |
| Gross BB capacity accessible | independent Z p=0.03 | K<=8 | 256 |

Surface cases use MWPM; BB cases use floating relay BP, 6 legs x 10 sweeps, alpha=0.875,
gamma=(-0.3,0.9), disorder seed 0x5E1A4B9C, full-schedule policy. Direct Monte Carlo uses
**the actual calls spent by stratification**, not the sum of possibly-unused declared budgets.
The direct and stratified decoders/plan are built once; setup times are reported separately.
Run times include sampling, decoding and the appropriate confidence interval calculation.
The comparison is of interval width and runtime at matched calls, not a claim that conditioning
makes each decoder call faster.

The offline coverage experiment uses a heterogeneous four-mechanism DEM with independently
enumerable logical failure probability. Its all-conditional design spends 32 calls in each
of five strata (160 total). A second mixed design declares [8,8,5,8,8]: complete enumeration
of k=0,1,3,4 and five draws in k=2 (15 actual calls). Both have predeclared endpoints and
record every final interval, empirical coverage and a separate binomial interval for coverage.
A conservative >=95% construction need not have coverage calibrated to exactly 95%.

Tiny CI tests enumerate equal/unequal/deterministic models, check conditional set frequencies
independently of decoder output, verify all lexicographic ranks, compare complete weighted
failure sums to exact enumeration, and test underflow, resource/input errors, omitted tails,
zero failures, fixed-budget status, worker determinism and v1/v2 witness replay/minimization.
Long repeated-seed campaigns stay outside CI.

This is fault-count stratification, not an implementation of the splitting/Bennett estimator in
[Bravyi and Vargo (2013)](https://arxiv.org/abs/1308.6270). Adaptive allocation, splitting/Metropolis,
correlated circuit proposals and physical-fault provenance are separate follow-ups.

## Measured fixed-budget results

Pinned Xeon Silver 4114, CPU3, Rayon1; 16 fixed seeds per case, no services stopped.
Direct MC receives the same **actual** calls as explicit stratification. Median widths and
runtime below are descriptive shared-host observations; this is a precision improvement,
not a claim of raw sampling speedup.

| Case | Calls/run | Stratified width | Direct width | Width ratio strat/direct | Stratified ms | Direct ms |
|---|---:|---:|---:|---:|---:|---:|
| Surface d3/r3 p=0.0001 | 1065 | 1.2651e-6 | 0.0034577 | 0.0003659 | 4.678 | 0.420 |
| Surface d3/r3 p=0.03 | 1065 | 0.056360 | 0.036894 | 1.5095 | 5.070 | 1.621 |
| Gross BB p=0.001 | 1169 | 0.00018396 | 0.0031506 | 0.058388 | 380.203 | 364.637 |
| Gross BB p=0.03 | 1937 | 0.050235 | 0.0053836 | 9.5495 | 640.054 | 630.952 |

Rare-case median widths improve about 2733x and 17.1x; higher-rate widths worsen.
Only one of the 16 rare-surface runs meets width 1e-6; the other 63 runs report
`BudgetExhausted`. Overlapping intervals in all matched pairs are a consistency check,
not proof that the estimators are equivalent. Surface conditional sampling is materially
slower per call; weighted precision, not decoder throughput, is the intended benefit.

Tiny exact-truth coverage is 2000/2000 for all-conditional allocation and 1984/2000 for
mixed exact/conditional allocation. Coverage's own 95% intervals are [0.9981573,1] and
[0.9870409,0.9954206]. These conservative results do not establish universal coverage
beyond the independent-draw model and documented numerical limitations.

Warmed Criterion observations: plan construction 38.237 us, conditional k=2 draw 1.7423 us,
fixed stratified campaign 4583.35 us, matched direct campaign 500.391 us (1065 calls each).
CPU3 external utilization during the statistical campaign was about 0.0043%, its SMT
sibling 0.0302%; the entire host was not isolated. All 44 exported Linux witnesses replay
on both Linux and Mac. The default PyMatching importer uses a different merge policy for
same-endpoint/different-observable edges; the oracle retains this discrepancy and uses an
explicit minimum-weight boundary graph for the tiny model's native truth 0.1925.

Raw campaign, every coverage interval, source fingerprints, independent oracle checks,
load trajectories and Criterion samples are in
[`../perf/data/p7-02-rare-events/`](../perf/data/p7-02-rare-events/README.md).

## Opt-in automatic selection

`run_automatic_rare_event(&plan, &decoder, &options)` selects before sampling; use
`select_rare_event_method` to inspect its decision without decoder calls. Existing explicit
methods remain available. The decision and both scores are returned in the result.

The direct score is the Clopper–Pearson width at floor(N/2) failures for the declared total
budget N. The stratified score adds mass-weighted central-count widths with the same
Bonferroni allocation used by the estimator, all omitted/unsampled mass, and exact-row
mass enclosure uncertainty. Stratification must score at most half the direct score.
This fixed margin is a conservative heuristic, **not a guaranteed width or runtime bound**.
It does not assume that low-weight faults are correctable, inspect observed errors, run a
pilot or switch methods mid-campaign. Consequently selection does not invalidate the
selected estimator's fixed-endpoint confidence construction.

Exact enumeration may use fewer calls than the declared envelope. Auto's direct branch
spends that full envelope (1536 calls for k<=5 or 2304 for k<=8 in the example), so its
comparison is budget-limited rather than the strict actual-call matching in the table.
Planning and the prebuilt decoder still cost time even if direct wins. Both runtime and
unknown conditional failure rates can defeat this score; retain the explicit override.

Direct selection delegates to the ordinary batched DEM harness and returns its failure tally
and a CP interval at `options.confidence`. `witness_limit` applies only when stratification
is selected; direct auto runs do not capture witness artifacts. Actual final bounds determine
`TargetMet`/`BudgetExhausted`, never the heuristic scores. The example records one automatic
run per case alongside the fixed explicit campaign; it uses the same fixed seed 544.


Supplemental final-source Linux runs and native tests select stratification for both rare
fixtures and direct MC for both accessible fixtures. Scores (stratified/direct) are
1.774e-6/0.05062, 0.06897/0.05062, 0.001489/0.05062 and 0.18414/0.04124,
respectively. Exact selected-method equality is tested for counts and bounds. These are
four regression fixtures, not held-out evidence of universal predictive accuracy.
