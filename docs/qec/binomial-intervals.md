# Direct-sampling logical-error confidence bounds

P8-16 / #566 changes the interpretation used by new direct Monte Carlo reports.
Zero observed failures are evidence about an upper bound, not proof of zero risk.

`LogicalErrorResult::try_new(shots, errors)` validates `errors <= shots`.
`interval95()` computes explicit `BinomialInterval95 { lower, upper }` endpoints
from the integer tallies, ignoring the public cached `rate` and `ci95` fields.
The method is a **two-sided, equal-tailed 95% Clopper–Pearson interval** for fixed,
independent, identically distributed Bernoulli trials. It inverts the binomial
tails through beta quantiles, as described in the
[NIST statistical handbook, §7.2.4.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
In exact arithmetic it has at least 95% coverage, generally above nominal coverage
because counts are discrete. It is not a Bayesian probability statement about the
parameter, or a decoder-equivalence test.

| Shots | Observed failures | Observed rate | 95% endpoints |
|---:|---:|---:|---:|
| 0 | 0 | no estimate | [0, 1] |
| 1 | 0 | 0 | [0, 0.975] |
| 10 | 1 | 0.1 | [0.002529, 0.445016] |
| 1000 | 0 | 0 | [0, 0.003682] |
| 1000 | 50 | 0.05 | [0.037335, 0.065390] |
| 1000 | 1000 | 1 | [0.996318, 1] |

For zero failures, the upper endpoint is `1 - 0.025^(1/n)`, evaluated with
`expm1` to avoid cancellation. A **one-sided 95%** upper bound would instead use
`0.05`; these are different contracts. No one-sided API is introduced here.
All-failure bounds use the reflected boundary formula and round the lower endpoint
outwards so even `u64::MAX` shots do not produce a singleton `[1, 1]`.
Zero shots have the full probability range; the legacy `rate = 0` field is not an
estimate in this case.

Interior quantiles use bounded bisection (at most 128 iterations) and the checked
regularized beta function from `statrs 0.18`, with default features disabled.
Only this one package is new in the lockfile; its `approx`/`num-traits` dependencies
were already present. Reusing a statistical special-function library avoids adding
an unreviewed incomplete-beta implementation. The upstream inverse uses unbounded
Newton loops, so it is not used. Nonfinite CDF values, a tail residual above `1e-7`,
unordered endpoints or interior tallies at/above `2^53` shots return
`Error::IntervalNumerics`; there is no silent Wald fallback. Floating-point
evaluation is still approximate even though the statistical construction is called
"exact". The offline reference campaign includes rare-event interior cases at
one billion shots, not a claim of validated accuracy for every possible `u64` tally.

## Compatibility and consumer inventory

The Rust result retains its four existing public fields, `Copy` behavior and
struct-literal compatibility. `new()` and `ci95` keep their old semantics for
valid inputs: `ci95 = 1.96 * sqrt(rate * (1-rate) / shots)` and zero-shot sentinel
fields remain zero. The legacy infallible constructor cannot return a validation error;
new callers use `try_new`. The experiment harness now uses the checked constructor.
Invalid legacy-constructor inputs now retain their counts with NaN rate/width,
so zero-shot invalid inputs cannot silently become a valid `(0, 0)` tally.
No asymmetric interval is disguised as an old half-width.

| Consumer | Policy in this change |
|---|---|
| Rust `run_dem_experiment` / `run_memory_experiment` | Same counts and legacy fields; checked tally construction, explicit bound method and updated example. |
| `qec_threshold` CSV | Existing columns retained; append `ci95_lower`, `ci95_upper`, `ci95_method=clopper-pearson`, `has_estimate`. |
| `scripts/qec_threshold_plot.py` | Prefer explicit endpoints, draw asymmetric error bars; omit zero-shot estimates and show zero-failure upper endpoints on the log axis. Historical schemas retain Wald half-widths with an explicit warning. |
| Q2/Q5/Q6/Q7 example CSV and hardware golden-format outputs | Keep historical numeric half-width columns for format compatibility; print a legacy-field notice. Published CSV/figures and hardware comparison scripts are historical Wald results, not retroactively corrected. |
| Q4/Q7 marginal comparison helpers | Replace the sum-of-Wald decision with actual interval overlap. Rename decision columns to `intervals_overlap95`; overlap is descriptive, never an accuracy/equivalence certification or sufficient reason to reduce hardware precision/budget. |
| `qec_q1_compare` shared-shot oracle report | Retain paired discordant counts; conditional exact binomial test of an equal split replaces the independent-rate formula. Changed columns name the paired test and legacy widths. A non-rejection does not establish equivalence; no family-wise correction is claimed. |
| Existing statistical regression tolerances | Existing `ci95` uses remain legacy heuristics; they do not gain a new interval meaning. New interval correctness tests are separate. |
| Python QEC bindings / main `aleph` CLI | Neither exposes `LogicalErrorResult` or its `ci95`; no Python wheel or CLI command migration is needed. Threshold sweeps are Rust examples. |
| CUDA `CellResult::ci95` and RTL/C++/Python hardware helpers | Separate legacy Wald APIs/data, unchanged by this Rust direct-result migration. Callers can pass integer direct-sampling counts into the new API when its assumptions apply. No device/kernel or hardware-format change is claimed. |

Paired experiments keep their original shared shot stream. Marginal overlap
discards pairing information and is not substituted for a paired comparison where
the comparison explicitly retains discordant outcomes. Weighted importance
sampling, stratified estimators, adaptive stopping and correlated sliding-window
observations require a separate uncertainty model: this API only accepts integer
Bernoulli tallies, with no weighted-estimate overload. Its independence/fixed-trial
assumptions remain the caller's responsibility. Phase 7 estimators must not treat
their effective sample size as an ordinary binomial shot count.

## Validation and reproduction

Hermetic Rust tests cover reference bounds, invalid tallies, no evidence, both
boundaries through `u64::MAX`, monotonicity, symmetry, randomized direct counts,
large rare-event counts and conditional paired comparisons. Finite-sum coverage
tests enumerate every possible count for small experiments; these have no Monte
Carlo uncertainty. Python loader tests cover asymmetric error bars, historical
schemas, zero shots, invalid endpoints and unsupported interval methods.

The offline campaign in `../perf/data/p8-16-binomial-coverage.json` compares 3633
Rust intervals against independently computed SciPy beta quantiles, enumerates
72 small-sample coverage cases and simulates 10 larger/rare operating points with
20,000 independent experiments each (seed 566). The report includes the simulation
standard errors and all operating points, including boundaries. It does not use
observed sample coverage as proof of coverage for unsampled probabilities.
The maximum absolute SciPy endpoint difference was `6.24e-13`; the minimum exact
finite-sum coverage was `0.9556901`. Simulated coverage ranged down to `0.95045`
with standard error up to `0.001535`; the predeclared check rejects a simulation
only if coverage plus five standard errors is below 0.95. No numerical failures
occurred in this campaign. The JSON records source/binary hashes and tool versions.

```sh
cargo test --locked -p aleph-qec --test binomial_interval
python3 -m unittest scripts/python/test_qec_threshold_plot.py -v
cargo build --locked -p aleph-qec --example binomial_intervals
# In a temporary environment containing scipy==1.16.2:
python scripts/python/check_binomial_intervals.py \
  --binary target/debug/examples/binomial_intervals --output coverage.json
```

This is a statistical-correctness change, not an execution optimization; there is
no throughput claim or GPU timing requirement. Bounds are computed on request
after tallying, not inside the shot/decoder hot loop.
