# Relay-BP iteration budgets and early exit (#514)

The f64 `RelayBpDecoder` now exposes an explicit **iterations per leg** budget
and an optional **first-valid early exit**. The same controls reach both
`relay-bp` and `relay-bp-osd` through native Python and the CUDA-Q adapter.
Defaults preserve the previous decoding schedule and selection policy.

| Configuration | Sweeps per leg | Stop/selection policy |
| --- | --- | --- |
| Omitted budget | `max(100 / max(legs, 1), 8)` using integer division | Full schedule, minimum Hamming-weight valid decision |
| `iters_per_leg=N` | `max(N, 1)` | Full schedule unless early exit is enabled |
| `early_exit=True` | Same configured cap | First syndrome-valid decision; otherwise exhaust the cap |

`legs` defaults to 4, so the legacy schedule remains **4×25**. Six legs with no
explicit iteration override means **6×16**, not 6×10. To request a 6×10 schedule,
specify both values. Zero legs and explicit zero iterations are promoted to one,
matching the fixed-point Rust constructor's policy. Python iteration overrides
must fit an unsigned 32-bit integer; negative, fractional and out-of-range values
are rejected. `early_exit` requires a boolean. Other decoder names reject these
keywords rather than silently ignore them.

## Rust

```rust
use aleph_qec::{DetectorErrorModel, RelayBpDecoder};
let dem = DetectorErrorModel::parse("error(0.1) D0 L0").unwrap();
let relay = RelayBpDecoder::with_budget(&dem, 6, 10, 0.875, (-0.3, 0.9), 7)
    .with_early_exit(true);
```

`new()` and `with_params()` retain the old full-schedule behavior.
`with_budget()` starts with early exit disabled; `with_early_exit(false)` can
disable it again. For relay-BP+OSD, pass the configured relay to the existing
`RelayBpOsdDecoder::with_parts(relay, osd)` constructor. The iteration setting
controls the relay front-end, not the OSD combination-sweep order.

## Python and CUDA-Q

```python
import aleph.qec as qec
dem = qec.DetectorErrorModel(dem_text)
decoder = qec.Decoder(dem, "relay-bp-osd", legs=6, iters_per_leg=10,
                      early_exit=True, osd_order=12, seed=7)
```

```python
import cudaq_qec as cq
import aleph.cudaq
decoder = cq.get_decoder("aleph-relay-bp-osd", H, O=O,
                         error_rate_vec=rates, legs=6, iters_per_leg=10,
                         early_exit=True, osd_order=12, seed=7)
```

The adapter forwards the parameters through the same typed native factory.
Prior/matrix validation, disorder generation, seed defaults, min-sum arithmetic
and OSD behavior are unchanged. A given seed/leg index has the same disorder
pattern regardless of how many iterations that leg is allowed to run.

## First valid is a different accuracy policy

With early exit disabled, relay-BP visits every configured iteration and keeps
the **lowest Hamming-weight syndrome-valid** estimate encountered. Its LLRs still
come from the final iteration, even if the retained estimate was seen earlier,
exactly as before. With early
exit enabled, it stops after the first iteration satisfying `H·ehat = syndrome`.
The hard decision and posterior LLRs returned by `decode_soft()` then come from
that stopping iteration. If no estimate is valid, both modes run the entire
schedule and return its final hard decision and LLRs with `converged=false`.

Syndrome validity does not establish the correct logical class. A pinned tiny
regression fixture finds a weight-5 estimate first and a weight-3 estimate later;
both satisfy the syndrome but have different observable predictions. Thus early
exit is an explicit latency/quality trade-off, not an unconditional improvement.
Relay-BP+OSD already accepts a converged relay estimate directly, so early exit
can change its output too. Non-convergence and logical mismatch remain separate
diagnostics; enabling early exit does not relabel an invalid result as valid.

This feature makes schedule-controlled comparisons possible. It does **not**
establish parity with NVIDIA's relay implementation or eliminate the historical
logical-error-rate gap: matching a 6×10 cap alone does not align arithmetic,
disorder distributions, stopping policies and OSD configurations across decoders.

## Validation and reproduction

Tests count actual sweeps, checking full-budget execution, first-valid stopping,
and cap exhaustion on an impossible syndrome. They compare stopping-iteration
LLRs bit-for-bit against an equivalent schedule prefix. Legacy/default and
explicit-budget paths have equal disorder and equal hard/soft output. Gross-code
samples verify exact serial/parallel replay and syndrome/observable projection.
The pinned fixture checks that both native Python names and the real CUDA-Q
adapter use the supplied controls, including single-shot and batch calls.

```bash
cargo test -p aleph-qec --lib relay_bp
cargo test --release -p aleph-qec --lib relay_bp
cargo test --release -p aleph-qec --test relay_budget
cargo test --release -p aleph-qec --test relay_bp
cargo test -p aleph-py --lib qec_core
PYTHONPATH=scripts/python python -m unittest test_qec.TestApi test_cudaq.TestRegistration
RAYON_NUM_THREADS=1 python scripts/python/audit_relay_budget.py --shots 256 --repeats 3
```

Run the audit with `--legacy-only` against the baseline module, then compare its
signatures with the candidate's `default`/`legacy-6` rows. The candidate also
checks those rows against explicit 4×25 and 6×16 budgets. The audit uses fixed
DEM-mechanism samples, preserves `^` correlation by reducing each original
mechanism's supports before constructing sampling matrices, and reports logical
mismatches separately from non-convergence. Its small samples are API/regression
checks, not a statistically powered quality comparison.


## Recorded results (2026-10-16)

Baseline: `bb128f7e3dcc3138b7bb89d20fba070f40f0c677`; candidate code:
`58606d38368e29bd60b04e0696b8218cb3c0a763`. All nine validated source hashes match the local patch.
The before/after audit compared 16 legacy-output signatures (two cases, two
relay names, two legacy schedules, two rounds); all agree. Explicit 4×25 and
6×16 candidate rows also agree with their corresponding omitted-budget rows.
The baseline overlay was verified to reject the new budget keyword.

Targeted Rust results: six relay unit tests in both debug and release, two new
budget integration tests, two existing relay integration tests, nine binding
factory tests and five doctests passed on Linux. The two existing nightly
Monte-Carlo tests remain ignored; no new quality claim depends on them.
Native Python passed 16 local checks. The actual CUDA-Q QEC 0.8.0 SDK passed
36 selected Python/API tests without skips on Linux, including matrix/DEM-string
and single-shot/batch routes. Workspace/all-target and Python-feature Clippy,
QEC/Python all-target Clippy, rustfmt and Python syntax checks passed. Review was
manual self-review; the full platform/workspace matrix is left to PR CI.

The illustrative audit below uses 256 fixed-seed shots per case. Rates are
medians of six timed CPU `decode_batch_errors()` batches (two ABBA-ordered
processes per version, three repeats each), excluding construction/sampling.
Both versions used standard release builds, one Rayon thread and pinned CPU 2
on an i5-13500; quiet core/SMT-sibling checks ran before each process. This is a
shared host with services left running, not an isolated performance campaign.

| Case | Mode | Relay mismatches / nonconverged | Relay+OSD mismatches / nonconverged | Relay shots/s | Relay+OSD shots/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy | 4×25 full, default seed | 63 / 74 | 62 / 0 | 83,011 | 81,736 |
| Toy | 6×16 full, seed 7 | 43 / 24 | 40 / 0 | 73,681 | 73,154 |
| Toy | 6×10 full, seed 7 | 45 / 39 | 40 / 0 | 114,621 | 113,256 |
| Toy | 6×10 first-valid, seed 7 | 61 / 39 | 56 / 0 | 346,511 | 331,963 |
| Gross circuit, r=2, p=0.003 | 4×25 full, default seed | 0 / 1 | 0 / 0 | 398 | 398 |
| Gross circuit, r=2, p=0.003 | 6×16 full, seed 7 | 0 / 1 | 0 / 0 | 393 | 392 |
| Gross circuit, r=2, p=0.003 | 6×10 full, seed 7 | 0 / 1 | 0 / 0 | 603 | 603 |
| Gross circuit, r=2, p=0.003 | 6×10 first-valid, seed 7 | 0 / 1 | 0 / 0 | 7,251 | 7,161 |

OSD order is **0** in this audit. Compare 6×16 with 6×10 full to isolate the
budget change at the same seed, and 6×10 full with first-valid to isolate stopping.
The toy case demonstrates a real quality trade-off: early exit increases logical
mismatches while non-convergence is unchanged. The gross case contains only one
empty syndrome out of 256; its fast early-exit result is not explained solely by
empty shots. Zero observed gross logical mismatches in 256 shots is far too weak
to resolve the historical LER gap. Non-convergence is reported independently.
All converged estimates satisfy the syndrome; all observable projections agree
with the separate native `decode_batch()` path.

Evidence: [raw audit samples, signatures and case composition](../perf/data/q5-514-audit.json),
[validation metadata](../perf/data/q5-514-validation.json),
[validation output](../perf/data/q5-514-validation.txt),
[source hashes](../perf/data/q5-514-source-sha256.txt),
[baseline control](../perf/data/q5-514-baseline-control.txt),
[CPU guards](../perf/data/q5-514-audit-host.jsonl), and
[paired audit driver](../perf/data/q5-514-run-audits.py).
