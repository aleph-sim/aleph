# P8-06: CPU sampling without single-shot alias tables

Issue #556 targets the public CPU sampler and the terminal sample inside every
`run_noisy` trajectory. The former path validated into a probability vector,
built five more alias scratch/table vectors, drew one index and discarded them.
The new path validates the complete immutable state without storing probabilities,
then scans a cumulative sum for one weighted draw. Zero shots still fully validate
the state, return an empty result and consume no RNG. FP64 AoS, FP64 SoA and FP32
all use this policy; FP32 probabilities are widened before squaring/accumulation.

The threshold is deliberately **one shot**. At two or more shots the existing
Vose alias algorithm remains, with unchanged allocation requirements and seed
mapping. No broader optimal low-shot crossover is claimed from this campaign.
The `1/2/4/16/1024` matrix measures the actual public dispatch, not a hypothetical
multi-draw streaming prototype. Larger streaming thresholds remain a measured
follow-up, especially because they would change more callers' seeded outputs.

## Correctness and reproducibility

Shape, finiteness and norm validation are shared with the existing probability-
vector queries. Validation completes before any RNG consumption or early CDF
exit, including when an invalid amplitude lies after a certain first outcome.
The existing FP64 `sqrt(dimension) * AMPLITUDE_TOL` and FP32 `1e-4` drift budgets
are unchanged. A draw uses the actual validated total mass, strict comparison
skips zero bins, and endpoint-rounding fallback selects the last positive bin.
No measurement-collapse degeneracy cutoff is imposed on sampling branches.

Single-shot sampling now uses one 53-bit uniform variate instead of an alias
column/32-bit threshold split. **Historical single-shot seed-to-outcome mapping
changes**, including noisy histograms; same-version/mode/configuration seeds are
reproducible. Both algorithms consume exactly one RNG `u64` even for a forced
outcome. The absolute noisy shot-seed derivation, channel/gate order and subsequent
readout RNG position remain unchanged. Noiseless fusion, trajectory state reuse,
concurrency limits and sparse histograms are not part of this issue (#557 follows).

Local validation: 346 SV unit/property tests, six stored Qiskit Aer noisy-density-
matrix/readout oracle fixtures, three noise-driver tests, two new public sampler/
worker-invariance tests, million-shot convergence and three FP32 equivalence tests
passed. Public analytic nonuniform sampling is checked through all three layouts;
worker invariance covers Pauli, Kraus and readout noise on 1/2/4 workers. Tests also
cover invalid shapes, empty/NaN/Inf/unnormalized states at 0/1/many shots, exact
zero-bin and tiny-branch boundaries, RNG advancement and legacy alias mapping.

## Measurements

Intel i5-13500, Linux 6.8, Rust 1.96; identical benchmark source compiled against
baseline `8c9223f` and the final isolated-helper candidate in **separate Cargo
targets**. Optimized Cargo profiles, no custom RUSTFLAGS. CPU affinity is logical
CPU0 (P-core0), `RAYON_NUM_THREADS=1`; shared services stayed running. Preflight
checked for competing build/bench jobs; load was approximately 0.2 before the
campaign and approached 1.0 while our one-core benchmark ran. This is a shared
host measurement, not dedicated-machine isolation.

Criterion order is A/B/B/A, with 10 samples per case per repeat, 0.3 s warmup and
0.7 s requested measurement (Criterion can extend expensive cases). States are
prepared outside sampler timings and the RNG advances between calls, avoiding
a benchmark of one fixed early/late CDF exit. Noisy timings include allocation,
execution, sampling, readout and the dense histogram. Numbers below are the mean
of the two **median point estimates**, not the fastest repeat. Raw samples and
per-repeat confidence intervals for all 48 cases × 4 repeats are committed.

| Public sampler, one shot | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| FP64 AoS n=4 | 102 ns | 32 ns | 3.15× |
| FP64 AoS n=12 | 12.85 µs | 3.06 µs | 4.20× |
| FP64 AoS n=20 | 10.366 ms | 0.820 ms | 12.65× |
| FP64 SoA n=20 | 10.308 ms | 0.964 ms | 10.69× |
| FP32 n=20 | 10.212 ms | 0.849 ms | 12.03× |

| Public noisy run | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| n=4, 1024 trajectories | 536.0 µs | 414.7 µs | 1.29× |
| n=16, 64 trajectories | 96.39 ms | 83.80 ms | 1.15× |
| n=20, 8 trajectories | 366.65 ms | 274.84 ms | 1.33× |

The noise model is H on every qubit, depolarizing(0.05, 1) attached to H, plus
qubit-0 readout `[[0.97,0.03],[0.07,0.93]]`, seed 556. These macro gains are much
smaller than sampler-only gains because state allocation and gate/channel work
remain. No end-to-end GPU, Python or Metal speedup is inferred from these numbers.

Allocation profiles count requested Rust heap allocations/bytes and additional
live-heap peak, **not RSS or allocator-retained pages**. Prepared sampler states
are excluded; the returned sample vector is included. For n=20 one draw, all
three layouts fall from **7 allocations / 36 MiB scratch peak to 1 / 8 bytes**.
Zero shots go from six allocations to zero. Many-shot allocation records match
baseline exactly. The warmed one-worker noisy n=20/8 call drops its additional
heap peak **52.00 → 16.00 MiB** and allocated bytes **424.00 → 136.00 MiB**;
the still-dense result and per-trajectory state are included in this measurement.
The n=4 noisy peak remains 16,688 bytes despite fewer allocation calls: eliminating
small sampler buffers does not reduce the dominating result-collection peak.

### Limits and negative results

The initial valid candidate put the low-shot scan in the alias function; several
tiny-state many-shot cells regressed up to 6.6%, including ~5–6% at 1024 shots.
Moving the scan into an `inline(never)` helper reduced that 1024-shot penalty to
approximately 0–1% in the repeat campaign. The initial raw campaign is retained.
The final many-shot time ratios range from 0.962 to 1.052: several tiny 2–4-shot
cases still cost roughly 3–6 ns more (up to 5.3%). This is a retained dispatch/
code-generation tradeoff, not a many-shot improvement claim; the algorithm,
allocations and seeded mapping are preserved. Small wins within repeat variability
are not advertised. Wider states and tested noisy callers provide the useful gain.

An earlier **invalid pilot** reused one Cargo target across two copied repository
roots: Cargo returned the same release executable for both. Matching binary hashes
and unchanged allocation counters exposed it. All pilot timings were discarded.
Accepted runs require independent targets, distinct binary hashes and public
allocation preflight (0 shots = zero allocation, 1 shot = exactly 8 bytes; many
shot records unchanged) before timings. This is why a feature build or passing
correctness oracle alone cannot establish that an optimization executed.

## Data and reproduction

- `data/p8-06-manifest.json`: entry points, policies, provenance, binary/source hashes and resource log.
- `data/p8-06-summary.csv`: all 48 cases and both repeat medians.
- `data/p8-06-criterion.json`: final per-repeat estimates, confidence intervals and raw samples.
- `data/p8-06-initial-helper-criterion.json`: valid initial candidate, including tiny regressions.
- `data/p8-06-allocations-{baseline,candidate}.csv`: public allocation profiles.

```sh
cargo test --locked -p aleph-sv --lib --test single_shot --test noise_driver --test noise_oracle
cargo clippy --locked -p aleph-sv --all-targets -- -D warnings
# Run separately against each revision; copy this identical harness to the baseline.
CARGO_TARGET_DIR=/tmp/target-for-this-revision \
  cargo bench --locked -p aleph-sv --bench sample --no-run
# After all builds finish, alternate the resulting executables A/B/B/A:
RAYON_NUM_THREADS=1 ALEPH_BENCH_OUTPUT=/tmp/results-this-repeat \
  taskset -c 0 /absolute/path/to/sample-bench p8_single_shot --bench --noplot
CARGO_TARGET_DIR=/tmp/target-for-this-revision \
  cargo run --locked --release -p aleph-sv --example sampling_allocations
```
