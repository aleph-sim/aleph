# P8-01 — optimized CPU/Metal CLI execution

Issue: [#551](https://github.com/aleph-sim/aleph/issues/551).

**Status:** implementation and CPU evidence complete; real Metal correctness
passed. Metal before/after timing remains pending on a free Mac, explicitly
deferred by the user. The PR remains a draft and the issue stays open.

## Execution and observable contract

CLI CPU FP64/FP32 and Metal SV now select an optimized executor independently
of the shared counts/amplitude/expectation formatter. `--raw` retains verbatim
execution for A/B comparisons. The original Rust `run_circuit` signature remains
available and delegates to default execution options. MPS, stabilizer, noisy
trajectories, CUDA and the legacy `bench` subcommand retain their existing drivers.

The measurement-oriented default pipeline's DCE is unsafe for CLI final-state
views: an X on an unmeasured qubit can disappear when another qubit is measured.
The new final-state pipelines omit DCE for **every** view, including counts that
resample the collapsed final state. They retain cancellation, fusion, barriers,
measurement order, and relabelling with final logical-order restoration.

Metal uses its supported final-state wrapper and flushes pending commands.
CPU uses the common optimized driver with its own final-state pass list. CPU
keeps small precomputed diagonal gates instead of introducing phase polynomials;
the pilot below showed that fewer instructions need not mean less work. The
existing default pipeline used by other callers is unchanged. FP32 supports
fused diagonal/dense gates and permutation restoration. Its documented
`TiledBlock` implementation replays contained gates correctly, rather than using
FP64's tile-major fast path. Unsupported reset still raises in both modes.

`--execution-stats` reports the actually executed plan's input/retained
instruction counts, top-level fused kinds and clone/pass planning time on stderr.
These are **IR counts**, not kernel launches or full-state memory passes; tiled
blocks can contain multiple gates, and replaying backends retain those costs.

Each mode is deterministic for a fixed version/backend/precision/seed. Fusion
changes floating-point rounding, which can change alias-table bins and the
seed-to-outcome mapping. Raw and optimized histograms therefore need not be
bit-identical, despite matching amplitudes/distributions within the precision
contract. CLI seed help documents this; `--raw` preserves the reference path.

## Public CPU before/after

Linux 6.8, Intel i5-13500 (20 logical CPUs), Rust 1.96.0 release build, Python
3.12 subprocess harness, `RAYON_NUM_THREADS=1`. This compares the **same binary**
with `--raw` against optimized execution, not two historical binaries. It measures
whole-process wall time: startup, parse, planning, execution, sampling and
formatting. Both modes request 32 shots, seed 7 and execution diagnostics.
Compilation finished before the final campaign. Services were not stopped;
host load before the first campaign was 0.34/0.30/0.12 and was below 1 at a later
check. Sustained dedicated-core isolation was not established.

Three fresh processes per mode, interleaved raw/optimized/optimized/raw/raw/optimized.
QFT uses a controlled-phase decomposition and final SWAPs; brickwall uses 12
deterministic rotation/nearest-neighbor CX layers. Exact source and output hashes,
all repeats/spread, retained kinds and planning times are saved in
[data/p8-01-cpu-f64.json](data/p8-01-cpu-f64.json) and
[data/p8-01-cpu-f32.json](data/p8-01-cpu-f32.json).

| Precision / workload | Raw → optimized median (ms) | Input → planned IR | Median planning (µs) |
|----------------------|----------------------------:|-------------------:|---------------------:|
| FP64 Bell n=2 | 1.051 → 1.095 | 2 → 1 | 9.285 |
| FP64 QFT n=20 | 929.441 → 955.058 | 980 → 79 | 834.926 |
| FP64 brickwall n=20 | 774.322 → 537.925 | 594 → 61 | 283.709 |
| FP32 Bell n=2 | 1.146 → 1.182 | 2 → 1 | 13.383 |
| FP32 QFT n=20 | 2085.930 → 1050.777 | 980 → 79 | 818.323 |
| FP32 brickwall n=20 | 1130.771 → 579.510 | 594 → 61 | 487.326 |

Useful gains are workload/precision dependent: FP64 brickwall is about 1.44×,
FP32 QFT/brickwall about 1.99×/1.95×. FP64 QFT is about 2.8% slower here; it is
not counted as a win. Tiny-input overhead is visible and process noise overlaps.
No universal speedup or calibrated confidence interval is claimed from three repeats.

### Why the first pipeline was rejected

The initial final-state pipeline retained `FuseDiagonalRuns`. On FP64 QFT n=20
it reduced 980 instructions to 44 but changed 941.429 ms raw into 1452.329 ms
optimized (about 54% slower). Per-amplitude phase-polynomial evaluation replaced
cheap constant phase multiplications. CPU planning now leaves those small
diagonals precomputed, reducing this regression to the much smaller no-gain case
above. Metal keeps the GPU-supported phase-polynomial path. Pilot evidence:
[data/p8-01-pilot-phase-polynomial.json](data/p8-01-pilot-phase-polynomial.json).
Local Mac pilot timings were discarded because load averages were around 160.

## Criterion over actual CLI processes

`crates/aleph-cli/benches/public_pipeline.rs` launches the public release binary,
including parse/planning/sampling/output work. No diagnostics are requested here;
stdout/stderr are redirected but formatting still runs. Same single-thread budget,
32 shots and seed 7. Each cell has 10 Criterion samples, 1 s warmup and 2 s
requested measurement time; n=15 QFT/brickwall fixtures are the committed
Qiskit-baseline inputs (brickwall depth 20, distinct from the n=20 depth-12 sweep).

| Precision / fixture | Raw → optimized median estimate (ms) |
|---------------------|-------------------------------------:|
| FP64 Bell n=2 | 0.800 → 0.810 |
| FP64 QFT n=15 | 19.650 → 20.008 |
| FP64 brickwall n=15 d=20 | 25.605 → 22.854 |
| FP32 Bell n=2 | 0.868 → 0.889 |
| FP32 QFT n=15 | 37.224 → 19.194 |
| FP32 brickwall n=15 d=20 | 40.369 → 24.355 |

FP64 QFT's median confidence intervals overlap (raw 18.437–19.980 ms,
optimized 19.245–20.629 ms), so this is a no-gain cell. Raw Criterion estimates
and samples, including outliers and confidence intervals:
[data/p8-01-criterion.json](data/p8-01-criterion.json).

## FP32 sampling rounding check

The n=20 brickwall FP32 seeded histogram differs between modes, while each
mode repeats exactly. A separate complete-state offline oracle found:

- raw FP32 probability L1 error vs raw FP64: 1.691713706e-6;
- optimized FP32 probability L1 error: 5.524349248e-7;
- optimized maximum amplitude error: 2.854471942e-9.

This supports the declared numerical contract; identical 32-shot histograms are
not a distribution-equivalence proof. The raw and optimized RNG seed handling is
unchanged; the plan changes arithmetic and hence the sampling table's rounding.

## Verification and remaining work

- 35 CLI unit tests, 40 existing command tests, GPU-init regression and three
  new public optimizer tests pass. The new tests compare counts, complete
  amplitudes and Y/Z expectations against raw FP64 with phases, controls,
  barriers and partial measurements; unsupported reset is tested in both modes.
- Final-state backend oracles cover CPU FP64/FP32, a nontrivial relabelling case,
  external-control fences and 64 generated gate/measurement/barrier circuits.
  Existing optimized-driver oracles also pass. The n=20 offline FP32 check passed
  on the server. Three gate/barrier fixtures also match the public CPU Python API.
- Real Metal CLI raw/optimized oracle tests pass with hardware explicitly enabled.
  Public Metal timing is **pending**, not a skipped benchmark success.
- Default/Metal CLI and workspace all-target clippy, formatting and diff checks
  pass. CUDA-enabled CLI compilation passes on Linux; its executor stays raw.

Before completing #551: collect paired Metal raw/optimized timings on a free Mac,
record planning/output overhead and any no-gain cells, then review the final head
and require all CI jobs green. No Mac programs/services were stopped by the agent.

```bash
cargo test -p aleph-backend --test final_state_oracle
RAYON_NUM_THREADS=1 cargo test --release -p aleph-backend --test final_state_oracle -- --ignored --nocapture
cargo test -p aleph-cli --test optimized_execution
ALEPH_TEST_CLI_METAL=1 cargo test -p aleph-cli --features metal --test optimized_execution
cargo build --release -p aleph-cli
RAYON_NUM_THREADS=1 python scripts/python/bench_cli_optimized.py --binary /absolute/path/to/aleph --precision f64 --qubits 20 --output cpu-f64.json
RAYON_NUM_THREADS=1 ALEPH_BENCH_CLI=/absolute/path/to/aleph cargo bench -p aleph-cli --bench public_pipeline
```

For Metal, build the CLI with `--features metal` and use the Python harness with
`--backend metal --precision auto`. Wait for the Mac to be available before timing.
