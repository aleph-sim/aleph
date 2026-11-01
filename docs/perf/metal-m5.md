# Metal suite on Apple M5 (MacBook Air)

> Re-run of the Metal SV + MPS benches on an **M5 MacBook Air**, next to the earlier M3 Air / M4 Mini
> numbers ([`phase5.5.md`](phase5.5.md), [`metal.md`](metal.md), [`phase5.7.md`](phase5.7.md),
> [`phase5.8.md`](phase5.8.md)). Issue #520. Perf is never gated in CI.
> Sections 1–4 preserve the **2026-10-03 historical campaign**; the separately
> identified 2026-11 follow-up provides correctness/reach evidence. Its timing
> attempt was stopped after external contention; the missing-cell/drift gate
> remains open. It is not a fresh rerun of every source-grid cell.

## Historical verdict (2026-10-03)

- **SV:** at n=28 the same-Mac GPU FP32 vs CPU FP32 speedup is **6.4–7.0×** (M4 Mini: 4.7–6.1×), and
  Grover n=20 is **8.7×** (M4: 2.8×). These ratios mix machine, power, code and
  long-run conditions; they do not isolate a chip-generation effect. CPU FP32
  QFT-28 was 32.2 s vs 22.8 s on the published M4 baseline. The historical GPU
  times are 20% lower on GHZ-28 and 15–21% higher on QFT/random-28; neither the CPU
  nor GPU difference is causally attributed to the fanless chassis without clocks
  and controlled repeats.
- **MPS:** the GPU-MPS / CPU-MPS gap narrows slightly, from 16.1× to **14.3×** at n=16 χ=256 and from
  ~22× to **19.2×** at n=20 χ=256. The Phase 5.8 exit metric (GPU MPS ≥ CPU MPS in one regime) is
  **still unmet on the M5**. The wall is unchanged: the single-threadgroup per-block factorisation.
- **Correctness holds on the new GPU:** `cargo test -p aleph-metal --features metal --release` has 80
  tests passing, plus the 2 `--ignored` long tests, with no failures. That includes the 1e-5 Aer oracle
  fixtures and the pre-timing GPU-vs-FP64 guard in `sv_vs_cpu`.

## Historical machine and conditions

| | |
|---|---|
| Model | MacBook Air, `Mac17,3`, **no fan** |
| Chip | Apple **M5**: 10-core CPU (**4P + 6E**), **10-core GPU** |
| RAM | **32 GB** unified |
| OS | macOS 27.0.1 |
| Toolchain | stable Rust, `--release`, default `RUSTFLAGS` |
| Date | 2026-10-03 |

**Honesty caveats:**

- **Live desktop**, not an idle box (7 login sessions, display driven by the same GPU). The load average
  was ~2 before the run. During the CPU arms it rose to 10–14, which is the bench's own threads on
  10 cores.
- **Power:** on AC at the start of every bench (logged per bench). Some time after 19:13 the laptop
  went back to battery, during `mps_vs_cpu`, so its `large_n` cells may have been partly on battery.
- **Single run per bench.** Run-to-run drift was not measured. On a fanless chassis, a run of more than
  an hour makes power/thermal drift plausible, but no clock/temperature trace
  established either steady state or peak frequency. Wall-clock per bench:

  | bench | wall-clock |
  |---|---|
  | `sv_fused` | 3 m 46 s |
  | `sv_vs_cpu` | 67 m 51 s |
  | `sv_readout` | 1 m 07 s |
  | `mps_batched` | 48 s |
  | `mps_vs_cpu` | stopped after ~2 h 40 m (see §4) |

- The M4 Mini comparison mixes chassis (fan vs no fan) as well as chip generations. The M3 Air is the
  like-for-like chassis, but only `sv_fused` was ever run there.

## 1. State vector, GPU vs CPU (`sv_vs_cpu`)

Same method as P5.5-05: all arms use `run_optimized`. `gpu` = `MetalSvBackend` (FP32),
`cpu_f32` = `Fp32SvBackend`, `cpu_f64` = `NaiveSvBackend`, 10 samples. Values below are **Criterion reported time point estimates**:
**mean for Flat sampling, slope for Linear sampling**, corrected from the earlier
incorrect “medians” label. Recovered October Criterion artifacts
([archive](data/metal-m5-520/recovered-local-criterion.json)) match, for example,
GHZ-28 GPU mean 794.789 ms (median 776.099 ms), GHZ-24 GPU slope 37.757 ms
(mean/median ~37.573 ms), and CPU n20/chi512 mean 6.725 s (median 6.698 s). The archive was recovered from unversioned local files; source
identity and process conditions cannot be reconstructed from mtimes alone.

| Workload | n | gpu | cpu_f32 | cpu_f64 | **cpu_f32/gpu** | cpu_f64/gpu | M4 cpu_f32/gpu |
|---|---|---|---|---|---|---|---|
| GHZ | 24 | 37.8 ms | 237 ms | 267 ms | **6.28×** | 7.07× | 4.31× |
| GHZ | 26 | 175 ms | 1.103 s | 1.208 s | **6.31×** | 6.91× | 4.40× |
| GHZ | 28 | 795 ms | 5.582 s | 7.054 s | **7.02×** | 8.87× | 4.71× |
| QFT | 24 | 205 ms | 1.587 s | 1.672 s | **7.75×** | 8.17× | 5.69× |
| QFT | 26 | 1.006 s | 7.535 s | 7.721 s | **7.49×** | 7.68× | 6.04× |
| QFT | 28 | 4.731 s | 32.16 s | 33.51 s | **6.80×** | 7.08× | 6.10× |
| random | 24 | 473 ms | 3.740 s | 4.074 s | **7.90×** | 8.61× | 4.31× |
| random | 26 | 2.423 s | 16.26 s | 18.44 s | **6.71×** | 7.61× | 4.53× |
| random | 28 | 10.72 s | 68.52 s | 77.90 s | **6.39×** | 7.27× | 4.67× |
| Grover | 20 | 2.471 s | 21.58 s | 22.36 s | **8.73×** | 9.05× | 2.82× |

Absolute times at n=28 vs the M4 Mini (M4 time ÷ M5 time; >1 means the M5 is faster):

| n=28 | GPU | CPU FP32 |
|---|---|---|
| GHZ | 1.20× | 0.81× |
| QFT | 0.79× | 0.71× |
| random | 0.85× | 0.62× |

The random-28 CPU arm alone is 10 samples × ~70 s of all-core load. Chassis,
power, runtime and thermal drift are possible contributors, not measured causes.
Grover's ratio change combines a slower published CPU time (14.0 → 21.6 s) and
lower GPU time (4.98 → 2.47 s), so it cannot be attributed to the CPU alone.

**Public in-core reach.** n=28 (1 GiB FP32 state) is the largest supported and
measured public Metal SV case. `MAX_METAL_QUBITS=28` existed in the historical
source too; n=29 is rejected before allocation. The prior text's n=29–31 “not
verified” label missed this policy cap. A 16 GiB state-size calculation at n=31
is not evidence of a runnable public backend: allocation also creates a temporary
host vector before copying into the Metal buffer, and readout/scratch need further
memory. The follow-up queries device limits and checks the cap without overriding
it or forcing an OOM.

## 2. Fusion on the GPU (`sv_fused`)

| Workload | n | unfused | fused | unfused/fused | M3 (P5.5-04) |
|---|---|---|---|---|---|
| QFT | 14 | 0.860 ms | 0.852 ms | 1.01× | 3.15× |
| QFT | 16 | 1.631 ms | 1.523 ms | 1.07× | 3.02× |
| QFT | 18 | 2.490 ms | 3.786 ms | 0.66× | 3.14× |
| random | 14 | 2.823 ms | 2.556 ms | 1.10× | 5.44× |
| random | 16 | 3.946 ms | 3.376 ms | 1.17× | 6.55× |
| random | 18 | 11.22 ms | 4.310 ms | 2.60× | 8.62× |
| GHZ | 16 | 1.313 ms | 1.628 ms | 0.81× | 2.27× |
| GHZ | 18 | 2.039 ms | 2.336 ms | 0.87× | 1.69× |
| Grover | 15 | 215 ms | 133 ms | 1.62× | 6.02× |

**The M3 column is not comparable.** When P5.5-04 measured it, `unfused` (`MetalSvBackend::run`) paid
one `wait_until_completed` per gate. P5.6-04 then moved `run` onto a single batched command buffer
(`sv/state.rs`), which removed the per-gate sync that fusion was saving. On today's code, fusion only
cuts kernel passes: a clear win on random-18 (2.6×) and Grover (1.6×), roughly neutral on small
QFT/random, and a regression on QFT-18 and GHZ. On GHZ, the CNOT chain fuses into dense k-qubit blocks
whose matvec costs more than the cheap per-gate kernels it replaces. This is a fusion-policy follow-up
for the Metal path (the CUDA path solved the same problem with width-3 caps and CNOT/diagonal kernels,
P5.9-02..06). It is not M5-specific.

## 3. Readout (`sv_readout`)

| Op | n=20 | n=22 |
|---|---|---|
| `measure` | 0.689 ms | 3.09 ms |
| `expectation` (X,Y,Z) | 2.03 ms | 9.68 ms |
| `probabilities` | 0.856 ms | 3.35 ms |

There is no earlier published table for this bench, so these are a first baseline.

## 4. MPS on Metal (`mps_vs_cpu`, `mps_batched`)

**Large-n, the Phase 5.8 exit regime** (`cpu` = f64 CPU MPS, `gpu` = GPU-resident FP32 MPS):

| Cell | cpu | gpu | **gpu ÷ cpu (M5)** | M4 (phase5.8) |
|---|---|---|---|---|
| n=16 χ=256 | 0.359 s | 5.143 s | **14.3×** | 16.1× |
| n=20 χ=256 | 4.229 s | 81.01 s | **19.2×** | ≈22× |
| n=20 χ=512 | 6.725 s | not measured | — | — |

The historical `n20_chi512` GPU cell was **stopped before it finished**. Criterion
estimated 2,233 s for its 10 samples, and the laptop had to go back to battery.
This was the outstanding cell carried forward from PR #526; its follow-up is
reported separately below. Larger source-grid cells were not part of the published
M3/M4 comparison and are not silently described as measured.

**Small cells** (best GPU path ÷ CPU):

| Workload | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| NN brickwall (`gpu_batched`) | 130× | 40× | 19.2× | 10.3× |
| bond-saturating (`gpu`) | 159× | 63× | 46× | 48× |

The NN-brickwall ratio keeps shrinking with n. Extrapolating it is not a crossover claim: the large-n
cells above (which have more bond dimension) move the other way.

**Layer batching (`mps_batched`, n=12):**

| Workload | gate-by-gate | batched | speedup |
|---|---|---|---|
| NN brickwall d=24 | 381 ms | 313 ms | 1.22× |
| bond-saturating l=16 | 203 ms | 222 ms | 0.91× |

The direction matches the M4 (P5.7-04: 1.30× and 1.05×). On the M5, batching loses slightly in the
bond-saturating case.

## 2026-11-01 follow-up — remaining timing gate still open

**Status: interrupted and then explicitly deferred by the user; not a completed
performance campaign. Issue #520 stays open.** The [frozen protocol](data/metal-m5-520/protocol.md) and helper commit
`c8e484b` targeted the one unfinished cell retained by PR #526, pre/post CPU and
SV drift controls, readout/batching refresh, and an API/device reach audit.
This does not replace the historical n24/26/28 SV or full MPS tables above.

### Correctness and operational reach

On the actual M5, the normal release Metal suite passed **82 tests**, with two
existing ignores. The large ignored `bond_saturating_norm_unit_large_n` test was
then explicitly executed: **1 passed** in 241.39 s, covering n16 and n20 with
chi=512 and the existing **2e-3 norm tolerance**. This is not a 1e-5 full-amplitude
oracle for a truncated large-n MPS. The separate SV oracle fixtures retain their
own tolerance. [Suite log](data/metal-m5-520/metal-tests.log),
[large-chi log](data/metal-m5-520/large-chi512-norm.log).

The [safe public reach probe](../../crates/aleph-metal/examples/metal_m5_audit.rs)
ran GHZ n28, checked every FP32 amplitude for finiteness, checked its two expected
peaks, and summed the full norm (0.9999999657714582, tolerance 1e-5). It also
required the explicit n29 `TooManyQubits { limit: 28 }` rejection. It performed no
n29–31 state allocation or cap override. The exact [device receipt](data/metal-m5-520/reach.json)
reports:

| Quantity | Observed bytes / result |
|---|---:|
| Logical n28 FP32 amplitude buffer | 1,073,741,824 (1 GiB) |
| Device maxBufferLength | 20,100,448,256 (~18.72 GiB) |
| Recommended working set | 26,800,603,136 (~24.96 GiB) |
| Current process Metal bytes while retaining the probe state | 2,150,039,552 |
| Public n29 allocation | rejected before allocation, limit 28 |

Device limits and the recommended working set are not a free-memory measurement,
a peak allocation audit or proof that a hypothetical n31 circuit is runnable.
Public measured/supported SV reach remains **n28**; lifting the policy cap and
accounting for host initialization copies/readout would be a different change.

### Power, contamination and incomplete measurements

The follow-up ran on **Battery Power**, explicitly logged, with 10 Rayon workers
on the same 32 GB M5 Air. It is not AC-equivalent to a historical M4 Mini run.
[Power/load telemetry](data/metal-m5-520/campaign/host-power.jsonl), timestamped
benchmark events and executable hashes are preserved under the
[campaign manifest](data/metal-m5-520/campaign/manifest.json).
The instrumented runner does not compile during measurements or stop user apps.

Before the observed external load spike, **13 fresh Criterion cells** completed: the two random-n18
SV controls, six readout cells, four layer-batching cells, and CPU n20/chi512.
These used 10 samples with default 3 s warmup / 5 s requested measurement. The
readout/SV control sample count differs from the original default-100 campaigns;
no chip-generation or thermal improvement is inferred from it. Partial early
median estimates (not the historical reported mean/slope columns). These are individual early
observations, not a completed drift-qualified campaign:

| Early cell | Follow-up median |
|---|---:|
| Random n18 unfused / fused | 18.112 / 7.962 ms |
| Measure n20 / n22 | 1.258 / 2.783 ms |
| Expectation XYZ n20 / n22 | 2.509 / 10.544 ms |
| Probabilities n20 / n22 | 1.320 / 3.333 ms |
| NN MPS n12 d24 gate-by-gate / batched | 379.287 / 305.045 ms |
| Saturating MPS n12 l16 gate-by-gate / batched | 202.388 / 225.368 ms |
| CPU n20/chi512 | 7.470 s median; 8.343 s mean |

At 22:45:27 UTC, while the GPU cell collected samples, external Java and emulator
work saturated the CPU: 0% idle, load 27.74/12.37/7.36 on 10 cores, compressed
memory displayed as 7421M. [Admission/abort receipt](data/metal-m5-520/campaign/admission.json)
records the observation and the operator SIGINT to only the benchmark child.
The actual GPU cell therefore has **no completed estimate/sample set**. Its
~2328 s collection message is an estimate, not a measured result. Post-controls
were not run, so no before/after drift conclusion or new GPU/CPU ratio is available.
The [summary validator](../../scripts/python/summarize_metal_m5_followup.py)
rejects this incomplete campaign rather than reporting it as a successful run.

The MPS crate/shader files and missing-cell benchmark **at measured base
`ae537ad`** match the published October revision across 36 hashed files ([comparison](data/metal-m5-520/source-comparison.json)).
This controls source provenance only; it does not reconstruct the original binary,
compiler, exact thread settings, power, display load or thermal conditions.
`pmset` saying “no warning recorded” is not evidence that throttling was absent.

### Remaining completion gate

The user deferred the long repeat and plans to connect AC power. No AC run is
claimed here. On an uncontended Mac, finish the existing GPU n20/chi512 cell and matched
CPU/SV controls under continuously recorded stable supply. Retain every sample,
interval and adverse case. A >10% median control change is reported as observed
drift, not attributed to temperature without independent evidence. Keep this
issue open until that gate is satisfied; do not treat partial logs as a pass.
The explicit [runner](../../scripts/python/bench_metal_m5_followup.py) uses
prebuilt executables and separate Criterion homes so a retry cannot overwrite
this rejected attempt or the recovered October data.

## Reproduce

```bash
cargo test  -p aleph-metal --features metal --release
cargo test  -p aleph-metal --features metal --release -- --ignored
for b in sv_fused sv_vs_cpu sv_readout mps_batched mps_vs_cpu; do
  cargo bench -p aleph-metal --features metal --bench $b
done
```
