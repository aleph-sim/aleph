# Metal suite on Apple M5 (MacBook Air)

> Re-run of the Metal SV + MPS benches on an **M5 MacBook Air**, next to the earlier M3 Air / M4 Mini
> numbers ([`phase5.5.md`](phase5.5.md), [`metal.md`](metal.md), [`phase5.7.md`](phase5.7.md),
> [`phase5.8.md`](phase5.8.md)). Issue #520. Perf is never gated in CI; these are local measurements.

## Verdict

- **SV:** at n=28 the same-Mac GPU FP32 vs CPU FP32 speedup is **6.4–7.0×** (M4 Mini: 4.7–6.1×), and
  Grover n=20 is **8.7×** (M4: 2.8×). The ratio rose, but mostly because this fanless Air's CPU is slower
  over a long run than the M4 Mini's (CPU FP32 QFT-28 32.2 s vs 22.8 s). The M5 GPU itself is +20 % on
  GHZ-28 and 15–21 % slower than the M4 Mini GPU on QFT-28 and random-28.
- **MPS:** the GPU-MPS / CPU-MPS gap narrows slightly, from 16.1× to **14.3×** at n=16 χ=256 and from
  ~22× to **19.2×** at n=20 χ=256. The Phase 5.8 exit metric (GPU MPS ≥ CPU MPS in one regime) is
  **still unmet on the M5**. The wall is unchanged: the single-threadgroup per-block factorisation.
- **Correctness holds on the new GPU:** `cargo test -p aleph-metal --features metal --release` has 80
  tests passing, plus the 2 `--ignored` long tests, with no failures. That includes the 1e-5 Aer oracle
  fixtures and the pre-timing GPU-vs-FP64 guard in `sv_vs_cpu`.

## Machine and conditions

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
  an hour is in thermal steady state, not at its peak clocks. Wall-clock per bench:

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
`cpu_f32` = `Fp32SvBackend`, `cpu_f64` = `NaiveSvBackend`, 10 samples. Values are criterion medians.

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

The CPU column is the fanless-chassis effect: the random-28 CPU arm alone is 10 samples × ~70 s of
all-core load. Grover's ratio jump (2.8× → 8.7×) is mainly the CPU side (14.0 s → 21.6 s) with the GPU
2× faster (4.98 s → 2.47 s).

**Largest in-core n.** n=28 (1 GiB FP32) is the largest measured. 32 GB of RAM would hold the FP32
state up to n=31 (16 GiB), but `MTLDevice.maxBufferLength` and wired-memory limits were not probed,
so n=29–31 are **not verified** here.

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

The `n20_chi512` GPU cell was **stopped before it finished**. Criterion estimated 2,233 s for its
10 samples, and the laptop had to go back to battery. It is the one open cell of #520.

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

## Reproduce

```bash
cargo test  -p aleph-metal --features metal --release
cargo test  -p aleph-metal --features metal --release -- --ignored
for b in sv_fused sv_vs_cpu sv_readout mps_batched mps_vs_cpu; do
  cargo bench -p aleph-metal --features metal --bench $b
done
```
