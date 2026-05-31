# Phase 1 performance report: aleph vs Qiskit Aer (single thread)

> **This is the canonical Phase-1 closure report.** It supersedes the Stage-0
> snapshot in [`phase1-vs-qiskit.md`](./phase1-vs-qiskit.md) (2026-05-27,
> pre-P1-06/07, n=20 only). The numbers here cover the full
> {GHZ, QFT, Grover, random} × n ∈ {15, 20, 22, 25} matrix on the post-P1-13
> backend.

**Date:** 2026-05-31
**Host:** AMD EPYC 8124P (16 cores / 32 threads, Zen 4), 123 GiB RAM,
Ubuntu 26.04 LTS, kernel 7.0.0-15-generic
**aleph toolchain:** Rust 1.95.0 (2026-04-14), `RUSTFLAGS="-C target-cpu=native"`.
AVX-512 emission verified — `objdump` shows 114× `vmulpd …zmm` in the bench
binary (the AoS + AVX-512 packed-complex kernel, ADR 0008).
**Reference:** Qiskit 1.2.4, Aer 0.15.1, numpy 2.1.3, scipy 1.17.1, Python
3.12.13. `AerSimulator(method='statevector', max_parallel_threads=1)`.
**Pinning:** Aer under `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 taskset -c 0`; the aleph bench under `taskset -c 0`
with `RUSTFLAGS="-C target-cpu=native"`. Otherwise-idle runner.
**Reproducibility:** [`scripts/qiskit-baseline/README.md`](../../scripts/qiskit-baseline/README.md).
Both engines load the SAME committed `circuits/*.qasm`, so gate counts are
identical by construction. Raw Aer numbers: `scripts/qiskit-baseline/results-qiskit.json`.

`aleph` = `NaiveSvBackend` (AoS + AVX-512, the canonical fast x86 path
post-P1-03). All times are criterion medians (≤ n20: 50 samples / 10 s; n22:
10 / 20 s; grover-n22/n25: 10 / 30 s — see the RSD table). Aer medians are over
10/5/3 timed iterations at n ≤ 20 / 22 / 25 (1 warm-up). Lower is faster;
**`aleph/Aer < 1` means aleph is faster**. `time/gate` = aleph median ÷
post-transpile gate count. `theoretical` = `2^n × 16 B`.

-----

## Exit verdict (ROADMAP § 7)

> *"Single-thread within 2× of Qiskit Aer for QFT, Grover, random circuits at
> 25 qubits."*

**MET — for every Tier-1 algorithm, with margin.** At n=25:

| algorithm        | aleph/Aer @ n25 | verdict |
|------------------|:---------------:|:-------:|
| QFT              | **1.73×**       | ✅ ≤ 2× |
| Grover           | **0.67×**       | ✅ aleph 1.5× faster |
| random brickwall | **0.72×**       | ✅ aleph 1.4× faster |
| GHZ *(bonus)*    | **0.25×**       | ✅ aleph 4× faster |

QFT is the only family where aleph is slower than Aer, and it stays comfortably
under the 2× bar (1.73×). **No cell anywhere in the matrix exceeds 2× — no
follow-up issues filed** (see Known gaps). For reference, the Stage-0 snapshot
measured `qft_n20` at **2.39×** (over target); P1-06/07 (diagonal + 2q kernels)
brought the same cell to **1.22×**.

-----

## Per-algorithm headline tables

### GHZ — H + CNOT chain (n gates)

| n  | aleph (ms) | Aer (ms) | aleph/Aer | time/gate (ms) | peak RSS (MiB) | theo (MiB) | ≤2× |
|----|-----------:|---------:|:---------:|---------------:|---------------:|-----------:|:---:|
| 15 |      0.431 |    3.198 |   0.13×   |         0.029  |  —             |       0.5  | ✅ |
| 20 |     20.506 |  116.360 |   0.18×   |         1.025  |  —             |      16.0  | ✅ |
| 22 |    134.540 |  547.858 |   0.25×   |         6.115  |  —             |      64.0  | ✅ |
| 25 |  1 312.000 | 5 257.424 |  0.25×   |        52.480  |  514.6         |     512.0  | ✅ |

### QFT — textbook, no closing SWAPs

| n  | aleph (ms) | Aer (ms) | aleph/Aer | time/gate (ms) | peak RSS (MiB) | theo (MiB) | ≤2× |
|----|-----------:|---------:|:---------:|---------------:|---------------:|-----------:|:---:|
| 15 |      9.455 |   17.488 |   0.54×   |        0.0175  |  —             |       0.5  | ✅ |
| 20 |    564.050 |  460.575 |   1.22×   |        0.581   |  —             |      16.0  | ✅ |
| 22 |  3 764.900 | 2 231.787 |  1.69×   |        3.199   |  —             |      64.0  | ✅ |
| 25 | 39 683.000 | 22 877.305 | 1.73×   |       26.022   |  515.1         |     512.0  | ✅ |

### Grover — 1 marked state, 5 iterations

| n  | aleph (ms)   | Aer (ms)     | aleph/Aer | time/gate (ms) | peak RSS (MiB) | theo (MiB) | ≤2× |
|----|-------------:|-------------:|:---------:|---------------:|---------------:|-----------:|:---:|
| 15 |      571.330 |    2 641.294 |   0.22×   |        0.0120  |  —             |       0.5  | ✅ |
| 20 |   54 937.000 |  116 136.191 |   0.47×   |        0.571   |  —             |      16.0  | ✅ |
| 22 |  382 120.000 |  594 340.183 |   0.64×   |        3.183   |  —             |      64.0  | ✅ |
| 25 | 4 409 500.000 | 6 580 084.835 | 0.67×   |       27.454   |  ~515 †        |     512.0  | ✅ |

### Random brickwall — depth 20, deterministic angles

| n  | aleph (ms) | Aer (ms)  | aleph/Aer | time/gate (ms) | peak RSS (MiB) | theo (MiB) | ≤2× |
|----|-----------:|----------:|:---------:|---------------:|---------------:|-----------:|:---:|
| 15 |     13.449 |    41.152 |   0.33×   |        0.0182  |  —             |       0.5  | ✅ |
| 20 |    637.500 | 1 146.579 |   0.56×   |        0.644   |  —             |      16.0  | ✅ |
| 22 |  3 708.400 | 5 153.504 |   0.72×   |        3.402   |  —             |      64.0  | ✅ |
| 25 | 37 061.000 | 51 314.705 |  0.72×   |       29.888   |  515.2         |     512.0  | ✅ |

† Grover-n25 RSS not separately measured: peak RSS is set by the single state
vector (`2^25 × 16 B = 512 MiB`), which is identical across all n=25 families
regardless of gate count. The 1.2-hour grover-n25 oneshot was skipped as it
would reproduce the GHZ/QFT/random n=25 figure (~515 MiB).

-----

## Memory

Peak RSS measured via `/usr/bin/time -v` on a single-shot run (aleph: the
`oneshot` binary; Aer: `run.py --workloads <name>`), at the headline n=25.

| family | aleph RSS (MiB) | Aer RSS (MiB) | theoretical (MiB) |
|--------|----------------:|--------------:|------------------:|
| GHZ    |  514.6          |  607.3        |  512.0            |
| QFT    |  515.1          |  612.3        |  512.0            |
| random |  515.2          |  716.6        |  512.0            |

aleph sits within ~3 MiB of the 512 MiB theoretical floor — it holds exactly
one state vector plus negligible runtime. Aer carries 95–205 MiB of extra
buffer/runtime overhead on top of the same vector.

-----

## Trend — aleph/Aer across n

| family | n15  | n20  | n22  | n25  | shape |
|--------|:----:|:----:|:----:|:----:|-------|
| GHZ    | 0.13 | 0.18 | 0.25 | 0.25 | aleph 4–7× faster; plateaus |
| QFT    | 0.54 | 1.22 | 1.69 | 1.73 | gap grows with n, plateaus ~1.7× |
| Grover | 0.22 | 0.47 | 0.64 | 0.67 | aleph 1.5× faster; plateaus |
| random | 0.33 | 0.56 | 0.72 | 0.72 | aleph 1.4× faster; plateaus |

Every family's ratio **plateaus** by n=22–25 rather than diverging — once the
state vector exceeds cache (16 MiB at n=20), both engines are memory-bandwidth
bound and the ratio reflects the steady-state per-gate work, not a widening
algorithmic gap. QFT is the lone family where aleph trails Aer; the gap grows
n15→n22 then flattens at ~1.7×, consistent with Aer's controlled-phase handling
being better tuned for the dense QFT gate mix while aleph's generic diagonal/2q
kernels carry slightly more per-gate overhead.

-----

## Backend appendix — NaiveSv (AoS+AVX-512) vs SoA, n ≤ 20

| workload     | NaiveSv (ms) | SoA (ms) | SoA/Naive |
|--------------|-------------:|---------:|:---------:|
| ghz_n15      |       0.431  |   0.427  |  0.99×    |
| ghz_n20      |      20.506  |  20.531  |  1.00×    |
| qft_n15      |       9.455  |  16.628  |  1.76×    |
| qft_n20      |     564.050  | 973.520  |  1.73×    |
| grover_n15   |     571.330  | 1 849.600 | 3.24×    |
| grover_n20   |  54 937.000  | 125 300.000 | 2.28× |
| random_n15   |      13.449  |  31.883  |  2.37×    |
| random_n20   |     637.500  | 1 440.000 | 2.26×    |

SoA is 1.7–3.2× slower than AoS on every non-trivial workload (GHZ's all-CNOT
mix is the lone tie — both paths hit the same memory pattern). This confirms
ADR 0008: the AoS 2-stream packed-complex layout vectorises to AVX-512 cleanly
where SoA's 4-stream load pattern does not. SoA is not measured at n ≥ 22 (the
ratio is established and the EPYC time is better spent on the headline matrix).

-----

## Sampling / variance (no silent caps)

Large-n cells run for minutes per iteration, so both engines reduce sample
counts at n ≥ 22 — disclosed here so noisy cells are visible. **Every cell was
measured.** Aer relative stdev (RSD) is the timing-loop spread; aleph uses
criterion's adaptive sampling at the listed `(sample_size, measurement_time)`.

| cell                | Aer runs | Aer RSD | aleph criterion budget |
|---------------------|:--------:|:-------:|------------------------|
| *_n15, *_n20 (non-grover) | 10 | ≤ 4.2 % | 50 samples / 10 s |
| *_n22 (non-grover)  | 5        | ≤ 0.4 % | 10 samples / 20 s |
| *_n25 (non-grover)  | 3        | ≤ 0.6 % | 10 samples / 20 s |
| grover_n15, grover_n20 | 10    | ≤ 0.4 % | 10 samples / 20 s |
| grover_n22          | 5        | 0.01 %  | 10 samples / 30 s |
| grover_n25          | 3        | 0.02 %  | 10 samples / 30 s |

Aer RSD is uniformly low (worst case 4.2 % at the sub-4 ms `ghz_n15`; everything
≥ n20 is < 1 %). The criterion "Unable to complete N samples" warnings at large
n are expected — `sample_size` is a floor, so criterion extends the wall-clock
to honour 10 iterations (grover_n25 took ~12 h for its 10 samples; median
4 409.5 s/iter, stdev < 1 s).

-----

## Interpretation

- **Phase 1's single-thread goal is met across the board.** The three ROADMAP
  Tier-1 targets (QFT, Grover, random) are all ≤ 2× Aer at n=25, and aleph is
  *faster* than Aer on Grover, random, and GHZ.
- **The P1-06/07 kernels paid off where it mattered.** Stage-0's lone failure
  (`qft_n20` at 2.39×) is now 1.22×; the diagonal-gate and 2q CNOT/CZ AVX-512
  paths directly attacked the QFT controlled-phase bottleneck.
- **Everything is memory-bandwidth bound past n=20.** The per-gate times at
  n=25 (26–52 ms/gate) are exactly what a 512 MiB read-modify-write pass costs
  at this host's single-core bandwidth; the ratios plateau because both engines
  hit the same wall.
- **aleph is memory-lean.** Within 3 MiB of the theoretical floor vs Aer's
  95–205 MiB of overhead — relevant for the Phase-5 multi-GPU memory budget.

## Known gaps

- **None exceed 2× → no follow-up issues filed.** The exit criterion is fully
  met. QFT at 1.73× is the closest to the bar and the natural target if Phase-2
  pursues further single-thread QFT tuning, but it is not a Phase-1 miss.
- SoA at n ≥ 22 is unmeasured (intentional; see appendix).
- Multi-thread / multi-GPU performance is out of scope (Phase 2+).

## Reproducibility

```bash
# On the EPYC host (ssh root@195.154.249.85), otherwise-idle:
git checkout main
RUSTFLAGS="-C target-cpu=native" cargo build --release -p aleph-benches --bin oneshot

# 1. Aer, single-thread-pinned, full matrix -> results-qiskit.json
cd scripts/qiskit-baseline
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 .venv/bin/python run.py

# 2. aleph, same QASM, AVX-512 verified
cd ../..
ALEPH_BENCH_FULL_MATRIX=1 RUSTFLAGS="-C target-cpu=native" \
  taskset -c 0 cargo bench -p aleph-benches --bench qiskit_baseline -- --save-baseline phase1-final

# 3. peak RSS at n=25
/usr/bin/time -v taskset -c 0 ./target/release/oneshot scripts/qiskit-baseline/circuits/qft_n25.qasm 2>&1 \
  | grep 'Maximum resident'
```

Grover/random n=25 are the long poles (grover_n25 ≈ 1.8 h/iter for Aer, ~1.2 h
for aleph; the full aleph matrix took ~12 h end-to-end, dominated by
grover_n25's 10 criterion samples). The non-grover matrix completes in minutes.

-----

## Optimized pipeline (`run_optimized`) — the honest comparison

The headline tables above are **raw single-gate kernel throughput**: the run
path executed each parsed gate verbatim, with the P1-09…P1-13 IR optimization
passes (cancellation, DCE, 1q/2q fusion) **not** applied. Qiskit Aer fuses gates
by default, so those tables compare un-fused aleph against fused Aer. This
section re-measures with `run_optimized` (aleph's `Circuit::optimize()` —
`default_pipeline` — then simulate), the apples-to-apples comparison. Same EPYC
8124P, same pinning, same committed QASM, same Aer numbers
(`results-qiskit.json`). aleph = `NaiveSvBackend` (AoS + AVX-512, 236× `vmulpd
zmm` verified in the bench binary). Aer columns are reproduced from the raw
tables (Aer is unchanged).

`default_pipeline` gate-count reduction (per family, representative):
qft_n20 970→190, qft_n25 1525→300, random_n20 990→192, grover_n15 47805→12083,
ghz_n20 20→19.

> **Scope note:** to keep this re-measurement to ~1–2 h, the long-pole
> grover cells (n=20/22/25) were not re-timed — Grover already cleared the
> exit bar in the raw tables and gains further from fusion (grover_n15 below).
> Every QFT cell (the only family that was over 1×) and all GHZ/random cells
> were re-measured. The `speedup` column is optimized vs the raw aleph median
> from the tables above.

### Optimized headline — aleph vs Aer

| family | n  | aleph_opt (ms) | Aer (ms)   | aleph/Aer | speedup vs raw | ≤2× |
|--------|----|---------------:|-----------:|:---------:|:--------------:|:---:|
| GHZ    | 15 |          0.343 |      3.198 |   0.11×   |     1.27×      | ✅ |
| GHZ    | 20 |         18.287 |    116.360 |   0.16×   |     1.12×      | ✅ |
| GHZ    | 22 |        119.250 |    547.858 |   0.22×   |     1.13×      | ✅ |
| GHZ    | 25 |      1 161.900 |  5 257.424 |   0.22×   |     1.13×      | ✅ |
| QFT    | 15 |          4.536 |     17.488 |   0.26×   |     2.08×      | ✅ |
| QFT    | 20 |        241.100 |    460.575 |   0.52×   |     2.34×      | ✅ |
| QFT    | 22 |      1 361.900 |  2 231.787 |   0.61×   |     2.76×      | ✅ |
| QFT    | 25 |     13 911.000 | 22 877.305 | **0.61×** |   **2.85×**    | ✅ |
| random | 15 |          4.639 |     41.152 |   0.11×   |     2.90×      | ✅ |
| random | 20 |        200.880 |  1 146.579 |   0.18×   |     3.17×      | ✅ |
| random | 22 |      1 033.900 |  5 153.504 |   0.20×   |     3.58×      | ✅ |
| random | 25 |     10 283.000 | 51 314.705 |   0.20×   |     3.58×      | ✅ |
| Grover | 15 |        402.870 |  2 641.294 |   0.15×   |     1.42×      | ✅ |

Peak RSS at n=25 (optimized `oneshot`): GHZ 515.0 MiB, QFT 515.6 MiB,
random 515.8 MiB — unchanged from raw (within ~4 MiB of the 512 MiB theoretical
floor); fusion rewrites the IR, not the state vector.

### ROADMAP § 7 exit verdict (optimized) — MET with margin on every family

With the optimization pipeline enabled, **QFT is at or below 1× Aer at every n**
(0.26× / 0.52× / 0.61× / 0.61×). The lone raw-path laggard — `qft_n25` at 1.73×
*behind* Aer — becomes **0.61×, i.e. 1.6× ahead**. aleph is now faster than Aer
on all four Tier-1 families at n=25:

| algorithm | aleph/Aer @ n25 (optimized) | raw | verdict |
|-----------|:---------------------------:|:---:|:-------:|
| QFT       | **0.61×** | 1.73× | ✅ 1.6× ahead |
| Grover    | 0.15× (n15; n25 not re-timed, ≤0.67× raw) | 0.67× | ✅ ahead |
| random    | **0.20×** | 0.72× | ✅ 5× ahead |
| GHZ       | **0.22×** | 0.25× | ✅ 4.5× ahead |

### Why it was raw before

P1-14's bench called the raw `run` driver, which never invoked
`Circuit::optimize()` — so the passes shipped in P1-09…P1-13 had no effect on the
measured numbers. The fix (the `qft-parity` change) adds an opt-in
`run_optimized` driver, proves it semantics-preserving end-to-end
(`crates/aleph-backend/tests/run_optimized_oracle.rs`, fused ≡ raw within
1e-12), and points the bench + `oneshot` at it. No new kernels, passes, or gate
variants — the win was entirely from running optimization that already existed.

Reproduce: as the raw matrix above, but the aleph side now runs `run_optimized`
automatically (the bench and `oneshot` were switched). Filter to skip the long
grover poles, e.g.:
```
ALEPH_BENCH_FULL_MATRIX=1 RUSTFLAGS="-C target-cpu=native" taskset -c 0 \
  cargo bench -p aleph-benches --bench qiskit_baseline -- --save-baseline qft-parity \
  "(ghz|qft|random_brickwall)_n|grover_n1[05]"
```
