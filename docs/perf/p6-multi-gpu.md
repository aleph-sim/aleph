# P6: multi-GPU state vector: model, prediction, AWS results

Issue #55 (refs). The code is `DistSvBackend::multi` and `NcclExchange` in `crates/aleph-cuda/src/dist/` (P6-01b),
on top of the P6-02 plan ([`p6-02-partitioning.md`](p6-02-partitioning.md)), the P6-03 router
([`p6-03-routing.md`](p6-03-routing.md)) and the P6-01a single-card executor ([`p6-01a-dist-gpu.md`](p6-01a-dist-gpu.md)).

This file has three parts:

1. The time model, and how it is calibrated.
2. The **prediction for 4× L4 (AWS g6.12xlarge)**, written down *before* the AWS run (spec §6).
3. The measured AWS columns (§4), from one g6.12xlarge session on 2026-10-04
   (`scripts/aws/p6-multi-gpu-session.sh`).

## 1. Time model

For D GPUs with one rank per GPU (R = D, g = log2 D):

```
pred(D) = T_onecard(R=D) / D  +  bytes_moved_per_GPU / link_BW
```

- **`T_onecard(R=D)`** is the *same* plan run with all D ranks on one GPU (`LocalExchange`, P6-01a). On D GPUs, each GPU
  does exactly one rank's share of that work: the same specialised, per-rank-fused kernels on a 2^(n−g) slice. So
  dividing by D calibrates compute on the real instruction stream.
  - It also contains the on-card exchange copies, so it is an **upper bound** on compute.
- **`bytes_moved_per_GPU`** is `CommStats::amps_moved_per_rank × amp_bytes` (16 FP64, 8 FP32). `NcclExchange`
  sends and receives concurrently (full duplex), so this is the one-way volume.
- **`link_BW`** is the per-GPU NCCL p2p bandwidth. `dist_nccl_bench` measures it (`xchg,…,GBps=` line). Before the
  run it is assumed (§2).

**`T_onecard` cannot show host serialisation, and one was fixed before AWS.** `DistSvBackend` issues every rank's
program from one host thread, so devices overlap only if no instruction blocks the host. The phase-polynomial kernel
(QFT is about one `DiagonalPhase` per H after fusion) used to `synchronize` after launch. That would have made multi-GPU
QFT run device after device, and the bug cannot show up on one card. The sync is removed: cudarc's buffer drop already
orders the free after the kernel. `tests/dist_host_overlap.rs` pins this, and the inputs below were re-measured after
the fix (no change beyond ±2 %).

**A pass-count model was tried first and rejected.** That model was `pred = passes · t_pass + bytes / BW`.
- A distributed rank issues about 2× the kernel passes of the fused single-GPU circuit (random-28 d=10: 162–188 vs 76).
  Fusion stops at every exchange, the router inserts local swaps, and a 2-local diagonal becomes a dense `Unitary2q`
  (#529).
- But many of those passes are cheap (swaps, diagonals), while the single-GPU passes are heavy fused 3-qubit blocks.
- Checked on one card at R=2, the pass model said ~11.7 s against 7.49 s measured. It over-predicts by ~1.6×, so it is
  not used.

The inputs come from `cargo test --release -p aleph-cuda --features nccl --test dist_nccl_bench model_inputs -- --ignored --nocapture`.
They were run on the RTX 4000 SFF Ada box, which was idle (load 0.1–0.5, GPU util 2 %; the embeddings container was on
CPU).

## 2. Prediction: strong scaling, n = 28, 4× L4

Assumptions:
- **Compute.** The L4 is close to the RTX 4000 SFF Ada used for calibration: same sm_89 architecture, 58 vs 48 SMs,
  ~300 vs 280 GB/s. So the D=1 times are taken as RTX 4000 numbers, and the **speedups** are the part of the
  prediction that should transfer.
- **Link.** g6.12xlarge connects its L4s over PCIe with no NVLink. Two scenarios: **20 GB/s** per GPU (PCIe Gen4 x16
  P2P) and **8 GB/s** (P2P unavailable, staged through host memory).

| prec | circuit | D | exch | GB/GPU | 1 GPU (s) | T_onecard(R=D) (s) | compute/GPU (s) | pred @20 GB/s (s) | speedup @20 | pred @8 GB/s (s) | speedup @8 | efficiency @20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP64 | QFT | 2 | 2 | 2.15 | 2.91 | 2.46 | 1.23 | 1.34 | 2.18× | 1.50 | 1.94× | 109% |
| FP64 | QFT | 4 | 2 | 1.61 | 2.91 | 2.49 | 0.62 | 0.70 | 4.15× | 0.82 | 3.54× | 104% |
| FP64 | random_d10 | 2 | 10 | 10.74 | 5.50 | 7.50 | 3.75 | 4.29 | 1.28× | 5.09 | 1.08× | 64% |
| FP64 | random_d10 | 4 | 11 | 8.86 | 5.50 | 8.72 | 2.18 | 2.62 | 2.10× | 3.29 | 1.67× | 52% |
| FP64 | GHZ | 2 | 1 | 1.07 | 0.96 | 1.00 | 0.50 | 0.56 | 1.72× | 0.64 | 1.51× | 86% |
| FP64 | GHZ | 4 | 1 | 0.81 | 0.96 | 0.97 | 0.24 | 0.28 | 3.39× | 0.34 | 2.79× | 85% |
| FP32 | QFT | 2 | 2 | 1.07 | 2.28 | 2.05 | 1.03 | 1.08 | 2.11× | 1.16 | 1.96× | 105% |
| FP32 | QFT | 4 | 2 | 0.81 | 2.28 | 2.03 | 0.51 | 0.55 | 4.15× | 0.61 | 3.74× | 104% |
| FP32 | random_d10 | 2 | 10 | 5.37 | 1.59 | 2.96 | 1.48 | 1.75 | 0.91× | 2.15 | 0.74× | 45% |
| FP32 | random_d10 | 4 | 11 | 4.43 | 1.59 | 3.57 | 0.89 | 1.11 | 1.42× | 1.45 | 1.10× | 36% |
| FP32 | GHZ | 2 | 1 | 0.54 | 0.24 | 0.25 | 0.13 | 0.15 | 1.57× | 0.19 | 1.24× | 78% |
| FP32 | GHZ | 4 | 1 | 0.40 | 0.24 | 0.26 | 0.06 | 0.08 | 2.86× | 0.11 | 2.10× | 71% |

Column notes:
- "1 GPU" is the single-GPU `CudaSvBackend` on the `fuse_for_gpu` circuit, from the D=1 rows of `dist_nccl_scaling`.
- Efficiency is speedup / D.

**What the prediction says:**
- **QFT and GHZ scale near-linearly**, even at 8 GB/s. QFT is predicted *super*-linear (>100 %), for the reason
  P6-01a measured: the plan turns QFT's 14 trailing SWAPs (full-state passes on one GPU) into free relabels. Its 2
  exchanges move only 1–2 slices.
- **The random brickwall is the weak case, and its cause is compute, not the link.**
  - `T_onecard` grows 5.5 → 7.5 → 8.7 s (FP64) and 1.6 → 3.0 → 3.6 s (FP32) *before any link cost*. That is the 2×
    pass blow-up above.
  - FP32 suffers more because its fused 3q blocks are cheap, so the extra memory-bound passes weigh more.
  - So at D=2 FP32 is predicted *slower than one GPU*. The levers are #529 (keep 2-local diagonals diagonal) and #59
    (communication-aware fusion across exchanges). Link bandwidth is not the lever.
- None of this tests the #55 exit metric (>70 % at **8** GPUs on QFT-32). g6.12xlarge has 4 GPUs, and that item stays
  open for p4d.

## 3. Prediction: weak scaling (fixed 16 GiB per GPU), 4× L4

The slice is m = 30 FP64 / 31 FP32, so n = m + log2 D, on random d=10. Per-rank compute is extrapolated from
`T_onecard(R=D, n=28) / D` by the slice ratio, because memory-bound passes scale linearly with slice size. Link time
is the n-specific `amps_moved_per_rank` (from `model_inputs`) at 20 GB/s. This is rougher than §2.

| prec | n | D | compute/GPU (s) | link @20 (s) | pred (s) | D=1 in-core at m (s) | weak efficiency |
|---|---|---|---|---|---|---|---|
| FP64 | 30 | 1 | – | – | 23.9 (measured) | 23.9 | 100 % |
| FP64 | 31 | 2 | 30.0 | 4.3 | 34.3 | 23.9 | 70 % |
| FP64 | 32 | 4 | 34.9 | 6.4 | 41.3 | 23.9 | 58 % |
| FP32 | 31 | 1 | – | – | 13.6 (measured) | 13.6 | 100 % |
| FP32 | 32 | 2 | 23.7 | 4.3 | 28.0 | 13.6 | 49 % |
| FP32 | 33 | 4 | 28.6 | 6.4 | 35.0 | 13.6 | 39 % |

The single-GPU baseline at n = 32 FP64 / 33 FP32 does not fit any one GPU. It is `run_paged`, the state in pinned host
RAM (P5.10-02 measured ~18.5× the in-core cost at n=31), run by `weak_paged_baseline`. The prediction is that
multi-GPU in-core beats it by roughly an order of magnitude.

## 4. Measured on AWS g6.12xlarge

One session on 2026-10-04: 4× NVIDIA L4 (driver 595.91, sm_89), DLAMI base Ubuntu 22.04, NCCL 2.29.7 (the DLAMI copy
under `/usr/local/cuda-13.2/lib` loads ahead of the apt 2.32). Raw logs are in `results/p6-aws/` (git-ignored).

**Verdict.**
- **Correct on real multi-GPU.** All 26 oracle tests pass at D=2 and D=4 (FP64 1e-10, FP32 1e-5), under both NCCL SHM
  modes below.
- **The time model holds once the link is measured per exchange shape.** It is within 4 % at D=2. At D=4 it is within
  2–4 % when two-bit exchanges are priced at their own bandwidth.
- **The predicted speedups were not reached, and the cause is the link, not the code.** The box has no GPU P2P, so the
  link is 7.2 GB/s at best, below even the 8 GB/s pessimistic scenario of §2. Each L4 has only PCIe Gen4 **x8**, and
  every byte is staged through host memory.
- **Best results (memcpy SHM):** QFT reaches 2.93× FP64 and 3.30× FP32 at D=4 (73 % / 82 % efficiency). GHZ reaches
  2.1–2.2×. random_d10 is about 1× (1.23× FP64; FP32 0.87× at D=2), as §2 predicted: the pass blow-up (#529, #59)
  dominates.
- **None of this tests the #55 exit metric** (>70 % at 8 GPUs, QFT-32). That still needs NVLink hardware such as p4d.

### 4.1 Link: no P2P, so NCCL goes through host memory

- `nvidia-smi topo -m` shows every GPU pair as `NODE` (PCIe across host bridges, one NUMA node).
- `nvidia-smi topo -p2p r` shows `NS` for every pair: P2P is not supported.
- Each L4 links at Gen4 x8.
- NCCL picks `SHM/direct/direct` for every channel.

NCCL's default SHM copy is SM-driven. The CUDA-memcpy variant uses the copy engines and is 2.5× faster:

| `NCCL_SHM_USE_CUDA_MEMCPY` | D=2 bit 0 | D=4 bit 0 (0↔1) | D=4 bit 1 (0↔2) | D=4 bits {0,1} (3 peers) |
|---|---|---|---|---|
| unset (default) | 2.86 GB/s | 2.79 | 2.72 | 1.94 |
| `1` | **7.30** | **7.20** | **7.25** | **4.35** |

These are FP64 numbers at m=26 (`xchg_probe`): GB moved per GPU per second, one way, both directions running.
- The top-bit pair (0↔2) is as fast as 0↔1. The topology is symmetric.
- A two-bit exchange, which sends to three peers at once, runs at about 60 % of single-peer bandwidth. That is the
  whole D=4 model gap below.
- `NCCL_MIN_NCHANNELS`, `NCCL_BUFFSIZE`, `NCCL_PROTO=Simple` and `NCCL_SHM_MEMCPY_MODE` changed nothing (±3 %).
- The runbook now exports `NCCL_SHM_USE_CUDA_MEMCPY=1`.

**The memcpy mode exposed a deadlock, now fixed.**
- **Symptom.** With `NCCL_SHM_USE_CUDA_MEMCPY=1`, weak FP64 n=31 at D=2 hung: GPUs 0 and 1 at 100 %, no progress
  for 28 min.
- **Stacks (gdb).**
  - The host thread was blocked inside `cuMemcpyDtoDAsync`, issued by `copy_back` when the push buffer was full, and
    held a libcuda lock.
  - The NCCL proxy thread was waiting on that lock in `cuMemcpyDtoHAsync`.
  - The pending NCCL kernel was waiting on the proxy.
- **Why the default mode was safe.** Its proxy never calls CUDA.
- **Fix.** `NcclExchange::exchange` now drains the streams every 16 rounds and before it returns, so gate kernels never
  queue behind pending NCCL work.
  - A sync after every round also fixed it, but cost 30 % of D=4 bandwidth (7.2 → 4.85 GB/s): a sync lets no device
    pair run ahead of another.
  - The window keeps the bandwidth (7.20 GB/s after the fix).
- **Regression test.** `tests/dist_nccl_memcpy_shm.rs` uses 1 GiB slices, runs unfused and needs ≥2 GPUs.
  - On the instance with the fix it passed in 123 s (debug build).
  - With the fix reverted (on the instance only) it hit the 300 s watchdog and aborted.
  - A panic would not end the process, because teardown hangs behind the stuck threads, so the watchdog aborts.

### 4.2 Strong scaling, n = 28

The table uses `NCCL_SHM_USE_CUDA_MEMCPY=1` and the fixed exchange.
- "pred" is the §1 model with the measured single-bit link (`xchg` = 7.16 GB/s FP64, 7.09 GB/s FP32).
- "pred₂" also prices the D=4 exchanges at the two-bit bandwidth (×7.16/4.35). Every D=4 plan here swaps both global
  bits.
- "default SHM" is the speedup from the first run, without the memcpy flag (2.97 GB/s link).

| prec | circuit | D | 1 GPU (s) | T_onecard(R=D) (s) | pred (s) | pred₂ (s) | meas (s) | meas/pred(₂) | speedup | efficiency | §2 pred @8 GB/s | default SHM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP64 | QFT | 2 | 2.717 | 2.275 | 1.437 | – | 1.422 | 0.99 | **1.91×** | 96 % | 1.94× | 1.48× |
| FP64 | QFT | 4 | 2.717 | 2.307 | 0.802 | 0.947 | 0.926 | 0.98 | **2.93×** | 73 % | 3.54× | 2.02× |
| FP64 | random_d10 | 2 | 4.798 | 6.755 | 4.878 | – | 4.679 | 0.96 | **1.03×** | 52 % | 1.08× | 0.70× |
| FP64 | random_d10 | 4 | 4.798 | 8.008 | 3.240 | 4.039 | 3.891 | 0.96 | **1.23×** | 31 % | 1.67× | 0.76× |
| FP64 | GHZ | 2 | 0.848 | 0.876 | 0.588 | – | 0.580 | 0.99 | **1.46×** | 73 % | 1.51× | 1.07× |
| FP64 | GHZ | 4 | 0.848 | 0.855 | 0.326 | 0.399 | 0.390 | 0.98 | **2.17×** | 54 % | 2.79× | 1.41× |
| FP32 | QFT | 2 | 2.076 | 1.814 | 1.059 | – | 1.059 | 1.00 | **1.96×** | 98 % | 1.96× | 1.64× |
| FP32 | QFT | 4 | 2.076 | 1.812 | 0.566 | 0.640 | 0.630 | 0.98 | **3.30×** | 82 % | 3.74× | 2.47× |
| FP32 | random_d10 | 2 | 2.118 | 3.525 | 2.519 | – | 2.422 | 0.96 | **0.87×** | 44 % | 0.74× | 0.61× |
| FP32 | random_d10 | 4 | 2.118 | 4.099 | 1.649 | 2.052 | 1.994 | 0.97 | **1.06×** | 26 % | 1.10× | 0.67× |
| FP32 | GHZ | 2 | 0.385 | 0.393 | 0.272 | – | 0.269 | 0.99 | **1.43×** | 72 % | 1.24× | 1.03× |
| FP32 | GHZ | 4 | 0.385 | 0.384 | 0.153 | 0.189 | 0.186 | 0.98 | **2.07×** | 52 % | 2.10× | 1.33× |

**Reading.**
- **The calibration transferred.** On the L4, the D=1 and `T_onecard` times are 7–15 % below the RTX 4000 numbers
  (58 vs 48 SMs) and scale the same way. QFT `T_onecard(R=D)` is still below 1 GPU (2.3 vs 2.7 s FP64): the plan's
  SWAP→relabel win.
- **The model is right.**
  - With measured inputs it is within 4 % on every cell at D=2.
  - At D=4 the single-bit link under-predicts by 11–22 %. Priced at the two-bit bandwidth, every cell is within 2–4 %
    (all slightly faster than predicted).
  - So neither the host-serialisation risk (§1) nor the untested `rank_device` placement shows up.
- **The speedups track the 8 GB/s column of §2.** At D=2 they match it to within 0.03×. At D=4 they fall short by the
  two-bit penalty that §2 did not model.
  - QFT FP64 at D=4 is 2.93× against 3.54× predicted.
  - The 20 GB/s column was the P2P case, and this box cannot run it.
- **What would move the numbers.**
  - Hardware with P2P or NVLink.
  - On this box, #529 and #59 for random_d10. random_d10 FP32 at D=2 is slower than one GPU at any link speed (§2).
  - A router cost model that prefers single-bit exchanges when the link penalises fan-out.

### 4.3 Weak scaling (16 GiB per GPU, random d=10)

| prec | n | D | meas memcpy SHM (s) | meas default SHM (s) | weak efficiency (memcpy) | §3 pred @20 GB/s | 1 GPU paged at n (s) | multi-GPU vs paged |
|---|---|---|---|---|---|---|---|---|
| FP64 | 30 | 1 | 20.70 | 20.65 | 100 % | 23.9 | – | – |
| FP64 | 31 | 2 | 40.94 | 58.27 | 51 % | 34.3 (70 %) | > 2 900 (lower bound) | ≥ 71× |
| FP64 | 32 | 4 | 67.56 | 97.35 | 31 % | 41.3 (58 %) | not run | – |
| FP32 | 31 | 1 | 18.87 | 18.81 | 100 % | 13.6 | – | – |
| FP32 | 32 | 2 | 43.45 | 59.96 | 43 % | 28.0 (49 %) | not run | – |
| FP32 | 33 | 4 | 70.71 | 88.46 | 27 % | 35.0 (39 %) | not run | – |

Notes:
- Weak efficiency is the D=1 in-core time at m divided by the measured time.
- The paged baseline is `run_paged`, the whole state in pinned host RAM (181 GiB on this instance). Its tile is 28 for
  FP64 and 29 for FP32.
- The first run asked for tile 29/30. A brickwall gate can touch two high qubits, so tile+2 overflowed the device and
  the bench fixed the tile.

Weak efficiency trails §3 for the same reason as §2: the link is 7.2 GB/s (4.35 GB/s for two-bit swaps), not
20 GB/s.

For FP64, re-pricing §3 lands within 8–12 %, as close as §3's extrapolated compute allows:
- scale its compute to the L4 (×20.7/23.9);
- put the link term at the measured bandwidth.

| n | D | compute (s) | link | pred (s) | meas (s) |
|---|---|---|---|---|---|
| 31 | 2 | 26.0 | 86 GB / 7.2 GB/s | 37.9 | 40.9 |
| 32 | 4 | 30.2 | 128 GB / 4.35 GB/s | 59.6 | 67.6 |

FP32 is not re-priced. The L4 is slower than the RTX 4000 on the FP32 n=31 in-core baseline (18.9 vs 13.6 s), so §3's
FP32 compute does not transfer.

**Paged baseline: only a lower bound.**
- The bench timed `run_paged` with a warm-up run, so FP64 n=31 ran twice. After 97 min the timed run had still not
  finished. One run is therefore over 48.5 min, against 40.9 s on 2 GPUs: **multi-GPU in-core is ≥ 71× faster**.
- That is far beyond P5.10-02's ~18.5×-of-in-core paging cost on the RTX 4000. Each unfused gate streams the whole
  32 GiB host state through the L4's Gen4 **x8** link, about 770 gates.
- n=32 and n=33 would each take hours and would not change the verdict, so they were not run. The bench now times the
  paged run once, with no warm-up.

## Communication volume (CPU, from P6-03)

The Naive vs Lookahead exchange counts and bytes for QFT-32, random-30, GHZ-32 and Grover-20 at g ∈ {2,3} are in
[`p6-03-routing.md`](p6-03-routing.md#measured-reduction). Lookahead cuts random-30 bytes 2.7–3.0× and Grover
4.7–7.1×; it is the router every run above uses.
