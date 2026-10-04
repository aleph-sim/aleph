# P6: multi-GPU state vector: model, prediction, AWS results

Issue #55 (refs). The code is `DistSvBackend::multi` and `NcclExchange` in `crates/aleph-cuda/src/dist/` (P6-01b),
on top of the P6-02 plan ([`p6-02-partitioning.md`](p6-02-partitioning.md)), the P6-03 router
([`p6-03-routing.md`](p6-03-routing.md)) and the P6-01a single-card executor ([`p6-01a-dist-gpu.md`](p6-01a-dist-gpu.md)).

This file has three parts:

1. The time model, and how it is calibrated.
2. The **prediction for 4× L4 (AWS g6.12xlarge)**, written down *before* the AWS run (spec §6).
3. The measured AWS columns. Their status is **pending**: they need the AWS session, which is launched only with the user's
   OK (`scripts/aws/p6-multi-gpu-session.sh`).

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

Status: **pending**. Run `scripts/aws/p6-multi-gpu-session.sh launch|setup|run|teardown`, only with the user's OK. The
script fills in `results/p6-aws/{topo.txt,oracle.log,bench.log}`. Paste the `xchg`, `strong` and `weak` CSV lines here
and compare them to §2 and §3.

## Communication volume (CPU, from P6-03)

The Naive vs Lookahead exchange counts and bytes for QFT-32, random-30, GHZ-32 and Grover-20 at g ∈ {2,3} are in
[`p6-03-routing.md`](p6-03-routing.md#measured-reduction). Lookahead cuts random-30 bytes 2.7–3.0× and Grover
4.7–7.1×; it is the router every run above uses.
