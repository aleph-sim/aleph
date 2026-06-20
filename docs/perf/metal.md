# Phase 5.5 — Apple/Metal GPU: consolidated performance report

> Exit report for the **Phase 5.5 (Apple/Metal GPU)** milestone. Consolidates the
> per-task numbers in [`phase5.5.md`](phase5.5.md) (P5.5-04 fusion, P5.5-05 SV
> exit bench, P5.5-06 MPS scaffold) into one headline verdict. Perf is never
> gated in CI; these are reproducible local measurements.

## Verdict

**MET.** The Metal FP32 state-vector backend (`MetalSvBackend`) clears the
Phase-5.5 exit gate — **≥2× the same-Mac CPU state vector on ≥2 Tier-1 workloads
at n≈28** — on **all** Tier-1 structures, by a wide margin (4.7–6.3× at n=28). Its
single-precision results are oracle-verified against the exact FP64 Qiskit-Aer
fixtures at 1e-5. The MPS-on-Metal backend (`MetalMpsBackend`) is **started**: a
correct nearest-neighbour scaffold whose dominant per-gate cost — the CPU SVD — is
measured and documented as the next lever.

## Hardware & honesty caveats

- **Box:** Apple **M4** base Mac Mini — 10-core CPU (4P+6E), 10-core integrated
  GPU, 24 GB unified memory. (The P5.5-04 fusion numbers predate the Mini and were
  taken on an M3 MacBook Air; noted inline below.)
- **Integrated GPU, unified memory.** Unlike a discrete GPU with dedicated VRAM,
  the M-series GPU draws from the *same* unified-memory bandwidth as the CPU
  cores. There is no separate high-bandwidth pool for the state vector — this
  shapes both the SV result (the GPU still wins, see below) and the MPS result
  (the CPU-SVD round-trip is serialization, not transfer).
- **Live desktop, not an idle box.** Every run was on the working Mini with
  `WindowServer` sharing the GPU (load ≈ 3 on 10 cores). CLAUDE.md's bench
  discipline wants an idle machine; that is not achievable here. All compared arms
  share identical contention, so the **ratios** are robust even though absolute
  times and criterion CIs are inflated. Sample sizes were reduced (10–20) for
  tractable wall-time at n=28. Treat ratios as order-of-magnitude.
- **FP32 ceiling.** The Metal backends are single-precision (the state vector is
  f32; gate matrices are materialized in f64 then narrowed). This is the project's
  GPU accuracy ceiling: amplitudes match the exact FP64 reference to ~1e-5, not the
  1e-10 the FP64 CPU backends hit. Fine for the Tier-1 workloads; not a drop-in for
  FP64-exact needs.

## 1. State vector — the exit gate (P5.5-05)

`MetalSvBackend` (FP32, fused `apply_kq`) vs the same-Mac CPU state vector, both
on the default-optimized IR pipeline, at the headline n=28 cell. Full sweep
(n∈{24,26,28} + Grover n=20) and method in [`phase5.5.md` § P5.5-05](phase5.5.md).

| Workload | n  | GPU (median) | CPU FP64 | **GPU speedup (vs FP64)** |
|----------|----|--------------|----------|---------------------------|
| QFT      | 28 | 3.735 s      | 23.40 s  | **6.27×** |
| GHZ      | 28 | 956 ms       | 4.746 s  | **4.97×** |
| random   | 28 | 9.128 s      | 43.59 s  | **4.78×** |
| Grover   | 20 | 4.978 s      | 14.49 s  | **2.91×** |

(GPU vs CPU **FP32** — the apples-to-apples precision match — is 4.7–6.1× over the
same cells.) The advantage **grows** with n (GHZ 4.41×→4.97× over n=24→28): the
larger the state, the better the GPU amortizes its per-dispatch overhead, so this
is a genuine win rather than a unified-memory bandwidth ceiling at this scale. A
pre-timing self-consistency guard (GPU vs FP64 CPU, sampled, 1e-5 at n=24) passed
for every workload, so the timed GPU work is correct, not merely fast.

**Correctness.** `MetalSvBackend` is oracle-tested against the committed exact-FP64
Aer fixtures on Tier-1 (GHZ/QFT/Grover/random) at 1e-5, covering both the verbatim
`run` and fused `run_optimized` paths (P5.5-05 Part A).

## 2. Gate fusion on the GPU (P5.5-04)

Fusing adjacent gates into dense `UnitaryKq` blocks cuts the per-gate CPU↔GPU
round-trip count. Measured on the **M3** MacBook Air (predates the Mini), fused
beats unfused on all nine Tier-1 cells: random brickwall up to **8.6×** (n=18),
QFT a steady **~3×**, GHZ **1.7–2.3×**, Grover-15 **6.0×**. The dominant unfused
cost is per-gate `wait_until_completed` synchronization, not arithmetic. Full
table in [`phase5.5.md` § P5.5-04](phase5.5.md).

## 3. MPS on Metal — scaffold start (P5.5-06)

`MetalMpsBackend` is a nearest-neighbour MPS scaffold: FP32 site tensors in
unified-memory buffers, GPU kernels for 1q apply / two-site contraction / 2q
gate-apply, and a host `faer` truncated SVD per NN 2q gate (the GPU has no SVD).

- **Correct.** Dense statevector matches the CPU `MpsBackend` **and** the exact
  FP64 `NaiveSvBackend` within 1e-5 on NN circuits (1q-only, GHZ n∈{3,5,8,10}, NN
  brickwall {4×6,6×8,8×6}).
- **CPU-SVD round-trip cost.** On an NN brickwall (n=12, d=24) the host SVD split
  is **94.5%** of per-gate tensor time — **17.3×** the GPU contract+apply. Unified
  memory makes the Θ read zero-copy, so the cost is **serialization** (a
  single-threaded CPU SVD blocking the GPU), not data transfer. This is the
  structural reason a unified-memory MPS-on-Metal scaffold cannot yet beat the CPU
  MPS, and it names the next lever: a GPU-resident or batched SVD.

## What's next

The SV exit gate is met; the MPS path has a correct scaffold and a measured
bottleneck. Beyond Phase 5.5, the open levers are (a) batched/whole-circuit
command-buffer dispatch to drop the remaining per-gate sync on the SV path, and
(b) a GPU/batched SVD to unblock the MPS path. Both are out of scope for the
Phase-5.5 milestone, which set out to stand up the Apple/Metal GPU backend and
clear the ≥2× SV exit bar — done.
