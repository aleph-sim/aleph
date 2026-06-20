# Phase 5.5 — Apple/Metal GPU: consolidated performance report

> Exit report for the **Phase 5.5 (Apple/Metal GPU)** milestone. Consolidates the
> per-task numbers in [`phase5.5.md`](phase5.5.md) (P5.5-04 fusion, P5.5-05 SV
> exit bench, P5.5-06 MPS scaffold) into one headline verdict. Perf is never
> gated in CI; these are reproducible local measurements.

## Verdict

**MET.** The Metal FP32 state-vector backend (`MetalSvBackend`) clears the
Phase-5.5 exit gate — **≥2× the same-Mac CPU state vector on ≥2 Tier-1 workloads
at n≈28** — on **all** Tier-1 structures, by a wide margin. The honest
apples-to-apples figure (GPU FP32 vs **CPU FP32**, same precision) is **4.67–6.10×
at n=28** (Grover 2.82× at n=20); the softer vs-FP64 ratios (4.78–6.27×) are
secondary, since the FP64 CPU moves 2× the bytes on a bandwidth-bound workload.
Its single-precision results are oracle-verified against the exact FP64 Qiskit-Aer
fixtures at 1e-5, and against a full (non-sampled) FP64-CPU amplitude compare at
n=26. The MPS-on-Metal backend (`MetalMpsBackend`) is **started**: a
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

The headline is **GPU FP32 vs CPU FP32** — both backends carry the same
single-precision state, so the ratio isolates the *backend*, not the precision.
The FP64 CPU column is shown alongside as a secondary number: it moves 2× the
bytes per amplitude on this bandwidth-bound workload, so the vs-FP64 ratios run
~3–4% higher and would flatter the GPU for a reason that has nothing to do with
the GPU.

| Workload | n  | GPU FP32 | CPU FP32 | **speedup (FP32, apples-to-apples)** | CPU FP64 | _vs FP64 (secondary)_ |
|----------|----|----------|----------|--------------------------------------|----------|-----------------------|
| QFT      | 28 | 3.735 s  | 22.80 s  | **6.10×** | 23.40 s | _6.27×_ |
| GHZ      | 28 | 956 ms   | 4.503 s  | **4.71×** | 4.746 s | _4.97×_ |
| random   | 28 | 9.128 s  | 42.59 s  | **4.67×** | 43.59 s | _4.78×_ |
| Grover   | 20 | 4.978 s  | 14.02 s  | **2.82×** | 14.49 s | _2.91×_ |

The advantage **grows** with n (GHZ 4.41×→4.71× FP32 over n=24→28): the larger the
state, the better the GPU amortizes its per-dispatch overhead, so this is a
genuine win rather than a unified-memory bandwidth ceiling at this scale. A
pre-timing self-consistency guard — a **full, non-sampled** GPU-vs-FP64-CPU
amplitude compare at **n=26** (every one of 2^26 amplitudes within 1e-5, not a
sampled stride) — passed for every workload, so the timed GPU work is correct on
*all* amplitudes, not merely fast. n=26 is the largest sweep cell whose full FP64
reference fits the 24 GB live-desktop budget; the kernels are size-invariant (n=28
is the same dispatch on a larger grid), so the full n=26 check gates the timed
n=28 path.

**Correctness.** `MetalSvBackend` is oracle-tested against the committed exact-FP64
Aer fixtures on Tier-1 (GHZ/QFT/Grover/random) at 1e-5, covering both the verbatim
`run` and fused `run_optimized` paths (P5.5-05 Part A) — plus the full-compare
scale guard above.

**No external Metal SV reference.** There is no apples-to-apples third-party
state-vector simulator on Metal to cross-check against: Apple's MPSGraph targets
ML tensor graphs, not arbitrary quantum-gate state-vector evolution, and the
established GPU simulators (cuQuantum, Qiskit-Aer-GPU) are CUDA-only. The
cross-checks are therefore the exact FP64 CPU reference (full compare, this
section) and the committed Aer FP64 fixtures (Part A oracle).

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
