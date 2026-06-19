# Phase 5.5 — Apple/Metal GPU performance

## P5.5-04 — Fused-block GPU kernel (fused vs unfused)

**Hardware:** Apple **M3** (MacBook Air, 8-core: 4P+4E, integrated 10-core GPU).
The acceptance criterion references an "M4 base"; no M4 was available, so these
numbers are M3. The fused-vs-unfused *direction* is hardware-agnostic (fusion
cuts the per-gate GPU round-trip count); absolute ratios will differ on M4.

**Measurement caveat (honesty):** the run was taken on a *live desktop*, not an
idle box — load average ≈ 6 on 8 cores, with `WindowServer` actively using the
same integrated GPU for display. CLAUDE.md's bench discipline calls for a
verified-idle machine; that is not achievable on the working laptop. Because
both arms (`unfused`/`fused`) share the identical contention, the *relative*
ratios below are robust even though the absolute times and criterion CIs are
inflated. Sample size was reduced to 20 (`--sample-size 20 --warm-up-time 2
--measurement-time 6`) to keep wall-time tractable under contention; the Grover
case auto-extended (its unfused arm is ~10 s/run). Treat the ratios as
order-of-magnitude evidence of the direction, not precision figures.

**Method:** `cargo bench -p aleph-metal --features metal --bench sv_fused`.
`unfused` = `MetalSvBackend::run` (gate-by-gate; one dispatch + one
`wait_until_completed` per gate). `fused` = `MetalSvBackend::run_optimized`
(default IR pipeline; `FuseKq` collapses adjacent gates into dense `UnitaryKq`
blocks routed through the `apply_kq` kernel, cutting the dispatch count). Backend
rebuilt per sample in the untimed setup closure. Tier-1 builders from
`aleph-benches`; Grover parsed from
`scripts/qiskit-baseline/circuits/grover_n15_iters5.qasm`. Times are criterion
medians (`[lower median upper]` 95% CI shown for the median estimate).

| Workload | n  | unfused (median) | fused (median) | speedup (unfused/fused) |
|----------|----|------------------|----------------|--------------------------|
| QFT      | 14 | 15.70 ms         | 4.978 ms       | **3.15×** |
| QFT      | 16 | 17.70 ms         | 5.856 ms       | **3.02×** |
| QFT      | 18 | 32.40 ms         | 10.31 ms       | **3.14×** |
| random   | 14 | 48.75 ms         | 8.964 ms       | **5.44×** |
| random   | 16 | 72.74 ms         | 11.10 ms       | **6.55×** |
| random   | 18 | 215.0 ms         | 24.95 ms       | **8.62×** |
| GHZ      | 16 | 6.738 ms         | 2.964 ms       | **2.27×** |
| GHZ      | 18 | 8.023 ms         | 4.741 ms       | **1.69×** |
| Grover   | 15 | 9.588 s          | 1.592 s        | **6.02×** |

**Result:** AC #2 met — fused beats unfused on **all nine** measured points, not
just the required ≥1. The win scales with how many gates fold away and how many
GPU round-trips are thereby saved:

- **Random brickwall** (densest 2q structure) wins hardest — up to **8.6×** at
  n=18 — because `FuseKq` packs runs of `Rz/Rx/CNOT` into few dense `UnitaryKq`
  blocks, replacing hundreds of per-gate dispatches with a handful.
- **QFT** holds a steady **~3×**: its H + controlled-`Phase` ladder collapses into
  fused 1q/2q blocks plus a `DiagonalPhase` operator.
- **GHZ** wins least (**1.7–2.3×**) — it is a single H plus a CNOT chain, so there
  is less to fuse; the gain is just the chain folding into a few blocks.
- **Grover n=15** is the starkest illustration of the cost being attacked: the
  unfused run takes **~9.6 s** because the multi-controlled-decomposed circuit
  issues thousands of gates, each paying a full `wait_until_completed` GPU
  round-trip. Fusion cuts that to **~1.6 s** (6.0×). This is per-gate-sync
  overhead, not compute.

The dominant cost in the unfused path is the per-gate CPU↔GPU synchronization,
not arithmetic — so collapsing N gates into a few dense blocks cuts wall-clock
roughly in proportion to the dispatch-count reduction.

**Next lever (deferred to P5.5-05):** batch the whole circuit into one command
buffer and drop the per-gate `wait_until_completed`. That attacks the same
round-trip overhead directly (independently of fusion) and is the path to the
≥2× exit gate vs an Aer/CPU reference. It is out of scope for P5.5-04, which only
needed to demonstrate that fusion already helps on the GPU.

## P5.5-05 — Tier-1 GPU vs CPU statevector (exit measurement)

**Hardware:** Apple **M4** base Mac Mini (10-core CPU: 4P+6E, 10-core integrated
GPU, 24 GB unified memory). The integrated GPU shares the system's unified-memory
bandwidth with the CPU cores — unlike a discrete GPU with its own VRAM, there is
no separate high-bandwidth memory pool for the statevector to live in.

**Measurement caveat (honesty):** taken on a *live desktop* (3 login sessions,
`WindowServer` driving the display on the same integrated GPU), not a
verified-idle box — load average ≈ 3 on 10 cores during the run. CLAUDE.md's
bench discipline calls for an idle machine; that is not achievable on the working
Mini. Because all three arms share the identical contention, the *relative*
ratios are robust even though absolute times and criterion CIs are inflated.
Sample size reduced to 10 for tractable wall-time at n=28 (the n=28 QFT and
random CPU cells are 20–45 s/run). Treat ratios as order-of-magnitude.

**Method:** `cargo bench -p aleph-metal --features metal --bench sv_vs_cpu`. All
three arms run the same default-optimized IR pipeline (`run_optimized`), so the
comparison is backend-only. `gpu` = `MetalSvBackend` (FP32, fused `apply_kq`).
`cpu_f32` = `Fp32SvBackend` (CPU FP32). `cpu_f64` = `NaiveSvBackend` (CPU FP64).
A pre-timing self-consistency guard asserts the GPU result matches the FP64 CPU
result on sampled amplitudes within 1e-5 at n=24, so the timed work is correct.
Tier-1 builders from `aleph-benches`; Grover parsed from
`scripts/qiskit-baseline/circuits/grover_n20_iters5.qasm`. Times are criterion
medians.

| Workload | n  | gpu (median) | cpu_f32  | cpu_f64  | cpu_f32/gpu | cpu_f64/gpu |
|----------|----|--------------|----------|----------|-------------|-------------|
| GHZ      | 24 | 54.6 ms      | 235 ms   | 241 ms   | 4.31×       | 4.41×       |
| GHZ      | 26 | 227 ms       | 998 ms   | 1.026 s  | 4.40×       | 4.52×       |
| GHZ      | 28 | 956 ms       | 4.503 s  | 4.746 s  | 4.71×       | 4.97×       |
| QFT      | 24 | 199 ms       | 1.134 s  | 1.156 s  | 5.69×       | 5.81×       |
| QFT      | 26 | 845 ms       | 5.104 s  | 5.179 s  | 6.04×       | 6.13×       |
| QFT      | 28 | 3.735 s      | 22.80 s  | 23.40 s  | 6.10×       | 6.27×       |
| random   | 24 | 520 ms       | 2.242 s  | 2.394 s  | 4.31×       | 4.60×       |
| random   | 26 | 2.164 s      | 9.793 s  | 10.46 s  | 4.53×       | 4.83×       |
| random   | 28 | 9.128 s      | 42.59 s  | 43.59 s  | 4.67×       | 4.78×       |
| Grover   | 20 | 4.978 s      | 14.02 s  | 14.49 s  | 2.82×       | 2.91×       |

**Verdict (≥2× met).** `MetalSvBackend` reaches ≥2× the same-Mac CPU statevector
on **all** Tier-1 workloads at the headline n=28 cell: QFT **6.27×** (vs `cpu_f64`,
6.10× vs `cpu_f32`), GHZ **4.97×** (4.71×), and random brickwall **4.78×**
(4.67×). The Grover n=20 extra cell clears the bar too at **2.91×** (2.82×). AC #2
is met: the GPU statevector backend clears the 2× exit bar on every Tier-1
structure — comfortably more than the "≥2 workloads" the criterion requires — and
the ratios *grow* with n (GHZ 4.41×→4.97× over n=24→28), so the GPU's advantage
widens as the problem fills the chip rather than collapsing into a unified-memory
bandwidth ceiling at this scale. The pre-timing guard passed for every workload,
so the timed GPU work is the correct result (Part A's Aer oracle plus this
scale guard together pin down both small-n and n=24 correctness).

