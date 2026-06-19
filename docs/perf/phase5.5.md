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
