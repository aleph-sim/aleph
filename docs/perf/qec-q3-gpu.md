# Phase Q3 — GPU decoder verdict

Phase Q3 set out to exploit aleph's CUDA depth with a **GPU decoder + massive on-device Monte-Carlo**
— the track's unique angle (ROADMAP §2.4): nobody else ships a co-designed GPU simulator *and* GPU
decoder. This report is the phase verdict, synthesising the three deliverables — the GPU Union-Find
decoder ([Q3-01]), the GPU belief-propagation decoder ([Q3-02]), and the end-to-end on-device
Monte-Carlo harness ([Q3-03]) — into one throughput/latency picture against the CPU decoders, with an
honest where-it-wins / where-it-loses statement.

## Exit metric — met

> *GPU decoder beats CPU MWPM/UF on decode throughput at large d / high shot counts; end-to-end
> (simulate noisy syndromes + decode) runs entirely on GPU.*

Both halves hold. The GPU Union-Find decoder beats CPU UF at every distance (3.5–18.9×) and exceeds
PyMatching's sparse-blossom core throughput at large `d`; the end-to-end threshold sweep runs entirely
on the device and is **11.8–107× the realistic CPU-MWPM threshold harness** at `d ≥ 7`. The honest
qualifier is that this is a **batch-throughput** win, not a single-shot **latency** win — quantified
below, and the reason real-time streaming (Q4) and FPGA (Q6) remain open.

## Hardware + methodology

- **GPU:** RTX 4000 SFF Ada Generation (sm_89, 20 GiB), CUDA 13.0 + NVRTC, box `openwebgui.splynx.com`.
- **CPU baselines:** same box, single-thread `Decoder::decode` (the comparable core; the Q1-05/Q2-03
  methodology). PyMatching figures are from [Q1-05] (Apple M4) — a *different machine*, cited for
  order-of-magnitude context only.
- All decoders consume the same surface-code memory-Z DEM (`p_data = p_meas`, `rounds = d`); shots are
  drawn from that DEM. Seed 2024, deterministic. Throughput is best-of-3 whole-batch decode; latency is
  best-of-5. One GPU thread per syndrome shot throughout (the design that keeps every GPU decoder
  bit/numerically identical to its CPU reference).

Sub-reports with full per-cell tables: [`qec-q3-gpu-uf.md`](qec-q3-gpu-uf.md) (Q3-01),
[`qec-q3-gpu-bp.md`](qec-q3-gpu-bp.md) (Q3-02), [`qec-q3-montecarlo.md`](qec-q3-montecarlo.md) (Q3-03).
Raw data under [`data/`](data/): `qec-q3-gpu-uf.csv`, `qec-q3-gpu-bp.csv`, `qec-q3-montecarlo.csv`,
`qec-q3-latency.csv`.

## Throughput (syndromes/second, single-thread CPU vs GPU, uniform p = 3 %)

All four columns are measured **on the same box, same methodology** (whole-batch decode, best of 3):
Union-Find from the Q3-01 bench (100 000-shot batch), belief propagation from the Q3-02 bench
(50 000-shot batch, fixed 64 iterations). UF is the workhorse; BP is the qLDPC-ready decoder — slower
in absolute terms but the only one here that handles non-graphlike checks.

| `d` | CPU UF | **GPU UF** | GPU UF speed-up | CPU BP | GPU BP | GPU BP speed-up |
|--:|--:|--:|--:|--:|--:|--:|
| 3 | 2 154 000 | **40 613 000** | 18.9× | 498 000 | 2 605 000 | 5.2× |
| 5 | 367 000 | **2 380 000** | 6.5× | 27 300 | 228 000 | 8.4× |
| 7 | 140 000 | **584 000** | 4.2× | 5 180 | 26 800 | 5.2× |
| 9 | 62 300 | **226 000** | 3.6× | 1 920 | 8 590 | 4.5× |
| 11 | 31 800 | **112 000** | 3.5× | 861 | 5 950 | 6.9× |

**Reading it.** GPU UF beats CPU UF at every `d` (3.5× at the large `d` that matters, up to 18.9× on
light syndromes); GPU BP beats CPU BP 4.5–8.4×. The UF speed-up *shrinks* with `d` because
one-thread-per-shot UF does more serial, divergent work as the defect count grows — the throughput
wall, see Limits.

**Versus MWPM.** On the same box the CPU-MWPM threshold harness (the accurate decoder Q0-05 used)
sustains only ~49 000/s at `d=7`, ~11 000/s at `d=9`, ~2 900/s at `d=11` and ~850/s at `d=13` (derived
from Q3-03's `mwpm_block` timings) — its dense-blossom cost explodes with the defect count. GPU UF's
112 000/s at `d=11` is **~39× that CPU-MWPM**, and also exceeds PyMatching 2.4.0's *sparse*-blossom
single-core throughput (~55 500/s at `d=11`, [Q1-05]) — though that PyMatching figure is from an Apple
M4, a different machine, so it is an order-of-magnitude comparison only.

## Per-shot latency (GPU UF decode, vs batch size)

The flip side of the throughput design. Per-shot GPU latency is `whole-batch-time / batch`; at batch 1
it is pure kernel-launch + transfer overhead.

| batch | d=5 GPU µs/shot | d=5 vs CPU | d=11 GPU µs/shot | d=11 vs CPU |
|--:|--:|--:|--:|--:|
| 1 | 310.3 | 121× slower | 2923.6 | 92× slower |
| 10 | 86.0 | 33× slower | 1095.2 | 34× slower |
| 100 | 17.4 | 6.8× slower | 272.2 | 8.6× slower |
| 1 000 | 1.94 | 0.75× (≈ parity) | 22.7 | 0.72× (≈ parity) |
| 10 000 | **0.31** | **8.3× faster** | **3.88** | **8.2× faster** |
| 100 000 | 0.42 | 6× faster | 8.92 | 3.6× faster |

(CPU per-shot: 2.57 µs at d=5, 31.75 µs at d=11. The batch-100 000 row regresses because the working
set exceeds the 2 GiB scratch budget and tiles — peak per-shot efficiency is at batch ≈ 10 000.)

**The crossover is ~1 000 shots.** Below it the GPU's per-shot latency is *worse* than the CPU — at a
single shot, 90–120× worse — because a kernel launch + the host↔device round-trip cost tens to
hundreds of microseconds regardless of work. Above ~10 000 shots the GPU is ~8× faster per shot. The
GPU decoder is a **batch-throughput engine, not a low-latency single-decode engine.**

## End-to-end on-device Monte-Carlo (Q3-03)

The headline capability: sample noisy syndromes + decode + score **entirely on the GPU**, only the
error count returning. Against the realistic CPU-MWPM threshold harness (the accurate decoder Q0-05
used), at matched-within-CI threshold:

| `d` | GPU-UF end-to-end (s) | CPU-MWPM harness (s) | speed-up |
|--:|--:|--:|--:|
| 7 | 0.17 | 2.03 | 11.8× |
| 9 | 0.44 | 9.04 | 20.6× |
| 11 | 0.89 | 34.7 | 38.8× |
| 13 | 1.10 | 117.5 | **107×** |

≥ 10× from `d=7` up, growing to 107× at `d=13` (CPU MWPM's blossom cost explodes with the defect
count; batch UF stays cheap). All 30 `(d,p)` cells agree with the CPU rate within CI. Against a
*same-decoder* CPU-UF harness the figure is a more modest 4.2× aggregate — the same one-thread-per-shot
UF wall.

## The differentiator and its limits

**What Phase Q3 delivers — the differentiator.** A correct (bit/numerically identical to the CPU
reference) GPU decoder family — Union-Find and belief propagation — plus a fused on-device Monte-Carlo
harness that runs the entire logical-error sweep without leaving the GPU. For the workload that
actually bottlenecks decoder research — *millions of shots per cell* — this is **10–100× the realistic
CPU harness**, on a single mid-range workstation GPU, and it scales with distance. Nobody else pairs a
GPU simulator with a GPU decoder this way.

**What it does not deliver — the limits, honestly stated.**

1. **Single-shot latency.** The GPU loses badly at batch 1 (90–120× slower than CPU): launch +
   transfer overhead dominates. A real fault-tolerant machine decodes *one syndrome per QEC round*
   under a hard ~1 µs budget — the opposite of batch. The GPU decoder does not address this; it is the
   remit of **streaming/sliding-window decoding (Phase Q4)** and ultimately a **dedicated FPGA/ASIC
   decoder (Phase Q6/Q7)** where the North Star (< 1 µs/round) lives.
2. **The one-thread-per-shot throughput wall.** Each GPU thread runs a serial, divergent decode, so
   the per-shot UF speed-up *falls* with distance (18.9× → 3.5×) and the same-decoder Monte-Carlo win
   is only 4.2×. A **block-cooperative / coalesced-scratch GPU UF kernel** is the open lever — it would
   lift both the standalone decoder throughput and the on-device harness, and is the obvious next
   optimisation.
3. **BP accuracy on surface codes.** Pure BP is degeneracy-limited; it reproduces the threshold but
   trails MWPM/UF in absolute logical rate. **BP+OSD (Phase Q5)** is the fix, and the GPU BP kernel is
   the substrate it will sit on.

## Verdict

Phase Q3's exit metric is **met**: the GPU decoders beat CPU MWPM/UF on throughput at large `d` and
high shot counts, and the end-to-end Monte-Carlo runs entirely on the GPU at 10–100× the realistic CPU
harness. The differentiator is real and quantified. The honest boundary is equally clear: this is a
**throughput** result, not a **latency** one — the per-round real-time budget that defines a deployable
decoder is untouched, and is exactly what Phases Q4 (streaming) and Q6 (FPGA) take on next.

## References

- N. Delfosse, N. H. Nickerson, **Almost-linear time decoding algorithm for topological codes**,
  Quantum 5, 595 (2021), [arXiv:1709.06218].
- O. Higgott & C. Gidney, **Sparse Blossom: correcting a million errors per core-second with MWPM**,
  [arXiv:2303.15933].
- J. Roffe et al., **Decoding across the quantum LDPC code landscape**, Phys. Rev. Research 2, 043423
  (2020) — BP+OSD.

[Q1-05]: qec-q1-mwpm.md
[Q3-01]: qec-q3-gpu-uf.md
[Q3-02]: qec-q3-gpu-bp.md
[Q3-03]: qec-q3-montecarlo.md
[arXiv:1709.06218]: https://arxiv.org/abs/1709.06218
[arXiv:2303.15933]: https://arxiv.org/abs/2303.15933
