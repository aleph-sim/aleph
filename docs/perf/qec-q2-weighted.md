# Q2-02 — Weighted Union-Find growth

Q2-01 shipped an **unweighted** Union-Find decoder: clusters grow isotropically, every edge at the
same rate. That ignores the information MWPM uses — that some errors are more likely than others —
so it sits a little below MWPM in accuracy. Q2-02 adds **weighted cluster growth**: each edge's
integer growth length is proportional to its matching weight `ln((1-p)/p)`, so clusters expand
cheaply along *likely* (low-weight) error paths first and slowly along unlikely ones. This recovers
most of MWPM's edge-weight awareness (Huang, Newman & Brown, [arXiv:2004.04693]) at near-identical
runtime.

To keep the round count ~equal to the unweighted decoder's, growth uses a **jump step**: each round
advances every odd cluster's boundary by the largest `δ` that completes the *next* edge anywhere,
rather than one unit at a time. Per-round cost stays one vertex scan plus a cheap edge pass — and
each round still completes ≥ 1 edge — so the absolute length magnitude does not affect runtime, only
the length *ratios* steer the result.

**Verdict — both acceptance criteria met:**

- **Accuracy improves over Q2-01 at every distance** (documented below). On a heterogeneous-weight
  model weighted UF **closes 80–98 % of the unweighted-UF→MWPM gap**; even on uniform noise it
  closes 40–96 %.
- **Runtime stays within 2× of Q2-01**: worst case **1.61×** (asymmetric, d=11), and essentially
  free (≤ 1.08×) on uniform noise.

## Method

Same phenomenological memory-Z surface code and DEM-sampling harness as Q1/Q2-01. All three
decoders — unweighted UF (Q2-01), weighted UF (Q2-02), MWPM (Q1) — decode the **same** sampled shots
(one seed) so every rate is apples-to-apples. 300 000 shots/cell, seed 2024, `rounds = d`.
Throughput is single-thread decode, best of three (the Q1-05 methodology).

Two noise models, because weighted growth only helps when edge weights actually differ:

- **Asymmetric** `p_data = 2 %, p_meas = 6 %` — measurement noisier than data, so vertical
  (time/measurement) and horizontal (data) edges carry distinctly different weights. The regime
  where weighting matters most.
- **Uniform** `p_data = p_meas = 3 %` — the standard model. Weights are *not* all equal even here
  (edge merging + boundary structure spread them), so weighting still helps, but less.

**Machine:** Apple M4 (10-core), macOS 26.5.1, rustc 1.96.0 `--release`. Reproduce:

```bash
cargo run --release -p aleph-qec --example qec_q2_weighted -- 0.02 0.06 300000 2024  # asymmetric
cargo run --release -p aleph-qec --example qec_q2_weighted -- 0.03 0.03 300000 2024  # uniform
```

Raw data: [`data/qec-q2-weighted-asym.csv`](data/qec-q2-weighted-asym.csv),
[`data/qec-q2-weighted-uniform.csv`](data/qec-q2-weighted-uniform.csv).

## Accuracy — asymmetric noise (p_data = 2 %, p_meas = 6 %)

Logical error rate (lower = better); "closes gap" is the fraction of the unweighted-UF→MWPM gap that
weighting recovers.

| d | UF (Q2-01) | **weighted UF** | MWPM | rel. improvement | closes gap | slowdown |
|--:|-----------:|----------------:|-----:|-----------------:|-----------:|---------:|
| 3 | 0.0772 | **0.0612** | 0.0609 | 20.7 % | 98 % | 1.23× |
| 5 | 0.1028 | **0.0710** | 0.0641 | 31.0 % | 82 % | 1.39× |
| 7 | 0.1303 | **0.0761** | 0.0636 | 41.6 % | 81 % | 1.48× |
| 9 | 0.1607 | **0.0834** | 0.0651 | 48.1 % | 81 % | 1.55× |
| 11 | 0.1922 | **0.0906** | 0.0654 | 52.9 % | 80 % | 1.61× |

The unweighted decoder degrades badly as `d` grows under asymmetric noise (it cannot tell the cheap
measurement edges from the expensive data edges); weighting fixes most of that, tracking MWPM within
~0.025 absolute at d=11 where unweighted UF is 0.127 worse. All differences are far outside the 95 %
CI (≈ 0.001).

## Accuracy — uniform noise (p_data = p_meas = 3 %)

| d | UF (Q2-01) | **weighted UF** | MWPM | rel. improvement | closes gap | slowdown |
|--:|-----------:|----------------:|-----:|-----------------:|-----------:|---------:|
| 3 | 0.1125 | **0.0993** | 0.0987 | 11.7 % | 96 % | 0.93× |
| 5 | 0.1269 | **0.1182** | 0.1076 | 6.9 % | 45 % | 1.01× |
| 7 | 0.1511 | **0.1299** | 0.1095 | 14.1 % | 51 % | 1.01× |
| 9 | 0.1674 | **0.1424** | 0.1109 | 14.9 % | 44 % | 1.06× |
| 11 | 0.1856 | **0.1566** | 0.1139 | 15.6 % | 40 % | 1.08× |

Smaller gains (the weight spread is smaller), but still a consistent 7–16 % relative improvement and
essentially **no runtime cost** — at uniform p the jump step actually makes some cells *faster* than
unweighted unit growth.

## Runtime budget

The Q2-02 cap is **≤ 2× Q2-01**. Measured slowdowns: **1.23–1.61×** (asymmetric), **0.93–1.08×**
(uniform). Comfortably inside budget. The cost comes from the extra delta-computation pass per round;
the jump step keeps the round count from blowing up with the (larger) weighted edge lengths.

## What this validates

- Weighted growth recovers most of MWPM's accuracy advantage over unweighted UF — **80–98 % of the
  gap on heterogeneous noise** — while remaining a near-linear, integer-only, fixed-array decoder
  (the hardware-friendly properties Q2-01 established are untouched: same Union-Find, same peeling).
- The accuracy/runtime trade-off is favourable: a ≤ 1.6× decode-time cost buys a 20–53 % logical-rate
  reduction where it matters most.

The full speed/accuracy **Pareto** across decoders and noise strengths — UF vs weighted UF vs MWPM,
errors/second and logical error rate, with the "when to use which / which goes to hardware"
recommendation — is the Q2-03 report (`docs/perf/qec-q2-unionfind.md`).

## References

- S. Huang, M. Newman, K. R. Brown, **Fault-tolerant weighted union-find decoding on the toric
  code**, Phys. Rev. A 102, 012419 (2020), [arXiv:2004.04693].
- N. Delfosse, N. H. Nickerson, **Almost-linear time decoding algorithm for topological codes**,
  Quantum 5, 595 (2021), [arXiv:1709.06218].

[`MwpmDecoder`]: ../../crates/aleph-qec/src/mwpm.rs
[`UnionFindDecoder`]: ../../crates/aleph-qec/src/union_find.rs
[arXiv:2004.04693]: https://arxiv.org/abs/2004.04693
[arXiv:1709.06218]: https://arxiv.org/abs/1709.06218
