# ADR 0009: Diagonal-gate fast path detected at kernel layer

**Date:** 2026-05-27
**Status:** Accepted (P1-06).
**Context:** ADR 0008 ([[0008-aos-avx512-beats-soa-simd]]) established the
AoS + AVX-512 packed-complex kernel as the canonical fast x86 path for the
generic 1q kernel. The post-P1-03 baseline (Stage 0 report,
`docs/perf/phase1-vs-qiskit.md`) showed QFT-20 at 2.39× Aer — over the
ROADMAP § 7 ≤ 2× target. P1-06 attacks the QFT bottleneck specifically:
59% of QFT-20's transpiled gates are uncontrolled `p` (Phase) — pure diagonal.

## Decision

Add a diagonal 1q fast path to both `kernels::aos::apply_1q` and
`kernels::soa::apply_1q`, dispatched by matrix-runtime detection via the
`is_diagonal_2x2(m)` helper in `kernels::mod`. Threshold: both off-diagonal
entries have squared magnitude below `DIAGONAL_EPS_SQ = 1e-30` (i.e. magnitude
< ~3.16e-16, just above FP64 machine epsilon).

The diagonal kernel walks the state vector once with a single complex multiply
per amplitude — no cross-term arithmetic, no paired-index access. On AVX-512
the inner loop is ~5 µops per 4 complex pairs vs the generic 1q kernel's ~16.

## Why matrix detection, not gate-tag dispatch

P0-09 deliberately kept kernels gate-tag-agnostic — they consume `GateMatrix`,
not `Gate`. A gate-tag dispatcher in `backend.rs::apply_gate` would (a) miss
user-supplied diagonal `GenericUnitary(M2x2)` matrices, and (b) require
maintenance every time a new diagonal gate is added to `Gate`. Matrix
detection costs ~5 ns per gate (two `norm_sqr` calls + two compares) and
catches both intrinsic and user-supplied diagonals.

## Consequences — measured

EPYC 8124P (Zen 4), single-thread, `RUSTFLAGS="-C target-cpu=native"`, criterion
`--sample-size 30 --measurement-time 15`. Comparison against the Stage 0
baseline (`docs/perf/phase1-vs-qiskit.md`):

| Workload                              | Stage 0 (ms) | P1-06 (ms) | Δ        |
|---------------------------------------|-------------:|-----------:|----------|
| `qft_n20` `NaiveSvBackend`            |       1098.1 |     1133.0 | **+3.2 %** (regression) |
| `qft_n20` `SoaSvBackend`              |       2553.5 |     2252.4 | -11.8 %  |
| `grover_n20_iters5` `NaiveSvBackend`  |     92 111.3 |   79 033.2 | **-14.2 %** (large win) |
| `grover_n20_iters5` `SoaSvBackend`    |    211 096.4 |  201 129.1 | -4.7 %   |
| `random_brickwall_n20_d20` `NaiveSvBackend` |     821.6 |     842.2 | **+2.5 %** (regression) |
| `random_brickwall_n20_d20` `SoaSvBackend`   |    2 238.4 |    2 004.0 | -10.5 %  |

All measurements stable (MAD < 0.3 % of median). **Four wins, two regressions**
— the regressions are real, not noise.

## Forensic notes on the AoS QFT regression

`perf stat -e cycles,instructions,fp_ret_sse_avx_ops.mult_flops,ls_dispatch.ld_dispatch,l2_cache_req_stat.ic_dc_miss_in_l2,branch-misses`
over 30 s of the `naive_aos_avx512/qft_n20` bench:

| Counter                          | Baseline (Stage 0) |     P1-06 | Δ          |
|----------------------------------|-------------------:|----------:|------------|
| cycles                           |             85.8 B |    87.4 B | **+1.8 %** |
| instructions                     |            390.7 B |   339.6 B | -13.1 %    |
| FP mul flops                     |            238.5 B |   195.9 B | -17.8 %    |
| `ls_dispatch.ld_dispatch` (µops) |             50.1 B |    51.3 B | **+2.4 %** |
| `l2_cache_req_stat.ic_dc_miss_in_l2` |        454 M  |     245 M | **-46 %**  |
| branch-misses                    |             12.6 M |    13.3 M | +5.2 %     |

Interpretation:

- The diagonal kernel **is** more compute-efficient (13 % fewer instructions,
  18 % fewer FP muls) — design works as intended.
- The diagonal kernel **is** more cache-friendly (46 % fewer L2 misses) —
  the block-walk pattern beats the pair-walk pattern on QFT-20's 16 MiB state.
- But cycles still increased by 1.8 %, driven by 2.4 % more load µops.
  The likely cause is **reduced instruction-level parallelism**: the generic
  kernel's two-stream interleave (`z0`, `z1` per iter) feeds the CPU two
  independent dependency chains, while the diagonal kernel's single-stream
  walk creates a tighter dependency chain that the OoO engine can't accelerate
  as much. Net: compute wins eaten by ILP loss on the load side.
- Grover sees a 14 % win because its dominant 1q work is `T`/`Tdg` from
  Toffoli decomposition — pure diagonal, but importantly **not** part of a
  larger pipelined kernel where ILP matters. The diagonal path's compute win
  there isn't offset.
- SoA wins across the board because the SoA generic 1q kernel is already
  scalar (per ADR 0008's "AoS dominates on x86" finding), so the diagonal
  fast path has more room to improve over a slower baseline.

## Layer separation preserved

`apply_gate` stays matrix-based (`GateMatrix::M2x2/M4x4/M8x8`). Kernels remain
gate-tag-agnostic — `is_diagonal_2x2` is a pure matrix predicate, not a tag
inspection. User-supplied `GenericUnitary(M2x2)` matrices benefit automatically.

## Open follow-up (not blocking this PR)

The AoS QFT regression suggests a Stage 1 follow-up worth pursuing in a
separate small ticket:

- **Two-stream interleave for `apply_1q_diagonal_avx512`.** Process two
  independent 4-amp blocks per inner iter (8 amps total), matching the
  generic kernel's z0/z1 ILP pattern. Should recover the missing 2 % cycles
  on QFT/random while keeping the Grover win.

This is a tactical refinement, not an architectural one — the matrix-detection
dispatch and the kernel shape stay as designed.

## Related

- ADR 0007 ([[0007-soa-x86-perf-finding]]) — SoA-on-x86 perf finding.
- ADR 0008 ([[0008-aos-avx512-beats-soa-simd]]) — generic AoS + AVX-512 kernel.
- Stage 0 report (`docs/perf/phase1-vs-qiskit.md`) — established the QFT
  bottleneck.
