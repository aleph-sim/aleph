# ADR 0008 — AoS + hand-written AVX-512 beats SoA + hand-written SIMD on QFT

**Status**: Accepted (2026-05-26)
**Issues**: [P1-01](../../BACKLOG.md), [P1-03](../../BACKLOG.md), [P1-04](../../BACKLOG.md)
**Builds on**: [ADR 0007](0007-soa-x86-perf-finding.md)
**Investigation**: PR #78 (SoA SIMD; not merged), PR #79 (AoS-AVX-512 experiment; not merged), `perf stat` + `objdump` on self-hosted EPYC 8124P (Zen 4)

## Context

ADR 0007 closed the P1-02 attempt with the finding that **SoA without
explicit SIMD loses to LLVM auto-vec'd AoS** on x86 (LLVM masked-loop
vectorisation extracts most of the work from the predicate-loop AoS
body; the SoA layout-only port hides that pattern from the
vectoriser). The ADR explicitly anticipated that SoA might still
win once paired with hand-written intrinsics.

P1-03 was scoped to test that hypothesis: hand-write AVX2 + AVX-512
intrinsics on top of the SoA layout from P1-01 and prove SoA finally
beats AoS on EPYC. The acceptance bar was `qft/n20/soa ≥ 2× faster
than P1-01 SoA on the EPYC bench server`.

## What we measured

**Side-by-side EPYC `qft/n20`** (commit `c1a7ce9` main vs PR #78
`05473c7` SoA-AVX-512 vs PR #79 `caa4321` AoS-AVX-512, same machine,
back-to-back runs):

| Backend / variant | `qft/n10` | `qft/n15` | `qft/n20` |
|---|---|---|---|
| P1-01 SoA (main) | 100.9 µs | 5.94 ms | 310 ms |
| naive AoS (LLVM auto-vec) | 87.0 µs | 5.16 ms | 305.7 ms |
| **PR #78 SoA + AVX-512 intrinsics** | 117 µs | 5.89 ms | **312 ms (flat)** |
| **PR #79 naive AoS + AVX-512 intrinsics** | **57.9 µs** | **2.95 ms** | **172 ms** |

PR #78's SoA-SIMD is **flat-to-regressed** vs the LLVM-auto-vec'd
scalar P1-01 SoA baseline. PR #79's AoS-AVX-512 is **1.80× faster
than P1-01 SoA on n20** and **2.01× faster on n15** — close to the
2× AC and the right direction.

## Root cause (perf-stat + objdump)

**Codegen on PR #78's SoA-AVX-512 path is correct.** `objdump
--disassemble=...apply_1q...` shows 16 `vmulpd zmm`, 8 `vfmsub231pd
zmm`, 8 `vfmadd231pd zmm`, 8 `vaddpd zmm`. The packed 8-lane
arithmetic is being emitted exactly as the intrinsics request. So
"the intrinsics aren't firing" is *not* the explanation.

**The bottleneck is load µops, not FLOPs.** `perf stat` over a 5-second
`--profile-time` window comparing the SoA-AVX-512 path vs the AoS
naive (LLVM `vmulpd xmm`) path:

| Per QFT-20 run | SoA P1-03 (AVX-512) | naive AoS (LLVM xmm) | SoA / AoS |
|---|---|---|---|
| Total flops retired (lane-counted) | 1.90 B | 1.91 B | ~1× |
| Mult flops | 0.69 B (zmm packed) | 1.08 B (xmm packed) | 0.64× |
| FMA flops | 0.76 B (`vfm*pd zmm`) | 0.005 B (none) | — |
| **Load µops** | **454 M** | **166 M** | **2.7× more in SoA** |
| Cycles | 0.9 B | 0.62 B | 1.45× |
| **Flops / cycle** | **2.11** | **3.08** | AoS is 1.46× more efficient |

SoA `apply_1q` retires **2.7× more load µops** for the same logical
work. The SoA `(re, im)` storage forces **four cache-line streams**
per gate (`re[i..]`, `re[j..]`, `im[i..]`, `im[j..]`), while AoS
`Vec<Complex>` forces only **two streams** (`state[i..]`,
`state[j..]`) because each `vmovupd xmm` load grabs both `re` and
`im` packed together. On the EPYC 8124P load/store unit, two streams
schedule better than four, even when the SoA path uses 8-lane zmm
arithmetic and the AoS path uses only 2-lane xmm arithmetic.

The PR #79 experiment confirms the diagnosis. AoS + AVX-512 keeps
the 2-stream pattern AND adds 8-lane packed-complex arithmetic via
`vfmaddsub_pd`:

* `z = vmovupd zmm [state]` — one load, 4 complex pairs.
* `z_swap = vpermilpd z, 0x55` — `(re, im)` → `(im, re)` lane-wise.
* `result = vfmaddsub(m_re_bcast, z, m_im_bcast × z_swap)` — alternating
  SUB/ADD across even/odd lanes produces `(re_out, im_out)` for each
  of the 4 packed complex.

Per inner iter: 2 loads + 2 permutes + 4 mul + 4 fmaddsub + 2 adds + 2
stores ≈ 16 µops for 4 complex pairs (4 µops/pair). Compare PR #78's
SoA-AVX-512: ~28 µops for 8 pairs across 4 streams (3.5 µops/pair
arithmetic-wise but the 4-stream load pressure dominates).

## Decision

1. **Drop the SoA-SIMD direction for the 1q kernel.** PR #78 closes
   without merge. The SoA layout's per-gate load-µop overhead is
   structural — not fixable by tweaking intrinsics.
2. **Ship AoS-AVX-512 as the actual P1-03 deliverable.** The
   dispatcher lives in `kernels::aos::apply_1q` (this PR). On hosts
   with AVX-512F and `target_bit ≥ LANES (= 4)` AND every control
   above target, the packed-complex kernel runs; otherwise the
   existing scalar body (which LLVM still auto-vec's to `vmulpd
   xmm`) runs as the fallback. ARM / WASM / RISC-V hit the scalar
   body unconditionally.
3. **Acceptance partial credit.** P1-03's stated AC was `qft/n20/soa
   ≥ 2× P1-01 SoA`. We hit `qft/n20/naive ≥ 1.80× P1-01 SoA` and
   `qft/n15/naive ≥ 2.01× P1-01 SoA`. The literal AC is missed
   because we ship the win on the *AoS* backend rather than the SoA
   one. We accept partial credit and revise the BACKLOG entry to
   match the actual deliverable; further gains come from algorithmic
   work (gate fusion P1-08) not more SIMD coverage.
4. **Keep P1-01 SoA in tree.** The SoA backend stays as
   `SoaSvBackend` for non-x86 hosts (where it's competitive with
   AoS) and for any workload where SoA's lower memory footprint per
   gate-sweep might matter. We do not flip the default backend in
   this ADR; that's a separate decision once we have more workload
   data.

## What's NOT in scope here

* **Gate fusion (P1-08).** CLAUDE.md's perf hierarchy puts
  IR-level algorithmic wins (#2) above SIMD (#4). The 1.80× factor
  we achieved is in the SIMD tier; gate fusion could give 2-5× on
  top by reducing total state-vector sweeps. That's where the next
  Phase-1 effort goes.
* **ARM NEON.** Apple silicon's SoA scalar auto-vec is already
  competitive with AoS; the NEON marginal win is smaller than what
  AVX-512 unlocked on x86. Deferred to a future ticket if any
  workload demonstrates need.
* **Specialised Pauli-X / diagonal-gate / 2q SIMD kernels** (P1-05,
  P1-06, P1-07) remain separate tickets. The AoS-AVX-512 generic
  2×2 kernel already covers all the listed 1q types; specialisations
  layer on top.

## Lessons (added to CLAUDE.md "Common Mistakes")

* **Don't assume the codegen is wrong when the bench is flat.**
  `objdump` first; if the intrinsics are there (`vmulpd zmm`,
  `vfm*pd zmm`), the issue is upstream — µop scheduling, memory
  bandwidth, cache pattern, gate-dispatch overhead. `perf stat -e
  ls_dispatch.ld_dispatch,...` distinguishes these.
* **Layout choice constrains SIMD upside on memory-bound workloads.**
  SoA's strided 4-stream load pattern eats the FLOPs/cycle win that
  8-lane zmm should give. ADR 0007 said "SoA without SIMD doesn't
  win"; this ADR closes the loop: **SoA with hand-written SIMD also
  doesn't win, for the same memory-pattern reason.**
* **The "right" perf hierarchy from CLAUDE.md is right.** We
  attempted SIMD (#4) before algorithmic wins (#2) on the assumption
  that SIMD would be the bigger unlock. The data says otherwise —
  algorithmic / cache-pattern wins matter more than SIMD coverage at
  the gate kernel granularity for n20+.

## References

* PR #78 forensic perf-stat output, EPYC 8124P, commit `05473c7`.
* PR #79 AoS-AVX-512 bench, EPYC 8124P, commit `caa4321`.
* AMD Zen 4 Software Optimization Guide (load/store unit, µop
  scheduling).
* Intel Intrinsics Guide entries for `_mm512_permute_pd`,
  `_mm512_fmaddsub_pd`.
* [ADR 0007](0007-soa-x86-perf-finding.md) — the predecessor finding.
