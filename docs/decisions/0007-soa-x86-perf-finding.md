# ADR 0007 — SoA layout-only optimization on x86 loses to LLVM masked-loop auto-vec

**Status**: Accepted (2026-05-26)
**Issues**: [P1-01](../../BACKLOG.md), [P1-02](../../BACKLOG.md), [P1-03](../../BACKLOG.md)
**Investigation**: PR #76 (closed without merge), `perf stat` + `objdump`
on self-hosted EPYC (commit `4d960e8` + main `07d3724` back-to-back)

## Context

P1-02's premise was that bit-manipulation indexing (replacing a per-iteration
`if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask` test with a
branch-free outer loop over exactly the pairs that mutate) would yield
~2-3× on QFT-20 over the P1-01 SoA backend. The BACKLOG entry cited
QuEST as the reference implementation.

On the canonical bench server (self-hosted EPYC, x86_64), the
implementation regressed by ~30% vs P1-01 SoA instead of improving.
Same-load measurements:

| Version | `qft/n20/naive` | `qft/n20/soa` | SoA / naive |
|---|---|---|---|
| P1-01 (main `07d3724`) | 250 ms | **332 ms** | 1.33× slower |
| P1-02 (no closure, two-path if/else) | 224 ms | **428 ms** | **1.92× slower** |

## Root cause

`objdump` of the release binary on EPYC (Rust 1.95, `RUSTFLAGS=-C
target-cpu=native`, AVX-512-capable host) revealed:

* **P1-01 `apply_1q`**: 16 vmulsd (scalar matrix muls) **+ 8 `vporq`
  / 5 `vpsllvq` / 5 `vpmovzxdq`** (AVX-512 packed-quadword ops). LLVM
  auto-vectorizer recognized the flat predicate-loop shape as a
  **masked loop**: load 4-8 indices into a vector register, evaluate
  `i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask` per lane via
  packed bitwise ops, masked-store the matrix-multiply result.
* **P1-02 `apply_1q`**: 32 vmulsd (2× P1-01) + 48 vmovsd, **zero
  `vp*q` packed-quadword ops**. The two-path `if controls.is_empty()
  { … } else { … }` split, plus the `expand_with_fixed` helper call in
  the controlled path, blocked the vectorizer. Both paths lowered to
  fully scalar.
* **AoS `apply_1q` (`NaiveSvBackend`)**: 8 `vmulpd` (packed-double),
  `vextracti64x4`, `vmovddup`, `vaddsubpd`. LLVM auto-vectorized the
  `Vec<Complex>` layout to AVX-512 packed-double — 4 consecutive
  `Complex<f64>` (= 64 B = one 512-bit register) processed per
  iteration, ~250 ms on EPYC.

The structural finding: **a "branchy" loop that LLVM can mask-vectorize
beats a "branch-free" loop that LLVM cannot vectorize.** P1-02's
restructure (motivated by the textbook "branch-free is faster" intuition)
specifically broke the LLVM transformation that was extracting most of
P1-01's perf.

A secondary, deeper finding: **SoA-without-SIMD is structurally slower
than AoS-without-SIMD on x86.** AoS `Vec<Complex>` (16 B per element)
maps cleanly onto AVX-512 packed-double — one 64-byte vector load
pulls 4 consecutive amplitudes. SoA needs two separate stream loads
(re + im) to compose the same 4-amplitude vector. Without explicit
SIMD intrinsics (P1-03), AoS auto-vec wins; SoA wins back only when
P1-03's SIMD path beats AoS auto-vec by exploiting separated streams
for cross-amplitude operations (Pauli-X swap, diagonal phase
multiply).

## Decision

1. **Roll back P1-02.** PR #76 closed without merge. The layout-only
   bit-manip optimization was incorrect — it pessimizes the very
   transformation it was trying to improve.
2. **Defer P1-02 work into P1-03.** The bit-manip indexing pattern
   (nested block/pair, branch-free) is still the right shape for
   manual SIMD: each outer block contains unit-stride inner pairs
   that AVX2 / AVX-512 `vmovupd` consumes directly. P1-03 will
   implement it as part of the SIMD kernel rather than as a
   standalone optimization. Update BACKLOG P1-02 to reflect this.
3. **Expect P1-01 SoA layout to remain `~1.3× slower than AoS
   naive` on x86 until P1-03 lands.** This is structural, not a bug.
   The SoA layout is correct *as a prerequisite* for SIMD work; do
   not measure it as an independent perf win.
4. **On ARM (Apple silicon)** the picture is different: NEON-friendly
   auto-vec on SoA gets closer to parity with AoS. P1-02 measured
   ~parity with P1-01 on M-series (245 → 248 ms on QFT-20). This is
   informational; the canonical bench server is EPYC and that is the
   AC reference.

## Consequences

* **BACKLOG P1-02** rewritten to fold its motivation into P1-03; no
  standalone P1-02 implementation will land.
* **Future Phase-1 perf tickets** must validate against the EPYC
  bench server, not local M-series, before claiming a win. M-series
  auto-vec behavior diverges enough from x86 that local-dev numbers
  are not a reliable proxy.
* **When evaluating a "branch-free is faster" optimization on x86**,
  inspect `objdump --no-show-raw-insn` of the inner loop first.
  Look for `vp*q` (packed-quadword index ops) and `vmulpd` (packed
  double mul). Their absence in the new version is the smoking gun
  for "you broke the vectorizer."
* **Tooling note**: this investigation used `perf stat -e
  cycles,instructions,branches,branch-misses,L1-dcache-loads,
  L1-dcache-load-misses` + `objdump -d --no-show-raw-insn`. Document
  in `docs/perf/` as the canonical x86 perf-triage flow when a Phase-1
  optimization regresses.

## References

* PR #76 (closed) — full perf-stat numbers, codegen dumps, and the
  closure-vs-no-closure ablation.
* QuEST `statevec_unitary` — the reference C implementation that
  the P1-02 spec was modeled on. QuEST presumably wins because (a)
  they hand-write SIMD intrinsics from day one, not relying on
  auto-vec, and (b) their build flags exercise specific SIMD paths.
* LLVM loop-vectorize docs on masked vectorization:
  https://llvm.org/docs/Vectorizers.html
