# ADR 0010: 2-qubit gate specialised paths

**Date:** 2026-05-27
**Status:** Accepted (P1-07).
**Supersedes:** None
**Extends:** ADR 0008 ([[0008-aos-avx512-beats-soa-simd]]), ADR 0009 ([[0009-diagonal-fast-path]])

## Context

Phase 1 Stage 0 baseline (`docs/perf/phase1-vs-qiskit.md`) confirmed
QFT-20 is the only Tier-1 workload over the ROADMAP § 7 ≤ 2× Aer
target (2.39× Aer), with 39 % of its transpiled gates being scalar
`apply_2q` CNOTs. P1-06's diagonal-1q fast path (ADR 0009) lifted
µop efficiency on 1q diagonals but at n=20 ran into memory-bandwidth
ceilings — the 1q surface alone couldn't close the QFT gap.

P1-07 attacks the 2q surface directly. This ADR documents the
2q-specialisation tree it introduces, extending the matrix-detection
pattern established in ADR 0009 from 1q to 2q.

## Decision

`apply_2q` dispatch is matrix-based (extends ADR 0009): runtime
inspection via `classify_2q_permutation` + `is_diagonal_4x4` +
`is_cz_signature` routes each call to one of six paths.

### Dispatch tree

The prelude in `kernels::{aos,soa}::apply_2q` runs detection in this
order; first match wins:

1. **Identity** (`Perm2qKind::Identity`, π = `[0,1,2,3]`) — no-op return.
2. **CnotHi** (`Perm2qKind::CnotHi`, π = `[0,1,3,2]`) — control = `targets[0]`
   (MSB per ADR 0004), target = `targets[1]`. Swap-pair via packed
   load/store.
3. **CnotLo** (`Perm2qKind::CnotLo`, π = `[0,3,2,1]`) — control =
   `targets[1]` (LSB), target = `targets[0]`. Same kernel as CnotHi
   with arguments swapped.
4. **Swap** (`Perm2qKind::Swap`, π = `[0,2,1,3]`) — symmetric swap of
   `|01⟩` and `|10⟩` subspaces.
5. **CZ** (diagonal with `(1,1,1,-1)` signature, via `is_cz_signature`)
   — `vxorpd` sign-flip of the `|11⟩` subspace only.
6. **2q-diagonal** (any diagonal not matching CZ, via `is_diagonal_4x4`)
   — single-stream multiply by one of four sub-block multipliers
   keyed by the (target_hi, target_lo) bit pair.
7. **Generic dense 4×4** (fallback) — packed-complex AVX-512 4×4
   matmul when the SIMD contract holds, else scalar `apply_2q_dense_scalar`.

Other valid 4-element permutations (e.g. `X⊗I = π[1,0,3,2]`) and
phased permutations fall through to the generic path; only the four
canonical patterns above are recognised. Cost of misses is ~30 ns
per gate (12 `norm_sqr` calls + a few compares) — negligible against
even a single n=20 inner walk.

## Three-tier SIMD coverage (CNOT and SWAP, AoS)

The permutation paths require different inner-loop shapes depending
on how the targets sit relative to the AVX-512 LANE boundary
(`LANES = 4` for `Vec<Complex>` = 8 doubles = one zmm). Three tiers
cover every legal `(t_lo, t_hi)` pair:

* **Tier A** — `1 << min(targets) ≥ LANES`. Classic LANES-aligned
  block-walk: each zmm-load contains 4 contiguous complex pairs all
  with the same target-bit pattern, so a swap-pair is two paired
  loads + two paired stores.
* **Tier B** — `1 << min(targets) < LANES ≤ 1 << max(targets)`. The
  low target is sub-LANES, so within a single zmm-load the target=0
  and target=1 amplitudes interleave. Resolved by an in-register
  `vpermt2pd` / `vpermutexvar_pd` shuffle that re-pairs lanes before
  the swap.
* **Tier C** — both qubits in `{0, 1}` (AoS) or `{0, 1, 2}` (SoA).
  One quartet's worth of state fits in a single zmm; one permute
  per zmm-load reorganises lanes for the swap. SoA reaches one
  higher index (qubit 2) because its f64-only LANES = 8 vs AoS
  Complex-pair LANES = 4.

Tier C in SoA additionally requires `re.len() ≥ LANES_SOA = 8` —
sub-LANES states fall back to scalar (see the Task 14 follow-up
fix on commit `e8293fc`).

CZ and 2q-diagonal ship Tier A only. Their per-amp footprints are
already small (CZ: 1/4 of state, zero multiplies; diagonal: 1 mul
per amp) so the marginal value of Tier B/C SIMD is below the
implementation cost. Sub-LANES inputs route to scalar.

## Renormalisation outer-walk pattern (lesson)

The first pass at Tier A's controlled outer-walk threaded a single
`fixed_above` list spanning every fixed bit (targets + external
controls). Two bug iterations during Task 5 surfaced a subtler
contract: `expand_with_fixed` produces an index relative to its
fixed list, so the outer block index must be **rebuilt** for each
iteration, not incremented. The pattern that works (and is now
duplicated across CNOT/SWAP/CZ/diagonal/generic kernels):

```rust
for k in 0..outer_count {
    let outer = crate::kernels::expand_with_fixed(k, &fixed);
    inner_walk(outer);
}
```

NOT:

```rust
let mut block = expand_with_fixed(0, &fixed);
while block < len {
    inner_walk(block);
    block += outer_step;  // WRONG: skips holes in the control sieve
}
```

The incremental form silently works when `controls.is_empty()`
(holes don't exist) and silently corrupts state when they don't.
This is now load-bearing in every controlled 2q path.

## Performance shape (EPYC 8124P, Zen 4, single-thread, n=20)

Per-amp µop counts (theoretical, no bandwidth ceiling):

* CNOT / SWAP specialised: ~0.5 µops per amp on the half/quartile
  of state that's touched. Zero multiplies. ~16× faster per-amp
  than the generic-2q matmul.
* CZ specialised: 0.25 µops per amp (touches 1/4 of state) + 4×
  bandwidth reduction.
* 2q-diagonal: 1.25 µops per amp vs ~3.5 for generic SIMD —
  ~2.8× per-amp.
* Generic dense AVX-512: 56 µops per 16 amps ≈ 3.5 µops per amp,
  vs ~16 µops per amp for scalar generic-2q (~4.5× per-amp).

Wall-clock results (EPYC, post-P1-06 baseline → post-P1-07,
`docs/perf/phase1-vs-qiskit.md`):

* `qft_n20` AoS: 1133 ms → <FILL> ms (<FILL>× faster; vs Aer <FILL>×).
* `grover_n20_iters5` AoS: 79 033 ms → <FILL> ms.
* `random_brickwall_n20_d20` AoS: 842 ms → <FILL> ms.
* `p1_07/cnot_specialized` vs `cnot_via_generic` micro-bench:
  <FILL>× (BACKLOG AC target ≥ 5×).

Numbers will be substituted from Task 17 EPYC measurement.

## Why matrix detection (not gate-tag dispatch)

Same three options as ADR 0009; same answer. The kernel layer is
deliberately gate-tag-agnostic (P0-09) — it consumes `GateMatrix`,
not `Gate`. A gate-tag dispatcher in `backend.rs::apply_gate` would
(a) miss user-supplied permutation / diagonal `GenericUnitary(M4x4)`
matrices, and (b) require maintenance every time a new 2q gate is
added to `Gate`. The 30 ns matrix-detection cost catches both
intrinsic and user-supplied permutations, plus any output of the
P1-09 / P1-10 IR-fusion passes that happens to synthesise a
canonical permutation or diagonal.

## Why both CnotHi and CnotLo

`Gate::Cnot` always emits CnotHi per ADR 0004's MSB-control
convention. CnotLo catches user-supplied or fusion-output CNOTs
in the LSB-control orientation (matrix `[0,3,2,1]`). Cost: ~20
lines + one classifier test; preserves the gate-tag-agnostic
kernel layering and avoids forcing the dispatch prelude to
reorder `targets`.

## AoS Tier C vs SoA Tier C

The two backends diverge on Tier C reach:

* **AoS** (LANES = 4 complex pairs = one zmm of doubles) reaches
  Tier C only for `{t_lo, t_hi} ⊆ {0, 1}` — one full quartet
  per zmm.
* **SoA** (LANES_SOA = 8 f64s per stream = one zmm) reaches
  Tier C for `{t_lo, t_hi} ⊆ {0, 1, 2}` — two quartets per
  zmm-pair (re + im stream).

The extra qubit-2 coverage on SoA is the main place the
two-stream layout pays off in P1-07. SoA Tier C additionally
guards `re.len() ≥ LANES_SOA = 8` because the in-register
permute path is undefined for sub-LANES states; the dispatch
falls back to scalar there.

## Open questions deferred

* **Adjacent-pair-specific kernel** (`t_hi = t_lo + 1`). Estimated
  10–15 % potential gain on the 30–40 % of QFT-20 cx pairs that
  are adjacent. Held for a P1-07 follow-up if QFT-20 remains over
  the 2× Aer target after Stage 1 closes.
* **AVX2 path** for pre-Skylake-X / pre-Zen-4 hosts. No production
  consumer requesting it; scalar fallback covers correctness.
* **iSWAP / sqrt-SWAP specialised paths.** Hold until a Phase 2+
  workload exercises them — current Tier-1 set (GHZ / QFT / Grover /
  random) has zero iSWAPs.
* **ADR 0008 open Q#2 ("kill SoA").** Stays open through Phase 1
  closure. P1-07 is a tie-breaker data point — if SoA Tier C
  doesn't show its own daylight on QFT-20, the case for retiring
  SoA strengthens for Phase 2 planning.

## Related

* ADR 0008 ([[0008-aos-avx512-beats-soa-simd]]) — generic AoS +
  AVX-512 kernel; established the layout dominance this ADR
  inherits.
* ADR 0009 ([[0009-diagonal-fast-path]]) — diagonal-1q fast path;
  established the matrix-detection-at-kernel-layer pattern this
  ADR extends to 2q.
* Stage 0 report (`docs/perf/phase1-vs-qiskit.md`) — QFT-20
  bottleneck identification motivating P1-07's scope.
* Spec: `docs/superpowers/specs/2026-05-27-p1-07-2q-kernel-design.md`.
* Plan: `docs/superpowers/plans/2026-05-27-p1-07-2q-kernel.md`.
