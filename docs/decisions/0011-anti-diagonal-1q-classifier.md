# 0011 — Anti-diagonal 1q classifier (P1-05)

## Status

Accepted, 2026-05-28.

## Context

P1-06 added a diagonal-1q fast path (ADR 0009). P1-07 added 2q
permutation + diagonal fast paths (ADR 0010). The remaining
non-arithmetic 1q gate-class is anti-diagonal (`[[0, a], [b, 0]]`),
covering Pauli-X (pure swap), Pauli-Y (swap + sign flip), and
user-supplied generic anti-diagonal unitaries.

## Decision

Add an `is_antidiagonal_2x2` classifier next to `is_diagonal_2x2` in
`kernels/mod.rs`, plus a `classify_1q_antidiag` Pauli-kind extractor
(`Perm1qKind::{X, YPos, YNeg}`). Dispatch routes Pauli-X to a pure-swap
kernel, Pauli-Y to a swap-with-sign-flip kernel, and generic anti-
diagonal to a complex-multiply-plus-swap kernel. Approach B from the
brainstorm (separate kernels per Pauli kind, 18 unsafe kernels total)
chosen over approach A (single generic with runtime branch) for
cleaner per-kind SIMD emission, and over approach C (X+Y only) for
completeness — user `GenericUnitary` anti-diagonals are rare but cheap
to cover.

## Three-tier coverage

Mirror of ADR 0010 (2q specialised paths):

- **Tier A**: `1 << target ≥ LANES`, controls > target → packed SIMD.
  LANES = 4 complex pairs per `__m512d` for AoS; LANES_SOA = 8 doubles
  per `__m512d` for SoA. Block-level outer walk + LANES-stride inner.
- **Tier B**: `target < log2(LANES)` → in-register lane permute via
  `_mm512_permutex_pd<0x4E>` (target=0), `_mm512_permutexvar_pd` with
  the cross-256 lane-pair index vector (target=1 for AoS, target ∈
  {1, 2} for SoA), or `_mm512_permute_pd<0x55>` (target=0 SoA).
- **Tier C**: any control < `log2(LANES)` OR non-AVX-512 host → scalar.

For SoA Tier B at `target ∈ {0, 1, 2}`, Y and generic anti-diag wrap
the scalar kernel (lane-by-lane sign-mask construction on split re/im
streams is bug-prone and the workload payoff is minimal — see Open
Questions).

## Correctness gotcha — anti-diag application direction

For `m = [[0, a], [b, 0]]` applying to `(z0, z1)`:

- `new_z0 = m[0][0]·z0 + m[0][1]·z1 = a · z1`
- `new_z1 = m[1][0]·z0 + m[1][1]·z1 = b · z0`

So `amps[i] ← a · amps[j]_old` and `amps[j] ← b · amps[i]_old` — the
upper-right entry `a` multiplies what was at the lower (j) index, and
the lower-left `b` multiplies what was at the upper (i) index. The
original brainstorm spec had this inverted; T2's oracle test
(`apply_1q_antidiag_scalar_matches_generic_phased`) caught the
inversion before merge. Future readers extending this dispatch (e.g.
to anti-diagonal 2q gates) should re-derive the row-multiply
formulation directly.

## Correctness gotcha — `_mm512_set_epi64` lane-index decoding

`_mm512_set_epi64(a, b, c, d, e, f, g, h)` packs `h` into lane 0 and
`a` into lane 7 (reversed argument order). When swapping complex pairs
across 256-bit lanes via `_mm512_permutexvar_pd`, each pair occupies
two adjacent lanes, so swapping `pair0` (lanes 0,1) with `pair2`
(lanes 4,5) requires the lane-order index vector `[4,5,6,7,0,1,2,3]`,
which in `_mm512_set_epi64` argument order is `(3, 2, 1, 0, 7, 6, 5, 4)`.
The initial T7 implementation used `(3, 2, 7, 6, 1, 0, 5, 4)` which is
`[4,5,0,1,6,7,2,3]` in lane order — that's a pair-shuffle WITHIN 256-bit
lanes, not a cross-lane swap. EPYC validation surfaced the bug for
target=1 X kernel; the single-line index correction repaired all three
Tier-B kernels.

## Correctness gotcha — Tier-B block-level control gate

Tier-B kernels iterate over LANES-aligned `block` addresses with the
gate `(block & ctrl_mask) == ctrl_mask`. Bits below `log2(LANES)` are
always 0 in `block` (LANES-aligned), so a control at qubit index
< `log2(LANES)` would alias to 0 in `block` and the gate would silently
no-op for amplitudes that DO have the control bit set within the LANES-
block. The original dispatch guard was `c > target`, which for target=0
allows `c=1` — within the LANES=4 in-block range. Backend proptest
`intrinsic_cnot_matches_external_control` with `c=1, t=0, preamble_q=1`
surfaced the bug on EPYC after T13 bumped X/Y weights in `arb_op`
(making the failing input more likely under shrinking). Fix: tighten
Tier-B dispatch to `controls.iter().all(|&c| c >= log2(LANES))` —
`c >= 2` for AoS, `c >= 3` for SoA. Kernel `debug_assert!` and SAFETY
block updated accordingly.

## NaN handling (ADR 0006 carryover)

`is_antidiagonal_2x2` explicitly rejects non-finite diagonal entries
via `is_finite` before the magnitude test. `classify_1q_antidiag` does
NOT need an explicit `is_finite` guard on off-diagonals: NaN
comparisons in the component-wise `close()` predicate yield `false`,
so NaN-poisoned off-diagonals fall through to the generic anti-diag
kernel which propagates NaN through its complex multiply. Test
`nan_off_diagonal_propagates_through_generic_antidiag_kernel` verifies.

## Performance shape

Micro-bench (L2-resident n=14, EPYC 8124P, single thread,
`RUSTFLAGS="-C target-cpu=native"`):

| Kernel       | Generic baseline | Specialised path | Speedup |
|--------------|------------------|-------------------|---------|
| AoS X        | 17.62 µs         | 5.22 µs           | 3.38×   |
| AoS Y        | 17.65 µs         | 5.00 µs           | 3.53×   |
| AoS antidiag | 17.62 µs         | 5.31 µs           | 3.32×   |

All three clear the BACKLOG AC of 3–10× speedup over the generic 2×2
kernel at L2-resident state. SoA micro-bench harness not added in
P1-05 (deferred to a P1-05-followup ticket if SoA backend remains
canonical after Phase-1 closure); SoA Tier-A AVX-512 correctness
validated on EPYC via T9's `apply_1q_x_soa_avx512_matches_scalar`,
`apply_1q_y_soa_avx512_pos_matches_scalar`, and
`apply_1q_antidiag_soa_avx512_matches_scalar` unit tests.

Workload (grover_n20_iters5, informational):

- Pre-P1-05 (post-P1-07 baseline): 58 491.74 ms
- Post-P1-05: 56 756.0 ms
- Delta: −2.97 % (−1 735.7 ms) — consistent with ADR 0008's bandwidth-bound prediction

Per ADR 0008 (bandwidth-bound regime), workload-level delta at n ≥ 20
is expected to be small (Grover's oracle/diffusion contains few X gates
at high-qubit targets); the micro AC is the gating metric.

## Open Questions

1. **Tier-B SoA Y and generic anti-diag wrap scalar.** Re-evaluate
   with profile data if very-low-target X/Y gates show up as hot in
   any real workload. The lane-by-lane sign-mask construction on split
   re/im streams was deemed bug-prone and the workload payoff
   minimal — but the call is reversible.
2. **Anti-diagonal 2q gates** (e.g. iSWAP `[[1,0,0,0],[0,0,i,0],
   [0,i,0,0],[0,0,0,1]]` or `[[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]`)
   — out of scope for P1-05; would extend ADR 0010's `classify_2q_*`
   family if needed.
3. **SoA micro-bench** — not added; the bench harness wraps
   `aleph_sv::kernels::aos::apply_1q` only. A symmetric SoA bench
   would re-validate the SoA Tier-A wall-clock at L2-resident sizes.
   Deferred to a P1-05-followup if SoA stays in tree post-Phase-1.

## Related

- ADR 0006 (NaN handling) — guard pattern carried over.
- ADR 0008 (AoS-AVX-512 beats SoA-SIMD) — bandwidth-bound regime at
  n=20, micro-bench at L2-resident n=14 explanation.
- ADR 0009 (diagonal fast path) — direct precedent for 1q classifier.
- ADR 0010 (2q specialised paths) — three-tier coverage pattern + the
  same `_mm512_set_pd` / `_mm512_set_epi64` lane-decoding traps.
