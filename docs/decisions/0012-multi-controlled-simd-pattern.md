# ADR 0012: Multi-controlled SIMD dispatch pattern (P1-08)

**Date:** 2026-05-28
**Status:** Accepted (P1-08).
**Supersedes:** None
**Extends:** ADR 0008 ([[0008-aos-avx512-beats-soa-simd]]), ADR 0010 ([[0010-2q-specialised-paths]]), ADR 0011 ([[0011-anti-diagonal-1q-classifier]])

## Context

The P1-03/05/06/07 ladder established the AoS + AVX-512 packed-complex
substrate and the matrix-shape-detection-at-kernel-layer pattern for 1q and
2q gates. After P1-07, all three Tier-1 workloads clear the ROADMAP § 7 ≤ 2×
Aer exit criterion. P1-08 extends the pattern to 3-qubit multi-controlled
gates: Toffoli (CCX, a.k.a. controlled-controlled-X) and CCZ
(controlled-controlled-Z).

Three architectural options were considered for routing CCX / CCZ to
specialised paths in `apply_3q`:

- **Option A — routing through existing 1q/2q kernels.** CCX ≡ Pauli-X with
  two inner controls. Route it through `apply_1q` (P1-05's anti-diagonal AVX-512
  path) with the two inner qubits promoted to external controls. CCZ ≡ CZ with
  one inner control, routing through `apply_2q` (P1-07's CZ diagonal path).
  Minimal new code; reuse tested kernels.
- **Option B — standalone 3q specialised kernels.** Write fresh
  `dispatch_toffoli` / `dispatch_ccz` with their own Tier A/B/C structure,
  isolated from the 1q/2q dispatch trees. Maximum freedom for 3q-specific
  tuning; explicit, auditable unsafe code.
- **Option C — hybrid.** Start with Option A, escalate to Option B if
  measurement shows routing overhead. Pragmatic but defers the design
  decision.

Option B was chosen to maintain the same isolation property that let P1-05,
P1-06, and P1-07 be audited and merged independently: each specialised kernel
owns its own SAFETY blocks and tier contracts. Routing through `apply_1q`
(Option A) would entangle the 3q control-masking with the 1q Tier-A/B tier
dispatch logic, complicating future tuning. Option C was rejected because the
"escalate if needed" gate never gets triggered in practice — bandwidth-bound
n=20 workloads show sub-percent wall-clock delta on diagonal gates (see ADR
0009 § performance shape), so the measurement bar for escalation would not be
cleared, and the Option-A code would survive untouched and rotting.

MCX with k ≥ 2 controls (Pauli-X with an arbitrary control count) is treated
separately: it routes through the existing `apply_1q` to P1-05's anti-diagonal
AVX-512 path, which already handles arbitrary `controls.len()`. No new kernel
is needed; verified by oracle test `multi_ctrl_mcx_k7_8q_oracle` and
`mcx_k{2,4,6}_n20` micro-benches.

## Decision

### Matrix-shape detectors

Add three new shape predicates to `kernels/mod.rs`, consistent with the
`is_diagonal_2x2` / `is_antidiagonal_2x2` / `is_cz_signature` family:

- `is_identity_8x8(m: &Matrix8)` — all 64 entries match the identity; used
  as a fast-return pre-check in `apply_3q`.
- `is_toffoli(m: &Matrix8)` — the 8×8 matrix has a `[0,0,0,0,0,0,0,1,…]`
  column-swap signature matching CCX. Tolerance: 1e-12 on each entry; NaN
  explicitly rejected via `is_finite` (ADR 0006 carryover).
- `is_ccz(m: &Matrix8)` — diagonal with the last entry equal to −1 and all
  others +1. Same tolerance and NaN guard.

### `apply_3q` dispatch prelude

`kernels::aos::apply_3q` becomes a prelude that runs detectors in order:

1. Identity → early return.
2. Toffoli → `dispatch_toffoli(state, targets, controls)`.
3. CCZ → `dispatch_ccz(state, targets, controls)`.
4. All other dense 8×8 → `apply_3q_generic` scalar fallback.

The same structure is mirrored in `kernels::soa::apply_3q`.

### Toffoli kernel — AoS tiers

Three SIMD tiers cover every legal `(t, c_lo, c_hi)` triple for the target
qubit `t` and the two inner controls (outer controls are threaded through
`expand_with_fixed` regardless of tier):

- **Tier A clean** (`1 << t ≥ LANES = 4`, `c_lo > t`): both inner controls
  are strictly above the target in qubit index. Each LANES-aligned block
  contains 4 complex pairs all sharing the same inner-control bits, so the
  check `(block & ctrl_mask) == ctrl_mask` resolves at the block level. The
  inner walk swaps two LANES-stride pairs via two paired zmm loads + two
  paired zmm stores — identical form to ADR 0010's CNOT Tier A, with a
  wider control mask.
- **Tier A outer-walk** (`1 << t ≥ LANES`, at least one `c < t`): one or
  both inner controls sit below the target. The outer loop still walks
  aligned blocks, but the per-block control test is a uniform lane mask built
  from all above-target bits, plus a scalar fallback for below-target
  sub-block lanes. Logically equivalent to Tier-A clean but the control-bit
  sieve operates at finer granularity.
- **Tier B.0 / B.1** (`t ∈ {0, 1}`, AoS): the target is below the LANES
  boundary — within-zmm permute paths. Tier B.0 uses
  `_mm512_permutex_pd<0x4E>` (adjacent-double swap, equivalent to target=0
  in P1-05's anti-diagonal path). Tier B.1 uses `_mm512_permutexvar_pd`
  with the cross-256 lane-pair index vector `(3,2,1,0,7,6,5,4)` in
  `_mm512_set_epi64` argument order (= lane vector `[4,5,6,7,0,1,2,3]`).
  Both Tier B paths perform a sign-conditional permute: lanes matching the
  inner-control mask are swapped; others are left untouched via
  `_mm512_mask_blend_pd`.
- **Tier C** (any qubit index out of range for the AVX-512 contract, or non-
  AVX-512 host): scalar `apply_toffoli_scalar`. Also the mandatory fallback
  for `state.len() < 2 * LANES`.

### CCZ kernel — AoS tiers

Two SIMD tiers cover the diagonal sign-flip (`|111⟩ → −|111⟩`):

- **Tier A clean** (`1 << mask_lo ≥ LANES = 4`, where `mask_lo` is the
  lowest of the three target qubits): all three control bits are above the
  LANES boundary, so the control check is at the block level.
  `_mm512_xor_pd(v, _mm512_set1_pd(-0.0))` is the sign-flip for matching
  blocks — 1 µop, latency 1, zero multiplies.
- **Tier A outer-walk** (`mask_lo < LANES`): one or more qubits are within
  the LANES block. Per-block `lane_mask` built from qubit positions, then
  `_mm512_mask_blend_pd(lane_mask, v, sign_flipped)` to flip only the
  matching lanes. Scalar fallback for sub-LANES states.

CZ in P1-07 (ADR 0010) established the Tier-A-only pattern for diagonal 2q
gates (Tier B/C value is marginal because the gate touches only a fraction of
state); CCZ follows the same reasoning — now 1/8 of state at most, so
bandwidth savings from Tier B/C are negligible.

### SoA mirroring

`kernels/soa.rs` gets a symmetric implementation. LANES_SOA = 8 f64s per
zmm (vs AoS LANES = 4 complex pairs). Tier B for Toffoli gains a third
sub-tier (`t = 2`), because SoA's f64-only LANES gives one additional
qubit of in-register reach:

- **SoA Tier B.0**: target = 0, `_mm512_permute_pd<0x55>` (adjacent swap).
- **SoA Tier B.1**: target = 1, `_mm512_permutexvar_pd` cross-pair within
  128-bit lane.
- **SoA Tier B.2**: target = 2, `_mm512_permutexvar_pd` cross-256 lane-pair.

Sub-LANES states (`re.len() < LANES_SOA = 8`) fall back to scalar in all SoA
tiers — same guard as ADR 0010 (Task 14 fix class, EPYC SIGSEGV on Bell
state n=2).

### MCX routing (no new kernel)

`MCX(k controls, target t)` with `k ≥ 2` is not a separate `apply_3q`
path. The backend routes it through `apply_1q(state, matrix_x, t,
all_controls)` directly. P1-05's anti-diagonal AVX-512 path in
`kernels::aos::apply_1q` already handles arbitrary `controls.len()` through
the `expand_with_fixed` outer-walk. Oracle test
`multi_ctrl_mcx_k7_8q_oracle` and `mcx_k{2,4,6}_n20` micro-benches confirm
this path covers k = 2 through 7 correctly.

## Consequences

**Positive:**

- Establishes "matrix-shape detector at the kernel-layer prelude" as the
  canonical pattern for future N-qubit specialised paths. P1-08 is the first
  extension beyond 2q; the pattern scales.
- Cleanly separates Toffoli / CCZ kernels from the generic 8×8 matmul, so
  each can be profiled, tuned, or replaced without touching the other.
- MCX-via-P1-05 routing is cost-free: no new kernel lines needed for the
  arbitrarily-controlled-X case, and the existing anti-diagonal SIMD path
  delivers full AVX-512 coverage for k = 2..7 controls.
- `_mm512_xor_pd` with `_mm512_set1_pd(-0.0)` sign-flip (CCZ Tier A) is
  1-µop latency-1 — zero multiplies. Documents the pattern for all future
  diagonal sign-mask operations.

**Negative:**

- Adds ~1 500 LOC of `unsafe` in `kernels/aos.rs` + `kernels/soa.rs`
  (kernels, dispatch, SAFETY blocks, unit tests). Maintenance budget grows.
- Tier-A clean and Tier-A outer-walk are near-clones for both Toffoli and
  CCZ. Code duplication accepted for clarity: each variant's SAFETY block
  documents a distinct contract, and unifying them would require a runtime
  branch inside the inner loop defeating the purpose. Could be deduplicated
  in a follow-up if SAFETY blocks can be unified.
- At Phase-1 workload scale (n = 20, 16 MiB state exceeding L3), the
  bandwidth ceiling (ADR 0008) severely limits wall-clock impact.
  `qft_n20` has zero Toffoli / CCZ gates; `random_brickwall_n20_d20` has
  zero; `grover_n20_iters5` has 5 CCZ instances. Real value is in micro-bench
  territory (L2-resident n = 14–15) and in future workloads (Shor, error
  correction) that are heavy on Toffoli.

**Deferred / open:**

- Generic n-control beyond CCX / CCZ (e.g. CCCCX as a dedicated 5q kernel)
  — covered by the MCX-via-P1-05 routing; no dedicated kernel needed.
- Potential dedup of Tier-A clean vs Tier-A outer-walk: defer until codegen
  analysis confirms the duplication is not load-bearing.
- Anti-diagonal 2q gates (iSWAP, sqrt-SWAP) — out of scope for P1-08; see
  ADR 0011 Open Questions.

## Lessons

1. **Codebase `LANES` constant is 4 amp-units, NOT 8 doubles.** The
   initial spec § 3.3 had Tier A require `t ≥ 3` (meaning `1 << t ≥ 8`)
   instead of `t ≥ 2` (meaning `1 << t ≥ 4 = LANES`). Always verify
   against existing kernel code (`grep "1usize << target) >=" kernels/aos.rs`)
   before designing new tier contracts. A single grep at spec-writing time
   would have saved two fix iterations on task T2.

2. **`_mm512_xor_pd` with `_mm512_set1_pd(-0.0)` is a 1-µop sign-flip;
   cheaper than `_mm512_mul_pd × -1.0`.** The multiply form uses a literal
   −1.0 constant that the compiler often fails to fold into `xorpd` when
   surrounded by FMA chains. Use `xorpd` + a `set1_pd(-0.0)` sign-mask
   constant for all future diagonal sign-mask operations; this form is
   unambiguous to the backend scheduler.

3. **Integer-only indexing-coverage tests catch bit-collision bugs that SIMD
   tests miss.** Per the P1-07 retrospective, a SIGSEGV on Bell state (n = 2
   < LANES_SOA = 8) only surfaced on the EPYC test run; the local aarch64
   host never triggered the underflow. Tasks T2 and T3 now require
   `classify_toffoli` / `ccz_pairs_unique` exhaustive integer-coverage tests
   (verifying `block | offsets[k] | j` pairwise-disjoint bits) before any
   SIMD kernel is written. This practice is now standard for every new 3q+
   kernel.

4. **MSB convention vs state-vector bit-index**: ADR 0004 documents the MSB
   matrix convention (qubit 0 is the most-significant bit in the gate matrix
   column ordering), but state-vector indexing uses bit-i = qubit-i (LSB-first
   amplitude index). The spec's initial basis-state mappings for CCX
   (`0b110 ↔ 0b111` for the swap) were incorrect under LSB indexing; the
   implementer caught and corrected them by cross-checking against the
   existing `apply_3q` scalar. Future spec authors: explicitly walk through
   one basis state by computing `|state_index| = Σ bit_i * 2^i` before
   writing test vectors.

5. **`_mm512_set_epi64` takes arguments in HIGH-to-LOW lane order.** Passing
   the desired lane vector `[4,5,6,7,0,1,2,3]` (index 0 = 4) naively
   yields `_mm512_set_epi64(4,5,6,7,0,1,2,3)`, which packs `3` into lane 0
   and `4` into lane 7 — the reverse. The correct call is
   `_mm512_set_epi64(3,2,1,0,7,6,5,4)`. ADR 0011 documented this trap for
   the Tier-B anti-diagonal cross-256 permute; P1-08 encountered it again
   for the Toffoli Tier B.1 index vector. Established policy: always write
   the desired lane vector in lane-order notation first, then reverse it for
   `_mm512_set_epi64`.

## Related

- ADR 0004 ([[0004-msb-qubit-ordering]]) — MSB convention; informs basis-state
  mappings in all kernel tests.
- ADR 0006 ([[0006-nan-handling]]) — NaN-guard pattern carried into
  `is_toffoli` / `is_ccz` / `is_identity_8x8`.
- ADR 0008 ([[0008-aos-avx512-beats-soa-simd]]) — bandwidth-bound regime at
  n = 20; explains why Toffoli/CCZ specialisation has minimal Tier-1 workload
  impact and why micro-bench at L2-resident n = 14 is the gating metric.
- ADR 0010 ([[0010-2q-specialised-paths]]) — 2q Tier A/B/C pattern and the
  `expand_with_fixed` outer-walk renormalisation idiom that all controlled 3q
  kernels inherit.
- ADR 0011 ([[0011-anti-diagonal-1q-classifier]]) — anti-diagonal 1q SIMD
  classification; MCX routing through this path is the P1-08 zero-new-kernel
  result.
- Spec: `docs/superpowers/specs/2026-05-28-p1-08-multi-controlled-design.md`.
- Plan: `docs/superpowers/plans/2026-05-28-p1-08-multi-controlled.md`.
