# P1-05 — Specialised Pauli-X/Y (anti-diagonal 1q) kernel (design)

> **Phase 1, Stage 1, ticket 3.** After P1-06 (diagonal 1q) and P1-07
> (2q kernel + CNOT/CZ/SWAP). P1-07 brought all three Tier-1 workloads
> under ROADMAP § 7's `≤ 2× Aer` exit (QFT-20 = 1.30× Aer, Grover -25.9 %,
> random -21 %), so P1-05 is no longer load-bearing for the phase exit.
> It ships for completeness and to pull Grover's oracle/diffusion path
> off the generic kernel.

## 1. Goal

Add a specialised anti-diagonal 1-qubit kernel family to the AoS and SoA
backends. Catches `Gate::X`, `Gate::Y`, and any user-supplied
`GenericUnitary(M2x2)` whose matrix is anti-diagonal (`[[0, a], [b, 0]]`).
Pauli-X is a pure amplitude swap (zero arithmetic). Pauli-Y is swap +
sign-flip via `vxorpd`. Generic anti-diagonal is one complex multiply
per amplitude + swap.

Classifier-driven dispatch mirrors P1-06's diagonal path and ADR 0010's
2q permutation path — anti-diagonal becomes the third recognised matrix
class.

## 2. Non-goals

- **No Z fast path.** Already covered by P1-06's diagonal kernel
  (Z = `diag(1, -1)`). The BACKLOG entry listing "X, Y, Z specialised"
  is amended in § 9.1.
- **No multi-controlled X (CCX/MCX).** That is P1-08.
- **No SoA backend removal.** Open question deferred to Phase 1 closure,
  as in P1-06/07. P1-05 ships symmetric kernels in both backends.
- **No new public API.** Backend trait, `GateInstance`, `GateMatrix` —
  unchanged. Detection happens at the kernel layer via matrix inspection.
- **No removal of intrinsic-gate paths.** Detection sits inside
  `apply_1q`, transparent to callers.

## 3. Deliverables

Single squash-merge PR titled `[P1-05] Specialised Pauli-X/Y (anti-diagonal 1q) kernel`:

```
crates/aleph-sv/src/kernels/
├── mod.rs   # + is_antidiagonal_2x2, Perm1qKind, classify_1q_antidiag
├── aos.rs   # + 9 unsafe kernels (X/Y/generic × Tier A/B/C) + dispatch
└── soa.rs   # + 9 unsafe kernels (X/Y/generic × Tier A/B/C) + dispatch
crates/aleph-sv/benches/
└── pauli_xy.rs   # NEW micro-bench
docs/decisions/0011-anti-diagonal-1q-classifier.md   # NEW ADR
BACKLOG.md            # § 15.1 amendment
docs/perf/phase1-vs-qiskit.md   # "P1-05 update" section
```

No new crates, no dependency changes.

## 4. Architecture

### 4.1 Dispatch tree (AoS, mirrored in SoA)

```
apply_1q(amps, target, controls, m):
    1. if is_diagonal_2x2(m):
           → apply_1q_diagonal_{avx512, scalar}    (P1-06, unchanged)

    2. if is_antidiagonal_2x2(m):                   ← NEW
           match classify_1q_antidiag(m):
               Some(X)     → apply_1q_x_{tier_a, tier_b, tier_c}
               Some(YPos)  → apply_1q_y_{tier_a, tier_b, tier_c}   (phase = +i,-i)
               Some(YNeg)  → apply_1q_y_{tier_a, tier_b, tier_c}   (phase = -i,+i)
               None        → apply_1q_antidiag_{tier_a, tier_b, tier_c}

    3. (fallthrough) generic 2×2 kernel             (unchanged)
```

Classifier called once per gate (not per amplitude); ~5 ns overhead
amortised against the inner kernel.

SoA dispatch (`apply_1q_soa` in `kernels/soa.rs`) is **structurally
identical** — same three steps in the same order, routing to
`apply_1q_soa_x_{tier}` etc. Spec drift across the two layouts is
prevented by the per-tier equivalence test (§ 6.1).

Diagonal and anti-diagonal matrix classes are **mutually disjoint** by
construction: diagonal requires off-diagonals ≈ 0, anti-diagonal
requires diagonals ≈ 0. No matrix can match both unless it is ~zero
(rejected by unitarity at the `apply_gate` level). Dispatch order
(diagonal → anti-diagonal → generic) is therefore correctness-neutral;
the chosen order matches frequency (diagonal gates are more common in
real circuits per the P1-06 QFT histogram).

### 4.2 Three-tier coverage (per ADR 0010 pattern)

For each gate-kind (X, Y, generic anti-diag) × each layout (AoS, SoA):

| Tier | Entry condition                                  | AoS kernel shape (LANES = 4) | SoA kernel shape (LANES_SOA = 8) |
|------|--------------------------------------------------|------------------------------|----------------------------------|
| A    | `1 << target ≥ LANES` ∧ every `control > target` | Packed block swap (`vmovupd` 2× + reg swap) | Symmetric block swap on `re[]`, `im[]` |
| B    | `target < LANES`                                  | In-register lane swap (`vpermq`/`vshufpd`) | In-register `vpermpd` on both streams |
| C    | Any `control < target`                            | Scalar swap loop with `expand_with_fixed` outer walk | Scalar swap loop |

Tier-A is the dominant path for typical n ≥ 6 circuits.

### 4.3 Y arithmetic (sign-flip via `vxorpd`)

`Y = [[0, -i], [i, 0]]` so:
- `amps[i0] ← (-i) · amps[i1] = (im[i1], -re[i1])`
- `amps[i1] ← (+i) · amps[i0] = (-im[i0], re[i0])`

In packed AoS (`re_0, im_0, re_1, im_1, ...`) this is **swap of (re, im)
within each pair** + **sign-flip on the imaginary lane of one side and
the real lane of the other**. Implemented as:

1. `vpermilpd` with control `0x55` — swaps (re, im) → (im, re) per pair.
2. `vxorpd` with two sign masks (one for the `i0`-side, one for the
   `i1`-side) — flips the right halves.
3. Block swap as in X kernel.

Total: ~3 µops per packed register vs. 4 `vfmadd231pd` for full complex
multiply.

In SoA, with `re[]` and `im[]` as separate streams, Y is:
1. Swap `im[i0]` ↔ `re[i1]`, `re[i0]` ↔ `-im[i1]` (with sign flip via
   `vxorpd` on the right stream block).

`YNeg` (`Y' = [[0, +i], [-i, 0]]`, an "anti-Pauli-Y" rarely seen but
trivial extension) uses opposite-sign masks.

### 4.4 Anti-diagonal classifier (`is_antidiagonal_2x2`)

Mirror of `is_diagonal_2x2`:

```rust
pub(crate) fn is_antidiagonal_2x2(m: &[[Complex; 2]; 2]) -> bool {
    let diag = [&m[0][0], &m[1][1]];
    for entry in diag {
        if !entry.re.is_finite() || !entry.im.is_finite() {
            return false;                          // ADR 0006: explicit reject
        }
        if entry.norm_sqr() >= DIAGONAL_EPS_SQ {
            return false;
        }
    }
    true
}
```

`is_finite` reject **precedes** the magnitude test — a NaN-poisoned
diagonal entry would otherwise silently classify the matrix as
anti-diagonal and route the NaN to the swap fast path (which only
consults `m[0][1]`, `m[1][0]`). Rejecting non-finite diagonals forces
the generic kernel to see and propagate the NaN. Three Phase-0 review
rounds regressed on the equivalent pattern for `is_diagonal_2x2` — the
guard is mandatory.

### 4.5 Pauli kind classifier (`classify_1q_antidiag`)

```rust
pub(crate) enum Perm1qKind { X, YPos, YNeg }

pub(crate) fn classify_1q_antidiag(
    m: &[[Complex; 2]; 2],
) -> Option<Perm1qKind> {
    // Caller already established is_antidiagonal_2x2(m).
    // Component-wise comparison within PERM_TOL = 1e-14 (reuse from is_cz_signature).
    let a = m[0][1];   // upper-right
    let b = m[1][0];   // lower-left

    let close = |z: Complex, re: f64, im: f64| {
        (z.re - re).abs() < PERM_TOL && (z.im - im).abs() < PERM_TOL
    };

    if close(a, 1.0, 0.0)  && close(b, 1.0, 0.0)  { return Some(Perm1qKind::X); }
    if close(a, 0.0, -1.0) && close(b, 0.0, 1.0)  { return Some(Perm1qKind::YPos); }
    if close(a, 0.0, 1.0)  && close(b, 0.0, -1.0) { return Some(Perm1qKind::YNeg); }
    None
}
```

Component-wise check (not `(z - target).norm_sqr() < PERM_TOL`) because
`norm_sqr < PERM_TOL` is effectively `|z - target| < sqrt(PERM_TOL) ≈
1e-7`, seven orders looser than the documented tolerance — the same
mistake caught in `is_cz_signature` during P1-07 review.

`None` falls through to `apply_1q_antidiag_generic_{tier}` which does
the full complex multiply on `a` and `b` per pair.

## 5. Correctness invariants

### 5.1 Pairwise-disjoint bits (ADR 0010 canonical pattern)

Every SIMD kernel's inner address is `block | offsets[k] | j` with
**pairwise disjoint** bit masks across `block`, `offsets[k]`, and the
within-pair offset `j ∈ {0, 1<<target}`. This is the canonical
invariant from P1-07 — the bug class hit twice during that ticket. The
spec is explicit: any kernel that produces a `block | offset | j`
address MUST verify pairwise-disjointness either by construction (free
bits in `block`, fixed bits in `offsets`, target bit in `j`) or by a
debug-assertion. Per-test enumeration via the portable indexing-coverage
test (§ 6.2) catches the violation.

### 5.2 Tier-C outer-walk renormalisation

For external controls below the target (`controls.iter().any(|&c| c <
target)`), Tier-C scalar kernel:

```rust
let mut external: Vec<u32> = controls.iter().filter(|&&c| c < target).copied().collect();
external.sort();
let fixed: Vec<(u32, bool)> = std::iter::once((target, false))
    .chain(external.iter().map(|&c| (c, true)))
    .collect();
let free_bits = n - fixed.len();
for k in 0..(1u64 << free_bits) {
    let i0 = expand_with_fixed(k, &fixed) as usize;
    let i1 = i0 | (1usize << target);
    // X: swap(amps[i0], amps[i1])
    // Y: amps[i0], amps[i1] = (∓i)·old_i1, (±i)·old_i0
    // generic: amps[i0], amps[i1] = b·old_i1, a·old_i0
}
```

`expand_with_fixed` is the existing helper in `kernels/mod.rs`. SoA
Tier-C mirror identical but operates on `re[]`/`im[]` separately.

### 5.3 NaN propagation (ADR 0006)

Matrices with non-finite entries MUST reach the generic kernel so the
NaN propagates into amplitudes:

- `is_diagonal_2x2(m)` and `is_antidiagonal_2x2(m)` both reject any
  non-finite entry up front.
- `classify_1q_antidiag` is only called after `is_antidiagonal_2x2`
  passes, so it never sees a non-finite `m[0][0]` or `m[1][1]`.
- However, `m[0][1]` or `m[1][0]` could still be non-finite. The
  component-wise `close()` predicate uses `(z.re - re).abs() <
  PERM_TOL`; a NaN difference yields `false`, so the matrix falls
  through to the generic anti-diagonal path which DOES propagate NaN
  through its complex multiply. No explicit `is_finite` guard needed in
  `classify_1q_antidiag` — the falsiness of NaN comparisons handles it.
  (Tested explicitly in § 6.4.)

## 6. Testing strategy

### 6.1 Per-kernel equivalence vs naive (18 unit-test groups)

For each of `{X, Y_pos, Y_neg, antidiag_generic}` × `{Tier A, Tier B,
Tier C}` × `{AoS, SoA}` — covers `(target ∈ 0..=6, controls ∈ subsets
of {0..=6}\{target} of size ≤ 2, n ∈ {3, 8, 14})`. Amplitudes compared
to `apply_1q` generic via `naive` reference, tolerance 1e-12.

Co-located in `kernels/aos.rs` / `kernels/soa.rs` `mod tests` blocks.

### 6.2 Portable indexing-coverage test (integer-only, no FP)

`kernels/mod.rs` `tests/` block: reproduces address generation `block |
offsets[k] | j` for the SAME `(target, controls, n)` enumeration as
§ 6.1. Assertions:

- Each generated address is in `0..(1 << n)`.
- No `(i0, i1)` pair appears twice across the enumeration.
- The set of pairs equals the canonical pair set
  `{ (a, a ^ (1 << target)) : a ∈ controls-respecting-subset }`.

Per P1-07 Task 14: integer model catches SIMD-indexing bugs that the
oracle path misses.

### 6.3 Boundary-`n` test (SoA Tier-C SIGSEGV regression guard)

Bell state `n=2` plus `n ∈ {LANES-1, LANES, LANES+1}` for AoS and
`n ∈ {LANES_SOA-1, LANES_SOA, LANES_SOA+1}` for SoA. Asserts apply_1q
on X and Y both completes without segfault and matches naive. Mirrors
the regression class caught only on EPYC during P1-07.

### 6.4 NaN-propagation test (ADR 0006)

Three matrices, one per "NaN-poisoning" class:

- `[[NaN, 1], [1, 0]]` — non-finite diagonal entry; MUST fall through
  to generic kernel (which propagates NaN to all touched amplitudes).
- `[[0, NaN], [1, 0]]` — non-finite off-diagonal; MUST fall through to
  generic anti-diagonal path with NaN multiply.
- `[[0, 1], [NaN, 0]]` — same, opposite off-diagonal.

For each: apply_1q to a non-trivial state, assert at least one touched
amplitude is `is_nan()`.

### 6.5 Oracle comparison vs Qiskit

Extend the existing `run_oracle_qiskit` random-circuit enumeration in
`crates/aleph-test` to include `Gate::X` and `Gate::Y` (currently mostly
H/Phase/CX). End-to-end equivalence to Aer, tolerance 1e-10 on
amplitudes.

### 6.6 Proptest (P0-05 infra)

Random `[[Complex; 2]; 2]` matrices. Property: if `is_antidiagonal_2x2(m)
== true`, then `apply_1q(m, ...) ≡ apply_1q(generic-path, m, ...)`
within 1e-12. Guarantees classifier doesn't strand a valid anti-diagonal
matrix on the generic-multiply path AND that the dispatched fast-path
matches the generic-path result.

## 7. Benchmarks

`crates/aleph-sv/benches/pauli_xy.rs` (NEW):

### 7.1 L2-resident micro (gating for BACKLOG AC)

`n = 14` (state = 256 KiB, fits EPYC's 1 MiB L2). For each of
`{AoS, SoA}` × `{X, Y, generic-anti-diag}`:

- Baseline: invoke the generic 2×2 kernel directly (already `pub(crate)`
  per P1-03; a bench-local re-export shim keeps the call site honest
  without leaking the symbol cross-crate).
- Specialised: `apply_1q` with classifier dispatch.

**AC: 3–10× speedup on the specialised path vs. generic 2×2.** Target
qubit at position 8 (Tier-A entry).

### 7.2 L3-resident reference (informational)

`n = 20` (state = 16 MiB, fits L3 not L2). Documented in
`docs/perf/phase1-vs-qiskit.md` "P1-05 update" but not gating —
bandwidth-bound per the P1-06 lesson.

### 7.3 Workload re-bench (informational)

`grover_n20` re-run vs. P1-07 baseline. Any delta documented in the
perf doc; not gating per the explicit user decision on AC strictness.

## 8. Performance hierarchy positioning

Per CLAUDE.md, the ROI hierarchy is `algorithm > IR opt > memory layout
> SIMD > threads > GPU`. P1-05 sits in the SIMD tier and is strictly
ranked below P1-09/10 (gate fusion) and any subsequent Stage 2
IR-optimisation pass. The justification for shipping it before Stage 2
is twofold:

1. The user has explicitly chosen to complete Stage 1 in full before
   Stage 2 (recorded in `[[phase1-completion-plan]]`).
2. The anti-diagonal classifier is a substrate dependency for future
   fusion: the 1q-fusion pass (P1-09) produces matrices that may be
   anti-diagonal (e.g. `X · S = [[0, i], [1, 0]]` — generic anti-diag);
   having the dispatch in place means fused matrices land on the right
   kernel from day one.

## 9. BACKLOG amendment

### 9.1 P1-05 entry rewrite (§ 15.1)

Current BACKLOG P1-05 says:

> Pauli-X swaps amplitudes at index i and i ⊕ 2^q. Implement as pure
> swap, no multiplication. For Y: swap + multiply one half by ±i …
> For Z: `re[i1] = -re[i1]; im[i1] = -im[i1];`

Amend to:

- Remove Z from scope — already covered by P1-06's diagonal kernel.
- Substrate: AoS + SoA parity, three-tier dispatch per ADR 0010.
- Add `Perm1qKind` enum + `classify_1q_antidiag` classifier; mirrors
  P1-06 diagonal classifier and ADR 0010 2q-permutation classifier.
- AC clarification: 3–10× speedup is **micro-bench-level at L2-resident
  state (`n ≤ 14`)**; n=20 wall-clock is informational, not gating —
  bandwidth-bound regime per ADR 0008.
- AC bullet rewrite:
  - `[x] X kernel (pure swap, AoS + SoA, three tiers)`
  - `[x] Y kernel (swap + sign-flip, AoS + SoA, three tiers)`
  - `[x] Generic anti-diagonal kernel (full multiply, AoS + SoA, three tiers)`
  - `[x] is_antidiagonal_2x2 + classify_1q_antidiag in kernels/mod.rs`
  - `[x] Micro-bench: 3–10× speedup over generic on L2-resident state`

### 9.2 ADR 0011

`docs/decisions/0011-anti-diagonal-1q-classifier.md` — new ADR. Closes
the loop with ADR 0009 (diagonal) and ADR 0010 (2q permutation):

- Context: third recognised matrix class.
- Approach B chosen (separate kernels per Pauli kind) over A (single
  generic + runtime fast-path) or C (X+Y only, no generic anti-diag).
- Tier coverage rationale (mirror ADR 0010).
- Tolerance: component-wise, `PERM_TOL = 1e-14`.

## 10. Out of scope (explicit)

- Gate fusion (P1-09/P1-10).
- Multi-controlled X (CCX, MCX) — P1-08.
- Anti-diagonal 2q gates (e.g. `iSWAP` analogues) — would be a P1-07
  follow-up if needed; not on Phase 1 critical path.
- Default x86 backend flip (AoS vs SoA) — deferred to Phase 1 closure.
- Removing the SoA backend — same.
