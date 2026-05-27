# P1-06 — Specialised diagonal-gate 1q kernel (design)

> **Phase 1, Stage 1, ticket 1.** Priority elevated above its original P1-05
> ordering by the Stage 0 baseline report (`docs/perf/phase1-vs-qiskit.md`,
> 2026-05-27): QFT-20 is the only workload over the ROADMAP § 7 ≤ 2× Aer target,
> and **59 % of QFT-20's transpiled gates are 1q `p` (Phase) — pure diagonal**.
> Specialising the diagonal path directly attacks the workload closest to the
> exit criterion.

## 1. Goal

Add a specialised diagonal 1-qubit kernel to the AoS and SoA backends.
Diagonal gates (`Z`, `S`, `T`, `Sdg`, `Tdg`, `Rz`, `Phase`, and any user-supplied
diagonal `GenericUnitary`) skip the cross-term arithmetic of the generic 2×2
kernel and run through a single-stream multiply, halving the µop count per
complex pair on the AVX-512 path and reducing total kernel work by ~30 % on
QFT-20.

## 2. Non-goals

- **No 2q diagonal kernel.** `Cz`, controlled-`Phase` decomposed as 2q matrices
  (i.e. when `Gate::Cz` lands in `apply_2q`), and other 2q diagonal forms are
  in scope for **P1-07** (2q kernel ticket). For QFT specifically this doesn't
  matter — Qiskit's `optimization_level=0` transpile decomposes controlled-`p`
  into 3× uncontrolled `p` + 2× `cx` so all `p` gates flow through the **1q**
  path with `controls=[]`. (Verified by gate-mnemonic histogram on
  `circuits/qft_n20.qasm`: 569 `p`, 380 `cx`, 19 `h`.)
- **No gate-tag-aware dispatch at the backend layer.** Detection happens at the
  kernel layer via matrix inspection so user-supplied diagonal
  `GenericUnitary(M2x2)` is caught as well. The matrix-vs-gate-tag dispatch
  alternative is rejected in § 4.3.
- **No new public API.** The `Backend` trait, `GateInstance`, `GateMatrix` —
  all unchanged. The work lives inside the SV backend kernels.
- **No SoA backend removal.** Open question #2 in ADR 0008 (whether to keep
  SoA at all) is **explicitly deferred to Phase 1 closure** per the phase 1
  completion plan. P1-06 ships symmetric diagonal kernels in both backends.

## 3. Deliverables

A single squash-merge PR titled `[P1-06] Specialised diagonal-gate 1q kernel`
adding:

```
crates/aleph-sv/src/kernels/
├── aos.rs            # add apply_1q_diagonal_avx512 + detection branch in apply_1q
├── soa.rs            # add apply_1q_diagonal_soa + detection branch in apply_1q_soa
└── mod.rs            # add `is_diagonal_2x2(&[[Complex; 2]; 2]) -> bool` helper
docs/decisions/0009-diagonal-fast-path.md   # ADR documenting the pattern
```

No production code changes outside `crates/aleph-sv/src/kernels/`. Tests added
inline (`#[cfg(test)]` modules in the kernel files) and via the existing
`aleph-test` property strategies. Bench measurements feed `docs/perf/` updates
at the end of Stage 1 (not this PR).

## 4. Design

### 4.1 Detection contract

A `2×2` complex matrix `M = [[m00, m01], [m10, m11]]` is **diagonal** iff
`m01.norm_sqr() < EPS_SQ && m10.norm_sqr() < EPS_SQ`.

`EPS_SQ` is a module-private constant set to `1e-30` (= `(1e-15)²`), one order
of magnitude looser than FP64 unit roundoff so genuine zeros and exact-on-paper
zero off-diagonals (`Phase`, `Rz`, `Z`, `S`, `T`, `Sdg`, `Tdg`) all detect
cleanly. A genuinely non-diagonal matrix with `|m01|² ≥ 1e-30` (i.e. `|m01| ≥
~3e-16`) falls through to the generic kernel. The asymmetry is acceptable: a
"barely-diagonal" gate the user explicitly built with `m01 = 1e-17` would be
mis-applied as diagonal, but the error per amplitude is bounded by
`|m01| · max|amp|` ≤ FP64 roundoff anyway.

Helper: `kernels/mod.rs::is_diagonal_2x2(m) -> bool`. Lives in the kernel
module (not `aleph-core`) — pure perf-internal heuristic, no public-facing
contract.

### 4.2 AoS diagonal kernel

**Math.** For each amplitude `z = state[i]`, multiply in place:

```
state[i] *= d   where d = if (i >> target) & 1 == 0 { m00 } else { m11 }
```

**Block structure.** The target qubit splits the basis index into blocks of
`target_bit = 1 << target` contiguous amps with the same multiplier. Outer
step `2 * target_bit`; the first `target_bit` amps in each outer block get
`m00`, the next `target_bit` get `m11`.

**AVX-512 path** (`apply_1q_diagonal_avx512`) — for `target ≥ LANES = 4` (i.e.
`target_bit ≥ 4` contiguous amps in a block):

```
for each outer block of 2 * target_bit amps starting at i0:
    if (i0 & ctrl_mask) != ctrl_mask: continue
    for j = 0..target_bit step LANES:   # 0-side, multiplier m00 = (d_re, d_im)
        z = vmovupd state[i0 + j]                # 4 complex pairs: (re,im,re,im,...)
        sz = vpermilpd 0x55 z                    # swap (re,im) per pair
        t = vmulpd(d_im_bcast, sz)               # (d_im·im, d_im·re, ...) per lane
        out = vfmaddsub(d_re_bcast, z, t)        # even: d_re·re - d_im·im = re_out
                                                  # odd:  d_re·im + d_im·re = im_out
        vmovupd state[i0 + j] = out
    for j = 0..target_bit step LANES:   # 1-side, multiplier m11 (same pattern)
        ... identical 5-µop sequence with d_re/d_im from m11 ...
```

Per inner iter (4 complex pairs): **1 vmovupd + 1 vpermilpd + 1 vmulpd +
1 vfmaddsub + 1 vmovupd ≈ 5 µops**, vs the generic `apply_1q_avx512`'s ~16
µops per 4 pairs (2 loads, 4 muls, 4 fmaddsub, 2 permutes, 2 adds, 2 stores).
Roughly **3× fewer µops per packed-complex** on the diagonal path.

The `m00_re_bcast`/`m00_im_bcast`/`m11_re_bcast`/`m11_im_bcast` broadcasts
are computed once at the top of the kernel — constant across the entire
state vector walk.

**Safety contract** (same shape as `apply_1q_avx512`):
- Host CPU supports AVX-512F (`is_x86_feature_detected!("avx512f")` gate).
- `1usize << target ≥ LANES` so each block has ≥ LANES contiguous pairs.
- Every control's qubit index is strictly greater than `target` so the inner
  SIMD walk's block offset doesn't toggle any control bit.
- Standard apply_gate invariants: `target` and `controls` distinct, in range.

**Scalar fallback** (for `target < LANES` or no AVX-512):

```
for i in 0..amps.len():
    if (i & ctrl_mask) != ctrl_mask: continue
    let d = if (i >> target) & 1 == 0 { m00 } else { m11 };
    amps[i] *= d;
```

LLVM should auto-vectorise this to 2-lane `vmulpd xmm` on x86_64 even without
explicit intrinsics — the AVX-512 path is a substantial extra win, not the
only one.

### 4.3 SoA diagonal kernel

Symmetric pattern in `kernels/soa.rs`. SoA stores `(re_arr, im_arr)` as
separate `Vec<f64>`. For a diagonal multiply `z * d = z * (d_re + d_im i)`:

```
new_re[i] = re[i] * d_re - im[i] * d_im
new_im[i] = re[i] * d_im + im[i] * d_re
```

The diagonal case still has cross-term mixing **between the re and im streams**
of *the same amplitude*, but unlike the generic 1q SoA which mixes 4 streams
(re/im of paired amps), this is only 2 streams. AVX-512 packed-double over
contiguous `f64` works naturally. Per LANES=8 doubles (= 8 complex amps via
re_arr + 8 complex amps via im_arr):

```
re_lane = vmovupd re_arr[i..i+8]
im_lane = vmovupd im_arr[i..i+8]
new_re = vfmsub231(re_lane, d_re_bcast, vmulpd(im_lane, d_im_bcast))
new_im = vfmadd231(re_lane, d_im_bcast, vmulpd(im_lane, d_re_bcast))
vmovupd re_arr[i..i+8] = new_re
vmovupd im_arr[i..i+8] = new_im
```

~7 µops per 8 amps = ~3.5 µops per amp; less per-amp than the AoS path (~1.25
µops per amp), but SoA's overall throughput is gated by the 2-stream load
pattern (per ADR 0008). The relative speedup vs SoA generic 1q is similar to
AoS's relative speedup.

The same block-walk structure applies — outer step `2 * target_bit`, two
sub-blocks with `(d_re, d_im) = (m00.re, m00.im)` then `(m11.re, m11.im)`.

### 4.4 Dispatch

Both backends' top-level `apply_1q` / `apply_1q_soa` get a short prelude:

```rust
pub(crate) fn apply_1q(amps: &mut [Complex], target: u32, controls: &[u32],
                      m: &[[Complex; 2]; 2]) {
    if super::is_diagonal_2x2(m) {
        apply_1q_diagonal(amps, target, controls, m[0][0], m[1][1]);
        return;
    }
    // ... existing AVX-512 path + scalar fallback ...
}
```

The detection cost: 2 `norm_sqr()` calls + 2 comparisons per gate. For QFT-20
(970 gates) that's ~5 µs total — three orders of magnitude below the
millisecond-scale measurement noise floor.

### 4.5 Why matrix-detection over gate-tag dispatch

Three alternatives were considered:

1. **Matrix detection in kernel (chosen).** Pros: catches user-supplied
   diagonal `GenericUnitary(M2x2)`, no backend-layer changes, kernels remain
   gate-tag-agnostic (preserves the layering established by P0-09's matrix-based
   dispatch). Cons: ~5 ns detection cost per gate (negligible).
2. **Gate-tag dispatch in backend.rs.** Pros: zero runtime detection cost. Cons:
   user-supplied diagonals don't benefit, backend gains a per-variant case for
   Z/S/T/Sdg/Tdg/Rz/Phase that must be updated whenever a new diagonal gate is
   added to `Gate`, layering violation. Rejected.
3. **Hybrid (tag hint + kernel verify).** Pros: same as #1 but compile-time
   guarantee for intrinsic diags. Cons: doubles API surface, complicates the
   `apply_1q` signature. Not worth it for ~5 ns / gate. Rejected.

### 4.6 Why include SoA path despite ADR 0008's "AoS dominates" finding

P1-01's `all_fixtures_match_naive` workhorse test pins SoA ≡ AoS within 1e-12
across 112 generated oracle fixtures. Letting the AoS path take a 30 % shortcut
on diagonal gates while SoA stays slow would make the comparison less honest
(the appendix table in `phase1-vs-qiskit.md` shows the cross-backend gap; we
shouldn't artificially widen it via incomplete optimisation). Symmetric
implementation also keeps the code paths' structure parallel, which helps
future readers and any potential SoA-revival decision.

Open ADR 0008 question #2 — "do we keep `SoaSvBackend` in tree at all?" —
**stays open**. P1-06 doesn't commit either way; it just keeps both backends'
diagonal handling at the same fidelity.

## 5. Acceptance criteria

- [ ] `kernels/mod.rs::is_diagonal_2x2(m) -> bool` implemented + unit-tested
      (exact zeros, FP-noise tolerance, non-diagonal rejection).
- [ ] `kernels/aos.rs::apply_1q_diagonal_avx512` implemented under
      `#[cfg(target_arch = "x86_64")]` + `#[target_feature(enable = "avx512f")]`,
      with safety contract comment matching the existing
      `apply_1q_avx512` style.
- [ ] Scalar `apply_1q_diagonal` fallback in `kernels/aos.rs` for
      `target < LANES` or non-AVX-512 hosts.
- [ ] `kernels/aos.rs::apply_1q` prelude routes to diagonal path when
      `is_diagonal_2x2(m)` returns true.
- [ ] Symmetric `apply_1q_diagonal_soa` in `kernels/soa.rs` + `apply_1q_soa`
      prelude routing.
- [ ] Unit tests in `kernels/aos.rs::tests` and `kernels/soa.rs::tests`:
  - `apply_1q_diagonal_z_matches_generic` (Z gate matrix)
  - `apply_1q_diagonal_phase_random_theta_matches_generic` (random θ over
    `prop_for_each` 32 iterations)
  - `apply_1q_diagonal_with_controls_matches_generic` (controlled-Phase)
  - `apply_1q_diagonal_user_supplied_matrix` (`GenericUnitary` containing a
    hand-built diagonal)
  - `apply_1q_almost_diagonal_falls_through_to_generic` (`m01 = 1e-8` rejected;
    confirms generic path still hit)
  - `apply_1q_diagonal_target_0_uses_scalar_fallback` (target = 0, no AVX-512
    block path)
- [ ] Property test `prop_1q_diagonal_equiv_to_generic` in `crates/aleph-test`
      strategy module: any diagonal `M2x2` over `prop_for_each 64`, both paths
      produce equal state to 1e-14 tolerance.
- [ ] Existing oracle harness (112 generated fixtures, vs-Qiskit equivalence)
      passes; `all_fixtures_match_naive` (SoA ≡ AoS) passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] **EPYC benchmark:** `qft_n20` ≥ 1.30× faster than the post-P1-03 baseline
      (1098 ms → ≤ 845 ms). Target rationale: 59 % of gates are 1q-diagonal,
      ~3× per-µop speedup on those → ~1.4× kernel speedup → ~1.3× wall-clock
      speedup accounting for non-diagonal gates and memory bandwidth.
- [ ] **EPYC benchmark:** `random_brickwall_n20_d20` ≥ 1.10× faster (33 % of
      gates are `rz`, diagonal; expected wall-clock improvement ~10–15 %).
- [ ] **EPYC benchmark:** `grover_n20_iters5` no regression (within 1.05×).
      Grover is multi-controlled-X heavy; diagonal speedup is small.
- [ ] **Bencher.dev:** no regression on `bell`, `ghz`, `qft/10`, `qft/15`,
      `qft/20`, `random` benches in `benches/benches/`. (`qft/20` should
      improve.)
- [ ] **ADR 0009** committed: `docs/decisions/0009-diagonal-fast-path.md`
      documenting the matrix-detection pattern, expected speedup shape, and
      why it lives in kernels (not backend dispatch).
- [ ] PR body includes EPYC bench numbers (before / after `qft_n20`,
      `random_brickwall_n20_d20`, `grover_n20_iters5`).

## 6. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Detection cost > expected on non-diagonal gates | Bench an artificial 1000-H circuit (no diagonal gates) before / after; expect < 0.5 % overhead. If higher, switch to enum-tag fast path. |
| FP-noise edge case: `m01 = 1e-300` from unitarity-normalisation drift mis-categorised | `EPS_SQ = 1e-30` ⇒ `|m01| > 3e-16` is the threshold. FP64 unit roundoff for a unit-magnitude unitary is ~2e-16, so any "true" off-diagonal value still detects as non-diagonal. Test `apply_1q_almost_diagonal_falls_through_to_generic` pins this. |
| Controls present + `target < LANES` corner | Scalar fallback handles target < LANES; controlled scalar fallback walks `i in 0..len()` with the existing ctrl_mask check. No SIMD shortcut for that path. |
| Block-walk math wrong at `target = 0` (`target_bit = 1`, every other amp same multiplier) | Scalar fallback (target < LANES = 4 always hits scalar). Test `apply_1q_diagonal_target_0_uses_scalar_fallback` confirms. |
| SoA path's expected speedup smaller than AoS (per ADR 0008's load-pattern finding) | Acceptable — symmetry is the goal, not absolute SoA perf. Final report will document the cross-backend gap; user can decide ADR 0008 open Q#2 at Phase 1 closure. |
| Test runtime explosion from property test 64 iterations × oracle harness | `prop_for_each 64` runs in milliseconds (no state vector larger than n=6 in property strategies). Oracle harness already runs the 112 fixtures; total CI test time increase < 5 s. |
| The 30 % QFT wall-clock improvement target may be optimistic if memory bandwidth dominates over µop count on QFT-20 | At n=20, the state vector is 16 MiB — fits in L3 but not L2. We're bandwidth-bound, not µop-bound, at large n. **The ~3× µop reduction translates to a smaller wall-clock improvement.** If we see < 1.2× wall-clock at the end, document as "diagonal path is correct but bandwidth-limited; combine with IR-level gate fusion (P1-09) for further QFT speedup". Don't gate the PR on hitting the bench target — acceptance criterion is **≥ 1.30× on qft_n20**, which is conservative. |

## 7. Open questions (deferred, not blockers)

1. **AVX2 path?** The AVX-512 path is gated by `is_x86_feature_detected!("avx512f")`.
   Hosts without AVX-512 (Intel pre-Skylake-X, AMD pre-Zen 4) hit the scalar
   fallback, which LLVM auto-vectorises to xmm. Adding an explicit AVX2
   diagonal path (~256-bit ymm with 2 complex pairs per register) might be
   worth ~1.5× over scalar on those hosts. **Out of scope; deferred to a
   separate `[infra]` ticket if Intel-laptop perf becomes a priority.**
2. **Generalised "diagonal in any basis" detection?** Some 1q gates are diagonal
   in the X or Y basis (e.g. Pauli-X is diagonal in the Hadamard basis). Doing
   a 1q similarity-transform fast path is its own design question, **out of
   scope here**, possibly Phase 2+ work.

## 8. Workflow

Per the established per-ticket workflow (P0-06 onwards): brainstorm (done) →
spec (**this doc**) → plan → execute → request code review → fix → squash-merge.

Next step after spec approval: invoke `writing-plans` to author
`docs/superpowers/plans/2026-05-27-p1-06-diagonal-1q-kernel.md`.
