# P1-07 — 2-qubit gate kernel + CNOT/CZ/SWAP specialised paths (design)

> **Phase 1, Stage 1, ticket 2.** Continues the Stage 1 SIMD-specialisation
> arc opened by P1-06. Priority anchored by the Stage 0 baseline report
> (`docs/perf/phase1-vs-qiskit.md`, 2026-05-27): QFT-20 remains the only
> workload over the ROADMAP § 7 ≤ 2× Aer target, **39 % of QFT-20's
> transpiled gates are 2q `cx`** flowing through the scalar `apply_2q`
> path with full 4×4 matmul per quadruplet, and the post-P1-06 baseline
> regressed slightly on QFT-20 to 2.47× Aer. Specialising the 2q path —
> permutations (CNOT/SWAP) at zero multiplies, diagonals at single-stream
> multiply, CZ at pure sign-flip — directly attacks the lone bandwidth-
> and µop-bound bottleneck blocking the Phase 1 exit criterion.

## 1. Goal

Add specialised 2-qubit kernels to the AoS and SoA backends:

- **Generic 2q AVX-512** — packed-complex, 4-substream-load, full 4×4
  dense matmul; echoes the structural pattern of `apply_1q_avx512`
  (ADR 0008) extended to two-qubit subspaces.
- **CNOT / SWAP** — pure permutation paths, ZERO multiplies; swap-pair
  trafic only. Detection via `classify_2q_permutation` (matrix-based,
  per ADR 0009 precedent), three SIMD tiers (A: `1 << min(targets) ≥
  LANES`; B: only `max` does; C: both targets in the lowest two
  qubits).
- **CZ** — pure sign-flip via `vxorpd` with a `-0.0` mask; touches only
  1/4 of the state vector. Bonus shortcut on top of the diagonal-4×4
  fast path.
- **2q-diagonal fast path** — generalised `is_diagonal_4x4(M)` detection,
  AVX-512 single-stream multiply across four sub-block multipliers.
  Catches CZ, controlled-Phase(θ), Rzz(θ), and any user-supplied
  diagonal `GenericUnitary([[Complex; 4]; 4])`.
- **Identity 2q** — free win: matrix-detected, returns immediately.

All five are hung off `apply_2q` as a detection prelude. The fundamental
goal is to push QFT-20 wall-clock from the post-P1-06 baseline of 1133 ms
down to ≤ 944 ms (1.20× hard AC, lands at 2.06× Aer); the ambition is
≤ 870 ms (1.30×, clears ROADMAP exit at 1.89× Aer).

## 2. Non-goals

- **Multi-controlled 2q (Toffoli, CCZ, MCX).** P1-08 owns the 3q-kernel
  layer; this ticket touches `apply_2q` only.
- **Adjacent-pair-specific dense-2q kernel** (the variant-B option from
  brainstorm). The Tier A SIMD path covers ~80 % of QFT-20's cx pairs;
  the adjacent special case adds 10–15 % on a 30–40 % slice. Deferred
  as Open Q #1; revisit only if the 1.30× ambition is unreached.
- **AVX2 path.** Pre-Skylake-X / pre-Zen-4 hosts fall through to scalar
  fallbacks (LLVM auto-vec on xmm). Separate `[infra]` ticket if Intel-
  laptop perf becomes a priority. (Mirrors P1-06 § 7.)
- **iSWAP / sqrt-SWAP / arbitrary permutation specialisations.**
  BACKLOG-AC list these as "specialised as needed" — not exercised by
  Tier-1 algorithms (QFT, Grover, random brickwall, GHZ). Out of scope.
- **`Backend` trait / `Gate` / `GateMatrix` API changes.** All code lives
  inside `crates/aleph-sv/src/kernels/`. Kernels remain gate-tag-agnostic.
- **SoA explicit-AVX-512 for the generic dense-2q path.** ADR 0008
  documented that 4-stream SoA SIMD lost to AoS auto-vec on dense
  matmul; not worth re-litigating here. SoA dense-2q stays scalar.
- **ADR 0008 open Q#2 — "do we keep SoA at all?"** stays deferred to
  Phase 1 closure (P1-14). P1-07 ships symmetric specialised paths in
  both backends to keep `all_fixtures_match_naive` honest.

## 3. Deliverables

A single squash-merge PR titled `[P1-07] 2q kernel + CNOT/CZ/SWAP
specialised paths`:

```
crates/aleph-sv/src/kernels/
├── aos.rs        # + apply_2q_avx512              (generic dense, Tier A)
│                 # + apply_2q_cnot_avx512         (perm, Tiers A+B+C)
│                 # + apply_2q_swap_avx512         (perm, Tiers A+B+C)
│                 # + apply_2q_cz_avx512           (sign-flip, Tier A)
│                 # + apply_2q_diagonal_avx512     (4-mult, Tier A)
│                 # + 5 *_scalar fallbacks
│                 # + dispatch prelude in apply_2q
├── soa.rs        # + symmetric SoA explicit-AVX-512 for cnot/swap/cz/diag
│                 # + apply_2q (dense) stays scalar
│                 # + dispatch prelude in apply_2q (SoA)
└── mod.rs        # + Perm2qKind enum {Identity, CnotHi, CnotLo, Swap}
                  # + is_diagonal_4x4(m: &[[Complex; 4]; 4]) -> bool
                  # + classify_2q_permutation(m) -> Option<Perm2qKind>
                  # + is_cz_signature(d0, d1, d2, d3) -> bool

docs/decisions/0010-2q-specialised-paths.md   # ADR — dispatch tree, tiering
benches/benches/p1_07_microbench.rs           # criterion: cnot vs generic_2q
```

Tests inline (`#[cfg(test)]` sub-modules in `aos.rs`, `soa.rs`,
`kernels/mod.rs`) plus new property strategies in `aleph-test`
(`arb_diagonal_4x4`, `arb_cnot_matrix`, `arb_swap_matrix`).
EPYC bench numbers feed `docs/perf/` updates at end of Stage 1
(not this PR).

## 4. Design

### 4.1 Detection contracts (`kernels/mod.rs`)

All three helpers operate on `&[[Complex; 4]; 4]`. `EPS_SQ = 1e-30` is
the same module-private constant introduced by P1-06; an additional
`EPS_PERM = 1e-14` is used for permutation-row magnitude checks (looser,
because unitarity-normalisation drift in user-built matrices can push
off-diagonal entries to ~1e-15 while keeping on-diagonal `|m|² ≈ 1`).

**`is_diagonal_4x4(m) -> bool`.** Returns `true` iff every off-diagonal
entry satisfies `|m[r][c]|² < EPS_SQ` for `r ≠ c`. 12 `norm_sqr` + 12
compares per call ≈ 30 ns. For QFT-20 (970 gates, 380 of them 2q),
detection overhead totals ~12 µs — three orders of magnitude under
measurement noise.

```rust
pub(crate) fn is_diagonal_4x4(m: &[[Complex; 4]; 4]) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            if r != c && m[r][c].norm_sqr() >= EPS_SQ {
                return false;
            }
        }
    }
    true
}
```

**`Perm2qKind` + `classify_2q_permutation`.**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Perm2qKind {
    Identity,  // π = [0, 1, 2, 3]
    CnotHi,    // π = [0, 1, 3, 2]  — control = t0 (MSB), as Gate::Cnot
    CnotLo,    // π = [0, 3, 2, 1]  — control = t1 (LSB)
    Swap,      // π = [0, 2, 1, 3]
}

pub(crate) fn classify_2q_permutation(
    m: &[[Complex; 4]; 4],
) -> Option<Perm2qKind> {
    let mut perm = [0u8; 4];
    for r in 0..4 {
        // Find the single column with |m[r][c]|² ≥ 1 - EPS_PERM and
        // require ≈ +1+0i (not just ≈ unit modulus — phase ≠ 0 means
        // this is not a "pure" permutation and the generic kernel
        // should handle it).
        let mut hit = None;
        for c in 0..4 {
            let nsq = m[r][c].norm_sqr();
            if nsq < EPS_SQ {
                continue;
            }
            if (nsq - 1.0).abs() < EPS_PERM
                && (m[r][c].re - 1.0).abs() < EPS_PERM
                && m[r][c].im.abs() < EPS_PERM
            {
                if hit.is_some() {
                    return None;
                }
                hit = Some(c as u8);
            } else {
                return None;
            }
        }
        perm[r] = hit?;
    }
    // Reject duplicate columns (not a permutation).
    let mut seen = [false; 4];
    for &c in &perm {
        if seen[c as usize] {
            return None;
        }
        seen[c as usize] = true;
    }
    match perm {
        [0, 1, 2, 3] => Some(Perm2qKind::Identity),
        [0, 1, 3, 2] => Some(Perm2qKind::CnotHi),
        [0, 3, 2, 1] => Some(Perm2qKind::CnotLo),
        [0, 2, 1, 3] => Some(Perm2qKind::Swap),
        _ => None, // e.g. [1,0,3,2] = X⊗I, not a 2q gate we specialise
    }
}
```

Worst-case cost on dense generic gates: full diagonal scan (12 norms)
+ partial permutation scan (≈ 4–16 norms before branching out) ≈ 50 ns.
On QFT-20: ~20 µs total — < 0.01 % of wall-clock.

**`is_cz_signature(d0, d1, d2, d3) -> bool`.** Detects exactly the CZ
phase pattern `(1, 1, 1, -1)` for the `vxorpd` sign-flip shortcut.
Tight test: `(d - target).norm_sqr() < EPS_PERM` for each of the four
expected entries.

```rust
pub(crate) fn is_cz_signature(d: [Complex; 4]) -> bool {
    let close = |z: Complex, target: Complex| (z - target).norm_sqr() < EPS_PERM;
    close(d[0], Complex::new(1.0, 0.0))
        && close(d[1], Complex::new(1.0, 0.0))
        && close(d[2], Complex::new(1.0, 0.0))
        && close(d[3], Complex::new(-1.0, 0.0))
}
```

### 4.2 Dispatch order in `apply_2q`

```rust
pub(crate) fn apply_2q(
    amps: &mut [Complex],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex; 4]; 4],
) {
    // 1. Permutation detection (Identity / CNOT / SWAP).
    match super::classify_2q_permutation(m) {
        Some(super::Perm2qKind::Identity) => return,
        Some(super::Perm2qKind::CnotHi) => {
            apply_2q_cnot(amps, targets[0], targets[1], controls);
            return;
        }
        Some(super::Perm2qKind::CnotLo) => {
            apply_2q_cnot(amps, targets[1], targets[0], controls);
            return;
        }
        Some(super::Perm2qKind::Swap) => {
            apply_2q_swap(amps, targets, controls);
            return;
        }
        None => {}
    }

    // 2. Diagonal-4x4 (catches Cz, controlled-Phase, Rzz, user diagonals).
    if super::is_diagonal_4x4(m) {
        let d = [m[0][0], m[1][1], m[2][2], m[3][3]];
        if super::is_cz_signature(d) {
            apply_2q_cz(amps, targets, controls);
        } else {
            apply_2q_diagonal(amps, targets, controls, d);
        }
        return;
    }

    // 3. Generic dense 4×4 — SIMD where contract holds, scalar otherwise.
    #[cfg(target_arch = "x86_64")]
    {
        let t_lo = targets[0].min(targets[1]);
        let t_hi = targets[0].max(targets[1]);
        if std::is_x86_feature_detected!("avx512f")
            && (1usize << t_lo) >= LANES
            && controls.iter().all(|&c| c > t_hi)
        {
            unsafe { apply_2q_avx512(amps, targets, controls, m); }
            return;
        }
    }
    apply_2q_dense_scalar(amps, targets, controls, m);
}
```

Each specialised entry function (`apply_2q_cnot`, `apply_2q_swap`,
`apply_2q_cz`, `apply_2q_diagonal`) internally picks its SIMD tier or
scalar fallback based on its own contract; the prelude doesn't have to
know about tier B/C.

### 4.3 SIMD safety contract (shared)

Each `*_avx512` entry checks all of these at the call site (debug-asserts
mirror them inside the unsafe function, in the P1-03 style):

- Host supports AVX-512F (`is_x86_feature_detected!("avx512f")`).
- For Tier A paths: `1 << min(targets) ≥ LANES = 4`.
  - For Tier B (`apply_2q_cnot/swap` only): `1 << max(targets) ≥ LANES`.
  - For Tier C: targets `{0, 1}` exactly (covers both orientations).
- Every external `controls[i] > max(targets)` so the inner SIMD walk's
  block offset can never toggle a control bit.
- Standard apply_gate invariants: distinct targets/controls, all in
  qubit range.

If any of those fail, fall through to the kernel's scalar mirror.

### 4.4 AoS — `apply_2q_avx512` (generic dense, Tier A only)

**Math.** For each base index `i` with `i & t_mask == 0`, load the four
amps of the quartet:

```
z00 = state[i],
z01 = state[i | t_lo_bit],
z10 = state[i | t_hi_bit],
z11 = state[i | t_mask]
```

Compute `new_z_r = Σ_c m[r][c] · z_c` for `r ∈ {0..4}`. Each
`m[r][c] · z_c` is one `vfmaddsub(re_bcast, z_c, im_bcast × swap(z_c))`
(the P1-03 packed-complex multiply idiom).

**Inner loop** (one outer block, processes `LANES` quartets = 16 amps):

```
broadcast 16 doubles: m_re[r][c], m_im[r][c]  (one bcast pair per cell)

for j = 0..t_lo_bit step LANES:
    z00 = vmovupd state[(block | j) * 2]
    z01 = vmovupd state[(block | t_lo_bit | j) * 2]
    z10 = vmovupd state[(block | t_hi_bit | j) * 2]
    z11 = vmovupd state[(block | t_mask | j) * 2]

    z00s = vpermilpd<0x55> z00
    z01s = vpermilpd<0x55> z01
    z10s = vpermilpd<0x55> z10
    z11s = vpermilpd<0x55> z11

    for r in 0..4:
        t0 = vmulpd(m_im[r][0], z00s)
        p  = vfmaddsub(m_re[r][0], z00, t0)
        t1 = vmulpd(m_im[r][1], z01s); p = vaddpd(p, vfmaddsub(m_re[r][1], z01, t1))
        t2 = vmulpd(m_im[r][2], z10s); p = vaddpd(p, vfmaddsub(m_re[r][2], z10, t2))
        t3 = vmulpd(m_im[r][3], z11s); p = vaddpd(p, vfmaddsub(m_re[r][3], z11, t3))
        new_z[r] = p

    vmovupd state[(block | j) * 2]            = new_z[0]
    vmovupd state[(block | t_lo_bit | j) * 2] = new_z[1]
    vmovupd state[(block | t_hi_bit | j) * 2] = new_z[2]
    vmovupd state[(block | t_mask | j) * 2]   = new_z[3]
```

**µop count per inner iter (16 complex):** 4 loads + 4 permutes + 16
mul + 16 fmaddsub + 12 add + 4 stores ≈ **56 µops**. Scalar 4-quad
equivalent: ~256 µops. **~4.5× per-amp**.

**Outer-walk.**
- `controls` empty: `block += 4 * t_hi_bit`; iterate `block < len`.
- `controls` present (all > `t_hi`): renormalise each control position
  (`c - t_hi - 1`), use `expand_with_fixed` over the renormalised mask,
  shift result back by `t_hi + 1`. Same idiom as `apply_1q_avx512`
  (aos.rs:234-248) extended with both targets fixed-zero.

**Scalar fallback.** `apply_2q_dense_scalar` — exactly today's
`apply_2q` walk (aos.rs:424-453), renamed and left untouched. Reused
when `t_lo < LANES` or non-AVX-512 host.

### 4.5 AoS — `apply_2q_cnot` (permutation, Tiers A + B + C)

**Math.** Inputs are `(control, target, external_controls)`. CNOT swaps
the amplitude pair `(state[i], state[i | t_bit])` for every `i` where
bit `control = 1` AND `bit target = 0` AND every external control bit
is set. Zero multiplies; pure load/store.

**Convention.** Within the kernel, `target` always names the qubit
defining the inner SIMD walk (the bit that gets toggled by the swap);
`control` is the qubit that gates the outer walk. The dispatch prelude
in `apply_2q` passes the matrix-detected (control, target) order
unchanged, so `CnotHi` calls `apply_2q_cnot(state, targets[0],
targets[1], extra_controls)` and `CnotLo` swaps the target arguments.

**Tier A — `1 << target ≥ LANES`.** Classic swap-pair over LANES amps:

```rust
let t_bit = 1usize << target;
let c_bit = 1usize << control;

for outer in walk_outer(c_bit = 1, t_bit = 0, external_controls, len) {
    for j in (0..t_bit).step_by(LANES) {
        let i0 = outer | j;                // control=1, target=0
        let i1 = i0 | t_bit;               // control=1, target=1
        let a = _mm512_loadu_pd(ptr.add(i0 * 2));
        let b = _mm512_loadu_pd(ptr.add(i1 * 2));
        _mm512_storeu_pd(ptr.add(i0 * 2), b);
        _mm512_storeu_pd(ptr.add(i1 * 2), a);
    }
}
```

**µop count per inner iter (LANES = 4 complex):** 2 loads + 2 stores =
**4 µops**. Per amp: **1.0**. Vs scalar generic-2q quadruplet
(~64 µops / 4 amps = 16 µops/amp): **16× per-amp**. Comfortably clears
BACKLOG-AC "5–10× faster than generic 2q".

**Tier B — `1 << target < LANES ≤ 1 << control`.** Inner-walk dimension
has fewer than LANES contiguous amps with matched (control, target)
bits, so a single LANES-wide `vmovupd` straddles the t_bit boundary
and contains amps with mixed target-bit values. Use `vpermt2pd` with
a precomputed index vector to swap the appropriate doubles in-register.

The permute-index for SWAP-style positions within a zmm is keyed by
`target` (since `target ∈ {0, 1}` is the only case Tier B serves here:
LANES = 4 ⇒ `1 << target ∈ {1, 2}` ⇒ `target ∈ {0, 1}`).

```
target = 0:
    # within each 4-amp zmm half, swap pos 0↔1 and pos 2↔3 (target bit toggle)
    # then mask by control: only do the swap on the half where control=1
    permute_idx = [2, 3, 0, 1, 6, 7, 4, 5]   # double-indexed (re,im pairs)

target = 1:
    # within each 4-amp zmm half, swap pos 0↔2 and pos 1↔3
    permute_idx = [4, 5, 6, 7, 0, 1, 2, 3]
```

Inner-walk for Tier B (target = 0, control ≥ 2):

```
c_bit = 1 << control
outer_step = c_bit << 1   # advance past one (control=0, control=1) pair

for outer in (0..len).step_by(outer_step):
    base = outer | c_bit                              # control=1 half
    for j in (0..c_bit).step_by(LANES):
        i = base | j
        z = vmovupd state[i * 2]
        z = vpermt2pd(z, permute_idx_target_0, z)
        vmovupd state[i * 2] = z
```

**µop count per inner iter (LANES amps):** 1 load + 1 permute + 1 store
= **3 µops**.

**Tier C — `1 << max(control, target) < LANES`, i.e. both qubits in
`{0, 1}`.** One zmm holds exactly one quartet (4 complex amps = 8 f64
= 1 zmm). Single in-register lane-permute per zmm; outer-walk steps by
4 amps (one quartet at a time).

```
permute_idx_c0_t1 = [0, 1, 6, 7, 4, 5, 2, 3]   # control = q0, target = q1
permute_idx_c1_t0 = [0, 1, 2, 3, 6, 7, 4, 5]   # control = q1, target = q0
```

Inner-walk:

```
for quartet_base in (0..len).step_by(4):
    z = vmovupd state[quartet_base * 2]
    z = vpermt2pd(z, permute_idx_<orient>, z)
    vmovupd state[quartet_base * 2] = z
```

**µop count per zmm (one quartet = 4 amps):** 3 µops; per amp:
**0.75 µops**.

**Scalar fallback.** All cases where SIMD contract doesn't hold:

```rust
let c_bit = 1usize << control;
let t_bit = 1usize << target;
let ctrl_mask = c_bit | super::control_mask(external_controls);
for i in 0..amps.len() {
    if (i & ctrl_mask) == ctrl_mask && (i & t_bit) == 0 {
        amps.swap(i, i | t_bit);
    }
}
```

LLVM auto-vec lifts the swap into 2-lane xmm where it can.

### 4.6 AoS — `apply_2q_swap` (permutation, Tiers A + B + C)

**Math.** SWAP[a, b]: for each base `i` with `i & t_mask == 0`, swap
`state[i | a_bit] ↔ state[i | b_bit]`. Amps with bits `(a, b) ∈ {(0,0),
(1,1)}` are unchanged.

**Tier A — `1 << min(a, b) ≥ LANES`.** Iterate outer blocks; within
each, swap LANES amps from the `(a=0, b=1)` sub-block with LANES amps
from the `(a=1, b=0)` sub-block.

```rust
let lo_bit = 1usize << lo;     // lo = min(a, b)
let hi_bit = 1usize << hi;
let outer_step = (hi_bit << 1) ;   // skip past full (hi, lo) quartet space

for outer in (0..len).step_by(outer_step) {
    let base_01 = outer | lo_bit;          // (hi=0, lo=1)
    let base_10 = outer | hi_bit;          // (hi=1, lo=0)
    for j in (0..lo_bit).step_by(LANES) {
        let a_vec = vmovupd state[(base_01 | j) * 2];
        let b_vec = vmovupd state[(base_10 | j) * 2];
        vmovupd state[(base_01 | j) * 2] = b_vec;
        vmovupd state[(base_10 | j) * 2] = a_vec;
    }
}
```

**µop count per inner iter:** 2 loads + 2 stores = **4 µops** for LANES
amps. Per amp: **1.0**.

**Tier B / Tier C.** Analogous to CNOT but with the
(a=0, b=1) ↔ (a=1, b=0) swap pattern instead of CNOT's (control=1, t=0)
↔ (control=1, t=1).

Permute-index tables (Tier C, targets `{0, 1}`):

```
permute_idx_swap_q0_q1 = [0, 1, 4, 5, 2, 3, 6, 7]   # swap doubles 2,3 ↔ 4,5
```

External controls handled the same way as CNOT (walk fixed by
`expand_with_fixed`).

**Scalar fallback** equivalent to CNOT's, with the swap condition
`((i >> a) & 1) == 0 && ((i >> b) & 1) == 1`.

### 4.7 AoS — `apply_2q_cz` (sign-flip, Tier A)

**Math.** For each amp `state[i]` with bits `(t0, t1) == (1, 1)` (and
all external controls = 1): negate. 75 % of state is skipped; only the
`(t_hi = 1, t_lo = 1)` sub-block is touched.

**Inner loop:**

```rust
let sign_mask = _mm512_set1_pd(-0.0_f64);
let t_lo_bit = 1usize << t_lo;
let t_hi_bit = 1usize << t_hi;
let outer_step = t_hi_bit << 1;

for outer in (0..len).step_by(outer_step) {
    let base_11 = outer | t_hi_bit | t_lo_bit;
    for j in (0..t_lo_bit).step_by(LANES) {
        let i = base_11 | j;
        let z = _mm512_loadu_pd(ptr.add(i * 2));
        let z = _mm512_xor_pd(z, sign_mask);
        _mm512_storeu_pd(ptr.add(i * 2), z);
    }
}
```

**µop count per inner iter (LANES amps):** 1 load + 1 xor + 1 store =
**3 µops**. Per amp: **0.75 µops**. Plus the 4× memory-bandwidth
reduction from touching only 1/4 of state.

**Scalar fallback** — same walk, scalar negate (`amps[i] = -amps[i]`).

**No Tier B/C.** When `1 << min(t0, t1) < LANES`, scalar fallback is
plenty fast (CZ touches 1/4 of state, so even at n = 20 the scalar
loop is tiny).

### 4.8 AoS — `apply_2q_diagonal_avx512` (general diagonal, Tier A)

**Math.** `state[i] *= d[k]` where `k = (((i >> t0) & 1) << 1) | ((i >>
t1) & 1)` — four multipliers gated by the bit-pattern. Each sub-block
gets one `d` value and runs the AVX-512 packed-complex multiply (same
5-µop shape as `apply_1q_diagonal_avx512` from P1-06).

**Outer-walk.** Iterate four sub-blocks per outer step (one per
`(t_hi, t_lo)` bit-pattern), choosing the appropriate `d[k]` for each.

```rust
for outer in (0..len).step_by(outer_step) {
    for (kind_bits, &d_k) in [(0, d[0]), (lo_bit, d[1]),
                              (hi_bit, d[2]), (mask, d[3])].iter() {
        let base = outer | kind_bits;
        let d_re_bc = _mm512_set1_pd(d_k.re);
        let d_im_bc = _mm512_set1_pd(d_k.im);
        for j in (0..t_lo_bit).step_by(LANES) {
            let z = _mm512_loadu_pd(ptr.add((base | j) * 2));
            let z_swap = _mm512_permute_pd::<0x55>(z);
            let t = _mm512_mul_pd(d_im_bc, z_swap);
            let z = _mm512_fmaddsub_pd(d_re_bc, z, t);
            _mm512_storeu_pd(ptr.add((base | j) * 2), z);
        }
    }
}
```

**µop count per inner iter (LANES = 4 complex):** 1 load + 1 permute +
1 mul + 1 fmaddsub + 1 store = **5 µops**. Per amp: **1.25 µops**.
Vs generic-2q SIMD (~3.5 µops/amp): **~2.8×**.

**Scalar fallback** — index-by-index `d` lookup, single multiply.

### 4.9 SoA mirror (per the hybrid decision)

| Path | SoA implementation |
|------|---|
| Generic 2q dense | Scalar only (existing walk). LLVM auto-vec is poor here (ADR 0008 4-stream anti-pattern); not worth explicit SIMD. |
| CNOT | Explicit AVX-512: parallel swap-pair on `re[..]` and `im[..]` streams (two `vmovupd zmm` per stream, four total). Tiers A + B + C identical structure, just operating on two `f64` arrays. |
| SWAP | Same as CNOT pattern, two `f64` arrays. |
| CZ | Explicit AVX-512: `vxorpd` on both `re[..]` and `im[..]` slices (sign-flip applies independently per stream). |
| 2q diagonal | Explicit AVX-512: same 2-stream cross-multiply shape as P1-06's `apply_1q_diagonal_soa`, extended to 4 sub-block multipliers. |
| Identity | No-op, return immediately. |

`LANES_SOA = 8` (zmm with packed f64). All SIMD-tiers enabled when
`1 << t_lo ≥ LANES_SOA`, host AVX-512F, controls > t_hi. The same
`Perm2qKind` / `is_diagonal_4x4` helpers from `kernels/mod.rs` are
reused — the dispatch prelude in `kernels/soa.rs::apply_2q` mirrors
§ 4.2.

### 4.10 Why matrix-detection over gate-tag dispatch

Same three options as P1-06 § 4.5; same answer:

1. **Matrix detection in kernel (chosen).** Catches user-supplied
   `GenericUnitary([[Complex; 4]; 4])` that's structurally CNOT/SWAP/
   diagonal, and any 2q matrix coming out of future IR-fusion passes
   (P1-09/P1-10) that synthesises diagonals or permutations. No backend
   churn; kernels stay gate-tag-agnostic.
2. **Gate-tag dispatch in backend.rs.** Zero runtime detection cost,
   but user `GenericUnitary` and fused matrices get no benefit, and
   `backend.rs` must learn each new `Gate` variant. Rejected.
3. **Hybrid (gate-tag hint + kernel verify).** Same drawbacks as P1-06.
   Not worth ~50 ns/gate. Rejected.

This continues the layering established by ADR 0009.

### 4.11 Why both CnotHi and CnotLo

`Gate::Cnot` always emits the `CnotHi` matrix (rows 2↔3 swap) per
ADR 0004's MSB convention and P0-06's matrix spec. `CnotLo` (rows 1↔3
swap) is only hit by user-supplied `GenericUnitary` or future
IR-fusion-output in the "control = low qubit" orientation. The extra
detection arm is ~20 lines + 1 test; cost negligible. Keeping it
preserves the matrix-detection layer's gate-tag-agnostic property.

## 5. Acceptance criteria

### Implementation

- [ ] `kernels/mod.rs::is_diagonal_4x4` implemented + unit-tested
      (exact zeros, FP-noise tolerance, non-diagonal rejection).
- [ ] `kernels/mod.rs::Perm2qKind` enum + `classify_2q_permutation`
      implemented + unit-tested (all four canonical perms, non-perm
      rejection, non-canonical-phase rejection, duplicate-column reject).
- [ ] `kernels/mod.rs::is_cz_signature` implemented + unit-tested.
- [ ] `kernels/aos.rs::apply_2q_avx512` (Tier A generic dense) under
      `#[cfg(target_arch = "x86_64")]` + `#[target_feature(enable =
      "avx512f")]`, safety-contract comment matching `apply_1q_avx512`.
- [ ] `kernels/aos.rs::apply_2q_cnot_avx512` Tiers A + B + C.
- [ ] `kernels/aos.rs::apply_2q_swap_avx512` Tiers A + B + C.
- [ ] `kernels/aos.rs::apply_2q_cz_avx512` (Tier A).
- [ ] `kernels/aos.rs::apply_2q_diagonal_avx512` (Tier A).
- [ ] Scalar fallbacks for each of the 5 specialised paths.
- [ ] `kernels/aos.rs::apply_2q` prelude per § 4.2.
- [ ] Symmetric SoA paths per § 4.9, with `kernels/soa.rs::apply_2q`
      prelude mirroring § 4.2.
- [ ] `apply_2q_dense_scalar` extracted from the existing scalar walk
      (no behavioural change; just a rename + symmetry with `apply_1q`'s
      structure).

### Tests

Inline (`#[cfg(test)]` in `kernels/aos.rs` and `kernels/soa.rs`):

- [ ] `apply_2q_cnot_matches_generic_canonical` (Gate::Cnot matrix,
      Tier A target).
- [ ] `apply_2q_cnot_matches_generic_lo_orientation` (CnotLo matrix).
- [ ] `apply_2q_cnot_tier_b_matches_scalar` (target = 0, control = 2).
- [ ] `apply_2q_cnot_tier_c_matches_scalar` (targets `{0, 1}`,
      both orientations).
- [ ] `apply_2q_swap_matches_generic` (Gate::Swap matrix, Tier A).
- [ ] `apply_2q_swap_tier_b_matches_scalar`.
- [ ] `apply_2q_swap_tier_c_matches_scalar`.
- [ ] `apply_2q_cz_matches_generic` (Gate::Cz matrix).
- [ ] `apply_2q_identity_is_noop`.
- [ ] `apply_2q_diagonal_random_phases_matches_generic`
      (`prop_for_each 32`, random θ-grid, diag-only).
- [ ] `apply_2q_almost_diagonal_falls_through_to_generic` (off-diag
      magnitude 1e-8, expect generic dispatch).
- [ ] `apply_2q_almost_permutation_falls_through_to_generic`
      (e.g. `|m[r][c]|² = 0.999`, expect generic dispatch).
- [ ] `apply_2q_dense_random_unitary_matches_scalar`
      (`prop_for_each 32`, n ∈ {6..10}, AoS-SIMD vs AoS-scalar).
- [ ] `apply_2q_dispatch_overhead_dense` micro-bench: 1000 sequential
      dense 2q gates, expect SIMD-path overhead < 2 %.

New `aleph-test` strategies (called from both backends' inline tests):

- [ ] `arb_diagonal_4x4(n)` — random complex diag (n ∈ {6..10}).
- [ ] `arb_cnot_matrix(orientation)` — CnotHi / CnotLo permutations.
- [ ] `arb_swap_matrix()` — Swap permutation.

Cross-backend / oracle (existing harness):

- [ ] `all_fixtures_match_naive` (112 fixtures, AoS ≡ SoA within 1e-12)
      passes unchanged.
- [ ] Qiskit oracle harness passes unchanged.

### Lint / format

- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.

### Benchmarks (EPYC bencher.dev)

- [ ] **Hard AC**: `qft_n20` ≥ **1.20×** faster than post-P1-06
      baseline (1133 ms → ≤ 944 ms ≈ 2.06× Aer).
- [ ] **Soft AC** (reported in PR-body, not a merge gate): `qft_n20`
      1.30× ambition (≤ 870 ms ≈ 1.89× Aer, clears ROADMAP exit at
      n = 20).
- [ ] **Hard AC**: no regression on `grover_n20_iters5` (within 1.05×
      of 79 033 ms).
- [ ] **Hard AC**: `random_brickwall_n20_d20` ≥ **1.05×** faster
      (CZ + diagonal pickup; circuit has small mixed-2q content).
- [ ] **Micro AC** (`benches/p1_07_microbench.rs`):
      `cnot_n20_specialized` ≥ **5×** `cnot_n20_via_generic`
      (BACKLOG-stated 5–10× target).
- [ ] No regression on `bell`, `ghz`, `qft/{10,15}` (within 1.05×).
- [ ] SoA-path benches (`qft_n20_soa`, `grover_n20_soa`): no regression
      within 1.05× of post-P1-06 baseline.

### Docs

- [ ] `docs/decisions/0010-2q-specialised-paths.md` ADR committed,
      documenting: dispatch tree, three-tier SIMD coverage, why matrix
      detection > gate-tag dispatch (extends ADR 0009), CnotLo
      inclusion rationale.
- [ ] PR body includes EPYC bench numbers: before/after for `qft_n20`,
      `grover_n20_iters5`, `random_brickwall_n20_d20`, micro
      `cnot_specialized` vs `cnot_via_generic`.

## 6. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Bandwidth-bound at n ≥ 20 — µop reduction doesn't translate to wall-clock (P1-06 lesson) | CNOT/SWAP also **reduce bandwidth** (touch only the half of state where control = 1 / one half of the swap pair, no extra reads); CZ touches only 1/4. Structurally stronger than P1-06's diagonal-1q (which still touched 100 % of state). If a regression still appears: perf-stat forensic (P1-06 protocol) + consider `[meta]` rollback / fold-back like P1-02 → P1-03. |
| Tier B/C lane-permute index tables — error-prone | Inline 8-double permute-index constants with bit-pattern derivation in comments; one property test per (tier, orientation) pair; matching scalar fallback as ground truth. |
| Detection cost on dense gates higher than 50 ns expected | `apply_2q_dispatch_overhead_dense` micro (1000 dense 2q gates sequentially) gates this — < 2 % overhead is the AC. If ≥ 2 %, add an optional `Gate`-tag hint plumbing without changing the kernel layering. |
| `CnotLo` detection false-positive on FP-noise unitary | `EPS_PERM = 1e-14` requires `|m[r][c]|² ≥ 1 - 1e-14` AND `(re - 1).abs() < 1e-14` AND `im.abs() < 1e-14` — any "almost-permutation" with off-diagonal magnitude ≥ 1e-7 fails the diagonal pre-test. Test `apply_2q_almost_permutation_falls_through_to_generic` pins this. |
| Bencher.dev runner stuck (Stage 0 lesson) | Same protocol: do not push to `benches/**` during manual EPYC measurement; `systemctl restart` broker if hung. Document in PR body operational issues. |
| Callsites depending on Identity matrices producing side-effects (unlikely but worth checking) | Audit: grep for `GenericUnitary` with 4×4 args across `crates/`; verify none constructs identity-shaped 2q. Add `apply_2q_identity_is_noop` test as documentation. |
| QFT-20 < 1.20× hard AC | Forensic perf-stat (load µops, L2 misses); document as "correct but bandwidth-limited"; do not merge until either (a) Tier-B/C consolidation finds a remaining µop win, or (b) the 2-stream interleave / blocked walk experiment yields ≥ 1.20×. Worst case: roll back like P1-02 with an ADR. |
| Unexpected interaction with controlled-`p` in transpiled QFT | QFT Qiskit-transpile output decomposes controlled-`p` into `p` + `cx` + `p` + `cx` + `p` (verified P1-06 gate-mnemonic histogram). All flow through 1q + 2q paths cleanly. No new gate matrix lands in `apply_2q` from QFT. |

## 7. Open questions (deferred, not blockers)

1. **Adjacent-pair-specific 2q kernel** (variant B from brainstorm).
   When `t_hi = t_lo + 1`, the four amps of a quartet are contiguous
   (16 bytes apart × 4 = one cache line). A specialised inner-walk that
   loads 8 contiguous f64 + shuffles, instead of 4 separate `vmovupd`,
   could win ~10–15 % on the 30–40 % of QFT-20 cx pairs that satisfy
   `t_hi = t_lo + 1`. **Defer to a follow-up ticket** if the 1.30×
   ambition is unreached.
2. **AVX2 path** for pre-Skylake-X / pre-Zen-4 hosts. Out of scope;
   separate `[infra]` ticket if Intel-laptop perf is prioritised.
3. **iSWAP / sqrt-SWAP specialisations.** Out of scope until a Phase 2+
   workload exercises them.
4. **`SoaSvBackend` removal (ADR 0008 open Q#2).** P1-07 ships
   symmetric specialised paths; the kill-SoA decision stays deferred to
   Phase 1 closure (P1-14).

## 8. Workflow

Per the established per-ticket workflow (P0-06 onwards): brainstorm
(done) → spec (**this doc**) → plan → execute → request code review →
fix → squash-merge.

Next step after spec approval: invoke `writing-plans` to author
`docs/superpowers/plans/2026-05-27-p1-07-2q-kernel.md`.
