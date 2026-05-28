# P1-08 — Multi-controlled gate kernels (Toffoli, CCZ, MCX) — Design

**Date:** 2026-05-28
**Backlog issue:** `[P1-08] Multi-controlled gate kernels (Toffoli, CCZ, MCX)`
**Phase:** 1 — Stage 1 (final SIMD ticket)
**Depends on:** P1-05 (anti-diagonal 1q), P1-06 (diagonal 1q), P1-07 (2q + CNOT/CZ/SWAP).
**ADRs touched in scope:** new ADR 0012 (multi-controlled SIMD pattern); confirms ADRs 0007/0008 (AoS+AVX-512 substrate).

---

## 1 — Spec amendment vs BACKLOG `[P1-08]`

The original BACKLOG entry (lines 907–937) was written before ADR 0008. It says only "specialized kernels for CCX, CCZ, MCX". This design **amends** the spec to:

1. **Substrate is AoS + AVX-512 packed-complex** (per ADR 0008). Fresh kernels live in `crates/aleph-sv/src/kernels/aos.rs` alongside the existing `dispatch_cnot`, `dispatch_cz`, `dispatch_swap` from P1-07. Symmetric implementation in `kernels/soa.rs` for non-x86 / SoA-host parity.
2. **MCX with `k` controls is NOT a separate kernel** — it is `Gate::X` with extra `controls`, dispatched through `apply_1q` to P1-05's specialised anti-diagonal kernel. P1-05's kernels already accept arbitrary `controls.len()` via `control_mask(controls)`. MCX is verified via a **new benchmark** (`mcx_k{2,4,6}_n20`), not a new kernel.
3. **Acceptance criterion "3–5× faster than decomposed equivalent"** is replaced by:
   - Micro-AC: Toffoli specialised ≥ **1.5×** vs scalar generic `apply_3q` on L2-resident `toffoli_chain_n15`. CCZ ≥ **2×** vs scalar generic on `ccz_chain_n15`.
   - Workload-AC: no regression > 2 % on `qft/grover/random` n=20 (anti-regression check on the `apply_3q` prelude).
4. **Generic MCX with up to 8 controls** — implicit via P1-05; `mcx_k6_n20` bench is the explicit verification anchor. Anything beyond k=6 is uncharted and not required for Phase-1 exit.

The original BACKLOG bullets get updated when this design lands (in the same `[P1-08]` PR).

---

## 2 — Strategic context

Phase-1 exit (ROADMAP §7, ≤ 2× Qiskit Aer single-thread for QFT/Grover/random at n=25) was **already cleared** as of P1-07 merge (qft_n20 1.30× Aer, grover -25.9 %, random -21 %). P1-08 is the final Stage-1 ticket and ships per the user-explicit "full Phase-1 backlog" decision (see [phase1-completion-plan](../../../docs/superpowers/plans/2026-05-26-phase1-completion.md) line 3).

**Workload coverage:**
- `qft_n20` — 0 Toffoli, 0 CCZ → expected wall-clock change ~0 % (anti-regression only).
- `random_brickwall_n20` — 0 Toffoli, 0 CCZ → ~0 %.
- `grover_iter5_n20` — 5 CCZ instances (one per Grover iteration as the diffusion-phase oracle) → small win (~1–3 %).

Real perf signal is on **new synthetic benches** (`toffoli_chain_n15/n20`, `ccz_chain_n15/n20`). Workload benches are the anti-regression net.

---

## 3 — Architecture

### 3.1 Dispatch table at `apply_3q` prelude

Top-level `apply_3q` in `kernels/aos.rs` gains a matrix-shape detector dispatch, mirroring `apply_2q`'s P1-07 prelude:

```rust
pub(crate) fn apply_3q(
    amps: &mut [Complex],
    targets: [u32; 3],
    controls: &[u32],
    m: &[[Complex; 8]; 8],
) {
    if super::is_identity_8x8(m) {
        return;
    }
    if super::is_toffoli(m) {
        dispatch_toffoli(amps, targets, controls);
        return;
    }
    if super::is_ccz(m) {
        dispatch_ccz(amps, targets, controls);
        return;
    }
    // Existing scalar-generic 3q kernel.
    apply_3q_generic(amps, targets, controls, m);
}
```

Detection is matrix-first (not `Gate::Toffoli` enum-based), so any future backend / IR pass that materialises a Toffoli-shaped 8×8 matrix from a different `Gate` variant (e.g. transpiled MCX of `Gate::X` lifted into 3q) hits the fast path.

### 3.2 Shape detectors (`kernels/mod.rs`)

Add:

```rust
pub(crate) fn is_identity_8x8(m: &[[Complex; 8]; 8]) -> bool { ... }
pub(crate) fn is_toffoli(m: &[[Complex; 8]; 8]) -> bool { ... }
pub(crate) fn is_ccz(m: &[[Complex; 8]; 8]) -> bool { ... }
```

**Toffoli detection:** rows 0..=5 == identity rows; row 6 == row 7-of-identity (e6 → e7); row 7 == row 6-of-identity (e7 → e6). All entries compared with `ABS_TOL = 1e-12`.

**CCZ detection:** diagonal with `d[k] == Complex::ONE` for k in 0..=6 and `d[7] == Complex::NEG_ONE`; off-diagonal entries below `ABS_TOL`. (Symmetric in qubit order — every permutation of `[q0, q1, q2]` produces the same 8×8 matrix because the matrix is diagonal.)

`is_identity_8x8` is dead-simple but worth adding because the existing generic kernel happily wastes cycles on identity; the prelude no-op shaves it.

### 3.3 Notation (pinned conventions)

- `LANES = 8` — doubles per zmm register (one AVX-512 zmm = 8 × `f64`).
- `LANES_AMPS = 4` — complex amplitudes per zmm (each amp is 2 doubles in AoS layout).
- `LANES_BITS = log2(LANES_AMPS) = 2` — the bit-position threshold for "target inside zmm".
- All dispatch contracts use the form `target_bit >= LANES` measured in **doubles-offset** (i.e. `(1<<t) * 2 >= LANES`, ≡ `t >= LANES_BITS+1 = 3`). This matches P1-05/P1-07 conventions verbatim — don't introduce a new arithmetic.
- Bit-position comparisons (`c_lo > t`) are over **qubit indices**, not byte/double offsets.

### 3.4 Tolerance constant

Reuse `DIAGONAL_EPS_SQ` from P1-06 for CCZ; reuse `PERM_TOL` from P1-05 (or its equivalent) for Toffoli's permutation-pattern check. **Lesson from P1-05 review:** keep `DIAGONAL_EPS_SQ` and `PERM_TOL` as **separate** constants (don't unify) — diagonal-magnitudes test wants tighter tolerance than permutation-shape.

---

## 4 — Toffoli (CCX) kernel

### 4.1 Math + dispatch contract

`CCX(c0, c1, t)` with external controls `ext = [e0, e1, ...]` performs:
```
for all i where (i & ctrl_mask) == ctrl_mask:
    swap(amps[i], amps[i ^ target_bit])
```
where `ctrl_mask = (1<<c0) | (1<<c1) | OR(1<<ext_k)` and `target_bit = 1<<t`.

Pre-dispatch sort:
- `ctrl_mask: u64` is built from `{c0, c1} ∪ ext` (de-duplicated, validated upstream).
- `c_lo = min` of those bit-positions.
- `c_hi = max` of those bit-positions.

### 4.2 Tier A — packed AVX-512 swap

**Contract:**
- `target_bit >= LANES` (== `t >= 3`).
- `c_lo > t` (every control bit strictly above target).
- `n >= 4` (state ≥ 16 amps, ≥ 2 LANES blocks).

**Inner loop:**
```rust
for block_base in (0..len).step_by(LANES).filter(in_outer_walk) {
    if (block_base & ctrl_mask) != ctrl_mask { continue; }
    let z_lo = _mm512_loadu_pd(amps_ptr.add(block_base));
    let z_hi = _mm512_loadu_pd(amps_ptr.add(block_base | target_bit));
    _mm512_storeu_pd(amps_ptr.add(block_base), z_hi);
    _mm512_storeu_pd(amps_ptr.add(block_base | target_bit), z_lo);
}
```

Mirrors `dispatch_cnot` Tier-A in P1-07 (aos.rs § ~1929) with `ctrl_mask` covering 2+ inner controls plus externals.

**Outer-walk for controls below target:** if user-supplied external controls include a `c < t`, the Tier-A contract fails. We extend to "outer-walk over fixed bits below `t`" — the **canonical pattern** from P1-07 (lesson logged in [P1-07 merged](../../../crates/aleph-sv/src/kernels/aos.rs) memory): `expand_with_fixed(k, &fixed_below) << (t_lo + 1)` with renormalised bit-positions for the controls above. Tier-C fallback is only for **sub-LANES** state size, not for "controls below target".

**Expected µops/amp:** 0.5 SIMD ops + branch on mask check per block.

### 4.3 Tier B — target inside LANES (`t < 3`)

**Contract:**
- `t < 3` (target_bit < LANES_PAIRS=4)
- `c_lo >= 3` (controls still above LANES boundary)
- `n >= 3`

**Inner loop:** load 4-complex zmm; `_mm512_permutexvar_pd` with target-dependent index constants:
- **Tier B.0** (`t = 0`) — swap pair-of-doubles within zmm: index `(2,3, 0,1, 6,7, 4,5)`.
- **Tier B.1** (`t = 1`) — swap two-pair-groups in the LOW 256, two-pair-groups in the HIGH 256: index `(4,5, 6,7, 0,1, 2,3)`. Pure in-zmm permute.
- **Tier B.2** (`t = 2`, cross-256) — swap LOW 256 ↔ HIGH 256. Cross-256 permute via `_mm512_shuffle_f64x2(z, z, 0b01_00_11_10)` (1-µop variant); validate codegen emits `vshuff64x2`, fall back to `_mm512_permutexvar_pd` if the shuffle compiles down to multiple µops.

These three sub-tiers share the same outer loop + mask-check skeleton; only the permute constant changes. Implementation: one kernel function with a runtime `match t { 0 => ..., 1 => ..., 2 => ... }` selecting the permute constant, **OR** three named kernels routed from `dispatch_toffoli` — pick whichever LLVM auto-resolves more cleanly during impl (revisit at codegen-inspection task in the plan).

Mask-check on `block_base` (single ctrl-mask comparison per block, because `c_lo >= 3` means the full block lies under one fixed ctrl-bit pattern).

**Expected µops/amp:** ~0.75 (1 load + 1 permute + 1 store per 4 amps).

### 4.4 Tier C — scalar fallback

**Fires when:**
- `n < 3` (state < 8 amps).
- Any other degenerate case not covered by the Tier-A outer-walk extension.

Implementation: straight loop:
```rust
for i in 0..1usize << n {
    if (i & ctrl_mask) == ctrl_mask && (i & target_bit) == 0 {
        amps.swap(i, i | target_bit);
    }
}
```

---

## 5 — CCZ kernel

### 5.1 Math + dispatch contract

`CCZ(q0, q1, q2)` with external controls `ext` performs:
```
ccz_mask = (1<<q0) | (1<<q1) | (1<<q2) | OR(1<<ext_k)
for all i where (i & ccz_mask) == ccz_mask:
    amps[i] = -amps[i]
```

Fully symmetric in the three qubits and in externals. No "target" — every bit in `ccz_mask` is equivalent.

Pre-dispatch:
- `ccz_mask: u64` built from `{q0, q1, q2} ∪ ext`.
- `mask_lo = min(bit-positions in ccz_mask)`.

### 5.2 Tier A — AVX-512 in-block sign-flip

**Contract:**
- `mask_lo >= LANES_BITS` (== `mask_lo >= 3`), so every 4-pair zmm block has a single value of `ccz_mask` bits.
- `n >= 4` (state ≥ 16 amps).

**Inner loop:**
```rust
let sign_mask = _mm512_set1_pd(-0.0);  // 0x8000000000000000 in all 8 lanes
for block_base in (0..len).step_by(LANES).filter(in_outer_walk) {
    if (block_base & ccz_mask) != ccz_mask { continue; }
    let z = _mm512_loadu_pd(amps_ptr.add(block_base));
    let neg = _mm512_xor_pd(z, sign_mask);
    _mm512_storeu_pd(amps_ptr.add(block_base), neg);
}
```

`_mm512_xor_pd` on the sign bit is **1 µop latency-1**, cheaper than `_mm512_mul_pd` by `-1.0` (4 µop latency).

**Outer-walk for `mask_lo < 3`:** mirror P1-06's diagonal-1q pattern — outer-walk over bits below LANES_BITS, fold them into the iteration's base offset, and Tier-A SIMD on the remaining bits. Same canonical renormalisation as Toffoli §4.2.

**Expected µops/amp:** 0.375 (1 load + 1 xor + 1 store per 8 amps).

### 5.3 No Tier B for CCZ

CCZ is structurally simpler than Toffoli — no permute needed because no swap. The "in-zmm permute" tier that Toffoli needs (for `t < LANES_BITS`) does not apply: if some mask bits are below LANES_BITS, the response is **outer-walk** (Tier A extension), not in-zmm permute (Tier B).

### 5.4 Tier C — scalar fallback

**Fires when:** `n < 3`. (For `mask_lo < 3` with `n >= 3`, the Tier-A outer-walk extension covers it.)

```rust
for i in 0..1usize << n {
    if (i & ccz_mask) == ccz_mask {
        amps[i] = -amps[i];
    }
}
```

---

## 6 — SoA mirror

Symmetric implementations in `kernels/soa.rs` for non-x86 hosts (the SoA backend is the dispatched default on aarch64 / SSE-only x86). The SoA path:
- Toffoli: swap `(re[i], re[i^t]) ↔ (re[i^t], re[i])` plus the same for `im[]`. AVX-512 packed-double SIMD with 8 doubles per zmm (lanes = 8 for one stream, processing 8 amps per zmm pair). Tier A/B/C same shape as AoS.
- CCZ: per-stream sign-flip via `_mm512_xor_pd` on both `re` and `im`. Tier A only.

Lesson from P1-07 ([P1-07 merged](../../../crates/aleph-sv/src/kernels/soa.rs) entry): SoA Tier-C sub-LANES handling must guard against `external_control - sentinel` underflow. Specifically: re-validate that `mask_lo >= LANES_SOA_BITS` before Tier-A entry; default to Tier-C scalar otherwise, never the unsafe Tier-A path.

---

## 7 — Testing strategy

### 7.1 Unit tests (`kernels::aos::tests`)

**Toffoli:**
- All 8 basis states under `CCX(0, 1, 2)` on `n=3` → expect `|110⟩ ↔ |111⟩`, identity elsewhere.
- External controls: `CCX(0, 1, 2) + ctx=[3]` on `n=4` → swap only when q3=1.
- Tier-boundary cases: `t=0` (Tier B.0 in-zmm swap), `t=1` (Tier B.1 dual-256), `t=2` (Tier B.2 cross-256), `t=3` (Tier A entry), `t=4+` (Tier A clean), control below target (Tier A outer-walk), `n=3/4` (sub-LANES → Tier C).

**CCZ:**
- All 8 basis states under `CCZ(0,1,2)` on `n=3` → expect sign flip on `|111⟩`.
- External controls: `CCZ(0,1,2) + ctx=[3]` on `n=4` → sign flip only on `|1111⟩`.
- Symmetry: `CCZ(0,1,2) == CCZ(1,0,2) == CCZ(2,0,1)` (assert equal output states).
- Tier-boundary: `mask_lo=0,1,2` (Tier A outer-walk), `mask_lo>=3` (Tier A clean), `n=3/4` (Tier C).

### 7.2 Indexing-coverage tests (integer-only, exhaustive)

Mirror the pattern from P1-07 / P1-05 post-review. Catches bit-collision bugs that SIMD-only tests miss (recall EPYC SIGSEGV in P1-07 Task 14 on n=2):

```rust
#[test]
fn toffoli_indexing_pairwise_disjoint_all_configs() {
    for c0 in 0..7 {
        for c1 in 0..7 {
            for t in 0..7 {
                if c0 == c1 || c0 == t || c1 == t { continue; }
                for ext in subsets_of_size_le_2_from({0..7} \ {c0,c1,t}) {
                    // assert dispatch contract classification matches expected tier
                    // assert (block | offsets[k] | j) bits are pairwise disjoint
                }
            }
        }
    }
}
```

Same shape for CCZ (with the mask-symmetry handled).

### 7.3 Property tests (proptest)

In `crates/aleph-test/`:
- `prop_ccx_involutive`: rand state, apply CCX twice → original within 1e-12.
- `prop_ccz_involutive`: same.
- `prop_ccx_equiv_generic`: rand state + rand `(c0,c1,t)` + rand `ext.len() ∈ {0,1,2}` → specialised result ≡ scalar-generic result within 1e-12.
- `prop_ccz_equiv_generic`: same.
- `prop_ccz_qubit_symmetry`: rand state, every permutation of `(q0,q1,q2)` produces equal output.

### 7.4 Oracle tests (vs Qiskit Aer)

Through existing oracle harness:
- 3-qubit pure-Toffoli circuit on all 8 basis inputs.
- 4-qubit CCCX (Toffoli + 1 external control) on all 16 basis inputs.
- Grover-style 3-qubit diffusion built around CCZ on `|+++⟩` start.
- **MCX with `k=7` controls** (`Pauli-X` lifted to 8-qubit circuit) — validates that P1-05's anti-diagonal kernel handles large `controls.len()` correctly without regression. This is the **MCX verification anchor** for the "MCX up to 8 controls" backlog bullet.

Tolerance: 1e-10 amplitudes (FP64), per `docs/testing.md`.

---

## 8 — Benchmarks

### 8.1 New benches (`crates/aleph-sv/benches/`)

- `toffoli_chain_n{15,20}` — synthetic: 100 Toffolis on rotating qubit triples `(i % n, (i+1) % n, (i+2) % n)`. Measures Tier-A throughput at n=15 (L2-resident, ~256 KiB state) and n=20 (L3 / DRAM-bound, ~16 MiB state).
- `ccz_chain_n{15,20}` — same shape for CCZ.
- `mcx_k{2,4,6}_n20` — Pauli-X with `k ∈ {2, 4, 6}` external controls. Verifies P1-05 path handles multi-control without regression.

### 8.2 Re-run existing benches (anti-regression)

- `qft_n20` — 0 Toffoli, 0 CCZ → ~0 % delta expected (validates the prelude detector doesn't slow apply_3q).
- `random_brickwall_n20` — 0 Toffoli, 0 CCZ → ~0 % delta.
- `grover_iter5_n20` — 5 CCZ → small win (~1-3 %).
- `bell_n2`, `ghz_n10` — sanity (small-state, Tier-C path).

### 8.3 Acceptance gates

- Toffoli specialised ≥ **1.5×** vs scalar-generic on `toffoli_chain_n15` (L2-resident).
- CCZ specialised ≥ **2×** vs scalar-generic on `ccz_chain_n15`.
- All workload benches: no regression > 2 % on EPYC reference runner.
- All numbers from `bencher.dev` EPYC baseline, not from aarch64 laptop. **Mid-stream EPYC validation per AVX-512 task group** (lesson from [P1-05 merged](../../../crates/aleph-sv/src/kernels/aos.rs) memory).

---

## 9 — Out of scope (explicit)

- ❌ Standalone MCX kernel — handled by routing through P1-05.
- ❌ Generic n-control Pauli-X / Pauli-Z beyond CCX/CCZ.
- ❌ Toffoli / CCZ with > 7 controls (BACKLOG "up to 8" satisfied by `mcx_k6_n20` + P1-05 generalisation).
- ❌ Modifications to P1-05/P1-06/P1-07 dispatch contracts.
- ❌ Default-backend selection logic (separate Phase-1-closure ADR).
- ❌ AVX2-only path (separate Phase-1-closure question per the plan's open issues).

---

## 10 — Risks and mitigations

- **R1 — `is_toffoli`/`is_ccz` detector cost on apply_3q's hot path.** Mitigation: keep detectors O(64) compares with early-exit on first mismatch; bench `qft_n20` (0 Toffoli/CCZ) as anti-regression check.
- **R2 — Tier-A outer-walk renormalisation bit-collision bug** (the P1-07 Task 5 / Task 14 lesson class). Mitigation: integer-only indexing-coverage tests **before** SIMD work, exhaustive over all `(c0,c1,t,ext)` permutations.
- **R3 — `_mm512_xor_pd` codegen.** Some Rust intrinsic versions emit `vpxor`/`vxorpd` ambiguously; ensure `objdump --disassemble` shows the expected `vxorpd zmm, zmm, zmm` pattern (latency 1). Mitigation: codegen inspection task in plan, with fallback to `_mm512_mul_pd` by `_mm512_set1_pd(-1.0)` if `vxorpd` doesn't materialise.
- **R4 — Bandwidth ceiling at n=20** (ADR 0008). Mitigation: the micro-AC is at n=15 (L2-resident) where bandwidth doesn't dominate; n=20 numbers are reported but not gating the AC.
- **R5 — SoA Tier-C sub-LANES underflow** (P1-07 Task 14 EPYC SIGSEGV). Mitigation: re-validate `mask_lo >= LANES_SOA_BITS` before any Tier-A entry; default to Tier-C otherwise.

---

## 11 — Done definition

- All unit / indexing / property / oracle tests green on CI (aarch64) and on EPYC.
- All new and existing benches re-run on EPYC; numbers in PR body.
- Micro-AC and workload-AC met.
- BACKLOG `[P1-08]` checkboxes ticked; bullets amended per §1 of this spec.
- ADR 0012 written (multi-controlled SIMD pattern: matrix-shape dispatch + Tier A/B/C + canonical outer-walk).
- PR title `[P1-08] Multi-controlled gate kernels (Toffoli, CCZ, MCX)`; body cites `Closes #<issue-number>` (not `#<PR-number>` — repeated mistake from P0-06..P0-11).
- Squash-merge.
