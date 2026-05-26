# P1-02 — Bit-manipulation indexing for 1-qubit gate application

**Issue:** P1-02 (see `BACKLOG.md`, GitHub #14)
**Depends on:** P1-01 (SoA backend + paired `Vec<f64>` storage)
**Date:** 2026-05-26

---

## 1. Goal

Replace the SoA `apply_1q` kernel's branch-heavy `for i in 0..len { if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask { ... } }` loop with a **nested block/pair** iteration over a free-bit outer index that visits each `(i0, i1)` pair exactly once — no per-iteration branch, deterministic order, unit-stride inner reads for `controls.is_empty()`.

The kernel handles uncontrolled and externally-controlled 1q gates through one unified `expand_index`-style outer loop. This is the foundation P1-03 (AVX2) and P1-04 (AVX-512) need: a unit-stride inner loop on aligned `f64` pairs that `_mm256_loadu_pd` can consume directly.

`apply_2q` and `apply_3q` are out of scope (P1-07 and P1-08). `NaiveSvBackend`'s AoS kernels are out of scope (reference oracle per BACKLOG).

The acceptance bar is a **2–3× speedup on QFT-20 vs P1-01** on the canonical EPYC bench server. M-series local-dev is informational only; bencher.dev historical (anchored at main `07d3724`) is the source of truth.

---

## 2. Scope

### In scope

- Rewrite `crates/aleph-sv/src/kernels/soa.rs::apply_1q` from the P1-01 branch-loop shape to a nested block/pair shape with controls integrated via free-bit outer iteration.
- Add `crates/aleph-sv/src/kernels/mod.rs::base_index_1q(k, target, controls) -> usize` helper (layout-agnostic; sits alongside `control_mask`).
- Extend the existing `apply_1q_soa_matches_aos` proptest in `kernels/soa.rs::tests` to draw a non-empty `controls` set, so AoS↔SoA equivalence is verified for the controlled path too.
- Add helper unit tests for `base_index_1q` covering the boundary cases that bit-manipulation typically gets wrong (target at LSB, target at MSB-of-state, multi-control with target between them).
- PR description carries before/after `qft/n{10,15,20}/soa` numbers from local-dev and from bencher.dev's EPYC run when available at merge time.

### Out of scope (deferred)

- `apply_2q` bit-manip — **P1-07**. The 2q kernel has a 4-element group per iteration with two target bits; same `base_index_*` shape applies but generalisation is its own ticket.
- `apply_3q` bit-manip — out of P1 perf path (3q is structurally Toffoli/CCZ and is rare in production circuits).
- SIMD intrinsics — **P1-03 (AVX2)**, **P1-04 (AVX-512)**. P1-02 leaves the inner loop in scalar f64 form; the auto-vectoriser may already exploit the unit-stride shape on M-series ARM SVE.
- Multi-controlled gate specialisation — **P1-08**. Today the unified bit-manip path handles any `controls.len()`, but P1-08 adds dedicated MCX/Toffoli kernels.
- AoS path — `NaiveSvBackend::apply_1q` stays at the P1-01 branch-loop shape. It is the reference oracle (BACKLOG explicitly preserves it).
- API churn — `SoaSvBackend::apply_gate`, `measure_soa::expectation_value_impl_soa::slow_path`, `kernels/soa.rs::apply_1q`'s public signature all stay 1-to-1 with P1-01.

---

## 3. Architecture

One file changes substantively: `crates/aleph-sv/src/kernels/soa.rs::apply_1q`. The helper lives in `crates/aleph-sv/src/kernels/mod.rs` next to `control_mask` (both are layout-agnostic — they return `usize` indices).

```
crates/aleph-sv/src/
├── kernels/
│   ├── mod.rs        + base_index_1q helper + 3 unit tests
│   ├── aos.rs        UNCHANGED — reference apply_1q/2q/3q on Vec<Complex>
│   └── soa.rs        REWRITE apply_1q body; apply_2q / apply_3q UNCHANGED
└── (everything else  UNCHANGED — soa_backend, measure_soa, validation,
    soa_state, lib.rs, …)
```

`SoaSvBackend::apply_gate` (which dispatches `GateMatrix::M2x2` → `kernels::soa::apply_1q`) is untouched — the kernel rewrite is internal to the function body.

No new dependencies. No public-API changes. `aleph-oracle` is untouched.

---

## 4. The `base_index_1q` helper

### 4.1 Contract

```rust
// crates/aleph-sv/src/kernels/mod.rs

/// For a 1-qubit gate on `target` with external `controls`, returns
/// the "base" amplitude index `i0` (the one with `target` bit = 0
/// and every control bit = 1) for the `k`-th iteration of the
/// free-bit outer loop.
///
/// `controls` must contain no element equal to `target` (callers
/// enforce this via the `DuplicateQubit` check before reaching the
/// kernel).  `target` and every control must be `< usize::BITS`
/// (caller enforces via `MAX_SOA_QUBITS = 28`).
///
/// `k` ranges over `0..(1usize << free_bits)` where
/// `free_bits = n_qubits - 1 - controls.len()`.  The full pair is
/// `(i0, i0 | (1usize << target))`.
pub(crate) fn base_index_1q(k: usize, target: u32, controls: &[u32]) -> usize;
```

### 4.2 Implementation

The function inserts `controls.len() + 1` fixed bits into `k`. Fixed positions are `target` (value 0) and every `controls[i]` (value 1).

```rust
pub(crate) fn base_index_1q(k: usize, target: u32, controls: &[u32]) -> usize {
    // Sort target + controls ascending; build the index by walking
    // free-bit chunks between fixed positions.  smallvec keeps the
    // allocation on the stack for the realistic `controls.len() ≤ 6`
    // range — see `SmallVec<[u32; 6]>` precedent in apply_gate.
    let mut fixed: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed.push((target, false));                         // target bit = 0
    for &c in controls {
        fixed.push((c, true));                            // control bit = 1
    }
    fixed.sort_unstable_by_key(|(pos, _)| *pos);

    let mut i = 0usize;
    let mut k_rem = k;
    let mut prev = 0u32;
    for (pos, val) in &fixed {
        let span = (*pos - prev) as usize;
        let chunk = k_rem & ((1usize << span) - 1);
        i |= chunk << (prev as usize);
        k_rem >>= span;
        if *val {
            i |= 1usize << pos;
        }
        prev = pos + 1;
    }
    // Remaining free bits above the highest fixed position.
    i |= k_rem << (prev as usize);
    i
}
```

### 4.3 Invariants (enforced by unit tests, §6.2)

- For `controls.is_empty()`, `target = 0`: `base_index_1q(k, 0, &[])` = `2 * k`
- For `controls.is_empty()`, `target = q`: bit `q` of result is always 0
- For `controls = [c]`: bit `c` of result is always 1
- For any `(target, controls)`, the returned `i0` has `(i0 >> target) & 1 == 0` and `(i0 >> c) & 1 == 1` for every control `c`
- `i0 ^ (1 << target)` is the partner index (no caller needs to compute it via `expand` again)

### 4.4 Cost

Per iteration: one short loop over `fixed.len() ≤ 7` elements (1 target + ≤6 controls — the `SmallVec<[u32; 6]>` cap in `apply_gate` validation). Inside the loop, 2 shifts + 1 mask + 1 or-into-i + 1 conditional or. For the common case (`controls.is_empty()`), that's a 1-iteration loop with ~4 cheap ops — comparable to one branchless miss-predicted branch from the P1-01 shape.

---

## 5. The rewritten `apply_1q`

### 5.1 Body

```rust
// crates/aleph-sv/src/kernels/soa.rs

pub(crate) fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    debug_assert_eq!(re.len(), im.len());
    let target_bit = 1usize << target;
    // n_qubits is encoded by re.len() = 2^n_qubits (validated upstream).
    // free_bits = n_qubits - 1 (target) - controls.len()
    let n_qubits = re.len().trailing_zeros();
    let n_free = n_qubits - 1 - controls.len() as u32;
    let outer_count = 1usize << n_free;

    let m00_re = m[0][0].re; let m00_im = m[0][0].im;
    let m01_re = m[0][1].re; let m01_im = m[0][1].im;
    let m10_re = m[1][0].re; let m10_im = m[1][0].im;
    let m11_re = m[1][1].re; let m11_im = m[1][1].im;

    for k in 0..outer_count {
        let i0 = super::base_index_1q(k, target, controls);
        let i1 = i0 | target_bit;

        let a0_re = re[i0]; let a0_im = im[i0];
        let a1_re = re[i1]; let a1_im = im[i1];

        re[i0] = m00_re * a0_re - m00_im * a0_im
               + m01_re * a1_re - m01_im * a1_im;
        im[i0] = m00_re * a0_im + m00_im * a0_re
               + m01_re * a1_im + m01_im * a1_re;
        re[i1] = m10_re * a0_re - m10_im * a0_im
               + m11_re * a1_re - m11_im * a1_im;
        im[i1] = m10_re * a0_im + m10_im * a0_re
               + m11_re * a1_im + m11_im * a1_re;
    }
}
```

### 5.2 What changed vs P1-01

- **No `if (i & t_bit == 0)` branch.** `base_index_1q` constructs `i0` with `target` bit guaranteed 0.
- **No `if (i & ctrl_mask) == ctrl_mask` branch.** `base_index_1q` constructs `i0` with every control bit guaranteed 1.
- **Loop count = exactly the number of pairs that mutate.** P1-01 iterates `re.len()` times and discards via two branches; P1-02 iterates `re.len() / 2 / 2^|controls|` times.
- **Inner block of mutation is unit-stride for `controls.is_empty()` and `target = 0`.** This is what AVX2 wants in P1-03: load 4 consecutive `re[i0..i0+4]`, load 4 consecutive `re[i1..i1+4]`, do 4 lanes of 2×2 multiply.

For `target > 0`, the inner block is unit-stride **within** each `2 * 2^target`-amplitude block. P1-03 SIMD can still vectorise by iterating outer blocks scalarly and inner pairs via SIMD.

### 5.3 What stayed the same

- `m: &[[Complex; 2]; 2]` (same signature)
- 8-mul + 4-add 2×2 matrix application formula
- `debug_assert_eq!(re.len(), im.len())` invariant guard
- MSB qubit-ordering convention (ADR 0004): `target` is the qubit index, `1usize << target` is the bit mask

---

## 6. Testing

### 6.1 Workhorse equivalence (no changes needed)

`crates/aleph-oracle/tests/soa_vs_naive.rs::all_fixtures_match_naive` runs all 28 oracle fixtures through both backends and asserts amplitude-wise agreement within `1e-12`. A bit-manip regression that gets even one `(i0, i1)` pair wrong on any of the 28 circuits panics here. **This is the gating correctness test.**

### 6.2 `base_index_1q` unit tests (`kernels/mod.rs::tests`)

```rust
#[test]
fn base_index_uncontrolled_target_zero_is_2k() {
    // target = 0, no controls → k-th iteration's i0 is 2*k.
    for k in 0..16 { assert_eq!(base_index_1q(k, 0, &[]), 2 * k); }
}

#[test]
fn base_index_uncontrolled_target_two_keeps_bit_zero() {
    // target = 2 → bit 2 of i0 is always 0.
    for k in 0..16 {
        let i = base_index_1q(k, 2, &[]);
        assert_eq!((i >> 2) & 1, 0, "k = {k}, i = {i:#06b}");
    }
}

#[test]
fn base_index_controlled_sets_control_bits() {
    // target = 0, control = 2 → bit 0 of i0 is 0, bit 2 of i0 is 1.
    for k in 0..8 {
        let i = base_index_1q(k, 0, &[2]);
        assert_eq!(i & 1, 0, "target bit must be 0");
        assert_eq!((i >> 2) & 1, 1, "control bit must be 1");
    }
}

#[test]
fn base_index_two_controls_target_between() {
    // target = 2, controls = [0, 5] → bit 0 = 1, bit 2 = 0, bit 5 = 1.
    for k in 0..16 {
        let i = base_index_1q(k, 2, &[0, 5]);
        assert_eq!(i & 1, 1);
        assert_eq!((i >> 2) & 1, 0);
        assert_eq!((i >> 5) & 1, 1);
    }
}

#[test]
fn base_index_enumerates_all_pairs_exactly_once() {
    // For target = 3, controls = [1] on n = 5 qubits, there are
    // 2^(5 - 1 - 1) = 8 valid `i0` values.  Collect them; assert
    // each i0 ⊕ target_bit covers the matching `i1` partition; assert
    // every (i0, i1) is distinct.
    let mut seen: std::collections::HashSet<usize> = Default::default();
    for k in 0..8 {
        let i0 = base_index_1q(k, 3, &[1]);
        let i1 = i0 | (1 << 3);
        assert!(seen.insert(i0), "duplicate i0 = {i0}");
        assert!(seen.insert(i1), "duplicate i1 = {i1}");
    }
    // 8 iterations × 2 indices per iter = 16 distinct values; the
    // n=5 space with control bit set = 2^(5-1) = 16.
    assert_eq!(seen.len(), 16);
}
```

### 6.3 Extended AoS↔SoA proptest

In `kernels/soa.rs::tests`, the existing `apply_1q_soa_matches_aos` proptest currently passes `controls = &[]`. Extend its strategy to draw a controls set:

```rust
#[test]
fn apply_1q_soa_matches_aos(
    gate in arb_1q_gate(),
    q in 0u32..5,
    n_ctrls in 0usize..=2,
    ctrl_seed in any::<u32>(),
    amps in arb_state_vector(5),
) {
    // Build a unique controls set from ctrl_seed, excluding `q`.
    let mut ctrls: Vec<u32> = (0u32..5).filter(|c| *c != q).collect();
    // Shuffle deterministically by ctrl_seed, then take first n_ctrls.
    // (proptest's Strategy combinators don't have shuffle directly;
    // simplest is `ctrls.sort_by_key(|c| (ctrl_seed.wrapping_mul(*c as u32 + 7))`)
    ctrls.sort_by_key(|c| ctrl_seed.wrapping_mul(*c + 7));
    ctrls.truncate(n_ctrls);
    ctrls.sort_unstable();  // kernels don't care about order; sort for determinism

    // … rest as today: AoS reference vs SoA candidate, assert |Δ| < 1e-12.
}
```

The 64 cases × `n_ctrls ∈ {0, 1, 2}` distribution gives roughly equal weight to uncontrolled and controlled paths.

### 6.4 Oracle suite (no changes needed)

The 112 generated tests in `crates/aleph-oracle/tests/_generated.rs` exercise SoA against Qiskit on every fixture. QFT-3, QFT-5, GHZ-3, GHZ-5, GHZ-10, kernel_p (controlled-Phase), grover_2q_mark11 — all heavy on controlled-1q paths. A correctness regression in P1-02's controlled path surfaces here.

---

## 7. Bench strategy

`crates/aleph-sv/benches/soa_vs_naive.rs` is **unchanged**. It already covers `qft/n{10,15,20}/{naive,soa}` and `ghz/n20/{naive,soa}`. The P1-02 effect shows up as:

- `qft/n20/soa` ratio vs P1-01 baseline (bencher.dev historical at `07d3724`) — AC target **2–3×**
- `qft/n15/soa` and `qft/n10/soa` — informational; smaller `n` gives the branch-predictor more chance to learn the P1-01 shape, so the P1-02 win shrinks at small `n`
- `ghz/n20/soa` — modest improvement expected (GHZ is mostly Cnot via `apply_2q`, which P1-02 doesn't touch); P1-02 reaches `ghz` only through its single initial `H` gate

PR body cites both local M-series numbers (captured via `RUSTFLAGS="-C target-cpu=native" cargo bench`) and EPYC numbers when bencher.dev posts them. M-series may underperform AC (Apple silicon's prefetch and branch predictor absorb a lot of P1-01's overhead); EPYC is the source of truth per spec §1.

---

## 8. Acceptance-criteria mapping (BACKLOG P1-02)

| BACKLOG AC | Where satisfied |
|---|---|
| All 1q gates implemented with this pattern | `kernels/soa.rs::apply_1q` body rewrite (§5); both uncontrolled and externally-controlled paths share one `base_index_1q`-driven loop |
| Benchmark: 2–3× improvement over P1-01 on QFT-20 | `benches/soa_vs_naive.rs::qft/n20/soa` vs P1-01 historical on bencher.dev; PR body reports both M-series and EPYC numbers |
| All correctness tests pass | Workhorse `all_fixtures_match_naive` (28 fixtures) + extended AoS↔SoA proptest (controls coverage) + 112 generated oracle tests + 5 `base_index_1q` unit tests |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `base_index_1q` bit-shift math wrong → wrong index → silent wrong amplitude | Helper unit tests (§6.2) cover boundary cases (target at LSB, target at MSB, multi-control with target between); AoS↔SoA proptest (§6.3) shrinks to minimal failing input on any divergence; workhorse equivalence (§6.1) over 28 fixtures is the production gate |
| QFT-20 SoA misses the 2–3× AC on EPYC | Same precedent as P1-01 spec §12: correctness gates take priority; if AC misses, PR body documents and defers the remaining win to P1-03 (AVX2). Don't block merge on a perf AC if correctness gates met. |
| Helper allocation in hot loop (`SmallVec` push) | `SmallVec<[(u32, bool); 7]>` is stack-only at `≤7` elements (1 target + ≤6 controls, capped by `apply_gate` validation). Zero heap allocations. |
| `trailing_zeros` on `re.len()` is brittle | `re.len()` is `1 << num_qubits` enforced by `validate_state_soa` (which `apply_gate` does not call, but `allocate` populates correctly). To be defense-in-depth, add `debug_assert!(re.len().is_power_of_two())` at the kernel entry. Production cost is zero (debug-only); failure surface for a future kernel that mutates length is loud. |
| Auto-vectoriser miscompiles the new shape on M-series | `cargo bench` measures end-to-end; equivalence proptest verifies correctness. If a regression appears on the bench server but not locally, `objdump` + `perf` triage per `docs/perf/phase0.md`. |

---

## 10. Workflow notes

Standard P0-06…P1-01 workflow:

- **Branch:** `p1-02-bit-manip-indexing` (already created from main `07d3724`).
- **Implementation order** (drives the plan, §11 below).
- **PR title:** `[P1-02] Bit-manipulation indexing for 1-qubit gates`.
- **PR body:** `Closes #14` (P1-02 = GitHub issue 14; pattern follows P0-12 → P1-01 lesson — never `Closes #<PR>`).
- **Squash-merge.**

---

## 11. Implementation order

1. Add `base_index_1q` to `crates/aleph-sv/src/kernels/mod.rs` with the 5 unit tests from §6.2.
2. Rewrite `crates/aleph-sv/src/kernels/soa.rs::apply_1q` body using `base_index_1q`. Run `cargo test -p aleph-sv --lib kernels::soa` — the existing uncontrolled tests pass, the existing AoS↔SoA proptest (still `controls=&[]`) passes.
3. Extend `apply_1q_soa_matches_aos` proptest (§6.3) to draw controls. Run it — must pass, confirms controlled-path correctness on random gates × random states.
4. Run workhorse `cargo test -p aleph-oracle --test soa_vs_naive` — must pass, confirms correctness across all 28 fixtures (including QFT, GHZ, CCX, controlled-Phase).
5. Run `cargo test -p aleph-oracle --tests` — 112 generated oracle tests must pass.
6. Run `RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-sv --bench soa_vs_naive` — capture local-dev numbers for PR body.
7. Tick BACKLOG ACs (correctness + benchmark + 1q-coverage all three).
8. `cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace && cargo build --release --workspace`.
9. Push, open PR, wait for CI, squash-merge.

Estimated ~7 plan tasks (vs P1-01's 17). Single substantive file (`kernels/soa.rs`) + single helper (`kernels/mod.rs`) + one test-strategy extension. Spec amendments unlikely — the algorithm is canonical (cf. QuEST `statevec_unitary` for the reference implementation).
