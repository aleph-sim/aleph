# P1-08 Multi-Controlled Gate Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Specialised AVX-512 kernels for Toffoli (CCX) and CCZ on AoS layout, plus a benchmark anchor verifying P1-05's anti-diagonal kernel handles MCX (Pauli-X with k≥2 controls) without regression.

**Architecture:** Add matrix-shape detectors (`is_identity_8x8`, `is_toffoli`, `is_ccz`) in `kernels/mod.rs`. Convert the existing scalar `apply_3q` into `apply_3q_generic`; introduce a new dispatch prelude `apply_3q` that matrix-detects Toffoli/CCZ shapes and routes to `dispatch_toffoli` / `dispatch_ccz`. Each dispatch has Tier A (packed AVX-512 + outer-walk renormalisation), Tier B (in-zmm permute; Toffoli only), Tier C (scalar fallback). Symmetric implementation in `kernels/soa.rs`. MCX routes through the existing `apply_1q` → P1-05 anti-diagonal path.

**Tech Stack:** Rust 2021 (edition), MSRV 1.89, `std::arch::x86_64::*` AVX-512F intrinsics, criterion 0.5 benchmarks, proptest 1.x, the existing `aleph-test` crate for shared property strategies, `bencher.dev` EPYC runner for perf validation.

**Spec:** `docs/superpowers/specs/2026-05-28-p1-08-multi-controlled-design.md`

**Reference patterns:** P1-05 anti-diagonal (Tier A/B), P1-06 diagonal (outer-walk), P1-07 dispatch_cnot/dispatch_cz (matrix-shape prelude).

---

## File Structure

**Modify:**
- `crates/aleph-sv/src/kernels/mod.rs` — shape detectors (`is_identity_8x8`, `is_toffoli`, `is_ccz`); reusable bit-position helpers if needed.
- `crates/aleph-sv/src/kernels/aos.rs` — rename existing `apply_3q` → `apply_3q_generic`; new public `apply_3q` prelude + `dispatch_toffoli` + `dispatch_ccz` with Tier A/B/C.
- `crates/aleph-sv/src/kernels/soa.rs` — same renames + SoA mirrors of dispatch_toffoli / dispatch_ccz.
- `crates/aleph-sv/Cargo.toml` — register new bench targets.
- `BACKLOG.md` — amend `[P1-08]` per spec §1.
- `crates/aleph-test/src/lib.rs` (or strategies module) — add CCX/CCZ-shaped circuit strategies if missing.

**Create:**
- `crates/aleph-sv/benches/multi_controlled.rs` — `toffoli_chain_n{15,20}`, `ccz_chain_n{15,20}`, `mcx_k{2,4,6}_n20`.
- `docs/decisions/0012-multi-controlled-simd-pattern.md` — ADR 0012.

**Indexing-coverage test files** live alongside kernels in inline `#[cfg(test)] mod tests`.

---

### Task 1: Shape detectors — `is_identity_8x8`, `is_toffoli`, `is_ccz`

**Files:**
- Modify: `crates/aleph-sv/src/kernels/mod.rs`

- [ ] **Step 1: Write failing tests**

Append to the existing `#[cfg(test)] mod tests` block in `crates/aleph-sv/src/kernels/mod.rs`:

```rust
#[cfg(test)]
mod shape_8x8_tests {
    use super::*;
    use aleph_core::Complex;

    fn identity_8x8() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..8 { m[i][i] = o; }
        m
    }

    fn toffoli_8x8() -> [[Complex; 8]; 8] {
        // Identity rows 0..=5; swap rows 6 ↔ 7.
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..6 { m[i][i] = o; }
        m[6][7] = o;
        m[7][6] = o;
        m
    }

    fn ccz_8x8() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..7 { m[i][i] = o; }
        m[7][7] = Complex::new(-1.0, 0.0);
        m
    }

    #[test]
    fn identity_detected() {
        assert!(is_identity_8x8(&identity_8x8()));
        assert!(!is_identity_8x8(&toffoli_8x8()));
        assert!(!is_identity_8x8(&ccz_8x8()));
    }

    #[test]
    fn toffoli_detected() {
        assert!(is_toffoli(&toffoli_8x8()));
        assert!(!is_toffoli(&identity_8x8()));
        assert!(!is_toffoli(&ccz_8x8()));
    }

    #[test]
    fn ccz_detected() {
        assert!(is_ccz(&ccz_8x8()));
        assert!(!is_ccz(&identity_8x8()));
        assert!(!is_ccz(&toffoli_8x8()));
    }

    #[test]
    fn toffoli_tolerates_tiny_noise() {
        let mut m = toffoli_8x8();
        m[0][1] = Complex::new(1e-14, 0.0);
        assert!(is_toffoli(&m));
    }

    #[test]
    fn toffoli_rejects_visible_noise() {
        let mut m = toffoli_8x8();
        m[0][1] = Complex::new(1e-6, 0.0);
        assert!(!is_toffoli(&m));
    }
}
```

- [ ] **Step 2: Run tests, expect failure (functions don't exist)**

```
cargo test -p aleph-sv kernels::shape_8x8_tests -- --nocapture
```
Expected: compile error — `is_identity_8x8`, `is_toffoli`, `is_ccz` not defined.

- [ ] **Step 3: Add detector functions to `kernels/mod.rs`**

```rust
use aleph_core::Complex;

const SHAPE_8X8_TOL: f64 = 1e-12;

/// Returns true if `m` is within `SHAPE_8X8_TOL` of the 8×8 identity.
pub(crate) fn is_identity_8x8(m: &[[Complex; 8]; 8]) -> bool {
    for r in 0..8 {
        for c in 0..8 {
            let expected = if r == c { 1.0 } else { 0.0 };
            if (m[r][c].re - expected).abs() > SHAPE_8X8_TOL { return false; }
            if m[r][c].im.abs() > SHAPE_8X8_TOL { return false; }
        }
    }
    true
}

/// Returns true if `m` is within `SHAPE_8X8_TOL` of the canonical
/// Toffoli (CCX) matrix: identity on rows 0..=5, swap rows 6 ↔ 7.
pub(crate) fn is_toffoli(m: &[[Complex; 8]; 8]) -> bool {
    // Rows 0..=5: identity rows.
    for r in 0..6 {
        for c in 0..8 {
            let expected = if r == c { 1.0 } else { 0.0 };
            if (m[r][c].re - expected).abs() > SHAPE_8X8_TOL { return false; }
            if m[r][c].im.abs() > SHAPE_8X8_TOL { return false; }
        }
    }
    // Row 6: e6 -> e7  ⇒  m[6][7] = 1, else 0.
    for c in 0..8 {
        let expected = if c == 7 { 1.0 } else { 0.0 };
        if (m[6][c].re - expected).abs() > SHAPE_8X8_TOL { return false; }
        if m[6][c].im.abs() > SHAPE_8X8_TOL { return false; }
    }
    // Row 7: e7 -> e6  ⇒  m[7][6] = 1, else 0.
    for c in 0..8 {
        let expected = if c == 6 { 1.0 } else { 0.0 };
        if (m[7][c].re - expected).abs() > SHAPE_8X8_TOL { return false; }
        if m[7][c].im.abs() > SHAPE_8X8_TOL { return false; }
    }
    true
}

/// Returns true if `m` is within `SHAPE_8X8_TOL` of the canonical
/// CCZ matrix: diagonal with d[0..7] = +1 and d[7] = -1.
pub(crate) fn is_ccz(m: &[[Complex; 8]; 8]) -> bool {
    for r in 0..8 {
        for c in 0..8 {
            let expected_re = match (r, c) {
                (i, j) if i == j && i < 7 => 1.0,
                (7, 7) => -1.0,
                _ => 0.0,
            };
            if (m[r][c].re - expected_re).abs() > SHAPE_8X8_TOL { return false; }
            if m[r][c].im.abs() > SHAPE_8X8_TOL { return false; }
        }
    }
    true
}
```

- [ ] **Step 4: Run tests, expect pass**

```
cargo test -p aleph-sv kernels::shape_8x8_tests
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/mod.rs
git commit -m "[P1-08] shape detectors: is_identity_8x8 / is_toffoli / is_ccz

Pure-function matrix-shape detectors with 1e-12 tolerance. Tested
against canonical matrices and tiny noise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Indexing-coverage tests — Toffoli (integer-only)

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs` (append to `#[cfg(test)] mod tests`)

**Why now:** The P1-07 EPYC SIGSEGV on n=2 was caught by indexing-coverage tests **after** SIMD landed. Doing the integer-only test first means the dispatch contract is verified pre-SIMD and the SIMD work has a known-good oracle.

- [ ] **Step 1: Write integer-only Toffoli classification test**

Append to `crates/aleph-sv/src/kernels/aos.rs`:

```rust
#[cfg(test)]
mod toffoli_indexing_tests {
    /// Classify (c0, c1, t, ext, n) into the expected dispatch tier
    /// per spec §4.2-§4.4. This is the source-of-truth oracle for the
    /// SIMD dispatch path; if the function below returns Tier X, the
    /// runtime SIMD path must match.
    #[derive(Debug, PartialEq, Eq)]
    enum Tier { A, B0, B1, C }

    fn classify_toffoli(c0: u32, c1: u32, t: u32, ext: &[u32], n: u32) -> Tier {
        const LANES_BITS: u32 = 2;
        if n < 3 { return Tier::C; }
        let target_bit_idx = t;
        let ctrl_bits: Vec<u32> = std::iter::once(c0)
            .chain(std::iter::once(c1))
            .chain(ext.iter().copied())
            .collect();
        let c_lo = *ctrl_bits.iter().min().unwrap();
        // Tier A: target_bit ≥ LANES (== t ≥ LANES_BITS) and c_lo > t.
        if target_bit_idx >= LANES_BITS && c_lo > target_bit_idx {
            return Tier::A;
        }
        // Tier A outer-walk: target_bit ≥ LANES but some controls below target.
        // Spec §4.2 says Tier-A handles this via expand_with_fixed renormalisation.
        if target_bit_idx >= LANES_BITS {
            return Tier::A;
        }
        // Tier B: target_bit < LANES (t ∈ {0,1}) and c_lo ≥ LANES_BITS.
        if c_lo >= LANES_BITS {
            return match t {
                0 => Tier::B0,
                1 => Tier::B1,
                _ => unreachable!("t<LANES_BITS but t not in {0,1}"),
            };
        }
        // Else: Tier C scalar.
        Tier::C
    }

    /// Compute the swap pair (i, i ^ target_bit) for a given dispatch
    /// configuration and verify pairwise-disjoint bits at the
    /// SIMD-block level (mirrors P1-07 Task 14's coverage tests).
    fn pairs_are_disjoint(c0: u32, c1: u32, t: u32, ext: &[u32], n: u32) -> bool {
        let target_bit = 1u64 << t;
        let mut ctrl_mask = (1u64 << c0) | (1u64 << c1);
        for &e in ext { ctrl_mask |= 1u64 << e; }
        // For every i with ctrl bits set and target bit clear, (i, i | target_bit)
        // must be in-range and distinct, and target_bit must not overlap ctrl_mask.
        if target_bit & ctrl_mask != 0 { return false; }
        let len = 1u64 << n;
        for i in 0..len {
            if (i & ctrl_mask) != ctrl_mask { continue; }
            if (i & target_bit) != 0 { continue; }
            let j = i | target_bit;
            if j >= len { return false; }
            if i == j { return false; }
        }
        true
    }

    #[test]
    fn toffoli_classification_clean_tier_a() {
        // c0=4, c1=5, t=2, n=6: t_bit_idx=2 ≥ LANES_BITS, c_lo=4 > t=2.
        assert_eq!(classify_toffoli(4, 5, 2, &[], 6), Tier::A);
    }

    #[test]
    fn toffoli_classification_tier_b0() {
        // c0=2, c1=3, t=0, n=4: t<LANES_BITS, c_lo=2 ≥ LANES_BITS.
        assert_eq!(classify_toffoli(2, 3, 0, &[], 4), Tier::B0);
    }

    #[test]
    fn toffoli_classification_tier_b1() {
        // c0=2, c1=3, t=1, n=4
        assert_eq!(classify_toffoli(2, 3, 1, &[], 4), Tier::B1);
    }

    #[test]
    fn toffoli_classification_tier_c_small_n() {
        // n=2 — must be Tier C.
        assert_eq!(classify_toffoli(0, 1, 0, &[], 2), Tier::C);
    }

    #[test]
    fn toffoli_classification_tier_c_mixed_low_controls() {
        // c0=0, c1=1, t=0: t < LANES_BITS, c_lo=0 < LANES_BITS → Tier C.
        assert_eq!(classify_toffoli(0, 1, 0, &[], 3), Tier::C);
    }

    #[test]
    fn toffoli_pairs_disjoint_exhaustive_n6() {
        // For all triples (c0,c1,t) in {0..6}^3 with c0 != c1 != t,
        // and ext subsets of size ≤ 1, verify pair disjointness.
        for c0 in 0..6 {
            for c1 in 0..6 {
                for t in 0..6 {
                    if c0 == c1 || c0 == t || c1 == t { continue; }
                    assert!(pairs_are_disjoint(c0, c1, t, &[], 6),
                            "c0={} c1={} t={} ext=[]", c0, c1, t);
                    for e in 0..6 {
                        if e == c0 || e == c1 || e == t { continue; }
                        assert!(pairs_are_disjoint(c0, c1, t, &[e], 6),
                                "c0={} c1={} t={} ext=[{}]", c0, c1, t, e);
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 2: Run tests, expect pass (pure integer math, no kernels yet)**

```
cargo test -p aleph-sv kernels::aos::toffoli_indexing_tests
```
Expected: 6 passed.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] indexing-coverage tests for Toffoli dispatch tiers

Pre-SIMD integer-only oracle: classify_toffoli() and pairs_are_disjoint()
exhaustively verified for (c0,c1,t,ext)∈{0..6}, n=6. Catches the
bit-collision class that bit P1-07 Task 14 on EPYC.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Indexing-coverage tests — CCZ

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs` (append new `#[cfg(test)] mod ccz_indexing_tests`)

- [ ] **Step 1: Write integer-only CCZ classification + coverage test**

Append:

```rust
#[cfg(test)]
mod ccz_indexing_tests {
    #[derive(Debug, PartialEq, Eq)]
    enum CczTier { A, C }

    /// CCZ has no target — every mask bit is symmetric. Tier A
    /// applies when mask_lo ≥ LANES_BITS (so each zmm block has a
    /// fixed ctrl-mask value); Tier C otherwise.
    fn classify_ccz(q0: u32, q1: u32, q2: u32, ext: &[u32], n: u32) -> CczTier {
        const LANES_BITS: u32 = 2;
        if n < 3 { return CczTier::C; }
        let mask_bits: Vec<u32> = [q0, q1, q2].iter().copied()
            .chain(ext.iter().copied()).collect();
        let mask_lo = *mask_bits.iter().min().unwrap();
        // Tier A outer-walk handles mask_lo < LANES_BITS (per spec §5.2).
        if n >= 3 { CczTier::A } else { CczTier::C }
    }

    fn ccz_pairs_unique(q0: u32, q1: u32, q2: u32, ext: &[u32], n: u32) -> bool {
        let mut mask = (1u64 << q0) | (1u64 << q1) | (1u64 << q2);
        for &e in ext { mask |= 1u64 << e; }
        let len = 1u64 << n;
        let mut count = 0u64;
        for i in 0..len {
            if (i & mask) == mask { count += 1; }
        }
        // Every full match is exactly one sign-flip; no pairs to swap.
        // Validate count = 2^(n - popcount(mask)).
        let expected = 1u64 << (n - mask.count_ones());
        count == expected
    }

    #[test]
    fn ccz_pairs_count_exhaustive_n6() {
        for q0 in 0..6 {
            for q1 in 0..6 {
                for q2 in 0..6 {
                    if q0 == q1 || q0 == q2 || q1 == q2 { continue; }
                    assert!(ccz_pairs_unique(q0, q1, q2, &[], 6),
                            "q0={} q1={} q2={}", q0, q1, q2);
                    for e in 0..6 {
                        if e == q0 || e == q1 || e == q2 { continue; }
                        assert!(ccz_pairs_unique(q0, q1, q2, &[e], 6));
                    }
                }
            }
        }
    }

    #[test]
    fn ccz_symmetry_mask_is_permutation_invariant() {
        // CCZ symmetric in qubit order: mask(q0,q1,q2) = mask(any permutation).
        let m1 = (1u64 << 3) | (1u64 << 4) | (1u64 << 5);
        let m2 = (1u64 << 5) | (1u64 << 3) | (1u64 << 4);
        assert_eq!(m1, m2);
    }

    #[test]
    fn ccz_classification_small_n_is_tier_c() {
        assert_eq!(classify_ccz(0, 1, 2, &[], 2), CczTier::C);
    }
}
```

- [ ] **Step 2: Run tests, expect pass**

```
cargo test -p aleph-sv kernels::aos::ccz_indexing_tests
```
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] indexing-coverage tests for CCZ dispatch

Validates mask uniqueness and symmetry for CCZ on n=6 exhaustively.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Scalar `apply_toffoli_scalar` (Tier-C reference impl)

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

- [ ] **Step 1: Write tests for `apply_toffoli_scalar`**

Append a new `#[cfg(test)] mod apply_toffoli_scalar_tests`:

```rust
#[cfg(test)]
mod apply_toffoli_scalar_tests {
    use super::*;
    use aleph_core::Complex;

    fn basis_state(n: u32, index: usize) -> Vec<Complex> {
        let mut amps = vec![Complex::new(0.0, 0.0); 1 << n];
        amps[index] = Complex::new(1.0, 0.0);
        amps
    }

    #[test]
    fn ccx_swaps_only_11_target_pair() {
        // qubits = [c0=0, c1=1, t=2], n=3.
        // Basis ordering: MSB convention from P0-06 — index k = (q0<<2)|(q1<<1)|q2.
        // So state |c0 c1 t⟩ = |q0 q1 q2⟩, indices map directly.
        for input in 0..8usize {
            let mut amps = basis_state(3, input);
            apply_toffoli_scalar(&mut amps, [0, 1, 2], &[]);
            let expected = if input == 0b110 { 0b111 }
                          else if input == 0b111 { 0b110 }
                          else { input };
            let mut want = vec![Complex::new(0.0, 0.0); 8];
            want[expected] = Complex::new(1.0, 0.0);
            assert_eq!(amps, want, "input {:03b} should map to {:03b}", input, expected);
        }
    }

    #[test]
    fn ccx_with_external_control_acts_only_when_ext_set() {
        // qubits = [0,1,2], ctx=[3], n=4. CCCX swaps |1110⟩ ↔ |1111⟩ only.
        for input in 0..16usize {
            let mut amps = basis_state(4, input);
            apply_toffoli_scalar(&mut amps, [0, 1, 2], &[3]);
            let expected = if input == 0b1110 { 0b1111 }
                          else if input == 0b1111 { 0b1110 }
                          else { input };
            let mut want = vec![Complex::new(0.0, 0.0); 16];
            want[expected] = Complex::new(1.0, 0.0);
            assert_eq!(amps, want);
        }
    }

    #[test]
    fn ccx_involutive() {
        let mut amps: Vec<Complex> = (0..16).map(|i| Complex::new(i as f64, 0.0)).collect();
        let original = amps.clone();
        apply_toffoli_scalar(&mut amps, [0, 1, 2], &[]);
        apply_toffoli_scalar(&mut amps, [0, 1, 2], &[]);
        assert_eq!(amps, original);
    }
}
```

- [ ] **Step 2: Run tests, expect failure**

```
cargo test -p aleph-sv kernels::aos::apply_toffoli_scalar_tests
```
Expected: compile error — `apply_toffoli_scalar` not defined.

- [ ] **Step 3: Add `apply_toffoli_scalar` to `kernels/aos.rs`**

```rust
/// Scalar Tier-C reference for Toffoli (CCX). Tier-C path of
/// dispatch_toffoli (spec §4.4).
///
/// `targets = [c0, c1, t]` matches `Gate::Toffoli`'s qubit layout.
pub(crate) fn apply_toffoli_scalar(
    amps: &mut [Complex],
    targets: [u32; 3],
    external_controls: &[u32],
) {
    let c0 = targets[0];
    let c1 = targets[1];
    let t = targets[2];
    let target_bit = 1usize << t;
    let mut ctrl_mask = (1usize << c0) | (1usize << c1);
    for &e in external_controls { ctrl_mask |= 1usize << e; }
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & ctrl_mask) == ctrl_mask && (i & target_bit) == 0 {
            amps.swap(i, i | target_bit);
        }
        i += 1;
    }
}
```

- [ ] **Step 4: Run tests, expect pass**

```
cargo test -p aleph-sv kernels::aos::apply_toffoli_scalar_tests
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] apply_toffoli_scalar — Tier-C reference for CCX

Basis-state + external-control + involutivity tests. Same MSB
indexing convention as apply_3q_generic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Scalar `apply_ccz_scalar` (Tier-C reference impl)

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

- [ ] **Step 1: Write tests**

Append:

```rust
#[cfg(test)]
mod apply_ccz_scalar_tests {
    use super::*;
    use aleph_core::Complex;

    fn basis_state(n: u32, index: usize) -> Vec<Complex> {
        let mut amps = vec![Complex::new(0.0, 0.0); 1 << n];
        amps[index] = Complex::new(1.0, 0.0);
        amps
    }

    #[test]
    fn ccz_sign_flips_only_111() {
        for input in 0..8usize {
            let mut amps = basis_state(3, input);
            apply_ccz_scalar(&mut amps, [0, 1, 2], &[]);
            let mut want = vec![Complex::new(0.0, 0.0); 8];
            let sign = if input == 0b111 { -1.0 } else { 1.0 };
            want[input] = Complex::new(sign, 0.0);
            assert_eq!(amps, want);
        }
    }

    #[test]
    fn ccz_with_external_control_acts_only_when_ext_set() {
        for input in 0..16usize {
            let mut amps = basis_state(4, input);
            apply_ccz_scalar(&mut amps, [0, 1, 2], &[3]);
            let mut want = vec![Complex::new(0.0, 0.0); 16];
            let sign = if input == 0b1111 { -1.0 } else { 1.0 };
            want[input] = Complex::new(sign, 0.0);
            assert_eq!(amps, want);
        }
    }

    #[test]
    fn ccz_involutive() {
        let mut amps: Vec<Complex> = (0..16).map(|i| Complex::new(i as f64, 0.0)).collect();
        let original = amps.clone();
        apply_ccz_scalar(&mut amps, [0, 1, 2], &[]);
        apply_ccz_scalar(&mut amps, [0, 1, 2], &[]);
        assert_eq!(amps, original);
    }

    #[test]
    fn ccz_symmetric_in_qubit_order() {
        let mut a = vec![Complex::new(1.0, 0.0); 16];
        let mut b = a.clone();
        apply_ccz_scalar(&mut a, [0, 1, 2], &[]);
        apply_ccz_scalar(&mut b, [2, 0, 1], &[]);
        assert_eq!(a, b);
    }
}
```

- [ ] **Step 2: Run tests, expect failure**

```
cargo test -p aleph-sv kernels::aos::apply_ccz_scalar_tests
```
Expected: compile error.

- [ ] **Step 3: Add `apply_ccz_scalar`**

```rust
/// Scalar Tier-C reference for CCZ (spec §5.4). Sign-flips the
/// single amplitude where all three qubits AND any external controls
/// are |1⟩.
pub(crate) fn apply_ccz_scalar(
    amps: &mut [Complex],
    targets: [u32; 3],
    external_controls: &[u32],
) {
    let mut mask = (1usize << targets[0])
                 | (1usize << targets[1])
                 | (1usize << targets[2]);
    for &e in external_controls { mask |= 1usize << e; }
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & mask) == mask {
            amps[i] = -amps[i];
        }
        i += 1;
    }
}
```

- [ ] **Step 4: Run tests, expect pass**

```
cargo test -p aleph-sv kernels::aos::apply_ccz_scalar_tests
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] apply_ccz_scalar — Tier-C reference for CCZ

Basis + symmetry + involutivity tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Wire `apply_3q` prelude — route Toffoli/CCZ to scalar path

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

**Goal:** Make `apply_3q` matrix-detect Toffoli/CCZ and route. Keep using scalar paths for now; SIMD comes in Tasks 7-12.

- [ ] **Step 1: Write equivalence test (scalar Toffoli/CCZ ≡ generic apply_3q on random state)**

Append:

```rust
#[cfg(test)]
mod apply_3q_prelude_tests {
    use super::*;
    use aleph_core::Complex;

    fn toffoli_matrix() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..6 { m[i][i] = o; }
        m[6][7] = o;
        m[7][6] = o;
        m
    }

    fn ccz_matrix() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..7 { m[i][i] = o; }
        m[7][7] = Complex::new(-1.0, 0.0);
        m
    }

    fn random_amps(n: u32, seed: u64) -> Vec<Complex> {
        // Linear congruential — deterministic, no rand crate dep.
        let mut s = seed;
        let mut step = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        let mut v: Vec<Complex> = (0..(1 << n)).map(|_| Complex::new(step(), step())).collect();
        // Normalise.
        let norm: f64 = v.iter().map(|c| c.re*c.re + c.im*c.im).sum::<f64>().sqrt();
        for c in &mut v { *c = Complex::new(c.re / norm, c.im / norm); }
        v
    }

    #[test]
    fn apply_3q_routes_toffoli_to_scalar() {
        let mut a = random_amps(5, 1);
        let mut b = a.clone();
        apply_3q(&mut a, [0, 1, 4], &[], &toffoli_matrix());
        apply_toffoli_scalar(&mut b, [0, 1, 4], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_3q_routes_ccz_to_scalar() {
        let mut a = random_amps(5, 2);
        let mut b = a.clone();
        apply_3q(&mut a, [0, 1, 4], &[], &ccz_matrix());
        apply_ccz_scalar(&mut b, [0, 1, 4], &[]);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_3q_generic_unchanged_on_arbitrary_matrix() {
        let z = Complex::new(0.0, 0.0);
        let mut m = [[z; 8]; 8];
        // Hadamard-like 3q matrix.
        let s = 1.0 / (8.0_f64.sqrt());
        for r in 0..8 { for c in 0..8 { m[r][c] = Complex::new(s, 0.0); } }
        for r in 0..8 {
            for c in 0..8 {
                if (r & c).count_ones() % 2 == 1 { m[r][c] = -m[r][c]; }
            }
        }
        let mut a = random_amps(5, 3);
        let mut b = a.clone();
        apply_3q(&mut a, [0, 1, 4], &[], &m);
        apply_3q_generic(&mut b, [0, 1, 4], &[], &m);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }
}
```

- [ ] **Step 2: Run tests, expect failure (apply_3q_generic not yet defined, dispatch missing)**

- [ ] **Step 3: Rename existing `apply_3q` → `apply_3q_generic`; add new prelude**

In `kernels/aos.rs`, locate the existing `pub(crate) fn apply_3q(...)` (around line 2683) and:

1. Rename it to `pub(crate) fn apply_3q_generic`.
2. Insert the new prelude above it:

```rust
/// Top-level 3q dispatch. Matrix-detects Toffoli (CCX) and CCZ shapes
/// per spec §3.1. Identity short-circuits to no-op.
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
    apply_3q_generic(amps, targets, controls, m);
}

/// Routes Toffoli to the best available tier (spec §4).
/// Tier-C-only for now; SIMD added in later tasks.
fn dispatch_toffoli(amps: &mut [Complex], targets: [u32; 3], controls: &[u32]) {
    apply_toffoli_scalar(amps, targets, controls);
}

/// Routes CCZ to the best available tier (spec §5).
/// Tier-C-only for now; SIMD added in later tasks.
fn dispatch_ccz(amps: &mut [Complex], targets: [u32; 3], controls: &[u32]) {
    apply_ccz_scalar(amps, targets, controls);
}
```

- [ ] **Step 4: Run all aleph-sv tests**

```
cargo test -p aleph-sv
```
Expected: every pre-existing test still green, plus 3 new prelude tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] apply_3q prelude: route Toffoli/CCZ to scalar dispatch

Matrix-shape dispatch (is_identity_8x8/is_toffoli/is_ccz) at apply_3q
entry. apply_3q_generic preserved as the fall-through. SIMD paths
land in later tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Toffoli Tier-A AVX-512 — packed swap (no outer-walk yet)

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

**Contract:** `t >= LANES_BITS (==2)`, `c_lo > t`, no external controls below `t`, `n >= 3`.

- [ ] **Step 1: Write SIMD-vs-scalar equivalence test under clean Tier-A contract**

Append:

```rust
#[cfg(test)]
mod toffoli_tier_a_tests {
    use super::*;
    use aleph_core::Complex;

    fn random_amps(n: u32, seed: u64) -> Vec<Complex> {
        let mut s = seed;
        let mut step = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        (0..(1 << n)).map(|_| Complex::new(step(), step())).collect()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tier_a_matches_scalar_clean_contract() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        // n=8 state, c0=5, c1=6, t=2 — c_lo=5 > t=2, target_bit=4 ≥ LANES=4. Clean Tier A.
        let mut simd = random_amps(8, 7);
        let mut scalar = simd.clone();
        // Direct SIMD invocation:
        unsafe {
            apply_toffoli_avx512_tier_a(&mut simd, 2, &[5, 6]);
        }
        apply_toffoli_scalar(&mut scalar, [5, 6, 2], &[]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tier_a_with_external_control_clean() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let mut simd = random_amps(8, 8);
        let mut scalar = simd.clone();
        // c0=3, c1=4, t=2, ext=[7]; full ctrl_mask = {3,4,7}, t_bit=4. c_lo=3 > t=2 ✓.
        unsafe {
            apply_toffoli_avx512_tier_a(&mut simd, 2, &[3, 4, 7]);
        }
        apply_toffoli_scalar(&mut scalar, [3, 4, 2], &[7]);
        for (x, y) in simd.iter().zip(scalar.iter()) {
            assert!((x.re - y.re).abs() < 1e-12);
            assert!((x.im - y.im).abs() < 1e-12);
        }
    }
}
```

- [ ] **Step 2: Run tests, expect failure (function not defined)**

- [ ] **Step 3: Implement `apply_toffoli_avx512_tier_a`**

Add after `apply_toffoli_scalar`:

```rust
/// # Safety
///
/// Caller MUST guarantee:
/// - host has AVX-512F (`is_x86_feature_detected!("avx512f")` true);
/// - `t_bit_idx = target as u32; (1 << target) >= LANES_AMPS (=4)`;
/// - every control bit position > target;
/// - `amps.len() == 1 << n` for some n ≥ 3;
/// - target qubit and all control qubits are distinct, all < n.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_a(
    amps: &mut [Complex],
    target: u32,
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 4; // complex amps per zmm
    const LANES_DOUBLES: usize = 8; // doubles per zmm
    let target_bit = 1usize << target;
    let mut ctrl_mask = 0usize;
    for &c in sorted_controls { ctrl_mask |= 1usize << c; }
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;
    let mut block_base = 0usize;
    while block_base < len {
        // Skip blocks where (a) target bit is set (we only handle pair-lo) or (b) ctrl_mask not satisfied.
        if (block_base & target_bit) != 0 {
            block_base += LANES;
            continue;
        }
        if (block_base & ctrl_mask) != ctrl_mask {
            block_base += LANES;
            continue;
        }
        // SAFETY: block_base + LANES ≤ len (n ≥ 3, target_bit ≥ LANES means len has at least 2 LANES blocks);
        // block_base | target_bit ≤ len - LANES because target_bit is in the "above-block" range.
        let lo_ptr = amps_ptr.add(block_base * 2);
        let hi_ptr = amps_ptr.add((block_base | target_bit) * 2);
        let z_lo = _mm512_loadu_pd(lo_ptr);
        let z_hi = _mm512_loadu_pd(hi_ptr);
        _mm512_storeu_pd(lo_ptr, z_hi);
        _mm512_storeu_pd(hi_ptr, z_lo);
        block_base += LANES;
    }
}
```

- [ ] **Step 4: Wire `dispatch_toffoli` to call Tier-A when contract holds**

Modify `dispatch_toffoli`:

```rust
fn dispatch_toffoli(amps: &mut [Complex], targets: [u32; 3], controls: &[u32]) {
    let c0 = targets[0];
    let c1 = targets[1];
    let t = targets[2];
    const LANES_BITS: u32 = 2;
    // Build sorted control vec including inner CCX controls.
    let mut all_ctrls: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
    all_ctrls.push(c0); all_ctrls.push(c1);
    for &c in controls { all_ctrls.push(c); }
    let c_lo = *all_ctrls.iter().min().unwrap();

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && t >= LANES_BITS
            && c_lo > t
        {
            // SAFETY: contract satisfied per spec §4.2.
            unsafe { apply_toffoli_avx512_tier_a(amps, t, &all_ctrls); }
            return;
        }
    }
    apply_toffoli_scalar(amps, targets, controls);
}
```

Add `smallvec` to `aleph-sv` `Cargo.toml` dependencies if not already present (it almost certainly is — see existing `SmallVec` uses in `backend.rs`).

- [ ] **Step 5: Run all tests**

```
cargo test -p aleph-sv
```
Expected: all green; new Tier-A tests pass on AVX-512 host, skip cleanly on non-AVX-512.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs crates/aleph-sv/Cargo.toml
git commit -m "[P1-08] Toffoli Tier-A AVX-512 packed swap (clean contract)

Inner loop: 2 zmm loads + 2 zmm stores per matching block.
Contract: t >= LANES_BITS, c_lo > t. Falls through to scalar otherwise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Toffoli Tier-A outer-walk — controls below target

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

**Contract:** `t >= LANES_BITS`, but **some** control bits below `t`. Use `expand_with_fixed` to outer-walk fixed-below bits and SIMD-walk the rest.

- [ ] **Step 1: Write equivalence test (control below target)**

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn tier_a_outer_walk_control_below_target() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    // c0=0, c1=5, t=2, n=7. c_lo=0 < t=2 — outer-walk over bit 0.
    let mut simd = random_amps(7, 11);
    let mut scalar = simd.clone();
    super::dispatch_toffoli(&mut simd, [0, 5, 2], &[]);
    apply_toffoli_scalar(&mut scalar, [0, 5, 2], &[]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run, expect failure (dispatch currently falls through to scalar — wrong tier classification)**

Hmm actually wait — current dispatch_toffoli falls to scalar when c_lo ≤ t, so the test passes accidentally via scalar. We need to PROVE that we're hitting the new Tier-A outer-walk path. Add an assert on a sentinel side-channel, OR (cleaner): write the test as a perf-sniff (skip equivalence-only).

Replace Step 1 test with two:

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn tier_a_outer_walk_control_below_matches_scalar() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    let mut simd = random_amps(7, 11);
    let mut scalar = simd.clone();
    // Force-call Tier-A outer-walk directly.
    let sorted = [0u32, 5u32];
    unsafe {
        apply_toffoli_avx512_tier_a_outer_walk(&mut simd, 2, &sorted);
    }
    apply_toffoli_scalar(&mut scalar, [0, 5, 2], &[]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}
```

- [ ] **Step 3: Implement `apply_toffoli_avx512_tier_a_outer_walk`**

Add after `apply_toffoli_avx512_tier_a`:

```rust
/// Tier-A outer-walk variant: handles controls *below* target by
/// folding their fixed-=1 contribution into the iteration base.
///
/// `sorted_controls` MUST be sorted ascending and de-duplicated.
///
/// # Safety
/// Same as `apply_toffoli_avx512_tier_a` plus:
/// - `target >= LANES_BITS`
/// - All elements of `sorted_controls` are distinct, ≠ target, < amps.len().log2().
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_a_outer_walk(
    amps: &mut [Complex],
    target: u32,
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let target_bit = 1usize << target;
    // Partition controls into below-target and above-target sets.
    let mut below: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
    let mut above: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
    for &c in sorted_controls {
        if c < target { below.push(c); } else { above.push(c); }
    }
    let ctrl_mask_above: usize = above.iter().map(|&c| 1usize << c).sum();
    let ctrl_mask_below: usize = below.iter().map(|&c| 1usize << c).sum();
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;
    // Outer-walk: every block_base must have ctrl_mask_below bits set AND
    // ctrl_mask_above bits set. We iterate block_base in LANES steps; the
    // skip check covers both masks.
    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & target_bit) != 0 {
            block_base += LANES;
            continue;
        }
        let mask_total = ctrl_mask_below | ctrl_mask_above;
        if (block_base & mask_total) != mask_total {
            block_base += LANES;
            continue;
        }
        let lo_ptr = amps_ptr.add(block_base * 2);
        let hi_ptr = amps_ptr.add((block_base | target_bit) * 2);
        let z_lo = _mm512_loadu_pd(lo_ptr);
        let z_hi = _mm512_loadu_pd(hi_ptr);
        _mm512_storeu_pd(lo_ptr, z_hi);
        _mm512_storeu_pd(hi_ptr, z_lo);
        block_base += LANES;
    }
}
```

**Note:** The simpler "skip blocks where ctrl_mask not satisfied" approach works for outer-walk too because the mask check covers both above and below bits uniformly. The `expand_with_fixed` outer-walk pattern from P1-07 is needed only when we want to *enumerate* fixed-above bits rather than mask-test; for Toffoli where we mask-test every block anyway, the simple form suffices.

- [ ] **Step 4: Update `dispatch_toffoli` to use outer-walk when needed**

```rust
fn dispatch_toffoli(amps: &mut [Complex], targets: [u32; 3], controls: &[u32]) {
    let c0 = targets[0];
    let c1 = targets[1];
    let t = targets[2];
    const LANES_BITS: u32 = 2;
    let mut all_ctrls: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
    all_ctrls.push(c0); all_ctrls.push(c1);
    for &c in controls { all_ctrls.push(c); }
    all_ctrls.sort();
    let c_lo = all_ctrls[0];

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && t >= LANES_BITS {
            if c_lo > t {
                // SAFETY: clean Tier-A contract (spec §4.2).
                unsafe { apply_toffoli_avx512_tier_a(amps, t, &all_ctrls); }
            } else {
                // SAFETY: Tier-A outer-walk contract (spec §4.2 ext).
                unsafe { apply_toffoli_avx512_tier_a_outer_walk(amps, t, &all_ctrls); }
            }
            return;
        }
    }
    apply_toffoli_scalar(amps, targets, controls);
}
```

- [ ] **Step 5: Run all tests**

```
cargo test -p aleph-sv
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] Toffoli Tier-A outer-walk for controls below target

Same packed-swap inner loop, mask test handles below-/above-target
controls uniformly. dispatch_toffoli picks clean vs outer-walk.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Toffoli Tier-B.0 — in-zmm swap for t=0

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

**Contract:** `t = 0`, `c_lo >= LANES_BITS = 2`, `n >= 3`.

- [ ] **Step 1: Write equivalence test**

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn tier_b0_matches_scalar() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    // c0=3, c1=4, t=0, n=6. c_lo=3 ≥ LANES_BITS=2.
    let mut simd = random_amps(6, 13);
    let mut scalar = simd.clone();
    super::dispatch_toffoli(&mut simd, [3, 4, 0], &[]);
    apply_toffoli_scalar(&mut scalar, [3, 4, 0], &[]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run, expect failure (dispatch_toffoli currently falls to scalar for t < LANES_BITS — that's correct but not exercising Tier B)**

Actually since dispatch falls to scalar (correct equivalent output), the test PASSES — but doesn't exercise the new path. Add a direct-call test:

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn tier_b0_direct_call_matches_scalar() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    let mut simd = random_amps(6, 14);
    let mut scalar = simd.clone();
    let sorted = [3u32, 4u32];
    unsafe { apply_toffoli_avx512_tier_b0(&mut simd, &sorted); }
    apply_toffoli_scalar(&mut scalar, [3, 4, 0], &[]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}
```

- [ ] **Step 3: Implement `apply_toffoli_avx512_tier_b0`**

`t = 0`: within a 4-amp zmm block, swap amp0 ↔ amp1 and amp2 ↔ amp3. As doubles within the zmm: indices `(0,1, 2,3, 4,5, 6,7)` → permute → `(2,3, 0,1, 6,7, 4,5)`.

```rust
/// # Safety
/// AVX-512F; target=0; all sorted_controls ≥ LANES_BITS=2;
/// amps.len() = 2^n for n ≥ 3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_b0(
    amps: &mut [Complex],
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let ctrl_mask: usize = sorted_controls.iter().map(|&c| 1usize << c).sum();
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;
    // Permute index: swap pairs (0,1) ↔ (2,3) within zmm — as doubles:
    // input lanes  = (a0re, a0im, a1re, a1im, a2re, a2im, a3re, a3im)
    // output lanes = (a1re, a1im, a0re, a0im, a3re, a3im, a2re, a2im)
    // Index vec    = (2, 3, 0, 1, 6, 7, 4, 5).
    let perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & ctrl_mask) != ctrl_mask {
            block_base += LANES;
            continue;
        }
        let p = amps_ptr.add(block_base * 2);
        let z = _mm512_loadu_pd(p);
        let z_perm = _mm512_permutexvar_pd(perm_idx, z);
        _mm512_storeu_pd(p, z_perm);
        block_base += LANES;
    }
}
```

**Endianness note:** `_mm512_set_epi64` takes args in HIGH-to-LOW order. The lane-7 index is the first argument. So `_mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2)` places `2` in lane-0, `3` in lane-1, `0` in lane-2, `1` in lane-3, `6` in lane-4, `7` in lane-5, `4` in lane-6, `5` in lane-7 — which is the desired `(2,3, 0,1, 6,7, 4,5)`. **Verify this with `objdump --disassemble` after Step 4.**

- [ ] **Step 4: Wire Tier-B.0 into dispatch_toffoli**

Replace the dispatch body's `#[cfg(target_arch = "x86_64")]` block:

```rust
#[cfg(target_arch = "x86_64")]
{
    if std::is_x86_feature_detected!("avx512f") {
        if t >= LANES_BITS {
            if c_lo > t {
                unsafe { apply_toffoli_avx512_tier_a(amps, t, &all_ctrls); }
            } else {
                unsafe { apply_toffoli_avx512_tier_a_outer_walk(amps, t, &all_ctrls); }
            }
            return;
        }
        if t == 0 && c_lo >= LANES_BITS {
            unsafe { apply_toffoli_avx512_tier_b0(amps, &all_ctrls); }
            return;
        }
    }
}
```

- [ ] **Step 5: Run tests**

```
cargo test -p aleph-sv kernels::aos::toffoli_tier_a_tests
cargo test -p aleph-sv kernels::aos::apply_3q_prelude_tests
```
Expected: green.

- [ ] **Step 6: Inspect codegen for Tier B.0**

```
cargo rustc -p aleph-sv --release -- --emit=asm
grep -A 10 "apply_toffoli_avx512_tier_b0" target/release/deps/aleph_sv-*.s | head -40
```
Verify the inner loop emits `vpermq` / `vpermpd zmm, zmm, zmm` (1 µop expected) plus `vmovupd` load + store. If LLVM splits the permute into multiple µops, capture and note for ADR 0012.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] Toffoli Tier-B.0 — in-zmm permute for t=0

_mm512_permutexvar_pd with idx (2,3,0,1, 6,7,4,5). Single permute
per matching 4-amp block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Toffoli Tier-B.1 — cross-128 swap for t=1

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

**Contract:** `t = 1`, `c_lo >= 2`, `n >= 3`.

- [ ] **Step 1: Write direct-call equivalence test**

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn tier_b1_direct_call_matches_scalar() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    let mut simd = random_amps(6, 15);
    let mut scalar = simd.clone();
    let sorted = [3u32, 4u32];
    unsafe { apply_toffoli_avx512_tier_b1(&mut simd, &sorted); }
    apply_toffoli_scalar(&mut scalar, [3, 4, 1], &[]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `apply_toffoli_avx512_tier_b1`**

`t = 1`: swap amp0 ↔ amp2 and amp1 ↔ amp3 within a 4-amp zmm. As doubles: input `(a0re,a0im, a1re,a1im, a2re,a2im, a3re,a3im)` → output `(a2re,a2im, a3re,a3im, a0re,a0im, a1re,a1im)`. Index vec `(4,5, 6,7, 0,1, 2,3)`.

```rust
/// # Safety
/// AVX-512F; target=1; sorted_controls all ≥ LANES_BITS; n ≥ 3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_toffoli_avx512_tier_b1(
    amps: &mut [Complex],
    sorted_controls: &[u32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let ctrl_mask: usize = sorted_controls.iter().map(|&c| 1usize << c).sum();
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;
    // Permute index (HIGH-to-LOW): want output (4,5, 6,7, 0,1, 2,3).
    // lane-0=4, lane-1=5, lane-2=6, lane-3=7, lane-4=0, lane-5=1, lane-6=2, lane-7=3.
    let perm_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & ctrl_mask) != ctrl_mask {
            block_base += LANES;
            continue;
        }
        let p = amps_ptr.add(block_base * 2);
        let z = _mm512_loadu_pd(p);
        let z_perm = _mm512_permutexvar_pd(perm_idx, z);
        _mm512_storeu_pd(p, z_perm);
        block_base += LANES;
    }
}
```

- [ ] **Step 4: Wire Tier-B.1 into dispatch_toffoli**

Add to the SIMD `cfg` block, after the Tier-B.0 branch:

```rust
if t == 1 && c_lo >= LANES_BITS {
    unsafe { apply_toffoli_avx512_tier_b1(amps, &all_ctrls); }
    return;
}
```

- [ ] **Step 5: Run all tests + codegen sanity check**

```
cargo test -p aleph-sv
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] Toffoli Tier-B.1 — cross-128 permute for t=1

_mm512_permutexvar_pd with idx (4,5,6,7, 0,1,2,3). Swaps lo-256/hi-256
amp groups within zmm.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: CCZ Tier-A AVX-512 — sign-flip via vxorpd

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

**Contract:** `mask_lo >= LANES_BITS = 2`, `n >= 3`.

- [ ] **Step 1: Write equivalence test**

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn ccz_tier_a_direct_call_matches_scalar() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    let mut simd = random_amps(7, 17);
    let mut scalar = simd.clone();
    let mask_bits = [2u32, 4u32, 6u32]; // mask_lo = 2 ≥ LANES_BITS.
    unsafe { apply_ccz_avx512_tier_a(&mut simd, &mask_bits); }
    apply_ccz_scalar(&mut scalar, [2, 4, 6], &[]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn ccz_tier_a_with_external_control() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    let mut simd = random_amps(7, 18);
    let mut scalar = simd.clone();
    let mask_bits = [2u32, 3u32, 4u32, 5u32];
    unsafe { apply_ccz_avx512_tier_a(&mut simd, &mask_bits); }
    apply_ccz_scalar(&mut scalar, [2, 3, 4], &[5]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `apply_ccz_avx512_tier_a`**

Add to `kernels/aos.rs` after the CCZ scalar:

```rust
/// # Safety
/// AVX-512F; mask_bits all distinct, all ≥ LANES_BITS=2 (so each 4-amp
/// zmm block has a single fixed value of mask_bits); n ≥ 3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ccz_avx512_tier_a(
    amps: &mut [Complex],
    mask_bits: &[u32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let mask: usize = mask_bits.iter().map(|&b| 1usize << b).sum();
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;
    // Sign-flip mask: 0x8000000000000000 in every double lane.
    let sign_mask = _mm512_set1_pd(-0.0_f64);
    let mut block_base = 0usize;
    while block_base < len {
        if (block_base & mask) != mask {
            block_base += LANES;
            continue;
        }
        let p = amps_ptr.add(block_base * 2);
        let z = _mm512_loadu_pd(p);
        let neg = _mm512_xor_pd(z, sign_mask);
        _mm512_storeu_pd(p, neg);
        block_base += LANES;
    }
}
```

- [ ] **Step 4: Wire `dispatch_ccz`**

Replace `dispatch_ccz`:

```rust
fn dispatch_ccz(amps: &mut [Complex], targets: [u32; 3], controls: &[u32]) {
    const LANES_BITS: u32 = 2;
    let mut all_mask: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
    for &q in &targets { all_mask.push(q); }
    for &c in controls { all_mask.push(c); }
    all_mask.sort();
    let mask_lo = all_mask[0];

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && mask_lo >= LANES_BITS {
            // SAFETY: clean Tier-A contract (spec §5.2).
            unsafe { apply_ccz_avx512_tier_a(amps, &all_mask); }
            return;
        }
    }
    apply_ccz_scalar(amps, targets, controls);
}
```

- [ ] **Step 5: Run tests + codegen check**

```
cargo test -p aleph-sv
cargo rustc -p aleph-sv --release -- --emit=asm 2>/dev/null
grep -A 10 "apply_ccz_avx512_tier_a" target/release/deps/aleph_sv-*.s | head -30
```
Verify `vxorpd zmm, zmm, zmm` emitted (1-µop sign flip). If LLVM emits `vmulpd` by `-1.0` instead, note for ADR 0012 risk R3 and try `_mm512_castsi512_pd(_mm512_xor_si512(...))` fallback.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] CCZ Tier-A AVX-512 — vxorpd sign-flip

Inner loop: 1 load + 1 xor + 1 store per matching block. Sign mask
is _mm512_set1_pd(-0.0). Codegen verified to emit vxorpd.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: CCZ Tier-A outer-walk — mask_lo < LANES_BITS

**Files:**
- Modify: `crates/aleph-sv/src/kernels/aos.rs`

**Contract:** `mask_lo < LANES_BITS` (some mask bits below LANES boundary), `n >= 3`. Outer-walk over below-LANES bits + SIMD over remaining.

- [ ] **Step 1: Write equivalence test**

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn ccz_tier_a_outer_walk_low_mask() {
    if !std::is_x86_feature_detected!("avx512f") { return; }
    let mut simd = random_amps(6, 19);
    let mut scalar = simd.clone();
    // mask = {0, 3, 5} — mask_lo=0 < LANES_BITS.
    unsafe { apply_ccz_avx512_tier_a_outer_walk(&mut simd, &[0, 3, 5]); }
    apply_ccz_scalar(&mut scalar, [0, 3, 5], &[]);
    for (x, y) in simd.iter().zip(scalar.iter()) {
        assert!((x.re - y.re).abs() < 1e-12);
        assert!((x.im - y.im).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement outer-walk variant**

For CCZ, a low mask bit (e.g. bit 0 or 1) means inside a 4-amp zmm block, only some lanes have `mask_bit = 1`. Approach: drop to a **per-lane-masked store** via AVX-512 mask intrinsics, OR fall back to scalar inside the block. Cleanest: compute the per-block lane mask once and use `_mm512_mask_xor_pd`.

For each 4-amp zmm block at `block_base`, the lane-`k` (k ∈ 0..4) corresponds to amp index `block_base + k`. The lane's contribution to mask-bits below LANES_BITS is `(k & mask_low_bits)`. So lanes where `k & mask_low_bits == mask_low_bits` need sign flip; others don't. Since each double lane is 2 lanes (re + im) per amp, the per-double-lane mask doubles up: `(k/2 & mask_low_bits) == mask_low_bits`.

```rust
/// # Safety
/// AVX-512F; mask_bits all distinct; n ≥ 3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_ccz_avx512_tier_a_outer_walk(
    amps: &mut [Complex],
    mask_bits: &[u32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 4;
    let mut mask_low: usize = 0;
    let mut mask_high: usize = 0;
    for &b in mask_bits {
        if b < 2 {
            mask_low |= 1usize << b;
        } else {
            mask_high |= 1usize << b;
        }
    }
    let full_mask = mask_low | mask_high;
    let len = amps.len();
    let amps_ptr = amps.as_mut_ptr() as *mut f64;
    let sign = _mm512_set1_pd(-0.0_f64);
    // Pre-compute the per-block lane mask: for amp k in {0,1,2,3}, flip if
    // (k & mask_low) == mask_low. Each amp occupies 2 doubles, so per-double
    // lane index is k = lane_idx >> 1.
    let lane_mask: u8 = {
        let mut m = 0u8;
        for k in 0..4u32 {
            if (k as usize & mask_low) == mask_low {
                m |= 1 << (2 * k);     // re lane
                m |= 1 << (2 * k + 1); // im lane
            }
        }
        m
    };
    let mut block_base = 0usize;
    while block_base < len {
        // For Tier-A semantics, we still need the HIGH mask satisfied at the block level.
        if (block_base & mask_high) != mask_high {
            block_base += LANES;
            continue;
        }
        let p = amps_ptr.add(block_base * 2);
        let z = _mm512_loadu_pd(p);
        let neg = _mm512_xor_pd(z, sign);
        // Mask-select: use `lane_mask` to choose between neg (where mask_low bits set)
        // and z (where not).
        let blended = _mm512_mask_blend_pd(lane_mask, z, neg);
        _mm512_storeu_pd(p, blended);
        block_base += LANES;
    }
}
```

**Edge case:** if `mask_low == 0` (no bits below), `lane_mask == 0xFF` if `(0 & 0) == 0` is true for all k → all lanes flipped. That's the Tier-A-clean behaviour, which is fine but wastes a blend; only call this function when `mask_low != 0`.

- [ ] **Step 4: Wire outer-walk into `dispatch_ccz`**

```rust
#[cfg(target_arch = "x86_64")]
{
    if std::is_x86_feature_detected!("avx512f") {
        if mask_lo >= LANES_BITS {
            unsafe { apply_ccz_avx512_tier_a(amps, &all_mask); }
        } else {
            unsafe { apply_ccz_avx512_tier_a_outer_walk(amps, &all_mask); }
        }
        return;
    }
}
```

- [ ] **Step 5: Run tests**

```
cargo test -p aleph-sv
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/kernels/aos.rs
git commit -m "[P1-08] CCZ Tier-A outer-walk for mask_lo < LANES_BITS

Per-block lane-mask + _mm512_mask_blend_pd to handle low-bit mask
configurations without leaving SIMD.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Property tests via aleph-test — involutivity, equivalence, symmetry

**Files:**
- Create or modify: `crates/aleph-sv/tests/multi_controlled_proptest.rs`

- [ ] **Step 1: Write proptest file**

Create `crates/aleph-sv/tests/multi_controlled_proptest.rs`:

```rust
//! Property tests for P1-08 specialised Toffoli + CCZ kernels.
//!
//! Verifies, on random state vectors:
//! - CCX∘CCX = I (involutivity).
//! - CCZ∘CCZ = I (involutivity).
//! - CCX(c0,c1,t) ≡ apply_3q_generic with Toffoli matrix.
//! - CCZ(q0,q1,q2) symmetric in qubit order.

use aleph_core::Complex;
use aleph_sv::backend::NaiveSvBackend;
use aleph_sv::state::StateVector;
use proptest::prelude::*;

fn random_state(n: u32, seed: u64) -> StateVector {
    // Use aleph-test's strategy if available; else local LCG.
    let mut s = seed;
    let mut step = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (u32::MAX as f64)
    };
    let mut amps: Vec<Complex> = (0..(1 << n))
        .map(|_| Complex::new(step() - 0.5, step() - 0.5)).collect();
    let norm: f64 = amps.iter().map(|c| c.re*c.re + c.im*c.im).sum::<f64>().sqrt();
    for c in &mut amps { *c = Complex::new(c.re/norm, c.im/norm); }
    StateVector { num_qubits: n, amps }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn ccx_is_involutive(
        n in 3u32..7,
        seed in any::<u64>(),
        c0 in 0u32..7,
        c1 in 0u32..7,
        t in 0u32..7,
    ) {
        prop_assume!(c0 != c1 && c0 != t && c1 != t);
        prop_assume!(c0 < n && c1 < n && t < n);
        let mut s = random_state(n, seed);
        let original = s.amps.clone();
        aleph_sv::kernels::aos::apply_3q(
            &mut s.amps,
            [c0, c1, t],
            &[],
            &toffoli_matrix(),
        );
        aleph_sv::kernels::aos::apply_3q(
            &mut s.amps,
            [c0, c1, t],
            &[],
            &toffoli_matrix(),
        );
        for (a, b) in s.amps.iter().zip(original.iter()) {
            prop_assert!((a.re - b.re).abs() < 1e-10);
            prop_assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn ccz_is_involutive(
        n in 3u32..7,
        seed in any::<u64>(),
        q0 in 0u32..7,
        q1 in 0u32..7,
        q2 in 0u32..7,
    ) {
        prop_assume!(q0 != q1 && q0 != q2 && q1 != q2);
        prop_assume!(q0 < n && q1 < n && q2 < n);
        let mut s = random_state(n, seed);
        let original = s.amps.clone();
        aleph_sv::kernels::aos::apply_3q(&mut s.amps, [q0, q1, q2], &[], &ccz_matrix());
        aleph_sv::kernels::aos::apply_3q(&mut s.amps, [q0, q1, q2], &[], &ccz_matrix());
        for (a, b) in s.amps.iter().zip(original.iter()) {
            prop_assert!((a.re - b.re).abs() < 1e-10);
            prop_assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn ccz_symmetric_in_qubit_order(
        n in 3u32..7,
        seed in any::<u64>(),
        q0 in 0u32..7,
        q1 in 0u32..7,
        q2 in 0u32..7,
    ) {
        prop_assume!(q0 != q1 && q0 != q2 && q1 != q2);
        prop_assume!(q0 < n && q1 < n && q2 < n);
        let s = random_state(n, seed);
        let mut a = s.amps.clone();
        let mut b = s.amps.clone();
        aleph_sv::kernels::aos::apply_3q(&mut a, [q0, q1, q2], &[], &ccz_matrix());
        aleph_sv::kernels::aos::apply_3q(&mut b, [q2, q0, q1], &[], &ccz_matrix());
        for (x, y) in a.iter().zip(b.iter()) {
            prop_assert!((x.re - y.re).abs() < 1e-10);
            prop_assert!((x.im - y.im).abs() < 1e-10);
        }
    }
}

fn toffoli_matrix() -> [[Complex; 8]; 8] {
    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    let mut m = [[z; 8]; 8];
    for i in 0..6 { m[i][i] = o; }
    m[6][7] = o;
    m[7][6] = o;
    m
}

fn ccz_matrix() -> [[Complex; 8]; 8] {
    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    let mut m = [[z; 8]; 8];
    for i in 0..7 { m[i][i] = o; }
    m[7][7] = Complex::new(-1.0, 0.0);
    m
}
```

**Note:** If `apply_3q` is `pub(crate)`, expose it through a `pub use` in `lib.rs` or temporarily make it `pub` for the duration of the test crate. The simpler path: re-export `kernels::aos` from `lib.rs` as `pub mod kernels` or via `#[cfg(test)] pub use`. Pick whichever matches existing test-file patterns in this crate.

- [ ] **Step 2: Run, expect failure (visibility / missing types)**

- [ ] **Step 3: Fix visibility — minimal `pub use` in `crates/aleph-sv/src/lib.rs`**

```rust
#[cfg(any(test, feature = "internal-test-api"))]
pub mod kernels;
```

Or simpler — make `kernels` `pub(crate)` already exists; this test is integration-style. Use the existing public Backend API if possible:

If routing through `NaiveSvBackend::apply_gate` is cleaner, rewrite the proptest with `GateInstance::new(Gate::Toffoli, vec![c0, c1, t]).unwrap()` and `backend.apply_gate(&mut state, &gi).unwrap()` instead of directly calling `kernels::aos::apply_3q`. Recommended path.

Rewrite the proptest accordingly. The internal SIMD dispatch will fire because `Backend::apply_gate` goes through `apply_3q` on the `M8x8` path.

- [ ] **Step 4: Run tests**

```
cargo test -p aleph-sv --test multi_controlled_proptest
```
Expected: all proptests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/tests/multi_controlled_proptest.rs crates/aleph-sv/src/lib.rs
git commit -m "[P1-08] property tests: CCX/CCZ involutivity, CCZ qubit-symmetry

64 random cases × 3 properties via proptest. Goes through the public
NaiveSvBackend::apply_gate path (exercises full dispatch chain).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Oracle tests vs Qiskit Aer — CCX, CCCX, Grover-CCZ, MCX k=7

**Files:**
- Modify or extend: `crates/aleph-sv/tests/oracle_qiskit.rs` (or wherever the existing oracle harness lives — check `tests/oracle*.rs` for the canonical pattern).

- [ ] **Step 1: Locate existing oracle harness**

```
find crates -name "oracle*" -type f
```

Read whatever oracle test pattern P0-10 set up. Mirror it for the new cases.

- [ ] **Step 2: Add CCX, CCCX, Grover-CCZ, MCX-k=7 fixtures**

For each new fixture, construct:
- The aleph circuit via `Circuit` builder or OpenQASM input.
- The equivalent Qiskit circuit JSON / OpenQASM string.
- Expected state vector tolerance: 1e-10 (per `docs/testing.md`).

Concrete fixtures:
1. `oracle_ccx_3q_all_basis` — `Toffoli(0,1,2)` on n=3, all 8 basis inputs.
2. `oracle_cccx_4q_all_basis` — Toffoli with 1 external control, n=4, all 16 basis inputs.
3. `oracle_grover_ccz_3q` — H⊗3 then Ccz(0,1,2) — expected: phase-marked uniform.
4. `oracle_mcx_k7_8q` — `Gate::X` on q7 with controls `[0,1,2,3,4,5,6]`, started from `|11111110⟩`. Expected: → `|11111111⟩`. **This validates P1-05's anti-diagonal kernel handles k=7 controls without bug.**

- [ ] **Step 3: Run oracle tests**

```
cargo test -p aleph-sv --test oracle_qiskit -- multi_controlled
```
Expected: all 4 new fixtures pass within 1e-10.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-sv/tests/oracle_qiskit.rs
git commit -m "[P1-08] oracle tests: CCX/CCCX/Grover-CCZ/MCX-k7 vs Qiskit

MCX-k7 fixture is the verification anchor for the BACKLOG bullet
'generic MCX with up to 8 controls' — routes through P1-05.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: SoA mirror — Toffoli + CCZ on SoA layout

**Files:**
- Modify: `crates/aleph-sv/src/kernels/soa.rs`

**Pattern:** Mirror Tasks 4-12 but on `(re: &mut [f64], im: &mut [f64])` instead of `&mut [Complex]`. Same dispatch contracts.

Read the existing `kernels::soa::apply_3q` for the SoA convention (already exists since P0-09). Convert to `apply_3q_generic` and add prelude + dispatch_toffoli_soa + dispatch_ccz_soa + Tier A/B/C variants.

**SoA LANES:** the existing P1-07 SoA path uses `LANES_SOA = 8` doubles per zmm (one stream at a time, no AoS interleave). Tier A contract is `target_bit >= LANES_SOA`. Adjust constants accordingly.

- [ ] **Step 1: Mirror Task 4 (scalar Toffoli SoA) — write test, fail, impl, pass.**

```rust
pub(crate) fn apply_toffoli_scalar_soa(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 3],
    external_controls: &[u32],
) {
    let target_bit = 1usize << targets[2];
    let mut ctrl_mask = (1usize << targets[0]) | (1usize << targets[1]);
    for &e in external_controls { ctrl_mask |= 1usize << e; }
    let len = re.len();
    for i in 0..len {
        if (i & ctrl_mask) == ctrl_mask && (i & target_bit) == 0 {
            re.swap(i, i | target_bit);
            im.swap(i, i | target_bit);
        }
    }
}
```

- [ ] **Step 2: Mirror Task 5 (scalar CCZ SoA)**

```rust
pub(crate) fn apply_ccz_scalar_soa(
    re: &mut [f64],
    im: &mut [f64],
    targets: [u32; 3],
    external_controls: &[u32],
) {
    let mut mask = (1usize << targets[0]) | (1usize << targets[1]) | (1usize << targets[2]);
    for &e in external_controls { mask |= 1usize << e; }
    let len = re.len();
    for i in 0..len {
        if (i & mask) == mask {
            re[i] = -re[i];
            im[i] = -im[i];
        }
    }
}
```

- [ ] **Step 3: Mirror Task 6 — `apply_3q` SoA prelude with detector routing.**

- [ ] **Step 4: Mirror Tasks 7-8 — Toffoli SoA Tier A (packed AVX-512 swap over re and im streams independently).**

SoA Tier A contract: `(1 << t) >= LANES_SOA = 8` (== `t >= 3`), `c_lo > t`. Per-stream swap with `_mm512_loadu_pd` / `_mm512_storeu_pd`. Apply twice — once for `re`, once for `im`.

**Pay attention to the P1-07 lesson:** SoA Tier-C sub-LANES underflow on n < log2(LANES_SOA) when external_control formulas use `c - sentinel - 1`. Re-validate against P1-07's pattern (memory entry [P1-07 merged]).

- [ ] **Step 5: Mirror Tasks 9-10 — Toffoli SoA Tier B for t < LANES_SOA_BITS=3.**

For SoA, `LANES_SOA_BITS = 3`. So Tier B covers `t ∈ {0, 1, 2}` — three sub-tiers. Permute indices for SoA are *single-stream*, not interleaved AoS pairs. Compute carefully:
- `t=0`: swap doubles within zmm at lane-pairs (0↔1, 2↔3, 4↔5, 6↔7). Perm idx `(1,0, 3,2, 5,4, 7,6)`.
- `t=1`: swap pairs (0↔2, 1↔3, 4↔6, 5↔7). Perm idx `(2,3, 0,1, 6,7, 4,5)`.
- `t=2`: swap (0↔4, 1↔5, 2↔6, 3↔7). Perm idx `(4,5,6,7, 0,1,2,3)`.

Each gets its own kernel; call once on `re` stream and once on `im` stream.

- [ ] **Step 6: Mirror Tasks 11-12 — CCZ SoA Tier A + outer-walk.**

CCZ on SoA: sign-flip via `_mm512_xor_pd` on the `re` stream and the `im` stream. Tier A contract: `mask_lo >= LANES_SOA_BITS = 3`. Outer-walk for lower.

- [ ] **Step 7: Run all SoA tests**

```
cargo test -p aleph-sv kernels::soa
```
Expected: green.

- [ ] **Step 8: Commit (one commit per kernel pair is fine — break this task into ≤ 4 commits for review)**

```bash
git add crates/aleph-sv/src/kernels/soa.rs
git commit -m "[P1-08] SoA mirror: scalar + Tier-A AVX-512 for Toffoli/CCZ"
# Optionally subsequent commits for Tier B and outer-walk variants.
```

---

### Task 16: Benchmarks — toffoli_chain, ccz_chain, mcx_k{2,4,6}

**Files:**
- Create: `crates/aleph-sv/benches/multi_controlled.rs`
- Modify: `crates/aleph-sv/Cargo.toml` — register the new bench target.

- [ ] **Step 1: Write the bench file**

```rust
//! P1-08 benchmarks: synthetic chains of Toffoli, CCZ, and MCX gates.

use aleph_core::{Complex, Gate, GateInstance};
use aleph_sv::backend::NaiveSvBackend;
use aleph_sv::state::StateVector;
use criterion::{criterion_group, criterion_main, Criterion};

fn toffoli_chain_bench(c: &mut Criterion, n: u32, gates: usize) {
    let bench_name = format!("toffoli_chain_n{}", n);
    c.bench_function(&bench_name, |b| {
        b.iter_with_setup(
            || StateVector::zero(n),
            |mut state| {
                let mut backend = NaiveSvBackend::new(0);
                for i in 0..gates {
                    let c0 = (i as u32) % n;
                    let c1 = ((i as u32) + 1) % n;
                    let t = ((i as u32) + 2) % n;
                    if c0 == c1 || c0 == t || c1 == t { continue; }
                    let gi = GateInstance::new(Gate::Toffoli, vec![c0, c1, t]).unwrap();
                    backend.apply_gate(&mut state, &gi).unwrap();
                }
                state
            },
        );
    });
}

fn ccz_chain_bench(c: &mut Criterion, n: u32, gates: usize) {
    let bench_name = format!("ccz_chain_n{}", n);
    c.bench_function(&bench_name, |b| {
        b.iter_with_setup(
            || StateVector::zero(n),
            |mut state| {
                let mut backend = NaiveSvBackend::new(0);
                for i in 0..gates {
                    let q0 = (i as u32) % n;
                    let q1 = ((i as u32) + 1) % n;
                    let q2 = ((i as u32) + 2) % n;
                    if q0 == q1 || q0 == q2 || q1 == q2 { continue; }
                    let gi = GateInstance::new(Gate::Ccz, vec![q0, q1, q2]).unwrap();
                    backend.apply_gate(&mut state, &gi).unwrap();
                }
                state
            },
        );
    });
}

fn mcx_bench(c: &mut Criterion, n: u32, k: u32) {
    let bench_name = format!("mcx_k{}_n{}", k, n);
    c.bench_function(&bench_name, |b| {
        b.iter_with_setup(
            || StateVector::zero(n),
            |mut state| {
                let mut backend = NaiveSvBackend::new(0);
                let controls: Vec<u32> = (0..k).collect();
                let target = k; // q_k is target, q_0..q_{k-1} are controls.
                let gi = GateInstance::new_with_controls(
                    Gate::X, vec![target], controls,
                ).unwrap();
                // Apply 100 times for measurement stability.
                for _ in 0..100 {
                    backend.apply_gate(&mut state, &gi).unwrap();
                }
                state
            },
        );
    });
}

fn benches(c: &mut Criterion) {
    toffoli_chain_bench(c, 15, 100);
    toffoli_chain_bench(c, 20, 100);
    ccz_chain_bench(c, 15, 100);
    ccz_chain_bench(c, 20, 100);
    mcx_bench(c, 20, 2);
    mcx_bench(c, 20, 4);
    mcx_bench(c, 20, 6);
}

criterion_group!(p108, benches);
criterion_main!(p108);
```

**Note:** verify `GateInstance::new_with_controls` exists; if not, use the public-fields constructor pattern (look at existing benches like `crates/aleph-sv/benches/soa_vs_naive.rs` for the canonical construction).

- [ ] **Step 2: Register bench in `crates/aleph-sv/Cargo.toml`**

```toml
[[bench]]
name = "multi_controlled"
harness = false
```

- [ ] **Step 3: Run bench locally (aarch64, smoke test)**

```
cargo bench -p aleph-sv --bench multi_controlled -- --warm-up-time 1 --measurement-time 3 toffoli_chain_n15
```
Expected: compiles, runs, produces criterion output. Numbers themselves are not the AC — that comes on EPYC.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-sv/benches/multi_controlled.rs crates/aleph-sv/Cargo.toml
git commit -m "[P1-08] benches: toffoli_chain, ccz_chain, mcx_k{2,4,6}

Synthetic chains for Tier-A perf measurement at n=15 (L2-resident)
and n=20 (DRAM-bound). MCX bench validates P1-05 anti-diagonal
kernel handles k=2,4,6 controls without regression.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: EPYC bench run — capture numbers

**Files:** none (operational task).

**Server:** `ssh root@195.154.249.85` (see memory `aleph-bench-server`).

- [ ] **Step 1: Confirm EPYC runner is healthy**

```
ssh root@195.154.249.85 'systemctl status github-runner'
```
If "stuck after timeout-cancellation" (memory pattern), `systemctl restart github-runner`. Wait for `Active: active (running)`.

- [ ] **Step 2: Sync working branch to EPYC**

Push the P1-08 branch to GitHub; do **not** merge to main while measuring. Memory lesson [Stage 0 merged]: "don't push to `benches/**` during manual EPYC measurement (CI Bench races on same runner)".

- [ ] **Step 3: Run benches with criterion baseline against pre-P1-08 main**

```
ssh root@195.154.249.85 << 'EOF'
cd /opt/aleph-bench/aleph
git fetch origin
git checkout p1-08-multi-controlled
RUSTFLAGS="-C target-cpu=native" cargo bench --bench multi_controlled -- --save-baseline p108
git checkout main
RUSTFLAGS="-C target-cpu=native" cargo bench --bench multi_controlled -- --baseline p108
EOF
```

Capture the criterion output for:
- `toffoli_chain_n15` (Tier A micro-AC: P1-08 ≥ 1.5× pre-P1-08 dispatch).
- `toffoli_chain_n20` (workload-AC: bandwidth-bound, expected small win).
- `ccz_chain_n15` (Tier A micro-AC: ≥ 2× pre).
- `ccz_chain_n20`.
- `mcx_k{2,4,6}_n20`.

- [ ] **Step 4: Re-run canonical workload benches for anti-regression**

```
ssh root@195.154.249.85 << 'EOF'
cd /opt/aleph-bench/aleph
RUSTFLAGS="-C target-cpu=native" cargo bench --bench soa_vs_naive -- qft_n20 grover_iter5_n20 random_brickwall_n20
EOF
```

Confirm no regression > 2 %.

- [ ] **Step 5: Save numbers to `docs/perf/p1-08-multi-controlled.md`**

Create that file with: side-by-side table, criterion baseline command used, RUSTFLAGS, host (EPYC 8124P), date.

- [ ] **Step 6: Commit numbers**

```bash
git add docs/perf/p1-08-multi-controlled.md
git commit -m "[P1-08] perf: EPYC numbers for Toffoli/CCZ/MCX benches"
```

---

### Task 18: BACKLOG amendment — `[P1-08]` per spec §1

**Files:**
- Modify: `BACKLOG.md` lines 907-937.

- [ ] **Step 1: Read existing entry**

Lines 907-937 (the `### [P1-08]` block).

- [ ] **Step 2: Amend per spec §1**

Update **Acceptance Criteria**:
```markdown
**Acceptance Criteria**

- [x] CCX, CCZ specialised (AoS + AVX-512 packed-complex)
- [x] Generic MCX with up to 8 controls — implicit via P1-05 anti-diagonal kernel; verified by `mcx_k6_n20` bench (mcx_k7 oracle test)
- [x] Benchmark: Toffoli specialised ≥ 1.5× scalar generic on `toffoli_chain_n15` (L2-resident, EPYC). CCZ ≥ 2× scalar generic on `ccz_chain_n15` (EPYC)
- [x] Workload-AC: no regression > 2 % on qft/grover/random n=20 (EPYC)
```

Update **Technical Details** to reference AoS + AVX-512 (per ADR 0008) and the spec doc.

Add a paragraph: `**Spec amendment:** see docs/superpowers/specs/2026-05-28-p1-08-multi-controlled-design.md §1.`

- [ ] **Step 3: Commit**

```bash
git add BACKLOG.md
git commit -m "[P1-08] BACKLOG: amend AC to AoS+AVX-512 + MCX-via-P1-05"
```

---

### Task 19: ADR 0012 — Multi-controlled SIMD pattern

**Files:**
- Create: `docs/decisions/0012-multi-controlled-simd-pattern.md`.

- [ ] **Step 1: Read existing ADRs for format reference**

```
ls docs/decisions/
```

Mirror the format of ADR 0008 (the most recent SIMD ADR).

- [ ] **Step 2: Write ADR 0012**

Outline:
- **Status:** Accepted.
- **Context:** P1-05/06/07 established the AoS+AVX-512 dispatch pattern. P1-08 extends it to 3q multi-controlled gates.
- **Decision:** Matrix-shape detector at apply_3q prelude → Tier A/B/C dispatch. Toffoli routes to fresh kernels; CCZ uses `vxorpd` sign-flip. MCX (Pauli-X with k controls) routes through P1-05's apply_1q anti-diagonal kernel — no new kernel needed.
- **Consequences:**
  - Adds ~600-900 LOC unsafe Toffoli/CCZ kernels (acceptable per Phase-1 perf budget).
  - Establishes "matrix-shape detector before SIMD" as the canonical pattern for all future N-qubit specialised paths.
  - Bandwidth ceiling (ADR 0008) applies: workload-AC is anti-regression, not win.
- **Lessons:**
  - Codebase `LANES` constant = 4 amp-units (NOT 8 doubles); always verify against existing kernel code before designing new tier contracts.
  - `_mm512_xor_pd` with `_mm512_set1_pd(-0.0)` is 1-µop sign-flip; faster than `vmulpd × -1.0`.
  - Pre-SIMD indexing-coverage tests (integer-only) catch bit-collision bugs that SIMD-only tests miss (P1-07 EPYC SIGSEGV class).

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0012-multi-controlled-simd-pattern.md
git commit -m "[P1-08] ADR 0012: multi-controlled SIMD dispatch pattern"
```

---

### Task 20: PR prep — squash-merge

**Files:** none (operational).

- [ ] **Step 1: Confirm all tests green on CI**

```
git push origin p1-08-multi-controlled
gh pr create --title "[P1-08] Multi-controlled gate kernels (Toffoli, CCZ, MCX)" --body "$(cat <<'EOF'
## Summary
- New AoS+AVX-512 dispatch_toffoli (Tier A clean + outer-walk + Tier B.0/B.1) and dispatch_ccz (Tier A clean + outer-walk via mask_blend).
- Matrix-shape prelude on apply_3q (is_identity_8x8 / is_toffoli / is_ccz) — fall-through to apply_3q_generic for arbitrary 8×8 matrices.
- Symmetric SoA mirror.
- MCX (Pauli-X with k controls) routes through P1-05's anti-diagonal kernel; verified by mcx_k6_n20 bench + mcx_k7 oracle test.

## Test plan
- [x] Unit tests: basis-state Toffoli/CCZ coverage on n=3,4 (all 8/16 inputs).
- [x] Indexing-coverage tests: exhaustive (c0,c1,t,ext) on n=6 for dispatch tier classification and pair disjointness.
- [x] Property tests (proptest, 64 cases): CCX/CCZ involutivity, CCZ qubit-symmetry, scalar-equivalence.
- [x] Oracle vs Qiskit Aer: CCX/CCCX/Grover-CCZ on n=3,4; MCX k=7 on n=8 (P1-05 anchor).
- [x] EPYC benches: toffoli_chain_n15/n20, ccz_chain_n15/n20, mcx_k{2,4,6}_n20. Numbers in docs/perf/p1-08-multi-controlled.md.
- [x] Anti-regression: qft/grover/random n=20 < 2 % delta on EPYC.

## Benchmark numbers (EPYC, single-thread, RUSTFLAGS=-C target-cpu=native)
[insert criterion table from Task 17]

## ADR
- New ADR 0012: multi-controlled SIMD dispatch pattern.

Closes #<issue-number>
EOF
)"
```

**Reminder:** use the **issue number**, NOT the PR number, in `Closes #` — repeated mistake P0-06..P0-11.

- [ ] **Step 2: Self-review the diff**

```
gh pr diff
```

Spend 5 minutes re-reading. Focus on unsafe blocks, dispatch contract guards, the smallvec stack-size sizing for outer-walk variants.

- [ ] **Step 3: Wait for `/code-review` agent + at least one EPYC bench-server validation cycle.**

Per memory pattern from P1-07/P1-05: code-review caught real ship-blockers in already-merged-CI work. Don't skip.

- [ ] **Step 4: Squash-merge once green + reviewed.**

```
gh pr merge --squash
```

- [ ] **Step 5: Update memory + close ticket.**

```bash
gh issue close <issue-number> --comment "Closed by PR #<pr-number>"
```

Add a memory file `p1-08-merged.md` summarising the merge per the pattern of prior memory entries.

---

## Wrap-up checklist (after Task 20)

- [ ] BACKLOG `[P1-08]` checkboxes all ticked.
- [ ] ADR 0012 committed.
- [ ] Memory file `p1-08-merged.md` added.
- [ ] Phase-1 Stage 1 is now **complete** — next step is Stage 2 (P1-09 onwards, IR optimisation passes) per `docs/superpowers/plans/2026-05-26-phase1-completion.md`.
