# [P2-06] Diagonal-Run Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse maximal runs of diagonal gates (absorbing interleaved `cx`s via monomial tracking) into a single `Instruction::DiagonalPhase` applied in one streaming state-vector pass, collapsing the QFT cphase ladder and cutting the memory-pass count ≥ 5×.

**Architecture:** A new `FuseDiagonalRuns` IR pass walks runs of `{diagonal gates ∪ Cnot}`, tracks a GF(2) bit-permutation `P` and an accumulated set of symbolic phase terms `D` (each term = AND-of-parity-masks → angle). When the run's net `P == I` it emits one `DiagonalPhase`; otherwise it leaves the run unchanged. A new SV kernel (scalar + AVX-512, AoS + SoA) evaluates the terms per amplitude and multiplies by `e^{iφ}`.

**Tech Stack:** Rust 2021, `rayon` (`par_units`), AVX-512 (`VPOPCNTQ`) intrinsics, `criterion`, `proptest`.

**Reference spec:** `docs/superpowers/specs/2026-06-02-p2-06-diagonal-fusion-design.md`.

**Golden-rule reminders for the worker:**
- Correctness first. Oracle/property tests gate every change; tolerance 1e-12 for amplitudes, **global phase included**.
- No `unwrap()`/`expect()` in library code (tests OK). `?` with `thiserror`.
- No `unsafe` without a `// SAFETY:` block. AVX-512 intrinsics are the legitimate use.
- AVX-512 kernels only build/run on x86_64 (EPYC); local dev is aarch64 and **silently skips the SIMD path** — validate SIMD tasks on the EPYC box ([[aleph-bench-server]], `ssh root@195.154.249.85`). `cargo check --target x86_64-unknown-linux-gnu` validates SIMD codegen locally without running.
- Verify the bench box is idle (`uptime` ~0; `pgrep -af "cargo bench|bencher run|Runner.Worker"`) before any perf measurement — CLAUDE.md rule.

---

## File Structure

**Create:**
- `crates/aleph-ir/src/diagonal_phase.rs` — `DiagonalPhase`, `PhaseTerm` types + their unit tests.
- `crates/aleph-ir/src/passes/fuse_diagonal.rs` — `FuseDiagonalRuns` pass + `Perm` (GF(2) tracker) + `diagonal_to_terms` extractor + tests.
- `crates/aleph-sv/src/kernels/diagonal_phase.rs` — scalar + AVX-512 phase-eval kernels (AoS & SoA), shared term-evaluation helpers + tests.
- `crates/aleph-oracle/tests/diagonal_fusion_oracle.rs` — Tier-1 equivalence (raw + pipeline) + generic-state property test.

**Modify:**
- `crates/aleph-ir/src/instruction.rs` — add `DiagonalPhase` variant; extend `used_qubits`/`used_clbits`.
- `crates/aleph-ir/src/lib.rs` — `pub mod diagonal_phase;` + re-exports.
- `crates/aleph-ir/src/passes/mod.rs` — `pub mod fuse_diagonal;`, re-export, wire into `default_pipeline()`.
- `crates/aleph-backend/src/lib.rs` — `Backend::apply_diagonal_phase` (default Err) + run-loop arms in `run_with_outcomes`/`run_optimized_with_outcomes`.
- `crates/aleph-sv/src/backend.rs` (AoS) and `crates/aleph-sv/src/soa_backend.rs` (SoA) — override `apply_diagonal_phase`.
- `crates/aleph-sv/src/kernels/mod.rs` — `pub(crate) mod diagonal_phase;`.
- `crates/aleph-ir/src/circuit.rs` and any QASM `emit` — refuse / pass-through `DiagonalPhase` (emit refuses).
- `benches/benches/tier1_scaling.rs` (and `benches/src/lib.rs` if a builder-QFT bench fixture is missing) — before/after fixture + builder QFT.
- `docs/perf/phase2.md` — append a P2-06 section with EPYC numbers.

---

## Task 1: IR types — `DiagonalPhase` and `PhaseTerm`

**Files:**
- Create: `crates/aleph-ir/src/diagonal_phase.rs`
- Modify: `crates/aleph-ir/src/lib.rs`

- [ ] **Step 1: Write the failing test** (in `crates/aleph-ir/src/diagonal_phase.rs`)

```rust
//! Symbolic multi-qubit diagonal operator produced by `FuseDiagonalRuns`.
//!
//! The amplitude at basis index `x` is multiplied by
//! `exp(i * Σ_t angle_t * [∀ m ∈ conds_t: parity(m & x) == 1])`,
//! where `parity(v) = v.count_ones() & 1`. An empty `conds` is a
//! vacuously-true (global-phase) term. Masks are `u64`; the producer
//! asserts `n_qubits <= 64`.

use smallvec::SmallVec;

#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTerm {
    /// AND of these parity-conditions. Empty == global phase.
    pub conds: SmallVec<[u64; 2]>,
    pub angle: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalPhase {
    pub n_qubits: u32,
    pub terms: Vec<PhaseTerm>,
}

impl DiagonalPhase {
    /// Real phase (radians) applied to amplitude index `x`.
    pub fn phase_at(&self, x: u64) -> f64 {
        let mut phi = 0.0;
        for t in &self.terms {
            if t.conds.iter().all(|&m| (m & x).count_ones() & 1 == 1) {
                phi += t.angle;
            }
        }
        phi
    }

    /// Union of all qubit indices referenced by any condition mask.
    pub fn support_mask(&self) -> u64 {
        self.terms
            .iter()
            .flat_map(|t| t.conds.iter())
            .fold(0u64, |acc, &m| acc | m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn phase_at_single_bit_term() {
        // p(θ) on qubit 1: fires when bit 1 set.
        let dp = DiagonalPhase {
            n_qubits: 3,
            terms: vec![PhaseTerm { conds: smallvec![0b010], angle: 0.5 }],
        };
        assert_eq!(dp.phase_at(0b000), 0.0);
        assert_eq!(dp.phase_at(0b010), 0.5);
        assert_eq!(dp.phase_at(0b011), 0.5);
        assert_eq!(dp.phase_at(0b101), 0.0);
    }

    #[test]
    fn phase_at_and_of_two_conds_is_controlled() {
        // controlled-Phase(θ), ctrl 0, tgt 1: fires only when both set.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm { conds: smallvec![0b01, 0b10], angle: 0.7 }],
        };
        assert_eq!(dp.phase_at(0b00), 0.0);
        assert_eq!(dp.phase_at(0b01), 0.0);
        assert_eq!(dp.phase_at(0b10), 0.0);
        assert_eq!(dp.phase_at(0b11), 0.7);
    }

    #[test]
    fn empty_conds_is_global_phase() {
        let dp = DiagonalPhase {
            n_qubits: 1,
            terms: vec![PhaseTerm { conds: smallvec![], angle: 1.1 }],
        };
        assert_eq!(dp.phase_at(0), 1.1);
        assert_eq!(dp.phase_at(1), 1.1);
    }

    #[test]
    fn support_mask_unions_all_conds() {
        let dp = DiagonalPhase {
            n_qubits: 4,
            terms: vec![
                PhaseTerm { conds: smallvec![0b0011], angle: 0.1 },
                PhaseTerm { conds: smallvec![0b1000, 0b0010], angle: 0.2 },
            ],
        };
        assert_eq!(dp.support_mask(), 0b1011);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-ir --lib diagonal_phase`
Expected: FAIL — `module 'diagonal_phase' not found` / unresolved.

- [ ] **Step 3: Wire the module**

In `crates/aleph-ir/src/lib.rs`, add near the other `pub mod` lines:

```rust
pub mod diagonal_phase;
pub use diagonal_phase::{DiagonalPhase, PhaseTerm};
```

(The type bodies are already written in Step 1.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-ir --lib diagonal_phase`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-ir/src/diagonal_phase.rs crates/aleph-ir/src/lib.rs
git commit -m "[P2-06] IR types: DiagonalPhase + PhaseTerm

Symbolic multi-qubit diagonal: phase(x) = Σ angle·[AND of parities].
phase_at + support_mask with unit tests.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `Instruction::DiagonalPhase` variant + instruction queries

**Files:**
- Modify: `crates/aleph-ir/src/instruction.rs`

- [ ] **Step 1: Write the failing test** (append to the `tests` mod in `instruction.rs`)

```rust
#[test]
fn used_qubits_diagonal_phase_unions_support() {
    use crate::diagonal_phase::{DiagonalPhase, PhaseTerm};
    use smallvec::smallvec as sv;
    let dp = DiagonalPhase {
        n_qubits: 5,
        terms: vec![
            PhaseTerm { conds: sv![0b00011], angle: 0.1 }, // bits 0,1
            PhaseTerm { conds: sv![0b10000, 0b00100], angle: 0.2 }, // bits 4,2
        ],
    };
    let inst = Instruction::DiagonalPhase(Box::new(dp));
    let mut q = inst.used_qubits().to_vec();
    q.sort();
    assert_eq!(q, vec![0, 1, 2, 4]);
    assert!(inst.used_clbits().is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-ir --lib used_qubits_diagonal_phase`
Expected: FAIL — no variant `DiagonalPhase`.

- [ ] **Step 3: Add the variant and extend the queries**

In `crates/aleph-ir/src/instruction.rs`:

Add `use crate::diagonal_phase::DiagonalPhase;` to the imports. Add the variant to the enum (after `Barrier`):

```rust
    /// A fused multi-qubit diagonal operator produced by
    /// `passes::FuseDiagonalRuns`. Never produced by the parser; only
    /// exists post-optimization. Boxed to keep the enum small.
    DiagonalPhase(Box<DiagonalPhase>),
```

Extend `used_qubits`'s match with:

```rust
            Instruction::DiagonalPhase(dp) => {
                let mask = dp.support_mask();
                for q in 0..dp.n_qubits {
                    if (mask >> q) & 1 == 1 {
                        out.push(q);
                    }
                }
            }
```

`used_clbits` already returns empty for non-`Measure`; its `if let` needs no change, but confirm it still compiles (the match is not exhaustive there — it uses `if let`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-ir --lib instruction`
Expected: PASS (existing + the new test).

- [ ] **Step 5: Fix any non-exhaustive-match build errors across the workspace**

Run: `cargo build -p aleph-ir`
Expected: PASS. If `layers.rs` or any pass has an exhaustive `match` on `Instruction`, add a minimal arm there now (e.g. in `layers.rs`, treat `DiagonalPhase` as occupying its `used_qubits()`; in passes, default to leaving it untouched / treating as a run-breaker). Show each added arm in the commit.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-ir/src/instruction.rs crates/aleph-ir/src/layers.rs
git commit -m "[P2-06] Add Instruction::DiagonalPhase variant + used_qubits

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: GF(2) permutation tracker `Perm`

**Files:**
- Create: `crates/aleph-ir/src/passes/fuse_diagonal.rs`
- Modify: `crates/aleph-ir/src/passes/mod.rs`

- [ ] **Step 1: Write the failing test** (in `fuse_diagonal.rs`)

```rust
//! `FuseDiagonalRuns` — fuses runs of {diagonal gates ∪ Cnot} into one
//! `DiagonalPhase`, absorbing interleaved `cx` via monomial tracking.
//! See docs/superpowers/specs/2026-06-02-p2-06-diagonal-fusion-design.md.

/// GF(2) bit-permutation tracker. `row[i]` is the mask such that the
/// i-th output bit equals `parity(row[i] & x)`. Starts as identity.
/// A `cx(c, t)` on the LEFT of the accumulated product does
/// `row[t] ^= row[c]` (design §1.2).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Perm {
    row: Vec<u64>,
}

impl Perm {
    pub(crate) fn identity(n: u32) -> Self {
        Perm { row: (0..n).map(|i| 1u64 << i).collect() }
    }
    pub(crate) fn cx(&mut self, control: u32, target: u32) {
        let c = self.row[control as usize];
        self.row[target as usize] ^= c;
    }
    /// Mask for the image of input bit `b` under the current permutation.
    pub(crate) fn image(&self, b: u32) -> u64 {
        self.row[b as usize]
    }
    pub(crate) fn is_identity(&self) -> bool {
        self.row.iter().enumerate().all(|(i, &m)| m == 1u64 << i)
    }
}

#[cfg(test)]
mod perm_tests {
    use super::*;

    #[test]
    fn identity_is_identity() {
        assert!(Perm::identity(4).is_identity());
        assert_eq!(Perm::identity(3).image(2), 0b100);
    }

    #[test]
    fn single_cx_xors_control_into_target() {
        let mut p = Perm::identity(3);
        p.cx(0, 1); // row[1] ^= row[0]
        assert_eq!(p.image(1), 0b011);
        assert_eq!(p.image(0), 0b001);
        assert!(!p.is_identity());
    }

    #[test]
    fn cx_pair_cancels_to_identity() {
        // The QFT invariant: cx(c,t) applied twice nets to identity.
        let mut p = Perm::identity(4);
        p.cx(3, 1);
        p.cx(3, 1);
        assert!(p.is_identity());
    }

    #[test]
    fn distinct_target_cx_pairs_all_cancel() {
        let mut p = Perm::identity(5);
        for t in [3u32, 2, 1, 0] {
            p.cx(4, t);
        }
        for t in [3u32, 2, 1, 0] {
            p.cx(4, t);
        }
        assert!(p.is_identity());
    }
}
```

- [ ] **Step 2: Wire the module**

In `crates/aleph-ir/src/passes/mod.rs`, add `pub mod fuse_diagonal;` with the other `pub mod`s.

- [ ] **Step 3: Run test to verify it passes**

Run: `cargo test -p aleph-ir --lib perm_tests`
Expected: PASS (4 tests). (The type is defined in Step 1; this task has no separate "fail" stage beyond module wiring — confirm it compiles and passes.)

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-ir/src/passes/fuse_diagonal.rs crates/aleph-ir/src/passes/mod.rs
git commit -m "[P2-06] GF(2) permutation tracker for diagonal fusion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `diagonal_to_terms` — gate → phase terms (multilinear expansion)

**Files:**
- Modify: `crates/aleph-ir/src/passes/fuse_diagonal.rs`

**Background for the worker:** Any diagonal gate over its `k` target qubits has
a diagonal whose phase function `f(b_1..b_k)` (radians) expands uniquely as a
multilinear polynomial `f = Σ_{S⊆targets} α_S · Π_{i∈S} b_i`, where the monomial
`Π b_i` is exactly the AND-of-ones condition. Coefficients come from Möbius
inversion: `α_S = Σ_{T⊆S} (-1)^{|S\T|} f(T)`. Each nonzero `α_S` → one
`PhaseTerm` with `conds = [perm.image(q) for q in S]`. External controls gate the
whole operator, so every emitted term gets the control images appended to its
`conds` (the global `S=∅` term thereby becomes conditioned on the controls).
`k ≤ 3` for all our gates (`Ccz`), so `2^k ≤ 8`.

- [ ] **Step 1: Write the failing test** (append to `fuse_diagonal.rs`, new `mod terms_tests`)

```rust
#[cfg(test)]
mod terms_tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;
    use std::f64::consts::PI;

    fn dp_phase(terms: &[crate::PhaseTerm], x: u64) -> f64 {
        let mut phi = 0.0;
        for t in terms {
            if t.conds.iter().all(|&m| (m & x).count_ones() & 1 == 1) {
                phi += t.angle;
            }
        }
        phi
    }

    #[test]
    fn plain_phase_is_one_single_bit_term() {
        let g = GateInstance::new(Gate::Phase(0.5.into()), smallvec![1u32]);
        let perm = Perm::identity(3);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        // fires only when bit 1 set
        assert!((dp_phase(&terms, 0b000) - 0.0).abs() < 1e-15);
        assert!((dp_phase(&terms, 0b010) - 0.5).abs() < 1e-15);
        assert!((dp_phase(&terms, 0b110) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn controlled_phase_fires_only_on_both_set() {
        let g = GateInstance::controlled(
            Gate::Phase(PI.into()),
            smallvec![0u32],   // target
            smallvec![1u32],   // control
        );
        let perm = Perm::identity(2);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        assert!(dp_phase(&terms, 0b00).abs() < 1e-15);
        assert!(dp_phase(&terms, 0b01).abs() < 1e-15);
        assert!(dp_phase(&terms, 0b10).abs() < 1e-15);
        assert!((dp_phase(&terms, 0b11) - PI).abs() < 1e-12);
    }

    #[test]
    fn cz_phase_pi_on_eleven() {
        let g = GateInstance::new(Gate::Cz, smallvec![0u32, 1u32]);
        let perm = Perm::identity(2);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        for x in [0b00u64, 0b01, 0b10] {
            assert!(dp_phase(&terms, x).abs() < 1e-15, "x={x:b}");
        }
        // e^{iπ} on |11>
        let p = dp_phase(&terms, 0b11);
        assert!(((p - PI).rem_euclid(2.0 * PI)).abs() < 1e-12 || (p + PI).abs() < 1e-12);
    }

    #[test]
    fn conjugated_phase_picks_up_control_bit() {
        // p(θ) on bit 1, but recorded while P has cx(0,1) applied
        // (perm.image(1) = bits {0,1}). Then it fires on parity(b0^b1).
        let g = GateInstance::new(Gate::Phase(0.5.into()), smallvec![1u32]);
        let mut perm = Perm::identity(2);
        perm.cx(0, 1);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        assert!(dp_phase(&terms, 0b00).abs() < 1e-15);        // b0^b1 = 0
        assert!((dp_phase(&terms, 0b01) - 0.5).abs() < 1e-15); // 1^0 = 1
        assert!((dp_phase(&terms, 0b10) - 0.5).abs() < 1e-15); // 0^1 = 1
        assert!(dp_phase(&terms, 0b11).abs() < 1e-15);        // 1^1 = 0
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-ir --lib terms_tests`
Expected: FAIL — `diagonal_to_terms` not found.

- [ ] **Step 3: Implement `diagonal_to_terms`** (in `fuse_diagonal.rs`)

```rust
use crate::PhaseTerm;
use aleph_core::{Gate, GateInstance, GateMatrix};
use smallvec::SmallVec;

/// Drop terms whose angle is within this of a 2π multiple (matches the
/// 1q-diagonal kernel's intent; see `kernels::DIAGONAL_EPS_SQ`).
pub(crate) const PHASE_EPS: f64 = 1e-12;

/// Expand a diagonal `GateInstance` into additive phase terms in the
/// current permuted basis. Returns `None` if the gate is not diagonal
/// (caller must have checked, but we re-guard) or has a symbolic/
/// non-finite parameter.
pub(crate) fn diagonal_to_terms(g: &GateInstance, perm: &Perm) -> Option<Vec<PhaseTerm>> {
    if !g.gate.is_diagonal() {
        return None;
    }
    let targets = &g.qubits; // Gate's own qubits (targets); arity == targets.len()
    let k = targets.len();
    debug_assert!(k <= 3, "diagonal gates have ≤3 targets in Phase 0/1/2");

    // f(pattern) = arg of the diagonal entry at that target-bit pattern.
    // pattern bit j (LSB..) corresponds to targets[k-1-j]? No: define
    // pattern bit position p in 0..k mapping to targets[p], and read the
    // matrix diagonal using the gate's MSB-first convention.
    let diag = diagonal_entries(g)?; // length 2^k, indexed MSB-first by targets
    // f over the k target bits, indexed so bit p (value 1<<p) == targets[p] set.
    // diag is MSB-first: row index r has targets[0] as the MSB.
    let f = |subset_pattern: usize| -> f64 {
        // subset_pattern bit p set => targets[p] == 1.
        // Build the MSB-first matrix row index.
        let mut r = 0usize;
        for p in 0..k {
            if (subset_pattern >> p) & 1 == 1 {
                r |= 1 << (k - 1 - p);
            }
        }
        diag[r].arg()
    };

    // Möbius inversion: α_S = Σ_{T⊆S} (-1)^{|S\T|} f(T).
    let mut terms: Vec<PhaseTerm> = Vec::new();
    for s in 0..(1usize << k) {
        // iterate subsets T of S
        let mut alpha = 0.0;
        let mut t = s;
        loop {
            let sign = if ((s ^ t).count_ones()) & 1 == 0 { 1.0 } else { -1.0 };
            alpha += sign * f(t);
            if t == 0 {
                break;
            }
            t = (t - 1) & s;
        }
        if alpha.rem_euclid(2.0 * std::f64::consts::PI).min(
            (2.0 * std::f64::consts::PI) - alpha.rem_euclid(2.0 * std::f64::consts::PI),
        ) < PHASE_EPS
        {
            continue; // negligible
        }
        // conds = images of target qubits in S, plus all external controls.
        let mut conds: SmallVec<[u64; 2]> = SmallVec::new();
        for p in 0..k {
            if (s >> p) & 1 == 1 {
                conds.push(perm.image(targets[p]));
            }
        }
        for &c in g.controls.iter() {
            conds.push(perm.image(c));
        }
        terms.push(PhaseTerm { conds, angle: alpha });
    }
    Some(terms)
}

/// Diagonal entries (length 2^arity) of a diagonal gate, MSB-first.
/// `None` for symbolic/non-finite params.
fn diagonal_entries(g: &GateInstance) -> Option<Vec<aleph_core::Complex>> {
    match g.gate.matrix().ok()? {
        GateMatrix::M2x2(m) => Some(vec![m[0][0], m[1][1]]),
        GateMatrix::M4x4(m) => Some(vec![m[0][0], m[1][1], m[2][2], m[3][3]]),
        GateMatrix::M8x8(m) => Some((0..8).map(|i| m[i][i]).collect()),
    }
}
```

**Note for the worker:** confirm the exact `GateMatrix` variant names
(`M2x2/M4x4/M8x8`) and the `Complex::arg()` method against
`crates/aleph-core/src/gate/kinds.rs`; adjust if they differ. If `matrix()`
returns a `Result`, the `.ok()?` is correct; if a 3-qubit `GateMatrix` variant
doesn't exist yet, restrict `diagonal_entries` to `k ≤ 2` and handle `Ccz`
explicitly (it's `diag(1,…,1,-1)` → single term `conds=[img(q0),img(q1),img(q2)], angle=π`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-ir --lib terms_tests`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-ir/src/passes/fuse_diagonal.rs
git commit -m "[P2-06] diagonal_to_terms: multilinear gate→phase-term extractor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: The pass — `FuseDiagonalRuns` (walk, fuse, conservative fallback)

**Files:**
- Modify: `crates/aleph-ir/src/passes/fuse_diagonal.rs`, `crates/aleph-ir/src/passes/mod.rs`

- [ ] **Step 1: Write the failing test** (new `mod pass_tests` in `fuse_diagonal.rs`)

```rust
#[cfg(test)]
mod pass_tests {
    use super::*;
    use crate::passes::Pass;
    use crate::{Circuit, Instruction};
    use std::f64::consts::PI;

    /// Brute-force the unfused circuit's diagonal as a full 2^n phase
    /// vector by applying each gate to basis states. Only valid for
    /// purely-diagonal circuits (no H etc.).
    fn brute_phase(c: &Circuit) -> Vec<f64> {
        // ... apply via NaiveSvBackend on each basis state would pull a dep;
        // instead reuse the oracle in Task 9. Here, assert structural facts.
        unimplemented!()
    }

    #[test]
    fn cx_p_cx_reconstructs_controlled_phase() {
        // p(π/4) q1 ; cx(1,0) ; p(-π/4) q0 ; cx(1,0) ; p(π/4) q0
        // == cp(π/2) on (control=1, target=0): phase π/2 iff bits 0&1 set.
        let mut c = Circuit::new(2, 0);
        c.add_gate(aleph_core::GateInstance::new(
            aleph_core::Gate::Phase((PI / 4.0).into()), smallvec::smallvec![1u32])).unwrap();
        c.cnot(1, 0).unwrap();
        c.add_gate(aleph_core::GateInstance::new(
            aleph_core::Gate::Phase((-PI / 4.0).into()), smallvec::smallvec![0u32])).unwrap();
        c.cnot(1, 0).unwrap();
        c.add_gate(aleph_core::GateInstance::new(
            aleph_core::Gate::Phase((PI / 4.0).into()), smallvec::smallvec![0u32])).unwrap();

        let stats = FuseDiagonalRuns.run(&mut c).unwrap();
        assert_eq!(c.len(), 1, "whole run collapses to one op");
        let dp = match &c.instructions()[0] {
            Instruction::DiagonalPhase(dp) => dp.clone(),
            other => panic!("expected DiagonalPhase, got {other:?}"),
        };
        // cp(π/2): phase π/2 only on |11>.
        for x in 0u64..4 {
            let want = if x == 0b11 { PI / 2.0 } else { 0.0 };
            let got = dp.phase_at(x);
            let d = (got - want).rem_euclid(2.0 * PI);
            assert!(d < 1e-12 || (2.0 * PI - d) < 1e-12, "x={x:b} got={got} want={want}");
        }
        assert!(stats.transformations >= 1);
    }

    #[test]
    fn run_with_nonidentity_perm_is_left_unchanged() {
        // A lone cx leaves net permutation != I → no fusion.
        let mut c = Circuit::new(2, 0);
        c.add_gate(aleph_core::GateInstance::new(
            aleph_core::Gate::Phase(0.3.into()), smallvec::smallvec![0u32])).unwrap();
        c.cnot(0, 1).unwrap();
        let before = c.instructions().to_vec();
        FuseDiagonalRuns.run(&mut c).unwrap();
        assert_eq!(c.len(), before.len(), "non-identity perm: run untouched");
        assert!(c.instructions().iter().all(|i| !matches!(i, Instruction::DiagonalPhase(_))));
    }

    #[test]
    fn lone_single_qubit_diag_run_not_fused() {
        // Cost model: 1-qubit-only diagonal run stays for Fuse1qRuns.
        let mut c = Circuit::new(1, 0);
        c.t(0).unwrap();
        c.s(0).unwrap();
        FuseDiagonalRuns.run(&mut c).unwrap();
        assert!(c.instructions().iter().all(|i| !matches!(i, Instruction::DiagonalPhase(_))));
    }

    #[test]
    fn barrier_is_a_hard_fence() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(aleph_core::GateInstance::new(
            aleph_core::Gate::Phase(0.3.into()), smallvec::smallvec![0u32])).unwrap();
        c.cnot(0, 1).unwrap();
        c.barrier(smallvec::smallvec![0u32, 1u32]).unwrap();
        c.add_gate(aleph_core::GateInstance::new(
            aleph_core::Gate::Phase(0.3.into()), smallvec::smallvec![1u32])).unwrap();
        c.cnot(0, 1).unwrap();
        // Each side of the barrier nets to identity? No: left side ends mid-perm.
        // Just assert the barrier still exists and nothing crosses it.
        FuseDiagonalRuns.run(&mut c).unwrap();
        assert!(c.instructions().iter().any(|i| matches!(i, Instruction::Barrier(_))));
    }
}
```

(Delete the `brute_phase`/`unimplemented!` stub — it was a thinking note; the
real cross-check lives in the Task 9 oracle.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-ir --lib pass_tests`
Expected: FAIL — `FuseDiagonalRuns` not found.

- [ ] **Step 3: Implement the pass** (in `fuse_diagonal.rs`)

```rust
use crate::passes::{Pass, PassError, PassStats};
use crate::{Circuit, DiagonalPhase, Instruction};

/// Fuses maximal runs of {diagonal gates ∪ Cnot} into one
/// `DiagonalPhase`, absorbing interleaved `cx` via the `Perm` tracker.
/// Conservative: a run whose net permutation is not the identity is
/// re-emitted verbatim. See the design doc for the monomial algebra.
pub struct FuseDiagonalRuns;

impl Pass for FuseDiagonalRuns {
    fn name(&self) -> &'static str {
        "FuseDiagonalRuns"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.num_qubits();
        let before = circuit.len();
        // u64 masks ⇒ ≤64 qubits. n is capped at 28 by the backends, but
        // assert here so the invariant is enforced at the IR boundary.
        if n > 64 {
            return Err(PassError::InternalInvariant("DiagonalPhase mask width > 64"));
        }

        let input = circuit.instructions().to_vec();
        let mut out: Vec<Instruction> = Vec::with_capacity(input.len());
        let mut transformations = 0u64;

        let mut i = 0usize;
        while i < input.len() {
            // Is this the start of a fusable run?
            if !is_run_member(&input[i]) {
                out.push(input[i].clone());
                i += 1;
                continue;
            }
            // Collect the maximal run [i, j).
            let mut j = i;
            while j < input.len() && is_run_member(&input[j]) {
                j += 1;
            }
            let run = &input[i..j];
            match fuse_run(run, n) {
                Some(dp) => {
                    out.push(Instruction::DiagonalPhase(Box::new(dp)));
                    transformations += 1;
                }
                None => out.extend_from_slice(run), // conservative re-emit
            }
            i = j;
        }

        let after = out.len();
        circuit.replace_instructions(out); // see note below
        Ok(PassStats { gates_before: before, gates_after: after, transformations })
    }
}

/// A run member is a diagonal `Gate` or a `Cnot` `Gate`. Everything else
/// (Measure/Reset/Barrier/non-diagonal gate/existing DiagonalPhase) breaks
/// the run.
fn is_run_member(inst: &Instruction) -> bool {
    match inst {
        Instruction::Gate(g) => g.gate.is_diagonal() || matches!(g.gate, aleph_core::Gate::Cnot),
        _ => false,
    }
}

/// Try to fuse a run into one `DiagonalPhase`. Returns `None` (caller
/// re-emits verbatim) when the net permutation is not identity, or the
/// cost model rejects the run, or any gate has a non-extractable matrix.
fn fuse_run(run: &[Instruction], n: u32) -> Option<DiagonalPhase> {
    let mut perm = Perm::identity(n);
    let mut terms: Vec<crate::PhaseTerm> = Vec::new();
    let mut diag_gate_count = 0usize;
    let mut support: u64 = 0;

    for inst in run {
        let g = match inst {
            Instruction::Gate(g) => g,
            _ => return None, // not reachable given is_run_member
        };
        if matches!(g.gate, aleph_core::Gate::Cnot) {
            // qubits = [control, target]
            perm.cx(g.qubits[0], g.qubits[1]);
            support |= 1u64 << g.qubits[0];
            support |= 1u64 << g.qubits[1];
        } else {
            let mut t = diagonal_to_terms(g, &perm)?;
            diag_gate_count += 1;
            for q in g.qubits.iter().chain(g.controls.iter()) {
                support |= 1u64 << q;
            }
            terms.append(&mut t);
        }
    }

    if !perm.is_identity() {
        return None; // conservative
    }
    // Cost model: only fuse multi-qubit runs that absorbed ≥2 diagonal gates.
    let span = support.count_ones();
    if span <= 1 || diag_gate_count < 2 {
        return None;
    }

    let terms = canonicalize(terms);
    if terms.is_empty() {
        // Whole run is (global-phase-free) identity — still replace it with an
        // empty DiagonalPhase to drop the gates? No: dropping changes count but
        // is correct. Return an empty DiagonalPhase (a no-op pass over state).
    }
    Some(DiagonalPhase { n_qubits: n, terms })
}

/// Merge terms with identical condition-sets (order-insensitive within a
/// term) by summing angles; drop negligible angles; deterministic order.
fn canonicalize(terms: Vec<crate::PhaseTerm>) -> Vec<crate::PhaseTerm> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<Vec<u64>, f64> = BTreeMap::new();
    for mut t in terms {
        t.conds.sort_unstable();
        let key: Vec<u64> = t.conds.to_vec();
        *map.entry(key).or_insert(0.0) += t.angle;
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    map.into_iter()
        .filter_map(|(conds, angle)| {
            let a = angle.rem_euclid(two_pi);
            if a < PHASE_EPS || (two_pi - a) < PHASE_EPS {
                None
            } else {
                Some(crate::PhaseTerm { conds: conds.into(), angle })
            }
        })
        .collect()
}
```

**Note for the worker:** check how the existing passes rebuild the instruction
vector. `fuse_1q.rs`/`fuse_2q.rs` mutate `circuit.instructions`. If there is no
`replace_instructions` setter, follow the exact mechanism those passes use
(e.g. a `pub(crate)` field or a `Circuit::set_instructions`). Do **not**
`std::mem::take` then fail (the P1-09 lesson — Err would leave the circuit
empty); build `out` fully, then swap in one shot.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-ir --lib pass_tests`
Expected: PASS (4 tests).

- [ ] **Step 5: Run full IR test suite (catch regressions)**

Run: `cargo test -p aleph-ir`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-ir/src/passes/fuse_diagonal.rs crates/aleph-ir/src/passes/mod.rs
git commit -m "[P2-06] FuseDiagonalRuns pass: monomial-run fusion + conservative fallback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Export the pass; do NOT wire into default_pipeline yet

**Files:**
- Modify: `crates/aleph-ir/src/passes/mod.rs`

- [ ] **Step 1: Add the re-export**

In `passes/mod.rs`, alongside the other `pub use`:

```rust
pub use fuse_diagonal::FuseDiagonalRuns;
```

- [ ] **Step 2: Build**

Run: `cargo build -p aleph-ir`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-ir/src/passes/mod.rs
git commit -m "[P2-06] Re-export FuseDiagonalRuns (not yet in default_pipeline)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(The pass is wired into `default_pipeline()` only in Task 11, after the kernel
exists — so `cargo test --workspace` never runs a `DiagonalPhase` through a
backend that can't apply it.)

---

## Task 7: Backend trait method + run-loop dispatch (default = error)

**Files:**
- Modify: `crates/aleph-backend/src/lib.rs`

- [ ] **Step 1: Write the failing test** (in the `tests` mod of `aleph-backend/src/lib.rs`, or a new `tests/diagonal_phase_dispatch.rs`)

```rust
#[test]
fn run_loop_dispatches_diagonal_phase_to_backend_method() {
    // A backend whose apply_diagonal_phase is the default impl must
    // surface UnsupportedInstruction when the circuit contains one.
    use aleph_ir::{Circuit, Instruction, DiagonalPhase, PhaseTerm};
    use smallvec::smallvec;
    let mut c = Circuit::new(1, 0);
    c.h(0).unwrap(); // ensure non-empty/allocate path
    c.push_instruction(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
        n_qubits: 1,
        terms: vec![PhaseTerm { conds: smallvec![1u64], angle: 0.5 }],
    }))); // use whatever raw-instruction API exists; else build via a pass
    // Pick a backend that does NOT override apply_diagonal_phase, or assert
    // the SV backend (Task 8) returns Ok. For the default-path test use a
    // minimal stub backend defined in this test module.
    // ... see note.
}
```

**Note for the worker:** if there's no public API to push a raw `Instruction`
onto a `Circuit`, define a tiny stub `Backend` in the test module and call
`run` on a hand-built circuit, OR fold this test into Task 8 against the real SV
backend (preferred — the SV backend *does* implement it, so test the happy path
there and keep only the trait-signature change here).

- [ ] **Step 2: Add the trait method with a default impl**

In the `Backend` trait (after `probabilities`):

```rust
    /// Apply a fused multi-qubit diagonal (`Instruction::DiagonalPhase`).
    ///
    /// Default implementation rejects it as unsupported, so backends that
    /// never see optimized circuits (MPS/stabilizer, for now) need not
    /// implement it. State-vector backends override this.
    fn apply_diagonal_phase(
        &mut self,
        _state: &mut Self::State,
        _dp: &aleph_ir::DiagonalPhase,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedInstruction { kind: "diagonal_phase" })
    }
```

- [ ] **Step 3: Add the run-loop arm**

In `run_with_outcomes`'s `match inst { … }` add:

```rust
            aleph_ir::Instruction::DiagonalPhase(dp) => {
                backend.apply_diagonal_phase(&mut state, dp)?;
            }
```

Do the same in `run_optimized_with_outcomes` (find its instruction loop —
it iterates the optimized circuit's instructions the same way).

- [ ] **Step 4: Build the workspace**

Run: `cargo build --workspace`
Expected: PASS. Fix any other exhaustive `match inst` sites the compiler flags
(e.g. CLI, oracle helpers) with the same dispatch or an explicit error.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-backend/src/lib.rs
git commit -m "[P2-06] Backend::apply_diagonal_phase (default Err) + run-loop dispatch

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Scalar kernel (AoS `CpuState`) + naive backend impl

**Files:**
- Create: `crates/aleph-sv/src/kernels/diagonal_phase.rs`
- Modify: `crates/aleph-sv/src/kernels/mod.rs`, `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Write the failing test** (in `kernels/diagonal_phase.rs`)

```rust
//! Diagonal-phase kernel: ψ[x] *= exp(i·phase(x)) in one streaming pass.

use aleph_core::Complex;
use aleph_ir::DiagonalPhase;

/// Evaluate the real phase for amplitude index `x`. Hot inner helper —
/// kept tiny so it inlines into both scalar and SIMD-tail paths.
#[inline(always)]
pub(crate) fn phase_at(dp: &DiagonalPhase, x: u64) -> f64 {
    let mut phi = 0.0;
    for t in &dp.terms {
        let mut all = true;
        for &m in &t.conds {
            if (m & x).count_ones() & 1 == 0 {
                all = false;
                break;
            }
        }
        if all {
            phi += t.angle;
        }
    }
    phi
}

/// Scalar, rayon-parallel application over an AoS amplitude slice.
pub(crate) fn apply_diagonal_phase_scalar_aos(amps: &mut [Complex], dp: &DiagonalPhase) {
    use crate::kernels::{par_blocks, ComplexPtr};
    use crate::kernels::tuning::default_policy; // adjust to actual policy accessor
    let len = amps.len();
    let p = ComplexPtr(amps.as_mut_ptr());
    par_blocks(default_policy(len), len, len, |k| k, move |k| {
        // SAFETY: each k is a distinct index; writes never alias across tasks.
        let amp = unsafe { &mut *p.ptr().add(k) };
        let phi = phase_at(dp, k as u64);
        let (s, co) = phi.sin_cos();
        let re = amp.re * co - amp.im * s;
        let im = amp.re * s + amp.im * co;
        amp.re = re;
        amp.im = im;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_ir::PhaseTerm;
    use smallvec::smallvec;

    #[test]
    fn applies_controlled_phase_to_amplitudes() {
        // cp(π/2) on (0,1): multiply ψ[11] by i, others unchanged.
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm { conds: smallvec![0b01, 0b10], angle: std::f64::consts::FRAC_PI_2 }],
        };
        let mut amps = vec![Complex::new(1.0, 0.0); 4];
        apply_diagonal_phase_scalar_aos(&mut amps, &dp);
        for x in 0..3 {
            assert!((amps[x] - Complex::new(1.0, 0.0)).norm() < 1e-12, "x={x}");
        }
        // ψ[11] = e^{iπ/2} = i
        assert!((amps[3] - Complex::new(0.0, 1.0)).norm() < 1e-12);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-sv --lib diagonal_phase`
Expected: FAIL — module not wired.

- [ ] **Step 3: Wire the module + naive override**

In `crates/aleph-sv/src/kernels/mod.rs`: `pub(crate) mod diagonal_phase;`

In `crates/aleph-sv/src/backend.rs`, inside `impl Backend for NaiveSvBackend`, add:

```rust
    fn apply_diagonal_phase(
        &mut self,
        state: &mut Self::State,
        dp: &aleph_ir::DiagonalPhase,
    ) -> Result<(), BackendError> {
        // CpuState::amps is AlignedBuf<Complex>; deref to &mut [Complex].
        crate::kernels::diagonal_phase::apply_diagonal_phase_scalar_aos(&mut state.amps, dp);
        Ok(())
    }
```

**Note:** confirm the AoS `CpuState` field/accessor for a mutable slice;
`state.amps` derefs via `AlignedBuf: DerefMut<[Complex]>` (P2-02). Also confirm
the chunk-policy accessor name in `kernels::tuning` (P2-04) and fix
`default_policy(len)` accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-sv --lib diagonal_phase`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/diagonal_phase.rs crates/aleph-sv/src/kernels/mod.rs crates/aleph-sv/src/backend.rs
git commit -m "[P2-06] Scalar AoS diagonal-phase kernel + naive backend impl

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Scalar kernel (SoA `SoaState`) + SoA backend impl

**Files:**
- Modify: `crates/aleph-sv/src/kernels/diagonal_phase.rs`, `crates/aleph-sv/src/soa_backend.rs`

- [ ] **Step 1: Write the failing test** (append to `kernels/diagonal_phase.rs` tests)

```rust
    #[test]
    fn applies_controlled_phase_soa() {
        use super::apply_diagonal_phase_scalar_soa;
        let dp = DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm { conds: smallvec![0b01, 0b10], angle: std::f64::consts::FRAC_PI_2 }],
        };
        let mut re = vec![1.0f64; 4];
        let mut im = vec![0.0f64; 4];
        apply_diagonal_phase_scalar_soa(&mut re, &mut im, &dp);
        for x in 0..3 {
            assert!((re[x] - 1.0).abs() < 1e-12 && im[x].abs() < 1e-12, "x={x}");
        }
        assert!(re[3].abs() < 1e-12 && (im[3] - 1.0).abs() < 1e-12);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p aleph-sv --lib diagonal_phase`
Expected: FAIL — `apply_diagonal_phase_scalar_soa` not found.

- [ ] **Step 3: Implement the SoA scalar kernel**

```rust
/// Scalar, rayon-parallel application over SoA (split re/im) arrays.
pub(crate) fn apply_diagonal_phase_scalar_soa(re: &mut [f64], im: &mut [f64], dp: &DiagonalPhase) {
    use crate::kernels::{par_blocks, BlockPtr};
    use crate::kernels::tuning::default_policy;
    let len = re.len();
    debug_assert_eq!(len, im.len());
    let rp = BlockPtr(re.as_mut_ptr());
    let ip = BlockPtr(im.as_mut_ptr());
    par_blocks(default_policy(len), len, len, |k| k, move |k| {
        // SAFETY: distinct k ⇒ disjoint writes; rp/ip never alias each other.
        let r = unsafe { &mut *rp.0.add(k) };
        let i = unsafe { &mut *ip.0.add(k) };
        let phi = phase_at(dp, k as u64);
        let (s, co) = phi.sin_cos();
        let nr = *r * co - *i * s;
        let ni = *r * s + *i * co;
        *r = nr;
        *i = ni;
    });
}
```

In `crates/aleph-sv/src/soa_backend.rs`, add the `apply_diagonal_phase` override
calling `apply_diagonal_phase_scalar_soa(&mut state.re, &mut state.im, dp)`
(confirm the SoA state's field names/accessors).

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p aleph-sv --lib diagonal_phase`
Expected: PASS (both AoS + SoA tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/diagonal_phase.rs crates/aleph-sv/src/soa_backend.rs
git commit -m "[P2-06] Scalar SoA diagonal-phase kernel + SoA backend impl

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Oracle + property tests (raw pass AND via pipeline)

**Files:**
- Create: `crates/aleph-oracle/tests/diagonal_fusion_oracle.rs`

This is the correctness gate. It must compare a fused circuit's final state to
the unfused circuit's final state at 1e-12 **including global phase**, on a
**generic** input state (per the P1-13 lesson).

- [ ] **Step 1: Write the test**

```rust
//! P2-06 oracle: FuseDiagonalRuns preserves the exact statevector
//! (global phase included) across Tier-1 fixtures, both as a standalone
//! pass and through default_pipeline().

use aleph_backend::{run, Backend};
use aleph_ir::passes::{FuseDiagonalRuns, Pass, PassPipeline};
use aleph_ir::Circuit;
use aleph_sv::SoaSvBackend; // or whichever SV backend the oracle harness uses

/// Build a generic (non-|0…0>) initial state by prepending a layer of
/// H and non-trivial rotations, then run; compare fused vs unfused.
fn assert_state_equiv(mut base: Circuit) {
    let mut fused = base.clone();
    FuseDiagonalRuns.run(&mut fused).unwrap();

    let mut be1 = SoaSvBackend::default();
    let mut be2 = SoaSvBackend::default();
    let s_unfused = run(&mut be1, &base).unwrap();
    let s_fused = run(&mut be2, &fused).unwrap();

    let a = s_unfused.amplitudes();
    let b = s_fused.amplitudes();
    assert_eq!(a.len(), b.len());
    for (x, (u, f)) in a.iter().zip(b.iter()).enumerate() {
        assert!((*u - *f).norm() < 1e-12, "amp {x}: {u:?} vs {f:?}");
    }
    let _ = &mut base;
}

fn generic_prefix(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
        c.rx(0.3 + 0.1 * q as f64, q).unwrap();
        c.ry(0.7 - 0.05 * q as f64, q).unwrap();
    }
    c
}

#[test]
fn builder_qft_equiv_raw_and_pipeline() {
    for n in [3u32, 5, 8] {
        // generic prefix + QFT body (diagonal ladder absorbs cx)
        let mut c = generic_prefix(n);
        // append builder QFT gates onto the SAME circuit
        append_qft(&mut c, n);
        assert_state_equiv(c.clone());

        // and via the full pipeline
        let mut piped = c.clone();
        PassPipeline::default_pipeline().run(&mut piped).unwrap();
        let mut be1 = SoaSvBackend::default();
        let mut be2 = SoaSvBackend::default();
        let s0 = run(&mut be1, &c).unwrap();
        let s1 = run(&mut be2, &piped).unwrap();
        for (u, f) in s0.amplitudes().iter().zip(s1.amplitudes().iter()) {
            assert!((*u - *f).norm() < 1e-12);
        }
    }
}

#[test]
fn decomposed_fixture_qft_equiv() {
    // Parse scripts/qiskit-baseline/circuits/qft_n08.qasm-style p+cx form,
    // or build the decomposed form directly, for small n; assert equiv.
    // Use the smallest decomposed QFT available; if only n25 exists,
    // construct a small p+cx decomposition inline.
    let n = 5u32;
    let mut c = generic_prefix(n);
    append_qft_decomposed(&mut c, n); // p + cx form
    assert_state_equiv(c);
}

// append_qft / append_qft_decomposed: copy the ladder construction from
// benches/src/lib.rs::qft_circuit (controlled-Phase form) and a p+cx
// decomposition (cp(θ) = p(θ/2)@c; cx; p(-θ/2)@t; cx; p(θ/2)@t).
```

**Note for the worker:** factor `append_qft` from `benches/src/lib.rs::qft_circuit`
(currently it builds a fresh `Circuit`; extract the ladder into a helper that
appends to an existing circuit, or inline it in the test). For
`append_qft_decomposed`, emit the textbook `cp` decomposition so the test
exercises the `cx`-absorption path explicitly.

- [ ] **Step 2: Run**

Run: `cargo test -p aleph-oracle --test diagonal_fusion_oracle`
Expected: PASS. If a global-phase mismatch appears, the bug is almost certainly a
dropped `arg(d0)` global-phase term in `diagonal_to_terms` (Task 4) — fix there.

- [ ] **Step 3: Add a proptest** (generic random diagonal+cx run ≡ sequential)

Append a `proptest!` that builds a random sequence of `{p(θ), rz(θ), z, s, t, cz, cnot}`
on `n ∈ 2..=6` qubits, applies the conservative pass, and asserts state
equivalence on a generic prefix to 1e-12. Reuse `assert_state_equiv`.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-oracle/tests/diagonal_fusion_oracle.rs benches/src/lib.rs
git commit -m "[P2-06] Oracle + proptest: fused ≡ unfused (global phase) on generic state

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Wire into `default_pipeline()` + idempotence

**Files:**
- Modify: `crates/aleph-ir/src/passes/mod.rs`

- [ ] **Step 1: Write the failing test** (in `passes/mod.rs` tests)

```rust
#[test]
fn default_pipeline_fuses_diagonal_ladder_and_is_idempotent() {
    // Builder-style controlled-Phase ladder between two H's collapses.
    let mut c = Circuit::new(3, 0);
    c.h(2).unwrap();
    for (t, k) in [(0u32, 2u32), (1, 2), (0, 1)] {
        c.add_gate(aleph_core::GateInstance::controlled(
            aleph_core::Gate::Phase(0.5.into()),
            smallvec::smallvec![t],
            smallvec::smallvec![k],
        )).unwrap();
    }
    let once = c.clone();
    let mut a = once.clone();
    let s1 = PassPipeline::default_pipeline().run(&mut a).unwrap();
    let mut b = a.clone();
    let s2 = PassPipeline::default_pipeline().run(&mut b).unwrap();
    // idempotent: running again changes nothing
    assert_eq!(a.len(), b.len());
    assert_eq!(s2.transformations, 0, "second pass is a no-op");
    // a DiagonalPhase was produced
    assert!(a.instructions().iter().any(|i| matches!(i, aleph_ir::Instruction::DiagonalPhase(_))));
    let _ = s1;
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p aleph-ir --lib default_pipeline_fuses_diagonal`
Expected: FAIL — no `DiagonalPhase` produced (pass not in pipeline).

- [ ] **Step 3: Wire into the pipeline**

In `default_pipeline()` change the vec to:

```rust
        Self::new(vec![
            Box::new(CancelInversePairs),
            Box::new(DeadCodeElim),
            Box::new(FuseDiagonalRuns),
            Box::new(Fuse1qRuns),
            Box::new(Fuse2q),
        ])
```

Update the doc comment above `default_pipeline` to mention `FuseDiagonalRuns`
runs before `Fuse2q` so raw `cx`s are still absorbable, and that an emitted
`DiagonalPhase` is a run-breaker for the pass (idempotence).

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p aleph-ir --lib`
Expected: PASS — including the two existing `default_pipeline_*` tests (confirm
they still hold; the diagonal pass must not disturb their non-diagonal inputs).

- [ ] **Step 5: Full workspace test**

Run: `cargo test --workspace`
Expected: PASS. The pipeline now emits `DiagonalPhase`, and every backend used in
tests either applies it (SV) or the test doesn't route optimized circuits through
a non-SV backend.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-ir/src/passes/mod.rs
git commit -m "[P2-06] Wire FuseDiagonalRuns into default_pipeline (before Fuse2q)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: QASM emit refuses `DiagonalPhase`

**Files:**
- Modify: the QASM emit module (find it: `grep -rn "fn emit\|UnsupportedGate\|to_qasm" crates/aleph-parser/src crates/aleph-ir/src`).

- [ ] **Step 1: Write the failing test**

In the emit module's tests, assert that emitting a circuit containing a
`DiagonalPhase` returns a clear error (e.g. `EmitError::UnsupportedInstruction`),
not a panic. Build the circuit by running `FuseDiagonalRuns` on a small ladder.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p aleph-parser emit` (adjust crate/test name)
Expected: FAIL (non-exhaustive match or wrong behavior).

- [ ] **Step 3: Implement the refusal**

Add the `Instruction::DiagonalPhase(_)` arm to the emit match returning the
unsupported-instruction error variant. Document the asymmetry: `DiagonalPhase`
exists only post-optimization and is not round-trippable in v1.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p aleph-parser emit`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "[P2-06] QASM emit refuses DiagonalPhase (post-opt-only, not round-trippable)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: AVX-512 kernels (AoS + SoA), EPYC-validated, measure-then-decide

**Files:**
- Modify: `crates/aleph-sv/src/kernels/diagonal_phase.rs` (add `#[cfg(target_arch = "x86_64")]` SIMD paths), and the AoS/SoA backend dispatch to prefer SIMD when `is_x86_feature_detected!("avx512f")` (+ `avx512vpopcntdq`).

**Read first:** the existing `apply_1q_avx512` in `crates/aleph-sv/src/kernels/aos.rs`
for the project's intrinsic style, `is_x86_feature_detected!` dispatch, and the
`// SAFETY:` block conventions. Mirror that structure.

- [ ] **Step 1: Local codegen check before writing intrinsics**

Run: `cargo check --target x86_64-unknown-linux-gnu -p aleph-sv`
Expected: PASS (validates SIMD code compiles without an EPYC box; per P2-04 lesson).

- [ ] **Step 2: Write the SIMD kernel (AoS)**

Process 8 amplitudes per `zmm`. For each lane index vector `x` (8 consecutive
indices), compute per-term: `parity = VPOPCNTQ(mask & x) & 1`; AND the per-cond
parities into a per-lane fire-mask; conditionally add `angle` via masked FMA to a
`phi` accumulator. Then `e^{iφ}` — **resolve the sincos by measurement (spec §4):**
implement option (ii) first (extract the 8 `phi` lanes to a `[f64;8]`, scalar
`sin_cos`, reload), since it is simplest and likely bandwidth-hidden. Multiply the
interleaved complex amplitudes by `(cos, sin)` with `zmm` shuffles (reuse the
complex-multiply pattern already in `aos.rs`).

Guard with `#[target_feature(enable = "avx512f,avx512vpopcntdq")]` and a
`// SAFETY:` block (feature-detected at the call site, disjoint per-block writes).

- [ ] **Step 3: Dispatch**

In the AoS backend `apply_diagonal_phase`, branch:
```rust
#[cfg(target_arch = "x86_64")]
{
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vpopcntdq") {
        // SAFETY: features detected above.
        unsafe { apply_diagonal_phase_avx512_aos(&mut state.amps, dp) };
        return Ok(());
    }
}
apply_diagonal_phase_scalar_aos(&mut state.amps, dp);
Ok(())
```
Same shape for SoA in Step 5.

- [ ] **Step 4: SIMD-equivalence test (runs on EPYC; auto-skips elsewhere)**

Add a test that builds a random `DiagonalPhase`, applies scalar vs SIMD to two
copies of a random state, and asserts bit-for-bit-close (1e-13) equality. It only
exercises SIMD on x86_64 with the features; on aarch64 it just compares scalar to
itself (still valid).

Run on EPYC: `ssh root@195.154.249.85`, then in the repo
`RUSTFLAGS="-C target-cpu=native" cargo test -p aleph-sv --lib diagonal_phase`
Expected: PASS.

- [ ] **Step 5: SoA SIMD kernel + dispatch + test** (mirror Steps 2–4 for split re/im).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/kernels/diagonal_phase.rs crates/aleph-sv/src/backend.rs crates/aleph-sv/src/soa_backend.rs
git commit -m "[P2-06] AVX-512 diagonal-phase kernels (AoS+SoA), scalar-extract sincos

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Benchmarks (fixture + builder), EPYC measurement, perf doc

**Files:**
- Modify: `benches/benches/tier1_scaling.rs` (and `benches/src/lib.rs` if a builder/decomposed pair isn't already exposed).
- Modify: `docs/perf/phase2.md`.

- [ ] **Step 1: Add before/after bench arms**

Ensure `tier1_scaling` benches both: (a) the decomposed fixture `qft_n25.qasm`
and (b) the builder `qft_circuit(n)`, each run through `run_optimized` (which now
includes `FuseDiagonalRuns`) vs the unoptimized `run`. Follow the existing
criterion group/baseline conventions in the file.

- [ ] **Step 2: Verify the bench box is idle, then measure**

```bash
ssh root@195.154.249.85
uptime                                   # load ≈ 0
pgrep -af "cargo bench|bencher run|Runner.Worker"   # empty
```
If anything competes, wait or restart per the [[feedback-check-server-clean]] rule.

- [ ] **Step 3: Run the benches on EPYC**

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p benches --bench tier1_scaling -- qft
```
Record: instruction-count before/after (≥5× pass reduction AC), and wall-clock
speedup for fixture and builder at n=25, @8 and @16 threads.

- [ ] **Step 4: Write the perf section**

Append a `## P2-06 — diagonal-run fusion` section to `docs/perf/phase2.md`:
pass-count reduction, fixture vs builder speedup, which sincos path won (scalar-
extract vs vectorized), and an honest note if the decomposed-fixture gain is
smaller than the builder's. Cite EPYC, idle-verified.

- [ ] **Step 5: Commit**

```bash
git add benches/benches/tier1_scaling.rs benches/src/lib.rs docs/perf/phase2.md
git commit -m "[P2-06] tier1_scaling diagonal-fusion benches + phase2.md report

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Final gates, self-review, PR

- [ ] **Step 1: Full CI-equivalent local run**

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --target x86_64-unknown-linux-gnu -p aleph-sv
```
All must pass. Fix clippy/fmt inline.

- [ ] **Step 2: Self-review the diff**

`git diff origin/main...HEAD` — read it with fresh eyes. Check: no `unwrap()` in
lib code, every `unsafe` has a `// SAFETY:` block, global phase preserved, the
`P != I` fallback path is covered by a test, the AC checkboxes in BACKLOG #106 are
all demonstrably met.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin p2-06-diagonal-fusion
gh pr create --title "[P2-06] Diagonal gate fusion pass" --body "$(cat <<'EOF'
Closes #106

## Approach
Monomial-run fusion: walk maximal runs of {diagonal gates ∪ cx}, track a GF(2)
permutation P and accumulate symbolic AND-of-parity phase terms; emit one
`Instruction::DiagonalPhase` when net P==I (conservative re-emit otherwise). New
scalar + AVX-512 (AoS+SoA) kernel applies it in one streaming pass. Absorbs the
interleaved `cx`s so BOTH builder and decomposed-fixture QFT ladders collapse.

## Tests
- IR: Perm tracker, diagonal_to_terms (multilinear), pass (cx·p·cx → cp, P≠I
  fallback, cost model, barrier fence), pipeline idempotence.
- Oracle: fused ≡ unfused (global phase, 1e-12) on a generic state, builder +
  decomposed QFT, raw pass and via default_pipeline; proptest over random runs.
- Kernel: scalar↔AVX-512 equivalence on EPYC.

## Benchmarks (EPYC, idle-verified)
<fill from Task 14: pass-count ≥5× reduction; fixture vs builder speedup @8/@16;
sincos path chosen>

## Notes / follow-ups
- v1 is whole-run-or-nothing on net permutation; Swap/X/Y absorption and run
  splitting deferred (spec §8). QASM emit refuses DiagonalPhase (post-opt-only).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Let the PR sit ~1h, re-review, run `/code-review high`, then merge** (project workflow). Address findings per superpowers:receiving-code-review.

---

## Self-review against the spec

- **§1.1/§1.2 (absorb cx, both QFT forms):** Tasks 3 (Perm), 4 (terms in permuted basis), 5 (fuse_run), 10 (builder + decomposed oracle). ✓
- **§2 (Instruction::DiagonalPhase, AND-of-parity terms, ≤64q assert):** Tasks 1, 2, 5 (`n > 64` guard). ✓
- **§2.1 (blast radius, emit refuses):** Tasks 2 (used_qubits/match arms), 7 (backend default), 12 (emit). ✓
- **§3 (run def, P-update, term emit, run-end P==I, cost model, determinism):** Task 5 (+ canonicalize BTreeMap for determinism). ✓
- **§4 (scalar + AVX-512 AoS+SoA, sincos measure-then-decide):** Tasks 8, 9, 13. ✓
- **§5 (pipeline before Fuse2q, idempotent):** Task 11. ✓
- **§6 (property/unit/standalone/pipeline/oracle/bench):** Tasks 4, 5, 10, 11, 13, 14. ✓
- **§7 (AC: ≥5× pass drop, 1e-12 oracle, EPYC criterion):** Task 14 (counts + speedup), Task 10 (oracle). ✓
