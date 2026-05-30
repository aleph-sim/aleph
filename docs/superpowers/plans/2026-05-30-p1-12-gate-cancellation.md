# [P1-12] Gate Cancellation Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an IR optimisation pass `CancelInversePairs` that deletes adjacent inverse-pair gates (H·H, X·X, CNOT·CNOT, S·Sdg, Rz(θ)·Rz(−θ), …), wired into the default pipeline before fusion.

**Architecture:** A single forward pass over `circuit.instructions` with a per-qubit stack of live instruction indices. Two gates cancel iff they are adjacent on their entire shared support, act on the same targets (positionally) and controls (as a set), and one is the other's `Gate::inverse()`. Non-gate instructions (Measure/Reset/Barrier) are hard barriers. Nested cancellation (`X H H X → ∅`) falls out of the stack discipline. Adjacent-only; commutation-aware cancellation is deferred to P1-13.

**Tech Stack:** Rust 2021, `aleph-ir` (`Pass`/`PassPipeline` from P1-09), `aleph-core` (`Gate::inverse`), `proptest`, `criterion`, `aleph-sv::NaiveSvBackend` for the oracle.

Design spec: `docs/superpowers/specs/2026-05-30-p1-12-gate-cancellation-design.md`.

---

## File Structure

- **Create** `crates/aleph-ir/src/passes/cancel.rs` — the pass + co-located unit/property tests.
- **Modify** `crates/aleph-ir/src/passes/mod.rs` — module decl, re-export, `default_pipeline` order, one pipeline test, doc update.
- **Modify** `crates/aleph-ir/src/bench_fixtures.rs` — add `cancel_redundant`.
- **Create** `crates/aleph-ir/benches/cancel.rs` — criterion bench + reduction-ratio print.
- **Modify** `crates/aleph-ir/Cargo.toml` — add `[[bench]] name = "cancel"`.
- **Create** `crates/aleph-sv/tests/cancel_oracle.rs` — SV amplitude equivalence (1e-12), hand-built + proptest.
- **Modify** `CLAUDE.md` — add a Quick Reference row for the cancel bench.
- **Modify** the design spec §7 — correct the note about `bench.yml` (no change needed; existing per-crate feature-gated steps cover it generically).

No `aleph-core` changes. No new dependencies.

---

## Task 1: Pass skeleton (no-op) + module wiring

**Files:**
- Create: `crates/aleph-ir/src/passes/cancel.rs`
- Modify: `crates/aleph-ir/src/passes/mod.rs:13-19`

- [ ] **Step 1: Write the failing test** (in a new `crates/aleph-ir/src/passes/cancel.rs`)

```rust
//! `CancelInversePairs` — removes adjacent inverse-pair gates
//! (H·H, X·X, CNOT·CNOT, S·Sdg, Rz(θ)·Rz(−θ), …). A single forward
//! pass with per-qubit stacks of live instruction indices; two gates
//! cancel iff they are adjacent on their entire shared support, act on
//! the same targets (positionally) and controls (as a set), and one is
//! the other's `Gate::inverse()`. Adjacent-only; commutation-aware
//! cancellation is P1-13. See
//! `docs/superpowers/specs/2026-05-30-p1-12-gate-cancellation-design.md`.

use super::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction};
use std::collections::HashMap;

pub struct CancelInversePairs;

impl Pass for CancelInversePairs {
    fn name(&self) -> &'static str {
        "CancelInversePairs"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.instructions.len();
        Ok(PassStats {
            gates_before: n,
            gates_after: n,
            transformations: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circuit;

    fn run_pass(c: &mut Circuit) -> PassStats {
        CancelInversePairs
            .run(c)
            .expect("CancelInversePairs is infallible")
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(CancelInversePairs.name(), "CancelInversePairs");
    }

    #[test]
    fn empty_circuit_is_a_no_op() {
        let mut c = Circuit::new(2, 0);
        let s = run_pass(&mut c);
        assert_eq!(s.gates_before, 0);
        assert_eq!(s.gates_after, 0);
        assert_eq!(s.transformations, 0);
    }
}
```

Note the unused `HashMap`/`Instruction` imports will warn under `-D warnings`; they are consumed by the full implementation in Task 2. To keep this task green on its own, temporarily import only what is used:

```rust
use super::{Pass, PassError, PassStats};
use crate::Circuit;
```

(Task 2 re-adds `Instruction` and `HashMap`.)

- [ ] **Step 2: Wire the module into `passes/mod.rs`**

In `crates/aleph-ir/src/passes/mod.rs`, add the module decl and re-export alongside the existing ones:

```rust
pub mod cancel;
pub mod dce;
pub mod fuse_1q;
pub mod fuse_2q;

pub use cancel::CancelInversePairs;
pub use dce::DeadCodeElim;
pub use fuse_1q::Fuse1qRuns;
pub use fuse_2q::Fuse2q;
```

(Do **not** touch `default_pipeline()` yet — that is Task 7.)

- [ ] **Step 3: Run the tests**

Run: `cargo test -p aleph-ir passes::cancel`
Expected: PASS (`name_is_stable`, `empty_circuit_is_a_no_op`).

- [ ] **Step 4: Lint**

Run: `cargo clippy -p aleph-ir --all-targets -- -D warnings`
Expected: clean (no unused-import warnings).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-ir/src/passes/cancel.rs crates/aleph-ir/src/passes/mod.rs
git commit -m "[P1-12] CancelInversePairs pass skeleton + module wiring"
```

---

## Task 2: Core algorithm — H·H cancellation

**Files:**
- Modify: `crates/aleph-ir/src/passes/cancel.rs`

- [ ] **Step 1: Write the failing test** (add to the `tests` module in `cancel.rs`)

```rust
#[test]
fn h_h_cancels_to_empty() {
    // H(0); H(0) → ∅
    let mut c = Circuit::new(1, 0);
    c.h(0).unwrap();
    c.h(0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.gates_before, 2);
    assert_eq!(s.gates_after, 0);
    assert_eq!(s.transformations, 1);
    assert!(c.instructions().is_empty());
}

#[test]
fn h_on_different_qubits_does_not_cancel() {
    // H(0); H(1) → unchanged (different support).
    let mut c = Circuit::new(2, 0);
    c.h(0).unwrap();
    c.h(1).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 2);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-ir passes::cancel::tests::h_h_cancels_to_empty`
Expected: FAIL — the no-op skeleton leaves both H gates in place (`gates_after` is 2, not 0).

- [ ] **Step 3: Replace the `run` body with the full algorithm**

Replace the imports and the `impl Pass` body in `cancel.rs` with:

```rust
use super::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction};
use std::collections::HashMap;

pub struct CancelInversePairs;

/// Order-independent equality of two external-control lists. Controls
/// are semantically a set: equal length plus equal sorted contents.
fn controls_match(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a: smallvec::SmallVec<[u32; 2]> = a.iter().copied().collect();
    let mut b: smallvec::SmallVec<[u32; 2]> = b.iter().copied().collect();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

impl Pass for CancelInversePairs {
    fn name(&self) -> &'static str {
        "CancelInversePairs"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let input: &[Instruction] = &circuit.instructions;
        let gates_before = input.len();

        let mut result: Vec<Instruction> = Vec::with_capacity(input.len());
        let mut removed: Vec<bool> = Vec::with_capacity(input.len());
        // Per qubit: stack of indices into `result` of still-live
        // instructions touching it. Top = most recent.
        let mut live: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut transformations: u64 = 0;

        for inst in input.iter() {
            // Non-gate instructions are hard barriers: push and never pop,
            // so no gate cancels across a Measure/Reset/Barrier.
            let gate = match inst {
                Instruction::Gate(g) => g,
                _ => {
                    let k = result.len();
                    result.push(inst.clone());
                    removed.push(false);
                    for q in inst.used_qubits() {
                        live.entry(q).or_default().push(k);
                    }
                    continue;
                }
            };

            let support = inst.used_qubits();

            // Candidate predecessor: the single live instruction that is the
            // current top on EVERY qubit of `support`. If the tops disagree
            // (something intervened on part of the support) or any support
            // qubit has no live predecessor, there is no candidate.
            let mut cand: Option<usize> = None;
            let mut shared = true;
            for &q in &support {
                match live.get(&q).and_then(|s| s.last().copied()) {
                    Some(top) => match cand {
                        None => cand = Some(top),
                        Some(c) if c == top => {}
                        Some(_) => {
                            shared = false;
                            break;
                        }
                    },
                    None => {
                        shared = false;
                        break;
                    }
                }
            }

            let cancels = shared
                && match cand {
                    Some(i) => match &result[i] {
                        // Same targets positionally, same controls as a set,
                        // and inverse gate. Target/control equality also
                        // guarantees `prev` touches exactly `support`, so
                        // popping it from every support stack is complete.
                        Instruction::Gate(prev) => {
                            prev.qubits == gate.qubits
                                && controls_match(&prev.controls, &gate.controls)
                                && prev.gate == gate.gate.inverse()
                        }
                        _ => false,
                    },
                    None => false,
                };

            if cancels {
                let i = cand.expect("cancels implies a candidate");
                removed[i] = true;
                for &q in &support {
                    // `i` is the top of every support stack (candidate
                    // condition), so this pops exactly `result[i]`.
                    live.get_mut(&q).expect("support qubit has a stack").pop();
                }
                transformations += 1;
                // `gate` itself is dropped (not pushed).
            } else {
                let k = result.len();
                result.push(inst.clone());
                removed.push(false);
                for &q in &support {
                    live.entry(q).or_default().push(k);
                }
            }
        }

        let kept: Vec<Instruction> = result
            .into_iter()
            .zip(removed)
            .filter_map(|(inst, dead)| if dead { None } else { Some(inst) })
            .collect();
        let gates_after = kept.len();
        circuit.instructions = kept;

        Ok(PassStats {
            gates_before,
            gates_after,
            transformations,
        })
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-ir passes::cancel`
Expected: PASS (all four tests so far).

- [ ] **Step 5: Lint**

Run: `cargo clippy -p aleph-ir --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-ir/src/passes/cancel.rs
git commit -m "[P1-12] CancelInversePairs core forward-pass algorithm"
```

---

## Task 3: Adjoint pairs and parametric cancellation

**Files:**
- Modify: `crates/aleph-ir/src/passes/cancel.rs`

- [ ] **Step 1: Write the tests** (add to the `tests` module)

```rust
#[test]
fn s_sdg_cancels() {
    // S(0); Sdg(0) → ∅  (adjoint pair, not self-inverse).
    let mut c = Circuit::new(1, 0);
    c.s(0).unwrap();
    c.sdg(0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 1);
    assert!(c.instructions().is_empty());
}

#[test]
fn t_tdg_cancels_both_orders() {
    for build in [
        |c: &mut Circuit| {
            c.t(0).unwrap();
            c.tdg(0).unwrap();
        },
        |c: &mut Circuit| {
            c.tdg(0).unwrap();
            c.t(0).unwrap();
        },
    ] {
        let mut c = Circuit::new(1, 0);
        build(&mut c);
        let s = run_pass(&mut c);
        assert_eq!(s.transformations, 1);
        assert!(c.instructions().is_empty());
    }
}

#[test]
fn rz_theta_rz_neg_theta_cancels() {
    // Rz(0.3); Rz(-0.3) → ∅  (exact f64 negation).
    let mut c = Circuit::new(1, 0);
    c.rz(0.3, 0).unwrap();
    c.rz(-0.3, 0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 1);
    assert!(c.instructions().is_empty());
}

#[test]
fn rz_same_sign_does_not_cancel() {
    // Rz(0.3); Rz(0.3) is Rz(0.6), NOT identity → kept.
    let mut c = Circuit::new(1, 0);
    c.rz(0.3, 0).unwrap();
    c.rz(0.3, 0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 2);
}

#[test]
fn rz_near_but_unequal_angle_does_not_cancel() {
    // Cancellation requires exact -θ; a 1e-12 mismatch must NOT cancel
    // (that is fusion/tolerance territory, not exact-inverse deletion).
    let mut c = Circuit::new(1, 0);
    c.rz(0.3, 0).unwrap();
    c.rz(-0.3 + 1e-12, 0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 2);
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p aleph-ir passes::cancel`
Expected: PASS (the algorithm already covers these via `Gate::inverse` + exact `PartialEq`).

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-ir/src/passes/cancel.rs
git commit -m "[P1-12] Tests: adjoint-pair + parametric cancellation"
```

---

## Task 4: Multi-qubit and externally-controlled cancellation

**Files:**
- Modify: `crates/aleph-ir/src/passes/cancel.rs`

- [ ] **Step 1: Write the tests** (add to the `tests` module; add imports at top of the test module)

At the top of `mod tests`, extend the imports:

```rust
use super::*;
use crate::Circuit;
use aleph_core::{Gate, GateInstance};
use smallvec::smallvec;
```

Then add:

```rust
#[test]
fn cnot_cnot_same_qubits_cancels() {
    // CNOT(0,1); CNOT(0,1) → ∅
    let mut c = Circuit::new(2, 0);
    c.cnot(0, 1).unwrap();
    c.cnot(0, 1).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 1);
    assert!(c.instructions().is_empty());
}

#[test]
fn cnot_reversed_roles_does_not_cancel() {
    // CNOT(0,1); CNOT(1,0) are different operations → kept.
    let mut c = Circuit::new(2, 0);
    c.cnot(0, 1).unwrap();
    c.cnot(1, 0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 2);
}

#[test]
fn cz_swap_toffoli_cancel() {
    // CZ(0,1)·CZ(0,1), SWAP(0,1)·SWAP(0,1), Toffoli·Toffoli all self-inverse.
    let mut c = Circuit::new(3, 0);
    c.cz(0, 1).unwrap();
    c.cz(0, 1).unwrap();
    c.swap(0, 1).unwrap();
    c.swap(0, 1).unwrap();
    c.ccx(0, 1, 2).unwrap();
    c.ccx(0, 1, 2).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 3);
    assert!(c.instructions().is_empty());
}

#[test]
fn iswap_iswapdg_cancels() {
    // Iswap(0,1); IswapDg(0,1) → ∅  (adjoint pair via add_gate).
    let mut c = Circuit::new(2, 0);
    c.add_gate(GateInstance::new(Gate::Iswap, smallvec![0u32, 1u32]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::IswapDg, smallvec![0u32, 1u32]))
        .unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 1);
    assert!(c.instructions().is_empty());
}

#[test]
fn controlled_x_same_controls_cancels_control_order_independent() {
    // ctrl-X target=3 controls={0,1}; then controls={1,0} → cancels
    // (controls compared as a set).
    let mut c = Circuit::new(4, 0);
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![3u32],
        smallvec![0u32, 1u32],
    ))
    .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![3u32],
        smallvec![1u32, 0u32],
    ))
    .unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 1);
    assert!(c.instructions().is_empty());
}

#[test]
fn controlled_x_different_controls_does_not_cancel() {
    // Same target, different control set → not the same operation → kept.
    let mut c = Circuit::new(4, 0);
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![3u32],
        smallvec![0u32],
    ))
    .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![3u32],
        smallvec![1u32],
    ))
    .unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 2);
}

#[test]
fn symmetric_gate_reordered_qubits_not_cancelled_v1() {
    // Cz(0,1) and Cz(1,0) ARE the same operation, but v1 compares targets
    // positionally and conservatively does NOT cancel them. Documents the
    // deferral (symmetric-gate normalisation is future work).
    let mut c = Circuit::new(2, 0);
    c.cz(0, 1).unwrap();
    c.cz(1, 0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 2);
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p aleph-ir passes::cancel`
Expected: PASS.

- [ ] **Step 3: Lint**

Run: `cargo clippy -p aleph-ir --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-ir/src/passes/cancel.rs
git commit -m "[P1-12] Tests: multi-qubit + controlled cancellation, symmetric-gate deferral"
```

---

## Task 5: Boundaries (barrier/measure/reset) and nested cancellation

**Files:**
- Modify: `crates/aleph-ir/src/passes/cancel.rs`

- [ ] **Step 1: Write the tests** (add to the `tests` module)

```rust
#[test]
fn barrier_blocks_cancellation() {
    // H(0); Barrier([0]); H(0) → kept (barrier severs adjacency).
    let mut c = Circuit::new(1, 0);
    c.h(0).unwrap();
    c.barrier([0u32]).unwrap();
    c.h(0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 3);
}

#[test]
fn measure_blocks_cancellation() {
    // X(0); Measure(0,0); X(0) → kept.
    let mut c = Circuit::new(1, 1);
    c.x(0).unwrap();
    c.measure(0, 0).unwrap();
    c.x(0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 3);
}

#[test]
fn reset_blocks_cancellation() {
    // X(0); Reset(0); X(0) → kept.
    let mut c = Circuit::new(1, 0);
    c.x(0).unwrap();
    c.reset(0).unwrap();
    c.x(0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 3);
}

#[test]
fn intervening_gate_on_partial_support_blocks_cancellation() {
    // CNOT(0,1); X(1); CNOT(0,1) → NOT cancelled: X(1) intervenes on
    // qubit 1, so the two CNOTs are not adjacent on their full support.
    let mut c = Circuit::new(2, 0);
    c.cnot(0, 1).unwrap();
    c.x(1).unwrap();
    c.cnot(0, 1).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 0);
    assert_eq!(c.instructions().len(), 3);
}

#[test]
fn nested_single_qubit_cancellation() {
    // X H H X (all on qubit 0): inner H·H cancels, then the X·X become
    // adjacent and cancel → ∅.
    let mut c = Circuit::new(1, 0);
    c.x(0).unwrap();
    c.h(0).unwrap();
    c.h(0).unwrap();
    c.x(0).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 2);
    assert!(c.instructions().is_empty());
}

#[test]
fn nested_through_two_qubit_gate() {
    // CNOT(0,1); X(0); X(0); CNOT(0,1) → inner X·X cancels, then the two
    // CNOTs share the top on both qubits and cancel → ∅.
    let mut c = Circuit::new(2, 0);
    c.cnot(0, 1).unwrap();
    c.x(0).unwrap();
    c.x(0).unwrap();
    c.cnot(0, 1).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 2);
    assert!(c.instructions().is_empty());
}

#[test]
fn partial_cancellation_keeps_survivors_in_order() {
    // H(0); X(0); X(0); Z(1) → X·X cancels; H(0) and Z(1) survive in order.
    let mut c = Circuit::new(2, 0);
    c.h(0).unwrap();
    c.x(0).unwrap();
    c.x(0).unwrap();
    c.z(1).unwrap();
    let s = run_pass(&mut c);
    assert_eq!(s.transformations, 1);
    assert_eq!(c.instructions().len(), 2);
    assert!(matches!(
        &c.instructions()[0],
        Instruction::Gate(g) if g.gate == aleph_core::Gate::H
    ));
    assert!(matches!(
        &c.instructions()[1],
        Instruction::Gate(g) if g.gate == aleph_core::Gate::Z
    ));
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p aleph-ir passes::cancel`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-ir/src/passes/cancel.rs
git commit -m "[P1-12] Tests: boundary blocking + nested cancellation"
```

---

## Task 6: Property tests — idempotence and non-growth

**Files:**
- Modify: `crates/aleph-ir/src/passes/cancel.rs`

- [ ] **Step 1: Write the property tests** (add a `proptests` module at the end of `cancel.rs`, outside `mod tests`)

```rust
#[cfg(test)]
mod proptests {
    use super::*;
    use aleph_test::arb_circuit_emittable;
    use proptest::prelude::*;

    proptest! {
        // Running the pass a second time changes nothing: after one pass
        // there are no adjacent inverse pairs left, so the fixed point is
        // reached in one sweep.
        #[test]
        fn idempotent(c in arb_circuit_emittable(4, 2, 24)) {
            let mut once = c.clone();
            CancelInversePairs.run(&mut once).unwrap();
            let len_once = once.instructions().len();

            let mut twice = once.clone();
            let s = CancelInversePairs.run(&mut twice).unwrap();
            prop_assert_eq!(s.transformations, 0);
            prop_assert_eq!(twice.instructions().len(), len_once);
        }

        // The pass never adds instructions and accounts for exactly two
        // removed gates per cancellation event.
        #[test]
        fn never_grows_and_counts_are_consistent(c in arb_circuit_emittable(4, 2, 24)) {
            let mut cc = c.clone();
            let s = CancelInversePairs.run(&mut cc).unwrap();
            prop_assert!(s.gates_after <= s.gates_before);
            prop_assert_eq!(s.gates_before - s.gates_after, (s.transformations as usize) * 2);
        }
    }
}
```

- [ ] **Step 2: Confirm the fixture signature**

Run: `grep -n "pub fn arb_circuit_emittable" crates/aleph-test/src/circuit.rs`
Expected: `pub fn arb_circuit_emittable(nq: u32, nc: u32, n_ops: usize) -> impl Strategy<Value = Circuit>` — matches the call `arb_circuit_emittable(4, 2, 24)`. If the signature differs, adjust the call arguments to match (qubit count, clbit count, op count).

- [ ] **Step 3: Run the property tests**

Run: `cargo test -p aleph-ir passes::cancel::proptests`
Expected: PASS (256 generated cases each).

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-ir/src/passes/cancel.rs
git commit -m "[P1-12] Property tests: idempotence + non-growth invariants"
```

---

## Task 7: Wire into `default_pipeline` (DCE → Cancel → Fuse1q → Fuse2q)

**Files:**
- Modify: `crates/aleph-ir/src/passes/mod.rs:64-72` (`default_pipeline`), `:1-8` (doc), tests

- [ ] **Step 1: Write the failing test** (add to `mod tests` in `passes/mod.rs`)

```rust
#[test]
fn default_pipeline_cancels_inverse_pair_before_fusion() {
    // H(0); H(0); H(1) — the H·H pair must be removed by cancellation;
    // fusion alone would instead fuse it into an identity Unitary1q that
    // still executes. After the pipeline: only H(1) remains.
    let mut c = Circuit::new(2, 0);
    c.h(0).unwrap();
    c.h(0).unwrap();
    c.h(1).unwrap();
    let stats = PassPipeline::default_pipeline().run(&mut c).unwrap();
    assert_eq!(stats.gates_before, 3);
    assert_eq!(stats.gates_after, 1);
    assert!(stats.transformations >= 1);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-ir passes::tests::default_pipeline_cancels_inverse_pair_before_fusion`
Expected: FAIL — without Cancel in the pipeline, `Fuse1qRuns` fuses H·H into a single `Unitary1qDiag`/`Unitary1q`, so `gates_after` is 2 (fused-H0 + H1), not 1.

- [ ] **Step 3: Add `CancelInversePairs` to `default_pipeline`**

In `crates/aleph-ir/src/passes/mod.rs`, change `default_pipeline`:

```rust
    /// Phase-1 default pipeline. Currently
    /// `[DeadCodeElim, CancelInversePairs, Fuse1qRuns, Fuse2q]`; later
    /// passes are appended here as they ship. Cancellation runs before
    /// fusion so exact inverse pairs (e.g. `Rz(θ)·Rz(−θ)`) are deleted
    /// rather than fused into an identity block that still executes.
    pub fn default_pipeline() -> Self {
        Self::new(vec![
            Box::new(DeadCodeElim),
            Box::new(CancelInversePairs),
            Box::new(Fuse1qRuns),
            Box::new(Fuse2q),
        ])
    }
```

- [ ] **Step 4: Update the module doc** at the top of `passes/mod.rs`

```rust
//! `aleph-ir::passes` — IR-level optimisation passes.
//!
//! Each pass implements [`Pass`]. A [`PassPipeline`] runs an ordered
//! sequence of passes over a [`Circuit`], aggregating per-pass
//! [`PassStats`]. Phase-1 ships [`dce::DeadCodeElim`],
//! [`cancel::CancelInversePairs`], [`fuse_1q::Fuse1qRuns`], and
//! [`fuse_2q::Fuse2q`]; later tickets (P1-13) add more passes that plug
//! in by being pushed onto the pipeline.
```

- [ ] **Step 5: Run the affected tests**

Run: `cargo test -p aleph-ir passes`
Expected: PASS — the new test plus the existing `default_pipeline_*` tests still pass (DCE-first behaviour unchanged; the `default_pipeline_includes_fuse_2q` case `Rx(0); CNOT(0,1)` has no inverse pair, so Cancel is a no-op there).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-ir/src/passes/mod.rs
git commit -m "[P1-12] Wire CancelInversePairs into default_pipeline before fusion"
```

---

## Task 8: SV oracle — amplitude equivalence vs original

**Files:**
- Create: `crates/aleph-sv/tests/cancel_oracle.rs`

- [ ] **Step 1: Confirm the backend API** used by the sibling oracle

Run: `sed -n '10,31p' crates/aleph-sv/tests/dce_oracle.rs`
Expected: shows `use aleph_backend::run;`, `aleph_sv::NaiveSvBackend::with_seed(0)`, and `run(&mut backend, c).unwrap().amplitudes().to_vec()`. The new oracle reuses this exact shape.

- [ ] **Step 2: Write the oracle test file**

Create `crates/aleph-sv/tests/cancel_oracle.rs`:

```rust
//! P1-12 oracle — a circuit with `CancelInversePairs` applied yields the
//! same full state vector as the original. Cancellation only ever deletes
//! a gate together with its exact inverse (an identity), so the entire
//! amplitude vector must match to 1e-12, not merely the measurement
//! marginal. Hand-built cases plus a proptest over random circuits guard
//! against false-positive removals.

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance};
use aleph_ir::passes::{CancelInversePairs, Pass};
use aleph_ir::{Circuit, Instruction};
use smallvec::smallvec;

const TOL: f64 = 1e-12;

/// Gate-only twin (drop Measure/Reset/Barrier) so `run` accepts it. The
/// cancellation cases here are unitary; this strips any terminal markers.
fn gate_only(c: &Circuit) -> Circuit {
    let mut out = Circuit::new(c.num_qubits(), c.num_clbits());
    for inst in c.instructions() {
        if let Instruction::Gate(g) = inst {
            out.add_gate(g.clone()).unwrap();
        }
    }
    out
}

fn amplitudes(c: &Circuit) -> Vec<Complex> {
    let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
    run(&mut backend, c).unwrap().amplitudes().to_vec()
}

fn assert_state_preserved(c: &Circuit) {
    let mut cancelled = c.clone();
    CancelInversePairs.run(&mut cancelled).unwrap();

    let before = amplitudes(&gate_only(c));
    let after = amplitudes(&gate_only(&cancelled));

    assert_eq!(before.len(), after.len(), "state dimension changed");
    for (k, (a, b)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            (a.re - b.re).abs() < TOL && (a.im - b.im).abs() < TOL,
            "amplitude[{k}] differs: before={a:?} after={b:?}"
        );
    }
}

#[test]
fn nested_cancellation_preserves_state() {
    // X H H X with a surrounding useful gate that survives.
    let mut c = Circuit::new(2, 0);
    c.ry(0.7, 0).unwrap();
    c.x(0).unwrap();
    c.h(0).unwrap();
    c.h(0).unwrap();
    c.x(0).unwrap();
    c.cnot(0, 1).unwrap();
    assert_state_preserved(&c);
}

#[test]
fn parametric_and_adjoint_pairs_preserve_state() {
    let mut c = Circuit::new(2, 0);
    c.h(0).unwrap();
    c.rz(0.41, 0).unwrap();
    c.rz(-0.41, 0).unwrap(); // cancels
    c.s(1).unwrap();
    c.sdg(1).unwrap(); // cancels
    c.cnot(0, 1).unwrap();
    assert_state_preserved(&c);
}

#[test]
fn two_qubit_and_controlled_pairs_preserve_state() {
    let mut c = Circuit::new(3, 0);
    c.h(0).unwrap();
    c.h(1).unwrap();
    c.cz(0, 1).unwrap();
    c.cz(0, 1).unwrap(); // cancels
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![2u32],
        smallvec![0u32, 1u32],
    ))
    .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::X,
        smallvec![2u32],
        smallvec![1u32, 0u32],
    ))
    .unwrap(); // cancels (controls as a set)
    assert_state_preserved(&c);
}
```

- [ ] **Step 3: Add the proptest guard** (append to `cancel_oracle.rs`)

```rust
use aleph_test::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    // For ANY random unitary circuit, cancelling inverse pairs must not
    // change the state vector. This is the real false-positive guard:
    // if the pass ever deletes a non-inverse pair, amplitudes diverge.
    #[test]
    fn random_circuit_state_preserved(c in arb_circuit_emittable(4, 0, 20)) {
        let mut cancelled = c.clone();
        CancelInversePairs.run(&mut cancelled).unwrap();

        let before = amplitudes(&gate_only(&c));
        let after = amplitudes(&gate_only(&cancelled));
        prop_assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            prop_assert!((a.re - b.re).abs() < TOL && (a.im - b.im).abs() < TOL);
        }
    }
}
```

Note: `arb_circuit_emittable(4, 0, 20)` requests 0 clbits so no `Measure` is generated; combined with `gate_only` this keeps every case a pure unitary whose full state is well-defined. If the strategy can still emit `Reset` (which `gate_only` drops, changing semantics), restrict by filtering: confirm with `sed -n '150,229p' crates/aleph-test/src/circuit.rs` whether `arb_op_emittable` includes `Reset`; if it does, add `.prop_filter("no reset", |c| c.instructions().iter().all(|i| !matches!(i, Instruction::Reset(_))))` to the strategy.

- [ ] **Step 4: Run the oracle**

Run: `cargo test -p aleph-sv --test cancel_oracle`
Expected: PASS (3 hand-built + 256 generated cases).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/tests/cancel_oracle.rs
git commit -m "[P1-12] SV oracle: cancellation preserves full state vector"
```

---

## Task 9: Benchmark fixture + criterion bench

**Files:**
- Modify: `crates/aleph-ir/src/bench_fixtures.rs` (append)
- Create: `crates/aleph-ir/benches/cancel.rs`
- Modify: `crates/aleph-ir/Cargo.toml` (add `[[bench]]`)

- [ ] **Step 1: Add the fixture** to `crates/aleph-ir/src/bench_fixtures.rs` (append at end of file)

```rust
/// Circuit dominated by cancellable redundancy. Per step: one surviving
/// `Rz` rotation (distinct angle, so consecutive survivors never cancel)
/// followed by an `H·H` and an `X·X` self-inverse pair that the
/// cancellation pass deletes. `pairs` steps → `5·pairs` gates in, `pairs`
/// gates out (5× reduction). Deterministic; `q` cycles across the
/// register so the redundancy is spread over all qubits.
pub fn cancel_redundant(n_qubits: u32, pairs: u32) -> Circuit {
    assert!(n_qubits >= 1, "cancel_redundant needs at least one qubit");
    let mut c = Circuit::new(n_qubits, 0);
    for p in 0..pairs {
        let q = p % n_qubits;
        c.rz(0.1 + 0.001 * (p as f64), q).unwrap(); // survives
        c.h(q).unwrap();
        c.h(q).unwrap(); // cancels
        c.x(q).unwrap();
        c.x(q).unwrap(); // cancels
    }
    c
}
```

- [ ] **Step 2: Add a fixture sanity test** to `bench_fixtures.rs` (in its `#[cfg(test)] mod tests`, or add one if absent)

First check whether a test module exists:

Run: `grep -n "mod tests" crates/aleph-ir/src/bench_fixtures.rs`

If it exists, add this test inside it; otherwise append a new module at the end of the file:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::{CancelInversePairs, Pass};

    #[test]
    fn cancel_redundant_reduces_5x() {
        let mut c = cancel_redundant(4, 20);
        assert_eq!(c.len(), 100); // 5 gates × 20 steps
        let s = CancelInversePairs.run(&mut c).unwrap();
        assert_eq!(s.gates_after, 20); // only the Rz survivors
        assert_eq!(c.len(), 20);
    }
}
```

(If a `mod tests` already exists with `use super::*;`, do not duplicate the import — add only the function and the `passes` import it needs.)

- [ ] **Step 3: Run the fixture test** (fixtures are `cfg(test)`-visible without the feature)

Run: `cargo test -p aleph-ir bench_fixtures`
Expected: PASS (`cancel_redundant_reduces_5x`).

- [ ] **Step 4: Create the bench** `crates/aleph-ir/benches/cancel.rs`

```rust
//! P1-12 benchmark: cost of `CancelInversePairs` on a redundancy-heavy
//! circuit, plus a printed gate-count reduction (the acceptance-criterion
//! figure).
//!
//! Run with:
//! `cargo bench -p aleph-ir --features bench-fixtures --bench cancel`

use aleph_ir::bench_fixtures::cancel_redundant;
use aleph_ir::passes::{CancelInversePairs, PassPipeline};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_cancel(c: &mut Criterion) {
    // Report the AC figure once, outside the timing loop.
    let probe = cancel_redundant(8, 200);
    let before = probe.len();
    let mut reduced = probe.clone();
    PassPipeline::new(vec![Box::new(CancelInversePairs)])
        .run(&mut reduced)
        .unwrap();
    eprintln!(
        "cancel_redundant(8,200): {} → {} ({:.2}× reduction)",
        before,
        reduced.len(),
        before as f64 / reduced.len() as f64
    );

    let mut group = c.benchmark_group("cancel");
    group.bench_function("cancel_redundant_n8_pairs200", |bch| {
        bch.iter_batched(
            || cancel_redundant(8, 200),
            |mut circ| {
                PassPipeline::new(vec![Box::new(CancelInversePairs)])
                    .run(&mut circ)
                    .unwrap();
                circ
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_cancel);
criterion_main!(benches);
```

- [ ] **Step 5: Register the bench** in `crates/aleph-ir/Cargo.toml` (after the `fuse_2q` `[[bench]]`)

```toml
[[bench]]
name = "cancel"
harness = false
required-features = ["bench-fixtures"]
```

- [ ] **Step 6: Compile and run the bench**

Run: `cargo bench -p aleph-ir --features bench-fixtures --bench cancel`
Expected: compiles and runs; prints `cancel_redundant(8,200): 1000 → 200 (5.00× reduction)` and a criterion timing for `cancel/cancel_redundant_n8_pairs200`.

- [ ] **Step 7: Lint the whole crate with benches**

Run: `cargo clippy -p aleph-ir --all-targets --features bench-fixtures -- -D warnings`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add crates/aleph-ir/src/bench_fixtures.rs crates/aleph-ir/benches/cancel.rs crates/aleph-ir/Cargo.toml
git commit -m "[P1-12] Bench: cancel_redundant fixture + criterion cancel bench (5x)"
```

---

## Task 10: Docs — CLAUDE.md quick-ref + spec correction

**Files:**
- Modify: `CLAUDE.md` (Quick Reference table)
- Modify: `docs/superpowers/specs/2026-05-30-p1-12-gate-cancellation-design.md` (§7 note)

- [ ] **Step 1: Add a Quick Reference row** to `CLAUDE.md`

Find the row `|Benchmark (IR fuse 2q)|...` and add immediately after it:

```
|Benchmark (IR cancel)|`cargo bench -p aleph-ir --features bench-fixtures --bench cancel`|
```

- [ ] **Step 2: Correct the spec §7 note** about `bench.yml`

In the design spec, replace the sentence beginning "Because `cargo bench --workspace` silently skips…" with:

```
- The crate-level `bench-fixtures` steps already in
  `.github/workflows/bench.yml` (`cargo bench -p aleph-ir --features
  bench-fixtures [--no-run]`) compile and run every feature-gated bench
  in the crate generically, so the new `cancel` bench needs no workflow
  change — only the `[[bench]]` registration in `Cargo.toml`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-05-30-p1-12-gate-cancellation-design.md
git commit -m "[P1-12] Docs: CLAUDE.md cancel-bench row + spec bench.yml note"
```

---

## Task 11: Full workspace verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `cargo test --workspace`
Expected: PASS (all crates, including the new `cancel` unit/property/oracle tests).

- [ ] **Step 2: Lint + format**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
Expected: clean.

- [ ] **Step 3: Confirm feature-gated benches still compile**

Run: `cargo bench -p aleph-ir --features bench-fixtures --no-run`
Expected: compiles `fuse_1q`, `fuse_2q`, and `cancel` benches.

- [ ] **Step 4: Final review of the diff**

Run: `git log --oneline main..HEAD`
Expected: a clean sequence of `[P1-12]` commits, one per task.

---

## Notes / follow-ups

- **EPYC validation:** the bench AC ratio (5×) is gate-count, machine-independent, so it can be confirmed locally. No SIMD/codegen concern here (pure IR rewrite), so the per-task EPYC discipline used for kernel tickets is unnecessary — local verification suffices.
- **Deferred to P1-13:** commutation-aware cancellation (reorder to bring pairs together) and symmetric-gate qubit-order normalisation (`Cz(0,1) ≡ Cz(1,0)`).
- **`/code-review` high-effort** after the branch is green, before opening the PR — consistent with prior Stage-2 tickets; pay special attention to the live-stack bookkeeping on cancellation (the pop-from-every-support-stack step is the analogue of the output-reorder bugs caught in Fuse2q).
