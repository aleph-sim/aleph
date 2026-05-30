# [P1-13] Commutation Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aleph_ir::passes::gates_commute(a, b) -> bool`, a sound, conservative commutation predicate over `GateInstance` pairs, for use by later optimisation passes.

**Architecture:** Pure function in a new `passes/commute.rs` module. A first-match-wins rule table (disjoint support → both-diagonal → structurally identical → CNOT control/target relations → else `false`). Sound by design: `true` means the operators provably commute; unsure → `false`. No `aleph-core` change, no `Pass`, no `default_pipeline` change. Correctness guarded by an `aleph-sv` state-vector oracle.

**Tech Stack:** Rust 2021, `aleph-ir` (consumes `aleph-core` `Gate`/`GateInstance`/`Param`), `proptest`, `aleph-sv::NaiveSvBackend` for the oracle.

Design spec: `docs/superpowers/specs/2026-05-30-p1-13-commutation-design.md`.

---

## File Structure

- **Create** `crates/aleph-ir/src/passes/commute.rs` — `gates_commute` + private helpers + co-located unit tests.
- **Modify** `crates/aleph-ir/src/passes/mod.rs` — `pub mod commute;`, `pub use commute::gates_commute;`, one line in the module doc.
- **Create** `crates/aleph-sv/tests/commute_oracle.rs` — state-vector reorder-equivalence oracle (deterministic true/false pairs + a `commute ⟹ equal` proptest).

No `aleph-core` change, no benchmark (this is a predicate, not an optimisation pass — there is no before/after wall-clock to report).

Key API facts (verified against the codebase):
- `aleph_core::{Gate, GateInstance, Param}` are all re-exported at the crate root. `Gate` derives `PartialEq`; `Gate::Rz(Param::Concrete(0.5))` constructs a parametric gate; `Gate::is_diagonal()` and `Gate::arity()` exist.
- `GateInstance { gate: Gate, qubits: SmallVec<[u32;4]>, controls: SmallVec<[u32;2]> }`; `GateInstance::new(gate, qubits)` and `GateInstance::controlled(gate, qubits, controls)`. `qubits`/`controls` deref to `&[u32]` and support `.contains(&q)`.
- `Circuit::new(n, c)`, `Circuit::add_gate(GateInstance)`, `circuit.instructions()`. `Instruction::Gate(GateInstance)`.
- Oracle pattern (mirror `crates/aleph-sv/tests/cancel_oracle.rs`): `use aleph_backend::run;`, `aleph_sv::NaiveSvBackend::with_seed(0)`, `run(&mut backend, &c).unwrap().amplitudes().to_vec()`.
- Property strategy: `aleph_test::circuit::arb_circuit_emittable(nq, nc, n_ops)`.

---

## Task 1: Module skeleton + wiring + disjoint-support rule

**Files:**
- Create: `crates/aleph-ir/src/passes/commute.rs`
- Modify: `crates/aleph-ir/src/passes/mod.rs`

- [ ] **Step 1: Create `commute.rs` with the failing test first**

```rust
//! `gates_commute` — a sound, conservative predicate answering whether
//! two gate instances commute (`A·B == B·A` as operators), so a later
//! pass may safely reorder them. First-match-wins rule table; when
//! unsure it returns `false` (a false negative only forgoes an
//! optimisation, a false positive would corrupt state). See
//! `docs/superpowers/specs/2026-05-30-p1-13-commutation-design.md`.

use aleph_core::{Gate, GateInstance};

/// True iff `a` and `b` provably commute as operators. Conservative:
/// returns `false` whenever commutation is not established by a rule.
/// Symmetric in its arguments.
pub fn gates_commute(a: &GateInstance, b: &GateInstance) -> bool {
    // Rule 1: disjoint support — operators on different qubits commute.
    if !supports_overlap(a, b) {
        return true;
    }
    false
}

/// Whether `a` and `b` touch any common qubit (targets ∪ controls).
fn supports_overlap(a: &GateInstance, b: &GateInstance) -> bool {
    let in_a = |q: &u32| a.qubits.contains(q) || a.controls.contains(q);
    b.qubits.iter().any(in_a) || b.controls.iter().any(in_a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
        GateInstance::new(gate, qubits.to_vec())
    }

    #[test]
    fn disjoint_support_commutes() {
        // H(0) and X(1) act on different qubits.
        assert!(gates_commute(&g(Gate::H, &[0]), &g(Gate::X, &[1])));
        // CNOT(0,1) and Z(2): q2 disjoint from {0,1}.
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[2])));
    }

    #[test]
    fn overlapping_non_commuting_is_false_for_now() {
        // X(0) and Z(0) overlap and (until rule table is built) must not
        // be falsely reported as commuting.
        assert!(!gates_commute(&g(Gate::X, &[0]), &g(Gate::Z, &[0])));
    }
}
```

Note: the top-level `use aleph_core::{Gate, GateInstance};` is used by the function signature/body; the test module re-imports what it needs. Keep imports tight for `-D warnings` (e.g. if `Gate` is unused at top level until Task 2, import only `GateInstance` — but Task 2 adds `Gate` usage, so importing both now is fine only if both are used; in this skeleton only `GateInstance` is used in the signature, so import `use aleph_core::GateInstance;` at top level and let the test module import `Gate` itself). Adjust to whatever compiles cleanly.

- [ ] **Step 2: Wire the module into `passes/mod.rs`**

Add alongside the existing module decls/re-exports:

```rust
pub mod cancel;
pub mod commute;
pub mod dce;
pub mod fuse_1q;
pub mod fuse_2q;

pub use cancel::CancelInversePairs;
pub use commute::gates_commute;
pub use dce::DeadCodeElim;
pub use fuse_1q::Fuse1qRuns;
pub use fuse_2q::Fuse2q;
```

(Do NOT touch `default_pipeline()` — `gates_commute` is not a pass.)

- [ ] **Step 3: Run the tests**

Run: `cargo test -p aleph-ir passes::commute`
Expected: PASS (`disjoint_support_commutes`, `overlapping_non_commuting_is_false_for_now`).

- [ ] **Step 4: Lint**

Run: `cargo clippy -p aleph-ir --all-targets -- -D warnings`
Expected: clean (no unused imports).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-ir/src/passes/commute.rs crates/aleph-ir/src/passes/mod.rs
git commit -m "[P1-13] gates_commute skeleton + disjoint-support rule"
```

---

## Task 2: Diagonal, identical, and CNOT rules

**Files:**
- Modify: `crates/aleph-ir/src/passes/commute.rs`

- [ ] **Step 1: Write the failing tests** (add to the `tests` module)

```rust
    fn rz(theta: f64, q: u32) -> GateInstance {
        GateInstance::new(Gate::Rz(aleph_core::Param::Concrete(theta)), [q].to_vec())
    }
    fn rx(theta: f64, q: u32) -> GateInstance {
        GateInstance::new(Gate::Rx(aleph_core::Param::Concrete(theta)), [q].to_vec())
    }

    #[test]
    fn both_diagonal_commute() {
        // Z·Rz, S·T, Phase·Cz, Rz·CRz, Cz·Ccz — all diagonal, all commute.
        assert!(gates_commute(&g(Gate::Z, &[0]), &rz(0.4, 0)));
        assert!(gates_commute(&g(Gate::S, &[0]), &g(Gate::T, &[0])));
        assert!(gates_commute(
            &GateInstance::new(Gate::Phase(aleph_core::Param::Concrete(0.3)), [0u32].to_vec()),
            &g(Gate::Cz, &[0, 1])
        ));
        assert!(gates_commute(
            &rz(0.2, 1),
            &GateInstance::new(Gate::CRz(aleph_core::Param::Concrete(0.5)), [0u32, 1u32].to_vec())
        ));
        assert!(gates_commute(&g(Gate::Cz, &[0, 1]), &g(Gate::Ccz, &[0, 1, 2])));
    }

    #[test]
    fn controlled_diagonal_is_diagonal_commutes() {
        // Externally-controlled Z is still diagonal; commutes with Rz.
        let cz_ext = GateInstance::controlled(Gate::Z, smallvec![1u32], smallvec![0u32]);
        assert!(gates_commute(&cz_ext, &rz(0.7, 1)));
    }

    #[test]
    fn structurally_identical_commute() {
        assert!(gates_commute(&g(Gate::H, &[0]), &g(Gate::H, &[0])));
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Cnot, &[0, 1])));
        // Identically-controlled X (controls compared as a set).
        let a = GateInstance::controlled(Gate::X, smallvec![3u32], smallvec![0u32, 1u32]);
        let b = GateInstance::controlled(Gate::X, smallvec![3u32], smallvec![1u32, 0u32]);
        assert!(gates_commute(&a, &b));
    }

    #[test]
    fn cnot_commutes_with_x_or_rx_on_target() {
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[1])));
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &rx(0.9, 1)));
    }

    #[test]
    fn cnot_commutes_with_diagonal_on_control() {
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[0])));
        assert!(gates_commute(&g(Gate::Cnot, &[0, 1]), &rz(0.5, 0)));
    }

    #[test]
    fn cnot_does_not_commute_with_z_target_or_x_control() {
        assert!(!gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[1]))); // Z on target
        assert!(!gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[0]))); // X on control
        assert!(!gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Y, &[1]))); // Y on target (deferred)
    }

    #[test]
    fn externally_controlled_cnot_skips_rule_4() {
        // A CNOT carrying an external control is not a bare CNOT; rule 4
        // must not fire. ctrl-CNOT(c=2; q=[0,1]) vs X(1): no rule applies → false.
        let cc = GateInstance::controlled(Gate::Cnot, smallvec![0u32, 1u32], smallvec![2u32]);
        assert!(!gates_commute(&cc, &g(Gate::X, &[1])));
    }
```

- [ ] **Step 2: Run to verify failures**

Run: `cargo test -p aleph-ir passes::commute`
Expected: the new tests FAIL (skeleton only has rule 1; e.g. `both_diagonal_commute` fails because `Z(0)`/`Rz(0)` overlap and return `false`).

- [ ] **Step 3: Implement rules 2–4**

Replace the body of `gates_commute` and add the helpers in `commute.rs`:

```rust
pub fn gates_commute(a: &GateInstance, b: &GateInstance) -> bool {
    // Rule 1: disjoint support — operators on different qubits commute.
    if !supports_overlap(a, b) {
        return true;
    }
    // Rule 2: both diagonal — diagonal matrices (incl. controlled-diagonal,
    // which is still diagonal) always commute, on any qubits.
    if a.gate.is_diagonal() && b.gate.is_diagonal() {
        return true;
    }
    // Rule 3: structurally identical — an operator commutes with itself.
    if instances_identical(a, b) {
        return true;
    }
    // Rule 4: CNOT control/target relations (symmetric over arg order).
    if cnot_commutes_with_1q(a, b) || cnot_commutes_with_1q(b, a) {
        return true;
    }
    false
}

/// Order-independent equality of two external-control lists.
fn controls_eq_set(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_unstable();
    y.sort_unstable();
    x == y
}

/// Same gate, same target qubits (positional), same controls (as a set).
fn instances_identical(a: &GateInstance, b: &GateInstance) -> bool {
    a.gate == b.gate && a.qubits == b.qubits && controls_eq_set(&a.controls, &b.controls)
}

/// True iff `cnot` is a bare `Cnot(c, t)` and `other` is a bare single-
/// qubit gate that commutes with it by the control/target relations:
/// a gate that commutes with X (`X`, `Rx`) on the target `t`, or any
/// diagonal gate on the control `c`. Both instances must have no
/// external controls.
fn cnot_commutes_with_1q(cnot: &GateInstance, other: &GateInstance) -> bool {
    if cnot.gate != Gate::Cnot || !cnot.controls.is_empty() {
        return false;
    }
    if other.gate.arity() != 1 || !other.controls.is_empty() {
        return false;
    }
    let control = cnot.qubits[0];
    let target = cnot.qubits[1];
    let q = other.qubits[0];
    if q == target {
        // Commutes with X on the target: gates that are functions of I and X.
        matches!(other.gate, Gate::X | Gate::Rx(_))
    } else if q == control {
        // Any diagonal gate on the control passes through CNOT.
        other.gate.is_diagonal()
    } else {
        // `q` overlaps the CNOT support only via {control, target}; if it
        // is neither, supports do not overlap (handled by rule 1).
        false
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-ir passes::commute`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Lint**

Run: `cargo clippy -p aleph-ir --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-ir/src/passes/commute.rs
git commit -m "[P1-13] gates_commute rules: diagonal, identical, CNOT control/target"
```

---

## Task 3: Non-commuting sanity + symmetry

**Files:**
- Modify: `crates/aleph-ir/src/passes/commute.rs`

- [ ] **Step 1: Write the tests** (add to the `tests` module)

```rust
    #[test]
    fn non_commuting_pairs_are_false() {
        assert!(!gates_commute(&g(Gate::X, &[0]), &g(Gate::Z, &[0])));
        assert!(!gates_commute(&g(Gate::H, &[0]), &g(Gate::X, &[0])));
        // CNOT(0,1) and CNOT(1,2): control of one is target of the other.
        assert!(!gates_commute(&g(Gate::Cnot, &[0, 1]), &g(Gate::Cnot, &[1, 2])));
    }

    #[test]
    fn commute_is_symmetric() {
        // gates_commute(a,b) must equal gates_commute(b,a) for every case.
        let cases: &[(GateInstance, GateInstance)] = &[
            (g(Gate::H, &[0]), g(Gate::X, &[1])),       // disjoint → true
            (g(Gate::Z, &[0]), rz(0.4, 0)),             // both diagonal → true
            (g(Gate::Cnot, &[0, 1]), g(Gate::X, &[1])), // cnot/target → true
            (g(Gate::Cnot, &[0, 1]), g(Gate::Z, &[0])), // cnot/control → true
            (g(Gate::X, &[0]), g(Gate::Z, &[0])),       // → false
            (g(Gate::Cnot, &[0, 1]), g(Gate::Y, &[1])), // → false
            (g(Gate::Cnot, &[0, 1]), g(Gate::Cnot, &[1, 2])), // → false
        ];
        for (a, b) in cases {
            assert_eq!(
                gates_commute(a, b),
                gates_commute(b, a),
                "asymmetry on {:?} / {:?}",
                a.gate,
                b.gate
            );
        }
    }
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p aleph-ir passes::commute`
Expected: PASS (the implementation is already symmetric — rule 4 tries both orders, rules 1–3 are symmetric).

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-ir/src/passes/commute.rs
git commit -m "[P1-13] Tests: non-commuting sanity + symmetry"
```

---

## Task 4: SV oracle — reorder-equivalence guards soundness

**Files:**
- Create: `crates/aleph-sv/tests/commute_oracle.rs`

- [ ] **Step 1: Confirm the backend API** used by the sibling oracle

Run: `sed -n '1,35p' crates/aleph-sv/tests/cancel_oracle.rs`
Expected: shows `use aleph_backend::run;`, `aleph_sv::NaiveSvBackend::with_seed(0)`, `run(&mut backend, &c).unwrap().amplitudes().to_vec()`. Reuse this shape.

- [ ] **Step 2: Write the oracle file**

Create `crates/aleph-sv/tests/commute_oracle.rs`:

```rust
//! P1-13 oracle — soundness guard for `gates_commute`. For every pair
//! the predicate calls commuting, applying the two gates in either order
//! must yield the same full state vector (1e-12). For a sample of
//! non-commuting pairs, the two orders must differ (BACKLOG sanity). A
//! proptest enforces the `commute ⟹ equal` direction over random pairs.

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_ir::passes::gates_commute;
use aleph_ir::{Circuit, Instruction};

const TOL: f64 = 1e-12;
const N: u32 = 3;

fn amplitudes(c: &Circuit) -> Vec<Complex> {
    let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
    run(&mut backend, c).unwrap().amplitudes().to_vec()
}

/// State vector after applying `gates` in order to |0…0⟩ on N qubits.
fn state_after(gates: &[GateInstance]) -> Vec<Complex> {
    let mut c = Circuit::new(N, 0);
    for gi in gates {
        c.add_gate(gi.clone()).unwrap();
    }
    amplitudes(&c)
}

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}
fn rz(theta: f64, q: u32) -> GateInstance {
    GateInstance::new(Gate::Rz(Param::Concrete(theta)), [q].to_vec())
}
fn rx(theta: f64, q: u32) -> GateInstance {
    GateInstance::new(Gate::Rx(Param::Concrete(theta)), [q].to_vec())
}

fn assert_reorder_equal(a: &GateInstance, b: &GateInstance) {
    assert!(
        gates_commute(a, b),
        "test bug: pair not reported commuting: {:?}/{:?}",
        a.gate,
        b.gate
    );
    let ab = state_after(&[a.clone(), b.clone()]);
    let ba = state_after(&[b.clone(), a.clone()]);
    for (k, (x, y)) in ab.iter().zip(ba.iter()).enumerate() {
        assert!(
            (x.re - y.re).abs() < TOL && (x.im - y.im).abs() < TOL,
            "commuting pair {:?}/{:?} changed amplitude[{k}]: {x:?} vs {y:?}",
            a.gate,
            b.gate
        );
    }
}

fn assert_reorder_differs(a: &GateInstance, b: &GateInstance) {
    let ab = state_after(&[a.clone(), b.clone()]);
    let ba = state_after(&[b.clone(), a.clone()]);
    let differs = ab
        .iter()
        .zip(ba.iter())
        .any(|(x, y)| (x.re - y.re).abs() > TOL || (x.im - y.im).abs() > TOL);
    assert!(
        differs,
        "expected non-commuting pair {:?}/{:?} to differ on |0…0⟩",
        a.gate,
        b.gate
    );
}

#[test]
fn commuting_pairs_preserve_state() {
    // Need a non-trivial input on the shared qubits, so prefix the system
    // into a superposition where order would matter if they didn't commute.
    // state_after starts from |0…0⟩; for pairs whose action on |0⟩ is
    // order-insensitive only by luck, we instead rely on the proptest for
    // breadth and use clearly order-sensitive operators here.
    assert_reorder_equal(&g(Gate::H, &[0]), &g(Gate::X, &[1])); // disjoint
    assert_reorder_equal(&g(Gate::Z, &[0]), &rz(0.4, 0)); // diagonal
    assert_reorder_equal(&g(Gate::S, &[0]), &g(Gate::T, &[0])); // diagonal
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[1])); // cnot/target
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &rx(0.9, 1)); // cnot/target
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &g(Gate::Z, &[0])); // cnot/control
    assert_reorder_equal(&g(Gate::Cnot, &[0, 1]), &rz(0.5, 0)); // cnot/control
}

#[test]
fn non_commuting_pairs_differ() {
    // Cases chosen so the two orders differ on |0…0⟩ directly (no state
    // prep needed). NOTE: a pair like Cnot(0,1) ∥ Z(1) (Z on target) does
    // NOT differ on |0…0⟩ — the control is |0⟩ so CNOT is identity and Z
    // fixes |0⟩ — so it is intentionally NOT used here; its
    // non-commutation is covered by the unit test
    // `cnot_does_not_commute_with_z_target_or_x_control`.
    assert_reorder_differs(&g(Gate::X, &[0]), &g(Gate::Z, &[0]));
    assert_reorder_differs(&g(Gate::H, &[0]), &g(Gate::X, &[0]));
    assert_reorder_differs(&g(Gate::Cnot, &[0, 1]), &g(Gate::X, &[0])); // X on control
}
```

Note on `commuting_pairs_preserve_state`: a commuting pair applied to `|0…0⟩` could match by coincidence even if it did NOT commute, so this test alone is weak for some pairs. Its job is to confirm the predicate's `true` entries are at least consistent on a concrete state; the proptest (next step) provides breadth, and the soundness math is in the spec. The chosen cases here (`Cnot ∥ X(1)`, `Cnot ∥ Rx(1)`, etc.) are genuinely order-sensitive on `|0…0⟩` when they do NOT commute, so they meaningfully exercise equality.

- [ ] **Step 3: Add the proptest** (append to `commute_oracle.rs`)

```rust
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    // For any random pair of gate instructions: if gates_commute says they
    // commute, reordering them must not change the state vector. This is
    // the strong false-positive guard. (We do not assert the converse —
    // some non-commuting pairs coincidentally agree on a given input.)
    #[test]
    fn commute_implies_reorder_equal(
        c in arb_circuit_emittable(N, 0, 2).prop_filter(
            "exactly two gate instructions",
            |c| c.instructions().len() == 2
                && c.instructions().iter().all(|i| matches!(i, Instruction::Gate(_)))
        )
    ) {
        let g0 = match &c.instructions()[0] {
            Instruction::Gate(gi) => gi.clone(),
            _ => unreachable!(),
        };
        let g1 = match &c.instructions()[1] {
            Instruction::Gate(gi) => gi.clone(),
            _ => unreachable!(),
        };
        if gates_commute(&g0, &g1) {
            let ab = state_after(&[g0.clone(), g1.clone()]);
            let ba = state_after(&[g1, g0]);
            for (x, y) in ab.iter().zip(ba.iter()) {
                prop_assert!((x.re - y.re).abs() < TOL && (x.im - y.im).abs() < TOL);
            }
        }
    }
}
```

- [ ] **Step 4: Run the oracle**

Run: `cargo test -p aleph-sv --test commute_oracle`
Expected: PASS (3 hand-built tests + the proptest, 256 cases with some filtered).

If `commute_implies_reorder_equal` ever FAILS, the predicate has a false positive (an unsound `true`) — that is a correctness bug in `gates_commute`'s rules, not a test problem; fix the rule.

- [ ] **Step 5: Lint**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/tests/commute_oracle.rs
git commit -m "[P1-13] SV oracle: commute => reorder preserves state vector"
```

---

## Task 5: Module doc + full verification

**Files:**
- Modify: `crates/aleph-ir/src/passes/mod.rs` (module doc only)

- [ ] **Step 1: Mention `gates_commute` in the module doc**

At the top of `crates/aleph-ir/src/passes/mod.rs`, update the doc paragraph to mention the new primitive, e.g. add a sentence after the list of passes:

```rust
//! This module also exports [`commute::gates_commute`], a sound,
//! conservative commutation predicate over `GateInstance` pairs that
//! future passes use to decide when gates may be reordered. It is a
//! free function, not a [`Pass`], and is not part of `default_pipeline`.
```

- [ ] **Step 2: Full workspace verification**

Run: `cargo test --workspace`
Expected: PASS — capture the result by exit code (`echo $?` == 0) and grep the log for `FAILED`/`; [1-9][0-9]* failed`/`panicked`; confirm none. (Do not trust a tail alone.)

- [ ] **Step 3: Lint + format**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
Expected: clean.

- [ ] **Step 4: Confirm `git status` is clean** (no stray `*.proptest-regressions` changes implying a hidden failure)

Run: `git status --porcelain`
Expected: empty. If a `*.proptest-regressions` file changed, a property test failed during the run — investigate before proceeding (P1-12 lesson).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-ir/src/passes/mod.rs
git commit -m "[P1-13] Docs: note gates_commute in passes module doc"
```

---

## Notes / follow-ups

- **No consuming pass** in this ticket (per spec). A future commutation-aware cancellation/fusion pass is the intended consumer; if it runs after DCE it may re-expose the single-pass idempotence concern from P1-12 — a run-to-fixpoint wrapper in `PassPipeline::run` would be the robust fix then.
- **No benchmark:** `gates_commute` is a predicate, not an optimisation; there is no before/after wall-clock to report, so the CLAUDE.md "benchmark every optimisation" rule does not apply.
- **`/code-review` high-effort** after the branch is green, before opening the PR (consistent with prior Stage-2 tickets). Pay special attention to soundness of every `true` rule — a false positive is a silent state-corruption bug; the oracle proptest is the guard.
- **Deferred (spec §4/§6):** CNOT·CNOT partial overlap, different non-diagonal 1q on the same qubit, `Y`/`Ry` on CNOT target, externally-controlled gates beyond rules 1–3.
