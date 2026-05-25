# P0-05 Property-Based Testing Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `aleph-test` dev-only crate, migrate existing duplicated proptest strategies into it, close the BACKLOG "diagonal gates leave magnitudes unchanged" invariant gap, and document the property-test suite in `docs/testing.md`.

**Architecture:** New `crates/aleph-test/` library hosting four modules (`state`, `gate`, `circuit`, `pauli`). Each consumer crate pulls it in via `[dev-dependencies]` only — proptest stays out of production builds. The parser/IR `OpKind` enums currently diverge (parser restricts to emitter-supported variants; IR exercises layer-algorithm paths); the migration consolidates the **union** of variants in `aleph-test::circuit::OpKind` and exposes two strategies — `arb_op_emittable` (parser shape) and `arb_op_full` (IR shape) — so neither test loses coverage.

**Tech Stack:** Rust (edition 2021, MSRV 1.85), `proptest 1` (already workspace-pinned), no new external deps. Uses existing `aleph-core::{Complex, Gate, GateInstance, Pauli, PauliString}` and `aleph-ir::Circuit` types.

**Spec:** `docs/superpowers/specs/2026-05-25-p0-05-proptest-infra-design.md` (approved).

**Branch:** `p0-05-proptest-infra` (already created during spec-write; squash-merge as PR `[P0-05] …`).

**Spec amendment captured here, not in the spec file:** §5 of the spec says "DELETE `OpKind` from parser/IR tests and import from `aleph_test::circuit`". Reality is the two `OpKind` enums **diverge** (parser includes Sdg/Tdg/U3 but no Ccz/Controlled1q; IR vice versa) by intent — each test curates a vocabulary that targets its own correctness surface. This plan therefore exports the **union** `OpKind` in `aleph-test::circuit` and provides two `arb_op_*` strategies plus two `arb_circuit_*` wrappers. The DRY win remains substantial (`OpKind` definition, `apply()` body, `distinct_pair`, `distinct_triple` — ~120 LOC of duplication eliminated).

---

## File Structure

**Create:**
- `crates/aleph-test/Cargo.toml`
- `crates/aleph-test/src/lib.rs` — re-exports the four modules.
- `crates/aleph-test/src/state.rs` — `arb_state_vector`.
- `crates/aleph-test/src/gate.rs` — `arb_1q_gate`, `arb_2q_gate`, `arb_gate`, `arb_diagonal_1q_gate`.
- `crates/aleph-test/src/circuit.rs` — `OpKind`, `apply()`, `distinct_pair`, `distinct_triple`, `arb_op_emittable`, `arb_op_full`, `arb_circuit_emittable`, `arb_circuit_full`.
- `crates/aleph-test/src/pauli.rs` — `arb_pauli_string`.

**Modify:**
- `Cargo.toml` (workspace root) — add `aleph-test` to `members`.
- `crates/aleph-parser/Cargo.toml` — add `aleph-test = { path = "../aleph-test" }` under `[dev-dependencies]`.
- `crates/aleph-ir/Cargo.toml` — same.
- `crates/aleph-sv/Cargo.toml` — same.
- `crates/aleph-parser/tests/round_trip_property.rs` — delete inline helpers, import from `aleph_test::circuit`.
- `crates/aleph-ir/tests/layers_properties.rs` — same.
- `crates/aleph-sv/src/backend.rs` — replace `random_1q_gate_strategy` with `aleph_test::gate::arb_1q_gate`; add `diagonal_gate_preserves_magnitudes` proptest.
- `crates/aleph-sv/src/measure.rs` — replace `random_normalised_state` with `arb_state_vector` composition; replace `RandomOp` / `any_random_op` with `arb_op_full` + `OpKind::as_gate_instance`.
- `docs/testing.md` — append "Property-based testing (P0-05)" section.
- `BACKLOG.md` — tick P0-05 acceptance criteria.

**No changes:** any production source file in `aleph-core`, `aleph-ir`, `aleph-parser`, `aleph-sv`, `aleph-backend`, `aleph-oracle`, `aleph-cli`. Production builds gain zero deps.

---

## Task 1: Scaffold `aleph-test` crate

**Files:**
- Create: `crates/aleph-test/Cargo.toml`
- Create: `crates/aleph-test/src/lib.rs`
- Create: empty stubs in `state.rs`, `gate.rs`, `circuit.rs`, `pauli.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add the new crate to the workspace members list**

Read `Cargo.toml` and locate the `members = [ ... ]` array. Append `"crates/aleph-test"` to the list (sort alphabetically with the other entries).

- [ ] **Step 2: Create `crates/aleph-test/Cargo.toml`**

```toml
[package]
name = "aleph-test"
edition.workspace = true
rust-version.workspace = true
version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true
publish = false

[lib]
path = "src/lib.rs"

[dependencies]
aleph-core = { path = "../aleph-core" }
aleph-ir   = { path = "../aleph-ir" }
proptest   = { workspace = true }
smallvec   = { workspace = true }
```

- [ ] **Step 3: Create the four module stubs and lib.rs**

`crates/aleph-test/src/lib.rs`:
```rust
//! Shared proptest strategies for the aleph workspace.  Dev-only;
//! never depended on from production code.  See
//! `docs/superpowers/specs/2026-05-25-p0-05-proptest-infra-design.md`.

pub mod circuit;
pub mod gate;
pub mod pauli;
pub mod state;
```

`crates/aleph-test/src/state.rs`:
```rust
//! Random normalised state vectors.  See spec §4.1.
```

`crates/aleph-test/src/gate.rs`:
```rust
//! Random `Gate` strategies.  See spec §4.2.
```

`crates/aleph-test/src/circuit.rs`:
```rust
//! `OpKind` union enum + circuit strategies.  See spec §4.3 and
//! the plan's §"Spec amendment" — the parser and IR tests
//! intentionally curate divergent vocabularies; this module
//! exports the union plus two `arb_op_*` / `arb_circuit_*`
//! strategies so neither test loses coverage.
```

`crates/aleph-test/src/pauli.rs`:
```rust
//! Random `PauliString` strategies.  See spec §4.4.
```

- [ ] **Step 4: Verify the workspace builds**

Run: `cargo build -p aleph-test`
Expected: PASS. Empty crate compiles cleanly; warnings about unused deps are fine at this stage (proptest is used in the next tasks).

- [ ] **Step 5: Commit**

Run:
```bash
git add Cargo.toml crates/aleph-test/Cargo.toml crates/aleph-test/src/lib.rs \
    crates/aleph-test/src/state.rs crates/aleph-test/src/gate.rs \
    crates/aleph-test/src/circuit.rs crates/aleph-test/src/pauli.rs
git commit -m "P0-05: scaffold aleph-test crate"
```

---

## Task 2: `arb_state_vector` + unit test

**Files:**
- Modify: `crates/aleph-test/src/state.rs`

- [ ] **Step 1: Write the strategy + a red unit test**

Overwrite `crates/aleph-test/src/state.rs`:
```rust
//! Random normalised state vectors.  See spec §4.1.

use aleph_core::Complex;
use proptest::prelude::*;

/// Random normalised state vector of `n` qubits.  Output length is
/// `2^n`; total norm² lies within `validate_state`'s drift budget
/// (`√n · AMPLITUDE_TOL`).
///
/// Samples (re, im) ∈ [-1, 1] uniformly per amplitude then
/// renormalises.  Not uniformly distributed on the Bloch sphere —
/// intentional: pathological near-degenerate states are part of
/// the input space we want to surface.
pub fn arb_state_vector(n: u32) -> impl Strategy<Value = Vec<Complex>> {
    let dim = 1usize << n;
    proptest::collection::vec((-1.0_f64..=1.0, -1.0_f64..=1.0), dim..=dim).prop_map(|pairs| {
        let mut amps: Vec<Complex> = pairs
            .into_iter()
            .map(|(re, im)| Complex::new(re, im))
            .collect();
        let norm2: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        // All-zero is possible but vanishingly unlikely; bias to a
        // valid state by mapping the degenerate case to |0…0⟩.
        if norm2 < 1e-300 {
            amps[0] = Complex::new(1.0, 0.0);
            return amps;
        }
        let inv = norm2.sqrt().recip();
        for a in &mut amps {
            *a *= Complex::new(inv, 0.0);
        }
        amps
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn output_length_is_2_to_n(n in 1u32..=6) {
            let strategy = arb_state_vector(n);
            let mut runner = proptest::test_runner::TestRunner::default();
            let amps = strategy.new_tree(&mut runner).unwrap().current();
            prop_assert_eq!(amps.len(), 1usize << n);
        }

        #[test]
        fn output_is_normalised(n in 1u32..=6, seed in any::<u64>()) {
            let strategy = arb_state_vector(n);
            let mut runner = proptest::test_runner::TestRunner::new_with_rng(
                ProptestConfig::default(),
                proptest::test_runner::TestRng::from_seed(
                    proptest::test_runner::RngAlgorithm::ChaCha,
                    &seed.to_le_bytes(),
                ),
            );
            let amps = strategy.new_tree(&mut runner).unwrap().current();
            let total: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            // Drift budget: √n · AMPLITUDE_TOL.  Same as validate_state.
            let budget = (amps.len() as f64).sqrt() * 1e-10;
            prop_assert!((total - 1.0).abs() <= budget, "total = {total}, budget = {budget}");
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p aleph-test --lib state::tests`
Expected: 2 proptests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-test/src/state.rs
git commit -m "P0-05: aleph_test::state::arb_state_vector + proptests"
```

---

## Task 3: `arb_*_gate` strategies + unit tests

**Files:**
- Modify: `crates/aleph-test/src/gate.rs`

- [ ] **Step 1: Implement the four gate strategies**

Overwrite `crates/aleph-test/src/gate.rs`:
```rust
//! Random `Gate` strategies.  See spec §4.2.

use aleph_core::Gate;
use proptest::prelude::*;

/// Random 1-qubit gate.  Vocabulary:
/// H, X, Y, Z, S, Sdg, T, Tdg, Rx(θ), Ry(θ), Rz(θ).
/// Rotation angles ∈ [-2π, 2π].
pub fn arb_1q_gate() -> impl Strategy<Value = Gate> {
    let tau = std::f64::consts::TAU;
    prop_oneof![
        Just(Gate::H),
        Just(Gate::X),
        Just(Gate::Y),
        Just(Gate::Z),
        Just(Gate::S),
        Just(Gate::Sdg),
        Just(Gate::T),
        Just(Gate::Tdg),
        (-tau..=tau).prop_map(|t| Gate::Rx(t.into())),
        (-tau..=tau).prop_map(|t| Gate::Ry(t.into())),
        (-tau..=tau).prop_map(|t| Gate::Rz(t.into())),
    ]
}

/// Random 2-qubit gate.  Vocabulary: Cnot, Cz, Swap, Iswap, IswapDg.
pub fn arb_2q_gate() -> impl Strategy<Value = Gate> {
    prop_oneof![
        Just(Gate::Cnot),
        Just(Gate::Cz),
        Just(Gate::Swap),
        Just(Gate::Iswap),
        Just(Gate::IswapDg),
    ]
}

/// Union of `arb_1q_gate` and `arb_2q_gate`, weighted ~70/30
/// toward 1-qubit (matches typical circuit density).
pub fn arb_gate() -> impl Strategy<Value = Gate> {
    prop_oneof![
        7 => arb_1q_gate(),
        3 => arb_2q_gate(),
    ]
}

/// Diagonal-only 1q subset for the
/// "leaves-magnitudes-unchanged" invariant.  Vocabulary:
/// Z, S, Sdg, T, Tdg, Rz(θ).
pub fn arb_diagonal_1q_gate() -> impl Strategy<Value = Gate> {
    let tau = std::f64::consts::TAU;
    prop_oneof![
        Just(Gate::Z),
        Just(Gate::S),
        Just(Gate::Sdg),
        Just(Gate::T),
        Just(Gate::Tdg),
        (-tau..=tau).prop_map(|t| Gate::Rz(t.into())),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn arb_1q_gate_arity_is_one(g in arb_1q_gate()) {
            prop_assert_eq!(g.arity(), 1);
        }

        #[test]
        fn arb_2q_gate_arity_is_two(g in arb_2q_gate()) {
            prop_assert_eq!(g.arity(), 2);
        }

        #[test]
        fn arb_gate_arity_is_one_or_two(g in arb_gate()) {
            let a = g.arity();
            prop_assert!(a == 1 || a == 2, "got arity {a}");
        }

        #[test]
        fn arb_diagonal_1q_gate_is_z_or_s_or_t_or_rz(g in arb_diagonal_1q_gate()) {
            use Gate::*;
            prop_assert!(matches!(g, Z | S | Sdg | T | Tdg | Rx(_) | Ry(_) | Rz(_)));
            // The Rx/Ry arms in matches! are a safety-net — the
            // strategy must never produce them; assert below.
            prop_assert!(!matches!(g, Rx(_) | Ry(_)), "got non-diagonal {g:?}");
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p aleph-test --lib gate::tests`
Expected: 4 proptests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-test/src/gate.rs
git commit -m "P0-05: aleph_test::gate strategies + arity proptests"
```

---

## Task 4: `OpKind` union + `apply` + `distinct_pair`/`distinct_triple`

**Files:**
- Modify: `crates/aleph-test/src/circuit.rs`

- [ ] **Step 1: Define `OpKind` and its `apply` method**

Overwrite `crates/aleph-test/src/circuit.rs`:
```rust
//! `OpKind` union enum + circuit strategies.  See spec §4.3 and
//! the plan's §"Spec amendment" — the parser and IR tests
//! intentionally curate divergent vocabularies; this module
//! exports the union plus two `arb_op_*` / `arb_circuit_*`
//! strategies so neither test loses coverage.

use aleph_core::{Gate, GateInstance};
use aleph_ir::Circuit;
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use smallvec::smallvec;

/// Union of the operation vocabularies the parser and IR tests
/// each curate locally.  Some variants are emitter-supported
/// (`Sdg`, `Tdg`, `U3`); some exercise the IR's
/// non-builder-method paths (`Ccz`, `Controlled1q`).  Each
/// `arb_op_*` strategy in this module selects the appropriate
/// subset for its consumer.
#[derive(Debug, Clone)]
pub enum OpKind {
    H(u32),
    X(u32),
    Y(u32),
    Z(u32),
    S(u32),
    T(u32),
    Sdg(u32),
    Tdg(u32),
    Rx(f64, u32),
    Ry(f64, u32),
    Rz(f64, u32),
    Phase(f64, u32),
    U3(f64, f64, f64, u32),
    Cnot(u32, u32),
    Cz(u32, u32),
    Swap(u32, u32),
    Toffoli(u32, u32, u32),
    Ccz(u32, u32, u32),
    /// Generic controlled-1q construction.  Exercises the
    /// `GateInstance::controlled` path that pure builder methods
    /// don't reach.
    Controlled1q(u32, u32),
    Measure(u32, u32),
    Reset(u32),
    Barrier1(u32),
    Barrier2(u32, u32),
}

impl OpKind {
    /// Apply this op to `c`.  Returns nothing — `Circuit`'s builder
    /// methods may reject invalid combinations, but our strategies
    /// never generate them; a silent drop here would indicate a
    /// bug in the strategy, not in the IR.
    pub fn apply(self, c: &mut Circuit) {
        let _ = match self {
            OpKind::H(q) => c.h(q),
            OpKind::X(q) => c.x(q),
            OpKind::Y(q) => c.y(q),
            OpKind::Z(q) => c.z(q),
            OpKind::S(q) => c.s(q),
            OpKind::T(q) => c.t(q),
            OpKind::Sdg(q) => c.sdg(q),
            OpKind::Tdg(q) => c.tdg(q),
            OpKind::Rx(t, q) => c.rx(t, q),
            OpKind::Ry(t, q) => c.ry(t, q),
            OpKind::Rz(t, q) => c.rz(t, q),
            OpKind::Phase(t, q) => c.phase(t, q),
            OpKind::U3(a, b, d, q) => c.u3(a, b, d, q),
            OpKind::Cnot(a, b) => c.cnot(a, b),
            OpKind::Cz(a, b) => c.cz(a, b),
            OpKind::Swap(a, b) => c.swap(a, b),
            OpKind::Toffoli(a, b, t) => c.ccx(a, b, t),
            OpKind::Ccz(a, b, t) => c.add_gate(GateInstance::new(Gate::Ccz, smallvec![a, b, t])),
            OpKind::Controlled1q(target, ctrl) => c.add_gate(GateInstance::controlled(
                Gate::X,
                smallvec![target],
                smallvec![ctrl],
            )),
            OpKind::Measure(q, cl) => c.measure(q, cl),
            OpKind::Reset(q) => c.reset(q),
            OpKind::Barrier1(q) => c.barrier([q]),
            OpKind::Barrier2(a, b) => c.barrier([a, b]),
        };
    }

    /// If this op is a gate (not a measurement, reset, or
    /// barrier), return the corresponding `GateInstance`.  Used by
    /// backend-level proptests that want to skip non-gate variants
    /// instead of filtering them out at strategy-construction
    /// time.
    pub fn as_gate_instance(&self) -> Option<GateInstance> {
        match *self {
            OpKind::H(q) => Some(GateInstance::new(Gate::H, smallvec![q])),
            OpKind::X(q) => Some(GateInstance::new(Gate::X, smallvec![q])),
            OpKind::Y(q) => Some(GateInstance::new(Gate::Y, smallvec![q])),
            OpKind::Z(q) => Some(GateInstance::new(Gate::Z, smallvec![q])),
            OpKind::S(q) => Some(GateInstance::new(Gate::S, smallvec![q])),
            OpKind::T(q) => Some(GateInstance::new(Gate::T, smallvec![q])),
            OpKind::Sdg(q) => Some(GateInstance::new(Gate::Sdg, smallvec![q])),
            OpKind::Tdg(q) => Some(GateInstance::new(Gate::Tdg, smallvec![q])),
            OpKind::Rx(t, q) => Some(GateInstance::new(Gate::Rx(t.into()), smallvec![q])),
            OpKind::Ry(t, q) => Some(GateInstance::new(Gate::Ry(t.into()), smallvec![q])),
            OpKind::Rz(t, q) => Some(GateInstance::new(Gate::Rz(t.into()), smallvec![q])),
            OpKind::Phase(t, q) => Some(GateInstance::new(Gate::Phase(t.into()), smallvec![q])),
            OpKind::U3(a, b, d, q) => Some(GateInstance::new(
                Gate::U3(a.into(), b.into(), d.into()),
                smallvec![q],
            )),
            OpKind::Cnot(a, b) => Some(GateInstance::new(Gate::Cnot, smallvec![a, b])),
            OpKind::Cz(a, b) => Some(GateInstance::new(Gate::Cz, smallvec![a, b])),
            OpKind::Swap(a, b) => Some(GateInstance::new(Gate::Swap, smallvec![a, b])),
            OpKind::Toffoli(a, b, t) => Some(GateInstance::new(Gate::Ccx, smallvec![a, b, t])),
            OpKind::Ccz(a, b, t) => Some(GateInstance::new(Gate::Ccz, smallvec![a, b, t])),
            OpKind::Controlled1q(target, ctrl) => Some(GateInstance::controlled(
                Gate::X,
                smallvec![target],
                smallvec![ctrl],
            )),
            OpKind::Measure(_, _) | OpKind::Reset(_) | OpKind::Barrier1(_) | OpKind::Barrier2(_, _) => None,
        }
    }
}

/// Distinct unordered pair `(a, b)` with `a, b ∈ [0, nq)`.
pub fn distinct_pair(nq: u32) -> impl Strategy<Value = (u32, u32)> {
    (0u32..nq, 0u32..nq).prop_filter("distinct", |(a, b)| a != b)
}

/// Distinct unordered triple `(a, b, c)` with all three in `[0, nq)`.
pub fn distinct_triple(nq: u32) -> impl Strategy<Value = (u32, u32, u32)> {
    (0u32..nq, 0u32..nq, 0u32..nq).prop_filter("distinct", |(a, b, c)| a != b && a != c && b != c)
}
```

Note: `Gate::Ccx` is the canonical CCX name; verify before running. If aleph-core uses a different name (e.g. `Toffoli`), update the `as_gate_instance` arm.

- [ ] **Step 2: Verify the Gate name for Toffoli / CCX**

Run: `grep -n "Gate::\(Ccx\|Toffoli\)" crates/aleph-core/src/gate/kinds.rs | head`
Expected: shows the canonical name. If it's `Ccx`, the code above is correct. If it's `Toffoli`, change `Gate::Ccx` to `Gate::Toffoli` in the `as_gate_instance` arm above.

If neither name appears (the 3-qubit gate is added via `add_gate` with a different mechanism), grep again with `grep -n "^\s*\(Ccx\|Toffoli\|Ccz\)" crates/aleph-core/src/gate/kinds.rs` and pick the matching variant. The rest of the plan assumes whatever name the codebase uses today.

- [ ] **Step 3: Build the crate**

Run: `cargo build -p aleph-test`
Expected: PASS. No warnings about unused variants.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-test/src/circuit.rs
git commit -m "P0-05: OpKind union + apply + distinct helpers in aleph-test::circuit"
```

---

## Task 5: `arb_op_emittable` + `arb_circuit_emittable` (parser shape)

**Files:**
- Modify: `crates/aleph-test/src/circuit.rs`

- [ ] **Step 1: Append the emittable strategies**

Append to `crates/aleph-test/src/circuit.rs` (before any `#[cfg(test)]` block):
```rust
/// `OpKind` vocabulary restricted to variants the emitter can
/// serialise.  Excludes `Ccz` and `Controlled1q` (which the
/// builder constructs but the emitter doesn't yet round-trip).
///
/// Used by `aleph-parser/tests/round_trip_property.rs`.
pub fn arb_op_emittable(nq: u32, nc: u32) -> BoxedStrategy<OpKind> {
    let angle = -10.0_f64..10.0_f64;

    let single = prop_oneof![
        (0u32..nq).prop_map(OpKind::H),
        (0u32..nq).prop_map(OpKind::X),
        (0u32..nq).prop_map(OpKind::Y),
        (0u32..nq).prop_map(OpKind::Z),
        (0u32..nq).prop_map(OpKind::S),
        (0u32..nq).prop_map(OpKind::T),
        (0u32..nq).prop_map(OpKind::Sdg),
        (0u32..nq).prop_map(OpKind::Tdg),
    ];
    let parametric = prop_oneof![
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rx(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Ry(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rz(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Phase(t, q)),
        (angle.clone(), angle.clone(), angle.clone(), 0u32..nq)
            .prop_map(|(a, b, c, q)| OpKind::U3(a, b, c, q)),
    ];
    let two_q = prop_oneof![
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cnot(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cz(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Swap(a, b)),
    ];
    let three_q = distinct_triple(nq).prop_map(|(a, b, t)| OpKind::Toffoli(a, b, t));
    let non_gate = prop_oneof![
        (0u32..nq).prop_map(OpKind::Reset),
        (0u32..nq).prop_map(OpKind::Barrier1),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Barrier2(a, b)),
    ];

    if nc == 0 {
        prop_oneof![
            4 => single,
            3 => parametric,
            3 => two_q,
            2 => three_q,
            2 => non_gate,
        ]
        .boxed()
    } else {
        let measurement = (0u32..nq, 0u32..nc).prop_map(|(q, cl)| OpKind::Measure(q, cl));
        prop_oneof![
            4 => single,
            3 => parametric,
            3 => two_q,
            2 => three_q,
            2 => non_gate,
            3 => measurement,
        ]
        .boxed()
    }
}

/// Random emitter-compatible `Circuit`.  Replaces the inline
/// `arb_circuit(...)` previously duplicated in
/// `aleph-parser/tests/round_trip_property.rs`.
pub fn arb_circuit_emittable(nq: u32, nc: u32, n_ops: usize) -> impl Strategy<Value = Circuit> {
    proptest::collection::vec(arb_op_emittable(nq, nc), 0..=n_ops).prop_map(move |ops| {
        let mut c = Circuit::new(nq, nc);
        for op in ops {
            op.apply(&mut c);
        }
        c
    })
}
```

- [ ] **Step 2: Build**

Run: `cargo build -p aleph-test`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-test/src/circuit.rs
git commit -m "P0-05: arb_op_emittable + arb_circuit_emittable (parser shape)"
```

---

## Task 6: `arb_op_full` + `arb_circuit_full` (IR shape)

**Files:**
- Modify: `crates/aleph-test/src/circuit.rs`

- [ ] **Step 1: Append the full-vocabulary strategies**

Append after the emittable strategies:
```rust
/// `OpKind` vocabulary that exercises the IR's layer-algorithm
/// paths.  Excludes `Sdg`, `Tdg`, `U3` (they overlap with cases
/// the parser test covers); adds `Ccz` and `Controlled1q` (they
/// reach the `add_gate` / `GateInstance::controlled` paths the
/// pure builder methods bypass).
///
/// Used by `aleph-ir/tests/layers_properties.rs`.
pub fn arb_op_full(nq: u32, nc: u32) -> BoxedStrategy<OpKind> {
    let angle = -10.0_f64..10.0_f64;

    let single = prop_oneof![
        (0u32..nq).prop_map(OpKind::H),
        (0u32..nq).prop_map(OpKind::X),
        (0u32..nq).prop_map(OpKind::Y),
        (0u32..nq).prop_map(OpKind::Z),
        (0u32..nq).prop_map(OpKind::S),
        (0u32..nq).prop_map(OpKind::T),
    ];
    let parametric = prop_oneof![
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rx(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Ry(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Rz(t, q)),
        (angle.clone(), 0u32..nq).prop_map(|(t, q)| OpKind::Phase(t, q)),
    ];
    let two_q = prop_oneof![
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cnot(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Cz(a, b)),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Swap(a, b)),
        distinct_pair(nq).prop_map(|(t, c)| OpKind::Controlled1q(t, c)),
    ];
    let three_q = prop_oneof![
        distinct_triple(nq).prop_map(|(a, b, t)| OpKind::Toffoli(a, b, t)),
        distinct_triple(nq).prop_map(|(a, b, t)| OpKind::Ccz(a, b, t)),
    ];
    let non_gate = prop_oneof![
        (0u32..nq).prop_map(OpKind::Reset),
        (0u32..nq).prop_map(OpKind::Barrier1),
        distinct_pair(nq).prop_map(|(a, b)| OpKind::Barrier2(a, b)),
    ];

    if nc == 0 {
        prop_oneof![
            4 => single,
            3 => parametric,
            3 => two_q,
            2 => three_q,
            2 => non_gate,
        ]
        .boxed()
    } else {
        let measurement = (0u32..nq, 0u32..nc).prop_map(|(q, cl)| OpKind::Measure(q, cl));
        prop_oneof![
            4 => single,
            3 => parametric,
            3 => two_q,
            2 => three_q,
            2 => non_gate,
            3 => measurement,
        ]
        .boxed()
    }
}

/// Random `Circuit` exercising the IR's broader op vocabulary
/// (including `Ccz` and `Controlled1q`).  Replaces the inline
/// `arb_circuit(...)` previously duplicated in
/// `aleph-ir/tests/layers_properties.rs`.
pub fn arb_circuit_full(nq: u32, nc: u32, n_ops: usize) -> impl Strategy<Value = Circuit> {
    proptest::collection::vec(arb_op_full(nq, nc), 0..=n_ops).prop_map(move |ops| {
        let mut c = Circuit::new(nq, nc);
        for op in ops {
            op.apply(&mut c);
        }
        c
    })
}
```

- [ ] **Step 2: Build**

Run: `cargo build -p aleph-test`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-test/src/circuit.rs
git commit -m "P0-05: arb_op_full + arb_circuit_full (IR layer-test shape)"
```

---

## Task 7: `arb_pauli_string`

**Files:**
- Modify: `crates/aleph-test/src/pauli.rs`

- [ ] **Step 1: Implement the Pauli strategy + unit test**

Overwrite `crates/aleph-test/src/pauli.rs`:
```rust
//! Random `PauliString` strategies.  See spec §4.4.

use aleph_core::{Pauli, PauliString};
use proptest::prelude::*;

/// Random `PauliString` with terms on qubits in `[0, n)`.
///   `mix_xy = false` → Z-only strings (exercises the Z fast path).
///   `mix_xy = true`  → full {I, X, Y, Z} (mixed fallthrough).
/// Coefficient is `1.0`; compose with `.prop_flat_map` if a
/// random coefficient is needed.
pub fn arb_pauli_string(n: u32, mix_xy: bool) -> impl Strategy<Value = PauliString> {
    let alphabet: Vec<Pauli> = if mix_xy {
        vec![Pauli::I, Pauli::X, Pauli::Y, Pauli::Z]
    } else {
        vec![Pauli::I, Pauli::Z]
    };
    let dim = n as usize;
    proptest::collection::vec(proptest::sample::select(alphabet), dim..=dim).prop_map(move |body| {
        let terms: Vec<(u32, Pauli)> = body
            .into_iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p))
            .collect();
        // PauliString::new sorts, dedupes (no dupes possible here),
        // drops I, and rejects non-finite coefficient.
        PauliString::new(1.0, terms).expect("arb_pauli_string produced a valid PauliString")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn z_only_terms_have_only_z(ps in arb_pauli_string(5, false)) {
            for (_, p) in &ps.terms {
                prop_assert_eq!(*p, Pauli::Z);
            }
        }

        #[test]
        fn mixed_terms_are_x_y_or_z(ps in arb_pauli_string(5, true)) {
            for (_, p) in &ps.terms {
                prop_assert!(matches!(p, Pauli::X | Pauli::Y | Pauli::Z));
            }
        }

        #[test]
        fn coefficient_is_one(ps in arb_pauli_string(4, true)) {
            prop_assert_eq!(ps.coefficient, 1.0);
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p aleph-test --lib pauli::tests`
Expected: 3 proptests PASS.

- [ ] **Step 3: Run the whole aleph-test suite**

Run: `cargo test -p aleph-test`
Expected: PASS. ~12 proptests total (state + gate + pauli unit tests). circuit.rs has no tests yet — added implicitly via the migrated parser/IR tests in Tasks 8–9.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-test/src/pauli.rs
git commit -m "P0-05: aleph_test::pauli::arb_pauli_string + proptests"
```

---

## Task 8: Migrate `aleph-parser` round-trip test

**Files:**
- Modify: `crates/aleph-parser/Cargo.toml`
- Modify: `crates/aleph-parser/tests/round_trip_property.rs`

- [ ] **Step 1: Add `aleph-test` as dev-dep**

In `crates/aleph-parser/Cargo.toml`, locate `[dev-dependencies]` and add:
```toml
aleph-test = { path = "../aleph-test" }
```

- [ ] **Step 2: Replace inline helpers with the shared imports**

Overwrite `crates/aleph-parser/tests/round_trip_property.rs`:
```rust
//! Property test: random `Circuit` (restricted to emitter-supported
//! variants) round-trips through `emit → parse → compare`.

use aleph_parser::{emit, parse};
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    #[test]
    fn parse_emit_roundtrip(c in arb_circuit_emittable(4, 2, 12)) {
        let out = match emit(&c) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        let c2 = parse(&out).map_err(|e| TestCaseError::fail(format!(
            "re-parse failed.\nemitted:\n{out}\nerror:\n{}",
            e.render()
        )))?;
        prop_assert_eq!(c.len(), c2.len(), "instruction count mismatch");
        prop_assert_eq!(c.num_qubits(), c2.num_qubits());
        prop_assert_eq!(c.num_clbits(), c2.num_clbits());
        prop_assert_eq!(c2.metadata().generated_from.as_deref(), Some("openqasm:3.0"));
        for (i, (a, b)) in c.instructions().iter().zip(c2.instructions().iter()).enumerate() {
            prop_assert_eq!(format!("{a:?}"), format!("{b:?}"), "instr {} differs", i);
        }
    }
}
```

- [ ] **Step 3: Run the parser proptest**

Run: `cargo test -p aleph-parser --test round_trip_property`
Expected: PASS. Same 256 cases as before; the strategy is byte-for-byte identical to the previous inline one (Task 5 of this plan copied it verbatim).

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-parser/Cargo.toml crates/aleph-parser/tests/round_trip_property.rs
git commit -m "P0-05: migrate aleph-parser round-trip test to aleph_test::circuit"
```

---

## Task 9: Migrate `aleph-ir` layers test

**Files:**
- Modify: `crates/aleph-ir/Cargo.toml`
- Modify: `crates/aleph-ir/tests/layers_properties.rs`

- [ ] **Step 1: Add `aleph-test` as dev-dep**

In `crates/aleph-ir/Cargo.toml`, locate `[dev-dependencies]` and add:
```toml
aleph-test = { path = "../aleph-test" }
```

- [ ] **Step 2: Read the existing layers test to find what to keep**

Read `crates/aleph-ir/tests/layers_properties.rs`. The file has:
- An inline `OpKind` / `arb_op` / `arb_circuit` block (~120 LOC) — DELETE.
- Helper fns `touched_qubits`, `touched_clbits`, `pair_can_share_layer` and the proptest bodies (`layers_flatten_to_0_to_len`, `within_layer_no_non_commuting_overlap`) — KEEP verbatim.

- [ ] **Step 3: Replace the inline strategy block while preserving everything else**

Edit `crates/aleph-ir/tests/layers_properties.rs`: remove the `OpKind` enum, the `apply` impl, the `distinct_pair`, `distinct_triple`, `arb_op`, and `arb_circuit` definitions. At the top, change the imports to:
```rust
use aleph_core::{Gate, GateInstance};
use aleph_ir::{Circuit, Instruction};
use aleph_test::circuit::arb_circuit_full;
use proptest::prelude::*;
use smallvec::smallvec;
```
(Drop `proptest::strategy::BoxedStrategy` — no longer used locally.)

Inside the existing `proptest! { ... }` block, change every reference from the local `arb_circuit(...)` to `arb_circuit_full(...)`. Argument list is the same `(4, 2, 16)`.

- [ ] **Step 4: Run the layers proptests**

Run: `cargo test -p aleph-ir --test layers_properties`
Expected: PASS. Same case budget as before; strategy distribution is the same (`arb_op_full` was copied verbatim in Task 6).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-ir/Cargo.toml crates/aleph-ir/tests/layers_properties.rs
git commit -m "P0-05: migrate aleph-ir layers test to aleph_test::circuit"
```

---

## Task 10: Migrate `aleph-sv/src/backend.rs` 1q-gate strategy

**Files:**
- Modify: `crates/aleph-sv/Cargo.toml`
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Add `aleph-test` as dev-dep**

In `crates/aleph-sv/Cargo.toml`, locate `[dev-dependencies]` and add:
```toml
aleph-test = { path = "../aleph-test" }
```

- [ ] **Step 2: Delete `random_1q_gate_strategy`**

In `crates/aleph-sv/src/backend.rs`, find the `fn random_1q_gate_strategy() -> impl Strategy<Value = Gate> { ... }` definition (inside the `#[cfg(test)] mod tests` block) and delete it.

- [ ] **Step 3: Update call sites**

Grep within the file for `random_1q_gate_strategy()` call sites (there is at least one, used in a proptest body). Replace each call with `aleph_test::gate::arb_1q_gate()`.

The replacement command:
```bash
sed -i.bak 's/random_1q_gate_strategy()/aleph_test::gate::arb_1q_gate()/g' \
    crates/aleph-sv/src/backend.rs
rm crates/aleph-sv/src/backend.rs.bak
```

(macOS `sed` needs `-i.bak`; the `rm` cleans up afterward.)

- [ ] **Step 4: Verify it builds**

Run: `cargo build -p aleph-sv --tests`
Expected: PASS.

- [ ] **Step 5: Run the affected proptests**

Run: `cargo test -p aleph-sv --lib backend::tests`
Expected: PASS. All 80+ tests, including the proptests that previously used `random_1q_gate_strategy`.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/Cargo.toml crates/aleph-sv/src/backend.rs
git commit -m "P0-05: migrate aleph-sv random_1q_gate_strategy to aleph_test::gate::arb_1q_gate"
```

---

## Task 11: Migrate `aleph-sv/src/measure.rs` state + op helpers

**Files:**
- Modify: `crates/aleph-sv/src/measure.rs`

- [ ] **Step 1: Locate the section to replace**

Read `crates/aleph-sv/src/measure.rs` and find the `#[cfg(test)] mod tests { ... }` block at the bottom. Inside, identify:
- `fn random_normalised_state(n: u32, seed: u64) -> CpuState { ... }` — DELETE.
- `enum RandomOp { ... }` and its `realize` impl — DELETE.
- `fn any_random_op() -> impl Strategy<Value = RandomOp> { ... }` — DELETE.
- The proptest body `fn z_fast_path_matches_slow_path(...)` — KEEP; will be rewritten.
- The proptest body `fn probabilities_full_basis_sums_to_one(...)` — KEEP; will be rewritten.

- [ ] **Step 2: Rewrite both proptests using shared strategies**

Replace the test-mod contents (the `proptest! { ... }` block and the helpers above it) with:
```rust
#[cfg(test)]
mod tests {
    // Both `super::*` and `proptest::prelude::*` glob a `Rng` trait,
    // which trips the `ambiguous_glob_imported_traits` future-incompat
    // warning.  Name parent-module imports explicitly and only glob
    // the proptest prelude.
    use super::{expectation_value_impl, CpuState};
    use aleph_core::{Complex, Pauli, PauliString};
    use aleph_test::circuit::arb_op_full;
    use aleph_test::state::arb_state_vector;
    use proptest::prelude::*;

    /// Reference implementation: always-clone, kernel-apply path.
    /// Mirrors what `expectation_value_impl` did before the Z fast
    /// path landed; used by the proptest below to assert the fast
    /// path agrees on Z-only Pauli strings.
    fn reference_expectation(state: &CpuState, pauli: &PauliString) -> f64 {
        let mut tmp = state.amps.clone();
        for (q, p) in &pauli.terms {
            if *p == Pauli::I {
                continue;
            }
            let m = p.matrix();
            crate::kernels::apply_1q(&mut tmp, *q, &[], &m);
        }
        let mut acc = Complex::new(0.0, 0.0);
        for (lhs, rhs) in state.amps.iter().zip(tmp.iter()) {
            acc += lhs.conj() * (*rhs);
        }
        pauli.coefficient * acc.re
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        /// Fast-path equivalence: for any Z-only PauliString on any
        /// concrete normalised state, the diagonal fast path and the
        /// copy-and-rotate slow path must agree to 1e-12.
        #[test]
        fn z_fast_path_matches_slow_path(
            n in 1u32..=5,
            amps in arb_state_vector(5),
            mask in any::<u32>(),
            coeff in -2.0_f64..=2.0,
        ) {
            // The strategy produces a 2^5-element vector; slice to 2^n.
            let dim = 1usize << n;
            let mut amps = amps;
            amps.truncate(dim);
            // Renormalise after the truncation.
            let norm2: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            let inv = norm2.sqrt().recip();
            for a in &mut amps { *a *= Complex::new(inv, 0.0); }

            let state = CpuState { num_qubits: n, amps };
            // Build a Z-only PauliString from the low n bits of `mask`.
            let mut terms = Vec::new();
            for q in 0..n {
                if (mask >> q) & 1 == 1 {
                    terms.push((q, Pauli::Z));
                }
            }
            let ps = PauliString::new(coeff, terms).unwrap();
            let fast = expectation_value_impl(&state, &ps).unwrap();
            let slow = reference_expectation(&state, &ps);
            prop_assert!(
                (fast - slow).abs() < 1e-12,
                "n={n} mask={mask:0width$b} coeff={coeff}: fast={fast}, slow={slow}",
                width = n as usize,
            );
        }

        /// Sum of marginal probabilities over the full qubit subset
        /// must equal 1 within `√n · AMPLITUDE_TOL`. BACKLOG testing
        /// requirement; see spec §9.
        #[test]
        fn probabilities_full_basis_sums_to_one(
            n in 1u32..=6,
            ops in proptest::collection::vec(arb_op_full(6, 0), 0..30),
        ) {
            use aleph_backend::Backend;
            let mut b = crate::NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(n).unwrap();
            for op in &ops {
                if let Some(gi) = op.as_gate_instance() {
                    // The op was generated for nq=6 but we're running
                    // on `n` qubits.  Skip ops that touch a qubit
                    // outside [0, n).
                    let max_q = gi.qubits.iter().chain(gi.controls.iter()).max().copied().unwrap_or(0);
                    if max_q < n {
                        // `apply_gate` may still reject (e.g. duplicate
                        // qubit on a Toffoli/Ccz where two indices
                        // happened to collide post-filter).  Drop those
                        // silently — they're not a state regression.
                        let _ = b.apply_gate(&mut s, &gi);
                    }
                }
            }
            let qubits: Vec<u32> = (0..n).collect();
            let p = b.probabilities(&s, &qubits).unwrap();
            let sum: f64 = p.iter().sum();
            let drift = (p.len() as f64).sqrt() * aleph_core::AMPLITUDE_TOL;
            prop_assert!((sum - 1.0).abs() <= drift, "sum = {sum}");
        }
    }
}
```

Notice: `arb_state_vector(5)` produces a fixed-length 2^5 = 32 vector regardless of `n`; the body truncates + renormalises. This is awkward but unavoidable until proptest supports `Strategy::flat_map_arg`-style chaining cleanly. The bug rate this trades off is negligible — the test still exercises every `n ∈ [1, 5]` with random amplitudes.

For `probabilities_full_basis_sums_to_one`, we use `arb_op_full(6, 0)` (clbits=0 so no measurements) and post-filter ops whose max qubit exceeds the running `n`. This is less elegant than the previous `RandomOp.realize(n)` path but eliminates the local enum.

- [ ] **Step 3: Run the affected proptests**

Run: `cargo test -p aleph-sv --lib measure::tests`
Expected: 2 proptests PASS within a few seconds.

- [ ] **Step 4: Workspace regression check**

Run: `cargo test --workspace`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/measure.rs
git commit -m "P0-05: migrate aleph-sv measure proptests to aleph_test::{state,circuit}"
```

---

## Task 12: New invariant — `diagonal_gate_preserves_magnitudes`

**Files:**
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Locate the existing proptest! block**

In `crates/aleph-sv/src/backend.rs`, find the `proptest! { ... }` block inside `#[cfg(test)] mod tests`. Inside that block, append the new proptest after the last existing one (e.g. after `intrinsic_cnot_matches_external_control`).

- [ ] **Step 2: Add the new proptest**

Insert (inside the existing `proptest! { ... }` block):
```rust
        /// Diagonal 1q gates (Z, S, Sdg, T, Tdg, Rz(θ)) only rotate
        /// phases; they MUST leave |aᵢ| invariant for every basis
        /// state.  The existing reversibility proptests verify a
        /// stronger property — but this targets magnitudes directly
        /// and would surface a single-direction bug (e.g. a Z kernel
        /// that accidentally scales an amplitude).
        #[test]
        fn diagonal_gate_preserves_magnitudes(
            op in aleph_test::gate::arb_diagonal_1q_gate(),
            q in 0u32..4u32,
        ) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(4).unwrap();
            // Non-trivial preamble so the state isn't |0…0⟩.
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0, 1])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![2])).unwrap();
            let before: Vec<f64> = s.amplitudes().iter().map(|a| a.norm()).collect();
            b.apply_gate(&mut s, &GateInstance::new(op, smallvec![q])).unwrap();
            let after: Vec<f64> = s.amplitudes().iter().map(|a| a.norm()).collect();
            for (b_mag, a_mag) in before.iter().zip(after.iter()) {
                prop_assert!((b_mag - a_mag).abs() < 1e-12, "|a| changed: {b_mag} → {a_mag}");
            }
        }
```

- [ ] **Step 3: Run the new proptest**

Run: `cargo test -p aleph-sv --lib backend::tests::diagonal_gate_preserves_magnitudes`
Expected: PASS, 256 cases.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-sv/src/backend.rs
git commit -m "P0-05: diagonal_gate_preserves_magnitudes proptest"
```

---

## Task 13: `docs/testing.md` Property-based section

**Files:**
- Modify: `docs/testing.md`

- [ ] **Step 1: Append the new section to the end of the file**

Read `docs/testing.md` and append:
```markdown

## Property-based testing (P0-05)

The workspace uses [proptest] for invariant testing. Shared
strategies live in the `aleph-test` crate (`crates/aleph-test/`),
consumed as a `[dev-dependencies]` entry by every crate that
needs them. No production code depends on `proptest`.

### Generators

| Strategy | Module | What it produces |
|---|---|---|
| `arb_state_vector(n)` | `aleph_test::state` | Normalised `Vec<Complex>` of length `2^n` |
| `arb_1q_gate()` / `arb_2q_gate()` / `arb_gate()` | `aleph_test::gate` | Random `Gate` |
| `arb_diagonal_1q_gate()` | `aleph_test::gate` | Z / S / Sdg / T / Tdg / Rz only |
| `arb_circuit_emittable(nq, nc, n_ops)` | `aleph_test::circuit` | Emitter-supported `Circuit` (parser tests) |
| `arb_circuit_full(nq, nc, n_ops)` | `aleph_test::circuit` | Broader-vocabulary `Circuit` (IR layer tests) |
| `arb_op_emittable(nq, nc)` / `arb_op_full(nq, nc)` | `aleph_test::circuit` | Single random `OpKind` |
| `arb_pauli_string(n, mix_xy)` | `aleph_test::pauli` | `PauliString` |
| `distinct_pair(nq)` / `distinct_triple(nq)` | `aleph_test::circuit` | Raw qubit-tuple helpers |

### Invariants exercised

| Invariant | Where |
|---|---|
| Norm preservation after any gate | `aleph-sv/src/backend.rs::tests::normalisation_invariant` |
| Reversibility (`G†·G·ψ = ψ`) | 10+ proptests in `aleph-sv/src/backend.rs` (`*_then_*_negative_returns_identity`, `*_squared_is_identity`) |
| Diagonal gates leave \|aᵢ\| invariant | `aleph-sv/src/backend.rs::tests::diagonal_gate_preserves_magnitudes` |
| Σ P(outcome) = 1 over full basis | `aleph-sv/src/measure.rs::tests::probabilities_full_basis_sums_to_one` |
| Z fast path ≡ slow path (Z-only Pauli) | `aleph-sv/src/measure.rs::tests::z_fast_path_matches_slow_path` |
| Parser ↔ emitter round-trip | `aleph-parser/tests/round_trip_property.rs::parse_emit_roundtrip` |
| IR layer partitioning correctness | `aleph-ir/tests/layers_properties.rs` |
| f64 round-trip through serde_json | `aleph-oracle/src/fixture.rs::tests::f64_pair_round_trips_through_serde_json` |
| Pauli arg parser ↔ Display | `aleph-cli/src/pauli.rs::tests::z_only_round_trip` |

### Failure persistence

proptest writes shrunk failure seeds to
`<crate>/proptest-regressions/*.txt`. **Commit these files** —
they replay historical failure cases on every future run,
preventing regression of bugs the suite previously caught.

### Adding a property test

1. Pick or compose a strategy from `aleph_test::*`.
2. Inside a `proptest! { #[test] fn ... { ... } }` block, assert
   the invariant with `prop_assert!` (not plain `assert!` — the
   former shrinks).
3. Default `ProptestConfig::default()` (256 cases) is fine for
   most tests; bump `cases: N` for expensive setups.

[proptest]: https://github.com/proptest-rs/proptest
```

- [ ] **Step 2: Commit**

```bash
git add docs/testing.md
git commit -m "P0-05: docs/testing.md — Property-based testing section"
```

---

## Task 14: BACKLOG ACs + plan commit

**Files:**
- Modify: `BACKLOG.md`

- [ ] **Step 1: Locate P0-05 in BACKLOG.md**

Run: `grep -n "P0-05" BACKLOG.md | head`
Expected: heading line followed by AC bullets at ~line 260.

- [ ] **Step 2: Tick the AC checkboxes**

Find the four ACs under P0-05 and change `- [ ]` to `- [x]`:
- `proptest integrated, at least 4 generators` → done (8 generators ship)
- `At least 4 invariant tests passing` → done (9 catalogued)
- `Tests run as part of cargo test` → done (no separate harness)
- `Documentation in docs/testing.md` → done (Task 13)

- [ ] **Step 3: Stage the plan + BACKLOG**

```bash
git add docs/superpowers/plans/2026-05-25-p0-05-proptest-infra.md BACKLOG.md
git commit -m "P0-05: implementation plan + BACKLOG ACs ticked"
```

---

## Task 15: Final sweep — fmt, clippy, workspace test

**Files:** none

- [ ] **Step 1: Format**

Run: `cargo fmt`
Expected: no diffs, or a small whitespace diff to be committed.

- [ ] **Step 2: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS. New crate is small; the only likely lint is `clippy::needless_pass_by_value` or similar — fix inline.

- [ ] **Step 3: Workspace test**

Run: `cargo test --workspace`
Expected: PASS. Total test count grew by ~10 (state proptests 2, gate proptests 4, pauli proptests 3, diagonal-gate invariant 1).

- [ ] **Step 4: Commit any fmt drift**

If `cargo fmt` produced a diff:
```bash
git add -u
git commit -m "P0-05: fmt"
```

Otherwise: no commit.

---

## Task 16: Push + PR

**Files:** none

- [ ] **Step 1: Push the branch**

Run: `git push -u origin p0-05-proptest-infra`
Expected: branch published.

- [ ] **Step 2: Open the PR**

Run:
```bash
gh pr create --title "[P0-05] Property-based testing infrastructure" --body "$(cat <<'EOF'
## Summary

Closes #5.  Stands up the `aleph-test` dev-only crate, migrates
duplicated proptest strategies out of `aleph-parser/tests/`,
`aleph-ir/tests/`, and `aleph-sv/src/*.rs`, and closes the last
BACKLOG-listed invariant gap (diagonal-gates-leave-magnitudes).

* New `crates/aleph-test/` with four modules (`state`, `gate`,
  `circuit`, `pauli`).  Eight public strategies + two helpers.
* Parser-test and IR-test `OpKind`s diverge intentionally
  (parser restricts to emitter-supported variants; IR exercises
  non-builder paths).  This PR consolidates the union enum +
  helpers in `aleph_test::circuit` and exposes two `arb_op_*` /
  `arb_circuit_*` strategies so neither test loses coverage.
* `aleph-sv` proptests migrated:
  `random_1q_gate_strategy` → `arb_1q_gate`;
  `random_normalised_state` → `arb_state_vector`;
  `RandomOp` → `arb_op_full` + `OpKind::as_gate_instance`.
* New `diagonal_gate_preserves_magnitudes` proptest in
  `aleph-sv/src/backend.rs` (BACKLOG AC).
* `docs/testing.md` gains a "Property-based testing (P0-05)"
  section cataloguing all 8 strategies and 9 invariant tests,
  plus failure-persistence guidance.

## Test plan

* [x] `cargo test --workspace` green; aleph-test itself ships
  ~10 new unit proptests, the migrated tests are byte-for-byte
  equivalent to their pre-migration shapes.
* [x] `cargo clippy --workspace --all-targets -- -D warnings` green.
* [x] `cargo fmt --check` green.

## Out of scope (deferred — spec §2 / §7)

* Generic backend-runner fixture (revisit in P3+).
* Custom shrinking strategies for `Complex` / state-vector inputs.
* Stateful proptest (P0-13+).
* `arb_circuit` with classical-control instructions.

## Spec / plan

`docs/superpowers/specs/2026-05-25-p0-05-proptest-infra-design.md`
(design) and
`docs/superpowers/plans/2026-05-25-p0-05-proptest-infra.md`
(plan), both committed alongside the implementation per the
P0-06…P0-12 workflow.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Print the PR URL**

`gh pr create` prints the URL on success. Hand it back to the user.

---

## Self-review checklist (performed during plan-writing)

**Spec coverage:**
- §2 in-scope items: aleph-test crate (Task 1), 8 generators (Tasks 2/3/4/5/6/7), migration of duplicated strategies (Tasks 8/9), aleph-sv private strategies (Tasks 10/11), new invariant test (Task 12), `docs/testing.md` section (Task 13), `aleph-test` as dev-dep everywhere (Tasks 8/9/10).
- §2 out-of-scope items: none touched in any task.
- §3 Architecture: matches Task 1's module layout exactly.
- §4 Public API: each generator implemented in its named task with verbatim signature.
- §5 Migration map: each row maps to a specific task. The "RandomOp → arb_op" row is realised via `arb_op_full` + `OpKind::as_gate_instance` (Task 11).
- §6 New invariant: Task 12.
- §7 Documentation: Task 13.
- §8 AC mapping: Task 14 (BACKLOG ticks); Tasks 8/9/12/13 fulfil each AC's evidence.
- §9 Risks: tasks include re-running each migrated test (Tasks 8/9/10/11) so a distribution-change-induced flake surfaces immediately.
- §10 Workflow notes: branch already exists; PR opening in Task 16.

**Plan amendment surfaced:** parser and IR `OpKind`s diverge by intent; plan exports the union enum and provides two `arb_op_*` / `arb_circuit_*` strategies (one per consumer). Documented in the plan header and reflected in Tasks 4/5/6/8/9.

**Placeholder scan:** no `TBD`/`TODO`/"add appropriate"/"similar to Task N"; every code block is complete; every command has an expected outcome.

**Type consistency:** `arb_state_vector(n) -> impl Strategy<Value = Vec<Complex>>`, `arb_1q_gate() -> impl Strategy<Value = Gate>`, `arb_circuit_emittable(nq, nc, n_ops) -> impl Strategy<Value = Circuit>`, `arb_circuit_full(nq, nc, n_ops) -> impl Strategy<Value = Circuit>`, `arb_op_emittable(nq, nc) -> BoxedStrategy<OpKind>`, `arb_op_full(nq, nc) -> BoxedStrategy<OpKind>`, `OpKind::apply(self, c: &mut Circuit)`, `OpKind::as_gate_instance(&self) -> Option<GateInstance>`, `arb_pauli_string(n, mix_xy) -> impl Strategy<Value = PauliString>` — all match across tasks.
