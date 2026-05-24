# P0-09 Backend Trait + Naive CPU State Vector Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `Backend` trait, shared `BackendError`, and a naive single-threaded CPU state-vector backend (`NaiveSvBackend`) that runs all Tier-1 algorithms at ≥ 20 qubits.

**Architecture:** Three crates collaborate. `aleph-core` gains a tiny `Pauli` / `PauliString` module. `aleph-backend` defines the `Backend` trait, the shared `BackendError`, and a `run<B>` driver. `aleph-sv` implements the trait with naive indexed-iteration kernels (1q / 2q / 3q) handling external controls via index masking. Measurement uses a per-backend `StdRng` seeded explicitly or from entropy.

**Tech Stack:** Rust 2021, MSRV 1.85, `num-complex`, `rand = "0.8"`, `thiserror`, `proptest`, `criterion`. Existing dependencies: `aleph-core` (Complex, Gate, GateMatrix, Param), `aleph-ir` (Circuit, Instruction, GateInstance), `aleph-parser` (parse) for integration tests.

**Spec:** `docs/superpowers/specs/2026-05-24-p0-09-backend-naive-sv-design.md`.

**Branch:** `p0-09-backend-naive-sv` (already created; spec already committed).

---

## File Structure

Files this plan creates or modifies:

| File | Responsibility |
|------|---------------|
| `Cargo.toml` | Add `rand = "0.8"` to `[workspace.dependencies]`. |
| `crates/aleph-core/src/pauli.rs` | NEW. `Pauli`, `PauliString`, `PauliError`. |
| `crates/aleph-core/src/lib.rs` | Add `pub mod pauli;` and re-exports. |
| `crates/aleph-backend/Cargo.toml` | Add `aleph-core`, `aleph-ir`, `thiserror` deps. |
| `crates/aleph-backend/src/lib.rs` | `Backend` trait, `BackendError`, `run<B>`. |
| `crates/aleph-sv/Cargo.toml` | Add deps + dev-deps + `[[bench]]`. |
| `crates/aleph-sv/src/lib.rs` | Module declarations + re-exports. |
| `crates/aleph-sv/src/state.rs` | `CpuState` struct + getters. |
| `crates/aleph-sv/src/backend.rs` | `NaiveSvBackend` struct + `impl Backend`. |
| `crates/aleph-sv/src/kernels.rs` | `apply_1q`, `apply_2q`, `apply_3q`. |
| `crates/aleph-sv/src/measure.rs` | `measure`, `sample`, `expectation_value`, `probabilities`. |
| `crates/aleph-sv/tests/tier1.rs` | Integration tests: GHZ, QFT, Grover, random via parser. |
| `crates/aleph-sv/benches/naive_sv.rs` | Criterion benchmark: H wall at n ∈ {10, 15, 20}. |

---

### Task 1: Add `rand` to workspace dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Update `[workspace.dependencies]`**

Edit `Cargo.toml`, find the `[workspace.dependencies]` block (after `nom_locate = "4"`) and add:

```toml
rand = "0.8"
```

After the change the block should read (relevant lines):

```toml
[workspace.dependencies]
num-complex = "0.4"
criterion = { version = "0.5", default-features = false, features = ["cargo_bench_support", "html_reports", "plotters"] }
smallvec = "1"
thiserror = "1"
proptest = "1"
nom = "7"
nom_locate = "4"
rand = "0.8"
```

- [ ] **Step 2: Verify the workspace still resolves**

Run: `cargo metadata --quiet > /dev/null`
Expected: exit 0 (no output).

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "[P0-09] Add rand 0.8 to workspace dependencies"
```

---

### Task 2: `aleph-core::pauli` — `Pauli` enum + `PauliString` + `PauliError`

**Files:**
- Create: `crates/aleph-core/src/pauli.rs`
- Modify: `crates/aleph-core/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/aleph-core/src/pauli.rs` with:

```rust
//! Pauli operators and Pauli strings — used by `Backend::expectation_value`.
//!
//! `Pauli` is the four single-qubit Pauli matrices `{I, X, Y, Z}`.
//! `PauliString` is a tensor product over named qubits with a real
//! coefficient, e.g. `0.5 · X₀ ⊗ Z₂`. Qubits not listed in `terms` are
//! implicit identity. `terms` is kept sorted by qubit and deduplicated.

use crate::Complex;

/// Single-qubit Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    /// 2×2 matrix in basis `|0⟩, |1⟩`.
    pub fn matrix(self) -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let i = Complex::new(0.0, 1.0);
        let neg_o = Complex::new(-1.0, 0.0);
        let neg_i = Complex::new(0.0, -1.0);
        match self {
            Pauli::I => [[o, z], [z, o]],
            Pauli::X => [[z, o], [o, z]],
            Pauli::Y => [[z, neg_i], [i, z]],
            Pauli::Z => [[o, z], [z, neg_o]],
        }
    }
}

/// A Pauli tensor-product with a real coefficient.
///
/// `terms` is sorted ascending by qubit index and contains no
/// duplicates. Construct via [`PauliString::new`] to enforce these
/// invariants, or [`PauliString::identity`] for the empty string.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliString {
    pub coefficient: f64,
    pub terms: Vec<(u32, Pauli)>,
}

impl PauliString {
    pub fn new(coefficient: f64, mut terms: Vec<(u32, Pauli)>) -> Result<Self, PauliError> {
        if !coefficient.is_finite() {
            return Err(PauliError::NonFiniteCoefficient);
        }
        terms.sort_by_key(|(q, _)| *q);
        for w in terms.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(PauliError::DuplicateQubit { qubit: w[0].0 });
            }
        }
        terms.retain(|(_, p)| *p != Pauli::I);
        Ok(Self { coefficient, terms })
    }

    pub fn identity(coefficient: f64) -> Self {
        Self {
            coefficient,
            terms: Vec::new(),
        }
    }
}

/// Errors from constructing a [`PauliString`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PauliError {
    #[error("duplicate qubit {qubit} in Pauli string")]
    DuplicateQubit { qubit: u32 },
    #[error("non-finite coefficient")]
    NonFiniteCoefficient,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_are_correct() {
        let m_x = Pauli::X.matrix();
        assert_eq!(m_x[0][1], Complex::new(1.0, 0.0));
        assert_eq!(m_x[1][0], Complex::new(1.0, 0.0));
        assert_eq!(m_x[0][0], Complex::new(0.0, 0.0));

        let m_y = Pauli::Y.matrix();
        assert_eq!(m_y[0][1], Complex::new(0.0, -1.0));
        assert_eq!(m_y[1][0], Complex::new(0.0, 1.0));

        let m_z = Pauli::Z.matrix();
        assert_eq!(m_z[0][0], Complex::new(1.0, 0.0));
        assert_eq!(m_z[1][1], Complex::new(-1.0, 0.0));
        assert_eq!(m_z[0][1], Complex::new(0.0, 0.0));

        let m_i = Pauli::I.matrix();
        assert_eq!(m_i[0][0], Complex::new(1.0, 0.0));
        assert_eq!(m_i[1][1], Complex::new(1.0, 0.0));
    }

    #[test]
    fn new_sorts_and_drops_identity() {
        let p = PauliString::new(
            1.0,
            vec![(2, Pauli::Z), (0, Pauli::X), (1, Pauli::I)],
        )
        .unwrap();
        assert_eq!(p.terms, vec![(0, Pauli::X), (2, Pauli::Z)]);
    }

    #[test]
    fn new_rejects_duplicates() {
        let err = PauliString::new(1.0, vec![(0, Pauli::X), (0, Pauli::Z)]).unwrap_err();
        assert_eq!(err, PauliError::DuplicateQubit { qubit: 0 });
    }

    #[test]
    fn new_rejects_non_finite_coefficient() {
        let err = PauliString::new(f64::NAN, vec![]).unwrap_err();
        assert_eq!(err, PauliError::NonFiniteCoefficient);
        let err = PauliString::new(f64::INFINITY, vec![]).unwrap_err();
        assert_eq!(err, PauliError::NonFiniteCoefficient);
    }

    #[test]
    fn identity_is_empty() {
        let p = PauliString::identity(0.5);
        assert!(p.terms.is_empty());
        assert_eq!(p.coefficient, 0.5);
    }
}
```

- [ ] **Step 2: Wire module into `lib.rs`**

Edit `crates/aleph-core/src/lib.rs`. After the line `pub mod gate;` (and its `pub use` line), add:

```rust
pub mod pauli;
pub use pauli::{Pauli, PauliError, PauliString};
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p aleph-core pauli`
Expected: 5 tests pass.

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-core --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-core/src/pauli.rs crates/aleph-core/src/lib.rs
git commit -m "[P0-09] aleph-core: add Pauli and PauliString"
```

---

### Task 3: `aleph-backend` — `BackendError` enum

**Files:**
- Modify: `crates/aleph-backend/Cargo.toml`
- Modify: `crates/aleph-backend/src/lib.rs`

- [ ] **Step 1: Update `Cargo.toml`**

Replace the contents of `crates/aleph-backend/Cargo.toml` with:

```toml
[package]
name = "aleph-backend"
edition.workspace = true
rust-version.workspace = true
version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]
aleph-core = { path = "../aleph-core" }
aleph-ir   = { path = "../aleph-ir" }
thiserror  = { workspace = true }
```

- [ ] **Step 2: Write `BackendError` and its smoke test in `lib.rs`**

Replace the contents of `crates/aleph-backend/src/lib.rs` with:

```rust
//! `aleph-backend`: the `Backend` trait, shared `BackendError`, and a
//! `run<B>` driver. Backend implementations live in `aleph-sv` (naive
//! CPU state vector), `aleph-mps`, `aleph-stab`, etc.
//!
//! See `docs/superpowers/specs/2026-05-24-p0-09-backend-naive-sv-design.md`.

/// Errors common to every backend.
///
/// Backends share one concrete error type rather than an associated
/// `type Error` so that the `run<B>` driver and downstream code (CLI,
/// Python bindings) don't have to be generic over an open-ended error.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BackendError {
    #[error("qubit {qubit} out of range for {num_qubits}-qubit state")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },

    #[error("duplicate qubit {qubit} in gate or query")]
    DuplicateQubit { qubit: u32 },

    #[error("circuit declares {circuit} qubits but state has {state}")]
    QubitCountMismatch { circuit: u32, state: u32 },

    #[error("gate `{kind}` is not supported by this backend")]
    UnsupportedGate { kind: &'static str },

    #[error("backend requires concrete parameters; got symbolic")]
    SymbolicParam,

    #[error("cannot run an empty circuit")]
    EmptyCircuit,

    #[error("measurement of qubit {qubit} on degenerate branch (p = {probability:e})")]
    DegenerateMeasurement { qubit: u32, probability: f64 },

    #[error("requested {requested} qubits exceeds backend limit of {limit}")]
    TooManyQubits { requested: u32, limit: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_render() {
        let e = BackendError::QubitOutOfRange {
            qubit: 7,
            num_qubits: 3,
        };
        assert_eq!(
            e.to_string(),
            "qubit 7 out of range for 3-qubit state"
        );

        let e = BackendError::DegenerateMeasurement {
            qubit: 0,
            probability: 1e-301,
        };
        // Format includes the scientific notation rendering.
        assert!(e.to_string().contains("p = 1e-301"));
    }
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p aleph-backend`
Expected: 1 test passes.

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-backend --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-backend/Cargo.toml crates/aleph-backend/src/lib.rs
git commit -m "[P0-09] aleph-backend: add BackendError"
```

---

### Task 4: `aleph-backend` — `Backend` trait

**Files:**
- Modify: `crates/aleph-backend/src/lib.rs`

- [ ] **Step 1: Append the trait to `lib.rs`**

Edit `crates/aleph-backend/src/lib.rs`. Insert the following block immediately **before** the `#[cfg(test)]` line at the bottom of the file:

```rust
use aleph_core::{GateInstance, PauliString};
use aleph_ir::Circuit;

/// A simulation backend.
///
/// Backends own no state vector; they construct and return one through
/// `allocate`, then mutate it in place via `apply_gate` / `measure`.
/// Query methods (`sample`, `expectation_value`, `probabilities`) take
/// `&Self::State` and do not mutate the state.
pub trait Backend {
    /// Backend-specific representation (state vector, MPS tensors,
    /// stabilizer tableau, …).
    type State;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError>;

    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError>;

    fn measure(
        &mut self,
        state: &mut Self::State,
        qubit: u32,
    ) -> Result<bool, BackendError>;

    fn sample(
        &mut self,
        state: &Self::State,
        shots: u32,
    ) -> Result<Vec<u64>, BackendError>;

    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError>;

    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError>;
}
```

- [ ] **Step 2: Verify the crate still builds**

Run: `cargo build -p aleph-backend`
Expected: success (no test added yet — trait is type-only).

- [ ] **Step 3: Lint and format**

Run: `cargo clippy -p aleph-backend --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-backend/src/lib.rs
git commit -m "[P0-09] aleph-backend: add Backend trait"
```

---

### Task 5: `aleph-backend` — `run<B>` driver

**Files:**
- Modify: `crates/aleph-backend/src/lib.rs`

- [ ] **Step 1: Append the function to `lib.rs`**

Edit `crates/aleph-backend/src/lib.rs`. Insert immediately after the `Backend` trait block (still before `#[cfg(test)]`):

```rust
/// Run `circuit` on `backend`, returning the final backend state.
///
/// Iterates instructions in order, dispatching `Instruction::Gate` to
/// `Backend::apply_gate`. Non-gate instructions are handled inline:
///
/// * `Measure { qubit, .. }` calls `Backend::measure` (and discards the
///   outcome — `run` is a state-producing driver, not a sampling one).
/// * `Reset(q)` is currently rejected as `UnsupportedGate { kind: "reset" }`
///   because the naive backend deals with mid-circuit reset via
///   measure-and-conditional-X, which the IR does not yet express
///   declaratively. P0-13+ may revisit.
/// * `Barrier(_)` is a no-op (semantic-only).
///
/// Returns `EmptyCircuit` only when the circuit declares zero qubits
/// **and** has zero instructions — the truly-degenerate input.
pub fn run<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<B::State, BackendError> {
    if circuit.num_qubits() == 0 && circuit.is_empty() {
        return Err(BackendError::EmptyCircuit);
    }
    let mut state = backend.allocate(circuit.num_qubits())?;
    for inst in circuit.instructions() {
        match inst {
            aleph_ir::Instruction::Gate(g) => backend.apply_gate(&mut state, g)?,
            aleph_ir::Instruction::Measure { qubit, .. } => {
                let _ = backend.measure(&mut state, *qubit)?;
            }
            aleph_ir::Instruction::Reset(_) => {
                return Err(BackendError::UnsupportedGate { kind: "reset" });
            }
            aleph_ir::Instruction::Barrier(_) => {}
        }
    }
    Ok(state)
}
```

- [ ] **Step 2: Verify the crate builds**

Run: `cargo build -p aleph-backend`
Expected: success.

- [ ] **Step 3: Lint and format**

Run: `cargo clippy -p aleph-backend --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-backend/src/lib.rs
git commit -m "[P0-09] aleph-backend: add run<B> driver"
```

---

### Task 6: `aleph-sv` — Cargo manifest + skeleton + `CpuState`

**Files:**
- Modify: `crates/aleph-sv/Cargo.toml`
- Modify: `crates/aleph-sv/src/lib.rs`
- Create: `crates/aleph-sv/src/state.rs`

- [ ] **Step 1: Update `Cargo.toml`**

Replace the contents of `crates/aleph-sv/Cargo.toml` with:

```toml
[package]
name = "aleph-sv"
edition.workspace = true
rust-version.workspace = true
version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]
aleph-core    = { path = "../aleph-core" }
aleph-ir      = { path = "../aleph-ir" }
aleph-backend = { path = "../aleph-backend" }
num-complex   = { workspace = true }
rand          = { workspace = true }
thiserror     = { workspace = true }

[dev-dependencies]
aleph-parser = { path = "../aleph-parser" }
proptest     = { workspace = true }
criterion    = { workspace = true }

[[bench]]
name = "naive_sv"
harness = false
```

- [ ] **Step 2: Create `state.rs`**

Create `crates/aleph-sv/src/state.rs` with:

```rust
//! `CpuState` — the dense `Vec<Complex>` of size 2^n used by the naive
//! CPU state-vector backend. Fields are private; consumers go through
//! the read-only getters.

use aleph_core::Complex;

/// State vector held by [`crate::NaiveSvBackend`].
///
/// Layout is array-of-structs (`Vec<Complex<f64>>`); `amps[i]` is the
/// amplitude of basis state `|i⟩` where bit `q` of `i` is the value of
/// qubit `q`. This is the textbook layout from Nielsen & Chuang §4.
#[derive(Debug, Clone)]
pub struct CpuState {
    pub(crate) num_qubits: u32,
    pub(crate) amps: Vec<Complex>,
}

impl CpuState {
    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Read-only view of the underlying amplitude buffer.
    pub fn amplitudes(&self) -> &[Complex] {
        &self.amps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn getters_match_construction() {
        let s = CpuState {
            num_qubits: 3,
            amps: vec![Complex::new(0.0, 0.0); 8],
        };
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.amplitudes().len(), 8);
    }
}
```

- [ ] **Step 3: Replace `lib.rs`**

Replace the contents of `crates/aleph-sv/src/lib.rs` with:

```rust
//! `aleph-sv`: naive single-threaded CPU state-vector backend.
//!
//! The reference implementation: simple, correct, and the yardstick
//! every other backend or future optimization is compared against. See
//! `docs/superpowers/specs/2026-05-24-p0-09-backend-naive-sv-design.md`.

mod backend;
mod kernels;
mod measure;
mod state;

pub use backend::NaiveSvBackend;
pub use state::CpuState;
```

Note: `backend`, `kernels`, and `measure` modules are declared here; their files land in subsequent tasks. After this step, the build will fail until Task 7 is in place — that's expected. Skip the build step in this task.

- [ ] **Step 4: Commit (without building)**

```bash
git add crates/aleph-sv/Cargo.toml crates/aleph-sv/src/lib.rs crates/aleph-sv/src/state.rs
git commit -m "[P0-09] aleph-sv: scaffold crate, deps, CpuState"
```

---

### Task 7: `aleph-sv::backend` — `NaiveSvBackend` struct + `allocate`

**Files:**
- Create: `crates/aleph-sv/src/backend.rs`
- Create (placeholder): `crates/aleph-sv/src/kernels.rs`
- Create (placeholder): `crates/aleph-sv/src/measure.rs`

- [ ] **Step 1: Create empty `kernels.rs` and `measure.rs`**

Create `crates/aleph-sv/src/kernels.rs` with:

```rust
//! Indexed gate application kernels. Filled in by P0-09 Tasks 8–10.
```

Create `crates/aleph-sv/src/measure.rs` with:

```rust
//! Measurement, sampling, expectation, marginals. Filled in by P0-09 Tasks 12–15.
```

- [ ] **Step 2: Write the failing tests**

Create `crates/aleph-sv/src/backend.rs` with:

```rust
//! `NaiveSvBackend` — the naive CPU state-vector backend.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateInstance, PauliString};
use rand::{rngs::StdRng, SeedableRng};

use crate::state::CpuState;

/// Soft cap on qubits. 2^28 × 16 bytes = 4 GiB, comfortable on a
/// 16 GiB development machine. Acceptance target is 20 qubits.
pub(crate) const MAX_NAIVE_QUBITS: u32 = 28;

/// Naive single-threaded CPU state-vector backend.
pub struct NaiveSvBackend {
    pub(crate) rng: StdRng,
}

impl NaiveSvBackend {
    /// Construct with an entropy-seeded RNG.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Construct with an explicit seed; runs are reproducible across
    /// processes and machines for a given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Default for NaiveSvBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for NaiveSvBackend {
    type State = CpuState;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_NAIVE_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: MAX_NAIVE_QUBITS,
            });
        }
        let dim = 1usize << num_qubits;
        let mut amps = vec![Complex::new(0.0, 0.0); dim];
        amps[0] = Complex::new(1.0, 0.0);
        Ok(CpuState { num_qubits, amps })
    }

    fn apply_gate(
        &mut self,
        _state: &mut Self::State,
        _gate: &GateInstance,
    ) -> Result<(), BackendError> {
        unimplemented!("apply_gate lands in P0-09 Task 11")
    }

    fn measure(
        &mut self,
        _state: &mut Self::State,
        _qubit: u32,
    ) -> Result<bool, BackendError> {
        unimplemented!("measure lands in P0-09 Task 12")
    }

    fn sample(
        &mut self,
        _state: &Self::State,
        _shots: u32,
    ) -> Result<Vec<u64>, BackendError> {
        unimplemented!("sample lands in P0-09 Task 13")
    }

    fn expectation_value(
        &mut self,
        _state: &Self::State,
        _pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        unimplemented!("expectation_value lands in P0-09 Task 15")
    }

    fn probabilities(
        &mut self,
        _state: &Self::State,
        _qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        unimplemented!("probabilities lands in P0-09 Task 14")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_initialises_zero_state() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(3).unwrap();
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.amplitudes().len(), 8);
        assert_eq!(s.amplitudes()[0], Complex::new(1.0, 0.0));
        for a in &s.amplitudes()[1..] {
            assert_eq!(*a, Complex::new(0.0, 0.0));
        }
    }

    #[test]
    fn allocate_rejects_too_many_qubits() {
        let mut b = NaiveSvBackend::with_seed(0);
        let err = b.allocate(MAX_NAIVE_QUBITS + 1).unwrap_err();
        assert_eq!(
            err,
            BackendError::TooManyQubits {
                requested: MAX_NAIVE_QUBITS + 1,
                limit: MAX_NAIVE_QUBITS,
            }
        );
    }

    #[test]
    fn allocate_zero_qubits_yields_unit_amplitude() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(0).unwrap();
        assert_eq!(s.amplitudes(), &[Complex::new(1.0, 0.0)]);
    }
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p aleph-sv backend::tests`
Expected: 3 tests pass.

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/backend.rs crates/aleph-sv/src/kernels.rs crates/aleph-sv/src/measure.rs
git commit -m "[P0-09] aleph-sv: NaiveSvBackend skeleton with allocate"
```

---

### Task 8: `aleph-sv::kernels::apply_1q`

**Files:**
- Modify: `crates/aleph-sv/src/kernels.rs`

- [ ] **Step 1: Write the failing tests**

Replace the contents of `crates/aleph-sv/src/kernels.rs` with:

```rust
//! Indexed gate application kernels.
//!
//! Convention from the P0-06 spec: `qubits[0]` is the LSB of the
//! matrix index. For a 2-qubit gate on `[a, b]`, basis order is
//! `|b a⟩` — i.e. matrix row/col `k` corresponds to `(a, b) =
//! (k & 1, (k >> 1) & 1)`. Same generalization for 3-qubit gates.

use aleph_core::Complex;

/// Apply a 1-qubit matrix to `target` (possibly with external
/// `controls`) in place.
///
/// Iterates the 2^(n-1) basis indices whose `target` bit is zero;
/// each defines a 2-element subspace `(i, i | t_bit)`. Skips
/// iterations whose `i` does not have every control bit set.
pub fn apply_1q(
    amps: &mut [Complex],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    let t_bit = 1usize << target;
    let mut ctrl_mask: usize = 0;
    for &c in controls {
        ctrl_mask |= 1usize << c;
    }
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let a = amps[i];
            let b = amps[j];
            amps[i] = m[0][0] * a + m[0][1] * b;
            amps[j] = m[1][0] * a + m[1][1] * b;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pauli_x() -> [[Complex; 2]; 2] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[z, o], [o, z]]
    }

    fn hadamard() -> [[Complex; 2]; 2] {
        let s = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        [[s, s], [s, -s]]
    }

    #[test]
    fn x_flips_single_qubit() {
        let mut amps = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        apply_1q(&mut amps, 0, &[], &pauli_x());
        assert_eq!(amps[0], Complex::new(0.0, 0.0));
        assert_eq!(amps[1], Complex::new(1.0, 0.0));
    }

    #[test]
    fn h_on_zero_yields_plus() {
        let mut amps = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        apply_1q(&mut amps, 0, &[], &hadamard());
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amps[0].re - s).abs() < 1e-12);
        assert!((amps[1].re - s).abs() < 1e-12);
    }

    #[test]
    fn x_on_target_1_in_2q_state() {
        // |10⟩ in our convention (q0 = LSB, q1 = MSB) is index 2.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[2] = Complex::new(1.0, 0.0);
        apply_1q(&mut amps, 1, &[], &pauli_x());
        // X on q1 sends |10⟩ → |00⟩, index 2 → index 0.
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        assert_eq!(amps[2], Complex::new(0.0, 0.0));
    }

    #[test]
    fn controls_skip_when_unset() {
        // 2-qubit state |01⟩: q0 = 1, q1 = 0 → index 1.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[1] = Complex::new(1.0, 0.0);
        // Apply CX with control q0, target q1: should flip q1, send → |11⟩ = index 3.
        apply_1q(&mut amps, 1, &[0], &pauli_x());
        assert_eq!(amps[3], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
    }

    #[test]
    fn controls_do_nothing_when_control_zero() {
        // |00⟩ = index 0. CX (c=q0, t=q1) leaves it unchanged.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[0] = Complex::new(1.0, 0.0);
        apply_1q(&mut amps, 1, &[0], &pauli_x());
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p aleph-sv kernels::tests`
Expected: 5 tests pass.

- [ ] **Step 3: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-sv/src/kernels.rs
git commit -m "[P0-09] aleph-sv: apply_1q kernel"
```

---

### Task 9: `aleph-sv::kernels::apply_2q`

**Files:**
- Modify: `crates/aleph-sv/src/kernels.rs`

- [ ] **Step 1: Append `apply_2q` to `kernels.rs`**

Add the following function **before** the `#[cfg(test)]` block in `crates/aleph-sv/src/kernels.rs`:

```rust
/// Apply a 2-qubit matrix to `targets = [t0, t1]` (with external
/// `controls`) in place.
///
/// **MSB convention (P0-06):** `targets[0]` is the *high* bit of the
/// matrix index `k`, `targets[1]` is the *low* bit. So matrix row 2
/// (binary `10`) corresponds to `(targets[0] = 1, targets[1] = 0)`.
/// This matches `Gate::Cnot` (`qubits = [control, target]`), whose
/// matrix swaps rows 2 ↔ 3.
///
/// Targets must be distinct; the caller (`apply_gate`) enforces this.
pub fn apply_2q(
    amps: &mut [Complex],
    targets: [u32; 2],
    controls: &[u32],
    m: &[[Complex; 4]; 4],
) {
    let t0_bit = 1usize << targets[0];
    let t1_bit = 1usize << targets[1];
    let t_mask = t0_bit | t1_bit;
    let mut ctrl_mask: usize = 0;
    for &c in controls {
        ctrl_mask |= 1usize << c;
    }
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            // MSB convention: matrix index k bit 1 → targets[0], bit 0 → targets[1].
            // So idx[k] sets t0_bit iff (k & 2) != 0, t1_bit iff (k & 1) != 0.
            let idx = [
                i,                 // k = 00
                i | t1_bit,        // k = 01
                i | t0_bit,        // k = 10
                i | t_mask,        // k = 11
            ];
            let v = [
                amps[idx[0]],
                amps[idx[1]],
                amps[idx[2]],
                amps[idx[3]],
            ];
            for r in 0..4 {
                amps[idx[r]] = m[r][0] * v[0]
                    + m[r][1] * v[1]
                    + m[r][2] * v[2]
                    + m[r][3] * v[3];
            }
        }
        i += 1;
    }
}
```

- [ ] **Step 2: Add tests inside the existing `#[cfg(test)] mod tests` block**

Append the following test functions (and helpers if needed) to the existing `#[cfg(test)] mod tests` block in `crates/aleph-sv/src/kernels.rs`:

```rust
    /// Canonical `Gate::Cnot` matrix (P0-06):
    /// swaps rows 2 ↔ 3 with `qubits = [control, target]` and
    /// control = MSB of the matrix index.
    fn cnot() -> [[Complex; 4]; 4] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, z, o],
            [z, z, o, z],
        ]
    }

    #[test]
    fn cnot_flips_target_when_control_set() {
        // Targets = [q0, q1]. State amps[1] = 1 corresponds to
        // (q0 = 1, q1 = 0) in the global state vector.
        // With MSB convention idx = [0, t1_bit, t0_bit, t_mask] = [0, 2, 1, 3],
        // amps[1] sits at matrix slot k = 2 (control set, target clear).
        // Cnot swaps slot 2 ↔ 3 ⇒ amps[1] moves to amps[3].
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[1] = Complex::new(1.0, 0.0);
        apply_2q(&mut amps, [0, 1], &[], &cnot());
        assert_eq!(amps[3], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
    }

    #[test]
    fn cnot_on_zero_state_unchanged() {
        // amps[0] = 1 (control = 0, target = 0) — Cnot leaves it alone.
        let mut amps = vec![Complex::new(0.0, 0.0); 4];
        amps[0] = Complex::new(1.0, 0.0);
        apply_2q(&mut amps, [0, 1], &[], &cnot());
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn apply_2q_external_control_skips_when_unset() {
        // 3 qubits, state amps[1] = 1 (q0 = 1, q1 = 0, q2 = 0).
        // Apply Cnot (q0 = ctrl, q1 = tgt) externally controlled by q2.
        // Since q2 = 0, gate should NOT fire.
        let mut amps = vec![Complex::new(0.0, 0.0); 8];
        amps[1] = Complex::new(1.0, 0.0);
        apply_2q(&mut amps, [0, 1], &[2], &cnot());
        assert_eq!(amps[1], Complex::new(1.0, 0.0));
    }
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p aleph-sv kernels::tests`
Expected: 8 tests pass total (5 from Task 8 + 3 new).

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels.rs
git commit -m "[P0-09] aleph-sv: apply_2q kernel"
```

---

### Task 10: `aleph-sv::kernels::apply_3q`

**Files:**
- Modify: `crates/aleph-sv/src/kernels.rs`

- [ ] **Step 1: Append `apply_3q` to `kernels.rs`**

Add the following function before the `#[cfg(test)]` block:

```rust
/// Apply a 3-qubit matrix to `targets = [t0, t1, t2]` (with external
/// `controls`) in place.
///
/// **MSB convention (P0-06):** matrix index `k`'s bits map to targets
/// from MSB to LSB — bit 2 of `k` is `targets[0]`, bit 1 is
/// `targets[1]`, bit 0 is `targets[2]`. So `k = 6` (binary `110`)
/// corresponds to `(targets[0] = 1, targets[1] = 1, targets[2] = 0)`.
/// This matches `Gate::Toffoli` (`qubits = [c0, c1, target]`), whose
/// matrix swaps rows 6 ↔ 7.
pub fn apply_3q(
    amps: &mut [Complex],
    targets: [u32; 3],
    controls: &[u32],
    m: &[[Complex; 8]; 8],
) {
    let t_bits = [
        1usize << targets[0],
        1usize << targets[1],
        1usize << targets[2],
    ];
    let t_mask = t_bits[0] | t_bits[1] | t_bits[2];
    let mut ctrl_mask: usize = 0;
    for &c in controls {
        ctrl_mask |= 1usize << c;
    }
    let len = amps.len();
    let mut i = 0usize;
    while i < len {
        if (i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask {
            let mut idx = [0usize; 8];
            for (k, slot) in idx.iter_mut().enumerate() {
                // MSB convention: k bit 2 → targets[0], bit 1 → targets[1], bit 0 → targets[2].
                let bit_t0 = if k & 4 != 0 { t_bits[0] } else { 0 };
                let bit_t1 = if k & 2 != 0 { t_bits[1] } else { 0 };
                let bit_t2 = if k & 1 != 0 { t_bits[2] } else { 0 };
                *slot = i | bit_t0 | bit_t1 | bit_t2;
            }
            let v = [
                amps[idx[0]], amps[idx[1]], amps[idx[2]], amps[idx[3]],
                amps[idx[4]], amps[idx[5]], amps[idx[6]], amps[idx[7]],
            ];
            for r in 0..8 {
                let mut acc = Complex::new(0.0, 0.0);
                for c in 0..8 {
                    acc += m[r][c] * v[c];
                }
                amps[idx[r]] = acc;
            }
        }
        i += 1;
    }
}
```

- [ ] **Step 2: Add tests to the existing `#[cfg(test)] mod tests` block**

Append:

```rust
    /// Canonical `Gate::Toffoli` matrix (P0-06): identity on rows 0..6,
    /// swap rows 6 ↔ 7. Matches `qubits = [c0, c1, target]` with
    /// `qubits[0]` as the MSB of the matrix index.
    fn toffoli() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for (i, row) in m.iter_mut().enumerate().take(6) {
            row[i] = o;
        }
        m[6][7] = o;
        m[7][6] = o;
        m
    }

    #[test]
    fn toffoli_flips_target_when_both_controls_set() {
        // Targets = [q0, q1, q2]. State amps[3] = 1 corresponds to
        // (q0 = 1, q1 = 1, q2 = 0) globally. With MSB convention this
        // maps to matrix slot k = 6 (bit 2 = q0 = 1, bit 1 = q1 = 1,
        // bit 0 = q2 = 0). Toffoli swaps slot 6 ↔ 7 ⇒ amps[3] → amps[7].
        let mut amps = vec![Complex::new(0.0, 0.0); 8];
        amps[3] = Complex::new(1.0, 0.0);
        apply_3q(&mut amps, [0, 1, 2], &[], &toffoli());
        assert_eq!(amps[7], Complex::new(1.0, 0.0));
        assert_eq!(amps[3], Complex::new(0.0, 0.0));
    }

    #[test]
    fn toffoli_with_single_control_set_is_identity() {
        // State amps[1] = 1 (q0 = 1, q1 = 0, q2 = 0). Only one control
        // bit set ⇒ Toffoli acts as identity.
        let mut amps = vec![Complex::new(0.0, 0.0); 8];
        amps[1] = Complex::new(1.0, 0.0);
        apply_3q(&mut amps, [0, 1, 2], &[], &toffoli());
        assert_eq!(amps[1], Complex::new(1.0, 0.0));
    }
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p aleph-sv kernels::tests`
Expected: 10 tests pass total.

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels.rs
git commit -m "[P0-09] aleph-sv: apply_3q kernel"
```

---

### Task 11: `aleph-sv` — `apply_gate` dispatch

**Files:**
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Replace the `apply_gate` stub**

Edit `crates/aleph-sv/src/backend.rs`. Replace the existing `fn apply_gate(...) { unimplemented!(...) }` method body with:

```rust
    fn apply_gate(
        &mut self,
        state: &mut Self::State,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let n = state.num_qubits;
        // Bounds + duplicate checks across qubits ∪ controls.
        let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for &q in gate.qubits.iter().chain(gate.controls.iter()) {
            if q >= n {
                return Err(BackendError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
            if seen.contains(&q) {
                return Err(BackendError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        // Materialise the matrix; map symbolic-param errors to BackendError.
        let matrix = gate
            .gate
            .matrix()
            .map_err(|_| BackendError::SymbolicParam)?;
        match matrix {
            aleph_core::GateMatrix::M2x2(m) => {
                let t = gate.qubits[0];
                crate::kernels::apply_1q(&mut state.amps, t, &gate.controls, &m);
            }
            aleph_core::GateMatrix::M4x4(m) => {
                let t = [gate.qubits[0], gate.qubits[1]];
                crate::kernels::apply_2q(&mut state.amps, t, &gate.controls, &m);
            }
            aleph_core::GateMatrix::M8x8(m) => {
                let t = [gate.qubits[0], gate.qubits[1], gate.qubits[2]];
                crate::kernels::apply_3q(&mut state.amps, t, &gate.controls, &m);
            }
        }
        Ok(())
    }
```

Then add `smallvec` to `aleph-sv`'s dependencies in `crates/aleph-sv/Cargo.toml` — under `[dependencies]`, append:

```toml
smallvec      = { workspace = true }
```

- [ ] **Step 2: Add `apply_gate` tests**

Append the following tests to the existing `#[cfg(test)] mod tests` block of `backend.rs`:

```rust
    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn apply_gate_x_on_q0() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let gate = GateInstance::new(Gate::X, smallvec![0u32]);
        b.apply_gate(&mut s, &gate).unwrap();
        assert_eq!(s.amplitudes()[0], Complex::new(0.0, 0.0));
        assert_eq!(s.amplitudes()[1], Complex::new(1.0, 0.0));
    }

    #[test]
    fn apply_gate_cnot_creates_bell() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        // H on q0, then CX(q0 → q1).
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
        let a = s.amplitudes();
        assert!((a[0].re - inv_s2).abs() < 1e-12);
        assert!((a[3].re - inv_s2).abs() < 1e-12);
        assert!(a[1].norm_sqr() < 1e-24);
        assert!(a[2].norm_sqr() < 1e-24);
    }

    #[test]
    fn apply_gate_external_control_matches_intrinsic_cnot() {
        // Path A: intrinsic CX.
        let mut b1 = NaiveSvBackend::with_seed(0);
        let mut s1 = b1.allocate(2).unwrap();
        b1.apply_gate(&mut s1, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b1.apply_gate(
            &mut s1,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        // Path B: X on q1 with external control = q0.
        let mut b2 = NaiveSvBackend::with_seed(0);
        let mut s2 = b2.allocate(2).unwrap();
        b2.apply_gate(&mut s2, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b2.apply_gate(
            &mut s2,
            &GateInstance::controlled(Gate::X, smallvec![1u32], smallvec![0u32]),
        )
        .unwrap();
        for (a, b) in s1.amplitudes().iter().zip(s2.amplitudes().iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn apply_gate_out_of_range() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let gate = GateInstance::new(Gate::X, smallvec![5u32]);
        let err = b.apply_gate(&mut s, &gate).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 1,
            }
        );
    }

    #[test]
    fn apply_gate_duplicate_qubit_via_controls() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(2).unwrap();
        let gate = GateInstance::controlled(Gate::X, smallvec![0u32], smallvec![0u32]);
        let err = b.apply_gate(&mut s, &gate).unwrap_err();
        assert_eq!(err, BackendError::DuplicateQubit { qubit: 0 });
    }
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p aleph-sv`
Expected: all backend + kernel tests pass.

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/backend.rs crates/aleph-sv/Cargo.toml
git commit -m "[P0-09] aleph-sv: apply_gate dispatch"
```

---

### Task 12: `aleph-sv::measure` — `measure`

**Files:**
- Modify: `crates/aleph-sv/src/measure.rs`
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Implement `measure_impl` in `measure.rs`**

Replace the contents of `crates/aleph-sv/src/measure.rs` with:

```rust
//! Measurement, sampling, expectation, marginals.

use aleph_backend::BackendError;
use aleph_core::Complex;
use rand::{rngs::StdRng, Rng};

use crate::state::CpuState;

/// Threshold under which we refuse to collapse the state — collapsing
/// on a branch of probability `< 1e-300` would scale amplitudes by
/// `≈ 1e150` and destroy any meaningful state.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;

pub(crate) fn measure_impl(
    rng: &mut StdRng,
    state: &mut CpuState,
    qubit: u32,
) -> Result<bool, BackendError> {
    let n = state.num_qubits;
    if qubit >= n {
        return Err(BackendError::QubitOutOfRange {
            qubit,
            num_qubits: n,
        });
    }
    let q_bit = 1usize << qubit;
    let mut p1 = 0.0_f64;
    for (i, a) in state.amps.iter().enumerate() {
        if i & q_bit != 0 {
            p1 += a.norm_sqr();
        }
    }
    let outcome: bool = rng.gen::<f64>() < p1;
    let p = if outcome { p1 } else { 1.0 - p1 };
    if p < DEGENERATE_BRANCH_THRESHOLD {
        return Err(BackendError::DegenerateMeasurement {
            qubit,
            probability: p,
        });
    }
    let norm = p.sqrt();
    for (i, a) in state.amps.iter_mut().enumerate() {
        let bit_set = (i & q_bit) != 0;
        if bit_set == outcome {
            *a /= Complex::new(norm, 0.0);
        } else {
            *a = Complex::new(0.0, 0.0);
        }
    }
    Ok(outcome)
}
```

- [ ] **Step 2: Wire `measure` in `backend.rs`**

In `crates/aleph-sv/src/backend.rs`, replace the existing `fn measure(...) { unimplemented!(...) }` method body with:

```rust
    fn measure(
        &mut self,
        state: &mut Self::State,
        qubit: u32,
    ) -> Result<bool, BackendError> {
        crate::measure::measure_impl(&mut self.rng, state, qubit)
    }
```

- [ ] **Step 3: Add tests to `backend.rs`**

Append to the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn measure_zero_state_returns_false() {
        let mut b = NaiveSvBackend::with_seed(42);
        let mut s = b.allocate(2).unwrap();
        let outcome = b.measure(&mut s, 0).unwrap();
        assert!(!outcome);
        assert_eq!(s.amplitudes()[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn measure_plus_state_collapses_to_basis() {
        let mut b = NaiveSvBackend::with_seed(123);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let outcome = b.measure(&mut s, 0).unwrap();
        // Post-collapse: amplitude 1 in the matching basis index, 0 elsewhere.
        let a = s.amplitudes();
        if outcome {
            assert!((a[1].norm() - 1.0).abs() < 1e-12);
            assert!(a[0].norm() < 1e-12);
        } else {
            assert!((a[0].norm() - 1.0).abs() < 1e-12);
            assert!(a[1].norm() < 1e-12);
        }
    }

    #[test]
    fn measure_qubit_out_of_range() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let err = b.measure(&mut s, 5).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 1,
            }
        );
    }
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-sv`
Expected: all tests pass (kernels + backend + state).

- [ ] **Step 5: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/measure.rs crates/aleph-sv/src/backend.rs
git commit -m "[P0-09] aleph-sv: measure"
```

---

### Task 13: `aleph-sv::measure` — `sample`

**Files:**
- Modify: `crates/aleph-sv/src/measure.rs`
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Add `sample_impl` to `measure.rs`**

Append to `crates/aleph-sv/src/measure.rs`:

```rust
/// Sample basis-state indices from `|amps[i]|²` via inverse-CDF.
///
/// Builds the CDF once, then binary-searches per shot. CDF is clamped
/// at 1.0 at the last index to absorb floating-point drift; a shot
/// with `u == 1.0` (rare but possible) maps to the last basis index.
pub(crate) fn sample_impl(
    rng: &mut StdRng,
    state: &CpuState,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let n = state.amps.len();
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0_f64;
    for a in &state.amps {
        acc += a.norm_sqr();
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        let u: f64 = rng.gen();
        let idx = cdf.partition_point(|&c| c < u);
        let idx = idx.min(n.saturating_sub(1));
        out.push(idx as u64);
    }
    Ok(out)
}
```

- [ ] **Step 2: Wire `sample` in `backend.rs`**

Replace the `fn sample(...) { unimplemented!(...) }` body with:

```rust
    fn sample(
        &mut self,
        state: &Self::State,
        shots: u32,
    ) -> Result<Vec<u64>, BackendError> {
        crate::measure::sample_impl(&mut self.rng, state, shots)
    }
```

- [ ] **Step 3: Add tests to `backend.rs`**

Append:

```rust
    #[test]
    fn sample_zero_state_only_returns_zero() {
        let mut b = NaiveSvBackend::with_seed(7);
        let s = b.allocate(3).unwrap();
        let shots = b.sample(&s, 100).unwrap();
        assert_eq!(shots.len(), 100);
        assert!(shots.iter().all(|&v| v == 0));
    }

    #[test]
    fn sample_bell_state_only_returns_00_or_11() {
        let mut b = NaiveSvBackend::with_seed(7);
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        let shots = b.sample(&s, 1000).unwrap();
        assert!(shots.iter().all(|&v| v == 0 || v == 3));
        // Sanity: both outcomes should appear in 1000 shots with overwhelming probability.
        let zeros = shots.iter().filter(|&&v| v == 0).count();
        let threes = shots.iter().filter(|&&v| v == 3).count();
        assert!(zeros > 100 && threes > 100, "zeros={zeros}, threes={threes}");
    }

    #[test]
    fn sample_zero_shots_returns_empty() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        assert!(b.sample(&s, 0).unwrap().is_empty());
    }
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-sv`
Expected: all tests pass.

- [ ] **Step 5: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/measure.rs crates/aleph-sv/src/backend.rs
git commit -m "[P0-09] aleph-sv: sample"
```

---

### Task 14: `aleph-sv::measure` — `probabilities`

**Files:**
- Modify: `crates/aleph-sv/src/measure.rs`
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Add `probabilities_impl` to `measure.rs`**

Append to `crates/aleph-sv/src/measure.rs`:

```rust
/// Marginal probabilities over the named qubit subset.
///
/// Returns a vector of length `2^qubits.len()`. The output is indexed
/// with `qubits[0]` as LSB to match the global gate-ordering convention.
pub(crate) fn probabilities_impl(
    state: &CpuState,
    qubits: &[u32],
) -> Result<Vec<f64>, BackendError> {
    let n = state.num_qubits;
    let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
    for &q in qubits {
        if q >= n {
            return Err(BackendError::QubitOutOfRange {
                qubit: q,
                num_qubits: n,
            });
        }
        if seen.contains(&q) {
            return Err(BackendError::DuplicateQubit { qubit: q });
        }
        seen.push(q);
    }
    if qubits.is_empty() {
        return Ok(vec![1.0]);
    }
    let out_dim = 1usize << qubits.len();
    let mut out = vec![0.0_f64; out_dim];
    for (i, a) in state.amps.iter().enumerate() {
        let mut k = 0usize;
        for (pos, &q) in qubits.iter().enumerate() {
            if (i >> q) & 1 == 1 {
                k |= 1usize << pos;
            }
        }
        out[k] += a.norm_sqr();
    }
    Ok(out)
}
```

- [ ] **Step 2: Wire `probabilities` in `backend.rs`**

Replace the `fn probabilities(...) { unimplemented!(...) }` body with:

```rust
    fn probabilities(
        &mut self,
        state: &Self::State,
        qubits: &[u32],
    ) -> Result<Vec<f64>, BackendError> {
        crate::measure::probabilities_impl(state, qubits)
    }
```

- [ ] **Step 3: Add tests to `backend.rs`**

Append:

```rust
    #[test]
    fn probabilities_zero_state() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        assert_eq!(b.probabilities(&s, &[0, 1]).unwrap(), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn probabilities_plus_state_uniform_marginal() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let p = b.probabilities(&s, &[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn probabilities_empty_subset_is_one() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(3).unwrap();
        assert_eq!(b.probabilities(&s, &[]).unwrap(), vec![1.0]);
    }

    #[test]
    fn probabilities_duplicate_qubit_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        let err = b.probabilities(&s, &[0, 0]).unwrap_err();
        assert_eq!(err, BackendError::DuplicateQubit { qubit: 0 });
    }

    #[test]
    fn probabilities_out_of_range_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        let err = b.probabilities(&s, &[5]).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 2,
            }
        );
    }
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-sv`
Expected: all tests pass.

- [ ] **Step 5: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/measure.rs crates/aleph-sv/src/backend.rs
git commit -m "[P0-09] aleph-sv: probabilities"
```

---

### Task 15: `aleph-sv::measure` — `expectation_value`

**Files:**
- Modify: `crates/aleph-sv/src/measure.rs`
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Add `expectation_value_impl` to `measure.rs`**

Append to `crates/aleph-sv/src/measure.rs`:

```rust
/// Naive expectation value: copy state, apply each non-identity Pauli
/// as a 1q gate to the copy, then take `Re(⟨ψ|φ⟩)`.
///
/// O(N · k) where N = 2^n and k = `pauli.terms.len()`. P0-11 will add
/// the Pauli-Z fast path that doesn't need a copy.
pub(crate) fn expectation_value_impl(
    state: &CpuState,
    pauli: &aleph_core::PauliString,
) -> Result<f64, BackendError> {
    let n = state.num_qubits;
    for (q, _) in &pauli.terms {
        if *q >= n {
            return Err(BackendError::QubitOutOfRange {
                qubit: *q,
                num_qubits: n,
            });
        }
    }
    let mut tmp = state.amps.clone();
    for (q, p) in &pauli.terms {
        if *p == aleph_core::Pauli::I {
            continue;
        }
        let m = p.matrix();
        crate::kernels::apply_1q(&mut tmp, *q, &[], &m);
    }
    let mut acc = Complex::new(0.0, 0.0);
    for (lhs, rhs) in state.amps.iter().zip(tmp.iter()) {
        acc += lhs.conj() * (*rhs);
    }
    Ok(pauli.coefficient * acc.re)
}
```

- [ ] **Step 2: Wire `expectation_value` in `backend.rs`**

Replace the `fn expectation_value(...) { unimplemented!(...) }` body with:

```rust
    fn expectation_value(
        &mut self,
        state: &Self::State,
        pauli: &PauliString,
    ) -> Result<f64, BackendError> {
        crate::measure::expectation_value_impl(state, pauli)
    }
```

- [ ] **Step 3: Add tests to `backend.rs`**

Append:

```rust
    use aleph_core::{Pauli, PauliString};

    #[test]
    fn expectation_z_on_zero_is_plus_one() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        let ev = b.expectation_value(&s, &z).unwrap();
        assert!((ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_x_on_plus_is_plus_one() {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        let ev = b.expectation_value(&s, &x).unwrap();
        assert!((ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_x_on_zero_is_zero() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        let ev = b.expectation_value(&s, &x).unwrap();
        assert!(ev.abs() < 1e-12);
    }

    #[test]
    fn expectation_z_on_minus_is_minus_one() {
        // |−⟩ = HZ|0⟩.
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![0u32]))
            .unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        let ev = b.expectation_value(&s, &z).unwrap();
        assert!((ev - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn expectation_identity_string_is_norm() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(2).unwrap();
        let ev = b
            .expectation_value(&s, &PauliString::identity(2.5))
            .unwrap();
        assert!((ev - 2.5).abs() < 1e-12);
    }

    #[test]
    fn expectation_out_of_range_rejected() {
        let mut b = NaiveSvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let p = PauliString::new(1.0, vec![(5, Pauli::Z)]).unwrap();
        let err = b.expectation_value(&s, &p).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 1,
            }
        );
    }
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-sv`
Expected: all tests pass.

- [ ] **Step 5: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/measure.rs crates/aleph-sv/src/backend.rs
git commit -m "[P0-09] aleph-sv: expectation_value"
```

---

### Task 16: Property tests — normalisation, reversibility, involution, control equivalence

**Files:**
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Add property tests inside the existing `mod tests`**

Append at the bottom of the existing `#[cfg(test)] mod tests` block in `crates/aleph-sv/src/backend.rs`:

```rust
    use proptest::prelude::*;

    fn random_1q_gate_strategy() -> impl Strategy<Value = Gate> {
        prop_oneof![
            Just(Gate::H),
            Just(Gate::X),
            Just(Gate::Y),
            Just(Gate::Z),
            Just(Gate::S),
            Just(Gate::Sdg),
            Just(Gate::T),
            Just(Gate::Tdg),
            (-6.28318_f64..=6.28318_f64).prop_map(|t| Gate::Rx(t.into())),
            (-6.28318_f64..=6.28318_f64).prop_map(|t| Gate::Ry(t.into())),
            (-6.28318_f64..=6.28318_f64).prop_map(|t| Gate::Rz(t.into())),
        ]
    }

    fn run_program(
        ops: &[(Gate, u32)],
        n: u32,
    ) -> CpuState {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(n).unwrap();
        for (g, q) in ops {
            b.apply_gate(&mut s, &GateInstance::new(g.clone(), smallvec![*q]))
                .unwrap();
        }
        s
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn normalisation_invariant(
            ops in proptest::collection::vec(
                (random_1q_gate_strategy(), 0u32..4u32),
                0..30,
            )
        ) {
            let s = run_program(&ops, 4);
            let total: f64 = s.amplitudes().iter().map(|a| a.norm_sqr()).sum();
            prop_assert!((total - 1.0).abs() < 1e-10, "norm² = {total}");
        }

        #[test]
        fn h_is_involution(q in 0u32..4u32) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(4).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            prop_assert!((s.amplitudes()[0].re - 1.0).abs() < 1e-12);
            for a in &s.amplitudes()[1..] {
                prop_assert!(a.norm() < 1e-12);
            }
        }

        #[test]
        fn cnot_is_involution(c in 0u32..3u32, t in 0u32..3u32) {
            prop_assume!(c != t);
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            // Put system in a superposition first so the test exercises non-trivial state.
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![c])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![c, t])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![c, t])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-12);
            }
        }

        #[test]
        fn rx_then_rx_negative_returns_identity(q in 0u32..3u32, theta in -3.0_f64..3.0_f64) {
            let mut b = NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(3).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q])).unwrap();
            let before: Vec<Complex> = s.amplitudes().to_vec();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Rx(theta.into()), smallvec![q])).unwrap();
            b.apply_gate(&mut s, &GateInstance::new(Gate::Rx((-theta).into()), smallvec![q])).unwrap();
            for (x, y) in before.iter().zip(s.amplitudes().iter()) {
                prop_assert!((x - y).norm() < 1e-10);
            }
        }

        #[test]
        fn intrinsic_cnot_matches_external_control(
            c in 0u32..3u32,
            t in 0u32..3u32,
            preamble_q in 0u32..3u32,
        ) {
            prop_assume!(c != t);
            // Path A: H on preamble, then intrinsic CX.
            let mut b1 = NaiveSvBackend::with_seed(0);
            let mut s1 = b1.allocate(3).unwrap();
            b1.apply_gate(&mut s1, &GateInstance::new(Gate::H, smallvec![preamble_q])).unwrap();
            b1.apply_gate(&mut s1, &GateInstance::new(Gate::Cnot, smallvec![c, t])).unwrap();
            // Path B: same preamble, then X on t with external control = c.
            let mut b2 = NaiveSvBackend::with_seed(0);
            let mut s2 = b2.allocate(3).unwrap();
            b2.apply_gate(&mut s2, &GateInstance::new(Gate::H, smallvec![preamble_q])).unwrap();
            b2.apply_gate(
                &mut s2,
                &GateInstance::controlled(Gate::X, smallvec![t], smallvec![c]),
            ).unwrap();
            for (a, b) in s1.amplitudes().iter().zip(s2.amplitudes().iter()) {
                prop_assert!((a - b).norm() < 1e-12);
            }
        }
    }
```

Note: this block uses `Param::from(f64)` via `.into()` (already in `aleph-core`).

- [ ] **Step 2: Run the property tests**

Run: `cargo test -p aleph-sv`
Expected: all unit + property tests pass.

- [ ] **Step 3: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-sv/src/backend.rs
git commit -m "[P0-09] aleph-sv: property tests (norm, involution, control equivalence)"
```

---

### Task 17: Integration tests — Tier-1 algorithms via parser

**Files:**
- Create: `crates/aleph-sv/tests/tier1.rs`

- [ ] **Step 1: Write the integration test file**

Create `crates/aleph-sv/tests/tier1.rs` with:

```rust
//! Tier-1 algorithm integration tests. Each test parses OpenQASM 3.0
//! text, runs it through `aleph_backend::run` with `NaiveSvBackend`,
//! and checks the final state against analytic expectations.
//!
//! Oracle comparison against Qiskit lands in P0-10.

use aleph_backend::run;
use aleph_core::Complex;
use aleph_parser::parse;
use aleph_sv::NaiveSvBackend;

const TOL: f64 = 1e-10;

fn ghz_qasm(n: u32) -> String {
    let mut out = format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{n}] q;\n");
    out.push_str("h q[0];\n");
    for i in 0..n - 1 {
        out.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
    }
    out
}

#[test]
fn ghz_2() {
    let circ = parse(&ghz_qasm(2)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[3].re - inv_s2).abs() < TOL);
    assert!(a[1].norm() < TOL);
    assert!(a[2].norm() < TOL);
}

#[test]
fn ghz_5() {
    let circ = parse(&ghz_qasm(5)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let n = 5;
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    let last = (1usize << n) - 1;
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[last].re - inv_s2).abs() < TOL);
    let mass: f64 = (1..last).map(|i| a[i].norm_sqr()).sum();
    assert!(mass < TOL);
}

#[test]
fn ghz_10() {
    let circ = parse(&ghz_qasm(10)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let n = 10;
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    let last = (1usize << n) - 1;
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[last].re - inv_s2).abs() < TOL);
    let mass: f64 = (1..last).map(|i| a[i].norm_sqr()).sum();
    assert!(mass < TOL);
}

#[test]
fn ghz_20_runs() {
    // Acceptance criterion: 20 qubits must run end-to-end.
    let circ = parse(&ghz_qasm(20)).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let n = 20;
    let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
    let a = s.amplitudes();
    let last = (1usize << n) - 1;
    assert!((a[0].re - inv_s2).abs() < TOL);
    assert!((a[last].re - inv_s2).abs() < TOL);
}

#[test]
fn qft_3_on_one() {
    // QFT-3 applied to |001⟩ (which is q0=1, q1=0, q2=0, index 1).
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
x q[0];
h q[2];
cp(pi/2) q[1], q[2];
cp(pi/4) q[0], q[2];
h q[1];
cp(pi/2) q[0], q[1];
h q[0];
swap q[0], q[2];
"#;
    let circ = parse(src).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    // Expected QFT|1⟩ amplitudes: 1/√8 · exp(2πi·k/8) for k ∈ 0..8.
    let n = 3;
    let dim = 1usize << n;
    let norm = (1.0 / (dim as f64)).sqrt();
    let a = s.amplitudes();
    for k in 0..dim {
        let phase = 2.0 * std::f64::consts::PI * (k as f64) / (dim as f64);
        let want = Complex::new(norm * phase.cos(), norm * phase.sin());
        assert!(
            (a[k] - want).norm() < 1e-9,
            "k={k}: got {:?}, want {want:?}",
            a[k]
        );
    }
}

#[test]
fn grover_3_one_marked() {
    // 3-qubit Grover with marked state |111⟩. One iteration gives
    // P(|111⟩) ≈ 0.7812.
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
h q[1];
h q[2];
ccz q[0], q[1], q[2];
h q[0];
h q[1];
h q[2];
x q[0];
x q[1];
x q[2];
ccz q[0], q[1], q[2];
x q[0];
x q[1];
x q[2];
h q[0];
h q[1];
h q[2];
"#;
    let circ = parse(src).unwrap();
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, &circ).unwrap();
    let p_marked = s.amplitudes()[7].norm_sqr();
    assert!(p_marked > 0.78, "p_marked = {p_marked}");
}

#[test]
fn random_clifford_t_8q_is_deterministic_and_normalised() {
    // 8-qubit random-ish Clifford+T circuit (depth ~10). Determinism
    // check: same circuit + same input ⇒ same final state.
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[8] q;
h q[0]; h q[1]; h q[2]; h q[3];
t q[4]; t q[5]; t q[6]; t q[7];
cx q[0], q[4];
cx q[1], q[5];
cx q[2], q[6];
cx q[3], q[7];
s q[0]; s q[1];
h q[4]; h q[5];
cx q[4], q[0];
cx q[5], q[1];
t q[2]; t q[3];
cx q[6], q[2];
cx q[7], q[3];
h q[6]; h q[7];
"#;
    let circ = parse(src).unwrap();
    let mut b1 = NaiveSvBackend::with_seed(0);
    let mut b2 = NaiveSvBackend::with_seed(0);
    let s1 = run(&mut b1, &circ).unwrap();
    let s2 = run(&mut b2, &circ).unwrap();
    // Determinism.
    for (a, b) in s1.amplitudes().iter().zip(s2.amplitudes().iter()) {
        assert!((a - b).norm() < 1e-15);
    }
    // Normalisation.
    let total: f64 = s1.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((total - 1.0).abs() < 1e-10, "norm² = {total}");
}
```

- [ ] **Step 2: Run the integration tests**

Run: `cargo test -p aleph-sv --test tier1`
Expected: 7 tests pass. (If any QASM construct isn't supported by the parser — `cp`, `ccz` — fall back to expressing them via the supported gates the parser exposes and re-run.)

- [ ] **Step 3: If parser rejects `cp` or `ccz`**

Open `crates/aleph-parser/src/lower.rs` and confirm which gate names are wired. If `cp` or `ccz` are missing, edit the QFT-3 and Grover-3 strings to expand them using gates the parser does support (e.g. `ccz q[0],q[1],q[2]` → `h q[2]; ccx q[0],q[1],q[2]; h q[2];`; `cp(λ) a,b` → `rz(λ/2) a; rz(λ/2) b; cx a,b; rz(-λ/2) b; cx a,b;`). Re-run the tests until green.

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/tests/tier1.rs
git commit -m "[P0-09] aleph-sv: Tier-1 integration tests (GHZ, QFT, Grover, random)"
```

---

### Task 18: Criterion benchmark — H wall baseline

**Files:**
- Create: `crates/aleph-sv/benches/naive_sv.rs`

- [ ] **Step 1: Create the benchmark file**

Create `crates/aleph-sv/benches/naive_sv.rs` with:

```rust
//! Baseline benchmark for [`aleph_sv::NaiveSvBackend`]: apply an H to
//! every qubit on `n ∈ {10, 15, 20}`. Establishes the curve P0-11 will
//! measure against.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallvec::smallvec;

fn h_wall(c: &mut Criterion) {
    let mut group = c.benchmark_group("h_wall");
    for &n in &[10u32, 15, 20] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &n| {
            bencher.iter_with_setup(
                || {
                    let mut b = NaiveSvBackend::with_seed(0);
                    let s = b.allocate(n).unwrap();
                    (b, s)
                },
                |(mut b, mut s)| {
                    for q in 0..n {
                        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![q]))
                            .unwrap();
                    }
                    criterion::black_box(&s);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, h_wall);
criterion_main!(benches);
```

- [ ] **Step 2: Smoke-build the benchmark**

Run: `cargo build -p aleph-sv --benches --release`
Expected: success.

- [ ] **Step 3: Run the benchmark briefly**

Run: `cargo bench -p aleph-sv --bench naive_sv -- --quick`
Expected: completes; produces measurements for n=10/15/20. No assertion — just confirming the bench harness works.

- [ ] **Step 4: Lint and format**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 5: Final workspace-wide check**

Run:

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/benches/naive_sv.rs
git commit -m "[P0-09] aleph-sv: criterion baseline benchmark (H wall)"
```

---

## Plan self-review notes

- **Spec coverage:** every spec section has at least one task. §6.1 → Task 2; §6.2 → Tasks 3–5; §6.3 → Tasks 6–7; §7.2–7.4 → Tasks 8–10; §7.5 → Task 11; §7.6 → Task 12; §7.7 → Task 13; §7.8 → Task 15; §7.9 → Task 14; §7.10 → Task 5; §8.1 unit tests are embedded in their respective tasks; §8.2 → Task 16; §8.3 → Task 17; §8.4 → Task 18.
- **No placeholders:** every step shows the code or the exact command.
- **Naming consistency:** `apply_1q` / `apply_2q` / `apply_3q`, `measure_impl` / `sample_impl` / `expectation_value_impl` / `probabilities_impl`, `NaiveSvBackend`, `CpuState` — used identically across tasks.
- **Known fragile spot:** Task 17 depends on the parser supporting `cp` and `ccz` (or a fallback). The task contains an explicit fallback step.
