# P3-03 Stabilizer Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the stabilizer simulator into the unified `Backend` trait (`StabilizerBackend`) and expose it through the CLI (`aleph run … --backend stabilizer`), so Clifford circuits run end-to-end alongside the state vector.

**Architecture:** New `StabilizerBackend` in `crates/aleph-stab/src/backend.rs` implementing `aleph_backend::Backend` with `State = Tableau` (allocate/apply_gate/measure/sample/expectation_value; probabilities→Unsupported). A read-only `Tableau::pauli_eigenvalue` computes Pauli expectations via the verified `rowsum`. The CLI gains a `--backend` flag and a stabilizer run path that supports `--shots`/`--expectation` but rejects `--statevector` (a `Tableau` has no dense amplitudes). No SIMD, no Stim — correctness is cross-checked **in-process** against `NaiveSvBackend`, so it all validates locally.

**Tech Stack:** Rust 2021, `aleph-backend` (Backend trait), `aleph-core` (Gate/Pauli/PauliString), `rand` 0.8, `aleph-sv` (oracle, dev-dep), `assert_cmd` (CLI tests).

**Reference:** P3-01/P3-02 (`Tableau`, `apply_gate`, `measure`, `rowsum`). Spec: `docs/superpowers/specs/2026-06-04-p3-03-stabilizer-backend-design.md`.

---

## File Structure

| File | Change |
|------|--------|
| `crates/aleph-stab/Cargo.toml` | add `aleph-backend` dep; `aleph-sv` dev-dep (oracle) |
| `crates/aleph-stab/src/backend.rs` | new: `StabilizerBackend` + `Backend` impl + `map_stab_err` |
| `crates/aleph-stab/src/tableau.rs` | add `pub(crate) fn pauli_eigenvalue` |
| `crates/aleph-stab/src/lib.rs` | `mod backend; pub use backend::StabilizerBackend;` |
| `crates/aleph-stab/tests/sv_cross.rs` | new: cross-backend oracle vs `NaiveSvBackend` |
| `oracle/circuits/surface-code-cycle.qasm` | new Clifford fixture |
| `crates/aleph-cli/Cargo.toml` | add `aleph-stab` dep |
| `crates/aleph-cli/src/cli.rs` | `BackendKind` enum + `backend` arg on `Run` |
| `crates/aleph-cli/src/exec.rs` | `run_stabilizer` path; `backend` param on `run_circuit` |
| `crates/aleph-cli/src/main.rs` | thread `backend` through |
| `crates/aleph-cli/tests/cli.rs` | assert_cmd stabilizer tests |

> Conventions (CLAUDE.md): no `unwrap`/`expect` in library code (tests OK); no `unsafe`; clippy `-D warnings`; `cargo fmt`; rustdoc on public items.

---

## Task 1: `StabilizerBackend` core (`Backend` impl minus expectation)

**Files:**
- Modify: `crates/aleph-stab/Cargo.toml`
- Create: `crates/aleph-stab/src/backend.rs`
- Modify: `crates/aleph-stab/src/lib.rs`

- [ ] **Step 1: Add deps**

In `crates/aleph-stab/Cargo.toml`, add to `[dependencies]` (after `rand`):

```toml
aleph-backend = { path = "../aleph-backend" }
```

(`aleph-sv` is already a dev-dependency from P3-01 — leave it.)

- [ ] **Step 2: Write the failing tests**

Create `crates/aleph-stab/src/backend.rs` with the test module first (implementation added next step):

```rust
//! `StabilizerBackend`: the `aleph_backend::Backend` implementation over
//! the CHP [`Tableau`]. Clifford circuits run end-to-end through the same
//! driver as the state-vector backends; non-Clifford gates are rejected.

#[cfg(test)]
mod tests {
    use super::StabilizerBackend;
    use aleph_backend::{Backend, BackendError};
    use aleph_core::{Gate, GateInstance};

    #[test]
    fn bell_apply_and_measure() {
        let mut be = StabilizerBackend::with_seed(0);
        let mut t = be.allocate(2).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32])).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![0u32, 1u32])).unwrap();
        let b0 = be.measure(&mut t, 0).unwrap();
        let b1 = be.measure(&mut t, 1).unwrap();
        assert_eq!(b0, b1, "Bell correlation through the backend");
    }

    #[test]
    fn rejects_non_clifford() {
        let mut be = StabilizerBackend::with_seed(0);
        let mut t = be.allocate(1).unwrap();
        let err = be
            .apply_gate(&mut t, &GateInstance::new(Gate::T, vec![0u32]))
            .unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedGate { kind } if kind == "T"));
    }

    #[test]
    fn sample_ghz_is_all_zero_or_all_one() {
        // GHZ-4: every shot must be 0000 or 1111 (bits 0..4).
        let mut be = StabilizerBackend::with_seed(42);
        let mut t = be.allocate(4).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32])).unwrap();
        for i in 0..3u32 {
            be.apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![i, i + 1])).unwrap();
        }
        let shots = be.sample(&t, 200).unwrap();
        for s in shots {
            assert!(s == 0b0000 || s == 0b1111, "unexpected GHZ sample {s:04b}");
        }
    }

    #[test]
    fn sample_rejects_over_64_qubits() {
        let mut be = StabilizerBackend::with_seed(0);
        let t = be.allocate(65).unwrap();
        let err = be.sample(&t, 1).unwrap_err();
        assert!(matches!(err, BackendError::TooManyQubits { requested: 65, limit: 64 }));
    }

    #[test]
    fn probabilities_unsupported() {
        let mut be = StabilizerBackend::with_seed(0);
        let t = be.allocate(2).unwrap();
        let err = be.probabilities(&t, &[0]).unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedInstruction { kind } if kind == "probabilities"));
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p aleph-stab --lib backend`
Expected: FAIL — `StabilizerBackend` not defined.

- [ ] **Step 4: Implement `StabilizerBackend`**

Prepend to `backend.rs` (above the test module):

```rust
use aleph_backend::{Backend, BackendError};
use aleph_core::{GateInstance, PauliString};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::{apply_gate, StabError, Tableau};

/// Stabilizer (Aaronson-Gottesman) backend. Simulates Clifford circuits
/// in O(n²) memory; rejects non-Clifford gates.
pub struct StabilizerBackend {
    rng: StdRng,
}

impl StabilizerBackend {
    /// Entropy-seeded RNG (for the random-measurement branch).
    pub fn new() -> Self {
        Self { rng: StdRng::from_entropy() }
    }

    /// Explicit seed; reproducible for a given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self { rng: StdRng::seed_from_u64(seed) }
    }
}

impl Default for StabilizerBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum qubit count `allocate` accepts (generous; stabilizer is
/// O(n²), so this guards only pathological allocations).
const MAX_QUBITS: u32 = 65_536;

fn map_stab_err(e: StabError) -> BackendError {
    match e {
        StabError::NonClifford { gate } => BackendError::UnsupportedGate { kind: gate },
        StabError::QubitOutOfRange { qubit, num_qubits } => {
            BackendError::QubitOutOfRange { qubit, num_qubits }
        }
    }
}

impl Backend for StabilizerBackend {
    type State = Tableau;

    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > MAX_QUBITS {
            return Err(BackendError::TooManyQubits { requested: num_qubits, limit: MAX_QUBITS });
        }
        Ok(Tableau::new(num_qubits as usize))
    }

    fn apply_gate(&mut self, state: &mut Self::State, gate: &GateInstance) -> Result<(), BackendError> {
        apply_gate(state, gate).map_err(map_stab_err)
    }

    fn measure(&mut self, state: &mut Self::State, qubit: u32) -> Result<bool, BackendError> {
        state.measure(qubit as usize, &mut self.rng).map_err(map_stab_err)
    }

    fn sample(&mut self, state: &Self::State, shots: u32) -> Result<Vec<u64>, BackendError> {
        let n = state.num_qubits();
        // One shot packs one bitstring into a u64 (qubit q → bit q, matching
        // the state-vector backends' `1 << qubit` convention), so n ≤ 64.
        if n > 64 {
            return Err(BackendError::TooManyQubits { requested: n as u32, limit: 64 });
        }
        let mut out = Vec::with_capacity(shots as usize);
        for _ in 0..shots {
            let mut t = state.clone();
            let mut bits = 0u64;
            for q in 0..n {
                if t.measure(q, &mut self.rng).map_err(map_stab_err)? {
                    bits |= 1u64 << q;
                }
            }
            out.push(bits);
        }
        Ok(out)
    }

    fn expectation_value(&mut self, _state: &Self::State, _pauli: &PauliString) -> Result<f64, BackendError> {
        // Implemented in Task 2.
        Err(BackendError::UnsupportedInstruction { kind: "expectation_value" })
    }

    fn probabilities(&mut self, _state: &Self::State, _qubits: &[u32]) -> Result<Vec<f64>, BackendError> {
        Err(BackendError::UnsupportedInstruction { kind: "probabilities" })
    }
}
```

- [ ] **Step 5: Export from lib.rs**

In `crates/aleph-stab/src/lib.rs`, add the module + re-export (alongside the existing `mod`s and `pub use`s):

```rust
mod backend;
pub use backend::StabilizerBackend;
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib backend`
Expected: PASS (5 tests).

- [ ] **Step 7: Gate + commit**

```bash
cargo clippy -p aleph-stab --all-targets -- -D warnings
cargo fmt -p aleph-stab
git add crates/aleph-stab/Cargo.toml crates/aleph-stab/src/backend.rs crates/aleph-stab/src/lib.rs
git commit -m "[P3-03] StabilizerBackend: allocate/apply_gate/measure/sample"
```

---

## Task 2: `pauli_eigenvalue` + `expectation_value`

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`
- Modify: `crates/aleph-stab/src/backend.rs`

- [ ] **Step 1: Write the failing tests (tableau unit)**

Add to `tableau.rs` `mod tests`:

```rust
    #[test]
    fn pauli_eigenvalue_bell() {
        // Bell |Φ+>: stabilized by +XX and +ZZ; anticommutes with Z⊗I.
        let mut t = Tableau::new(2);
        t.h(0).unwrap();
        t.cnot(0, 1).unwrap();
        assert_eq!(t.pauli_eigenvalue(&[true, true], &[false, false]), 1); // XX
        assert_eq!(t.pauli_eigenvalue(&[false, false], &[true, true]), 1); // ZZ
        assert_eq!(t.pauli_eigenvalue(&[false, false], &[true, false]), 0); // ZI
        // -ZZ would be -1: prepare |Φ->: apply Z on q0 of the Bell state.
        t.z_gate(0).unwrap();
        assert_eq!(t.pauli_eigenvalue(&[false, false], &[true, true]), -1); // now -ZZ
    }

    #[test]
    fn pauli_eigenvalue_zero_state() {
        let t = Tableau::new(1);
        assert_eq!(t.pauli_eigenvalue(&[false], &[true]), 1); // Z on |0> = +1
        assert_eq!(t.pauli_eigenvalue(&[true], &[false]), 0); // X on |0> = 0
    }
```

- [ ] **Step 2: Run to verify fail**

Run: `cargo test -p aleph-stab --lib pauli_eigenvalue`
Expected: FAIL — `pauli_eigenvalue` not defined.

- [ ] **Step 3: Implement `pauli_eigenvalue` on `Tableau`**

Add to `impl Tableau` in `tableau.rs`:

```rust
    /// `⟨ψ|P|ψ⟩` for the unsigned Pauli `P` given by `(x_p, z_p)` per
    /// qubit (`x` bit = X-component, `z` bit = Z-component; both set = Y).
    /// Returns `+1`/`-1` if `P` (up to sign) is in the stabilizer group,
    /// `0` if `P` anticommutes with some stabilizer generator.
    ///
    /// `x_p` and `z_p` must each have length `self.num_qubits()`.
    pub(crate) fn pauli_eigenvalue(&self, x_p: &[bool], z_p: &[bool]) -> i8 {
        debug_assert_eq!(x_p.len(), self.n);
        debug_assert_eq!(z_p.len(), self.n);
        // Symplectic product of P with row `r`: odd ⇒ anticommute.
        let anti_with = |t: &Tableau, r: usize| -> bool {
            let mut acc = false;
            for j in 0..t.n {
                acc ^= (x_p[j] & t.z.get(r, j)) ^ (z_p[j] & t.x.get(r, j));
            }
            acc
        };
        // 1. Anticommutes with any stabilizer generator ⇒ expectation 0.
        for k in self.n..2 * self.n {
            if anti_with(self, k) {
                return 0;
            }
        }
        // 2. P commutes with all stabilizers ⇒ (pure stabilizer state,
        //    maximal abelian group) P ∈ ⟨generators⟩ up to sign. The
        //    stabilizers whose product equals P are those whose paired
        //    destabilizer anticommutes with P; accumulate them into a
        //    scratch row (on a clone) and read the resulting sign.
        let mut t = self.clone();
        let scratch = 2 * t.n;
        t.zero_row(scratch);
        for k in 0..t.n {
            if anti_with(&t, k) {
                t.rowsum(scratch, k + t.n);
            }
        }
        debug_assert!(
            (0..t.n).all(|j| t.x.get(scratch, j) == x_p[j] && t.z.get(scratch, j) == z_p[j]),
            "accumulated stabilizer product does not equal P"
        );
        if t.sign[scratch] {
            -1
        } else {
            1
        }
    }
```

- [ ] **Step 4: Run tableau tests**

Run: `cargo test -p aleph-stab --lib pauli_eigenvalue`
Expected: PASS.

- [ ] **Step 5: Write the failing backend test**

Add to `backend.rs` `mod tests`:

```rust
    use aleph_core::{Pauli, PauliString};

    #[test]
    fn expectation_value_bell() {
        let mut be = StabilizerBackend::with_seed(0);
        let mut t = be.allocate(2).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::H, vec![0u32])).unwrap();
        be.apply_gate(&mut t, &GateInstance::new(Gate::Cnot, vec![0u32, 1u32])).unwrap();
        let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        let xx = PauliString::new(1.0, vec![(0, Pauli::X), (1, Pauli::X)]).unwrap();
        let zi = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        assert!((be.expectation_value(&t, &zz).unwrap() - 1.0).abs() < 1e-12);
        assert!((be.expectation_value(&t, &xx).unwrap() - 1.0).abs() < 1e-12);
        assert!(be.expectation_value(&t, &zi).unwrap().abs() < 1e-12);
        // coefficient is honoured.
        let half_zz = PauliString::new(0.5, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        assert!((be.expectation_value(&t, &half_zz).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn expectation_value_qubit_out_of_range() {
        let mut be = StabilizerBackend::with_seed(0);
        let t = be.allocate(2).unwrap();
        let p = PauliString::new(1.0, vec![(5, Pauli::Z)]).unwrap();
        let err = be.expectation_value(&t, &p).unwrap_err();
        assert!(matches!(err, BackendError::QubitOutOfRange { qubit: 5, num_qubits: 2 }));
    }
```

- [ ] **Step 6: Run to verify fail**

Run: `cargo test -p aleph-stab --lib expectation_value_bell`
Expected: FAIL (current impl returns Unsupported).

- [ ] **Step 7: Wire `expectation_value`**

Replace the placeholder `expectation_value` in `backend.rs` with:

```rust
    fn expectation_value(&mut self, state: &Self::State, pauli: &PauliString) -> Result<f64, BackendError> {
        let n = state.num_qubits();
        let mut x_p = vec![false; n];
        let mut z_p = vec![false; n];
        for (q, p) in &pauli.terms {
            let qi = *q as usize;
            if qi >= n {
                return Err(BackendError::QubitOutOfRange { qubit: *q, num_qubits: n as u32 });
            }
            match p {
                aleph_core::Pauli::I => {}
                aleph_core::Pauli::X => x_p[qi] = true,
                aleph_core::Pauli::Z => z_p[qi] = true,
                aleph_core::Pauli::Y => {
                    x_p[qi] = true;
                    z_p[qi] = true;
                }
            }
        }
        let s = state.pauli_eigenvalue(&x_p, &z_p);
        Ok(pauli.coefficient * s as f64)
    }
```

- [ ] **Step 8: Run backend tests**

Run: `cargo test -p aleph-stab --lib backend`
Expected: PASS (all, including the two new).

- [ ] **Step 9: Gate + commit**

```bash
cargo clippy -p aleph-stab --all-targets -- -D warnings
cargo fmt -p aleph-stab
git add crates/aleph-stab/src/tableau.rs crates/aleph-stab/src/backend.rs
git commit -m "[P3-03] Stabilizer Pauli expectation_value (pauli_eigenvalue)"
```

---

## Task 3: Cross-backend oracle vs `NaiveSvBackend`

**Files:**
- Create: `crates/aleph-stab/tests/sv_cross.rs`

- [ ] **Step 1: Write the oracle tests**

Create `crates/aleph-stab/tests/sv_cross.rs`:

```rust
//! Stabilizer ≡ state vector on Clifford circuits. `expectation_value`
//! is deterministic, so it's compared exactly against `NaiveSvBackend`;
//! `sample` is checked for support (no impossible outcomes) + known
//! correlations (RNG sequences differ between backends, so exact counts
//! are not comparable).

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance, Pauli, PauliString};
use aleph_stab::StabilizerBackend;
use aleph_sv::NaiveSvBackend;

const N: usize = 5;

/// Deterministic xorshift so circuits are reproducible without an RNG dep.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_clifford(seed: u64) -> Vec<GateInstance> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for _ in 0..40 {
        let q = rng.below(N as u64) as u32;
        match rng.below(6) {
            0 => out.push(GateInstance::new(Gate::H, vec![q])),
            1 => out.push(GateInstance::new(Gate::S, vec![q])),
            2 => out.push(GateInstance::new(Gate::X, vec![q])),
            3 => out.push(GateInstance::new(Gate::Z, vec![q])),
            _ => {
                let a = q;
                let mut b = rng.below(N as u64) as u32;
                if a == b {
                    b = (b + 1) % N as u32;
                }
                out.push(GateInstance::new(Gate::Cnot, vec![a, b]));
            }
        }
    }
    out
}

fn apply_all<B: Backend>(be: &mut B, st: &mut B::State, circ: &[GateInstance]) {
    for g in circ {
        be.apply_gate(st, g).unwrap();
    }
}

#[test]
fn expectation_matches_state_vector() {
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    for k in 0..50u64 {
        let circ = random_clifford(k);

        let mut sb = StabilizerBackend::with_seed(0);
        let mut st = sb.allocate(N as u32).unwrap();
        apply_all(&mut sb, &mut st, &circ);

        let mut nb = NaiveSvBackend::with_seed(0);
        let mut nv = nb.allocate(N as u32).unwrap();
        apply_all(&mut nb, &mut nv, &circ);

        // A handful of random Pauli observables per circuit.
        let mut rng = Rng(0x5151 ^ k);
        for _ in 0..6 {
            let terms: Vec<(u32, Pauli)> = (0..N as u32)
                .filter_map(|q| {
                    let p = paulis[rng.below(4) as usize];
                    if p == Pauli::I {
                        None
                    } else {
                        Some((q, p))
                    }
                })
                .collect();
            let ps = PauliString::new(1.0, terms).unwrap();
            let s = sb.expectation_value(&st, &ps).unwrap();
            let v = nb.expectation_value(&nv, &ps).unwrap();
            assert!(
                (s - v).abs() < 1e-9,
                "circuit {k}: stabilizer ⟨P⟩={s} != state-vector {v} for {ps:?}"
            );
        }
    }
}

#[test]
fn sample_support_is_physical() {
    // Every stabilizer-sampled bitstring must have nonzero probability in
    // the state-vector amplitudes for the same circuit.
    for k in 0..20u64 {
        let circ = random_clifford(k);

        let mut sb = StabilizerBackend::with_seed(k);
        let mut st = sb.allocate(N as u32).unwrap();
        apply_all(&mut sb, &mut st, &circ);
        let shots = sb.sample(&st, 200).unwrap();

        let mut nb = NaiveSvBackend::with_seed(0);
        let mut nv = nb.allocate(N as u32).unwrap();
        apply_all(&mut nb, &mut nv, &circ);
        // `CpuState::amplitudes(&self) -> &[Complex]` (inherent method,
        // crates/aleph-sv/src/state.rs); index = basis state, qubit q at bit q.
        let amps = nv.amplitudes();

        for s in &shots {
            let idx = *s as usize;
            let p = amps[idx].norm_sqr();
            assert!(p > 1e-12, "circuit {k}: sampled |{s:0width$b}⟩ has prob {p}", width = N);
        }
    }
}
```

> `Complex::norm_sqr()` is the `num_complex` method for `|z|²`. `amps`
> has length `2^N`; `idx` (the sampled `u64`) is always `< 2^N` since
> each qubit contributes one bit.

- [ ] **Step 2: Run the oracle tests**

Run: `cargo test -p aleph-stab --test sv_cross`
Expected: PASS (both). If `expectation_matches_state_vector` fails, the
bug is in `pauli_eigenvalue` (Task 2) — fix there, not the test.

- [ ] **Step 3: Commit**

```bash
cargo clippy -p aleph-stab --all-targets -- -D warnings
git add crates/aleph-stab/tests/sv_cross.rs
git commit -m "[P3-03] Cross-backend oracle: stabilizer ≡ state vector"
```

---

## Task 4: `surface-code-cycle.qasm` fixture

**Files:**
- Create: `oracle/circuits/surface-code-cycle.qasm`
- Modify: `crates/aleph-stab/tests/sv_cross.rs` (smoke test)

- [ ] **Step 1: Create the fixture**

Create `oracle/circuits/surface-code-cycle.qasm` — one Clifford
stabilizer-measurement round of a 2×2 plaquette (q0–3 data, q4 Z-ancilla
for `Z0Z1Z2Z3`, q5 X-ancilla for `X0X1X2X3`):

```text
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
bit[2] c;
// Z-parity stabilizer on data q0..q3 via ancilla q4.
cx q[0], q[4];
cx q[1], q[4];
cx q[2], q[4];
cx q[3], q[4];
measure q[4] -> c[0];
// X-parity stabilizer on data q0..q3 via ancilla q5.
h q[5];
cx q[5], q[0];
cx q[5], q[1];
cx q[5], q[2];
cx q[5], q[3];
h q[5];
measure q[5] -> c[1];
```

- [ ] **Step 2: Write a smoke test**

Add to `crates/aleph-stab/tests/sv_cross.rs`:

```rust
#[test]
fn surface_code_cycle_runs_on_stabilizer() {
    let src = std::fs::read_to_string(
        aleph_oracle::workspace_path("oracle/circuits/surface-code-cycle.qasm"),
    )
    .unwrap();
    let circuit = aleph_parser::parse(&src).unwrap();
    let mut be = StabilizerBackend::with_seed(0);
    let state = aleph_backend::run(&mut be, &circuit).unwrap();
    assert_eq!(state.num_qubits(), 6);
}
```

This requires `aleph-parser` and `aleph-oracle` as dev-deps of
`aleph-stab`.

- [ ] **Step 3: Add the dev-deps**

In `crates/aleph-stab/Cargo.toml` `[dev-dependencies]`, add:

```toml
aleph-parser = { path = "../aleph-parser" }
aleph-oracle = { path = "../aleph-oracle" }
```

- [ ] **Step 4: Run the smoke test**

Run: `cargo test -p aleph-stab --test sv_cross surface_code_cycle_runs_on_stabilizer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add oracle/circuits/surface-code-cycle.qasm crates/aleph-stab/Cargo.toml crates/aleph-stab/tests/sv_cross.rs
git commit -m "[P3-03] surface-code-cycle.qasm fixture + stabilizer smoke test"
```

---

## Task 5: CLI `--backend` flag + stabilizer run path

**Files:**
- Modify: `crates/aleph-cli/Cargo.toml`
- Modify: `crates/aleph-cli/src/cli.rs`
- Modify: `crates/aleph-cli/src/exec.rs`
- Modify: `crates/aleph-cli/src/main.rs`
- Modify: `crates/aleph-cli/tests/cli.rs`

- [ ] **Step 1: Add the dep**

In `crates/aleph-cli/Cargo.toml` `[dependencies]`, add:

```toml
aleph-stab = { path = "../aleph-stab" }
```

- [ ] **Step 2: Add `BackendKind` + the `--backend` arg**

In `crates/aleph-cli/src/cli.rs`, near the `Precision` enum, add:

```rust
/// Simulation backend selector for `aleph run`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum BackendKind {
    /// Dense state vector (default). Exact; memory grows as 2^n.
    #[default]
    Statevector,
    /// Stabilizer (Clifford-only). O(n²) memory; thousands of qubits.
    Stabilizer,
}
```

And add the field to the `Run` variant (after `precision`):

```rust
        /// Simulation backend: `statevector` (default) or `stabilizer`
        /// (Clifford-only; rejects non-Clifford gates and --statevector).
        #[arg(long, value_enum, default_value_t = BackendKind::Statevector)]
        backend: BackendKind,
```

- [ ] **Step 3: Thread `backend` through `main.rs`**

In `crates/aleph-cli/src/main.rs`, add `backend` to the `Cmd::Run`
destructuring and pass it to `run_circuit` (as a new last-before-`out`
argument):

```rust
        Cmd::Run {
            qasm,
            shots,
            statevector,
            force_statevector,
            expectation,
            seed,
            precision,
            backend,
        } => run_circuit(
            &qasm,
            shots,
            statevector,
            force_statevector,
            &expectation,
            seed,
            precision,
            backend,
            &mut out,
        )?,
```

- [ ] **Step 4: Add the `backend` param + stabilizer branch in `exec.rs`**

In `crates/aleph-cli/src/exec.rs`:

(a) add `use crate::cli::{BackendKind, Precision};` (extend the existing
`use crate::cli::Precision;`), and `use aleph_stab::StabilizerBackend;`.

(b) change `run_circuit`'s signature to accept `backend: BackendKind`
just before `out`:

```rust
pub fn run_circuit<W: Write>(
    qasm_path: &Path,
    shots_opt: Option<u32>,
    print_statevector: bool,
    force_statevector: bool,
    expectations: &[String],
    seed: Option<u64>,
    precision: Precision,
    backend: BackendKind,
    out: &mut W,
) -> Result<()> {
```

(c) Just before the existing `match precision { … }` block (the one that
builds `NaiveSvBackend`/`Fp32SvBackend`), branch on `backend`. When
`Stabilizer`, route to a new helper and return early:

```rust
    if backend == BackendKind::Stabilizer {
        return run_stabilizer(
            &circuit,
            effective_shots,
            print_statevector || force_statevector,
            &paulis,
            n,
            seed,
            &seed_label,
            out,
        );
    }
```

(d) Add the `run_stabilizer` helper (place it after `run_with_backend`):

```rust
/// Stabilizer-backend run path. Supports `--shots` and `--expectation`;
/// rejects `--statevector` (a tableau has no dense amplitudes).
#[allow(clippy::too_many_arguments)]
fn run_stabilizer<W: Write>(
    circuit: &aleph_ir::Circuit,
    effective_shots: Option<u32>,
    statevector_requested: bool,
    paulis: &[(String, aleph_core::PauliString)],
    n: u32,
    seed: Option<u64>,
    seed_label: &str,
    out: &mut W,
) -> Result<()> {
    if statevector_requested {
        return Err(anyhow!(
            "the stabilizer backend has no dense state vector; drop --statevector \
             (use --shots and/or --expectation instead)"
        ));
    }
    let mut backend = match seed {
        Some(s) => StabilizerBackend::with_seed(s),
        None => StabilizerBackend::new(),
    };
    let state = run(&mut backend, circuit).context("running circuit (stabilizer)")?;

    if let Some(shots) = effective_shots {
        let samples = backend.sample(&state, shots).context("sampling final state")?;
        output::format_counts(out, &samples, shots, n, seed_label)?;
    }
    if !paulis.is_empty() {
        writeln!(out, "expectation values:")?;
        for (raw, ps) in paulis {
            let v = backend
                .expectation_value(&state, ps)
                .with_context(|| format!("computing expectation value for {raw:?}"))?;
            output::format_expectation(out, raw, v)?;
        }
    }
    Ok(())
}
```

> Note `effective_shots` and `seed_label` are computed earlier in
> `run_circuit` (steps 4–5 of the existing body) and are in scope at the
> branch point. The `print_statevector`/`force_statevector` cap check
> (existing step 3) runs before the branch for the state-vector path; for
> stabilizer we reject statevector inside `run_stabilizer` instead. Ensure
> the early `return` for `Stabilizer` happens BEFORE the `match precision`
> block but AFTER `effective_shots`/`seed_label` are computed.

- [ ] **Step 5: Add assert_cmd tests**

Add to `crates/aleph-cli/tests/cli.rs`:

```rust
fn surface_code_path() -> std::path::PathBuf {
    aleph_oracle::workspace_path("oracle/circuits/surface-code-cycle.qasm")
}

#[test]
fn stabilizer_backend_runs_surface_code() {
    aleph()
        .args(["run"])
        .arg(surface_code_path())
        .args(["--backend", "stabilizer", "--shots", "1024", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("counts (1024 shots, seed=0):"));
}

#[test]
fn stabilizer_backend_rejects_non_clifford() {
    // qft_5 contains non-Clifford phase gates.
    let qft = aleph_oracle::workspace_path("oracle/circuits/qft_5.qasm");
    aleph()
        .args(["run"])
        .arg(qft)
        .args(["--backend", "stabilizer", "--shots", "16"])
        .assert()
        .failure()
        .stderr(contains("not supported"));
}

#[test]
fn stabilizer_backend_rejects_statevector() {
    aleph()
        .args(["run"])
        .arg(surface_code_path())
        .args(["--backend", "stabilizer", "--statevector"])
        .assert()
        .failure()
        .stderr(contains("no dense state vector"));
}
```

> `aleph_oracle` is already a dev-dependency of `aleph-cli` (used by the
> existing tests via `workspace_path`). The non-Clifford error text:
> `BackendError::UnsupportedGate` renders as "gate `…` is not supported by
> this backend", so `contains("not supported")` matches. Verify `qft_5.qasm`
> indeed contains a non-Clifford gate (it has controlled-phase `cp`/`p`);
> if not, use `oracle/circuits/kernel_t.qasm` (a bare `T` gate) instead.

- [ ] **Step 6: Run the CLI tests**

Run: `cargo test -p aleph-cli`
Expected: PASS (existing + 3 new).

- [ ] **Step 7: Gate + commit**

```bash
cargo clippy -p aleph-cli --all-targets -- -D warnings
cargo fmt -p aleph-cli
git add crates/aleph-cli
git commit -m "[P3-03] CLI --backend stabilizer run path + tests"
```

---

## Task 6: Workspace gate + final review + PR

**Files:** none (validation + PR).

- [ ] **Step 1: Full-workspace gate**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
Expected: all green. (No SIMD/Stim/EPYC needed — stabilizer is scalar and
the oracle is the in-process `NaiveSvBackend`.) Fix any issue in the
relevant task's files.

- [ ] **Step 2: Self-review the diff**

```bash
git diff origin/main..HEAD --stat
git diff origin/main..HEAD
```
Re-read with fresh eyes: error mapping correct, no `unwrap` in library
code, no `unsafe`, `pauli_eigenvalue` purity (no mutation of the passed
state), CLI help text accurate.

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin p3-03-stabilizer-backend
gh pr create --title "[P3-03] Stabilizer backend integration with Backend trait" --body "$(cat <<'EOF'
Closes #34

## Summary
Wires the stabilizer simulator into the unified `Backend` trait and the
CLI, completing the stabilizer chain (P3-01 tableau → P3-02 measurement →
P3-03 integration):
- `StabilizerBackend` (`crates/aleph-stab/src/backend.rs`) implements
  `Backend` with `State = Tableau`: `allocate`, `apply_gate` (rejects
  non-Clifford → `UnsupportedGate`), `measure`, `sample` (≤64 qubits),
  `expectation_value`. `probabilities` returns a clear Unsupported error.
- `Tableau::pauli_eigenvalue` computes Pauli expectations via the verified
  `rowsum` (commute-check + scratch-row product), read-only.
- CLI `aleph run … --backend {statevector,stabilizer}`. The stabilizer
  path supports `--shots`/`--expectation` and rejects `--statevector`
  (a tableau has no dense amplitudes).
- New `oracle/circuits/surface-code-cycle.qasm` Clifford fixture.

## Tests
- Unit: apply/measure round-trip, non-Clifford rejection, `sample` GHZ
  all-0/all-1, `sample` >64 qubits → `TooManyQubits`, `expectation_value`
  on Bell (⟨ZZ⟩=⟨XX⟩=+1, ⟨ZI⟩=0, coefficient honoured), `probabilities`
  Unsupported, out-of-range Pauli qubit.
- **Cross-backend oracle vs `NaiveSvBackend`:** `expectation_value` matches
  the state vector exactly (50 random Clifford circuits × 6 random Paulis,
  1e-9); `sample` support is physical (no impossible outcomes).
- **CLI (`assert_cmd`):** `--backend stabilizer` runs surface-code-cycle
  and prints counts; rejects a non-Clifford circuit; rejects `--statevector`.

All workspace tests / clippy `-D warnings` / fmt green. (No SIMD/Stim —
correctness is cross-checked in-process against the state vector.)

## AC mapping
- [x] Stabilizer reachable through unified API
- [x] Clear errors on non-Clifford gates
- [x] CLI option works
- [x] Integration: run surface-code-cycle.qasm through the stabilizer backend

## Follow-ups
- P3-07 auto-selection can now route Clifford-only circuits here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Confirm CI green.**

Run: `gh pr checks --watch`.

---

## Self-Review (plan vs spec)

**Spec coverage:**
- §2 crate wiring (aleph-backend dep, exports) → Task 1. ✓
- §3 `StabilizerBackend` methods (allocate/apply_gate/measure/sample, probabilities Unsupported, inherited defaults) → Task 1; error mapping §3.2 → Task 1 `map_stab_err`. ✓
- §4 `expectation_value` + `pauli_eigenvalue` → Task 2. ✓
- §5 CLI `--backend` + `run_stabilizer` (rejects statevector) → Task 5. ✓
- §6 surface-code-cycle.qasm → Task 4. ✓
- §7.1 unit → Tasks 1, 2; §7.2 sample oracle + §7.3 expectation oracle → Task 3; §7.4 CLI assert_cmd → Task 5. ✓
- §8 AC mapping → Task 6 PR body. ✓

**Placeholder scan:** none. One executor note (non-Clifford fixture choice in Task 5) is an explicit "verify, with fallback" instruction — not a gap. (The Task 3 amplitude accessor was resolved to `CpuState::amplitudes()`.)

**Type consistency:** `StabilizerBackend::{new,with_seed}`, `map_stab_err`, `MAX_QUBITS`, `pauli_eigenvalue(&self, &[bool], &[bool]) -> i8`, `BackendKind::{Statevector,Stabilizer}`, `run_stabilizer(...)` consistent across tasks. `Backend` method signatures match the trait (`&mut self`, `&Self::State` for queries). Bit convention `1u64 << q` matches `measure_impl`. `sample` returns `Vec<u64>`; cross-check reads `amps[idx]` with qubit q at bit q.

**Known soft spot (flagged in-task):** the non-Clifford CLI fixture — Task 5 note gives `kernel_t.qasm` as fallback if `qft_5.qasm` isn't parsed as non-Clifford. Concrete fallback; does not block.
