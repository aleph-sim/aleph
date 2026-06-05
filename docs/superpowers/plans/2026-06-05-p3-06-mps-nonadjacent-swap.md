# P3-06 MPS — non-adjacent 2q gates via SWAP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let `MpsState::apply_2q` apply a 2q gate between non-adjacent qubits by inserting a nearest-neighbor SWAP network (always-swap-back), preserving the site = qubit invariant so all readout paths stay unchanged.

**Architecture:** Refactor the current nearest-neighbor `apply_2q` body into a private `apply_2q_adjacent(q0, q1, u)`. `apply_2q(g, u)` becomes a dispatcher: adjacent → call it directly; non-adjacent → SWAP the higher-site qubit down next to the lower one, apply the gate, undo the SWAPs. A `swap_adjacent(k)` helper applies `Gate::Swap` on sites `(k, k+1)` through `apply_2q_adjacent`.

**Tech Stack:** Rust 2021, existing MPS machinery (`apply_2q` contraction + `truncated_svd`), `Gate::Swap`, `proptest`, `criterion`.

**Spec:** `docs/superpowers/specs/2026-06-05-p3-06-mps-nonadjacent-swap-design.md`
**Branch:** `p3-06-mps-nonadjacent-swap` (off `main`).
**Conventions:** no `unwrap`/`expect` in lib code (tests OK); `cargo clippy --workspace --all-targets -- -D warnings`; **`cargo fmt --all --check`** (workspace form). Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Current code (ground truth)

`crates/aleph-mps/src/mps.rs` `apply_2q` (lines ~78-199): signature `pub(crate) fn apply_2q(&mut self, g: &GateInstance, u: &[[Complex;4];4]) -> Result<(), MpsError>`. Body: reads `qa=g.qubits[0]`, `qb=g.qubits[1]`; `if qa.abs_diff(qb) != 1 { return Err(MpsError::NonNearestNeighbor { a: qa, b: qb }); }`; `i=qa.min(qb) as usize; j=i+1; self.move_center_to(i);` then builds Θ, applies the gate via an `out(phys_i, phys_j)` closure that reads `g.qubits[0]`/`g.qubits[1]`, reshapes to `m`, `truncated_svd(&m, &self.policy)`, updates `trunc_error`/`max_bond_seen`, writes `sites[i]`/`sites[j]`, `center=j`.
`mps.rs` imports: `use aleph_core::{Complex, GateInstance, PauliString};`. `Gate::Swap` exists with a 4×4 matrix; `crate::gate::matrix_4x4(&GateInstance) -> Result<[[Complex;4];4], MpsError>`.
Existing tests that assert non-adjacent rejection (MUST be removed/replaced): `mps.rs` `rejects_non_adjacent` (asserts `CNOT(0,2)` → `MpsError::NonNearestNeighbor`); `backend.rs` `rejects_non_adjacent` (asserts `CNOT(0,2)` → `BackendError::InvalidState`).

---

## Task 1: Refactor into `apply_2q_adjacent` + `swap_adjacent` + non-adjacent dispatcher

**Files:** Modify `crates/aleph-mps/src/mps.rs`, `crates/aleph-mps/src/backend.rs`.

- [ ] **Step 1: Add `Gate` to imports.** Change `use aleph_core::{Complex, GateInstance, PauliString};` to `use aleph_core::{Complex, Gate, GateInstance, PauliString};`.

- [ ] **Step 2: Rename the current `apply_2q` to `apply_2q_adjacent(q0: u32, q1: u32, u)`.** Replace the function signature/header:
```rust
    /// Apply a 2q gate (4×4 matrix `u`) to ADJACENT sites `q0`, `q1`
    /// (`|q0 − q1| == 1`), in `q0`=MSB / `q1`=LSB convention (ADR-0004).
    /// Callers guarantee adjacency; `apply_2q` is the public entry point.
    fn apply_2q_adjacent(&mut self, q0: u32, q1: u32, u: &[[Complex; 4]; 4]) -> Result<(), MpsError> {
        let i = q0.min(q1) as usize;
        let j = i + 1;
        self.move_center_to(i);
        // ... (unchanged Θ build) ...
```
Inside the body: delete the `let qa = g.qubits[0]; let qb = g.qubits[1]; if qa.abs_diff(qb) != 1 { return Err(...) }` lines (the dispatcher now guarantees adjacency). In the `out` closure, replace `g.qubits[0]` with `q0` and `g.qubits[1]` with `q1` (both the `as usize == i` comparisons). Everything else in the body is byte-identical.

- [ ] **Step 3: Add the `swap_adjacent` helper** (right after `apply_2q_adjacent`):
```rust
    /// Swap the qubit states on adjacent sites `(k, k+1)` via a SWAP gate.
    fn swap_adjacent(&mut self, k: usize) -> Result<(), MpsError> {
        let g = GateInstance::new(Gate::Swap, vec![k as u32, (k + 1) as u32]);
        let u = crate::gate::matrix_4x4(&g)?;
        self.apply_2q_adjacent(k as u32, (k + 1) as u32, &u)
    }
```

- [ ] **Step 4: Add the new public `apply_2q` dispatcher** (place it just above `apply_2q_adjacent`, keeping the original rustdoc about the MSB convention but updating the "only nearest-neighbor" note):
```rust
    /// Apply a 2q gate (4×4 matrix `u`) on the qubits named by `g`
    /// (`g.qubits[0]`=MSB, ADR-0004). Adjacent pairs apply directly; non-adjacent
    /// pairs are brought together by a nearest-neighbor SWAP network, the gate is
    /// applied, then the SWAPs are undone (always-swap-back), so site = qubit is
    /// preserved. `MpsError::NonNearestNeighbor` is retained as a defensive
    /// invariant guard but is no longer reached on the normal 2q path.
    pub(crate) fn apply_2q(&mut self, g: &GateInstance, u: &[[Complex; 4]; 4]) -> Result<(), MpsError> {
        let qa = g.qubits[0];
        let qb = g.qubits[1];
        if qa.abs_diff(qb) == 1 {
            return self.apply_2q_adjacent(qa, qb, u);
        }
        let lo = qa.min(qb);
        let hi = qa.max(qb);
        // Forward ladder: move the qubit at site `hi` down to site `lo+1`.
        for k in (lo as usize + 1..=hi as usize - 1).rev() {
            self.swap_adjacent(k)?;
        }
        // Now qubit `lo` is at site `lo`, qubit `hi` is at site `lo+1`. Apply the
        // gate on the adjacent pair, preserving the original control/target order.
        let (s0, s1) = if qa < qb { (lo, lo + 1) } else { (lo + 1, lo) };
        self.apply_2q_adjacent(s0, s1, u)?;
        // Reverse ladder: undo the SWAPs, restoring site = qubit.
        for k in lo as usize + 1..=hi as usize - 1 {
            self.swap_adjacent(k)?;
        }
        Ok(())
    }
```

- [ ] **Step 5: Remove the now-invalid `rejects_non_adjacent` test in `mps.rs`** and replace with non-adjacent correctness unit tests. Delete the existing `rejects_non_adjacent` test (the one asserting `CNOT(0,2)` → `NonNearestNeighbor`). Add:
```rust
    #[test]
    fn ghz_via_nonadjacent_cnots() {
        // H(0); CNOT(0,2); CNOT(0,3) on 4 qubits. After CNOT(0,2): q2=q0.
        // After CNOT(0,3): q3=q0. State = (|0000⟩ + |1011⟩... wait compute via dense.
        let mut s = MpsState::new(4, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for tgt in [2u32, 3u32] {
            let gi = GateInstance::new(Gate::Cnot, smallvec![0u32, tgt]);
            let cnot = crate::gate::matrix_4x4(&gi).unwrap();
            s.apply_2q(&gi, &cnot).unwrap();
        }
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        // q0 control: |0000⟩ → stays; |1...⟩ flips q2 and q3 → |1⟩ at bits 0,2,3 = 0b1101 = 13.
        assert!((v[0].re - inv).abs() < 1e-10, "|0000>");
        assert!((v[0b1101].re - inv).abs() < 1e-10, "|1101>");
        // everything else ~0
        for (k, amp) in v.iter().enumerate() {
            if k != 0 && k != 0b1101 { assert!(amp.norm() < 1e-10, "idx {k} nonzero"); }
        }
    }

    #[test]
    fn swap_via_nonadjacent() {
        // X(0); SWAP(0,3) → qubit 3 becomes 1, qubit 0 becomes 0 → |1000⟩ (bit3) = 8.
        let mut s = MpsState::new(4, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap();
        s.apply_1q(0, &x);
        let gi = GateInstance::new(Gate::Swap, smallvec![0u32, 3u32]);
        let sw = crate::gate::matrix_4x4(&gi).unwrap();
        s.apply_2q(&gi, &sw).unwrap();
        let v = s.dense_statevector();
        assert!((v[0b1000].re - 1.0).abs() < 1e-10, "expected |1000> (q3=1)");
    }
```
> NOTE: verify the expected indices by reasoning under ADR-0004 (qubit q at bit q). For `ghz_via_nonadjacent_cnots`, q0=1 ⇒ CNOT(0,2) sets q2=1, CNOT(0,3) sets q3=1 ⇒ bits {0,2,3} set = `0b1101` = 13. If the dense check fails, the SWAP-network control/target mapping is wrong — that is the bug to fix, NOT the test.

- [ ] **Step 6: Fix the backend `rejects_non_adjacent` test in `backend.rs`.** It asserts `CNOT(0,2)` → `BackendError::InvalidState`. Since non-adjacent now succeeds, replace it with a test that non-adjacent now WORKS through the backend:
```rust
    #[test]
    fn nonadjacent_cnot_runs() {
        let mut be = MpsBackend::with_seed(0);
        let mut s = be.allocate(3).unwrap();
        be.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        // CNOT(0,2) is non-adjacent — must now succeed.
        be.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0u32, 2u32])).unwrap();
        // GHZ-on-ends: samples are 000 or 101 (q0=q2, q1=0).
        for sh in be.sample(&s, 100).unwrap() { assert!(sh == 0b000 || sh == 0b101); }
    }
```

- [ ] **Step 7: Run** `cargo test -p aleph-mps` → PASS. clippy `-D warnings` + `cargo fmt --all -- --check` clean.

- [ ] **Step 8: Commit.**
```bash
git add crates/aleph-mps/src/mps.rs crates/aleph-mps/src/backend.rs
git commit
```
subject `[P3-06] non-adjacent 2q via SWAP network (always-swap-back)` + trailer.

---

## Task 2: Oracle equivalence vs NaiveSvBackend + proptest

**Files:** Modify `crates/aleph-mps/tests/sv_equivalence.rs`.

The file has `g(gate, &[u32])`, `mps_dense(circuit, chi)`, `sv_dense(circuit)` and imports `aleph_core::{Complex, Gate, GateInstance, Param, Pauli, PauliString}`, `proptest::prelude::*` with an existing `proptest! { … }` block.

- [ ] **Step 1: Add non-adjacent oracle tests** (append):
```rust
#[test]
fn nonadjacent_matches_sv() {
    // Asymmetric control/target + various distances, χ large = exact.
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n { c.add_gate(g(Gate::H, &[q])).unwrap(); }
    c.add_gate(g(Gate::Cnot, &[0, 3])).unwrap();   // distance 3
    c.add_gate(g(Gate::Cnot, &[4, 1])).unwrap();   // reversed, distance 3
    c.add_gate(g(Gate::Cz, &[0, 4])).unwrap();     // distance 4 (symmetric)
    c.add_gate(g(Gate::Cnot, &[2, 0])).unwrap();   // reversed, distance 2
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) { assert!((x - y).norm() < 1e-10); }
}
```
> If `Gate::Cz` is not the variant name, grep `crates/aleph-core/src/gate/kinds.rs` for the controlled-Z variant and use the real name.

- [ ] **Step 2: Add a proptest** with random arbitrary-distance 2q gates. Add INSIDE the existing `proptest! { … }` block:
```rust
    #[test]
    fn random_long_range_matches_sv(seq in prop::collection::vec((0u8..5, 0u8..5, 0u8..5), 0..20)) {
        let n = 5u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        for (op, x, y) in seq {
            let a = (x as u32) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[a])).unwrap(); }
                1 => { c.add_gate(g(Gate::S, &[a])).unwrap(); }
                _ => {
                    let b = (y as u32) % n;
                    if a != b { c.add_gate(g(Gate::Cnot, &[a, b])).unwrap(); }
                }
            }
        }
        if c.is_empty() { return Ok(()); }
        let am = mps_dense(&c, 64);
        let bm = sv_dense(&c);
        for (x, y) in am.iter().zip(bm.iter()) { prop_assert!((x - y).norm() < 1e-9); }
    }
```

- [ ] **Step 3: Run** `cargo test -p aleph-mps --test sv_equivalence` → PASS. Run `PROPTEST_CASES=500 cargo test -p aleph-mps --test sv_equivalence random_long_range` once. clippy + `cargo fmt --all -- --check` clean.

- [ ] **Step 4: Commit.**
```bash
git add crates/aleph-mps/tests/sv_equivalence.rs
git commit
```
subject `[P3-06] oracle equivalence for non-adjacent 2q gates vs NaiveSv` + trailer.

> If `nonadjacent_matches_sv` or the proptest FAILS, the SWAP-network control/target mapping is wrong — report BLOCKED with the failing circuit + elementwise diff; do NOT loosen tolerances.

---

## Task 3: Distance-scaling benchmark

**Files:** Create `crates/aleph-mps/benches/long_range.rs`; modify `crates/aleph-mps/Cargo.toml`.

- [ ] **Step 1: Add the bench target** to `crates/aleph-mps/Cargo.toml` (after the existing `[[bench]] name = "nn_qaoa"`):
```toml
[[bench]]
name = "long_range"
harness = false
```

- [ ] **Step 2: Create `crates/aleph-mps/benches/long_range.rs`:**
```rust
//! Wall-clock cost of a single non-adjacent 2q gate as a function of qubit
//! distance — documents the O(distance) SWAP-network overhead of the
//! always-swap-back strategy. The lazy (permutation-tracking) strategy is
//! deferred (see the P3-06 design); this curve reflects always-swap-back, which
//! performs 2·(distance−1) nearest-neighbor SWAPs per non-local gate.

use aleph_backend::{run, Backend};
use aleph_core::{Gate, GateInstance};
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, Criterion};

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}

/// n-qubit circuit: spread entanglement with an H layer + NN ladder, then a
/// single CNOT(0, dist) whose SWAP cost we are measuring.
fn long_range_circuit(n: u32, dist: u32) -> aleph_ir::Circuit {
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, dist])).unwrap();
    c
}

fn bench(cr: &mut Criterion) {
    let mut grp = cr.benchmark_group("long_range_cnot_n12_chi32");
    let n = 12u32;
    for dist in [1u32, 4, 8, 11] {
        let c = long_range_circuit(n, dist);
        grp.bench_function(format!("dist{dist}"), |b| {
            b.iter(|| {
                let mut be = MpsBackend::with_seed(0).with_max_bond(32);
                run(&mut be, &c).unwrap()
            })
        });
    }
    grp.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

- [ ] **Step 3:** `cargo build -p aleph-mps --benches` → success. (No need to run the full bench.)

- [ ] **Step 4: Commit.**
```bash
git add crates/aleph-mps/Cargo.toml crates/aleph-mps/benches/long_range.rs
git commit
```
subject `[P3-06] long-range distance-scaling benchmark` + trailer.

---

## Task 4: Docs + final gate

**Files:** Modify `crates/aleph-mps/src/lib.rs`.

- [ ] **Step 1: Crate doc.** Append to the `//!` module-doc block in `lib.rs`:
```rust
//!
//! 2q gates between non-adjacent qubits are handled by a nearest-neighbor SWAP
//! network (always-swap-back): the targets are brought together, the gate is
//! applied, and the SWAPs are undone, so `site = qubit` always holds. A lazy
//! permutation-tracking strategy that would avoid the swap-back is a future
//! optimization.
```

- [ ] **Step 2: Full gate.**
```bash
cargo test -p aleph-mps
cargo test -p aleph-cli
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo build -p aleph-mps --benches
```
All pass/clean.

- [ ] **Step 3: Commit.**
```bash
git add crates/aleph-mps/src/lib.rs
git commit
```
subject `[P3-06] docs: non-adjacent SWAP-network note` + trailer.

---

## Final verification (before PR)
- [ ] `cargo test --workspace` green; `cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo fmt --all --check` clean.
- [ ] Self-review the diff.

## PR
- Title: `[P3-06] MPS — non-adjacent 2q gates via SWAP`.
- Body: `Closes #37`. Summary, test results (oracle incl. asymmetric/long-range + proptest), bench note (distance scaling; lazy deferred).

## Notes for the implementer
1. The SWAP-network control/target correctness is the one subtle point — the `nonadjacent_matches_sv` oracle (asymmetric `CNOT(0,3)`, `CNOT(3,0)`, `CNOT(2,0)`) is the guard, exactly as the 2q-convention oracle was in P3-04.
2. `swap_adjacent` builds `GateInstance::new(Gate::Swap, vec![k, k+1])` — use `vec!` (Into<SmallVec>) to avoid importing the `smallvec!` macro in non-test code.
3. Use `cargo fmt --all` (CI runs `--all`).
4. Don't touch readout paths (measure/sample/expectation/probabilities/dense) — site = qubit is preserved by swap-back, so they need no change.
