# P4-07 Surface-code 1-cycle benchmark (stabilizer) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one rotated-surface-code syndrome-extraction cycle (d = 3,5,7,9,11) on the stabilizer backend, prove it matches Stim, verify logical X/Z detection, and produce a time-per-cycle report (`docs/perf/surface_code.md`).

**Architecture:** A hand-rolled rotated-surface-code builder in `benches/src/lib.rs` produces both `Vec<GateInstance>` (for `StabilizerBackend`) and an identical Stim program string. Correctness is proven two ways: a fast deterministic logical/physical-detection gate (no Stim) and an `#[ignore]`d postselected canonical-stabilizer-group oracle vs Stim. Timing is a criterion bench on the aleph side plus a Stim timing script, merged into a dedicated report. The Aer-bound `run.py`/`report.py` are untouched.

**Tech Stack:** Rust (criterion, `aleph-stab` `StabilizerBackend`, `aleph-core` `Gate`/`GateInstance`), Python 3 + `stim` (oracle + timing), stdlib `unittest` (golden test).

**Spec:** `docs/superpowers/specs/2026-06-09-p4-07-surface-code-design.md`

---

## Key facts (verified against the codebase)

- `GateInstance::new(gate: Gate, qubits: impl Into<SmallVec<[u32;4]>>)` — `vec![q]` / `vec![a,b]` coerce. Variants used: `Gate::H`, `Gate::X`, `Gate::Z`, `Gate::Cnot`.
- `StabilizerBackend::with_seed(u64)`, `.allocate(n: u32) -> Result<Tableau, _>`, `.apply_gate(&mut Tableau, &GateInstance) -> Result<(), _>`, `.measure(&mut Tableau, q: u32) -> Result<bool, _>`. Per-qubit `measure` has **no** 64-qubit cap (that cap is only on `sample`).
- Stim-subprocess idiom: see `crates/aleph-stab/tests/stim_measure_oracle.rs` (python3 via `std::process::Command`, stdin program, `#[ignore]`).
- `aleph-benches` does **not** yet depend on `aleph-stab` — Task 6 adds it.
- The `[[bench]]` entries in `benches/Cargo.toml` use `harness = false`.

## Geometry (rotated surface code, d odd ≥ 3) — the construction this plan implements

- **Data qubits:** `d×d`, index `r*d + c` for `r,c ∈ 0..d`. Total `d²`.
- **Ancilla candidates:** centers `(r,c)` for `r,c ∈ {-1,0,…,d-1}`, each owning the ≤4 data qubits `{(r,c),(r,c+1),(r+1,c),(r+1,c+1)}` that lie in-grid. Type **X if `(r+c)` even, else Z**.
- **Keep rule:** keep a candidate iff it has **4** in-grid neighbours (bulk), **or** it has **2** in-grid neighbours and its type matches its boundary (X on a horizontal edge `r∈{-1,d-1}`, Z on a vertical edge `c∈{-1,d-1}`). Corners (1 neighbour) are dropped.
- This yields exactly `d²-1` ancillas, split evenly `(d²-1)/2` X and `(d²-1)/2` Z, all pairwise commuting (any two plaquettes share 0 or 2 data qubits). Counts: d=3→8, d=5→24, d=7→48, d=9→80, d=11→120 ancillas → total qubits `2d²-1` = 17/49/97/161/**241**.
- **Logical operators:** `logical_x` = X on data **column 0** `{r*d+0 : r∈0..d}` (connects the top/bottom X boundaries); `logical_z` = Z on data **row 0** `{0*d+c : c∈0..d}` (connects the left/right Z boundaries). They overlap only at data qubit `0` (anticommute); each commutes with every stabilizer (even overlap).

**The invariant tests in Task 1 are the real specification of the geometry.** If a boundary parity is wrong, the commutation/independence/logical tests fail — fix the construction until they pass.

## Cycle definition

One extraction cycle = (for every X-ancilla `a`: `H a`; `CX a d` for each data neighbour `d`; `H a`) then (for every Z-ancilla `a`: `CX d a` for each data neighbour `d`). Measurements are **not** part of `cycle_gates()` — the caller measures every ancilla in `ancilla_order()` afterwards. Gates-then-measure-all is valid because all stabilizers commute. Neighbours are applied in a fixed canonical order (the order produced by the construction loop, i.e. sorted by `(r,c)`).

---

## Task 1: Surface-code geometry (`SurfaceCode::new`)

**Files:**
- Modify: `benches/src/lib.rs` (append new module / items)
- Test: same file, `#[cfg(test)] mod surface_tests`

- [ ] **Step 1: Write the failing tests (geometry invariants)**

Append to `benches/src/lib.rs`:

```rust
/// Rotated surface code (Fowler et al. 2012; rotated variant per Tomita &
/// Svore 2014). Distance `d` (odd ≥ 3): `d²` data qubits + `d²−1` ancillas
/// (`2d²−1` total). See docs/superpowers/specs/2026-06-09-p4-07-surface-code-design.md.
#[derive(Clone, Debug)]
pub struct Ancilla {
    pub index: u32,
    pub is_x: bool,
    pub data_neighbours: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct SurfaceCode {
    pub distance: usize,
    pub num_qubits: usize,
    pub data: Vec<u32>,
    pub ancillas: Vec<Ancilla>,
    pub logical_x: Vec<u32>,
    pub logical_z: Vec<u32>,
}
```

And the tests:

```rust
#[cfg(test)]
mod surface_tests {
    use super::*;

    // Symplectic anticommutation of two supports given as (data-set, is_x):
    // two Paulis anticommute iff the X-support of one overlaps the Z-support
    // of the other in an odd total count. For an all-X op P and all-Z op Q on
    // data sets A, B: they anticommute iff |A ∩ B| is odd.
    fn anticommute_xz(x_support: &[u32], z_support: &[u32]) -> bool {
        let zset: std::collections::HashSet<u32> = z_support.iter().copied().collect();
        x_support.iter().filter(|q| zset.contains(q)).count() % 2 == 1
    }

    #[test]
    fn counts_are_correct() {
        for d in [3usize, 5, 7, 9, 11] {
            let sc = SurfaceCode::new(d);
            assert_eq!(sc.data.len(), d * d, "d={d} data count");
            assert_eq!(sc.ancillas.len(), d * d - 1, "d={d} ancilla count");
            assert_eq!(sc.num_qubits, 2 * d * d - 1, "d={d} total");
            let xs = sc.ancillas.iter().filter(|a| a.is_x).count();
            let zs = sc.ancillas.iter().filter(|a| !a.is_x).count();
            assert_eq!(xs, (d * d - 1) / 2, "d={d} X-ancilla count");
            assert_eq!(zs, (d * d - 1) / 2, "d={d} Z-ancilla count");
            // Every ancilla weight is 2 or 4; indices are unique and contiguous.
            for a in &sc.ancillas {
                assert!(
                    a.data_neighbours.len() == 2 || a.data_neighbours.len() == 4,
                    "d={d} ancilla {} weight {}", a.index, a.data_neighbours.len()
                );
            }
            let mut idx: Vec<u32> = sc.ancillas.iter().map(|a| a.index).collect();
            idx.sort_unstable();
            let expect: Vec<u32> = ((d * d) as u32..(2 * d * d - 1) as u32).collect();
            assert_eq!(idx, expect, "d={d} ancilla indices contiguous after data");
        }
    }

    #[test]
    fn all_stabilizers_commute() {
        // X-ancilla (all-X) vs Z-ancilla (all-Z) must share an even number of
        // data qubits. Same-type pairs always commute.
        for d in [3usize, 5, 7, 9, 11] {
            let sc = SurfaceCode::new(d);
            for ax in sc.ancillas.iter().filter(|a| a.is_x) {
                for az in sc.ancillas.iter().filter(|a| !a.is_x) {
                    assert!(
                        !anticommute_xz(&ax.data_neighbours, &az.data_neighbours),
                        "d={d}: X-anc {} and Z-anc {} anticommute", ax.index, az.index
                    );
                }
            }
        }
    }

    #[test]
    fn logicals_commute_with_stabilizers_and_anticommute_each_other() {
        for d in [3usize, 5, 7, 9, 11] {
            let sc = SurfaceCode::new(d);
            assert_eq!(sc.logical_x.len(), d, "d={d} logical X weight");
            assert_eq!(sc.logical_z.len(), d, "d={d} logical Z weight");
            // logical_x (all-X) commutes with every Z-stabilizer.
            for az in sc.ancillas.iter().filter(|a| !a.is_x) {
                assert!(
                    !anticommute_xz(&sc.logical_x, &az.data_neighbours),
                    "d={d}: logical_x anticommutes with Z-anc {}", az.index
                );
            }
            // logical_z (all-Z) commutes with every X-stabilizer.
            for ax in sc.ancillas.iter().filter(|a| a.is_x) {
                assert!(
                    !anticommute_xz(&ax.data_neighbours, &sc.logical_z),
                    "d={d}: logical_z anticommutes with X-anc {}", ax.index
                );
            }
            // The two logicals anticommute (overlap on exactly one data qubit).
            assert!(
                anticommute_xz(&sc.logical_x, &sc.logical_z),
                "d={d}: logicals must anticommute"
            );
        }
    }

    #[test]
    #[should_panic]
    fn rejects_even_distance() {
        let _ = SurfaceCode::new(4);
    }
}
```

- [ ] **Step 2: Run the tests; verify they fail to compile (no `SurfaceCode::new`)**

Run: `cargo test -p aleph-benches --lib surface_tests 2>&1 | head -20`
Expected: compile error — `SurfaceCode::new` not found.

- [ ] **Step 3: Implement `SurfaceCode::new`**

Append to `benches/src/lib.rs` (after the struct defs):

```rust
impl SurfaceCode {
    /// Build the rotated surface code of distance `d` (odd, ≥ 3).
    ///
    /// # Panics
    /// Panics if `d < 3` or `d` is even.
    #[must_use]
    pub fn new(distance: usize) -> Self {
        let d = distance;
        assert!(d >= 3 && d % 2 == 1, "distance must be odd and >= 3, got {d}");
        let di = d as i32;
        let didx = |r: i32, c: i32| -> u32 { (r as u32) * d as u32 + c as u32 };

        let data: Vec<u32> = (0..(d * d) as u32).collect();
        let mut ancillas: Vec<Ancilla> = Vec::with_capacity(d * d - 1);
        let mut next = (d * d) as u32;

        // Candidate plaquette centres (r,c), r,c ∈ {-1,…,d-1}; owns the in-grid
        // members of {(r,c),(r,c+1),(r+1,c),(r+1,c+1)}. Type X iff (r+c) even.
        for r in -1..di {
            for c in -1..di {
                let mut nbrs: Vec<u32> = Vec::with_capacity(4);
                for (rr, cc) in [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)] {
                    if (0..di).contains(&rr) && (0..di).contains(&cc) {
                        nbrs.push(didx(rr, cc));
                    }
                }
                let is_x = (r + c).rem_euclid(2) == 0;
                let keep = match nbrs.len() {
                    4 => true,
                    2 => {
                        let horizontal_edge = r == -1 || r == di - 1;
                        let vertical_edge = c == -1 || c == di - 1;
                        (horizontal_edge && is_x) || (vertical_edge && !is_x)
                    }
                    _ => false, // corners (1 neighbour) dropped
                };
                if keep {
                    ancillas.push(Ancilla { index: next, is_x, data_neighbours: nbrs });
                    next += 1;
                }
            }
        }

        // Logical X = data column 0 (top↔bottom); logical Z = data row 0 (left↔right).
        let logical_x: Vec<u32> = (0..d as u32).map(|r| r * d as u32).collect();
        let logical_z: Vec<u32> = (0..d as u32).collect();

        Self {
            distance: d,
            num_qubits: 2 * d * d - 1,
            data,
            ancillas,
            logical_x,
            logical_z,
        }
    }

    /// Ancilla measurement order (construction order; matches the Stim program).
    #[must_use]
    pub fn ancilla_order(&self) -> Vec<u32> {
        self.ancillas.iter().map(|a| a.index).collect()
    }
}
```

- [ ] **Step 4: Run the tests; verify they pass**

Run: `cargo test -p aleph-benches --lib surface_tests`
Expected: all 4 tests PASS. If `all_stabilizers_commute` or the logical test fails, a boundary parity is wrong — re-check the `keep` rule before changing tests.

- [ ] **Step 5: Commit**

```bash
git add benches/src/lib.rs
git commit -m "[P4-07] Rotated surface-code geometry with commutation invariants"
```

---

## Task 2: Cycle gates + deterministic Z-syndrome through the backend

**Files:**
- Modify: `benches/src/lib.rs` (add `cycle_gates`)
- Modify: `benches/Cargo.toml` (add `aleph-stab`, `rand` deps — needed so the test can drive the backend)
- Test: `benches/src/lib.rs` `surface_tests`

- [ ] **Step 1: Add dependencies to `benches/Cargo.toml`**

Under `[dependencies]` add:

```toml
aleph-stab   = { path = "../crates/aleph-stab" }
```

Under `[dev-dependencies]` add (used by tests/bench setup):

```toml
rand          = { workspace = true }
```

- [ ] **Step 2: Write the failing test (Z-syndrome is all-zero from |0…0⟩)**

Add to `surface_tests` in `benches/src/lib.rs`:

```rust
    use aleph_backend::Backend;
    use aleph_stab::StabilizerBackend;

    /// Run one cycle from data |0…0⟩ and return the measured outcome for each
    /// ancilla, in `ancilla_order()`.
    fn run_cycle(sc: &SurfaceCode, seed: u64, pre: &[GateInstance]) -> Vec<bool> {
        let mut be = StabilizerBackend::with_seed(seed);
        let mut t = be.allocate(sc.num_qubits as u32).unwrap();
        for g in pre {
            be.apply_gate(&mut t, g).unwrap();
        }
        for g in sc.cycle_gates() {
            be.apply_gate(&mut t, &g).unwrap();
        }
        sc.ancilla_order()
            .iter()
            .map(|&a| be.measure(&mut t, a).unwrap())
            .collect()
    }

    #[test]
    fn z_syndrome_is_zero_from_ground_state() {
        // From |0…0⟩, every Z-stabilizer is +1 ⇒ its ancilla measures 0,
        // for any seed. (X-ancillas are random and not asserted here.)
        for d in [3usize, 5] {
            let sc = SurfaceCode::new(d);
            let order = sc.ancilla_order();
            for seed in [0u64, 1, 7, 42] {
                let out = run_cycle(&sc, seed, &[]);
                for (k, &anc) in order.iter().enumerate() {
                    let is_z = sc.ancillas.iter().find(|a| a.index == anc).unwrap().is_x == false;
                    if is_z {
                        assert!(!out[k], "d={d} seed={seed}: Z-ancilla {anc} fired from |0>");
                    }
                }
            }
        }
    }
```

- [ ] **Step 3: Run; verify failure (no `cycle_gates`)**

Run: `cargo test -p aleph-benches --lib surface_tests::z_syndrome 2>&1 | head -20`
Expected: compile error — `cycle_gates` not found.

- [ ] **Step 4: Implement `cycle_gates`**

Add to `impl SurfaceCode` in `benches/src/lib.rs`:

```rust
    /// One syndrome-extraction cycle as gates (no measurements). X-ancillas:
    /// `H a; CX a d…; H a`. Z-ancillas: `CX d… a`. Caller measures ancillas
    /// in `ancilla_order()` afterwards.
    #[must_use]
    pub fn cycle_gates(&self) -> Vec<GateInstance> {
        let mut g = Vec::new();
        for a in self.ancillas.iter().filter(|a| a.is_x) {
            g.push(GateInstance::new(Gate::H, vec![a.index]));
            for &d in &a.data_neighbours {
                g.push(GateInstance::new(Gate::Cnot, vec![a.index, d]));
            }
            g.push(GateInstance::new(Gate::H, vec![a.index]));
        }
        for a in self.ancillas.iter().filter(|a| !a.is_x) {
            for &d in &a.data_neighbours {
                g.push(GateInstance::new(Gate::Cnot, vec![d, a.index]));
            }
        }
        g
    }
```

- [ ] **Step 5: Run; verify pass**

Run: `cargo test -p aleph-benches --lib surface_tests`
Expected: all tests PASS (5 now).

- [ ] **Step 6: Commit**

```bash
git add benches/src/lib.rs benches/Cargo.toml
git commit -m "[P4-07] cycle_gates + deterministic Z-syndrome from |0>"
```

---

## Task 3: Logical / physical detection gate (fast CI test)

**Files:**
- Create: `benches/tests/surface_code_logical.rs`

This is the AC "logical X/Z operator detection works as expected" gate. Fully deterministic via Z-stabilizers (|0…0⟩) and X-stabilizers (|+…+⟩). No Stim.

- [ ] **Step 1: Write the test file**

Create `benches/tests/surface_code_logical.rs`:

```rust
//! P4-07 acceptance test: surface-code logical/physical error detection.
//!
//! Deterministic, no Stim. From logical |0⟩_L (data |0…0⟩) the Z-stabilizers
//! are deterministic: a single physical X error fires exactly its adjacent
//! Z-ancillas, while a *logical* X̄ (a full data column) fires none — the
//! defining "undetectable" property of a logical operator. The X-basis mirror
//! (|+…+⟩, X-stabilizers, physical/logical Z) is symmetric.

use aleph_backend::Backend;
use aleph_benches::{Ancilla, SurfaceCode};
use aleph_core::{Gate, GateInstance};
use aleph_stab::StabilizerBackend;

/// Measure every ancilla after applying `pre` (errors/logicals) then one cycle,
/// from the all-|0⟩ start. Returns outcomes keyed by ancilla index.
fn syndrome(sc: &SurfaceCode, seed: u64, pre: &[GateInstance]) -> std::collections::HashMap<u32, bool> {
    let mut be = StabilizerBackend::with_seed(seed);
    let mut t = be.allocate(sc.num_qubits as u32).unwrap();
    for g in pre {
        be.apply_gate(&mut t, g).unwrap();
    }
    for g in sc.cycle_gates() {
        be.apply_gate(&mut t, &g).unwrap();
    }
    sc.ancilla_order()
        .iter()
        .map(|&a| (a, be.measure(&mut t, a).unwrap()))
        .collect()
}

fn z_ancillas(sc: &SurfaceCode) -> Vec<&Ancilla> {
    sc.ancillas.iter().filter(|a| !a.is_x).collect()
}
fn x_ancillas(sc: &SurfaceCode) -> Vec<&Ancilla> {
    sc.ancillas.iter().filter(|a| a.is_x).collect()
}

#[test]
fn physical_x_error_fires_adjacent_z_ancillas() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        // Pick an interior data qubit so it has ≥1 Z-neighbour.
        let q = (d / 2 * d + d / 2) as u32;
        let pre = vec![GateInstance::new(Gate::X, vec![q])];
        let s = syndrome(&sc, 0, &pre);
        let expected_fired: std::collections::HashSet<u32> = z_ancillas(&sc)
            .iter()
            .filter(|a| a.data_neighbours.contains(&q))
            .map(|a| a.index)
            .collect();
        assert!(!expected_fired.is_empty(), "d={d}: chosen qubit has no Z-neighbour");
        for a in z_ancillas(&sc) {
            let want = expected_fired.contains(&a.index);
            assert_eq!(s[&a.index], want, "d={d}: Z-ancilla {} fired={:?}, want {want}", a.index, s[&a.index]);
        }
    }
}

#[test]
fn logical_x_is_undetectable() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        let pre: Vec<GateInstance> =
            sc.logical_x.iter().map(|&q| GateInstance::new(Gate::X, vec![q])).collect();
        let s = syndrome(&sc, 0, &pre);
        for a in z_ancillas(&sc) {
            assert!(!s[&a.index], "d={d}: logical X̄ fired Z-ancilla {}", a.index);
        }
    }
}

#[test]
fn physical_z_error_fires_adjacent_x_ancillas() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        let q = (d / 2 * d + d / 2) as u32;
        // Prepare |+…+⟩ via H on all data, then inject Z, then cycle.
        let mut pre: Vec<GateInstance> =
            sc.data.iter().map(|&dq| GateInstance::new(Gate::H, vec![dq])).collect();
        pre.push(GateInstance::new(Gate::Z, vec![q]));
        let s = syndrome(&sc, 0, &pre);
        let expected: std::collections::HashSet<u32> = x_ancillas(&sc)
            .iter()
            .filter(|a| a.data_neighbours.contains(&q))
            .map(|a| a.index)
            .collect();
        assert!(!expected.is_empty(), "d={d}: chosen qubit has no X-neighbour");
        for a in x_ancillas(&sc) {
            let want = expected.contains(&a.index);
            assert_eq!(s[&a.index], want, "d={d}: X-ancilla {} fired={:?}, want {want}", a.index, s[&a.index]);
        }
    }
}

#[test]
fn logical_z_is_undetectable() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        let mut pre: Vec<GateInstance> =
            sc.data.iter().map(|&dq| GateInstance::new(Gate::H, vec![dq])).collect();
        for &q in &sc.logical_z {
            pre.push(GateInstance::new(Gate::Z, vec![q]));
        }
        let s = syndrome(&sc, 0, &pre);
        for a in x_ancillas(&sc) {
            assert!(!s[&a.index], "d={d}: logical Z̄ fired X-ancilla {}", a.index);
        }
    }
}
```

- [ ] **Step 2: Run; verify the tests pass**

Run: `cargo test -p aleph-benches --test surface_code_logical`
Expected: 4 tests PASS. If `physical_x_error_fires_adjacent_z_ancillas` fails because the centre qubit happens to have no Z-neighbour at small d, switch `q` to a data qubit known to border a Z-plaquette (e.g. iterate `sc.data` for the first with a Z-neighbour) — but with the column-0/row-0 logicals and centre pick this holds for d=3,5.

- [ ] **Step 3: Commit**

```bash
git add benches/tests/surface_code_logical.rs
git commit -m "[P4-07] Logical/physical detection gate (deterministic, no Stim)"
```

---

## Task 4: Stim program emission + committed `.stim` corpus

**Files:**
- Modify: `benches/src/lib.rs` (`cycle_stim_gates`, `surface_code_stim_program`)
- Create: `benches/src/bin/surface_dump.rs`
- Create: `scripts/surface_code/circuits/surface_d{3,5,7,9,11}.stim`
- Modify: `benches/Cargo.toml` (`[[bin]] surface_dump`)

- [ ] **Step 1: Write the failing test (program shape)**

Add to `surface_tests` in `benches/src/lib.rs`:

```rust
    #[test]
    fn stim_program_has_one_m_per_ancilla() {
        for d in [3usize, 5] {
            let sc = SurfaceCode::new(d);
            let prog = surface_code_stim_program(d);
            let m_targets: usize = prog
                .lines()
                .filter(|l| l.starts_with("M "))
                .map(|l| l.split_whitespace().count() - 1)
                .sum();
            assert_eq!(m_targets, sc.ancillas.len(), "d={d}: M target count");
            // Gate-only form has no M lines.
            assert!(!cycle_stim_gates(d).lines().any(|l| l.starts_with("M ")));
        }
    }
```

- [ ] **Step 2: Run; verify failure**

Run: `cargo test -p aleph-benches --lib surface_tests::stim_program 2>&1 | head`
Expected: compile error — `surface_code_stim_program` not found.

- [ ] **Step 3: Implement the emitters**

Append to `benches/src/lib.rs` (free functions):

```rust
/// The cycle's gates as a Stim program (H/CX only, no measurements). Targets
/// match `SurfaceCode::cycle_gates`. Used by the postselect oracle.
#[must_use]
pub fn cycle_stim_gates(d: usize) -> String {
    let sc = SurfaceCode::new(d);
    let mut s = String::new();
    for g in sc.cycle_gates() {
        let q = &g.qubits;
        match g.gate {
            Gate::H => s.push_str(&format!("H {}\n", q[0])),
            Gate::Cnot => s.push_str(&format!("CX {} {}\n", q[0], q[1])),
            _ => unreachable!("cycle uses only H and CX"),
        }
    }
    s
}

/// Full Stim program for one cycle: gates followed by a single `M` over all
/// ancillas in `ancilla_order()`. Used to time Stim and as committed corpus.
#[must_use]
pub fn surface_code_stim_program(d: usize) -> String {
    let sc = SurfaceCode::new(d);
    let mut s = cycle_stim_gates(d);
    let targets: Vec<String> = sc.ancilla_order().iter().map(|a| a.to_string()).collect();
    s.push_str(&format!("M {}\n", targets.join(" ")));
    s
}
```

- [ ] **Step 4: Run; verify pass**

Run: `cargo test -p aleph-benches --lib surface_tests::stim_program`
Expected: PASS.

- [ ] **Step 5: Write the dumper bin**

Create `benches/src/bin/surface_dump.rs`:

```rust
//! Dump the Stim program for one surface-code cycle to stdout. Used to
//! regenerate the committed timing corpus under
//! `scripts/surface_code/circuits/surface_d{d}.stim`.
//!
//!   cargo run -q -p aleph-benches --bin surface_dump -- 5 > surface_d5.stim

fn main() {
    let d: usize = std::env::args()
        .nth(1)
        .expect("usage: surface_dump <distance>")
        .parse()
        .expect("distance must be a positive odd integer");
    print!("{}", aleph_benches::surface_code_stim_program(d));
}
```

Register in `benches/Cargo.toml` (after the existing `[[bin]] oneshot`):

```toml
[[bin]]
name = "surface_dump"
path = "src/bin/surface_dump.rs"
```

- [ ] **Step 6: Generate and commit the corpus**

```bash
mkdir -p scripts/surface_code/circuits
for d in 3 5 7 9 11; do
  cargo run -q -p aleph-benches --bin surface_dump -- $d > scripts/surface_code/circuits/surface_d$d.stim
done
head -5 scripts/surface_code/circuits/surface_d3.stim
wc -l scripts/surface_code/circuits/surface_d*.stim
```
Expected: each file starts with `H …` / `CX … …` lines and ends with one `M …` line.

- [ ] **Step 7: Commit**

```bash
git add benches/src/lib.rs benches/src/bin/surface_dump.rs benches/Cargo.toml scripts/surface_code/circuits/
git commit -m "[P4-07] Stim program emission + committed .stim corpus + surface_dump bin"
```

---

## Task 5: Stim group-equivalence oracle (`#[ignore]`)

**Files:**
- Create: `benches/tests/surface_code_stim_oracle.rs`

Postselected canonical-stabilizer-group equivalence, structurally identical to `crates/aleph-stab/tests/stim_measure_oracle.rs` but parametrised over d and over the full ancilla set.

- [ ] **Step 1: Write the oracle test file**

Create `benches/tests/surface_code_stim_oracle.rs`:

```rust
//! P4-07 oracle: surface-code cycle post-state matches Stim. Run our cycle,
//! collect ancilla outcomes b[], postselect Stim's ancillas to b[] in the same
//! order, compare canonical stabilizer groups. Requires python3 + stim;
//! `#[ignore]`d (run on the EPYC oracle venv):
//!
//!   cargo test -p aleph-benches --test surface_code_stim_oracle -- --ignored
//!
//! Comparison is sign-and-generator canonical (sorted set), not row-order
//! sensitive. Mirrors crates/aleph-stab/tests/stim_measure_oracle.rs.

use aleph_backend::Backend;
use aleph_benches::{cycle_stim_gates, SurfaceCode};
use aleph_core::Pauli;
use aleph_stab::{StabilizerBackend, Tableau};
use std::process::Command;

/// Our post-cycle stabilizer generators in Stim "+XZ_Y" format.
fn ours_generators(t: &Tableau, n: usize) -> Vec<String> {
    t.stabilizers()
        .iter()
        .map(|p| {
            let mut chars = vec![b'_'; n];
            for (q, pauli) in &p.terms {
                chars[*q as usize] = match pauli {
                    Pauli::I => b'_',
                    Pauli::X => b'X',
                    Pauli::Y => b'Y',
                    Pauli::Z => b'Z',
                };
            }
            let sign = if p.coefficient < 0.0 { '-' } else { '+' };
            format!("{sign}{}", String::from_utf8(chars).unwrap())
        })
        .collect()
}

/// Run our cycle from |0…0⟩, return (ancilla outcomes in order, our generators).
fn run_ours(sc: &SurfaceCode, seed: u64) -> (Vec<bool>, Vec<String>) {
    let mut be = StabilizerBackend::with_seed(seed);
    let mut t = be.allocate(sc.num_qubits as u32).unwrap();
    for g in sc.cycle_gates() {
        be.apply_gate(&mut t, &g).unwrap();
    }
    let outcomes: Vec<bool> = sc
        .ancilla_order()
        .iter()
        .map(|&a| be.measure(&mut t, a).unwrap())
        .collect();
    let gens = ours_generators(&t, sc.num_qubits);
    (outcomes, gens)
}

/// Returns (ref_canon, ours_canon) or None if the helper failed.
fn stim_canonical(
    d: usize,
    order: &[u32],
    outcomes: &[bool],
    ours: &[String],
) -> Option<(Vec<String>, Vec<String>)> {
    // stdin layout: gates --- "<a0> <b0>\n<a1> <b1>\n…" --- ours generators.
    let py = r#"
import sys, stim
parts = sys.stdin.read().split("---\n")
prog = parts[0]
post = [l for l in parts[1].splitlines() if l]
ours = [l for l in parts[2].splitlines() if l]
sim = stim.TableauSimulator()
sim.do(stim.Circuit(prog))
for line in post:
    a, b = line.split()
    sim.postselect_z(int(a), desired_value=(b == "1"))
ref = stim.Tableau.from_stabilizers(
    sim.canonical_stabilizers(), allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
oursc = stim.Tableau.from_stabilizers(
    [stim.PauliString(s) for s in ours], allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
print("\n".join(str(p) for p in ref))
print("===")
print("\n".join(str(p) for p in oursc))
"#;
    let mut input = cycle_stim_gates(d);
    input.push_str("---\n");
    for (a, b) in order.iter().zip(outcomes) {
        input.push_str(&format!("{a} {}\n", if *b { 1 } else { 0 }));
    }
    input.push_str("---\n");
    input.push_str(&ours.join("\n"));

    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take()?.write_all(input.as_bytes()).ok()?;
            child.wait_with_output().ok()
        })?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut it = text.split("===");
    let refs: Vec<String> = it.next()?.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let oursc: Vec<String> = it.next()?.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    Some((refs, oursc))
}

#[test]
#[ignore = "requires python3 + stim; run on the EPYC oracle venv"]
fn surface_cycle_matches_stim() {
    for d in [3usize, 5, 7, 9, 11] {
        let sc = SurfaceCode::new(d);
        let (outcomes, ours) = run_ours(&sc, 0xC0FFEE ^ d as u64);
        let order = sc.ancilla_order();
        let (refs, oursc) = match stim_canonical(d, &order, &outcomes, &ours) {
            Some(v) => v,
            None => panic!("stim helper failed at d={d} (is `stim` installed in python3?)"),
        };
        let mut a = refs.clone();
        let mut b = oursc.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b, "d={d}: post-cycle stabilizer group disagrees with Stim");
    }
}
```

- [ ] **Step 2: Verify it compiles (and is skipped by default)**

Run: `cargo test -p aleph-benches --test surface_code_stim_oracle`
Expected: builds; reports the test as `ignored` (0 run). Full run happens on EPYC in Task 8.

- [ ] **Step 3: Commit**

```bash
git add benches/tests/surface_code_stim_oracle.rs
git commit -m "[P4-07] Stim postselected group-equivalence oracle (ignored)"
```

---

## Task 6: Criterion bench (aleph time per cycle)

**Files:**
- Create: `benches/benches/phase4_surface_code.rs`
- Modify: `benches/Cargo.toml` (`[[bench]] phase4_surface_code`)

- [ ] **Step 1: Write the bench**

Create `benches/benches/phase4_surface_code.rs`:

```rust
//! P4-07 surface-code cycle timing on the stabilizer backend. One rotated
//! surface-code syndrome-extraction cycle per distance d ∈ {3,5,7,9,11}
//! (2d²−1 qubits, up to 241 at d=11). The aleph half of the
//! `docs/perf/surface_code.md` report row (baseline = Stim, timed separately).
//!
//!   cargo bench -p aleph-benches --bench phase4_surface_code
//!
//! `iter_batched` excludes tableau allocation from the timed cycle (apply
//! gates + measure all ancillas).

use aleph_backend::Backend;
use aleph_benches::SurfaceCode;
use aleph_stab::StabilizerBackend;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::hint::black_box;

const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];

fn bench_surface(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface_code");
    for &d in DISTANCES {
        let sc = SurfaceCode::new(d);
        let gates = sc.cycle_gates();
        let order = sc.ancilla_order();
        let n = sc.num_qubits as u32;
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            let mut be = StabilizerBackend::with_seed(0);
            b.iter_batched(
                || be.allocate(n).unwrap(),
                |mut t| {
                    for g in &gates {
                        be.apply_gate(&mut t, g).unwrap();
                    }
                    let mut acc = false;
                    for &a in &order {
                        acc ^= be.measure(&mut t, a).unwrap();
                    }
                    black_box(acc)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_surface);
criterion_main!(benches);
```

Note: `be` is borrowed mutably by both the setup and routine closures of `iter_batched`. If the borrow checker rejects sharing `be` across both closures, construct the backend inside the routine instead and move `allocate` into setup as `|| n` returning the qubit count, allocating + applying + measuring in the routine (this includes allocation in the timed path — acceptable, allocation is O(n²) and tiny relative to the cycle; document the choice in the bench header if you take this route).

- [ ] **Step 2: Register the bench in `benches/Cargo.toml`**

After the other `[[bench]]` entries:

```toml
[[bench]]
name = "phase4_surface_code"
harness = false
```

- [ ] **Step 3: Run a quick bench smoke test**

Run: `cargo bench -p aleph-benches --bench phase4_surface_code -- --quick 2>&1 | tail -20`
Expected: criterion runs groups `surface_code/3 … surface_code/11`, all complete without panic. (If `--quick` is unsupported on the pinned criterion, run without it and Ctrl-C after the first sizes report.)

- [ ] **Step 4: Commit**

```bash
git add benches/benches/phase4_surface_code.rs benches/Cargo.toml
git commit -m "[P4-07] Criterion surface-code cycle timing bench (d=3..11)"
```

---

## Task 7: Report tooling (Stim timing + render + golden)

**Files:**
- Create: `scripts/surface_code/stim_timing.py`
- Create: `scripts/surface_code/render_report.py`
- Create: `scripts/surface_code/test_render.py`
- Create: `scripts/surface_code/testdata/` golden fixtures

Mirrors `scripts/qaoa/{render_report.py,test_render.py}` (stdlib `unittest`, deterministic pure render).

- [ ] **Step 1: Write the Stim timing script**

Create `scripts/surface_code/stim_timing.py`:

```python
"""Time one Stim surface-code cycle per committed .stim file. Writes
surface-stim.json: {"workloads": {"3": {"d":3,"qubits":17,"median_s":...}, ...}}.
Single-thread; median of N runs (default 20)."""
import argparse
import json
import statistics
import time
from pathlib import Path

import stim


def time_one(path: Path, runs: int) -> float:
    circuit = stim.Circuit(path.read_text())
    samples = []
    for _ in range(runs):
        sim = stim.TableauSimulator()
        t0 = time.perf_counter()
        sim.do(circuit)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=Path, default=Path(__file__).parent / "circuits")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()
    workloads = {}
    for d in [3, 5, 7, 9, 11]:
        p = args.circuits / f"surface_d{d}.stim"
        median = time_one(p, args.runs)
        workloads[str(d)] = {"d": d, "qubits": 2 * d * d - 1, "median_s": median}
    args.out.write_text(json.dumps({"workloads": workloads}, indent=2) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the render script**

Create `scripts/surface_code/render_report.py`:

```python
"""Render docs/perf/surface_code.md from aleph + Stim JSON (deterministic, pure).
aleph JSON: {"workloads": {"3": {"d":3,"median_s":...}, ...}}.
Stim JSON: {"workloads": {"3": {"d":3,"qubits":17,"median_s":...}, ...}}."""
import argparse
import json
from pathlib import Path

CAVEAT = (
    "_Single-thread both sides. aleph = `StabilizerBackend` (CHP tableau, "
    "O(n²)); baseline = Stim `TableauSimulator`. One rotated surface-code "
    "syndrome-extraction cycle (2d²−1 qubits). **Stim is a purpose-built, "
    "heavily-optimised stabilizer simulator; aleph is expected to be slower "
    "per cycle** — the deliverable is correctness parity (postselected "
    "stabilizer-group equivalence, `surface_code_stim_oracle`) plus a "
    "documented time-per-cycle row, not beating Stim._"
)


def render(aleph: dict, stim: dict, meta: dict) -> str:
    out = [
        "# Phase 4 — Surface-code syndrome extraction (stabilizer)\n",
        "> Auto-generated by `scripts/surface_code/render_report.py`. Do not edit by hand.\n",
        f"**Date:** {meta['date']}  ",
        f"**Host:** {meta['host']}  ",
        f"**Toolchain:** {meta['toolchain']}\n",
        CAVEAT,
        "",
        "## Time per cycle\n",
        "| d | qubits | aleph (ms) | Stim (ms) | aleph / Stim |",
        "|--:|-------:|-----------:|----------:|-------------:|",
    ]
    for d in [3, 5, 7, 9, 11]:
        k = str(d)
        a = aleph["workloads"].get(k)
        s = stim["workloads"].get(k)
        if not a or not s:
            continue
        a_ms = a["median_s"] * 1000.0
        s_ms = s["median_s"] * 1000.0
        ratio = a_ms / s_ms if s_ms else float("nan")
        out.append(f"| {d} | {s['qubits']} | {a_ms:.3f} | {s_ms:.3f} | {ratio:.2f}× |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aleph", required=True, type=Path)
    ap.add_argument("--stim", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    aleph = json.loads(args.aleph.read_text())
    stim = json.loads(args.stim.read_text())
    meta = json.loads(args.meta.read_text())
    args.out.write_text(render(aleph, stim, meta))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the golden test + fixtures**

Create `scripts/surface_code/testdata/aleph.json`:

```json
{"workloads": {"3": {"d": 3, "median_s": 0.00001}, "5": {"d": 5, "median_s": 0.00004}, "7": {"d": 7, "median_s": 0.00012}, "9": {"d": 9, "median_s": 0.00030}, "11": {"d": 11, "median_s": 0.00065}}}
```

Create `scripts/surface_code/testdata/stim.json`:

```json
{"workloads": {"3": {"d": 3, "qubits": 17, "median_s": 0.000002}, "5": {"d": 5, "qubits": 49, "median_s": 0.000006}, "7": {"d": 7, "qubits": 97, "median_s": 0.000013}, "9": {"d": 9, "qubits": 161, "median_s": 0.000028}, "11": {"d": 11, "qubits": 241, "median_s": 0.000055}}}
```

Create `scripts/surface_code/testdata/meta.json`:

```json
{"date": "2026-01-01", "host": "test", "toolchain": "test"}
```

Create `scripts/surface_code/test_render.py`:

```python
"""Golden test for render_report.render (stdlib unittest)."""
import json
import unittest
from pathlib import Path

import render_report

HERE = Path(__file__).parent
TD = HERE / "testdata"


class RenderTest(unittest.TestCase):
    def test_render_golden(self):
        aleph = json.loads((TD / "aleph.json").read_text())
        stim = json.loads((TD / "stim.json").read_text())
        meta = json.loads((TD / "meta.json").read_text())
        md = render_report.render(aleph, stim, meta)
        # Structural assertions (robust to float formatting).
        self.assertIn("# Phase 4 — Surface-code syndrome extraction (stabilizer)", md)
        self.assertIn("| 11 | 241 |", md)
        self.assertIn("aleph / Stim", md)
        # d=3: 0.01 ms aleph / 0.002 ms stim = 5.00×
        self.assertIn("| 3 | 17 | 0.010 | 0.002 | 5.00× |", md)
        # All five distances present.
        for d in (3, 5, 7, 9, 11):
            self.assertRegex(md, rf"\n\| {d} \|")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the golden test**

Run: `cd scripts/surface_code && python3 -m unittest test_render -v; cd -`
Expected: 1 test PASS. (Adjust the exact `| 3 | 17 | … |` assertion string if your formatting differs — compute `0.00001*1000=0.010`, `0.000002*1000=0.002`, ratio `5.00×`.)

- [ ] **Step 5: Commit**

```bash
git add scripts/surface_code/
git commit -m "[P4-07] Surface-code report tooling: Stim timing + render + golden"
```

---

## Task 8: Local full-suite check, EPYC measurement, report, ACs

**Files:**
- Create: `docs/perf/surface_code.md` (EPYC numbers)
- Create: `docs/perf/data/surface-aleph.json`, `docs/perf/data/surface-stim.json`, `docs/perf/data/surface-meta.json`
- Modify: `BACKLOG.md` (check the four P4-07 ACs)

- [ ] **Step 1: Local clippy/fmt/test gate**

Run:
```bash
cargo fmt --check
cargo clippy -p aleph-benches --all-targets -- -D warnings
cargo test -p aleph-benches --lib surface_tests
cargo test -p aleph-benches --test surface_code_logical
cargo test -p aleph-benches --test surface_code_stim_oracle   # builds, ignored
```
Expected: all green; oracle reports `ignored`.

- [ ] **Step 2: EPYC — build, run the Stim oracle, measure both sides**

Per [[phase4-status]] / [[aleph_bench_server]] ops notes. Transfer via git bundle into the reusable `/tmp/aleph-p114/aleph` checkout (keeps the qiskit/stim venv + cargo registry); ensure `stim` is in the venv (`uv pip install --python <venv>/bin/python stim`). Set cargo PATH: `export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH`. Verify the box is idle (`uptime`, no competing `cargo bench`) before timing.

```bash
# correctness vs Stim (all distances)
RUSTFLAGS="-C target-cpu=native" \
  cargo test -p aleph-benches --test surface_code_stim_oracle -- --ignored --nocapture
# aleph timing
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench -p aleph-benches --bench phase4_surface_code
# extract aleph medians -> surface-aleph.json (per d, from target/criterion/surface_code/<d>/new/estimates.json)
python3 scripts/bench-report/extract_criterion.py ...   # or a thin inline reader; group="surface_code"
# Stim timing
<venv>/bin/python scripts/surface_code/stim_timing.py --out docs/perf/data/surface-stim.json
```
Expected: oracle prints PASS for d=3,5,7,9,11; criterion + stim JSON produced.

Note: `extract_criterion.py` emits a `phase4-aleph.json`-style structure keyed by `workload_key`; the surface group key is the bare distance (`3`,`5`,…). If its schema doesn't match `render_report.py`'s `{"workloads": {"<d>": {"median_s": …}}}`, write a ~15-line reader inline that walks `target/criterion/surface_code/<d>/new/estimates.json` (`["median"]["point_estimate"]` is in **ns**) → `surface-aleph.json`. Prefer this small reader over bending `extract_criterion.py`.

- [ ] **Step 3: Render the report**

```bash
# surface-meta.json: {"date":"2026-06-..","host":"AMD EPYC 8124P …","toolchain":"rustc … RAYON_NUM_THREADS=1; stim <ver>"}
python3 scripts/surface_code/render_report.py \
  --aleph docs/perf/data/surface-aleph.json \
  --stim docs/perf/data/surface-stim.json \
  --meta docs/perf/data/surface-meta.json \
  --out docs/perf/surface_code.md
cat docs/perf/surface_code.md
```
Expected: the time-per-cycle table with d=3..11, honest aleph/Stim ratios (likely > 1× — aleph slower, as the caveat states).

- [ ] **Step 4: Check the BACKLOG ACs**

Edit `BACKLOG.md` P4-07 section: tick `[x]` for "Cycles run to d = 11", "Match Stim output", "Benchmark report row", and the logical X/Z testing requirement.

- [ ] **Step 5: Commit + push + PR**

```bash
git add docs/perf/surface_code.md docs/perf/data/surface-*.json BACKLOG.md
git commit -m "[P4-07] Surface-code report (EPYC, vs Stim) + tick ACs"
git push -u origin p4-07-surface-code
gh pr create --title "[P4-07] Surface code 1-cycle benchmark (stabilizer)" --body "<see below>"
```

PR body must include: `Closes #45` (the **issue** number, not the PR), approach summary, test results (logical gate + Stim oracle PASS at all d), the surface_code.md table, and the honesty caveat that aleph is slower than Stim by design.

---

## Self-review notes

- **Spec coverage:** circuit source (Task 1–2), Stim match (Task 5), report row (Task 6–8), logical X/Z detection (Task 3), d=11/241 qubits (Task 1 counts test + bench), dedicated `surface_code.md` (Task 7–8), no `run.py`/`report.py` changes (none touched). ✓
- **Type consistency:** `SurfaceCode { distance, num_qubits, data, ancillas, logical_x, logical_z }`, `Ancilla { index, is_x, data_neighbours }`, methods `new`/`cycle_gates`/`ancilla_order`, free fns `cycle_stim_gates`/`surface_code_stim_program` — used identically across Tasks 1–7. ✓
- **`iter_batched` borrow caveat** is flagged in Task 6 Step 1 with a concrete fallback.
- **Geometry risk** is contained by the Task 1 invariant tests (commutation + counts + logical (anti)commutation) — the tests, not the prose, define correctness.
```
