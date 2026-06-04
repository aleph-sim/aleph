# P3-02 Stabilizer Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add projective Z-basis measurement with collapse to the `aleph-stab` `Tableau` (Aaronson-Gottesman §3): the `g` phase-exponent helper, the `rowsum` sign-tracking primitive, and `measure<R: rand::Rng>(qubit, rng)` with the deterministic/random case split.

**Architecture:** All logic lands in `crates/aleph-stab/src/tableau.rs` on the existing `Tableau` (rows `0..n` destabilizers, `n..2n` stabilizers, row `2n` scratch over packed-`u64` `BitGrid` x/z grids + `sign: Vec<bool>`). `rowsum` uses the scratch row P3-01 reserved. Randomness is an injected `rand::Rng` (mirrors `aleph-sv`). Correctness is gated by Bell/GHZ unit tests, a symplectic proptest, and a Stim postselect oracle.

**Tech Stack:** Rust 2021, `aleph-core`, `rand` 0.8 (workspace dep), `proptest`, Python `stim` 1.16 (EPYC oracle, venv at `/root/stim312`).

**Reference:** Aaronson & Gottesman 2004, §2 (`g`, `rowsum`) and §3 (measurement). Spec: `docs/superpowers/specs/2026-06-04-p3-02-stabilizer-measurement-design.md`.

---

## File Structure

| File | Change |
|------|--------|
| `crates/aleph-stab/Cargo.toml` | add `rand = { workspace = true }` to `[dependencies]` |
| `crates/aleph-stab/src/tableau.rs` | add `g`, `rowsum`, `copy_row`, `zero_row`, `measure` + unit/statistical tests |
| `crates/aleph-stab/tests/properties.rs` | add a symplectic-after-measurement proptest |
| `crates/aleph-stab/tests/stim_measure_oracle.rs` | new: Stim postselect equivalence (`#[ignore]`) |

> Conventions (CLAUDE.md): no `unwrap`/`expect` in library code (tests OK); no `unsafe`; clippy `-D warnings`; `cargo fmt`; rustdoc on public items; cite AG §2/§3 in comments. `BitGrid` has **no** `toggle` (removed in P3-01) — use `get`/`set` for bit XOR.

---

## Task 1: `rand` dependency + `g` phase-exponent helper

**Files:**
- Modify: `crates/aleph-stab/Cargo.toml`
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Add the `rand` dependency**

In `crates/aleph-stab/Cargo.toml`, under `[dependencies]`, after the `thiserror` line, add:

```toml
rand       = { workspace = true }
```

(Result: `[dependencies]` lists `aleph-core`, `thiserror`, `rand`.)

- [ ] **Step 2: Write the failing test for `g`**

Add to the existing `#[cfg(test)] mod tests` block in `tableau.rs`:

```rust
    #[test]
    fn g_phase_exponent_table() {
        use super::g;
        // (x1,z1)=(0,0) → always 0
        for &(x2, z2) in &[(false, false), (true, false), (false, true), (true, true)] {
            assert_eq!(g(false, false, x2, z2), 0);
        }
        // (x1,z1)=(1,0): z2*(2*x2-1)
        assert_eq!(g(true, false, false, false), 0); // z2=0
        assert_eq!(g(true, false, true, false), 0); // z2=0
        assert_eq!(g(true, false, false, true), -1); // z2=1,x2=0 → 1*(−1)
        assert_eq!(g(true, false, true, true), 1); // z2=1,x2=1 → 1*(1)
        // (x1,z1)=(0,1): x2*(1-2*z2)
        assert_eq!(g(false, true, false, false), 0); // x2=0
        assert_eq!(g(false, true, true, false), 1); // x2=1,z2=0 → 1*(1)
        assert_eq!(g(false, true, true, true), -1); // x2=1,z2=1 → 1*(−1)
        // (x1,z1)=(1,1): z2 - x2
        assert_eq!(g(true, true, false, false), 0);
        assert_eq!(g(true, true, true, false), -1); // 0-1
        assert_eq!(g(true, true, false, true), 1); // 1-0
        assert_eq!(g(true, true, true, true), 0); // 1-1
    }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib g_phase_exponent_table`
Expected: FAIL — `g` not defined.

- [ ] **Step 4: Implement `g`**

Add as a free function in `tableau.rs` (module level, e.g. just above `impl Tableau` — `g` needs no `self`). Place it after the `use` lines:

```rust
/// Aaronson-Gottesman §2 phase exponent: the power of `i` introduced when
/// the single-qubit Pauli `(x1,z1)` is left-multiplied onto `(x2,z2)`.
/// Returns a value in `{-1, 0, 1}`. Used by [`Tableau::rowsum`].
fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
    let x2 = x2 as i32;
    let z2 = z2 as i32;
    match (x1, z1) {
        (false, false) => 0,
        (true, false) => z2 * (2 * x2 - 1),
        (false, true) => x2 * (1 - 2 * z2),
        (true, true) => z2 - x2,
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p aleph-stab --lib g_phase_exponent_table`
Expected: PASS.

- [ ] **Step 6: Verify crate builds + lints**

Run: `cargo build -p aleph-stab && cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-stab/Cargo.toml crates/aleph-stab/src/tableau.rs
git commit -m "[P3-02] rand dep + AG g phase-exponent helper"
```

---

## Task 2: `rowsum` + `copy_row` + `zero_row`

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the failing test**

Add to `tableau.rs` `mod tests`:

```rust
    // rowsum(h,i) does x_h ^= x_i, z_h ^= z_i (XOR involution): applying
    // it twice restores row h's bits. Sign tracking is exercised more
    // thoroughly by the measurement + Stim oracle; here we pin the bit
    // involution and that the sign stays in {false,true} (no panic on the
    // mod-4 debug_assert) over a generic state.
    #[test]
    fn rowsum_bit_involution() {
        let mut t = generic_state(); // 3-qubit entangled Clifford state
        // snapshot rows 0 (destab) and 3 (stab) bits
        let snap = |t: &Tableau, r: usize| -> Vec<(bool, bool)> {
            (0..t.num_qubits()).map(|j| (t.x(r, j), t.z(r, j))).collect()
        };
        let before = snap(&t, 0);
        t.rowsum(0, 3);
        t.rowsum(0, 3); // second application cancels the bit XORs
        assert_eq!(snap(&t, 0), before, "rowsum bit XOR is not involutive");
    }

    // copy_row duplicates a full row; zero_row clears it.
    #[test]
    fn copy_and_zero_row() {
        let mut t = generic_state();
        t.copy_row(0, 4); // row 0 (destab) ← row 4 (stab 1)
        for j in 0..t.num_qubits() {
            assert_eq!(t.x(0, j), t.x(4, j));
            assert_eq!(t.z(0, j), t.z(4, j));
        }
        assert_eq!(t.sign(0), t.sign(4));
        t.zero_row(0);
        for j in 0..t.num_qubits() {
            assert!(!t.x(0, j) && !t.z(0, j));
        }
        assert!(!t.sign(0));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-stab --lib rowsum_bit_involution`
Expected: FAIL — `rowsum` not defined.

- [ ] **Step 3: Implement `rowsum`, `copy_row`, `zero_row`**

Add to `impl Tableau` in `tableau.rs`:

```rust
    /// AG §2 `rowsum`: set generator `row h ← (row i)·(row h)`, tracking
    /// the sign. The phase accumulates as `2·r_h + 2·r_i + Σ_j g(...)`
    /// reduced mod 4, which is always 0 or 2 (real `±1`) for a product of
    /// two Pauli generators.
    fn rowsum(&mut self, h: usize, i: usize) {
        let mut acc: i32 = 2 * self.sign[h] as i32 + 2 * self.sign[i] as i32;
        for j in 0..self.n {
            acc += g(self.x.get(i, j), self.z.get(i, j), self.x.get(h, j), self.z.get(h, j));
        }
        let m = acc.rem_euclid(4);
        debug_assert!(m == 0 || m == 2, "rowsum phase {m} not in {{0, 2}}");
        self.sign[h] = m == 2;
        for j in 0..self.n {
            let xh = self.x.get(h, j) ^ self.x.get(i, j);
            let zh = self.z.get(h, j) ^ self.z.get(i, j);
            self.x.set(h, j, xh);
            self.z.set(h, j, zh);
        }
    }

    /// Copy a full generator row (x bits, z bits, sign) from `src` to `dst`.
    fn copy_row(&mut self, dst: usize, src: usize) {
        for j in 0..self.n {
            self.x.set(dst, j, self.x.get(src, j));
            self.z.set(dst, j, self.z.get(src, j));
        }
        self.sign[dst] = self.sign[src];
    }

    /// Reset a row to the identity Pauli with `+` sign.
    fn zero_row(&mut self, r: usize) {
        for j in 0..self.n {
            self.x.set(r, j, false);
            self.z.set(r, j, false);
        }
        self.sign[r] = false;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib rowsum_bit_involution copy_and_zero_row`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-02] rowsum sign-tracking primitive + row helpers (AG §2)"
```

---

## Task 3: `measure` — deterministic + random with collapse

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the failing tests**

Add to `tableau.rs` `mod tests`. (Add `use rand::SeedableRng;` and
`use rand::rngs::StdRng;` at the top of the `mod tests` block if not
already imported.)

```rust
    #[test]
    fn measure_zero_state_is_deterministic_zero() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut t = Tableau::new(3);
        for a in 0..3 {
            assert_eq!(t.measure(a, &mut rng).unwrap(), false, "|0> qubit {a}");
        }
        // Out-of-range rejected.
        assert!(t.measure(3, &mut rng).is_err());
    }

    #[test]
    fn measure_bell_forces_correlation() {
        // |Φ+> = (|00>+|11>)/√2: measuring q0 is random; q1 must equal q0.
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..32 {
            let mut t = Tableau::new(2);
            t.h(0).unwrap();
            t.cnot(0, 1).unwrap();
            let b0 = t.measure(0, &mut rng).unwrap();
            let b1 = t.measure(1, &mut rng).unwrap();
            assert_eq!(b0, b1, "Bell correlation broken");
            // Re-measuring q0 after collapse returns the same value.
            let b0_again = t.measure(0, &mut rng).unwrap();
            assert_eq!(b0, b0_again, "post-collapse determinism broken");
        }
    }

    #[test]
    fn measure_plus_state_is_random() {
        // H|0> = |+>: measuring in Z is random; over many seeds we should
        // see both outcomes.
        let mut saw_false = false;
        let mut saw_true = false;
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut t = Tableau::new(1);
            t.h(0).unwrap();
            match t.measure(0, &mut rng).unwrap() {
                false => saw_false = true,
                true => saw_true = true,
            }
        }
        assert!(saw_false && saw_true, "|+> measurement never produced both outcomes");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p aleph-stab --lib measure_`
Expected: FAIL — `measure` not defined.

- [ ] **Step 3: Implement `measure`**

Add to `impl Tableau` in `tableau.rs`:

```rust
    /// Projective Z-basis measurement of qubit `a` with state collapse
    /// (Aaronson-Gottesman §3). Returns the outcome bit (`true` = `|1>`).
    ///
    /// If `Z_a` anticommutes with some stabilizer the outcome is random
    /// (drawn from `rng`) and the tableau collapses accordingly; otherwise
    /// the outcome is determined by the current state. `rng` is consumed
    /// only in the random case.
    pub fn measure<R: rand::Rng>(&mut self, a: usize, rng: &mut R) -> Result<bool, crate::StabError> {
        self.check_qubit(a)?;
        // A stabilizer row anticommuting with Z_a (i.e. with an X/Y on a)
        // ⇒ random outcome.
        let p = (self.n..2 * self.n).find(|&row| self.x.get(row, a));
        match p {
            Some(p) => {
                // Random outcome: eliminate column `a`'s X from every other
                // row, promote p to a destabilizer, install Z_a as the new
                // stabilizer with a random sign.
                for i in 0..2 * self.n {
                    if i != p && self.x.get(i, a) {
                        self.rowsum(i, p);
                    }
                }
                self.copy_row(p - self.n, p);
                self.zero_row(p);
                self.z.set(p, a, true);
                let outcome = rng.gen::<bool>();
                self.sign[p] = outcome;
                Ok(outcome)
            }
            None => {
                // Deterministic outcome: accumulate the relevant stabilizers
                // into the scratch row; its resulting sign is the outcome.
                let scratch = 2 * self.n;
                self.zero_row(scratch);
                for i in 0..self.n {
                    if self.x.get(i, a) {
                        self.rowsum(scratch, i + self.n);
                    }
                }
                Ok(self.sign[scratch])
            }
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p aleph-stab --lib measure_`
Expected: PASS (all three).

- [ ] **Step 5: Full crate gate**

Run: `cargo test -p aleph-stab && cargo clippy -p aleph-stab --all-targets -- -D warnings && cargo fmt -p aleph-stab --check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-02] Projective measurement with collapse (AG §3)"
```

---

## Task 4: Statistical test — GHZ 50/50 + intra-trial agreement

**Files:**
- Modify: `crates/aleph-stab/src/tableau.rs`

- [ ] **Step 1: Write the test**

Add to `tableau.rs` `mod tests`:

```rust
    #[test]
    fn measure_ghz_is_balanced_and_consistent() {
        // GHZ_n = (|0…0> + |1…1>)/√2. Each trial: all qubits agree; over
        // many trials q0 is ~50/50.
        const N: usize = 5;
        const TRIALS: u32 = 4000;
        let mut ones = 0u32;
        for seed in 0..TRIALS as u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut t = Tableau::new(N);
            t.h(0).unwrap();
            for i in 0..N - 1 {
                t.cnot(i, i + 1).unwrap();
            }
            let b0 = t.measure(0, &mut rng).unwrap();
            for q in 1..N {
                assert_eq!(t.measure(q, &mut rng).unwrap(), b0, "GHZ qubit {q} disagreed");
            }
            if b0 {
                ones += 1;
            }
        }
        // Binomial(4000, 0.5): mean 2000, sd ≈ 31.6. ±5 sd ≈ ±158 → [1842,2158].
        assert!(
            (1842..=2158).contains(&ones),
            "GHZ q0 balance out of range: {ones}/4000 ones"
        );
    }
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test -p aleph-stab --lib measure_ghz_is_balanced_and_consistent`
Expected: PASS. (Deterministic — fixed seeds; the band is wide enough that a correct implementation never flakes.)

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-stab/src/tableau.rs
git commit -m "[P3-02] Statistical GHZ measurement test (50/50 + agreement)"
```

---

## Task 5: Symplectic-invariant-after-measurement proptest

**Files:**
- Modify: `crates/aleph-stab/tests/properties.rs`

- [ ] **Step 1: Add the proptest**

`properties.rs` already has the `Op` enum, `op_strategy`, `apply`, and a
`symplectic_invariant_preserved` test from P3-01, plus a top-level
`use proptest::prelude::*;`. Add `use rand::SeedableRng;` near the top,
then append this new `proptest! { ... }` block at the end of the file:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Measurement leaves a well-formed tableau: after a random Clifford
    /// circuit and one measurement, the destabilizer/stabilizer structure
    /// is still symplectic.
    #[test]
    fn symplectic_invariant_preserved_after_measure(
        ops in {
            let n = 6;
            proptest::collection::vec(op_strategy(n), 0..40)
        },
        target in 0usize..6,
        seed in any::<u64>(),
    ) {
        let n = 6;
        let mut t = Tableau::new(n);
        for op in &ops {
            apply(&mut t, op);
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let _ = t.measure(target, &mut rng).unwrap();
        for i in 0..n {
            prop_assert!(t.rows_anticommute(i, n + i), "destab {i} ⊥ stab {i} broken after measure");
            for j in 0..n {
                if j != i {
                    prop_assert!(!t.rows_anticommute(i, n + j));
                    prop_assert!(!t.rows_anticommute(n + i, n + j));
                }
            }
        }
    }
}
```

`Tableau` and `rand` are usable here: `aleph-stab` exports `Tableau`, and
`rand` is now a normal dependency so it is available to integration tests
without a separate dev-dep entry. The invariant check reuses the existing
`pub fn rows_anticommute` from P3-01.

- [ ] **Step 2: Run the proptest**

Run: `cargo test -p aleph-stab --test properties symplectic_invariant_preserved_after_measure`
Expected: PASS (200 cases).

- [ ] **Step 3: Run the whole properties file + commit**

Run: `cargo test -p aleph-stab --test properties`
Expected: both proptests pass.

```bash
git add crates/aleph-stab/tests/properties.rs
git commit -m "[P3-02] Proptest: symplectic invariant survives measurement"
```

---

## Task 6: Stim postselect oracle (`#[ignore]`, EPYC)

**Files:**
- Create: `crates/aleph-stab/tests/stim_measure_oracle.rs`

This mirrors `tests/stim_oracle.rs` (P3-01) but adds a measurement +
Stim postselect step.

- [ ] **Step 1: Write the oracle test**

Create `crates/aleph-stab/tests/stim_measure_oracle.rs`:

```rust
//! Oracle: measurement + collapse equivalence vs Stim. For each random
//! Clifford circuit and target qubit, we measure in our tableau (outcome
//! `b`), then postselect Stim's qubit to `b` and compare post-measurement
//! canonical stabilizer groups. Also cross-checks determinism via Stim
//! `peek_z`. Requires python3 + stim; `#[ignore]`d (run on EPYC):
//!
//!   cargo test -p aleph-stab --test stim_measure_oracle -- --ignored
//!
//! Group comparison is sign-and-generator canonical (sorted set), not
//! row-order sensitive.

use aleph_core::{Gate, GateInstance};
use aleph_stab::{apply_gate, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::process::Command;

const N: usize = 10;
const DEPTH: usize = 25;
const CIRCUITS: usize = 100;

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

fn random_circuit(seed: u64) -> Vec<GateInstance> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for _ in 0..DEPTH {
        for _ in 0..N {
            let q = rng.below(N as u64) as u32;
            match rng.below(7) {
                0 => out.push(GateInstance::new(Gate::H, vec![q])),
                1 => out.push(GateInstance::new(Gate::S, vec![q])),
                2 => out.push(GateInstance::new(Gate::X, vec![q])),
                3 => out.push(GateInstance::new(Gate::Y, vec![q])),
                4 => out.push(GateInstance::new(Gate::Z, vec![q])),
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
    }
    out
}

fn stim_program(circ: &[GateInstance]) -> String {
    let mut s = String::new();
    for g in circ {
        let q = &g.qubits;
        match g.gate {
            Gate::H => s.push_str(&format!("H {}\n", q[0])),
            Gate::S => s.push_str(&format!("S {}\n", q[0])),
            Gate::X => s.push_str(&format!("X {}\n", q[0])),
            Gate::Y => s.push_str(&format!("Y {}\n", q[0])),
            Gate::Z => s.push_str(&format!("Z {}\n", q[0])),
            Gate::Cnot => s.push_str(&format!("CX {} {}\n", q[0], q[1])),
            _ => unreachable!("oracle circuits only use H/S/Paulis/CX"),
        }
    }
    s
}

/// Our post-measurement stabilizer generators in stim "+XZ_Y" format.
fn ours_generators(t: &Tableau) -> Vec<String> {
    t.stabilizers()
        .iter()
        .map(|p| {
            let mut chars = vec![b'_'; N];
            for (q, pauli) in &p.terms {
                chars[*q as usize] = match pauli {
                    aleph_core::Pauli::I => b'_',
                    aleph_core::Pauli::X => b'X',
                    aleph_core::Pauli::Y => b'Y',
                    aleph_core::Pauli::Z => b'Z',
                };
            }
            let sign = if p.coefficient < 0.0 { '-' } else { '+' };
            format!("{sign}{}", String::from_utf8(chars).unwrap())
        })
        .collect()
}

/// Returns `(peek, ref_canon, ours_canon)`: Stim's `peek_z(a)` (+1/-1/0),
/// the reference canonical stabilizers after postselecting `a→b`, and our
/// canonical generators. `None` if the helper failed to run.
fn stim_check(
    circ: &[GateInstance],
    a: usize,
    b: bool,
    ours: &[String],
) -> Option<(i64, Vec<String>, Vec<String>)> {
    // Pinned to stim 1.16. peek_z is read BEFORE postselect (peek does not
    // collapse). We always postselect to OUR outcome b, which has nonzero
    // probability in the same state, so postselect never rejects.
    let py = r#"
import sys, stim
data = sys.stdin.read().split("---\n")
prog = data[0]
meta = data[1].splitlines()
a, b = meta[0].split()
a = int(a); b = (b == "1")
ours = [l for l in meta[1:] if l]
sim = stim.TableauSimulator()
sim.do(stim.Circuit(prog))
peek = sim.peek_z(a)
sim.postselect_z(a, desired_value=b)
ref_canon = stim.Tableau.from_stabilizers(
    sim.canonical_stabilizers(), allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
ours_canon = stim.Tableau.from_stabilizers(
    [stim.PauliString(s) for s in ours], allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
print(peek)
print("===")
print("\n".join(str(p) for p in ref_canon))
print("===")
print("\n".join(str(p) for p in ours_canon))
"#;
    let mut input = stim_program(circ);
    input.push_str("---\n");
    input.push_str(&format!("{a} {}\n", if b { 1 } else { 0 }));
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
    let mut parts = text.split("===");
    let peek: i64 = parts.next()?.trim().parse().ok()?;
    let refs: Vec<String> = parts
        .next()?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let ours_c: Vec<String> = parts
        .next()?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Some((peek, refs, ours_c))
}

#[test]
#[ignore = "requires python3 + stim; run on the EPYC oracle venv"]
fn measurement_matches_stim() {
    let mut failures = 0;
    for k in 0..CIRCUITS {
        let circ = random_circuit(0xBEEF ^ (k as u64).wrapping_mul(0x100000001B3));
        let a = (k * 7) % N; // vary the measured qubit
        let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ k as u64);

        let mut t = Tableau::new(N);
        for g in &circ {
            apply_gate(&mut t, g).unwrap();
        }
        let b = t.measure(a, &mut rng).unwrap();
        let ours = ours_generators(&t);

        let (peek, refs, ours_c) = match stim_check(&circ, a, b, &ours) {
            Some(v) => v,
            None => panic!("stim helper failed (is `stim` installed in the active python3?)"),
        };

        // Determinism cross-check: peek_z == +1 → outcome must be 0(false);
        // -1 → 1(true); 0 → random (no constraint).
        if peek == 1 {
            assert!(!b, "circuit {k}: stim says deterministic |0> but we measured 1");
        } else if peek == -1 {
            assert!(b, "circuit {k}: stim says deterministic |1> but we measured 0");
        }

        let mut x = refs.clone();
        let mut y = ours_c.clone();
        x.sort();
        y.sort();
        if x != y {
            failures += 1;
            eprintln!("circuit {k} (measure q{a}→{b}) post-state mismatch:\n  stim: {x:?}\n  ours: {y:?}");
        }
    }
    assert_eq!(failures, 0, "{failures}/{CIRCUITS} circuits disagreed with Stim");
}
```

> **Executor note (stim version):** like P3-01, the embedded Python is
> pinned to stim 1.16 (`postselect_z(target, desired_value=...)`,
> `peek_z`, `Tableau.to_stabilizers(canonicalize=True)`). If the EPYC
> stim version differs, adjust the helper to its API — the contract is:
> read `peek_z`, postselect to `b`, emit `peek`, reference canonical
> generators, and our canonical generators, in that 3-section format. The
> Rust comparison is stable.

- [ ] **Step 2: Compile-check (ignored locally)**

Run: `cargo test -p aleph-stab --test stim_measure_oracle --no-run`
Expected: compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-stab/tests/stim_measure_oracle.rs
git commit -m "[P3-02] Stim postselect measurement oracle (EPYC-gated)"
```

---

## Task 7: EPYC validation

Validate on the EPYC box (`ssh root@195.154.249.85`). The stim venv from
P3-01 is preserved at `/root/stim312` (stim 1.16). Stabilizer code is
scalar, so x86 correctness is the goal (no perf AC for P3-02).

**Files:** none (validation only; any fix re-commits the relevant file).

- [ ] **Step 1: Local full-workspace gate**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
Expected: all green.

- [ ] **Step 2: Ship branch to EPYC via git bundle**

```bash
git bundle create /tmp/p3-02.bundle p3-02-stabilizer-measurement
scp /tmp/p3-02.bundle root@195.154.249.85:/root/
ssh root@195.154.249.85 'cd /root && rm -rf aleph-p302 && git clone -q /root/p3-02.bundle aleph-p302 && cd aleph-p302 && git checkout -q p3-02-stabilizer-measurement && git log --oneline -3'
```

- [ ] **Step 3: Idle-check, run tests on EPYC (x86 scalar correctness)**

```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" | grep -v pgrep || echo IDLE'
ssh root@195.154.249.85 'export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH; cd /root/aleph-p302; cargo test -p aleph-stab 2>&1 | tail -20'
```
Expected: all aleph-stab tests pass on x86_64.

- [ ] **Step 4: Run the Stim measurement oracle on EPYC**

```bash
ssh root@195.154.249.85 'export PATH=/root/stim312/bin:$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH; cd /root/aleph-p302; python3 -c "import stim; print(stim.__version__)"; cargo test -p aleph-stab --test stim_measure_oracle -- --ignored --nocapture 2>&1 | tail -20'
```
Expected: `0/100 circuits disagreed with Stim`. If the stim API differs,
fix the helper (Task 6 note), re-bundle, re-run.

- [ ] **Step 5: Clean up EPYC disk (keep `/root/stim312`)**

```bash
ssh root@195.154.249.85 'rm -rf /root/aleph-p302 /root/p3-02.bundle; df -h / | tail -1'
```

- [ ] **Step 6: Record the result. If a helper fix was needed, commit it.**

```bash
git add crates/aleph-stab/tests/stim_measure_oracle.rs
git commit -m "[P3-02] Pin measurement oracle to installed stim API"
```

---

## Task 8: PR

**Files:** none.

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin p3-02-stabilizer-measurement
gh pr create --title "[P3-02] Stabilizer measurement with collapse" --body "$(cat <<'EOF'
Closes #33

## Summary
Adds projective Z-basis measurement with collapse to the `aleph-stab`
`Tableau` (Aaronson-Gottesman §3), building on P3-01:
- `g(x1,z1,x2,z2)` phase-exponent helper (§2) and `rowsum(h,i)`
  sign-tracking generator multiply (uses the P3-01-reserved scratch row).
- `measure<R: rand::Rng>(&mut self, qubit, rng) -> Result<bool, StabError>`
  with the deterministic / random case split and full collapse.
- Injected `rand::Rng` (mirrors `aleph-sv`), so P3-03's `Backend::measure`
  is a direct call.

Scope: no `Backend`/CLI integration (P3-03), no multi-qubit sample/reset.

## Tests
- Unit: `g` 4-case table, `rowsum` bit involution, `copy_row`/`zero_row`,
  |0…0⟩ deterministic-zero, |+⟩ random, out-of-range rejection.
- **Bell forcing (AC):** measuring q0 of |Φ+⟩ → random `b`; q1 returns the
  same `b`; re-measuring q0 is stable.
- **Statistical (AC):** GHZ-5 over 4000 seeded trials → q0 ~50/50 within a
  ±5σ band; all qubits agree every trial.
- **Property:** symplectic invariant survives a measurement (200 cases).
- **Stim oracle** (`#[ignore]`, EPYC): 100 random Clifford circuits;
  measure → postselect Stim to our outcome → compare canonical stabilizer
  groups; plus a `peek_z` determinism cross-check. Validated on EPYC:
  0/100 disagreements (stim 1.16.0).

All workspace tests / clippy `-D warnings` / fmt green.

## AC mapping
- [x] Measurement implemented for the stabilizer backend
- [x] Deterministic + random cases correct
- [x] Bell pair: measuring q0 forces q1
- [x] Equivalence vs Stim (100 circuits)
- [x] GHZ measurements 50/50

## Follow-ups
- P3-03: wire `Backend::measure` to this (`--backend stabilizer`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Confirm CI green; self-review the diff.**

Run: `gh pr checks --watch`.

---

## Self-Review (plan vs spec)

**Spec coverage:**
- §2.1 `g` → Task 1. ✓
- §2.2 `rowsum` → Task 2. ✓
- §2.3 `measure` (det/random/collapse) → Task 3. ✓
- §3 `rand` dep → Task 1 Step 1. ✓
- §4.1 unit g/rowsum → Tasks 1, 2. ✓
- §4.2 measurement units (Bell, deterministic, random) → Task 3. ✓
- §4.3 statistical GHZ → Task 4. ✓
- §4.4 Stim postselect oracle + determinism cross-check → Task 6 + Task 7. ✓
- §4.5 symplectic proptest → Task 5. ✓
- §5 error handling (only QubitOutOfRange) → Task 3 (`check_qubit`). ✓
- §6 AC mapping → Task 8 PR body. ✓

**Placeholder scan:** none (Task 8 has no `<...>` placeholders — issue
number #33 is known and hardcoded).

**Type consistency:** `g(bool,bool,bool,bool)->i32` (free fn) used by
`rowsum` (Task 2) and tested in Task 1. `rowsum(h,i)`, `copy_row(dst,src)`,
`zero_row(r)` private; `measure<R: rand::Rng>(&mut self, a, &mut R)->Result<bool,StabError>`
pub — consistent across Tasks 3, 5, 6. Uses existing `x()/z()/sign()`
accessors, `check_qubit`, `rows_anticommute` (pub), `stabilizers()`,
`apply_gate` from P3-01. `BitGrid::toggle` deliberately NOT used (removed
in P3-01) — `rowsum` uses get/set XOR. Task 5 inlines the symplectic check
via `rows_anticommute` (the `measure_does_not_break_symplectic` name is
flagged as non-existent and explicitly replaced).

**Note:** Task 5 reuses the existing `pub fn rows_anticommute` (P3-01) for
the inlined symplectic check — no new helper introduced.
