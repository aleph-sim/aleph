# #538 follow-up: zero-tracking state-class walk — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The state-class walk tracks qubits known to be |0⟩ (Z0), so an instruction that acts trivially on the actual state (e.g. QFT's controlled phases whose controls are still |0⟩) no longer makes it generic. The change is validated on three new held-out cells fixed in the spec.

**Architecture:**
- **Rule core.** A private `instr_makes_generic(instr, rule, z0: &mut u64)` in `crates/aleph-cuda/src/dist/cost.rs` implements spec §2. It restricts the target matrix to the columns that the Z0 bits allow, runs an identity check, judges R2 on the restricted entries, and updates Z0.
- **Public wrappers.** `makes_generic_under` keeps its signature and calls the core with Z0 = ∅. `state_classes` walks the plan with Z0 = all qubits and swaps Z0 bits at each `Exchange`.
- **Gate.** `dist_cost_gate` gains a third table for H1–H3. The table shows the old-rule and new-rule generic-step counts.

**Tech Stack:** Rust 2021 (MSRV 1.89), aleph-core / aleph-ir / aleph-cuda. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-p6-05-zero-tracking-design.md` (commit `b9bd87d`). Base spec: `docs/superpowers/specs/2026-10-05-p6-05-state-class-cost-design.md`. Report: `docs/perf/p6-05-compiler.md` §4.

## Plan rulings (binding)

1. **Z0 is a `u64` bitmask over physical qubits** (bit q = qubit q). `DistLayout::new` enforces `n ≤ 64`, and `DiagonalPhase` cond masks are `u64` over the same physical indices (`plan.rs` lowers them). A qubit index ≥ 64 counts as "not in Z0" (conservative). An update to such an index is skipped, never a panic.
2. **Matrix index convention.** `qubits[0]` is the MSB of the matrix index (as `Gate::Unitary2q` documents), so qubit `qubits[i]` is bit `k − 1 − i` of a row or column index, where `k = qubits.len()`. If `dim != 1 << k`, return `DistError::Unsupported { kind: "internal: matrix size does not match qubit count" }`.
3. **Identity check (spec §2 step 3).** For each column `c ∈ S`, every entry `(r, c)` must satisfy `|z − δ_rc| ≤ 1e-9`. A global phase on S, such as `Rz` on a Z0 qubit, is **not** the identity and is judged by R2. Spec §2 asks for exactly this.
4. **`makes_generic_under` with Z0 = ∅ is unchanged on every non-identity gate.** Two side effects:
   - An exact identity matrix is now "not generic" by the identity check. It was already not generic under R1/R2, since all its entries are 0 or 1 with phase 0.
   - A `DiagonalPhase` cond mask of `0` (parity always 0, so the cond is always false) now makes its term dead.

   Both are correct semantics. No rule-table row has either case.
5. **The existing test `globally_controlled_generic_gate_flips_all_ranks` must change.** Its control, global qubit 20, is |0⟩, so under the new rule the gate is the identity. The test is rewritten so the control is in superposition after an exchange (Task 1 gives the code). Its intent is kept: the class is plan-wide, even where one rank drops the gate.
6. **Old-rule class column.** The gate prints both counts:
   - the new walk: `state_classes`;
   - the old walk, a test-side helper `old_rule_generic_steps(p)` in `tests/common/dist.rs`: `generic` starts false, and a `Local` step is generic if `makes_generic` is true for any of its instructions (Z0 = ∅), monotone.

   This lets the report show old vs new class per cell (spec §6).
7. **Runs.** One script runs gate ×2 and compile bench ×2. Before each run it waits until the 1-minute load is below 0.1 and GPU utilisation is 0 %, then logs `uptime` and `nvidia-smi` (spec §4). The code commit is pushed before the script starts.

## Global Constraints

- Branch `p6-05-state-class-cost` (PR #541), main checkout `/Users/ex/GitHub/aleph`. **No git worktrees.** Do not push unless the task says so.
- Library code has no `unwrap()`/`expect()`/`panic!` on input. Float comparisons gating correctness check `is_finite()` first (ADR 0006).
- `aleph-ir` is not touched.
- Frozen: `RTX4000_FP64` / `RTX4000_FP32` constants and `STATE_RULE = R2` (spec §4 exit 4).
- Tolerance 1e-9; R2 as defined in the base spec §2 rule 3.
- The Mac cannot build the `cuda` feature. CUDA checks run on the box:
  - sync: `rsync -a --delete --exclude target --exclude .git ./ root@openwebgui.splynx.com:/root/aleph-p6/`;
  - run: `ssh root@openwebgui.splynx.com 'cd /root/aleph-p6 && source ~/.cargo/env && <cmd>'`;
  - checks: `cargo test --release -p aleph-cuda --features cuda --lib dist`, `cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings`, `cargo test --release -p aleph-cuda --features cuda --test dist_gpu_oracle`;
  - never `pkill -f` / `pgrep -f` over ssh;
  - no `--ignored` runs except in Task 3.
- Mac: `export PATH="$(brew --prefix rustup)/bin:$PATH"`, then `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo +nightly clippy --workspace --all-targets -- -D warnings`.
- Commit trailers:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn
  ```
  (a subagent uses its own model name in `Co-Authored-By` if its instructions say so).

## Review Focus

1. **A Swap that moves a |0⟩ onto a qubit that a later gate uses as a control.** Expected: that control is now in Z0, so the later gate counts as the identity. Pinned by `swap_moves_zero` (Task 1).
2. **An exchange followed by a gate controlled by the newly local qubit.** Expected: Z0 follows the exchange. Pinned by `exchange_swaps_zero_membership` (Task 1).
3. **A global phase on the Z0 subspace (`Rz(0.3)` on a |0⟩ qubit).** Expected: not the identity, so R2 marks it generic. Pinned in `zero_tracking_rule_cases` (Task 1).
4. **A CP whose *target* is in Z0 but whose control is superposed.** Expected: the identity on the state (the phase fires only when target = 1). Pinned in `zero_tracking_rule_cases` (Task 1).
5. **Plans whose first step puts `H` on every qubit.** Expected: classes identical to the old rule's. Pinned by `h_layer_first_matches_old_rule` (Task 1).

---

### Task 1: Zero-tracking rule core and walk

**Files:**
- Modify: `crates/aleph-cuda/src/dist/cost.rs` (new private `instr_makes_generic`, bit helpers; `makes_generic_under` and `state_classes` delegate; tests)

**Interfaces:**
- Consumes (existing, same file): `matrix_entries(&Gate) -> Result<(usize, Vec<Complex>), DistError>`, `special_magnitude`, `quarter_pi_phase`, `CLASS_TOL`, `NON_FINITE`, `StateRule`, `STATE_RULE`, `makes_generic`.
- Produces:
  ```rust
  fn instr_makes_generic(instr: &Instruction, rule: StateRule, z0: &mut u64) -> Result<bool, DistError>; // private
  pub fn makes_generic_under(instr: &Instruction, rule: StateRule) -> Result<bool, DistError>; // unchanged signature, Z0 = ∅
  pub fn state_classes(plan: &DistPlan) -> Result<Vec<bool>, DistError>; // unchanged signature, zero-tracking walk
  ```

- [ ] **Step 1: Write the failing tests** (in `mod tests`)

```rust
    fn ctl(gate: Gate, targets: &[u32], controls: &[u32]) -> Instruction {
        Instruction::Gate(GateInstance::controlled(gate, targets.to_vec(), controls.to_vec()))
    }

    fn qft_circuit(n: u32) -> aleph_ir::Circuit {
        let mut c = aleph_ir::Circuit::new(n, 0);
        for j in (0..n).rev() {
            c.h(j).unwrap();
            for k in (0..j).rev() {
                let th = std::f64::consts::PI / f64::from(1u32 << (j - k));
                c.add_gate(GateInstance::controlled(
                    Gate::Phase(Param::Concrete(th)),
                    vec![j],
                    vec![k],
                ))
                .unwrap();
            }
        }
        for q in 0..n / 2 {
            c.swap(q, n - 1 - q).unwrap();
        }
        c
    }

    fn walk(instrs: &[Instruction], z0: &mut u64) -> Vec<bool> {
        instrs
            .iter()
            .map(|i| instr_makes_generic(i, StateRule::R2, z0).unwrap())
            .collect()
    }

    #[test]
    fn zero_tracking_rule_cases() {
        let p = Param::Concrete;
        let all = |n: u32| (1u64 << n) - 1;
        // Cnot / controlled-Rx with control in Z0: identity, Z0 unchanged.
        let mut z = all(4);
        assert_eq!(walk(&[g(Gate::Cnot, &[0, 1]), ctl(Gate::Rx(p(0.3)), &[1], &[0])], &mut z), vec![false, false]);
        assert_eq!(z, all(4));
        // CRx(0.3) on [c, t] with c in Z0: restricted columns are identity (rule steps 2-3).
        let mut z = all(4);
        assert_eq!(walk(&[g(Gate::CRx(p(0.3)), &[0, 1])], &mut z), vec![false]);
        // H removes its qubit from Z0 (and is simple).
        let mut z = all(4);
        assert_eq!(walk(&[g(Gate::H, &[0])], &mut z), vec![false]);
        assert_eq!(z, all(4) & !1);
        // CP(0.3) control superposed (q0), target in Z0 (q1): identity on the state.
        assert_eq!(walk(&[ctl(Gate::Phase(p(0.3)), &[1], &[0])], &mut z), vec![false]);
        // After H(1) the same CP fires: generic.
        assert_eq!(walk(&[g(Gate::H, &[1]), ctl(Gate::Phase(p(0.3)), &[1], &[0])], &mut z), vec![false, true]);
        // A global phase on the Z0 subspace is not the identity: Rz(0.3) on a |0> qubit is generic under R2.
        let mut z = all(4);
        assert_eq!(walk(&[g(Gate::Rz(p(0.3)), &[2])], &mut z), vec![true]);
        // Rx(0.3) on a |0> qubit: generic (cos/sin magnitudes), and the qubit leaves Z0.
        let mut z = all(4);
        assert_eq!(walk(&[g(Gate::Rx(p(0.3)), &[3])], &mut z), vec![true]);
        assert_eq!(z & 0b1000, 0);
    }

    #[test]
    fn swap_moves_zero() {
        // q0 superposed, q1 in Z0; Swap(0, 1) moves the |0> to q0.
        let mut z = 0b10u64;
        assert_eq!(walk(&[g(Gate::Swap, &[0, 1])], &mut z), vec![false]);
        assert_eq!(z, 0b01);
        // A later Cnot controlled by q0 is now the identity.
        let before = z;
        assert_eq!(walk(&[g(Gate::Cnot, &[0, 1])], &mut z), vec![false]);
        assert_eq!(z, before);
    }

    #[test]
    fn diagonal_phase_dead_terms() {
        let dp = |m: u64, angle: f64| {
            Instruction::DiagonalPhase(Box::new(DiagonalPhase {
                n_qubits: 3,
                terms: vec![PhaseTerm { conds: smallvec![m], angle }],
            }))
        };
        // q0 superposed, q1 and q2 in Z0.
        let mut z = 0b110u64;
        assert_eq!(walk(&[dp(0b110, 0.3)], &mut z), vec![false]); // cond ⊆ Z0: dead
        assert_eq!(walk(&[dp(0b011, 0.3)], &mut z), vec![true]); // live cond, odd angle
        assert_eq!(z, 0b110);
    }

    #[test]
    fn qft_from_zero_is_simple_and_x_input_is_generic() {
        use aleph_ir::dist::{plan, Router};
        let l = DistLayout::new(6, 1).unwrap();
        let p = plan(&qft_circuit(6), l, Router::Lookahead).unwrap();
        assert!(state_classes(&p).unwrap().iter().all(|&g| !g));
        let mut c = aleph_ir::Circuit::new(6, 0);
        for q in (1..6).step_by(2) {
            c.x(q).unwrap();
        }
        for i in qft_circuit(6).instructions() {
            c.add_instruction(i.clone()).unwrap();
        }
        let p = plan(&c, l, Router::Lookahead).unwrap();
        assert!(*state_classes(&p).unwrap().last().unwrap());
    }

    #[test]
    fn exchange_swaps_zero_membership() {
        // two_step_plan: n=21, g=1 (m=20); Exchange swaps physical 20 <-> 19.
        // Step a superposes q19 and q0; after the exchange q20 holds that superposed
        // qubit and q19 holds |0>.
        let a = vec![g(Gate::H, &[19]), g(Gate::H, &[0])];
        let via20 = two_step_plan(a.clone(), vec![ctl(Gate::Phase(Param::Concrete(0.3)), &[0], &[20])]);
        assert_eq!(state_classes(&via20).unwrap(), vec![false, false, true]);
        let via19 = two_step_plan(a, vec![ctl(Gate::Phase(Param::Concrete(0.3)), &[0], &[19])]);
        assert_eq!(state_classes(&via19).unwrap(), vec![false, false, false]);
    }

    #[test]
    fn h_layer_first_matches_old_rule() {
        use aleph_ir::dist::{plan, Router};
        let mut c = aleph_ir::Circuit::new(6, 0);
        for q in 0..6 {
            c.h(q).unwrap();
        }
        c.add_gate(GateInstance::controlled(Gate::Phase(Param::Concrete(0.3)), vec![2], vec![0])).unwrap();
        c.cnot(1, 5).unwrap();
        c.rz(0.7, 3).unwrap();
        let p = plan(&c, DistLayout::new(6, 1).unwrap(), Router::Lookahead).unwrap();
        let mut generic = false;
        let old: Vec<bool> = p
            .steps
            .iter()
            .map(|s| {
                if let DistStep::Local(v) = s {
                    generic |= v.iter().any(|i| makes_generic(i).unwrap());
                }
                generic
            })
            .collect();
        assert_eq!(state_classes(&p).unwrap(), old);
    }
```

Replace the body of the existing `globally_controlled_generic_gate_flips_all_ranks` (plan ruling 5) with:

```rust
        // q19 is superposed in step a; after the exchange it sits in global slot 20,
        // so Rx(0.3) on q0 controlled by q20 really fires on rank 1 and is dropped on
        // rank 0. The class is plan-wide: both ranks price step b generic.
        let ctl_rx = Instruction::Gate(GateInstance::controlled(
            Gate::Rx(Param::Concrete(0.3)),
            vec![0u32],
            vec![20u32],
        ));
        let p = two_step_plan(vec![g(Gate::H, &[19])], vec![ctl_rx, g(Gate::H, &[1])]);
        assert_eq!(state_classes(&p).unwrap(), vec![false, false, true]);
        let m = model(unit_times_generic());
        let l = p.layout;
        let DistStep::Local(instrs) = &p.steps[2] else {
            unreachable!()
        };
        let r0 = m.rank_segment(instrs, l, 0, true).unwrap();
        let r1 = m.rank_segment(instrs, l, 1, true).unwrap();
        assert_eq!(r0, 10.0); // H only, generic
        assert_eq!(r1, 20.0); // Rx + H, generic
        // Step a: H(19) simple on both ranks (1.0 each).
        assert_eq!(m.all_ranks(&p).unwrap(), 32.0);
```

`rustfmt` will reflow the long lines; that is fine.

- [ ] **Step 2: Sync and verify the tests fail on the box**

Run: sync, then `cargo test --release -p aleph-cuda --features cuda --lib dist::cost` on the box.
Expected: compile error, `instr_makes_generic` not found.

- [ ] **Step 3: Implement** (in `cost.rs`, replacing the body of `makes_generic_under` and of `state_classes`)

```rust
/// Bit `q` of `z0` (qubits ≥ 64 are never known to be |0⟩).
fn in_z0(z0: u64, q: u32) -> bool {
    q < 64 && (z0 >> q) & 1 == 1
}

/// Sets bit `q` of `z0` to `on` (no-op for q ≥ 64).
fn set_z0(z0: &mut u64, q: u32, on: bool) {
    if q < 64 {
        if on {
            *z0 |= 1u64 << q;
        } else {
            *z0 &= !(1u64 << q);
        }
    }
}

/// Whether `instr` makes the state generic under `rule`, given the physical
/// qubits `z0` known to be |0⟩, and updates `z0` (zero-tracking spec §2).
///
/// A gate with an external control in Z0, or whose target matrix is the
/// identity on the columns Z0 allows, acts trivially. Otherwise `rule` is
/// judged on those columns only. A `DiagonalPhase` term with a cond mask inside
/// Z0 is dead (its parity is 0).
fn instr_makes_generic(instr: &Instruction, rule: StateRule, z0: &mut u64) -> Result<bool, DistError> {
    let g = match instr {
        Instruction::Gate(g) => g,
        Instruction::DiagonalPhase(dp) => {
            let mut generic = false;
            for t in &dp.terms {
                if !t.angle.is_finite() {
                    return Err(NON_FINITE);
                }
                let dead = t.conds.iter().any(|&m| m & !*z0 == 0);
                generic |= !dead && rule == StateRule::R2 && !quarter_pi_phase(t.angle);
            }
            return Ok(generic);
        }
        Instruction::Barrier(_) => return Ok(false),
        _ => {
            return Err(DistError::Unsupported {
                kind: "cost: unsupported instruction",
            })
        }
    };
    let (dim, entries) = matrix_entries(&g.gate)?;
    if entries.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(NON_FINITE);
    }
    // Step 1: an external control in Z0 never fires.
    if g.controls.iter().any(|&c| in_z0(*z0, c)) {
        return Ok(false);
    }
    let k = g.qubits.len();
    if k >= usize::BITS as usize || dim != 1usize << k {
        return Err(DistError::Unsupported {
            kind: "internal: matrix size does not match qubit count",
        });
    }
    // qubits[i] is bit k-1-i of a matrix index (MSB-first).
    let bit = |i: usize| 1usize << (k - 1 - i);
    let zero_bits: usize = (0..k)
        .filter(|&i| in_z0(*z0, g.qubits[i]))
        .map(bit)
        .sum();
    // Step 2: the columns the state occupies.
    let cols: Vec<usize> = (0..dim).filter(|c| c & zero_bits == 0).collect();
    let at = |r: usize, c: usize| entries[r * dim + c];
    // Step 3: identity on those columns.
    let identity = cols.iter().all(|&c| {
        (0..dim).all(|r| {
            let want = if r == c { 1.0 } else { 0.0 };
            (at(r, c) - Complex::new(want, 0.0)).norm() <= CLASS_TOL
        })
    });
    if identity {
        return Ok(false);
    }
    // Step 4: R1/R2 on the restricted entries.
    let (mut non_diagonal, mut odd_magnitude, mut odd_phase) = (false, false, false);
    for &c in &cols {
        for r in 0..dim {
            let z = at(r, c);
            let mag = z.norm();
            if r != c && mag > CLASS_TOL {
                non_diagonal = true;
            }
            odd_magnitude |= !special_magnitude(mag);
            odd_phase |= mag > CLASS_TOL && !quarter_pi_phase(z.arg());
        }
    }
    // Step 5: Z0 update for the target qubits.
    for (i, &q) in g.qubits.iter().enumerate() {
        let stays_zero = cols
            .iter()
            .all(|&c| (0..dim).all(|r| at(r, c).norm() <= CLASS_TOL || r & bit(i) == 0));
        let now = if g.controls.is_empty() {
            stays_zero
        } else {
            in_z0(*z0, q) && stays_zero
        };
        set_z0(z0, q, now);
    }
    let r1 = non_diagonal && odd_magnitude;
    Ok(match rule {
        StateRule::R1 => r1,
        StateRule::R2 => r1 || odd_phase,
    })
}
```

`makes_generic_under` becomes (keep its doc, and add one line: `/// No qubit is assumed |0⟩ (Z0 = ∅); [`state_classes`] tracks Z0.`):

```rust
pub fn makes_generic_under(instr: &Instruction, rule: StateRule) -> Result<bool, DistError> {
    instr_makes_generic(instr, rule, &mut 0)
}
```

`state_classes` becomes. Replace its doc's middle sentence with: "The walk starts simple with every qubit known |0⟩ (zero-tracking spec §2). A `Local` step that holds any instruction that makes the state generic, given the qubits still |0⟩ at that point, is priced generic in full, and so is every later step. Exchanges swap the paired bits' |0⟩ status and keep the class."

```rust
pub fn state_classes(plan: &DistPlan) -> Result<Vec<bool>, DistError> {
    let l = plan.layout;
    let mut z0: u64 = if l.n >= 64 { u64::MAX } else { (1u64 << l.n) - 1 };
    let mut generic = false;
    let mut out = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        if !generic {
            match step {
                DistStep::Local(instrs) => {
                    for i in instrs {
                        if instr_makes_generic(i, STATE_RULE, &mut z0)? {
                            generic = true;
                            break;
                        }
                    }
                }
                DistStep::Exchange { global_bits } => {
                    let k = global_bits.len() as u32;
                    for (j, &gb) in global_bits.iter().enumerate() {
                        let lb = (l.m() + j as u32)
                            .checked_sub(k)
                            .ok_or(DistError::Unsupported {
                                kind: "internal: exchange wider than the local slice",
                            })?;
                        let (a, b) = (in_z0(z0, gb), in_z0(z0, lb));
                        set_z0(&mut z0, gb, b);
                        set_z0(&mut z0, lb, a);
                    }
                }
            }
        }
        out.push(generic);
    }
    Ok(out)
}
```

- [ ] **Step 4: Run tests and clippy on the box**

Run: sync, then `cargo test --release -p aleph-cuda --features cuda --lib dist && cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings && cargo test --release -p aleph-cuda --features cuda --test dist_gpu_oracle`.
Expected: all PASS, including the existing `makes_generic_rule_tables`, `class_walk_survives_exchanges`, `no_generic_gate_prices_like_pr3` and `empty_locals_cost_nothing`.

If `qft_from_zero_is_simple_and_x_input_is_generic` fails on the first assertion, print the plan's instructions per step and find the instruction the walk calls generic. Report it rather than weakening the test. It would mean the spec §3 expectation is wrong.

- [ ] **Step 5: Mac checks and commit**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add crates/aleph-cuda/src/dist/cost.rs
git commit -m "[P6-05] Zero-tracking state-class walk (#538 QFT follow-up)

state_classes now tracks the physical qubits known to be |0>: a gate whose
external control is |0>, or whose matrix is the identity on the columns the
state occupies, no longer makes the state generic; Exchange and Swap move
the |0> status. makes_generic_under keeps its meaning (Z0 = empty).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
```

---

### Task 2: Held-out cells H1–H3 in the gate

**Files:**
- Modify: `crates/aleph-cuda/tests/common/dist.rs` (add `qft_x_odd`, `zero_phase_ladder`, `zero_crx`, `old_rule_generic_steps`)
- Modify: `crates/aleph-cuda/tests/dist_cost_gate.rs` (third table; an `old-rule generic steps` column in every table; assert covers all three tables)

**Interfaces:**
- Consumes: `qft(n)` and `clifford_layers` (tests/common/dist.rs), `aleph_cuda::{makes_generic, state_classes}`, the existing `gate_table(title, cases, n, sync, d, model) -> f64`.
- Produces: the H1–H3 builders and the old-rule column.

- [ ] **Step 1: Add the circuits and helper** (append to `tests/common/dist.rs`)

```rust
/// Zero-tracking held-out H1: `X` on every odd qubit, then `qft(n)`.
pub fn qft_x_odd(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in (1..n).step_by(2) {
        c.x(q).unwrap();
    }
    for i in qft(n).instructions() {
        c.add_instruction(i.clone()).unwrap();
    }
    c
}

/// Zero-tracking held-out H2: `H` on qubits n/2..n; controlled `Phase(π/2^((t − n/2) + 1))`
/// from every |0> qubit c < n/2 (external control) onto every t ≥ n/2; then `Cz(t, t+1)`
/// on the upper half.
pub fn zero_phase_ladder(n: u32) -> Circuit {
    let h = n / 2;
    let mut c = Circuit::new(n, 0);
    for t in h..n {
        c.h(t).unwrap();
    }
    for ctrl in 0..h {
        for t in h..n {
            let th = std::f64::consts::PI / f64::from(1u32 << ((t - h) + 1));
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(th)),
                vec![t],
                vec![ctrl],
            ))
            .unwrap();
        }
    }
    for t in h..n - 1 {
        c.add_gate(GateInstance::new(Gate::Cz, vec![t, t + 1])).unwrap();
    }
    c
}

/// Zero-tracking held-out H3: 4 Clifford layers on qubits n/2..n (H on even / S on odd,
/// then nearest-neighbour CNOTs within the upper half), then `CRx(0.3)` on `[c, n/2 + c]`
/// for every c < n/2.
pub fn zero_crx(n: u32) -> Circuit {
    let h = n / 2;
    let mut c = Circuit::new(n, 0);
    for d in 0..4usize {
        for q in h..n {
            if q % 2 == 0 {
                c.h(q).unwrap();
            } else {
                c.s(q).unwrap();
            }
        }
        let mut q = h + (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    for ctrl in 0..h {
        c.add_gate(GateInstance::new(Gate::CRx(Param::Concrete(0.3)), vec![ctrl, h + ctrl]))
            .unwrap();
    }
    c
}

/// Generic `Local` steps under the **old** (pre-zero-tracking) walk: a step is
/// generic if `makes_generic` is true for any of its instructions, monotone.
pub fn old_rule_generic_steps(p: &DistPlan) -> usize {
    let mut generic = false;
    let mut count = 0;
    for s in &p.steps {
        if let DistStep::Local(v) = s {
            generic |= v.iter().any(|i| aleph_cuda::makes_generic(i).unwrap());
            count += usize::from(generic);
        }
    }
    count
}
```

At n = 28 the half is 14, which matches spec §4's "14..27" and "0..13". `Gate`, `GateInstance` and `Param` are already imported in this file.

- [ ] **Step 2: Extend the gate**

In `tests/dist_cost_gate.rs`:
- Add a column `old-rule generic` right after `generic steps` in `gate_table`'s header, separator and rows. Its value is `old_rule_generic_steps(&p)` over the same `Local`-step count, e.g. `3/11`.
- After the held-out table, add:
  ```rust
  let zero: Vec<(&str, Circuit)> = vec![
      ("H1 QFT on X-odd input", qft_x_odd(n)),
      ("H2 |0>-controlled phase ladder", zero_phase_ladder(n)),
      ("H3 |0>-controlled CRx", zero_crx(n)),
  ];
  let w_zero = gate_table("zero-tracking held-out cells (fixed in the zero-tracking spec)", &zero, n, &sync, &mut d, &model);
  ```
- Replace the summary line and the assert with:
  ```rust
  println!(
      "worst |model/measured − 1|: old {:.1} %, held-out {:.1} %, zero-tracking held-out {:.1} %",
      100.0 * w_old, 100.0 * w_new, 100.0 * w_zero
  );
  assert!(
      w_old.max(w_new).max(w_zero) <= 0.10,
      "every old and held-out cell within ±10 %"
  );
  ```
- Module doc: add the line `//! Zero-tracking follow-up: three more held-out cells (H1–H3) and an old-rule column.`
- Update the imports.

- [ ] **Step 3: Compile-check on the box (no timed runs)**

Run: sync, then on the box `cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings && cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate --test dist_compile_bench --no-run`.
Expected: clean.

- [ ] **Step 4: Mac checks and commit**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
git add crates/aleph-cuda/tests/common/dist.rs crates/aleph-cuda/tests/dist_cost_gate.rs
git commit -m "[P6-05] Gate: zero-tracking held-out cells H1-H3 + old-rule column (#538)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
```

---

### Task 3: Push, then idle-enforced runs (controller)

**Files:** save `docs/perf/p6-05-state-class/zt-gate-run{1,2}.log`, `zt-compile-run{1,2}.log` and `zt-runner.out`.

- [ ] **Step 1: Push the code first** (spec §6: the code is pushed before any run)

```bash
git push origin p6-05-state-class-cost
```

- [ ] **Step 2: Launch the runs, waiting for idle before each**

```bash
rsync -a --delete --exclude target --exclude .git ./ root@openwebgui.splynx.com:/root/aleph-p6/
ssh root@openwebgui.splynx.com 'cat > /root/zt.sh <<'"'"'EOF'"'"'
source ~/.cargo/env
cd /root/aleph-p6
cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate --test dist_compile_bench --no-run
idle() {
  while :; do
    l=$(cut -d" " -f1 /proc/loadavg); u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    awk -v l="$l" -v u="$u" "BEGIN{exit !(l < 0.1 && u == 0)}" && break
    sleep 30
  done
  uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
}
for r in 1 2; do
  idle; cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture > /root/zt-gate-run$r.log 2>&1
  idle; cargo test --release -p aleph-cuda --features cuda --test dist_compile_bench -- --ignored --nocapture > /root/zt-compile-run$r.log 2>&1
done
echo ZT_DONE
EOF
(setsid nohup bash /root/zt.sh > /root/zt-runner.out 2>&1 < /dev/null &)'
```

- [ ] **Step 3: Wait for `ZT_DONE`, then fetch the logs**

Fetch the four logs and `zt-runner.out` into `docs/perf/p6-05-state-class/`.

---

### Task 4: Report §4.6, PR update

**Files:**
- Modify: `docs/perf/p6-05-compiler.md` (new `### 4.6 Zero-tracking walk (QFT follow-up)`)
- Commit the Task 3 logs.

- [ ] **Step 1: Write §4.6**
  - the rule, in short;
  - the expected-vs-measured class table per cell (old-rule and new-rule generic steps, from the logs);
  - both runs' gate tables (all three tables);
  - the compile-bench `exit1` / `exit3b` verdicts and any change of chosen candidate against §4.4;
  - the idle-wait `uptime` lines;
  - spec §4's exit criteria 1–4 with PASS/MISS per run;
  - a Reading.

  QFT is labelled **in-sample**. Every quoted number appears in a table. A MISS is reported as one, without re-tuning.
- [ ] **Step 2: Commit and push**

```bash
git add docs/perf/p6-05-compiler.md docs/perf/p6-05-state-class/zt-*
git commit -m "[P6-05] Zero-tracking validation: report §4.6 (#538)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
git push origin p6-05-state-class-cost
```

- [ ] **Step 3: Update the PR #541 body** (`gh pr edit 541 --body-file …`). Add the zero-tracking result table and the new exit verdicts, and drop or keep the "Exit 1 is MISSED" banner according to the results.
