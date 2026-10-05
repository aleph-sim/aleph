# P6-05 follow-up: state-class GPU cost model (#538) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Price dense GPU kernels by a plan-tracked **state class** (simple vs generic). The rule and the constants are fixed by a pre-registered microbench (Stage A) before a held-out accuracy check (Stage C) runs.

**Architecture:**
- `aleph-ir`: `CostModel` gains `step_costs(&DistPlan)`, a whole-plan hook whose default prices each step alone, exactly as today. `plan_cost` calls it and keeps its validation. aleph-ir learns nothing about kernels or states.
- `aleph-cuda`:
  - `makes_generic_under(instr, StateRule)` implements both candidate rules, R1 and R2. `STATE_RULE` names the rule Stage A chose.
  - `KindTimes` gets a `generic: GenericTimes` of per-kind `Option<f64>`. `None` means the kind keeps one constant.
  - `GpuCostModel` overrides `step_costs` with a class walk: once the state is generic, it stays generic. It also gets `all_ranks(plan)`.
- Stage A is a new `#[ignore]` GPU microbench that replaces `dist_cost_calibrate.rs`. Stage C extends `dist_cost_gate.rs` with four held-out cells and re-runs `dist_compile_bench`.

**Tech Stack:** Rust 2021 (MSRV 1.89), the existing aleph-core / aleph-ir / aleph-cuda crates. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-p6-05-state-class-cost-design.md` (commit `d0b9395`). Background: `docs/perf/p6-05-compiler.md` §2.3–2.6 and §3.5.

## Spec rulings for this plan (binding; Task 6 records them in the spec)

1. **Every kind gets an optional generic constant in code. Rule 2 decides which ones are `Some`.** The spec says "the exact field set follows Stage A". This plan adds `GenericTimes` with an `Option<f64>` for each of the nine constants, so the code does not depend on the measurement. Stage A fills in `Some(..)` only where rule 2 passes. A `None` generic constant prices exactly like today's single constant.
2. **Both candidate rules are implemented.** `makes_generic_under(instr, StateRule::{R1, R2})` implements both. `const STATE_RULE` selects one, and `makes_generic(instr)` uses it. Stage A needs both rules to decide between them. Tests pin both rules' tables, so they do not depend on the Stage A outcome. Until Task 5, `STATE_RULE = R1` and every generic constant is `None`, so prices are bit-identical to PR 3 whichever rule is set.
3. **Rule 3 is judged on FP64.** The state effect lives on FP64 (report §2.3). If FP32's rule 2 selects no kind, FP32's measured classes do not matter, because its prices are class-independent. If FP32's rule 2 selects any kind, the chosen rule must also match FP32's measured classes; otherwise **stop and ask the user**. Without this ruling, an FP32 card with no state effect (every state measures simple) would match neither rule, and a literal reading of the spec would stop for no reason.
4. **`dist_cost_states.rs` replaces `dist_cost_calibrate.rs`.** After this PR the constants come from Stage A (rule 4: simple = state (a), generic = state (d)). The old calibrator would print a `KindTimes` literal in the old format with a stale Dense2 method. It is removed with `git rm`. Its payload builders move verbatim into `tests/common/calib.rs`, and git history keeps the original. The test runs both passes itself and prints the per-run values and their mean. Spec rule 4 asks for "mean of 2 runs", and a single invocation makes that reproducible.
5. **Rule 2 is applied to all nine constants as written, with a guard.** A constant whose (a) value is ≤ 0 or non-finite cannot form a ratio: it is reported `INVALID` and keeps one constant. The phase slopes are small fitted differences, so this can happen.
6. **The non-diagonal test is numeric**, like `classify`: some off-diagonal entry has `|z| > 1e-9`. `UnitaryKq` is read from its `data` directly, because `Gate::matrix()` rejects it. A non-finite matrix entry or `DiagonalPhase` angle is an error, never a silent class (ADR 0006). `Barrier` is never generic. Every other non-gate instruction is rejected, as `classify` does.
7. **Commits.** The spec's "two commits" are read as two **phase boundaries**. The code-only tasks before Stage A (Tasks 1–4) hold no measured values, and each commits on its own, so review stays per task. The audit trail the spec asks for is that the commit carrying Stage A's table, rule and constants (Task 5) is **pushed to origin before any Stage C timed run** (Task 6). The push timestamp is the evidence.
8. **The gate keeps its `assert`, over old and held-out cells together, and asserts only after every table is printed.** A held-out MISS is then still fully reported (spec §4: "reported as a MISS with its per-kind breakdown").

## Global Constraints

- Branch: `p6-05-state-class-cost` (exists locally, carries the spec commit `d0b9395`, not yet pushed), in the main checkout `/Users/ex/GitHub/aleph`. **No git worktrees.**
- Library code has no `unwrap()`/`expect()`/`panic!` on input. Internal impossibilities return `DistError::Unsupported { kind: "internal: …" }`.
- `aleph-ir` stays backend-agnostic: no CUDA, kernel, GPU or "state class" names in `aleph-ir/src`. The hook is just "steps in order".
- New `aleph-cuda` items sit under the crate's existing `#[cfg(all(target_os = "linux", feature = "cuda"))]` gating; `src/dist/` is already gated. Every new test file starts with `#![cfg(all(target_os = "linux", feature = "cuda"))]`.
- Float comparisons gating correctness check `is_finite()` first (ADR 0006).
- Thresholds, verbatim from the spec:
  - generic state: `dense2` ≥ 5 % slower than on (a);
  - two constants for a kind: its time on (d) ≥ 5 % above (a);
  - class tolerance: 1e-9;
  - magnitude set: {0, 1, 1/√2};
  - phase set: multiples of π/4;
  - accuracy band: ±10 %;
  - `m_ref` = 27 (FP64) and 28 (FP32).
- CPU checks on the Mac: `export PATH="$(brew --prefix rustup)/bin:$PATH"`, then `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo +nightly clippy --workspace --all-targets -- -D warnings` (CI runs beta lints), and `cargo test -p aleph-ir dist`. The Mac cannot build the `cuda` feature, so CUDA code is checked on the GPU box.
- GPU box: `ssh root@openwebgui.splynx.com` (RTX 4000 SFF Ada, sm_89, 20 GiB, 70 W cap).
  - Sync: `rsync -a --delete --exclude target --exclude .git ./ root@openwebgui.splynx.com:/root/aleph-p6/`.
  - Run: `ssh root@openwebgui.splynx.com 'cd /root/aleph-p6 && source ~/.cargo/env && <cmd>'`.
  - Before **every** timed run: `uptime` load ≈ 0 and `nvidia-smi` utilisation 0 %. About 1.3 GiB held by a resident service is expected.
  - Never `pkill -f`/`pgrep -f` over ssh: the pattern matches its own ssh command line. Kill by PID.
  - Long runs: `(setsid nohup bash script.sh > log 2>&1 < /dev/null &)`.
- CUDA checks on the box: `cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings` and `cargo test --release -p aleph-cuda --features cuda --lib dist`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn
  ```
- PR title `[P6-05] State-class GPU cost model (#538)`. Body starts with `Closes #538` and ends with the Claude Code attribution lines.

## Review Focus

1. **A model whose `step_costs` returns the wrong number of costs.** Expected: `plan_cost` returns an error; it never sums a truncated list. Pinned by `plan_cost_rejects_step_count_mismatch` (Task 1).
2. **Non-finite or unpriceable input to the rule:**
   - a `DiagonalPhase` with a NaN angle;
   - `Rx(NaN)`;
   - `Measure`.

   Expected: `Err`, never a silent "simple". Pinned by `makes_generic_rejects_bad_input` (Task 2).
3. **An exchange between the transition and later steps.** Expected: the exchange keeps the class, the step holding the transition is priced generic in full, and every later `Local` step stays generic. Pinned by `class_walk_survives_exchanges` (Task 3).
4. **The only generic gate is controlled by a *global* qubit.** On some ranks it then specialises away. Expected: the class is judged pre-specialise and is plan-wide, so every rank prices generic from that step on. Pinned by `globally_controlled_generic_gate_flips_all_ranks` (Task 3).
5. **The exchange-only plan (`comm_only`, every `Local` empty) and a plan with no generic gate.** Expected:
   - the exchange-only plan has 0 compute, and its class stays simple;
   - a plan with no generic gate prices bit-for-bit like the PR 3 model, even with generic constants set.

   Pinned by `no_generic_gate_prices_like_pr3` and `empty_locals_cost_nothing` (Task 3).

---

### Task 1: `CostModel::step_costs` hook in aleph-ir

**Files:**
- Modify: `crates/aleph-ir/src/dist/cost.rs` (trait, `plan_cost`, tests)

**Interfaces:**
- Consumes (existing): `DistPlan { layout, steps, .. }`, `DistStep::{Local(Vec<Instruction>), Exchange { global_bits }}`, `DistLayout::m()`.
- Produces (used by Task 3):
  ```rust
  fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> // trait method with default
  ```
  `plan_cost(plan, model)` keeps its signature and its `"internal: non-finite or negative cost"` error. It gains `"internal: step_costs length mismatch"`.

- [ ] **Step 1: Write the failing tests** (append inside `mod tests` in `crates/aleph-ir/src/dist/cost.rs`)

```rust
    /// Charges local steps 1 s per instruction, ×2 once any earlier step was
    /// an exchange: a model that carries state across steps.
    struct Stateful;
    impl CostModel for Stateful {
        fn local_segment(&self, instrs: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
            Ok(instrs.len() as f64)
        }
        fn exchange(&self, _: u32, _: u32) -> f64 {
            0.0
        }
        fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> {
            let mut seen_exchange = false;
            let mut out = Vec::new();
            for s in &plan.steps {
                out.push(match s {
                    DistStep::Local(v) => {
                        let f = if seen_exchange { 2.0 } else { 1.0 };
                        f * self.local_segment(v, plan.layout)?
                    }
                    DistStep::Exchange { .. } => {
                        seen_exchange = true;
                        0.0
                    }
                });
            }
            Ok(out)
        }
    }

    #[test]
    fn plan_cost_uses_step_costs_override() {
        // n=4, g=1: h(0) local, h(3) needs an exchange, then h(3) and h(0) local.
        let mut c = Circuit::new(4, 0);
        c.h(0).unwrap();
        c.h(3).unwrap();
        c.h(0).unwrap();
        let p = plan(&c, DistLayout::new(4, 1).unwrap(), Router::Naive).unwrap();
        let mut want = 0.0;
        let mut seen = false;
        for s in &p.steps {
            match s {
                DistStep::Local(v) => want += if seen { 2.0 } else { 1.0 } * v.len() as f64,
                DistStep::Exchange { .. } => seen = true,
            }
        }
        assert!(seen, "the plan must contain an exchange");
        assert_eq!(plan_cost(&p, &Stateful).unwrap(), want);
        // The default hook prices each step alone: Stub's sum is unchanged.
        assert_eq!(
            Stub.step_costs(&p).unwrap().iter().sum::<f64>(),
            plan_cost(&p, &Stub).unwrap()
        );
    }

    #[test]
    fn plan_cost_rejects_step_count_mismatch() {
        struct Short;
        impl CostModel for Short {
            fn local_segment(&self, _: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                Ok(1.0)
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                1.0
            }
            fn step_costs(&self, _: &DistPlan) -> Result<Vec<f64>, DistError> {
                Ok(vec![1.0])
            }
        }
        let mut c = Circuit::new(4, 0);
        c.h(0).unwrap();
        c.h(3).unwrap();
        let p = plan(&c, DistLayout::new(4, 1).unwrap(), Router::Naive).unwrap();
        assert!(p.steps.len() > 1);
        assert_eq!(
            plan_cost(&p, &Short),
            Err(DistError::Unsupported {
                kind: "internal: step_costs length mismatch"
            })
        );
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `export PATH="$(brew --prefix rustup)/bin:$PATH"; cargo test -p aleph-ir dist::cost`
Expected: compile error, `step_costs` is not a member of trait `CostModel`.

- [ ] **Step 3: Implement the hook**

Replace the trait and `plan_cost` in `crates/aleph-ir/src/dist/cost.rs`:

```rust
/// Seconds a distributed plan's pieces take on one rank's device.
pub trait CostModel {
    /// One rank's time for one `Local` step (physical indices, pre-specialise).
    fn local_segment(&self, instrs: &[Instruction], layout: DistLayout) -> Result<f64, DistError>;
    /// One `k`-bit exchange of a `2^m` slice.
    fn exchange(&self, k: u32, m: u32) -> f64;
    /// Per-step costs of `plan`, one per step, in order.
    ///
    /// Default: each step priced alone (`local_segment` / `exchange`). A model
    /// may override this to carry state from one step to the next.
    fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> {
        let m = plan.layout.m();
        plan.steps
            .iter()
            .map(|step| match step {
                DistStep::Local(instrs) => self.local_segment(instrs, plan.layout),
                DistStep::Exchange { global_bits } => {
                    Ok(self.exchange(global_bits.len() as u32, m))
                }
            })
            .collect()
    }
}

/// Predicted per-device time of `plan` under `model`: the sum of
/// `model.step_costs(plan)`.
///
/// Rejects a step-cost list whose length differs from the plan's step count,
/// and a step cost or total that is not finite or is negative (a model bug or
/// a missing table entry), so callers never rank plans on NaN/∞.
pub fn plan_cost(plan: &DistPlan, model: &dyn CostModel) -> Result<f64, DistError> {
    const BAD: DistError = DistError::Unsupported {
        kind: "internal: non-finite or negative cost",
    };
    let costs = model.step_costs(plan)?;
    if costs.len() != plan.steps.len() {
        return Err(DistError::Unsupported {
            kind: "internal: step_costs length mismatch",
        });
    }
    let mut t = 0.0;
    for c in costs {
        // is_finite first: NaN compares false (ADR 0006).
        if !c.is_finite() || c < 0.0 {
            return Err(BAD);
        }
        t += c;
    }
    if !t.is_finite() {
        return Err(BAD);
    }
    Ok(t)
}
```

Update the module doc's objective line to: `//! Objective (spec §6.1): T = Σ step_costs(plan); by default Σ_Local local_segment + Σ_Exchange exchange(k, m).`

- [ ] **Step 4: Run all aleph-ir dist tests**

Run: `cargo test -p aleph-ir dist`
Expected: PASS, including the three existing `plan_cost_*` tests unchanged and the `compile` tests.

- [ ] **Step 5: Lint and commit**

```bash
cargo fmt && cargo clippy -p aleph-ir --all-targets -- -D warnings && cargo +nightly clippy -p aleph-ir --all-targets -- -D warnings
git add crates/aleph-ir/src/dist/cost.rs
git commit -m "[P6-05] CostModel::step_costs whole-plan hook (#538)

plan_cost now sums model.step_costs(plan); the default prices each step
alone, so every existing model is unchanged. Lets a backend model carry
state across steps (the GPU state class) without aleph-ir knowing about it.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
```

---

### Task 2: Transition rules `makes_generic_under` (R1, R2)

**Files:**
- Modify: `crates/aleph-cuda/src/dist/cost.rs` (new items after `classify`; tests)
- Modify: `crates/aleph-cuda/src/lib.rs:54` (re-export)

**Interfaces:**
- Consumes (existing): `Instruction::{Gate, DiagonalPhase, Barrier}`, `Gate::matrix() -> Result<GateMatrix, GateError>` (`DistError: From<GateError>`), `GateMatrix::{M2x2, M4x4, M8x8}`, `Gate::UnitaryKq { k: u8, data: Box<[Complex]> }`, `PhaseTerm { conds, angle: f64 }`.
- Produces (used by Tasks 3, 4, 5):
  ```rust
  pub enum StateRule { R1, R2 }                      // Debug, Clone, Copy, PartialEq, Eq
  pub const STATE_RULE: StateRule;                   // R1 until Task 5
  pub fn makes_generic_under(instr: &Instruction, rule: StateRule) -> Result<bool, DistError>;
  pub fn makes_generic(instr: &Instruction) -> Result<bool, DistError>; // = under STATE_RULE
  ```
  Re-exported from `aleph_cuda`: `makes_generic, makes_generic_under, StateRule, STATE_RULE`.

- [ ] **Step 1: Write the failing tests** (in `mod tests` of `crates/aleph-cuda/src/dist/cost.rs`)

```rust
    /// (instruction, generic under R1, generic under R2) — spec §3.3's list.
    fn rule_table() -> Vec<(&'static str, Instruction, bool, bool)> {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let p = Param::Concrete;
        let ctl = |gate: Gate, t: u32, c: u32| {
            Instruction::Gate(GateInstance::controlled(gate, vec![t], vec![c]))
        };
        let dp = |angle: f64| {
            Instruction::DiagonalPhase(Box::new(DiagonalPhase {
                n_qubits: 3,
                terms: vec![PhaseTerm {
                    conds: smallvec![0b011],
                    angle,
                }],
            }))
        };
        // Rx(0.3) ⊗ I as a fused 2q block: non-diagonal, |cos 0.15|, |sin 0.15|.
        let (a, b) = ((0.15f64).cos(), (0.15f64).sin());
        let z = Complex::new(0.0, 0.0);
        let (ca, mib) = (Complex::new(a, 0.0), Complex::new(0.0, -b));
        let rx_i = [
            [ca, z, mib, z],
            [z, ca, z, mib],
            [mib, z, ca, z],
            [z, mib, z, ca],
        ];
        // 3q permutation block (simple) read from UnitaryKq data.
        let mut perm = vec![z; 64];
        for r in 0..8 {
            perm[r * 8 + (r ^ 1)] = Complex::new(1.0, 0.0);
        }
        let kq = Gate::UnitaryKq {
            k: 3,
            data: perm.into_boxed_slice(),
        };
        vec![
            ("H", g(Gate::H, &[0]), false, false),
            ("S", g(Gate::S, &[0]), false, false),
            ("T", g(Gate::T, &[0]), false, false),
            ("X", g(Gate::X, &[0]), false, false),
            ("Y", g(Gate::Y, &[0]), false, false),
            ("CNOT", g(Gate::Cnot, &[0, 1]), false, false),
            ("Toffoli", g(Gate::Toffoli, &[0, 1, 2]), false, false),
            ("CZ", g(Gate::Cz, &[0, 1]), false, false),
            ("Iswap", g(Gate::Iswap, &[0, 1]), false, false),
            ("MCZ", ctl(Gate::Z, 3, 0), false, false),
            ("Rx(0.3)", g(Gate::Rx(p(0.3)), &[0]), true, true),
            ("Ry(0.3)", g(Gate::Ry(p(0.3)), &[0]), true, true),
            ("Rz(0.3)", g(Gate::Rz(p(0.3)), &[0]), false, true),
            ("Rz(pi/2)", g(Gate::Rz(p(FRAC_PI_2)), &[0]), false, false),
            ("Phase(pi/4)", g(Gate::Phase(p(FRAC_PI_4)), &[0]), false, false),
            ("Phase(0.3)", g(Gate::Phase(p(0.3)), &[0]), false, true),
            ("CRz(0.3)", g(Gate::CRz(p(0.3)), &[0, 1]), false, true),
            ("Unitary2q Rx⊗I", g(Gate::Unitary2q(Box::new(rx_i)), &[0, 1]), true, true),
            ("UnitaryKq perm", g(kq, &[0, 1, 2]), false, false),
            ("DiagonalPhase pi/4", dp(FRAC_PI_4), false, false),
            ("DiagonalPhase 0.3", dp(0.3), false, true),
            ("ext-ctl Rx(0.3)", ctl(Gate::Rx(p(0.3)), 0, 1), true, true),
            ("ext-ctl Rz(0.3)", ctl(Gate::Rz(p(0.3)), 0, 1), false, true),
            ("Barrier", Instruction::Barrier(smallvec![0, 1]), false, false),
        ]
    }

    #[test]
    fn makes_generic_rule_tables() {
        for (name, instr, r1, r2) in rule_table() {
            assert_eq!(makes_generic_under(&instr, StateRule::R1).unwrap(), r1, "R1 {name}");
            assert_eq!(makes_generic_under(&instr, StateRule::R2).unwrap(), r2, "R2 {name}");
            assert_eq!(
                makes_generic(&instr).unwrap(),
                makes_generic_under(&instr, STATE_RULE).unwrap(),
                "STATE_RULE {name}"
            );
        }
    }

    #[test]
    fn makes_generic_rejects_bad_input() {
        let nan_dp = Instruction::DiagonalPhase(Box::new(DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![0b01],
                angle: f64::NAN,
            }],
        }));
        let nan_rx = g(Gate::Rx(Param::Concrete(f64::NAN)), &[0]);
        let measure = Instruction::Measure { qubit: 0, clbit: 0 };
        for rule in [StateRule::R1, StateRule::R2] {
            assert!(makes_generic_under(&nan_dp, rule).is_err(), "{rule:?} NaN angle");
            assert!(makes_generic_under(&nan_rx, rule).is_err(), "{rule:?} Rx(NaN)");
            assert!(makes_generic_under(&measure, rule).is_err(), "{rule:?} Measure");
        }
    }
```

- [ ] **Step 2: Sync to the box and verify the tests fail**

Run: `rsync -a --delete --exclude target --exclude .git ./ root@openwebgui.splynx.com:/root/aleph-p6/ && ssh root@openwebgui.splynx.com 'cd /root/aleph-p6 && source ~/.cargo/env && cargo test --release -p aleph-cuda --features cuda --lib dist::cost'`
Expected: compile error, `makes_generic_under` / `StateRule` not found.

- [ ] **Step 3: Implement the rules** (in `crates/aleph-cuda/src/dist/cost.rs`, after `classify`)

Add the imports at the top: `use aleph_core::{Complex, Gate, GateMatrix};` (replacing `use aleph_core::Gate;`) and `use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};`.

```rust
/// Candidate state-class transition rules (#538, spec §2 rule 3). An
/// instruction that "makes the state generic" moves the amplitudes off the
/// small value set where FP64 kernels run fastest at the card's power cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateRule {
    /// Non-diagonal, and some matrix entry has magnitude outside {0, 1, 1/√2}.
    R1,
    /// R1, or some entry (diagonal or not) or `DiagonalPhase` term angle has a
    /// phase that is not a multiple of π/4.
    R2,
}

/// The rule Stage A chose (docs/perf/p6-05-compiler.md §4.1).
pub const STATE_RULE: StateRule = StateRule::R1;

/// Tolerance on magnitudes and phases (spec §2 rule 3).
const CLASS_TOL: f64 = 1e-9;

const NON_FINITE: DistError = DistError::Unsupported {
    kind: "cost: non-finite matrix entry or phase angle",
};

fn special_magnitude(r: f64) -> bool {
    [0.0, 1.0, FRAC_1_SQRT_2]
        .iter()
        .any(|v| (r - v).abs() <= CLASS_TOL)
}

fn quarter_pi_phase(a: f64) -> bool {
    (a - (a / FRAC_PI_4).round() * FRAC_PI_4).abs() <= CLASS_TOL
}

/// Dimension and row-major entries of `gate`'s target matrix. `UnitaryKq` is
/// read from its data, as the backend does (`Gate::matrix` rejects it).
fn matrix_entries(gate: &Gate) -> Result<(usize, Vec<Complex>), DistError> {
    if let Gate::UnitaryKq { k, data } = gate {
        return Ok((1usize << k, data.to_vec()));
    }
    Ok(match gate.matrix()? {
        GateMatrix::M2x2(m) => (2, m.iter().flatten().copied().collect()),
        GateMatrix::M4x4(m) => (4, m.iter().flatten().copied().collect()),
        GateMatrix::M8x8(m) => (8, m.iter().flatten().copied().collect()),
    })
}

/// Whether `instr` makes the state generic under `rule`.
///
/// Judged on the logical, pre-specialise instruction; external controls are
/// ignored (the target matrix decides, as in [`classify`]). `Barrier` is never
/// generic. Errors on a non-finite entry or angle (ADR 0006) and on
/// instructions `classify` rejects.
pub fn makes_generic_under(instr: &Instruction, rule: StateRule) -> Result<bool, DistError> {
    let g = match instr {
        Instruction::Gate(g) => g,
        Instruction::DiagonalPhase(dp) => {
            let mut generic = false;
            for t in &dp.terms {
                if !t.angle.is_finite() {
                    return Err(NON_FINITE);
                }
                generic |= rule == StateRule::R2 && !quarter_pi_phase(t.angle);
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
    let (mut non_diagonal, mut odd_magnitude, mut odd_phase) = (false, false, false);
    for (idx, z) in entries.iter().enumerate() {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Err(NON_FINITE);
        }
        let r = z.norm();
        if idx / dim != idx % dim && r > CLASS_TOL {
            non_diagonal = true;
        }
        odd_magnitude |= !special_magnitude(r);
        odd_phase |= r > CLASS_TOL && !quarter_pi_phase(z.arg());
    }
    let r1 = non_diagonal && odd_magnitude;
    Ok(match rule {
        StateRule::R1 => r1,
        StateRule::R2 => r1 || odd_phase,
    })
}

/// [`makes_generic_under`] the chosen [`STATE_RULE`].
pub fn makes_generic(instr: &Instruction) -> Result<bool, DistError> {
    makes_generic_under(instr, STATE_RULE)
}
```

In `crates/aleph-cuda/src/lib.rs:54` change the re-export to:

```rust
pub use dist::cost::{
    classify, makes_generic, makes_generic_under, GpuCostModel, KernelKind, KindTimes,
    LinkModel, StateRule, STATE_RULE,
};
```

- [ ] **Step 4: Run the tests and clippy on the box**

Run: sync as in Step 2, then `ssh root@openwebgui.splynx.com 'cd /root/aleph-p6 && source ~/.cargo/env && cargo test --release -p aleph-cuda --features cuda --lib dist::cost && cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings'`
Expected: PASS, including `makes_generic_rule_tables`, `makes_generic_rejects_bad_input`, and the existing `classify_*` tests. If a table row fails, re-derive that gate's matrix from `crates/aleph-core/src/gate/kinds.rs` before you touch the rule. The table is spec §3.3's intent: Rz(θ) = diag(e^{−iθ/2}, e^{iθ/2}), so Rz(π/2) has phases ±π/4.

- [ ] **Step 5: Mac checks and commit**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add crates/aleph-cuda/src/dist/cost.rs crates/aleph-cuda/src/lib.rs
git commit -m "[P6-05] State-class transition rules R1/R2 (#538)

makes_generic_under implements both pre-registered candidate rules; the
Stage A microbench decides which one STATE_RULE names. Non-finite entries
and angles are errors, never a silent class (ADR 0006).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
```

---

### Task 3: Generic constants and the class walk in `GpuCostModel`

**Files:**
- Modify: `crates/aleph-cuda/src/dist/cost.rs` (`GenericTimes`, `KindTimes`, `seconds`, `rank_segment`, `state_classes`, `all_ranks`, `step_costs`, constants, tests)
- Modify: `crates/aleph-cuda/src/lib.rs` (re-export `GenericTimes`, `state_classes`)
- Modify: `crates/aleph-cuda/tests/common/dist.rs:223-234` (`all_ranks` delegates)
- Modify: `crates/aleph-cuda/tests/dist_cost_gate.rs:21-40,94-99` (`only` keeps generic values; `rep` via `step_costs`)
- Modify: `crates/aleph-cuda/tests/dist_compile_kinds.rs:21-66` (`ZERO`, `KINDS` pick generic too)

**Interfaces:**
- Consumes: Task 1's `CostModel::step_costs`; Task 2's `makes_generic`.
- Produces (used by Tasks 4–6):
  ```rust
  pub struct GenericTimes { pub dense1: Option<f64>, pub dense2: Option<f64>, pub dense3: Option<f64>,
      pub diag1: Option<f64>, pub diag_k: Option<f64>, pub cnot: Option<f64>,
      pub phase_base: Option<f64>, pub phase_term: Option<f64>, pub phase_term_multi: Option<f64> }
  impl GenericTimes { pub const NONE: Self; }               // all None
  // KindTimes gains: pub generic: GenericTimes
  impl KindTimes { pub fn seconds(&self, k: KernelKind, m: u32, generic: bool) -> f64; }
  impl GpuCostModel {
      pub fn rank_segment(&self, instrs: &[Instruction], layout: DistLayout, rank: u32, generic: bool) -> Result<f64, DistError>;
      pub fn all_ranks(&self, plan: &DistPlan) -> Result<f64, DistError>;
  }
  pub fn state_classes(plan: &DistPlan) -> Result<Vec<bool>, DistError>; // per step: priced generic?
  // CostModel for GpuCostModel: step_costs overridden; local_segment prices simple.
  ```

- [ ] **Step 1: Write the failing tests** (in `mod tests`; change `unit_times()` to add `generic: GenericTimes::NONE`)

```rust
    fn rx(q: u32) -> Instruction {
        g(Gate::Rx(Param::Concrete(0.3)), &[q])
    }

    /// unit_times with every kind's generic value = 10 × simple.
    fn unit_times_generic() -> KindTimes {
        let s = unit_times();
        KindTimes {
            generic: GenericTimes {
                dense1: Some(10.0 * s.dense1),
                dense2: Some(10.0 * s.dense2),
                dense3: Some(10.0 * s.dense3),
                diag1: Some(10.0 * s.diag1),
                diag_k: Some(10.0 * s.diag_k),
                cnot: Some(10.0 * s.cnot),
                phase_base: Some(10.0 * s.phase_base),
                phase_term: Some(10.0 * s.phase_term),
                phase_term_multi: Some(10.0 * s.phase_term_multi),
            },
            ..s
        }
    }

    fn model(kinds: KindTimes) -> GpuCostModel {
        GpuCostModel {
            kinds,
            link: LinkModel::aws_g6_fp64(),
            amp_bytes: 16.0,
            fuse: false,
        }
    }

    /// n=21, g=1 (m=20=m_ref): steps = [Local(a), Exchange(top), Local(b)].
    fn two_step_plan(a: Vec<Instruction>, b: Vec<Instruction>) -> DistPlan {
        use aleph_ir::dist::DistStep;
        let l = DistLayout::new(21, 1).unwrap();
        DistPlan {
            layout: l,
            steps: vec![
                DistStep::Local(a),
                DistStep::Exchange {
                    global_bits: smallvec![20],
                },
                DistStep::Local(b),
            ],
            ..aleph_ir::dist::plan(&aleph_ir::Circuit::new(21, 0), l, aleph_ir::dist::Router::Naive)
                .unwrap()
        }
    }

    #[test]
    fn none_generic_prices_like_simple() {
        let t = unit_times();
        for k in [KernelKind::Dense1, KernelKind::Dense3, KernelKind::Cnot] {
            assert_eq!(t.seconds(k, 21, true).to_bits(), t.seconds(k, 21, false).to_bits());
        }
        let tg = unit_times_generic();
        assert_eq!(tg.seconds(KernelKind::Dense2, 20, true), 20.0);
        assert_eq!(tg.seconds(KernelKind::Dense2, 20, false), 2.0);
    }

    #[test]
    fn class_walk_survives_exchanges() {
        // Step 0 Clifford (simple), step 2 has Rx(0.3): step 0 simple, 2 generic.
        let p = two_step_plan(vec![g(Gate::H, &[0])], vec![g(Gate::H, &[0]), rx(1)]);
        assert_eq!(state_classes(&p).unwrap(), vec![false, false, true]);
        let costs = model(unit_times_generic()).step_costs(&p).unwrap();
        // H = Dense1: simple 1.0, generic 10.0; Rx = Dense1 too.
        assert_eq!(costs[0], 1.0);
        assert_eq!(costs[2], 20.0);
        // Transition first: everything after it stays generic across the exchange.
        let q = two_step_plan(vec![rx(0)], vec![g(Gate::H, &[0])]);
        assert_eq!(state_classes(&q).unwrap(), vec![true, true, true]);
        assert_eq!(model(unit_times_generic()).step_costs(&q).unwrap()[2], 10.0);
    }

    #[test]
    fn no_generic_gate_prices_like_pr3() {
        let p = two_step_plan(
            vec![g(Gate::H, &[0]), g(Gate::Cnot, &[0, 1])],
            vec![g(Gate::T, &[1]), g(Gate::Toffoli, &[0, 1, 2])],
        );
        let pr3 = model(unit_times());
        let new = model(unit_times_generic());
        let a = aleph_ir::dist::plan_cost(&p, &new).unwrap();
        let b = aleph_ir::dist::plan_cost(&p, &pr3).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(new.all_ranks(&p).unwrap().to_bits(), pr3.all_ranks(&p).unwrap().to_bits());
    }

    #[test]
    fn empty_locals_cost_nothing() {
        let p = two_step_plan(vec![], vec![]);
        let m = model(unit_times_generic());
        assert_eq!(state_classes(&p).unwrap(), vec![false, false, false]);
        assert_eq!(m.all_ranks(&p).unwrap(), 0.0);
        let costs = m.step_costs(&p).unwrap();
        assert_eq!(costs[0], 0.0);
        assert_eq!(costs[1], m.exchange(1, 20));
    }

    #[test]
    fn globally_controlled_generic_gate_flips_all_ranks() {
        // Rx(0.3) on local q0 controlled by global q20: rank 0 drops it, rank 1
        // keeps it. The class is plan-wide, so both ranks price H(1) generic.
        let ctl_rx = Instruction::Gate(GateInstance::controlled(
            Gate::Rx(Param::Concrete(0.3)),
            vec![0u32],
            vec![20u32],
        ));
        let p = two_step_plan(vec![ctl_rx, g(Gate::H, &[1])], vec![]);
        assert!(state_classes(&p).unwrap()[0]);
        let m = model(unit_times_generic());
        let l = p.layout;
        let DistStep::Local(instrs) = &p.steps[0] else { unreachable!() };
        let r0 = m.rank_segment(instrs, l, 0, true).unwrap();
        let r1 = m.rank_segment(instrs, l, 1, true).unwrap();
        assert_eq!(r0, 10.0); // H only, generic
        assert_eq!(r1, 20.0); // Rx + H, generic
        assert_eq!(m.all_ranks(&p).unwrap(), 30.0);
    }

    #[test]
    fn local_segment_alone_prices_simple() {
        let m = model(unit_times_generic());
        let l = DistLayout::new(21, 1).unwrap();
        assert_eq!(m.local_segment(&[rx(0)], l).unwrap(), 1.0);
    }
```

Also change the existing `rank_segment_prices_the_fused_program` call to `model.rank_segment(&instrs, l, 1, false)`, and the `kind_seconds_scale_with_slice` / `scaling_handles_m_below_ref` calls to `seconds(.., .., false)`. The test module needs `use aleph_ir::dist::{DistLayout, DistPlan, DistStep};`.

`DistPlan`'s fields are `layout`, `steps`, `final_map`, `stats`; pricing reads only `layout` and `steps`, so `two_step_plan` borrows `final_map`/`stats` from an empty-circuit plan via struct update.

- [ ] **Step 2: Sync and verify the tests fail on the box**

Run: sync, then `cargo test --release -p aleph-cuda --features cuda --lib dist::cost` on the box.
Expected: compile errors (`GenericTimes`, `state_classes`, `all_ranks`, and the arity of `seconds` / `rank_segment`).

- [ ] **Step 3: Implement**

In `crates/aleph-cuda/src/dist/cost.rs`:

```rust
/// Per-launch seconds at `m_ref` on a **generic** state, for the kinds whose
/// time depends on the state class (#538, spec §2 rule 2). `None`: the kind
/// keeps its one constant, the simple value in [`KindTimes`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GenericTimes {
    pub dense1: Option<f64>,
    pub dense2: Option<f64>,
    pub dense3: Option<f64>,
    pub diag1: Option<f64>,
    pub diag_k: Option<f64>,
    pub cnot: Option<f64>,
    pub phase_base: Option<f64>,
    pub phase_term: Option<f64>,
    pub phase_term_multi: Option<f64>,
}

impl GenericTimes {
    /// No kind is state-dependent: prices exactly as one constant per kind.
    pub const NONE: Self = Self {
        dense1: None,
        dense2: None,
        dense3: None,
        diag1: None,
        diag_k: None,
        cnot: None,
        phase_base: None,
        phase_term: None,
        phase_term_multi: None,
    };
}
```

Give each `GenericTimes` field a one-line doc (`/// Generic-state \`Dense1\` seconds.` etc.); the crate enforces `missing_docs`. Add `pub generic: GenericTimes` as the last `KindTimes` field, with the doc `/// Generic-state values; the fields above are the simple-state (uniform) values.` Then:

```rust
impl KindTimes {
    /// Seconds for one launch of kind `k` on a `2^m` slice, on a generic state
    /// if `generic` (a kind without a generic value uses its simple one).
    ///
    /// Valid near `m_ref`: the `2^(m − m_ref)` scaling has no launch-latency
    /// floor, so costs at small `m` are not meaningful.
    pub fn seconds(&self, k: KernelKind, m: u32, generic: bool) -> f64 {
        let pick = |simple: f64, gen: Option<f64>| if generic { gen.unwrap_or(simple) } else { simple };
        let g = &self.generic;
        let at_ref = match k {
            KernelKind::Dense1 => pick(self.dense1, g.dense1),
            KernelKind::Dense2 => pick(self.dense2, g.dense2),
            KernelKind::Dense3 => pick(self.dense3, g.dense3),
            KernelKind::Diag1 => pick(self.diag1, g.diag1),
            KernelKind::DiagK => pick(self.diag_k, g.diag_k),
            KernelKind::Cnot => pick(self.cnot, g.cnot),
            KernelKind::PhasePoly { single, multi } => {
                pick(self.phase_base, g.phase_base)
                    + pick(self.phase_term, g.phase_term) * single as f64
                    + pick(self.phase_term_multi, g.phase_term_multi) * multi as f64
            }
            KernelKind::Free => 0.0,
        };
        at_ref * 2f64.powi(m as i32 - self.m_ref as i32)
    }
}

/// Whether each step of `plan` is priced on a generic state (spec §3.2).
///
/// The walk starts simple. A `Local` step that holds any instruction for
/// which [`makes_generic`] is true is priced generic in full, and so is every
/// later step. Exchanges permute amplitudes, so they keep the class.
pub fn state_classes(plan: &DistPlan) -> Result<Vec<bool>, DistError> {
    let mut generic = false;
    let mut out = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        if let DistStep::Local(instrs) = step {
            if !generic {
                for i in instrs {
                    if makes_generic(i)? {
                        generic = true;
                        break;
                    }
                }
            }
        }
        out.push(generic);
    }
    Ok(out)
}
```

In `impl GpuCostModel`:
- `rank_segment` gains a `generic: bool` parameter after `rank` and calls `self.kinds.seconds(classify(i)?, m, generic)`. Its doc gains: `` /// `generic`: price on a generic state (see [`state_classes`]). ``
- Add:

```rust
    /// All-ranks compute of `plan` (Σ over `Local` steps and ranks), with the
    /// state-class walk: the quantity the §6.3 gate compares to one card
    /// running every rank.
    pub fn all_ranks(&self, plan: &DistPlan) -> Result<f64, DistError> {
        let l = plan.layout;
        let mut all = 0.0;
        for (step, generic) in plan.steps.iter().zip(state_classes(plan)?) {
            if let DistStep::Local(instrs) = step {
                for r in 0..l.ranks() {
                    all += self.rank_segment(instrs, l, r, generic)?;
                }
            }
        }
        Ok(all)
    }
```

In `impl CostModel for GpuCostModel`:
- `local_segment` becomes `self.rank_segment(instrs, layout, layout.ranks() - 1, false)`. Prepend to its doc: `` /// Without plan context this prices a **simple** state; `plan_cost` uses `step_costs`, which carries the class. ``
- Add:

```rust
    /// Per-step costs with the state-class walk ([`state_classes`]); `Local`
    /// steps priced at rank R − 1 as in [`CostModel::local_segment`].
    fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> {
        let l = plan.layout;
        plan.steps
            .iter()
            .zip(state_classes(plan)?)
            .map(|(step, generic)| match step {
                DistStep::Local(instrs) => self.rank_segment(instrs, l, l.ranks() - 1, generic),
                DistStep::Exchange { global_bits } => {
                    Ok(self.exchange(global_bits.len() as u32, l.m()))
                }
            })
            .collect()
    }
```

Imports: `use aleph_ir::dist::{CostModel, DistError, DistLayout, DistPlan, DistStep};`. Add `generic: GenericTimes::NONE,` to `RTX4000_FP64` and `RTX4000_FP32` (Task 5 fills them). In `lib.rs`, add `GenericTimes` and `state_classes` to the `dist::cost` re-export.

Test-side updates:
- `tests/common/dist.rs`, replace the body of `all_ranks`:
  ```rust
  /// All-ranks model compute with the state-class walk (`GpuCostModel::all_ranks`).
  pub fn all_ranks(model: &GpuCostModel, p: &DistPlan) -> f64 {
      model.all_ranks(p).unwrap()
  }
  ```
  Drop the now-unused `DistStep` import if clippy flags it. `comm_only` still uses it.
- `tests/dist_cost_gate.rs`:
  - In `only`, add `generic: GenericTimes::NONE,` to the zeroed literal. The `keep` closures copy the generic value too, e.g. `|k, z| { z.dense2 = k.dense2; z.generic.dense2 = k.generic.dense2; }`, for every kind; `phase` copies all three generic fields.
  - Replace the `rep` loop (lines 93–99) with:
    ```rust
    let costs = model.step_costs(&p).unwrap();
    let rep: f64 = p
        .steps
        .iter()
        .zip(&costs)
        .filter(|(s, _)| matches!(s, DistStep::Local(_)))
        .map(|(_, c)| f64::from(l.ranks()) * c)
        .sum();
    ```
  - Imports: `GenericTimes` from `aleph_cuda`, and `aleph_ir::dist::CostModel`.
- `tests/dist_compile_kinds.rs`:
  - Add `generic: GenericTimes::NONE,` to `ZERO`.
  - Each `KINDS` pick also copies the generic value (`|k, z| { z.dense1 = k.dense1; z.generic.dense1 = k.generic.dense1; }`, and the `phase` pick copies all three).
  - The counters stay simple-only. A `None` generic value falls back to the simple 1.0, so launch counts are unchanged.
  - Import `GenericTimes`.

- [ ] **Step 4: Run every CUDA check on the box**

Run: sync, then on the box:
```bash
cargo test --release -p aleph-cuda --features cuda --lib dist && \
cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings && \
cargo test --release -p aleph-cuda --features cuda --test dist_gpu_oracle
```
Expected: all PASS. The oracle test checks that compile/run behaviour is unchanged: with `GenericTimes::NONE` every price is bit-identical to PR 3.

- [ ] **Step 5: Mac checks and commit**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings && cargo test -p aleph-ir dist
git add crates/aleph-cuda/src crates/aleph-cuda/tests/common/dist.rs crates/aleph-cuda/tests/dist_cost_gate.rs crates/aleph-cuda/tests/dist_compile_kinds.rs
git commit -m "[P6-05] GpuCostModel state-class walk + generic constants (#538)

KindTimes gains optional generic-state values per kind; GpuCostModel
overrides step_costs to walk the plan (once generic, always generic; the
transition step priced generic in full). No generic values are set yet,
so every price is bit-identical to PR 3.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
```

---

### Task 4: Stage A microbench `dist_cost_states.rs`

**Files:**
- Create: `crates/aleph-cuda/tests/common/calib.rs` (payload builders moved from `dist_cost_calibrate.rs`, state prefixes)
- Modify: `crates/aleph-cuda/tests/common/mod.rs` (add `pub mod calib;`)
- Modify: `crates/aleph-cuda/tests/common/dist.rs` (add `clifford_layers`, `clifford_brickwall`)
- Create: `crates/aleph-cuda/tests/dist_cost_states.rs`
- Delete: `crates/aleph-cuda/tests/dist_cost_calibrate.rs` (`git rm`)
- Modify: `crates/aleph-cuda/src/dist/cost.rs` (doc comments naming `dist_cost_calibrate` → `dist_cost_states`)

**Interfaces:**
- Consumes: Task 2's `makes_generic_under`, `StateRule`.
- Produces (used by Tasks 5, 6):
  ```rust
  // tests/common/dist.rs
  pub fn clifford_layers(c: &mut Circuit, n: u32, depth: usize); // per layer: H even / S odd, then NN CNOTs offset d%2
  pub fn clifford_brickwall(n: u32, depth: usize) -> Circuit;
  // tests/common/calib.rs
  pub enum State { A, B, C, D, E, F }                 // Debug, Clone, Copy, PartialEq, Eq
  pub const STATES: [State; 6];
  pub fn prefix(state: State, n: u32) -> Circuit;
  pub const KIND_NAMES: [&str; 9];                    // dense1 … phase_term_multi
  pub fn kinds(n: u32, state: State, time: &mut dyn FnMut(&Circuit) -> f64) -> [f64; 9];
  ```

- [ ] **Step 1: Add the Clifford helpers** (append to `tests/common/dist.rs`)

```rust
/// `depth` Clifford layers: `H` on even qubits and `S` on odd ones, then
/// nearest-neighbour `CNOT`s starting at qubit `d % 2` (#538 state (f) and the
/// held-out Clifford brickwall).
pub fn clifford_layers(c: &mut Circuit, n: u32, depth: usize) {
    for d in 0..depth {
        for q in 0..n {
            if q % 2 == 0 {
                c.h(q).unwrap();
            } else {
                c.s(q).unwrap();
            }
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
}

pub fn clifford_brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    clifford_layers(&mut c, n, depth);
    c
}
```

- [ ] **Step 2: Create `tests/common/calib.rs`**

Move `LAUNCHES`, `g`, `dp`, `single_term`, `multi_term` **verbatim** from `dist_cost_calibrate.rs`. Then:

```rust
//! Shared per-kernel-kind timing for the state microbench (#538 Stage A):
//! the interleaved-with-H method of P6-05 PR 2, on a chosen prepared state.
#![allow(dead_code)]

use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_ir::{Circuit, DiagonalPhase, Instruction, PhaseTerm};

use super::dist::clifford_layers;

// LAUNCHES, g, dp, single_term, multi_term: moved verbatim from dist_cost_calibrate.rs.

/// The six prepared states of spec §2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Uniform: H on every qubit.
    A,
    /// A + Rz(0.7 + 0.11q): equal magnitudes, varied phases.
    B,
    /// A + Ry(0.3 + 0.17q): varied magnitudes, real.
    C,
    /// A + Rx(0.3 + 0.17q) + Rz(0.7 + 0.11q): generic complex.
    D,
    /// GHZ-like: H(0), CNOT chain.
    E,
    /// Clifford: 4 `clifford_layers`.
    F,
}

pub const STATES: [State; 6] = [State::A, State::B, State::C, State::D, State::E, State::F];

/// The state's preparation from |0…0⟩, in both payload and baseline circuits.
pub fn prefix(state: State, n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    match state {
        State::E => {
            c.h(0).unwrap();
            for q in 0..n - 1 {
                c.cnot(q, q + 1).unwrap();
            }
        }
        State::F => clifford_layers(&mut c, n, 4),
        _ => {
            for q in 0..n {
                c.h(q).unwrap();
            }
            for q in 0..n {
                let f = f64::from(q);
                match state {
                    State::B => {
                        c.rz(0.7 + 0.11 * f, q).unwrap();
                    }
                    State::C => {
                        c.ry(0.3 + 0.17 * f, q).unwrap();
                    }
                    State::D => {
                        c.rx(0.3 + 0.17 * f, q).unwrap();
                        c.rz(0.7 + 0.11 * f, q).unwrap();
                    }
                    _ => {}
                }
            }
        }
    }
    c
}

/// `prefix(state)`, then `LAUNCHES` × (`H` on qubit `i % 8`, then `payload(i)`
/// if any). With no payload this is the baseline.
fn circuit(n: u32, state: State, payload: Option<&dyn Fn(usize) -> Instruction>) -> Circuit {
    let mut c = prefix(state, n);
    for i in 0..LAUNCHES {
        c.h((i % 8) as u32).unwrap();
        if let Some(make) = payload {
            c.add_instruction(make(i)).unwrap();
        }
    }
    c
}

/// Best-of-5 seconds per launch of the payload, the interleaved-H baseline on
/// the same state subtracted.
fn per_launch(
    n: u32,
    state: State,
    time: &mut dyn FnMut(&Circuit) -> f64,
    make: impl Fn(usize) -> Instruction,
) -> f64 {
    let base = circuit(n, state, None);
    let full = circuit(n, state, Some(&make));
    let t_base = (0..5).map(|_| time(&base)).fold(f64::INFINITY, f64::min);
    let t_full = (0..5).map(|_| time(&full)).fold(f64::INFINITY, f64::min);
    (t_full - t_base) / LAUNCHES as f64
}

pub const KIND_NAMES: [&str; 9] = [
    "dense1", "dense2", "dense3", "diag1", "diag_k", "cnot", "phase_base", "phase_term",
    "phase_term_multi",
];
```

Then `pub fn kinds(n, state, time) -> [f64; 9]`. It is `dist_cost_calibrate.rs`'s `kinds` body, with exactly these changes:
- every `per_launch(n, &mut *time, …)` / `per_launch_on(n, true, &mut *time, …)` becomes `per_launch(n, state, &mut *time, …)`;
- the H8 scrambling comment on `dense2` is deleted. Every kind now runs on the same `state`, which is the point of Stage A.

The payloads (`u2` = Iswap, the `kq` permutation, `Rz`, `Cz`, `Cnot`, `dp` 1/64-term fits) stay byte-for-byte.

- [ ] **Step 3: Create `tests/dist_cost_states.rs`**

```rust
//! #538 Stage A: every kernel kind's per-launch time on the six prepared
//! states of the spec (§2), FP64 at m_ref=27 and FP32 at 28, two passes and
//! their mean. Prints the measured state classes (rule 1), each candidate
//! rule's predictions (rule 3), the per-kind split (rule 2) and the
//! `KindTimes` literals (rule 4). It measures only: the rules are fixed in
//! the spec, and the human applies the printed decision in `src/dist/cost.rs`.
//! Replaces `dist_cost_calibrate.rs` (P6-05 PR 2).
//! Run (idle box): cargo test --release -p aleph-cuda --features cuda --test dist_cost_states -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use std::time::Instant;

use aleph_backend::run;
use aleph_cuda::{makes_generic_under, CudaContext, CudaSvBackend, CudaSvBackendF32, StateRule};
use aleph_ir::Circuit;
use common::calib::{kinds, prefix, State, KIND_NAMES, STATES};

/// Spec §2 thresholds.
const GENERIC_STATE: f64 = 1.05;
const SPLIT_KIND: f64 = 1.05;

type Table = [[f64; 9]; 6]; // [state][kind]

fn pass(n: u32, time: &mut dyn FnMut(&Circuit) -> f64) -> Table {
    let mut t = [[0.0; 9]; 6];
    for (s, &state) in STATES.iter().enumerate() {
        t[s] = kinds(n, state, time);
    }
    t
}

fn mean(a: &Table, b: &Table) -> Table {
    let mut m = [[0.0; 9]; 6];
    for s in 0..6 {
        for k in 0..9 {
            m[s][k] = 0.5 * (a[s][k] + b[s][k]);
        }
    }
    m
}

fn predicted(rule: StateRule, state: State, n: u32) -> bool {
    prefix(state, n)
        .instructions()
        .iter()
        .any(|i| makes_generic_under(i, rule).unwrap())
}

/// Prints the tables, decisions and literal for one precision.
fn report(tag: &str, name: &str, n: u32, r1: &Table, r2: &Table) {
    let m = mean(r1, r2);
    println!("\n### {tag} (n = m_ref = {n}), ms per launch, mean of 2 (run1/run2)\n");
    println!("| kind | a | b | c | d | e | f | d/a |");
    println!("|---|---|---|---|---|---|---|---|");
    for k in 0..9 {
        let cells: Vec<String> = (0..6)
            .map(|s| format!("{:.3} ({:.3}/{:.3})", 1e3 * m[s][k], 1e3 * r1[s][k], 1e3 * r2[s][k]))
            .collect();
        println!("| {} | {} | {:.3} |", KIND_NAMES[k], cells.join(" | "), m[3][k] / m[0][k]);
    }
    // Rule 1 (dense2 is kind index 1) and rule 3.
    let a2 = m[0][1];
    let measured: Vec<bool> = (0..6).map(|s| m[s][1] / a2 >= GENERIC_STATE).collect();
    println!("\n| state | dense2 / a | measured | R1 predicts | R2 predicts |");
    println!("|---|---|---|---|---|");
    let mut r1_ok = true;
    let mut r2_ok = true;
    for (s, &state) in STATES.iter().enumerate() {
        let (p1, p2) = (predicted(StateRule::R1, state, n), predicted(StateRule::R2, state, n));
        r1_ok &= p1 == measured[s];
        r2_ok &= p2 == measured[s];
        let cls = |g: bool| if g { "generic" } else { "simple" };
        println!(
            "| {state:?} | {:.3} | {} | {} | {} |",
            m[s][1] / a2,
            cls(measured[s]),
            cls(p1),
            cls(p2)
        );
    }
    let rule = match (r1_ok, r2_ok) {
        (true, _) => "R1 (matches every state; R1 preferred on a tie)",
        (false, true) => "R2 (only R2 matches every state)",
        (false, false) => "NONE MATCHES -> STOP, decide with the user (spec rule 3)",
    };
    println!("{tag} rule 3: {rule}");
    // Rule 2 and rule 4.
    let mut fields = Vec::new();
    let mut gens = Vec::new();
    for k in 0..9 {
        let (a, d) = (m[0][k], m[3][k]);
        let verdict = if !a.is_finite() || a <= 0.0 || !d.is_finite() {
            "INVALID (one constant)".to_string()
        } else if d / a >= SPLIT_KIND {
            gens.push(format!("{}: Some({d:.6e})", KIND_NAMES[k]));
            format!("SPLIT (d/a = {:.3})", d / a)
        } else {
            format!("one constant (d/a = {:.3})", d / a)
        };
        println!("{tag} rule 2: {} → {verdict}", KIND_NAMES[k]);
        fields.push(format!("{}: {a:.6e}", KIND_NAMES[k]));
    }
    let generic = if gens.is_empty() {
        "GenericTimes::NONE".to_string()
    } else {
        format!("GenericTimes {{ {}, ..GenericTimes::NONE }}", gens.join(", "))
    };
    println!(
        "const {name}: KindTimes = KindTimes {{ m_ref: {n}, {}, generic: {generic} }};",
        fields.join(", ")
    );
}

#[test]
#[ignore]
fn state_microbench() {
    let Ok(sync) = CudaContext::new(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let Ok(mut b64) = CudaSvBackend::with_seed(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let mut t64 = |c: &Circuit| {
        sync.synchronize().unwrap();
        let t = Instant::now();
        let st = run(&mut b64, c).unwrap();
        sync.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64();
        drop(st);
        s
    };
    let a = pass(27, &mut t64);
    let b = pass(27, &mut t64);
    report("FP64", "RTX4000_FP64", 27, &a, &b);
    let Ok(mut b32) = CudaSvBackendF32::with_seed(0) else {
        eprintln!("skipped: no CUDA FP32");
        return;
    };
    let mut t32 = |c: &Circuit| {
        sync.synchronize().unwrap();
        let t = Instant::now();
        let st = run(&mut b32, c).unwrap();
        sync.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64();
        drop(st);
        s
    };
    let a = pass(28, &mut t32);
    let b = pass(28, &mut t32);
    report("FP32", "RTX4000_FP32", 28, &a, &b);
}
```

`git rm crates/aleph-cuda/tests/dist_cost_calibrate.rs`. In `src/dist/cost.rs`, replace each `dist_cost_calibrate` mention (the `KindTimes` doc, `rtx4000_fp64` doc, `RTX4000_FP64` doc) with `dist_cost_states`. The provenance text itself is rewritten in Task 5.

- [ ] **Step 4: Compile-check on the box (no timed run yet)**

Run: sync, then on the box:
```bash
cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings && \
cargo test --release -p aleph-cuda --features cuda --test dist_cost_states --no-run && \
cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate --no-run
```
Expected: builds clean. The other `mod common` binaries also compile `calib.rs`; `#![allow(dead_code)]` covers them.

- [ ] **Step 5: Mac checks and commit**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add -A crates/aleph-cuda/tests crates/aleph-cuda/src/dist/cost.rs
git commit -m "[P6-05] Stage A state microbench, replaces dist_cost_calibrate (#538)

Times every kernel kind on the six prepared states of the spec, two passes
and their mean, and prints the pre-registered decisions: measured classes
(rule 1), the per-kind split (rule 2), which candidate rule matches
(rule 3) and the KindTimes literals (rule 4). No timings committed yet.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
```

---

### Task 5: Run Stage A, apply the decision, commit and push (phase boundary)

**Files:**
- Modify: `crates/aleph-cuda/src/dist/cost.rs` (`STATE_RULE`, `RTX4000_FP64`, `RTX4000_FP32`, their provenance docs, the `classify`/`KindTimes` notes)
- Modify: `docs/perf/p6-05-compiler.md` (new `## 4. State-class cost model (#538)`, `### 4.1 Stage A`)
- Save: `docs/perf/p6-05-state-class/stage-a.log` (raw output)

**Interfaces:**
- Consumes: Task 4's bench output.
- Produces: the final `STATE_RULE` and constants that Task 6 measures against. **Frozen after this task.**

- [ ] **Step 1: Run Stage A on the idle box**

```bash
rsync -a --delete --exclude target --exclude .git ./ root@openwebgui.splynx.com:/root/aleph-p6/
ssh root@openwebgui.splynx.com 'uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
# load ≈ 0 and utilisation 0 % required; otherwise wait and re-check.
ssh root@openwebgui.splynx.com 'cd /root/aleph-p6 && cat > /root/stage-a.sh <<EOF
source ~/.cargo/env
cd /root/aleph-p6
cargo test --release -p aleph-cuda --features cuda --test dist_cost_states -- --ignored --nocapture
EOF
(setsid nohup bash /root/stage-a.sh > /root/stage-a.log 2>&1 < /dev/null &)'
```
Expected runtime is roughly 1 h (6 states × 11 timings × 10 runs × 2 passes × 2 precisions). Poll `tail /root/stage-a.log` every ~15 min. When it is done, copy the log locally: `mkdir -p docs/perf/p6-05-state-class && scp root@openwebgui.splynx.com:/root/stage-a.log docs/perf/p6-05-state-class/stage-a.log`.

- [ ] **Step 2: Apply the printed decision exactly as pre-registered**

- **FP64 `rule 3: NONE MATCHES`:** **STOP.** Do not edit constants. Report the class table to the user and wait for their decision (spec rule 3; plan ruling 3).
- **FP64 rule 3 picks R1 or R2:** set `pub const STATE_RULE: StateRule = StateRule::R1;` (or `R2`).
- **FP32:**
  - If FP32's rule 2 line shows any `SPLIT`, FP32's rule 3 line must name the **same** rule. If it names a different rule, or NONE: STOP and ask (ruling 3).
  - If FP32 shows no SPLIT, FP32's class table is reported but not used.
- **Constants:** paste the printed `const RTX4000_FP64` / `RTX4000_FP32` literals over the existing ones, then run `cargo fmt`. The mean-of-2 rounding is already in the printed numbers. Do **not** edit any value by hand.
- **Provenance:** rewrite the `RTX4000_FP64` doc comment:
  - calibrated 2026-10-xx with `dist_cost_states` (command);
  - simple values = state (a), generic = state (d), mean of 2 passes in one invocation (rule 4);
  - which kinds split, and why (rule 2: (d) ≥ 5 % over (a));
  - the largest run-to-run spread from the table.

  Delete the PR 2 text about "dense2 alone on a scrambled state" and "dense3 carries a known state-dependent error": both are superseded. Point to `docs/perf/p6-05-compiler.md` §4.1.
- **`STATE_RULE` doc:** one line on why this rule (e.g. "state (b) measured simple/generic, which only R1/R2 predicts").

- [ ] **Step 3: Write report §4.1**

Append to `docs/perf/p6-05-compiler.md`:

```markdown
## 4. State-class cost model (#538)

Spec: `docs/superpowers/specs/2026-10-05-p6-05-state-class-cost-design.md`. The rule and the constants below were
fixed and pushed (commit `<sha>`) before any Stage C run.

### 4.1 Stage A: per-kind time by state

Bench: `cargo test --release -p aleph-cuda --features cuda --test dist_cost_states -- --ignored --nocapture`
(RTX 4000 SFF Ada, <date>, idle box, two passes in one invocation; raw log `p6-05-state-class/stage-a.log`).

<FP64 per-kind table, copied from the log>
<FP64 class table: measured vs R1 vs R2>
<FP32 tables, same shape>

**Decision (pre-registered rules, spec §2):**
- Rule 1: generic states = <list>; simple = <list>.
- Rule 3: <R1|R2>, because <state (b) measured …>.
- Rule 2: FP64 kinds with two constants: <list with d/a>; one constant: <rest>. FP32: <…>.
- Rule 4: constants as committed in `crates/aleph-cuda/src/dist/cost.rs` (`RTX4000_FP64`, `RTX4000_FP32`).
```

Every number in the prose must appear in a table above it. The `<sha>` is filled in after Step 5.

- [ ] **Step 4: Re-run the CUDA unit tests and the oracle on the box**

Run: sync, then `cargo test --release -p aleph-cuda --features cuda --lib dist && cargo test --release -p aleph-cuda --features cuda --test dist_gpu_oracle && cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings`.
Expected: PASS. If `makes_generic_rule_tables` was written to assert against `STATE_RULE`, it still passes, because it compares `makes_generic` to `makes_generic_under(.., STATE_RULE)`.

- [ ] **Step 5: Commit (Stage A+B boundary) and push**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add crates/aleph-cuda/src/dist/cost.rs docs/perf/p6-05-compiler.md docs/perf/p6-05-state-class/stage-a.log
git commit -m "[P6-05] Stage A results: state rule + generic constants (#538)

Stage A+B boundary: the transition rule and the simple/generic constants
are fixed here, from the pre-registered rules, before any Stage C run.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
git push -u origin p6-05-state-class-cost
```

Then replace `<sha>` in §4 with the short SHA of this commit. Leave that edit **uncommitted**; it goes in Task 6's commit.

---

### Task 6: Stage C held-out validation, report, PR

**Files:**
- Modify: `crates/aleph-cuda/tests/common/dist.rs` (add `hea_bench`, `qaoa_ring_skip7`)
- Modify: `crates/aleph-cuda/tests/dist_cost_gate.rs` (old + held-out tables; assert after printing)
- Modify: `crates/aleph-cuda/tests/dist_compile_bench.rs:1-7` (module doc: exit3b is now #538 exit 2)
- Modify: `docs/perf/p6-05-compiler.md` (§4.2–4.5)
- Modify: `docs/superpowers/specs/2026-10-05-p6-05-state-class-cost-design.md` (append "§7 Plan rulings")
- Save: `docs/perf/p6-05-state-class/stage-c-gate-run{1,2}.log`, `stage-c-compile-run{1,2}.log`

**Interfaces:**
- Consumes: frozen constants and rule from Task 5; `clifford_brickwall` from Task 4; `aleph_ir::build_hea(n, depth, &[f64]) -> Result<Circuit, AnsatzError>` (needs `n·(depth+1)` params); `aleph_ir::build_qaoa(n, &edges, &gammas, &betas)`.
- Produces: the Stage C evidence and the PR.

- [ ] **Step 1: Add the held-out circuits** (append to `tests/common/dist.rs`)

```rust
/// #538 held-out HEA: `build_hea(n, 4, params)` with `params[i] = 0.1 + 0.07·i`.
pub fn hea_bench(n: u32) -> Circuit {
    let depth = 4;
    let params: Vec<f64> = (0..n as usize * (depth as usize + 1))
        .map(|i| 0.1 + 0.07 * i as f64)
        .collect();
    aleph_ir::build_hea(n, depth, &params).unwrap()
}

/// #538 held-out QAOA p=2 on a 3-regular-like graph: the ring `(i, i+1 mod n)`
/// plus `(i, i+7 mod n)` for every `i`; γ = [0.4, 0.7], β = [0.3, 0.5].
pub fn qaoa_ring_skip7(n: u32) -> Circuit {
    let mut edges: Vec<(u32, u32)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    edges.extend((0..n).map(|i| (i, (i + 7) % n)));
    build_qaoa(n, &edges, &[0.4, 0.7], &[0.3, 0.5]).unwrap()
}
```

If `build_qaoa` rejects duplicate or reversed edges at n=28 (e.g. `(i, i+7)` against `(i+21, i+28 ≡ i)`), check its rules in `crates/aleph-ir/src/ansatz.rs`. The skip-7 set at n=28 holds no duplicates, since 7 ≠ 21 mod 28. If the builder still errors, **stop and report**; do not alter the graph silently.

- [ ] **Step 2: Extend the gate**

In `tests/dist_cost_gate.rs`:
- Refactor the body of `model_gate_n28_fp64` into `fn gate_table(title: &str, cases: &[(&str, Circuit)], …) -> f64`. It prints `### {title}`, the existing table header and rows, and the `kinds:` breakdown lines (the `only` shares), and it returns that table's worst |ratio − 1|.
- The test then calls it twice:
  ```rust
  let old: Vec<(&str, Circuit)> = vec![
      ("QFT", qft(n)), ("GHZ", ghz(n)), ("random d=10", brickwall_bench(n, 10)),
      ("QAOA p=2", qaoa_ring_chords(n)), ("CCZ ladder d=4", ccz_ladder(n, 4)), ("Grover K=3", grover_iters(n, 3)),
  ];
  let held_out: Vec<(&str, Circuit)> = vec![
      ("HEA d=4", hea_bench(n)), ("random d=20", brickwall_bench(n, 20)),
      ("Clifford brickwall d=10", clifford_brickwall(n, 10)), ("QAOA p=2 skip-7", qaoa_ring_skip7(n)),
  ];
  let w_old = gate_table("old cells (seen during design)", &old, …);
  let w_new = gate_table("held-out cells (never used to choose anything)", &held_out, …);
  println!("worst |model/measured − 1|: old {:.1} %, held-out {:.1} %", 100.0 * w_old, 100.0 * w_new);
  assert!(w_old.max(w_new) <= 0.10, "#538 exit 1: every old and held-out cell within ±10 %");
  ```
- Each row also prints the cell's class walk: count of steps priced generic over total `Local` steps, via `aleph_cuda::state_classes(&p)`. Add a column `generic steps`, so the report can show which cells the new rule touched.
- Update the module doc: name #538's Stage C, and give the run command unchanged.

In `tests/dist_compile_bench.rs`, change lines 5–6 of the module doc to: `` //! (spec §8); it does not assert them. The `exit3b` lines (compiled-plan model/measured) are #538's exit 2 and use the state-class model. ``

- [ ] **Step 3: Compile-check, then the two Stage C runs (idle box each time)**

```bash
rsync -a --delete --exclude target --exclude .git ./ root@openwebgui.splynx.com:/root/aleph-p6/
ssh root@openwebgui.splynx.com 'cd /root/aleph-p6 && source ~/.cargo/env && cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings && cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate --test dist_compile_bench --no-run'
ssh root@openwebgui.splynx.com 'uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
ssh root@openwebgui.splynx.com 'cat > /root/stage-c.sh <<EOF
source ~/.cargo/env
cd /root/aleph-p6
for r in 1 2; do
  cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture > /root/stage-c-gate-run\$r.log 2>&1
  cargo test --release -p aleph-cuda --features cuda --test dist_compile_bench -- --ignored --nocapture > /root/stage-c-compile-run\$r.log 2>&1
done
EOF
(setsid nohup bash /root/stage-c.sh > /root/stage-c.out 2>&1 < /dev/null &)'
```
Check idleness again before launching. The script runs back to back, so nothing else should start on the box while it runs. Then copy the four logs into `docs/perf/p6-05-state-class/`. A gate `assert` failure still leaves the full tables in the log; that is a reported MISS, not a crash to fix.

- [ ] **Step 4: Write report §4.2–4.5**

Append to `docs/perf/p6-05-compiler.md` §4:
- **§4.2 Old cells:** the gate table for both runs, including the `generic steps` column, compared with the PR 2/PR 3 ratios (§2.5).
- **§4.3 Held-out cells:** both runs. Each MISS gets its `kinds:` breakdown line.
- **§4.4 Compiled plans (exit 2) and compile bench (exit 3):** from `dist_compile_bench` both runs: the `exit3b` lines (FP64), the `exit1` lines, and any change in the chosen candidate versus §3.1 (with the new model, `compile` may choose differently).
- **§4.5 Exit criteria and Reading:**
  - list exits 1–3 of spec §4 with PASS/MISS per run;
  - a Reading on what moved and why;
  - if anything missed: "not re-tuned in this PR (spec §4); next step is the user's call".

Every number quoted in prose appears in a table.

Append to the spec a `## 7. Plan rulings (2026-10-05)` section that copies rulings 1–8 from this plan's "Spec rulings" section, one line each.

- [ ] **Step 5: Final checks, commit, PR**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings && cargo test -p aleph-ir dist
git add crates/aleph-cuda/tests docs/perf/p6-05-compiler.md docs/perf/p6-05-state-class docs/superpowers/specs/2026-10-05-p6-05-state-class-cost-design.md docs/superpowers/plans/2026-10-05-p6-05-state-class-cost.md
git commit -m "[P6-05] Stage C: held-out validation of the state-class model (#538)

Extends the §6.3 gate with four held-out cells (HEA, random d=20, Clifford
brickwall, QAOA skip-7) and re-runs dist_compile_bench; two idle-box runs.
Constants and rule are unchanged from the Stage A commit.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn"
git push
gh pr create --title "[P6-05] State-class GPU cost model (#538)" --body "$(cat <<'EOF'
Closes #538

## Summary
- aleph-ir: `CostModel::step_costs` whole-plan hook (default = per-step, unchanged).
- aleph-cuda: transition rules R1/R2 (`makes_generic_under`), `GenericTimes`, class walk in `GpuCostModel::step_costs` / `all_ranks`.
- Stage A microbench `dist_cost_states` (replaces `dist_cost_calibrate`); rule + constants committed and pushed in <stage-A sha> before Stage C ran.
- Stage C: §6.3 gate + 4 held-out cells, `dist_compile_bench` re-run, two idle-box runs.

## Results
<exit 1/2/3 verdicts per run, copied from report §4.5>

## Tests
- `cargo test -p aleph-ir dist`, `cargo test --release -p aleph-cuda --features cuda --lib dist`, `dist_gpu_oracle`: pass.

Report: `docs/perf/p6-05-compiler.md` §4.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01LUCZ7e8BnGsfk98a5GBkUn
EOF
)"
```

CI must be green before merge. The CLAUDE.md overview line, if any, is a separate `[meta]` PR (CLAUDE.md rule).
