# P6-05 follow-up: state-class GPU cost model (#538)

Issue: #538. Follows P6-05 (#59; report `docs/perf/p6-05-compiler.md` §2.3–2.4, §3.3–3.5).

## 1. Problem and goal

`GpuCostModel` prices each kernel kind with one constant. On the RTX 4000 SFF Ada at its 70 W cap, FP64 kernel time
depends on the amplitude **values** (report §2.3, H8):
- Dense2 costs ×1.196 more on a generic complex state than on the uniform H state.
- Dense3 costs ×1.12 more.
- Dense1 costs +1.5 % more.

`dense2` is calibrated on a scrambled state and `dense3` on the uniform state. A single constant cannot fit both GHZ
(Dense3 on a two-amplitude state) and random d=10 (Dense3 on a generic state). PR 3's compiled random plan, with
120 Dense3 launches against Lookahead's 18, is therefore under-priced by ~10 % (model/measured 0.898–0.912).

**Goal.** Price dense kernels by a **state class** (simple vs generic) that the model tracks along the plan. The
model must hold ±10 % on held-out workloads it was never tuned on. The rule and the constants are fixed **before**
the accuracy check runs (unlike PR 2's Dense2 choice, which was made after seeing gate results).

**Non-goals:** a continuous entropy model; simulating amplitudes; changing kernels or `DistPlan`; recalibrating kinds
whose state effect is < 5 %.

## 2. Stage A: state microbench (decides the rule; runs first)

New `#[ignore]` test `crates/aleph-cuda/tests/dist_cost_states.rs`. It times every kind (`dense1`, `dense2`,
`dense3`, `diag1`, `diag_k`, `cnot`, `phase_base`, `phase_term`, `phase_term_multi`) with the existing
interleaved-with-H method (best of 5 × 32, H-only baseline subtracted, mean of two runs). It runs at m_ref = 27 for
FP64 and m_ref = 28 for FP32, on each of these prepared states (the prefix is in both the payload and the baseline
circuit):

| id | state | prefix |
|---|---|---|
| a | uniform | H on every qubit |
| b | equal magnitudes, varied phases | a + `Rz(0.7 + 0.11q)` per qubit |
| c | varied magnitudes, real | a + `Ry(0.3 + 0.17q)` per qubit |
| d | generic complex (today's scrambled) | a + `Rx(0.3 + 0.17q)` + `Rz(0.7 + 0.11q)` per qubit |
| e | GHZ-like, two nonzero amplitudes | `H(0)`, then `CNOT(q, q+1)` chain |
| f | Clifford, few distinct values | 4 layers of (`H` / `S` alternating per qubit, then nearest-neighbour `CNOT`s) |

(c) gives real amplitudes with varied magnitudes (`Rx` would add phases); it is the HEA-like case. (b) is the only
state on which R1 and R2 (below) disagree: R1 calls it simple, R2 generic.

The interleaving `H` itself changes the state class of the touched qubits only within its existing set (`H` is
Clifford). States (b)–(d) stay generic under `H`; (e) and (f) stay in the Clifford set.

**Pre-registered decision rules** (fixed in this spec; Stage A only measures):
1. A state is **generic** iff `dense2` on it is ≥ 5 % slower than on (a). Otherwise it is **simple**.
2. A kind gets **two constants** (`simple`, `generic`) iff its own time on (d) is ≥ 5 % above (a). Otherwise it keeps
   one constant, the existing value. FP64 and FP32 are decided separately.
3. **Transition rule:** one of two candidates. An instruction **makes the state generic** if:
   - **R1:** it is non-diagonal and some matrix entry has magnitude outside {0, 1, 1/√2} (tolerance 1e-9).
   - **R2:** R1, **or** some entry (diagonal or not), or some `DiagonalPhase` term angle, has a phase that is not a
     multiple of π/4 (tolerance 1e-9).

   The chosen rule is the one whose prediction matches the measured class of every state (a)–(f). It is judged on
   each state's prefix, from |0…0⟩. If both match, choose **R1** (fewer generic transitions, closer to today's model). If
   neither matches, **stop**: report the table and decide with the user. No third rule is invented after the fact.
4. **Generic constants** are the (d) timings; **simple constants** are the (a) timings, which are today's values
   re-measured in the same session. Both come from the same two Stage A runs, mean of 2.
5. **Externally controlled gates** are judged by their target matrix (as `classify` does).

Stage A's table, the chosen rule and the constants are committed (code + report §4.1) **before** Stage C runs.

## 3. Stage B: model change

### 3.1 `aleph-ir` (backend-neutral)

`CostModel` gains a whole-plan hook with a default that equals today's behaviour:

```rust
pub trait CostModel {
    fn local_segment(&self, instrs: &[Instruction], layout: DistLayout) -> Result<f64, DistError>;
    fn exchange(&self, k: u32, m: u32) -> f64;
    /// Per-step costs of `plan`, in order. Default: each step priced alone.
    /// A model may override this to carry state across steps.
    fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> { /* default */ }
}
```

`plan_cost` calls `step_costs` and keeps its validation: every step cost and the total are finite and ≥ 0, and the
error is the same. `compile` is unchanged (it calls `plan_cost`). aleph-ir learns nothing about kernels or states: the
hook only lets a model see the steps in order.

### 3.2 `aleph-cuda`

- `KindTimes` gets optional generic values for the kinds that rule 2 selects. The exact field set follows Stage A,
  e.g. `dense2_generic: Option<f64>`; `None` means one constant. Absent generic values price exactly as today.
- `fn makes_generic(instr: &Instruction) -> Result<bool, DistError>` implements the chosen rule on **logical-order,
  pre-specialise** instructions of a `Local` step (physical indices do not matter to the rule).
- `GpuCostModel::step_costs` walks `plan.steps` in order with `generic: bool = false`:
  - Before pricing a `Local` step, if any instruction in it makes the state generic, set `generic = true`. The **whole
    step** is then priced generic. This is conservative for the step where the transition happens; it is usually the
    first step.
  - Price the step with `rank_segment_class(instrs, layout, R−1, generic)`.
  - `Exchange` steps are priced as today and do not change the class. An exchange is a permutation of amplitudes, so
    it keeps the values.
- `rank_segment` (public, used by the gate and the bench `all_ranks` helper) gets a `generic: bool` parameter.
  `GpuCostModel::all_ranks(plan) -> Result<f64, DistError>` (new, public) gives the all-ranks compute with the class
  walk. The test helper `all_ranks` in `tests/common/dist.rs` delegates to it.
- `CostModel::local_segment` alone (no plan context) prices **simple**, documented: it is the per-step fallback,
  unused by `plan_cost` once `step_costs` is overridden.

### 3.3 Tests (CPU-checkable parts on the box, model-only)

- `makes_generic` unit tests: a fixed list of instructions and their expected class under the chosen rule (H, S, T,
  X, CNOT, Toffoli, CZ, MCZ, `Rx(0.3)`, `Ry(0.3)`, `Rz(0.3)`, `Rz(π/2)`, `Phase(π/4)`, `Unitary2q` from a fused
  random block, `DiagonalPhase` with angle π/4 and with 0.3, and a gate with external controls).
- The class walk: a plan whose first step is Clifford and whose second contains `Rx(0.3)` prices step 1 simple and
  step 2 onward generic. A plan with no generic instruction prices exactly as the PR 3 model, compared bit for bit
  with absent generic values.
- `plan_cost` with the default `step_costs` is unchanged (the existing aleph-ir cost tests pass untouched), plus one
  test with a stateful stub model overriding `step_costs`.

## 4. Stage C: held-out validation (after Stage A+B are committed)

Extend `dist_cost_gate.rs` (Lookahead plans; same method, two idle-box runs, n=28, D∈{2,4}, FP64, ±10 % on model
all-ranks / measured compute):
- **Old cells (seen during design):** QFT, GHZ, random d=10, QAOA p=2 ring+chords, CCZ ladder d=4, Grover K=3.
- **Held-out cells (never used to choose anything):**
  - **HEA:** `build_hea(28, 4, params)` with `params[i] = 0.1 + 0.07·i`.
  - **random d=20:** `brickwall_bench(28, 20)`.
  - **Clifford brickwall d=10:** per layer, `H` on even qubits and `S` on odd ones, then alternating nearest-neighbour
    `CNOT`s (the state (f) construction, depth 10).
  - **QAOA p=2 on another graph:** a 3-regular-like graph, the ring `(i, i+1 mod n)` plus `(i, i+7 mod n)` for every
    `i`, with γ = [0.4, 0.7] and β = [0.3, 0.5].

Also re-run `dist_compile_bench` and report the PR 3 "3b" compiled-plan check with the new model. It needs the same
±10 %, now as a pass criterion of this follow-up.

**Exit criteria:**
1. Every old and held-out cell is within ±10 % in both runs.
2. The compiled-plan check (3b) is within ±10 % on every FP64 cell in both runs.
3. `dist_compile_bench` exit 1 (compiled ≤ min(Naive, Lookahead) + 3 %) still holds on every cell.

If a held-out cell fails, it is reported as a MISS with its per-kind breakdown (`dist_compile_kinds`-style). The rule
and constants are **not** re-tuned in this PR; the user decides the next step.

## 5. Report and delivery

- Report: `docs/perf/p6-05-compiler.md` new §4 "State-class cost model (#538)": Stage A table and decision, the
  chosen rule, the constants, the Stage C tables (old and held-out separately), the 3b re-check, and a Reading. Every
  quoted number appears in a table.
- One PR, `Closes #538`, with two commits fixed in order: Stage A+B (microbench, rule, constants, model), then
  Stage C (gate extension, runs, report). Stage A's results are committed before Stage C is run. The commit timestamps
  are the audit trail.
- GPU box: same ops as PR 3 (idle check before every timed run; never `pkill -f` over ssh).

## 6. Risks

- **Rule ambiguity.** The (a)–(f) states may not separate cleanly; Stage A's stop rule handles that.
- **Partial genericity.** A state can be generic on a subset of qubits only (e.g. Rx on one qubit). Any generic
  factor makes every amplitude differ, so the rule treats the whole state as generic; Stage A's states do not test a
  one-qubit-generic state. Accepted: it errs toward the generic (higher) price.
- **Un-scrambling.** A circuit that returns to a simple state (e.g. a circuit followed by its inverse) stays priced
  generic. Accepted, rare in practice.
- **Run-to-run noise** is ~3 %, the same order as the effect on Dense1. Rule 2's 5 % threshold keeps noise from
  splitting a kind.
