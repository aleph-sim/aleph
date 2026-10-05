# #538 follow-up: track |0⟩ qubits in the state-class walk (QFT blind spot)

Issue: #538, same PR (#541). Builds on `2026-10-05-p6-05-state-class-cost-design.md` (the #538 spec, "the base spec")
and report `docs/perf/p6-05-compiler.md` §4.

## 1. Problem and goal

The base spec's rule R2 judges an instruction by its matrix alone. QFT's controlled phases have angles π/2^k, so R2
marks QFT's steps generic and prices `phase_base` at its generic value. But QFT runs from |0…0⟩, and in our
construction (`tests/common/dist.rs::qft`) every controlled phase's control k < j has not yet had its `H`: the phase
acts as the identity and the state stays simple. Result: QFT model/measured 1.144 / 1.147 in Stage C run 1 (exit 1
MISS); pricing those phases simple gives 1.067 / 1.016 (report §4.5, diagnostic).

**Goal.** Make the class walk see which qubits are still |0⟩, so an instruction that acts trivially on the actual
state does not make it generic. Validate on new held-out cells fixed in this spec before any run.

**Non-goals:** changing Stage A constants or the choice of R2 (both frozen); tracking qubits known to be |1⟩;
modelling ranks that hold all-zero amplitudes; any change to kernels, `DistPlan` or `aleph-ir`.

## 2. The rule

`state_classes(plan)` (aleph-cuda `src/dist/cost.rs`) keeps, besides `generic: bool`, a set **Z0** of physical
qubits known to be |0⟩. Initially Z0 = all `n` qubits (every plan starts from |0…0⟩, and a non-identity initial map
is still |0…0⟩). Instructions of each `Local` step are processed **in order**:

- **`Barrier`:** skipped.
- **Gate `g`** (target matrix `U` on `g.qubits`, MSB-first as in `Gate::matrix`; external controls `g.controls`):
  1. If any external control ∈ Z0: the gate is the identity on the state. No class change, Z0 unchanged.
  2. Let **S** = the column indices of `U` whose bits for the target qubits in Z0 are 0 (the input subspace the
     state actually occupies).
  3. If every column `c ∈ S` equals the unit vector `e_c` (tolerance 1e-9): the gate is the identity on the state.
     No class change, Z0 unchanged.
  4. Otherwise judge R2 on the **restricted matrix** (the columns in S): "non-diagonal" means some entry `(r, c)`,
     `c ∈ S`, `r ≠ c`, has `|z| > 1e-9`; magnitudes and phases are taken over those columns' entries. If R2 fires,
     the state becomes generic.
  5. Update Z0 for each target qubit q: q ∈ Z0 afterwards iff every entry `(r, c)` with `c ∈ S` and `|z| > 1e-9` has
     bit q of `r` = 0. If `g.controls` is non-empty (all controls ∉ Z0 by step 1), a target may only **leave** Z0,
     never join it (the control-0 branch keeps the old value). Qubits outside `g.qubits` are unchanged.
- **`DiagonalPhase`:** a term is dead if some cond mask has all its bits in Z0 (its parity is 0, so the cond is
  false). Live terms' angles are judged by R2 as before. Z0 unchanged.
- **`Exchange { global_bits }`:** swap the Z0 membership of `global_bits[j]` and `m − k + j` (`k = global_bits.len()`),
  as the step swaps those physical bits.
- Non-finite entries/angles and unsupported instructions still return the errors they return today.

As before: the class is monotone (once generic, every later step is generic; Z0 need not be tracked after that), the
step holding the transition is priced generic in full, and exchanges keep the class. Under this rule a `Swap` with one
qubit in Z0 moves the |0⟩ to the other qubit (step 5), and `H` on a Z0 qubit removes it from Z0 (and is simple, as
today).

`makes_generic_under(instr, rule)` keeps its current signature and meaning (no Z0, i.e. Z0 = ∅) for the Stage A bench
and the rule-table tests. The walk uses a new private function that takes Z0.

## 3. Expected effect (stated before any run)

- QFT from |0…0⟩: every step simple (all controlled phases hit step 1 or 3).
- The other nine §4 cells start with `H`/`Rx`/`Ry` on every qubit, or (GHZ, Clifford brickwall) use only gates that
  are simple anyway: their classes and prices are unchanged.

## 4. Validation

**Held-out cells (new, fixed here; never run before this spec is committed).** n = 28, D ∈ {2, 4}, FP64, Lookahead
plans, ±10 % on model all-ranks / measured compute, same method as `dist_cost_gate`. The |0⟩ qubits are the low,
local ones, so no rank holds all-zero amplitudes.

| id | circuit | old R2 | new rule | checks |
|---|---|---|---|---|
| H1 | `X` on every odd qubit, then `qft(28)` | generic | generic | no under-pricing where phases really fire |
| H2 | `H` on qubits 14..27; then for each control c in 0..13 and each target t in 14..27, `GateInstance::controlled(Phase(π/2^((t−14)+1)), [t], [c])` (external control c); then `Cz(t, t+1)` for t in 14..26 | generic | simple | the fix on a structure other than QFT |
| H3 | 4 `clifford_layers` restricted to qubits 14..27 (H on even / S on odd, then nearest-neighbour CNOTs within 14..27); then `Gate::CRx(0.3)` on qubits `[c, 14 + c]` (2-qubit gate, control c first) for c in 0..13 | generic | simple | a trivial non-diagonal gate (cos/sin magnitudes) does not flip the class; exercises rule steps 2–3 (H2 exercises step 1) |

**Old cells:** the ten cells of report §4.2–4.3 re-run unchanged. QFT is reported as **in-sample** for this rule.

**Exit criteria (two runs):**
1. H1–H3 within ±10 % in both runs.
2. All ten old cells within ±10 % in both runs.
3. `dist_compile_bench`: exit1 (compiled ≤ min(N, L) + 3 %) and 3b (±10 %) on every FP64 cell in both runs.
4. Stage A constants and `STATE_RULE` unchanged.

A failure is reported as a MISS with its per-kind breakdown; nothing is re-tuned; the user decides.

**Idle box (hard precondition).** The run script waits before each run until the 1-minute load average is < 0.1 and
GPU utilisation is 0 %, and writes `uptime` + `nvidia-smi` to the committed log.

## 5. Tests (CPU, on the box's `--lib`)

Walk unit tests on hand-built plans:
- `qft(6)` planned at g = 1: every step simple.
- `X` on odd qubits then `qft(6)`: generic.
- Cnot and controlled-Rx with control in Z0: identity (simple; Z0 unchanged).
- `H` on a Z0 qubit removes it from Z0; a later CP controlled by it then counts.
- `Swap(a, b)` with a ∈ Z0, b ∉ Z0: afterwards b ∈ Z0, a ∉ Z0.
- An `Exchange` swaps Z0 membership of the paired bits.
- `DiagonalPhase` term whose cond mask ⊆ Z0 is dead; a term with a live cond and angle 0.3 makes the state generic.
- A plan whose first step puts `H` on every qubit prices bit-for-bit as before this change.

## 6. Report and delivery

- Report `docs/perf/p6-05-compiler.md` new §4.6 "Zero-tracking walk": the rule, the held-out table (old vs new class
  per cell, from `state_classes`), both runs' gate tables, the compile-bench verdicts, the exit criteria, and a
  Reading. QFT is labelled in-sample. Every quoted number in a table.
- Same PR #541; commits: spec, code + tests, then runs + report. The code commit is pushed before any Stage C' run.
- `Closes #538` stays; the PR body is updated with the new verdicts.

## 7. Risks

- **Zero-amplitude ranks.** A global qubit in Z0 makes half the ranks all-zero; the model prices them like the
  others. The held-out cells avoid it; GHZ and QFT plans may still have it (unchanged by this rule).
- **|1⟩ not tracked.** `X` then a control on that qubit is judged non-trivially (conservative: generic).
- **Plan order.** The walk follows plan order, which may differ from circuit order (Reorder). Plans are equivalent
  circuits, so Z0 is still exact for the plan as executed.
