# P6-05: Communication-aware circuit compiler for distributed SV

Issue: #59 (P6-05). Depends on P6-03 (#528, lookahead router). Refs #55.

## 1. Goal and success metric

Build a compiler for the distributed state-vector path that minimizes **predicted multi-GPU wall time**. Today the
planner keeps the circuit's gate order. The compiler instead reorders gates within what commutation allows, chooses the
initial placement of global qubits, and chooses the exchange width.

The objective is the time model validated on AWS g6.12xlarge (`docs/perf/p6-multi-gpu.md` §1, §4; within 2–4 % of
measured on every cell once the link is priced per exchange shape):

```
T(D) = compute per GPU + Σ_exchanges bytes(k) / BW(k)
```

No new hardware is needed. Compute is measured on one GPU as `T_onecard(R=D)/D` (RTX 4000 Ada), and `BW(k)` comes from
the AWS calibration. A real multi-GPU re-run is optional and out of scope.

Why this matters (from the AWS report):
- random d=10 at n=28 reaches only 1.03× (FP64, D=2) and 0.87× (FP32, D=2).
- Its distributed rank issues about 2× the kernel passes of the fused single-GPU circuit, for three reasons: fusion
  stops at every exchange, the router inserts local swaps, and the gate order interleaves local and global work.
- The loss at D=2 is therefore mostly compute, not link.

Non-goals (YAGNI): simulated annealing / ILP / RL search; a memory objective (per-rank memory is fixed at `2^m`);
inter-node MPI; changing `DistPlan`, `specialize`, or any backend execution path.

## 2. Architecture

```
aleph-ir/src/dist/
  dag.rs        NEW  commutation-respecting dependency DAG (block counters)
  schedule.rs   NEW  reordering list scheduler → DistPlan   (Router::Reorder)
  placement.rs  NEW  initial global-qubit choice
  cost.rs       NEW  CostModel trait, LinkModel, compile() candidate selection
  plan.rs       unchanged: Naive / Lookahead remain (baselines and candidates)
aleph-cuda/src/dist/
  cost.rs       NEW  GpuCostModel: calibrated per-kernel-kind weights
  mod.rs        + DistSvBackend::run_compiled
```

- **Output format is unchanged.** Every strategy produces the existing `DistPlan`, so `specialize`, `DistSvBackend`,
  `NcclExchange` and the CPU reference `run_dist` run compiled plans as they are.
- **The IR stays backend-agnostic** (CLAUDE.md golden rule 4). `aleph-ir` only declares the `CostModel` trait. Kernel
  kinds, calibration constants and the link table live in `aleph-cuda`.
- **Initial placement is free.** Every distributed run starts from |0…0⟩, which is invariant under any qubit
  permutation. A plan may therefore start from a non-identity logical→physical map without emitting any instruction.
  `final_map` already maps the result back for readout. The invariant "plans assume a |0…0⟩ initial state" is
  documented on `DistPlan`.
- **Never worse by construction.** `compile` chooses among candidates that include today's Naive and Lookahead plans, so
  under the model it is never worse than either.

### Public API

```rust
// aleph-ir::dist
pub enum Router { Naive, Lookahead, Reorder { max_k: u32 } }   // Reorder new; max_k clamped to 1..=g

pub trait CostModel {
    /// Seconds for one rank's `m`-qubit program of one `Local` step (physical indices, pre-specialise).
    fn local_segment(&self, instrs: &[Instruction], layout: DistLayout) -> f64;
    /// Seconds for one k-bit exchange of a 2^m slice.
    fn exchange(&self, k: u32, m: u32) -> f64;
}

/// Build all candidate plans, price each with `cost`, return the cheapest (ties: fewer exchanges).
pub fn compile(circuit: &Circuit, layout: DistLayout, cost: &dyn CostModel) -> Result<DistPlan, DistError>;

// aleph-cuda
pub struct GpuCostModel { /* per-kind seconds at m_ref, amp_bytes, LinkModel */ }
pub struct LinkModel { pub bw_by_k: Vec<f64> /* bytes/s, index k-1 */ }  // presets: aws_g6_fp64(), aws_g6_fp32()
impl DistSvBackend { pub fn run_compiled(&mut self, c: &Circuit, g: u32, cost: &dyn CostModel) -> … }
```

## 3. DAG (`dag.rs`)

Each (gate, qubit) pair gets an **action type**:

| gate / role on qubit q | type |
|---|---|
| diagonal gate (`is_diagonal()`), any of its qubits | Z |
| external control | Z |
| in-qubit control of `Cnot`, `CRx`, `CRy`, `Toffoli` | Z |
| `Cnot` / `Toffoli` target, `X`, `Rx`, `CRx` target | X |
| `DiagonalPhase`, every qubit in any cond mask | Z |
| everything else (`H`, `Y`, `Ry`, `U3`, `CRy` target, `Swap`, `Iswap`, `Unitary*` non-diag, …) | Other |

**Commutation rule.** Two gates commute if, on every shared qubit, they have the same type and that type is Z or X.
- Soundness: on each shared qubit, both operators are block-diagonal in the same single-qubit basis (computational for
  Z, Hadamard for X). Operators that are block-diagonal in a common product basis over their shared support, and act
  arbitrarily on unshared qubits, commute.
- This is a subset of what `passes::commute::gates_commute` proves, and it is conservative.

**Representation: block counters, not edges.**
- Per qubit, the gate sequence is split into maximal **blocks** of consecutive same-type gates. Z and X blocks may hold
  many gates; Other is always a block of one.
- A gate is **ready** when, for each of its qubits, every gate of the *previous* block on that qubit has been scheduled.
- Gates inside one block are mutually unordered on that qubit. Order on other qubits still constrains them.
- Bookkeeping per (qubit, block): a remaining-count. When a block's count hits 0, gates of the next block on that qubit
  lose one pending dependency. Build is O(Σ arity); scheduling is O(Σ arity) plus the ready-set cost.
- `Barrier` is a full fence: it starts a new Other block on every qubit it lists. `Measure` / `Reset` / `TiledBlock`
  stay rejected as today (`DistError::Unsupported`).
- `Swap` (unconditioned) is Other on both qubits. The scheduler applies it as a free relabel, exactly like `plan.rs`.

## 4. Scheduler (`schedule.rs`)

State: the logical↔physical `Map` (as in `plan.rs`), the ready set (ordered by original index), and per-qubit lazy
queues of unscheduled original indices of gates that need that qubit local (`required_local`).

Loop until no gates remain:
1. **Drain.** Repeatedly take the lowest-index ready gate whose `required_local` qubits are all local and emit it
   (physical indices) into the current `Local` step. Diagonal-only and control-only global use needs no exchange, as
   today. Lowest-index-first keeps adjacent gates together, which fusion needs.
2. **Exchange.** When every ready gate needs at least one global qubit:
   - **Bring set B.** Seed with the missing qubits of the lowest-index ready gate. While `|B| < max_k`, consider each
     other global qubit whose next local-need (smallest unscheduled original index that needs it local) lies within
     the window of original indices `[seed, seed + H)`, `H = 4·n` (so it may hold fewer than `4·n` unscheduled ones). Add the one with the earliest next local-need, but only if that
     need comes before the next local-need of the victim it would displace. This is P6-03's prefetch rule, applied to
     the DAG frontier. Stop when no qubit qualifies.
   - **Victims.** The `|B|` local qubits with the farthest next local-need (Belady). Ties prefer higher physical slots.
     Never evict a qubit the seed gate needs.
   - Move victims to the top `|B|` local slots with local `Swap`s (counted in `CommStats::local_swaps`). Close the
     current `Local` step, emit `Exchange { global_bits }`, and update the map. This is the same contiguity contract as
     `exchange_lookahead`.
3. `Swap` gates are applied as relabels when they become ready (`CommStats::relabels`).

`max_k ∈ 1..=g` lets the cost model trade batched k-bit exchanges against single-bit ones. The AWS link penalises
fan-out: 4.35 GB/s two-bit vs 7.16 GB/s single-bit, FP64.

`Router::Reorder` uses the identity placement. `compile` also runs it from the placement in §5.

## 5. Placement (`placement.rs`)

Choose the initial global set as the `g` logical qubits with the **latest first local-need**. Ties go to the fewest
total local-needs, then to the highest logical index. Qubits that never need to be local come first. The resulting map
puts these qubits in physical slots `m..n`, and the rest in order in `0..m`.

## 6. Cost model

### 6.1 Objective

`T(plan) = Σ_{Local steps} cost.local_segment(step) + Σ_{Exchange steps} cost.exchange(k, m)`.

### 6.2 `GpuCostModel` (`aleph-cuda/src/dist/cost.rs`)

- **`local_segment`:**
  - Specialise the step for the representative rank **R−1** (all global bits 1). There, every global-controlled gate
    is live, so this is the busiest rank.
  - Run `fuse_for_gpu` on the result.
  - Sum, over the fused instructions, the per-kind time `t_kind(m_ref) · 2^(m − m_ref)`. These passes are
    bandwidth-bound.
  - Kinds are exactly the dispatch arms that `CudaSvBackend` takes for a fused circuit. PR 2 enumerates them from the
    dispatch code and pins each with a unit test mapping instruction → kind. The expected set is: dense block k=1/2/3,
    numerically diagonal (`apply_diag`), `DiagonalPhase` (`apply_phase_poly`; per-pass plus per-term), `apply_cnot`,
    and 2q permutation (`Swap`). If the dispatch has a kind outside this set, it gets its own weight.
  - *PR 2 correction (verified against the dispatch code):* the shipped set is `Dense1/2/3`, `Diag1`, `DiagK`, `Cnot`,
    `PhasePoly`.
    - There is no `Swap` kind: `Swap` is not diagonal and goes to dense k=2 `apply_kq_tiled`, so it costs as `Dense2`.
    - Diagonals split into `Diag1` (`apply_diag_1q`, no scratch upload) and `DiagK` (`apply_diag`, k=2/3).
    - No layer kind: the dist path applies instructions one at a time (`apply_one`), never `apply_1q_multi`.
    - `UnitaryKq` is tested before the diagonal check, so it is always dense; k=4/5 (TF32) is unreachable because
      `fuse_for_gpu` caps fusion at 3.
    - `PhasePoly` is priced by term shape: `(t_phase_base + t_phase_term · n_single + t_phase_term_multi · n_multi)
      · 2^(m − m_ref)`, where single terms have ≤ 1 cond and multi terms are an AND of ≥ 2 conds. A term's cost tracks
      how often it fires (1/2 vs 1/4 of amplitudes), as measured by a kernel sweep.
  - `local_segment` returns `Result<f64, DistError>` (specialisation can fail; library code must not panic).
  - *PR 2 correction (calibration):* each launch is timed **interleaved with an `H`** (H-only baseline subtracted),
    not 32 identical launches back to back, which read 2–10 % slow at the card's power cap. Constants are the mean of
    two runs.
  - *PR 2 correction (Dense2 state):* `Dense2` alone is calibrated on a scrambled state (H layer, then one `Rx` and one
    `Rz` per qubit, in both payload and baseline). At the 70 W cap FP64 kernel time depends on the amplitude data, and
    `Dense2` costs ×1.196 on a generic complex state. Every other kind stays on the uniform H state (scrambling `Dense3`
    would mis-price GHZ). This per-kind choice was made after seeing the §6.3 gate results.
- **`exchange(k, m)`:** `(1 − 2^−k) · 2^m · amp_bytes / bw_by_k[k−1]`.
  - `LinkModel::aws_g6_fp64()` = [7.16e9, 4.35e9] B/s. For k > 2 it extrapolates the two-bit value, and the extrapolation
    is documented.
  - The FP32 preset uses 7.09e9 single-bit and scales the two-bit value by the same ratio. Callers can supply other
    tables, e.g. a P2P preset.
- **Calibration:** an `#[ignore]`d test, `tests/dist_cost_calibrate.rs`. On the RTX 4000 at m_ref = 27 (FP64) and 28
  (FP32), it times each kind as the best of 5 over a batch of 32 launches (method superseded by the PR 2 calibration
  corrections above: interleaved with `H`, `Dense2` on a scrambled state, mean of two runs). The constants are committed in `cost.rs`
  with the date, GPU and command.

### 6.3 Model accuracy gate (must pass before `compile` lands)

(Superseded below.) On every workload in §8 at n=28, D∈{2,4}, FP64, the model's compute term `Σ local_segment` for the
Lookahead plan must be within **±10 %** of measured `T_onecard(R=D)/D`. Measured `T_onecard` includes on-card exchange
copies; the gate subtracts those, using the measured `LocalExchange` copy time for the plan's exchanges.

*PR 2 correction:* one card runs every rank, so the gate compares the model's compute summed over **all ranks**
(`Σ_steps Σ_r rank_segment(step, r)`) against measured `T_onecard − T_exchange`, not one rank against `T_onecard/D`. The
exchange copy time is measured by running the same plan with every `Local` step emptied (an exchange-only plan) through
`DistSvBackend::run_plan`; no timing hook is needed and allocation cancels in the subtraction. The representative-rank
estimate `compile` uses, `R · cost(R−1)`, is reported alongside but not gated (an upper bound by design). Bench:
`crates/aleph-cuda/tests/dist_cost_gate.rs`.

If the gate fails, the per-kind model is revised and re-checked before going further, and the finding is recorded in
the report. Shipping an optimizer that aims at a wrong objective is not acceptable.

### 6.4 Candidates in `compile`

{Naive, Lookahead, Reorder{max_k = 1..=g}} × {identity placement, §5 placement}. Naive and Lookahead get the
non-identity placement by starting `plan.rs` from that map. This needs a `plan_from(map)` entry, which is public and lands in PR 1;
`plan()` stays identity.

That is ≤ 2·(2+g) plans, 12 at g=4. The cheapest by `T` wins; ties go to fewer exchanges, then to fewer local passes.
Compile time is reported, and the target is < 50 ms for ~1k gates at g=2.

## 7. Correctness and testing

- **DAG soundness (proptest, CPU):**
  - Random circuits, n ≤ 8, with H/X/Y/Z/S/T/Rx/Ry/Rz/U3, Cnot, Cz, CRx/CRy/CRz, Toffoli, Ccz, Swap, externally
    controlled gates, and `DiagonalPhase`.
  - For each circuit, sample 8 **random** topological orders allowed by the DAG. Each order must equal the original on
    the naive SV backend to 1e-10.
  - Random orders test the DAG itself, not just the scheduler's choice.
- **Classification unit tests** for each table row in §3, including the tricky ones: CNOT control vs target, CRy target
  is Other, external controls, multi-mask `DiagonalPhase`.
- **Plan equivalence (proptest, CPU):** `Router::Reorder{max_k}` for g ∈ {1,2,3}, all `max_k`, with identity and §5
  placement, through `aleph_sv::dist_ref::run_dist`, must equal the single-node oracle to 1e-10.
  `compile` with a stub `CostModel` must do the same.
- **Plan invariants:**
  - every original gate is emitted exactly once;
  - every `Local` instruction's required-local qubits are local (`specialize` never returns `GlobalTarget`);
  - `final_map` is a permutation.
- **GPU oracle:** extend `dist_gpu_oracle` cases (ghz, qft, brick, grover, diag) to `Router::Reorder` and
  `run_compiled`, FP64 and FP32.
- **Regression pin:** Reorder is never worse than Lookahead in exchange count on GHZ/QFT (like P6-03's pin). This holds
  for exchange count, not necessarily bytes; if bytes differ, `compile` handles it.

## 8. Benchmark and acceptance

**Workloads:** n=28, D∈{2,4}, FP64. QFT, GHZ, random brickwall d=10, Grover (multi-controlled), QAOA Max-Cut p=2 on a
ring plus 7 chords `(i, i + n/2)`, even `i < n/2`, CCZ ladder d=4. FP32 is reported for QFT and random.

**Metric:** predicted `T` with *measured* compute. That is `T_onecard(R=D)/D` from `dist_local_bench` on the RTX 4000,
minus on-card copy time, plus `bytes/BW(k)` with the AWS FP64 table. It is computed for the Naive, Lookahead and
compiled plans, so the model's compute estimate is not the evidence here.

**Exit criteria:**
1. compiled ≤ min(Naive, Lookahead) + 3 % on every cell.
2. random d=10 at D=2 is at least **15 %** better than Lookahead on predicted `T`. If this is missed, the report records
   it honestly with the cause, and no claim is inflated.
3. The model gate (§6.3) holds on every cell.

**Also reported:**
- CPU comm stats for n=30–32 at g ∈ {2,3} (exchanges, amps moved, local swaps), extending the P6-03 table. This is
  `aleph-sv/examples/dist_comm_counts.rs`.
- Compile time per workload.
- Which candidate `compile` chose, per cell.

**Report:** `docs/perf/p6-05-compiler.md`, linked from `docs/perf/p6-multi-gpu.md`.

## 9. Delivery

Three PRs, each with green CI. GPU-specific tests run on the CUDA box.
1. **DAG + scheduler + placement.** `dag.rs`, `schedule.rs`, `placement.rs`, `Router::Reorder`, the CPU proptests and
   invariants, and the CPU comm-stats table.
2. **Cost model.** The `CostModel` trait, `LinkModel`, `GpuCostModel`, the calibration test and constants, and the
   model accuracy gate (§6.3) with its numbers.
3. **Compile.** `compile`, `run_compiled`, the GPU oracle extension, the §8 benchmark, the report, and a
   CLAUDE.md overview line.

## 10. Risks

- **Reordering can hurt fusion.** Moving gates across an exchange splits blocks that used to fuse. Mitigation: drain in
  lowest-index-first order, and `compile` still has Lookahead as a fallback candidate.
- **Model error on cheap kinds.** Diagonal and swap launches are latency-dominated at small m. Mitigation: calibrate at
  m_ref near the target m; the §6.3 gate catches any remaining error.
- **Representative rank.** Ranks other than R−1 skip global-controlled gates and therefore run less. Using R−1 is an
  upper bound on per-rank compute. Ranks also synchronise at each exchange, so the slowest rank sets the pace anyway.
