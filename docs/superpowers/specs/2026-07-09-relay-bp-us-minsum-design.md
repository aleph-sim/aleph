# Q7-02 M7 — µs-latency relay-BP: synthesis-friendly full-unroll min-sum

**Status:** design (2026-07-09). Continues the Q7-02 FPGA relay-BP track (Advances #322).
**Predecessor:** M6 (PR #452) put circuit-level relay-BP on KV260 silicon at **6.72 ms** (`bp_relay_bram_dp`,
2× the Arty purely from clock) and proved the µs premise dead with the *current* RTL.

## 1. Motivation & root cause (from M6)

M6's OOC sweep showed the two cores that could exploit the KV260 fabric do **not** synthesize at circuit
scale (`BP_N=864 / BP_C=144 / BP_E=2952`, max check-degree 25):

- `bp_relay_fast` (full spatial unroll): all 144 deg-25 min-sums **and** ~2952×2 constant-blend multiplies
  in **one flat combinational cloud** per phase → Vivado's Cross-Boundary-Area-Optimization OOM'd (~43 GB,
  ~1 h, killed).
- `bp_relay_partial_fast`: to avoid runtime-index muxing it **replicates the gather/mux across all groups**
  (constant-index, per §6), so the mux network approaches full-unroll size → same area-opt stall (>40 min).
- BRAM cores fit but are edge-serial (≤2 edges/cycle) → ~672 000 cycles → ms-scale.

**Key latency physics.** BP iterations are strictly sequential (iteration *n+1* consumes *n*'s messages), so
a pipeline **cannot** be overlapped across iterations. 6×10 = 60 iters × 2 phases = ~120 dependent phases.
To land ≤3 µs @100 MHz (~300 cycles) the design must process **≈ all 2952 edges per phase** — i.e. near-full
spatial unroll, messages register-resident. BRAM tops out at ~288 accesses/cycle (144 tiles × 2 ports) →
≈10 µs floor. **µs ⇒ register-resident full unroll; the only open problem is making it synthesize + fit.**

**Root cause of the wall is largely RTL *structure*, not fundamental logic:** `bp_relay_fast` writes the whole
unroll as flat inline `for` loops inside one `always_ff`, so Vivado sees a single ~300k-cell flattened block
that area-opt cannot converge on. The same logic, expressed as **stamped hierarchical submodules with
registered pipeline stages**, gives the optimizer small repeated units — the standard escape hatch.

## 2. Goal

- **Target latency:** true **~1–3 µs** worst-case (60-iter full schedule) on KV260 silicon.
- **Correctness bar:** LER-preserving (see §5). Bit-exact to `FixedRelayBp` is kept **wherever the change is
  pure pipelining**; the fixed-point may be re-derived (narrower width / reordered reduction) **only** if fit
  forces it and the Monte-Carlo LER oracle stays within band.
- **Drop-in:** same top-level ports as `bp_relay_fast` → rides the existing `bp_axi_wrap_wide` (IDCODE
  `0x4250_0002`), no wrapper change, reuses the M6 board-build tcl + `bp_circ_kv260.py` runner.

## 3. Architecture — `bp_relay_unroll_pipe.sv`

Register-resident full unroll (as `bp_relay_fast`) but **hierarchical + shallow-pipelined**.

**Unchanged from `bp_relay_fast`:** `m_vc[BP_E]`, `e_cv[BP_E]`, `ehat[BP_N]` in flop arrays accessed **only at
compile-time-constant edge indices** (no runtime mux — the partial_fast trap); 6×10 schedule; SAT-overlap;
top ports (`clk,rst_n,in_valid,syndrome_in[BP_C],busy,out_valid,corr_out[BP_N],obs_flip,valid_flag,latency_cycles`).

**Two structural changes (the fix):**

1. **Hierarchical modularization** — replace the flat inline loops with stamped submodules:
   - **`check_minsum`** — one check's min-sum. Inputs: its ≤`BP_CHK_DEG` `m_vc` values + syndrome bit.
     Outputs: its ≤`BP_CHK_DEG` `e_cv` values. Internals: min1/min2/argmin/sign reduction → `e_cv = ±(exmin −
     (exmin>>3))` (α=7/8, multiply-free). **144 instances**, each parameterized/wired to its check's constant
     edge list. (Degree varies; either one degree-parameterized module or a small set of degree variants.)
   - **`var_update`** — one variable's update. Inputs: its ≤`BP_VAR_DEG` `e_cv` + `λ` + `γ(leg)`. Outputs: its
     `e_cv`-edge `m_vc` values + `ehat` bit. Internals: `total = λ + Σ e_cv`; per edge
     `blend = γ·old + (1−γ)·(total−e_cv)` with `γ` a **per-var compile-time constant** (from `BP_GAMMA`,
     one of `BP_LEGS`=6 constants selected by `leg`) → constant-coefficient multiply = LUT shift-add, no DSP
     needed. **864 instances**.
   - `synth_design -flatten_hierarchy none` (+ `-mode out_of_context` for the gate) so Vivado synthesizes each
     small module **once** and stamps copies instead of flattening one mega-cloud.

2. **Shallow pipeline (2 stages / phase)** inside each submodule: stage-1 registers the reduction
   (min1/min2 for check; `total` for var), stage-2 computes+registers the outputs. Bounds combinational depth
   for timing and hands the optimizer small clouds. Depth stays **2** (not deep) because sequential iterations
   cannot hide pipeline latency.

**FSM / cycle count.** Per iteration: `S_CHECK` (2 cyc, all 144 `check_minsum`) → `S_VAR` (2 cyc, all 864
`var_update`), SAT overlapped as in `bp_relay_fast`. ~4 cyc/iter × 60 + init/SAT/emit ≈ **~250 cycles →
2.5 µs @100 MHz**; a 1-cyc-phase variant and/or a higher FCLK (the -2LV part reports OOC Fmax >300 MHz for
light logic) are levers toward the ~1 µs end.

## 4. Step-0 fit gate (go/no-go) — the de-risk

Before building the full FSM core, synthesize a **representative slice** to answer "does this fit + synthesize
at all":

- Build `check_minsum` + `var_update` + a skeleton top that instantiates **all 144 + 864** (constant-wired to
  the real edge lists) with a trivial/no FSM — enough for OOC to place the real logic.
- OOC synth at `xck26-sfvc784-2LV-c`, `-flatten_hierarchy none`. Measure **LUT vs 117 120, DSP vs 1248, FF**,
  and **whether area-opt COMPLETES** (the flat version did not).
- **Decision:** completes + fits (<~90% LUT) + timing plausible → **GO** build the full core. Overflows → pull
  the MSG_BITS-narrowing lever (§5), re-measure. Still overflows / still stalls → **fall back to B** and
  document the honest result.

## 5. Correctness & verification (Verilator-first, §4)

Two tiers, preferring the stronger:

- **Primary — bit-exact.** As long as the change is **pure re-timing** (registered stages, identical
  arithmetic / order / width — values unchanged), the core stays bit-exact to `FixedRelayBp`. Reuse the
  existing bit-exact TB (`tb_bp_relay.cpp`) → all 40 circuit vectors bit-identical, plus a thread/pipeline-flush
  invariance check. New make target `bpunrollpipe`.
- **LER oracle — only on the precision lever.** If fit forces narrower `MSG_BITS` (values change), add a
  Monte-Carlo LER check: decode N≈10 000 circuit-noise shots through the RTL and through `FixedRelayBp`, assert
  the RTL LER is within the 5σ band of the reference (reuse `aleph_oracle::assert_distribution_close`). Bits are
  dropped only here, only if LER holds.
- **On silicon.** Reuse the M6 flow: `bp_axi_wrap_wide` + `kv260_bp_circ_bd.tcl` (swap the core), 40-vector
  decode via `bp_circ_kv260.py` (Overlay-bypass), IDCODE `0x4250_0002`, both full + early-exit, latency report.

## 6. Latency & fit model (to confirm against silicon)

| | cycles | @100 MHz | @ ~200 MHz |
|---|---|---|---|
| target (2-cyc phases) | ~250 | 2.5 µs | 1.25 µs |
| stretch (1-cyc phases) | ~130 | 1.3 µs | 0.65 µs |
| M6 baseline (`bram_dp`) | 672 000 | 6.72 ms | — |

Fit budget (Step-0 decides): 144 min-sums + 864 var-updates + ~2952 constant-blends on **117 120 LUT / 1248
DSP**. The min-sum comparator trees dominate LUT; constant-blends are shift-adds (cheap). Narrowing `MSG_BITS`
(LER permitting) is the primary fit lever.

## 7. Fallback B (if Step-0 = no-go)

Banked-BRAM streaming: partition `BP_E` across many BRAM banks (≤288 accesses/cycle) + a pipelined reduction →
~10 µs floor. Fit-safe, synthesizes, but misses 1–3 µs by ~3–10×. Documented as the honest alternative; the
"µs unreachable at this precision/fabric" outcome is itself a valid result (cf. M6).

## 8. Deliverables

- `hw/bp_relay_unroll_pipe.sv` (+ `check_minsum` / `var_update` submodules, in-file or split).
- `hw/tb_*.cpp` reuse / new target `bpunrollpipe`; LER oracle only if the precision lever is used.
- Step-0 OOC tcl (reuse `hw/syn/ooc.tcl` pattern) + a KV260 board build (reuse `kv260_bp_circ_bd.tcl`, swap core).
- `docs/perf/qec-q7-fixed-bp.md` — M7 section (fit, silicon latency, honest scope).
- PR `[Q7-02] M7: …`, **Advances #322** (umbrella, not Closes).

## 9. Honest scope

µs is a genuine stretch with real fit/timing risk; Step-0 exists precisely so we learn go/no-go from **one**
synthesis before committing the full build. If it fits, this is the first µs-class circuit-level qLDPC decoder
on the KV260; if it doesn't, we bank the ~10 µs banked-BRAM result and the documented reason. Either way the
claim stays measured on silicon, never projected.

---

## AMENDMENT 2 (2026-07-09) — banked-store pivot: check-major banking + König edge-coloring

**Why.** Both non-banked paths are now empirically dead on `xck26` (see the plan's Amendment 1 and the M7
handoff): the modular **full unroll synthesizes but does not fit** (453k LUT = 386 %), and the modular
**partial unroll fits but does not synthesize** — area-opt stalls on the runtime-`grp` gather muxes over the
`m_vc`/`e_cv` flop arrays (~O(BP_E)-wide operand muxes), independent of the compute modularization and of
`-directive RuntimeOptimized`. §7's banked-BRAM fallback is therefore the path; this amendment specifies it
concretely, and the offline analysis below makes it *stronger* than §7's ~10 µs guess.

### A2.1 Architecture — `bp_relay_banked.sv`, parameterized `(W, V)`

Scale `bp_relay_bram_dp`'s 2-bank idea to hundreds of tiny LUTRAM banks, banked **check-major with a
β split** so the check passes are hardwired and the scattered var passes stay ≤1 access/bank/cycle:

- **Message stores.** Three LUTRAM bank arrays (all tiny; **zero BRAM tiles**). Let `GC = ⌈144/W⌉`,
  `GV = ⌈864/V⌉`; the emitter assigns each check a `(group g, slot j)` and each var a `(group h, slot i)`.
  - `m_cm[W·25·2]` half-banks: variable→check messages. Edge `e` = check `c`'s **logical** position-`k`
    edge lives at half-bank `(slot j(c), k, β(e))`, `β(e) ∈ {0,1}` chosen offline; row = a per-half-bank
    packed index (ROM of its ≤GC resident groups). Logical `k` is **never permuted** — β only splits.
  - `e_cm[W·25]` banks × `GC` rows: check→variable messages, banked `(j, k)`, row `g` — no split
    (reads tolerate 2/bank: both LUTRAM read ports; written only by the CHK phase, conflict-free).
  - `m_vm[V·6]` banks × `GV` rows: a **var-major shadow copy** of `m_vc` (bank `(i, d)`, row `h`).
    Written together with `m_cm`; read only by the VAR phase (the "old message" blend operand).
- **Access pattern (conflict-free by construction, II=1):**
  - CHK phase, group `g`: each `check_minsum` lane `(j,k)` reads `m_cm` via a **2:1 β-mux** + row ROM
    (select/row are compile-time functions of `g`); two cycles later slot outputs write `e_cm[(j,k)]`
    row `g−2` hardwired. One check-group per cycle → `GC+4` cyc/phase.
  - VAR phase, group `h`: V stamped `var_update` slots read `e_cm` scattered (≤GV:1 ROM-driven mux per
    operand; ≤2 reads/bank guaranteed) + `m_vm` mux-free; two cycles later write blended `m_vc` to
    **both** `m_cm` (≤1 write/half-bank/cycle guaranteed, ≤GC:1 source mux per half-bank) and `m_vm`
    (mux-free). One var-group per cycle → `GV+4` cyc/phase.
  - SAT overlapped on `ehat` flops (per-(j,k) 1-bit GC:1 gathers), INIT via the VAR write path (λ),
    EMIT V vars/cycle — FSM structure carried from `bp_relay_unroll_pipe` (launch/scatter software
    pipeline), plus `early_exit` + 32-bit `latency_cycles` for a clean `bp_axi_wrap_wide` drop-in.

### A2.2 The conflict-freedom result (measured on the real graph, not conjectured)

The one hard constraint is the VAR phase: within a var-group, its edges must hit distinct half-banks.
Distinct slots/positions are distinct banks automatically; the residual collisions — same `(j,k)` from
two same-slot checks — are handled by the emitter's offline solve: (1) slot assignment + var grouping
such that every var-group hits every `(j,k)` **≤2** times (greedy + local repair; verified exactly on
`bb_gross_tanner.svh` at all four configs below), then (2) β splits each such pair — always possible.
If a future graph defeats the cap-2 grouping search, the guaranteed fallback is per-check bank
permutations via bipartite edge-coloring (König, ≤25 colors — exists for any graph of this degree;
costs 25:1 CHK muxes ≈ +20k LUT, measured feasible on this graph too). Verified configs:

| W (chk/cyc) | V (var/cyc) | m_cm half-banks | cyc/decode | @100 MHz | @200 MHz | est. LUT |
|---|---|---|---|---|---|---|
| 8  | 24 | 400×≤18 | ~3 946 | 39.5 µs | 19.7 µs | ~40–45k |
| 12 | 36 | 600×≤12 | ~2 836 | 28.4 µs | 14.2 µs | ~55–60k |
| 16 | 48 | 800×≤9  | ~2 281 | 22.8 µs | 11.4 µs | ~65–75k |

Residual muxes are **bounded by construction**: `e_cm` reads ≤GV:1 per var operand (measured avg 15–22),
`m_cm` writes ≤GC:1 per half-bank (avg 7–13), CHK β-muxes 2:1 — vs the ~O(BP_E) operand muxes that
stalled partial-unroll's area-opt. Rejected on data: naive `edge % K` (up to 15 same-bank edges/cycle →
stalls, ~100 µs class) and unrestricted search-based banking (feasible but ~128:1 muxes/operand ≈
60–130k LUT of routing).

### A2.3 Correctness — bit-exact, **no golden regeneration**

Banking never touches logical edge order: `check_minsum`'s min1/min2/argmin reduction consumes positions
`k` exactly as today (β/row are pure physical placement), so tie-breaks — and every message value — are
unchanged; var-side sums are wrap-add order-invariant, and grouping is the same pure-scheduling change
already proven G-invariant for `bp_relay_unroll_pipe`. The core stays **bit-exact to the existing
`FixedRelayBp` golden and `bp_circ_vectors.txt`** (same TB pattern; plus a (W,V)-invariance check).
§5's LER oracle stays unused (no precision change). The `MSG_BITS` narrowing lever remains available but
unneeded (LUTRAM stores bits cheaply).

### A2.4 Emitter & deliverables delta

- `qec_q7_bp_graph` gains the offline solve (`(W, V)` parameters): slot assignment, cap-2 var grouping,
  β assignment — emitting `check→(group, slot)`, `var→(group, slot)`, `β(e)` alongside the existing CSR
  tables (which stay byte-identical). Address/select ROMs derive from these in RTL `generate`
  (compile-time constants — the partial_fast rule). The solve asserts feasibility loudly.
- New core `hw/bp_relay_banked.sv`; OOC fit/Fmax probes at (8,24), (12,36), (16,48) → pick the fastest
  that fits ≤ ~105k LUT and closes timing; then the M6 board flow (`kv260_bp_circ_bd.tcl` swap,
  `bp_circ_kv260.py`), silicon numbers, `docs/perf/qec-q7-fixed-bp.md` M7 section, PR **Advances #322**.
- Honest expectation vs M6's 6.72 ms: **~11–40 µs on silicon** (~200–500×), decided by fit + Fmax, not
  projection.
