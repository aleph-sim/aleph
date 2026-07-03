# Q7-02 M0 — fixed-point relay-BP golden (RTL sizing)

**Track:** Q7-02 (RTL implementation of the core decoder block), milestone **M0**.
**Depends on:** Q5-03 (relay-BP), Q5-01 (gross code + DEM).
**Status:** done.

## What and why

The pre-ASIC frontier gap (ROADMAP §7, lever #4) is that our on-silicon decoder is UF surface-code
only — the **qLDPC frontier decoder is software-only**. #4 puts it on silicon. The chosen hardware
target is **relay-BP** (Q5-03), not classic BP+OSD: relay-BP already beats BP+OSD at every `p`
(`qec-q5-qldpc.md`) *and* drops OSD's data-dependent GF(2) Gauss–Jordan elimination entirely, so it
is pure fixed-schedule message passing — the same bounded-pass-per-cycle datapath shape that carried
UF to silicon (Q6).

An FPGA/ASIC cannot carry `f64` through 432 message edges. M0 builds the **fixed-point golden** —
the exact integer arithmetic the RTL will implement — and finds the **narrowest message word** whose
logical-error rate still matches the `f64` relay-BP decoder. That word width is the primary RTL
sizing input (message RAM, adder/comparator widths, routing).

`FixedRelayBp` (`crates/aleph-qec/src/fixed_bp.rs`) is the bit-accurate twin of `RelayBpDecoder`:
identical Tanner layout, `γ` seeding (SplitMix64), leg/iteration schedule, and
keep-lowest-weight-valid rule — only the arithmetic is quantised.

### Fixed-point scheme (= the RTL spec)

- A value is stored as signed `round(x·2^F)`, saturated to `±(2^(W−1)−1)`; `W` = `msg_bits`,
  `F` = `frac_bits`. Both message buffers (var→check, check→var) live in this `W`-bit word.
- **α = 0.875 = 7/8 is multiply-free**: `mag − (mag >> 3)`. The whole min-sum check update is
  compare / min / sign — **no multiplier**.
- The **only multiply in the datapath** is the relay memory blend `(1−γ_v)·computed + γ_v·m_old`,
  and `γ_v` is a *per-variable constant* (seeded ROM) — a fixed-coefficient multiply / small LUT,
  not a general one.
- The blend truncates by arithmetic shift (`num >> F`, floor) — the cheapest RTL rounding (no adder).
  The golden adopts it so software and silicon agree bit-for-bit.
- The per-node accumulator is kept wider than a message (here `i64`; RTL sizes it
  `W + ⌈log2(deg)⌉`); only stored messages re-saturate to `W`.

## Result — message width sweep

Gross `[[144,12,12]]`, independent-`Z` code capacity, 20 000 shots, seed 2024. `f64` = `RelayBpDecoder::new`.
`cargo run --release -p aleph-qec --example qec_q7_fixed_bp`. Data: `docs/perf/data/qec-q7-fixed-bp.csv`.

LER as a ratio to the `f64` relay-BP baseline (1.00 = identical); every cell below is **within
Monte-Carlo CI** of the `f64` rate:

| word `(W,F)` | range / step | p=0.02 | p=0.03 | p=0.04 | p=0.05 |
|--------------|--------------|--------|--------|--------|--------|
| f64 (ref)    | —            | 1.00e-4 | 2.00e-3 | 1.27e-2 | 4.35e-2 |
| (6, 2)       | ±7.75 / 0.25   | 2.00× | 1.20× | 1.06× | 0.97× |
| (7, 3)       | ±7.88 / 0.125  | 2.50× | 1.18× | 1.04× | 0.99× |
| **(8, 3)**   | **±15.9 / 0.125** | **1.00×** | **0.97×** | **0.95×** | **0.96×** |
| (8, 4)       | ±7.94 / 0.0625 | 1.50× | 1.18× | 0.98× | 0.99× |
| (10, 4)      | ±31.9 / 0.0625 | 1.00× | 0.97× | 0.97× | 0.98× |
| (12, 6)      | ±31.9 / 0.016  | 1.00× | 0.95× | 0.99× | 0.99× |

### Verdict: **8-bit signed, 3 fractional bits (Q5.3)** is the RTL message word.

`(8,3)` tracks `f64` tightest of all candidates *and* is the only 8-bit word that reproduces the
`f64` rate exactly at the p=0.02 floor (both 1.0e-4). The reason it beats the finer `(8,4)` there:
at low `p` the priors are largest (`λ(0.02) ≈ 3.9`, `λ(0.01) ≈ 4.6`) and messages accumulate, so the
±15.9 integer range matters more than the extra fractional bit. Below 8 bits `(6,2)/(7,3)` stay
within CI at higher `p` but bias high at the floor (2–2.5×) where accuracy matters most — so 8 bits
is the floor, not a comfort margin.

The truncating blend acts as a mild extra damping, which is why `(8,3)` sits slightly *below* `f64`
(0.95–0.97×) at higher `p` — a harmless, even favourable, quantisation artefact.

### Silicon sizing at (8,3)

| structure | size |
|-----------|------|
| message RAM (m_vc + e_cv), 432 edges × 2 × 8 b | **864 B** (was ~7 KB at f64 → **8× smaller**) |
| `γ` disorder ROM, 4 legs × 144 vars × 8 b | 576 B |
| `λ` prior ROM, 144 × 8 b | 144 B |
| syndrome register | 72 b |
| datapath multipliers | **1 fixed-coeff** (relay blend); min-sum is multiply-free |

The floor regime (p ≤ 0.02) needs more than 20 k shots to be conclusive on its own; `(8,3)` already
matches `f64` exactly there, so it is not gating.

## Acceptance

- [x] Fixed-point relay-BP golden implemented, deterministic, valid decisions reproduce the syndrome
      (unit tests in `fixed_bp.rs`).
- [x] Width sweep vs `f64` relay-BP produced; the narrowest matching word identified (**Q5.3**).

## Files

- `crates/aleph-qec/src/fixed_bp.rs` — `FixedRelayBp` + `FixedHwView` + unit tests.
- `crates/aleph-qec/examples/qec_q7_fixed_bp.rs` — the width sweep.
- `docs/perf/data/qec-q7-fixed-bp.csv` — committed run.

-----

# Q7-02 M1 — Tanner `.svh` + combinational check-update RTL

**Status:** done (Verilator).

## What

The RTL half of the check→variable min-sum update, plus the generator that bakes the gross-code
graph into a SystemVerilog header (as `uf_surface_graph.svh` does for UF).

- `qec_q7_bp_graph.rs` emits **`hw/bb_gross_tanner.svh`**: `BP_N=144`, `BP_C=72`, `BP_E=432`,
  `MSG_BITS=8`, `FRAC_BITS=3`, `MAX_MAG=127`, `BP_LEGS=4`, `BP_ITERS=25`, the flattened CSR
  (`BP_VAR_OFF`, `BP_EDGE_VAR`, `BP_CHECK_OFF`, `BP_CHECK_EDGES`), quantised priors `BP_LAMBDA[144]`,
  and disorder `BP_GAMMA[4*144]`. `BP_VAR_OFF = {0,3,6,…}` confirms the constant variable-degree 3;
  the whole graph (M2 needs all of it) is in one header.
- `hw/bp_check_update.sv` is the combinational datapath: for each degree-6 check, the two-pass
  exclusive-minimum (min1/min2 + argmin), the check parity / per-edge sign, and **α = 7/8 as
  `mag − (mag >> 3)`** — no multiplier. Like the Q6-02 combinational UF draft it is one `always_comb`
  cloud (M2 sequentialises it).

## Verification

`make -C hw bpcheck` replays 256 random `(syndrome, m_vc)` vectors — `m_vc` spanning the full signed
Q5.3 range — through the RTL and checks all 432 outputs per vector against
`FixedRelayBp::check_update_once`:

> **PASS: 110 592 / 110 592 check-update outputs bit-identical to the fixed-point golden.**

The RTL check-update is the exact silicon twin of the M0 golden. Next (M2): sequentialise into a
clocked FSM (S_CHECK → S_VAR-with-memory → leg/iteration loop → keep-best-valid) for the full
relay-BP decode, reusing this `.svh`.

## Files

- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — `.svh` + vector generator.
- `hw/bp_check_update.sv` — combinational check-update RTL.
- `hw/tb_bp_check.cpp`, `hw/Makefile` (`bpcheck`) — Verilator TB.
- `hw/bb_gross_tanner.svh`, `hw/bp_check_vectors.txt` — generated.

-----

# Q7-02 M2 — sequential FSM full relay-BP decoder

**Status:** done (Verilator).

## What

`hw/bp_relay_decoder.sv` — the complete relay-BP decode of `FixedRelayBp` time-multiplexed into a
clocked FSM, **one bounded pass per cycle** (the Q6-04 UF methodology): a cycle touches exactly one
check (≤ `BP_CHK_DEG`=6 edges) or one variable (≤ `BP_VAR_DEG`=3 edges), so per-cycle combinational
depth is bounded and graph-size-independent. Per iteration:

- `S_CHECK` (72 cyc): `e_cv ← min-sum(m_vc, s)`, one check/cycle (multiply-free α=7/8).
- `S_VAR` (144 cyc): `m_vc, ehat ← var-update-with-memory`, one variable/cycle — the **memory blend
  `(1−γ)·computed + γ·m_old`, the one multiply**, γ from the per-leg disorder ROM; running Hamming
  weight tracked here.
- `S_SAT` (72 cyc): `all_sat ← (H·ehat == s)`; keep the lowest-weight syndrome-valid `ehat` seen.

looped `BP_LEGS`×`BP_ITERS` = 4×25, then `S_EMIT` (144 cyc) reduces the chosen `ehat` → observable
flips, `S_DONE` pulses `out_valid`. Handshake mirrors the UF core (`in_valid`/`busy`/`out_valid`/
`latency_cycles`).

## Verification

`make -C hw bprelay` drives 65 syndromes (empty + every single-variable error + 40 random low-weight)
through the FSM and checks the chosen error `corr_out[144]`, observable flips `obs_flip[12]`, and the
validity flag bit-for-bit against `FixedRelayBp::decode_fixed_ehat`:

> **PASS: 65 / 65 full decodes bit-identical to the fixed-point golden.**

Bit-exact agreement with the golden means the RTL's logical-error rate **is** the golden's (which is
within CI of `f64` relay-BP, M0) — no separate statistical LER run needed.

## Honest latency headline (the M3/M4 target)

> **Worst-case latency = 28 944 cycles/decode.**

That is `4·25·(72+144+72) + 144` — the full legs×iterations schedule with **no early exit** (relay-BP
must keep the best across *all* legs). At a nominal ~150 MHz that is ~193 µs — far over a ~1 µs round
budget. This is the honest cost of a real qLDPC decoder and exactly what M3 (measure it in silicon)
and M4 attack: **process K checks/variables per cycle** (the UF `FOREST_UNROLL` lever — the graph's
constant degree 6/3 and 72/144 independent nodes per pass make this embarrassingly parallel, so a
K-wide datapath cuts the pass length by K), plus revisiting the leg/iteration budget. The point of M2
is a *correct, bounded-depth, synthesizable* baseline to optimise from.

## Files

- `hw/bp_relay_decoder.sv` — sequential relay-BP FSM.
- `hw/tb_bp_relay.cpp`, `hw/Makefile` (`bprelay`) — Verilator TB.
- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (`decvectors` mode) — full-decode golden vectors.
- `crates/aleph-qec/src/fixed_bp.rs` — `decode_fixed_ehat` exposes the chosen `ehat` for the TB.
- `hw/bp_dec_vectors.txt` — generated.

-----

# Q7-02 M3 — Vivado OOC synth (Zybo + KV260)

**Status:** done. Vivado 2024.2, out-of-context synth + place + route, both target parts (same flow as
the UF `syn/synth.tcl`). Ran on `openwebgui`.

## Result — the M2 baseline is nowhere near real-time (as expected)

| part | LUTs | FFs | DSP | BRAM | Fmax | latency (28 944 clk) | vs ~1 µs budget |
|------|------|-----|-----|------|------|----------------------|-----------------|
| Zybo `xc7z020` | 23 596 (44%) | 7 601 | 0 | 0 | **28.3 MHz** | **1023 µs** | ~1000× over |
| KV260 `xck26`  | 24 119 (21%) | 7 616 | 0 | 0 | **67.8 MHz** | **427 µs**  | ~430× over |

It **fits** (LUT-heavy, no DSP, no BRAM) but is 2–3 orders of magnitude off real-time. Both the
cycle count (28 944) *and* Fmax are bad. This is the honest baseline M4 optimises.

## Critical path — the cursor mux, not the multiply

The binding path on both parts is `idx_reg → e_cv_reg[*]` (Zybo **44 logic levels**, 35.4 ns,
**74% routing**; KV260 37 levels, 14.8 ns, 64% routing). Diagnosis:

- **The `idx` cursor is the wall.** Reading `BP_CHECK_OFF[idx]` then `BP_CHECK_EDGES[off+k]` with a
  *runtime* check cursor synthesises to a 72-way select feeding the min-sum, whose result then fans
  out through a 432-way address-decode to write `e_cv` (observed net fanouts of **586 / 626**). That
  giant mux/demux — not arithmetic — is the deep path. Same root cause the UF decoder never hit
  because its per-cycle graph was tiny (d=3: 18 edges); here it is 432 edges / 72 checks.
- **The multiply is free.** `DSP = 0` on both parts: the relay memory blend's `γ`/`(1−γ)` are small
  per-variable *constants*, so Vivado folded the "multiply" into a few LUTs. Confirms the M0 design
  call — the one multiply is not the bottleneck.
- **Routing-dominated** (74% on Zybo) — the same tax the Q6-03 UF report flagged, and the tax an
  ASIC removes. Feeds lever #3 (the logic-vs-routing ASIC argument).

## Verdict → M4 architecture

The per-node **cursor** FSM was the wrong micro-architecture for a 432-edge graph. The fix is
**spatial unrolling**: wire each check's min-sum to its *constant* 6 edges and each variable's update
to its constant 3 edges, and evaluate a whole layer (all 72 checks, then all 144 variables) per
cycle. That deletes the `idx` mux entirely (edges become constant wiring), collapses a
check-pass + var-pass to **2 cycles** (so `legs·iters·2 ≈ 200` cycles, ~140× fewer), and is how real
BP silicon is built. Area grows (72 + 144 small fixed datapaths) but each unit is tiny; the register
count barely moves (the messages already exist as flops). M4 will also then pipeline the layer and
revisit the leg/iteration budget.

## Files

- `hw/syn/synth_bp.tcl` — OOC synth/impl flow for `bp_relay_decoder` (any part).
- `docs/perf/data/` — reports live on `openwebgui:/root/q7synth/reports/` (fmax + util + timing).

-----

# Q7-02 M4 — spatial unrolling (all checks/vars per cycle)

**Status:** done (Verilator + Vivado 2024.2 OOC on `openwebgui`).

## What

`hw/bp_relay_unrolled.sv` replaces M2's runtime node cursor `idx` with a **spatially-unrolled**
datapath: every check's 6 edges and every variable's 3 edges are compile-time constants, so a whole BP
layer evaluates in **one cycle**. The schedule collapses from M2's 72/144/72 cycles-per-phase to

- `S_CHECK` (1 cyc): all 72 min-sum check updates in parallel,
- `S_VAR`   (1 cyc): all 144 var-updates-with-memory in parallel (the one relay blend multiply per var),
- `S_SAT`   (1 cyc): all 72 parity checks + keep-lowest-weight-valid,

looped 4×25, then `S_EMIT` (1 cyc). **Worst-case latency 301 cycles**, down from 28 944 — **96× fewer**.

Bit-exactness is structural: within each M2 phase the nodes are already independent (each Tanner edge
belongs to exactly one check and one variable), so folding 72/144 sequential cycles into one parallel
cycle changes only timing. `make -C hw bpunroll` runs the same golden vectors as M2 (TB shared via
`-DUNROLL`):

> **PASS: 65 / 65 full decodes bit-identical to the fixed-point golden; worst latency = 301 cycles.**

## Two synthesis gotchas the unroll exposed (both real, both fixed in-RTL)

The spatial unroll makes **every array index a compile-time constant**, which turns on Vivado
optimisations that M2's runtime `idx` mux had been (accidentally) suppressing. Two of them silently
**deleted the entire message datapath**, leaving a 158-FF control shell that "closed timing" at a
*meaningless* ~485 MHz. A netlist probe (`get_cells -filter IS_SEQUENTIAL`) caught it: `m_vc`, `e_cv`,
`ehat`, `best_e`, `corr_out` all at 0 FFs. Root causes and fixes:

1. **Async-reset set/reset conflict** (`Synth 8-7137`, *"has both Set and reset with same priority …
   may cause simulation mismatches"*). M2's `@(posedge clk or negedge rst_n)` makes each datapath FF
   an async-reset FF whose S_IDLE constant load (`m_vc←λ`) collides with the async reset. → switched
   to a **synchronous reset**. (Necessary hygiene, but not sufficient here — the fold persisted.)
2. **Sequential constant-propagation folding the 100-iteration feedback to `ehat≡0`.** With constant
   indices, Vivado chases the message recurrence to a bogus fixpoint and proves the decision constant
   (the tell: `found` — which reduces to `syndrome == 0` — was the *only* datapath FF that survived).
   → anchored the message/decision registers with **`(* dont_touch = "true" *)`**. This disables only
   that fold; the reachable logic is still fully optimised. Verilator computes the real
   syndrome-dependent decode (65/65), so this is an over-aggressive synth pass, not an RTL bug.

Lesson for M5+: on constant-indexed unrolled decoders, **verify the post-synth register count against
elaboration** (`report_utilization` FFs ≈ the RTL's ~7.6 k), never trust an Fmax alone.

## Result — the idx-mux wall is gone; the new wall is arithmetic depth + congestion

Full datapath (post-`dont_touch`), OOC synth + place + route:

| part | LUTs | FFs | DSP | Fmax | latency (301 clk) | vs M2/M3 |
|------|------|-----|-----|------|-------------------|----------|
| KV260 `xck26` | 94 194 (80%) | 7 578 | 0 | **95.2 MHz** | **3.16 µs** | **135× faster** than M3's 427 µs |
| Zybo `xc7z020` | 91 829 LUTs = **172.6% of cap** | 7 616 | 0 | **does not fit** | — | full unroll is a KV260-class design |

- **Latency: 427 µs → 3.16 µs on KV260** — 96× from the cycle count, ×1.40 from Fmax (67.8→95.2 MHz).
  The `idx` cursor mux (M3's binding path, 44 logic levels / 74% routing) is **gone**.
- **New critical path = the S_VAR blend.** 25 logic levels (5 CARRY8 chains) through
  `(1−γ)·computed + γ·m_old` + the λ+Σe_cv accumulate — real fixed-point arithmetic, not a mux. DSP
  still 0 (γ folded to LUTs). Fmax is also held back by **routing congestion at 80% LUT**.
- **Area: the honest cost of full unroll.** 72 min-sum + 144 var-update + 72 parity units in parallel
  = 94 k LUTs, fitting the KV260 (80%) but **overflowing the small Zybo** (53 k LUT). M4 is therefore
  a **KV260-targeted** design; a Zybo fit needs *partial* unrolling (K<full nodes/cycle — a middle
  point on the area/latency curve).

## Verdict → still ~3× over the ~1 µs budget; M5 closes it

M4 delivered the two headline wins — **96× fewer cycles** and the **removal of the idx-mux wall** — but
sub-µs needs two more levers, now that the bottleneck is arithmetic-depth-bound not mux-bound:

1. **Pipeline the S_VAR blend** (register the multiply-add) to lift Fmax past the 25-level CARRY8 path.
2. **Revisit the leg/iteration budget** (4×25): fewer legs/iters cuts the 301 cycles directly, at a
   quantified LER cost (re-run the M0 sweep). Early-exit-on-valid is off the table for relay-BP (it
   keeps the best across *all* legs), but the budget itself is tunable.
3. **Partial unroll** for area/congestion relief (and Zybo fit), trading some of the 96× back.

## Files

- `hw/bp_relay_unrolled.sv` — spatially-unrolled relay-BP decoder (sync reset + `dont_touch` anchors).
- `hw/tb_bp_relay.cpp` (built `-DUNROLL`), `hw/Makefile` (`bpunroll`) — shared Verilator TB.
- `hw/syn/synth_bp.tcl` — now takes `[top] [rtl.sv]` args (M3 sequential = default, M4 = unrolled).
- reports on `openwebgui:/root/q7synth/reports/{kv260_unroll,zybo_unroll}/`.

-----

# Q7-02 M5 (step 1) — leg/iteration budget study (the cycle-count lever)

**Status:** budget study done; RTL pipelining + schedule-swap is M5 step 2.

## Why

M4's 301 cycles = `legs·iters·3 + 1` (4 legs × 25 iters) is 3.16 µs at the KV260's 95.2 MHz — ~3× over
the ~1 µs budget. The dominant term is the schedule length `legs·iters`, so the first (and cheapest)
lever is to **stop doing sweeps the decode doesn't need**. `FixedRelayBp::with_budget(legs,
iters_per_leg, …)` makes the schedule explicit; `qec_q7_budget` sweeps it at the hardware word (Q5.3)
and reports each schedule's LER **relative to the full 4×25** plus the RTL cycles/latency it costs.

## Result — the split matters more than the total (80 000 shots, gross code, indep-Z)

LER as a ratio to the full 4×25 relay-BP; `within_ci` at the discriminating p=0.05 (tightest relative
CI). Data: `docs/perf/data/qec-q7-budget.csv`.

| schedule | sweeps | cycles | latency @95.2 MHz | LER ratio (p=0.05) | within CI |
|----------|--------|--------|-------------------|--------------------|-----------|
| **4×25** (full) | 100 | 301 | 3.16 µs | 1.000 | ✓ |
| 4×20 | 80 | 241 | 2.53 µs | 1.028 | ✓ |
| **6×10** | 60 | 181 | 1.90 µs | **1.057** | **✓** |
| 5×12 | 60 | 181 | 1.90 µs | 1.065 | ✗ |
| 4×15 | 60 | 181 | 1.90 µs | 1.067 | ✗ |
| 3×20 | 60 | 181 | 1.90 µs | 1.073 | ✗ |
| 4×12 | 48 | 145 | 1.52 µs | 1.109 | ✗ |
| 2×15 | 30 | 91  | **0.96 µs** | 1.295 | ✗ |

Two findings:

1. **At a fixed sweep budget, more legs / fewer iters-per-leg wins.** All four 60-sweep schedules cost
   the same 181 cycles, but only **6×10 stays within CI** (1.057×) — beating 4×15 (1.067×), 5×12, and
   3×20. Relay-BP's strength is the **leg disorder diversity** (each leg reseeds γ and relays the
   messages); iterations-per-leg have diminishing returns once BP has settled. So the schedule to spend
   cycles on is *many short legs*, not *few long ones*.
2. **The budget lever alone cannot reach sub-µs.** The only sub-µs schedule (2×15, 91 cyc, 0.96 µs)
   costs 1.30× LER — outside CI at every p. Even the aggressive-but-safe 6×10 is 1.90 µs.

## Verdict → M5 step 2 (the sub-µs recipe)

**6×10 (181 cyc, 1.06× LER, within CI at all p)** is the M5 schedule — a **40% cycle cut for no
measurable LER cost**. But 181 cyc is still 1.90 µs at 95.2 MHz, so sub-µs also needs an Fmax lever.
Step 2 (below) applies it and re-measures — and finds the Fmax bottleneck is **not** where this section
predicted (it is the S_CHECK min-sum, not the S_VAR blend). Regenerating the `.svh` at
`BP_LEGS=6 BP_ITERS=10` is a drop-in — the M4 RTL consumes it unchanged.

## Files

- `crates/aleph-qec/src/fixed_bp.rs` — `FixedRelayBp::with_budget` (explicit `legs`/`iters_per_leg`).
- `crates/aleph-qec/examples/qec_q7_budget.rs` — the schedule sweep.
- `docs/perf/data/qec-q7-budget.csv` — committed 80 k-shot run.

-----

# Q7-02 M5 step 2 — adopt 6×10 + right-size the accumulator; re-measure

**Status:** done (Verilator + Vivado KV260 OOC). Two RTL changes, both bit-exact with the golden.

## Changes

1. **6×10 schedule** (2b). The graph emitter switched from `FixedRelayBp::new` (4×25) to
   `with_budget(6, 10, …)` — the step-1 study's schedule. `bb_gross_tanner.svh` regenerates with
   `BP_LEGS=6 BP_ITERS=10` and a 6-leg γ ROM (legs 0–3 keep their original γ, 4–5 are new), so the
   golden and RTL stay bit-exact. **Worst-case latency 301 → 181 cycles** (`60·3 + 1`). Both decoders
   re-verified: `bpunroll` (M4) PASS 65/65 @ 181 cyc; `bprelay` (M2) PASS 65/65 @ 17 424 cyc.
2. **WACC 32 → 16** (2a). M4 carried M2's generous 32-bit blend accumulator, so every
   `total`/`computed`/`num` add was a 32-bit CARRY chain. `|blend| ≤ ~5 600` fits signed 16 bits with
   5× margin, so 16 is bit-exact with 32 at half the CARRY depth.

## Result — 1.68× faster than M4, at no LER cost (KV260 OOC synth + P&R)

| build | schedule | cycles | LUTs | Fmax | latency | vs M4 | LER |
|-------|----------|--------|------|------|---------|-------|-----|
| M4 | 4×25 | 301 | 94 194 | 95.2 MHz | 3.16 µs | 1.0× | 1.00× |
| WACC=16 only | 4×25 | 301 | 86 352 | **100.9 MHz** | 2.98 µs | 1.06× | 1.00× |
| **M5 (6×10 + WACC=16)** | **6×10** | **181** | 93 562 | **96.0 MHz** | **1.89 µs** | **1.68×** | **1.06×** |

The gain is **almost entirely the cycle cut** (301 → 181). The two honest surprises:

- **WACC narrowing's Fmax win (95.2 → 100.9 MHz at 4×25) is *cancelled* at 6×10.** The 6-leg γ ROM is
  ~50% larger than 4-leg, which pushed LUT utilisation back to 80% (86 k → 93.5 k) and the extra
  routing congestion ate the WACC gain — net Fmax 96.0 MHz, essentially M4's 95.2. Right-sizing WACC is
  still kept (strictly smaller adders, headroom at 4×25), but at 6×10 it is masked by congestion.
- **The Fmax wall is the S_CHECK min-sum, not the S_VAR blend.** The binding path on both the WACC and
  6×10 builds is `m_vc_reg[*] → e_cv_reg[*]` — the **check→variable min-sum update** (two-pass
  two-smallest-magnitude + exclude), **route-dominated (55% of the delay) at ~80% util**. The M4→M5
  plan named the S_VAR relay blend as the target; the timing report says otherwise. Pipelining or
  narrowing the *blend* would not have moved Fmax.

## Verdict → the real sub-µs levers (M5 follow-up)

181 cyc / 96 MHz = 1.89 µs is ~1.9× over the ~1 µs budget, and the wall is now **min-sum logic depth +
routing congestion**, so:

1. **Pipeline the S_CHECK min-sum** (register between the two magnitude passes), not the blend — the +1
   cycle/iteration (→ 4/iter, 241 cyc at 6×10) only pays off if it lifts Fmax by more than the 1.33×
   cycle penalty; the route-dominated path means the win is uncertain until measured.
2. **Relieve congestion** — a *partial* unroll (K < all-72 checks/cycle) trades some of M4's 96× cycle
   win back for far less area (M4 is 80% of the KV260) and shorter routes, which the 55%-route path
   says matters more than logic here. This also un-does the γ-ROM congestion penalty.

Both are microarchitecture work with uncertain payoff on a route-bound design — the honest state is
**M5 landed a 1.68× latency cut for free (no LER cost)**; closing the remaining ~1.9× to sub-µs is a
congestion/pipelining problem, not a schedule one.

## Files

- `hw/bp_relay_unrolled.sv` — `WACC=16`.
- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — `LEGS=6 ITERS=10` via `with_budget`.
- `hw/bb_gross_tanner.svh` — regenerated at 6×10 (committed).

-----

# Q7-02 M5-followup — partial unroll: a fast relay-BP decoder that fits the Arty

**Status:** done — `bp_relay_partial` synthesises clean on the xc7z020 and is **20× faster than the
sequential M2 at the same area**. It is the Arty-fitting decoder.

## Why

M4/M5's full unroll is a **KV260-class** design (93.5 k LUT = 80% of the KV260, **172% of a xc7z020**),
so it cannot run on the Zybo/Arty parts we already have. `bp_relay_partial` is parameterised
(`CHK_UNROLL`/`VAR_UNROLL`) to sit between M2 (1 node/cycle) and M4 (all nodes/cycle): process
`CHK_UNROLL` checks / `VAR_UNROLL` variables per cycle via a group cursor. Bit-exact in Verilator at
12/24 (65/65, 1 086 cycles) and at the uneven 16/32 (905 cycles, exercises the `c < BP_C` guard).

## Result — 12/24 on the xc7z020 (= Arty/Zybo part)

| variant | nodes/cycle | cycles @6×10 | LUTs | Fmax | latency | fits xc7z020? |
|---------|-------------|--------------|------|------|---------|---------------|
| M2 sequential (`bp_relay_decoder`) | 1 | 17 424 | 23 596 (44%) | 28.3 MHz | 616 µs | yes |
| **partial 12/24 (`bp_relay_partial`)** | 12 chk / 24 var | **1 086** | **24 790 (47%)** | 35.5 MHz | **30.6 µs** | **yes** |
| M4 unrolled (`bp_relay_unrolled`) | all 72 / 144 | 181 | 93 562 (172%) | 96.0 MHz | 1.89 µs | no |

The partial is the sweet spot for the small part: **~same LUTs as M2 (47% vs 44%) but 16× fewer cycles
(1 086 vs 17 424) → 20× lower latency (30.6 µs vs 616 µs)**. Datapath survives (7 684 FF, no shell), no
DSP, no BRAM. Larger factors (e.g. 24/48) trade area for fewer cycles; 12/24 clears the part at 47%.

## The finding that got here — do the mux on the inputs, not the addresses

The **first** partial draft was Verilator-correct but **synthesis-hostile**: with a runtime cursor an
edge read written as `m_vc[BP_CHECK_EDGES[BP_CHECK_OFF[grp·CU+i] + k]]` is a *nested runtime
indirection* (runtime → offset → edge index → message), which Vivado expanded explosively — it ground
at 100% CPU / 6.6 GB for 18 min in `synth_design` and was killed. **The same cursor-mux wall M3 hit**
(M4 has the identical expression, but `grp·CU+i` is compile-time constant there → a direct wire).

The fix — the shipped RTL — does the time-multiplexing on the **inputs** with **compile-time-constant
addresses**: a *gather → compute → scatter* per slot, where the group selection is an unrolled
`for (g) if (grp == g)` over **literal** edge indices:

```
gather:  for (g) if (grp==g) mm[k] = m_vc[<constant edge of check g*CU+i>];   // G:1 mux of direct wires
compute: one min-sum / var-update on the gathered mm[];                        // one shared unit / slot
scatter: for (g) if (grp==g) e_cv[<constant edge>] <= result[k];
```

Vivado unrolls `g` (constant per iteration), so every array index is a literal and the `if(grp==g)`
chain is a clean `G:1` mux — no runtime address arithmetic reaches the arrays. Same behaviour
(bit-exact), synthesises in minutes instead of grinding. **Lesson: to time-multiplex a
constant-indexed unrolled datapath, mux the operands, never the indices.**

## On silicon — the Arty Z7-20 board bring-up (done)

`bp_relay_partial` now runs on the **real Arty Z7-20** (`xc7z020clg400-1`), the first **qLDPC frontier
relay-BP decoder on owned silicon** (all prior on-hardware decodes were UF surface-code). The value here
is *correctness on real hardware*, not real-time latency — the gross code on this small part is over the
~1 µs surface budget, exactly as expected.

**PS↔PL interface — AXI4-Lite (`bp_axi_wrap.sv` + Verilog top `bp_axi_top.v`).** One syndrome in / one
correction out (code-capacity), so — unlike the Q6 UF throughput path (AXI-Stream + DMA) — the plain
AXI4-Lite control plane is the right fit (same shape as `uf_axi_wrap`/`uf_pynq.py`). The gross code is
wider than a 32-bit word, so the register map spreads the ports: syndrome (72 b) over **3 write words**,
correction (144 b) over **5 read words**, `obs_flip` (12 b) in one; plus CTRL/STATUS/LATENCY and an
IDCODE constant `0x4250_0001` ('BP', v1) for PS↔PL sanity.

**Verified before Vivado.** `hw/tb_bp_axi.cpp` (`make -C hw bpaxi`) drives *real* AXI4-Lite transactions
through the wrapper per golden vector — **65/65 bit-exact, worst latency 1086 cycles**. The Python driver
`hw/sw/bp_pynq.py` is its twin (pynq.MMIO on-board / a `GoldenModel` software backend off-board), so the
protocol self-tests 65/65 with no board.

**Board build (`hw/syn/arty_z7_bp_bd.tcl`, openwebgui Vivado 2024.2).** Zynq-7 PS GP0 → `bp_axi_top`,
generic PS7 (no board files; DDR/MIO from the PYNQ-Z1 FSBL). At **FCLK 25 MHz** (the 12/24 partial's OOC
Fmax was 35.5 MHz; 25 MHz gives in-context margin): **WNS +4.57 ns → TIMING_MET** (in-context Fmax
≈ 28 MHz), **23 881 LUT (44.9 %), 8 400 FF (7.9 %), 0 DSP, 0 BRAM** — the datapath survives placement.

**On-silicon result (`bp_pynq.py bp_arty.bit`).** IDCODE ok; **65/65 decodes bit-identical to the
fixed-point golden**; worst latency **1086 cycles = 43.4 µs @ 25 MHz**. The cycle count matches the
Verilator sim and the Rust `FixedRelayBp` golden exactly — the silicon is the bit-for-bit twin. Board
dir `~/bp`; artifacts routed openwebgui → Mac → board (the cloud box can't reach the private LAN).

## Files

- `hw/bp_axi_wrap.sv`, `hw/bp_axi_top.v` — AXI4-Lite PS↔PL wrapper + Verilog module-ref top.
- `hw/tb_bp_axi.cpp` (`make -C hw bpaxi`) — Verilator TB driving real AXI4-Lite per golden vector.
- `hw/sw/bp_pynq.py` — host driver (on-board pynq.MMIO / off-board GoldenModel self-test).
- `hw/syn/arty_z7_bp_bd.tcl` — Arty Z7-20 block design + bitstream (default FCLK 25 MHz).

- `hw/bp_relay_partial.sv` — parameterised partial unroll (gather/compute/scatter, synth-friendly).
- `hw/tb_bp_relay.cpp` (`-DPARTIAL`), `hw/Makefile` (`bppartial`) — shared Verilator TB.

-----

# Q7-02 M5-followup — the sub-µs levers, measured (KV260 OOC): SAT-overlap wins, min-sum pipeline loses

**Status:** done. The M5 verdict named two *uncertain* levers to close the remaining ~1.9× to sub-µs —
pipeline the min-sum, or relieve congestion. Both are now **measured on KV260** (OOC, `synth_bp.tcl`,
xck26-sfvc784-2LV-c, 3 ns target). The result is a clean **1.48× latency cut for free**, plus a decisive
finding on *why* sub-µs stays out of reach.

## Result — 6×10 schedule, KV260 OOC P&R

| variant | RTL | states/iter | cycles | Fmax | **latency** | LUT | verdict |
|---------|-----|-------------|--------|------|-------------|-----|---------|
| M5 baseline | `bp_relay_unrolled` | S_CHECK·S_VAR·S_SAT (3) | 181 | 96.0 MHz | 1.885 µs | 93 562 (79.9%) | — |
| **SAT-overlap** | **`bp_relay_fast`** | **S_CHECK‖S_SAT · S_VAR (2)** | **122** | **95.9 MHz** | **1.272 µs** | **93 487 (79.8%)** | **✓ 1.48× win, free** |
| min-sum pipeline | `bp_relay_pipe` | S_CHK1·S_CHK2·S_VAR·S_SAT (4) | 241 | 111.3 MHz | 2.165 µs | 100 102 (85.5%) | ✗ 15% regression |

All three are **bit-identical to the golden** in Verilator (`make -C hw bpunroll|bpfast|bppipe` →
181 / 122 / 241 cycles, 65/65 each).

## SAT-overlap (`bp_relay_fast`) — the win: cut a cycle, keep the wall

M4/M5 spent a whole cycle on `S_SAT` (`H·ehat == s` + keep-lowest-weight-valid). But `S_SAT` reads only
`ehat` (from the just-finished `S_VAR`), while the *next* `S_CHECK` reads only `m_vc` — **independent
register→register clouds**. So `S_SAT` is folded to run **in parallel** with the next iteration's
`S_CHECK` (a trailing `S_SATF` handles the last iteration, whose `ehat` has no following `S_CHECK`).
That is **2 cyc/iter instead of 3** → 122 cycles at 6×10.

Crucially the per-cycle critical path is **unchanged**: the `S_SAT` parity XOR is far shallower than the
`S_CHECK` min-sum, so running them concurrently keeps Fmax at the min-sum bound (**95.9 vs 96.0 MHz**),
at the **same area** (79.8 vs 79.9% — folding a state removes control logic, adds none). Pure
cycle-count lever, **no Fmax gamble** — 1.885 → **1.272 µs, 1.48× faster, for free.** This is the
production unrolled decoder.

## Min-sum pipeline (`bp_relay_pipe`) — the measured negative result

Splitting `S_CHECK` into `S_CHK1` (the deep 6-way exclusive-minimum tournament, registered) + `S_CHK2`
(the shallow output shift) **did** raise Fmax — but only **96.0 → 111.3 MHz (1.16×)**, well under the
**1.33×** cycle penalty (+1 cyc/iter → 241 cyc). Net: **2.165 µs, a 15% regression.** And the extra
pipeline registers pushed util **79.9 → 85.5%**, *worsening* congestion.

This is the key finding: **the min-sum wall is routing/congestion at ~80% util, not logic depth.**
Pipelining shortens logic levels but the path is route-dominated (M5's 55%-routing diagnosis), so the
Fmax gain is small — and adding area to an already-congested part makes it worse, not better. The lever
that helps is the *opposite* of pipelining: remove work (SAT-overlap), don't add registers.

## Verdict → sub-µs needs a bigger fabric, not more microarchitecture

`bp_relay_fast` at **1.27 µs** is the practical floor for the 6×10 gross-code decoder on the KV260: the
schedule is LER-optimal (§ M5 step 1), the cycle count is at its 2-cyc/iter minimum for a
CHECK→VAR message-passing loop, and Fmax is pinned by min-sum *routing* at 80% util — which the pipeline
experiment showed cannot be bought with pipelining. Closing the last ~1.3× to sub-µs is now a **fabric**
problem: a larger / faster part (more routing resource → higher Fmax at this util, or the same util
spread thinner), or an ASIC — which is the North-Star target anyway. Latency ladder for the gross code:
M2 616 µs → M4 3.16 µs → M5 1.89 µs → **fast 1.27 µs**.

## Files

- `hw/bp_relay_fast.sv` — SAT-overlapped unrolled decoder (the 1.27 µs production variant).
- `hw/bp_relay_pipe.sv` — min-sum-pipelined variant (kept as the measured negative-result experiment).
- `hw/tb_bp_relay.cpp` (`-DFAST` / `-DPIPE`), `hw/Makefile` (`bpfast` / `bppipe`) — shared Verilator TB.
