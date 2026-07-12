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

-----

# Q7-02 M5-followup — the OSD-0 tail, measured: not worth the silicon (ships pure relay-BP)

**Status:** done — measured and rejected. Relay-BP on a degenerate qLDPC code occasionally leaves a
hard decision that does not satisfy `H ê = s`. The classic fix is an **OSD** tail (Fossorier–Lin: order
the variables by BP reliability, most-reliable-basis GF(2) Gauss–Jordan, solve the pivots). We built it
on the fixed-point golden — [`FixedRelayBpOsd`], a rare **slow-path escape** (the RTL emits `valid_flag`;
the PS runs OSD only on `!valid_flag` shots, since OSD's data-dependent Gauss–Jordan is exactly the
hardware-hostile work Q7-02 chose relay-BP to avoid) — and measured whether it earns its place.

## Result — OSD-0 does not cut LER; the win needs an order-12 sweep

`qec_q7_osd` sweeps the OSD combination-sweep **order** at code-capacity and circuit-level (gross code,
Q5.3 front-end, fixed **and** float relay-BP as reference).

**Code capacity** (20 000 shots): OSD-0 is **LER-neutral** at every `p` (within CI), tail-rate grows
0.15 % → 8.6 % over p = 0.03 → 0.06. Relay-BP's failures here are mostly *uncorrectable* (weight > d/2),
so a valid OSD decode is a ~coin-flip coset — no gain.

**Circuit-level** (depth-7 extraction, rounds = 6, 3 000 shots), fixed vs float, by OSD order:

| p | plain (fx / fl) | +OSD-0 | +OSD-4 | +OSD-12 |
|---|-----------------|--------|--------|---------|
| 0.002 | 4.0e-3 / 3.3e-3 | 6.7e-3 / 5.7e-3 **worse** | 4.0e-3 / 4.3e-3 ≈ | **1.7e-3 / 1.3e-3 wins** |
| 0.003 | 2.0e-2 / 1.9e-2 | 2.7e-2 / 2.8e-2 **worse** | 2.3e-2 / 2.6e-2 ≈ | **1.3e-2 / 1.5e-2 wins** |

Two decisive findings:

1. **OSD-0 (order 0) hurts** — in *both* float and fixed. Replacing BP's invalid guess with a valid but
   often-wrong-coset decode loses more shots than it rescues. The beneficial regime is **order ≈ 12**
   (`2^12` = 4096 flip patterns re-solved per shot), which reproduces the Q5-05 relay-BP+OSD win
   (`qec-q5-circuit-dem.md` used order 12). Order 4 is roughly break-even.
2. **Fixed Q5.3 tracks float at every order** (order-12: fixed 1.3e-2 vs float 1.5e-2). So the Q5.3
   hardware word is **not** the limiter for OSD — the *order* is. (This also re-validates Q5.3: even the
   OSD reliability ordering survives the quantisation.)

## Verdict → no OSD tail on the Q7-02 hardware

The only hardware-tractable OSD order (0) does not help, and the order that helps (12) is a 4096-way
reliability-ordered GF(2) Gauss–Jordan per failure shot — utterly impractical as an RTL datapath or even
a PS slow-path tail, and precisely the data-dependent variable-latency decoder Q7-02 rejected up front.
**So Q7-02 ships pure relay-BP; the OSD tail is not worth the silicon or the PS cycles.** This is the
data that *validates* the original architecture call (relay-BP over BP+OSD). `FixedRelayBpOsd` remains as
the measured evidence and a validity-guarantee option (its decode always satisfies `H ê = s`), which the
logical-observable readout use case does not need.

## Files

- `crates/aleph-qec/src/fixed_bp.rs` — `FixedRelayBp::decode_fixed_soft` (exposes the fixed-point
  posterior LLR) + `FixedRelayBpOsd` (the OSD-0 tail decoder + `decode_fixed_osd` returning the
  tail-ran flag).
- `crates/aleph-qec/examples/qec_q7_osd.rs` — the order/precision sweep (`capacity` | `circuit`).

-----

# Q7-02 M5-followup — circuit-level DEM + sim↔RTL co-sim: the decoder generalises past code capacity

**Status:** done. Everything above targets the **code-capacity** gross graph (144 vars / 72 checks,
regular degree 6 / 3) baked into `bb_gross_tanner.svh`. This closes the gap to **circuit-level** noise —
the depth-7 syndrome-extraction DEM (Q5-04, CNOT depolarising + idle/prep/meas), a much larger and
**irregular** hypergraph — on two fronts: software LER, and a bit-exact RTL co-sim.

## Software — relay-BP decodes circuit-level gross noise (~0.3% threshold)

`qec_q7_osd -- circuit` runs the fixed-point relay-BP over `BBCode::gross().circuit_level_dem(rounds, p)`
(depth-7 gadget, uniform noise). Plain relay-BP (Q5.3, 4×25) tracks the float decoder and clears a
per-cycle threshold of **~0.3 %** (LER falls with `p` below it): p=0.002 → 4e-3, p=0.003 → 2e-2,
p=0.004 → 6.5e-2 (rounds=6). So the *same* fixed-point decoder the RTL implements handles realistic
gate-noise syndromes, not just code-capacity independent-Z — the qLDPC-frontier claim the whole track is
about.

## RTL co-sim — the parametric M2 decoder is graph-generic (`make -C hw bpcirc`)

The emitter grew a `circgraph` / `circvectors` mode: it emits the **circuit-level** Tanner graph in the
same `.svh` format and **real DEM-shot** golden vectors (sampled from the circuit DEM, decoded by the
same `FixedRelayBp`). The `bpcirc` target cp's the circuit header over `bb_gross_tanner.svh` in a build
dir and decodes the vectors through the **M2 sequential** decoder (`bp_relay_decoder.sv`) — its runtime
node cursor handles any graph size, unlike the spatially-unrolled variants which would need the graph
baked in (and the circuit graph is far too large to unroll).

At **rounds = 1** the circuit graph is **864 vars / 144 checks / 2952 edges**, max check-degree **25**
(vs code capacity's uniform 6) — a genuinely irregular, 6× larger graph. Result:

```
make -C hw bpcirc
→ PASS: 40 full decodes bit-identical to the fixed-point golden
```

The parametric RTL decodes the circuit-level graph **bit-for-bit** against the software golden — the
sim↔RTL co-sim proving the decoder generalises past the baked-in code-capacity graph with **zero RTL
change** (only the generated header differs).

**One honest caveat (cosmetic):** the M2 decode over this graph runs ~70 000 cycles
(`6×10 · (144+864+144) + 864`), which overflows the shared **16-bit** `latency_cycles` port (wraps mod
65 536 → the TB reports 4448). Correctness is unaffected — the co-sim compares `ehat`/`obs`/`valid`, not
latency — and the *perf* decoders (unrolled, ~122–181 cycles) are nowhere near the limit; only the M2
correctness vehicle at circuit scale wraps. Widening the port would ripple through every bp decoder and
the AXI wrapper for a number the co-sim does not check, so it is left as-is and noted here.

## Files

- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — `circgraph` / `circvectors` modes (circuit-level
  graph + real-shot golden vectors); `build()` selects code-capacity vs circuit-level DEM.
- `hw/Makefile` (`bpcirc`) — the circuit-level co-sim: emit → cp header → M2 Verilator → bit-exact.

-----

# Q7-02 M5-followup — SAT-overlap on the Arty partial: 43.4 → 29.3 µs on real silicon

**Status:** done, on the board. The `bp_relay_fast` SAT-overlap lever (§ sub-µs levers, #443) — fold the
S_SAT parity check to run in parallel with the next iteration's S_CHECK, since one reads only `ehat` and
the other only `m_vc` — applies just as well to the **Arty-fit partial** decoder. The partial sweeps
`G_CHK` check-groups per S_CHECK and per S_SAT over the same `grp` cursor, so S_SAT folds group-for-group
into S_CHECK: **`G_CHK + G_VAR` cycles/iteration instead of `2·G_CHK + G_VAR`** (12/24 build: 12 not 18).

`bp_relay_partial_fast.sv`, `make -C hw bppartialfast` → **65/65 bit-exact, 732 cycles** (was 1086).

**On the real Arty Z7-20** (same `bp_axi_wrap` AXI4-Lite path, rebuilt bitstream at 25 MHz):

| decoder | cycles | LUT | Fmax (WNS) | on-silicon latency |
|---------|--------|-----|-----------|--------------------|
| `bp_relay_partial` (#442) | 1086 | 23 881 (44.9%) | +4.57 ns | 43.4 µs |
| **`bp_relay_partial_fast`** | **732** | **23 454 (44.1%)** | **+4.59 ns** | **29.3 µs** |

`65/65 decodes match golden; IDCODE ok; worst latency 732 clk = 29.3 µs @ 25 MHz`. **1.48× faster on the
board at *slightly less* area** (folding the S_SAT state removed control logic — SAT parity XOR is
shallower than the min-sum, so timing is unchanged), bit-exact on silicon. This is the SAT-overlap win
(measured 1.48× OOC on the KV260 unrolled decoder) landed on the hardware we actually have — no KV260
needed. `bp_relay_partial_fast` supersedes `bp_relay_partial` as the board decoder; the latter stays as
the M2↔M4 curve point.

## Files

- `hw/bp_relay_partial_fast.sv` — SAT-overlapped partial (the Arty board decoder).
- `hw/tb_bp_relay.cpp` (`-DPARTIALFAST`), `hw/Makefile` (`bppartialfast`); `hw/bp_axi_wrap.sv` +
  `hw/syn/arty_z7_bp_bd.tcl` now instantiate it.

-----

# Q7-02 M5-followup — circuit-level decode on the Arty: the M2 cursor-mux wall is fatal (needs BRAM)

**Status:** attempted; the wrapper/co-sim/driver are validated, but the **M2 decoder core does not fit the
xc7z020** at circuit scale. This records exactly why, and the path that would.

The plan: bring up the circuit-level relay-BP decode on the Arty via the **graph-generic M2 sequential
decoder** (the only variant that isn't baked to a fixed graph) behind a **size-generic AXI4-Lite wrapper**
(`bp_axi_wrap_wide`, deriving `NS = ⌈BP_C/32⌉` syndrome words / `NC = ⌈BP_N/32⌉` correction words from the
header — 5 / 27 for the rounds=1 circuit graph). Everything **off the fabric works**:

- `make -C hw bpaxiwide` → **40/40 circuit-level decodes bit-exact over the wide AXI4-Lite regmap**, and
- `bp_circ_pynq.py` (GoldenModel) → **40/40**, and the M2 latency counter, widened 16→32-bit, now reports
  the true **69 984** cycles (this also fixes the § circuit co-sim 16-bit overflow — `bpcirc` no longer
  wraps to 4448).

**But the M2 core will not synthesise for the xc7z020.** M2 reads its message arrays by a *runtime* node
cursor — `m_vc[BP_CHECK_EDGES[off+k]]` — which the M3 study already found to be the wall (a large select
feeding the min-sum, demuxed back across the edges). At code capacity that mux is over `BP_E = 432` edges
(M2 = 23 596 LUT, 44%). The circuit-level graph has **`BP_E = 2952`** edges (6.8×), and the mux blows up
super-linearly: an OOC synth on the 62 GB build box was **OOM-killed** —

```
oom-kill: task=vivado ... Out of memory: Killed process (vivado)
          total-vm:86 GB, anon-rss:63.8 GB
```

Vivado consumed **~64 GB** in RTL/timing optimization on the cursor mux and died before place. Even had it
completed, the mux extrapolates to **~90 k+ LUT ≫ the 53 200-LUT part**. So the runtime-cursor M2 path is a
dead end at circuit scale — not merely too big, but not synthesisable on the box.

## The path that would fit — BRAM-backed messages (an M2 redesign, not a wrapper change)

The cursor mux exists because M2 does *many scattered message reads per cycle* (all `deg` edges of a check
in one cycle). Mapping `m_vc`/`e_cv` to **block RAM** (the xc7z020 has 140 BRAM tiles, unused here) removes
the mux — but a single-port BRAM serves *one* access per cycle, so the check/var updates must be
restructured to an **edge-serial FSM** (one edge per cycle, ~`deg`× more cycles). That is a genuine core
redesign, deferred. Until then, circuit-level relay-BP is validated end-to-end **in simulation** (the wide
wrapper + co-sim + driver above), and runs on real silicon only at **code capacity** (§ board bring-up);
the circuit-scale core wants BRAM-backed messages or a larger fabric.

## Files

- `hw/bp_axi_wrap_wide.sv`, `hw/bp_axi_top_wide.v` — size-generic wide AXI4-Lite wrapper (validated 40/40).
- `hw/tb_bp_axi_wide.cpp` (`make -C hw bpaxiwide`), `hw/sw/bp_circ_pynq.py` — wide-regmap sim + driver.
- `hw/bp_relay_decoder.sv` — M2 latency widened to 32-bit (real circuit latency; fixes the co-sim wrap).
- `hw/syn/arty_z7_bp_circ_bd.tcl` — the (currently un-synthesisable) circuit board BD, kept for the future
  BRAM-backed core.

-----

# Q7-02 M2-BRAM — circuit-level relay-BP FITS the Arty: block-RAM messages + edge-serial updates

**Status:** done. The BRAM redesign the previous section called for is built (`bp_relay_bram`), verified
bit-for-bit against the golden, and **synthesises + closes timing on the xc7z020** — the first
circuit-level qLDPC decode to fit commodity Arty silicon. It resolves the § "cursor-mux wall is fatal"
OOM directly.

## Why the flop-array M2 couldn't fit (recap)

`bp_relay_decoder` holds the two per-edge message tables `m_vc`/`e_cv` (`BP_E` deep) in **flip-flops** and
reads them by a *runtime* node cursor — `m_vc[BP_CHECK_EDGES[off+k]]` — up to `BP_CHK_DEG = 25` times
*combinationally per cycle*. That is a 2952:1 × 8-bit register-file select replicated ~25× per port. At
circuit scale (`BP_E = 2952`) Vivado spent **~64 GB** optimising the mux and was **OOM-killed before place**;
the mux extrapolated to **~90 k LUT ≫ 53 200**.

## The fix — `bp_relay_bram`: map the messages to BRAM, serialise the updates

Two changes, both mechanical once framed right:

1. **`m_vc`/`e_cv` → block RAM** (one synchronous read port + one write port each). Vivado infers **2 BRAM
   tiles** and the register-file mux *disappears* — a BRAM is addressed, not demuxed.
2. **Edge-serial check/var updates.** A single-port BRAM serves one access per cycle, so the min-sum and
   memory-blend loops that touched all `deg` edges of a node in one cycle are unrolled *in time*: one edge
   per BRAM access, the node/leg/iter loops advancing across FSM states. The registered BRAM read gives a
   2-cycle address/data handshake on the read passes. Check pass-2 (`e_cv` write) and the SAT parity scan
   need no read latency (inputs are registered / `ehat` is a 1-bit flop array), so they stay 1 cycle/edge.

Everything else (syndrome, hard decisions `ehat`, best-so-far `best_e`, observable reduce) stays in flops —
1-bit or single-access, cheap. **The arithmetic is byte-for-byte the M2 golden**: same Q5.3 quantisation,
multiply-free α=7/8 check-update, single memory-blend multiply `(1−γ)·computed + γ·m_old`, truncating
arithmetic-shift rounding, keep-lowest-weight-valid rule.

## Correctness — bit-exact, unchanged golden

```
make -C hw bpbram      → PASS: 40 full decodes bit-identical to the fixed-point golden
make -C hw bpaxiwide   → PASS: 40 circuit-level decodes bit-identical to golden over wide AXI4-Lite
```

Both replay the same `bp_circ_vectors.txt` (real depth-7 circuit-level DEM shots) the flop-M2 was checked
against. `bp_relay_bram` is the exact silicon twin of the M2 reference — the redesign is a *microarchitecture*
change, not an algorithm change.

## It fits — OOC synth + P&R on the xc7z020 (`xc7z020clg400-1`)

Peak Vivado RSS **3.2 GB** (was 64 GB → OOM). Post-route utilisation:

| Resource       | Used | Avail  |     %  |
|----------------|-----:|-------:|-------:|
| Slice LUTs     | 6147 | 53 200 | 11.5 % |
| Slice Registers| 3235 |106 400 |  3.0 % |
| **Block RAM**  |  **2** |  140 |  1.4 % |
| DSP48E1        |    6 |    220 |  2.7 % |

From ~90 k-LUT-that-wouldn't-build to **6147 LUT (11.5 %) + 2 BRAM**. The message tables are the 2 BRAMs;
the DSP is the memory-blend multiply. Huge headroom remains on the part.

## Timing — closes at 50 MHz; Fmax 55.4 MHz

`Fmax = 55.4 MHz` (WNS −8.056 ns at a 100 MHz target). The board BD clocks the PL at **50 MHz**, **under**
Fmax, so the integrated PS+PL design **meets timing**. The critical path is the S_SAT *keep-best* decision:
the edge cursor `p` → `deg−1` compare → parity/`ehat` mux → `final_sat` fanning out to the **864 `best_e`
register clock-enables** (route-dominated, 24 logic levels, 69 % routing). Registering that decision one
cycle would break it — a cheap Fmax lever left for later, since this is a *reach* result (fit + correct),
not a latency result.

## On silicon — the first circuit-level qLDPC decode on the Arty Z7-20

The full board build (`arty_z7_bp_circ_bd.tcl`, PS7 + wide AXI4-Lite + `bp_relay_bram`, PL @ 50 MHz)
**closes timing** (`WNS +0.037 ns, TIMING_MET`) and produces a bitstream. Loaded onto the real Arty
(PYNQ, `hw/sw/bp_circ_pynq.py bp_arty_circ.bit bp_circ_vectors.txt`):

```
[board] IDCODE ok (0x42500002)
bp-circ driver: decodes=40 fails=0 worst latency=1489896 clk = 29797920 ns @ 50 MHz
RESULT: PASS (40/40 circuit-level decodes match golden; IDCODE ok)
```

**40/40 circuit-level decodes bit-identical to the golden on owned silicon.** The on-hardware cycle count
(1 489 896) equals the Verilator sim equals `FixedRelayBp` — silicon, sim, and reference agree exactly.
This is the first qLDPC **circuit-level** decode on the Arty; prior on-silicon relay-BP (#442/#446) ran only
at code capacity, and the flop-M2 circuit core wouldn't build at all (#447).

## Cost — latency, honestly

Edge-serial is O(`BP_E`) per pass, not O(`BP_C + BP_N`): worst-case **1 489 896 cycles/decode** (vs the flop
M2's 69 984). At 50 MHz that is **~29.8 ms/decode** — slow, but this is the *reach* deliverable: circuit-level
relay-BP that **fits and is correct on a $200 commodity board**, proving the graph-generic core works at
circuit scale on real silicon. Latency stays the KV260-fabric / ASIC story. A dual-port-BRAM or pipelined-read
follow-up (2 edges/cycle) would roughly halve the cycle count; deeper unrolling needs more BRAM banks.

## Pipelined-read follow-up — `bp_relay_bram_fast`, 1.39× on the same silicon

The BRAM core above spent **2 cycles/edge** in its three edge-serial *read* passes (`S_CHK1` min-sum scan,
`S_VAR1` e_cv accumulate, `S_VAR2` old-m_vc read): a `ph` sub-phase presented the address on `ph==0` and
consumed the registered read datum on `ph==1`. The read port can accept a new address every cycle, so this
is pure idle. `bp_relay_bram_fast` **pipelines** the reads — an address cursor `p` runs `0..deg`, presenting
the read for edge `p` (when `p<deg`) while the seq block consumes the registered read for edge `p-1` (when
`p>=1`). Each read pass drops from `2·deg` to `deg+1` cycles (one-cycle fill). `S_VAR2` is a pipelined
**read-modify-write**: it presents the read of `m_vc[lo+p]` while the comb block writes the blend for edge
`p-1`; adjacent edges are distinct addresses, so the BRAM read and write ports never collide (no RAW hazard).

Second lever: the **S_SAT keep-best commit is registered**. The slow core folded, in a single cycle at the
last check of each iteration, the parity scan + `ehat_w < best_w` compare into the 864-wide `best_e`
register-enable fan-out (its route-dominated critical path). The fast core latches that decision (`do_commit`)
in `S_SAT1` and runs the wide copy the next cycle in a dedicated `S_SAT2` state, so the 864 enables are driven
by a plain flop instead of the deep combinational chain. Costs +1 cycle/iteration (60 total — negligible).

Arithmetic is byte-for-byte the M2 golden — a microarchitecture change, not an algorithm change.

**Cycle count.** Per iteration `5·BP_E + 3·BP_C + 3·BP_N` (vs the slow core's `8·BP_E + 2·BP_C + BP_N`):
on the rounds=1 circuit graph 17 785 vs 24 768 cyc/iter → worst-case **1 070 916** cycles/decode (vs
1 489 896). Verilator `make -C hw bpbramfast` and `make -C hw bpaxiwide` (the wide wrapper now instantiates
the fast core) are both **40/40 bit-exact**.

**xc7z020 OOC** (`synth_bp.tcl`, `bp_relay_bram_fast`): **6127 LUT (11.5 %), 3230 FF, 2 BRAM, 6 DSP,
Fmax 57.9 MHz** (WNS −7.269 ns at 100 MHz) — slightly *smaller* than the slow core (6147 LUT; removing the
`ph` handshake logic offsets the extra state) at a slightly higher Fmax (55.4 → 57.9 MHz).

**On silicon** (rebuilt board bitstream, PL @ 50 MHz, **WNS +0.567 ns TIMING_MET** — comfortable margin vs the
slow core's +0.037 ns, the registered commit paying off in-context too):

```
[board] IDCODE ok (0x42500002)
bp-circ driver: decodes=40 fails=0 worst latency=1070916 clk = 21418320 ns @ 50 MHz
RESULT: PASS (40/40 circuit-level decodes match golden; IDCODE ok)
```

**40/40 bit-identical on the real Arty at 1 070 916 cycles = 21.42 ms/decode** — a **1.39× speed-up on owned
silicon** (29.8 → 21.42 ms), silicon == sim == `FixedRelayBp` exactly. Honest ceiling: the read passes are now
1 cyc/edge, so the remaining edge-serial cost is inherent to single-port BRAM; a further ~2× needs *dual-port*
(true 2 edges/cycle) or multiple BRAM banks, and real-time stays the KV260-fabric / ASIC story (M6).

## Dual-port follow-up — `bp_relay_bram_dp`, 2 edges/cycle everywhere, 2.22× on silicon

The pipelined core got each edge-serial pass to **1 cyc/edge**; the remaining cost is the single BRAM
read/write port. To go to **2 edges/cycle** every pass needs two message accesses per cycle. The catch is
the edge numbering is *variable-major*: the variable passes (INIT, VAR1, VAR2) touch **contiguous** edge
indices, but the check passes (`CHK1` reads m_vc, `CHK2` writes e_cv) touch **scattered** ones through
`BP_CHECK_EDGES` (the Tanner-graph transpose). A plain even/odd 2-bank split is conflict-free only for the
contiguous passes — two scattered edges can hit the same bank.

`bp_relay_bram_dp` makes each message table **two banks, each a TRUE dual-port BRAM** (bank = `edge&1`,
row = `edge>>1`). With two independent R/W ports per bank, even two *same-bank* scattered accesses use the
bank's two ports, so **every** pass runs 2/cycle:

- **INIT / CHK2** — 2 writes/cycle (slot0→portA, slot1→portB of each slot's bank).
- **CHK1 / VAR1** — 2 pipelined reads/cycle; 1-cycle read latency, consume the pair presented last cycle.
- **VAR2** — 2 pipelined read-modify-writes/cycle: portA reads the leading pair, portB writes the lagging
  pair's blend. Contiguous ⇒ the pair straddles both banks, so per bank portA (read) + portB (write) never
  collide and hit distinct rows.
- **SAT1** — 2 ehat (flop) reads/cycle folded into the running parity.

The 2-wide min-sum folds both magnitudes into the running top-2 **in edge order**, so argmin ties match the
sequential golden exactly. Arithmetic is byte-for-byte the M2 golden.

> **TDP inference gotcha.** Vivado (`Synth 8-4767`) refuses BRAM if a memory is written from **multiple
> ports in one process** — the first attempt put all 8 port-blocks in one `always_ff` and every bank
> *dissolved into registers*. Fix: **one `always_ff` per port** (two processes writing the same array →
> one true dual-port BRAM). After the split: 4× RAMB18E1, each `1K×8 (READ_FIRST)`, PORT A and B both W+R.

**Cycle count.** Halving every pass → worst-case **672 000** cycles/decode. Verilator `make -C hw bpbramdp`
and `make -C hw bpaxiwide` (wide wrapper now on the dp core) are **40/40 bit-exact**.

**xc7z020 OOC**: **8328 LUT (15.7 %), 3224 FF, 4 RAMB18 (2 tiles), 11 DSP, Fmax 54.3 MHz** — LUT roughly
doubles the fast core (the 2-wide datapath + port muxing) but still a small fraction of the part; the two
per-cycle blends need 2 DSP multiplies (6→11 DSP).

**On silicon** (board bitstream, PL @ 50 MHz, **WNS +0.113 ns TIMING_MET**):

```
[board] IDCODE ok (0x42500002)
bp-circ driver: decodes=40 fails=0 worst latency=672000 clk = 13440000 ns @ 50 MHz
RESULT: PASS (40/40 circuit-level decodes match golden; IDCODE ok)
```

**40/40 bit-identical on the real Arty at 672 000 cycles = 13.44 ms/decode** — **2.22× the original
edge-serial core** (29.8 → 13.44 ms) and 1.59× the pipelined one, silicon == sim == `FixedRelayBp`. This is
the practical floor for the Arty: 2 edges/cycle is the max a 2-bank dual-port layout gives without a
graph-dependent scattered-access scheme; more parallelism needs the KV260 fabric (which fits the unrolled
core at circuit scale for real-time µs decoding — M6) or an ASIC.

Ladder on silicon: **29.8 ms (#448) → 21.42 ms (#449) → 13.44 ms (dp)**.

## Early termination — average-case latency, ~18× on silicon, zero LER cost

Everything above optimises the **worst case**: the fixed `LEGS×ITERS` (6×10 = 60) schedule runs to the end,
so worst == every case. Standard BP instead stops the moment the hard decision satisfies the syndrome. That
**changes the result** (the first valid `ê` rather than the lowest-weight valid one over the whole schedule),
so it is an *algorithm* change, verified end-to-end against a matching golden — not a microarchitecture one.

**Does it hurt LER?** No, measurably. `qec_q7_early` (circuit-level rounds=1, 40 000 shots) compares the two:

| p | LER full | LER early | within CI | converged | iters mean / p50 / p99 / max |
|------|-----------|-----------|-----------|-----------|------------------------------|
| 0.001 | 2.5e-5 | 2.5e-5 | ✓ | 100.0 % | 1.45 / 1 / 9 / 60 |
| 0.002 | 1.75e-4 | 1.75e-4 | ✓ | 100.0 % | 2.26 / 1 / 14 / 60 |
| 0.003 | 7.0e-4 | 7.0e-4 | ✓ | 99.9 % | 3.28 / 2 / 20 / 60 |
| 0.005 | 6.98e-3 | 6.98e-3 | ✓ | 99.2 % | 5.97 / 4 / 48 / 60 |

The LER is identical within Monte-Carlo CI at every p — the extra relay legs almost never change the
predicted observable — while the decode converges in a *handful* of iterations on average (3.3 of 60 at
p=0.003), not the full schedule.

**RTL.** `bp_relay_bram_dp` gains an `early_exit` input; `S_SAT2` jumps to `S_EMIT` the moment an iteration's
decision satisfies the syndrome (`found`). The wide wrapper exposes it as **CTRL bit1 (sticky)**, so the
*same bitstream* runs either mode at runtime. Verilator `make -C hw bpbramdpearly` is 40/40 bit-exact against
the early-exit golden (`circvectorsearly`).

**On silicon** (same board bitstream, PL @ 50 MHz, both modes, the 40 circuit shots at p=0.003):

```
[full-schedule]  min=p50=mean=p99=max = 672 000 cyc = 13.440 ms   (fixed)
[early-exit]     min=13 501  p50=24 662  mean=38 055  p99=max=180 916 cyc
                 min=0.270  p50=0.493  mean=0.761  p99=max=3.618 ms
```

**Average 0.761 ms vs 13.44 ms = 17.7× on real silicon** (median 27×), 40/40 still bit-exact, LER unchanged.
Worst-case is **not** improved (a non-converging shot still runs the full 60 iters → 13.44 ms), so this is an
average-throughput / energy lever, not a hard-deadline one — the CTRL bit lets you pick per workload.

Silicon ladder (worst-case): **29.8 ms (#448) → 21.42 ms (#449) → 13.44 ms (dp)**; early-exit adds an
**average** 0.76 ms path on the same dp bitstream.

## Files

- `hw/bp_relay_bram.sv` — the original BRAM edge-serial core (1 cyc/edge via a `ph` sub-phase; curve point).
  `hw/bp_relay_bram_fast.sv` — pipelined-read variant (1 cyc/edge, no `ph`; 1.39×; curve point).
  `hw/bp_relay_bram_dp.sv` — **dual-port 2-edges/cycle variant (2.22×) with a runtime `early_exit` input;
  the current board decoder.**
- `hw/tb_bp_relay.cpp` (`-DBRAM` / `-DBRAMFAST` / `-DBRAMDP`; optional `early` arg), `hw/Makefile`
  (`make -C hw bpbram` / `bpbramfast` / `bpbramdp` / `bpbramdpearly`) — bit-exact verification of each core
  (and both dp modes) vs the same golden.
- `crates/aleph-qec/examples/qec_q7_early.rs` — full-vs-early LER + iteration-distribution study.
- `hw/bp_axi_wrap_wide.sv` — wide AXI4-Lite wrapper instantiates `bp_relay_bram_dp`; **CTRL bit1 = sticky
  early-exit enable** (`make -C hw bpaxiwide`).
- `hw/sw/bp_circ_pynq.py` — board driver; `early` arg sets CTRL bit1 and reports the latency distribution.
- `hw/syn/arty_z7_bp_circ_bd.tcl`, `hw/syn/arty_z7.xdc` — circuit board BD (now on the dp core) + Arty OOC clock.

-----

# Q7-02 M6 — KV260 (Zynq UltraScale+): circuit-level relay-BP on the bigger fabric

The KV260 (XCK26, `xck26-sfvc784-2LV-c`) has ~117k 6-input LUTs vs the Arty's ~53k. The M6 hypothesis
(from the pre-KV260 handoff) was that the bigger fabric would fit an **unrolled** circuit-scale core and
turn the Arty's 13.44 ms edge-serial *reach* result into a *real-time* (~1–3 µs) *latency* result.

**That hypothesis is false, and the OOC sweep is why.** The circuit-level gross-code graph is
`BP_N=864 / BP_C=144 / BP_E=2952`, irregular, **max check-degree 25**. A spatially-unrolled min-sum
evaluates every check's 25-input min₁/min₂ reduction combinationally every cycle — a colossal comparator
network that Vivado's synthesis cannot handle at this scale:

| core | result @ xck26 | verdict |
|---|---|---|
| `bp_relay_fast` (full unroll, ~123 cyc) | ~43 GB RSS, **stuck in Cross-Boundary-Area-Optimization ~1 h, killed** | infeasible (OOM) |
| `bp_relay_partial_fast` (CHK12/VAR24, ~2.9k cyc) | ~20 GB RSS, **stalled the same area-opt phase >40 min, no progress** | infeasible (non-convergent) |
| `bp_relay_bram_fast` (edge-serial) | LUT 6466 (**5.5%**), FF 3217, BRAM 2, DSP 6, Fmax 122.5 MHz | fits |
| `bp_relay_bram_dp` (2 edges/cyc)  | LUT 8509 (**7.3%**), FF 3216, BRAM 2, DSP 11, Fmax 121.3 MHz | fits |

The cores that **fit** are BRAM edge-serial (fabric-light); the cores that would **exploit** the fabric
don't synthesize. **On this decoder the KV260's larger fabric buys clock, not unrolling.** The wall is
synthesis of the deg-25 min-sum, not placement/area — it is the same class of wall the flop-array M2 hit
on the Arty (#447), just triggered in the optimizer instead of at elaboration.

**Chosen core: `bp_relay_bram_dp`** — fewest cycles of the fitting family (672 000 cyc worst-case, matching
the Arty silicon), and already the core instantiated in `bp_axi_wrap_wide`, so **zero RTL change**.

**Board build** (`hw/syn/kv260_bp_circ_bd.tcl`): `zynq_ultra_ps_e` (KV260) → `M_AXI_HPM0_FPD` AXI4-Lite →
`bp_axi_top_wide` (IDCODE `0x4250_0002`) at base `0xA000_0000`, `pl_clk0` FCLK **100 MHz**. Implementation
**closes timing** (WNS **+0.310 ns**), fit 7.3% LUT.

**On silicon** (KV260, PL @ 100 MHz, the 40 circuit shots at p=0.003, IDCODE probed `0x4250_0002`,
**40/40 bit-exact to the fixed-point golden in both modes**):

```
[full-schedule]  min=p50=mean=p99=max = 672 000 cyc = 6.720 ms   (fixed)
[early-exit]     min=13 501  p50=24 662  mean=38 055  p99=max=180 916 cyc
                 min=0.135  p50=0.247  mean=0.381  p99=max=1.809 ms
```

vs the Arty (dp core @ 50 MHz): identical cycle counts, **half the wall-clock** — worst-case
**13.44 → 6.72 ms = 2.0×**, early-exit average **0.761 → 0.381 ms = 2.0×** (17.6× the average lever, as on
the Arty). The 2× is purely the clock (100 vs 50 MHz); the OOC Fmax (~121 MHz) leaves ~1.2× more clock on
the table for a future tightened build, but not the orders of magnitude that only unrolling would give.

**Silicon ladder (worst-case):** Arty 29.8 ms (#448) → 21.42 ms (#449) → 13.44 ms (dp) → **KV260 6.72 ms (dp)**.

**PYNQ gotcha (KV260 / Kria-PYNQ 3.0.1).** `pynq.Overlay(bit)` fails on a design with **no PL DRAM banks**:
the image ships a *stub* `/usr/bin/xclbinutil` that wraps `unwrapped/xclbinutil` and `exit 0`s regardless of
its return code, so the empty `MEM_TOPOLOGY` makes the real tool fail, the failure is masked, `t.xclbin` is
never written, and `Overlay` dies with `FileNotFoundError: .../t.xclbin`. **Bypass:** program the PL with
`pynq.Bitstream(bit).download()` (skips the metadata/xclbin path) and talk to the IP via
`pynq.MMIO(0xA000_0000, 0x1000)` directly — no Overlay. See `hw/sw/bp_circ_kv260.py`.

**Honest scope.** M6 did **not** reach µs real-time, and OOC proved it is unreachable with the current RTL:
it needs an unrolled datapath, and the deg-25 min-sum does not synthesize unrolled at circuit scale. The
route to µs is a **synthesis-friendly restructure of the min-sum** (pipelined/registered comparator
reduction so no single cycle carries the whole deg-25 tree, or a different parallel-but-registered
architecture), not a bigger fabric. What M6 delivers is the **first UltraScale+ bring-up of the decoder**,
correct on silicon, at 2× the Arty clock plus the early-exit average-case lever.

## Files (M6)

- `hw/syn/kv260_bp_circ_bd.tcl` — KV260 block design + bitstream (`zynq_ultra_ps_e` + wide wrap on
  `bp_relay_bram_dp`; `bdonly` 4th arg does a fast BD-assembly pre-flight). Usage in the header.
- `hw/sw/bp_circ_kv260.py` — KV260 runner: `Bitstream.download()` + raw `MMIO` (Overlay-bypass, see gotcha),
  reuses `bp_circ_pynq.BpCircDecoder` / `load_vectors` / `run_check`; runs full-schedule + early-exit.
- Cores unchanged (`bp_relay_bram_dp.sv`, `bp_axi_wrap_wide.sv`, `bp_axi_top_wide.v`); no RTL edits for M6.

# Q7-02 M7 — µs-class banked relay-BP on the KV260: 6.72 ms → 32.8 µs worst-case (2.5 µs early-exit avg)

**Goal.** M6 ended at 6.72 ms with the edge-serial `bp_relay_bram_dp` (2 edges/cycle). M7 asked for µs-class:
process ~all 2952 edges per BP phase instead of 2. Target restated after the fit-gate (below): ~20–40 µs
worst-case on this part.

## The design space is a trilemma (all three corners measured)

| approach | fit | synthesizes? | verdict |
|---|---|---|---|
| full spatial unroll (`bp_unroll_skeleton`, 144+864 stamped submodules) | **453 k LUT = 386 %** (CARRY8 276 %) | yes, ~3 min once modular | needs a ~4× bigger part |
| partial unroll (`bp_relay_unroll_pipe`, runtime-`grp` gather from flop arrays) | ~est. fits | **area-opt stalls** (O(BP_E) operand muxes; `RuntimeOptimized` doesn't help) | dead — the M2 cursor-mux wall again |
| **banked LUTRAM store (this milestone)** | 88 k LUT = 75 % | yes (~30 min) | **shipped** |

## The banked design (`bp_relay_banked`, spec amendment 2)

Scale `bram_dp`'s 2 banks to hundreds of tiny LUTRAM banks so W checks + V vars update per cycle
(M7 pick: **W=12, V=36** → GC=12 check groups, GV=24 var groups):

- **Check-major banking with a β split**: edge `e` lives at half-bank `(slot(check(e))·25 + pos(e))·2 + β(e)`,
  row = its check's group. CHK phases are then *hardwired* (broadcast row, 2:1 β-muxes); the only scattered
  access is the VAR phase, and an **offline solve in the emitter** (slot assignment + cap-2 var grouping +
  β assignment, deterministic, exactly verified on every regen) guarantees ≤1 write per m_cm half-bank and
  ≤2 reads per e_cm bank per cycle. König edge-coloring is the guaranteed-existence fallback if the greedy
  ever fails on a future graph; on this graph greedy succeeds at 8/24, 12/36, 16/48.
- **Three stores**: `m_cm` (v→c, check-major, β-split), `e_cm` (c→v, check-major, 2 read ports), `m_vm`
  (v→c shadow, var-major — makes the VAR read of the "old" message mux-free). All distributed RAM;
  **0 BRAM tiles**; the message fabric costs ~7 k LUT.
- **Bit-exact by construction**: banking never touches the logical in-check edge order (min-sum tie-breaks)
  or the wrap-add order — the SAME `FixedRelayBp` golden and vectors as every core since M2. 40/40 at all
  three (W,V) on Verilator (Mac + EPYC), through the AXI wrapper, and on silicon.

## What it took to synthesize (the whale hunt — each step measured OOC at 5 ns)

1. **Flat top** (functionally perfect): area-opt churns **~8 h**, converges to a garbage netlist —
   386 k LUT, DSP 747, Fmax 24 MHz.
2. **Memory cells modularized** (`bp_mcm/ecm/mvm_cell`, one per bank): same numbers — the fabric was
   never the whale.
3. **Literal resolution tables** (emitter-baked `BP_CHK_GRP/SLOT`, `BP_VAR_GRP/SLOT`, `BP_EDGE_HB/EB/ROW/EPORT`,
   replacing scan-loop helper functions Vivado can't constant-fold): faster synth, same size.
4. **`BP_GAMMA[leg·BP_N+v]` was the whale**: `leg` is a runtime 32-bit int, so Vivado built full 32-bit
   index arithmetic (a DSP multiply) + a 5184:1 ROM mux at each of ~864 gather sites — Verilator
   range-folds this, Vivado does not. Constant-folding over `leg` (6:1 mux): **386 k → 71 k LUT**.
5. **Serial reductions capped Fmax at 26 MHz** (worst path: S_EMIT obs fold, 137 logic levels; then
   `check_minsum`'s 25-deep serial min chain). Tree restructuring — a tournament (min1,min2,argmin) tree
   with first-occurrence tie-breaks, XOR/add trees for the folds — all provably bit-exact: **26 → 93 MHz**.

Every step re-proved bit-exact (40/40, cycle counts unchanged) before the next synthesis. Lint fences
Verilator↔Vivado disagreements (`ifndef SYNTHESIS` around the elaboration guards).

## OOC probe sweep (tree core, xck26-sfvc784-2LV-c, 5 ns)

| (W,V) | LUT | DSP | Fmax | cyc/decode | latency @Fmax |
|---|---|---|---|---|---|
| 8/24 | 68.8 k (59 %) | 738 (59 %) | 92.8 MHz | 3 570 | 38.5 µs |
| **12/36** | **88.0 k (75 %)** | **1 116 (89 %)** | **91.1 MHz** | **2 460** | **27.0 µs** |
| 16/48 | 105.3 k (90 %) | **1 476 (118 %)** | 93.8 MHz | 1 905 | **no fit** (DSP) |

DSP scales ~31/var (the `var_update` blend multiplies) and kills 16/48; 12/36 is the largest fitting config.

> **Correction (M8).** The DSP column above is **9× inflated**: the OOC tcl counted
> `REF_NAME =~ DSP*`, which also matches the ~9 sub-primitives every DSP48E2 macro expands into
> (DSP_ALU, DSP_A_B_DATA, …). Real usage: 8/24 → **82 (6.6 %)**, 12/36 → **124 (10 %)**,
> 16/48 → **164 (13 %)**. The "no fit (DSP)" verdict for 16/48 — and the "DSP … kills 16/48"
> sentence — were wrong: 16/48 was only ever **LUT-bound (90 %)** and is rehabilitated (and shipped)
> in M8 below. The M7 silicon result itself is unaffected.

## Board + silicon

Board build (`kv260_bp_circ_banked_bd.tcl`, `bp_axi_wrap_banked`, IDCODE `0x4250_0003`,
`FLATTEN_HIERARCHY none` on synth_1 — load-bearing): @90 MHz WNS −1.53 ns (post-route eats the OOC margin);
**@75 MHz (exact PS grid 1500/20) TIMING_MET, WNS +0.224 ns**.

**On silicon** (KV260, PL @ 75 MHz, 40 circuit shots p=0.003, IDCODE `0x4250_0003`,
**40/40 bit-exact in both modes**):

```
[full-schedule]  min=p50=mean=p99=max = 2 460 cyc = 32.8 µs   (fixed)
[early-exit]     min=100  p50=140  mean=188  p99=700  max=700 cyc
                 min=1.3  p50=1.9  mean=2.5  p99=9.3  max=9.3 µs
```

**Silicon ladder (worst-case):** Arty 29.8 ms → 13.44 ms (dp) → KV260 6.72 ms (M6) → **32.8 µs (M7) = 205×
M6**. Early-exit average 381 µs → **2.5 µs = 152×**; the average case now sits in the original 1–3 µs band.

## Honest scope & levers left on the table

- True 1–3 µs **worst-case** stays out of reach on this part (needs the full unroll = a ~4× bigger FPGA).
- Clock: the worst path (bank read → gather → `check_minsum` stage-1) allows ~91 MHz OOC but post-route
  closed only at 75 MHz with the default strategy. Registering the bank outputs (+~2 cyc/phase, ~+5 %
  cycles) should push past 120 MHz → ~20 µs; a Performance_Explore impl run may also recover 90 MHz as-is.
- DSP pressure (89 %) is the wall to 16/48's 1 905 cycles (~25 µs @75): forcing the blend multiplies to
  LUT fabric (`use_dsp = "no"`) would trade ~10–15 k LUT for the 360 extra DSPs — untested.
- The offline solve generalizes (parameterized (W,V), loud failure + a proven fallback), so a future
  bigger part lifts straight to 16/48 or beyond.

# Q7-02 M8 — squeeze the KV260: 32.8 → 15.6 µs worst-case, sub-µs median early-exit

**Goal.** M7 shipped 32.8 µs at 75 MHz with one long unregistered path (bank read → gather →
min-sum stage-1, ~12.6 ns routed) and a mistakenly-condemned 16/48 config (see the M7 correction
note above). M8 pipelines the path and ships 16/48.

## Levers

1. **L1 — bank-output register plane.** The gather OUTPUTS (not the raw banks) latch one cycle:
   group-g data stays paired with group-g context, and the gather mux leaves the min-sum's input
   path. Submodule `en` delayed 1 cycle; addresses stay combinational.
2. **L3 — 3-stage `check_minsum`.** One register plane after tournament-tree level 3, behind a
   `STAGES` parameter (default 2 — `bp_relay_unroll_pipe`/`bp_unroll_skeleton` byte-unchanged,
   regression-gated). Values provably unchanged; an elaboration guard rejects unsupported
   STAGES/degenerate small-DEG splits.
3. **L4 — 16/48.** With the real DSP numbers it was always viable; with L1+L3 it times *better*
   than 12/36 (shallower per-group muxes at GC=9/GV=18).
4. **L2 — impl-strategy tclarg** on the board flow (used in the ladder below).

New timing contract: CHK scatter lag pc−4 (phase GC+4 cyc), VAR pc−3 (GV+3) → +3 cyc/iteration.
Bit-exact in values to the same golden at all three configs (Mac + EPYC co-sim, AXI gate,
`checkminsum` at both STAGES, `bpunrollpipe`/`bpbramdp` regressions): **8/24 → 3 750 cyc,
12/36 → 2 640, 16/48 → 2 085** (M7: 3 570 / 2 460 / 1 905).

## OOC probes (5 ns, corrected DSP counter)

| (W,V) | LUT | DSP (real) | WNS | Fmax |
|---|---|---|---|---|
| 12/36 | 85.5 k (73 %) | 124 (10 %) | −3.02 | 124.7 MHz |
| **16/48** | **102.8 k (88 %)** | **164 (13 %)** | **−0.63** | **177.7 MHz** |

## Board ladder + silicon

16/48: 150 MHz default → WNS −0.238; 150 `Performance_Explore` → −0.147; 136-request (PS grants
**133.332 MHz**, 1500-grid) default → **TIMING_MET, WNS 0.000**. 12/36 fallback unneeded.

**On silicon** (KV260, PL @ 133.332 MHz, 40 circuit shots p=0.003, IDCODE `0x4250_0003`,
**40/40 bit-exact in both modes**):

```
[full-schedule]  min=p50=mean=p99=max = 2 085 cyc = 15.64 µs   (fixed)
[early-exit]     min=79  p50=113  mean=153  p99=589  max=589 cyc
                 min=0.59  p50=0.85  mean=1.15  p99=4.4  max=4.4 µs
```

**Silicon ladder (worst-case):** 6.72 ms (M6) → 32.8 µs (M7) → **15.64 µs (M8) = 2.1× over M7,
430× over M6**. Early-exit average 2.5 → **1.15 µs**; the **median decode is now sub-microsecond
(0.85 µs)** — the original 1–3 µs target band is met by the average case with room to spare.

## Levers left on the table

- The closed path at 133 MHz still has the var_update stage-2 blend (DSP mult + clamp) unregistered
  behind its gather; a 3-stage var_update would chase ~150+ MHz (~13.9 µs) for +60 cycles.
- LUT 88 % is the ceiling for wider configs on this part; a bigger part lifts the same RTL to
  W=24+ (full-unroll territory ~1–3 µs worst-case).

-----

# M9a — sliding-window streaming golden + (W, C) sweep (Q7-04, PR #460)

The banked core decodes one rounds=1 batch per START; real-time QEC is a continuous round stream.
M9 builds the BB-code analog of the surface-code UF streaming decoder (Q6-20/22) in three stages
(design spec `docs/superpowers/specs/2026-07-11-q7-04-streaming-relay-bp-design.md`): **M9a**
(this section) = software golden + the (W, C, seam) decision; **M9b** = emitter + streaming RTL +
bit-exact co-sim; **M9c** = KV260 silicon + sustained round rate.

## What shipped

`SlidingWindowBp` (`crates/aleph-qec/src/relay_window.rs`) — residual-carry sliding window over
multi-round circuit-level BB DEMs, per-window base decoder = the frozen `FixedRelayBp` operating
point (Q5.3, 6×10 schedule, γ ∈ [−0.3, 0.9], seed `0x5E1A_4B9C`). Two BP-specific deltas vs the
UF streaming pattern:

1. **Hypergraph time cut by truncation.** A mechanism straddling the window edge keeps its
   probability and observables but loses its out-of-window detectors (open temporal boundary —
   BP decodes hypergraphs directly, so no temporal-sink nodes are needed).
2. **Commit on error-vars.** A fired variable with an in-window detector at round < commit
   boundary XORs its observable mask into the running logical and toggles its in-window detectors
   in the residual; buffer rounds (W−C) re-decode next window. Non-convergence =
   report-and-flag: the best-kept decision commits anyway and `StreamStats.nonconverged` counts
   it (feeds Q7-07).

**Pins (all green):** one-window case (W = stream) **bit-exact to the batch `FixedRelayBp`**
(50/50 shots); interior windows compile to the **identical local DEM** (strict `assert_eq!` — the
translation invariance that lets M9b bake ONE window-graph header); all-windows-converged streams
drain the residual to zero; SoftPriors ≡ ResidualOnly on a single window; decode is a pure
function.

## The sweep (EPYC, gross [[144,12,12]], rounds=12, 100 000 shots/point, seed 2024)

Same-shots discipline: batch and every windowed config decode the identical `sample_shots`
stream, so differences are windowing cost, not sampling noise. `nonconv` = fraction of shots with
≥1 non-converged window; `resid` = fraction with a non-empty final residual.

| p | W | C | seam | LER win ±CI | LER batch ±CI | within CI | nonconv | resid |
|---|---|---|------|------------|---------------|-----------|---------|-------|
| 0.001 | 3 | 1 | residual | 4.58e-3 ± 4.2e-4 | 8.20e-4 ± 1.8e-4 | ✗ | 26.2 % | 2.1 % |
| 0.001 | 3 | 1 | soft     | 3.15e-3 ± 3.5e-4 | 8.20e-4 ± 1.8e-4 | ✗ | 21.7 % | 0.7 % |
| 0.001 | 4 | 2 | residual | 2.29e-3 ± 3.0e-4 | 8.20e-4 ± 1.8e-4 | ✗ | 14.3 % | 0.5 % |
| 0.001 | 4 | 2 | soft     | 2.46e-3 ± 3.1e-4 | 8.20e-4 ± 1.8e-4 | ✗ | 14.0 % | 0.6 % |
| 0.001 | 6 | 2 | residual | **1.11e-3 ± 2.1e-4** | 8.20e-4 ± 1.8e-4 | **✓** | 11.9 % | 0.2 % |
| 0.001 | 6 | 2 | soft     | 1.10e-3 ± 2.1e-4 | 8.20e-4 ± 1.8e-4 | ✓ | 11.6 % | 0.2 % |
| 0.001 | 6 | 3 | residual | 1.18e-3 ± 2.1e-4 | 8.20e-4 ± 1.8e-4 | ✓ | 9.2 % | 0.2 % |
| 0.001 | 6 | 3 | soft     | 1.37e-3 ± 2.3e-4 | 8.20e-4 ± 1.8e-4 | ✗ | 9.0 % | 0.3 % |
| 0.003 | 3 | 1 | residual | 1.51e-1 ± 2.2e-3 | 4.43e-2 ± 1.3e-3 | ✗ | 91.1 % | 30.3 % |
| 0.003 | 3 | 1 | soft     | 1.55e-1 ± 2.2e-3 | 4.43e-2 ± 1.3e-3 | ✗ | 87.1 % | 23.7 % |
| 0.003 | 4 | 2 | residual | 9.92e-2 ± 1.9e-3 | 4.43e-2 ± 1.3e-3 | ✗ | 73.7 % | 16.7 % |
| 0.003 | 4 | 2 | soft     | 1.16e-1 ± 2.0e-3 | 4.43e-2 ± 1.3e-3 | ✗ | 72.5 % | 18.6 % |
| 0.003 | 6 | 2 | residual | **5.47e-2 ± 1.4e-3** | 4.43e-2 ± 1.3e-3 | ✗ (1.24×) | 66.9 % | 8.2 % |
| 0.003 | 6 | 2 | soft     | 7.77e-2 ± 1.7e-3 | 4.43e-2 ± 1.3e-3 | ✗ | 65.3 % | 11.6 % |
| 0.003 | 6 | 3 | residual | 5.93e-2 ± 1.5e-3 | 4.43e-2 ± 1.3e-3 | ✗ (1.34×) | 57.8 % | 9.3 % |
| 0.003 | 6 | 3 | soft     | 7.96e-2 ± 1.7e-3 | 4.43e-2 ± 1.3e-3 | ✗ | 56.8 % | 12.2 % |
| 0.005 | 3 | 1 | residual | 6.26e-1 ± 3.0e-3 | 3.51e-1 ± 3.0e-3 | ✗ | 99.8 % | 79.6 % |
| 0.005 | 3 | 1 | soft     | 6.59e-1 ± 2.9e-3 | 3.51e-1 ± 3.0e-3 | ✗ | 99.6 % | 76.8 % |
| 0.005 | 4 | 2 | residual | 5.29e-1 ± 3.1e-3 | 3.51e-1 ± 3.0e-3 | ✗ | 97.6 % | 66.2 % |
| 0.005 | 4 | 2 | soft     | 5.76e-1 ± 3.1e-3 | 3.51e-1 ± 3.0e-3 | ✗ | 97.3 % | 69.9 % |
| 0.005 | 6 | 2 | residual | 3.89e-1 ± 3.0e-3 | 3.51e-1 ± 3.0e-3 | ✗ (1.11×) | 95.7 % | 48.1 % |
| 0.005 | 6 | 2 | soft     | 4.77e-1 ± 3.1e-3 | 3.51e-1 ± 3.0e-3 | ✗ | 95.2 % | 57.0 % |
| 0.005 | 6 | 3 | residual | 4.11e-1 ± 3.1e-3 | 3.51e-1 ± 3.0e-3 | ✗ | 92.4 % | 51.2 % |
| 0.005 | 6 | 3 | soft     | 4.74e-1 ± 3.1e-3 | 3.51e-1 ± 3.0e-3 | ✗ | 91.8 % | 58.0 % |

## Verdict — (W=6, C=2, residual-only) ships to RTL

- **Seam: residual-only.** Soft priors never win at a viable window: at W=6, p=0.003 they are
  **42 % worse** (7.77e-2 vs 5.47e-2); they help only the too-short W=3 window (partially
  compensating a buffer that is simply too small). The extra carried state (posterior LLRs per
  buffer var) buys nothing — the disorder restart from DEM priors each window is at least as
  good. This is the UF-streaming discipline exactly: the seam carries only the binary residual.
- **W=6 is the floor.** W=3/W=4 pay 2.8–5.6× at p=0.001 and 2.2–3.4× at p=0.003 — a buffer of
  W−C < 4 rounds cannot protect seam commits on this code.
- **C=2 over C=3 on LER**: at p=0.003 the gap (5.47 vs 5.93e-2) exceeds the summed CIs; at
  p=0.001 they tie. (6,3) remains the throughput-optimized alternate: −33 % window invocations
  per round and a 3-round commit budget, for +8 % LER at threshold.
- **The honest cost of streaming**: sub-threshold (p=0.001, the operating regime) the chosen
  config overlaps batch under the sweep's unpaired CI test (1.11e-3 vs 8.2e-4) — but the shots
  are paired (same stream), and a paired read (82 vs 111 errors on the same 100 k shots,
  McNemar p ≈ 1e-7) resolves a small **~1.35× sub-threshold cost**: the same accepted-gap
  category as threshold, where the penalty is **1.24× batch** (the pilot's 2 k-shot "within CI"
  there was lack of statistics, not absence of cost). The M9c re-sweep should upgrade the
  within-CI flag to a paired test. At p=0.005 (deep supra-threshold) everything is broken
  including batch — recorded for completeness only.
- **Non-convergence input for Q7-07**: at (6,2), 12 % / 67 % / 96 % of shots see ≥1
  non-converged window at p = 1/3/5 × 10⁻³ — window BP on the truncated graph converges notably
  less often than batch on the full graph; the resid column shows most of those still commit a
  syndrome-consistent stream (resid ≪ nonconv).

**AC-1 met:** multi-round (rounds = 12 ≥ 3) circuit-level windows generated and decoded; the
golden matches batch bit-exactly on the one-window anchor and within the documented band at the
chosen config.

## M9b fit pre-probe (parallel de-risk, openwebgui OOC @5 ns)

The interior W=6 window graph is **432 checks / 4 824 vars / 15 120 edges** (5.1× the rounds=1
graph), max check degree **35** (vs 25). The new emitter mode
(`qec_q7_bp_graph -- wingraph rounds p W C bankW bankV`) solves banking at 8/24 (GC=54, GV=201)
and 4/12 (GC=108, GV=402); the M8 `bp_relay_banked` RTL synthesises against the swapped header
unchanged:

| config | CLB LUT | % of 117 120 | LUT-as-logic | LUTRAM | DSP | Fmax OOC | closed-form worst-case @133 MHz |
|--------|---------|--------------|--------------|--------|-----|----------|--------------------------------|
| 8/24   | 197 701 | **169 %**    | 183 491      | 14 210 | 78  | 177.7 MHz | ~16.0 k cyc ≈ 120 µs |
| 4/12   | ~160 k  | **~138 %**   | —            | 13 880 | 40  | 174.7 MHz | ~31.8 k cyc ≈ 238 µs |

**Neither banking fits the KV260, and the failure mode is informative**: narrowing 8/24 → 4/12
cut only 19 % — the area is dominated by an E-proportional constant (the literal edge-table ROMs
`BP_CHK_EDGES`/`BP_EDGE_*`/λ/γ, all in LUT logic), not by the bank/compute fabric (LUTRAM is a
cheap 14 k, DSP negligible, timing comfortable at ~175 MHz). Meanwhile **144 BRAM36 + 64 URAM sit
idle (RAMB=0)**. The M9b architecture lever is therefore to move the E-indexed tables into
BRAM/URAM (+1 read-latency stage, absorbed by the existing software pipeline) — or a bigger part.
Worst-case real-time at 1 µs/round was already out of reach per the spec's honest-expectations
section; M9b targets the architecture + honestly measured max round rate (the Q7-01 ASIC de-risk
input), with early-exit medians expected ~5 % of worst-case per the M8 distribution.

-----

# M9b — streaming RTL: uniform-graph schedule, bit-exact co-sim, KV260 fit in progress (Q7-04)

**Status:** AC-2 (streaming schedule bit-exact to the windowed golden, Verilator co-sim) is
**met**. Vivado synthesis of the BRAM-ified core is **in progress** at time of writing — see the
KV260 fit subsection; the fit outcome does not gate AC-2.

M9a froze the software operating point (W=6, C=2, residual-only seam). M9b builds the RTL that
implements it: a hardware-schedule golden that one baked RTL header can reproduce, a BRAM-ified
sibling of the M8 banked core sized for the 5.1×-bigger window graph, a WARM/RUN/WAIT/COMMIT/
SLIDE/RELOAD streaming FSM lifted from the Q6-20 UF pattern, and an AXI-Stream front end — all
gated bit-exact against `FixedRelayBp` at every layer.

## What shipped

- **`HwSlidingWindowBp`** (`crates/aleph-qec/src/relay_window.rs`) — the hardware-schedule golden.
  Unlike M9a's `SlidingWindowBp` (a different, exact DEM per window position), it compiles **one**
  interior window graph and decodes every slot on it, zero-padding past the stream end; commit is
  the baked rule "var has an in-window detector at relative round < C". `WindowTrace` (per-slot
  `committed`/`obs`/`valid`/`commit_clean`) is the unit the RTL co-sim compares bit-for-bit.
  `with_early_exit(bool)` passes through to `FixedRelayBp`'s first-valid-leg break.
- **Emitter modes** `streamgraph`/`streamvectors`/`streamvectorsearly`
  (`crates/aleph-qec/examples/qec_q7_bp_graph.rs`) — window header + streaming metadata, and
  per-mode golden round-streams/slot-decisions (`hw/bb_stream_tanner.svh`,
  `hw/bp_stream_vectors.txt`, `hw/bp_stream_vectors_early.txt`, all committed).
- **`bp_relay_banked_bram(_m)`** (`hw/bp_relay_banked_bram.sv`, `hw/bp_relay_banked_bram_m.sv`) —
  the M8 `bp_relay_banked` core with its O(E) literal-constant fabrics (edge tables, λ/γ, obs
  masks, scatter addresses) moved into sync-read ROMs; `_m` further wraps each ROM in a stamped
  cell module (14 cells) for the Vivado fit attempt. Decision-equal to M8's `bp_relay_banked`.
- **`bp_streaming_decoder.sv`** — the WARM→RUN→WAIT→COMMIT→SLIDE→RELOAD FSM (lift of
  `uf_streaming_decoder.sv`) implementing the W=6/C=2 window slide over the BRAM core.
- **`bp_stream_win_core.sv` / `bp_stream_win.v`** — the AXI4-Stream front end (3-beat 72-bit round
  framing, 32-bit result word, per-frame `frame_rst` re-arm) plus the Verilog-2001 BD-top
  passthrough, both M9c board-build units.

## Why a uniform graph, and its measured price

M9a's exact-schedule golden compiles a *different* DEM per window position (left boundary, right
boundary, interior, tail all differ). A single baked RTL header cannot replay that bit-exactly, so
M9b's hardware contract is: **every slot decodes on the one interior window graph** — the strict
translation-invariant DEM M9a already proved identical across interior positions — with local
zero-pad drain past the stream end and the baked commit rule. This is a **documented deviation from
the design spec**, not a shortcut: it is the only way one RTL header can serve every slot, and its
cost is measured, not assumed.

**The cost, measured (EPYC, 20 000 shots, gross code, rounds=12, seed 2024, W=6/C=2):**

| p | LER batch | LER exact-schedule (M9a) | LER hw-schedule | within CI | hw nonconv | hw discarded>0 |
|---|-----------|---------------------------|------------------|-----------|------------|-----------------|
| 0.001 | 1.050e-3 ± 4.5e-4 | 1.400e-3 ± 5.2e-4 | 2.000e-3 ± 6.2e-4 | ✓ | 11.82 % | 0.33 % |
| 0.003 | 4.565e-2 ± 2.9e-3 | 5.250e-2 ± 3.1e-3 | 5.840e-2 ± 3.3e-3 | ✓ | 66.82 % | 8.91 % |
| 0.005 | 3.517e-1 ± 6.6e-3 | 3.857e-1 ± 6.7e-3 | 4.158e-1 ± 6.8e-3 | ✗ | 96.01 % | 51.02 % |

(`m9b-hw-sweep-20k.csv`, `--hw` mode of `qec_q7_stream_sweep`.) At p=0.003 the hw schedule is
**1.11× the exact schedule / 1.28× batch** — worse than M9a's accepted **1.24× batch** at the same
config, an honest widening the uniform-graph simplification costs. At p=0.005 it **DIFFERS**
outside CI (4.16e-1 vs 3.86e-1, +8 %) with 96 % non-convergence — recorded for completeness, the
same "everything is broken here" regime M9a flagged. At the sub-threshold operating point
(p=0.001) hw is within CI of both references.

**Discarded-bits metric (`commit_clean`).** The first cut at a residual observable — the frame's
lit-bit count after the final slide — is **vacuous by construction**: after the last slide the
frame holds only zero-padded rounds, so it is always 0 (found in review; the initial smoke's
`hw_resid_frac = 0.0000` at every p was the tell, not a clean result). The fix:
`WindowTrace.commit_clean` = commit region `[0, C·dpr)` all-zero **after** the slot's commit toggle
but **before** the slide — a per-slot, non-vacuous drain check that maps directly to the RTL result
word's `residual_empty` bit. `StreamStats.residual` becomes the cumulative popcount of bits
discarded by slides. Measured discard rate: **0.33 % / 8.91 % / 51.02 %** of shots at p = 1/3/5×
10⁻³ — well below the non-convergence rate (12 %/67 %/96 %) at every p, because most non-converged
windows still drain their commit region cleanly; discarded-bits is the sharper RTL health signal
for Q7-07's fallback-policy design, non-convergence alone overstates the problem.

## Per-mode goldens — first-valid ≠ best-kept

The plan's original assumption ("the golden is schedule-independent — identical decisions both
modes") was wrong, caught by the first co-sim run: the RTL core's early-exit commits the **first**
syndrome-valid leg, while the software golden (as in M9a) keeps the **best-kept** (lowest-weight
valid) decision over the whole schedule. These differ whenever a later leg finds a lower-weight
valid decision than an earlier one — **25 of 280 slots** at the operating point (40 trials × 7
slots). This is exactly the house pattern already established at M6–M8 (`circvectors` /
`circvectorsearly`): each mode gets **its own golden**. `HwSlidingWindowBp::with_early_exit`
generates `hw/bp_stream_vectors_early.txt` alongside the best-kept `hw/bp_stream_vectors.txt`
(identical header, identical 520 r-lines/shots, exactly 25/280 w-lines differing — the divergence
set was cross-validated software-golden-diff vs RTL-co-sim-diff and the two sets match exactly).
Both files are committed; the co-sim gates each mode against its own file.

## Co-sim gates

- **`bpstream`** (`hw/tb_bp_stream.cpp`, `hw/bp_streaming_decoder.sv`): **40 trials × 7 slots ×
  2 early-exit modes, each bit-exact vs its own HW-schedule golden.**
- **`bpstreamaxi`** (`hw/tb_bp_stream_axi.cpp`, `hw/bp_stream_win_core.sv`): **5 gates × 2 modes,
  all green** — zero-stream, golden-equality, back-pressure-invariance, frame-independence, and a
  review-driven **adversarial drain-stall** gate with a **negative control**. The drain-stall gate
  holds `m_axis_tready` low for 40 000 cycles spanning the post-`in_last` tail (the decoder
  self-drives ⌈W/C⌉=3 zero-pad slots with no input handshake to gate on) and checks all 7 words
  survive intact. The original 1-deep result slot **failed** this gate 6/7 and 5/7 words under the
  same stall (the negative control, run from the pre-fix commit outside the tracked tree) — proof
  the new gate discriminates the exact defect the review caught: a stalled consumer during the
  drain silently overwrites a parked result. The fix is a **result FIFO of depth ⌈W/C⌉+1 = 4**,
  sized to the drain's own bound (the drain starts from an empty FIFO by construction — the
  streaming-phase input gate only ever admits a round when the prior slot's result has already been
  consumed — and emits at most ⌈W/C⌉ slots with no further input), because the internal zero-pad
  drain removes the decoder's own input-gating back-pressure guarantee once `in_last` lands.

## Latencies (core → window, at the operating point)

| level | config | cycles | wall-clock |
|-------|--------|--------|------------|
| core, rounds=1 (M8 comparison) | 16/48 | 2085 (M8) → **2206** (BRAM, +121) | 15.64 → 16.5 µs @133.332 MHz |
| core, rounds=1 (M8 comparison) | 8/24 | 3750 (M8) → **3871** (BRAM, +121) | — |
| window decode, full schedule | W=6, 16/48-class fabric | **16 298** cyc (fixed, 280/280 slots) | **≈ 122 µs** @133.332 MHz |
| window decode, early-exit | W=6 | min **722**, mean **4898** (236/280 windows exit early; 44/280 run full) | min ≈ **5.4 µs**, mean ≈ **37 µs** |

The BRAM re-lag (+121 cyc, not the originally-projected +122 — one cycle recovered when the
S_EMIT/CHK-sbit ROM reverts below removed a tail cycle) is the honest price of moving the E-fabric
off LUT-constant taps onto sync-read ROMs: one extra register stage per launch/scatter, uniformly.
It is unrelated to the window-vs-core jump — the window's 16 298 cyc is a **6.8×** larger schedule
than the rounds=1 core's 2206 (the W=6 window graph is 5.1× the rounds=1 edge count, plus one
extra CHK/VAR-fabric register stage from the BRAM core).

**The honest framing (feeds M9c):** at 122 µs worst-case and ~37 µs mean, the window decoder is
**far over** the ~C µs (2 µs) real-time budget a continuous round stream needs — exactly the
spec's own honest-expectations section anticipated. M9b's deliverable is not sub-µs streaming; it
is the **architecture** (bit-exact schedule, robust AXI shell) plus the **honestly measured**
worst/mean rate, which is the Q7-01 ASIC de-risk input and the M9c starting line, not a claimed
real-time result.

## KV260 fit — Vivado synthesis in progress, honest status

**Pre-probe (parallel de-risk, done before Task 3/4 started; recorded in M9a's tail).** The W=6
window graph (432 checks / 4824 vars / 15 120 edges, 5.1× the rounds=1 graph, max check-degree 35)
run through the **unmodified M8 LUT core** does not fit: 8/24 → 169 % LUT, 4/12 → 138 % LUT.
Narrowing the banking barely helped (only 19 %), because the area is dominated by an
**E-proportional constant** — the literal edge-table ROMs (`BP_CHK_EDGES`/`BP_EDGE_*`/λ/γ), all
synthesized as LUT logic — not by the bank/compute fabric. Meanwhile 144 BRAM36 + 64 URAM sat
idle. The lever: move the E-indexed tables into BRAM/URAM. This became the M9b Task 3 mandate.

**The rule, re-learned (cites the M7-era Vivado-folding lessons already in this doc: constant-fold
over `leg`, tree-restructure serial reductions, modularize before flattening).** The BRAM-ify pass
went through five structural iterations before landing on a synthesizable shape, each one a genuine
Vivado behavior (not a simulator quirk — Verilator constant-folds all of these; Vivado does not):

1. **2-D unpacked ROMs stall.** The first BRAM-ified core used two-dimensional unpacked ROM
   arrays; Vivado silently ignores `rom_style="block"` on that shape and decomposes it into
   per-element registers + a read network — reproducing the exact register-fan-out explosion the
   BRAM conversion exists to remove. Fixed by repacking every ROM to one unpacked dim (depth) ×
   one packed row word.
2. **Repacked 1-D ROMs still OOM.** The cost was not the ROM shape but Vivado *elaborating* the
   computed `initial`-block fills (helper functions over header localparams) — independent of
   `rom_style` (a `distributed` variant hit the same ~50 GB peak). Fixed by moving all ROM content
   computation into the Rust emitter as packed-row hex literal tables; the RTL fills became pure
   literal copies, with a `rom_contract` guard recomputing every row the old way and `$fatal`ing on
   any mismatch.
3. **Same stall persisted — the real whale was ROM-fed runtime indices into big register arrays.**
   Two sites used a ROM output to index a `BP_C`/`BP_N`-scale register array at runtime
   (`s_reg[chk_sbit_idx_r[j]]`, the S_EMIT `obs`/`corr_out` reduction) — Vivado does not fold
   runtime-indexed reads/writes into large register arrays; it port-splits and explodes, the exact
   M7-era cursor-mux lesson recurring one level down. Fixed by reverting those two sites to the
   original constant-folded `pc==g` form and deleting their ROMs (14 ROMs remain from the original
   19). This produced the documented **rule**, now written into the core's own header comment: **ROM
   outputs may feed data or real-memory addresses only, never register-array indices** — C/N-scale
   index sites stay constant-folded; only the genuinely E-scale edge fabric needs the ROM lever.
4. **Literal-table fills alone did not clear the wall** — confirms (3) was the actual root cause,
   not the fills.
5. **A rule-clean flat core was, at time of writing, still grinding** through Vivado's elaboration
   at the rounds=1 fit-probe scale (part-load ~2× faster than run 2, plateaued after part-load with
   no phase banner for over 1.5 hours) — left running as the A/B control. In parallel, a
   **modular** sibling (`bp_relay_banked_bram_m.sv`, 14 stamped ROM cells, the same
   `-flatten_hierarchy none` lesson that cleared the M7 full-unroll wall in ~3 minutes where the
   flat top stalled) was built and is racing it, on the theory that hierarchy — not ROM shape —
   is what Vivado's Cross-Boundary-Area-Optimization needs on this fabric.

Both cores are **bit-exact / decision-equal to `FixedRelayBp` and to the M8 LUT core** in Verilator
(40/40 at every banking, worst latency unchanged at 2206/3871 for the flat core and identical for
the modular sibling) — the open question is Vivado synthesizability of the W=6 window header at
fit-probe scale, not correctness.

<!-- FIT-RESULT-PENDING: table to be filled from RESULT lines once run 5 / the modular A/B converges -->

| config | CLB LUT | LUTRAM | DSP | RAMB | URAM | Fmax |
|--------|---------|--------|-----|------|------|------|
| TBD (flat, rule-clean) | — | — | — | — | — | — |
| TBD (modular, `_m`) | — | — | — | — | — | — |

**Honest framing.** AC-2 (streaming schedule bit-exact to the windowed golden, Verilator co-sim) is
**met and independent of this fit outcome** — it was gated entirely on the software golden and
Verilator, not on Vivado. The BRAM-ify lever is **validated in simulation**; Vivado synthesis of
the W=6 window header is **in progress**, and this table will be filled from the run-5 / modular-A-B
`RESULT` lines once one of them converges (or from whichever escalation — a bigger part, or a
further structural rework — the outcome demands).

## Deviations from the design spec

- **Uniform hardware schedule, not per-slot exact DEMs** — one baked interior window graph
  replays every slot (boundary/tail slots use the same graph as interior ones); the LER cost is
  measured above (1.11× exact-schedule at threshold), not assumed.
- **No `BP_VAR_DET` array** — a committed var's residual toggle set is derived from the existing
  CSR tables (`BP_VAR_OFF`/`BP_EDGE_CHK`) already in the header; only the new 1-bit/var
  `BP_VAR_COMMIT` was added, per the plan's design-decision #2.
- **Emitter CLI arg order** — `streamgraph`/`streamvectors[early] rounds p W C [bankW bankV | n
  seed]` puts (W, C) at positions 4/5 (matching `wingraph`), not the spec's `bankW bankV W C`.
- **Per-mode goldens, not one schedule-independent golden** — see the "first-valid ≠ best-kept"
  section above; a plan-execution finding, not a pre-planned design choice.
- **`commit_clean`/residual redefinition** — the spec's original end-of-frame residual metric was
  vacuous (always 0); redefined to the per-slot pre-slide drain check documented above.

## Reproduce

```bash
# Software: hardware-schedule golden vs exact-schedule vs batch, at the frozen op point
cargo run --release -p aleph-qec --example qec_q7_stream_sweep -- 12 20000 2024 --hw

# Emitter: window header + streaming metadata, and per-mode golden vectors (W=6, C=2)
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- streamgraph 12 0.003 6 2 8 24
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- streamvectors 12 0.003 6 2 40 7
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- streamvectorsearly 12 0.003 6 2 40 7

# RTL co-sim gates
make -C hw bpbankedbram      # BRAM-ified core vs M8 core, both bankings, 40/40 decision-equal
make -C hw bpbankedbramm     # modular ROM-cell sibling, same gate
make -C hw bpstream          # streaming FSM, 40 trials x 7 slots x 2 modes vs own goldens
make -C hw bpstreamaxi       # AXI shell, 5 gates x 2 modes incl. adversarial drain-stall
```

## Files

- `crates/aleph-qec/src/relay_window.rs` — `HwSlidingWindowBp`, `WindowTrace`, `with_early_exit`.
- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — `streamgraph`/`streamvectors`/
  `streamvectorsearly` emitter modes.
- `crates/aleph-qec/examples/qec_q7_stream_sweep.rs` — `--hw` comparison mode.
- `hw/bp_relay_banked_bram.sv`, `hw/bp_relay_banked_bram_m.sv` — BRAM-ified banked core (flat +
  modular-ROM-cell sibling).
- `hw/bp_streaming_decoder.sv` — WARM/RUN/WAIT/COMMIT/SLIDE/RELOAD streaming FSM.
- `hw/bp_stream_win_core.sv`, `hw/bp_stream_win.v` — AXI4-Stream front end + BD-top passthrough.
- `hw/tb_bp_stream.cpp`, `hw/tb_bp_stream_axi.cpp` — Verilator TBs (`bpstream`, `bpstreamaxi`).
- `hw/bb_stream_tanner.svh`, `hw/bp_stream_vectors.txt`, `hw/bp_stream_vectors_early.txt` —
  committed generated artifacts.
- `m9b-hw-sweep-20k.csv` (EPYC 20k-shot `--hw` sweep; scratchpad-only, not committed to the repo,
  matching the M9a-sweep-CSV convention) — source of the batch/exact/hw LER table above.
