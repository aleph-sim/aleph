# Q7-02 — ASAP7 placed timing and power: what is measured, and what is not

Status: **closed for #322 (2026-08-28).** This file records Q7-02 AC-2 ("synthesis reports meet, or
quantify the gap to, the Q7-01 budgets"): the timing half (Tasks A1–A3), the power half with real
switching activity from gate-level simulation (Task A4, §4), and the first gate-level co-simulation of
an ASIC netlist for this project — which does **not** pass, for a reason STA names (§4b). The closure
note for issue #322 is §7.

Design under measurement: the M8 banked relay-BP core with the Q7-08 register-file style
(`bp_m8rf_elab.v`, 16/48 banking), through OpenROAD-flow-scripts on the **ASAP7** predictive 7 nm
platform. ASAP7 is a proxy, not a fabbable node — see `docs/qec/regfile-plan.md` for why sky130hd was
abandoned for placed timing.

Tooling: OpenROAD `26Q3-528-g20d2d5c16e`, config `orfs/m8rf_asap7/`, on the EPYC bench box.

-----

## 1. Headline: the Fmax number straddles the budget

The Q7-01 budget is **≥ 600 MHz**. Two defensible measurements exist and they fall on opposite sides
of it, because they are taken at different flow stages under different parasitic models:

| Measurement | Stage | Parasitics | Timing repair | Fmax |
|---|---|---|---|---|
| `6_finish.rpt` (FLOW_DONE run) | post-route | **extracted (RCX, `6_final.spef`)** | **skipped** | **686.13 MHz** |
| Task A3 standalone run | post-CTS | placement estimate + platform `setRC.tcl` | **ran to completion** | **527.66 MHz** |

**These two numbers are not comparable, and neither should be quoted alone.** One is routed but
unrepaired; the other is repaired but not routed. The honest statement today is that the placed Fmax of
this core on ASAP7 lies in the **528–686 MHz band**, i.e. it straddles the ≥ 600 MHz budget, and a
single defensible figure requires one more run (§5).

> **Added by Task A4 (§4b): both figures are setup-only numbers on a netlist that fails hold.** The
> same `6_finish.rpt` reports **44 704 hold violations, worst −745.9 ps**, 98 % of them on the D pins
> of the latch register file, and hold was never repaired (the repair pass that crashed in A3 is the
> one that would have). Hold is frequency-independent, so this netlist would not decode correctly at
> *any* clock; the gate-level co-simulation in §4b shows exactly that. "686 MHz" is the speed of the
> datapath, not of a working chip. The 144/864 run (Task B3) carries the same defect: 35 616 hold
> violations, worst −1486.9 ps.

Supporting detail for the post-route figure (`6_finish.rpt`, SDC period 1000 ps):

```
clk period_min = 1457.44   fmax = 686.13
wns max        = -622.73
tns max        = -3947471.25
setup skew     = 940.67
```

Supporting detail for the post-CTS repaired figure:

```
before repair (ODB parasitics, no wire RC -- not quotable): fmax 681.23, WNS -507.25, TNS -1185678
repair start  (setRC + estimate_parasitics -placement):     WNS -1057.2, TNS -25525560, 52817 violating endpoints
after repair:                                               WNS  -957.02, hold WNS -958.15, TNS -20761226
                                                            clk period_min = 1895.16  fmax = 527.66
                                                            design area 83367 um^2 (51% util), +2.8% vs pre-repair
```

-----

## 2. What actually limits Fmax: the gated clock of the latch register file

Not the datapath. The worst setup path in the routed design starts at a **latch**:

```
Startpoint: gmcm[250].u_mcm.mem[18]$_DLATCH_P_  (positive level-sensitive latch clocked by clk')
Endpoint:   gchk[5].m_in_jr[194]$_DFF_P_
data arrival 2040.15 ps, of which ~1370 ps is clock insertion; required 1417.41; slack -622.73
```

That latch's clock arrives **1.37 ns** after the clock root, against **0.47 ns** for an ordinary flop
on the same tree. The path shows why:

1. The CTS tree reaches `clkbuf_leaf_1980_clk_regs` at ~472 ps of insertion — healthy, and comparable
   to the flop branch.
2. It then passes through `_0623647__192769/A (INVx1)` and `_0623648_/C (AND3x1)` — a **generic gating
   cell**, not a clock-gating cell.
3. Beyond that gate the clock is buffered by a **serial daisy-chain** of ~12 `load_slew*` BUFx12f
   buffers, each with fanout ~18–20, inserted by `repair_design` rather than by CTS. Delay accumulates
   linearly down the chain.

TritonCTS does not push a balanced tree through arbitrary combinational logic, so everything past the
AND gate is outside the clock tree it built. And the gate is a plain AND rather than an integrated
clock-gating cell for a platform reason: **ORFS's ASAP7 `config.mk` sets `DONT_USE_CELLS += SDF* ICG*`**
— ICG cells are banned, so synthesis had nothing else to map the enable onto.

**The Fmax limiter is therefore a flow artefact interacting with the Q7-08 latch register file, not the
decoder datapath.** Two independent pieces of evidence support that:

- CTS-stage Fmax (681.23 MHz) is within 1 % of post-route Fmax (686.13 MHz). Routing is not the
  constraint; the clock structure is, and it is fully formed by CTS.
- Timing repair, once unblocked (§3), moved WNS by only ~9 % (−1057 → −957 ps) and left all 52,817
  violating endpoints in place, while improving TNS by 19 %. It fixes the bulk of ordinary paths and
  cannot touch the worst one.

### Scale of the problem (Task A1)

| | latches | flops |
|---|---|---|
| cells | 55,703 | 43,025 |
| **distinct clock nets** | **15,951** | 2,321 |
| sinks per net | ~3.5 (max 25) | ~18 |

15,951 gated-clock domains is the number that decides what can be done about this:

- **Declaring them in SDC** (`create_generated_clock` per gated net, so CTS balances them) is not
  viable at that count — STA would be unusable.
- **Re-synthesising with ICG cells enabled** inherits the same problem: ~16 k clock-gating cells.
- The tractable lever is an **RTL change coarsening enable granularity** — one enable per bank rather
  than per register (~3.5 latches per enable today). That is design work, not a tooling flag, and it
  is not in Q7-02's scope. Recorded here so the next person does not rediscover it.

-----

## 3. The `repair_timing` crash, root-caused and worked around

The ASAP7 flow ran with both `SKIP_CTS_REPAIR_TIMING = 1` and `SKIP_INCREMENTAL_REPAIR = 1`, which is
why the post-route figure above is unrepaired. Both were set on the assumption that the sky130
`[CRITICAL ODB-0445]` CTS crash applied here too. **That assumption was wrong, and only one of the two
skips was ever necessary.**

What actually crashes is the **GRT-stage incremental repair**, with a null dereference inside OpenSTA's
CRPR arrival pruning, reached through the incremental parasitics update:

```
rsz::Resizer::repairSetup
  -> rsz::SetupLegacyPolicy::iterate -> runMainRepairLoop -> repairEndpoint
  -> est::EstimateParasitics::updateParasitics
  -> grt::GlobalRouter::updateDirtyRoutes -> grt::FastRouteCore::run -> layerAssignment
  -> grt::FastRouteCore::updateSlacks -> sta::Sta::slack -> Sta::findRequired
  -> sta::Search::findAllArrivals -> findArrivals1 -> ArrivalVisitor::visit
  -> sta::ArrivalVisitor::pruneCrprArrivals -> sta::Path::minMax   <-- SIGSEGV
```

It died on `do-5_1_grt` (make Error 245) after 1 h 05 min, at repair iteration 30.

Two workarounds were considered:

- **Disable CRPR** — the function that crashes. **Not possible:** `set_crpr_enabled` does not exist in
  this OpenROAD build (`info commands` returns nothing for it); CRPR is not a user-settable toggle.
- **Avoid the GRT incremental path** — repair at the CTS stage with `estimate_parasitics -placement`,
  which never calls `updateDirtyRoutes`. **This works.** The run completed, passing iteration 30 where
  the GRT-stage run died, and produced the numbers in §1.

A standalone `openroad` invocation also needs the platform's `setRC.tcl`; ORFS supplies layer/wire RC
via `SET_RC_TCL`, and without it the resizer aborts with `[ERROR RSZ-0089] Could not find a resistance
value for any corner` before doing any work.

**Flow consequence: `SKIP_CTS_REPAIR_TIMING` should be unset for ASAP7.** Only `SKIP_INCREMENTAL_REPAIR`
is genuinely required.

-----

## 4. Power — measured with real switching activity (Task A4)

**Method.** The routed 16/48 netlist (`6_final.v`, 639 322 cells: 72 k INV, 56 k AO21, **55 703 DHLx1
latches, 43 025 DFFHQNx1/2/3**, no other sequential cells) was compiled with Verilator 5.032 against
the ASAP7 behavioural cell models — the combinational files with their guarded `primitive…endprimitive`
UDP definitions stripped (never instantiated), ORFS's `dff.v` for `DFFHQNx*`, and a two-line
`always_latch` model for `DHLx1` (`hw/asap7_latch.v`) — driven by the *same* `tb_bp_banked.cpp` and the
same 40 circuit-level golden vectors the RTL passes (`-DBP_GATE_PORTS` addresses the packed vector
ports a netlist exposes; `-DBP_TRACE` adds a VCD window). Build: 25 min, **95 GB peak RSS** on the
EPYC box — the 144/864 netlist (433 MB) does not fit in 123 GB and was not attempted. Run: 37 s for
40 decodes (83 485 cycles). Two 300-cycle VCD windows were dumped: cycles 1000–1300 (steady-state BP
iteration inside decode 0) and 300 cycles of the parked decoder after the last decode (`+idle=400`).
Each window was fed to OpenSTA through the ORFS environment (`read_vcd -scope TOP/bp_relay_banked`,
`6_final.spef`, TT NLDM libs, SDC period 1000 ps): **2 128 987 pin activities annotated, 0 unannotated.**

| Activity source | Total | Sequential | Combinational | Clock | Internal / switching |
|---|---|---|---|---|---|
| ORFS default (`6_finish.rpt`, not a decoder number) | 0.656 W | 0.115 W | 0.466 W | 0.075 W | 52 % / 48 % |
| **Gate-level VCD, steady BP iteration** | **0.149 W** | 0.064 W | **0.0115 W** | 0.0736 W | 75 % / 25 % |
| Gate-level VCD, idle (parked, clock running) | 0.138 W | 0.065 W | ≈ 0 | 0.0730 W | 79 % / 21 % |

Leakage is 6e-5 W throughout (7 nm predictive TT, negligible).

**What the numbers say.**

1. The default-activity figure over-estimates the decoder by **4.4×**, almost entirely in the
   combinational group (0.466 → 0.0115 W). The datapath of this core is quiet: min-sum messages are
   8-bit, most banks are idle in any given cycle, and the decode converges in a few relay legs.
2. **93 % of the real power is clock + sequential internal** — 43 k flops and 55 k latches being
   clocked at 1 GHz, whether or not they do anything (idle 0.138 W vs active 0.149 W). This is the
   direct cost of the clock structure diagnosed in §2: no integrated clock gating (ASAP7 platform
   `DONT_USE ICG*`), so every register toggles its clock pin every cycle. A gated design would remove
   most of the 0.13 W floor; the datapath itself is ~10 mW.
3. **Energy per decode window: 149 pJ/cycle × 2085 cycles = 0.31 µJ** (16/48, full best-kept schedule;
   early-exit windows are shorter in proportion). Dynamic power scales with clock, so the per-window
   energy is clock-independent to first order; at the 686 MHz of §1 the power would be ~0.10 W. This is
   inside the §5 "≤ 1 µJ per window" budget of `asic-architecture.md` with 3× margin — **at 7 nm
   predictive**. Energy is roughly banking-invariant (the same messages are computed whichever bank
   computes them), while a 28 nm part draws several times a 7 nm figure at the same activity; the
   budget needs restating at the real target node, as §4 of the previous revision already warned.

**Caveats, in order of weight.** (a) The activity comes from a simulation that is *not* bit-exact
(§4b): ~1 % of output bits differ from the golden, which changes toggle statistics negligibly but
means this is the activity of a nearly-correct decode, not of the exact one. (b) ASAP7 is a predictive
PDK with NLDM tables; treat the absolute watts as ±30 %. (c) 300 cycles of one decode; the
distribution across the 40 shots was not sampled (the decoder is schedule-driven, so cycle-to-cycle
activity is nearly periodic, and the idle window bounds the floor).

## 4b. Gate-level co-simulation: the netlist is not bit-exact, and STA says why

**This is the first gate-level co-simulation of an ASIC netlist in this project** — the claim in
`asic-architecture.md` §6 that the harness "retargets to gate-level unchanged" is now tested. It
retargets (same testbench, same vectors, two `#ifdef`s), and it **fails**:

```
latency distribution (cycles -> shots): 2085:40          <- schedule/control path exact
FAIL: 40/40 decodes mismatched                           <- 8–15 fields per decode of 877
    corr_out[81]: got 1 want 0 ... valid_flag: got 0 want 1
```

The control FSM runs the exact 2085-cycle schedule on every shot; a handful of message bits are wrong
and the decode therefore fails to converge (`valid_flag = 0`). The RTL from the very directory the
netlist was elaborated from passes 40/40 on the same vectors (checked, same session), so this is a
netlist-level effect, not a vector/graph mismatch.

**STA names the mechanism.** From `6_finish.rpt` and a per-endpoint query (`hold_m8rf.tcl`):

| | 16/48 (`m8rf_asap7`) | 144/864 (`fp864_asap7`, Task B3) |
|---|---|---|
| hold violations | **44 704** | 35 616 |
| worst hold slack | **−745.9 ps** | −1486.9 ps |
| … of which on latch D pins (DHLx1) | **43 802** (worst −745.9) | not queried |
| … on flop D pins | 889 (worst −268) | |
| … clock-gating checks / other | 388 | |
| setup WNS (for scale) | −622.7 ps | −1118 ps |

Worst path: `gvar[0].u_var.m_out[34]` (flop) → `gmcm[200].u_mcm.mem[58]` (latch D). The latch rows are
transparent-low with enable `we & ~clk` (RTL: `bp_mcm_cell`, Q7-08), realised as `AND2(we, clk_leaf)`
feeding a CTS-built sub-tree with **1.37 ns insertion delay against 0.47 ns for the flops** (§2). The
latch therefore closes ~0.9 ns *after* the flops launch new data; any flop→latch path shorter than that
is a hold violation, and 43 802 of the 55 703 latch inputs are. A zero-delay simulator has no notion of
this skew, but its evaluation order lets the new data leak into the still-open latch on some paths —
the same failure the timing says silicon would have. Hold does not scale with the clock period: this
netlist is wrong at every frequency.

**Hold repair cannot fix it.** `repair_timing -hold -allow_setup_violations` on `6_final.odb` with
placement-estimated parasitics (avoiding the GRT path that segfaults, A3): 70 iterations, 335 delay
buffers, worst hold −746 → −728 ps, then `DPL-0038 Utilization greater than 100%`. Closing 0.9 ns on
44 k endpoints by buffering needs ~1 M buffers; the die has room for ~0.5 M cells. The fix is the clock
structure (§2, Task A2 / "enable granularity"), not the netlist — the same root cause now blocks Fmax,
power (§4 item 2) *and* correctness.

**Arbitration with an event-driven simulator was attempted and is inconclusive.** Icarus 14 with the
vendor UDP sequential models (`hw/tb_bp_gate_asap7.sv`, Verilog-2001 I/O because Icarus's SV string
support crashes its backend on this design): the full 2085-cycle schedule runs, but the datapath is X
from cycle 0 — the vendor latch model drives its UDP through `delayed_*` nets that only `$setuphold`
in the `specify` block connects, and with timing checks disabled they float; with those wired
directly and an `initial q = 0` added to the UDP, an 8-cycle probe (`xprobe.vcd`) still shows
676 493 of 677 307 signals at X from t = 0 — Icarus does not honour the UDP initial, and 55 703
uninitialised latches flood the datapath. Stopped there; it does not change the STA verdict.

**What the co-simulation *does* establish.** The netlist's control path, schedule, bank
gather/scatter and I/O contract are exact through synthesis, placement, CTS and routing — the 2085
cycles land on every shot and 99 % of the 877 output fields match. What is broken is precisely the
block that the flow already flagged (§2), and it is broken in the way the flow's own hold report says.

-----

## 5. Open items

1. **One defensible Fmax.** Re-run the ASAP7 flow from CTS with `SKIP_CTS_REPAIR_TIMING` unset and
   `SKIP_INCREMENTAL_REPAIR` kept, then extract post-route and report `report_clock_min_period`. That
   is the only way to get a repaired *and* routed number, and it collapses the 528–686 MHz band to one
   figure.
2. ~~Power with real activity~~ — done (§4). Open instead: **hold closure on the latch register
   file** (§4b). Not buffer-fixable; requires the clock-structure change in item 4. Until then no
   netlist from this flow is functionally sign-off-able, and Fmax figures are datapath-only.
3. **The 85 `%Warning-LATCH` in `bp_relay_banked.sv:729`** that Verilator 5.032 emits and 5.050 does
   not. Not a functional bug — the core is bit-exact over 10⁶ shots on silicon — but an `always_comb`
   that infers latches deserves an explanation rather than a version pin.
4. **Enable granularity** (§2) — no longer optional: it is the fix for items 1, 2 and the 0.13 W
   clock/sequential power floor (§4 item 2) alike.
5. **Gate-level arbitration with vendor models** (§4b, Icarus X sources) — nice-to-have; STA already
   settles the question.

## 6. Reproduction

Box `root@195.154.249.85`, `/data/asicprobe`:

- FLOW_DONE outputs: `orfs_out_m8rf_asap7/` (config `orfs/m8rf_asap7/`)
- Clock-structure query (Task A1): `clk_struct.tcl`, `run_clkstruct.sh` → `clk_struct.log`
- CTS-stage repair (Task A3): `repair_cts.tcl`, `run_repair_cts.sh` → `repair_cts.log`,
  `asap7_cts_repaired.odb`
- Task A4, all under `gate/`: `build_m8rf.sh` (Verilator build; models `cells_*.v`, `dff.v`,
  `asap7_latch.v`; 25 min, 95 GB), `obj_m8rf/sim_gate_m8rf` (co-sim + `+trace`/`+idle` windows →
  `win_active.vcd`, `win_idle.vcd`), `power_m8rf.tcl` → `power_m8rf.log` (run through ORFS:
  `make DESIGN_CONFIG=/work/orfs/m8rf_asap7/config.mk WORK_HOME=/work/orfs_out_m8rf_asap7
  RUN_SCRIPT=/work/gate/power_m8rf.tcl run` with `-e VCDS=… -e VCD_SCOPE=TOP/bp_relay_banked`),
  `hold_m8rf.tcl` → `hold_m8rf.log` (hold endpoints by cell kind), `holdfix_m8rf.tcl` →
  `holdfix_m8rf.log` (the failed hold repair), Icarus side `seq_sim.v`, `sim_iv_m8rf`,
  `bp_circ_vectors.gate.txt` (from `hw/sw/gate_vectors.py`). In-repo: `make -C hw bpgate-asap7`.
- Traps: `read_power_activities` is deprecated in this OpenSTA and its wrapper has the wrong arity —
  use `read_vcd -scope <scope> <file>`; Verilator's `commandArgsPlusMatch` returns the whole `+key=…`
  argument; a VCD of the *whole* run is 50 GB — always window it; the vendor UDP `altos_latch` is
  guard-defined in every model file, so a patched copy must come first *and* predefine
  `_udp_def_altos_latch_`.

Trap worth repeating: in these wrapper scripts the redirect `> /work/x.log` runs on the **host**, not
inside the container — `/work` exists only inside Docker. Use the full host path.

-----

## 7. Closure note for #322 (Task A5)

Q7-02 asked for two things. Stated without softening:

**AC-1, "RTL passes full co-simulation against the golden" — met, with one boundary.** The evidence
is the ~25 Verilator gates in `hw/Makefile` (`bpbanked` 40/40 at three bankings, `bpbanked-highweight`
2000/2000, `bpbankedrf` 40/40, `bpbankedscale` over eight geometries up to 144/864, the Beneš/AS-Waksman
fabrics at 10⁴ cases each), run in CI since #484, plus silicon: M8 on KV260 40/40 at 133.3 MHz, Q7-06
10⁶ × 3 shots with 0 mismatches, Q7-07 10⁵ shots `valid_mismatch = 0`. The boundary: **gate-level
co-simulation of the ASIC netlist now exists and fails (§4b)** — not because the RTL is wrong, but
because the routed netlist violates hold on 43 802 latch inputs, a flow/clock-structure defect that no
amount of RTL co-simulation can see. Also unmet, as before: the streaming core never fitted silicon
(M9c: LUT 162 %, BRAM 113 %); runt frames (`slices < W`) remain uncovered by co-sim; 64/192 and 144/864
are cycle-counted and (144/864) synthesised, not run on hardware.

**AC-2, "synthesis meets, or quantifies the gap to, the Q7-01 budgets" — quantified, not met.**

| Budget (Q7-01) | Measured | Verdict |
|---|---|---|
| Area | 16/48: 0.163 mm² die on ASAP7, 0 congestion; 144/864: sub-linear scaling (B3) | met, off-FPGA |
| Fmax ≥ 600 MHz | 528–686 MHz setup-only band (§1); 614.6 MHz at 144/864 (B3); **hold unclosed** | gap: not a working netlist |
| Power / energy | **0.149 W at 1 GHz, 0.31 µJ per window, real activity (§4)** vs ≤ 1 µJ budget | met at 7 nm; restate at 28 nm |

The single root cause behind the Fmax band, the hold failure and the 0.13 W clock floor is the
un-gated latch-register-file clock (§2). It is a physical-design task, not an RTL one, and it is the
first item of Phase E hardening in `docs/qec/open-silicon-program.md`. #322 is closed as the RTL issue
it was; the clock-structure work is tracked there.
