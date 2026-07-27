# Q7-02 — ASAP7 placed timing and power: what is measured, and what is not

Status: **in progress.** This file records the timing half of Q7-02 AC-2 ("synthesis reports meet, or
quantify the gap to, the Q7-01 budgets"). The power half (Task A4, gate-level switching activity) is
not here yet and is called out as open below.

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

## 4. Power — not yet measured honestly

The FLOW_DONE run reports:

```
finish__power__internal__total  0.339059 W
finish__power__switching__total 0.316803 W
finish__power__leakage__total   6.36414e-05 W
finish__power__total            0.655926 W
PSM worst IR drop 33.9 mV (4.40 %), average 0.766 mV
```

**This is at ORFS's default switching activity, not ours**, so it is not a decoder power number and
must not be quoted as one. Task A4 replaces it with activity derived from gate-level simulation of the
routed netlist (`6_final.v`, 78 MB) against the existing co-simulation vectors — which also closes the
long-standing gap that no gate-level co-simulation of an ASIC netlist has ever been run, despite
`asic-architecture.md` §6 asserting the chain "retargets to gate-level unchanged".

The ASAP7 platform does ship behavioural Verilog for its standard cells
(`platforms/asap7/verilog/stdcell/*.v`), so the path is open.

Note for when the number arrives: energy per window is roughly **banking-invariant** while latency is
not, and a 28 nm part will draw several times a 7 nm figure — the §5 "≤ 1 µJ per window" budget almost
certainly needs restating at whatever node is actually targeted.

-----

## 5. Open items

1. **One defensible Fmax.** Re-run the ASAP7 flow from CTS with `SKIP_CTS_REPAIR_TIMING` unset and
   `SKIP_INCREMENTAL_REPAIR` kept, then extract post-route and report `report_clock_min_period`. That
   is the only way to get a repaired *and* routed number, and it collapses the 528–686 MHz band to one
   figure.
2. **Power with real activity** (Task A4, §4).
3. **The 85 `%Warning-LATCH` in `bp_relay_banked.sv:729`** that Verilator 5.032 emits and 5.050 does
   not. Not a functional bug — the core is bit-exact over 10⁶ shots on silicon — but an `always_comb`
   that infers latches deserves an explanation rather than a version pin.
4. **Enable granularity** (§2) if the clock structure is ever to be fixed rather than characterised.

## 6. Reproduction

Box `root@195.154.249.85`, `/data/asicprobe`:

- FLOW_DONE outputs: `orfs_out_m8rf_asap7/` (config `orfs/m8rf_asap7/`)
- Clock-structure query (Task A1): `clk_struct.tcl`, `run_clkstruct.sh` → `clk_struct.log`
- CTS-stage repair (Task A3): `repair_cts.tcl`, `run_repair_cts.sh` → `repair_cts.log`,
  `asap7_cts_repaired.odb`

Trap worth repeating: in these wrapper scripts the redirect `> /work/x.log` runs on the **host**, not
inside the container — `/work` exists only inside Docker. Use the full host path.
