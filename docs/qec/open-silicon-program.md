# Open-Silicon QEC Decoder Program — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended)
> or `superpowers:executing-plans` to implement Phase A task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Track O, Track P (P1) and Track S (Phase A, Phase B) are implementable now. Phases C–E are
> money/decision gates executed by a human, and each names its exit criterion explicitly.

**Goal:** Ship a cheap, ready-to-use, openly-licensed real-time decoder for bivariate-bicycle qLDPC codes
— as a working product from day one, and as a sub-microsecond open ASIC at the end — so that any QEC lab
can deploy real-time decoding for a few hundred euro instead of the $10–30 k FPGA or $30 k+ GPU it
replaces. Design, data, results and bring-up software permanently open. Silicon funded from a fixed
personal budget of €50–100 k.

**Architecture:** Three tracks run **in parallel**, not in sequence.
**Track O (open)** makes the existing work usable and citable by outsiders — licence, CI, size, docs,
paper, data DOI, ecosystem plugins. **Track P (product)** ships a usable decoder appliance at every
stage of capability, starting with hardware that already exists and works today. **Track S (silicon)**
makes the numbers trustworthy, proves the full-parallel configuration on rented FPGA, rehearses the
tape-out flow on a free fully-open 130 nm shuttle, then tapes out the 28 nm part and gives the dies away.

The tracks are coupled at exactly one point: **Track P is Track S's demand gate.** We do not ask labs
whether they would use a decoder — we ship them one that works, count who actually deploys it, and let
that number authorise the silicon spend.

**Tech stack:** SystemVerilog RTL (`hw/`), Verilator co-simulation against the Rust golden model
(`crates/aleph-qec`), Vivado for FPGA, OpenROAD-flow-scripts for ASIC P&R, ASAP7 (predictive proxy) and
IHP SG13G2 / SKY130 (open PDKs) and TSMC 28 nm (production target).

## Global constraints

- **Everything published stays open.** RTL, scripts, campaign data, P&R configs, bring-up software.
  No NDA-encumbered artefact may become a dependency of the *published* flow.
- **Budget ceiling: €100 k total, self-funded.** Any stage that would exceed it stops the program instead.
- **No stage may start before its predecessor's exit criterion is met, in writing, in this file.**
- **Honest numbers only.** Every performance claim in public material names its measurement conditions
  and its comparison baseline, including where we lose. This is the project's reputational asset.
- **Correctness gates are non-negotiable.** Bit-exact co-simulation against the software golden model
  gates every RTL change (repo Golden Rule 1); a banking change is an RTL change.
- Existing repo conventions apply: no git worktrees, one issue one PR, `Closes #<issue-number>`,
  `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` must pass.

---

## 0. The decision that shapes the whole program

Two facts, established 2026-07-26, that cannot both be satisfied:

1. **Open PDKs stop at 130 nm.** SKY130, GF180MCU and IHP SG13G2 are the complete set. There is no
   open 28 nm or 22 nm PDK. At 130 nm this core runs ~90–150 MHz.
2. **Sub-microsecond decoding needs an ASIC clock.** The cycle model
   (`cycles = LEGS·ITERS·(GC+GV+7) + (2·GV+GC)`, `docs/qec/asic-architecture.md` §5) floors at
   **543 cycles** for the full-parallel 144/864 configuration — measured, not modelled, since Task B0
   Option A (2026-07-30). 543 cycles is 0.91 µs only at 600 MHz and 4.07 µs at the KV260's 133 MHz.

Therefore: **a fully-open-flow chip cannot be faster than the $300 FPGA we already have, and a chip that
beats it cannot have a fully-open flow.** The program resolves this by splitting "open" into two claims
and honouring both separately:

- **The design is open** — RTL, testbenches, campaign data, P&R scripts, bring-up software, results.
  This holds at every node, including 28 nm. Anyone with PDK access can re-run and re-fabricate.
- **The flow is open** — reproducible end-to-end by anyone with no NDA. This holds only at 130 nm, and
  is delivered by the Phase D shuttle, which is deliberately kept in the program for that reason alone.

The PDK itself is the one thing we cannot open, because it is not ours.

---

## 1. Who needs this chip and why — the case the money rests on

### 1.1 The one-line proposition

> A ~€500–1000 chip that does what a $10–30 k FPGA or a $30 k+ GPU does today: real-time relay-BP
> decoding of bivariate-bicycle qLDPC codes in under a microsecond — free to any lab that asks, with
> the RTL published.

### 1.2 Why an FPGA is not already the answer

It is, until you need sub-µs. Measured, on our own hardware:

| Platform | Config | Cycles | Clock | Worst-case latency |
|---|---|---|---|---|
| KV260 (~$300) — **shipped, measured** | 16/48 banked | 2085 | 133.332 MHz | **15.64 µs** (0.85 µs median early-exit) |
| KV260 | 144/864 full-parallel | 543 | — | **does not fit** (M9c: LUT 162 %, BRAM 113 %) |
| Large FPGA ($10–30 k), projected | 144/864 | 543 | ~200–300 MHz | 1.8–2.7 µs |
| **28 nm ASIC, projected** | 144/864 | 543 | 600 MHz–1 GHz | **0.54–0.91 µs** |

The full-parallel configuration — the only one that reaches the latency floor — **does not fit in an
affordable FPGA**. That is the entire technical justification for silicon, and it is a measured fact,
not an aspiration.

> **Correction (2026-07-26, from Task B1 — read this before quoting any number above).** The 144/864
> rows are **arithmetic from the cycle model, not a built design**. That configuration *cannot
> currently be generated at all*: `qec_q7_bp_graph -- circgraph` panics on every GC = 1 geometry (see
> Task B1). What is actually buildable today, and what the honest table looks like:
>
> | Config | Cycles | @600 MHz | @686 MHz (our measured ASAP7 Fmax) | @1 GHz |
> |---|---|---|---|---|
> | 64/192 — generates, in the `bpbankedscale` sweep | 913 | 1.52 µs | 1.33 µs | 0.91 µs |
> | 144/864 — **does not generate** | 544 | 0.91 µs | 0.79 µs | 0.54 µs |
>
> **On today's evidence sub-microsecond is not reachable**, because both levers are blocked at once:
> the full-parallel configuration does not build (Task B0), and the clock cannot currently exceed
> ~686 MHz because of the gated-clock structure — `repair_timing`, once unblocked, moved WNS from
> −1057 ps to ~−960 ps and then stalled with all 52,817 violating endpoints intact.
>
> This is Phase B doing its job: the finding cost about €0 and a day, instead of €45 k spent taping out
> a configuration that does not exist. It does, however, change the positioning honestly — against
> Riverlane's published <1 µs we are at 1.33–1.52 µs until B0 lands.

> **Correction 2 (2026-07-30, from Task B0 Option A — supersedes the "does not generate" half of the
> block above).** B0 landed. **144/864 now generates, elaborates, and is bit-exact at 543 cycles**,
> 40/40 against the golden on `bp_relay_banked` (`bpbankedscale`; see the Task B0 Option A RESULT
> section below). The generator defect was three defects, all fixed. So of the two levers named above:
>
> - **"The full-parallel configuration does not build" is no longer true** as a *generation* claim. It
>   remains unanswered as a *fit* claim — whether a 144/864 instance can be placed and clocked on any
>   FPGA, and at what Fmax, is Task B2 and is being measured now.
> - **"The clock cannot currently exceed ~686 MHz" still stands**, unchanged. Phase A owns it.
>
> The cycle-model rows in the table above are therefore no longer arithmetic-only for 144/864 — 543 is
> measured. Every microsecond figure derived from it still is a projection, because it divides a
> measured cycle count by an unmeasured clock. Sub-microsecond remains **unproven**, now for one
> reason instead of two.

### 1.3 Named user segments, honestly ranked

1. **Superconducting-qubit QEC groups without an NVIDIA budget.** Round time ~1 µs, so they genuinely
   need sub-µs. Today their options are a large FPGA or a GPU on NVQLink. **This is the real user.**
2. **Control-electronics builders** (Qblox, Quantum Machines, QuantWare, the QICK/Fermilab community).
   They want a decode block inside the controller. They will take the RTL; some would take a chip.
3. **Neutral-atom and trapped-ion groups.** Round times of 100 µs–ms mean our *existing FPGA build is
   already real-time for them*. They need the design and the bitstream, **not** the chip. Do not
   count them in the silicon case.
4. **Teaching and reproducibility use.** Served by Phase D's fully-open 130 nm part, not by the 28 nm one.

### 1.4 What we are not

We are not first, and the plan must never claim otherwise:

- **Riverlane** has a hardware decoder (Local Clustering Decoder) in Nature Communications and Nature
  Electronics, <1 µs/round, deployed with Rigetti, OQC, Infleqtion, ORNL; Deltaflow 3 (FPGA/ASIC) is
  due late 2026. They have raised $120 M+ and employ ~198 people.
- **NVIDIA** ships relay-BP — our algorithm — inside CUDA-Q QEC 0.6 over NVQLink, adopted by
  Quantinuum, IQM, Pasqal, Alice&Bob, Q-CTRL.
- **QEC Labs** is a startup on hardware-native QLDPC decoders (GPU/FPGA/ASIC).
- In the *open* lane, arXiv:2603.16203 demonstrates a 446 ns open-source decode-feedback loop on ZCU216
  — but for the **surface code at d=3**, a different and easier code family than BB/gross qLDPC.

Our defensible claim is narrow and true: **the first open-source, silicon-validated, sub-µs relay-BP
decoder for bivariate-bicycle qLDPC codes, distributed free.** Everything public must stay inside that
claim.

### 1.5 Assets that make this credible

- Bit-exact co-simulation harness, 25+ Verilator targets in `hw/Makefile`, software golden ↔ RTL.
- **0 mismatches in 10⁶ × 3 shots on silicon** (Q7-06, matched-prior campaign).
- Q7-07: `valid_flag` heralds nearly every logical error (A(p) ≈ 1.0) — an independently publishable
  architectural result.
- A measured banking scaling law with a banking-invariant pipeline tail (Q7-08 / PR #475).
- OpenROAD P&R evidence on two nodes: sky130hd (4.54 mm², routing-infeasible at 5 metals) and ASAP7
  (die 0.163 mm² @45 % util, 0 GRT congestion, 207 residual DRC, route completes).

---

## 2. The three tracks at a glance

| Track | Phase | What | Direct cost | Exit criterion (written into this file before anything downstream starts) |
|---|---|---|---|---|
| **O** | F1–F8 | Licence, CI, repo size, docs, paper, DOI, CUDA-Q plugin, grants | €0 | `hw/` re-licensed with a patent grant, hardware gates green in CI, paper on arXiv, data DOI minted |
| **P** | **P1** | **Appliance v1 — KV260, ships now** | ~€0.3–3 k | One-command deploy; ≥ 1 external lab running it on their own hardware |
| **P** | P2 | Appliance v2 — large-FPGA build, sub-2 µs | €0 (bitstream only) | Published bitstream + utilisation/Fmax on a real large part |
| **P** | P3 | Appliance v3 — ASIC module on a carrier board | €5–15 k | Module BOM under €200 at 100 units, schematic and layout open |
| **S** | A | Trustworthy ASIC numbers (#322) | €0 (compute) | Post-route Fmax ≥ 600 MHz with timing repair actually run, **and** a gate-level-sim-derived power number |
| **S** | **B0** | **Make a full-parallel config exist** (found necessary by B1) | €0 | A GC = 1 geometry generates and passes bit-exact co-sim — today none does |
| **S** | B | Full-parallel proof on rented FPGA | ~€100–500 | 64/192 **and** the B0 full-parallel config pass bit-exact co-sim; P&R on a real large part under 90 % utilisation with a quoted Fmax |
| **S** | **C** | **Demand gate — measured, not surveyed** | €0 | **≥ 5 labs have deployed appliance v1/v2 on their own hardware; ≥ 2 named as early adopters for silicon** |
| **S** | D | Fully-open 130 nm shuttle (flow rehearsal) | €0–20 k | Working silicon back, co-sim vectors pass on it, measured power published |
| **S** | E | 28 nm part | €45–93 k fab | Dies distributed; measured sub-µs on silicon published |

**Start now, in parallel:** O (F1–F3), P1, S-A, S-B. Phase C is not a separate activity — it is a
counter on Track P that reaches its threshold or does not. Phases D and E are locked until it does.

Budget envelope: everything after Phase C ≤ €100 k inclusive of packaging, test boards and shuttle fees.

---

## Track P — the cheap ready product

**Design rule for the whole track:** at every capability level, something works and is documented well
enough that a stranger can deploy it in an afternoon. We never hold a release waiting for silicon.

**Latency tiers and who each one already serves:**

| Version | Hardware | Config | Latency | Real-time for |
|---|---|---|---|---|
| **v1 — now** | Kria KV260, ~$300 | 16/48 banked | 15.64 µs worst, 0.85 µs median early-exit | neutral atoms, trapped ions (round times 100 µs–ms) — **already real-time for them today** |
| v2 | large FPGA ($10–30 k) | 144/864 | ~1.8–2.7 µs projected | most platforms except the fastest superconducting loops |
| v3 | AD-1 ASIC module | 144/864 | 0.54–0.91 µs projected | superconducting, ~1 µs round times |

v1 is the important row: **for two of the four qubit modalities the product is finished and merely
undistributed.** That is the cheapest possible demand experiment.

### Task P1: Appliance v1 — make the KV260 build deployable by a stranger

**Files:**
- Create: `hw/product/README.md`
- Create: `hw/product/deploy.sh`
- Create: `hw/product/interface-spec.md`
- Modify: `.github/workflows/release.yml` (attach the bitstream to releases)

- [ ] **Step 1: Publish the prebuilt bitstream as a release artefact**

Do not ask users to run Vivado. The bitstream `bp_p005.bit` used for the Q7-07 silicon campaign already
exists and is validated at `mismatch = 0` over 100 000 shots.

- [ ] **Step 2: Write the interface specification**

The product's contract with the outside world, and the single most important design decision on this
track. Specify all three paths and state which is primary:
  - **AXI-Lite / AXI-DMA** for designs embedding the core in their own FPGA — already implemented
    (`hw/bp_axi_top_banked.v`, `hw/bp_axi_wrap_banked.sv`), and the Q7-06 batched DMA path is measured.
  - **Low-latency serial** (LVDS or Aurora over SFP+) for use as an external decoder box — the path a
    control-electronics vendor would actually wire up. Not yet implemented; specify before building.
  - **Slow/simple** (SPI or plain parallel) for atom and ion platforms where 15 µs is already ample.

- [ ] **Step 3: Write `deploy.sh`**

One command from a bare KV260 to a running decoder: flash, load bitstream, install the Python driver
(`hw/sw/bp_stream_banked_ler_kv260.py` is the working starting point), run a self-test against the
shipped vectors, print the measured latency.

- [ ] **Step 4: Verify on a fresh board, not on the development board**

Expected: a person who has never seen this repository gets a decoding self-test to pass without asking
us a question. If that fails, the product is not ready, regardless of what the RTL does.

- [ ] **Step 5: Commit and cut a release**

```bash
git add hw/product/
git commit -m "[product] Appliance v1: one-command KV260 deploy, bitstream in releases"
```

### Task P2: Appliance v2 — publish the large-FPGA build

Depends on Phase B, which produces exactly this artefact as a by-product.

- [ ] **Step 1: Publish the 144/864 bitstream and constraints for the part Phase B used**
- [ ] **Step 2: State the achieved Fmax and utilisation, and the resulting latency, in the release notes**
- [ ] **Step 3: Do not buy these boards for other people.** Publish; let the labs that own them use them.

### Task P3: Appliance v3 — the ASIC module

Locked until Phase E produces dies.

- [ ] **Step 1: Design an open carrier board** — chip, power, clock, the serial interface from P1 Step 2,
  and a USB or Ethernet control path. Schematic and layout published under CERN-OHL-S.
- [ ] **Step 2: Target a BOM under €200 at 100 units**, so the module genuinely replaces a $10–30 k FPGA
  rather than merely being smaller.
- [ ] **Step 3: Assemble a first batch of 20–50**, ship free to the Phase C early adopters first.
- [ ] **Step 4: Publish the recipient list and every measured latency number obtained on real setups.**

### Task P4: Support policy — the thing that makes it a product rather than a repo

- [ ] **Step 1: Write it down.** What we answer, how fast, what we do not support, how versions are
  numbered, what "stable" means for the interface spec. One page.
- [ ] **Step 2: Commit to interface stability across v1→v2→v3.** A lab that integrates against v1 must
  not have to rewrite when the ASIC arrives. This single promise is what makes the cheap product a
  bridge to the chip instead of a distraction from it.

---

## Phase A — Make the ASIC numbers trustworthy (issue #322)

**Why this is first:** every later number — the €45 k fab quote, the sub-µs claim, the datasheet —
descends from Fmax and power. Today both are provisional: the ASAP7 flow ran with
`SKIP_INCREMENTAL_REPAIR=1` and `SKIP_CTS_REPAIR_TIMING=1`, and power used ORFS's default switching
activity rather than our traffic.

**Root cause already established (2026-07-26):** `repair_timing` segfaults with a null dereference in
OpenSTA's CRPR arrival pruning, reached through GRT's incremental parasitics update:

```
rsz::Resizer::repairSetup → SetupLegacyPolicy::repairEndpoint
  → est::EstimateParasitics::updateParasitics → grt::GlobalRouter::updateDirtyRoutes
  → grt::FastRouteCore::layerAssignment → updateSlacks → sta::Sta::slack
  → sta::Search::findAllArrivals → ArrivalVisitor::pruneCrprArrivals → sta::Path::minMax  ← SIGSEGV
```

**Second finding:** the critical path is launched by a *latch* (`gmcm[250].u_mcm.mem[18]$_DLATCH_P_`)
whose clock insertion is **1.37 ns**, against 0.47 ns for an ordinary flop. TritonCTS stops at the
generic `AND3x1` gating cell and the 55,703-latch array beyond it is buffered by a *serial*
`load_slew` daisy-chain. The ASAP7 platform sets `DONT_USE_CELLS += ICG*`, so no integrated
clock-gating cell was available to synthesis. **Fmax is limited by a flow artefact, not by the datapath.**

### Task A1: Quantify the gated-clock penalty

**Files:**
- Create: `/data/asicprobe/clk_struct.tcl` (on the EPYC box, already staged)
- Create: `docs/perf/q7-02-asap7-timing.md`

- [ ] **Step 1: Collect the clock-network structure**

Run on `root@195.154.249.85`:

```bash
cd /data/asicprobe && setsid nohup ./run_clkstruct.sh </dev/null >/dev/null 2>&1 &
```

Expected output in `/data/asicprobe/clk_struct.log`: `DISTINCT_LATCH_CLK_NETS`, `DISTINCT_DFF_CLK_NETS`,
`LATCH_NET_SINK_TOP20`, `LATCH_SINKS_TOTAL`.

- [x] **Step 2: Record how many distinct gated-clock nets exist**

If the count is small (< ~2000), the SDC route in Task A2 is viable. If it is very large, skip A2 and
go straight to A3 (re-synthesis with ICG cells enabled).

**Result (2026-07-26):**

| | latches | flops |
|---|---|---|
| cells | 55,703 | 43,025 |
| **distinct clock nets** | **15,951** | 2,321 |
| sinks per net | ~3.5 (max 25) | ~18 |

**15,951 gated-clock domains is an order of magnitude past the viability threshold this task set for
itself.** Declaring that many generated clocks would make STA unusable, so **Task A2 is dead as
written** and is superseded by A3. The structure is fine-grained per-register write enables — roughly
3.5 latches per enable — which is also why re-synthesis with ICG cells would insert ~16 k clock-gating
cells rather than a handful. If the clock tree is ever attacked directly, the cheaper lever is an RTL
change that coarsens the enable granularity (one enable per bank rather than per register), not a
tooling flag.

- [ ] **Step 3: Commit the finding**

```bash
git add docs/perf/q7-02-asap7-timing.md
git commit -m "[Q7-02] ASAP7 clock-network structure: gated-clock nets driving the latch array"
```

### Task A2: Declare the gated clocks and re-run from CTS

**Files:**
- Create: `/data/asicprobe/orfs/m8rf_asap7_gclk/config.mk`
- Create: `/data/asicprobe/orfs/m8rf_asap7_gclk/constraint.sdc`

- [ ] **Step 1: Generate the `create_generated_clock` statements**

For every net found in A1 driving latch `CLK` pins, emit into the SDC:

```tcl
create_generated_clock -name gclk_<n> -source [get_ports clk] -divide_by 1 [get_pins <gate>/Y]
```

- [ ] **Step 2: Re-run the flow from CTS onward with the new SDC**

TritonCTS builds balanced trees for declared clocks, which the plain AND gate previously prevented.

- [ ] **Step 3: Compare `report_clock_min_period` against the 686.13 MHz baseline**

Baseline for comparison (unrepaired, post-route, RCX parasitics, SDC period 1000 ps):
`period_min = 1457.44 ps, fmax = 686.13 MHz`, `wns max = -622.73`, setup skew 940.67 ps.

- [ ] **Step 4: Commit the result, pass or fail**

A negative result is a result — record it either way.

### Task A3: Work around the `repair_timing` segfault

Two independent hypotheses; test the cheap one first, one variable at a time.

**Files:**
- Create: `/data/asicprobe/repair_h1_crpr.tcl`
- Create: `/data/asicprobe/repair_h2_placement.tcl`

- [x] **Step 1: H1 — disable CRPR, the exact function that crashed**

**H1 is not testable: `set_crpr_enabled` does not exist in this build.** `info commands` returns
nothing for it in OpenROAD `26Q3-528-g20d2d5c16e`. CRPR is not exposed as a user-settable toggle, so
the crashing code path cannot be switched off from Tcl. Abandoned without spending a run on it.

- [x] **Step 2: H2 — avoid the GRT incremental path entirely**

```tcl
read_db /work/orfs_out_m8rf_asap7/results/asap7/bp_relay_banked/base/4_1_cts.odb
read_sdc .../4_cts.sdc
set_propagated_clock [all_clocks]
source $plat/setRC.tcl          # else RSZ-0089: no resistance value for any corner
estimate_parasitics -placement
repair_timing -setup_margin 0 -hold_margin 0 -repair_tns 100 -verbose
```

The crash is reached through `updateDirtyRoutes`; placement-based parasitics never call it.

**Result (2026-07-26): H2 holds — `repair_timing` runs at the CTS stage without crashing.** It entered
its iteration loop and reported the endpoint table, where the GRT-stage run died after ~1 h at
iteration 30. Two things follow:

1. The `SKIP_CTS_REPAIR_TIMING = 1` in `orfs/m8rf_asap7/config.mk` was set on the *assumption* that the
   sky130 ODB-0445 CTS crash also applied to ASAP7. **That assumption was never tested and is wrong.**
   The flow has been skipping a repair stage that works.
2. Only `SKIP_INCREMENTAL_REPAIR` is genuinely required. The fix for the flow is to unset
   `SKIP_CTS_REPAIR_TIMING` and re-run from CTS.

A standalone run also needs the platform's `setRC.tcl` — ORFS supplies layer/wire RC via `SET_RC_TCL`,
which a bare `openroad` invocation does not pick up, and without it the resizer aborts with RSZ-0089
before doing any work.

Pre-repair timing at the CTS stage, for comparison with the 686.13 MHz post-route figure:
`fmax = 681.23 MHz` (period_min 1467.94 ps), WNS −507.25 ps with the ODB's own parasitics; after
`setRC` + placement estimation the honest starting point is WNS −1057.2 ps over 52,817 violating
endpoints. **The near-identical CTS and post-route Fmax is itself a finding: routing is not what limits
this design — the clock structure is, and it is already fully formed at CTS.**

- [ ] **Step 3: If both fail, report upstream and stop**

File an OpenROAD issue with the stack trace and the ODB. Do not attempt a third workaround —
per `superpowers:systematic-debugging`, three failures means questioning the approach, and the
fallback (quote the unrepaired number with its conditions stated) is legitimate.

- [ ] **Step 4: Commit whichever outcome occurred**

### Task A4: Real switching-activity power via gate-level simulation

This simultaneously closes the long-standing gap that no gate-level co-simulation of an ASIC netlist
has ever been run (`docs/qec/asic-architecture.md` §6 claims the chain "retargets to gate-level
unchanged" — never tested).

**Files:**
- Create: `hw/tb_bp_gate_asap7.sv`
- Create: `hw/Makefile` target `bpgate-asap7`
- Modify: `docs/perf/q7-02-asap7-timing.md`

**Inputs available:** netlist `6_final.v` (78 MB), ASAP7 behavioural models at
`/OpenROAD-flow-scripts/flow/platforms/asap7/verilog/stdcell/*.v`, and the existing co-sim vectors
(`hw/bp_dec_vectors.txt`, `hw/bp_circ_vectors.txt`).

- [ ] **Step 1: Compile the netlist with Verilator, zero-delay functional mode**

Toggle counts do not need timing annotation. `specify` blocks are ignored by default.

- [ ] **Step 2: Drive one representative decode window and dump VCD**

Use ~300 cycles of steady-state BP iteration rather than the full 2085-cycle window — a full-window VCD
over 654 k cells is tens of GB. Dump a second, separate stretch covering the idle/early-exit regime.

- [ ] **Step 3: Verify the gate-level netlist reproduces the golden output**

This is the co-simulation gate, not an optional extra. Expected: bit-exact match on the same vectors
the RTL passes.

- [ ] **Step 4: Feed the VCD to OpenSTA and report power**

```tcl
read_power_activities -vcd /work/gate_window.vcd
report_power
```

Baseline to beat for honesty, not for pride: ORFS default-activity power was **0.655926 W**
(internal 0.339 W, switching 0.317 W, leakage 6.4e-5 W) with a PSM worst IR drop of 33.9 mV (4.40 %).

- [ ] **Step 5: Convert to energy per window and compare with the §5 budget**

`E = P × cycles / f`. Note in the write-up that energy is roughly **banking-invariant** while latency is
not, and that a 28 nm part will draw several times the 7 nm figure — the §5 "≤1 µJ per window" budget
almost certainly needs restating at the real target node.

- [ ] **Step 6: Commit**

### Task A5: Close #322 with the honest note

**Files:**
- Modify: `docs/qec/BACKLOG.md` (Q7-02 acceptance criteria, ~lines 1318–1319)
- Modify: `docs/qec/asic-architecture.md` §8 open items
- Create: `docs/perf/q7-02-asap7-timing.md` (final form)

- [ ] **Step 1: Write the closure note**

It must state, without softening: gate-level co-sim now exists (A4) or does not; the streaming core
never fit silicon (M9c: LUT 162 %, BRAM 113 %); runt frames (`slices < W`) remain uncovered by co-sim;
64/192 and 144/864 were cycle-counted in Verilator before Phase B, not synthesised; and none of the
hardware gates run in CI.

- [ ] **Step 2: Open the PR with `Closes #322`**

Use the issue number, not the PR number — P0-06/07/08/11 all merged with the wrong reference.

---

## Phase B — Prove the full-parallel configuration (unblocks everything downstream)

**Why:** the entire silicon case rests on 144/864 at 544 cycles. That number is currently a Verilator
cycle count, never synthesised. If it does not close timing or does not fit, the chip is not worth
building and we find out for ~€300 instead of ~€45 000.

### Task B1: Bit-exact co-simulation at 64/192 and 144/864

**Files:**
- Modify: `hw/Makefile` (extend the existing `bpbankedscale` geometry list)

- [x] **Step 1: Add both geometries to the scale sweep**

**64/192 was already there** — `bpbankedscale` sweeps `8 24`, `12 36`, `16 48`, `32 96`, `64 192`.
Only the full-parallel geometry was missing.

- [x] **Step 2: Run and confirm bit-exactness**

**Result (2026-07-26): the full-parallel geometry does not exist. It cannot even be generated.**

`qec_q7_bp_graph -- circgraph 1 0.003 <W> <V>` panics for every full-parallel or ratio-6 geometry:

| W / V | groups (GC = ⌈144/W⌉, GV = ⌈864/V⌉) | result |
|---|---|---|
| 16/48, 64/192, 72/216 | GC ≥ 2 | generates |
| 48/288, 72/432 | GC ≥ 2, V = 6W | **panic** at `qec_q7_bp_graph.rs:969`, `len = 25·W` |
| 144/432 | **GC = 1** | **panic** at `:1290` — `RomRow: zero-width row (degenerate graph parameter)` |
| **144/864** — the full-parallel target | **GC = GV = 1** | **panic** at `:969`, `len is 3600 but the index is 3600` |

`bank_w`/`bank_v` are unvalidated positional arguments, so nothing rejects these inputs — and
144/864 is exactly the configuration `asic-architecture.md` §5 line 163 names as the hard floor
("even full-parallel 144/864 (GC = GV = 1) is 60·9 + 4 = 544 cycles").

**Root cause of the `:969` family.** `benes_group_matchings` allocates
`dest_ecm`/`dest_mcm` sized by its `ecm_m`/`mcm_m` parameters, then indexes them by *tap*
`s = i·var_deg + d`, which ranges over `v·var_deg`. Two of the three call sites (`:1172`, `:1645`,
both on the AS-Waksman path added in M9c / PR #466) pass the **real bank counts** `neb`/`nhb` for
those parameters, not a tap count. Bank count and tap count are different quantities that happen to
satisfy `neb ≥ taps` for the qualified geometries and stop doing so outside them. The third call site
(`:1593`) passes the power-of-two-padded `ecm_m`/`mcm_m` and is unaffected. This is a latent sizing
defect, not a deliberate restriction.

**Superseded (2026-07-30):** every row of the table above now generates and co-simulates bit-exactly —
see "Task B0 Option A RESULT" below. The root cause named here was correct but incomplete: two further
defects sat behind it, both only reachable once the sizing was fixed.

- [x] **Step 3: Commit**

### Task B0 RESULT (2026-07-27): Option B decodes in 181 cycles but does not fit, and is slow

Option B was chosen and executed. Both halves of the answer are now measured, and they point in
opposite directions.

**The good half.** `bp_relay_unrolled.sv` needed no RTL change at all: it lints clean at circuit-DEM
scale and, driven by the same golden and the same compare logic as the banked core
(`make -C hw bpunrollcirc`), passes **40/40 bit-identical at 181 cycles** — against 913 for the banked
core at 64/192 and 544 for the 144/864 configuration that cannot be generated. It spends 3 cycles per
sweep regardless of graph size and carries none of the banked core's banking-invariant 7-cycle tail.

**The bad half — OOC synthesis, `hw/syn/ooc_unrolled.tcl`, xck26-sfvc784-2LV-c, period 5.0 ns:**

```
RESULT cellLUT=1117790 FF=50112 CARRY8=46811 DSP=9 period=5.00 WNS=-27.524 Fmax=30.7MHz
```

| resource | used | available on KV260 | utilisation |
|---|---|---|---|
| **CLB LUTs** | **981,402** | 117,120 | **838 %** |
| CLB registers | 50,106 | 234,240 | 21 % |
| CARRY8 | 46,811 | 14,640 | 320 % |
| F7 / F8 muxes | 143,590 / 62,941 | 58,560 / 29,280 | 245 % / 215 % |

Two independent disqualifications, either of which alone would be fatal:

1. **It needs 8.4× the entire KV260's LUTs.** Registers sit at 21 %, so this is not a storage problem —
   the *combinational* logic explodes. That is the cost of evaluating 144 checks and 864 variables in
   one cycle each.
2. **Fmax is 30.7 MHz**, against the ~200 MHz that 181 cycles needs for sub-microsecond. WNS is
   −27.5 ns at a 5 ns target, i.e. a ~32.5 ns combinational path. Even given infinite area, 181 cycles
   at 30.7 MHz is **5.9 µs**.

**Verdict: sub-microsecond is not reachable by either road today.** The banked road is blocked by
cycles (its pipeline tail is banking-invariant, and its full-parallel geometry does not generate); the
unrolled road is blocked by area and Fmax simultaneously. This is not a tuning gap — 838 % is an order
of magnitude, not a directive-tweak.

It also retrospectively explains why the banked core exists at all. Banking is not an optimisation
bolted onto a working full-parallel design; it is the thing that made the design implementable. M3
diagnosed M2's runtime cursor mux as the wall and M4 removed it — and hit a larger wall behind it.

**Where the real design space is.** Latency is `cycles / Fmax`, and the two roads sit at opposite
extremes of a curve neither of them optimises:

| core | cycles | Fmax | latency | fits KV260 |
|---|---|---|---|---|
| banked 16/48 | 2085 | 133.3 MHz (measured on silicon) | 15.64 µs | yes |
| unrolled | 181 | 30.7 MHz (OOC synth) | 5.9 µs | **no — 838 %** |

The unroll buys 2.65× in latency for 8.4× the area, and cannot be built. The interesting configurations
are in between, and the RTL for them **already exists**: `bp_relay_unroll_pipe.sv` is the M7 partial-unroll
core, parameterised by `NGROUP`, and `hw/syn/ooc_core.tcl` already probes exactly it for fit and Fmax.

**Next experiment (supersedes B2 as written): sweep `NGROUP` and minimise `cycles / Fmax`, subject to
fitting.** That is a cheap OOC sweep on hardware already available, and it is the measurement that
should have preceded any silicon costing.

### Task B0 (NEW, prerequisite for B2): make a full-parallel configuration exist at all

Discovered by B1. Until one of these lands, there is no sub-microsecond design to place, route or cost,
and Phase E has nothing to tape out.

- [x] **Option A — fix the generator.** Give `benes_group_matchings` a tap count rather than a bank
  count at `:1172` and `:1645` (or size the vectors to `max(neb, v·var_deg)`), then handle the
  zero-width ROM row at `:1290` for the GC = 1 case. Add the failing geometries to `bpbankedscale`
  so the regression is permanent. **Done — see the result section below.**

- [ ] **Option B — qualify the M4 unrolled core instead.** `hw/bp_relay_unrolled.sv` already exists
  and already computes all checks and variables per cycle — and `asic-architecture.md:89` calls
  full-parallel "M4-style", which suggests this, not a 144/864 banked instance, was always the
  intended realisation. Its `bpunroll` target runs against the **code-capacity** graph (`graph` /
  `decvectors`), not the circuit-level DEM (`circgraph` / `circvectors`) the silicon path uses, so
  qualifying it means porting it to the circuit-level flow and re-running the bit-exactness gate.

- [x] **Decide between them before doing either.** Option A gets a full-parallel *banked* core, which
  keeps one RTL and one regression suite. Option B may be closer to the intended architecture but
  forks the verification effort. Whichever wins, the deliverable is the same: a generated,
  co-simulated, bit-exact full-parallel configuration with a measured cycle count.
  **Option A was chosen** — one RTL, one golden, one regression suite, and B2 had by then shown the
  unrolled family to be uncompetitive at every knob setting, so forking verification onto it bought
  nothing.

### Task B0 Option A RESULT (2026-07-30): the full-parallel configuration exists, and it decodes in 543 cycles

The deliverable the task asked for is met: **144/864 generates, elaborates, and is bit-exact at 543
cycles.** `bp_relay_banked` at the full-parallel geometry passes the same 40-shot circuit-level golden
every other geometry passes, 40/40 bit-identical, worst = mean = 543 cycles.

| W / V | GC / GV | cycles | co-sim | note |
|---|---|---|---|---|
| 8 / 24 | 18 / 36 | 3750 | 40/40 | regression — unchanged |
| 16 / 48 | 9 / 18 | 2085 | 40/40 | regression — unchanged, the silicon geometry |
| 48 / 288 | 3 / 3 | 789 | 40/40 | **new** — was a generator panic (ratio-6) |
| 144 / 432 | 1 / 2 | 605 | 40/40 | **new** — was a generator panic (GC = 1) |
| **144 / 864** | **1 / 1** | **543** | **40/40** | **new — the full-parallel target** |

543 against the `asic-architecture.md` § 5 prediction of 544. The closed form there reads exactly one
cycle high at every measured point (2086/2085 at 16/48, 3751/3750 at 8/24, 790/789 at 48/288,
606/605 at 144/432), so 543 is the model's answer, not a surprise. § 5 has been corrected in place.

**Three defects, not one.** B1 named the first; the other two only surfaced once it was fixed.

1. **Routing-network sizing.** `benes_group_matchings` allocated its `dest_ecm`/`dest_mcm` vectors by
   *bank* count and indexed them by *tap* (`s = i·var_deg + d`). A rearrangeable network is square, so
   its lane count must cover both endpoints; passing only the bank count worked by accident while
   `neb = 25W ≥ nvb = 6V`, i.e. for every `V = 3W` geometry, and broke for every ratio-6 one. Fixed by
   `asw_network_sizes(neb, nhb, nvb) = (max(neb,nvb), max(nhb,nvb))`, one helper feeding all three
   consumers (the gen-time guard, the emitted control ROMs, the `BP_ASW_*_N` localparams) so they
   cannot drift. **The RTL needed no change for this**: `bp_relay_banked_bram_m.sv` already 0-pads its
   `din` lanes past the real bank count and reads `dout` only at real indices.
2. **Zero-width row address.** `$clog2(1) = 0`, so at GC = 1 (and GV = 1) the row-address field
   collapsed — a zero-width `RomRow` in the emitter and an illegal `logic [-1:0]` in the RTL. Floored
   at 1 bit in both, `row_addr_width` mirroring `BB_BWC`/`BB_BWV`. The bit is always zero; multi-group
   geometries are unaffected.
3. **A tool wall nobody had hit.** At single-group geometries a BRAM-core ROM row carries a whole group
   in one literal, and several cross **Verilator's 65536-bit number limit** (144/432:
   `BP_ROM_BENES_ECMRD` = 78210 bits; 144/864: `BP_ROM_SCAT_HB` 67392, `BP_ROM_BENES_MCMWR` 85409).
   `bp_relay_banked` reads none of those tables but had to *parse* them, so it could not elaborate. The
   block is now `` `ifdef BP_BRAM_ROMS ``-gated and the two BRAM cores opt in ahead of their `include`.

**Regression safety.** At all five previously-qualified geometries (8/24, 12/36, 16/48, 32/96, 64/192)
the emitted header is **byte-identical** to before the fix apart from the two `ifdef` guard lines —
`max(neb, nvb) = neb` and `clog2(GC) ≥ 1` are both no-ops there. 8/24 and 16/48 were re-co-simulated to
confirm it: same cycle counts, 40/40. The shipped 16/48 bitstream's header is unchanged.

**What this does and does not settle.** It settles that the configuration `asic-architecture.md` § 5
rests its whole silicon case on is real and correct, and what it costs in cycles. It settles nothing
about whether it can be built: 543 cycles at the silicon-measured 133.3 MHz would be 4.1 µs, and sub-µs
needs ~600 MHz on a core whose smaller siblings already congest. **Fit and Fmax for 144/864 remain
Task B2**, and B0/B2 have already shown that this core family's area is dominated by a crossbar that
does not shrink. The honest reading is that B0 Option A removes an *excuse* — "we cannot even generate
it" — not the wall.

**Not done here:** 144/864 was co-simulated on `bp_relay_banked` only. That core uses the mux crossbar,
not the AS-Waksman fabric, so the widened network is checked at the new geometries by the generator's
own round-trip guard (`complete_partial` → `aswaksman_control` → `aswaksman_apply`, asserting every tap
lands on its bank) and by construction in the RTL, but **not by an RTL simulation**. The two BRAM cores
take the same `ifdef` opt-in and the same width floors, and their existing 8/24 and 16/48 gates still
pass, but neither was built at a single-group geometry — at 144/864 their fabric would be a 5184-lane
network with 59201 switches per port, which is not a design anyone should synthesise.

### Task B2 RESULT (2026-07-28): the `NGROUP` knob has no feasible setting, and the banked core dominates it outright

The B0 verdict ended by naming the next experiment: sweep `NGROUP` on `bp_relay_unroll_pipe` and minimise
`cycles / Fmax` subject to fitting. That sweep has been run — five points synthesised out-of-context on
the KV260 part, same flow and 5.0 ns period as B0. Full report: `docs/perf/q7-02-ngroup-sweep.md`.

**There is no feasible setting.** Every point is over the device budget *and* slower than the banked core
that already ships:

| NGROUP | cycles | CLB LUTs | % of KV260 | Fmax | latency |
|---|---|---|---|---|---|
| 144 | 17,808 | 703,698 | 601 % | 17.5 MHz | 1017.6 µs |
| 72 | 9,024 | 858,952 | 733 % | 17.4 MHz | 518.6 µs |
| 48 | 6,096 | 1,042,940 | 890 % | 17.3 MHz | 352.4 µs |
| 24 | 3,168 | 1,433,697 | 1224 % | 16.8 MHz | 188.6 µs |
| 16 | 2,192 | 1,726,227 | 1474 % | 16.5 MHz | 132.8 µs |
| **banked 16/48** | **2,085** | **fits** | — | **133.3 MHz** | **15.64 µs** |

Cycles are exactly `122 · NGROUP + 240` and every value is bit-exact against the golden (40/40), so the
knob is a pure latency knob as designed — it just does not buy anything. Three findings:

1. **Area moves the wrong way**: shrinking `NGROUP` makes the core bigger, so the cheapest member has the
   worst cycle count.
2. **Fmax is flat at 16.5–17.5 MHz across a 9× span of `NGROUP`** — a 9× change in stamped arithmetic
   barely moves the critical path, because the critical path is not the arithmetic.
3. **The best member is 8.5× slower than the banked core at 14.7× the device's LUTs.** The banked core
   wins on both axes simultaneously; there is no trade to make.

The cause is structural, not a tuning gap. Each slot gathers across `NGROUP` groups and there are
`BP_C/NGROUP` slots, so the product cancels and the crossbar is `NGROUP`-invariant by construction. Area
fits `615,985 + 18,415,474/NGROUP` (residuals within 5.7 %), and the floor was then **measured, not
extrapolated**: at `NGROUP = 144`, with one check slot and six variable slots stamped, the core still
needs **703,698 LUTs = 601 % of the whole device**. The crossbar is simultaneously the area floor and the
critical path.

**Consequences for this plan.** B0 closed the fully-unrolled road; B2 now closes the partially-unrolled
road at every knob setting. Sub-µs is not reachable by *any* degree of unrolling layered on a flat
message-register array — the remaining attack surface is the message store and its access pattern, which
is precisely what banking already attacks. Step 4 of the task below should be read in that light: the
question is no longer "which unroll degree do we place on a large FPGA", it is whether a *banked*
full-parallel geometry can be made to generate at all (Task B0 Option A, still open).

### Task B2: Synthesise and place-and-route on a large FPGA

**Files:**
- Create: `hw/syn/f1_144x864.tcl`
- Create: `docs/perf/q7-02-fullparallel-fpga.md`

- [x] **Step 0: Probe the banking curve out-of-context first — it is free** — **DONE 2026-07-30, verdict: proceed**

Full report: `docs/perf/q7-02-fullparallel-fpga.md`. Four geometries out-of-context on the KV260 part
at 5.0 ns, ascending and serial; the whole sweep took 70 minutes and peaked at 9.1 GB.

| W/V | GC/GV | cycles | CLB LUTs | % KV260 | Fmax |
|---|---|---|---|---|---|
| 16/48 *(ships)* | 9/18 | 2085 | 94,182 | 80.4 % | 177.7 MHz |
| 48/288 | 3/3 | 789 | 291,098 | 248.6 % | 155.8 MHz |
| 144/432 | 1/2 | 605 | 490,944 | 419.2 % | 164.3 MHz |
| **144/864** | **1/1** | **543** | **803,518** | **686.1 %** | **154.0 MHz** |

Two findings:

1. **Banking preserves the clock.** Fmax falls only 13 % across an 8.5× area growth, against the
   unrolled core's 30.7 MHz and the `NGROUP` family's flat 16.5–17.5 MHz. The banked core's critical
   path is deep arithmetic (25 levels, check min-sum → `ehat_w`), not a fabric-wide crossbar.
2. **144/864 is 61.6 % of a VU47P**, the part AWS rents today — inside Step 4's 90 % gate, with every
   non-LUT resource nearly empty. Against 838 % (unrolled) and 601–1474 % (`NGROUP`), this is the
   first configuration in the program that looks buildable.

**Calibrated, not just estimated:** 16/48 is in this sweep *and* on silicon, so OOC's 1.33× optimism is
measured. De-rating 144/864 the same way gives **~115 MHz → ~4.7 µs**, against the shipped 15.64 µs.
Sub-µs remains off the FPGA road entirely — 543 cycles in 1 µs needs 543 MHz.

- [ ] **Step 1: Rent the build instance** *(rewritten 2026-07-30 — the original text is obsolete)*

**Step-by-step runbook: `docs/qec/b2-aws-build-runbook.md`** — CLI-first, with the vCPU-quota trap that
blocks a new account, a five-minute licence smoke test to run before committing hours, and the
shut-it-down checklist.

Two corrections to what this step used to say:

- **AWS F1 is end-of-life** (end of 2025, closed to new users). Its replacement is **F2**, carrying the
  **Virtex UltraScale+ HBM VU47P**. Target that part; do not create `hw/syn/f1_144x864.tcl`.
- **No FPGA instance is needed.** Steps 2–3 want utilisation and Fmax, i.e. synthesis and
  implementation — not a running image. AWS's own dev-kit documentation says builds do not require an
  F-family instance and recommends ≥ 4 vCPU / ≥ 32 GiB, x86 only.

Rent **z1d.2xlarge** (8 vCPU, 64 GiB, ~$0.744/h — highest sustained clock, and Vivado P&R is largely
single-threaded) with the **FPGA Developer AMI**, ~100 GB gp3, ~40 instance-hours for both configs plus
retries. **Budget $30–60, not €100–500.**

**The licence is the only reason AWS is involved.** Vivado ML Standard (free) does not support Virtex
UltraScale+; Enterprise is ~$4,395 node-locked; the AWS AMI bundles a licence valid on EC2 for AWS's
parts only. Vultr/Hetzner/our own EPYC box are all adequate *machines* — the EPYC box just ran Step 0
for €0 and exceeds AWS's recommended build spec — and all are blocked on the licence alone. If a
licence is ever obtained by another route (purchase, or the AMD University Program / Europractice
academic route this program already contemplates for 28 nm sign-off), every future large-part build
including the Track P2 appliance-v2 bitstream runs on hardware we own at zero marginal cost. Price that
deliberately against Track P2 rather than renting by reflex.

- [x] **Steps 2–3: implementation and numbers** — **DONE 2026-07-31**

`z1d.2xlarge` + FPGA Developer AMI (Vivado 2025.2), part `xcvu47p-fsvh2892-2-e`, full
synth→opt→place→phys_opt→route via `hw/syn/impl_vu47p.tcl`. **7.3 instance-hours, ~$5.50.**

| | cycles | CLB LUTs | % VU47P | Fmax (post-route) | latency |
|---|---|---|---|---|---|
| 64/192 | 913 | 247,434 | **19.0 %** | 150.4 MHz | **6.07 µs** |
| 144/864 | 543 | 994,700 | **76.3 %** | 97.3 MHz | **5.58 µs** |
| *(shipped 16/48, KV260)* | 2085 | — | — | 133.3 MHz | 15.64 µs |

**The fit gate passes and the latency case does not.** 1.68× fewer cycles bought 1.55× less clock —
net **1.09× for 4.0× the area**. Full report: `docs/perf/q7-02-fullparallel-fpga.md` §8–13.

The clock wall is not logic depth. The critical path has **8 logic levels and 92.7 % routing**, running
from the `pc_reg[1]` control counter to a min-sum cell indexed 5562 — one register driving thousands of
loads across three SLRs, with Laguna crossing tiles inside level-5/6 congestion windows. That is the
fanout risk Step 0 named, on the same net. It was a default-directive run: `MAX_FANOUT`, per-bank
replication, SLR floorplanning and aggressive `phys_opt` are all **untried**, so 97.3 MHz is a floor
for this design rather than its ceiling.

Step 0's projections were optimistic in both directions that matter — 61.6 % vs 76.3 % measured, and
~4.7 µs vs 5.58 µs. Cross-part LUT extrapolation is worth about ±25 %, not the precision implied.

- [x] **Step 4: Decide** — **DECIDED 2026-07-31**

1. **The ASIC area estimate is trustworthy** in the sense this step intended: 144/864 places and routes
   under 90 % on a rentable part.
2. **Track P2's appliance v2 ships 64/192, not 144/864** — 92 % of the latency in a quarter of the
   device, leaving room for a host interface, and a 65-minute build instead of six hours.
3. **The sub-µs assumption is now an evidenced risk.** The program has been quoting 543 cycles ÷
   600–1000 MHz as if the clock were geometry-independent. It is not: the full-parallel geometry lost a
   third of its clock to distribution effects that scale with size. The one ASIC-node number we own —
   686 MHz on ASAP7 — was measured on **16/48**, not on this geometry. If even half the penalty
   transferred, 543 cycles at ~450 MHz is 1.2 µs and sub-µs is gone.
4. Therefore **Task B3 (144/864 through the ASIC flow) is now the decisive measurement of the silicon
   track**, ahead of any further FPGA work. It is also free — OpenROAD on our own box.

- [ ] **Step 5 (optional, ~$4): one follow-up FPGA run with the control registers replicated**, to
  separate "this design is slow" from "these were the default settings". Cheap, and it is the only open
  question the FPGA road still has.

- [x] **Step 5: Commit both results** — this document and `docs/perf/q7-02-fullparallel-fpga.md`.

### Task B3: Re-derive the 28 nm area and fab cost from real synthesis

**Files:**
- Modify: `docs/qec/asic-architecture.md` §7

Current estimate, from log-interpolating our own two ORFS data points (4.54 mm² @130 nm sky130hd,
0.163 mm² @7 nm ASAP7; fitted slope 1.139 in log-node):

| Config | Est. area @28 nm | mini@sic 28 nm fab cost |
|---|---|---|
| 16/48 | ~0.8 mm² | ~€10.6 k |
| 64/192 | ~2.4 mm² | ~€23.5 k |
| **144/864** | **~4.7 mm²** | **~€45 k** |

Pricing basis: TSMC 28 nm HPC+ RF mini@sic, €10,609 for the first mm² plus €919 per additional 0.1 mm²,
1 mm² minimum, registration 3 months ahead.

- [ ] **Step 1: Replace the interpolation with the Phase B synthesis-derived cell count**

- [ ] **Step 2: State the uncertainty band explicitly** — a two-point node interpolation is worth about
  ±2×; at the pessimistic end 144/864 is ~10 mm² ≈ €93 k, which is still inside budget but leaves no
  room for tooling.

- [ ] **Step 3: Commit**

---

## Phase C — Demand gate (measured by deployments, not by interviews)

**This phase costs nothing and blocks all silicon spending.** Its purpose is to answer "who needs it and
why" with names rather than reasoning — and, because Track P ships a working product first, with
*deployments* rather than opinions. A lab that has gone to the trouble of running our decoder on their
own hardware has demonstrated demand in a way no survey can.

### Task C1: Write the one-page chip spec sheet

**Files:**
- Create: `docs/qec/ad1-datasheet-draft.md`

- [ ] **Step 1: State, on one page:** target latency and the configuration that achieves it, host
  interface, power, die area, package, licence, price to the recipient (free), and what it does *not*
  do (surface-code MWPM, arbitrary codes, on-chip syndrome extraction).

- [ ] **Step 2: State the comparison honestly** — against a large FPGA, against GPU+NVQLink, and
  against Riverlane's published <1 µs.

### Task C2: Interview 12–20 named groups

- [ ] **Step 1: Build the target list** across the four segments in §1.3, prioritising superconducting
  groups not already committed to Riverlane (Rigetti, OQC, Infleqtion, ORNL, NQCC, QuEra, Atlantic
  Quantum) or to NVQLink (Quantinuum, IQM, Pasqal, Alice&Bob, Q-CTRL).

- [ ] **Step 2: Ask each the same four questions.** Do you decode in real time today, and with what?
  What is your per-QPU budget for decoding? Would a free chip plus open RTL change what you do?
  What interface would it have to speak?

- [ ] **Step 3: Record every answer verbatim** in `docs/qec/ad1-demand-validation.md`, including the
  refusals and the silences. A thin file is the most valuable possible outcome if it is true.

### Task C3: The gate

- [ ] **Decision.** Proceed to Phase D/E only if **≥ 5 labs have actually deployed appliance v1 or v2 on
  their own hardware, and ≥ 2 agree in writing to be named early adopters for the chip.** Interviews
  (C2) seed the funnel and shape the interface spec; they do not satisfy the gate on their own —
  deployments do.

- [ ] **Fallback if the gate fails.** Stop at FPGA plus published RTL, which is a complete and valuable
  outcome — `ROADMAP.md` §0 already says so — and redirect the budget into Track O and Track P: more
  seeded boards, better documentation, the paper, the CUDA-Q plugin. Note honestly that a failed gate
  most likely means the product is not yet good enough to deploy, not that nobody wants decoding; in
  that case the correct response is to fix the product, not to abandon the goal.

---

## Phase D — Fully-open 130 nm shuttle (flow rehearsal, and the only fully-open artefact)

**Purpose is explicitly *not* performance.** At 130 nm this core cannot beat the KV260. What it buys:
the whole tape-out flow learned end-to-end at near-zero PDK cost, a real measured-power datapoint, and
the one deliverable that is reproducible by anyone on Earth with no NDA.

- [ ] **Task D1:** Apply for IHP SG13G2 free MPW area. IHP offers free area in MPW runs for
  non-economic use (university education and research), the PDK is open, and OpenROAD already supports
  it. A sustainable free/low-cost MPW scheme for the open-source community was announced as under
  development for 2026 — confirm current terms at
  `https://www.ihp-microelectronics.com/services/research-and-prototyping-service/fast-design-enablement/open-source-pdk`.

- [ ] **Task D2:** If IHP free area is unavailable, fall back to ChipFoundry chipIgnite (SKY130,
  **$14,950**, ~10 mm² Caravel user area, 100 QFN packaged parts, ~5 months, 3 shuttles planned for
  2026). Note that Efabless, the original operator, shut down; ChipFoundry took the platform over.

- [ ] **Task D3:** Tape out a *subset* — one check-bank plus one variable-bank plus the regfile and the
  AXI interface — not the whole core. The goal is flow rehearsal and a power datapoint.

- [ ] **Task D4:** Bring-up on the existing PYNQ harness re-pointed at the chip; run the same co-sim
  vectors; publish the measured power.

**Exit criterion:** working silicon back, co-sim vectors pass on it, power published.

---

## Phase E — The 28 nm part (the chip that is actually useful)

**Entry:** Phase A, B and C exits all met. Not before.

- [ ] **Task E1:** Secure PDK and sign-off tooling. This is the real blocker, not the fab fee.
  TSMC 28 nm needs foundry sign-off DRC/LVS decks (Calibre-class), which OpenROAD does not replace.
  Routes worth pursuing, in order: an academic/institutional partner with Europractice membership;
  the Chips JU **EuroCDP** platform, which has a framework agreement with Siemens EDA explicitly aimed
  at lowering EDA cost for SMEs and start-ups; a commercial broker as the expensive fallback.

- [ ] **Task E2:** Harden the design for a real PDK — memory strategy in particular. The Q7-08 latch
  regfile avoids a memory-compiler licence, which is a real cost saving; verify it survives at 28 nm
  or budget for an SRAM compiler.

- [ ] **Task E3:** Register at least 3 months before the shuttle deadline (mini@sic requirement).

- [ ] **Task E4:** Tape out 144/864 (or 64/192 if Phase B forced the fallback). Fab cost ≈ €45 k at the
  central area estimate; ≈ €93 k at the pessimistic end.

- [ ] **Task E5:** Packaging, test board, bring-up. Budget €10–40 k. This is routinely underestimated.

- [ ] **Task E6:** Distribute the dies. 40 diced samples come with the mini@sic slot; order more.
  Publish a one-page application form; ship free to labs; keep a public recipient list.

**Exit criterion:** measured sub-µs decode latency on silicon, published, with dies in other people's hands.

---

## Track O — Openness and distribution (starts now, runs continuously)

This track is not a fallback and not a finishing touch. It is the precondition for Track P having any
users at all: an undiscoverable, unlicensed, untested repository ships to nobody, and a product with no
users cannot serve as Track S's demand gate. F1–F3 are prerequisites for the appliance v1 release.

- [ ] **Task F1: Re-licence `hw/`.** The repo is MIT, which has **no explicit patent grant** — a blocker
  for anyone whose lawyers must approve pulling RTL into a chip. Move `hw/` to **Apache-2.0** or
  **CERN-OHL-P v2**; keep the Rust crates MIT or dual MIT/Apache-2.0 as is idiomatic in that ecosystem.

- [ ] **Task F2: Put the hardware gates in CI.** `.github/workflows/` currently contains only
  `bench.yml`, `ci.yml` and `release.yml` — no `verilator`, no `make -C hw`. 58 make targets exist and
  none of them are enforced automatically. An open hardware project whose hardware is not tested in CI
  does not earn trust.

- [ ] **Task F3: Document the Q7 hardware in `hw/README.md`.** ~~Shrink `hw/`~~ — **withdrawn, the
  premise was wrong.** `hw/` is 2.8 GB *on disk*, but that is entirely `_bp*build/` Verilator output,
  already covered by `hw/.gitignore:52`. Tracked content is **6.4 MB across 165 files**, and the whole
  repository packs to **7.17 MiB**. A fresh clone is small; nothing needs shrinking.

  The real documentation gap is different and worse: `hw/README.md` is **entirely Q6-centric** — surface
  code, Union-Find, the Arty/KV260 bring-up — and contains **no section at all** on relay-BP, the banked
  M8 core, the streaming core, or anything else from Q7-02 onward. The directory's own front page does
  not mention the design this whole program is about. Fix that before pointing outsiders at the repo.

- [ ] **Task F4: Ship "decoder-in-a-box".** Pre-built KV260 bitstream, Python driver, one command,
  one page of documentation. This is the artefact most people will actually use.

- [ ] **Task F5: Publish the paper.** The M0→M9c ladder plus silicon campaigns plus the Q7-07 heralding
  result. `docs/perf/qec-q7-fixed-bp.md` is already a 1863-line master record. Mint a Zenodo DOI for
  the campaign CSVs so the data is citable.

- [ ] **Task F6: Write a CUDA-Q QEC decoder plugin.** NVQLink is where every real-time QEC integrator
  already is; a plugin lets them A/B our decoder against the GPU path on their own hardware. Using the
  competitor's platform as our distribution channel is the cheapest reach available.

- [ ] **Task F7: Register in the community indexes** — Error Correction Zoo decoder list,
  `qosf/awesome-quantum-software`.

- [ ] **Task F8: Apply for aligned funding.** Unitary Foundation microgrants ($4 k, worldwide, open
  quantum projects). NGI Zero Commons (NLnet, €21.6 M committed across 2026–27, **submissions currently
  paused** pending the EC Tech Sovereignty package of 2026-06-03 — watch for reopening).

---

## 3. Corrections owed to existing documents

`docs/qec/asic-architecture.md` §7 is now known to be wrong in three ways and must be fixed as part of
Phase A:

1. **Efabless is defunct.** chipIgnite is operated by ChipFoundry at **$14,950**, not "~$10–15 k" from
   Efabless, with 3 shuttles planned for 2026 rather than quarterly.
2. **22FDX is under-priced by roughly 3–5×.** The table says "~€15–60 k"; Europractice 2026 lists
   **€17,820/mm²** (€16,200 discounted) with a 4 mm² minimum — i.e. **~€71 k floor**.
3. **The TSMC mini@sic ladder is missing entirely**, and it is the cheapest real route to a
   performance-class node: 65 nm €4,491 + €419/0.1 mm²; 28 nm €10,609 + €919/0.1 mm²;
   16 nm €30,592 + €2,827/0.1 mm², each with a 1 mm² minimum.

`docs/qec/BACKLOG.md` Q7-03 (issue #323) must be rewritten: the acceptance criterion "explicit gate
decision recorded" is now satisfiable, but the decision is **not** the no-go of 2026-07-26. It is
**conditional go, open-hardware, staged, self-funded, gated on Phase C**. Record the date and the
reasoning, including that the commercial thesis was rejected and why.

`ROADMAP.md` §0 and §102 say the Q7 trigger is "funding + a committed QPU-company customer". That rule
was written for a commercial tape-out. It does not fit a community-funded open chip and must be amended
rather than quietly ignored — the replacement trigger is Phase C's gate.

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ~~144/864 does not generate~~ **RETIRED 2026-07-30**: B0 Option A fixed the three generator defects; 144/864 is bit-exact at 543 cycles | — | — | Cost ~€0 to find and ~€0 to fix, before any FPGA rental let alone silicon |
| ~~144/864 does not fit~~ **RETIRED 2026-07-31**: it places and routes at **76.3 % of a VU47P**, under the 90 % gate | — | — | Settled for $5.50 of instance time |
| **The clock is geometry-dependent, and the sub-µs case assumed it was not** — *new, and now the top risk* | **High.** Measured: going 64/192 → 144/864 on the same part cost **150.4 → 97.3 MHz**, a third of the clock, to control-net distribution across three SLRs. The 686 MHz ASAP7 figure the silicon case rests on was measured on **16/48**, a geometry 8.5× smaller | If even half that penalty transfers to 28 nm, 543 cycles at ~450 MHz is **1.2 µs** and sub-microsecond is gone — which removes the entire technical justification for the chip | **Task B3: run 144/864 through the ASIC flow.** Free, on our own box, and now the decisive measurement of the track. Separately, the FPGA penalty is partly an artefact of fixed interconnect and Laguna crossings that an ASIC does not have — do not assume it transfers, but do not assume it does not |
| **The 97.3 MHz may be the defaults rather than the design** | Medium | If replication recovers the clock, the FPGA appliance-v2 story improves; if it does not, the control-distribution problem is structural and follows the design into silicon | One ~$4 follow-up run with `MAX_FANOUT` / per-bank replication on the `pc` counter. Untried: this was a default-directive run with no floorplanning |
| Sub-µs unreachable even after B0, because Fmax is capped ~686 MHz | **High** | 64/192 lands at 1.33 µs, not sub-µs; Riverlane already ships <1 µs | Needs the clock-structure work (A1: 15,951 gated-clock nets), which is an RTL enable-granularity change, not a tooling flag. Decide whether that is in scope before Phase E |
| No sign-off EDA access at 28 nm | **High** | Blocks Phase E entirely | Phase E Task E1 is a gate, not a step; EuroCDP and academic partnership are the routes; Phase D proves the flow at 130 nm regardless |
| First silicon dead on arrival | Medium (30–50 % is normal) | €45–95 k lost | Phase D rehearsal on a free/cheap shuttle; conservative interface design; on-chip observability |
| Demand does not materialise (Phase C fails) | Medium | Silicon stops; product and open corpus continue | A legitimate outcome, not a failure — budget redirects into Tracks O and P |
| Product ships but nobody can deploy it unaided | Medium | Phase C gate misreads as "no demand" | P1 Step 4 tests deployment on a fresh board by a fresh person; kill criterion 2 distinguishes a bad product from absent demand |
| GPU path wins before we ship (12–18 months) | **High** | Chip is obsolete on arrival | Accept it. The open RTL, data and paper retain value independently; do not let the chip become the only deliverable |
| Area estimate off by 2× | Medium | €45 k → €93 k | Still inside budget, but leaves nothing for tooling; Phase B replaces the estimate with synthesis |
| Dual-use / export-control friction | Low | Delay | Open publication is the safest posture; take advice before shipping dies across borders |

## 5. Kill criteria — when to stop, decided in advance

Stop the **silicon** track — Tracks O and P continue regardless — if **any** of these becomes true:

1. Phase B shows 144/864 needs more than ~10 mm² at 28 nm, pushing fab past €93 k with no tooling budget left.
2. Phase C's counter stalls below 5 real deployments **after** appliance v1 has been genuinely deployable
   for six months. (Below 5 *before* that, the fault is the product, not the demand — fix the product.)
3. Phase E Task E1 finds no viable sign-off-tooling route within 6 months.
4. Total committed spend would exceed €100 k.
5. A functionally equivalent open chip ships from someone else first — in which case help them instead.

Nothing on this list stops Track O or Track P. The cheap product and the open corpus are the deliverables
that survive every bad outcome, and they are the reason the silicon is worth attempting at all.

## 6. Budget summary

| Line | Low | High |
|---|---|---|
| Track O (licence, CI, repo hygiene, paper, DOI, plugin) | €0 | €0 |
| Track P — appliance v1, seed boards for early labs (5–10 × KV260) | €0 | €3 k |
| Track P — appliance v3 carrier board design + first 20–50 modules | €5 k | €15 k |
| Phase A (compute only) | €0 | €0 |
| Phase B (rented FPGA) | €100 | €500 |
| Phase C | €0 | €0 |
| Phase D (open 130 nm shuttle + test board) | €0 (IHP free area) | €20 k (chipIgnite + board) |
| Phase E fab (144/864 @28 nm) | €45 k | €93 k |
| Phase E packaging, test board, bring-up | €10 k | €40 k |
| **Total** | **~€60 k** | **exceeds ceiling — see kill criterion 4** |

The high column overshoots deliberately: at the pessimistic end of *both* area and tooling, this program
does not fit in €100 k, and the plan says so up front rather than discovering it at the shuttle deadline.
Phase B exists to collapse that uncertainty before any money moves — it costs about €300 and removes the
single largest unknown in the budget.

Note the shape of the spend: **everything that produces a usable product costs under €20 k**, and the
€45–93 k is bought purely to move 1.8–2.7 µs down to 0.54–0.91 µs. That is the honest framing of what the
chip is for, and it is worth restating to anyone who asks why an ASIC is needed when an FPGA exists.
